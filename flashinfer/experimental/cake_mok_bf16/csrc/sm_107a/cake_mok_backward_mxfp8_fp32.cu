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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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
#define TMEM_NCOLS 572
#define TMEM_TMEM_SFA_OFFSET 512
#define TMEM_TMEM_SFB_OFFSET 524
#define TMEM_TMEM_SFB_HI_OFFSET 548
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
#define SMEM_MN_VIEW_OFF 1024
#define SMEM_MN_VIEW_STAGE_BYTES 16384
#define SMEM_MN_VIEW_STRIDE 16384
#define SMEM_B_NT_OFF 66560
#define SMEM_B_NT_STAGE_BYTES 16384
#define SMEM_B_NT_STRIDE 16384
#define SMEM_MN_VIEW_0_OFF 66560
#define SMEM_MN_VIEW_0_STAGE_BYTES 16384
#define SMEM_MN_VIEW_0_STRIDE 16384
#define SMEM_B_NT_HI_OFF 132096
#define SMEM_B_NT_HI_STAGE_BYTES 16384
#define SMEM_B_NT_HI_STRIDE 16384
#define SMEM_MN_VIEW_1_OFF 132096
#define SMEM_MN_VIEW_1_STAGE_BYTES 16384
#define SMEM_MN_VIEW_1_STRIDE 16384
#define SMEM_D_SMEM_OFF 207872
#define SMEM_D_SMEM_STAGE_BYTES 8192
#define SMEM_D_SMEM_STRIDE 8192
#define SMEM_FLAT_VIEW_OFF 207872
#define SMEM_FLAT_VIEW_STAGE_BYTES 24576
#define SMEM_FLAT_VIEW_STRIDE 24576
#define SMEM_SW_DH_OFF 1024
#define SMEM_SW_DH_STAGE_BYTES 32768
#define SMEM_SW_DH_STRIDE 32768
#define SMEM_SW_GATE_OFF 66560
#define SMEM_SW_GATE_STAGE_BYTES 32768
#define SMEM_SW_GATE_STRIDE 32768
#define SMEM_SW_UP_OFF 132096
#define SMEM_SW_UP_STAGE_BYTES 32768
#define SMEM_SW_UP_STRIDE 32768
#define SMEM_FLAT_VIEW_0_OFF 197632
#define SMEM_FLAT_VIEW_0_STAGE_BYTES 1024
#define SMEM_FLAT_VIEW_0_STRIDE 1024
#define SMEM_FLAT_VIEW_1_OFF 1024
#define SMEM_FLAT_VIEW_1_STAGE_BYTES 65536
#define SMEM_FLAT_VIEW_1_STRIDE 65536
#define SMEM_FLAT_VIEW_2_OFF 66560
#define SMEM_FLAT_VIEW_2_STAGE_BYTES 65536
#define SMEM_FLAT_VIEW_2_STRIDE 65536
#define SMEM_FLAT_VIEW_3_OFF 132096
#define SMEM_FLAT_VIEW_3_STAGE_BYTES 65536
#define SMEM_FLAT_VIEW_3_STRIDE 65536
#define SMEM_REPLAY_GATE_OFF 1024
#define SMEM_REPLAY_GATE_STAGE_BYTES 32768
#define SMEM_REPLAY_GATE_STRIDE 32768
#define SMEM_REPLAY_UP_OFF 99328
#define SMEM_REPLAY_UP_STAGE_BYTES 32768
#define SMEM_REPLAY_UP_STRIDE 32768
#define SMEM_FLAT_VIEW_4_OFF 197632
#define SMEM_FLAT_VIEW_4_STAGE_BYTES 34816
#define SMEM_FLAT_VIEW_4_STRIDE 34816
#define SMEM_FLAT_VIEW_5_OFF 197632
#define SMEM_FLAT_VIEW_5_STAGE_BYTES 34816
#define SMEM_FLAT_VIEW_5_STRIDE 34816
#define SMEM_SMEM_V19_OFF 1024
#define SMEM_SMEM_V19_STAGE_BYTES 81920
#define SMEM_SMEM_V19_STRIDE 81920
#define SMEM_SMEM_V19_0_OFF 17408
#define SMEM_SMEM_V19_0_STAGE_BYTES 66048
#define SMEM_SMEM_V19_0_STRIDE 66048
#define SMEM_SMEM_V19_1_OFF 99328
#define SMEM_SMEM_V19_1_STAGE_BYTES 81920
#define SMEM_SMEM_V19_1_STRIDE 81920
#define SMEM_SMEM_V19_2_OFF 115712
#define SMEM_SMEM_V19_2_STAGE_BYTES 66048
#define SMEM_SMEM_V19_2_STRIDE 66048
#define SMEM_FLAT_VIEW_6_OFF 1024
#define SMEM_FLAT_VIEW_6_STAGE_BYTES 131072
#define SMEM_FLAT_VIEW_6_STRIDE 131072
#define SMEM_FLAT_VIEW_7_OFF 199680
#define SMEM_FLAT_VIEW_7_STAGE_BYTES 512
#define SMEM_FLAT_VIEW_7_STRIDE 512
#define SMEM_SMEM_V25_OFF 1024
#define SMEM_SMEM_V25_STAGE_BYTES 131072
#define SMEM_SMEM_V25_STRIDE 131072
#define SMEM_SMEM_V26_OFF 1280
#define SMEM_SMEM_V26_STAGE_BYTES 130816
#define SMEM_SMEM_V26_STRIDE 130816
#define SMEM_SMEM_V27_OFF 1536
#define SMEM_SMEM_V27_STAGE_BYTES 130560
#define SMEM_SMEM_V27_STRIDE 130560
#define SMEM_SMEM_V28_OFF 1792
#define SMEM_SMEM_V28_STAGE_BYTES 130304
#define SMEM_SMEM_V28_STRIDE 130304
#define SMEM_SMEM_V29_OFF 132096
#define SMEM_SMEM_V29_STAGE_BYTES 16384
#define SMEM_SMEM_V29_STRIDE 16384
#define SMEM_SMEM_V30_OFF 148480
#define SMEM_SMEM_V30_STAGE_BYTES 16384
#define SMEM_SMEM_V30_STRIDE 16384
#define SMEM_SMEM_V31_OFF 164864
#define SMEM_SMEM_V31_STAGE_BYTES 512
#define SMEM_SMEM_V31_STRIDE 512
#define SMEM_SMEM_V32_OFF 165376
#define SMEM_SMEM_V32_STAGE_BYTES 512
#define SMEM_SMEM_V32_STRIDE 512
#define SMEM_SMEM_V33_OFF 165888
#define SMEM_SMEM_V33_STAGE_BYTES 16384
#define SMEM_SMEM_V33_STRIDE 16384
#define SMEM_SMEM_V34_OFF 182272
#define SMEM_SMEM_V34_STAGE_BYTES 16384
#define SMEM_SMEM_V34_STRIDE 16384
#define SMEM_SMEM_V35_OFF 198656
#define SMEM_SMEM_V35_STAGE_BYTES 512
#define SMEM_SMEM_V35_STRIDE 512
#define SMEM_SMEM_V36_OFF 199168
#define SMEM_SMEM_V36_STAGE_BYTES 512
#define SMEM_SMEM_V36_STRIDE 512
#define SMEM_SMEM_V37_OFF 1024
#define SMEM_SMEM_V37_STAGE_BYTES 65536
#define SMEM_SMEM_V37_STRIDE 65536
#define SMEM_SMEM_V38_OFF 1280
#define SMEM_SMEM_V38_STAGE_BYTES 65280
#define SMEM_SMEM_V38_STRIDE 65280
#define SMEM_SMEM_V39_OFF 66560
#define SMEM_SMEM_V39_STAGE_BYTES 65536
#define SMEM_SMEM_V39_STRIDE 65536
#define SMEM_SMEM_V40_OFF 66816
#define SMEM_SMEM_V40_STAGE_BYTES 65280
#define SMEM_SMEM_V40_STRIDE 65280
#define SMEM_FLAT_VIEW_8_OFF 1024
#define SMEM_FLAT_VIEW_8_STAGE_BYTES 229376
#define SMEM_FLAT_VIEW_8_STRIDE 229376
#define SMEM_SMEM_V42_OFF 1024
#define SMEM_SMEM_V42_STAGE_BYTES 16384
#define SMEM_SMEM_V42_STRIDE 16384
#define SMEM_SMEM_V43_OFF 66560
#define SMEM_SMEM_V43_STAGE_BYTES 16384
#define SMEM_SMEM_V43_STRIDE 16384
#define SMEM_SMEM_V44_OFF 132096
#define SMEM_SMEM_V44_STAGE_BYTES 16384
#define SMEM_SMEM_V44_STRIDE 16384
#define SMEM_SMEM_V45_OFF 197632
#define SMEM_SMEM_V45_STAGE_BYTES 512
#define SMEM_SMEM_V45_STRIDE 512
#define SMEM_SMEM_V46_OFF 199680
#define SMEM_SMEM_V46_STAGE_BYTES 1024
#define SMEM_SMEM_V46_STRIDE 1024
#define SMEM_SMEM_V47_OFF 203776
#define SMEM_SMEM_V47_STAGE_BYTES 1024
#define SMEM_SMEM_V47_STRIDE 1024
#define SMEM_SMEM_V48_OFF 1024
#define SMEM_SMEM_V48_STAGE_BYTES 16384
#define SMEM_SMEM_V48_STRIDE 16384
#define SMEM_SMEM_V49_OFF 99328
#define SMEM_SMEM_V49_STAGE_BYTES 16384
#define SMEM_SMEM_V49_STRIDE 16384
#define SMEM_SMEM_V50_OFF 197632
#define SMEM_SMEM_V50_STAGE_BYTES 512
#define SMEM_SMEM_V50_STRIDE 512
#define SMEM_SMEM_V51_OFF 200704
#define SMEM_SMEM_V51_STAGE_BYTES 1024
#define SMEM_SMEM_V51_STRIDE 1024
#define SMEM_SMEM_V52_OFF 207872
#define SMEM_SMEM_V52_STAGE_BYTES 16384
#define SMEM_SMEM_V52_STRIDE 16384
#define SMEM_SMEM_V53_OFF 224256
#define SMEM_SMEM_V53_STAGE_BYTES 4096
#define SMEM_SMEM_V53_STRIDE 4096
#define SMEM_SMEM_V54_OFF 228352
#define SMEM_SMEM_V54_STAGE_BYTES 512
#define SMEM_SMEM_V54_STRIDE 512
#define SMEM_SMEM_V55_OFF 228864
#define SMEM_SMEM_V55_STAGE_BYTES 512
#define SMEM_SMEM_V55_STRIDE 512
#define SMEM_SMEM_V56_OFF 1024
#define SMEM_SMEM_V56_STAGE_BYTES 65536
#define SMEM_SMEM_V56_STRIDE 65536
#define SMEM_SMEM_V57_OFF 1024
#define SMEM_SMEM_V57_STAGE_BYTES 65536
#define SMEM_SMEM_V57_STRIDE 65536
#define SMEM_SMEM_V57_0_OFF 66560
#define SMEM_SMEM_V57_0_STAGE_BYTES 32768
#define SMEM_SMEM_V57_0_STRIDE 32768
#define SMEM_SMEM_V57_1_OFF 99328
#define SMEM_SMEM_V57_1_STAGE_BYTES 32768
#define SMEM_SMEM_V57_1_STRIDE 32768
#define SMEM_SMEM_V57_2_OFF 132096
#define SMEM_SMEM_V57_2_STAGE_BYTES 1024
#define SMEM_SMEM_V57_2_STRIDE 1024
#define SMEM_SMEM_V57_3_OFF 133120
#define SMEM_SMEM_V57_3_STAGE_BYTES 1024
#define SMEM_SMEM_V57_3_STRIDE 1024
#define SMEM_SMEM_V62_OFF 132096
#define SMEM_SMEM_V62_STAGE_BYTES 1024
#define SMEM_SMEM_V62_STRIDE 1024
#define SMEM_SMEM_V62_0_OFF 133120
#define SMEM_SMEM_V62_0_STAGE_BYTES 1024
#define SMEM_SMEM_V62_0_STRIDE 1024
#define SMEM_SMEM_V56_0_OFF 134144
#define SMEM_SMEM_V56_0_STAGE_BYTES 32768
#define SMEM_SMEM_V56_0_STRIDE 32768
#define SMEM_SMEM_V57_4_OFF 134144
#define SMEM_SMEM_V57_4_STAGE_BYTES 32768
#define SMEM_SMEM_V57_4_STRIDE 32768
#define SMEM_SMEM_V57_5_OFF 166912
#define SMEM_SMEM_V57_5_STAGE_BYTES 16384
#define SMEM_SMEM_V57_5_STRIDE 16384
#define SMEM_SMEM_V57_6_OFF 183296
#define SMEM_SMEM_V57_6_STAGE_BYTES 16384
#define SMEM_SMEM_V57_6_STRIDE 16384
#define SMEM_SMEM_V57_7_OFF 199680
#define SMEM_SMEM_V57_7_STAGE_BYTES 512
#define SMEM_SMEM_V57_7_STRIDE 512
#define SMEM_SMEM_V57_8_OFF 200192
#define SMEM_SMEM_V57_8_STAGE_BYTES 512
#define SMEM_SMEM_V57_8_STRIDE 512
#define SMEM_SMEM_V70_OFF 200704
#define SMEM_SMEM_V70_STAGE_BYTES 1024
#define SMEM_SMEM_V70_STRIDE 1024
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 704
#define LAUNCH_MIN_BLOCKS 1

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



__device__ __forceinline__ void tcgen05_mma_mxf8f6f4_bs_k64_cta2(
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
    return (x & 0x7FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mok_backward_mxfp8_fp32(const __grid_constant__ CUtensorMap dy_s, const __grid_constant__ CUtensorMap dg_s, const __grid_constant__ CUtensorMap du_s, const __grid_constant__ CUtensorMap dy_atb_s, const __grid_constant__ CUtensorMap dg_atb_s, const __grid_constant__ CUtensorMap du_atb_s, const __grid_constant__ CUtensorMap x_atb_s, const __grid_constant__ CUtensorMap h_atb_s, const __grid_constant__ CUtensorMap wg_s, const __grid_constant__ CUtensorMap wu_s, const __grid_constant__ CUtensorMap wd_s, const __grid_constant__ CUtensorMap dh_s, const __grid_constant__ CUtensorMap dh_r, const __grid_constant__ CUtensorMap dx_s, const __grid_constant__ CUtensorMap dx_r, const __grid_constant__ CUtensorMap gate_out_r, const __grid_constant__ CUtensorMap up_out_r, const __grid_constant__ CUtensorMap dwg_s, const __grid_constant__ CUtensorMap dwu_s, const __grid_constant__ CUtensorMap dwd_s, const __grid_constant__ CUtensorMap dwg_r, const __grid_constant__ CUtensorMap dwu_r, const __grid_constant__ CUtensorMap dwd_r, const __grid_constant__ CUtensorMap dh_sw_s, const __grid_constant__ CUtensorMap gate_sw_s, const __grid_constant__ CUtensorMap up_sw_s, const __grid_constant__ CUtensorMap dg_sw_s, const __grid_constant__ CUtensorMap du_sw_s, const __grid_constant__ CUtensorMap gate_sw_r, const __grid_constant__ CUtensorMap up_sw_r, const __grid_constant__ CUtensorMap dy_q_store, const __grid_constant__ CUtensorMap dy_sc_store, const __grid_constant__ CUtensorMap dy_t_store, const __grid_constant__ CUtensorMap dy_sc_t_store, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap dy_q, const __grid_constant__ CUtensorMap dy_sc, const __grid_constant__ CUtensorMap dy_t, const __grid_constant__ CUtensorMap dy_sc_t, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap x_t, const __grid_constant__ CUtensorMap x_sc_t, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap wd_t_q, const __grid_constant__ CUtensorMap wd_t_sc, const __grid_constant__ CUtensorMap wg_t_q, const __grid_constant__ CUtensorMap wg_t_sc, const __grid_constant__ CUtensorMap wu_t_q, const __grid_constant__ CUtensorMap wu_t_sc, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap h_q_store, const __grid_constant__ CUtensorMap h_sc_store, const __grid_constant__ CUtensorMap h_t_store, const __grid_constant__ CUtensorMap h_sc_t_store, const __grid_constant__ CUtensorMap h_t, const __grid_constant__ CUtensorMap h_sc_t, const __grid_constant__ CUtensorMap dh_tile, const __grid_constant__ CUtensorMap gate_tile, const __grid_constant__ CUtensorMap up_tile, const __grid_constant__ CUtensorMap gate_sc, const __grid_constant__ CUtensorMap up_sc, const __grid_constant__ CUtensorMap dg_store, const __grid_constant__ CUtensorMap dg_sc_store, const __grid_constant__ CUtensorMap du_store, const __grid_constant__ CUtensorMap du_sc_store, const __grid_constant__ CUtensorMap dg_t_store, const __grid_constant__ CUtensorMap dg_sc_t_store, const __grid_constant__ CUtensorMap du_t_store, const __grid_constant__ CUtensorMap du_sc_t_store, const __grid_constant__ CUtensorMap dg_q, const __grid_constant__ CUtensorMap dg_sc, const __grid_constant__ CUtensorMap du_q, const __grid_constant__ CUtensorMap du_sc, const __grid_constant__ CUtensorMap dg_t, const __grid_constant__ CUtensorMap dg_sc_t, const __grid_constant__ CUtensorMap du_t, const __grid_constant__ CUtensorMap du_sc_t, __nv_bfloat16* __restrict__ dx_routed_ptr, float* __restrict__ weights, float* __restrict__ partials, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ dy_peers, unsigned long long* __restrict__ dx_peers, unsigned long long* __restrict__ weight_peers, unsigned long long* __restrict__ dweight_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ dh_ready, int* __restrict__ dg_ready, int* __restrict__ dy_ready, int* __restrict__ dx_ready, int* __restrict__ replay_x, int* __restrict__ replay_gu, int* __restrict__ replay_h, int* __restrict__ buffers_done, int* __restrict__ weight_ready, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, CakeFastDivmod mini_size_div, CakeFastDivmod mini_bwd_div, CakeFastDivmod mini_replay_div, CakeFastDivmod replay_macro_tasks_div, CakeFastDivmod weight_tasks_div, CakeFastDivmod shared_wgrad_div, CakeFastDivmod routed_wgrad_div, CakeFastDivmod dh_col_blocks_div, CakeFastDivmod dx_col_blocks_div, CakeFastDivmod replay_col_blocks_div)
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
    #define scales_arrived_addr (mbar_base + 72)
    #define gemm_finished_addr (mbar_base + 104)
    #define scales_finished_addr (mbar_base + 136)
    #define output_arrived_addr (mbar_base + 168)
    #define output_finished_addr (mbar_base + 176)
    #define schedule_arrived_addr (mbar_base + 184)
    #define schedule_finished_addr (mbar_base + 192)
    #define drain_arrived_0_addr (mbar_base + 200)
    #define drain_arrived_1_addr (mbar_base + 208)
    #define drain_arrived_2_addr (mbar_base + 216)
    #define drain_arrived_3_addr (mbar_base + 224)
    #define drain_arrived_4_addr (mbar_base + 232)
    #define drain_arrived_5_addr (mbar_base + 240)
    #define drain_arrived_6_addr (mbar_base + 248)
    #define drain_arrived_7_addr (mbar_base + 256)
    #define drain_finished_0_addr (mbar_base + 264)
    #define drain_finished_1_addr (mbar_base + 272)
    #define drain_finished_2_addr (mbar_base + 280)
    #define drain_finished_3_addr (mbar_base + 288)
    #define drain_finished_4_addr (mbar_base + 296)
    #define drain_finished_5_addr (mbar_base + 304)
    #define drain_finished_6_addr (mbar_base + 312)
    #define drain_finished_7_addr (mbar_base + 320)
    #define dispatch_arrived_addr (mbar_base + 328)
    #define dispatch_arrived_hi_addr (mbar_base + 336)
    #define combine_arrived_addr (mbar_base + 344)
    #define decode_ready_addr (mbar_base + 400)
    #define decode_words_0_addr (mbar_base + 656)
    #define decode_words_1_addr (mbar_base + 672)
    #define decode_words_2_addr (mbar_base + 688)

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
    __nv_bfloat16* mn_view = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int mn_view_addr = smem + 1024;
    __nv_bfloat16* b_nt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int b_nt_addr = smem + 66560;
    __nv_bfloat16* mn_view_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int mn_view_addr_0 = smem + 66560;
    __nv_bfloat16* b_nt_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int b_nt_hi_addr = smem + 132096;
    __nv_bfloat16* mn_view_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int mn_view_addr_1 = smem + 132096;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 207872);
    const int d_smem_addr = smem + 207872;
    unsigned int* flat_view = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int flat_view_addr = smem + 207872;
    __nv_bfloat16* sw_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_dh_addr = smem + 1024;
    __nv_bfloat16* sw_gate = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int sw_gate_addr = smem + 66560;
    __nv_bfloat16* sw_up = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int sw_up_addr = smem + 132096;
    float* flat_view_0 = reinterpret_cast<float*>(smem_raw + 197632);
    const int flat_view_addr_0 = smem + 197632;
    unsigned int* flat_view_1 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_1 = smem + 1024;
    unsigned int* flat_view_2 = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int flat_view_addr_2 = smem + 66560;
    unsigned int* flat_view_3 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int flat_view_addr_3 = smem + 132096;
    __nv_bfloat16* replay_gate = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int replay_gate_addr = smem + 1024;
    __nv_bfloat16* replay_up = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int replay_up_addr = smem + 99328;
    __nv_bfloat16* flat_view_4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int flat_view_addr_4 = smem + 197632;
    unsigned int* flat_view_5 = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int flat_view_addr_5 = smem + 197632;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_v19_addr = smem + 1024;
    unsigned int* smem_v19_0 = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int smem_v19_addr_0 = smem + 17408;
    unsigned int* smem_v19_1 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int smem_v19_addr_1 = smem + 99328;
    unsigned int* smem_v19_2 = reinterpret_cast<unsigned int*>(smem_raw + 115712);
    const int smem_v19_addr_2 = smem + 115712;
    unsigned int* flat_view_6 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_6 = smem + 1024;
    float* flat_view_7 = reinterpret_cast<float*>(smem_raw + 199680);
    const int flat_view_addr_7 = smem + 199680;
    __nv_bfloat16* smem_v25 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v25_addr = smem + 1024;
    __nv_bfloat16* smem_v26 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v26_addr = smem + 1280;
    __nv_bfloat16* smem_v27 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_v27_addr = smem + 1536;
    __nv_bfloat16* smem_v28 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1792);
    const int smem_v28_addr = smem + 1792;
    unsigned int* smem_v29 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v29_addr = smem + 132096;
    unsigned int* smem_v30 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v30_addr = smem + 148480;
    unsigned int* smem_v31 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v31_addr = smem + 164864;
    unsigned int* smem_v32 = reinterpret_cast<unsigned int*>(smem_raw + 165376);
    const int smem_v32_addr = smem + 165376;
    unsigned int* smem_v33 = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_v33_addr = smem + 165888;
    unsigned int* smem_v34 = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int smem_v34_addr = smem + 182272;
    unsigned int* smem_v35 = reinterpret_cast<unsigned int*>(smem_raw + 198656);
    const int smem_v35_addr = smem + 198656;
    unsigned int* smem_v36 = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int smem_v36_addr = smem + 199168;
    __nv_bfloat16* smem_v37 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v37_addr = smem + 1024;
    __nv_bfloat16* smem_v38 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v38_addr = smem + 1280;
    __nv_bfloat16* smem_v39 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v39_addr = smem + 66560;
    __nv_bfloat16* smem_v40 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66816);
    const int smem_v40_addr = smem + 66816;
    __nv_bfloat16* flat_view_8 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr_8 = smem + 1024;
    uint8_t* smem_v42 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v42_addr = smem + 1024;
    uint8_t* smem_v43 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_v43_addr = smem + 66560;
    uint8_t* smem_v44 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v44_addr = smem + 132096;
    uint8_t* smem_v45 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v45_addr = smem + 197632;
    uint8_t* smem_v46 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v46_addr = smem + 199680;
    uint8_t* smem_v47 = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_v47_addr = smem + 203776;
    uint8_t* smem_v48 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v48_addr = smem + 1024;
    uint8_t* smem_v49 = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v49_addr = smem + 99328;
    uint8_t* smem_v50 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v50_addr = smem + 197632;
    uint8_t* smem_v51 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_v51_addr = smem + 200704;
    unsigned int* smem_v52 = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int smem_v52_addr = smem + 207872;
    unsigned int* smem_v53 = reinterpret_cast<unsigned int*>(smem_raw + 224256);
    const int smem_v53_addr = smem + 224256;
    unsigned int* smem_v54 = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_v54_addr = smem + 228352;
    unsigned int* smem_v55 = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_v55_addr = smem + 228864;
    __nv_bfloat16* smem_v56 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v56_addr = smem + 1024;
    unsigned int* smem_v57 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_v57_addr = smem + 1024;
    unsigned int* smem_v57_0 = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int smem_v57_addr_0 = smem + 66560;
    unsigned int* smem_v57_1 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int smem_v57_addr_1 = smem + 99328;
    unsigned int* smem_v57_2 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v57_addr_2 = smem + 132096;
    unsigned int* smem_v57_3 = reinterpret_cast<unsigned int*>(smem_raw + 133120);
    const int smem_v57_addr_3 = smem + 133120;
    uint16_t* smem_v62 = reinterpret_cast<uint16_t*>(smem_raw + 132096);
    const int smem_v62_addr = smem + 132096;
    uint16_t* smem_v62_0 = reinterpret_cast<uint16_t*>(smem_raw + 133120);
    const int smem_v62_addr_0 = smem + 133120;
    __nv_bfloat16* smem_v56_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 134144);
    const int smem_v56_addr_0 = smem + 134144;
    unsigned int* smem_v57_4 = reinterpret_cast<unsigned int*>(smem_raw + 134144);
    const int smem_v57_addr_4 = smem + 134144;
    unsigned int* smem_v57_5 = reinterpret_cast<unsigned int*>(smem_raw + 166912);
    const int smem_v57_addr_5 = smem + 166912;
    unsigned int* smem_v57_6 = reinterpret_cast<unsigned int*>(smem_raw + 183296);
    const int smem_v57_addr_6 = smem + 183296;
    unsigned int* smem_v57_7 = reinterpret_cast<unsigned int*>(smem_raw + 199680);
    const int smem_v57_addr_7 = smem + 199680;
    unsigned int* smem_v57_8 = reinterpret_cast<unsigned int*>(smem_raw + 200192);
    const int smem_v57_addr_8 = smem + 200192;
    float* smem_v70 = reinterpret_cast<float*>(smem_raw + 200704);
    const int smem_v70_addr = smem + 200704;
    int tokens = num_tokens[0];
    int i_tiles = intermediate / 128;
    int shared_down = local_tokens / 256 * ((intermediate + 511) / 512);
    int shared_swiglu = (local_tokens / 128 * (intermediate / 128) + 3) / 4;
    int shared_dx = local_tokens / 256 * ((hidden + 511) / 512);
    int shared_wgrad = ((intermediate / 256 * ((hidden + 512 - 1) / 512)) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (intermediate / 256 * ((hidden + 512 - 1) / 512)) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
    int routed_wgrad = ((intermediate / 256 * ((hidden + 512 - 1) / 512)) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (intermediate / 256 * ((hidden + 512 - 1) / 512)) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
    int shared_tasks = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
    int mini_down = mini_size / 256 * ((intermediate + 512 - 1) / 512);
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 3) / 4;
    int mini_dx = mini_size / 256 * ((hidden + 512 - 1) / 512);
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
    int weight_tasks = experts * routed_wgrad;
    int saved_tasks = saved_minis * mini_bwd + 3 * weight_tasks;
    int replay_macro_tasks = macro_size / mini_size * (mini_replay + mini_bwd) + 3 * weight_tasks;
    int comm_clusters = comm_sms / 2;
    int true_clusters = comm_clusters + true_compute;
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (33 pipeline groups, 0 ordered-sequence groups, 57 barriers)
    // Mbarriers at smem_raw[0..704)

    if (threadIdx.x == 0) {
        // replay_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        // swiglu_arrived: 2 barriers, init_count=1
        mbarrier_init(smem + 24, 1);
        mbarrier_init(smem + 32, 1);
        // gemm_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        mbarrier_init(smem + 56, 1);
        mbarrier_init(smem + 64, 1);
        // scales_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 72, 1);
        mbarrier_init(smem + 80, 1);
        mbarrier_init(smem + 88, 1);
        mbarrier_init(smem + 96, 1);
        // gemm_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 104, 1);
        mbarrier_init(smem + 112, 1);
        mbarrier_init(smem + 120, 1);
        mbarrier_init(smem + 128, 1);
        // scales_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        mbarrier_init(smem + 144, 1);
        mbarrier_init(smem + 152, 1);
        mbarrier_init(smem + 160, 1);
        // output_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        // output_finished: 1 barriers, init_count=2
        mbarrier_init(smem + 176, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 184, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 192, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 200, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 208, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 216, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 224, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 232, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 240, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 248, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 256, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 264, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 272, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 280, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 288, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 296, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 304, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 312, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 320, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 328, 1);
        // dispatch_arrived_hi: 1 barriers, init_count=1
        mbarrier_init(smem + 336, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 344, 1);
        mbarrier_init(smem + 352, 1);
        mbarrier_init(smem + 360, 1);
        mbarrier_init(smem + 368, 1);
        mbarrier_init(smem + 376, 1);
        mbarrier_init(smem + 384, 1);
        mbarrier_init(smem + 392, 1);
        // decode_ready: 1 barriers, init_count=1
        mbarrier_init(smem + 400, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (576 columns, 572 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 704);
    if (warp == 0) {
        int _tmem_hold = smem + 704;
        asm volatile("tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(576) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_sfa = taddr + 512;
    const int tmem_tmem_sfb = taddr + 524;
    const int tmem_tmem_sfb_hi = taddr + 548;
    const int tmem_accumulator = taddr;
    unsigned int taddr_1 = reinterpret_cast<const volatile unsigned int*>(reinterpret_cast<uint8_t*>(smem_raw) + CAKE_TMEM_HOLD_OFFSET)[0];
    int cta_rank_0 = cta_rank;
    int cluster = bid / 2;
    int gemm_deep = 0;
    int gemm_wide = 0;
    int gemm_layout = 0;
    int gemm_out0 = 0;
    int gemm_out1 = 0;
    int gemm_acc_tiles = 0;
    int scales_wide = 0;
    int swiglu_seq = 0;
    int replay_seq = 0;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int dispatch_seq = 0;
        int dispatch_seq_hi = 0;
        int combine_seq_0 = 0;
        int combine_seq_1 = 0;
        int combine_seq_2 = 0;
        int combine_seq_3 = 0;
        int combine_seq_4 = 0;
        int combine_seq_5 = 0;
        int combine_seq_6 = 0;
        if (tid == 0) {
        }
        __syncthreads();
        int offset = 0;
        int rows = ((macro_size) < (tokens - offset) ? (macro_size) : (tokens - offset));
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
            int poll_first;
            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_first) : "l"(weight_ready + 0) : "memory");
            int poll = poll_first;
            while (poll < comm_sms) {
                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                int poll_next;
                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_next) : "l"(weight_ready + 0) : "memory");
                poll = poll_next;
            }
            asm volatile("fence.acquire.gpu;" ::: "memory");
        }
        __syncthreads();
        int col_blocks = (hidden + 511) / 512;
        int macro_offset = 0;
        int macro_tokens = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
        int tasks = macro_tokens / 128 * col_blocks;
        int task = cluster * 2 + cta_rank_0;
        if (task < tasks) {
            int min_1 = ((256) < (hidden - task % col_blocks * 512) ? (256) : (hidden - task % col_blocks * 512));
            int peer_1 = -1;
            int peer_token = -1;
            if (tid < 128) {
                peer_1 = schedule_rank[macro_offset + task / col_blocks * 128 + tid];
                peer_token = schedule_token[macro_offset + task / col_blocks * 128 + tid];
            }
            uint32_t cta_count = __syncthreads_count(peer_1 >= 0);
            if (tid == 0) {
                mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count * (unsigned int)(min_1 * 2));
            }
            __syncthreads();
            if (peer_1 >= 0) {
                cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(task % col_blocks * 512)) * (unsigned long long)2)), min_1 * 2, dispatch_arrived_addr);
            } else if (tid < 128) {
                #pragma unroll
                for (int vec = 0; vec < 32; vec++) {
                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (tid * 512 + vec * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                }
            }
        }
        while (task < tasks) {
            int row_1 = task / col_blocks * 128;
            int col_block = task % col_blocks;
            int chunk_cols = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
            int lo_cols = ((256) < (chunk_cols) ? (256) : (chunk_cols));
            int hi_cols = chunk_cols - lo_cols;
            if (hi_cols > 0) {
                int peer_2 = -1;
                int peer_token_1 = -1;
                if (tid < 128) {
                    peer_2 = schedule_rank[macro_offset + row_1 + tid];
                    peer_token_1 = schedule_token[macro_offset + row_1 + tid];
                }
                uint32_t cta_count_1 = __syncthreads_count(peer_2 >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(dispatch_arrived_hi_addr, cta_count_1 * (unsigned int)(hi_cols * 2));
                }
                __syncthreads();
                if (peer_2 >= 0) {
                    cp_async_bulk_gmem2smem(smem_v39_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_2])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512) + 256) * (unsigned long long)2)), hi_cols * 2, dispatch_arrived_hi_addr);
                } else if (tid < 128) {
                    #pragma unroll
                    for (int vec_1 = 0; vec_1 < 32; vec_1++) {
                        asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (65536 + tid * 512 + vec_1 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                    }
                }
            }
            int weight_peer = -1;
            if (tid < 128) {
                weight_peer = schedule_rank[macro_offset + row_1 + tid];
                flat_view_7[tid] = ((weight_peer >= 0) ? weights[row_1 + tid] : 0.0f);
            }
            int _parity_dispatch_arrived = (int)((unsigned int)dispatch_seq % 2);
            mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived);
            dispatch_seq += 1;
            int row_tile = row_1 / 128;
            int k_tiles = hidden / 128;
            int macro_tiles = macro_size / 128;
            if (lo_cols / 128 > 0) {
                if (tid == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncthreads();
                if (tid < 128) {
                    int r = tid % 64;
                    int row0 = r * 2;
                    int row1 = row0 + 1;
                    int half = tid / 64;
                    int word_stride = 128;
                    int scale_base0 = (row0 % 32 * 4 + row0 / 32) * 4;
                    int scale_base1 = (row1 % 32 * 4 + row1 / 32) * 4;
                    #pragma unroll 1
                    for (int j = 0; j < 2; j++) {
                        int k_block = (half * 2 + j + tid / 8) % 4;
                        unsigned int pairs0[16];
                        unsigned int pairs1[16];
                        #pragma unroll
                        for (int k = 0; k < 16; k++) {
                            int col = k_block * 32 + (tid * 4 + k * 2) % 32;
                            unsigned int w0 = flat_view_6[col * word_stride + r];
                            unsigned int w1 = flat_view_6[(col + 1) * word_stride + r];
                            uint32_t lo;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo) : "r"(w0), "r"(w1));
                            uint32_t hi;
                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi) : "r"(w0), "r"(w1));
                            float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(lo));
                            pairs0[k] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32.x * flat_view_7[col], cvt_f32.y * flat_view_7[col + 1])));
                            float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(hi));
                            pairs1[k] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1.x * flat_view_7[col], cvt_f32_1.y * flat_view_7[col + 1])));
                        }
                        uint32_t bf16x2_abs;
                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs) : "r"(pairs0[0]));
                        unsigned int amax_pair = bf16x2_abs;
                        #pragma unroll
                        for (int i = 1; i < 16; i++) {
                            uint32_t bf16x2_abs_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1) : "r"(pairs0[i]));
                            uint32_t bf16x2_max;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max) : "r"(amax_pair), "r"(bf16x2_abs_1));
                            amax_pair = bf16x2_max;
                        }
                        uint16_t bf16_max;
                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max) : "h"((uint16_t)(amax_pair & 65535)), "h"((uint16_t)(amax_pair >> 16)));
                        float amax;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax) : "h"((uint16_t)(bf16_max)));
                        float scale = fmaxf(amax * 0.002232143f, 1e-12f);
                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_0;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_0) : "f"(scale), "f"(scale));
                        unsigned int scale_byte = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_0 & 255;
                        unsigned int inverse_lane = 254 - scale_byte << 7;
                        unsigned int inverse = inverse_lane | inverse_lane << 16;
                        unsigned int words[8];
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 8; i_1++) {
                            uint32_t bf16x2_mul;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(pairs0[i_1 * 2]), "r"(inverse));
                            uint16_t e4m3x2_bf16x2;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2) : "r"(bf16x2_mul));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t bf16x2_mul_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(pairs0[i_1 * 2 + 1]), "r"(inverse));
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_1;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_1) : "r"(bf16x2_mul_1));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            words[i_1] = (unsigned int)e4m3x2_bf16x2 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_1 << 16;
                        }
                        uint32_t bf16x2_abs_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_0) : "r"(pairs1[0]));
                        unsigned int amax_pair_0 = bf16x2_abs_0;
                        #pragma unroll
                        for (int i_2 = 1; i_2 < 16; i_2++) {
                            uint32_t bf16x2_abs_1_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_1) : "r"(pairs1[i_2]));
                            uint32_t bf16x2_max_1;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_1) : "r"(amax_pair_0), "r"(bf16x2_abs_1_1));
                            amax_pair_0 = bf16x2_max_1;
                        }
                        uint16_t bf16_max_0;
                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_0) : "h"((uint16_t)(amax_pair_0 & 65535)), "h"((uint16_t)(amax_pair_0 >> 16)));
                        float amax_0;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_0) : "h"((uint16_t)(bf16_max_0)));
                        float scale_0 = fmaxf(amax_0 * 0.002232143f, 1e-12f);
                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_2;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_2) : "f"(scale_0), "f"(scale_0));
                        unsigned int scale_byte_0 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_2 & 255;
                        unsigned int inverse_lane_0 = 254 - scale_byte_0 << 7;
                        unsigned int inverse_0 = inverse_lane_0 | inverse_lane_0 << 16;
                        unsigned int words_0[8];
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 8; i_3++) {
                            uint32_t bf16x2_mul_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(pairs1[i_3 * 2]), "r"(inverse_0));
                            uint16_t e4m3x2_bf16x2_1;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_1) : "r"(bf16x2_mul_2));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_2);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t bf16x2_mul_1_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(pairs1[i_3 * 2 + 1]), "r"(inverse_0));
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_3;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "r"(bf16x2_mul_1_1));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            words_0[i_3] = (unsigned int)e4m3x2_bf16x2_1 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_3 << 16;
                        }
                        #pragma unroll
                        for (int k_1 = 0; k_1 < 8; k_1++) {
                            int col_1 = k_block * 32 + (tid * 4 + k_1 * 4) % 32;
                            smem_v33[(row0 * 128 + col_1) / 4] = words[k_1];
                            smem_v33[(row1 * 128 + col_1) / 4] = words_0[k_1];
                        }
                        {
                            uint32_t _byte_4 = static_cast<uint32_t>(scale_byte) & 0xFFu;
                            uint32_t _addr_4 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base0 + k_block));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_4), "r"(_byte_4) : "memory");
                        }
                        {
                            uint32_t _byte_5 = static_cast<uint32_t>(scale_byte_0) & 0xFFu;
                            uint32_t _addr_5 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base1 + k_block));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_5), "r"(_byte_5) : "memory");
                        }
                    }
                } else {
                    int row_0 = tid - 128;
                    unsigned int scale_word = 0;
                    #pragma unroll 1
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        int k_block_1 = (j_1 + (tid - 128) / 8) % 4;
                        unsigned int pairs[16];
                        #pragma unroll
                        for (int k_2 = 0; k_2 < 16; k_2++) {
                            int col_2 = k_block_1 * 32 + ((tid - 128) * 4 + k_2 * 2) % 32;
                            float x0 = 0.0f;
                            float x1 = 0.0f;
                            float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[(row_0 * 256 + col_2) / 2]));
                            pairs[k_2] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_2.x * flat_view_7[row_0], cvt_f32_2.y * flat_view_7[row_0])));
                        }
                        uint32_t bf16x2_abs_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_2) : "r"(pairs[0]));
                        unsigned int amax_pair_1 = bf16x2_abs_2;
                        #pragma unroll
                        for (int i_4 = 1; i_4 < 16; i_4++) {
                            uint32_t bf16x2_abs_1_2;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_2) : "r"(pairs[i_4]));
                            uint32_t bf16x2_max_2;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_2) : "r"(amax_pair_1), "r"(bf16x2_abs_1_2));
                            amax_pair_1 = bf16x2_max_2;
                        }
                        uint16_t bf16_max_1;
                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_1) : "h"((uint16_t)(amax_pair_1 & 65535)), "h"((uint16_t)(amax_pair_1 >> 16)));
                        float amax_1;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1) : "h"((uint16_t)(bf16_max_1)));
                        float scale_1 = fmaxf(amax_1 * 0.002232143f, 1e-12f);
                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_6;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_6) : "f"(scale_1), "f"(scale_1));
                        unsigned int scale_byte_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_6 & 255;
                        unsigned int inverse_lane_1 = 254 - scale_byte_1 << 7;
                        unsigned int inverse_1 = inverse_lane_1 | inverse_lane_1 << 16;
                        unsigned int words_1[8];
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 8; i_5++) {
                            uint32_t bf16x2_mul_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(pairs[i_5 * 2]), "r"(inverse_1));
                            uint16_t e4m3x2_bf16x2_2;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_2) : "r"(bf16x2_mul_3));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_3);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t bf16x2_mul_1_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(pairs[i_5 * 2 + 1]), "r"(inverse_1));
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_7;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_7) : "r"(bf16x2_mul_1_2));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            words_1[i_5] = (unsigned int)e4m3x2_bf16x2_2 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_7 << 16;
                        }
                        scale_word = scale_word | scale_byte_1 << (unsigned int)(k_block_1 * 8);
                        #pragma unroll
                        for (int k_3 = 0; k_3 < 8; k_3++) {
                            int col_3 = k_block_1 * 32 + ((tid - 128) * 4 + k_3 * 4) % 32;
                            smem_v29[(row_0 * 128 + col_3) / 4] = words_1[k_3];
                        }
                    }
                    smem_v31[row_0 % 32 * 4 + row_0 / 32] = scale_word;
                }
                __syncthreads();
                if (tid == 0) {
                    int col_tile_0 = col_block * 4;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d((&dy_q_store), col_tile_0 * 128, row_1, smem_v29_addr);
                    tma_store_3d((&dy_sc_store), 0, 0, row_tile * k_tiles + col_tile_0, smem_v31_addr);
                    tma_store_2d((&dy_t_store), row_1, col_tile_0 * 128, smem_v33_addr);
                    tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_0 * macro_tiles + row_tile, smem_v35_addr);
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            if (lo_cols / 128 > 1) {
                if (tid == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                __syncthreads();
                if (tid < 128) {
                    int r_1 = tid % 64;
                    int row0_1 = r_1 * 2;
                    int row1_1 = row0_1 + 1;
                    int half_1 = tid / 64;
                    int word_stride_1 = 128;
                    int scale_base0_1 = (row0_1 % 32 * 4 + row0_1 / 32) * 4;
                    int scale_base1_1 = (row1_1 % 32 * 4 + row1_1 / 32) * 4;
                    #pragma unroll 1
                    for (int j_2 = 0; j_2 < 2; j_2++) {
                        int k_block_2 = (half_1 * 2 + j_2 + tid / 8) % 4;
                        unsigned int pairs0_1[16];
                        unsigned int pairs1_1[16];
                        #pragma unroll
                        for (int k_4 = 0; k_4 < 16; k_4++) {
                            int col_4 = k_block_2 * 32 + (tid * 4 + k_4 * 2) % 32;
                            unsigned int w0_1 = flat_view_6[64 + col_4 * word_stride_1 + r_1];
                            unsigned int w1_1 = flat_view_6[64 + (col_4 + 1) * word_stride_1 + r_1];
                            uint32_t lo_1;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_1) : "r"(w0_1), "r"(w1_1));
                            uint32_t hi_1;
                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_1) : "r"(w0_1), "r"(w1_1));
                            float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_1));
                            pairs0_1[k_4] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_3.x * flat_view_7[col_4], cvt_f32_3.y * flat_view_7[col_4 + 1])));
                            float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_1));
                            pairs1_1[k_4] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_1.x * flat_view_7[col_4], cvt_f32_1_1.y * flat_view_7[col_4 + 1])));
                        }
                        uint32_t bf16x2_abs_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_3) : "r"(pairs0_1[0]));
                        unsigned int amax_pair_2 = bf16x2_abs_3;
                        #pragma unroll
                        for (int i_6 = 1; i_6 < 16; i_6++) {
                            uint32_t bf16x2_abs_1_3;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_3) : "r"(pairs0_1[i_6]));
                            uint32_t bf16x2_max_3;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_3) : "r"(amax_pair_2), "r"(bf16x2_abs_1_3));
                            amax_pair_2 = bf16x2_max_3;
                        }
                        uint16_t bf16_max_2;
                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_2) : "h"((uint16_t)(amax_pair_2 & 65535)), "h"((uint16_t)(amax_pair_2 >> 16)));
                        float amax_2;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2) : "h"((uint16_t)(bf16_max_2)));
                        float scale_2 = fmaxf(amax_2 * 0.002232143f, 1e-12f);
                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_8;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_8) : "f"(scale_2), "f"(scale_2));
                        unsigned int scale_byte_2 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_8 & 255;
                        unsigned int inverse_lane_2 = 254 - scale_byte_2 << 7;
                        unsigned int inverse_2 = inverse_lane_2 | inverse_lane_2 << 16;
                        unsigned int words_2[8];
                        #pragma unroll
                        for (int i_7 = 0; i_7 < 8; i_7++) {
                            uint32_t bf16x2_mul_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(pairs0_1[i_7 * 2]), "r"(inverse_2));
                            uint16_t e4m3x2_bf16x2_3;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_3) : "r"(bf16x2_mul_4));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_4);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t bf16x2_mul_1_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(pairs0_1[i_7 * 2 + 1]), "r"(inverse_2));
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_9;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_9) : "r"(bf16x2_mul_1_3));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            words_2[i_7] = (unsigned int)e4m3x2_bf16x2_3 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_9 << 16;
                        }
                        uint32_t bf16x2_abs_1_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_4) : "r"(pairs1_1[0]));
                        unsigned int amax_pair_1_1 = bf16x2_abs_1_4;
                        #pragma unroll
                        for (int i_8 = 1; i_8 < 16; i_8++) {
                            uint32_t bf16x2_abs_1_0;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_0) : "r"(pairs1_1[i_8]));
                            uint32_t bf16x2_max_4;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_4) : "r"(amax_pair_1_1), "r"(bf16x2_abs_1_0));
                            amax_pair_1_1 = bf16x2_max_4;
                        }
                        uint16_t bf16_max_1_1;
                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_1_1) : "h"((uint16_t)(amax_pair_1_1 & 65535)), "h"((uint16_t)(amax_pair_1_1 >> 16)));
                        float amax_1_1;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_1) : "h"((uint16_t)(bf16_max_1_1)));
                        float scale_1_1 = fmaxf(amax_1_1 * 0.002232143f, 1e-12f);
                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_10;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_10) : "f"(scale_1_1), "f"(scale_1_1));
                        unsigned int scale_byte_1_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_10 & 255;
                        unsigned int inverse_lane_1_1 = 254 - scale_byte_1_1 << 7;
                        unsigned int inverse_1_1 = inverse_lane_1_1 | inverse_lane_1_1 << 16;
                        unsigned int words_1_1[8];
                        #pragma unroll
                        for (int i_9 = 0; i_9 < 8; i_9++) {
                            uint32_t bf16x2_mul_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(pairs1_1[i_9 * 2]), "r"(inverse_1_1));
                            uint16_t e4m3x2_bf16x2_4;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_4) : "r"(bf16x2_mul_5));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_5);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t bf16x2_mul_1_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(pairs1_1[i_9 * 2 + 1]), "r"(inverse_1_1));
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_11;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_11) : "r"(bf16x2_mul_1_4));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            words_1_1[i_9] = (unsigned int)e4m3x2_bf16x2_4 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_11 << 16;
                        }
                        #pragma unroll
                        for (int k_5 = 0; k_5 < 8; k_5++) {
                            int col_5 = k_block_2 * 32 + (tid * 4 + k_5 * 4) % 32;
                            smem_v34[(row0_1 * 128 + col_5) / 4] = words_2[k_5];
                            smem_v34[(row1_1 * 128 + col_5) / 4] = words_1_1[k_5];
                        }
                        {
                            uint32_t _byte_12 = static_cast<uint32_t>(scale_byte_2) & 0xFFu;
                            uint32_t _addr_12 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base0_1 + k_block_2));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_12), "r"(_byte_12) : "memory");
                        }
                        {
                            uint32_t _byte_13 = static_cast<uint32_t>(scale_byte_1_1) & 0xFFu;
                            uint32_t _addr_13 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base1_1 + k_block_2));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_13), "r"(_byte_13) : "memory");
                        }
                    }
                } else {
                    int row_1_1 = tid - 128;
                    unsigned int scale_word_1 = 0;
                    #pragma unroll 1
                    for (int j_3 = 0; j_3 < 4; j_3++) {
                        int k_block_3 = (j_3 + (tid - 128) / 8) % 4;
                        unsigned int pairs_1[16];
                        #pragma unroll
                        for (int k_6 = 0; k_6 < 16; k_6++) {
                            int col_6 = k_block_3 * 32 + ((tid - 128) * 4 + k_6 * 2) % 32;
                            float x0_1 = 0.0f;
                            float x1_1 = 0.0f;
                            float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[64 + (row_1_1 * 256 + col_6) / 2]));
                            pairs_1[k_6] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_4.x * flat_view_7[row_1_1], cvt_f32_4.y * flat_view_7[row_1_1])));
                        }
                        uint32_t bf16x2_abs_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_4) : "r"(pairs_1[0]));
                        unsigned int amax_pair_3 = bf16x2_abs_4;
                        #pragma unroll
                        for (int i_10 = 1; i_10 < 16; i_10++) {
                            uint32_t bf16x2_abs_1_5;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_5) : "r"(pairs_1[i_10]));
                            uint32_t bf16x2_max_5;
                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_5) : "r"(amax_pair_3), "r"(bf16x2_abs_1_5));
                            amax_pair_3 = bf16x2_max_5;
                        }
                        uint16_t bf16_max_3;
                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_3) : "h"((uint16_t)(amax_pair_3 & 65535)), "h"((uint16_t)(amax_pair_3 >> 16)));
                        float amax_3;
                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_3) : "h"((uint16_t)(bf16_max_3)));
                        float scale_3 = fmaxf(amax_3 * 0.002232143f, 1e-12f);
                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_14;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_14) : "f"(scale_3), "f"(scale_3));
                        unsigned int scale_byte_3 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_14 & 255;
                        unsigned int inverse_lane_3 = 254 - scale_byte_3 << 7;
                        unsigned int inverse_3 = inverse_lane_3 | inverse_lane_3 << 16;
                        unsigned int words_3[8];
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 8; i_11++) {
                            uint32_t bf16x2_mul_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(pairs_1[i_11 * 2]), "r"(inverse_3));
                            uint16_t e4m3x2_bf16x2_5;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_5) : "r"(bf16x2_mul_6));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_6);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t bf16x2_mul_1_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(pairs_1[i_11 * 2 + 1]), "r"(inverse_3));
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_15;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_15) : "r"(bf16x2_mul_1_5));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            words_3[i_11] = (unsigned int)e4m3x2_bf16x2_5 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_15 << 16;
                        }
                        scale_word_1 = scale_word_1 | scale_byte_3 << (unsigned int)(k_block_3 * 8);
                        #pragma unroll
                        for (int k_7 = 0; k_7 < 8; k_7++) {
                            int col_7 = k_block_3 * 32 + ((tid - 128) * 4 + k_7 * 4) % 32;
                            smem_v30[(row_1_1 * 128 + col_7) / 4] = words_3[k_7];
                        }
                    }
                    smem_v32[row_1_1 % 32 * 4 + row_1_1 / 32] = scale_word_1;
                }
                __syncthreads();
                if (tid == 0) {
                    int col_tile_1 = col_block * 4 + 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d((&dy_q_store), col_tile_1 * 128, row_1, smem_v30_addr);
                    tma_store_3d((&dy_sc_store), 0, 0, row_tile * k_tiles + col_tile_1, smem_v32_addr);
                    tma_store_2d((&dy_t_store), row_1, col_tile_1 * 128, smem_v34_addr);
                    tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_1 * macro_tiles + row_tile, smem_v36_addr);
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            int next_task = task + comm_sms;
            if (next_task < tasks) {
                int min_2 = ((256) < (hidden - next_task % col_blocks * 512) ? (256) : (hidden - next_task % col_blocks * 512));
                int peer_3 = -1;
                int peer_token_2 = -1;
                if (tid < 128) {
                    peer_3 = schedule_rank[macro_offset + next_task / col_blocks * 128 + tid];
                    peer_token_2 = schedule_token[macro_offset + next_task / col_blocks * 128 + tid];
                }
                uint32_t cta_count_2 = __syncthreads_count(peer_3 >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_2 * (unsigned int)(min_2 * 2));
                }
                __syncthreads();
                if (peer_3 >= 0) {
                    cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_3])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task % col_blocks * 512)) * (unsigned long long)2)), min_2 * 2, dispatch_arrived_addr);
                } else if (tid < 128) {
                    #pragma unroll
                    for (int vec_2 = 0; vec_2 < 32; vec_2++) {
                        asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (tid * 512 + vec_2 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                    }
                }
            }
            if (hi_cols > 0) {
                int _parity_dispatch_arrived_hi = (int)((unsigned int)dispatch_seq_hi % 2);
                mbarrier_wait(dispatch_arrived_hi_addr, _parity_dispatch_arrived_hi);
                dispatch_seq_hi += 1;
                int row_tile_0 = row_1 / 128;
                int k_tiles_0 = hidden / 128;
                int macro_tiles_0 = macro_size / 128;
                if (hi_cols / 128 > 0) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncthreads();
                    if (tid < 128) {
                        int r_2 = tid % 64;
                        int row0_2 = r_2 * 2;
                        int row1_2 = row0_2 + 1;
                        int half_2 = tid / 64;
                        int word_stride_2 = 128;
                        int scale_base0_2 = (row0_2 % 32 * 4 + row0_2 / 32) * 4;
                        int scale_base1_2 = (row1_2 % 32 * 4 + row1_2 / 32) * 4;
                        #pragma unroll 1
                        for (int j_4 = 0; j_4 < 2; j_4++) {
                            int k_block_4 = (half_2 * 2 + j_4 + tid / 8) % 4;
                            unsigned int pairs0_2[16];
                            unsigned int pairs1_2[16];
                            #pragma unroll
                            for (int k_8 = 0; k_8 < 16; k_8++) {
                                int col_8 = k_block_4 * 32 + (tid * 4 + k_8 * 2) % 32;
                                unsigned int w0_2 = flat_view_6[16384 + col_8 * word_stride_2 + r_2];
                                unsigned int w1_2 = flat_view_6[16384 + (col_8 + 1) * word_stride_2 + r_2];
                                uint32_t lo_2;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_2) : "r"(w0_2), "r"(w1_2));
                                uint32_t hi_2;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_2) : "r"(w0_2), "r"(w1_2));
                                float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_2));
                                pairs0_2[k_8] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_5.x * flat_view_7[col_8], cvt_f32_5.y * flat_view_7[col_8 + 1])));
                                float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_2));
                                pairs1_2[k_8] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_2.x * flat_view_7[col_8], cvt_f32_1_2.y * flat_view_7[col_8 + 1])));
                            }
                            uint32_t bf16x2_abs_5;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_5) : "r"(pairs0_2[0]));
                            unsigned int amax_pair_4 = bf16x2_abs_5;
                            #pragma unroll
                            for (int i_12 = 1; i_12 < 16; i_12++) {
                                uint32_t bf16x2_abs_1_6;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_6) : "r"(pairs0_2[i_12]));
                                uint32_t bf16x2_max_6;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_6) : "r"(amax_pair_4), "r"(bf16x2_abs_1_6));
                                amax_pair_4 = bf16x2_max_6;
                            }
                            uint16_t bf16_max_4;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_4) : "h"((uint16_t)(amax_pair_4 & 65535)), "h"((uint16_t)(amax_pair_4 >> 16)));
                            float amax_4;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_4) : "h"((uint16_t)(bf16_max_4)));
                            float scale_4 = fmaxf(amax_4 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_16;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_16) : "f"(scale_4), "f"(scale_4));
                            unsigned int scale_byte_4 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_16 & 255;
                            unsigned int inverse_lane_4 = 254 - scale_byte_4 << 7;
                            unsigned int inverse_4 = inverse_lane_4 | inverse_lane_4 << 16;
                            unsigned int words_4[8];
                            #pragma unroll
                            for (int i_13 = 0; i_13 < 8; i_13++) {
                                uint32_t bf16x2_mul_7;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(pairs0_2[i_13 * 2]), "r"(inverse_4));
                                uint16_t e4m3x2_bf16x2_6;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_6) : "r"(bf16x2_mul_7));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_7);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_6;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(pairs0_2[i_13 * 2 + 1]), "r"(inverse_4));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_17;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_17) : "r"(bf16x2_mul_1_6));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_4[i_13] = (unsigned int)e4m3x2_bf16x2_6 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_17 << 16;
                            }
                            uint32_t bf16x2_abs_2_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_2_1) : "r"(pairs1_2[0]));
                            unsigned int amax_pair_2_1 = bf16x2_abs_2_1;
                            #pragma unroll
                            for (int i_14 = 1; i_14 < 16; i_14++) {
                                uint32_t bf16x2_abs_1_7;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_7) : "r"(pairs1_2[i_14]));
                                uint32_t bf16x2_max_7;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_7) : "r"(amax_pair_2_1), "r"(bf16x2_abs_1_7));
                                amax_pair_2_1 = bf16x2_max_7;
                            }
                            uint16_t bf16_max_2_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_2_1) : "h"((uint16_t)(amax_pair_2_1 & 65535)), "h"((uint16_t)(amax_pair_2_1 >> 16)));
                            float amax_2_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2_1) : "h"((uint16_t)(bf16_max_2_1)));
                            float scale_2_1 = fmaxf(amax_2_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_18;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_18) : "f"(scale_2_1), "f"(scale_2_1));
                            unsigned int scale_byte_2_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_18 & 255;
                            unsigned int inverse_lane_2_1 = 254 - scale_byte_2_1 << 7;
                            unsigned int inverse_2_1 = inverse_lane_2_1 | inverse_lane_2_1 << 16;
                            unsigned int words_2_1[8];
                            #pragma unroll
                            for (int i_15 = 0; i_15 < 8; i_15++) {
                                uint32_t bf16x2_mul_8;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(pairs1_2[i_15 * 2]), "r"(inverse_2_1));
                                uint16_t e4m3x2_bf16x2_7;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_7) : "r"(bf16x2_mul_8));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_8);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_7;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(pairs1_2[i_15 * 2 + 1]), "r"(inverse_2_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_19;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "r"(bf16x2_mul_1_7));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_2_1[i_15] = (unsigned int)e4m3x2_bf16x2_7 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_19 << 16;
                            }
                            #pragma unroll
                            for (int k_9 = 0; k_9 < 8; k_9++) {
                                int col_9 = k_block_4 * 32 + (tid * 4 + k_9 * 4) % 32;
                                smem_v33[(row0_2 * 128 + col_9) / 4] = words_4[k_9];
                                smem_v33[(row1_2 * 128 + col_9) / 4] = words_2_1[k_9];
                            }
                            {
                                uint32_t _byte_20 = static_cast<uint32_t>(scale_byte_4) & 0xFFu;
                                uint32_t _addr_20 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base0_2 + k_block_4));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_20), "r"(_byte_20) : "memory");
                            }
                            {
                                uint32_t _byte_21 = static_cast<uint32_t>(scale_byte_2_1) & 0xFFu;
                                uint32_t _addr_21 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base1_2 + k_block_4));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_21), "r"(_byte_21) : "memory");
                            }
                        }
                    } else {
                        int row_2 = tid - 128;
                        unsigned int scale_word_2 = 0;
                        #pragma unroll 1
                        for (int j_5 = 0; j_5 < 4; j_5++) {
                            int k_block_5 = (j_5 + (tid - 128) / 8) % 4;
                            unsigned int pairs_2[16];
                            #pragma unroll
                            for (int k_10 = 0; k_10 < 16; k_10++) {
                                int col_10 = k_block_5 * 32 + ((tid - 128) * 4 + k_10 * 2) % 32;
                                float x0_2 = 0.0f;
                                float x1_2 = 0.0f;
                                float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[16384 + (row_2 * 256 + col_10) / 2]));
                                pairs_2[k_10] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_6.x * flat_view_7[row_2], cvt_f32_6.y * flat_view_7[row_2])));
                            }
                            uint32_t bf16x2_abs_6;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_6) : "r"(pairs_2[0]));
                            unsigned int amax_pair_5 = bf16x2_abs_6;
                            #pragma unroll
                            for (int i_16 = 1; i_16 < 16; i_16++) {
                                uint32_t bf16x2_abs_1_8;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_8) : "r"(pairs_2[i_16]));
                                uint32_t bf16x2_max_8;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_8) : "r"(amax_pair_5), "r"(bf16x2_abs_1_8));
                                amax_pair_5 = bf16x2_max_8;
                            }
                            uint16_t bf16_max_5;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_5) : "h"((uint16_t)(amax_pair_5 & 65535)), "h"((uint16_t)(amax_pair_5 >> 16)));
                            float amax_5;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_5) : "h"((uint16_t)(bf16_max_5)));
                            float scale_5 = fmaxf(amax_5 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_22;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_22) : "f"(scale_5), "f"(scale_5));
                            unsigned int scale_byte_5 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_22 & 255;
                            unsigned int inverse_lane_5 = 254 - scale_byte_5 << 7;
                            unsigned int inverse_5 = inverse_lane_5 | inverse_lane_5 << 16;
                            unsigned int words_5[8];
                            #pragma unroll
                            for (int i_17 = 0; i_17 < 8; i_17++) {
                                uint32_t bf16x2_mul_9;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(pairs_2[i_17 * 2]), "r"(inverse_5));
                                uint16_t e4m3x2_bf16x2_8;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_8) : "r"(bf16x2_mul_9));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_9);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_8) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_8;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(pairs_2[i_17 * 2 + 1]), "r"(inverse_5));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_23;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_23) : "r"(bf16x2_mul_1_8));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_5[i_17] = (unsigned int)e4m3x2_bf16x2_8 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_23 << 16;
                            }
                            scale_word_2 = scale_word_2 | scale_byte_5 << (unsigned int)(k_block_5 * 8);
                            #pragma unroll
                            for (int k_11 = 0; k_11 < 8; k_11++) {
                                int col_11 = k_block_5 * 32 + ((tid - 128) * 4 + k_11 * 4) % 32;
                                smem_v29[(row_2 * 128 + col_11) / 4] = words_5[k_11];
                            }
                        }
                        smem_v31[row_2 % 32 * 4 + row_2 / 32] = scale_word_2;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_2 = col_block * 4 + 2;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&dy_q_store), col_tile_2 * 128, row_1, smem_v29_addr);
                        tma_store_3d((&dy_sc_store), 0, 0, row_tile_0 * k_tiles_0 + col_tile_2, smem_v31_addr);
                        tma_store_2d((&dy_t_store), row_1, col_tile_2 * 128, smem_v33_addr);
                        tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_2 * macro_tiles_0 + row_tile_0, smem_v35_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (hi_cols / 128 > 1) {
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncthreads();
                    if (tid < 128) {
                        int r_3 = tid % 64;
                        int row0_3 = r_3 * 2;
                        int row1_3 = row0_3 + 1;
                        int half_3 = tid / 64;
                        int word_stride_3 = 128;
                        int scale_base0_3 = (row0_3 % 32 * 4 + row0_3 / 32) * 4;
                        int scale_base1_3 = (row1_3 % 32 * 4 + row1_3 / 32) * 4;
                        #pragma unroll 1
                        for (int j_6 = 0; j_6 < 2; j_6++) {
                            int k_block_6 = (half_3 * 2 + j_6 + tid / 8) % 4;
                            unsigned int pairs0_3[16];
                            unsigned int pairs1_3[16];
                            #pragma unroll
                            for (int k_12 = 0; k_12 < 16; k_12++) {
                                int col_12 = k_block_6 * 32 + (tid * 4 + k_12 * 2) % 32;
                                unsigned int w0_3 = flat_view_6[16448 + col_12 * word_stride_3 + r_3];
                                unsigned int w1_3 = flat_view_6[16448 + (col_12 + 1) * word_stride_3 + r_3];
                                uint32_t lo_3;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_3) : "r"(w0_3), "r"(w1_3));
                                uint32_t hi_3;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_3) : "r"(w0_3), "r"(w1_3));
                                float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_3));
                                pairs0_3[k_12] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_7.x * flat_view_7[col_12], cvt_f32_7.y * flat_view_7[col_12 + 1])));
                                float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_3));
                                pairs1_3[k_12] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_3.x * flat_view_7[col_12], cvt_f32_1_3.y * flat_view_7[col_12 + 1])));
                            }
                            uint32_t bf16x2_abs_7;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_7) : "r"(pairs0_3[0]));
                            unsigned int amax_pair_6 = bf16x2_abs_7;
                            #pragma unroll
                            for (int i_18 = 1; i_18 < 16; i_18++) {
                                uint32_t bf16x2_abs_1_9;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_9) : "r"(pairs0_3[i_18]));
                                uint32_t bf16x2_max_9;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_9) : "r"(amax_pair_6), "r"(bf16x2_abs_1_9));
                                amax_pair_6 = bf16x2_max_9;
                            }
                            uint16_t bf16_max_6;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_6) : "h"((uint16_t)(amax_pair_6 & 65535)), "h"((uint16_t)(amax_pair_6 >> 16)));
                            float amax_6;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_6) : "h"((uint16_t)(bf16_max_6)));
                            float scale_6 = fmaxf(amax_6 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_24;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_24) : "f"(scale_6), "f"(scale_6));
                            unsigned int scale_byte_6 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_24 & 255;
                            unsigned int inverse_lane_6 = 254 - scale_byte_6 << 7;
                            unsigned int inverse_6 = inverse_lane_6 | inverse_lane_6 << 16;
                            unsigned int words_6[8];
                            #pragma unroll
                            for (int i_19 = 0; i_19 < 8; i_19++) {
                                uint32_t bf16x2_mul_10;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(pairs0_3[i_19 * 2]), "r"(inverse_6));
                                uint16_t e4m3x2_bf16x2_9;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_9) : "r"(bf16x2_mul_10));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_10);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_9;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(pairs0_3[i_19 * 2 + 1]), "r"(inverse_6));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_25;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_25) : "r"(bf16x2_mul_1_9));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_6[i_19] = (unsigned int)e4m3x2_bf16x2_9 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_25 << 16;
                            }
                            uint32_t bf16x2_abs_3_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_3_1) : "r"(pairs1_3[0]));
                            unsigned int amax_pair_3_1 = bf16x2_abs_3_1;
                            #pragma unroll
                            for (int i_20 = 1; i_20 < 16; i_20++) {
                                uint32_t bf16x2_abs_1_10;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_10) : "r"(pairs1_3[i_20]));
                                uint32_t bf16x2_max_10;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_10) : "r"(amax_pair_3_1), "r"(bf16x2_abs_1_10));
                                amax_pair_3_1 = bf16x2_max_10;
                            }
                            uint16_t bf16_max_3_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_3_1) : "h"((uint16_t)(amax_pair_3_1 & 65535)), "h"((uint16_t)(amax_pair_3_1 >> 16)));
                            float amax_3_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_3_1) : "h"((uint16_t)(bf16_max_3_1)));
                            float scale_3_1 = fmaxf(amax_3_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_26;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_26) : "f"(scale_3_1), "f"(scale_3_1));
                            unsigned int scale_byte_3_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_26 & 255;
                            unsigned int inverse_lane_3_1 = 254 - scale_byte_3_1 << 7;
                            unsigned int inverse_3_1 = inverse_lane_3_1 | inverse_lane_3_1 << 16;
                            unsigned int words_3_1[8];
                            #pragma unroll
                            for (int i_21 = 0; i_21 < 8; i_21++) {
                                uint32_t bf16x2_mul_11;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(pairs1_3[i_21 * 2]), "r"(inverse_3_1));
                                uint16_t e4m3x2_bf16x2_10;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_10) : "r"(bf16x2_mul_11));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_11);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_10;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(pairs1_3[i_21 * 2 + 1]), "r"(inverse_3_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_27;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_27) : "r"(bf16x2_mul_1_10));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_3_1[i_21] = (unsigned int)e4m3x2_bf16x2_10 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_27 << 16;
                            }
                            #pragma unroll
                            for (int k_13 = 0; k_13 < 8; k_13++) {
                                int col_13 = k_block_6 * 32 + (tid * 4 + k_13 * 4) % 32;
                                smem_v34[(row0_3 * 128 + col_13) / 4] = words_6[k_13];
                                smem_v34[(row1_3 * 128 + col_13) / 4] = words_3_1[k_13];
                            }
                            {
                                uint32_t _byte_28 = static_cast<uint32_t>(scale_byte_6) & 0xFFu;
                                uint32_t _addr_28 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base0_3 + k_block_6));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_28), "r"(_byte_28) : "memory");
                            }
                            {
                                uint32_t _byte_29 = static_cast<uint32_t>(scale_byte_3_1) & 0xFFu;
                                uint32_t _addr_29 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base1_3 + k_block_6));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_29), "r"(_byte_29) : "memory");
                            }
                        }
                    } else {
                        int row_3 = tid - 128;
                        unsigned int scale_word_3 = 0;
                        #pragma unroll 1
                        for (int j_7 = 0; j_7 < 4; j_7++) {
                            int k_block_7 = (j_7 + (tid - 128) / 8) % 4;
                            unsigned int pairs_3[16];
                            #pragma unroll
                            for (int k_14 = 0; k_14 < 16; k_14++) {
                                int col_14 = k_block_7 * 32 + ((tid - 128) * 4 + k_14 * 2) % 32;
                                float x0_3 = 0.0f;
                                float x1_3 = 0.0f;
                                float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[16448 + (row_3 * 256 + col_14) / 2]));
                                pairs_3[k_14] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_8.x * flat_view_7[row_3], cvt_f32_8.y * flat_view_7[row_3])));
                            }
                            uint32_t bf16x2_abs_8;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_8) : "r"(pairs_3[0]));
                            unsigned int amax_pair_7 = bf16x2_abs_8;
                            #pragma unroll
                            for (int i_22 = 1; i_22 < 16; i_22++) {
                                uint32_t bf16x2_abs_1_11;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_11) : "r"(pairs_3[i_22]));
                                uint32_t bf16x2_max_11;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_11) : "r"(amax_pair_7), "r"(bf16x2_abs_1_11));
                                amax_pair_7 = bf16x2_max_11;
                            }
                            uint16_t bf16_max_7;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_7) : "h"((uint16_t)(amax_pair_7 & 65535)), "h"((uint16_t)(amax_pair_7 >> 16)));
                            float amax_7;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_7) : "h"((uint16_t)(bf16_max_7)));
                            float scale_7 = fmaxf(amax_7 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_30;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_30) : "f"(scale_7), "f"(scale_7));
                            unsigned int scale_byte_7 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_30 & 255;
                            unsigned int inverse_lane_7 = 254 - scale_byte_7 << 7;
                            unsigned int inverse_7 = inverse_lane_7 | inverse_lane_7 << 16;
                            unsigned int words_7[8];
                            #pragma unroll
                            for (int i_23 = 0; i_23 < 8; i_23++) {
                                uint32_t bf16x2_mul_12;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(pairs_3[i_23 * 2]), "r"(inverse_7));
                                uint16_t e4m3x2_bf16x2_11;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_11) : "r"(bf16x2_mul_12));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_12);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_11;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(pairs_3[i_23 * 2 + 1]), "r"(inverse_7));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_31;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_31) : "r"(bf16x2_mul_1_11));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_7[i_23] = (unsigned int)e4m3x2_bf16x2_11 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_31 << 16;
                            }
                            scale_word_3 = scale_word_3 | scale_byte_7 << (unsigned int)(k_block_7 * 8);
                            #pragma unroll
                            for (int k_15 = 0; k_15 < 8; k_15++) {
                                int col_15 = k_block_7 * 32 + ((tid - 128) * 4 + k_15 * 4) % 32;
                                smem_v30[(row_3 * 128 + col_15) / 4] = words_7[k_15];
                            }
                        }
                        smem_v32[row_3 % 32 * 4 + row_3 / 32] = scale_word_3;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_3 = col_block * 4 + 2 + 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&dy_q_store), col_tile_3 * 128, row_1, smem_v30_addr);
                        tma_store_3d((&dy_sc_store), 0, 0, row_tile_0 * k_tiles_0 + col_tile_3, smem_v32_addr);
                        tma_store_2d((&dy_t_store), row_1, col_tile_3 * 128, smem_v34_addr);
                        tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_3 * macro_tiles_0 + row_tile_0, smem_v36_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset + row_1) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
            }
            __syncthreads();
            task = next_task;
        }
        #pragma unroll 1
        for (int macro = 0; macro < macros; macro++) {
            int rows_0 = ((tokens - macro * macro_size) < (macro_size) ? (tokens - macro * macro_size) : (macro_size));
            int combine_tasks = (rows_0 / 16 * ((hidden + 1023) / 1024) + 6) / 7;
            #pragma unroll 1
            for (int task_1 = cluster * 2 + cta_rank_0; task_1 < combine_tasks; task_1 += comm_sms) {
                int col_blocks_0 = (hidden + 1023) / 1024;
                int first_tile = task_1 * 7;
                int macro_offset_1 = macro * macro_size;
                int macro_tokens_2 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                int valid_tiles = ((7) < (macro_tokens_2 / 16 * col_blocks_0 - first_tile) ? (7) : (macro_tokens_2 / 16 * col_blocks_0 - first_tile));
                if (valid_tiles > 0) {
                    int first_row = first_tile / col_blocks_0 * 16 + tid;
                    int first_col = first_tile % col_blocks_0;
                    int rows_1[7];
                    int columns[7];
                    int peers[7];
                    int tokens_2[7];
                    unsigned int counts_3[7];
                    int row_4 = first_row;
                    int column = first_col;
                    #pragma unroll
                    for (int stage = 0; stage < 7; stage++) {
                        rows_1[stage] = row_4;
                        columns[stage] = column;
                        peers[stage] = -1;
                        tokens_2[stage] = -1;
                        if (valid_tiles > stage && tid < 16) {
                            peers[stage] = schedule_rank[macro_offset_1 + row_4];
                            tokens_2[stage] = schedule_token[macro_offset_1 + row_4];
                        }
                        counts_3[stage] = 0;
                        if (valid_tiles > stage) {
                            if (stage == 0 || column == 0) {
                                uint32_t _barrier_vote_32 = __syncthreads_count(peers[stage] >= 0);
                                counts_3[stage] = _barrier_vote_32;
                            } else {
                                counts_3[stage] = counts_3[stage - 1];
                            }
                        }
                        column = column + 1;
                        if (column == col_blocks_0) {
                            column = 0;
                            row_4 = row_4 + 16;
                        }
                    }
                    if (tid == 0) {
                        int first_mini = (macro_offset_1 + first_row) / mini_size;
                        int last_mini = (macro_offset_1 + (first_tile + valid_tiles - 1) / col_blocks_0 * 16) / mini_size;
                        #pragma unroll 1
                        for (int mini = first_mini; mini < last_mini + 1; mini++) {
                            int mini_rows = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                            int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                            int poll_1_first;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_first) : "l"(dx_ready + mini) : "memory");
                            int poll_1 = poll_1_first;
                            while (poll_1 < required) {
                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                int poll_1_next;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_next) : "l"(dx_ready + mini) : "memory");
                                poll_1 = poll_1_next;
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                        }
                        if (valid_tiles > 0) {
                            int _min_33 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                            unsigned int chunk_bytes = (unsigned int)(_min_33 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr, counts_3[0] * chunk_bytes);
                        }
                        if (valid_tiles > 1) {
                            int _min_34 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                            unsigned int chunk_bytes_1 = (unsigned int)(_min_34 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 8, counts_3[1] * chunk_bytes_1);
                        }
                        if (valid_tiles > 2) {
                            int _min_35 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                            unsigned int chunk_bytes_2 = (unsigned int)(_min_35 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 16, counts_3[2] * chunk_bytes_2);
                        }
                        if (valid_tiles > 3) {
                            int _min_36 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                            unsigned int chunk_bytes_3 = (unsigned int)(_min_36 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 24, counts_3[3] * chunk_bytes_3);
                        }
                        if (valid_tiles > 4) {
                            int _min_37 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                            unsigned int chunk_bytes_4 = (unsigned int)(_min_37 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 32, counts_3[4] * chunk_bytes_4);
                        }
                        if (valid_tiles > 5) {
                            int _min_38 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                            unsigned int chunk_bytes_5 = (unsigned int)(_min_38 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 40, counts_3[5] * chunk_bytes_5);
                        }
                        if (valid_tiles > 6) {
                            int _min_39 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                            unsigned int chunk_bytes_6 = (unsigned int)(_min_39 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 48, counts_3[6] * chunk_bytes_6);
                        }
                    }
                    __syncthreads();
                    if (peers[0] >= 0) {
                        int chunk_cols_1 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)(tid * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)) * (unsigned long long)2)), chunk_cols_1 * 2, combine_arrived_addr);
                    }
                    if (peers[1] >= 0) {
                        int chunk_cols_2 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)((16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)) * (unsigned long long)2)), chunk_cols_2 * 2, combine_arrived_addr + 8);
                    }
                    if (peers[2] >= 0) {
                        int chunk_cols_3 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)((32 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)) * (unsigned long long)2)), chunk_cols_3 * 2, combine_arrived_addr + 16);
                    }
                    if (peers[3] >= 0) {
                        int chunk_cols_4 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)((48 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)) * (unsigned long long)2)), chunk_cols_4 * 2, combine_arrived_addr + 24);
                    }
                    if (peers[4] >= 0) {
                        int chunk_cols_5 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)((64 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)) * (unsigned long long)2)), chunk_cols_5 * 2, combine_arrived_addr + 32);
                    }
                    if (peers[5] >= 0) {
                        int chunk_cols_6 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)((80 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)) * (unsigned long long)2)), chunk_cols_6 * 2, combine_arrived_addr + 40);
                    }
                    if (peers[6] >= 0) {
                        int chunk_cols_7 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_8 + (unsigned int)((96 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)) * (unsigned long long)2)), chunk_cols_7 * 2, combine_arrived_addr + 48);
                    }
                    if (valid_tiles > 0) {
                        int _parity_combine_arrived = (int)((unsigned int)combine_seq_0 % 2);
                        mbarrier_wait(combine_arrived_addr, _parity_combine_arrived);
                        combine_seq_0 += 1;
                        if (peers[0] >= 0) {
                            int _min_40 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                            unsigned int chunk_bytes_7 = (unsigned int)(_min_40 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_41 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[0]]) + ((unsigned long long)tokens_2[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_41), "r"(flat_view_addr_8 + (unsigned int)(tid * 2048)), "r"((uint32_t)(chunk_bytes_7))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[0] == 0) {
                                float gradient = 0.0f;
                                #pragma unroll 1
                                for (int col_16 = 0; col_16 < intermediate / 128; col_16++) {
                                    gradient = gradient + partials[rows_1[0] * (intermediate / 128) + col_16];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[0]])[tokens_2[0]] = gradient;
                            }
                        }
                    }
                    if (valid_tiles > 1) {
                        int _parity_combine_arrived_1 = (int)((unsigned int)combine_seq_1 % 2);
                        mbarrier_wait(combine_arrived_addr + 8, _parity_combine_arrived_1);
                        combine_seq_1 += 1;
                        if (peers[1] >= 0) {
                            int _min_42 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                            unsigned int chunk_bytes_8 = (unsigned int)(_min_42 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_43 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[1]]) + ((unsigned long long)tokens_2[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_43), "r"(flat_view_addr_8 + (unsigned int)((16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_8))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[1] == 0) {
                                float gradient_1 = 0.0f;
                                #pragma unroll 1
                                for (int col_17 = 0; col_17 < intermediate / 128; col_17++) {
                                    gradient_1 = gradient_1 + partials[rows_1[1] * (intermediate / 128) + col_17];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[1]])[tokens_2[1]] = gradient_1;
                            }
                        }
                    }
                    if (valid_tiles > 2) {
                        int _parity_combine_arrived_2 = (int)((unsigned int)combine_seq_2 % 2);
                        mbarrier_wait(combine_arrived_addr + 16, _parity_combine_arrived_2);
                        combine_seq_2 += 1;
                        if (peers[2] >= 0) {
                            int _min_44 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                            unsigned int chunk_bytes_9 = (unsigned int)(_min_44 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_45 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[2]]) + ((unsigned long long)tokens_2[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_45), "r"(flat_view_addr_8 + (unsigned int)((32 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_9))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[2] == 0) {
                                float gradient_2 = 0.0f;
                                #pragma unroll 1
                                for (int col_18 = 0; col_18 < intermediate / 128; col_18++) {
                                    gradient_2 = gradient_2 + partials[rows_1[2] * (intermediate / 128) + col_18];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[2]])[tokens_2[2]] = gradient_2;
                            }
                        }
                    }
                    if (valid_tiles > 3) {
                        int _parity_combine_arrived_3 = (int)((unsigned int)combine_seq_3 % 2);
                        mbarrier_wait(combine_arrived_addr + 24, _parity_combine_arrived_3);
                        combine_seq_3 += 1;
                        if (peers[3] >= 0) {
                            int _min_46 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                            unsigned int chunk_bytes_10 = (unsigned int)(_min_46 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_47 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[3]]) + ((unsigned long long)tokens_2[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_47), "r"(flat_view_addr_8 + (unsigned int)((48 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_10))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[3] == 0) {
                                float gradient_3 = 0.0f;
                                #pragma unroll 1
                                for (int col_19 = 0; col_19 < intermediate / 128; col_19++) {
                                    gradient_3 = gradient_3 + partials[rows_1[3] * (intermediate / 128) + col_19];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[3]])[tokens_2[3]] = gradient_3;
                            }
                        }
                    }
                    if (valid_tiles > 4) {
                        int _parity_combine_arrived_4 = (int)((unsigned int)combine_seq_4 % 2);
                        mbarrier_wait(combine_arrived_addr + 32, _parity_combine_arrived_4);
                        combine_seq_4 += 1;
                        if (peers[4] >= 0) {
                            int _min_48 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                            unsigned int chunk_bytes_11 = (unsigned int)(_min_48 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_49 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[4]]) + ((unsigned long long)tokens_2[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_49), "r"(flat_view_addr_8 + (unsigned int)((64 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_11))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[4] == 0) {
                                float gradient_4 = 0.0f;
                                #pragma unroll 1
                                for (int col_20 = 0; col_20 < intermediate / 128; col_20++) {
                                    gradient_4 = gradient_4 + partials[rows_1[4] * (intermediate / 128) + col_20];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[4]])[tokens_2[4]] = gradient_4;
                            }
                        }
                    }
                    if (valid_tiles > 5) {
                        int _parity_combine_arrived_5 = (int)((unsigned int)combine_seq_5 % 2);
                        mbarrier_wait(combine_arrived_addr + 40, _parity_combine_arrived_5);
                        combine_seq_5 += 1;
                        if (peers[5] >= 0) {
                            int _min_50 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                            unsigned int chunk_bytes_12 = (unsigned int)(_min_50 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_51 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[5]]) + ((unsigned long long)tokens_2[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_51), "r"(flat_view_addr_8 + (unsigned int)((80 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_12))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[5] == 0) {
                                float gradient_5 = 0.0f;
                                #pragma unroll 1
                                for (int col_21 = 0; col_21 < intermediate / 128; col_21++) {
                                    gradient_5 = gradient_5 + partials[rows_1[5] * (intermediate / 128) + col_21];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[5]])[tokens_2[5]] = gradient_5;
                            }
                        }
                    }
                    if (valid_tiles > 6) {
                        int _parity_combine_arrived_6 = (int)((unsigned int)combine_seq_6 % 2);
                        mbarrier_wait(combine_arrived_addr + 48, _parity_combine_arrived_6);
                        combine_seq_6 += 1;
                        if (peers[6] >= 0) {
                            int _min_52 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                            unsigned int chunk_bytes_13 = (unsigned int)(_min_52 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_53 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[peers[6]]) + ((unsigned long long)tokens_2[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_53), "r"(flat_view_addr_8 + (unsigned int)((96 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_13))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            if (columns[6] == 0) {
                                float gradient_6 = 0.0f;
                                #pragma unroll 1
                                for (int col_22 = 0; col_22 < intermediate / 128; col_22++) {
                                    gradient_6 = gradient_6 + partials[rows_1[6] * (intermediate / 128) + col_22];
                                }
                                reinterpret_cast<float*>(dweight_peers[peers[6]])[tokens_2[6]] = gradient_6;
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
            }
            if (macros > macro + 1) {
                int minis_0 = (rows_0 + mini_size - 1) / mini_size;
                int required_1 = 2 * (minis_0 * mini_bwd + wgrad_tasks) + combine_tasks;
                if (tid == 0) {
                    bool enabled_value_1 = 1;
                    if (enabled_value_1 != 0) {
                        int poll_2_first;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_first) : "l"(buffers_done + macro) : "memory");
                        int poll_2 = poll_2_first;
                        while (poll_2 < required_1) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int poll_2_next;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_next) : "l"(buffers_done + macro) : "memory");
                            poll_2 = poll_2_next;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                    }
                }
                __syncthreads();
                int offset_1 = (macro + 1) * macro_size;
                int rows_2 = ((macro_size) < (tokens - offset_1) ? (macro_size) : (tokens - offset_1));
                #pragma unroll 1
                for (int row_5 = (cluster * 2 + cta_rank_0) * 256 + tid; row_5 < rows_2; row_5 += comm_sms * 256) {
                    int peer_4 = schedule_rank[offset_1 + row_5];
                    int token_1 = schedule_token[offset_1 + row_5];
                    float value_1 = 0.0f;
                    if (peer_4 >= 0) {
                        value_1 = reinterpret_cast<float*>(weight_peers[peer_4])[(int)token_1];
                    }
                    weights[row_5] = value_1;
                }
                __syncthreads();
                if (tid == 0) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(weight_ready)) + (macro + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    int poll_3_first;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_first) : "l"(weight_ready + (macro + 1)) : "memory");
                    int poll_3 = poll_3_first;
                    while (poll_3 < comm_sms) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int poll_3_next;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_next) : "l"(weight_ready + (macro + 1)) : "memory");
                        poll_3 = poll_3_next;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                }
                __syncthreads();
                int col_blocks_3 = (hidden + 511) / 512;
                int macro_offset_4 = (macro + 1) * macro_size;
                int macro_tokens_5 = ((macro_size) < (tokens - macro_offset_4) ? (macro_size) : (tokens - macro_offset_4));
                int tasks_6 = macro_tokens_5 / 128 * col_blocks_3;
                int task_7 = cluster * 2 + cta_rank_0;
                if (task_7 < tasks_6) {
                    int _min_2 = ((256) < (hidden - task_7 % col_blocks_3 * 512) ? (256) : (hidden - task_7 % col_blocks_3 * 512));
                    int peer_5 = -1;
                    int peer_token_3 = -1;
                    if (tid < 128) {
                        peer_5 = schedule_rank[macro_offset_4 + task_7 / col_blocks_3 * 128 + tid];
                        peer_token_3 = schedule_token[macro_offset_4 + task_7 / col_blocks_3 * 128 + tid];
                    }
                    uint32_t cta_count_3 = __syncthreads_count(peer_5 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_3 * (unsigned int)(_min_2 * 2));
                    }
                    __syncthreads();
                    if (peer_5 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_5])) + ((unsigned long long)((unsigned long long)(peer_token_3 / topk) * (unsigned long long)hidden + (unsigned long long)(task_7 % col_blocks_3 * 512)) * (unsigned long long)2)), _min_2 * 2, dispatch_arrived_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_3 = 0; vec_3 < 32; vec_3++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (tid * 512 + vec_3 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                while (task_7 < tasks_6) {
                    int row_6 = task_7 / col_blocks_3 * 128;
                    int col_block_1 = task_7 % col_blocks_3;
                    int chunk_cols_8 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                    int lo_cols_1 = ((256) < (chunk_cols_8) ? (256) : (chunk_cols_8));
                    int hi_cols_1 = chunk_cols_8 - lo_cols_1;
                    if (hi_cols_1 > 0) {
                        int peer_6 = -1;
                        int peer_token_4 = -1;
                        if (tid < 128) {
                            peer_6 = schedule_rank[macro_offset_4 + row_6 + tid];
                            peer_token_4 = schedule_token[macro_offset_4 + row_6 + tid];
                        }
                        uint32_t cta_count_4 = __syncthreads_count(peer_6 >= 0);
                        if (tid == 0) {
                            mbarrier_arrive_expect_tx(dispatch_arrived_hi_addr, cta_count_4 * (unsigned int)(hi_cols_1 * 2));
                        }
                        __syncthreads();
                        if (peer_6 >= 0) {
                            cp_async_bulk_gmem2smem(smem_v39_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_6])) + ((unsigned long long)((unsigned long long)(peer_token_4 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512) + 256) * (unsigned long long)2)), hi_cols_1 * 2, dispatch_arrived_hi_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_4 = 0; vec_4 < 32; vec_4++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (65536 + tid * 512 + vec_4 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                    int _parity_dispatch_arrived_1 = (int)((unsigned int)dispatch_seq % 2);
                    mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived_1);
                    dispatch_seq += 1;
                    int row_tile_1 = row_6 / 128;
                    int k_tiles_1 = hidden / 128;
                    int macro_tiles_1 = macro_size / 128;
                    if (lo_cols_1 / 128 > 0) {
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        __syncthreads();
                        if (tid < 128) {
                            int r_4 = tid % 64;
                            int row0_4 = r_4 * 2;
                            int row1_4 = row0_4 + 1;
                            int half_4 = tid / 64;
                            int word_stride_4 = 128;
                            int scale_base0_4 = (row0_4 % 32 * 4 + row0_4 / 32) * 4;
                            int scale_base1_4 = (row1_4 % 32 * 4 + row1_4 / 32) * 4;
                            #pragma unroll 1
                            for (int j_8 = 0; j_8 < 2; j_8++) {
                                int k_block_8 = (half_4 * 2 + j_8 + tid / 8) % 4;
                                unsigned int pairs0_4[16];
                                unsigned int pairs1_4[16];
                                #pragma unroll
                                for (int k_16 = 0; k_16 < 16; k_16++) {
                                    int col_23 = k_block_8 * 32 + (tid * 4 + k_16 * 2) % 32;
                                    unsigned int w0_4 = flat_view_6[col_23 * word_stride_4 + r_4];
                                    unsigned int w1_4 = flat_view_6[(col_23 + 1) * word_stride_4 + r_4];
                                    uint32_t lo_4;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_4) : "r"(w0_4), "r"(w1_4));
                                    uint32_t hi_4;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_4) : "r"(w0_4), "r"(w1_4));
                                    pairs0_4[k_16] = lo_4;
                                    pairs1_4[k_16] = hi_4;
                                }
                                uint32_t bf16x2_abs_9;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_9) : "r"(pairs0_4[0]));
                                unsigned int amax_pair_8 = bf16x2_abs_9;
                                #pragma unroll
                                for (int i_24 = 1; i_24 < 16; i_24++) {
                                    uint32_t bf16x2_abs_1_12;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_12) : "r"(pairs0_4[i_24]));
                                    uint32_t bf16x2_max_12;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_12) : "r"(amax_pair_8), "r"(bf16x2_abs_1_12));
                                    amax_pair_8 = bf16x2_max_12;
                                }
                                uint16_t bf16_max_8;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_8) : "h"((uint16_t)(amax_pair_8 & 65535)), "h"((uint16_t)(amax_pair_8 >> 16)));
                                float amax_8;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8) : "h"((uint16_t)(bf16_max_8)));
                                float scale_8 = fmaxf(amax_8 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_54;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_54) : "f"(scale_8), "f"(scale_8));
                                unsigned int scale_byte_8 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_54 & 255;
                                unsigned int inverse_lane_8 = 254 - scale_byte_8 << 7;
                                unsigned int inverse_8 = inverse_lane_8 | inverse_lane_8 << 16;
                                unsigned int words_8[8];
                                #pragma unroll
                                for (int i_25 = 0; i_25 < 8; i_25++) {
                                    uint32_t bf16x2_mul_13;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(pairs0_4[i_25 * 2]), "r"(inverse_8));
                                    uint16_t e4m3x2_bf16x2_12;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_12) : "r"(bf16x2_mul_13));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_13);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_12;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(pairs0_4[i_25 * 2 + 1]), "r"(inverse_8));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_55;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_55) : "r"(bf16x2_mul_1_12));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_55) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_8[i_25] = (unsigned int)e4m3x2_bf16x2_12 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_55 << 16;
                                }
                                uint32_t bf16x2_abs_4_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_4_1) : "r"(pairs1_4[0]));
                                unsigned int amax_pair_4_1 = bf16x2_abs_4_1;
                                #pragma unroll
                                for (int i_26 = 1; i_26 < 16; i_26++) {
                                    uint32_t bf16x2_abs_1_13;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_13) : "r"(pairs1_4[i_26]));
                                    uint32_t bf16x2_max_13;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_13) : "r"(amax_pair_4_1), "r"(bf16x2_abs_1_13));
                                    amax_pair_4_1 = bf16x2_max_13;
                                }
                                uint16_t bf16_max_4_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_4_1) : "h"((uint16_t)(amax_pair_4_1 & 65535)), "h"((uint16_t)(amax_pair_4_1 >> 16)));
                                float amax_4_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_4_1) : "h"((uint16_t)(bf16_max_4_1)));
                                float scale_4_1 = fmaxf(amax_4_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_56;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_56) : "f"(scale_4_1), "f"(scale_4_1));
                                unsigned int scale_byte_4_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_56 & 255;
                                unsigned int inverse_lane_4_1 = 254 - scale_byte_4_1 << 7;
                                unsigned int inverse_4_1 = inverse_lane_4_1 | inverse_lane_4_1 << 16;
                                unsigned int words_4_1[8];
                                #pragma unroll
                                for (int i_27 = 0; i_27 < 8; i_27++) {
                                    uint32_t bf16x2_mul_14;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(pairs1_4[i_27 * 2]), "r"(inverse_4_1));
                                    uint16_t e4m3x2_bf16x2_13;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_13) : "r"(bf16x2_mul_14));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_14);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_13;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(pairs1_4[i_27 * 2 + 1]), "r"(inverse_4_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_57;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_57) : "r"(bf16x2_mul_1_13));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_57) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_4_1[i_27] = (unsigned int)e4m3x2_bf16x2_13 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_57 << 16;
                                }
                                #pragma unroll
                                for (int k_17 = 0; k_17 < 8; k_17++) {
                                    int col_24 = k_block_8 * 32 + (tid * 4 + k_17 * 4) % 32;
                                    smem_v33[(row0_4 * 128 + col_24) / 4] = words_8[k_17];
                                    smem_v33[(row1_4 * 128 + col_24) / 4] = words_4_1[k_17];
                                }
                                {
                                    uint32_t _byte_58 = static_cast<uint32_t>(scale_byte_8) & 0xFFu;
                                    uint32_t _addr_58 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base0_4 + k_block_8));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                                }
                                {
                                    uint32_t _byte_59 = static_cast<uint32_t>(scale_byte_4_1) & 0xFFu;
                                    uint32_t _addr_59 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base1_4 + k_block_8));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_59), "r"(_byte_59) : "memory");
                                }
                            }
                        } else {
                            int row_4_1 = tid - 128;
                            unsigned int scale_word_4 = 0;
                            #pragma unroll 1
                            for (int j_9 = 0; j_9 < 4; j_9++) {
                                int k_block_9 = (j_9 + (tid - 128) / 8) % 4;
                                unsigned int pairs_4[16];
                                #pragma unroll
                                for (int k_18 = 0; k_18 < 16; k_18++) {
                                    int col_25 = k_block_9 * 32 + ((tid - 128) * 4 + k_18 * 2) % 32;
                                    float x0_4 = 0.0f;
                                    float x1_4 = 0.0f;
                                    pairs_4[k_18] = flat_view_6[(row_4_1 * 256 + col_25) / 2];
                                }
                                uint32_t bf16x2_abs_10;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_10) : "r"(pairs_4[0]));
                                unsigned int amax_pair_9 = bf16x2_abs_10;
                                #pragma unroll
                                for (int i_28 = 1; i_28 < 16; i_28++) {
                                    uint32_t bf16x2_abs_1_14;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_14) : "r"(pairs_4[i_28]));
                                    uint32_t bf16x2_max_14;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_14) : "r"(amax_pair_9), "r"(bf16x2_abs_1_14));
                                    amax_pair_9 = bf16x2_max_14;
                                }
                                uint16_t bf16_max_9;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_9) : "h"((uint16_t)(amax_pair_9 & 65535)), "h"((uint16_t)(amax_pair_9 >> 16)));
                                float amax_9;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_9) : "h"((uint16_t)(bf16_max_9)));
                                float scale_9 = fmaxf(amax_9 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_60;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_60) : "f"(scale_9), "f"(scale_9));
                                unsigned int scale_byte_9 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_60 & 255;
                                unsigned int inverse_lane_9 = 254 - scale_byte_9 << 7;
                                unsigned int inverse_9 = inverse_lane_9 | inverse_lane_9 << 16;
                                unsigned int words_9[8];
                                #pragma unroll
                                for (int i_29 = 0; i_29 < 8; i_29++) {
                                    uint32_t bf16x2_mul_15;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(pairs_4[i_29 * 2]), "r"(inverse_9));
                                    uint16_t e4m3x2_bf16x2_14;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_14) : "r"(bf16x2_mul_15));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_15);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_14;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(pairs_4[i_29 * 2 + 1]), "r"(inverse_9));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_61;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_61) : "r"(bf16x2_mul_1_14));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_61) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_9[i_29] = (unsigned int)e4m3x2_bf16x2_14 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_61 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_9 << (unsigned int)(k_block_9 * 8);
                                #pragma unroll
                                for (int k_19 = 0; k_19 < 8; k_19++) {
                                    int col_26 = k_block_9 * 32 + ((tid - 128) * 4 + k_19 * 4) % 32;
                                    smem_v29[(row_4_1 * 128 + col_26) / 4] = words_9[k_19];
                                }
                            }
                            smem_v31[row_4_1 % 32 * 4 + row_4_1 / 32] = scale_word_4;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_4 = col_block_1 * 4;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_4 * 128, row_6, smem_v29_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_4, smem_v31_addr);
                            tma_store_2d((&x_t_store), row_6, col_tile_4 * 128, smem_v33_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_4 * macro_tiles_1 + row_tile_1, smem_v35_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (lo_cols_1 / 128 > 1) {
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        __syncthreads();
                        if (tid < 128) {
                            int r_5 = tid % 64;
                            int row0_5 = r_5 * 2;
                            int row1_5 = row0_5 + 1;
                            int half_5 = tid / 64;
                            int word_stride_5 = 128;
                            int scale_base0_5 = (row0_5 % 32 * 4 + row0_5 / 32) * 4;
                            int scale_base1_5 = (row1_5 % 32 * 4 + row1_5 / 32) * 4;
                            #pragma unroll 1
                            for (int j_10 = 0; j_10 < 2; j_10++) {
                                int k_block_10 = (half_5 * 2 + j_10 + tid / 8) % 4;
                                unsigned int pairs0_5[16];
                                unsigned int pairs1_5[16];
                                #pragma unroll
                                for (int k_20 = 0; k_20 < 16; k_20++) {
                                    int col_27 = k_block_10 * 32 + (tid * 4 + k_20 * 2) % 32;
                                    unsigned int w0_5 = flat_view_6[64 + col_27 * word_stride_5 + r_5];
                                    unsigned int w1_5 = flat_view_6[64 + (col_27 + 1) * word_stride_5 + r_5];
                                    uint32_t lo_5;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_5) : "r"(w0_5), "r"(w1_5));
                                    uint32_t hi_5;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_5) : "r"(w0_5), "r"(w1_5));
                                    pairs0_5[k_20] = lo_5;
                                    pairs1_5[k_20] = hi_5;
                                }
                                uint32_t bf16x2_abs_11;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_11) : "r"(pairs0_5[0]));
                                unsigned int amax_pair_10 = bf16x2_abs_11;
                                #pragma unroll
                                for (int i_30 = 1; i_30 < 16; i_30++) {
                                    uint32_t bf16x2_abs_1_15;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_15) : "r"(pairs0_5[i_30]));
                                    uint32_t bf16x2_max_15;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_15) : "r"(amax_pair_10), "r"(bf16x2_abs_1_15));
                                    amax_pair_10 = bf16x2_max_15;
                                }
                                uint16_t bf16_max_10;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_10) : "h"((uint16_t)(amax_pair_10 & 65535)), "h"((uint16_t)(amax_pair_10 >> 16)));
                                float amax_10;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10) : "h"((uint16_t)(bf16_max_10)));
                                float scale_10 = fmaxf(amax_10 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_62;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_62) : "f"(scale_10), "f"(scale_10));
                                unsigned int scale_byte_10 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_62 & 255;
                                unsigned int inverse_lane_10 = 254 - scale_byte_10 << 7;
                                unsigned int inverse_10 = inverse_lane_10 | inverse_lane_10 << 16;
                                unsigned int words_10[8];
                                #pragma unroll
                                for (int i_31 = 0; i_31 < 8; i_31++) {
                                    uint32_t bf16x2_mul_16;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(pairs0_5[i_31 * 2]), "r"(inverse_10));
                                    uint16_t e4m3x2_bf16x2_15;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_15) : "r"(bf16x2_mul_16));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_16);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_15;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(pairs0_5[i_31 * 2 + 1]), "r"(inverse_10));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_63;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_63) : "r"(bf16x2_mul_1_15));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_63) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_10[i_31] = (unsigned int)e4m3x2_bf16x2_15 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_63 << 16;
                                }
                                uint32_t bf16x2_abs_5_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_5_1) : "r"(pairs1_5[0]));
                                unsigned int amax_pair_5_1 = bf16x2_abs_5_1;
                                #pragma unroll
                                for (int i_32 = 1; i_32 < 16; i_32++) {
                                    uint32_t bf16x2_abs_1_16;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_16) : "r"(pairs1_5[i_32]));
                                    uint32_t bf16x2_max_16;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_16) : "r"(amax_pair_5_1), "r"(bf16x2_abs_1_16));
                                    amax_pair_5_1 = bf16x2_max_16;
                                }
                                uint16_t bf16_max_5_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_5_1) : "h"((uint16_t)(amax_pair_5_1 & 65535)), "h"((uint16_t)(amax_pair_5_1 >> 16)));
                                float amax_5_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_5_1) : "h"((uint16_t)(bf16_max_5_1)));
                                float scale_5_1 = fmaxf(amax_5_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_64;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_64) : "f"(scale_5_1), "f"(scale_5_1));
                                unsigned int scale_byte_5_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_64 & 255;
                                unsigned int inverse_lane_5_1 = 254 - scale_byte_5_1 << 7;
                                unsigned int inverse_5_1 = inverse_lane_5_1 | inverse_lane_5_1 << 16;
                                unsigned int words_5_1[8];
                                #pragma unroll
                                for (int i_33 = 0; i_33 < 8; i_33++) {
                                    uint32_t bf16x2_mul_17;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(pairs1_5[i_33 * 2]), "r"(inverse_5_1));
                                    uint16_t e4m3x2_bf16x2_16;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_16) : "r"(bf16x2_mul_17));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_17);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_16;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(pairs1_5[i_33 * 2 + 1]), "r"(inverse_5_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_65;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_65) : "r"(bf16x2_mul_1_16));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_65) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_5_1[i_33] = (unsigned int)e4m3x2_bf16x2_16 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_65 << 16;
                                }
                                #pragma unroll
                                for (int k_21 = 0; k_21 < 8; k_21++) {
                                    int col_28 = k_block_10 * 32 + (tid * 4 + k_21 * 4) % 32;
                                    smem_v34[(row0_5 * 128 + col_28) / 4] = words_10[k_21];
                                    smem_v34[(row1_5 * 128 + col_28) / 4] = words_5_1[k_21];
                                }
                                {
                                    uint32_t _byte_66 = static_cast<uint32_t>(scale_byte_10) & 0xFFu;
                                    uint32_t _addr_66 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base0_5 + k_block_10));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_66), "r"(_byte_66) : "memory");
                                }
                                {
                                    uint32_t _byte_67 = static_cast<uint32_t>(scale_byte_5_1) & 0xFFu;
                                    uint32_t _addr_67 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base1_5 + k_block_10));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_67), "r"(_byte_67) : "memory");
                                }
                            }
                        } else {
                            int row_5_1 = tid - 128;
                            unsigned int scale_word_5 = 0;
                            #pragma unroll 1
                            for (int j_11 = 0; j_11 < 4; j_11++) {
                                int k_block_11 = (j_11 + (tid - 128) / 8) % 4;
                                unsigned int pairs_5[16];
                                #pragma unroll
                                for (int k_22 = 0; k_22 < 16; k_22++) {
                                    int col_29 = k_block_11 * 32 + ((tid - 128) * 4 + k_22 * 2) % 32;
                                    float x0_5 = 0.0f;
                                    float x1_5 = 0.0f;
                                    pairs_5[k_22] = flat_view_6[64 + (row_5_1 * 256 + col_29) / 2];
                                }
                                uint32_t bf16x2_abs_12;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_12) : "r"(pairs_5[0]));
                                unsigned int amax_pair_11 = bf16x2_abs_12;
                                #pragma unroll
                                for (int i_34 = 1; i_34 < 16; i_34++) {
                                    uint32_t bf16x2_abs_1_17;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_17) : "r"(pairs_5[i_34]));
                                    uint32_t bf16x2_max_17;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_17) : "r"(amax_pair_11), "r"(bf16x2_abs_1_17));
                                    amax_pair_11 = bf16x2_max_17;
                                }
                                uint16_t bf16_max_11;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_11) : "h"((uint16_t)(amax_pair_11 & 65535)), "h"((uint16_t)(amax_pair_11 >> 16)));
                                float amax_11;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11) : "h"((uint16_t)(bf16_max_11)));
                                float scale_11 = fmaxf(amax_11 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_68;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_68) : "f"(scale_11), "f"(scale_11));
                                unsigned int scale_byte_11 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_68 & 255;
                                unsigned int inverse_lane_11 = 254 - scale_byte_11 << 7;
                                unsigned int inverse_11 = inverse_lane_11 | inverse_lane_11 << 16;
                                unsigned int words_11[8];
                                #pragma unroll
                                for (int i_35 = 0; i_35 < 8; i_35++) {
                                    uint32_t bf16x2_mul_18;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(pairs_5[i_35 * 2]), "r"(inverse_11));
                                    uint16_t e4m3x2_bf16x2_17;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_17) : "r"(bf16x2_mul_18));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_18);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_17;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(pairs_5[i_35 * 2 + 1]), "r"(inverse_11));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_69;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_69) : "r"(bf16x2_mul_1_17));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_69) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_11[i_35] = (unsigned int)e4m3x2_bf16x2_17 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_69 << 16;
                                }
                                scale_word_5 = scale_word_5 | scale_byte_11 << (unsigned int)(k_block_11 * 8);
                                #pragma unroll
                                for (int k_23 = 0; k_23 < 8; k_23++) {
                                    int col_30 = k_block_11 * 32 + ((tid - 128) * 4 + k_23 * 4) % 32;
                                    smem_v30[(row_5_1 * 128 + col_30) / 4] = words_11[k_23];
                                }
                            }
                            smem_v32[row_5_1 % 32 * 4 + row_5_1 / 32] = scale_word_5;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_5 = col_block_1 * 4 + 1;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_5 * 128, row_6, smem_v30_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_5, smem_v32_addr);
                            tma_store_2d((&x_t_store), row_6, col_tile_5 * 128, smem_v34_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_5 * macro_tiles_1 + row_tile_1, smem_v36_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    int next_task_1 = task_7 + comm_sms;
                    if (next_task_1 < tasks_6) {
                        int _min_3 = ((256) < (hidden - next_task_1 % col_blocks_3 * 512) ? (256) : (hidden - next_task_1 % col_blocks_3 * 512));
                        int peer_7 = -1;
                        int peer_token_5 = -1;
                        if (tid < 128) {
                            peer_7 = schedule_rank[macro_offset_4 + next_task_1 / col_blocks_3 * 128 + tid];
                            peer_token_5 = schedule_token[macro_offset_4 + next_task_1 / col_blocks_3 * 128 + tid];
                        }
                        uint32_t cta_count_5 = __syncthreads_count(peer_7 >= 0);
                        if (tid == 0) {
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_5 * (unsigned int)(_min_3 * 2));
                        }
                        __syncthreads();
                        if (peer_7 >= 0) {
                            cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_7])) + ((unsigned long long)((unsigned long long)(peer_token_5 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task_1 % col_blocks_3 * 512)) * (unsigned long long)2)), _min_3 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_5 = 0; vec_5 < 32; vec_5++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (tid * 512 + vec_5 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                    if (hi_cols_1 > 0) {
                        int _parity_dispatch_arrived_hi_1 = (int)((unsigned int)dispatch_seq_hi % 2);
                        mbarrier_wait(dispatch_arrived_hi_addr, _parity_dispatch_arrived_hi_1);
                        dispatch_seq_hi += 1;
                        int row_tile_1_1 = row_6 / 128;
                        int k_tiles_1_1 = hidden / 128;
                        int macro_tiles_1_1 = macro_size / 128;
                        if (hi_cols_1 / 128 > 0) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int r_6 = tid % 64;
                                int row0_6 = r_6 * 2;
                                int row1_6 = row0_6 + 1;
                                int half_6 = tid / 64;
                                int word_stride_6 = 128;
                                int scale_base0_6 = (row0_6 % 32 * 4 + row0_6 / 32) * 4;
                                int scale_base1_6 = (row1_6 % 32 * 4 + row1_6 / 32) * 4;
                                #pragma unroll 1
                                for (int j_12 = 0; j_12 < 2; j_12++) {
                                    int k_block_12 = (half_6 * 2 + j_12 + tid / 8) % 4;
                                    unsigned int pairs0_6[16];
                                    unsigned int pairs1_6[16];
                                    #pragma unroll
                                    for (int k_24 = 0; k_24 < 16; k_24++) {
                                        int col_31 = k_block_12 * 32 + (tid * 4 + k_24 * 2) % 32;
                                        unsigned int w0_6 = flat_view_6[16384 + col_31 * word_stride_6 + r_6];
                                        unsigned int w1_6 = flat_view_6[16384 + (col_31 + 1) * word_stride_6 + r_6];
                                        uint32_t lo_6;
                                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_6) : "r"(w0_6), "r"(w1_6));
                                        uint32_t hi_6;
                                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_6) : "r"(w0_6), "r"(w1_6));
                                        pairs0_6[k_24] = lo_6;
                                        pairs1_6[k_24] = hi_6;
                                    }
                                    uint32_t bf16x2_abs_13;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_13) : "r"(pairs0_6[0]));
                                    unsigned int amax_pair_12 = bf16x2_abs_13;
                                    #pragma unroll
                                    for (int i_36 = 1; i_36 < 16; i_36++) {
                                        uint32_t bf16x2_abs_1_18;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_18) : "r"(pairs0_6[i_36]));
                                        uint32_t bf16x2_max_18;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_18) : "r"(amax_pair_12), "r"(bf16x2_abs_1_18));
                                        amax_pair_12 = bf16x2_max_18;
                                    }
                                    uint16_t bf16_max_12;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_12) : "h"((uint16_t)(amax_pair_12 & 65535)), "h"((uint16_t)(amax_pair_12 >> 16)));
                                    float amax_12;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_12) : "h"((uint16_t)(bf16_max_12)));
                                    float scale_12 = fmaxf(amax_12 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_70;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_70) : "f"(scale_12), "f"(scale_12));
                                    unsigned int scale_byte_12 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_70 & 255;
                                    unsigned int inverse_lane_12 = 254 - scale_byte_12 << 7;
                                    unsigned int inverse_12 = inverse_lane_12 | inverse_lane_12 << 16;
                                    unsigned int words_12[8];
                                    #pragma unroll
                                    for (int i_37 = 0; i_37 < 8; i_37++) {
                                        uint32_t bf16x2_mul_19;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(pairs0_6[i_37 * 2]), "r"(inverse_12));
                                        uint16_t e4m3x2_bf16x2_18;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_18) : "r"(bf16x2_mul_19));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_19);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_18) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_18;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(pairs0_6[i_37 * 2 + 1]), "r"(inverse_12));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_71;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_71) : "r"(bf16x2_mul_1_18));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_71) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_12[i_37] = (unsigned int)e4m3x2_bf16x2_18 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_71 << 16;
                                    }
                                    uint32_t bf16x2_abs_6_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_6_1) : "r"(pairs1_6[0]));
                                    unsigned int amax_pair_6_1 = bf16x2_abs_6_1;
                                    #pragma unroll
                                    for (int i_38 = 1; i_38 < 16; i_38++) {
                                        uint32_t bf16x2_abs_1_19;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_19) : "r"(pairs1_6[i_38]));
                                        uint32_t bf16x2_max_19;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_19) : "r"(amax_pair_6_1), "r"(bf16x2_abs_1_19));
                                        amax_pair_6_1 = bf16x2_max_19;
                                    }
                                    uint16_t bf16_max_6_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_6_1) : "h"((uint16_t)(amax_pair_6_1 & 65535)), "h"((uint16_t)(amax_pair_6_1 >> 16)));
                                    float amax_6_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_6_1) : "h"((uint16_t)(bf16_max_6_1)));
                                    float scale_6_1 = fmaxf(amax_6_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_72;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_72) : "f"(scale_6_1), "f"(scale_6_1));
                                    unsigned int scale_byte_6_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_72 & 255;
                                    unsigned int inverse_lane_6_1 = 254 - scale_byte_6_1 << 7;
                                    unsigned int inverse_6_1 = inverse_lane_6_1 | inverse_lane_6_1 << 16;
                                    unsigned int words_6_1[8];
                                    #pragma unroll
                                    for (int i_39 = 0; i_39 < 8; i_39++) {
                                        uint32_t bf16x2_mul_20;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(pairs1_6[i_39 * 2]), "r"(inverse_6_1));
                                        uint16_t e4m3x2_bf16x2_19;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_19) : "r"(bf16x2_mul_20));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_20);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_19;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(pairs1_6[i_39 * 2 + 1]), "r"(inverse_6_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_73;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_73) : "r"(bf16x2_mul_1_19));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_73) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_6_1[i_39] = (unsigned int)e4m3x2_bf16x2_19 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_73 << 16;
                                    }
                                    #pragma unroll
                                    for (int k_25 = 0; k_25 < 8; k_25++) {
                                        int col_32 = k_block_12 * 32 + (tid * 4 + k_25 * 4) % 32;
                                        smem_v33[(row0_6 * 128 + col_32) / 4] = words_12[k_25];
                                        smem_v33[(row1_6 * 128 + col_32) / 4] = words_6_1[k_25];
                                    }
                                    {
                                        uint32_t _byte_74 = static_cast<uint32_t>(scale_byte_12) & 0xFFu;
                                        uint32_t _addr_74 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base0_6 + k_block_12));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_74), "r"(_byte_74) : "memory");
                                    }
                                    {
                                        uint32_t _byte_75 = static_cast<uint32_t>(scale_byte_6_1) & 0xFFu;
                                        uint32_t _addr_75 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base1_6 + k_block_12));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_75), "r"(_byte_75) : "memory");
                                    }
                                }
                            } else {
                                int row_6_1 = tid - 128;
                                unsigned int scale_word_6 = 0;
                                #pragma unroll 1
                                for (int j_13 = 0; j_13 < 4; j_13++) {
                                    int k_block_13 = (j_13 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_6[16];
                                    #pragma unroll
                                    for (int k_26 = 0; k_26 < 16; k_26++) {
                                        int col_33 = k_block_13 * 32 + ((tid - 128) * 4 + k_26 * 2) % 32;
                                        float x0_6 = 0.0f;
                                        float x1_6 = 0.0f;
                                        pairs_6[k_26] = flat_view_6[16384 + (row_6_1 * 256 + col_33) / 2];
                                    }
                                    uint32_t bf16x2_abs_14;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_14) : "r"(pairs_6[0]));
                                    unsigned int amax_pair_13 = bf16x2_abs_14;
                                    #pragma unroll
                                    for (int i_40 = 1; i_40 < 16; i_40++) {
                                        uint32_t bf16x2_abs_1_20;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_20) : "r"(pairs_6[i_40]));
                                        uint32_t bf16x2_max_20;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_20) : "r"(amax_pair_13), "r"(bf16x2_abs_1_20));
                                        amax_pair_13 = bf16x2_max_20;
                                    }
                                    uint16_t bf16_max_13;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_13) : "h"((uint16_t)(amax_pair_13 & 65535)), "h"((uint16_t)(amax_pair_13 >> 16)));
                                    float amax_13;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_13) : "h"((uint16_t)(bf16_max_13)));
                                    float scale_13 = fmaxf(amax_13 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_76;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_76) : "f"(scale_13), "f"(scale_13));
                                    unsigned int scale_byte_13 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_76 & 255;
                                    unsigned int inverse_lane_13 = 254 - scale_byte_13 << 7;
                                    unsigned int inverse_13 = inverse_lane_13 | inverse_lane_13 << 16;
                                    unsigned int words_13[8];
                                    #pragma unroll
                                    for (int i_41 = 0; i_41 < 8; i_41++) {
                                        uint32_t bf16x2_mul_21;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(pairs_6[i_41 * 2]), "r"(inverse_13));
                                        uint16_t e4m3x2_bf16x2_20;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_20) : "r"(bf16x2_mul_21));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_21);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_20;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(pairs_6[i_41 * 2 + 1]), "r"(inverse_13));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_77;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_77) : "r"(bf16x2_mul_1_20));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_77) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_13[i_41] = (unsigned int)e4m3x2_bf16x2_20 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_77 << 16;
                                    }
                                    scale_word_6 = scale_word_6 | scale_byte_13 << (unsigned int)(k_block_13 * 8);
                                    #pragma unroll
                                    for (int k_27 = 0; k_27 < 8; k_27++) {
                                        int col_34 = k_block_13 * 32 + ((tid - 128) * 4 + k_27 * 4) % 32;
                                        smem_v29[(row_6_1 * 128 + col_34) / 4] = words_13[k_27];
                                    }
                                }
                                smem_v31[row_6_1 % 32 * 4 + row_6_1 / 32] = scale_word_6;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_6 = col_block_1 * 4 + 2;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_6 * 128, row_6, smem_v29_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_1_1 * k_tiles_1_1 + col_tile_6, smem_v31_addr);
                                tma_store_2d((&x_t_store), row_6, col_tile_6 * 128, smem_v33_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_6 * macro_tiles_1_1 + row_tile_1_1, smem_v35_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (hi_cols_1 / 128 > 1) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int r_7 = tid % 64;
                                int row0_7 = r_7 * 2;
                                int row1_7 = row0_7 + 1;
                                int half_7 = tid / 64;
                                int word_stride_7 = 128;
                                int scale_base0_7 = (row0_7 % 32 * 4 + row0_7 / 32) * 4;
                                int scale_base1_7 = (row1_7 % 32 * 4 + row1_7 / 32) * 4;
                                #pragma unroll 1
                                for (int j_14 = 0; j_14 < 2; j_14++) {
                                    int k_block_14 = (half_7 * 2 + j_14 + tid / 8) % 4;
                                    unsigned int pairs0_7[16];
                                    unsigned int pairs1_7[16];
                                    #pragma unroll
                                    for (int k_28 = 0; k_28 < 16; k_28++) {
                                        int col_35 = k_block_14 * 32 + (tid * 4 + k_28 * 2) % 32;
                                        unsigned int w0_7 = flat_view_6[16448 + col_35 * word_stride_7 + r_7];
                                        unsigned int w1_7 = flat_view_6[16448 + (col_35 + 1) * word_stride_7 + r_7];
                                        uint32_t lo_7;
                                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_7) : "r"(w0_7), "r"(w1_7));
                                        uint32_t hi_7;
                                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_7) : "r"(w0_7), "r"(w1_7));
                                        pairs0_7[k_28] = lo_7;
                                        pairs1_7[k_28] = hi_7;
                                    }
                                    uint32_t bf16x2_abs_15;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_15) : "r"(pairs0_7[0]));
                                    unsigned int amax_pair_14 = bf16x2_abs_15;
                                    #pragma unroll
                                    for (int i_42 = 1; i_42 < 16; i_42++) {
                                        uint32_t bf16x2_abs_1_21;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_21) : "r"(pairs0_7[i_42]));
                                        uint32_t bf16x2_max_21;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_21) : "r"(amax_pair_14), "r"(bf16x2_abs_1_21));
                                        amax_pair_14 = bf16x2_max_21;
                                    }
                                    uint16_t bf16_max_14;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_14) : "h"((uint16_t)(amax_pair_14 & 65535)), "h"((uint16_t)(amax_pair_14 >> 16)));
                                    float amax_14;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_14) : "h"((uint16_t)(bf16_max_14)));
                                    float scale_14 = fmaxf(amax_14 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_78;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_78) : "f"(scale_14), "f"(scale_14));
                                    unsigned int scale_byte_14 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_78 & 255;
                                    unsigned int inverse_lane_14 = 254 - scale_byte_14 << 7;
                                    unsigned int inverse_14 = inverse_lane_14 | inverse_lane_14 << 16;
                                    unsigned int words_14[8];
                                    #pragma unroll
                                    for (int i_43 = 0; i_43 < 8; i_43++) {
                                        uint32_t bf16x2_mul_22;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(pairs0_7[i_43 * 2]), "r"(inverse_14));
                                        uint16_t e4m3x2_bf16x2_21;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_21) : "r"(bf16x2_mul_22));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_22);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_21;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(pairs0_7[i_43 * 2 + 1]), "r"(inverse_14));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_79;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_79) : "r"(bf16x2_mul_1_21));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_79) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_14[i_43] = (unsigned int)e4m3x2_bf16x2_21 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_79 << 16;
                                    }
                                    uint32_t bf16x2_abs_7_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_7_1) : "r"(pairs1_7[0]));
                                    unsigned int amax_pair_7_1 = bf16x2_abs_7_1;
                                    #pragma unroll
                                    for (int i_44 = 1; i_44 < 16; i_44++) {
                                        uint32_t bf16x2_abs_1_22;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_22) : "r"(pairs1_7[i_44]));
                                        uint32_t bf16x2_max_22;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_22) : "r"(amax_pair_7_1), "r"(bf16x2_abs_1_22));
                                        amax_pair_7_1 = bf16x2_max_22;
                                    }
                                    uint16_t bf16_max_7_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_7_1) : "h"((uint16_t)(amax_pair_7_1 & 65535)), "h"((uint16_t)(amax_pair_7_1 >> 16)));
                                    float amax_7_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_7_1) : "h"((uint16_t)(bf16_max_7_1)));
                                    float scale_7_1 = fmaxf(amax_7_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_80;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_80) : "f"(scale_7_1), "f"(scale_7_1));
                                    unsigned int scale_byte_7_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_80 & 255;
                                    unsigned int inverse_lane_7_1 = 254 - scale_byte_7_1 << 7;
                                    unsigned int inverse_7_1 = inverse_lane_7_1 | inverse_lane_7_1 << 16;
                                    unsigned int words_7_1[8];
                                    #pragma unroll
                                    for (int i_45 = 0; i_45 < 8; i_45++) {
                                        uint32_t bf16x2_mul_23;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(pairs1_7[i_45 * 2]), "r"(inverse_7_1));
                                        uint16_t e4m3x2_bf16x2_22;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_22) : "r"(bf16x2_mul_23));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_23);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_22;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(pairs1_7[i_45 * 2 + 1]), "r"(inverse_7_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_81;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_81) : "r"(bf16x2_mul_1_22));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_81) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_7_1[i_45] = (unsigned int)e4m3x2_bf16x2_22 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_81 << 16;
                                    }
                                    #pragma unroll
                                    for (int k_29 = 0; k_29 < 8; k_29++) {
                                        int col_36 = k_block_14 * 32 + (tid * 4 + k_29 * 4) % 32;
                                        smem_v34[(row0_7 * 128 + col_36) / 4] = words_14[k_29];
                                        smem_v34[(row1_7 * 128 + col_36) / 4] = words_7_1[k_29];
                                    }
                                    {
                                        uint32_t _byte_82 = static_cast<uint32_t>(scale_byte_14) & 0xFFu;
                                        uint32_t _addr_82 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base0_7 + k_block_14));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_82), "r"(_byte_82) : "memory");
                                    }
                                    {
                                        uint32_t _byte_83 = static_cast<uint32_t>(scale_byte_7_1) & 0xFFu;
                                        uint32_t _addr_83 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base1_7 + k_block_14));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_83), "r"(_byte_83) : "memory");
                                    }
                                }
                            } else {
                                int row_7 = tid - 128;
                                unsigned int scale_word_7 = 0;
                                #pragma unroll 1
                                for (int j_15 = 0; j_15 < 4; j_15++) {
                                    int k_block_15 = (j_15 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_7[16];
                                    #pragma unroll
                                    for (int k_30 = 0; k_30 < 16; k_30++) {
                                        int col_37 = k_block_15 * 32 + ((tid - 128) * 4 + k_30 * 2) % 32;
                                        float x0_7 = 0.0f;
                                        float x1_7 = 0.0f;
                                        pairs_7[k_30] = flat_view_6[16448 + (row_7 * 256 + col_37) / 2];
                                    }
                                    uint32_t bf16x2_abs_16;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_16) : "r"(pairs_7[0]));
                                    unsigned int amax_pair_15 = bf16x2_abs_16;
                                    #pragma unroll
                                    for (int i_46 = 1; i_46 < 16; i_46++) {
                                        uint32_t bf16x2_abs_1_23;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_23) : "r"(pairs_7[i_46]));
                                        uint32_t bf16x2_max_23;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_23) : "r"(amax_pair_15), "r"(bf16x2_abs_1_23));
                                        amax_pair_15 = bf16x2_max_23;
                                    }
                                    uint16_t bf16_max_15;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_15) : "h"((uint16_t)(amax_pair_15 & 65535)), "h"((uint16_t)(amax_pair_15 >> 16)));
                                    float amax_15;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_15) : "h"((uint16_t)(bf16_max_15)));
                                    float scale_15 = fmaxf(amax_15 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_84;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_84) : "f"(scale_15), "f"(scale_15));
                                    unsigned int scale_byte_15 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_84 & 255;
                                    unsigned int inverse_lane_15 = 254 - scale_byte_15 << 7;
                                    unsigned int inverse_15 = inverse_lane_15 | inverse_lane_15 << 16;
                                    unsigned int words_15[8];
                                    #pragma unroll
                                    for (int i_47 = 0; i_47 < 8; i_47++) {
                                        uint32_t bf16x2_mul_24;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(pairs_7[i_47 * 2]), "r"(inverse_15));
                                        uint16_t e4m3x2_bf16x2_23;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_23) : "r"(bf16x2_mul_24));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_24);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_23;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(pairs_7[i_47 * 2 + 1]), "r"(inverse_15));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_85;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_85) : "r"(bf16x2_mul_1_23));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_85) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_15[i_47] = (unsigned int)e4m3x2_bf16x2_23 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_85 << 16;
                                    }
                                    scale_word_7 = scale_word_7 | scale_byte_15 << (unsigned int)(k_block_15 * 8);
                                    #pragma unroll
                                    for (int k_31 = 0; k_31 < 8; k_31++) {
                                        int col_38 = k_block_15 * 32 + ((tid - 128) * 4 + k_31 * 4) % 32;
                                        smem_v30[(row_7 * 128 + col_38) / 4] = words_15[k_31];
                                    }
                                }
                                smem_v32[row_7 % 32 * 4 + row_7 / 32] = scale_word_7;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_7 = col_block_1 * 4 + 2 + 1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_7 * 128, row_6, smem_v30_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_1_1 * k_tiles_1_1 + col_tile_7, smem_v32_addr);
                                tma_store_2d((&x_t_store), row_6, col_tile_7 * 128, smem_v34_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_7 * macro_tiles_1_1 + row_tile_1_1, smem_v36_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_x)) + ((macro_offset_4 + row_6) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                    __syncthreads();
                    task_7 = next_task_1;
                }
                int col_blocks_0_1 = (hidden + 511) / 512;
                int macro_offset_0 = (macro + 1) * macro_size;
                int macro_tokens_0 = ((macro_size) < (tokens - macro_offset_0) ? (macro_size) : (tokens - macro_offset_0));
                int tasks_0 = macro_tokens_0 / 128 * col_blocks_0_1;
                int task_0 = cluster * 2 + cta_rank_0;
                if (task_0 < tasks_0) {
                    int min_1_1 = ((256) < (hidden - task_0 % col_blocks_0_1 * 512) ? (256) : (hidden - task_0 % col_blocks_0_1 * 512));
                    int peer_8 = -1;
                    int peer_token_6 = -1;
                    if (tid < 128) {
                        peer_8 = schedule_rank[macro_offset_0 + task_0 / col_blocks_0_1 * 128 + tid];
                        peer_token_6 = schedule_token[macro_offset_0 + task_0 / col_blocks_0_1 * 128 + tid];
                    }
                    uint32_t cta_count_6 = __syncthreads_count(peer_8 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_6 * (unsigned int)(min_1_1 * 2));
                    }
                    __syncthreads();
                    if (peer_8 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_8])) + ((unsigned long long)((unsigned long long)(peer_token_6 / topk) * (unsigned long long)hidden + (unsigned long long)(task_0 % col_blocks_0_1 * 512)) * (unsigned long long)2)), min_1_1 * 2, dispatch_arrived_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_6 = 0; vec_6 < 32; vec_6++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (tid * 512 + vec_6 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                while (task_0 < tasks_0) {
                    int row_8 = task_0 / col_blocks_0_1 * 128;
                    int col_block_2 = task_0 % col_blocks_0_1;
                    int chunk_cols_9 = ((512) < (hidden - col_block_2 * 512) ? (512) : (hidden - col_block_2 * 512));
                    int lo_cols_2 = ((256) < (chunk_cols_9) ? (256) : (chunk_cols_9));
                    int hi_cols_2 = chunk_cols_9 - lo_cols_2;
                    if (hi_cols_2 > 0) {
                        int peer_9 = -1;
                        int peer_token_7 = -1;
                        if (tid < 128) {
                            peer_9 = schedule_rank[macro_offset_0 + row_8 + tid];
                            peer_token_7 = schedule_token[macro_offset_0 + row_8 + tid];
                        }
                        uint32_t cta_count_7 = __syncthreads_count(peer_9 >= 0);
                        if (tid == 0) {
                            mbarrier_arrive_expect_tx(dispatch_arrived_hi_addr, cta_count_7 * (unsigned int)(hi_cols_2 * 2));
                        }
                        __syncthreads();
                        if (peer_9 >= 0) {
                            cp_async_bulk_gmem2smem(smem_v39_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_9])) + ((unsigned long long)((unsigned long long)(peer_token_7 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_2 * 512) + 256) * (unsigned long long)2)), hi_cols_2 * 2, dispatch_arrived_hi_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_7 = 0; vec_7 < 32; vec_7++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (65536 + tid * 512 + vec_7 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                    int weight_peer_1 = -1;
                    if (tid < 128) {
                        weight_peer_1 = schedule_rank[macro_offset_0 + row_8 + tid];
                        flat_view_7[tid] = ((weight_peer_1 >= 0) ? weights[row_8 + tid] : 0.0f);
                    }
                    int _parity_dispatch_arrived_2 = (int)((unsigned int)dispatch_seq % 2);
                    mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived_2);
                    dispatch_seq += 1;
                    int row_tile_2 = row_8 / 128;
                    int k_tiles_2 = hidden / 128;
                    int macro_tiles_2 = macro_size / 128;
                    if (lo_cols_2 / 128 > 0) {
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        __syncthreads();
                        if (tid < 128) {
                            int r_8 = tid % 64;
                            int row0_8 = r_8 * 2;
                            int row1_8 = row0_8 + 1;
                            int half_8 = tid / 64;
                            int word_stride_8 = 128;
                            int scale_base0_8 = (row0_8 % 32 * 4 + row0_8 / 32) * 4;
                            int scale_base1_8 = (row1_8 % 32 * 4 + row1_8 / 32) * 4;
                            #pragma unroll 1
                            for (int j_16 = 0; j_16 < 2; j_16++) {
                                int k_block_16 = (half_8 * 2 + j_16 + tid / 8) % 4;
                                unsigned int pairs0_8[16];
                                unsigned int pairs1_8[16];
                                #pragma unroll
                                for (int k_32 = 0; k_32 < 16; k_32++) {
                                    int col_39 = k_block_16 * 32 + (tid * 4 + k_32 * 2) % 32;
                                    unsigned int w0_8 = flat_view_6[col_39 * word_stride_8 + r_8];
                                    unsigned int w1_8 = flat_view_6[(col_39 + 1) * word_stride_8 + r_8];
                                    uint32_t lo_8;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_8) : "r"(w0_8), "r"(w1_8));
                                    uint32_t hi_8;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_8) : "r"(w0_8), "r"(w1_8));
                                    float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_8));
                                    pairs0_8[k_32] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_9.x * flat_view_7[col_39], cvt_f32_9.y * flat_view_7[col_39 + 1])));
                                    float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_8));
                                    pairs1_8[k_32] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_4.x * flat_view_7[col_39], cvt_f32_1_4.y * flat_view_7[col_39 + 1])));
                                }
                                uint32_t bf16x2_abs_17;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_17) : "r"(pairs0_8[0]));
                                unsigned int amax_pair_16 = bf16x2_abs_17;
                                #pragma unroll
                                for (int i_48 = 1; i_48 < 16; i_48++) {
                                    uint32_t bf16x2_abs_1_24;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_24) : "r"(pairs0_8[i_48]));
                                    uint32_t bf16x2_max_24;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_24) : "r"(amax_pair_16), "r"(bf16x2_abs_1_24));
                                    amax_pair_16 = bf16x2_max_24;
                                }
                                uint16_t bf16_max_16;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_16) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                float amax_16;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_16) : "h"((uint16_t)(bf16_max_16)));
                                float scale_16 = fmaxf(amax_16 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_86;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_86) : "f"(scale_16), "f"(scale_16));
                                unsigned int scale_byte_16 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_86 & 255;
                                unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                unsigned int words_16[8];
                                #pragma unroll
                                for (int i_49 = 0; i_49 < 8; i_49++) {
                                    uint32_t bf16x2_mul_25;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(pairs0_8[i_49 * 2]), "r"(inverse_16));
                                    uint16_t e4m3x2_bf16x2_24;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_24) : "r"(bf16x2_mul_25));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_25);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_24) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_24;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(pairs0_8[i_49 * 2 + 1]), "r"(inverse_16));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_87;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_87) : "r"(bf16x2_mul_1_24));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_87) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_16[i_49] = (unsigned int)e4m3x2_bf16x2_24 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_87 << 16;
                                }
                                uint32_t bf16x2_abs_8_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_8_1) : "r"(pairs1_8[0]));
                                unsigned int amax_pair_8_1 = bf16x2_abs_8_1;
                                #pragma unroll
                                for (int i_50 = 1; i_50 < 16; i_50++) {
                                    uint32_t bf16x2_abs_1_25;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_25) : "r"(pairs1_8[i_50]));
                                    uint32_t bf16x2_max_25;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_25) : "r"(amax_pair_8_1), "r"(bf16x2_abs_1_25));
                                    amax_pair_8_1 = bf16x2_max_25;
                                }
                                uint16_t bf16_max_8_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_8_1) : "h"((uint16_t)(amax_pair_8_1 & 65535)), "h"((uint16_t)(amax_pair_8_1 >> 16)));
                                float amax_8_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8_1) : "h"((uint16_t)(bf16_max_8_1)));
                                float scale_8_1 = fmaxf(amax_8_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_88;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_88) : "f"(scale_8_1), "f"(scale_8_1));
                                unsigned int scale_byte_8_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_88 & 255;
                                unsigned int inverse_lane_8_1 = 254 - scale_byte_8_1 << 7;
                                unsigned int inverse_8_1 = inverse_lane_8_1 | inverse_lane_8_1 << 16;
                                unsigned int words_8_1[8];
                                #pragma unroll
                                for (int i_51 = 0; i_51 < 8; i_51++) {
                                    uint32_t bf16x2_mul_26;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(pairs1_8[i_51 * 2]), "r"(inverse_8_1));
                                    uint16_t e4m3x2_bf16x2_25;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_25) : "r"(bf16x2_mul_26));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_26);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_25;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(pairs1_8[i_51 * 2 + 1]), "r"(inverse_8_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_89;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_89) : "r"(bf16x2_mul_1_25));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_89) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_8_1[i_51] = (unsigned int)e4m3x2_bf16x2_25 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_89 << 16;
                                }
                                #pragma unroll
                                for (int k_33 = 0; k_33 < 8; k_33++) {
                                    int col_40 = k_block_16 * 32 + (tid * 4 + k_33 * 4) % 32;
                                    smem_v33[(row0_8 * 128 + col_40) / 4] = words_16[k_33];
                                    smem_v33[(row1_8 * 128 + col_40) / 4] = words_8_1[k_33];
                                }
                                {
                                    uint32_t _byte_90 = static_cast<uint32_t>(scale_byte_16) & 0xFFu;
                                    uint32_t _addr_90 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base0_8 + k_block_16));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_90), "r"(_byte_90) : "memory");
                                }
                                {
                                    uint32_t _byte_91 = static_cast<uint32_t>(scale_byte_8_1) & 0xFFu;
                                    uint32_t _addr_91 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base1_8 + k_block_16));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_91), "r"(_byte_91) : "memory");
                                }
                            }
                        } else {
                            int row_8_1 = tid - 128;
                            unsigned int scale_word_8 = 0;
                            #pragma unroll 1
                            for (int j_17 = 0; j_17 < 4; j_17++) {
                                int k_block_17 = (j_17 + (tid - 128) / 8) % 4;
                                unsigned int pairs_8[16];
                                #pragma unroll
                                for (int k_34 = 0; k_34 < 16; k_34++) {
                                    int col_41 = k_block_17 * 32 + ((tid - 128) * 4 + k_34 * 2) % 32;
                                    float x0_8 = 0.0f;
                                    float x1_8 = 0.0f;
                                    float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[(row_8_1 * 256 + col_41) / 2]));
                                    pairs_8[k_34] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_10.x * flat_view_7[row_8_1], cvt_f32_10.y * flat_view_7[row_8_1])));
                                }
                                uint32_t bf16x2_abs_18;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_18) : "r"(pairs_8[0]));
                                unsigned int amax_pair_17 = bf16x2_abs_18;
                                #pragma unroll
                                for (int i_52 = 1; i_52 < 16; i_52++) {
                                    uint32_t bf16x2_abs_1_26;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_26) : "r"(pairs_8[i_52]));
                                    uint32_t bf16x2_max_26;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_26) : "r"(amax_pair_17), "r"(bf16x2_abs_1_26));
                                    amax_pair_17 = bf16x2_max_26;
                                }
                                uint16_t bf16_max_17;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_17) : "h"((uint16_t)(amax_pair_17 & 65535)), "h"((uint16_t)(amax_pair_17 >> 16)));
                                float amax_17;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_17) : "h"((uint16_t)(bf16_max_17)));
                                float scale_17 = fmaxf(amax_17 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_92;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_92) : "f"(scale_17), "f"(scale_17));
                                unsigned int scale_byte_17 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_92 & 255;
                                unsigned int inverse_lane_17 = 254 - scale_byte_17 << 7;
                                unsigned int inverse_17 = inverse_lane_17 | inverse_lane_17 << 16;
                                unsigned int words_17[8];
                                #pragma unroll
                                for (int i_53 = 0; i_53 < 8; i_53++) {
                                    uint32_t bf16x2_mul_27;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(pairs_8[i_53 * 2]), "r"(inverse_17));
                                    uint16_t e4m3x2_bf16x2_26;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_26) : "r"(bf16x2_mul_27));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_27);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_26;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(pairs_8[i_53 * 2 + 1]), "r"(inverse_17));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_93;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_93) : "r"(bf16x2_mul_1_26));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_93) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_17[i_53] = (unsigned int)e4m3x2_bf16x2_26 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_93 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_17 << (unsigned int)(k_block_17 * 8);
                                #pragma unroll
                                for (int k_35 = 0; k_35 < 8; k_35++) {
                                    int col_42 = k_block_17 * 32 + ((tid - 128) * 4 + k_35 * 4) % 32;
                                    smem_v29[(row_8_1 * 128 + col_42) / 4] = words_17[k_35];
                                }
                            }
                            smem_v31[row_8_1 % 32 * 4 + row_8_1 / 32] = scale_word_8;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_8 = col_block_2 * 4;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&dy_q_store), col_tile_8 * 128, row_8, smem_v29_addr);
                            tma_store_3d((&dy_sc_store), 0, 0, row_tile_2 * k_tiles_2 + col_tile_8, smem_v31_addr);
                            tma_store_2d((&dy_t_store), row_8, col_tile_8 * 128, smem_v33_addr);
                            tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_8 * macro_tiles_2 + row_tile_2, smem_v35_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (lo_cols_2 / 128 > 1) {
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        __syncthreads();
                        if (tid < 128) {
                            int r_9 = tid % 64;
                            int row0_9 = r_9 * 2;
                            int row1_9 = row0_9 + 1;
                            int half_9 = tid / 64;
                            int word_stride_9 = 128;
                            int scale_base0_9 = (row0_9 % 32 * 4 + row0_9 / 32) * 4;
                            int scale_base1_9 = (row1_9 % 32 * 4 + row1_9 / 32) * 4;
                            #pragma unroll 1
                            for (int j_18 = 0; j_18 < 2; j_18++) {
                                int k_block_18 = (half_9 * 2 + j_18 + tid / 8) % 4;
                                unsigned int pairs0_9[16];
                                unsigned int pairs1_9[16];
                                #pragma unroll
                                for (int k_36 = 0; k_36 < 16; k_36++) {
                                    int col_43 = k_block_18 * 32 + (tid * 4 + k_36 * 2) % 32;
                                    unsigned int w0_9 = flat_view_6[64 + col_43 * word_stride_9 + r_9];
                                    unsigned int w1_9 = flat_view_6[64 + (col_43 + 1) * word_stride_9 + r_9];
                                    uint32_t lo_9;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_9) : "r"(w0_9), "r"(w1_9));
                                    uint32_t hi_9;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_9) : "r"(w0_9), "r"(w1_9));
                                    float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_9));
                                    pairs0_9[k_36] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_11.x * flat_view_7[col_43], cvt_f32_11.y * flat_view_7[col_43 + 1])));
                                    float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_9));
                                    pairs1_9[k_36] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_5.x * flat_view_7[col_43], cvt_f32_1_5.y * flat_view_7[col_43 + 1])));
                                }
                                uint32_t bf16x2_abs_19;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_19) : "r"(pairs0_9[0]));
                                unsigned int amax_pair_18 = bf16x2_abs_19;
                                #pragma unroll
                                for (int i_54 = 1; i_54 < 16; i_54++) {
                                    uint32_t bf16x2_abs_1_27;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_27) : "r"(pairs0_9[i_54]));
                                    uint32_t bf16x2_max_27;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_27) : "r"(amax_pair_18), "r"(bf16x2_abs_1_27));
                                    amax_pair_18 = bf16x2_max_27;
                                }
                                uint16_t bf16_max_18;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_18) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                float amax_18;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18) : "h"((uint16_t)(bf16_max_18)));
                                float scale_18 = fmaxf(amax_18 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_94;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_94) : "f"(scale_18), "f"(scale_18));
                                unsigned int scale_byte_18 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_94 & 255;
                                unsigned int inverse_lane_18 = 254 - scale_byte_18 << 7;
                                unsigned int inverse_18 = inverse_lane_18 | inverse_lane_18 << 16;
                                unsigned int words_18[8];
                                #pragma unroll
                                for (int i_55 = 0; i_55 < 8; i_55++) {
                                    uint32_t bf16x2_mul_28;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(pairs0_9[i_55 * 2]), "r"(inverse_18));
                                    uint16_t e4m3x2_bf16x2_27;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_27) : "r"(bf16x2_mul_28));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_28);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_27;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(pairs0_9[i_55 * 2 + 1]), "r"(inverse_18));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_95;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_95) : "r"(bf16x2_mul_1_27));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_95) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_18[i_55] = (unsigned int)e4m3x2_bf16x2_27 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_95 << 16;
                                }
                                uint32_t bf16x2_abs_9_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_9_1) : "r"(pairs1_9[0]));
                                unsigned int amax_pair_9_1 = bf16x2_abs_9_1;
                                #pragma unroll
                                for (int i_56 = 1; i_56 < 16; i_56++) {
                                    uint32_t bf16x2_abs_1_28;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_28) : "r"(pairs1_9[i_56]));
                                    uint32_t bf16x2_max_28;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_28) : "r"(amax_pair_9_1), "r"(bf16x2_abs_1_28));
                                    amax_pair_9_1 = bf16x2_max_28;
                                }
                                uint16_t bf16_max_9_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_9_1) : "h"((uint16_t)(amax_pair_9_1 & 65535)), "h"((uint16_t)(amax_pair_9_1 >> 16)));
                                float amax_9_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_9_1) : "h"((uint16_t)(bf16_max_9_1)));
                                float scale_9_1 = fmaxf(amax_9_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_96;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_96) : "f"(scale_9_1), "f"(scale_9_1));
                                unsigned int scale_byte_9_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_96 & 255;
                                unsigned int inverse_lane_9_1 = 254 - scale_byte_9_1 << 7;
                                unsigned int inverse_9_1 = inverse_lane_9_1 | inverse_lane_9_1 << 16;
                                unsigned int words_9_1[8];
                                #pragma unroll
                                for (int i_57 = 0; i_57 < 8; i_57++) {
                                    uint32_t bf16x2_mul_29;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(pairs1_9[i_57 * 2]), "r"(inverse_9_1));
                                    uint16_t e4m3x2_bf16x2_28;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_28) : "r"(bf16x2_mul_29));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_29);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_28;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(pairs1_9[i_57 * 2 + 1]), "r"(inverse_9_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_97;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_97) : "r"(bf16x2_mul_1_28));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_97) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_9_1[i_57] = (unsigned int)e4m3x2_bf16x2_28 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_97 << 16;
                                }
                                #pragma unroll
                                for (int k_37 = 0; k_37 < 8; k_37++) {
                                    int col_44 = k_block_18 * 32 + (tid * 4 + k_37 * 4) % 32;
                                    smem_v34[(row0_9 * 128 + col_44) / 4] = words_18[k_37];
                                    smem_v34[(row1_9 * 128 + col_44) / 4] = words_9_1[k_37];
                                }
                                {
                                    uint32_t _byte_98 = static_cast<uint32_t>(scale_byte_18) & 0xFFu;
                                    uint32_t _addr_98 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base0_9 + k_block_18));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_98), "r"(_byte_98) : "memory");
                                }
                                {
                                    uint32_t _byte_99 = static_cast<uint32_t>(scale_byte_9_1) & 0xFFu;
                                    uint32_t _addr_99 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base1_9 + k_block_18));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_99), "r"(_byte_99) : "memory");
                                }
                            }
                        } else {
                            int row_9 = tid - 128;
                            unsigned int scale_word_9 = 0;
                            #pragma unroll 1
                            for (int j_19 = 0; j_19 < 4; j_19++) {
                                int k_block_19 = (j_19 + (tid - 128) / 8) % 4;
                                unsigned int pairs_9[16];
                                #pragma unroll
                                for (int k_38 = 0; k_38 < 16; k_38++) {
                                    int col_45 = k_block_19 * 32 + ((tid - 128) * 4 + k_38 * 2) % 32;
                                    float x0_9 = 0.0f;
                                    float x1_9 = 0.0f;
                                    float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[64 + (row_9 * 256 + col_45) / 2]));
                                    pairs_9[k_38] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_12.x * flat_view_7[row_9], cvt_f32_12.y * flat_view_7[row_9])));
                                }
                                uint32_t bf16x2_abs_20;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_20) : "r"(pairs_9[0]));
                                unsigned int amax_pair_19 = bf16x2_abs_20;
                                #pragma unroll
                                for (int i_58 = 1; i_58 < 16; i_58++) {
                                    uint32_t bf16x2_abs_1_29;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_29) : "r"(pairs_9[i_58]));
                                    uint32_t bf16x2_max_29;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_29) : "r"(amax_pair_19), "r"(bf16x2_abs_1_29));
                                    amax_pair_19 = bf16x2_max_29;
                                }
                                uint16_t bf16_max_19;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_19) : "h"((uint16_t)(amax_pair_19 & 65535)), "h"((uint16_t)(amax_pair_19 >> 16)));
                                float amax_19;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_19) : "h"((uint16_t)(bf16_max_19)));
                                float scale_19 = fmaxf(amax_19 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_100;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_100) : "f"(scale_19), "f"(scale_19));
                                unsigned int scale_byte_19 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_100 & 255;
                                unsigned int inverse_lane_19 = 254 - scale_byte_19 << 7;
                                unsigned int inverse_19 = inverse_lane_19 | inverse_lane_19 << 16;
                                unsigned int words_19[8];
                                #pragma unroll
                                for (int i_59 = 0; i_59 < 8; i_59++) {
                                    uint32_t bf16x2_mul_30;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(pairs_9[i_59 * 2]), "r"(inverse_19));
                                    uint16_t e4m3x2_bf16x2_29;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_29) : "r"(bf16x2_mul_30));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_30);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_29;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(pairs_9[i_59 * 2 + 1]), "r"(inverse_19));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_101;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_101) : "r"(bf16x2_mul_1_29));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_101) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_19[i_59] = (unsigned int)e4m3x2_bf16x2_29 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_101 << 16;
                                }
                                scale_word_9 = scale_word_9 | scale_byte_19 << (unsigned int)(k_block_19 * 8);
                                #pragma unroll
                                for (int k_39 = 0; k_39 < 8; k_39++) {
                                    int col_46 = k_block_19 * 32 + ((tid - 128) * 4 + k_39 * 4) % 32;
                                    smem_v30[(row_9 * 128 + col_46) / 4] = words_19[k_39];
                                }
                            }
                            smem_v32[row_9 % 32 * 4 + row_9 / 32] = scale_word_9;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_9 = col_block_2 * 4 + 1;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&dy_q_store), col_tile_9 * 128, row_8, smem_v30_addr);
                            tma_store_3d((&dy_sc_store), 0, 0, row_tile_2 * k_tiles_2 + col_tile_9, smem_v32_addr);
                            tma_store_2d((&dy_t_store), row_8, col_tile_9 * 128, smem_v34_addr);
                            tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_9 * macro_tiles_2 + row_tile_2, smem_v36_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    int next_task_2 = task_0 + comm_sms;
                    if (next_task_2 < tasks_0) {
                        int min_2_1 = ((256) < (hidden - next_task_2 % col_blocks_0_1 * 512) ? (256) : (hidden - next_task_2 % col_blocks_0_1 * 512));
                        int peer_10 = -1;
                        int peer_token_8 = -1;
                        if (tid < 128) {
                            peer_10 = schedule_rank[macro_offset_0 + next_task_2 / col_blocks_0_1 * 128 + tid];
                            peer_token_8 = schedule_token[macro_offset_0 + next_task_2 / col_blocks_0_1 * 128 + tid];
                        }
                        uint32_t cta_count_8 = __syncthreads_count(peer_10 >= 0);
                        if (tid == 0) {
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_8 * (unsigned int)(min_2_1 * 2));
                        }
                        __syncthreads();
                        if (peer_10 >= 0) {
                            cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_10])) + ((unsigned long long)((unsigned long long)(peer_token_8 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task_2 % col_blocks_0_1 * 512)) * (unsigned long long)2)), min_2_1 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_8 = 0; vec_8 < 32; vec_8++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_6) + (tid * 512 + vec_8 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                    if (hi_cols_2 > 0) {
                        int _parity_dispatch_arrived_hi_2 = (int)((unsigned int)dispatch_seq_hi % 2);
                        mbarrier_wait(dispatch_arrived_hi_addr, _parity_dispatch_arrived_hi_2);
                        dispatch_seq_hi += 1;
                        int row_tile_2_1 = row_8 / 128;
                        int k_tiles_2_1 = hidden / 128;
                        int macro_tiles_2_1 = macro_size / 128;
                        if (hi_cols_2 / 128 > 0) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int r_10 = tid % 64;
                                int row0_10 = r_10 * 2;
                                int row1_10 = row0_10 + 1;
                                int half_10 = tid / 64;
                                int word_stride_10 = 128;
                                int scale_base0_10 = (row0_10 % 32 * 4 + row0_10 / 32) * 4;
                                int scale_base1_10 = (row1_10 % 32 * 4 + row1_10 / 32) * 4;
                                #pragma unroll 1
                                for (int j_20 = 0; j_20 < 2; j_20++) {
                                    int k_block_20 = (half_10 * 2 + j_20 + tid / 8) % 4;
                                    unsigned int pairs0_10[16];
                                    unsigned int pairs1_10[16];
                                    #pragma unroll
                                    for (int k_40 = 0; k_40 < 16; k_40++) {
                                        int col_47 = k_block_20 * 32 + (tid * 4 + k_40 * 2) % 32;
                                        unsigned int w0_10 = flat_view_6[16384 + col_47 * word_stride_10 + r_10];
                                        unsigned int w1_10 = flat_view_6[16384 + (col_47 + 1) * word_stride_10 + r_10];
                                        uint32_t lo_10;
                                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_10) : "r"(w0_10), "r"(w1_10));
                                        uint32_t hi_10;
                                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_10) : "r"(w0_10), "r"(w1_10));
                                        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_10));
                                        pairs0_10[k_40] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_13.x * flat_view_7[col_47], cvt_f32_13.y * flat_view_7[col_47 + 1])));
                                        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_10));
                                        pairs1_10[k_40] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_6.x * flat_view_7[col_47], cvt_f32_1_6.y * flat_view_7[col_47 + 1])));
                                    }
                                    uint32_t bf16x2_abs_21;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_21) : "r"(pairs0_10[0]));
                                    unsigned int amax_pair_20 = bf16x2_abs_21;
                                    #pragma unroll
                                    for (int i_60 = 1; i_60 < 16; i_60++) {
                                        uint32_t bf16x2_abs_1_30;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_30) : "r"(pairs0_10[i_60]));
                                        uint32_t bf16x2_max_30;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_30) : "r"(amax_pair_20), "r"(bf16x2_abs_1_30));
                                        amax_pair_20 = bf16x2_max_30;
                                    }
                                    uint16_t bf16_max_20;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_20) : "h"((uint16_t)(amax_pair_20 & 65535)), "h"((uint16_t)(amax_pair_20 >> 16)));
                                    float amax_20;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_20) : "h"((uint16_t)(bf16_max_20)));
                                    float scale_20 = fmaxf(amax_20 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_102;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_102) : "f"(scale_20), "f"(scale_20));
                                    unsigned int scale_byte_20 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_102 & 255;
                                    unsigned int inverse_lane_20 = 254 - scale_byte_20 << 7;
                                    unsigned int inverse_20 = inverse_lane_20 | inverse_lane_20 << 16;
                                    unsigned int words_20[8];
                                    #pragma unroll
                                    for (int i_61 = 0; i_61 < 8; i_61++) {
                                        uint32_t bf16x2_mul_31;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(pairs0_10[i_61 * 2]), "r"(inverse_20));
                                        uint16_t e4m3x2_bf16x2_30;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_30) : "r"(bf16x2_mul_31));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_31);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_30;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(pairs0_10[i_61 * 2 + 1]), "r"(inverse_20));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_103;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "r"(bf16x2_mul_1_30));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_20[i_61] = (unsigned int)e4m3x2_bf16x2_30 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_103 << 16;
                                    }
                                    uint32_t bf16x2_abs_10_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_10_1) : "r"(pairs1_10[0]));
                                    unsigned int amax_pair_10_1 = bf16x2_abs_10_1;
                                    #pragma unroll
                                    for (int i_62 = 1; i_62 < 16; i_62++) {
                                        uint32_t bf16x2_abs_1_31;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_31) : "r"(pairs1_10[i_62]));
                                        uint32_t bf16x2_max_31;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_31) : "r"(amax_pair_10_1), "r"(bf16x2_abs_1_31));
                                        amax_pair_10_1 = bf16x2_max_31;
                                    }
                                    uint16_t bf16_max_10_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_10_1) : "h"((uint16_t)(amax_pair_10_1 & 65535)), "h"((uint16_t)(amax_pair_10_1 >> 16)));
                                    float amax_10_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10_1) : "h"((uint16_t)(bf16_max_10_1)));
                                    float scale_10_1 = fmaxf(amax_10_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_104;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_104) : "f"(scale_10_1), "f"(scale_10_1));
                                    unsigned int scale_byte_10_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_104 & 255;
                                    unsigned int inverse_lane_10_1 = 254 - scale_byte_10_1 << 7;
                                    unsigned int inverse_10_1 = inverse_lane_10_1 | inverse_lane_10_1 << 16;
                                    unsigned int words_10_1[8];
                                    #pragma unroll
                                    for (int i_63 = 0; i_63 < 8; i_63++) {
                                        uint32_t bf16x2_mul_32;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_32) : "r"(pairs1_10[i_63 * 2]), "r"(inverse_10_1));
                                        uint16_t e4m3x2_bf16x2_31;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_31) : "r"(bf16x2_mul_32));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_32);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_31;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_31) : "r"(pairs1_10[i_63 * 2 + 1]), "r"(inverse_10_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_105;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_105) : "r"(bf16x2_mul_1_31));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_31);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_105) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_10_1[i_63] = (unsigned int)e4m3x2_bf16x2_31 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_105 << 16;
                                    }
                                    #pragma unroll
                                    for (int k_41 = 0; k_41 < 8; k_41++) {
                                        int col_48 = k_block_20 * 32 + (tid * 4 + k_41 * 4) % 32;
                                        smem_v33[(row0_10 * 128 + col_48) / 4] = words_20[k_41];
                                        smem_v33[(row1_10 * 128 + col_48) / 4] = words_10_1[k_41];
                                    }
                                    {
                                        uint32_t _byte_106 = static_cast<uint32_t>(scale_byte_20) & 0xFFu;
                                        uint32_t _addr_106 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base0_10 + k_block_20));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_106), "r"(_byte_106) : "memory");
                                    }
                                    {
                                        uint32_t _byte_107 = static_cast<uint32_t>(scale_byte_10_1) & 0xFFu;
                                        uint32_t _addr_107 = static_cast<uint32_t>(smem_v35_addr + (unsigned int)(scale_base1_10 + k_block_20));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_107), "r"(_byte_107) : "memory");
                                    }
                                }
                            } else {
                                int row_10 = tid - 128;
                                unsigned int scale_word_10 = 0;
                                #pragma unroll 1
                                for (int j_21 = 0; j_21 < 4; j_21++) {
                                    int k_block_21 = (j_21 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_10[16];
                                    #pragma unroll
                                    for (int k_42 = 0; k_42 < 16; k_42++) {
                                        int col_49 = k_block_21 * 32 + ((tid - 128) * 4 + k_42 * 2) % 32;
                                        float x0_10 = 0.0f;
                                        float x1_10 = 0.0f;
                                        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[16384 + (row_10 * 256 + col_49) / 2]));
                                        pairs_10[k_42] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_14.x * flat_view_7[row_10], cvt_f32_14.y * flat_view_7[row_10])));
                                    }
                                    uint32_t bf16x2_abs_22;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_22) : "r"(pairs_10[0]));
                                    unsigned int amax_pair_21 = bf16x2_abs_22;
                                    #pragma unroll
                                    for (int i_64 = 1; i_64 < 16; i_64++) {
                                        uint32_t bf16x2_abs_1_32;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_32) : "r"(pairs_10[i_64]));
                                        uint32_t bf16x2_max_32;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_32) : "r"(amax_pair_21), "r"(bf16x2_abs_1_32));
                                        amax_pair_21 = bf16x2_max_32;
                                    }
                                    uint16_t bf16_max_21;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_21) : "h"((uint16_t)(amax_pair_21 & 65535)), "h"((uint16_t)(amax_pair_21 >> 16)));
                                    float amax_21;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_21) : "h"((uint16_t)(bf16_max_21)));
                                    float scale_21 = fmaxf(amax_21 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_108;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_108) : "f"(scale_21), "f"(scale_21));
                                    unsigned int scale_byte_21 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_108 & 255;
                                    unsigned int inverse_lane_21 = 254 - scale_byte_21 << 7;
                                    unsigned int inverse_21 = inverse_lane_21 | inverse_lane_21 << 16;
                                    unsigned int words_21[8];
                                    #pragma unroll
                                    for (int i_65 = 0; i_65 < 8; i_65++) {
                                        uint32_t bf16x2_mul_33;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_33) : "r"(pairs_10[i_65 * 2]), "r"(inverse_21));
                                        uint16_t e4m3x2_bf16x2_32;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_32) : "r"(bf16x2_mul_33));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_33);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_32) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_32;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_32) : "r"(pairs_10[i_65 * 2 + 1]), "r"(inverse_21));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_109;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_109) : "r"(bf16x2_mul_1_32));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_32);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_109) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_21[i_65] = (unsigned int)e4m3x2_bf16x2_32 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_109 << 16;
                                    }
                                    scale_word_10 = scale_word_10 | scale_byte_21 << (unsigned int)(k_block_21 * 8);
                                    #pragma unroll
                                    for (int k_43 = 0; k_43 < 8; k_43++) {
                                        int col_50 = k_block_21 * 32 + ((tid - 128) * 4 + k_43 * 4) % 32;
                                        smem_v29[(row_10 * 128 + col_50) / 4] = words_21[k_43];
                                    }
                                }
                                smem_v31[row_10 % 32 * 4 + row_10 / 32] = scale_word_10;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_10 = col_block_2 * 4 + 2;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&dy_q_store), col_tile_10 * 128, row_8, smem_v29_addr);
                                tma_store_3d((&dy_sc_store), 0, 0, row_tile_2_1 * k_tiles_2_1 + col_tile_10, smem_v31_addr);
                                tma_store_2d((&dy_t_store), row_8, col_tile_10 * 128, smem_v33_addr);
                                tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_10 * macro_tiles_2_1 + row_tile_2_1, smem_v35_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (hi_cols_2 / 128 > 1) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int r_11 = tid % 64;
                                int row0_11 = r_11 * 2;
                                int row1_11 = row0_11 + 1;
                                int half_11 = tid / 64;
                                int word_stride_11 = 128;
                                int scale_base0_11 = (row0_11 % 32 * 4 + row0_11 / 32) * 4;
                                int scale_base1_11 = (row1_11 % 32 * 4 + row1_11 / 32) * 4;
                                #pragma unroll 1
                                for (int j_22 = 0; j_22 < 2; j_22++) {
                                    int k_block_22 = (half_11 * 2 + j_22 + tid / 8) % 4;
                                    unsigned int pairs0_11[16];
                                    unsigned int pairs1_11[16];
                                    #pragma unroll
                                    for (int k_44 = 0; k_44 < 16; k_44++) {
                                        int col_51 = k_block_22 * 32 + (tid * 4 + k_44 * 2) % 32;
                                        unsigned int w0_11 = flat_view_6[16448 + col_51 * word_stride_11 + r_11];
                                        unsigned int w1_11 = flat_view_6[16448 + (col_51 + 1) * word_stride_11 + r_11];
                                        uint32_t lo_11;
                                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_11) : "r"(w0_11), "r"(w1_11));
                                        uint32_t hi_11;
                                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_11) : "r"(w0_11), "r"(w1_11));
                                        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(lo_11));
                                        pairs0_11[k_44] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_15.x * flat_view_7[col_51], cvt_f32_15.y * flat_view_7[col_51 + 1])));
                                        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(hi_11));
                                        pairs1_11[k_44] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_1_7.x * flat_view_7[col_51], cvt_f32_1_7.y * flat_view_7[col_51 + 1])));
                                    }
                                    uint32_t bf16x2_abs_23;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_23) : "r"(pairs0_11[0]));
                                    unsigned int amax_pair_22 = bf16x2_abs_23;
                                    #pragma unroll
                                    for (int i_66 = 1; i_66 < 16; i_66++) {
                                        uint32_t bf16x2_abs_1_33;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_33) : "r"(pairs0_11[i_66]));
                                        uint32_t bf16x2_max_33;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_33) : "r"(amax_pair_22), "r"(bf16x2_abs_1_33));
                                        amax_pair_22 = bf16x2_max_33;
                                    }
                                    uint16_t bf16_max_22;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_22) : "h"((uint16_t)(amax_pair_22 & 65535)), "h"((uint16_t)(amax_pair_22 >> 16)));
                                    float amax_22;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_22) : "h"((uint16_t)(bf16_max_22)));
                                    float scale_22 = fmaxf(amax_22 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_110;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_110) : "f"(scale_22), "f"(scale_22));
                                    unsigned int scale_byte_22 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_110 & 255;
                                    unsigned int inverse_lane_22 = 254 - scale_byte_22 << 7;
                                    unsigned int inverse_22 = inverse_lane_22 | inverse_lane_22 << 16;
                                    unsigned int words_22[8];
                                    #pragma unroll
                                    for (int i_67 = 0; i_67 < 8; i_67++) {
                                        uint32_t bf16x2_mul_34;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_34) : "r"(pairs0_11[i_67 * 2]), "r"(inverse_22));
                                        uint16_t e4m3x2_bf16x2_33;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_33) : "r"(bf16x2_mul_34));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_34);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_33) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_33;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_33) : "r"(pairs0_11[i_67 * 2 + 1]), "r"(inverse_22));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_111;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_111) : "r"(bf16x2_mul_1_33));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_33);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_111) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_22[i_67] = (unsigned int)e4m3x2_bf16x2_33 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_111 << 16;
                                    }
                                    uint32_t bf16x2_abs_11_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_11_1) : "r"(pairs1_11[0]));
                                    unsigned int amax_pair_11_1 = bf16x2_abs_11_1;
                                    #pragma unroll
                                    for (int i_68 = 1; i_68 < 16; i_68++) {
                                        uint32_t bf16x2_abs_1_34;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_34) : "r"(pairs1_11[i_68]));
                                        uint32_t bf16x2_max_34;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_34) : "r"(amax_pair_11_1), "r"(bf16x2_abs_1_34));
                                        amax_pair_11_1 = bf16x2_max_34;
                                    }
                                    uint16_t bf16_max_11_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_11_1) : "h"((uint16_t)(amax_pair_11_1 & 65535)), "h"((uint16_t)(amax_pair_11_1 >> 16)));
                                    float amax_11_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11_1) : "h"((uint16_t)(bf16_max_11_1)));
                                    float scale_11_1 = fmaxf(amax_11_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_112;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_112) : "f"(scale_11_1), "f"(scale_11_1));
                                    unsigned int scale_byte_11_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_112 & 255;
                                    unsigned int inverse_lane_11_1 = 254 - scale_byte_11_1 << 7;
                                    unsigned int inverse_11_1 = inverse_lane_11_1 | inverse_lane_11_1 << 16;
                                    unsigned int words_11_1[8];
                                    #pragma unroll
                                    for (int i_69 = 0; i_69 < 8; i_69++) {
                                        uint32_t bf16x2_mul_35;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_35) : "r"(pairs1_11[i_69 * 2]), "r"(inverse_11_1));
                                        uint16_t e4m3x2_bf16x2_34;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_34) : "r"(bf16x2_mul_35));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_35);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_34) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_34;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_34) : "r"(pairs1_11[i_69 * 2 + 1]), "r"(inverse_11_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_113;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_113) : "r"(bf16x2_mul_1_34));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_34);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_113) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_11_1[i_69] = (unsigned int)e4m3x2_bf16x2_34 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_113 << 16;
                                    }
                                    #pragma unroll
                                    for (int k_45 = 0; k_45 < 8; k_45++) {
                                        int col_52 = k_block_22 * 32 + (tid * 4 + k_45 * 4) % 32;
                                        smem_v34[(row0_11 * 128 + col_52) / 4] = words_22[k_45];
                                        smem_v34[(row1_11 * 128 + col_52) / 4] = words_11_1[k_45];
                                    }
                                    {
                                        uint32_t _byte_114 = static_cast<uint32_t>(scale_byte_22) & 0xFFu;
                                        uint32_t _addr_114 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base0_11 + k_block_22));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_114), "r"(_byte_114) : "memory");
                                    }
                                    {
                                        uint32_t _byte_115 = static_cast<uint32_t>(scale_byte_11_1) & 0xFFu;
                                        uint32_t _addr_115 = static_cast<uint32_t>(smem_v36_addr + (unsigned int)(scale_base1_11 + k_block_22));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_115), "r"(_byte_115) : "memory");
                                    }
                                }
                            } else {
                                int row_11 = tid - 128;
                                unsigned int scale_word_11 = 0;
                                #pragma unroll 1
                                for (int j_23 = 0; j_23 < 4; j_23++) {
                                    int k_block_23 = (j_23 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_11[16];
                                    #pragma unroll
                                    for (int k_46 = 0; k_46 < 16; k_46++) {
                                        int col_53 = k_block_23 * 32 + ((tid - 128) * 4 + k_46 * 2) % 32;
                                        float x0_11 = 0.0f;
                                        float x1_11 = 0.0f;
                                        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(flat_view_6[16448 + (row_11 * 256 + col_53) / 2]));
                                        pairs_11[k_46] = __as_u32(__float22bfloat162_rn(make_float2(cvt_f32_16.x * flat_view_7[row_11], cvt_f32_16.y * flat_view_7[row_11])));
                                    }
                                    uint32_t bf16x2_abs_24;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_24) : "r"(pairs_11[0]));
                                    unsigned int amax_pair_23 = bf16x2_abs_24;
                                    #pragma unroll
                                    for (int i_70 = 1; i_70 < 16; i_70++) {
                                        uint32_t bf16x2_abs_1_35;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_35) : "r"(pairs_11[i_70]));
                                        uint32_t bf16x2_max_35;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_35) : "r"(amax_pair_23), "r"(bf16x2_abs_1_35));
                                        amax_pair_23 = bf16x2_max_35;
                                    }
                                    uint16_t bf16_max_23;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_23) : "h"((uint16_t)(amax_pair_23 & 65535)), "h"((uint16_t)(amax_pair_23 >> 16)));
                                    float amax_23;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_23) : "h"((uint16_t)(bf16_max_23)));
                                    float scale_23 = fmaxf(amax_23 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_116;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_116) : "f"(scale_23), "f"(scale_23));
                                    unsigned int scale_byte_23 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_116 & 255;
                                    unsigned int inverse_lane_23 = 254 - scale_byte_23 << 7;
                                    unsigned int inverse_23 = inverse_lane_23 | inverse_lane_23 << 16;
                                    unsigned int words_23[8];
                                    #pragma unroll
                                    for (int i_71 = 0; i_71 < 8; i_71++) {
                                        uint32_t bf16x2_mul_36;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_36) : "r"(pairs_11[i_71 * 2]), "r"(inverse_23));
                                        uint16_t e4m3x2_bf16x2_35;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_35) : "r"(bf16x2_mul_36));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_36);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_35) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t bf16x2_mul_1_35;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_35) : "r"(pairs_11[i_71 * 2 + 1]), "r"(inverse_23));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_117;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_117) : "r"(bf16x2_mul_1_35));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_35);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_117) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_23[i_71] = (unsigned int)e4m3x2_bf16x2_35 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_117 << 16;
                                    }
                                    scale_word_11 = scale_word_11 | scale_byte_23 << (unsigned int)(k_block_23 * 8);
                                    #pragma unroll
                                    for (int k_47 = 0; k_47 < 8; k_47++) {
                                        int col_54 = k_block_23 * 32 + ((tid - 128) * 4 + k_47 * 4) % 32;
                                        smem_v30[(row_11 * 128 + col_54) / 4] = words_23[k_47];
                                    }
                                }
                                smem_v32[row_11 % 32 * 4 + row_11 / 32] = scale_word_11;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_11 = col_block_2 * 4 + 2 + 1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&dy_q_store), col_tile_11 * 128, row_8, smem_v30_addr);
                                tma_store_3d((&dy_sc_store), 0, 0, row_tile_2_1 * k_tiles_2_1 + col_tile_11, smem_v32_addr);
                                tma_store_2d((&dy_t_store), row_8, col_tile_11 * 128, smem_v34_addr);
                                tma_store_3d((&dy_sc_t_store), 0, 0, col_tile_11 * macro_tiles_2_1 + row_tile_2_1, smem_v36_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset_0 + row_8) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                    __syncthreads();
                    task_0 = next_task_2;
                }
            }
        }
    } else {
        int iteration = 0;
        int kind = -1;
        int task_2 = 0;
        int macro_1 = 0;
        int mini_1 = 0;
        int shared = 0;
        if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
            if (shared_tasks > cluster - comm_clusters) {
                shared = 1;
                if (shared_down > cluster - comm_clusters) {
                    kind = 0;
                    task_2 = cluster - comm_clusters;
                } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                    kind = 1;
                    task_2 = cluster - comm_clusters - shared_down;
                } else {
                    if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                        kind = 2;
                        task_2 = cluster - comm_clusters - shared_down - shared_swiglu;
                    } else {
                        int weight_task = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                        uint32_t _fast_div_q_0 = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)weight_task);
                        uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)weight_task) - _fast_div_q_0 * (uint32_t)(shared_wgrad_div.divisor);
                        kind = (int)_fast_div_q_0 + 3;
                        task_2 = (int)_fast_div_r_0;
                    }
                }
            } else {
                int routed = cluster - comm_clusters - shared_tasks;
                int macro_task = routed;
                int replay_tasks = 0;
                if (routed >= saved_tasks) {
                    uint32_t _fast_div_q_1 = (replay_macro_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(routed - saved_tasks)), replay_macro_tasks_div.multiplier) >> replay_macro_tasks_div.shift_right) : (uint32_t)((unsigned int)(routed - saved_tasks));
                    uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)(routed - saved_tasks)) - _fast_div_q_1 * (uint32_t)(replay_macro_tasks_div.divisor);
                    macro_1 = (int)_fast_div_q_1 + 1;
                    macro_task = (int)_fast_div_r_1;
                }
                int _min_118 = ((tokens - macro_1 * macro_size) < (macro_size) ? (tokens - macro_1 * macro_size) : (macro_size));
                uint32_t _fast_div_q_2 = (mini_size_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(_min_118 + mini_size - 1)), mini_size_div.multiplier) >> mini_size_div.shift_right) : (uint32_t)((unsigned int)(_min_118 + mini_size - 1));
                uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)(_min_118 + mini_size - 1)) - _fast_div_q_2 * (uint32_t)(mini_size_div.divisor);
                int macro_minis = (int)_fast_div_q_2;
                if (routed >= saved_tasks) {
                    replay_tasks = macro_minis * mini_replay;
                }
                if (macro_task < replay_tasks) {
                    uint32_t _fast_div_q_3 = (mini_replay_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)macro_task), mini_replay_div.multiplier) >> mini_replay_div.shift_right) : (uint32_t)((unsigned int)macro_task);
                    uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)macro_task) - _fast_div_q_3 * (uint32_t)(mini_replay_div.divisor);
                    mini_1 = (int)_fast_div_q_3;
                    int mini_task = (int)_fast_div_r_3;
                    if (mini_task < mini_replay_gate) {
                        kind = 6;
                        task_2 = mini_task;
                    } else if (mini_task < 2 * mini_replay_gate) {
                        kind = 7;
                        task_2 = mini_task - mini_replay_gate;
                    } else {
                        kind = 8;
                        task_2 = mini_task - 2 * mini_replay_gate;
                    }
                } else {
                    int bwd_task = macro_task - replay_tasks;
                    if (bwd_task < macro_minis * mini_bwd) {
                        uint32_t _fast_div_q_4 = (mini_bwd_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)bwd_task), mini_bwd_div.multiplier) >> mini_bwd_div.shift_right) : (uint32_t)((unsigned int)bwd_task);
                        uint32_t _fast_div_r_4 = (uint32_t)((unsigned int)bwd_task) - _fast_div_q_4 * (uint32_t)(mini_bwd_div.divisor);
                        mini_1 = (int)_fast_div_q_4;
                        int mini_task_1 = (int)_fast_div_r_4;
                        if (mini_task_1 < mini_down) {
                            kind = 0;
                            task_2 = mini_task_1;
                        } else if (mini_task_1 < mini_down + mini_swiglu) {
                            kind = 1;
                            task_2 = mini_task_1 - mini_down;
                        } else {
                            kind = 2;
                            task_2 = mini_task_1 - mini_down - mini_swiglu;
                        }
                    } else {
                        int weight_task_1 = bwd_task - macro_minis * mini_bwd;
                        uint32_t _fast_div_q_5 = (weight_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task_1), weight_tasks_div.multiplier) >> weight_tasks_div.shift_right) : (uint32_t)((unsigned int)weight_task_1);
                        uint32_t _fast_div_r_5 = (uint32_t)((unsigned int)weight_task_1) - _fast_div_q_5 * (uint32_t)(weight_tasks_div.divisor);
                        kind = (int)_fast_div_q_5 + 3;
                        task_2 = (int)_fast_div_r_5;
                    }
                }
            }
        }
        int tile_word_0 = kind;
        int tile_word_1 = task_2;
        int tile_word_2 = macro_1;
        int tile_word_3 = mini_1;
        int tile_word_4 = shared;
        int word0 = 0;
        int word1 = 0;
        int word2 = 0;
        if (shared == 0) {
            if (kind == 0) {
                int word0_0 = 0;
                int word1_1 = 0;
                int word2_2 = 0;
                int first_block = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                uint32_t fast_div_q = (dh_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2), dh_col_blocks_div.multiplier) >> dh_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_2);
                uint32_t fast_div_r = (uint32_t)((unsigned int)task_2) - fast_div_q * (uint32_t)(dh_col_blocks_div.divisor);
                int block = first_block + (int)fast_div_q;
                if (block < end_block) {
                    word0_0 = counts[3 * experts + block];
                    word1_1 = counts[experts + word0_0];
                    word2_2 = counts[word0_0];
                }
                word0 = word0_0;
                word1 = word1_1;
                word2 = word2_2;
            } else if (kind == 2) {
                int word0_0_1 = 0;
                int word1_1_1 = 0;
                int word2_2_1 = 0;
                int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                uint32_t fast_div_q_1 = (dx_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2), dx_col_blocks_div.multiplier) >> dx_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_2);
                uint32_t fast_div_r_1 = (uint32_t)((unsigned int)task_2) - fast_div_q_1 * (uint32_t)(dx_col_blocks_div.divisor);
                int block_1 = first_block_1 + (int)fast_div_q_1;
                if (block_1 < end_block_1) {
                    word0_0_1 = counts[3 * experts + block_1];
                    word1_1_1 = counts[experts + word0_0_1];
                    word2_2_1 = counts[word0_0_1];
                }
                word0 = word0_0_1;
                word1 = word1_1_1;
                word2 = word2_2_1;
            } else {
                if (kind == 6 || kind == 7) {
                    int word0_0_2 = 0;
                    int word1_1_2 = 0;
                    int word2_2_2 = 0;
                    int first_block_2 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                    int end_block_2 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                    uint32_t fast_div_q_2 = (replay_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2), replay_col_blocks_div.multiplier) >> replay_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_2);
                    uint32_t fast_div_r_2 = (uint32_t)((unsigned int)task_2) - fast_div_q_2 * (uint32_t)(replay_col_blocks_div.divisor);
                    int block_2 = first_block_2 + (int)fast_div_q_2;
                    if (block_2 < end_block_2) {
                        word0_0_2 = counts[3 * experts + block_2];
                        word1_1_2 = counts[experts + word0_0_2];
                        word2_2_2 = counts[word0_0_2];
                    }
                    word0 = word0_0_2;
                    word1 = word1_1_2;
                    word2 = word2_2_2;
                } else if (kind >= 3 && kind <= 5) {
                    int word0_0_3 = 0;
                    int word1_1_3 = 0;
                    int word2_2_3 = 0;
                    uint32_t fast_div_q_3 = (routed_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2), routed_wgrad_div.multiplier) >> routed_wgrad_div.shift_right) : (uint32_t)((unsigned int)task_2);
                    uint32_t fast_div_r_3 = (uint32_t)((unsigned int)task_2) - fast_div_q_3 * (uint32_t)(routed_wgrad_div.divisor);
                    int expert_idx = (int)fast_div_q_3;
                    word0_0_3 = counts[experts + expert_idx];
                    word1_1_3 = counts[2 * experts + expert_idx];
                    word0 = word0_0_3;
                    word1 = word1_1_3;
                    word2 = word2_2_3;
                }
            }
        }
        int tile_word_5 = word0;
        int tile_word_6 = word1;
        int tile_word_7 = word2;
        while (cluster >= 0 && cluster < true_clusters) {
            if (tid / 32 == 5) {
                if (warp == 5) {
                    if (elect_sync()) {
                        if (cta_rank_0 == 0) {
                            int _parity_schedule_finished = (int)((unsigned int)iteration % 2) ^ 1;
                            mbarrier_wait(schedule_finished_addr, _parity_schedule_finished);
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
            int row_count = 2 * (intermediate / 128);
            int shared_rows = local_tokens / 256;
            int shared_down_0 = shared_rows * (intermediate / 256);
            int i_tiles_1 = intermediate / 128;
            int h_tiles = hidden / 128;
            if (tile_word_4 != 0) {
                if (tile_word_0 == 0) {
                    int col_blocks_1 = (hidden + 512 - 1) / 512;
                    {
                        col_blocks_1 = (intermediate + 512 - 1) / 512;
                    }
                    int x = -1;
                    int y = -1;
                    int expert = -1;
                    int k_start = 0;
                    int k_end = 0;
                    int first = 0;
                    int row_blocks = local_tokens / 256;
                    if (tile_word_1 < row_blocks * col_blocks_1) {
                        int supergroup = tile_word_1 / (row_blocks * 8);
                        int full_cols = col_blocks_1 / 8 * 8;
                        int row_12 = 0;
                        int col_55 = 0;
                        if (tile_word_1 < row_blocks * full_cols) {
                            row_12 = tile_word_1 % (row_blocks * 8) / 8;
                            col_55 = supergroup * 8 + tile_word_1 % 8;
                        } else {
                            row_12 = (tile_word_1 - row_blocks * full_cols) / (col_blocks_1 - full_cols);
                            col_55 = full_cols + (tile_word_1 - row_blocks * full_cols) % (col_blocks_1 - full_cols);
                        }
                        if ((supergroup & 1) != 0) {
                            row_12 = row_blocks - row_12 - 1;
                        }
                        x = row_12;
                        y = col_55;
                        expert = 0;
                    }
                    int has_hi = 0;
                    has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                    int global_mini = 0;
                    int macro_rows = 0;
                    int iterations = hidden / 64;
                    if (expert < 0) {
                        if (tid == 0) {
                        }
                    } else {
                        if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_119 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                        int mini_rows_1 = ((0) > (_min_119) ? (0) : (_min_119));
                                        int required_2 = (mini_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                                    }
                                    #pragma unroll 1
                                    for (int idx = 0; idx < iterations; idx++) {
                                        int _stage_gemm_finished = (int)((unsigned int)(gemm_wide + idx) % 4);
                                        int _parity_gemm_finished = (int)((unsigned int)(gemm_wide + idx) / 4 % 2) ^ 1;
                                        mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished) * 8, _parity_gemm_finished);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&dy_s)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wd_s)), "r"(0), "r"(idx * 64), "r"(y * 2 * 4 + cta_rank_0 * 2), "r"(expert), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wd_s)), "r"(0), "r"(idx * 64), "r"((y * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int _parity_output_finished = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr, _parity_output_finished);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_1 = 0; idx_1 < iterations; idx_1++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_1) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived = (int)((unsigned int)(gemm_wide + idx_1) % 4);
                                        int _parity_gemm_arrived = (int)((unsigned int)(gemm_wide + idx_1) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived) * 8, _parity_gemm_arrived);
                                        int _smem_desc_lo_120 = (((a_nt_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                        int _smem_desc_lo_121 = ((((mn_view_addr_0) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_120), "r"(_smem_desc_lo_121), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                        int _smem_desc_lo_122 = (((a_nt_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                        int _smem_desc_lo_123 = ((((mn_view_addr_1) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_122), "r"(_smem_desc_lo_123), "r"((tmem_accumulator + (256))), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_1) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else {
                            if (tid < 128) {
                                int _parity_output_arrived = (int)((unsigned int)gemm_out0 % 2);
                                mbarrier_wait(output_arrived_addr, _parity_output_arrived);
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
                                            packed[chunk * 16 + sub * 8 + pair] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1])));
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
                                    int previous_offset = macro_size;
                                    int output_row = x * 256 + cta_rank_0 * 128;
                                    int _min_124 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                    if (output_row < _min_124) {
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
                                    for (int half_12 = 0; half_12 < 2; half_12++) {
                                        #pragma unroll
                                        for (int col_tile = 0; col_tile < 2; col_tile++) {
                                            int row_13 = warp_0 * 32 + half_12 * 16 + lane_1 % 16;
                                            int col_56 = col_tile * 16 + lane_1 / 16 * 8;
                                            unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_13 * 32 + col_56) * 2);
                                            address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                            int offset_2 = chunk_1 * 16 + half_12 * 8 + col_tile * 4;
                                            uint32_t _stmatrix_addr_125 = static_cast<uint32_t>(address_1);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_125), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 3]))
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
                                                packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[pair_1 * 2], _tmem_load_1[pair_1 * 2 + 1])));
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
                                        for (int half_13 = 0; half_13 < 2; half_13++) {
                                            #pragma unroll
                                            for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                                int row_14 = warp_0_1 * 32 + half_13 * 16 + lane_2 % 16;
                                                int col_57 = col_tile_12 * 16 + lane_2 / 16 * 8;
                                                unsigned int address_3 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_14 * 32 + col_57) * 2);
                                                address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                                int offset_3 = chunk_3 * 16 + half_13 * 8 + col_tile_12 * 4;
                                                uint32_t _stmatrix_addr_126 = static_cast<uint32_t>(address_3);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_126), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 3]))
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
                        gemm_wide += iterations;
                        gemm_out0 += 1;
                        gemm_acc_tiles += 1;
                    }
                } else if (tile_word_0 == 1) {
                    int col_blocks_2 = intermediate / 128;
                    int num_tiles = local_tokens / 128 * col_blocks_2;
                    int macro_row_offset = 0;
                    int first_tile_1 = tile_word_1 * 16 + cta_rank_0 * 8;
                    int tile_end = num_tiles;
                    int _min_127 = ((8) < (tile_end - first_tile_1) ? (8) : (tile_end - first_tile_1));
                    int tiles = ((0) > (_min_127) ? (0) : (_min_127));
                    if (tiles > 0) {
                        if (tid == 0) {
                            int _min_6 = ((tiles) < (2) ? (tiles) : (2));
                            #pragma unroll 1
                            for (int stage_1 = 0; stage_1 < _min_6; stage_1++) {
                                int col_blocks_1_1 = intermediate / 128;
                                int row_15 = (first_tile_1 + stage_1) / col_blocks_1_1;
                                int col_58 = (first_tile_1 + stage_1) % col_blocks_1_1;
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_1) % 2)) * 8, 98304);
                                int parent = row_15 / 2 * (intermediate / 256) + col_58 / 2;
                                int poll_4_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_first) : "l"(dh_ready + parent) : "memory");
                                int poll_4 = poll_4_first;
                                while (poll_4 < 2) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_4_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_next) : "l"(dh_ready + parent) : "memory");
                                    poll_4 = poll_4_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_dh_addr + (unsigned int)((swiglu_seq + stage_1) % 2 * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_15 - macro_row_offset) * 128), "r"(col_58 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_1) % 2)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_gate_addr + (unsigned int)((swiglu_seq + stage_1) % 2 * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_15 - macro_row_offset) * 128), "r"(col_58 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_1) % 2)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_up_addr + (unsigned int)((swiglu_seq + stage_1) % 2 * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_15 - macro_row_offset) * 128), "r"(col_58 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage_1) % 2)) * 8) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int index = 0; index < tiles; index++) {
                            int stage_2 = (swiglu_seq + index) % 2;
                            int _parity_swiglu_arrived = (int)((unsigned int)(swiglu_seq + index) / 2 % 2);
                            mbarrier_wait(swiglu_arrived_addr + (stage_2) * 8, _parity_swiglu_arrived);
                            int row_16 = (first_tile_1 + index) / col_blocks_2;
                            int col_59 = (first_tile_1 + index) % col_blocks_2;
                            float gate[64];
                            float up[64];
                            float dhidden[64];
                            int warp_1 = tid / 32;
                            int local_warp = warp_1 / 4 + warp_1 % 4 * 2;
                            int lane_3 = tid % 32;
                            #pragma unroll
                            for (int tile_col = 0; tile_col < 8; tile_col++) {
                                unsigned int packed_1[4];
                                unsigned int address_4 = sw_gate_addr + (unsigned int)(stage_2 * 32768) + (unsigned int)(((tile_col * 16 + lane_3 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_3 % 16) * 64 + (tile_col * 16 + lane_3 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_1[0]), "=r"(packed_1[1]), "=r"(packed_1[2]), "=r"(packed_1[3])
                                    : "r"(address_4 ^ (address_4 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                    float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_1[pair_2]));
                                    gate[tile_col * 8 + pair_2 * 2] = cvt_f32_17.x;
                                    gate[tile_col * 8 + pair_2 * 2 + 1] = cvt_f32_17.y;
                                }
                            }
                            #pragma unroll
                            for (int elem = 0; elem < 64; elem++) {
                                dhidden[elem] = gate[elem] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                float _exp_128 = expf(dhidden[elem_1]);
                                dhidden[elem_1] = _exp_128;
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
                            int warp_0_2 = tid / 32;
                            int local_warp_0 = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                            int lane_0 = tid % 32;
                            #pragma unroll
                            for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                unsigned int packed_2[4];
                                unsigned int address_5 = sw_dh_addr + (unsigned int)(stage_2 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_0 / 16 * 8) / 64 * 128 * 64 + (local_warp_0 * 16 + lane_0 % 16) * 64 + (tile_col_1 * 16 + lane_0 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                    : "r"(address_5 ^ (address_5 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                    float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_2[pair_3]));
                                    dhidden[tile_col_1 * 8 + pair_3 * 2] = cvt_f32_18.x;
                                    dhidden[tile_col_1 * 8 + pair_3 * 2 + 1] = cvt_f32_18.y;
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
                            int warp_1_1 = tid / 32;
                            int local_warp_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                            int lane_1_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                unsigned int packed_3[4];
                                unsigned int address_6 = sw_up_addr + (unsigned int)(stage_2 * 32768) + (unsigned int)(((tile_col_2 * 16 + lane_1_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_1_1 % 16) * 64 + (tile_col_2 * 16 + lane_1_1 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                    : "r"(address_6 ^ (address_6 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                    float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_3[pair_4]));
                                    up[tile_col_2 * 8 + pair_4 * 2] = cvt_f32_19.x;
                                    up[tile_col_2 * 8 + pair_4 * 2 + 1] = cvt_f32_19.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_10 = 0; elem_10 < 64; elem_10++) {
                                dhidden[elem_10] = dhidden[elem_10] * up[elem_10];
                            }
                            int warp_2 = tid / 32;
                            int local_warp_2 = warp_2 / 4 + warp_2 % 4 * 2;
                            int lane_2_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                unsigned int packed_4[4];
                                #pragma unroll
                                for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                    packed_4[pair_5] = __as_u32(__float22bfloat162_rn(make_float2(dhidden[tile_col_3 * 8 + pair_5 * 2], dhidden[tile_col_3 * 8 + pair_5 * 2 + 1])));
                                }
                                unsigned int address_7 = sw_gate_addr + (unsigned int)(stage_2 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_2_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_2_1 % 16) * 64 + (tile_col_3 * 16 + lane_2_1 / 16 * 8) % 64) * 2);
                                uint32_t _stmatrix_addr_129 = static_cast<uint32_t>(address_7 ^ (address_7 & 1023) >> 7 << 4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_129), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                    : "memory");
                            }
                            int warp_3 = tid / 32;
                            int local_warp_3 = warp_3 / 4 + warp_3 % 4 * 2;
                            int lane_3_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                unsigned int packed_5[4];
                                #pragma unroll
                                for (int pair_6 = 0; pair_6 < 4; pair_6++) {
                                    packed_5[pair_6] = __as_u32(__float22bfloat162_rn(make_float2(gate[tile_col_4 * 8 + pair_6 * 2], gate[tile_col_4 * 8 + pair_6 * 2 + 1])));
                                }
                                unsigned int address_8 = sw_up_addr + (unsigned int)(stage_2 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_3 * 16 + lane_3_1 % 16) * 64 + (tile_col_4 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                uint32_t _stmatrix_addr_130 = static_cast<uint32_t>(address_8 ^ (address_8 & 1023) >> 7 << 4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_130), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&dg_sw_s), 0, (row_16 - macro_row_offset) * 128, col_59 * 2, 0, 0, sw_gate_addr + (unsigned int)(stage_2 * 32768));
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&du_sw_s), 0, (row_16 - macro_row_offset) * 128, col_59 * 2, 0, 0, sw_up_addr + (unsigned int)(stage_2 * 32768));
                                asm volatile("cp.async.bulk.commit_group;");
                                if (tiles > index + 2) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                    int col_blocks_2_1 = intermediate / 128;
                                    int row_12_1 = (first_tile_1 + index + 2) / col_blocks_2_1;
                                    int col_0 = (first_tile_1 + index + 2) % col_blocks_2_1;
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index + 2) % 2)) * 8, 98304);
                                    int parent_1 = row_12_1 / 2 * (intermediate / 256) + col_0 / 2;
                                    int poll_5_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_first) : "l"(dh_ready + parent_1) : "memory");
                                    int poll_5 = poll_5_first;
                                    while (poll_5 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_5_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_next) : "l"(dh_ready + parent_1) : "memory");
                                        poll_5 = poll_5_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_dh_addr + (unsigned int)((swiglu_seq + index + 2) % 2 * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_12_1 - macro_row_offset) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index + 2) % 2)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_gate_addr + (unsigned int)((swiglu_seq + index + 2) % 2 * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_12_1 - macro_row_offset) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index + 2) % 2)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_up_addr + (unsigned int)((swiglu_seq + index + 2) % 2 * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_12_1 - macro_row_offset) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + index + 2) % 2)) * 8) : "memory");
                                }
                            }
                            __syncthreads();
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll 1
                            for (int index_1 = 0; index_1 < tiles; index_1++) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + ((first_tile_1 + index_1) / col_blocks_2 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    if (tid == 0) {
                    }
                    swiglu_seq += tiles;
                } else {
                    if (tile_word_0 == 2) {
                        int col_blocks_4 = (intermediate + 512 - 1) / 512;
                        {
                            col_blocks_4 = (hidden + 512 - 1) / 512;
                        }
                        int x_1 = -1;
                        int y_1 = -1;
                        int expert_1 = -1;
                        int k_start_1 = 0;
                        int k_end_1 = 0;
                        int first_1 = 0;
                        int row_blocks_1 = local_tokens / 256;
                        if (tile_word_1 < row_blocks_1 * col_blocks_4) {
                            int supergroup_1 = tile_word_1 / (row_blocks_1 * 8);
                            int full_cols_1 = col_blocks_4 / 8 * 8;
                            int row_17 = 0;
                            int col_60 = 0;
                            if (tile_word_1 < row_blocks_1 * full_cols_1) {
                                row_17 = tile_word_1 % (row_blocks_1 * 8) / 8;
                                col_60 = supergroup_1 * 8 + tile_word_1 % 8;
                            } else {
                                row_17 = (tile_word_1 - row_blocks_1 * full_cols_1) / (col_blocks_4 - full_cols_1);
                                col_60 = full_cols_1 + (tile_word_1 - row_blocks_1 * full_cols_1) % (col_blocks_4 - full_cols_1);
                            }
                            if ((supergroup_1 & 1) != 0) {
                                row_17 = row_blocks_1 - row_17 - 1;
                            }
                            x_1 = row_17;
                            y_1 = col_60;
                            expert_1 = 0;
                        }
                        int has_hi_1 = 0;
                        has_hi_1 = (int)((y_1 * 2 + 1) * 256 < hidden);
                        int global_mini_1 = 0;
                        int macro_rows_1 = 0;
                        int iterations_1 = intermediate / 64 + intermediate / 64;
                        if (expert_1 < 0) {
                            if (tid == 0) {
                            }
                        } else {
                            if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_3 = 1;
                                            if (enabled_value_3 != 0) {
                                                int poll_6_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_first) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                                int poll_6 = poll_6_first;
                                                while (poll_6 < row_count) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int poll_6_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_next) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                                    poll_6 = poll_6_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_131 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                            int mini_rows_2 = ((0) > (_min_131) ? (0) : (_min_131));
                                            int required_3 = (mini_rows_2 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        #pragma unroll 1
                                        for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                            int _stage_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_2) % 4);
                                            int _parity_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_2) / 4 % 2) ^ 1;
                                            mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_1) * 8, _parity_gemm_finished_1);
                                            if (idx_2 < intermediate / 64) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&dg_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wg_s)), "r"(0), "r"(idx_2 * 64), "r"(y_1 * 2 * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wg_s)), "r"(0), "r"(idx_2 * 64), "r"((y_1 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            } else {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_nt_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&du_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2 - intermediate / 64), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wu_s)), "r"(0), "r"((idx_2 - intermediate / 64) * 64), "r"(y_1 * 2 * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wu_s)), "r"(0), "r"((idx_2 - intermediate / 64) * 64), "r"((y_1 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                }
                            } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int _parity_output_finished_1 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr, _parity_output_finished_1);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_3 = 0; idx_3 < iterations_1; idx_3++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) % 4);
                                            int _parity_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_1) * 8, _parity_gemm_arrived_1);
                                            int _smem_desc_lo_132 = (((a_nt_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                            int _smem_desc_lo_133 = ((((mn_view_addr_0) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_132), "r"(_smem_desc_lo_133), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                            int _smem_desc_lo_134 = (((a_nt_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                            int _smem_desc_lo_135 = ((((mn_view_addr_1) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_134), "r"(_smem_desc_lo_135), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    int _parity_output_arrived_1 = (int)((unsigned int)gemm_out0 % 2);
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_1);
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
                                                packed_6[chunk_4 * 16 + sub_2 * 8 + pair_7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[pair_7 * 2], _tmem_load_2[pair_7 * 2 + 1])));
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
                                        int previous_offset_1 = macro_size;
                                        int output_row_1 = x_1 * 256 + cta_rank_0 * 128;
                                        int _min_136 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                                        if (output_row_1 < _min_136) {
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
                                        for (int half_14 = 0; half_14 < 2; half_14++) {
                                            #pragma unroll
                                            for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                                int row_18 = warp_0_3 * 32 + half_14 * 16 + lane_4 % 16;
                                                int col_61 = col_tile_13 * 16 + lane_4 / 16 * 8;
                                                unsigned int address_10 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_18 * 32 + col_61) * 2);
                                                address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                                int offset_4 = chunk_5 * 16 + half_14 * 8 + col_tile_13 * 4;
                                                uint32_t _stmatrix_addr_137 = static_cast<uint32_t>(address_10);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_137), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 3]))
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
                                                    packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[pair_8 * 2], _tmem_load_3[pair_8 * 2 + 1])));
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
                                            for (int half_15 = 0; half_15 < 2; half_15++) {
                                                #pragma unroll
                                                for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                                    int row_19 = warp_0_4 * 32 + half_15 * 16 + lane_5 % 16;
                                                    int col_62 = col_tile_14 * 16 + lane_5 / 16 * 8;
                                                    unsigned int address_12 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_19 * 32 + col_62) * 2);
                                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                                    int offset_5 = chunk_7 * 16 + half_15 * 8 + col_tile_14 * 4;
                                                    uint32_t _stmatrix_addr_138 = static_cast<uint32_t>(address_12);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_138), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 3]))
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
                            gemm_wide += iterations_1;
                            gemm_out0 += 1;
                            gemm_acc_tiles += 1;
                        }
                    } else if (tile_word_0 == 3) {
                        int col_blocks_5 = (local_tokens + 512 - 1) / 512;
                        {
                            col_blocks_5 = (intermediate + 512 - 1) / 512;
                        }
                        int x_2 = -1;
                        int y_2 = -1;
                        int expert_2 = -1;
                        int k_start_2 = 0;
                        int k_end_2 = 0;
                        int first_2 = 0;
                        int row_blocks_2 = hidden / 256;
                        int expert_idx_1 = 0;
                        int local_task = tile_word_1;
                        k_end_2 = local_tokens;
                        first_2 = 1;
                        if (k_start_2 < k_end_2 && local_task < row_blocks_2 * col_blocks_5) {
                            int supergroup_2 = local_task / (row_blocks_2 * 8);
                            int full_cols_2 = col_blocks_5 / 8 * 8;
                            int row_20 = 0;
                            int col_63 = 0;
                            if (local_task < row_blocks_2 * full_cols_2) {
                                row_20 = local_task % (row_blocks_2 * 8) / 8;
                                col_63 = supergroup_2 * 8 + local_task % 8;
                            } else {
                                row_20 = (local_task - row_blocks_2 * full_cols_2) / (col_blocks_5 - full_cols_2);
                                col_63 = full_cols_2 + (local_task - row_blocks_2 * full_cols_2) % (col_blocks_5 - full_cols_2);
                            }
                            if ((supergroup_2 & 1) != 0) {
                                row_20 = row_blocks_2 - row_20 - 1;
                            }
                            x_2 = row_20;
                            y_2 = col_63;
                            expert_2 = expert_idx_1;
                        }
                        int has_hi_2 = 0;
                        has_hi_2 = (int)((y_2 * 2 + 1) * 256 < intermediate);
                        int global_mini_2 = 0;
                        int macro_rows_2 = 0;
                        int iterations_2 = hidden / 64;
                        iterations_2 = (k_end_2 - k_start_2) / 64;
                        if (expert_2 < 0) {
                            if (tid == 0) {
                            }
                        } else {
                            if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        #pragma unroll 1
                                        for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                            int token_row = k_start_2 + idx_4 * 64;
                                            if (idx_4 == 0 || token_row % 256 == 0) {
                                            }
                                            if (idx_4 == 0 || token_row % mini_size == 0) {
                                                int input_mini = token_row / mini_size;
                                                int input_rows = ((mini_size) < (tokens - input_mini * mini_size) ? (mini_size) : (tokens - input_mini * mini_size));
                                                int input_count = (input_rows + 127) / 128 * ((hidden + 511) / 512);
                                            }
                                            int _stage_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_4) % 4);
                                            int _parity_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_4) / 4 % 2) ^ 1;
                                            mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_2) * 8, _parity_gemm_finished_2);
                                            int local_row = k_start_2 + idx_4 * 64;
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(mn_view_addr + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&dy_atb_s)), "r"(0), "r"(local_row), "r"(x_2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&h_atb_s)), "r"(0), "r"(local_row), "r"(y_2 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&h_atb_s)), "r"(0), "r"(local_row), "r"((y_2 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        }
                                    }
                                }
                            } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int _parity_output_finished_2 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr, _parity_output_finished_2);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_5 = 0; idx_5 < iterations_2; idx_5++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_5) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_5) % 4);
                                            int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_5) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
                                            int _smem_desc_lo_139 = ((((mn_view_addr) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_5) % 4) * 1024;
                                            int _smem_desc_lo_140 = ((((mn_view_addr_0) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_5) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_139), "r"(_smem_desc_lo_140), "r"(tmem_accumulator), "r"(((idx_5 == 0) ? 0 : 1)) : "memory");
                                            int _smem_desc_lo_141 = ((((mn_view_addr) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_5) % 4) * 1024;
                                            int _smem_desc_lo_142 = ((((mn_view_addr_1) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_5) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_141), "r"(_smem_desc_lo_142), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)) : "memory");
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_5) % 4)) * 8, (uint16_t)(3));
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    int _parity_output_arrived_2 = (int)((unsigned int)gemm_out0 % 2);
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_2);
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
                                        for (int vec_9 = 0; vec_9 < 4; vec_9++) {
                                            unsigned int address_13 = d_smem_addr + (unsigned int)(chunk_8 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_9 * 16);
                                            address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_13 - flat_view_addr)), "f"(_tmem_load_4[vec_9 * 4]), "f"(_tmem_load_4[vec_9 * 4 + 1]), "f"(_tmem_load_4[vec_9 * 4 + 2]), "f"(_tmem_load_4[vec_9 * 4 + 3]) : "memory");
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
                                            for (int vec_10 = 0; vec_10 < 4; vec_10++) {
                                                unsigned int address_14 = d_smem_addr + (unsigned int)((16 + chunk_9) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_10 * 16);
                                                address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_14 - flat_view_addr)), "f"(_tmem_load_5[vec_10 * 4]), "f"(_tmem_load_5[vec_10 * 4 + 1]), "f"(_tmem_load_5[vec_10 * 4 + 2]), "f"(_tmem_load_5[vec_10 * 4 + 3]) : "memory");
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
                            gemm_wide += iterations_2;
                            gemm_out0 += 1;
                            gemm_acc_tiles += 1;
                        }
                    } else {
                        if (tile_word_0 == 4) {
                            int col_blocks_6 = (local_tokens + 512 - 1) / 512;
                            {
                                col_blocks_6 = (hidden + 512 - 1) / 512;
                            }
                            int x_3 = -1;
                            int y_3 = -1;
                            int expert_3 = -1;
                            int k_start_3 = 0;
                            int k_end_3 = 0;
                            int first_3 = 0;
                            int row_blocks_3 = intermediate / 256;
                            int expert_idx_2 = 0;
                            int local_task_1 = tile_word_1;
                            k_end_3 = local_tokens;
                            first_3 = 1;
                            if (k_start_3 < k_end_3 && local_task_1 < row_blocks_3 * col_blocks_6) {
                                int supergroup_3 = local_task_1 / (row_blocks_3 * 8);
                                int full_cols_3 = col_blocks_6 / 8 * 8;
                                int row_21 = 0;
                                int col_64 = 0;
                                if (local_task_1 < row_blocks_3 * full_cols_3) {
                                    row_21 = local_task_1 % (row_blocks_3 * 8) / 8;
                                    col_64 = supergroup_3 * 8 + local_task_1 % 8;
                                } else {
                                    row_21 = (local_task_1 - row_blocks_3 * full_cols_3) / (col_blocks_6 - full_cols_3);
                                    col_64 = full_cols_3 + (local_task_1 - row_blocks_3 * full_cols_3) % (col_blocks_6 - full_cols_3);
                                }
                                if ((supergroup_3 & 1) != 0) {
                                    row_21 = row_blocks_3 - row_21 - 1;
                                }
                                x_3 = row_21;
                                y_3 = col_64;
                                expert_3 = expert_idx_2;
                            }
                            int has_hi_3 = 0;
                            has_hi_3 = (int)((y_3 * 2 + 1) * 256 < hidden);
                            int global_mini_3 = 0;
                            int macro_rows_3 = 0;
                            int iterations_3 = intermediate / 64;
                            iterations_3 = (k_end_3 - k_start_3) / 64;
                            if (expert_3 < 0) {
                                if (tid == 0) {
                                }
                            } else {
                                if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            #pragma unroll 1
                                            for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                                int token_row_1 = k_start_3 + idx_6 * 64;
                                                if (idx_6 == 0 || token_row_1 % 256 == 0) {
                                                    bool enabled_value_4 = 1;
                                                    if (enabled_value_4 != 0) {
                                                        int poll_7_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_first) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                        int poll_7 = poll_7_first;
                                                        while (poll_7 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_7_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_next) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                            poll_7 = poll_7_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_6 == 0 || token_row_1 % mini_size == 0) {
                                                    int input_mini_1 = token_row_1 / mini_size;
                                                    int input_rows_1 = ((mini_size) < (tokens - input_mini_1 * mini_size) ? (mini_size) : (tokens - input_mini_1 * mini_size));
                                                    int input_count_1 = (input_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                                                }
                                                int _stage_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_6) % 4);
                                                int _parity_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_6) / 4 % 2) ^ 1;
                                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_3) * 8, _parity_gemm_finished_3);
                                                int local_row_1 = k_start_3 + idx_6 * 64;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&dg_atb_s)), "r"(0), "r"(local_row_1), "r"(x_3 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_1), "r"(y_3 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_1), "r"((y_3 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int _parity_output_finished_3 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr, _parity_output_finished_3);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_7) % 4);
                                                int _parity_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_7) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_3) * 8, _parity_gemm_arrived_3);
                                                int _smem_desc_lo_143 = ((((mn_view_addr) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_7) % 4) * 1024;
                                                int _smem_desc_lo_144 = ((((mn_view_addr_0) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_7) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_143), "r"(_smem_desc_lo_144), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)) : "memory");
                                                int _smem_desc_lo_145 = ((((mn_view_addr) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_7) % 4) * 1024;
                                                int _smem_desc_lo_146 = ((((mn_view_addr_1) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_7) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_145), "r"(_smem_desc_lo_146), "r"((tmem_accumulator + (256))), "r"(((idx_7 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8, (uint16_t)(3));
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        int _parity_output_arrived_3 = (int)((unsigned int)gemm_out0 % 2);
                                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_3);
                                        int warp_row_1 = tid / 32 * 32;
                                        #pragma unroll
                                        for (int chunk_10 = 0; chunk_10 < 16; chunk_10++) {
                                            float tmem_load[16];
                                            tmem_ld_x16(&tmem_load[0], taddr_1 + (unsigned int)(warp_row_1 << 16) + (unsigned int)(chunk_10 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_11 = 0; vec_11 < 4; vec_11++) {
                                                unsigned int address_15 = d_smem_addr + (unsigned int)(chunk_10 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_11 * 16);
                                                address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_15 - flat_view_addr)), "f"(tmem_load[vec_11 * 4]), "f"(tmem_load[vec_11 * 4 + 1]), "f"(tmem_load[vec_11 * 4 + 2]), "f"(tmem_load[vec_11 * 4 + 3]) : "memory");
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
                                                float tmem_load_1[16];
                                                tmem_ld_x16(&tmem_load_1[0], taddr_1 + (unsigned int)(warp_row_1 << 16) + 256 + (unsigned int)(chunk_11 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_12 = 0; vec_12 < 4; vec_12++) {
                                                    unsigned int address_16 = d_smem_addr + (unsigned int)((16 + chunk_11) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_12 * 16);
                                                    address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_16 - flat_view_addr)), "f"(tmem_load_1[vec_12 * 4]), "f"(tmem_load_1[vec_12 * 4 + 1]), "f"(tmem_load_1[vec_12 * 4 + 2]), "f"(tmem_load_1[vec_12 * 4 + 3]) : "memory");
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
                                gemm_wide += iterations_3;
                                gemm_out0 += 1;
                                gemm_acc_tiles += 1;
                            }
                        } else if (tile_word_0 == 5) {
                            int col_blocks_7 = (local_tokens + 512 - 1) / 512;
                            {
                                col_blocks_7 = (hidden + 512 - 1) / 512;
                            }
                            int x_4 = -1;
                            int y_4 = -1;
                            int expert_4 = -1;
                            int k_start_4 = 0;
                            int k_end_4 = 0;
                            int first_4 = 0;
                            int row_blocks_4 = intermediate / 256;
                            int expert_idx_3 = 0;
                            int local_task_2 = tile_word_1;
                            k_end_4 = local_tokens;
                            first_4 = 1;
                            if (k_start_4 < k_end_4 && local_task_2 < row_blocks_4 * col_blocks_7) {
                                int supergroup_4 = local_task_2 / (row_blocks_4 * 8);
                                int full_cols_4 = col_blocks_7 / 8 * 8;
                                int row_22 = 0;
                                int col_65 = 0;
                                if (local_task_2 < row_blocks_4 * full_cols_4) {
                                    row_22 = local_task_2 % (row_blocks_4 * 8) / 8;
                                    col_65 = supergroup_4 * 8 + local_task_2 % 8;
                                } else {
                                    row_22 = (local_task_2 - row_blocks_4 * full_cols_4) / (col_blocks_7 - full_cols_4);
                                    col_65 = full_cols_4 + (local_task_2 - row_blocks_4 * full_cols_4) % (col_blocks_7 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_22 = row_blocks_4 - row_22 - 1;
                                }
                                x_4 = row_22;
                                y_4 = col_65;
                                expert_4 = expert_idx_3;
                            }
                            int has_hi_4 = 0;
                            has_hi_4 = (int)((y_4 * 2 + 1) * 256 < hidden);
                            int global_mini_4 = 0;
                            int macro_rows_4 = 0;
                            int iterations_4 = intermediate / 64;
                            iterations_4 = (k_end_4 - k_start_4) / 64;
                            if (expert_4 < 0) {
                                if (tid == 0) {
                                }
                            } else {
                                if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            #pragma unroll 1
                                            for (int idx_8 = 0; idx_8 < iterations_4; idx_8++) {
                                                int token_row_2 = k_start_4 + idx_8 * 64;
                                                if (idx_8 == 0 || token_row_2 % 256 == 0) {
                                                    bool enabled_value_5 = 1;
                                                    if (enabled_value_5 != 0) {
                                                        int poll_8_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_8_first) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                        int poll_8 = poll_8_first;
                                                        while (poll_8 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_8_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_8_next) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                            poll_8 = poll_8_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_8 == 0 || token_row_2 % mini_size == 0) {
                                                    int input_mini_2 = token_row_2 / mini_size;
                                                    int input_rows_2 = ((mini_size) < (tokens - input_mini_2 * mini_size) ? (mini_size) : (tokens - input_mini_2 * mini_size));
                                                    int input_count_2 = (input_rows_2 + 127) / 128 * ((hidden + 511) / 512);
                                                }
                                                int _stage_gemm_finished_4 = (int)((unsigned int)(gemm_wide + idx_8) % 4);
                                                int _parity_gemm_finished_4 = (int)((unsigned int)(gemm_wide + idx_8) / 4 % 2) ^ 1;
                                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_4) * 8, _parity_gemm_finished_4);
                                                int local_row_2 = k_start_4 + idx_8 * 64;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr + (unsigned int)((gemm_wide + idx_8) % 4 * 16384)), "l"((&du_atb_s)), "r"(0), "r"(local_row_2), "r"(x_4 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_0 + (unsigned int)((gemm_wide + idx_8) % 4 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_2), "r"(y_4 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(mn_view_addr_1 + (unsigned int)((gemm_wide + idx_8) % 4 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_2), "r"((y_4 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int _parity_output_finished_4 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr, _parity_output_finished_4);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_9 = 0; idx_9 < iterations_4; idx_9++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) % 4);
                                                int _parity_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_4) * 8, _parity_gemm_arrived_4);
                                                int _smem_desc_lo_147 = ((((mn_view_addr) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_9) % 4) * 1024;
                                                int _smem_desc_lo_148 = ((((mn_view_addr_0) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_9) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_147), "r"(_smem_desc_lo_148), "r"(tmem_accumulator), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                                int _smem_desc_lo_149 = ((((mn_view_addr) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_9) % 4) * 1024;
                                                int _smem_desc_lo_150 = ((((mn_view_addr_1) >> 4) & 0x7FFF) | 0x2000000) + ((gemm_wide + idx_9) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_149), "r"(_smem_desc_lo_150), "r"((tmem_accumulator + (256))), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        int _parity_output_arrived_4 = (int)((unsigned int)gemm_out0 % 2);
                                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_4);
                                        int warp_row_2 = tid / 32 * 32;
                                        #pragma unroll
                                        for (int chunk_12 = 0; chunk_12 < 16; chunk_12++) {
                                            float tmem_load_2[16];
                                            tmem_ld_x16(&tmem_load_2[0], taddr_1 + (unsigned int)(warp_row_2 << 16) + (unsigned int)(chunk_12 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_13 = 0; vec_13 < 4; vec_13++) {
                                                unsigned int address_17 = d_smem_addr + (unsigned int)(chunk_12 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_13 * 16);
                                                address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_17 - flat_view_addr)), "f"(tmem_load_2[vec_13 * 4]), "f"(tmem_load_2[vec_13 * 4 + 1]), "f"(tmem_load_2[vec_13 * 4 + 2]), "f"(tmem_load_2[vec_13 * 4 + 3]) : "memory");
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
                                                float tmem_load_1_1[16];
                                                tmem_ld_x16(&tmem_load_1_1[0], taddr_1 + (unsigned int)(warp_row_2 << 16) + 256 + (unsigned int)(chunk_13 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_14 = 0; vec_14 < 4; vec_14++) {
                                                    unsigned int address_18 = d_smem_addr + (unsigned int)((16 + chunk_13) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_14 * 16);
                                                    address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_18 - flat_view_addr)), "f"(tmem_load_1_1[vec_14 * 4]), "f"(tmem_load_1_1[vec_14 * 4 + 1]), "f"(tmem_load_1_1[vec_14 * 4 + 2]), "f"(tmem_load_1_1[vec_14 * 4 + 3]) : "memory");
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
                                gemm_wide += iterations_4;
                                gemm_out0 += 1;
                                gemm_acc_tiles += 1;
                            }
                        }
                    }
                }
            } else if (tile_word_0 == 6) {
                int col_blocks_8 = (intermediate + 256 - 1) / 256;
                int x_5 = -1;
                int y_5 = -1;
                int expert_5 = -1;
                int k_start_5 = 0;
                int k_end_5 = 0;
                int first_5 = 0;
                int first_block_3 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                int end_block_3 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                int block_3 = first_block_3;
                uint32_t fast_div_q_4 = (replay_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), replay_col_blocks_div.multiplier) >> replay_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                uint32_t fast_div_r_4 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_4 * (uint32_t)(replay_col_blocks_div.divisor);
                block_3 = first_block_3 + (int)fast_div_q_4;
                if (block_3 < end_block_3) {
                    int index_2 = tile_word_5;
                    int offset_6 = tile_word_6 / 256;
                    int first_row_1 = ((first_block_3) > (offset_6) ? (first_block_3) : (offset_6));
                    int _min_151 = ((end_block_3) < (offset_6 + tile_word_7 / 256) ? (end_block_3) : (offset_6 + tile_word_7 / 256));
                    int rows_3 = _min_151 - first_row_1;
                    int supergroup_5 = (tile_word_1 - (first_row_1 - first_block_3) * col_blocks_8) / (rows_3 * 8);
                    int full_cols_5 = col_blocks_8 / 8 * 8;
                    int row_23 = 0;
                    int col_66 = 0;
                    if (tile_word_1 - (first_row_1 - first_block_3) * col_blocks_8 < rows_3 * full_cols_5) {
                        row_23 = (tile_word_1 - (first_row_1 - first_block_3) * col_blocks_8) % (rows_3 * 8) / 8;
                        col_66 = supergroup_5 * 8 + (tile_word_1 - (first_row_1 - first_block_3) * col_blocks_8) % 8;
                    } else {
                        row_23 = (tile_word_1 - (first_row_1 - first_block_3) * col_blocks_8 - rows_3 * full_cols_5) / (col_blocks_8 - full_cols_5);
                        col_66 = full_cols_5 + (tile_word_1 - (first_row_1 - first_block_3) * col_blocks_8 - rows_3 * full_cols_5) % (col_blocks_8 - full_cols_5);
                    }
                    if ((supergroup_5 & 1) != 0) {
                        row_23 = rows_3 - row_23 - 1;
                    }
                    x_5 = first_row_1 + row_23 - tile_word_2 * (macro_size / 256);
                    y_5 = col_66;
                    expert_5 = index_2;
                }
                int has_hi_5 = 0;
                int global_mini_5 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                int macro_rows_5 = tile_word_2 * (macro_size / 256);
                int iterations_5 = hidden / 128;
                int macro_k = tile_word_2 * (macro_size / 128);
                if (expert_5 < 0) {
                    if (tid == 0) {
                    }
                } else {
                    if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_152 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                    int mini_rows_3 = ((0) > (_min_152) ? (0) : (_min_152));
                                    int required_4 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_6 = 1;
                                    if (enabled_value_6 != 0) {
                                        int poll_9_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_9_first) : "l"(replay_x + global_mini_5) : "memory");
                                        int poll_9 = poll_9_first;
                                        while (poll_9 < required_4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_9_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_9_next) : "l"(replay_x + global_mini_5) : "memory");
                                            poll_9 = poll_9_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                #pragma unroll 1
                                for (int idx_10 = 0; idx_10 < iterations_5; idx_10++) {
                                    int _stage_gemm_finished_5 = (int)((unsigned int)(gemm_wide + idx_10) % 4);
                                    int _parity_gemm_finished_5 = (int)((unsigned int)(gemm_wide + idx_10) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_5) * 8, _parity_gemm_finished_5);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_10) % 4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_10) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_10) % 4 * 16384)), "l"((&wg_q)), "r"(0), "r"(y_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(expert_5), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_10) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                }
                            }
                        }
                    } else if (tid / 32 == 6) {
                        if (warp == 6) {
                            if (elect_sync()) {
                                {
                                    int _min_153 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                    int mini_rows_4 = ((0) > (_min_153) ? (0) : (_min_153));
                                    int required_5 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_7 = 1;
                                    if (enabled_value_7 != 0) {
                                        int poll_10_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_10_first) : "l"(replay_x + global_mini_5) : "memory");
                                        int poll_10 = poll_10_first;
                                        while (poll_10 < required_5) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_10_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_10_next) : "l"(replay_x + global_mini_5) : "memory");
                                            poll_10 = poll_10_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                #pragma unroll 1
                                for (int idx_11 = 0; idx_11 < iterations_5; idx_11++) {
                                    int _stage_scales_finished = (int)((unsigned int)(scales_wide + idx_11) % 4);
                                    int _parity_scales_finished = (int)((unsigned int)(scales_wide + idx_11) / 4 % 2) ^ 1;
                                    mbarrier_wait(scales_finished_addr + (_stage_scales_finished) * 8, _parity_scales_finished);
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_11) % 4 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_5 * 2 + cta_rank_0) * (hidden / 128) + idx_11),
                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_11) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_11) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((expert_5 * i_tiles_1 + y_5 * 2 + cta_rank_0) * (hidden / 128) + idx_11),
                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_11) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int _parity_output_finished_5 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr, _parity_output_finished_5);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_12 = 0; idx_12 < iterations_5; idx_12++) {
                                        mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_12) % 4)) * 8, 3072);
                                        int _stage_scales_arrived = (int)((unsigned int)(scales_wide + idx_12) % 4);
                                        int _parity_scales_arrived = (int)((unsigned int)(scales_wide + idx_12) / 4 % 2);
                                        mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived) * 8, _parity_scales_arrived);
                                        int buffer = idx_12 % 3;
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_12) % 4 * 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_12) % 4 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_12) % 4 * 1024) + 512)));
                                        tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_12) % 4)) * 8, (uint16_t)(3));
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8, 65536);
                                        int _stage_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_12) % 4);
                                        int _parity_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_12) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_5) * 8, _parity_gemm_arrived_5);
                                        int _smem_desc_lo_154 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_12) % 4) * 1024;
                                        int _smem_desc_lo_155 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_12) % 4) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_154) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_155) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb + buffer * 8 + 0)), ((idx_12 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer * 8 + 0) | 0x80000000), 1);
                                        }
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            int _parity_output_arrived_5 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_5);
                            int warp_row_3 = tid / 32 * 32;
                            unsigned int packed_7[128];
                            #pragma unroll
                            for (int i_72 = 0; i_72 < 8; i_72++) {
                                float tmem_load_3[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(tmem_load_3[0]), "=f"(tmem_load_3[1]), "=f"(tmem_load_3[2]), "=f"(tmem_load_3[3]), "=f"(tmem_load_3[4]), "=f"(tmem_load_3[5]), "=f"(tmem_load_3[6]), "=f"(tmem_load_3[7]), "=f"(tmem_load_3[8]), "=f"(tmem_load_3[9]), "=f"(tmem_load_3[10]), "=f"(tmem_load_3[11]), "=f"(tmem_load_3[12]), "=f"(tmem_load_3[13]), "=f"(tmem_load_3[14]), "=f"(tmem_load_3[15]), "=f"(tmem_load_3[16]), "=f"(tmem_load_3[17]), "=f"(tmem_load_3[18]), "=f"(tmem_load_3[19]), "=f"(tmem_load_3[20]), "=f"(tmem_load_3[21]), "=f"(tmem_load_3[22]), "=f"(tmem_load_3[23]), "=f"(tmem_load_3[24]), "=f"(tmem_load_3[25]), "=f"(tmem_load_3[26]), "=f"(tmem_load_3[27]), "=f"(tmem_load_3[28]), "=f"(tmem_load_3[29]), "=f"(tmem_load_3[30]), "=f"(tmem_load_3[31])
                                    : "r"(taddr_1 + (unsigned int)(warp_row_3 << 16) + (unsigned int)(i_72 * 32)));
                                #pragma unroll
                                for (int j_24 = 0; j_24 < 16; j_24++) {
                                    packed_7[i_72 * 16 + j_24] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_3[2 * j_24], tmem_load_3[2 * j_24 + 1])));
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            }
                            unsigned int scale_word_12 = 0;
                            unsigned int block_0[16];
                            #pragma unroll
                            for (int j_0 = 0; j_0 < 16; j_0++) {
                                block_0[j_0] = packed_7[j_0];
                            }
                            #pragma unroll
                            for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                                unsigned int address_0 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_1_1 * 16);
                                address_0 = address_0 ^ (address_0 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_0 - smem_v52_addr)), "r"(packed_7[4 * j_1_1]), "r"(packed_7[4 * j_1_1 + 1]), "r"(packed_7[4 * j_1_1 + 2]), "r"(packed_7[4 * j_1_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8, 0, 0, d_smem_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_25;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_25) : "r"(block_0[0]));
                            unsigned int amax_pair_24 = bf16x2_abs_25;
                            #pragma unroll
                            for (int i_73 = 1; i_73 < 16; i_73++) {
                                uint32_t bf16x2_abs_1_36;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_36) : "r"(block_0[i_73]));
                                uint32_t bf16x2_max_36;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_36) : "r"(amax_pair_24), "r"(bf16x2_abs_1_36));
                                amax_pair_24 = bf16x2_max_36;
                            }
                            uint16_t bf16_max_24;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_24) : "h"((uint16_t)(amax_pair_24 & 65535)), "h"((uint16_t)(amax_pair_24 >> 16)));
                            float amax_24;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_24) : "h"((uint16_t)(bf16_max_24)));
                            float scale_24 = fmaxf(amax_24 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_156;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_156) : "f"(scale_24), "f"(scale_24));
                            unsigned int scale_byte_24 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_156 & 255;
                            unsigned int inverse_lane_24 = 254 - scale_byte_24 << 7;
                            unsigned int inverse_24 = inverse_lane_24 | inverse_lane_24 << 16;
                            unsigned int words_24[8];
                            #pragma unroll
                            for (int i_74 = 0; i_74 < 8; i_74++) {
                                uint32_t bf16x2_mul_37;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_37) : "r"(block_0[i_74 * 2]), "r"(inverse_24));
                                uint16_t e4m3x2_bf16x2_36;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_36) : "r"(bf16x2_mul_37));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_37);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_36) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_36;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_36) : "r"(block_0[i_74 * 2 + 1]), "r"(inverse_24));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_157;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "r"(bf16x2_mul_1_36));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_36);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_24[i_74] = (unsigned int)e4m3x2_bf16x2_36 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_157 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_24[0]), "r"(words_24[1]), "r"(words_24[2]), "r"(words_24[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24[4]), "r"(words_24[5]), "r"(words_24[6]), "r"(words_24[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_1_1[16];
                            #pragma unroll
                            for (int j_2_1 = 0; j_2_1 < 16; j_2_1++) {
                                block_1_1[j_2_1] = packed_7[16 + j_2_1];
                            }
                            #pragma unroll
                            for (int j_3_1 = 0; j_3_1 < 4; j_3_1++) {
                                unsigned int address_1_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_3_1 * 16);
                                address_1_1 = address_1_1 ^ (address_1_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_1_1 - smem_v52_addr)), "r"(packed_7[16 + 4 * j_3_1]), "r"(packed_7[16 + 4 * j_3_1 + 1]), "r"(packed_7[16 + 4 * j_3_1 + 2]), "r"(packed_7[16 + 4 * j_3_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 1, 0, 0, d_smem_addr + 8192);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_12_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_12_1) : "r"(block_1_1[0]));
                            unsigned int amax_pair_12_1 = bf16x2_abs_12_1;
                            #pragma unroll
                            for (int i_75 = 1; i_75 < 16; i_75++) {
                                uint32_t bf16x2_abs_1_37;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_37) : "r"(block_1_1[i_75]));
                                uint32_t bf16x2_max_37;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_37) : "r"(amax_pair_12_1), "r"(bf16x2_abs_1_37));
                                amax_pair_12_1 = bf16x2_max_37;
                            }
                            uint16_t bf16_max_12_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_12_1) : "h"((uint16_t)(amax_pair_12_1 & 65535)), "h"((uint16_t)(amax_pair_12_1 >> 16)));
                            float amax_12_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_12_1) : "h"((uint16_t)(bf16_max_12_1)));
                            float scale_12_1 = fmaxf(amax_12_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_158;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_158) : "f"(scale_12_1), "f"(scale_12_1));
                            unsigned int scale_byte_12_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_158 & 255;
                            unsigned int inverse_lane_12_1 = 254 - scale_byte_12_1 << 7;
                            unsigned int inverse_12_1 = inverse_lane_12_1 | inverse_lane_12_1 << 16;
                            unsigned int words_12_1[8];
                            #pragma unroll
                            for (int i_76 = 0; i_76 < 8; i_76++) {
                                uint32_t bf16x2_mul_38;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_38) : "r"(block_1_1[i_76 * 2]), "r"(inverse_12_1));
                                uint16_t e4m3x2_bf16x2_37;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_37) : "r"(bf16x2_mul_38));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_38);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_37) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_37;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_37) : "r"(block_1_1[i_76 * 2 + 1]), "r"(inverse_12_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_159;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "r"(bf16x2_mul_1_37));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_37);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_12_1[i_76] = (unsigned int)e4m3x2_bf16x2_37 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_159 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_12_1 << 8;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_12_1[0]), "r"(words_12_1[1]), "r"(words_12_1[2]), "r"(words_12_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_12_1[4]), "r"(words_12_1[5]), "r"(words_12_1[6]), "r"(words_12_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 32), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_2_1[16];
                            #pragma unroll
                            for (int j_4_1 = 0; j_4_1 < 16; j_4_1++) {
                                block_2_1[j_4_1] = packed_7[32 + j_4_1];
                            }
                            #pragma unroll
                            for (int j_5_1 = 0; j_5_1 < 4; j_5_1++) {
                                unsigned int address_2_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_5_1 * 16);
                                address_2_1 = address_2_1 ^ (address_2_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_2_1 - smem_v52_addr)), "r"(packed_7[32 + 4 * j_5_1]), "r"(packed_7[32 + 4 * j_5_1 + 1]), "r"(packed_7[32 + 4 * j_5_1 + 2]), "r"(packed_7[32 + 4 * j_5_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 2, 0, 0, d_smem_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_13_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_13_1) : "r"(block_2_1[0]));
                            unsigned int amax_pair_13_1 = bf16x2_abs_13_1;
                            #pragma unroll
                            for (int i_77 = 1; i_77 < 16; i_77++) {
                                uint32_t bf16x2_abs_1_38;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_38) : "r"(block_2_1[i_77]));
                                uint32_t bf16x2_max_38;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_38) : "r"(amax_pair_13_1), "r"(bf16x2_abs_1_38));
                                amax_pair_13_1 = bf16x2_max_38;
                            }
                            uint16_t bf16_max_13_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_13_1) : "h"((uint16_t)(amax_pair_13_1 & 65535)), "h"((uint16_t)(amax_pair_13_1 >> 16)));
                            float amax_13_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_13_1) : "h"((uint16_t)(bf16_max_13_1)));
                            float scale_13_1 = fmaxf(amax_13_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_160;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_160) : "f"(scale_13_1), "f"(scale_13_1));
                            unsigned int scale_byte_13_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_160 & 255;
                            unsigned int inverse_lane_13_1 = 254 - scale_byte_13_1 << 7;
                            unsigned int inverse_13_1 = inverse_lane_13_1 | inverse_lane_13_1 << 16;
                            unsigned int words_13_1[8];
                            #pragma unroll
                            for (int i_78 = 0; i_78 < 8; i_78++) {
                                uint32_t bf16x2_mul_39;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_39) : "r"(block_2_1[i_78 * 2]), "r"(inverse_13_1));
                                uint16_t e4m3x2_bf16x2_38;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_38) : "r"(bf16x2_mul_39));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_39);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_38) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_38;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_38) : "r"(block_2_1[i_78 * 2 + 1]), "r"(inverse_13_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_161;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_161) : "r"(bf16x2_mul_1_38));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_38);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_161) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_13_1[i_78] = (unsigned int)e4m3x2_bf16x2_38 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_161 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_13_1 << 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_13_1[0]), "r"(words_13_1[1]), "r"(words_13_1[2]), "r"(words_13_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_13_1[4]), "r"(words_13_1[5]), "r"(words_13_1[6]), "r"(words_13_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 64), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_3_1[16];
                            #pragma unroll
                            for (int j_6_1 = 0; j_6_1 < 16; j_6_1++) {
                                block_3_1[j_6_1] = packed_7[48 + j_6_1];
                            }
                            #pragma unroll
                            for (int j_7_1 = 0; j_7_1 < 4; j_7_1++) {
                                unsigned int address_3_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_7_1 * 16);
                                address_3_1 = address_3_1 ^ (address_3_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_3_1 - smem_v52_addr)), "r"(packed_7[48 + 4 * j_7_1]), "r"(packed_7[48 + 4 * j_7_1 + 1]), "r"(packed_7[48 + 4 * j_7_1 + 2]), "r"(packed_7[48 + 4 * j_7_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 3, 0, 0, d_smem_addr + 8192);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_14_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_14_1) : "r"(block_3_1[0]));
                            unsigned int amax_pair_14_1 = bf16x2_abs_14_1;
                            #pragma unroll
                            for (int i_79 = 1; i_79 < 16; i_79++) {
                                uint32_t bf16x2_abs_1_39;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_39) : "r"(block_3_1[i_79]));
                                uint32_t bf16x2_max_39;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_39) : "r"(amax_pair_14_1), "r"(bf16x2_abs_1_39));
                                amax_pair_14_1 = bf16x2_max_39;
                            }
                            uint16_t bf16_max_14_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_14_1) : "h"((uint16_t)(amax_pair_14_1 & 65535)), "h"((uint16_t)(amax_pair_14_1 >> 16)));
                            float amax_14_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_14_1) : "h"((uint16_t)(bf16_max_14_1)));
                            float scale_14_1 = fmaxf(amax_14_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_162;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_162) : "f"(scale_14_1), "f"(scale_14_1));
                            unsigned int scale_byte_14_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_162 & 255;
                            unsigned int inverse_lane_14_1 = 254 - scale_byte_14_1 << 7;
                            unsigned int inverse_14_1 = inverse_lane_14_1 | inverse_lane_14_1 << 16;
                            unsigned int words_14_1[8];
                            #pragma unroll
                            for (int i_80 = 0; i_80 < 8; i_80++) {
                                uint32_t bf16x2_mul_40;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_40) : "r"(block_3_1[i_80 * 2]), "r"(inverse_14_1));
                                uint16_t e4m3x2_bf16x2_39;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_39) : "r"(bf16x2_mul_40));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_40);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_39) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_39;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_39) : "r"(block_3_1[i_80 * 2 + 1]), "r"(inverse_14_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_163;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "r"(bf16x2_mul_1_39));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_39);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_14_1[i_80] = (unsigned int)e4m3x2_bf16x2_39 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_163 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_14_1 << 24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_14_1[0]), "r"(words_14_1[1]), "r"(words_14_1[2]), "r"(words_14_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_14_1[4]), "r"(words_14_1[5]), "r"(words_14_1[6]), "r"(words_14_1[7]) : "memory");
                            smem_v54[tid % 32 * 4 + tid / 32] = scale_word_12;
                            scale_word_12 = 0;
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 96), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                tma_store_3d((&gate_sc_store), 0, 0, (x_5 * 2 + cta_rank_0) * i_tiles + y_5 * 2, smem_v54_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_4[16];
                            #pragma unroll
                            for (int j_8_1 = 0; j_8_1 < 16; j_8_1++) {
                                block_4[j_8_1] = packed_7[64 + j_8_1];
                            }
                            #pragma unroll
                            for (int j_9_1 = 0; j_9_1 < 4; j_9_1++) {
                                unsigned int address_4_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_9_1 * 16);
                                address_4_1 = address_4_1 ^ (address_4_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_4_1 - smem_v52_addr)), "r"(packed_7[64 + 4 * j_9_1]), "r"(packed_7[64 + 4 * j_9_1 + 1]), "r"(packed_7[64 + 4 * j_9_1 + 2]), "r"(packed_7[64 + 4 * j_9_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 4, 0, 0, d_smem_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_15_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_15_1) : "r"(block_4[0]));
                            unsigned int amax_pair_15_1 = bf16x2_abs_15_1;
                            #pragma unroll
                            for (int i_81 = 1; i_81 < 16; i_81++) {
                                uint32_t bf16x2_abs_1_40;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_40) : "r"(block_4[i_81]));
                                uint32_t bf16x2_max_40;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_40) : "r"(amax_pair_15_1), "r"(bf16x2_abs_1_40));
                                amax_pair_15_1 = bf16x2_max_40;
                            }
                            uint16_t bf16_max_15_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_15_1) : "h"((uint16_t)(amax_pair_15_1 & 65535)), "h"((uint16_t)(amax_pair_15_1 >> 16)));
                            float amax_15_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_15_1) : "h"((uint16_t)(bf16_max_15_1)));
                            float scale_15_1 = fmaxf(amax_15_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_164;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_164) : "f"(scale_15_1), "f"(scale_15_1));
                            unsigned int scale_byte_15_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_164 & 255;
                            unsigned int inverse_lane_15_1 = 254 - scale_byte_15_1 << 7;
                            unsigned int inverse_15_1 = inverse_lane_15_1 | inverse_lane_15_1 << 16;
                            unsigned int words_15_1[8];
                            #pragma unroll
                            for (int i_82 = 0; i_82 < 8; i_82++) {
                                uint32_t bf16x2_mul_41;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_41) : "r"(block_4[i_82 * 2]), "r"(inverse_15_1));
                                uint16_t e4m3x2_bf16x2_40;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_40) : "r"(bf16x2_mul_41));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_41);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_40) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_40;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_40) : "r"(block_4[i_82 * 2 + 1]), "r"(inverse_15_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_165;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "r"(bf16x2_mul_1_40));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_40);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_15_1[i_82] = (unsigned int)e4m3x2_bf16x2_40 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_165 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_15_1;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_15_1[0]), "r"(words_15_1[1]), "r"(words_15_1[2]), "r"(words_15_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_15_1[4]), "r"(words_15_1[5]), "r"(words_15_1[6]), "r"(words_15_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 128), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_5[16];
                            #pragma unroll
                            for (int j_10_1 = 0; j_10_1 < 16; j_10_1++) {
                                block_5[j_10_1] = packed_7[80 + j_10_1];
                            }
                            #pragma unroll
                            for (int j_11_1 = 0; j_11_1 < 4; j_11_1++) {
                                unsigned int address_5_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_11_1 * 16);
                                address_5_1 = address_5_1 ^ (address_5_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_5_1 - smem_v52_addr)), "r"(packed_7[80 + 4 * j_11_1]), "r"(packed_7[80 + 4 * j_11_1 + 1]), "r"(packed_7[80 + 4 * j_11_1 + 2]), "r"(packed_7[80 + 4 * j_11_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 5, 0, 0, d_smem_addr + 8192);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_16_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_16_1) : "r"(block_5[0]));
                            unsigned int amax_pair_16_1 = bf16x2_abs_16_1;
                            #pragma unroll
                            for (int i_83 = 1; i_83 < 16; i_83++) {
                                uint32_t bf16x2_abs_1_41;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_41) : "r"(block_5[i_83]));
                                uint32_t bf16x2_max_41;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_41) : "r"(amax_pair_16_1), "r"(bf16x2_abs_1_41));
                                amax_pair_16_1 = bf16x2_max_41;
                            }
                            uint16_t bf16_max_16_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_16_1) : "h"((uint16_t)(amax_pair_16_1 & 65535)), "h"((uint16_t)(amax_pair_16_1 >> 16)));
                            float amax_16_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_16_1) : "h"((uint16_t)(bf16_max_16_1)));
                            float scale_16_1 = fmaxf(amax_16_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_166;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_166) : "f"(scale_16_1), "f"(scale_16_1));
                            unsigned int scale_byte_16_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_166 & 255;
                            unsigned int inverse_lane_16_1 = 254 - scale_byte_16_1 << 7;
                            unsigned int inverse_16_1 = inverse_lane_16_1 | inverse_lane_16_1 << 16;
                            unsigned int words_16_1[8];
                            #pragma unroll
                            for (int i_84 = 0; i_84 < 8; i_84++) {
                                uint32_t bf16x2_mul_42;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_42) : "r"(block_5[i_84 * 2]), "r"(inverse_16_1));
                                uint16_t e4m3x2_bf16x2_41;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_41) : "r"(bf16x2_mul_42));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_42);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_41) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_41;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_41) : "r"(block_5[i_84 * 2 + 1]), "r"(inverse_16_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_167;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_167) : "r"(bf16x2_mul_1_41));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_41);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_167) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_16_1[i_84] = (unsigned int)e4m3x2_bf16x2_41 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_167 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_16_1 << 8;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_16_1[0]), "r"(words_16_1[1]), "r"(words_16_1[2]), "r"(words_16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16_1[4]), "r"(words_16_1[5]), "r"(words_16_1[6]), "r"(words_16_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 160), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_6[16];
                            #pragma unroll
                            for (int j_12_1 = 0; j_12_1 < 16; j_12_1++) {
                                block_6[j_12_1] = packed_7[96 + j_12_1];
                            }
                            #pragma unroll
                            for (int j_13_1 = 0; j_13_1 < 4; j_13_1++) {
                                unsigned int address_6_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_13_1 * 16);
                                address_6_1 = address_6_1 ^ (address_6_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_6_1 - smem_v52_addr)), "r"(packed_7[96 + 4 * j_13_1]), "r"(packed_7[96 + 4 * j_13_1 + 1]), "r"(packed_7[96 + 4 * j_13_1 + 2]), "r"(packed_7[96 + 4 * j_13_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 6, 0, 0, d_smem_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_17_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_17_1) : "r"(block_6[0]));
                            unsigned int amax_pair_17_1 = bf16x2_abs_17_1;
                            #pragma unroll
                            for (int i_85 = 1; i_85 < 16; i_85++) {
                                uint32_t bf16x2_abs_1_42;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_42) : "r"(block_6[i_85]));
                                uint32_t bf16x2_max_42;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_42) : "r"(amax_pair_17_1), "r"(bf16x2_abs_1_42));
                                amax_pair_17_1 = bf16x2_max_42;
                            }
                            uint16_t bf16_max_17_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_17_1) : "h"((uint16_t)(amax_pair_17_1 & 65535)), "h"((uint16_t)(amax_pair_17_1 >> 16)));
                            float amax_17_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_17_1) : "h"((uint16_t)(bf16_max_17_1)));
                            float scale_17_1 = fmaxf(amax_17_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_168;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_168) : "f"(scale_17_1), "f"(scale_17_1));
                            unsigned int scale_byte_17_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_168 & 255;
                            unsigned int inverse_lane_17_1 = 254 - scale_byte_17_1 << 7;
                            unsigned int inverse_17_1 = inverse_lane_17_1 | inverse_lane_17_1 << 16;
                            unsigned int words_17_1[8];
                            #pragma unroll
                            for (int i_86 = 0; i_86 < 8; i_86++) {
                                uint32_t bf16x2_mul_43;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_43) : "r"(block_6[i_86 * 2]), "r"(inverse_17_1));
                                uint16_t e4m3x2_bf16x2_42;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_42) : "r"(bf16x2_mul_43));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_43);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_42) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_42;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_42) : "r"(block_6[i_86 * 2 + 1]), "r"(inverse_17_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_169;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_169) : "r"(bf16x2_mul_1_42));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_42);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_169) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_17_1[i_86] = (unsigned int)e4m3x2_bf16x2_42 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_169 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_17_1 << 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_17_1[0]), "r"(words_17_1[1]), "r"(words_17_1[2]), "r"(words_17_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_17_1[4]), "r"(words_17_1[5]), "r"(words_17_1[6]), "r"(words_17_1[7]) : "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 192), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int block_7[16];
                            #pragma unroll
                            for (int j_14_1 = 0; j_14_1 < 16; j_14_1++) {
                                block_7[j_14_1] = packed_7[112 + j_14_1];
                            }
                            #pragma unroll
                            for (int j_15_1 = 0; j_15_1 < 4; j_15_1++) {
                                unsigned int address_7_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_15_1 * 16);
                                address_7_1 = address_7_1 ^ (address_7_1 & 511) >> 7 << 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_7_1 - smem_v52_addr)), "r"(packed_7[112 + 4 * j_15_1]), "r"(packed_7[112 + 4 * j_15_1 + 1]), "r"(packed_7[112 + 4 * j_15_1 + 2]), "r"(packed_7[112 + 4 * j_15_1 + 3]) : "memory");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&gate_out_r), 0, x_5 * 256 + cta_rank_0 * 128, y_5 * 8 + 7, 0, 0, d_smem_addr + 8192);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            uint32_t bf16x2_abs_18_1;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_18_1) : "r"(block_7[0]));
                            unsigned int amax_pair_18_1 = bf16x2_abs_18_1;
                            #pragma unroll
                            for (int i_87 = 1; i_87 < 16; i_87++) {
                                uint32_t bf16x2_abs_1_43;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_43) : "r"(block_7[i_87]));
                                uint32_t bf16x2_max_43;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_43) : "r"(amax_pair_18_1), "r"(bf16x2_abs_1_43));
                                amax_pair_18_1 = bf16x2_max_43;
                            }
                            uint16_t bf16_max_18_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_18_1) : "h"((uint16_t)(amax_pair_18_1 & 65535)), "h"((uint16_t)(amax_pair_18_1 >> 16)));
                            float amax_18_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18_1) : "h"((uint16_t)(bf16_max_18_1)));
                            float scale_18_1 = fmaxf(amax_18_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_170;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_170) : "f"(scale_18_1), "f"(scale_18_1));
                            unsigned int scale_byte_18_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_170 & 255;
                            unsigned int inverse_lane_18_1 = 254 - scale_byte_18_1 << 7;
                            unsigned int inverse_18_1 = inverse_lane_18_1 | inverse_lane_18_1 << 16;
                            unsigned int words_18_1[8];
                            #pragma unroll
                            for (int i_88 = 0; i_88 < 8; i_88++) {
                                uint32_t bf16x2_mul_44;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_44) : "r"(block_7[i_88 * 2]), "r"(inverse_18_1));
                                uint16_t e4m3x2_bf16x2_43;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_43) : "r"(bf16x2_mul_44));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_44);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_43) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t bf16x2_mul_1_43;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_43) : "r"(block_7[i_88 * 2 + 1]), "r"(inverse_18_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_171;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "r"(bf16x2_mul_1_43));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_43);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_18_1[i_88] = (unsigned int)e4m3x2_bf16x2_43 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_171 << 16;
                            }
                            scale_word_12 = scale_word_12 | scale_byte_18_1 << 24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_18_1[0]), "r"(words_18_1[1]), "r"(words_18_1[2]), "r"(words_18_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_18_1[4]), "r"(words_18_1[5]), "r"(words_18_1[6]), "r"(words_18_1[7]) : "memory");
                            smem_v55[tid % 32 * 4 + tid / 32] = scale_word_12;
                            scale_word_12 = 0;
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2}], [%3], %4;"
                                    :: "l"((&gate_q_store)), "r"(y_5 * 256 + 224), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                tma_store_3d((&gate_sc_store), 0, 0, (x_5 * 2 + cta_rank_0) * i_tiles + y_5 * 2 + 1, smem_v55_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
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
                    gemm_wide += iterations_5;
                    scales_wide += iterations_5;
                    gemm_out0 += 1;
                    gemm_acc_tiles += 1;
                }
            } else {
                if (tile_word_0 == 7) {
                    int col_blocks_9 = (intermediate + 256 - 1) / 256;
                    int x_6 = -1;
                    int y_6 = -1;
                    int expert_6 = -1;
                    int k_start_6 = 0;
                    int k_end_6 = 0;
                    int first_6 = 0;
                    int first_block_4 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                    int end_block_4 = ((first_block_4 + mini_size / 256) < (tokens / 256) ? (first_block_4 + mini_size / 256) : (tokens / 256));
                    int block_8 = first_block_4;
                    uint32_t fast_div_q_5 = (replay_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), replay_col_blocks_div.multiplier) >> replay_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                    uint32_t fast_div_r_5 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_5 * (uint32_t)(replay_col_blocks_div.divisor);
                    block_8 = first_block_4 + (int)fast_div_q_5;
                    if (block_8 < end_block_4) {
                        int index_3 = tile_word_5;
                        int offset_7 = tile_word_6 / 256;
                        int first_row_2 = ((first_block_4) > (offset_7) ? (first_block_4) : (offset_7));
                        int _min_172 = ((end_block_4) < (offset_7 + tile_word_7 / 256) ? (end_block_4) : (offset_7 + tile_word_7 / 256));
                        int rows_4 = _min_172 - first_row_2;
                        int supergroup_6 = (tile_word_1 - (first_row_2 - first_block_4) * col_blocks_9) / (rows_4 * 8);
                        int full_cols_6 = col_blocks_9 / 8 * 8;
                        int row_24 = 0;
                        int col_67 = 0;
                        if (tile_word_1 - (first_row_2 - first_block_4) * col_blocks_9 < rows_4 * full_cols_6) {
                            row_24 = (tile_word_1 - (first_row_2 - first_block_4) * col_blocks_9) % (rows_4 * 8) / 8;
                            col_67 = supergroup_6 * 8 + (tile_word_1 - (first_row_2 - first_block_4) * col_blocks_9) % 8;
                        } else {
                            row_24 = (tile_word_1 - (first_row_2 - first_block_4) * col_blocks_9 - rows_4 * full_cols_6) / (col_blocks_9 - full_cols_6);
                            col_67 = full_cols_6 + (tile_word_1 - (first_row_2 - first_block_4) * col_blocks_9 - rows_4 * full_cols_6) % (col_blocks_9 - full_cols_6);
                        }
                        if ((supergroup_6 & 1) != 0) {
                            row_24 = rows_4 - row_24 - 1;
                        }
                        x_6 = first_row_2 + row_24 - tile_word_2 * (macro_size / 256);
                        y_6 = col_67;
                        expert_6 = index_3;
                    }
                    int has_hi_6 = 0;
                    int global_mini_6 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                    int macro_rows_6 = tile_word_2 * (macro_size / 256);
                    int iterations_6 = hidden / 128;
                    int macro_k_1 = tile_word_2 * (macro_size / 128);
                    if (expert_6 < 0) {
                        if (tid == 0) {
                        }
                    } else {
                        if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_173 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                        int mini_rows_5 = ((0) > (_min_173) ? (0) : (_min_173));
                                        int required_6 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_9 = 1;
                                        if (enabled_value_9 != 0) {
                                            int poll_11_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_11_first) : "l"(replay_x + global_mini_6) : "memory");
                                            int poll_11 = poll_11_first;
                                            while (poll_11 < required_6) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_11_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_11_next) : "l"(replay_x + global_mini_6) : "memory");
                                                poll_11 = poll_11_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    #pragma unroll 1
                                    for (int idx_13 = 0; idx_13 < iterations_6; idx_13++) {
                                        int _stage_gemm_finished_6 = (int)((unsigned int)(gemm_wide + idx_13) % 4);
                                        int _parity_gemm_finished_6 = (int)((unsigned int)(gemm_wide + idx_13) / 4 % 2) ^ 1;
                                        mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_6) * 8, _parity_gemm_finished_6);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_13) % 4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(idx_13), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_13) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_13) % 4 * 16384)), "l"((&wu_q)), "r"(0), "r"(y_6 * 256 + cta_rank_0 * 128), "r"(idx_13), "r"(expert_6), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_13) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        } else if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        int _min_174 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                        int mini_rows_6 = ((0) > (_min_174) ? (0) : (_min_174));
                                        int required_7 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_10 = 1;
                                        if (enabled_value_10 != 0) {
                                            int poll_12_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_12_first) : "l"(replay_x + global_mini_6) : "memory");
                                            int poll_12 = poll_12_first;
                                            while (poll_12 < required_7) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_12_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_12_next) : "l"(replay_x + global_mini_6) : "memory");
                                                poll_12 = poll_12_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    #pragma unroll 1
                                    for (int idx_14 = 0; idx_14 < iterations_6; idx_14++) {
                                        int _stage_scales_finished_1 = (int)((unsigned int)(scales_wide + idx_14) % 4);
                                        int _parity_scales_finished_1 = (int)((unsigned int)(scales_wide + idx_14) / 4 % 2) ^ 1;
                                        mbarrier_wait(scales_finished_addr + (_stage_scales_finished_1) * 8, _parity_scales_finished_1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_14) % 4 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_6 * 2 + cta_rank_0) * (hidden / 128) + idx_14),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_14) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_14) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((expert_6 * i_tiles_1 + y_6 * 2 + cta_rank_0) * (hidden / 128) + idx_14),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_14) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int _parity_output_finished_6 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr, _parity_output_finished_6);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_15 = 0; idx_15 < iterations_6; idx_15++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_15) % 4)) * 8, 3072);
                                            int _stage_scales_arrived_1 = (int)((unsigned int)(scales_wide + idx_15) % 4);
                                            int _parity_scales_arrived_1 = (int)((unsigned int)(scales_wide + idx_15) / 4 % 2);
                                            mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_1) * 8, _parity_scales_arrived_1);
                                            int buffer_1 = idx_15 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_1 * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_15) % 4 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_1 * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_15) % 4 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_15) % 4 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_15) % 4)) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_15) % 4)) * 8, 65536);
                                            int _stage_gemm_arrived_6 = (int)((unsigned int)(gemm_wide + idx_15) % 4);
                                            int _parity_gemm_arrived_6 = (int)((unsigned int)(gemm_wide + idx_15) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_6) * 8, _parity_gemm_arrived_6);
                                            int _smem_desc_lo_175 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                            int _smem_desc_lo_176 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_smem_desc_lo_175) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_smem_desc_lo_176) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0)), ((idx_15 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0) | 0x80000000), 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_15) % 4)) * 8, (uint16_t)(3));
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                int _parity_output_arrived_6 = (int)((unsigned int)gemm_out0 % 2);
                                mbarrier_wait(output_arrived_addr, _parity_output_arrived_6);
                                int warp_row_4 = tid / 32 * 32;
                                unsigned int packed_8[128];
                                #pragma unroll
                                for (int i_89 = 0; i_89 < 8; i_89++) {
                                    float tmem_load_4[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(tmem_load_4[0]), "=f"(tmem_load_4[1]), "=f"(tmem_load_4[2]), "=f"(tmem_load_4[3]), "=f"(tmem_load_4[4]), "=f"(tmem_load_4[5]), "=f"(tmem_load_4[6]), "=f"(tmem_load_4[7]), "=f"(tmem_load_4[8]), "=f"(tmem_load_4[9]), "=f"(tmem_load_4[10]), "=f"(tmem_load_4[11]), "=f"(tmem_load_4[12]), "=f"(tmem_load_4[13]), "=f"(tmem_load_4[14]), "=f"(tmem_load_4[15]), "=f"(tmem_load_4[16]), "=f"(tmem_load_4[17]), "=f"(tmem_load_4[18]), "=f"(tmem_load_4[19]), "=f"(tmem_load_4[20]), "=f"(tmem_load_4[21]), "=f"(tmem_load_4[22]), "=f"(tmem_load_4[23]), "=f"(tmem_load_4[24]), "=f"(tmem_load_4[25]), "=f"(tmem_load_4[26]), "=f"(tmem_load_4[27]), "=f"(tmem_load_4[28]), "=f"(tmem_load_4[29]), "=f"(tmem_load_4[30]), "=f"(tmem_load_4[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row_4 << 16) + (unsigned int)(i_89 * 32)));
                                    #pragma unroll
                                    for (int j_25 = 0; j_25 < 16; j_25++) {
                                        packed_8[i_89 * 16 + j_25] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_4[2 * j_25], tmem_load_4[2 * j_25 + 1])));
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                                unsigned int scale_word_13 = 0;
                                unsigned int block_8_1[16];
                                #pragma unroll
                                for (int j_16_1 = 0; j_16_1 < 16; j_16_1++) {
                                    block_8_1[j_16_1] = packed_8[j_16_1];
                                }
                                #pragma unroll
                                for (int j_17_1 = 0; j_17_1 < 4; j_17_1++) {
                                    unsigned int address_8_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_17_1 * 16);
                                    address_8_1 = address_8_1 ^ (address_8_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_8_1 - smem_v52_addr)), "r"(packed_8[4 * j_17_1]), "r"(packed_8[4 * j_17_1 + 1]), "r"(packed_8[4 * j_17_1 + 2]), "r"(packed_8[4 * j_17_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_26;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_26) : "r"(block_8_1[0]));
                                unsigned int amax_pair_25 = bf16x2_abs_26;
                                #pragma unroll
                                for (int i_90 = 1; i_90 < 16; i_90++) {
                                    uint32_t bf16x2_abs_1_44;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_44) : "r"(block_8_1[i_90]));
                                    uint32_t bf16x2_max_44;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_44) : "r"(amax_pair_25), "r"(bf16x2_abs_1_44));
                                    amax_pair_25 = bf16x2_max_44;
                                }
                                uint16_t bf16_max_25;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_25) : "h"((uint16_t)(amax_pair_25 & 65535)), "h"((uint16_t)(amax_pair_25 >> 16)));
                                float amax_25;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_25) : "h"((uint16_t)(bf16_max_25)));
                                float scale_25 = fmaxf(amax_25 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_177;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_177) : "f"(scale_25), "f"(scale_25));
                                unsigned int scale_byte_25 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_177 & 255;
                                unsigned int inverse_lane_25 = 254 - scale_byte_25 << 7;
                                unsigned int inverse_25 = inverse_lane_25 | inverse_lane_25 << 16;
                                unsigned int words_25[8];
                                #pragma unroll
                                for (int i_91 = 0; i_91 < 8; i_91++) {
                                    uint32_t bf16x2_mul_45;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_45) : "r"(block_8_1[i_91 * 2]), "r"(inverse_25));
                                    uint16_t e4m3x2_bf16x2_44;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_44) : "r"(bf16x2_mul_45));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_45);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_44) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_44;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_44) : "r"(block_8_1[i_91 * 2 + 1]), "r"(inverse_25));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_178;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_178) : "r"(bf16x2_mul_1_44));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_44);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_178) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_25[i_91] = (unsigned int)e4m3x2_bf16x2_44 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_178 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_25;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_25[0]), "r"(words_25[1]), "r"(words_25[2]), "r"(words_25[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_25[4]), "r"(words_25[5]), "r"(words_25[6]), "r"(words_25[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_9[16];
                                #pragma unroll
                                for (int j_18_1 = 0; j_18_1 < 16; j_18_1++) {
                                    block_9[j_18_1] = packed_8[16 + j_18_1];
                                }
                                #pragma unroll
                                for (int j_19_1 = 0; j_19_1 < 4; j_19_1++) {
                                    unsigned int address_9_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_19_1 * 16);
                                    address_9_1 = address_9_1 ^ (address_9_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_9_1 - smem_v52_addr)), "r"(packed_8[16 + 4 * j_19_1]), "r"(packed_8[16 + 4 * j_19_1 + 1]), "r"(packed_8[16 + 4 * j_19_1 + 2]), "r"(packed_8[16 + 4 * j_19_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 1, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_19_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_19_1) : "r"(block_9[0]));
                                unsigned int amax_pair_19_1 = bf16x2_abs_19_1;
                                #pragma unroll
                                for (int i_92 = 1; i_92 < 16; i_92++) {
                                    uint32_t bf16x2_abs_1_45;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_45) : "r"(block_9[i_92]));
                                    uint32_t bf16x2_max_45;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_45) : "r"(amax_pair_19_1), "r"(bf16x2_abs_1_45));
                                    amax_pair_19_1 = bf16x2_max_45;
                                }
                                uint16_t bf16_max_19_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_19_1) : "h"((uint16_t)(amax_pair_19_1 & 65535)), "h"((uint16_t)(amax_pair_19_1 >> 16)));
                                float amax_19_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_19_1) : "h"((uint16_t)(bf16_max_19_1)));
                                float scale_19_1 = fmaxf(amax_19_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_179;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_179) : "f"(scale_19_1), "f"(scale_19_1));
                                unsigned int scale_byte_19_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_179 & 255;
                                unsigned int inverse_lane_19_1 = 254 - scale_byte_19_1 << 7;
                                unsigned int inverse_19_1 = inverse_lane_19_1 | inverse_lane_19_1 << 16;
                                unsigned int words_19_1[8];
                                #pragma unroll
                                for (int i_93 = 0; i_93 < 8; i_93++) {
                                    uint32_t bf16x2_mul_46;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_46) : "r"(block_9[i_93 * 2]), "r"(inverse_19_1));
                                    uint16_t e4m3x2_bf16x2_45;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_45) : "r"(bf16x2_mul_46));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_46);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_45) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_45;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_45) : "r"(block_9[i_93 * 2 + 1]), "r"(inverse_19_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_180;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_180) : "r"(bf16x2_mul_1_45));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_45);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_180) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_19_1[i_93] = (unsigned int)e4m3x2_bf16x2_45 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_180 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_19_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_19_1[0]), "r"(words_19_1[1]), "r"(words_19_1[2]), "r"(words_19_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_19_1[4]), "r"(words_19_1[5]), "r"(words_19_1[6]), "r"(words_19_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 32), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_10[16];
                                #pragma unroll
                                for (int j_20_1 = 0; j_20_1 < 16; j_20_1++) {
                                    block_10[j_20_1] = packed_8[32 + j_20_1];
                                }
                                #pragma unroll
                                for (int j_21_1 = 0; j_21_1 < 4; j_21_1++) {
                                    unsigned int address_10_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_21_1 * 16);
                                    address_10_1 = address_10_1 ^ (address_10_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_10_1 - smem_v52_addr)), "r"(packed_8[32 + 4 * j_21_1]), "r"(packed_8[32 + 4 * j_21_1 + 1]), "r"(packed_8[32 + 4 * j_21_1 + 2]), "r"(packed_8[32 + 4 * j_21_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 2, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_20_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_20_1) : "r"(block_10[0]));
                                unsigned int amax_pair_20_1 = bf16x2_abs_20_1;
                                #pragma unroll
                                for (int i_94 = 1; i_94 < 16; i_94++) {
                                    uint32_t bf16x2_abs_1_46;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_46) : "r"(block_10[i_94]));
                                    uint32_t bf16x2_max_46;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_46) : "r"(amax_pair_20_1), "r"(bf16x2_abs_1_46));
                                    amax_pair_20_1 = bf16x2_max_46;
                                }
                                uint16_t bf16_max_20_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_20_1) : "h"((uint16_t)(amax_pair_20_1 & 65535)), "h"((uint16_t)(amax_pair_20_1 >> 16)));
                                float amax_20_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_20_1) : "h"((uint16_t)(bf16_max_20_1)));
                                float scale_20_1 = fmaxf(amax_20_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_181;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_181) : "f"(scale_20_1), "f"(scale_20_1));
                                unsigned int scale_byte_20_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_181 & 255;
                                unsigned int inverse_lane_20_1 = 254 - scale_byte_20_1 << 7;
                                unsigned int inverse_20_1 = inverse_lane_20_1 | inverse_lane_20_1 << 16;
                                unsigned int words_20_1[8];
                                #pragma unroll
                                for (int i_95 = 0; i_95 < 8; i_95++) {
                                    uint32_t bf16x2_mul_47;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_47) : "r"(block_10[i_95 * 2]), "r"(inverse_20_1));
                                    uint16_t e4m3x2_bf16x2_46;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_46) : "r"(bf16x2_mul_47));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_47);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_46) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_46;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_46) : "r"(block_10[i_95 * 2 + 1]), "r"(inverse_20_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_182;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_182) : "r"(bf16x2_mul_1_46));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_46);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_182) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_20_1[i_95] = (unsigned int)e4m3x2_bf16x2_46 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_182 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_20_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_20_1[0]), "r"(words_20_1[1]), "r"(words_20_1[2]), "r"(words_20_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_20_1[4]), "r"(words_20_1[5]), "r"(words_20_1[6]), "r"(words_20_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 64), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_11[16];
                                #pragma unroll
                                for (int j_22_1 = 0; j_22_1 < 16; j_22_1++) {
                                    block_11[j_22_1] = packed_8[48 + j_22_1];
                                }
                                #pragma unroll
                                for (int j_23_1 = 0; j_23_1 < 4; j_23_1++) {
                                    unsigned int address_11_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_23_1 * 16);
                                    address_11_1 = address_11_1 ^ (address_11_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_11_1 - smem_v52_addr)), "r"(packed_8[48 + 4 * j_23_1]), "r"(packed_8[48 + 4 * j_23_1 + 1]), "r"(packed_8[48 + 4 * j_23_1 + 2]), "r"(packed_8[48 + 4 * j_23_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 3, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_21_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_21_1) : "r"(block_11[0]));
                                unsigned int amax_pair_21_1 = bf16x2_abs_21_1;
                                #pragma unroll
                                for (int i_96 = 1; i_96 < 16; i_96++) {
                                    uint32_t bf16x2_abs_1_47;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_47) : "r"(block_11[i_96]));
                                    uint32_t bf16x2_max_47;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_47) : "r"(amax_pair_21_1), "r"(bf16x2_abs_1_47));
                                    amax_pair_21_1 = bf16x2_max_47;
                                }
                                uint16_t bf16_max_21_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_21_1) : "h"((uint16_t)(amax_pair_21_1 & 65535)), "h"((uint16_t)(amax_pair_21_1 >> 16)));
                                float amax_21_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_21_1) : "h"((uint16_t)(bf16_max_21_1)));
                                float scale_21_1 = fmaxf(amax_21_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_183;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_183) : "f"(scale_21_1), "f"(scale_21_1));
                                unsigned int scale_byte_21_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_183 & 255;
                                unsigned int inverse_lane_21_1 = 254 - scale_byte_21_1 << 7;
                                unsigned int inverse_21_1 = inverse_lane_21_1 | inverse_lane_21_1 << 16;
                                unsigned int words_21_1[8];
                                #pragma unroll
                                for (int i_97 = 0; i_97 < 8; i_97++) {
                                    uint32_t bf16x2_mul_48;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_48) : "r"(block_11[i_97 * 2]), "r"(inverse_21_1));
                                    uint16_t e4m3x2_bf16x2_47;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_47) : "r"(bf16x2_mul_48));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_48);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_47) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_47;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_47) : "r"(block_11[i_97 * 2 + 1]), "r"(inverse_21_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_184;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_184) : "r"(bf16x2_mul_1_47));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_47);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_184) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_21_1[i_97] = (unsigned int)e4m3x2_bf16x2_47 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_184 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_21_1 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_21_1[0]), "r"(words_21_1[1]), "r"(words_21_1[2]), "r"(words_21_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_21_1[4]), "r"(words_21_1[5]), "r"(words_21_1[6]), "r"(words_21_1[7]) : "memory");
                                smem_v54[tid % 32 * 4 + tid / 32] = scale_word_13;
                                scale_word_13 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 96), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_6 * 2 + cta_rank_0) * i_tiles + y_6 * 2, smem_v54_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_12[16];
                                #pragma unroll
                                for (int j_24_1 = 0; j_24_1 < 16; j_24_1++) {
                                    block_12[j_24_1] = packed_8[64 + j_24_1];
                                }
                                #pragma unroll
                                for (int j_25_1 = 0; j_25_1 < 4; j_25_1++) {
                                    unsigned int address_12_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_25_1 * 16);
                                    address_12_1 = address_12_1 ^ (address_12_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_12_1 - smem_v52_addr)), "r"(packed_8[64 + 4 * j_25_1]), "r"(packed_8[64 + 4 * j_25_1 + 1]), "r"(packed_8[64 + 4 * j_25_1 + 2]), "r"(packed_8[64 + 4 * j_25_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 4, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_22_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_22_1) : "r"(block_12[0]));
                                unsigned int amax_pair_22_1 = bf16x2_abs_22_1;
                                #pragma unroll
                                for (int i_98 = 1; i_98 < 16; i_98++) {
                                    uint32_t bf16x2_abs_1_48;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_48) : "r"(block_12[i_98]));
                                    uint32_t bf16x2_max_48;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_48) : "r"(amax_pair_22_1), "r"(bf16x2_abs_1_48));
                                    amax_pair_22_1 = bf16x2_max_48;
                                }
                                uint16_t bf16_max_22_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_22_1) : "h"((uint16_t)(amax_pair_22_1 & 65535)), "h"((uint16_t)(amax_pair_22_1 >> 16)));
                                float amax_22_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_22_1) : "h"((uint16_t)(bf16_max_22_1)));
                                float scale_22_1 = fmaxf(amax_22_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_185;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_185) : "f"(scale_22_1), "f"(scale_22_1));
                                unsigned int scale_byte_22_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_185 & 255;
                                unsigned int inverse_lane_22_1 = 254 - scale_byte_22_1 << 7;
                                unsigned int inverse_22_1 = inverse_lane_22_1 | inverse_lane_22_1 << 16;
                                unsigned int words_22_1[8];
                                #pragma unroll
                                for (int i_99 = 0; i_99 < 8; i_99++) {
                                    uint32_t bf16x2_mul_49;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_49) : "r"(block_12[i_99 * 2]), "r"(inverse_22_1));
                                    uint16_t e4m3x2_bf16x2_48;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_48) : "r"(bf16x2_mul_49));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_49);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_48) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_48;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_48) : "r"(block_12[i_99 * 2 + 1]), "r"(inverse_22_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_186;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_186) : "r"(bf16x2_mul_1_48));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_48);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_186) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_22_1[i_99] = (unsigned int)e4m3x2_bf16x2_48 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_186 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_22_1;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_22_1[0]), "r"(words_22_1[1]), "r"(words_22_1[2]), "r"(words_22_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_22_1[4]), "r"(words_22_1[5]), "r"(words_22_1[6]), "r"(words_22_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 128), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_13[16];
                                #pragma unroll
                                for (int j_26 = 0; j_26 < 16; j_26++) {
                                    block_13[j_26] = packed_8[80 + j_26];
                                }
                                #pragma unroll
                                for (int j_27 = 0; j_27 < 4; j_27++) {
                                    unsigned int address_13_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_27 * 16);
                                    address_13_1 = address_13_1 ^ (address_13_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_13_1 - smem_v52_addr)), "r"(packed_8[80 + 4 * j_27]), "r"(packed_8[80 + 4 * j_27 + 1]), "r"(packed_8[80 + 4 * j_27 + 2]), "r"(packed_8[80 + 4 * j_27 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 5, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_23_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_23_1) : "r"(block_13[0]));
                                unsigned int amax_pair_23_1 = bf16x2_abs_23_1;
                                #pragma unroll
                                for (int i_100 = 1; i_100 < 16; i_100++) {
                                    uint32_t bf16x2_abs_1_49;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_49) : "r"(block_13[i_100]));
                                    uint32_t bf16x2_max_49;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_49) : "r"(amax_pair_23_1), "r"(bf16x2_abs_1_49));
                                    amax_pair_23_1 = bf16x2_max_49;
                                }
                                uint16_t bf16_max_23_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_23_1) : "h"((uint16_t)(amax_pair_23_1 & 65535)), "h"((uint16_t)(amax_pair_23_1 >> 16)));
                                float amax_23_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_23_1) : "h"((uint16_t)(bf16_max_23_1)));
                                float scale_23_1 = fmaxf(amax_23_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_187;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_187) : "f"(scale_23_1), "f"(scale_23_1));
                                unsigned int scale_byte_23_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_187 & 255;
                                unsigned int inverse_lane_23_1 = 254 - scale_byte_23_1 << 7;
                                unsigned int inverse_23_1 = inverse_lane_23_1 | inverse_lane_23_1 << 16;
                                unsigned int words_23_1[8];
                                #pragma unroll
                                for (int i_101 = 0; i_101 < 8; i_101++) {
                                    uint32_t bf16x2_mul_50;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_50) : "r"(block_13[i_101 * 2]), "r"(inverse_23_1));
                                    uint16_t e4m3x2_bf16x2_49;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_49) : "r"(bf16x2_mul_50));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_50);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_49) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_49;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_49) : "r"(block_13[i_101 * 2 + 1]), "r"(inverse_23_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_188;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_188) : "r"(bf16x2_mul_1_49));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_49);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_188) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_23_1[i_101] = (unsigned int)e4m3x2_bf16x2_49 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_188 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_23_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_23_1[0]), "r"(words_23_1[1]), "r"(words_23_1[2]), "r"(words_23_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_23_1[4]), "r"(words_23_1[5]), "r"(words_23_1[6]), "r"(words_23_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 160), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_14[16];
                                #pragma unroll
                                for (int j_28 = 0; j_28 < 16; j_28++) {
                                    block_14[j_28] = packed_8[96 + j_28];
                                }
                                #pragma unroll
                                for (int j_29 = 0; j_29 < 4; j_29++) {
                                    unsigned int address_14_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_29 * 16);
                                    address_14_1 = address_14_1 ^ (address_14_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_14_1 - smem_v52_addr)), "r"(packed_8[96 + 4 * j_29]), "r"(packed_8[96 + 4 * j_29 + 1]), "r"(packed_8[96 + 4 * j_29 + 2]), "r"(packed_8[96 + 4 * j_29 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 6, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_24_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_24_1) : "r"(block_14[0]));
                                unsigned int amax_pair_24_1 = bf16x2_abs_24_1;
                                #pragma unroll
                                for (int i_102 = 1; i_102 < 16; i_102++) {
                                    uint32_t bf16x2_abs_1_50;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_50) : "r"(block_14[i_102]));
                                    uint32_t bf16x2_max_50;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_50) : "r"(amax_pair_24_1), "r"(bf16x2_abs_1_50));
                                    amax_pair_24_1 = bf16x2_max_50;
                                }
                                uint16_t bf16_max_24_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_24_1) : "h"((uint16_t)(amax_pair_24_1 & 65535)), "h"((uint16_t)(amax_pair_24_1 >> 16)));
                                float amax_24_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_24_1) : "h"((uint16_t)(bf16_max_24_1)));
                                float scale_24_1 = fmaxf(amax_24_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_189;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_189) : "f"(scale_24_1), "f"(scale_24_1));
                                unsigned int scale_byte_24_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_189 & 255;
                                unsigned int inverse_lane_24_1 = 254 - scale_byte_24_1 << 7;
                                unsigned int inverse_24_1 = inverse_lane_24_1 | inverse_lane_24_1 << 16;
                                unsigned int words_24_1[8];
                                #pragma unroll
                                for (int i_103 = 0; i_103 < 8; i_103++) {
                                    uint32_t bf16x2_mul_51;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_51) : "r"(block_14[i_103 * 2]), "r"(inverse_24_1));
                                    uint16_t e4m3x2_bf16x2_50;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_50) : "r"(bf16x2_mul_51));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_51);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_50) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_50;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_50) : "r"(block_14[i_103 * 2 + 1]), "r"(inverse_24_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_190;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_190) : "r"(bf16x2_mul_1_50));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_50);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_190) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_24_1[i_103] = (unsigned int)e4m3x2_bf16x2_50 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_190 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_24_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_24_1[0]), "r"(words_24_1[1]), "r"(words_24_1[2]), "r"(words_24_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24_1[4]), "r"(words_24_1[5]), "r"(words_24_1[6]), "r"(words_24_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 192), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_15[16];
                                #pragma unroll
                                for (int j_30 = 0; j_30 < 16; j_30++) {
                                    block_15[j_30] = packed_8[112 + j_30];
                                }
                                #pragma unroll
                                for (int j_31 = 0; j_31 < 4; j_31++) {
                                    unsigned int address_15_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_31 * 16);
                                    address_15_1 = address_15_1 ^ (address_15_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v52_addr + (address_15_1 - smem_v52_addr)), "r"(packed_8[112 + 4 * j_31]), "r"(packed_8[112 + 4 * j_31 + 1]), "r"(packed_8[112 + 4 * j_31 + 2]), "r"(packed_8[112 + 4 * j_31 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_out_r), 0, x_6 * 256 + cta_rank_0 * 128, y_6 * 8 + 7, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_25_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_25_1) : "r"(block_15[0]));
                                unsigned int amax_pair_25_1 = bf16x2_abs_25_1;
                                #pragma unroll
                                for (int i_104 = 1; i_104 < 16; i_104++) {
                                    uint32_t bf16x2_abs_1_51;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_51) : "r"(block_15[i_104]));
                                    uint32_t bf16x2_max_51;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_51) : "r"(amax_pair_25_1), "r"(bf16x2_abs_1_51));
                                    amax_pair_25_1 = bf16x2_max_51;
                                }
                                uint16_t bf16_max_25_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_25_1) : "h"((uint16_t)(amax_pair_25_1 & 65535)), "h"((uint16_t)(amax_pair_25_1 >> 16)));
                                float amax_25_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_25_1) : "h"((uint16_t)(bf16_max_25_1)));
                                float scale_25_1 = fmaxf(amax_25_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_191;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_191) : "f"(scale_25_1), "f"(scale_25_1));
                                unsigned int scale_byte_25_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_191 & 255;
                                unsigned int inverse_lane_25_1 = 254 - scale_byte_25_1 << 7;
                                unsigned int inverse_25_1 = inverse_lane_25_1 | inverse_lane_25_1 << 16;
                                unsigned int words_25_1[8];
                                #pragma unroll
                                for (int i_105 = 0; i_105 < 8; i_105++) {
                                    uint32_t bf16x2_mul_52;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_52) : "r"(block_15[i_105 * 2]), "r"(inverse_25_1));
                                    uint16_t e4m3x2_bf16x2_51;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_51) : "r"(bf16x2_mul_52));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_52);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_51) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t bf16x2_mul_1_51;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_51) : "r"(block_15[i_105 * 2 + 1]), "r"(inverse_25_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_192;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_192) : "r"(bf16x2_mul_1_51));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_51);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_192) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_25_1[i_105] = (unsigned int)e4m3x2_bf16x2_51 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_192 << 16;
                                }
                                scale_word_13 = scale_word_13 | scale_byte_25_1 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32)), "r"(words_25_1[0]), "r"(words_25_1[1]), "r"(words_25_1[2]), "r"(words_25_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v53_addr + (unsigned int)(tid * 32 + 16)), "r"(words_25_1[4]), "r"(words_25_1[5]), "r"(words_25_1[6]), "r"(words_25_1[7]) : "memory");
                                smem_v55[tid % 32 * 4 + tid / 32] = scale_word_13;
                                scale_word_13 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_6 * 256 + 224), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(smem_v53_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_6 * 2 + cta_rank_0) * i_tiles + y_6 * 2 + 1, smem_v55_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
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
                        gemm_wide += iterations_6;
                        scales_wide += iterations_6;
                        gemm_out0 += 1;
                        gemm_acc_tiles += 1;
                    }
                } else if (tile_word_0 == 8) {
                    int col_blocks_10 = intermediate / 128;
                    int num_tiles_1 = tokens / 128 * col_blocks_10;
                    int macro_row_offset_1 = tile_word_2 * (macro_size / 128);
                    int first_tile_2 = tile_word_1 * 6 + cta_rank_0 * 3;
                    int global_mini_7 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                    int mini_tiles = mini_size / 128 * col_blocks_10;
                    first_tile_2 = first_tile_2 + global_mini_7 * mini_tiles;
                    int tile_end_1 = ((num_tiles_1) < ((global_mini_7 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_7 + 1) * mini_tiles));
                    int macro_tiles_3 = macro_size / 128;
                    if (first_tile_2 < tile_end_1) {
                        int first_row_3 = first_tile_2 / col_blocks_10;
                        int first_col_1 = first_tile_2 % col_blocks_10;
                        if (tid == 0) {
                            if (tile_end_1 > first_tile_2) {
                                int row_13_1 = first_row_3;
                                int col_1_1 = first_col_1;
                                if (col_1_1 >= col_blocks_10) {
                                    row_13_1 = row_13_1 + 1;
                                    col_1_1 = col_1_1 - col_blocks_10;
                                }
                                mbarrier_arrive_expect_tx(replay_arrived_addr + ((int)((unsigned int)replay_seq % 3)) * 8, 65536);
                                int parent_2 = row_13_1 / 2 * (intermediate / 256) + col_1_1 / 2;
                                int poll_13_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_13_first) : "l"(replay_gu + parent_2) : "memory");
                                int poll_13 = poll_13_first;
                                while (poll_13 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_13_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_13_next) : "l"(replay_gu + parent_2) : "memory");
                                    poll_13 = poll_13_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_gate_addr + (unsigned int)(replay_seq % 3 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_13_1 - macro_row_offset_1) * 128), "r"(col_1_1 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)replay_seq % 3)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_up_addr + (unsigned int)(replay_seq % 3 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_13_1 - macro_row_offset_1) * 128), "r"(col_1_1 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)replay_seq % 3)) * 8) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 1) {
                                int row_14_1 = first_row_3;
                                int col_2_1 = first_col_1 + 1;
                                if (col_2_1 >= col_blocks_10) {
                                    row_14_1 = row_14_1 + 1;
                                    col_2_1 = col_2_1 - col_blocks_10;
                                }
                                mbarrier_arrive_expect_tx(replay_arrived_addr + ((int)((unsigned int)(replay_seq + 1) % 3)) * 8, 65536);
                                int parent_3 = row_14_1 / 2 * (intermediate / 256) + col_2_1 / 2;
                                int poll_14_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_14_first) : "l"(replay_gu + parent_3) : "memory");
                                int poll_14 = poll_14_first;
                                while (poll_14 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_14_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_14_next) : "l"(replay_gu + parent_3) : "memory");
                                    poll_14 = poll_14_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_gate_addr + (unsigned int)((replay_seq + 1) % 3 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_14_1 - macro_row_offset_1) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)(replay_seq + 1) % 3)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_up_addr + (unsigned int)((replay_seq + 1) % 3 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_14_1 - macro_row_offset_1) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)(replay_seq + 1) % 3)) * 8) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 2) {
                                int row_15_1 = first_row_3;
                                int col_3_1 = first_col_1 + 2;
                                if (col_3_1 >= col_blocks_10) {
                                    row_15_1 = row_15_1 + 1;
                                    col_3_1 = col_3_1 - col_blocks_10;
                                }
                                mbarrier_arrive_expect_tx(replay_arrived_addr + ((int)((unsigned int)(replay_seq + 2) % 3)) * 8, 65536);
                                int parent_4 = row_15_1 / 2 * (intermediate / 256) + col_3_1 / 2;
                                int poll_15_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_15_first) : "l"(replay_gu + parent_4) : "memory");
                                int poll_15 = poll_15_first;
                                while (poll_15 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_15_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_15_next) : "l"(replay_gu + parent_4) : "memory");
                                    poll_15 = poll_15_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_gate_addr + (unsigned int)((replay_seq + 2) % 3 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_15_1 - macro_row_offset_1) * 128), "r"(col_3_1 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)(replay_seq + 2) % 3)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(replay_up_addr + (unsigned int)((replay_seq + 2) % 3 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_15_1 - macro_row_offset_1) * 128), "r"(col_3_1 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + ((int)((unsigned int)(replay_seq + 2) % 3)) * 8) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int stage_3 = 0; stage_3 < 3; stage_3++) {
                            if (tile_end_1 > first_tile_2 + stage_3) {
                                int _stage_replay_arrived = (int)((unsigned int)(replay_seq + stage_3) % 3);
                                int _parity_replay_arrived = (int)((unsigned int)(replay_seq + stage_3) / 3 % 2);
                                mbarrier_wait(replay_arrived_addr + (_stage_replay_arrived) * 8, _parity_replay_arrived);
                                int slot = (replay_seq + stage_3) % 3;
                                int row_25 = first_row_3;
                                int col_68 = first_col_1 + stage_3;
                                if (col_68 >= col_blocks_10) {
                                    row_25 = row_25 + 1;
                                    col_68 = col_68 - col_blocks_10;
                                }
                                float gate_1[64];
                                float up_1[64];
                                float denominator[64];
                                int warp_4 = tid / 32;
                                int local_warp_4 = warp_4 / 4 + warp_4 % 4 * 2;
                                int lane_6 = tid % 32;
                                #pragma unroll
                                for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                    unsigned int packed_9[4];
                                    unsigned int address_19 = replay_gate_addr + (unsigned int)(slot * 32768) + (unsigned int)(((tile_col_5 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_4 * 16 + lane_6 % 16) * 64 + (tile_col_5 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_9[0]), "=r"(packed_9[1]), "=r"(packed_9[2]), "=r"(packed_9[3])
                                        : "r"(address_19 ^ (address_19 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_9[pair_9]));
                                        gate_1[tile_col_5 * 8 + pair_9 * 2] = cvt_f32_20.x;
                                        gate_1[tile_col_5 * 8 + pair_9 * 2 + 1] = cvt_f32_20.y;
                                    }
                                }
                                int warp_4_1 = tid / 32;
                                int local_warp_4_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                                int lane_4_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_6 = 0; tile_col_6 < 8; tile_col_6++) {
                                    unsigned int packed_10[4];
                                    unsigned int address_20 = replay_up_addr + (unsigned int)(slot * 32768) + (unsigned int)(((tile_col_6 * 16 + lane_4_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_4_1 * 16 + lane_4_1 % 16) * 64 + (tile_col_6 * 16 + lane_4_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_10[0]), "=r"(packed_10[1]), "=r"(packed_10[2]), "=r"(packed_10[3])
                                        : "r"(address_20 ^ (address_20 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_10[pair_10]));
                                        up_1[tile_col_6 * 8 + pair_10 * 2] = cvt_f32_21.x;
                                        up_1[tile_col_6 * 8 + pair_10 * 2 + 1] = cvt_f32_21.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem_11 = 0; elem_11 < 64; elem_11++) {
                                    denominator[elem_11] = gate_1[elem_11] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_12 = 0; elem_12 < 64; elem_12++) {
                                    float _exp_193 = expf(denominator[elem_12]);
                                    denominator[elem_12] = _exp_193;
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
                                int warp_4_0 = tid / 32;
                                int local_warp_5 = warp_4_0 / 4 + warp_4_0 % 4 * 2;
                                int lane_6_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_7 = 0; tile_col_7 < 8; tile_col_7++) {
                                    unsigned int packed_11[4];
                                    #pragma unroll
                                    for (int pair_11 = 0; pair_11 < 4; pair_11++) {
                                        packed_11[pair_11] = __as_u32(__float22bfloat162_rn(make_float2(gate_1[tile_col_7 * 8 + pair_11 * 2], gate_1[tile_col_7 * 8 + pair_11 * 2 + 1])));
                                    }
                                    int row_0_1 = local_warp_5 * 16 + lane_6_1 % 16;
                                    int col_1_2 = tile_col_7 * 16 + lane_6_1 / 16 * 8;
                                    uint32_t _stmatrix_addr_194 = static_cast<uint32_t>(flat_view_addr_4 + (unsigned int)((row_0_1 * 136 + col_1_2) * 2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_194), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                int out_base = slot * 8192;
                                if (tid < 128) {
                                    int r_12 = tid % 64;
                                    int row0_12 = r_12 * 2;
                                    int row1_12 = row0_12 + 1;
                                    int half_16 = tid / 64;
                                    int word_stride_12 = 68;
                                    int scale_base0_12 = (row0_12 % 32 * 4 + row0_12 / 32) * 4;
                                    int scale_base1_12 = (row1_12 % 32 * 4 + row1_12 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_32 = 0; j_32 < 2; j_32++) {
                                        int k_block_24 = (half_16 * 2 + j_32 + tid / 8) % 4;
                                        unsigned int pairs0_12[16];
                                        unsigned int pairs1_12[16];
                                        #pragma unroll
                                        for (int k_48 = 0; k_48 < 16; k_48++) {
                                            int col_4_1 = k_block_24 * 32 + (tid * 4 + k_48 * 2) % 32;
                                            unsigned int w0_12 = flat_view_5[col_4_1 * word_stride_12 + r_12];
                                            unsigned int w1_12 = flat_view_5[(col_4_1 + 1) * word_stride_12 + r_12];
                                            uint32_t lo_12;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_12) : "r"(w0_12), "r"(w1_12));
                                            uint32_t hi_12;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_12) : "r"(w0_12), "r"(w1_12));
                                            pairs0_12[k_48] = lo_12;
                                            pairs1_12[k_48] = hi_12;
                                        }
                                        uint32_t bf16x2_abs_27;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_27) : "r"(pairs0_12[0]));
                                        unsigned int amax_pair_26 = bf16x2_abs_27;
                                        #pragma unroll
                                        for (int i_106 = 1; i_106 < 16; i_106++) {
                                            uint32_t bf16x2_abs_1_52;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_52) : "r"(pairs0_12[i_106]));
                                            uint32_t bf16x2_max_52;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_52) : "r"(amax_pair_26), "r"(bf16x2_abs_1_52));
                                            amax_pair_26 = bf16x2_max_52;
                                        }
                                        uint16_t bf16_max_26;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_26) : "h"((uint16_t)(amax_pair_26 & 65535)), "h"((uint16_t)(amax_pair_26 >> 16)));
                                        float amax_26;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_26) : "h"((uint16_t)(bf16_max_26)));
                                        float scale_26 = fmaxf(amax_26 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_195;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_195) : "f"(scale_26), "f"(scale_26));
                                        unsigned int scale_byte_26 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_195 & 255;
                                        unsigned int inverse_lane_26 = 254 - scale_byte_26 << 7;
                                        unsigned int inverse_26 = inverse_lane_26 | inverse_lane_26 << 16;
                                        unsigned int words_26[8];
                                        #pragma unroll
                                        for (int i_107 = 0; i_107 < 8; i_107++) {
                                            uint32_t bf16x2_mul_53;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_53) : "r"(pairs0_12[i_107 * 2]), "r"(inverse_26));
                                            uint16_t e4m3x2_bf16x2_52;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_52) : "r"(bf16x2_mul_53));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_53);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_52) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t bf16x2_mul_1_52;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_52) : "r"(pairs0_12[i_107 * 2 + 1]), "r"(inverse_26));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_196;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_196) : "r"(bf16x2_mul_1_52));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_52);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_196) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_26[i_107] = (unsigned int)e4m3x2_bf16x2_52 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_196 << 16;
                                        }
                                        uint32_t bf16x2_abs_26_1;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_26_1) : "r"(pairs1_12[0]));
                                        unsigned int amax_pair_26_1 = bf16x2_abs_26_1;
                                        #pragma unroll
                                        for (int i_108 = 1; i_108 < 16; i_108++) {
                                            uint32_t bf16x2_abs_1_53;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_53) : "r"(pairs1_12[i_108]));
                                            uint32_t bf16x2_max_53;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_53) : "r"(amax_pair_26_1), "r"(bf16x2_abs_1_53));
                                            amax_pair_26_1 = bf16x2_max_53;
                                        }
                                        uint16_t bf16_max_26_1;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_26_1) : "h"((uint16_t)(amax_pair_26_1 & 65535)), "h"((uint16_t)(amax_pair_26_1 >> 16)));
                                        float amax_26_1;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_26_1) : "h"((uint16_t)(bf16_max_26_1)));
                                        float scale_26_1 = fmaxf(amax_26_1 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_197;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_197) : "f"(scale_26_1), "f"(scale_26_1));
                                        unsigned int scale_byte_26_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_197 & 255;
                                        unsigned int inverse_lane_26_1 = 254 - scale_byte_26_1 << 7;
                                        unsigned int inverse_26_1 = inverse_lane_26_1 | inverse_lane_26_1 << 16;
                                        unsigned int words_26_1[8];
                                        #pragma unroll
                                        for (int i_109 = 0; i_109 < 8; i_109++) {
                                            uint32_t bf16x2_mul_54;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_54) : "r"(pairs1_12[i_109 * 2]), "r"(inverse_26_1));
                                            uint16_t e4m3x2_bf16x2_53;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_53) : "r"(bf16x2_mul_54));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_54);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_53) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t bf16x2_mul_1_53;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_53) : "r"(pairs1_12[i_109 * 2 + 1]), "r"(inverse_26_1));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_198;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_198) : "r"(bf16x2_mul_1_53));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_53);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_198) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_26_1[i_109] = (unsigned int)e4m3x2_bf16x2_53 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_198 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_49 = 0; k_49 < 8; k_49++) {
                                            int col_5_1 = k_block_24 * 32 + (tid * 4 + k_49 * 4) % 32;
                                            smem_v19_1[out_base + (row0_12 * 128 + col_5_1) / 4] = words_26[k_49];
                                            smem_v19_1[out_base + (row1_12 * 128 + col_5_1) / 4] = words_26_1[k_49];
                                        }
                                        {
                                            uint32_t _byte_199 = static_cast<uint32_t>(scale_byte_26) & 0xFFu;
                                            uint32_t _addr_199 = static_cast<uint32_t>(smem_v19_addr_2 + (unsigned int)(out_base * 4 + scale_base0_12 + k_block_24));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_199), "r"(_byte_199) : "memory");
                                        }
                                        {
                                            uint32_t _byte_200 = static_cast<uint32_t>(scale_byte_26_1) & 0xFFu;
                                            uint32_t _addr_200 = static_cast<uint32_t>(smem_v19_addr_2 + (unsigned int)(out_base * 4 + scale_base1_12 + k_block_24));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_200), "r"(_byte_200) : "memory");
                                        }
                                    }
                                } else {
                                    int row_16_1 = tid - 128;
                                    unsigned int scale_word_14 = 0;
                                    #pragma unroll 1
                                    for (int j_33 = 0; j_33 < 4; j_33++) {
                                        int k_block_25 = (j_33 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_12[16];
                                        #pragma unroll
                                        for (int k_50 = 0; k_50 < 16; k_50++) {
                                            int col_6_1 = k_block_25 * 32 + ((tid - 128) * 4 + k_50 * 2) % 32;
                                            float x0_12 = 0.0f;
                                            float x1_12 = 0.0f;
                                            pairs_12[k_50] = flat_view_5[(row_16_1 * 136 + col_6_1) / 2];
                                        }
                                        uint32_t bf16x2_abs_28;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_28) : "r"(pairs_12[0]));
                                        unsigned int amax_pair_27 = bf16x2_abs_28;
                                        #pragma unroll
                                        for (int i_110 = 1; i_110 < 16; i_110++) {
                                            uint32_t bf16x2_abs_1_54;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_54) : "r"(pairs_12[i_110]));
                                            uint32_t bf16x2_max_54;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_54) : "r"(amax_pair_27), "r"(bf16x2_abs_1_54));
                                            amax_pair_27 = bf16x2_max_54;
                                        }
                                        uint16_t bf16_max_27;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_27) : "h"((uint16_t)(amax_pair_27 & 65535)), "h"((uint16_t)(amax_pair_27 >> 16)));
                                        float amax_27;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_27) : "h"((uint16_t)(bf16_max_27)));
                                        float scale_27 = fmaxf(amax_27 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_201;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_201) : "f"(scale_27), "f"(scale_27));
                                        unsigned int scale_byte_27 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_201 & 255;
                                        unsigned int inverse_lane_27 = 254 - scale_byte_27 << 7;
                                        unsigned int inverse_27 = inverse_lane_27 | inverse_lane_27 << 16;
                                        unsigned int words_27[8];
                                        #pragma unroll
                                        for (int i_111 = 0; i_111 < 8; i_111++) {
                                            uint32_t bf16x2_mul_55;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_55) : "r"(pairs_12[i_111 * 2]), "r"(inverse_27));
                                            uint16_t e4m3x2_bf16x2_54;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_54) : "r"(bf16x2_mul_55));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_55);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_54) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t bf16x2_mul_1_54;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_54) : "r"(pairs_12[i_111 * 2 + 1]), "r"(inverse_27));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_202;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_202) : "r"(bf16x2_mul_1_54));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_54);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_202) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_27[i_111] = (unsigned int)e4m3x2_bf16x2_54 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_202 << 16;
                                        }
                                        scale_word_14 = scale_word_14 | scale_byte_27 << (unsigned int)(k_block_25 * 8);
                                        #pragma unroll
                                        for (int k_51 = 0; k_51 < 8; k_51++) {
                                            int col_7_1 = k_block_25 * 32 + ((tid - 128) * 4 + k_51 * 4) % 32;
                                            smem_v19[out_base + (row_16_1 * 128 + col_7_1) / 4] = words_27[k_51];
                                        }
                                    }
                                    smem_v19_0[out_base + row_16_1 % 32 * 4 + row_16_1 / 32] = scale_word_14;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_3 = row_25 - macro_row_offset_1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&h_q_store), col_68 * 128, local_row_3 * 128, smem_v19_addr + (unsigned int)(slot * 32768));
                                    tma_store_3d((&h_sc_store), 0, 0, local_row_3 * col_blocks_10 + col_68, smem_v19_addr_0 + (unsigned int)(slot * 32768));
                                    tma_store_2d((&h_t_store), local_row_3 * 128, col_68 * 128, smem_v19_addr_1 + (unsigned int)(slot * 32768));
                                    tma_store_3d((&h_sc_t_store), 0, 0, col_68 * macro_tiles_3 + local_row_3, smem_v19_addr_2 + (unsigned int)(slot * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            if (tile_end_1 > first_tile_2) {
                                int row_17_1 = first_row_3;
                                if (col_blocks_10 <= first_col_1) {
                                    row_17_1 = row_17_1 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_17_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 1) {
                                int row_18_1 = first_row_3;
                                if (col_blocks_10 <= first_col_1 + 1) {
                                    row_18_1 = row_18_1 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_18_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end_1 > first_tile_2 + 2) {
                                int row_19_1 = first_row_3;
                                if (col_blocks_10 <= first_col_1 + 2) {
                                    row_19_1 = row_19_1 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_19_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    int _min_204 = ((3) < (tile_end_1 - first_tile_2) ? (3) : (tile_end_1 - first_tile_2));
                    int _max_203 = ((0) > (_min_204) ? (0) : (_min_204));
                    replay_seq += _max_203;
                } else {
                    if (tile_word_0 == 0) {
                        int col_blocks_11 = (intermediate + 512 - 1) / 512;
                        int x_7 = -1;
                        int y_7 = -1;
                        int expert_7 = -1;
                        int k_start_7 = 0;
                        int k_end_7 = 0;
                        int first_7 = 0;
                        int first_block_5 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                        int end_block_5 = ((first_block_5 + mini_size / 256) < (tokens / 256) ? (first_block_5 + mini_size / 256) : (tokens / 256));
                        int block_16 = first_block_5;
                        uint32_t fast_div_q_6 = (dh_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), dh_col_blocks_div.multiplier) >> dh_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                        uint32_t fast_div_r_6 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_6 * (uint32_t)(dh_col_blocks_div.divisor);
                        block_16 = first_block_5 + (int)fast_div_q_6;
                        if (block_16 < end_block_5) {
                            int index_4 = tile_word_5;
                            int offset_8 = tile_word_6 / 256;
                            int first_row_4 = ((first_block_5) > (offset_8) ? (first_block_5) : (offset_8));
                            int _min_205 = ((end_block_5) < (offset_8 + tile_word_7 / 256) ? (end_block_5) : (offset_8 + tile_word_7 / 256));
                            int rows_5 = _min_205 - first_row_4;
                            int supergroup_7 = (tile_word_1 - (first_row_4 - first_block_5) * col_blocks_11) / (rows_5 * 8);
                            int full_cols_7 = col_blocks_11 / 8 * 8;
                            int row_26 = 0;
                            int col_69 = 0;
                            if (tile_word_1 - (first_row_4 - first_block_5) * col_blocks_11 < rows_5 * full_cols_7) {
                                row_26 = (tile_word_1 - (first_row_4 - first_block_5) * col_blocks_11) % (rows_5 * 8) / 8;
                                col_69 = supergroup_7 * 8 + (tile_word_1 - (first_row_4 - first_block_5) * col_blocks_11) % 8;
                            } else {
                                row_26 = (tile_word_1 - (first_row_4 - first_block_5) * col_blocks_11 - rows_5 * full_cols_7) / (col_blocks_11 - full_cols_7);
                                col_69 = full_cols_7 + (tile_word_1 - (first_row_4 - first_block_5) * col_blocks_11 - rows_5 * full_cols_7) % (col_blocks_11 - full_cols_7);
                            }
                            if ((supergroup_7 & 1) != 0) {
                                row_26 = rows_5 - row_26 - 1;
                            }
                            x_7 = first_row_4 + row_26 - tile_word_2 * (macro_size / 256);
                            y_7 = col_69;
                            expert_7 = index_4;
                        }
                        int has_hi_7 = 0;
                        has_hi_7 = (int)((y_7 * 2 + 1) * 256 < intermediate);
                        int global_mini_8 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                        int macro_rows_7 = tile_word_2 * (macro_size / 256);
                        int iterations_7 = hidden / 128;
                        int macro_k_2 = tile_word_2 * (macro_size / 128);
                        if (expert_7 < 0) {
                            if (tid == 0) {
                                bool enabled_value_12 = macros > 1;
                                if (enabled_value_12 != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        } else {
                            if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            int _min_206 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                            int mini_rows_7 = ((0) > (_min_206) ? (0) : (_min_206));
                                            int required_8 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_13 = 1;
                                            if (enabled_value_13 != 0) {
                                                int poll_16_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_16_first) : "l"(dy_ready + global_mini_8) : "memory");
                                                int poll_16 = poll_16_first;
                                                while (poll_16 < required_8) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int poll_16_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_16_next) : "l"(dy_ready + global_mini_8) : "memory");
                                                    poll_16 = poll_16_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int idx_16 = 0; idx_16 < iterations_7; idx_16++) {
                                            int _stage_gemm_finished_7 = (int)((unsigned int)(gemm_wide + idx_16) % 4);
                                            int _parity_gemm_finished_7 = (int)((unsigned int)(gemm_wide + idx_16) / 4 % 2) ^ 1;
                                            mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_7) * 8, _parity_gemm_finished_7);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_16) % 4 * 16384)), "l"((&dy_q)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(idx_16), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_16) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_16) % 4 * 16384)), "l"((&wd_t_q)), "r"(0), "r"(y_7 * 512 + cta_rank_0 * 128), "r"(idx_16), "r"(expert_7), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_16) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v44_addr + (unsigned int)((gemm_wide + idx_16) % 4 * 16384)), "l"((&wd_t_q)), "r"(0), "r"(y_7 * 512 + 256 + cta_rank_0 * 128), "r"(idx_16), "r"(expert_7), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_16) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        }
                                    }
                                }
                            } else if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_207 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                            int mini_rows_8 = ((0) > (_min_207) ? (0) : (_min_207));
                                            int required_9 = (mini_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_14 = 1;
                                            if (enabled_value_14 != 0) {
                                                int poll_17_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_17_first) : "l"(dy_ready + global_mini_8) : "memory");
                                                int poll_17 = poll_17_first;
                                                while (poll_17 < required_9) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int poll_17_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_17_next) : "l"(dy_ready + global_mini_8) : "memory");
                                                    poll_17 = poll_17_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int idx_17 = 0; idx_17 < iterations_7; idx_17++) {
                                            int _stage_scales_finished_2 = (int)((unsigned int)(scales_wide + idx_17) % 4);
                                            int _parity_scales_finished_2 = (int)((unsigned int)(scales_wide + idx_17) / 4 % 2) ^ 1;
                                            mbarrier_wait(scales_finished_addr + (_stage_scales_finished_2) * 8, _parity_scales_finished_2);
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_17) % 4 * 512)), "l"((&dy_sc)), "r"(0), "r"(0), "r"((x_7 * 2 + cta_rank_0) * (hidden / 128) + idx_17),
                                                   "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_17) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_17) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_t_sc)), "r"(0), "r"(0), "r"((expert_7 * i_tiles_1 + y_7 * 4 + cta_rank_0) * (hidden / 128) + idx_17),
                                                   "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_17) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v47_addr + (unsigned int)((scales_wide + idx_17) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_t_sc)), "r"(0), "r"(0), "r"((expert_7 * i_tiles_1 + y_7 * 4 + 2 + cta_rank_0) * (hidden / 128) + idx_17),
                                                   "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_17) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int _parity_output_finished_7 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                            mbarrier_wait(output_finished_addr, _parity_output_finished_7);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_18 = 0; idx_18 < iterations_7; idx_18++) {
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_18) % 4)) * 8, 5120);
                                                int _stage_scales_arrived_2 = (int)((unsigned int)(scales_wide + idx_18) % 4);
                                                int _parity_scales_arrived_2 = (int)((unsigned int)(scales_wide + idx_18) / 4 % 2);
                                                mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_2) * 8, _parity_scales_arrived_2);
                                                int buffer_2 = idx_18 % 3;
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_2 * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_18) % 4 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_2 * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_18) % 4 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_2 * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_18) % 4 * 1024) + 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_2 * 8, make_sf_cp_desc_sbo128(smem_v47_addr + (unsigned int)((scales_wide + idx_18) % 4 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_2 * 8 + 4), make_sf_cp_desc_sbo128((smem_v47_addr + (unsigned int)((scales_wide + idx_18) % 4 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_18) % 4)) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8, 98304);
                                                int _stage_gemm_arrived_7 = (int)((unsigned int)(gemm_wide + idx_18) % 4);
                                                int _parity_gemm_arrived_7 = (int)((unsigned int)(gemm_wide + idx_18) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_7) * 8, _parity_gemm_arrived_7);
                                                int _smem_desc_lo_208 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_18) % 4) * 1024;
                                                int _smem_desc_lo_209 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_18) % 4) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)_smem_desc_lo_208) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)_smem_desc_lo_209) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        0x90c00000U, (int)((tmem_tmem_sfa + buffer_2 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_2 * 8 + 0)), ((idx_18 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        0xd0c00020U, (int)((tmem_tmem_sfa + buffer_2 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_2 * 8 + 0) | 0x80000000), 1);
                                                }
                                                int _smem_desc_lo_210 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_18) % 4) * 1024;
                                                int _smem_desc_lo_211 = (((smem_v44_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_18) % 4) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)_smem_desc_lo_210) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)_smem_desc_lo_211) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                        0x90c00000U, (int)((tmem_tmem_sfa + buffer_2 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_2 * 8 + 0)), ((idx_18 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                        0xd0c00020U, (int)((tmem_tmem_sfa + buffer_2 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_2 * 8 + 0) | 0x80000000), 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8, (uint16_t)(3));
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    int _parity_output_arrived_7 = (int)((unsigned int)gemm_out0 % 2);
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_7);
                                    unsigned int packed_12[128];
                                    #pragma unroll
                                    for (int chunk_14 = 0; chunk_14 < 8; chunk_14++) {
                                        #pragma unroll
                                        for (int sub_4 = 0; sub_4 < 2; sub_4++) {
                                            unsigned int address_21 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_4 * 16 << 16) + (unsigned int)(chunk_14 * 32);
                                            float _tmem_load_12[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                                                : "r"(address_21));
                                            #pragma unroll
                                            for (int pair_12 = 0; pair_12 < 8; pair_12++) {
                                                packed_12[chunk_14 * 16 + sub_4 * 8 + pair_12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[pair_12 * 2], _tmem_load_12[pair_12 * 2 + 1])));
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_3 = 1;
                                    last_3 = 1 - has_hi_7;
                                    if (last_3 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    if (tid == 0) {
                                        int previous_offset_2 = (tile_word_2 + 1) * macro_size;
                                        int output_row_2 = x_7 * 256 + cta_rank_0 * 128;
                                        int _min_212 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                        if (output_row_2 < _min_212) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_15 = 0; chunk_15 < 8; chunk_15++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_5 = tid / 32;
                                        int lane_7 = tid % 32;
                                        #pragma unroll
                                        for (int half_17 = 0; half_17 < 2; half_17++) {
                                            #pragma unroll
                                            for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                                int row_27 = warp_0_5 * 32 + half_17 * 16 + lane_7 % 16;
                                                int col_70 = col_tile_15 * 16 + lane_7 / 16 * 8;
                                                unsigned int address_22 = d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192) + (unsigned int)((row_27 * 32 + col_70) * 2);
                                                address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                                int offset_9 = chunk_15 * 16 + half_17 * 8 + col_tile_15 * 4;
                                                uint32_t _stmatrix_addr_213 = static_cast<uint32_t>(address_22);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_213), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_9 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_5d((&dh_r), 0, x_7 * 256 + cta_rank_0 * 128, y_7 * 2 * 8 + chunk_15, 0, 0, d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192));
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (has_hi_7 != 0) {
                                        unsigned int packed_0_2[128];
                                        #pragma unroll
                                        for (int chunk_16 = 0; chunk_16 < 8; chunk_16++) {
                                            #pragma unroll
                                            for (int sub_5 = 0; sub_5 < 2; sub_5++) {
                                                unsigned int address_23 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_5 * 16 << 16) + 256 + (unsigned int)(chunk_16 * 32);
                                                float _tmem_load_13[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                                                    : "r"(address_23));
                                                #pragma unroll
                                                for (int pair_13 = 0; pair_13 < 8; pair_13++) {
                                                    packed_0_2[chunk_16 * 16 + sub_5 * 8 + pair_13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[pair_13 * 2], _tmem_load_13[pair_13 * 2 + 1])));
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
                                            int warp_0_6 = tid / 32;
                                            int lane_8 = tid % 32;
                                            #pragma unroll
                                            for (int half_18 = 0; half_18 < 2; half_18++) {
                                                #pragma unroll
                                                for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                                    int row_28 = warp_0_6 * 32 + half_18 * 16 + lane_8 % 16;
                                                    int col_71 = col_tile_16 * 16 + lane_8 / 16 * 8;
                                                    unsigned int address_24 = d_smem_addr + (unsigned int)((8 + chunk_17) % 3 * 8192) + (unsigned int)((row_28 * 32 + col_71) * 2);
                                                    address_24 = address_24 ^ (address_24 & 511) >> 7 << 4;
                                                    int offset_10 = chunk_17 * 16 + half_18 * 8 + col_tile_16 * 4;
                                                    uint32_t _stmatrix_addr_214 = static_cast<uint32_t>(address_24);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_214), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_10 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                tma_store_5d((&dh_r), 0, x_7 * 256 + cta_rank_0 * 128, (y_7 * 2 + 1) * 8 + chunk_17, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_17) % 3 * 8192));
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
                                                bool enabled_value_15 = 1;
                                                if (enabled_value_15 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_0 + (macro_rows_7 + x_7) * (intermediate / 256) + y_7 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                                bool enabled_value_1_1 = macros > 1;
                                                if (enabled_value_1_1 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_2_1 = 1;
                                                    if (enabled_value_2_1 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                                if (has_hi_7 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_3_1 = 1;
                                                    if (enabled_value_3_1 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_0 + (macro_rows_7 + x_7) * (intermediate / 256) + y_7 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_wide += iterations_7;
                            scales_wide += iterations_7;
                            gemm_out0 += 1;
                            gemm_acc_tiles += 1;
                        }
                    } else if (tile_word_0 == 1) {
                        int col_blocks_12 = intermediate / 128;
                        int num_tiles_2 = tokens / 128 * col_blocks_12;
                        int macro_row_offset_2 = tile_word_2 * (macro_size / 128);
                        int macro_tiles_4 = macro_size / 128;
                        int first_tile_3 = tile_word_1 * 4 + cta_rank_0 * 2;
                        int global_mini_9 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                        int mini_tiles_1 = mini_size / 128 * col_blocks_12;
                        first_tile_3 = first_tile_3 + global_mini_9 * mini_tiles_1;
                        int tile_end_2 = ((num_tiles_2) < ((global_mini_9 + 1) * mini_tiles_1) ? (num_tiles_2) : ((global_mini_9 + 1) * mini_tiles_1));
                        if (first_tile_3 < tile_end_2) {
                            int first_row_5 = first_tile_3 / col_blocks_12;
                            int first_col_2 = first_tile_3 % col_blocks_12;
                            if (tid == 0) {
                                if (tile_end_2 > first_tile_3) {
                                    int row_29 = first_row_5;
                                    int col_72 = first_col_2;
                                    if (col_72 >= col_blocks_12) {
                                        row_29 = row_29 + 1;
                                        col_72 = col_72 - col_blocks_12;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 2)) * 8, 66560);
                                    int parent_5 = row_29 / 2 * (intermediate / 256) + col_72 / 2;
                                    int poll_18_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_18_first) : "l"(dh_ready + (shared_down_0 + parent_5)) : "memory");
                                    int poll_18 = poll_18_first;
                                    while (poll_18 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_18_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_18_next) : "l"(dh_ready + (shared_down_0 + parent_5)) : "memory");
                                        poll_18 = poll_18_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    bool enabled_value_16 = tile_word_2 > 0;
                                    if (enabled_value_16 != 0) {
                                        int poll_0_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_0_first) : "l"(replay_gu + parent_5) : "memory");
                                        int poll_0 = poll_0_first;
                                        while (poll_0 < 4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_0_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_0_next) : "l"(replay_gu + parent_5) : "memory");
                                            poll_0 = poll_0_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int local_row_4 = row_29 - macro_row_offset_2;
                                    tma_2d_gmem2smem(smem_v56_addr + (unsigned int)(swiglu_seq % 2 * 32768), (&dh_tile), col_72 * 128, local_row_4 * 128, swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 2)) * 8);
                                    tma_2d_gmem2smem(smem_v57_addr_0 + (unsigned int)(swiglu_seq % 2 * 16384), (&gate_tile), col_72 * 128, local_row_4 * 128, swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 2)) * 8);
                                    tma_2d_gmem2smem(smem_v57_addr_1 + (unsigned int)(swiglu_seq % 2 * 16384), (&up_tile), col_72 * 128, local_row_4 * 128, swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 2)) * 8);
                                    tma_3d_gmem2smem(smem_v57_addr_2 + (unsigned int)(swiglu_seq % 2 * 512), (&gate_sc), 0, 0, local_row_4 * col_blocks_12 + col_72, swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 2)) * 8);
                                    tma_3d_gmem2smem(smem_v57_addr_3 + (unsigned int)(swiglu_seq % 2 * 512), (&up_sc), 0, 0, local_row_4 * col_blocks_12 + col_72, swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 2)) * 8);
                                }
                                if (tile_end_2 > first_tile_3 + 1) {
                                    int row_30 = first_row_5;
                                    int col_73 = first_col_2 + 1;
                                    if (col_73 >= col_blocks_12) {
                                        row_30 = row_30 + 1;
                                        col_73 = col_73 - col_blocks_12;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 2)) * 8, 66560);
                                    int parent_6 = row_30 / 2 * (intermediate / 256) + col_73 / 2;
                                    int poll_19_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_19_first) : "l"(dh_ready + (shared_down_0 + parent_6)) : "memory");
                                    int poll_19 = poll_19_first;
                                    while (poll_19 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_19_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_19_next) : "l"(dh_ready + (shared_down_0 + parent_6)) : "memory");
                                        poll_19 = poll_19_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    bool enabled_value_17 = tile_word_2 > 0;
                                    if (enabled_value_17 != 0) {
                                        int poll_1_1_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_1_first) : "l"(replay_gu + parent_6) : "memory");
                                        int poll_1_1 = poll_1_1_first;
                                        while (poll_1_1 < 4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_1_1_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_1_next) : "l"(replay_gu + parent_6) : "memory");
                                            poll_1_1 = poll_1_1_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int local_row_5 = row_30 - macro_row_offset_2;
                                    tma_2d_gmem2smem(smem_v56_addr + (unsigned int)((swiglu_seq + 1) % 2 * 32768), (&dh_tile), col_73 * 128, local_row_5 * 128, swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 2)) * 8);
                                    tma_2d_gmem2smem(smem_v57_addr_0 + (unsigned int)((swiglu_seq + 1) % 2 * 16384), (&gate_tile), col_73 * 128, local_row_5 * 128, swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 2)) * 8);
                                    tma_2d_gmem2smem(smem_v57_addr_1 + (unsigned int)((swiglu_seq + 1) % 2 * 16384), (&up_tile), col_73 * 128, local_row_5 * 128, swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 2)) * 8);
                                    tma_3d_gmem2smem(smem_v57_addr_2 + (unsigned int)((swiglu_seq + 1) % 2 * 512), (&gate_sc), 0, 0, local_row_5 * col_blocks_12 + col_73, swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 2)) * 8);
                                    tma_3d_gmem2smem(smem_v57_addr_3 + (unsigned int)((swiglu_seq + 1) % 2 * 512), (&up_sc), 0, 0, local_row_5 * col_blocks_12 + col_73, swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 2)) * 8);
                                }
                            }
                            int tile_row = tid % 128;
                            int half_19 = tid / 128;
                            int scale_index = tile_row % 32 * 4 + tile_row / 32;
                            int k_pair = tile_row >> 1 & 1 ^ half_19;
                            #pragma unroll 1
                            for (int stage_4 = 0; stage_4 < 2; stage_4++) {
                                if (tile_end_2 > first_tile_3 + stage_4) {
                                    int _stage_swiglu_arrived = (int)((unsigned int)(swiglu_seq + stage_4) % 2);
                                    int _parity_swiglu_arrived_1 = (int)((unsigned int)(swiglu_seq + stage_4) / 2 % 2);
                                    mbarrier_wait(swiglu_arrived_addr + (_stage_swiglu_arrived) * 8, _parity_swiglu_arrived_1);
                                    int slot_1 = (swiglu_seq + stage_4) % 2;
                                    int row_31 = first_row_5;
                                    int col_74 = first_col_2 + stage_4;
                                    if (col_74 >= col_blocks_12) {
                                        row_31 = row_31 + 1;
                                        col_74 = col_74 - col_blocks_12;
                                    }
                                    int tile_base = slot_1 * 4096;
                                    int dh_base = slot_1 * 8192;
                                    int sc_base = slot_1 * 256;
                                    int local_token = (row_31 - macro_row_offset_2) * 128 + tile_row;
                                    int peer_11 = schedule_rank[row_31 * 128 + tile_row];
                                    float weight = weights[local_token];
                                    float inverse_28 = ((weight > 0.0f) ? 1.0f / weight : 0.0f);
                                    float router_gradient = 0.0f;
                                    unsigned int gate_scales = (unsigned int)smem_v62[sc_base + scale_index * 2 + k_pair];
                                    unsigned int up_scales = (unsigned int)smem_v62_0[sc_base + scale_index * 2 + k_pair];
                                    unsigned int dgate_bytes[2];
                                    unsigned int dup_bytes[2];
                                    #pragma unroll
                                    for (int j_34 = 0; j_34 < 2; j_34++) {
                                        int k_block_26 = k_pair * 2 + (tile_row + j_34 & 1);
                                        int pair_byte = tile_row + j_34 & 1;
                                        float gate_scale = __uint_as_float((gate_scales >> (unsigned int)(pair_byte * 8) & 255) << 23);
                                        float up_scale = __uint_as_float((up_scales >> (unsigned int)(pair_byte * 8) & 255) << 23);
                                        float dgate_values[32];
                                        float dup_values[32];
                                        #pragma unroll
                                        for (int k_52 = 0; k_52 < 8; k_52++) {
                                            int col_idx = k_block_26 * 32 + (tile_row / 4 + k_52) % 8 * 4;
                                            unsigned int gate_word = smem_v57_0[tile_base + (tile_row * 128 + col_idx) / 4];
                                            unsigned int up_word = smem_v57_1[tile_base + (tile_row * 128 + col_idx) / 4];
                                            float2 fp8x2_decode;
                                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                "mov.b32 {lo, hi}, pair;\n"
                                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                : "=f"(fp8x2_decode.x), "=f"(fp8x2_decode.y) : "h"((uint16_t)(gate_word & 65535)));
                                            float2 fp8x2_decode_1;
                                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                "mov.b32 {lo, hi}, pair;\n"
                                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                : "=f"(fp8x2_decode_1.x), "=f"(fp8x2_decode_1.y) : "h"((uint16_t)(gate_word >> 16)));
                                            float2 fp8x2_decode_0;
                                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                "mov.b32 {lo, hi}, pair;\n"
                                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                : "=f"(fp8x2_decode_0.x), "=f"(fp8x2_decode_0.y) : "h"((uint16_t)(up_word & 65535)));
                                            float2 fp8x2_decode_1_0;
                                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                "mov.b32 {lo, hi}, pair;\n"
                                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                : "=f"(fp8x2_decode_1_0.x), "=f"(fp8x2_decode_1_0.y) : "h"((uint16_t)(up_word >> 16)));
                                            float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(smem_v57[dh_base + (tile_row * 128 + col_idx) / 2]));
                                            float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(smem_v57[dh_base + (tile_row * 128 + col_idx) / 2 + 1]));
                                            float dg = 0.0f;
                                            float du = 0.0f;
                                            float h = 0.0f;
                                            float _exp_215 = expf(-(fp8x2_decode.x * gate_scale));
                                            float sigmoid = 1.0f / (1.0f + _exp_215);
                                            float silu = fp8x2_decode.x * gate_scale * sigmoid;
                                            float dsilu = (1.0f - silu) * sigmoid + silu;
                                            dg = dsilu * (fp8x2_decode_0.x * up_scale) * _cvt_f32_29.x;
                                            du = silu * _cvt_f32_29.x;
                                            h = silu * (fp8x2_decode_0.x * up_scale);
                                            router_gradient = router_gradient + _cvt_f32_29.x * inverse_28 * h;
                                            dgate_values[k_52 * 4] = dg;
                                            dup_values[k_52 * 4] = du;
                                            float dg_0 = 0.0f;
                                            float du_1 = 0.0f;
                                            float h_2 = 0.0f;
                                            float _exp_216 = expf(-(fp8x2_decode.y * gate_scale));
                                            float sigmoid_3 = 1.0f / (1.0f + _exp_216);
                                            float silu_4 = fp8x2_decode.y * gate_scale * sigmoid_3;
                                            float dsilu_5 = (1.0f - silu_4) * sigmoid_3 + silu_4;
                                            dg_0 = dsilu_5 * (fp8x2_decode_0.y * up_scale) * _cvt_f32_29.y;
                                            du_1 = silu_4 * _cvt_f32_29.y;
                                            h_2 = silu_4 * (fp8x2_decode_0.y * up_scale);
                                            router_gradient = router_gradient + _cvt_f32_29.y * inverse_28 * h_2;
                                            dgate_values[k_52 * 4 + 1] = dg_0;
                                            dup_values[k_52 * 4 + 1] = du_1;
                                            float dg_6 = 0.0f;
                                            float du_7 = 0.0f;
                                            float h_8 = 0.0f;
                                            float _exp_217 = expf(-(fp8x2_decode_1.x * gate_scale));
                                            float sigmoid_9 = 1.0f / (1.0f + _exp_217);
                                            float silu_10 = fp8x2_decode_1.x * gate_scale * sigmoid_9;
                                            float dsilu_11 = (1.0f - silu_10) * sigmoid_9 + silu_10;
                                            dg_6 = dsilu_11 * (fp8x2_decode_1_0.x * up_scale) * _cvt_f32_30.x;
                                            du_7 = silu_10 * _cvt_f32_30.x;
                                            h_8 = silu_10 * (fp8x2_decode_1_0.x * up_scale);
                                            router_gradient = router_gradient + _cvt_f32_30.x * inverse_28 * h_8;
                                            dgate_values[k_52 * 4 + 2] = dg_6;
                                            dup_values[k_52 * 4 + 2] = du_7;
                                            float dg_12 = 0.0f;
                                            float du_13 = 0.0f;
                                            float h_14 = 0.0f;
                                            float _exp_218 = expf(-(fp8x2_decode_1.y * gate_scale));
                                            float sigmoid_15 = 1.0f / (1.0f + _exp_218);
                                            float silu_16 = fp8x2_decode_1.y * gate_scale * sigmoid_15;
                                            float dsilu_17 = (1.0f - silu_16) * sigmoid_15 + silu_16;
                                            dg_12 = dsilu_17 * (fp8x2_decode_1_0.y * up_scale) * _cvt_f32_30.y;
                                            du_13 = silu_16 * _cvt_f32_30.y;
                                            h_14 = silu_16 * (fp8x2_decode_1_0.y * up_scale);
                                            router_gradient = router_gradient + _cvt_f32_30.y * inverse_28 * h_14;
                                            dgate_values[k_52 * 4 + 3] = dg_12;
                                            dup_values[k_52 * 4 + 3] = du_13;
                                        }
                                        unsigned int dgate_bf16[16];
                                        unsigned int dup_bf16[16];
                                        #pragma unroll
                                        for (int p = 0; p < 16; p++) {
                                            dgate_bf16[p] = __as_u32(__float22bfloat162_rn(make_float2(dgate_values[2 * p], dgate_values[2 * p + 1])));
                                            dup_bf16[p] = __as_u32(__float22bfloat162_rn(make_float2(dup_values[2 * p], dup_values[2 * p + 1])));
                                        }
                                        uint32_t bf16x2_abs_29;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_29) : "r"(dgate_bf16[0]));
                                        unsigned int amax_pair_28 = bf16x2_abs_29;
                                        #pragma unroll
                                        for (int i_112 = 1; i_112 < 16; i_112++) {
                                            uint32_t bf16x2_abs_1_55;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_55) : "r"(dgate_bf16[i_112]));
                                            uint32_t bf16x2_max_55;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_55) : "r"(amax_pair_28), "r"(bf16x2_abs_1_55));
                                            amax_pair_28 = bf16x2_max_55;
                                        }
                                        uint16_t bf16_max_28;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_28) : "h"((uint16_t)(amax_pair_28 & 65535)), "h"((uint16_t)(amax_pair_28 >> 16)));
                                        float amax_28;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_28) : "h"((uint16_t)(bf16_max_28)));
                                        float scale_28 = fmaxf(amax_28 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_219;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_219) : "f"(scale_28), "f"(scale_28));
                                        unsigned int scale_byte_28 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_219 & 255;
                                        unsigned int inverse_lane_28 = 254 - scale_byte_28 << 7;
                                        unsigned int inverse_27_1 = inverse_lane_28 | inverse_lane_28 << 16;
                                        unsigned int words_28[8];
                                        #pragma unroll
                                        for (int i_113 = 0; i_113 < 8; i_113++) {
                                            uint32_t bf16x2_mul_56;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_56) : "r"(dgate_bf16[i_113 * 2]), "r"(inverse_27_1));
                                            uint16_t e4m3x2_bf16x2_55;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_55) : "r"(bf16x2_mul_56));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_56);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_55) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t bf16x2_mul_1_55;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_55) : "r"(dgate_bf16[i_113 * 2 + 1]), "r"(inverse_27_1));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_220;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_220) : "r"(bf16x2_mul_1_55));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_55);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_220) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_28[i_113] = (unsigned int)e4m3x2_bf16x2_55 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_220 << 16;
                                        }
                                        uint32_t bf16x2_abs_27_1;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_27_1) : "r"(dup_bf16[0]));
                                        unsigned int amax_pair_27_1 = bf16x2_abs_27_1;
                                        #pragma unroll
                                        for (int i_114 = 1; i_114 < 16; i_114++) {
                                            uint32_t bf16x2_abs_1_56;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_56) : "r"(dup_bf16[i_114]));
                                            uint32_t bf16x2_max_56;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_56) : "r"(amax_pair_27_1), "r"(bf16x2_abs_1_56));
                                            amax_pair_27_1 = bf16x2_max_56;
                                        }
                                        uint16_t bf16_max_27_1;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_27_1) : "h"((uint16_t)(amax_pair_27_1 & 65535)), "h"((uint16_t)(amax_pair_27_1 >> 16)));
                                        float amax_27_1;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_27_1) : "h"((uint16_t)(bf16_max_27_1)));
                                        float scale_27_1 = fmaxf(amax_27_1 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_221;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_221) : "f"(scale_27_1), "f"(scale_27_1));
                                        unsigned int scale_byte_27_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_221 & 255;
                                        unsigned int inverse_lane_27_1 = 254 - scale_byte_27_1 << 7;
                                        unsigned int inverse_28_1 = inverse_lane_27_1 | inverse_lane_27_1 << 16;
                                        unsigned int words_27_1[8];
                                        #pragma unroll
                                        for (int i_115 = 0; i_115 < 8; i_115++) {
                                            uint32_t bf16x2_mul_57;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_57) : "r"(dup_bf16[i_115 * 2]), "r"(inverse_28_1));
                                            uint16_t e4m3x2_bf16x2_56;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_56) : "r"(bf16x2_mul_57));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_57);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_56) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t bf16x2_mul_1_56;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_56) : "r"(dup_bf16[i_115 * 2 + 1]), "r"(inverse_28_1));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_222;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_222) : "r"(bf16x2_mul_1_56));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_56);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_222) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_27_1[i_115] = (unsigned int)e4m3x2_bf16x2_56 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_222 << 16;
                                        }
                                        dgate_bytes[j_34] = scale_byte_28;
                                        dup_bytes[j_34] = scale_byte_27_1;
                                        #pragma unroll
                                        for (int k_53 = 0; k_53 < 8; k_53++) {
                                            int col_idx_1 = k_block_26 * 32 + (tile_row / 4 + k_53) % 8 * 4;
                                            smem_v57_0[tile_base + (tile_row * 128 + col_idx_1) / 4] = words_28[k_53];
                                            smem_v57_1[tile_base + (tile_row * 128 + col_idx_1) / 4] = words_27_1[k_53];
                                            smem_v57[dh_base + (tile_row * 128 + col_idx_1) / 2] = dgate_bf16[2 * k_53];
                                            smem_v57[dh_base + (tile_row * 128 + col_idx_1) / 2 + 1] = dgate_bf16[2 * k_53 + 1];
                                            smem_v57_4[(tile_row * 128 + col_idx_1) / 2] = dup_bf16[2 * k_53];
                                            smem_v57_4[(tile_row * 128 + col_idx_1) / 2 + 1] = dup_bf16[2 * k_53 + 1];
                                        }
                                    }
                                    unsigned int pair_g = dgate_bytes[0] | dgate_bytes[1] << 8;
                                    unsigned int pair_u = dup_bytes[0] | dup_bytes[1] << 8;
                                    if ((tile_row & 1) != 0) {
                                        pair_g = dgate_bytes[1] | dgate_bytes[0] << 8;
                                        pair_u = dup_bytes[1] | dup_bytes[0] << 8;
                                    }
                                    int half_index = (tile_row % 32 * 16 + tile_row / 32 * 4 + k_pair * 2) / 2;
                                    smem_v62[sc_base + half_index] = (uint16_t)pair_g;
                                    smem_v62_0[sc_base + half_index] = (uint16_t)pair_u;
                                    if (stage_4 > 0) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                    }
                                    __syncthreads();
                                    if (tid == 0) {
                                        int local_row_6 = row_31 - macro_row_offset_2;
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_2d((&dg_store), col_74 * 128, local_row_6 * 128, smem_v57_addr_0 + (unsigned int)(slot_1 * 16384));
                                        tma_store_3d((&dg_sc_store), 0, 0, local_row_6 * col_blocks_12 + col_74, smem_v57_addr_2 + (unsigned int)(slot_1 * 512));
                                        tma_store_2d((&du_store), col_74 * 128, local_row_6 * 128, smem_v57_addr_1 + (unsigned int)(slot_1 * 16384));
                                        tma_store_3d((&du_sc_store), 0, 0, local_row_6 * col_blocks_12 + col_74, smem_v57_addr_3 + (unsigned int)(slot_1 * 512));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    if (tid < 128) {
                                        if (tid < 128) {
                                            int r_13 = tid % 64;
                                            int row0_13 = r_13 * 2;
                                            int row1_13 = row0_13 + 1;
                                            int half_0 = tid / 64;
                                            int word_stride_13 = 64;
                                            int scale_base0_13 = (row0_13 % 32 * 4 + row0_13 / 32) * 4;
                                            int scale_base1_13 = (row1_13 % 32 * 4 + row1_13 / 32) * 4;
                                            #pragma unroll 1
                                            for (int j_35 = 0; j_35 < 2; j_35++) {
                                                int k_block_27 = (half_0 * 2 + j_35 + tid / 8) % 4;
                                                unsigned int pairs0_13[16];
                                                unsigned int pairs1_13[16];
                                                #pragma unroll
                                                for (int k_54 = 0; k_54 < 16; k_54++) {
                                                    int col_8_1 = k_block_27 * 32 + (tid * 4 + k_54 * 2) % 32;
                                                    unsigned int w0_13 = smem_v57[dh_base + col_8_1 * word_stride_13 + r_13];
                                                    unsigned int w1_13 = smem_v57[dh_base + (col_8_1 + 1) * word_stride_13 + r_13];
                                                    uint32_t lo_13;
                                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_13) : "r"(w0_13), "r"(w1_13));
                                                    uint32_t hi_13;
                                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_13) : "r"(w0_13), "r"(w1_13));
                                                    pairs0_13[k_54] = lo_13;
                                                    pairs1_13[k_54] = hi_13;
                                                }
                                                uint32_t bf16x2_abs_30;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_30) : "r"(pairs0_13[0]));
                                                unsigned int amax_pair_29 = bf16x2_abs_30;
                                                #pragma unroll
                                                for (int i_116 = 1; i_116 < 16; i_116++) {
                                                    uint32_t bf16x2_abs_1_57;
                                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_57) : "r"(pairs0_13[i_116]));
                                                    uint32_t bf16x2_max_57;
                                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_57) : "r"(amax_pair_29), "r"(bf16x2_abs_1_57));
                                                    amax_pair_29 = bf16x2_max_57;
                                                }
                                                uint16_t bf16_max_29;
                                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_29) : "h"((uint16_t)(amax_pair_29 & 65535)), "h"((uint16_t)(amax_pair_29 >> 16)));
                                                float amax_29;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_29) : "h"((uint16_t)(bf16_max_29)));
                                                float scale_29 = fmaxf(amax_29 * 0.002232143f, 1e-12f);
                                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_223;
                                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_223) : "f"(scale_29), "f"(scale_29));
                                                unsigned int scale_byte_29 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_223 & 255;
                                                unsigned int inverse_lane_29 = 254 - scale_byte_29 << 7;
                                                unsigned int inverse_29 = inverse_lane_29 | inverse_lane_29 << 16;
                                                unsigned int words_29[8];
                                                #pragma unroll
                                                for (int i_117 = 0; i_117 < 8; i_117++) {
                                                    uint32_t bf16x2_mul_58;
                                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_58) : "r"(pairs0_13[i_117 * 2]), "r"(inverse_29));
                                                    uint16_t e4m3x2_bf16x2_57;
                                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_57) : "r"(bf16x2_mul_58));
                                                    #else
                                                    {
                                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_58);
                                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_57) : "f"(_hi), "f"(_lo));
                                                    }
                                                    #endif
                                                    uint32_t bf16x2_mul_1_57;
                                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_57) : "r"(pairs0_13[i_117 * 2 + 1]), "r"(inverse_29));
                                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_224;
                                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_224) : "r"(bf16x2_mul_1_57));
                                                    #else
                                                    {
                                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_57);
                                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_224) : "f"(_hi), "f"(_lo));
                                                    }
                                                    #endif
                                                    words_29[i_117] = (unsigned int)e4m3x2_bf16x2_57 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_224 << 16;
                                                }
                                                uint32_t bf16x2_abs_28_1;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_28_1) : "r"(pairs1_13[0]));
                                                unsigned int amax_pair_28_1 = bf16x2_abs_28_1;
                                                #pragma unroll
                                                for (int i_118 = 1; i_118 < 16; i_118++) {
                                                    uint32_t bf16x2_abs_1_58;
                                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_58) : "r"(pairs1_13[i_118]));
                                                    uint32_t bf16x2_max_58;
                                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_58) : "r"(amax_pair_28_1), "r"(bf16x2_abs_1_58));
                                                    amax_pair_28_1 = bf16x2_max_58;
                                                }
                                                uint16_t bf16_max_28_1;
                                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_28_1) : "h"((uint16_t)(amax_pair_28_1 & 65535)), "h"((uint16_t)(amax_pair_28_1 >> 16)));
                                                float amax_28_1;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_28_1) : "h"((uint16_t)(bf16_max_28_1)));
                                                float scale_28_1 = fmaxf(amax_28_1 * 0.002232143f, 1e-12f);
                                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_225;
                                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_225) : "f"(scale_28_1), "f"(scale_28_1));
                                                unsigned int scale_byte_28_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_225 & 255;
                                                unsigned int inverse_lane_28_1 = 254 - scale_byte_28_1 << 7;
                                                unsigned int inverse_30 = inverse_lane_28_1 | inverse_lane_28_1 << 16;
                                                unsigned int words_28_1[8];
                                                #pragma unroll
                                                for (int i_119 = 0; i_119 < 8; i_119++) {
                                                    uint32_t bf16x2_mul_59;
                                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_59) : "r"(pairs1_13[i_119 * 2]), "r"(inverse_30));
                                                    uint16_t e4m3x2_bf16x2_58;
                                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_58) : "r"(bf16x2_mul_59));
                                                    #else
                                                    {
                                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_59);
                                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_58) : "f"(_hi), "f"(_lo));
                                                    }
                                                    #endif
                                                    uint32_t bf16x2_mul_1_58;
                                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_58) : "r"(pairs1_13[i_119 * 2 + 1]), "r"(inverse_30));
                                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_226;
                                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_226) : "r"(bf16x2_mul_1_58));
                                                    #else
                                                    {
                                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_58);
                                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_226) : "f"(_hi), "f"(_lo));
                                                    }
                                                    #endif
                                                    words_28_1[i_119] = (unsigned int)e4m3x2_bf16x2_58 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_226 << 16;
                                                }
                                                #pragma unroll
                                                for (int k_55 = 0; k_55 < 8; k_55++) {
                                                    int col_9_1 = k_block_27 * 32 + (tid * 4 + k_55 * 4) % 32;
                                                    smem_v57_5[(row0_13 * 128 + col_9_1) / 4] = words_29[k_55];
                                                    smem_v57_5[(row1_13 * 128 + col_9_1) / 4] = words_28_1[k_55];
                                                }
                                                {
                                                    uint32_t _byte_227 = static_cast<uint32_t>(scale_byte_29) & 0xFFu;
                                                    uint32_t _addr_227 = static_cast<uint32_t>(smem_v57_addr_7 + (unsigned int)(scale_base0_13 + k_block_27));
                                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_227), "r"(_byte_227) : "memory");
                                                }
                                                {
                                                    uint32_t _byte_228 = static_cast<uint32_t>(scale_byte_28_1) & 0xFFu;
                                                    uint32_t _addr_228 = static_cast<uint32_t>(smem_v57_addr_7 + (unsigned int)(scale_base1_13 + k_block_27));
                                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_228), "r"(_byte_228) : "memory");
                                                }
                                            }
                                        }
                                    } else if (tid - 128 < 128) {
                                        int r_14 = (tid - 128) % 64;
                                        int row0_14 = r_14 * 2;
                                        int row1_14 = row0_14 + 1;
                                        int half_1_1 = (tid - 128) / 64;
                                        int word_stride_14 = 64;
                                        int scale_base0_14 = (row0_14 % 32 * 4 + row0_14 / 32) * 4;
                                        int scale_base1_14 = (row1_14 % 32 * 4 + row1_14 / 32) * 4;
                                        #pragma unroll 1
                                        for (int j_36 = 0; j_36 < 2; j_36++) {
                                            int k_block_28 = (half_1_1 * 2 + j_36 + (tid - 128) / 8) % 4;
                                            unsigned int pairs0_14[16];
                                            unsigned int pairs1_14[16];
                                            #pragma unroll
                                            for (int k_56 = 0; k_56 < 16; k_56++) {
                                                int col_10_1 = k_block_28 * 32 + ((tid - 128) * 4 + k_56 * 2) % 32;
                                                unsigned int w0_14 = smem_v57_4[col_10_1 * word_stride_14 + r_14];
                                                unsigned int w1_14 = smem_v57_4[(col_10_1 + 1) * word_stride_14 + r_14];
                                                uint32_t lo_14;
                                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_14) : "r"(w0_14), "r"(w1_14));
                                                uint32_t hi_14;
                                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_14) : "r"(w0_14), "r"(w1_14));
                                                pairs0_14[k_56] = lo_14;
                                                pairs1_14[k_56] = hi_14;
                                            }
                                            uint32_t bf16x2_abs_31;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_31) : "r"(pairs0_14[0]));
                                            unsigned int amax_pair_30 = bf16x2_abs_31;
                                            #pragma unroll
                                            for (int i_120 = 1; i_120 < 16; i_120++) {
                                                uint32_t bf16x2_abs_1_59;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_59) : "r"(pairs0_14[i_120]));
                                                uint32_t bf16x2_max_59;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_59) : "r"(amax_pair_30), "r"(bf16x2_abs_1_59));
                                                amax_pair_30 = bf16x2_max_59;
                                            }
                                            uint16_t bf16_max_30;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_30) : "h"((uint16_t)(amax_pair_30 & 65535)), "h"((uint16_t)(amax_pair_30 >> 16)));
                                            float amax_30;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_30) : "h"((uint16_t)(bf16_max_30)));
                                            float scale_30 = fmaxf(amax_30 * 0.002232143f, 1e-12f);
                                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_229;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_229) : "f"(scale_30), "f"(scale_30));
                                            unsigned int scale_byte_30 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_229 & 255;
                                            unsigned int inverse_lane_30 = 254 - scale_byte_30 << 7;
                                            unsigned int inverse_31 = inverse_lane_30 | inverse_lane_30 << 16;
                                            unsigned int words_30[8];
                                            #pragma unroll
                                            for (int i_121 = 0; i_121 < 8; i_121++) {
                                                uint32_t bf16x2_mul_60;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_60) : "r"(pairs0_14[i_121 * 2]), "r"(inverse_31));
                                                uint16_t e4m3x2_bf16x2_59;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_59) : "r"(bf16x2_mul_60));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_60);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_59) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                uint32_t bf16x2_mul_1_59;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_59) : "r"(pairs0_14[i_121 * 2 + 1]), "r"(inverse_31));
                                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_230;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_230) : "r"(bf16x2_mul_1_59));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_59);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_230) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                words_30[i_121] = (unsigned int)e4m3x2_bf16x2_59 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_230 << 16;
                                            }
                                            uint32_t bf16x2_abs_29_1;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_29_1) : "r"(pairs1_14[0]));
                                            unsigned int amax_pair_29_1 = bf16x2_abs_29_1;
                                            #pragma unroll
                                            for (int i_122 = 1; i_122 < 16; i_122++) {
                                                uint32_t bf16x2_abs_1_60;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_60) : "r"(pairs1_14[i_122]));
                                                uint32_t bf16x2_max_60;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_60) : "r"(amax_pair_29_1), "r"(bf16x2_abs_1_60));
                                                amax_pair_29_1 = bf16x2_max_60;
                                            }
                                            uint16_t bf16_max_29_1;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_29_1) : "h"((uint16_t)(amax_pair_29_1 & 65535)), "h"((uint16_t)(amax_pair_29_1 >> 16)));
                                            float amax_29_1;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_29_1) : "h"((uint16_t)(bf16_max_29_1)));
                                            float scale_29_1 = fmaxf(amax_29_1 * 0.002232143f, 1e-12f);
                                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_231;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_231) : "f"(scale_29_1), "f"(scale_29_1));
                                            unsigned int scale_byte_29_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_231 & 255;
                                            unsigned int inverse_lane_29_1 = 254 - scale_byte_29_1 << 7;
                                            unsigned int inverse_32 = inverse_lane_29_1 | inverse_lane_29_1 << 16;
                                            unsigned int words_29_1[8];
                                            #pragma unroll
                                            for (int i_123 = 0; i_123 < 8; i_123++) {
                                                uint32_t bf16x2_mul_61;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_61) : "r"(pairs1_14[i_123 * 2]), "r"(inverse_32));
                                                uint16_t e4m3x2_bf16x2_60;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_60) : "r"(bf16x2_mul_61));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_61);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_60) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                uint32_t bf16x2_mul_1_60;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_60) : "r"(pairs1_14[i_123 * 2 + 1]), "r"(inverse_32));
                                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_232;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_232) : "r"(bf16x2_mul_1_60));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_60);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_232) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                words_29_1[i_123] = (unsigned int)e4m3x2_bf16x2_60 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_232 << 16;
                                            }
                                            #pragma unroll
                                            for (int k_57 = 0; k_57 < 8; k_57++) {
                                                int col_11_1 = k_block_28 * 32 + ((tid - 128) * 4 + k_57 * 4) % 32;
                                                smem_v57_6[(row0_14 * 128 + col_11_1) / 4] = words_30[k_57];
                                                smem_v57_6[(row1_14 * 128 + col_11_1) / 4] = words_29_1[k_57];
                                            }
                                            {
                                                uint32_t _byte_233 = static_cast<uint32_t>(scale_byte_30) & 0xFFu;
                                                uint32_t _addr_233 = static_cast<uint32_t>(smem_v57_addr_8 + (unsigned int)(scale_base0_14 + k_block_28));
                                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_233), "r"(_byte_233) : "memory");
                                            }
                                            {
                                                uint32_t _byte_234 = static_cast<uint32_t>(scale_byte_29_1) & 0xFFu;
                                                uint32_t _addr_234 = static_cast<uint32_t>(smem_v57_addr_8 + (unsigned int)(scale_base1_14 + k_block_28));
                                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_234), "r"(_byte_234) : "memory");
                                            }
                                        }
                                    }
                                    if (half_19 != 0) {
                                        smem_v70[stage_4 * 128 + tile_row] = router_gradient;
                                    }
                                    __syncthreads();
                                    if (tid == 0) {
                                        int local_row_7 = row_31 - macro_row_offset_2;
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_2d((&dg_t_store), local_row_7 * 128, col_74 * 128, smem_v57_addr_5);
                                        tma_store_3d((&dg_sc_t_store), 0, 0, col_74 * macro_tiles_4 + local_row_7, smem_v57_addr_7);
                                        tma_store_2d((&du_t_store), local_row_7 * 128, col_74 * 128, smem_v57_addr_6);
                                        tma_store_3d((&du_sc_t_store), 0, 0, col_74 * macro_tiles_4 + local_row_7, smem_v57_addr_8);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    if (half_19 == 0 && peer_11 >= 0) {
                                        router_gradient = router_gradient + smem_v70[stage_4 * 128 + tile_row];
                                        partials[local_token * col_blocks_12 + col_74] = router_gradient;
                                    }
                                    if (stage_4 == 1 || tile_end_2 <= first_tile_3 + 1) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                    }
                                    __syncthreads();
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                if (tile_end_2 > first_tile_3) {
                                    int row_20_1 = first_row_5;
                                    if (col_blocks_12 <= first_col_2) {
                                        row_20_1 = row_20_1 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (shared_rows + row_20_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                                if (tile_end_2 > first_tile_3 + 1) {
                                    int row_21_1 = first_row_5;
                                    if (col_blocks_12 <= first_col_2 + 1) {
                                        row_21_1 = row_21_1 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (shared_rows + row_21_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        if (tid == 0) {
                            bool enabled_value_18 = macros > 1;
                            if (enabled_value_18 != 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        int _min_236 = ((2) < (tile_end_2 - first_tile_3) ? (2) : (tile_end_2 - first_tile_3));
                        int _max_235 = ((0) > (_min_236) ? (0) : (_min_236));
                        swiglu_seq += _max_235;
                    } else {
                        if (tile_word_0 == 2) {
                            int col_blocks_13 = (hidden + 512 - 1) / 512;
                            int x_8 = -1;
                            int y_8 = -1;
                            int expert_8 = -1;
                            int k_start_8 = 0;
                            int k_end_8 = 0;
                            int first_8 = 0;
                            int first_block_6 = (tile_word_2 * (macro_size / mini_size) + tile_word_3) * (mini_size / 256);
                            int end_block_6 = ((first_block_6 + mini_size / 256) < (tokens / 256) ? (first_block_6 + mini_size / 256) : (tokens / 256));
                            int block_17 = first_block_6;
                            uint32_t fast_div_q_7 = (dx_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), dx_col_blocks_div.multiplier) >> dx_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                            uint32_t fast_div_r_7 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_7 * (uint32_t)(dx_col_blocks_div.divisor);
                            block_17 = first_block_6 + (int)fast_div_q_7;
                            if (block_17 < end_block_6) {
                                int index_5 = tile_word_5;
                                int offset_11 = tile_word_6 / 256;
                                int first_row_6 = ((first_block_6) > (offset_11) ? (first_block_6) : (offset_11));
                                int _min_237 = ((end_block_6) < (offset_11 + tile_word_7 / 256) ? (end_block_6) : (offset_11 + tile_word_7 / 256));
                                int rows_6 = _min_237 - first_row_6;
                                int supergroup_8 = (tile_word_1 - (first_row_6 - first_block_6) * col_blocks_13) / (rows_6 * 8);
                                int full_cols_8 = col_blocks_13 / 8 * 8;
                                int row_32 = 0;
                                int col_75 = 0;
                                if (tile_word_1 - (first_row_6 - first_block_6) * col_blocks_13 < rows_6 * full_cols_8) {
                                    row_32 = (tile_word_1 - (first_row_6 - first_block_6) * col_blocks_13) % (rows_6 * 8) / 8;
                                    col_75 = supergroup_8 * 8 + (tile_word_1 - (first_row_6 - first_block_6) * col_blocks_13) % 8;
                                } else {
                                    row_32 = (tile_word_1 - (first_row_6 - first_block_6) * col_blocks_13 - rows_6 * full_cols_8) / (col_blocks_13 - full_cols_8);
                                    col_75 = full_cols_8 + (tile_word_1 - (first_row_6 - first_block_6) * col_blocks_13 - rows_6 * full_cols_8) % (col_blocks_13 - full_cols_8);
                                }
                                if ((supergroup_8 & 1) != 0) {
                                    row_32 = rows_6 - row_32 - 1;
                                }
                                x_8 = first_row_6 + row_32 - tile_word_2 * (macro_size / 256);
                                y_8 = col_75;
                                expert_8 = index_5;
                            }
                            int has_hi_8 = 0;
                            has_hi_8 = (int)((y_8 * 2 + 1) * 256 < hidden);
                            int global_mini_10 = tile_word_2 * (macro_size / mini_size) + tile_word_3;
                            int macro_rows_8 = tile_word_2 * (macro_size / 256);
                            int iterations_8 = intermediate / 128 + intermediate / 128;
                            int macro_k_3 = tile_word_2 * (macro_size / 128);
                            if (expert_8 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_19 = macros > 1;
                                    if (enabled_value_19 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else {
                                if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            {
                                                bool enabled_value_20 = 1;
                                                if (enabled_value_20 != 0) {
                                                    int poll_20_first;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_20_first) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    int poll_20 = poll_20_first;
                                                    while (poll_20 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int poll_20_next;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_20_next) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                        poll_20 = poll_20_next;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                                int _min_238 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                                int mini_rows_9 = ((0) > (_min_238) ? (0) : (_min_238));
                                                int required_10 = (mini_rows_9 + 127) / 128 * ((intermediate + 511) / 512);
                                            }
                                            #pragma unroll 1
                                            for (int idx_19 = 0; idx_19 < iterations_8; idx_19++) {
                                                int _stage_gemm_finished_8 = (int)((unsigned int)(gemm_wide + idx_19) % 4);
                                                int _parity_gemm_finished_8 = (int)((unsigned int)(gemm_wide + idx_19) / 4 % 2) ^ 1;
                                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_8) * 8, _parity_gemm_finished_8);
                                                if (idx_19 < intermediate / 128) {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_19) % 4 * 16384)), "l"((&dg_q)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_19), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_19) % 4 * 16384)), "l"((&wg_t_q)), "r"(0), "r"(y_8 * 512 + cta_rank_0 * 128), "r"(idx_19), "r"(expert_8), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v44_addr + (unsigned int)((gemm_wide + idx_19) % 4 * 16384)), "l"((&wg_t_q)), "r"(0), "r"(y_8 * 512 + 256 + cta_rank_0 * 128), "r"(idx_19), "r"(expert_8), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                } else {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_19) % 4 * 16384)), "l"((&du_q)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_19) % 4 * 16384)), "l"((&wu_t_q)), "r"(0), "r"(y_8 * 512 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(expert_8), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v44_addr + (unsigned int)((gemm_wide + idx_19) % 4 * 16384)), "l"((&wu_t_q)), "r"(0), "r"(y_8 * 512 + 256 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(expert_8), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 6) {
                                    if (warp == 6) {
                                        if (elect_sync()) {
                                            {
                                                bool enabled_value_21 = 1;
                                                if (enabled_value_21 != 0) {
                                                    int poll_21_first;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_21_first) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    int poll_21 = poll_21_first;
                                                    while (poll_21 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int poll_21_next;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_21_next) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                        poll_21 = poll_21_next;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                                int _min_239 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                                int mini_rows_10 = ((0) > (_min_239) ? (0) : (_min_239));
                                                int required_11 = (mini_rows_10 + 127) / 128 * ((intermediate + 511) / 512);
                                            }
                                            #pragma unroll 1
                                            for (int idx_20 = 0; idx_20 < iterations_8; idx_20++) {
                                                int _stage_scales_finished_3 = (int)((unsigned int)(scales_wide + idx_20) % 4);
                                                int _parity_scales_finished_3 = (int)((unsigned int)(scales_wide + idx_20) / 4 % 2) ^ 1;
                                                mbarrier_wait(scales_finished_addr + (_stage_scales_finished_3) * 8, _parity_scales_finished_3);
                                                if (idx_20 < intermediate / 128) {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_20) % 4 * 512)), "l"((&dg_sc)), "r"(0), "r"(0), "r"((x_8 * 2 + cta_rank_0) * (intermediate / 128) + idx_20),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_20) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_20) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_t_sc)), "r"(0), "r"(0), "r"((expert_8 * h_tiles + y_8 * 4 + cta_rank_0) * (intermediate / 128) + idx_20),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_20) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v47_addr + (unsigned int)((scales_wide + idx_20) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_t_sc)), "r"(0), "r"(0), "r"((expert_8 * h_tiles + y_8 * 4 + 2 + cta_rank_0) * (intermediate / 128) + idx_20),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_20) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                } else {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_20) % 4 * 512)), "l"((&du_sc)), "r"(0), "r"(0), "r"((x_8 * 2 + cta_rank_0) * (intermediate / 128) + (idx_20 - intermediate / 128)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_20) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_20) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_t_sc)), "r"(0), "r"(0), "r"((expert_8 * h_tiles + y_8 * 4 + cta_rank_0) * (intermediate / 128) + (idx_20 - intermediate / 128)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_20) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v47_addr + (unsigned int)((scales_wide + idx_20) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_t_sc)), "r"(0), "r"(0), "r"((expert_8 * h_tiles + y_8 * 4 + 2 + cta_rank_0) * (intermediate / 128) + (idx_20 - intermediate / 128)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_20) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int _parity_output_finished_8 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                                mbarrier_wait(output_finished_addr, _parity_output_finished_8);
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_21 = 0; idx_21 < iterations_8; idx_21++) {
                                                    mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_21) % 4)) * 8, 5120);
                                                    int _stage_scales_arrived_3 = (int)((unsigned int)(scales_wide + idx_21) % 4);
                                                    int _parity_scales_arrived_3 = (int)((unsigned int)(scales_wide + idx_21) / 4 % 2);
                                                    mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_3) * 8, _parity_scales_arrived_3);
                                                    int buffer_3 = idx_21 % 3;
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_3 * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_21) % 4 * 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_3 * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_21) % 4 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_3 * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_21) % 4 * 1024) + 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_3 * 8, make_sf_cp_desc_sbo128(smem_v47_addr + (unsigned int)((scales_wide + idx_21) % 4 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_3 * 8 + 4), make_sf_cp_desc_sbo128((smem_v47_addr + (unsigned int)((scales_wide + idx_21) % 4 * 1024) + 512)));
                                                    tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_21) % 4)) * 8, (uint16_t)(3));
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_21) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_8 = (int)((unsigned int)(gemm_wide + idx_21) % 4);
                                                    int _parity_gemm_arrived_8 = (int)((unsigned int)(gemm_wide + idx_21) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_8) * 8, _parity_gemm_arrived_8);
                                                    int _smem_desc_lo_240 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                                    int _smem_desc_lo_241 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)_smem_desc_lo_240) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)_smem_desc_lo_241) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                            0x90c00000U, (int)((tmem_tmem_sfa + buffer_3 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_3 * 8 + 0)), ((idx_21 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                            0xd0c00020U, (int)((tmem_tmem_sfa + buffer_3 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_3 * 8 + 0) | 0x80000000), 1);
                                                    }
                                                    int _smem_desc_lo_242 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                                    int _smem_desc_lo_243 = (((smem_v44_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)_smem_desc_lo_242) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)_smem_desc_lo_243) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                            0x90c00000U, (int)((tmem_tmem_sfa + buffer_3 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_3 * 8 + 0)), ((idx_21 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                            0xd0c00020U, (int)((tmem_tmem_sfa + buffer_3 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_3 * 8 + 0) | 0x80000000), 1);
                                                    }
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_21) % 4)) * 8, (uint16_t)(3));
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else if (tid < 128) {
                                        int _parity_output_arrived_8 = (int)((unsigned int)gemm_out0 % 2);
                                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_8);
                                        unsigned int packed_13[128];
                                        #pragma unroll
                                        for (int chunk_18 = 0; chunk_18 < 8; chunk_18++) {
                                            #pragma unroll
                                            for (int sub_6 = 0; sub_6 < 2; sub_6++) {
                                                unsigned int address_25 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_6 * 16 << 16) + (unsigned int)(chunk_18 * 32);
                                                float _tmem_load_14[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                                                    : "r"(address_25));
                                                #pragma unroll
                                                for (int pair_14 = 0; pair_14 < 8; pair_14++) {
                                                    packed_13[chunk_18 * 16 + sub_6 * 8 + pair_14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[pair_14 * 2], _tmem_load_14[pair_14 * 2 + 1])));
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        int last_4 = 1;
                                        last_4 = 1 - has_hi_8;
                                        if (last_4 != 0) {
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                        if (tid == 0) {
                                            int previous_offset_3 = (tile_word_2 + 1) * macro_size;
                                            int output_row_3 = x_8 * 256 + cta_rank_0 * 128;
                                            int _min_244 = ((macro_size) < (tokens - previous_offset_3) ? (macro_size) : (tokens - previous_offset_3));
                                            if (output_row_3 < _min_244) {
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_19 = 0; chunk_19 < 8; chunk_19++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_7 = tid / 32;
                                            int lane_9 = tid % 32;
                                            #pragma unroll
                                            for (int half_20 = 0; half_20 < 2; half_20++) {
                                                #pragma unroll
                                                for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                                    int row_33 = warp_0_7 * 32 + half_20 * 16 + lane_9 % 16;
                                                    int col_76 = col_tile_17 * 16 + lane_9 / 16 * 8;
                                                    unsigned int address_26 = d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192) + (unsigned int)((row_33 * 32 + col_76) * 2);
                                                    address_26 = address_26 ^ (address_26 & 511) >> 7 << 4;
                                                    int offset_12 = chunk_19 * 16 + half_20 * 8 + col_tile_17 * 4;
                                                    uint32_t _stmatrix_addr_245 = static_cast<uint32_t>(address_26);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_245), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_12 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                tma_store_5d((&dx_r), 0, x_8 * 256 + cta_rank_0 * 128, y_8 * 2 * 8 + chunk_19, 0, 0, d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192));
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        if (has_hi_8 != 0) {
                                            unsigned int packed_0_3[128];
                                            #pragma unroll
                                            for (int chunk_20 = 0; chunk_20 < 8; chunk_20++) {
                                                #pragma unroll
                                                for (int sub_7 = 0; sub_7 < 2; sub_7++) {
                                                    unsigned int address_27 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_7 * 16 << 16) + 256 + (unsigned int)(chunk_20 * 32);
                                                    float _tmem_load_15[16];
                                                    asm volatile(
                                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                                                        : "r"(address_27));
                                                    #pragma unroll
                                                    for (int pair_15 = 0; pair_15 < 8; pair_15++) {
                                                        packed_0_3[chunk_20 * 16 + sub_7 * 8 + pair_15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[pair_15 * 2], _tmem_load_15[pair_15 * 2 + 1])));
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
                                                int warp_0_8 = tid / 32;
                                                int lane_10 = tid % 32;
                                                #pragma unroll
                                                for (int half_21 = 0; half_21 < 2; half_21++) {
                                                    #pragma unroll
                                                    for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                                        int row_34 = warp_0_8 * 32 + half_21 * 16 + lane_10 % 16;
                                                        int col_77 = col_tile_18 * 16 + lane_10 / 16 * 8;
                                                        unsigned int address_28 = d_smem_addr + (unsigned int)((8 + chunk_21) % 3 * 8192) + (unsigned int)((row_34 * 32 + col_77) * 2);
                                                        address_28 = address_28 ^ (address_28 & 511) >> 7 << 4;
                                                        int offset_13 = chunk_21 * 16 + half_21 * 8 + col_tile_18 * 4;
                                                        uint32_t _stmatrix_addr_246 = static_cast<uint32_t>(address_28);
                                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                            :: "r"(_stmatrix_addr_246), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_13 + 3]))
                                                            : "memory");
                                                    }
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    tma_store_5d((&dx_r), 0, x_8 * 256 + cta_rank_0 * 128, (y_8 * 2 + 1) * 8 + chunk_21, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_21) % 3 * 8192));
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
                                                    bool enabled_value_22 = 1;
                                                    if (enabled_value_22 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_10))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                    bool enabled_value_4_1 = macros > 1;
                                                    if (enabled_value_4_1 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_5_1 = 1;
                                                        if (enabled_value_5_1 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                    if (has_hi_8 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_6_1 = 1;
                                                        if (enabled_value_6_1 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_10))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                gemm_wide += iterations_8;
                                scales_wide += iterations_8;
                                gemm_out0 += 1;
                                gemm_acc_tiles += 1;
                            }
                        } else if (tile_word_0 == 3) {
                            int col_blocks_14 = (macro_size + 512 - 1) / 512;
                            {
                                col_blocks_14 = (intermediate + 512 - 1) / 512;
                            }
                            int x_9 = -1;
                            int y_9 = -1;
                            int expert_9 = -1;
                            int k_start_9 = 0;
                            int k_end_9 = 0;
                            int first_9 = 0;
                            int row_blocks_5 = hidden / 256;
                            int expert_idx_4 = 0;
                            int local_task_3 = tile_word_1;
                            uint32_t fast_div_q_8 = (routed_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), routed_wgrad_div.multiplier) >> routed_wgrad_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                            uint32_t fast_div_r_8 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_8 * (uint32_t)(routed_wgrad_div.divisor);
                            expert_idx_4 = (int)fast_div_q_8;
                            local_task_3 = (int)fast_div_r_8;
                            int offset_14 = tile_word_5;
                            int real = tile_word_6;
                            real = (real + 128 - 1) / 128 * 128;
                            int _max_247 = ((offset_14) > (tile_word_2 * macro_size) ? (offset_14) : (tile_word_2 * macro_size));
                            k_start_9 = _max_247;
                            int _min_249 = (((tile_word_2 + 1) * macro_size) < (tokens) ? ((tile_word_2 + 1) * macro_size) : (tokens));
                            int _min_248 = ((offset_14 + real) < (_min_249) ? (offset_14 + real) : (_min_249));
                            k_end_9 = _min_248;
                            first_9 = (int)(k_start_9 == offset_14);
                            if (k_start_9 < k_end_9 && local_task_3 < row_blocks_5 * col_blocks_14) {
                                int supergroup_9 = local_task_3 / (row_blocks_5 * 8);
                                int full_cols_9 = col_blocks_14 / 8 * 8;
                                int row_35 = 0;
                                int col_78 = 0;
                                if (local_task_3 < row_blocks_5 * full_cols_9) {
                                    row_35 = local_task_3 % (row_blocks_5 * 8) / 8;
                                    col_78 = supergroup_9 * 8 + local_task_3 % 8;
                                } else {
                                    row_35 = (local_task_3 - row_blocks_5 * full_cols_9) / (col_blocks_14 - full_cols_9);
                                    col_78 = full_cols_9 + (local_task_3 - row_blocks_5 * full_cols_9) % (col_blocks_14 - full_cols_9);
                                }
                                if ((supergroup_9 & 1) != 0) {
                                    row_35 = row_blocks_5 - row_35 - 1;
                                }
                                x_9 = row_35;
                                y_9 = col_78;
                                expert_9 = expert_idx_4;
                            }
                            int has_hi_9 = 0;
                            has_hi_9 = (int)((y_9 * 2 + 1) * 256 < intermediate);
                            int global_mini_11 = tile_word_2 * (macro_size / mini_size);
                            int macro_rows_9 = tile_word_2 * (macro_size / 256);
                            int iterations_9 = hidden / 128;
                            int macro_k_4 = tile_word_2 * (macro_size / 128);
                            iterations_9 = (k_end_9 - k_start_9) / 128;
                            if (expert_9 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_23 = macros > 1;
                                    if (enabled_value_23 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else {
                                if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            #pragma unroll 1
                                            for (int idx_22 = 0; idx_22 < iterations_9; idx_22++) {
                                                int token_row_3 = k_start_9 + idx_22 * 128;
                                                if (idx_22 == 0 || token_row_3 % 256 == 0) {
                                                    bool enabled_value_24 = tile_word_2 > 0;
                                                    if (enabled_value_24 != 0) {
                                                        int poll_22_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_22_first) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                        int poll_22 = poll_22_first;
                                                        while (poll_22 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_22_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_22_next) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                            poll_22 = poll_22_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_22 == 0 || token_row_3 % mini_size == 0) {
                                                    int input_mini_3 = token_row_3 / mini_size;
                                                    int input_rows_3 = ((mini_size) < (tokens - input_mini_3 * mini_size) ? (mini_size) : (tokens - input_mini_3 * mini_size));
                                                    int input_count_3 = (input_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_25 = 1;
                                                    if (enabled_value_25 != 0) {
                                                        int poll_23_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_23_first) : "l"(dy_ready + input_mini_3) : "memory");
                                                        int poll_23 = poll_23_first;
                                                        while (poll_23 < input_count_3) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_23_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_23_next) : "l"(dy_ready + input_mini_3) : "memory");
                                                            poll_23 = poll_23_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                int _stage_gemm_finished_9 = (int)((unsigned int)(gemm_wide + idx_22) % 4);
                                                int _parity_gemm_finished_9 = (int)((unsigned int)(gemm_wide + idx_22) / 4 % 2) ^ 1;
                                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_9) * 8, _parity_gemm_finished_9);
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_22) % 4 * 16384)), "l"((&dy_t)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(k_start_9 / 128 + idx_22 - macro_k_4), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_22) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_22) % 4 * 16384)), "l"((&h_t)), "r"(0), "r"(y_9 * 512 + cta_rank_0 * 128), "r"(k_start_9 / 128 + idx_22 - macro_k_4), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_22) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(smem_v44_addr + (unsigned int)((gemm_wide + idx_22) % 4 * 16384)), "l"((&h_t)), "r"(0), "r"(y_9 * 512 + 256 + cta_rank_0 * 128), "r"(k_start_9 / 128 + idx_22 - macro_k_4), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_22) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 6) {
                                    if (warp == 6) {
                                        if (elect_sync()) {
                                            #pragma unroll 1
                                            for (int idx_23 = 0; idx_23 < iterations_9; idx_23++) {
                                                int token_row_4 = k_start_9 + idx_23 * 128;
                                                if (idx_23 == 0 || token_row_4 % 256 == 0) {
                                                    bool enabled_value_26 = tile_word_2 > 0;
                                                    if (enabled_value_26 != 0) {
                                                        int poll_24_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_24_first) : "l"(replay_h + (token_row_4 / 256)) : "memory");
                                                        int poll_24 = poll_24_first;
                                                        while (poll_24 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_24_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_24_next) : "l"(replay_h + (token_row_4 / 256)) : "memory");
                                                            poll_24 = poll_24_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_23 == 0 || token_row_4 % mini_size == 0) {
                                                    int input_mini_4 = token_row_4 / mini_size;
                                                    int input_rows_4 = ((mini_size) < (tokens - input_mini_4 * mini_size) ? (mini_size) : (tokens - input_mini_4 * mini_size));
                                                    int input_count_4 = (input_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_27 = 1;
                                                    if (enabled_value_27 != 0) {
                                                        int poll_25_first;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_25_first) : "l"(dy_ready + input_mini_4) : "memory");
                                                        int poll_25 = poll_25_first;
                                                        while (poll_25 < input_count_4) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int poll_25_next;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_25_next) : "l"(dy_ready + input_mini_4) : "memory");
                                                            poll_25 = poll_25_next;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                int _stage_scales_finished_4 = (int)((unsigned int)(scales_wide + idx_23) % 4);
                                                int _parity_scales_finished_4 = (int)((unsigned int)(scales_wide + idx_23) / 4 % 2) ^ 1;
                                                mbarrier_wait(scales_finished_addr + (_stage_scales_finished_4) * 8, _parity_scales_finished_4);
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_23) % 4 * 512)), "l"((&dy_sc_t)), "r"(0), "r"(0), "r"((x_9 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_9 / 128 + idx_23 - macro_k_4)),
                                                       "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_23) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_23) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&h_sc_t)), "r"(0), "r"(0), "r"((y_9 * 4 + cta_rank_0) * (macro_size / 128) + (k_start_9 / 128 + idx_23 - macro_k_4)),
                                                       "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_23) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(smem_v47_addr + (unsigned int)((scales_wide + idx_23) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&h_sc_t)), "r"(0), "r"(0), "r"((y_9 * 4 + 2 + cta_rank_0) * (macro_size / 128) + (k_start_9 / 128 + idx_23 - macro_k_4)),
                                                       "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_23) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int _parity_output_finished_9 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                                mbarrier_wait(output_finished_addr, _parity_output_finished_9);
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_24 = 0; idx_24 < iterations_9; idx_24++) {
                                                    mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_24) % 4)) * 8, 5120);
                                                    int _stage_scales_arrived_4 = (int)((unsigned int)(scales_wide + idx_24) % 4);
                                                    int _parity_scales_arrived_4 = (int)((unsigned int)(scales_wide + idx_24) / 4 % 2);
                                                    mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_4) * 8, _parity_scales_arrived_4);
                                                    int buffer_4 = idx_24 % 3;
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_4 * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_24) % 4 * 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_4 * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_24) % 4 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_4 * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_24) % 4 * 1024) + 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_4 * 8, make_sf_cp_desc_sbo128(smem_v47_addr + (unsigned int)((scales_wide + idx_24) % 4 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_4 * 8 + 4), make_sf_cp_desc_sbo128((smem_v47_addr + (unsigned int)((scales_wide + idx_24) % 4 * 1024) + 512)));
                                                    tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_24) % 4)) * 8, (uint16_t)(3));
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_24) % 4)) * 8, 98304);
                                                    int _stage_gemm_arrived_9 = (int)((unsigned int)(gemm_wide + idx_24) % 4);
                                                    int _parity_gemm_arrived_9 = (int)((unsigned int)(gemm_wide + idx_24) / 4 % 2);
                                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_9) * 8, _parity_gemm_arrived_9);
                                                    int _smem_desc_lo_250 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_24) % 4) * 1024;
                                                    int _smem_desc_lo_251 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_24) % 4) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)_smem_desc_lo_250) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)_smem_desc_lo_251) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                            0x90c00000U, (int)((tmem_tmem_sfa + buffer_4 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_4 * 8 + 0)), ((idx_24 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                            0xd0c00020U, (int)((tmem_tmem_sfa + buffer_4 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_4 * 8 + 0) | 0x80000000), 1);
                                                    }
                                                    int _smem_desc_lo_252 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_24) % 4) * 1024;
                                                    int _smem_desc_lo_253 = (((smem_v44_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_24) % 4) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)_smem_desc_lo_252) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)_smem_desc_lo_253) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                            0x90c00000U, (int)((tmem_tmem_sfa + buffer_4 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_4 * 8 + 0)), ((idx_24 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                            0xd0c00020U, (int)((tmem_tmem_sfa + buffer_4 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_4 * 8 + 0) | 0x80000000), 1);
                                                    }
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_24) % 4)) * 8, (uint16_t)(3));
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else if (tid < 128) {
                                        int _parity_output_arrived_9 = (int)((unsigned int)gemm_out0 % 2);
                                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_9);
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
                                            for (int vec_15 = 0; vec_15 < 4; vec_15++) {
                                                unsigned int address_29 = d_smem_addr + (unsigned int)(chunk_22 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_15 * 16);
                                                address_29 = address_29 ^ (address_29 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_29 - flat_view_addr)), "f"(_tmem_load_16[vec_15 * 4]), "f"(_tmem_load_16[vec_15 * 4 + 1]), "f"(_tmem_load_16[vec_15 * 4 + 2]), "f"(_tmem_load_16[vec_15 * 4 + 3]) : "memory");
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
                                                    :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(y_9 * 2 * 16 + chunk_22), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_22 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                for (int vec_16 = 0; vec_16 < 4; vec_16++) {
                                                    unsigned int address_30 = d_smem_addr + (unsigned int)((16 + chunk_23) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_16 * 16);
                                                    address_30 = address_30 ^ (address_30 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_30 - flat_view_addr)), "f"(_tmem_load_17[vec_16 * 4]), "f"(_tmem_load_17[vec_16 * 4 + 1]), "f"(_tmem_load_17[vec_16 * 4 + 2]), "f"(_tmem_load_17[vec_16 * 4 + 3]) : "memory");
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
                                                        :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"((y_9 * 2 + 1) * 16 + chunk_23), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_23) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                    bool enabled_value_28 = macros > 1;
                                                    if (enabled_value_28 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_7_1 = 1;
                                                        if (enabled_value_7_1 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                    if (has_hi_9 != 0) {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                gemm_wide += iterations_9;
                                scales_wide += iterations_9;
                                gemm_out0 += 1;
                                gemm_acc_tiles += 1;
                            }
                        } else {
                            if (tile_word_0 == 4) {
                                int col_blocks_15 = (macro_size + 512 - 1) / 512;
                                {
                                    col_blocks_15 = (hidden + 512 - 1) / 512;
                                }
                                int x_10 = -1;
                                int y_10 = -1;
                                int expert_10 = -1;
                                int k_start_10 = 0;
                                int k_end_10 = 0;
                                int first_10 = 0;
                                int row_blocks_6 = intermediate / 256;
                                int expert_idx_5 = 0;
                                int local_task_4 = tile_word_1;
                                uint32_t fast_div_q_9 = (routed_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), routed_wgrad_div.multiplier) >> routed_wgrad_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                                uint32_t fast_div_r_9 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_9 * (uint32_t)(routed_wgrad_div.divisor);
                                expert_idx_5 = (int)fast_div_q_9;
                                local_task_4 = (int)fast_div_r_9;
                                int offset_15 = tile_word_5;
                                int real_1 = tile_word_6;
                                real_1 = (real_1 + 128 - 1) / 128 * 128;
                                int _max_254 = ((offset_15) > (tile_word_2 * macro_size) ? (offset_15) : (tile_word_2 * macro_size));
                                k_start_10 = _max_254;
                                int _min_256 = (((tile_word_2 + 1) * macro_size) < (tokens) ? ((tile_word_2 + 1) * macro_size) : (tokens));
                                int _min_255 = ((offset_15 + real_1) < (_min_256) ? (offset_15 + real_1) : (_min_256));
                                k_end_10 = _min_255;
                                first_10 = (int)(k_start_10 == offset_15);
                                if (k_start_10 < k_end_10 && local_task_4 < row_blocks_6 * col_blocks_15) {
                                    int supergroup_10 = local_task_4 / (row_blocks_6 * 8);
                                    int full_cols_10 = col_blocks_15 / 8 * 8;
                                    int row_36 = 0;
                                    int col_79 = 0;
                                    if (local_task_4 < row_blocks_6 * full_cols_10) {
                                        row_36 = local_task_4 % (row_blocks_6 * 8) / 8;
                                        col_79 = supergroup_10 * 8 + local_task_4 % 8;
                                    } else {
                                        row_36 = (local_task_4 - row_blocks_6 * full_cols_10) / (col_blocks_15 - full_cols_10);
                                        col_79 = full_cols_10 + (local_task_4 - row_blocks_6 * full_cols_10) % (col_blocks_15 - full_cols_10);
                                    }
                                    if ((supergroup_10 & 1) != 0) {
                                        row_36 = row_blocks_6 - row_36 - 1;
                                    }
                                    x_10 = row_36;
                                    y_10 = col_79;
                                    expert_10 = expert_idx_5;
                                }
                                int has_hi_10 = 0;
                                has_hi_10 = (int)((y_10 * 2 + 1) * 256 < hidden);
                                int global_mini_12 = tile_word_2 * (macro_size / mini_size);
                                int macro_rows_10 = tile_word_2 * (macro_size / 256);
                                int iterations_10 = intermediate / 128;
                                int macro_k_5 = tile_word_2 * (macro_size / 128);
                                iterations_10 = (k_end_10 - k_start_10) / 128;
                                if (expert_10 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_29 = macros > 1;
                                        if (enabled_value_29 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 7) {
                                        if (warp == 7) {
                                            if (elect_sync()) {
                                                #pragma unroll 1
                                                for (int idx_25 = 0; idx_25 < iterations_10; idx_25++) {
                                                    int token_row_5 = k_start_10 + idx_25 * 128;
                                                    if (idx_25 == 0 || token_row_5 % 256 == 0) {
                                                        bool enabled_value_30 = 1;
                                                        if (enabled_value_30 != 0) {
                                                            int poll_26_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_26_first) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                            int poll_26 = poll_26_first;
                                                            while (poll_26 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_26_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_26_next) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                                poll_26 = poll_26_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_25 == 0 || token_row_5 % mini_size == 0) {
                                                        int input_mini_5 = token_row_5 / mini_size;
                                                        int input_rows_5 = ((mini_size) < (tokens - input_mini_5 * mini_size) ? (mini_size) : (tokens - input_mini_5 * mini_size));
                                                        int input_count_5 = (input_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_31 = tile_word_2 > 0;
                                                        if (enabled_value_31 != 0) {
                                                            int poll_27_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_27_first) : "l"(replay_x + input_mini_5) : "memory");
                                                            int poll_27 = poll_27_first;
                                                            while (poll_27 < input_count_5) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_27_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_27_next) : "l"(replay_x + input_mini_5) : "memory");
                                                                poll_27 = poll_27_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    int _stage_gemm_finished_10 = (int)((unsigned int)(gemm_wide + idx_25) % 4);
                                                    int _parity_gemm_finished_10 = (int)((unsigned int)(gemm_wide + idx_25) / 4 % 2) ^ 1;
                                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_10) * 8, _parity_gemm_finished_10);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_25) % 4 * 16384)), "l"((&dg_t)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(k_start_10 / 128 + idx_25 - macro_k_5), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_25) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_25) % 4 * 16384)), "l"((&x_t)), "r"(0), "r"(y_10 * 512 + cta_rank_0 * 128), "r"(k_start_10 / 128 + idx_25 - macro_k_5), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_25) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v44_addr + (unsigned int)((gemm_wide + idx_25) % 4 * 16384)), "l"((&x_t)), "r"(0), "r"(y_10 * 512 + 256 + cta_rank_0 * 128), "r"(k_start_10 / 128 + idx_25 - macro_k_5), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_25) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                }
                                            }
                                        }
                                    } else if (tid / 32 == 6) {
                                        if (warp == 6) {
                                            if (elect_sync()) {
                                                #pragma unroll 1
                                                for (int idx_26 = 0; idx_26 < iterations_10; idx_26++) {
                                                    int token_row_6 = k_start_10 + idx_26 * 128;
                                                    if (idx_26 == 0 || token_row_6 % 256 == 0) {
                                                        bool enabled_value_32 = 1;
                                                        if (enabled_value_32 != 0) {
                                                            int poll_28_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_28_first) : "l"(dg_ready + (shared_rows + token_row_6 / 256)) : "memory");
                                                            int poll_28 = poll_28_first;
                                                            while (poll_28 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_28_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_28_next) : "l"(dg_ready + (shared_rows + token_row_6 / 256)) : "memory");
                                                                poll_28 = poll_28_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_26 == 0 || token_row_6 % mini_size == 0) {
                                                        int input_mini_6 = token_row_6 / mini_size;
                                                        int input_rows_6 = ((mini_size) < (tokens - input_mini_6 * mini_size) ? (mini_size) : (tokens - input_mini_6 * mini_size));
                                                        int input_count_6 = (input_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_33 = tile_word_2 > 0;
                                                        if (enabled_value_33 != 0) {
                                                            int poll_29_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_29_first) : "l"(replay_x + input_mini_6) : "memory");
                                                            int poll_29 = poll_29_first;
                                                            while (poll_29 < input_count_6) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_29_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_29_next) : "l"(replay_x + input_mini_6) : "memory");
                                                                poll_29 = poll_29_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    int _stage_scales_finished_5 = (int)((unsigned int)(scales_wide + idx_26) % 4);
                                                    int _parity_scales_finished_5 = (int)((unsigned int)(scales_wide + idx_26) / 4 % 2) ^ 1;
                                                    mbarrier_wait(scales_finished_addr + (_stage_scales_finished_5) * 8, _parity_scales_finished_5);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_26) % 4 * 512)), "l"((&dg_sc_t)), "r"(0), "r"(0), "r"((x_10 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_10 / 128 + idx_26 - macro_k_5)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_26) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_26) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t)), "r"(0), "r"(0), "r"((y_10 * 4 + cta_rank_0) * (macro_size / 128) + (k_start_10 / 128 + idx_26 - macro_k_5)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_26) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v47_addr + (unsigned int)((scales_wide + idx_26) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t)), "r"(0), "r"(0), "r"((y_10 * 4 + 2 + cta_rank_0) * (macro_size / 128) + (k_start_10 / 128 + idx_26 - macro_k_5)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_26) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                }
                                            }
                                        }
                                    } else {
                                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                                            if (warp == 4) {
                                                if (elect_sync()) {
                                                    int _parity_output_finished_10 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                                    mbarrier_wait(output_finished_addr, _parity_output_finished_10);
                                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                                    #pragma unroll 1
                                                    for (int idx_27 = 0; idx_27 < iterations_10; idx_27++) {
                                                        mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_27) % 4)) * 8, 5120);
                                                        int _stage_scales_arrived_5 = (int)((unsigned int)(scales_wide + idx_27) % 4);
                                                        int _parity_scales_arrived_5 = (int)((unsigned int)(scales_wide + idx_27) / 4 % 2);
                                                        mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_5) * 8, _parity_scales_arrived_5);
                                                        int buffer_5 = idx_27 % 3;
                                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_5 * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_27) % 4 * 512)));
                                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_5 * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_27) % 4 * 1024)));
                                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_5 * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_27) % 4 * 1024) + 512)));
                                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_5 * 8, make_sf_cp_desc_sbo128(smem_v47_addr + (unsigned int)((scales_wide + idx_27) % 4 * 1024)));
                                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_5 * 8 + 4), make_sf_cp_desc_sbo128((smem_v47_addr + (unsigned int)((scales_wide + idx_27) % 4 * 1024) + 512)));
                                                        tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_27) % 4)) * 8, (uint16_t)(3));
                                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_27) % 4)) * 8, 98304);
                                                        int _stage_gemm_arrived_10 = (int)((unsigned int)(gemm_wide + idx_27) % 4);
                                                        int _parity_gemm_arrived_10 = (int)((unsigned int)(gemm_wide + idx_27) / 4 % 2);
                                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_10) * 8, _parity_gemm_arrived_10);
                                                        int _smem_desc_lo_257 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_27) % 4) * 1024;
                                                        int _smem_desc_lo_258 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_27) % 4) * 1024;
                                                        {
                                                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_257) | ((uint64_t)0x40004040 << 32);
                                                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_258) | ((uint64_t)0x40004040 << 32);

                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer_5 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_5 * 8 + 0)), ((idx_27 == 0) ? 0 : 1));
                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer_5 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_5 * 8 + 0) | 0x80000000), 1);
                                                        }
                                                        int _smem_desc_lo_259 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_27) % 4) * 1024;
                                                        int _smem_desc_lo_260 = (((smem_v44_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_27) % 4) * 1024;
                                                        {
                                                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_259) | ((uint64_t)0x40004040 << 32);
                                                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_260) | ((uint64_t)0x40004040 << 32);

                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer_5 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_5 * 8 + 0)), ((idx_27 == 0) ? 0 : 1));
                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer_5 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_5 * 8 + 0) | 0x80000000), 1);
                                                        }
                                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_27) % 4)) * 8, (uint16_t)(3));
                                                    }
                                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                                }
                                            }
                                        } else if (tid < 128) {
                                            int _parity_output_arrived_10 = (int)((unsigned int)gemm_out0 % 2);
                                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_10);
                                            int warp_row_6 = tid / 32 * 32;
                                            #pragma unroll
                                            for (int chunk_24 = 0; chunk_24 < 16; chunk_24++) {
                                                float tmem_load_5[16];
                                                tmem_ld_x16(&tmem_load_5[0], taddr_1 + (unsigned int)(warp_row_6 << 16) + (unsigned int)(chunk_24 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_17 = 0; vec_17 < 4; vec_17++) {
                                                    unsigned int address_31 = d_smem_addr + (unsigned int)(chunk_24 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_17 * 16);
                                                    address_31 = address_31 ^ (address_31 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_31 - flat_view_addr)), "f"(tmem_load_5[vec_17 * 4]), "f"(tmem_load_5[vec_17 * 4 + 1]), "f"(tmem_load_5[vec_17 * 4 + 2]), "f"(tmem_load_5[vec_17 * 4 + 3]) : "memory");
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
                                                        :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(y_10 * 2 * 16 + chunk_24), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_24 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                            if (has_hi_10 != 0) {
                                                #pragma unroll
                                                for (int chunk_25 = 0; chunk_25 < 16; chunk_25++) {
                                                    float tmem_load_1_2[16];
                                                    tmem_ld_x16(&tmem_load_1_2[0], taddr_1 + (unsigned int)(warp_row_6 << 16) + 256 + (unsigned int)(chunk_25 * 16));
                                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                    if (tid == 0) {
                                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                                    }
                                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                    #pragma unroll
                                                    for (int vec_18 = 0; vec_18 < 4; vec_18++) {
                                                        unsigned int address_32 = d_smem_addr + (unsigned int)((16 + chunk_25) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_18 * 16);
                                                        address_32 = address_32 ^ (address_32 & 511) >> 7 << 4;
                                                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_32 - flat_view_addr)), "f"(tmem_load_1_2[vec_18 * 4]), "f"(tmem_load_1_2[vec_18 * 4 + 1]), "f"(tmem_load_1_2[vec_18 * 4 + 2]), "f"(tmem_load_1_2[vec_18 * 4 + 3]) : "memory");
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
                                                            :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"((y_10 * 2 + 1) * 16 + chunk_25), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_25) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                        bool enabled_value_34 = macros > 1;
                                                        if (enabled_value_34 != 0) {
                                                            asm volatile("cp.async.bulk.wait_group 0;");
                                                            bool enabled_value_8_1 = 1;
                                                            if (enabled_value_8_1 != 0) {
                                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                            }
                                                        }
                                                        if (has_hi_10 != 0) {
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    gemm_wide += iterations_10;
                                    scales_wide += iterations_10;
                                    gemm_out0 += 1;
                                    gemm_acc_tiles += 1;
                                }
                            } else if (tile_word_0 == 5) {
                                int col_blocks_16 = (macro_size + 512 - 1) / 512;
                                {
                                    col_blocks_16 = (hidden + 512 - 1) / 512;
                                }
                                int x_11 = -1;
                                int y_11 = -1;
                                int expert_11 = -1;
                                int k_start_11 = 0;
                                int k_end_11 = 0;
                                int first_11 = 0;
                                int row_blocks_7 = intermediate / 256;
                                int expert_idx_6 = 0;
                                int local_task_5 = tile_word_1;
                                uint32_t fast_div_q_10 = (routed_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), routed_wgrad_div.multiplier) >> routed_wgrad_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                                uint32_t fast_div_r_10 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_10 * (uint32_t)(routed_wgrad_div.divisor);
                                expert_idx_6 = (int)fast_div_q_10;
                                local_task_5 = (int)fast_div_r_10;
                                int offset_16 = tile_word_5;
                                int real_2 = tile_word_6;
                                real_2 = (real_2 + 128 - 1) / 128 * 128;
                                int _max_261 = ((offset_16) > (tile_word_2 * macro_size) ? (offset_16) : (tile_word_2 * macro_size));
                                k_start_11 = _max_261;
                                int _min_263 = (((tile_word_2 + 1) * macro_size) < (tokens) ? ((tile_word_2 + 1) * macro_size) : (tokens));
                                int _min_262 = ((offset_16 + real_2) < (_min_263) ? (offset_16 + real_2) : (_min_263));
                                k_end_11 = _min_262;
                                first_11 = (int)(k_start_11 == offset_16);
                                if (k_start_11 < k_end_11 && local_task_5 < row_blocks_7 * col_blocks_16) {
                                    int supergroup_11 = local_task_5 / (row_blocks_7 * 8);
                                    int full_cols_11 = col_blocks_16 / 8 * 8;
                                    int row_37 = 0;
                                    int col_80 = 0;
                                    if (local_task_5 < row_blocks_7 * full_cols_11) {
                                        row_37 = local_task_5 % (row_blocks_7 * 8) / 8;
                                        col_80 = supergroup_11 * 8 + local_task_5 % 8;
                                    } else {
                                        row_37 = (local_task_5 - row_blocks_7 * full_cols_11) / (col_blocks_16 - full_cols_11);
                                        col_80 = full_cols_11 + (local_task_5 - row_blocks_7 * full_cols_11) % (col_blocks_16 - full_cols_11);
                                    }
                                    if ((supergroup_11 & 1) != 0) {
                                        row_37 = row_blocks_7 - row_37 - 1;
                                    }
                                    x_11 = row_37;
                                    y_11 = col_80;
                                    expert_11 = expert_idx_6;
                                }
                                int has_hi_11 = 0;
                                has_hi_11 = (int)((y_11 * 2 + 1) * 256 < hidden);
                                int global_mini_13 = tile_word_2 * (macro_size / mini_size);
                                int macro_rows_11 = tile_word_2 * (macro_size / 256);
                                int iterations_11 = intermediate / 128;
                                int macro_k_6 = tile_word_2 * (macro_size / 128);
                                iterations_11 = (k_end_11 - k_start_11) / 128;
                                if (expert_11 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_35 = macros > 1;
                                        if (enabled_value_35 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 7) {
                                        if (warp == 7) {
                                            if (elect_sync()) {
                                                #pragma unroll 1
                                                for (int idx_28 = 0; idx_28 < iterations_11; idx_28++) {
                                                    int token_row_7 = k_start_11 + idx_28 * 128;
                                                    if (idx_28 == 0 || token_row_7 % 256 == 0) {
                                                        bool enabled_value_36 = 1;
                                                        if (enabled_value_36 != 0) {
                                                            int poll_30_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_30_first) : "l"(dg_ready + (shared_rows + token_row_7 / 256)) : "memory");
                                                            int poll_30 = poll_30_first;
                                                            while (poll_30 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_30_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_30_next) : "l"(dg_ready + (shared_rows + token_row_7 / 256)) : "memory");
                                                                poll_30 = poll_30_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_28 == 0 || token_row_7 % mini_size == 0) {
                                                        int input_mini_7 = token_row_7 / mini_size;
                                                        int input_rows_7 = ((mini_size) < (tokens - input_mini_7 * mini_size) ? (mini_size) : (tokens - input_mini_7 * mini_size));
                                                        int input_count_7 = (input_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_37 = tile_word_2 > 0;
                                                        if (enabled_value_37 != 0) {
                                                            int poll_31_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_31_first) : "l"(replay_x + input_mini_7) : "memory");
                                                            int poll_31 = poll_31_first;
                                                            while (poll_31 < input_count_7) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_31_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_31_next) : "l"(replay_x + input_mini_7) : "memory");
                                                                poll_31 = poll_31_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    int _stage_gemm_finished_11 = (int)((unsigned int)(gemm_wide + idx_28) % 4);
                                                    int _parity_gemm_finished_11 = (int)((unsigned int)(gemm_wide + idx_28) / 4 % 2) ^ 1;
                                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_11) * 8, _parity_gemm_finished_11);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v42_addr + (unsigned int)((gemm_wide + idx_28) % 4 * 16384)), "l"((&du_t)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(k_start_11 / 128 + idx_28 - macro_k_6), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_28) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v43_addr + (unsigned int)((gemm_wide + idx_28) % 4 * 16384)), "l"((&x_t)), "r"(0), "r"(y_11 * 512 + cta_rank_0 * 128), "r"(k_start_11 / 128 + idx_28 - macro_k_6), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_28) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                        :: "r"(smem_v44_addr + (unsigned int)((gemm_wide + idx_28) % 4 * 16384)), "l"((&x_t)), "r"(0), "r"(y_11 * 512 + 256 + cta_rank_0 * 128), "r"(k_start_11 / 128 + idx_28 - macro_k_6), "r"(0), "r"(0),
                                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_28) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                }
                                            }
                                        }
                                    } else if (tid / 32 == 6) {
                                        if (warp == 6) {
                                            if (elect_sync()) {
                                                #pragma unroll 1
                                                for (int idx_29 = 0; idx_29 < iterations_11; idx_29++) {
                                                    int token_row_8 = k_start_11 + idx_29 * 128;
                                                    if (idx_29 == 0 || token_row_8 % 256 == 0) {
                                                        bool enabled_value_38 = 1;
                                                        if (enabled_value_38 != 0) {
                                                            int poll_32_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_32_first) : "l"(dg_ready + (shared_rows + token_row_8 / 256)) : "memory");
                                                            int poll_32 = poll_32_first;
                                                            while (poll_32 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_32_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_32_next) : "l"(dg_ready + (shared_rows + token_row_8 / 256)) : "memory");
                                                                poll_32 = poll_32_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_29 == 0 || token_row_8 % mini_size == 0) {
                                                        int input_mini_8 = token_row_8 / mini_size;
                                                        int input_rows_8 = ((mini_size) < (tokens - input_mini_8 * mini_size) ? (mini_size) : (tokens - input_mini_8 * mini_size));
                                                        int input_count_8 = (input_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_39 = tile_word_2 > 0;
                                                        if (enabled_value_39 != 0) {
                                                            int poll_33_first;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_33_first) : "l"(replay_x + input_mini_8) : "memory");
                                                            int poll_33 = poll_33_first;
                                                            while (poll_33 < input_count_8) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int poll_33_next;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_33_next) : "l"(replay_x + input_mini_8) : "memory");
                                                                poll_33 = poll_33_next;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    int _stage_scales_finished_6 = (int)((unsigned int)(scales_wide + idx_29) % 4);
                                                    int _parity_scales_finished_6 = (int)((unsigned int)(scales_wide + idx_29) / 4 % 2) ^ 1;
                                                    mbarrier_wait(scales_finished_addr + (_stage_scales_finished_6) * 8, _parity_scales_finished_6);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v45_addr + (unsigned int)((scales_wide + idx_29) % 4 * 512)), "l"((&du_sc_t)), "r"(0), "r"(0), "r"((x_11 * 2 + cta_rank_0) * (macro_size / 128) + (k_start_11 / 128 + idx_29 - macro_k_6)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_29) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v46_addr + (unsigned int)((scales_wide + idx_29) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t)), "r"(0), "r"(0), "r"((y_11 * 4 + cta_rank_0) * (macro_size / 128) + (k_start_11 / 128 + idx_29 - macro_k_6)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_29) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(smem_v47_addr + (unsigned int)((scales_wide + idx_29) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t)), "r"(0), "r"(0), "r"((y_11 * 4 + 2 + cta_rank_0) * (macro_size / 128) + (k_start_11 / 128 + idx_29 - macro_k_6)),
                                                           "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_29) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                }
                                            }
                                        }
                                    } else {
                                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                                            if (warp == 4) {
                                                if (elect_sync()) {
                                                    int _parity_output_finished_11 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                                    mbarrier_wait(output_finished_addr, _parity_output_finished_11);
                                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                                    #pragma unroll 1
                                                    for (int idx_30 = 0; idx_30 < iterations_11; idx_30++) {
                                                        mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_30) % 4)) * 8, 5120);
                                                        int _stage_scales_arrived_6 = (int)((unsigned int)(scales_wide + idx_30) % 4);
                                                        int _parity_scales_arrived_6 = (int)((unsigned int)(scales_wide + idx_30) / 4 % 2);
                                                        mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_6) * 8, _parity_scales_arrived_6);
                                                        int buffer_6 = idx_30 % 3;
                                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_6 * 4, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)((scales_wide + idx_30) % 4 * 512)));
                                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_6 * 8, make_sf_cp_desc_sbo128(smem_v46_addr + (unsigned int)((scales_wide + idx_30) % 4 * 1024)));
                                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_6 * 8 + 4), make_sf_cp_desc_sbo128((smem_v46_addr + (unsigned int)((scales_wide + idx_30) % 4 * 1024) + 512)));
                                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_6 * 8, make_sf_cp_desc_sbo128(smem_v47_addr + (unsigned int)((scales_wide + idx_30) % 4 * 1024)));
                                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_6 * 8 + 4), make_sf_cp_desc_sbo128((smem_v47_addr + (unsigned int)((scales_wide + idx_30) % 4 * 1024) + 512)));
                                                        tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_30) % 4)) * 8, (uint16_t)(3));
                                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_30) % 4)) * 8, 98304);
                                                        int _stage_gemm_arrived_11 = (int)((unsigned int)(gemm_wide + idx_30) % 4);
                                                        int _parity_gemm_arrived_11 = (int)((unsigned int)(gemm_wide + idx_30) / 4 % 2);
                                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_11) * 8, _parity_gemm_arrived_11);
                                                        int _smem_desc_lo_264 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_30) % 4) * 1024;
                                                        int _smem_desc_lo_265 = (((smem_v43_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_30) % 4) * 1024;
                                                        {
                                                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_264) | ((uint64_t)0x40004040 << 32);
                                                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_265) | ((uint64_t)0x40004040 << 32);

                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer_6 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_6 * 8 + 0)), ((idx_30 == 0) ? 0 : 1));
                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer_6 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_6 * 8 + 0) | 0x80000000), 1);
                                                        }
                                                        int _smem_desc_lo_266 = (((smem_v42_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_30) % 4) * 1024;
                                                        int _smem_desc_lo_267 = (((smem_v44_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_30) % 4) * 1024;
                                                        {
                                                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_266) | ((uint64_t)0x40004040 << 32);
                                                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_267) | ((uint64_t)0x40004040 << 32);

                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer_6 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_6 * 8 + 0)), ((idx_30 == 0) ? 0 : 1));
                                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer_6 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_6 * 8 + 0) | 0x80000000), 1);
                                                        }
                                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_30) % 4)) * 8, (uint16_t)(3));
                                                    }
                                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                                }
                                            }
                                        } else if (tid < 128) {
                                            int _parity_output_arrived_11 = (int)((unsigned int)gemm_out0 % 2);
                                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_11);
                                            int warp_row_7 = tid / 32 * 32;
                                            #pragma unroll
                                            for (int chunk_26 = 0; chunk_26 < 16; chunk_26++) {
                                                float tmem_load_6[16];
                                                tmem_ld_x16(&tmem_load_6[0], taddr_1 + (unsigned int)(warp_row_7 << 16) + (unsigned int)(chunk_26 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_19 = 0; vec_19 < 4; vec_19++) {
                                                    unsigned int address_33 = d_smem_addr + (unsigned int)(chunk_26 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_19 * 16);
                                                    address_33 = address_33 ^ (address_33 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_33 - flat_view_addr)), "f"(tmem_load_6[vec_19 * 4]), "f"(tmem_load_6[vec_19 * 4 + 1]), "f"(tmem_load_6[vec_19 * 4 + 2]), "f"(tmem_load_6[vec_19 * 4 + 3]) : "memory");
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
                                                        :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(y_11 * 2 * 16 + chunk_26), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_26 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    asm volatile("cp.async.bulk.commit_group;");
                                                }
                                            }
                                            if (has_hi_11 != 0) {
                                                #pragma unroll
                                                for (int chunk_27 = 0; chunk_27 < 16; chunk_27++) {
                                                    float tmem_load_1_3[16];
                                                    tmem_ld_x16(&tmem_load_1_3[0], taddr_1 + (unsigned int)(warp_row_7 << 16) + 256 + (unsigned int)(chunk_27 * 16));
                                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                    if (tid == 0) {
                                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                                    }
                                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                    #pragma unroll
                                                    for (int vec_20 = 0; vec_20 < 4; vec_20++) {
                                                        unsigned int address_34 = d_smem_addr + (unsigned int)((16 + chunk_27) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_20 * 16);
                                                        address_34 = address_34 ^ (address_34 & 511) >> 7 << 4;
                                                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(flat_view_addr + (address_34 - flat_view_addr)), "f"(tmem_load_1_3[vec_20 * 4]), "f"(tmem_load_1_3[vec_20 * 4 + 1]), "f"(tmem_load_1_3[vec_20 * 4 + 2]), "f"(tmem_load_1_3[vec_20 * 4 + 3]) : "memory");
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
                                                            :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"((y_11 * 2 + 1) * 16 + chunk_27), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_27) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                        bool enabled_value_40 = macros > 1;
                                                        if (enabled_value_40 != 0) {
                                                            asm volatile("cp.async.bulk.wait_group 0;");
                                                            bool enabled_value_9_1 = 1;
                                                            if (enabled_value_9_1 != 0) {
                                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (tile_word_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                            }
                                                        }
                                                        if (has_hi_11 != 0) {
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    gemm_wide += iterations_11;
                                    scales_wide += iterations_11;
                                    gemm_out0 += 1;
                                    gemm_acc_tiles += 1;
                                }
                            }
                        }
                    }
                }
            }
            if (tid / 32 == 5) {
                if (warp == 5) {
                    if (elect_sync()) {
                        int _parity_schedule_arrived = (int)((unsigned int)iteration % 2);
                        mbarrier_wait(schedule_arrived_addr, _parity_schedule_arrived);
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
                        int decoded_cluster = -1;
                        if (_clc_valid_0 != 0) {
                            decoded_cluster = (int)(_clc_ctaid_x_0 / 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        }
                        int kind_1 = -1;
                        int task_2_1 = 0;
                        int macro_3 = 0;
                        int mini_4 = 0;
                        int shared_5 = 0;
                        if (decoded_cluster - comm_clusters >= 0 && true_compute > decoded_cluster - comm_clusters) {
                            if (shared_tasks > decoded_cluster - comm_clusters) {
                                shared_5 = 1;
                                if (shared_down > decoded_cluster - comm_clusters) {
                                    kind_1 = 0;
                                    task_2_1 = decoded_cluster - comm_clusters;
                                } else if (decoded_cluster - comm_clusters < shared_down + shared_swiglu) {
                                    kind_1 = 1;
                                    task_2_1 = decoded_cluster - comm_clusters - shared_down;
                                } else {
                                    if (decoded_cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                                        kind_1 = 2;
                                        task_2_1 = decoded_cluster - comm_clusters - shared_down - shared_swiglu;
                                    } else {
                                        int weight_task_2 = decoded_cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                                        uint32_t _fast_div_q_17 = (shared_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task_2), shared_wgrad_div.multiplier) >> shared_wgrad_div.shift_right) : (uint32_t)((unsigned int)weight_task_2);
                                        uint32_t _fast_div_r_17 = (uint32_t)((unsigned int)weight_task_2) - _fast_div_q_17 * (uint32_t)(shared_wgrad_div.divisor);
                                        kind_1 = (int)_fast_div_q_17 + 3;
                                        task_2_1 = (int)_fast_div_r_17;
                                    }
                                }
                            } else {
                                int routed_1 = decoded_cluster - comm_clusters - shared_tasks;
                                int macro_task_1 = routed_1;
                                int replay_tasks_1 = 0;
                                if (routed_1 >= saved_tasks) {
                                    uint32_t _fast_div_q_18 = (replay_macro_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(routed_1 - saved_tasks)), replay_macro_tasks_div.multiplier) >> replay_macro_tasks_div.shift_right) : (uint32_t)((unsigned int)(routed_1 - saved_tasks));
                                    uint32_t _fast_div_r_18 = (uint32_t)((unsigned int)(routed_1 - saved_tasks)) - _fast_div_q_18 * (uint32_t)(replay_macro_tasks_div.divisor);
                                    macro_3 = (int)_fast_div_q_18 + 1;
                                    macro_task_1 = (int)_fast_div_r_18;
                                }
                                int _min_268 = ((tokens - macro_3 * macro_size) < (macro_size) ? (tokens - macro_3 * macro_size) : (macro_size));
                                uint32_t _fast_div_q_19 = (mini_size_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(_min_268 + mini_size - 1)), mini_size_div.multiplier) >> mini_size_div.shift_right) : (uint32_t)((unsigned int)(_min_268 + mini_size - 1));
                                uint32_t _fast_div_r_19 = (uint32_t)((unsigned int)(_min_268 + mini_size - 1)) - _fast_div_q_19 * (uint32_t)(mini_size_div.divisor);
                                int macro_minis_1 = (int)_fast_div_q_19;
                                if (routed_1 >= saved_tasks) {
                                    replay_tasks_1 = macro_minis_1 * mini_replay;
                                }
                                if (macro_task_1 < replay_tasks_1) {
                                    uint32_t _fast_div_q_20 = (mini_replay_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)macro_task_1), mini_replay_div.multiplier) >> mini_replay_div.shift_right) : (uint32_t)((unsigned int)macro_task_1);
                                    uint32_t _fast_div_r_20 = (uint32_t)((unsigned int)macro_task_1) - _fast_div_q_20 * (uint32_t)(mini_replay_div.divisor);
                                    mini_4 = (int)_fast_div_q_20;
                                    int mini_task_2 = (int)_fast_div_r_20;
                                    if (mini_task_2 < mini_replay_gate) {
                                        kind_1 = 6;
                                        task_2_1 = mini_task_2;
                                    } else if (mini_task_2 < 2 * mini_replay_gate) {
                                        kind_1 = 7;
                                        task_2_1 = mini_task_2 - mini_replay_gate;
                                    } else {
                                        kind_1 = 8;
                                        task_2_1 = mini_task_2 - 2 * mini_replay_gate;
                                    }
                                } else {
                                    int bwd_task_1 = macro_task_1 - replay_tasks_1;
                                    if (bwd_task_1 < macro_minis_1 * mini_bwd) {
                                        uint32_t _fast_div_q_21 = (mini_bwd_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)bwd_task_1), mini_bwd_div.multiplier) >> mini_bwd_div.shift_right) : (uint32_t)((unsigned int)bwd_task_1);
                                        uint32_t _fast_div_r_21 = (uint32_t)((unsigned int)bwd_task_1) - _fast_div_q_21 * (uint32_t)(mini_bwd_div.divisor);
                                        mini_4 = (int)_fast_div_q_21;
                                        int mini_task_3 = (int)_fast_div_r_21;
                                        if (mini_task_3 < mini_down) {
                                            kind_1 = 0;
                                            task_2_1 = mini_task_3;
                                        } else if (mini_task_3 < mini_down + mini_swiglu) {
                                            kind_1 = 1;
                                            task_2_1 = mini_task_3 - mini_down;
                                        } else {
                                            kind_1 = 2;
                                            task_2_1 = mini_task_3 - mini_down - mini_swiglu;
                                        }
                                    } else {
                                        int weight_task_3 = bwd_task_1 - macro_minis_1 * mini_bwd;
                                        uint32_t _fast_div_q_22 = (weight_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)weight_task_3), weight_tasks_div.multiplier) >> weight_tasks_div.shift_right) : (uint32_t)((unsigned int)weight_task_3);
                                        uint32_t _fast_div_r_22 = (uint32_t)((unsigned int)weight_task_3) - _fast_div_q_22 * (uint32_t)(weight_tasks_div.divisor);
                                        kind_1 = (int)_fast_div_q_22 + 3;
                                        task_2_1 = (int)_fast_div_r_22;
                                    }
                                }
                            }
                        }
                        int word0_6 = 0;
                        int word1_7 = 0;
                        int word2_8 = 0;
                        if (shared_5 == 0) {
                            if (kind_1 == 0) {
                                int word0_0_4 = 0;
                                int word1_1_4 = 0;
                                int word2_2_4 = 0;
                                int first_block_7 = (macro_3 * (macro_size / mini_size) + mini_4) * (mini_size / 256);
                                int end_block_7 = ((first_block_7 + mini_size / 256) < (tokens / 256) ? (first_block_7 + mini_size / 256) : (tokens / 256));
                                uint32_t fast_div_q_11 = (dh_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2_1), dh_col_blocks_div.multiplier) >> dh_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_2_1);
                                uint32_t fast_div_r_11 = (uint32_t)((unsigned int)task_2_1) - fast_div_q_11 * (uint32_t)(dh_col_blocks_div.divisor);
                                int block_18 = first_block_7 + (int)fast_div_q_11;
                                if (block_18 < end_block_7) {
                                    word0_0_4 = counts[3 * experts + block_18];
                                    word1_1_4 = counts[experts + word0_0_4];
                                    word2_2_4 = counts[word0_0_4];
                                }
                                word0_6 = word0_0_4;
                                word1_7 = word1_1_4;
                                word2_8 = word2_2_4;
                            } else if (kind_1 == 2) {
                                int word0_0_5 = 0;
                                int word1_1_5 = 0;
                                int word2_2_5 = 0;
                                int first_block_8 = (macro_3 * (macro_size / mini_size) + mini_4) * (mini_size / 256);
                                int end_block_8 = ((first_block_8 + mini_size / 256) < (tokens / 256) ? (first_block_8 + mini_size / 256) : (tokens / 256));
                                uint32_t fast_div_q_12 = (dx_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2_1), dx_col_blocks_div.multiplier) >> dx_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_2_1);
                                uint32_t fast_div_r_12 = (uint32_t)((unsigned int)task_2_1) - fast_div_q_12 * (uint32_t)(dx_col_blocks_div.divisor);
                                int block_19 = first_block_8 + (int)fast_div_q_12;
                                if (block_19 < end_block_8) {
                                    word0_0_5 = counts[3 * experts + block_19];
                                    word1_1_5 = counts[experts + word0_0_5];
                                    word2_2_5 = counts[word0_0_5];
                                }
                                word0_6 = word0_0_5;
                                word1_7 = word1_1_5;
                                word2_8 = word2_2_5;
                            } else {
                                if (kind_1 == 6 || kind_1 == 7) {
                                    int word0_0_6 = 0;
                                    int word1_1_6 = 0;
                                    int word2_2_6 = 0;
                                    int first_block_9 = (macro_3 * (macro_size / mini_size) + mini_4) * (mini_size / 256);
                                    int end_block_9 = ((first_block_9 + mini_size / 256) < (tokens / 256) ? (first_block_9 + mini_size / 256) : (tokens / 256));
                                    uint32_t fast_div_q_13 = (replay_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2_1), replay_col_blocks_div.multiplier) >> replay_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_2_1);
                                    uint32_t fast_div_r_13 = (uint32_t)((unsigned int)task_2_1) - fast_div_q_13 * (uint32_t)(replay_col_blocks_div.divisor);
                                    int block_20 = first_block_9 + (int)fast_div_q_13;
                                    if (block_20 < end_block_9) {
                                        word0_0_6 = counts[3 * experts + block_20];
                                        word1_1_6 = counts[experts + word0_0_6];
                                        word2_2_6 = counts[word0_0_6];
                                    }
                                    word0_6 = word0_0_6;
                                    word1_7 = word1_1_6;
                                    word2_8 = word2_2_6;
                                } else if (kind_1 >= 3 && kind_1 <= 5) {
                                    int word0_0_7 = 0;
                                    int word1_1_7 = 0;
                                    int word2_2_7 = 0;
                                    uint32_t fast_div_q_14 = (routed_wgrad_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_2_1), routed_wgrad_div.multiplier) >> routed_wgrad_div.shift_right) : (uint32_t)((unsigned int)task_2_1);
                                    uint32_t fast_div_r_14 = (uint32_t)((unsigned int)task_2_1) - fast_div_q_14 * (uint32_t)(routed_wgrad_div.divisor);
                                    int expert_idx_7 = (int)fast_div_q_14;
                                    word0_0_7 = counts[experts + expert_idx_7];
                                    word1_1_7 = counts[2 * experts + expert_idx_7];
                                    word0_6 = word0_0_7;
                                    word1_7 = word1_1_7;
                                    word2_8 = word2_2_7;
                                }
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_0_addr), "r"((unsigned int)decoded_cluster), "r"((unsigned int)kind_1), "r"((unsigned int)task_2_1), "r"((unsigned int)macro_3) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_1_addr), "r"((unsigned int)mini_4), "r"((unsigned int)shared_5), "r"((unsigned int)word0_6), "r"((unsigned int)word1_7) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_2_addr), "r"((unsigned int)word2_8), "r"(0), "r"(0), "r"(0) : "memory");
                        mbarrier_arrive(decode_ready_addr);
                    }
                }
            }
            int _parity_decode_ready = (int)((unsigned int)iteration % 2);
            mbarrier_wait(decode_ready_addr, _parity_decode_ready);
            unsigned int decode_quad[4];
            __int128_t _smem_raw_269;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_269) : "r"(decode_words_0_addr) : "memory");
            decode_quad[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_269)[0];
            decode_quad[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_269)[1];
            decode_quad[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_269)[2];
            decode_quad[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_269)[3];
            unsigned int decode_quad_2[4];
            __int128_t _smem_raw_270;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_270) : "r"(decode_words_1_addr) : "memory");
            decode_quad_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_270)[0];
            decode_quad_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_270)[1];
            decode_quad_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_270)[2];
            decode_quad_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_270)[3];
            unsigned int decode_quad_3[4];
            __int128_t _smem_raw_271;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_271) : "r"(decode_words_2_addr) : "memory");
            decode_quad_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_271)[0];
            decode_quad_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_271)[1];
            decode_quad_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_271)[2];
            decode_quad_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_271)[3];
            cluster = (int)decode_quad[0];
            int tile_word_8 = (int)decode_quad[1];
            int tile_word_9 = (int)decode_quad[2];
            int tile_word_10 = (int)decode_quad[3];
            int tile_word_11 = (int)decode_quad_2[0];
            int tile_word_12 = (int)decode_quad_2[1];
            __syncwarp();
            bool _elect_sync_0 = elect_sync();
            if (_elect_sync_0) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((schedule_finished_addr) & 0xFEFFFFFF) : "memory");
            }
            if ((tile_word_0 == 1 || tile_word_0 == 8) && cluster >= 0 && tile_word_8 != 1 && tile_word_8 != 8) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            tile_word_0 = tile_word_8;
            tile_word_1 = tile_word_9;
            tile_word_2 = tile_word_10;
            tile_word_3 = tile_word_11;
            tile_word_4 = tile_word_12;
            tile_word_5 = (int)decode_quad_2[2];
            tile_word_6 = (int)decode_quad_2[3];
            tile_word_7 = (int)decode_quad_3[0];
            iteration += 1;
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (cluster >= 0 && tid == 0) {
            int drain_iteration = 0;
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
                int _parity_drain_arrived_0 = (int)((unsigned int)drain_iteration % 2);
                mbarrier_wait(drain_arrived_0_addr, _parity_drain_arrived_0);
                uint32_t clc_valid = 0;
                uint32_t clc_ctaid_x;
                uint32_t clc_ctaid_y;
                uint32_t clc_ctaid_z;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x), "=r"(clc_ctaid_y), "=r"(clc_ctaid_z), "=r"(clc_valid)
                    : "r"(smem + 528 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_0_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_0 = (int)((unsigned int)drain_iteration % 2);
                    mbarrier_wait(drain_finished_0_addr, _parity_drain_finished_0);
                }
                if (clc_valid == 0) {
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
                drain_iteration += 1;
            }
        }
        if (cluster >= 0 && tid == 32) {
            int drain_iteration_1 = 0;
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
                int _parity_drain_arrived_1 = (int)((unsigned int)drain_iteration_1 % 2);
                mbarrier_wait(drain_arrived_1_addr, _parity_drain_arrived_1);
                uint32_t clc_valid_1 = 0;
                uint32_t clc_ctaid_x_1;
                uint32_t clc_ctaid_y_1;
                uint32_t clc_ctaid_z_1;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_1), "=r"(clc_ctaid_y_1), "=r"(clc_ctaid_z_1), "=r"(clc_valid_1)
                    : "r"(smem + 544 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_1 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_1_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_1 = (int)((unsigned int)drain_iteration_1 % 2);
                    mbarrier_wait(drain_finished_1_addr, _parity_drain_finished_1);
                }
                if (clc_valid_1 == 0) {
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
                drain_iteration_1 += 1;
            }
        }
        if (cluster >= 0 && tid == 64) {
            int drain_iteration_2 = 0;
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
                int _parity_drain_arrived_2 = (int)((unsigned int)drain_iteration_2 % 2);
                mbarrier_wait(drain_arrived_2_addr, _parity_drain_arrived_2);
                uint32_t clc_valid_2 = 0;
                uint32_t clc_ctaid_x_2;
                uint32_t clc_ctaid_y_2;
                uint32_t clc_ctaid_z_2;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_2), "=r"(clc_ctaid_y_2), "=r"(clc_ctaid_z_2), "=r"(clc_valid_2)
                    : "r"(smem + 560 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_2 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_2_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_2 = (int)((unsigned int)drain_iteration_2 % 2);
                    mbarrier_wait(drain_finished_2_addr, _parity_drain_finished_2);
                }
                if (clc_valid_2 == 0) {
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
                drain_iteration_2 += 1;
            }
        }
        if (cluster >= 0 && tid == 96) {
            int drain_iteration_3 = 0;
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
                int _parity_drain_arrived_3 = (int)((unsigned int)drain_iteration_3 % 2);
                mbarrier_wait(drain_arrived_3_addr, _parity_drain_arrived_3);
                uint32_t clc_valid_3 = 0;
                uint32_t clc_ctaid_x_3;
                uint32_t clc_ctaid_y_3;
                uint32_t clc_ctaid_z_3;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_3), "=r"(clc_ctaid_y_3), "=r"(clc_ctaid_z_3), "=r"(clc_valid_3)
                    : "r"(smem + 576 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_3 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_3_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_3 = (int)((unsigned int)drain_iteration_3 % 2);
                    mbarrier_wait(drain_finished_3_addr, _parity_drain_finished_3);
                }
                if (clc_valid_3 == 0) {
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
                drain_iteration_3 += 1;
            }
        }
        if (cluster >= 0 && tid == 128) {
            int drain_iteration_4 = 0;
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
                int _parity_drain_arrived_4 = (int)((unsigned int)drain_iteration_4 % 2);
                mbarrier_wait(drain_arrived_4_addr, _parity_drain_arrived_4);
                uint32_t clc_valid_4 = 0;
                uint32_t clc_ctaid_x_4;
                uint32_t clc_ctaid_y_4;
                uint32_t clc_ctaid_z_4;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_4), "=r"(clc_ctaid_y_4), "=r"(clc_ctaid_z_4), "=r"(clc_valid_4)
                    : "r"(smem + 592 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_4 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_4_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_4 = (int)((unsigned int)drain_iteration_4 % 2);
                    mbarrier_wait(drain_finished_4_addr, _parity_drain_finished_4);
                }
                if (clc_valid_4 == 0) {
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
                drain_iteration_4 += 1;
            }
        }
        if (cluster >= 0 && tid == 160) {
            int drain_iteration_5 = 0;
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
                int _parity_drain_arrived_5 = (int)((unsigned int)drain_iteration_5 % 2);
                mbarrier_wait(drain_arrived_5_addr, _parity_drain_arrived_5);
                uint32_t clc_valid_5 = 0;
                uint32_t clc_ctaid_x_5;
                uint32_t clc_ctaid_y_5;
                uint32_t clc_ctaid_z_5;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_5), "=r"(clc_ctaid_y_5), "=r"(clc_ctaid_z_5), "=r"(clc_valid_5)
                    : "r"(smem + 608 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_5 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_5_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_5 = (int)((unsigned int)drain_iteration_5 % 2);
                    mbarrier_wait(drain_finished_5_addr, _parity_drain_finished_5);
                }
                if (clc_valid_5 == 0) {
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
                drain_iteration_5 += 1;
            }
        }
        if (cluster >= 0 && tid == 192) {
            int drain_iteration_6 = 0;
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
                int _parity_drain_arrived_6 = (int)((unsigned int)drain_iteration_6 % 2);
                mbarrier_wait(drain_arrived_6_addr, _parity_drain_arrived_6);
                uint32_t clc_valid_6 = 0;
                uint32_t clc_ctaid_x_6;
                uint32_t clc_ctaid_y_6;
                uint32_t clc_ctaid_z_6;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_6), "=r"(clc_ctaid_y_6), "=r"(clc_ctaid_z_6), "=r"(clc_valid_6)
                    : "r"(smem + 624 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_6 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_6_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_6 = (int)((unsigned int)drain_iteration_6 % 2);
                    mbarrier_wait(drain_finished_6_addr, _parity_drain_finished_6);
                }
                if (clc_valid_6 == 0) {
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
                drain_iteration_6 += 1;
            }
        }
        if (cluster >= 0 && tid == 224) {
            int drain_iteration_7 = 0;
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
                int _parity_drain_arrived_7 = (int)((unsigned int)drain_iteration_7 % 2);
                mbarrier_wait(drain_arrived_7_addr, _parity_drain_arrived_7);
                uint32_t clc_valid_7 = 0;
                uint32_t clc_ctaid_x_7;
                uint32_t clc_ctaid_y_7;
                uint32_t clc_ctaid_z_7;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(clc_ctaid_x_7), "=r"(clc_ctaid_y_7), "=r"(clc_ctaid_z_7), "=r"(clc_valid_7)
                    : "r"(smem + 640 + 0 * 16 + 0 * 16)
                    : "memory");
                if (clc_valid_7 != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((drain_finished_7_addr) & 0xFEFFFFFF) : "memory");
                if (cta_rank_0 == 0) {
                    int _parity_drain_finished_7 = (int)((unsigned int)drain_iteration_7 % 2);
                    mbarrier_wait(drain_finished_7_addr, _parity_drain_finished_7);
                }
                if (clc_valid_7 == 0) {
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
                drain_iteration_7 += 1;
            }
        }
    }

    // ---- Role: warp_role_0 ----
    if (warp == 0) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_1 ----
    if (warp == 1) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_2 ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_3 ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_4 ----
    if (warp == 4) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_5 ----
    if (warp == 5) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_6 ----
    if (warp == 6) {
        // idle — no tasks assigned
    }
    // ---- Role: warp_role_7 ----
    if (warp == 7) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(576));
    }
}

} // extern "C"
