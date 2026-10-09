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
#define SMEM_A_SMEM_OFF 1024
#define SMEM_A_SMEM_STAGE_BYTES 16384
#define SMEM_A_SMEM_STRIDE 16384
#define SMEM_B_SMEM_OFF 66560
#define SMEM_B_SMEM_STAGE_BYTES 16384
#define SMEM_B_SMEM_STRIDE 16384
#define SMEM_B_HI_OFF 132096
#define SMEM_B_HI_STAGE_BYTES 16384
#define SMEM_B_HI_STRIDE 16384
#define SMEM_D_SMEM_OFF 207872
#define SMEM_D_SMEM_STAGE_BYTES 8192
#define SMEM_D_SMEM_STRIDE 8192
#define SMEM_GATE_SMEM_OFF 1024
#define SMEM_GATE_SMEM_STAGE_BYTES 32768
#define SMEM_GATE_SMEM_STRIDE 32768
#define SMEM_UP_SMEM_OFF 99328
#define SMEM_UP_SMEM_STAGE_BYTES 32768
#define SMEM_UP_SMEM_STRIDE 32768
#define SMEM_HIDDEN_STAGING_OFF 197632
#define SMEM_HIDDEN_STAGING_STAGE_BYTES 34816
#define SMEM_HIDDEN_STAGING_STRIDE 34816
#define SMEM_HIDDEN_WORDS_OFF 197632
#define SMEM_HIDDEN_WORDS_STAGE_BYTES 34816
#define SMEM_HIDDEN_WORDS_STRIDE 34816
#define SMEM_SMEM_V8_OFF 1024
#define SMEM_SMEM_V8_STAGE_BYTES 16384
#define SMEM_SMEM_V8_STRIDE 16384
#define SMEM_SMEM_V9_OFF 17408
#define SMEM_SMEM_V9_STAGE_BYTES 512
#define SMEM_SMEM_V9_STRIDE 512
#define SMEM_SMEM_V10_OFF 99328
#define SMEM_SMEM_V10_STAGE_BYTES 16384
#define SMEM_SMEM_V10_STRIDE 16384
#define SMEM_SMEM_V11_OFF 115712
#define SMEM_SMEM_V11_STAGE_BYTES 512
#define SMEM_SMEM_V11_STRIDE 512
#define SMEM_SMEM_V12_OFF 33792
#define SMEM_SMEM_V12_STAGE_BYTES 16384
#define SMEM_SMEM_V12_STRIDE 16384
#define SMEM_SMEM_V13_OFF 50176
#define SMEM_SMEM_V13_STAGE_BYTES 512
#define SMEM_SMEM_V13_STRIDE 512
#define SMEM_SMEM_V14_OFF 132096
#define SMEM_SMEM_V14_STAGE_BYTES 16384
#define SMEM_SMEM_V14_STRIDE 16384
#define SMEM_SMEM_V15_OFF 148480
#define SMEM_SMEM_V15_STAGE_BYTES 512
#define SMEM_SMEM_V15_STRIDE 512
#define SMEM_SMEM_V16_OFF 66560
#define SMEM_SMEM_V16_STAGE_BYTES 16384
#define SMEM_SMEM_V16_STRIDE 16384
#define SMEM_SMEM_V17_OFF 82944
#define SMEM_SMEM_V17_STAGE_BYTES 512
#define SMEM_SMEM_V17_STRIDE 512
#define SMEM_SMEM_V18_OFF 164864
#define SMEM_SMEM_V18_STAGE_BYTES 16384
#define SMEM_SMEM_V18_STRIDE 16384
#define SMEM_SMEM_V19_OFF 181248
#define SMEM_SMEM_V19_STAGE_BYTES 512
#define SMEM_SMEM_V19_STRIDE 512
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 131072
#define SMEM_DISPATCH_WORDS_STRIDE 131072
#define SMEM_DISPATCH_WEIGHTS_OFF 199680
#define SMEM_DISPATCH_WEIGHTS_STAGE_BYTES 512
#define SMEM_DISPATCH_WEIGHTS_STRIDE 512
#define SMEM_SMEM_V22_OFF 1024
#define SMEM_SMEM_V22_STAGE_BYTES 131072
#define SMEM_SMEM_V22_STRIDE 131072
#define SMEM_SMEM_V23_OFF 1280
#define SMEM_SMEM_V23_STAGE_BYTES 130816
#define SMEM_SMEM_V23_STRIDE 130816
#define SMEM_SMEM_V24_OFF 1536
#define SMEM_SMEM_V24_STAGE_BYTES 130560
#define SMEM_SMEM_V24_STRIDE 130560
#define SMEM_SMEM_V25_OFF 1792
#define SMEM_SMEM_V25_STAGE_BYTES 130304
#define SMEM_SMEM_V25_STRIDE 130304
#define SMEM_SMEM_V26_OFF 132096
#define SMEM_SMEM_V26_STAGE_BYTES 16384
#define SMEM_SMEM_V26_STRIDE 16384
#define SMEM_SMEM_V27_OFF 148480
#define SMEM_SMEM_V27_STAGE_BYTES 16384
#define SMEM_SMEM_V27_STRIDE 16384
#define SMEM_SMEM_V28_OFF 164864
#define SMEM_SMEM_V28_STAGE_BYTES 512
#define SMEM_SMEM_V28_STRIDE 512
#define SMEM_SMEM_V29_OFF 165376
#define SMEM_SMEM_V29_STAGE_BYTES 512
#define SMEM_SMEM_V29_STRIDE 512
#define SMEM_SMEM_V30_OFF 165888
#define SMEM_SMEM_V30_STAGE_BYTES 16384
#define SMEM_SMEM_V30_STRIDE 16384
#define SMEM_SMEM_V31_OFF 182272
#define SMEM_SMEM_V31_STAGE_BYTES 16384
#define SMEM_SMEM_V31_STRIDE 16384
#define SMEM_SMEM_V32_OFF 198656
#define SMEM_SMEM_V32_STAGE_BYTES 512
#define SMEM_SMEM_V32_STRIDE 512
#define SMEM_SMEM_V33_OFF 199168
#define SMEM_SMEM_V33_STAGE_BYTES 512
#define SMEM_SMEM_V33_STRIDE 512
#define SMEM_SMEM_V34_OFF 1024
#define SMEM_SMEM_V34_STAGE_BYTES 65536
#define SMEM_SMEM_V34_STRIDE 65536
#define SMEM_SMEM_V35_OFF 1280
#define SMEM_SMEM_V35_STAGE_BYTES 65280
#define SMEM_SMEM_V35_STRIDE 65280
#define SMEM_SMEM_V36_OFF 66560
#define SMEM_SMEM_V36_STAGE_BYTES 65536
#define SMEM_SMEM_V36_STRIDE 65536
#define SMEM_SMEM_V37_OFF 66816
#define SMEM_SMEM_V37_STAGE_BYTES 65280
#define SMEM_SMEM_V37_STRIDE 65280
#define SMEM_COMBINE_SMEM_OFF 1024
#define SMEM_COMBINE_SMEM_STAGE_BYTES 229376
#define SMEM_COMBINE_SMEM_STRIDE 229376
#define SMEM_COMBINE_SLOT_OFF 232440
#define SMEM_COMBINE_SLOT_STAGE_BYTES 8
#define SMEM_COMBINE_SLOT_STRIDE 8
#define SMEM_SMEM_V40_OFF 1024
#define SMEM_SMEM_V40_STAGE_BYTES 16384
#define SMEM_SMEM_V40_STRIDE 16384
#define SMEM_SMEM_V41_OFF 66560
#define SMEM_SMEM_V41_STAGE_BYTES 16384
#define SMEM_SMEM_V41_STRIDE 16384
#define SMEM_SMEM_V42_OFF 132096
#define SMEM_SMEM_V42_STAGE_BYTES 16384
#define SMEM_SMEM_V42_STRIDE 16384
#define SMEM_SMEM_V43_OFF 197632
#define SMEM_SMEM_V43_STAGE_BYTES 512
#define SMEM_SMEM_V43_STRIDE 512
#define SMEM_SMEM_V44_OFF 199680
#define SMEM_SMEM_V44_STAGE_BYTES 1024
#define SMEM_SMEM_V44_STRIDE 1024
#define SMEM_SMEM_V45_OFF 203776
#define SMEM_SMEM_V45_STAGE_BYTES 1024
#define SMEM_SMEM_V45_STRIDE 1024
#define SMEM_SMEM_V46_OFF 1024
#define SMEM_SMEM_V46_STAGE_BYTES 16384
#define SMEM_SMEM_V46_STRIDE 16384
#define SMEM_SMEM_V47_OFF 99328
#define SMEM_SMEM_V47_STAGE_BYTES 16384
#define SMEM_SMEM_V47_STRIDE 16384
#define SMEM_SMEM_V48_OFF 197632
#define SMEM_SMEM_V48_STAGE_BYTES 512
#define SMEM_SMEM_V48_STRIDE 512
#define SMEM_SMEM_V49_OFF 200704
#define SMEM_SMEM_V49_STAGE_BYTES 1024
#define SMEM_SMEM_V49_STRIDE 1024
#define SMEM_SMEM_V50_OFF 207872
#define SMEM_SMEM_V50_STAGE_BYTES 16384
#define SMEM_SMEM_V50_STRIDE 16384
#define SMEM_SMEM_V51_OFF 224256
#define SMEM_SMEM_V51_STAGE_BYTES 4096
#define SMEM_SMEM_V51_STRIDE 4096
#define SMEM_SMEM_V52_OFF 228352
#define SMEM_SMEM_V52_STAGE_BYTES 512
#define SMEM_SMEM_V52_STRIDE 512
#define SMEM_SMEM_V53_OFF 228864
#define SMEM_SMEM_V53_STAGE_BYTES 512
#define SMEM_SMEM_V53_STRIDE 512
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 384

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
kernel_cake_mok_forward_mxfp8_clamped(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap wd_q, const __grid_constant__ CUtensorMap wd_sc, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_q_store, const __grid_constant__ CUtensorMap hidden_sc_store, const __grid_constant__ CUtensorMap hidden_t_store, const __grid_constant__ CUtensorMap hidden_sc_t_store, const __grid_constant__ CUtensorMap hidden_q, const __grid_constant__ CUtensorMap hidden_sc, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, unsigned int* __restrict__ combine_next, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit)
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
    #define swiglu_arrived_addr (mbar_base + 0)
    #define gemm_arrived_addr (mbar_base + 24)
    #define scales_arrived_addr (mbar_base + 56)
    #define gemm_finished_addr (mbar_base + 88)
    #define scales_finished_addr (mbar_base + 120)
    #define output_arrived_addr (mbar_base + 152)
    #define output_finished_addr (mbar_base + 160)
    #define schedule_arrived_addr (mbar_base + 168)
    #define schedule_finished_addr (mbar_base + 176)
    #define drain_arrived_0_addr (mbar_base + 184)
    #define drain_arrived_1_addr (mbar_base + 192)
    #define drain_arrived_2_addr (mbar_base + 200)
    #define drain_arrived_3_addr (mbar_base + 208)
    #define drain_arrived_4_addr (mbar_base + 216)
    #define drain_arrived_5_addr (mbar_base + 224)
    #define drain_arrived_6_addr (mbar_base + 232)
    #define drain_arrived_7_addr (mbar_base + 240)
    #define drain_finished_0_addr (mbar_base + 248)
    #define drain_finished_1_addr (mbar_base + 256)
    #define drain_finished_2_addr (mbar_base + 264)
    #define drain_finished_3_addr (mbar_base + 272)
    #define drain_finished_4_addr (mbar_base + 280)
    #define drain_finished_5_addr (mbar_base + 288)
    #define drain_finished_6_addr (mbar_base + 296)
    #define drain_finished_7_addr (mbar_base + 304)
    #define dispatch_arrived_addr (mbar_base + 312)
    #define dispatch_arrived_hi_addr (mbar_base + 320)
    #define combine_arrived_addr (mbar_base + 328)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* a_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int a_smem_addr = smem + 1024;
    __nv_bfloat16* b_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int b_smem_addr = smem + 66560;
    __nv_bfloat16* b_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int b_hi_addr = smem + 132096;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 207872);
    const int d_smem_addr = smem + 207872;
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int gate_smem_addr = smem + 1024;
    __nv_bfloat16* up_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int up_smem_addr = smem + 99328;
    __nv_bfloat16* hidden_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_staging_addr = smem + 197632;
    unsigned int* hidden_words = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int hidden_words_addr = smem + 197632;
    unsigned int* smem_v8 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_v8_addr = smem + 1024;
    unsigned int* smem_v9 = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int smem_v9_addr = smem + 17408;
    unsigned int* smem_v10 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int smem_v10_addr = smem + 99328;
    unsigned int* smem_v11 = reinterpret_cast<unsigned int*>(smem_raw + 115712);
    const int smem_v11_addr = smem + 115712;
    unsigned int* smem_v12 = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int smem_v12_addr = smem + 33792;
    unsigned int* smem_v13 = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int smem_v13_addr = smem + 50176;
    unsigned int* smem_v14 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v14_addr = smem + 132096;
    unsigned int* smem_v15 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v15_addr = smem + 148480;
    unsigned int* smem_v16 = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int smem_v16_addr = smem + 66560;
    unsigned int* smem_v17 = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int smem_v17_addr = smem + 82944;
    unsigned int* smem_v18 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v18_addr = smem + 164864;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + 181248);
    const int smem_v19_addr = smem + 181248;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int dispatch_words_addr = smem + 1024;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + 199680);
    const int dispatch_weights_addr = smem + 199680;
    __nv_bfloat16* smem_v22 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v22_addr = smem + 1024;
    __nv_bfloat16* smem_v23 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v23_addr = smem + 1280;
    __nv_bfloat16* smem_v24 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_v24_addr = smem + 1536;
    __nv_bfloat16* smem_v25 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1792);
    const int smem_v25_addr = smem + 1792;
    unsigned int* smem_v26 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v26_addr = smem + 132096;
    unsigned int* smem_v27 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v27_addr = smem + 148480;
    unsigned int* smem_v28 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v28_addr = smem + 164864;
    unsigned int* smem_v29 = reinterpret_cast<unsigned int*>(smem_raw + 165376);
    const int smem_v29_addr = smem + 165376;
    unsigned int* smem_v30 = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_v30_addr = smem + 165888;
    unsigned int* smem_v31 = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int smem_v31_addr = smem + 182272;
    unsigned int* smem_v32 = reinterpret_cast<unsigned int*>(smem_raw + 198656);
    const int smem_v32_addr = smem + 198656;
    unsigned int* smem_v33 = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int smem_v33_addr = smem + 199168;
    __nv_bfloat16* smem_v34 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v34_addr = smem + 1024;
    __nv_bfloat16* smem_v35 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v35_addr = smem + 1280;
    __nv_bfloat16* smem_v36 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v36_addr = smem + 66560;
    __nv_bfloat16* smem_v37 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66816);
    const int smem_v37_addr = smem + 66816;
    __nv_bfloat16* combine_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int combine_smem_addr = smem + 1024;
    int* combine_slot = reinterpret_cast<int*>(smem_raw + 232440);
    const int combine_slot_addr = smem + 232440;
    uint8_t* smem_v40 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v40_addr = smem + 1024;
    uint8_t* smem_v41 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_v41_addr = smem + 66560;
    uint8_t* smem_v42 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v42_addr = smem + 132096;
    uint8_t* smem_v43 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v43_addr = smem + 197632;
    uint8_t* smem_v44 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v44_addr = smem + 199680;
    uint8_t* smem_v45 = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_v45_addr = smem + 203776;
    uint8_t* smem_v46 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v46_addr = smem + 1024;
    uint8_t* smem_v47 = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v47_addr = smem + 99328;
    uint8_t* smem_v48 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v48_addr = smem + 197632;
    uint8_t* smem_v49 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_v49_addr = smem + 200704;
    unsigned int* smem_v50 = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int smem_v50_addr = smem + 207872;
    unsigned int* smem_v51 = reinterpret_cast<unsigned int*>(smem_raw + 224256);
    const int smem_v51_addr = smem + 224256;
    unsigned int* smem_v52 = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_v52_addr = smem + 228352;
    unsigned int* smem_v53 = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_v53_addr = smem + 228864;
    int tokens = num_tokens[0];
    int shared_rows = local_tokens / 256;
    int gate_base = shared_rows * (intermediate / 256);
    int shared_fused = shared_rows * (intermediate / 256);
    int gate_tiles = mini_size / 256 * (intermediate / 256);
    int mini_gate = gate_tiles;
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int shared_tasks = shared_fused + shared_rows * ((hidden + 511) / 512);
    int mini_tasks = mini_gate + mini_swiglu + mini_size / 256 * ((hidden + 512 - 1) / 512);
    int comm_clusters = comm_sms / 2;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis_per_macro = macro_size / mini_size;
    int true_minis = (tokens + mini_size - 1) / mini_size;
    int last_minis = true_minis - (macros - 1) * minis_per_macro;
    int true_clusters = comm_clusters + shared_tasks + true_minis * mini_tasks;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (28 pipeline groups, 0 ordered-sequence groups, 48 barriers)
    // Mbarriers at smem_raw[0..384)

    if (threadIdx.x == 0) {
        // swiglu_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        // gemm_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 24, 1);
        mbarrier_init(smem + 32, 1);
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        // scales_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 56, 1);
        mbarrier_init(smem + 64, 1);
        mbarrier_init(smem + 72, 1);
        mbarrier_init(smem + 80, 1);
        // gemm_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 88, 1);
        mbarrier_init(smem + 96, 1);
        mbarrier_init(smem + 104, 1);
        mbarrier_init(smem + 112, 1);
        // scales_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 120, 1);
        mbarrier_init(smem + 128, 1);
        mbarrier_init(smem + 136, 1);
        mbarrier_init(smem + 144, 1);
        // output_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 152, 1);
        // output_finished: 1 barriers, init_count=2
        mbarrier_init(smem + 160, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 176, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 184, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 192, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 200, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 208, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 216, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 224, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 232, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 240, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 248, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 256, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 264, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 272, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 280, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 288, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 296, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 304, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 312, 1);
        // dispatch_arrived_hi: 1 barriers, init_count=1
        mbarrier_init(smem + 320, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 328, 1);
        mbarrier_init(smem + 336, 1);
        mbarrier_init(smem + 344, 1);
        mbarrier_init(smem + 352, 1);
        mbarrier_init(smem + 360, 1);
        mbarrier_init(smem + 368, 1);
        mbarrier_init(smem + 376, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (576 columns, 572 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 384);
    if (warp == 0) {
        int _tmem_hold = smem + 384;
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
    int cluster = bid / 2;
    int cta_rank_0 = cta_rank;
    unsigned int gemm_bits = 4294901760;
    unsigned int swiglu_bits = 4294901760;
    unsigned int dispatch_bits = 4294901760;
    unsigned int combine_bits = 4294901760;
    int i_tiles = intermediate / 128;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        if (macros > 0) {
            int _min_0 = ((macro_size) < (tokens - (macros - 1) * macro_size) ? (macro_size) : (tokens - (macros - 1) * macro_size));
            int last_rows = _min_0;
            int last_dispatch = last_rows / 128 * ((hidden + 511) / 512);
            unsigned int phase_bits = dispatch_bits;
            int col_blocks = (hidden + 511) / 512;
            int macro_offset = (macros - 1) * macro_size;
            int _min_1 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
            int macro_tokens = _min_1;
            int tasks = macro_tokens / 128 * col_blocks;
            int task = comm_cta;
            if (task < tasks) {
                int _min_2 = ((256) < (hidden - task % col_blocks * 512) ? (256) : (hidden - task % col_blocks * 512));
                int peer = -1;
                int peer_token = -1;
                if (tid < 128) {
                    peer = schedule_rank[macro_offset + task / col_blocks * 128 + tid];
                    peer_token = schedule_token[macro_offset + task / col_blocks * 128 + tid];
                }
                uint32_t _cta_count_0 = __syncthreads_count(peer >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_0 * (unsigned int)(_min_2 * 2));
                }
                __syncthreads();
                if (peer >= 0) {
                    cp_async_bulk_gmem2smem(smem_v34_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(task % col_blocks * 512)) * (unsigned long long)2)), _min_2 * 2, dispatch_arrived_addr);
                } else if (tid < 128) {
                    #pragma unroll
                    for (int vec = 0; vec < 32; vec++) {
                        asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 512 + vec * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                    }
                }
            }
            while (task < tasks) {
                int row = task / col_blocks * 128;
                int col_block = task % col_blocks;
                int _min_3 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                int chunk_cols = _min_3;
                int _min_4 = ((256) < (chunk_cols) ? (256) : (chunk_cols));
                int lo_cols = _min_4;
                int hi_cols = chunk_cols - lo_cols;
                if (hi_cols > 0) {
                    int peer_1 = -1;
                    int peer_token_1 = -1;
                    if (tid < 128) {
                        peer_1 = schedule_rank[macro_offset + row + tid];
                        peer_token_1 = schedule_token[macro_offset + row + tid];
                    }
                    uint32_t _cta_count_1 = __syncthreads_count(peer_1 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_hi_addr, _cta_count_1 * (unsigned int)(hi_cols * 2));
                    }
                    __syncthreads();
                    if (peer_1 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v36_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512) + 256) * (unsigned long long)2)), hi_cols * 2, dispatch_arrived_hi_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_1 = 0; vec_1 < 32; vec_1++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (65536 + tid * 512 + vec_1 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                unsigned int bits = phase_bits;
                mbarrier_wait(dispatch_arrived_addr, bits & 1);
                bits = bits ^ 1;
                int row_tile = row / 128;
                int k_tiles = hidden / 128;
                int macro_tiles = macro_size / 128;
                if (lo_cols / 128 > 0) {
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
                                x0 = (float)smem_v34[col * 256 + row_0];
                                x1 = (float)smem_v34[(col + 1) * 256 + row_0];
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
                                smem_v30[(row_0 * 128 + col_1) / 4] = words[k_1];
                            }
                        }
                        smem_v32[row_0 % 32 * 4 + row_0 / 32] = scale_word;
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
                                pairs_1[k_2] = dispatch_words[(row_0_1 * 256 + col_2) / 2];
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
                                smem_v26[(row_0_1 * 128 + col_3) / 4] = words_1[k_3];
                            }
                        }
                        smem_v28[row_0_1 % 32 * 4 + row_0_1 / 32] = scale_word_1;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile = col_block * 4;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile * 128, row, smem_v26_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile, smem_v28_addr);
                        tma_store_2d((&x_t_store), row, col_tile * 128, smem_v30_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile * macro_tiles + row_tile, smem_v32_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (lo_cols / 128 > 1) {
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
                                x0_2 = (float)smem_v35[col_4 * 256 + row_0_2];
                                x1_2 = (float)smem_v35[(col_4 + 1) * 256 + row_0_2];
                                __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(x0_2, x1_2));
                                pairs_2[k_4] = __as_u32(_bf16x2_1);
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
                                smem_v31[(row_0_2 * 128 + col_5) / 4] = words_2[k_5];
                            }
                        }
                        smem_v33[row_0_2 % 32 * 4 + row_0_2 / 32] = scale_word_2;
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
                                pairs_3[k_6] = dispatch_words[64 + (row_0_3 * 256 + col_6) / 2];
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
                                smem_v27[(row_0_3 * 128 + col_7) / 4] = words_3[k_7];
                            }
                        }
                        smem_v29[row_0_3 % 32 * 4 + row_0_3 / 32] = scale_word_3;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_1 = col_block * 4 + 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_1 * 128, row, smem_v27_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile_1, smem_v29_addr);
                        tma_store_2d((&x_t_store), row, col_tile_1 * 128, smem_v31_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_1 * macro_tiles + row_tile, smem_v33_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                phase_bits = bits;
                int next_task = task + comm_sms;
                if (next_task < tasks) {
                    int _min_5 = ((256) < (hidden - next_task % col_blocks * 512) ? (256) : (hidden - next_task % col_blocks * 512));
                    int peer_2 = -1;
                    int peer_token_2 = -1;
                    if (tid < 128) {
                        peer_2 = schedule_rank[macro_offset + next_task / col_blocks * 128 + tid];
                        peer_token_2 = schedule_token[macro_offset + next_task / col_blocks * 128 + tid];
                    }
                    uint32_t _cta_count_2 = __syncthreads_count(peer_2 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * (unsigned int)(_min_5 * 2));
                    }
                    __syncthreads();
                    if (peer_2 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v34_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task % col_blocks * 512)) * (unsigned long long)2)), _min_5 * 2, dispatch_arrived_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_2 = 0; vec_2 < 32; vec_2++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 512 + vec_2 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                if (hi_cols > 0) {
                    unsigned int bits_0 = phase_bits;
                    mbarrier_wait(dispatch_arrived_hi_addr, bits_0 >> 1 & 1);
                    bits_0 = bits_0 ^ 2;
                    int row_tile_1 = row / 128;
                    int k_tiles_2 = hidden / 128;
                    int macro_tiles_3 = macro_size / 128;
                    if (hi_cols / 128 > 0) {
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
                                    x0_4 = (float)smem_v36[col_8 * 256 + row_0_4];
                                    x1_4 = (float)smem_v36[(col_8 + 1) * 256 + row_0_4];
                                    __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(x0_4, x1_4));
                                    pairs_4[k_8] = __as_u32(_bf16x2_2);
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
                                    smem_v30[(row_0_4 * 128 + col_9) / 4] = words_4[k_9];
                                }
                            }
                            smem_v32[row_0_4 % 32 * 4 + row_0_4 / 32] = scale_word_4;
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
                                    pairs_5[k_10] = dispatch_words[16384 + (row_0_5 * 256 + col_10) / 2];
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
                                    smem_v26[(row_0_5 * 128 + col_11) / 4] = words_5[k_11];
                                }
                            }
                            smem_v28[row_0_5 % 32 * 4 + row_0_5 / 32] = scale_word_5;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_2 = col_block * 4 + 2;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_2 * 128, row, smem_v26_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_2 + col_tile_2, smem_v28_addr);
                            tma_store_2d((&x_t_store), row, col_tile_2 * 128, smem_v30_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_2 * macro_tiles_3 + row_tile_1, smem_v32_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (hi_cols / 128 > 1) {
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
                                    x0_6 = (float)smem_v37[col_12 * 256 + row_0_6];
                                    x1_6 = (float)smem_v37[(col_12 + 1) * 256 + row_0_6];
                                    __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(x0_6, x1_6));
                                    pairs_6[k_12] = __as_u32(_bf16x2_3);
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
                                    smem_v31[(row_0_6 * 128 + col_13) / 4] = words_6[k_13];
                                }
                            }
                            smem_v33[row_0_6 % 32 * 4 + row_0_6 / 32] = scale_word_6;
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
                                    pairs_7[k_14] = dispatch_words[16448 + (row_0_7 * 256 + col_14) / 2];
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
                                    smem_v27[(row_0_7 * 128 + col_15) / 4] = words_7[k_15];
                                }
                            }
                            smem_v29[row_0_7 % 32 * 4 + row_0_7 / 32] = scale_word_7;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_3 = col_block * 4 + 2 + 1;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_3 * 128, row, smem_v27_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_2 + col_tile_3, smem_v29_addr);
                            tma_store_2d((&x_t_store), row, col_tile_3 * 128, smem_v31_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_3 * macro_tiles_3 + row_tile_1, smem_v33_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    phase_bits = bits_0;
                }
                if (tid == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + row) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
                __syncthreads();
                task = next_task;
            }
            dispatch_bits = phase_bits;
            int macro = macros - 1;
            while (macro >= 0) {
                int _min_6 = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int macro_rows = _min_6;
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                int dispatch_tasks = 0;
                if (macro > 0) {
                    int _min_7 = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                    int previous_rows = _min_7;
                    dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                }
                if (macro == 0) {
                    unsigned int phase_bits_0 = combine_bits;
                    int col_blocks_1 = (hidden + 1023) / 1024;
                    int macro_offset_2 = 0;
                    int _min_8 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
                    int macro_tokens_3 = _min_8;
                    int num_tasks = (macro_tokens_3 / 16 * col_blocks_1 + 6) / 7;
                    if (tid == 0) {
                        unsigned int _atomic_old_0 = atomicAdd(&combine_next[0], 1);
                        combine_slot[0] = (int)_atomic_old_0;
                    }
                    __syncthreads();
                    int task_4 = combine_slot[0];
                    __syncthreads();
                    while (task_4 < num_tasks) {
                        unsigned int phase_bits_1 = phase_bits_0;
                        int col_blocks_2 = (hidden + 1023) / 1024;
                        int first_tile = task_4 * 7;
                        int macro_offset_3 = 0;
                        int _min_9 = ((macro_size) < (tokens - macro_offset_3) ? (macro_size) : (tokens - macro_offset_3));
                        int macro_tokens_4 = _min_9;
                        int _min_10 = ((7) < (macro_tokens_4 / 16 * col_blocks_2 - first_tile) ? (7) : (macro_tokens_4 / 16 * col_blocks_2 - first_tile));
                        int valid_tiles = _min_10;
                        if (valid_tiles > 0) {
                            int first_row = first_tile / col_blocks_2 * 16 + tid;
                            int first_col = first_tile % col_blocks_2;
                            int rows[7];
                            int columns[7];
                            int peers[7];
                            int tokens_0[7];
                            unsigned int counts_1[7];
                            int row_1 = first_row;
                            int column = first_col;
                            #pragma unroll
                            for (int stage = 0; stage < 7; stage++) {
                                rows[stage] = row_1;
                                columns[stage] = column;
                                peers[stage] = -1;
                                tokens_0[stage] = -1;
                                if (valid_tiles > stage && tid < 16) {
                                    peers[stage] = schedule_rank[macro_offset_3 + row_1];
                                    tokens_0[stage] = schedule_token[macro_offset_3 + row_1];
                                }
                                counts_1[stage] = 0;
                                if (valid_tiles > stage) {
                                    if (stage == 0 || column == 0) {
                                        uint32_t _cta_count_3 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _cta_count_3;
                                    } else {
                                        counts_1[stage] = counts_1[stage - 1];
                                    }
                                }
                                column = column + 1;
                                if (column == col_blocks_2) {
                                    column = 0;
                                    row_1 = row_1 + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini = (macro_offset_3 + first_row) / mini_size;
                                int last_mini = (macro_offset_3 + (first_tile + valid_tiles - 1) / col_blocks_2 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini = first_mini; mini < last_mini + 1; mini++) {
                                    int _min_11 = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                                    int mini_rows = _min_11;
                                    int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                                    int32_t _relaxed_ld_0;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(y_ready + mini) : "memory");
                                    int value = _relaxed_ld_0;
                                    while (value < required) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_1;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_1) : "l"(y_ready + mini) : "memory");
                                        value = _relaxed_ld_1;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                #pragma unroll
                                for (int stage_1 = 0; stage_1 < 7; stage_1++) {
                                    if (valid_tiles > stage_1) {
                                        int _min_12 = ((1024) < (hidden - columns[stage_1] * 1024) ? (1024) : (hidden - columns[stage_1] * 1024));
                                        unsigned int chunk_bytes = (unsigned int)(_min_12 * 2);
                                        mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_1) * 8, counts_1[stage_1] * chunk_bytes);
                                    }
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 7; stage_2++) {
                                if (peers[stage_2] >= 0) {
                                    int _min_13 = ((1024) < (hidden - columns[stage_2] * 1024) ? (1024) : (hidden - columns[stage_2] * 1024));
                                    int chunk_cols_1 = _min_13;
                                    cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_2 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows[stage_2] * (unsigned long long)hidden + (unsigned long long)(columns[stage_2] * 1024)) * (unsigned long long)2)), chunk_cols_1 * 2, combine_arrived_addr + (stage_2) * 8);
                                }
                            }
                            #pragma unroll
                            for (int stage_3 = 0; stage_3 < 7; stage_3++) {
                                if (valid_tiles > stage_3) {
                                    mbarrier_wait(combine_arrived_addr + (stage_3) * 8, phase_bits_1 >> (unsigned int)stage_3 & 1);
                                    phase_bits_1 = phase_bits_1 ^ (unsigned int)(1 << stage_3);
                                    if (peers[stage_3] >= 0) {
                                        int _min_14 = ((1024) < (hidden - columns[stage_3] * 1024) ? (1024) : (hidden - columns[stage_3] * 1024));
                                        unsigned int chunk_bytes_1 = (unsigned int)(_min_14 * 2);
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        {
                                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[stage_3]]) + ((unsigned long long)tokens_0[stage_3] * (unsigned long long)hidden + (unsigned long long)(columns[stage_3] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_0), "r"(combine_smem_addr + (unsigned int)((stage_3 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_1))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            int warp_1 = tid / 32;
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                        phase_bits_0 = phase_bits_1;
                        if (tid == 0) {
                            unsigned int _atomic_old_1 = atomicAdd(&combine_next[0], 1);
                            combine_slot[0] = (int)_atomic_old_1;
                        }
                        __syncthreads();
                        task_4 = combine_slot[0];
                        __syncthreads();
                    }
                    combine_bits = phase_bits_0;
                }
                int _max_0 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task_1 = comm_cta; task_1 < ((macro > 0) ? _max_0 : 0); task_1 += comm_sms) {
                    if (combine_tasks > task_1) {
                        unsigned int phase_bits_0_1 = combine_bits;
                        int col_blocks_1_1 = (hidden + 1023) / 1024;
                        int first_tile_1 = task_1 * 7;
                        int macro_offset_2_1 = macro * macro_size;
                        int _min_15 = ((macro_size) < (tokens - macro_offset_2_1) ? (macro_size) : (tokens - macro_offset_2_1));
                        int macro_tokens_3_1 = _min_15;
                        int _min_16 = ((7) < (macro_tokens_3_1 / 16 * col_blocks_1_1 - first_tile_1) ? (7) : (macro_tokens_3_1 / 16 * col_blocks_1_1 - first_tile_1));
                        int valid_tiles_1 = _min_16;
                        if (valid_tiles_1 > 0) {
                            int first_row_1 = first_tile_1 / col_blocks_1_1 * 16 + tid;
                            int first_col_1 = first_tile_1 % col_blocks_1_1;
                            int rows_1[7];
                            int columns_1[7];
                            int peers_1[7];
                            int tokens_0_1[7];
                            unsigned int counts_1_1[7];
                            int row_2 = first_row_1;
                            int column_1 = first_col_1;
                            #pragma unroll
                            for (int stage_4 = 0; stage_4 < 7; stage_4++) {
                                rows_1[stage_4] = row_2;
                                columns_1[stage_4] = column_1;
                                peers_1[stage_4] = -1;
                                tokens_0_1[stage_4] = -1;
                                if (valid_tiles_1 > stage_4 && tid < 16) {
                                    peers_1[stage_4] = schedule_rank[macro_offset_2_1 + row_2];
                                    tokens_0_1[stage_4] = schedule_token[macro_offset_2_1 + row_2];
                                }
                                counts_1_1[stage_4] = 0;
                                if (valid_tiles_1 > stage_4) {
                                    if (stage_4 == 0 || column_1 == 0) {
                                        uint32_t _cta_count_4 = __syncthreads_count(peers_1[stage_4] >= 0);
                                        counts_1_1[stage_4] = _cta_count_4;
                                    } else {
                                        counts_1_1[stage_4] = counts_1_1[stage_4 - 1];
                                    }
                                }
                                column_1 = column_1 + 1;
                                if (column_1 == col_blocks_1_1) {
                                    column_1 = 0;
                                    row_2 = row_2 + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini_1 = (macro_offset_2_1 + first_row_1) / mini_size;
                                int last_mini_1 = (macro_offset_2_1 + (first_tile_1 + valid_tiles_1 - 1) / col_blocks_1_1 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini_1 = first_mini_1; mini_1 < last_mini_1 + 1; mini_1++) {
                                    int _min_17 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                                    int mini_rows_1 = _min_17;
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    int32_t _relaxed_ld_2;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_2) : "l"(y_ready + mini_1) : "memory");
                                    int value_1 = _relaxed_ld_2;
                                    while (value_1 < required_1) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_3;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_3) : "l"(y_ready + mini_1) : "memory");
                                        value_1 = _relaxed_ld_3;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                #pragma unroll
                                for (int stage_5 = 0; stage_5 < 7; stage_5++) {
                                    if (valid_tiles_1 > stage_5) {
                                        int _min_18 = ((1024) < (hidden - columns_1[stage_5] * 1024) ? (1024) : (hidden - columns_1[stage_5] * 1024));
                                        unsigned int chunk_bytes_2 = (unsigned int)(_min_18 * 2);
                                        mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_5) * 8, counts_1_1[stage_5] * chunk_bytes_2);
                                    }
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int stage_6 = 0; stage_6 < 7; stage_6++) {
                                if (peers_1[stage_6] >= 0) {
                                    int _min_19 = ((1024) < (hidden - columns_1[stage_6] * 1024) ? (1024) : (hidden - columns_1[stage_6] * 1024));
                                    int chunk_cols_2 = _min_19;
                                    cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_6 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[stage_6] * (unsigned long long)hidden + (unsigned long long)(columns_1[stage_6] * 1024)) * (unsigned long long)2)), chunk_cols_2 * 2, combine_arrived_addr + (stage_6) * 8);
                                }
                            }
                            #pragma unroll
                            for (int stage_7 = 0; stage_7 < 7; stage_7++) {
                                if (valid_tiles_1 > stage_7) {
                                    mbarrier_wait(combine_arrived_addr + (stage_7) * 8, phase_bits_0_1 >> (unsigned int)stage_7 & 1);
                                    phase_bits_0_1 = phase_bits_0_1 ^ (unsigned int)(1 << stage_7);
                                    if (peers_1[stage_7] >= 0) {
                                        int _min_20 = ((1024) < (hidden - columns_1[stage_7] * 1024) ? (1024) : (hidden - columns_1[stage_7] * 1024));
                                        unsigned int chunk_bytes_3 = (unsigned int)(_min_20 * 2);
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        {
                                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[stage_7]]) + ((unsigned long long)tokens_0_1[stage_7] * (unsigned long long)hidden + (unsigned long long)(columns_1[stage_7] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_1), "r"(combine_smem_addr + (unsigned int)((stage_7 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_3))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            int warp_2 = tid / 32;
                            if (tid % 32 == 0 && warp_2 < valid_tiles_1 && macro > 0) {
                                int row_done = (first_tile_1 + warp_2) / col_blocks_1_1 * 16;
                                bool enabled_value = 1;
                                if (enabled_value != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_2_1 + row_done) / 128))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                        combine_bits = phase_bits_0_1;
                    }
                    if (dispatch_tasks > task_1) {
                        unsigned int phase_bits_0_2 = dispatch_bits;
                        int col_blocks_1_2 = (hidden + 511) / 512;
                        int macro_offset_2_2 = (macro - 1) * macro_size;
                        int _min_21 = ((macro_size) < (tokens - macro_offset_2_2) ? (macro_size) : (tokens - macro_offset_2_2));
                        int macro_tokens_3_2 = _min_21;
                        if (task_1 < macro_tokens_3_2 / 128 * col_blocks_1_2) {
                            int row_3 = task_1 / col_blocks_1_2 * 128;
                            int col_block_1 = task_1 % col_blocks_1_2;
                            int _min_22 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                            int chunk_cols_3 = _min_22;
                            unsigned int chunk_bytes_4 = (unsigned int)(chunk_cols_3 * 2);
                            int peer_3 = -1;
                            int peer_token_3 = -1;
                            if (tid < 128) {
                                peer_3 = schedule_rank[macro_offset_2_2 + row_3 + tid];
                                peer_token_3 = schedule_token[macro_offset_2_2 + row_3 + tid];
                            }
                            uint32_t _cta_count_5 = __syncthreads_count(peer_3 >= 0);
                            if (tid == 0) {
                                int previous_offset = macro * macro_size;
                                int _min_23 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                int previous_tokens = _min_23;
                                if (row_3 < previous_tokens) {
                                    int previous_mini = (previous_offset + row_3) / mini_size;
                                    int _min_24 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                    int mini_rows_2 = _min_24;
                                    int required_2 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                                    bool enabled_value_1 = 1;
                                    if (enabled_value_1 != 0) {
                                        int32_t _relaxed_ld_4;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(y_ready + previous_mini) : "memory");
                                        int value_2 = _relaxed_ld_4;
                                        while (value_2 < required_2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_5;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(y_ready + previous_mini) : "memory");
                                            value_2 = _relaxed_ld_5;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_5 * chunk_bytes_4);
                            }
                            __syncthreads();
                            if (peer_3 >= 0) {
                                cp_async_bulk_gmem2smem(smem_v22_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_3])) + ((unsigned long long)((unsigned long long)(peer_token_3 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2)), chunk_cols_3 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_3 = 0; vec_3 < 64; vec_3++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 1024 + vec_3 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                            mbarrier_wait(dispatch_arrived_addr, phase_bits_0_2 & 1);
                            phase_bits_0_2 = phase_bits_0_2 ^ 1;
                            int row_tile_2 = row_3 / 128;
                            int k_tiles_1 = hidden / 128;
                            int macro_tiles_1 = macro_size / 128;
                            if (chunk_cols_3 / 128 > 0) {
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
                                            int col_16 = k_block_8 * 32 + (tid * 4 + k_16 * 2) % 32;
                                            float x0_8 = 0.0f;
                                            float x1_8 = 0.0f;
                                            x0_8 = (float)smem_v22[col_16 * 512 + row_0_8];
                                            x1_8 = (float)smem_v22[(col_16 + 1) * 512 + row_0_8];
                                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(x0_8, x1_8));
                                            pairs_8[k_16] = __as_u32(_bf16x2_4);
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
                                            int col_17 = k_block_8 * 32 + (tid * 4 + k_17 * 4) % 32;
                                            smem_v30[(row_0_8 * 128 + col_17) / 4] = words_8[k_17];
                                        }
                                    }
                                    smem_v32[row_0_8 % 32 * 4 + row_0_8 / 32] = scale_word_8;
                                } else {
                                    int row_0_9 = tid - 128;
                                    unsigned int scale_word_9 = 0;
                                    #pragma unroll 1
                                    for (int j_9 = 0; j_9 < 4; j_9++) {
                                        int k_block_9 = (j_9 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_9[16];
                                        #pragma unroll
                                        for (int k_18 = 0; k_18 < 16; k_18++) {
                                            int col_18 = k_block_9 * 32 + ((tid - 128) * 4 + k_18 * 2) % 32;
                                            float x0_9 = 0.0f;
                                            float x1_9 = 0.0f;
                                            pairs_9[k_18] = dispatch_words[(row_0_9 * 512 + col_18) / 2];
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
                                            int col_19 = k_block_9 * 32 + ((tid - 128) * 4 + k_19 * 4) % 32;
                                            smem_v26[(row_0_9 * 128 + col_19) / 4] = words_9[k_19];
                                        }
                                    }
                                    smem_v28[row_0_9 % 32 * 4 + row_0_9 / 32] = scale_word_9;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_4 = col_block_1 * 4;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_4 * 128, row_3, smem_v26_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_4, smem_v28_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_4 * 128, smem_v30_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_4 * macro_tiles_1 + row_tile_2, smem_v32_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (chunk_cols_3 / 128 > 1) {
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
                                            int col_20 = k_block_10 * 32 + (tid * 4 + k_20 * 2) % 32;
                                            float x0_10 = 0.0f;
                                            float x1_10 = 0.0f;
                                            x0_10 = (float)smem_v23[col_20 * 512 + row_0_10];
                                            x1_10 = (float)smem_v23[(col_20 + 1) * 512 + row_0_10];
                                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(x0_10, x1_10));
                                            pairs_10[k_20] = __as_u32(_bf16x2_5);
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
                                            int col_21 = k_block_10 * 32 + (tid * 4 + k_21 * 4) % 32;
                                            smem_v31[(row_0_10 * 128 + col_21) / 4] = words_10[k_21];
                                        }
                                    }
                                    smem_v33[row_0_10 % 32 * 4 + row_0_10 / 32] = scale_word_10;
                                } else {
                                    int row_0_11 = tid - 128;
                                    unsigned int scale_word_11 = 0;
                                    #pragma unroll 1
                                    for (int j_11 = 0; j_11 < 4; j_11++) {
                                        int k_block_11 = (j_11 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_11[16];
                                        #pragma unroll
                                        for (int k_22 = 0; k_22 < 16; k_22++) {
                                            int col_22 = k_block_11 * 32 + ((tid - 128) * 4 + k_22 * 2) % 32;
                                            float x0_11 = 0.0f;
                                            float x1_11 = 0.0f;
                                            pairs_11[k_22] = dispatch_words[64 + (row_0_11 * 512 + col_22) / 2];
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
                                            int col_23 = k_block_11 * 32 + ((tid - 128) * 4 + k_23 * 4) % 32;
                                            smem_v27[(row_0_11 * 128 + col_23) / 4] = words_11[k_23];
                                        }
                                    }
                                    smem_v29[row_0_11 % 32 * 4 + row_0_11 / 32] = scale_word_11;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_5 = col_block_1 * 4 + 1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_5 * 128, row_3, smem_v27_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_5, smem_v29_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_5 * 128, smem_v31_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_5 * macro_tiles_1 + row_tile_2, smem_v33_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (chunk_cols_3 / 128 > 2) {
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
                                            int col_24 = k_block_12 * 32 + (tid * 4 + k_24 * 2) % 32;
                                            float x0_12 = 0.0f;
                                            float x1_12 = 0.0f;
                                            x0_12 = (float)smem_v24[col_24 * 512 + row_0_12];
                                            x1_12 = (float)smem_v24[(col_24 + 1) * 512 + row_0_12];
                                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(x0_12, x1_12));
                                            pairs_12[k_24] = __as_u32(_bf16x2_6);
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
                                            int col_25 = k_block_12 * 32 + (tid * 4 + k_25 * 4) % 32;
                                            smem_v30[(row_0_12 * 128 + col_25) / 4] = words_12[k_25];
                                        }
                                    }
                                    smem_v32[row_0_12 % 32 * 4 + row_0_12 / 32] = scale_word_12;
                                } else {
                                    int row_0_13 = tid - 128;
                                    unsigned int scale_word_13 = 0;
                                    #pragma unroll 1
                                    for (int j_13 = 0; j_13 < 4; j_13++) {
                                        int k_block_13 = (j_13 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_13[16];
                                        #pragma unroll
                                        for (int k_26 = 0; k_26 < 16; k_26++) {
                                            int col_26 = k_block_13 * 32 + ((tid - 128) * 4 + k_26 * 2) % 32;
                                            float x0_13 = 0.0f;
                                            float x1_13 = 0.0f;
                                            pairs_13[k_26] = dispatch_words[128 + (row_0_13 * 512 + col_26) / 2];
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
                                            int col_27 = k_block_13 * 32 + ((tid - 128) * 4 + k_27 * 4) % 32;
                                            smem_v26[(row_0_13 * 128 + col_27) / 4] = words_13[k_27];
                                        }
                                    }
                                    smem_v28[row_0_13 % 32 * 4 + row_0_13 / 32] = scale_word_13;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_6 = col_block_1 * 4 + 2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_6 * 128, row_3, smem_v26_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_6, smem_v28_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_6 * 128, smem_v30_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_6 * macro_tiles_1 + row_tile_2, smem_v32_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (chunk_cols_3 / 128 > 3) {
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
                                            int col_28 = k_block_14 * 32 + (tid * 4 + k_28 * 2) % 32;
                                            float x0_14 = 0.0f;
                                            float x1_14 = 0.0f;
                                            x0_14 = (float)smem_v25[col_28 * 512 + row_0_14];
                                            x1_14 = (float)smem_v25[(col_28 + 1) * 512 + row_0_14];
                                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(x0_14, x1_14));
                                            pairs_14[k_28] = __as_u32(_bf16x2_7);
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
                                            int col_29 = k_block_14 * 32 + (tid * 4 + k_29 * 4) % 32;
                                            smem_v31[(row_0_14 * 128 + col_29) / 4] = words_14[k_29];
                                        }
                                    }
                                    smem_v33[row_0_14 % 32 * 4 + row_0_14 / 32] = scale_word_14;
                                } else {
                                    int row_0_15 = tid - 128;
                                    unsigned int scale_word_15 = 0;
                                    #pragma unroll 1
                                    for (int j_15 = 0; j_15 < 4; j_15++) {
                                        int k_block_15 = (j_15 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_15[16];
                                        #pragma unroll
                                        for (int k_30 = 0; k_30 < 16; k_30++) {
                                            int col_30 = k_block_15 * 32 + ((tid - 128) * 4 + k_30 * 2) % 32;
                                            float x0_15 = 0.0f;
                                            float x1_15 = 0.0f;
                                            pairs_15[k_30] = dispatch_words[192 + (row_0_15 * 512 + col_30) / 2];
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
                                            int col_31 = k_block_15 * 32 + ((tid - 128) * 4 + k_31 * 4) % 32;
                                            smem_v27[(row_0_15 * 128 + col_31) / 4] = words_15[k_31];
                                        }
                                    }
                                    smem_v29[row_0_15 % 32 * 4 + row_0_15 / 32] = scale_word_15;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_7 = col_block_1 * 4 + 3;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_7 * 128, row_3, smem_v27_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_7, smem_v29_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_7 * 128, smem_v31_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_7 * macro_tiles_1 + row_tile_2, smem_v33_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_2_2 + row_3) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            __syncthreads();
                        }
                        dispatch_bits = phase_bits_0_2;
                    }
                }
                macro = macro - 1;
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
            int compute = cluster - comm_clusters;
            int result = 0;
            if (compute >= 0) {
                if (compute < shared_tasks) {
                    if (compute >= shared_tasks && compute < shared_tasks) {
                        result = 1;
                    }
                } else {
                    int mini_task = (compute - shared_tasks) % mini_tasks;
                    if (mini_task >= mini_gate && mini_task < mini_gate + mini_swiglu) {
                        result = 1;
                    }
                }
            }
            if (compute < shared_fused) {
                int col_blocks_3 = (intermediate + 256 - 1) / 256;
                int x = -1;
                int y = -1;
                int expert = -1;
                int k_start = 0;
                int k_end = 0;
                int first = 0;
                int row_blocks = local_tokens / 256;
                if (compute < row_blocks * col_blocks_3) {
                    int supergroup = compute / (row_blocks * 8);
                    int full_cols = col_blocks_3 / 8 * 8;
                    int row_4 = 0;
                    int col_32 = 0;
                    if (compute < row_blocks * full_cols) {
                        row_4 = compute % (row_blocks * 8) / 8;
                        col_32 = supergroup * 8 + compute % 8;
                    } else {
                        row_4 = (compute - row_blocks * full_cols) / (col_blocks_3 - full_cols);
                        col_32 = full_cols + (compute - row_blocks * full_cols) % (col_blocks_3 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_4 = row_blocks - row_4 - 1;
                    }
                    x = row_4;
                    y = col_32;
                    expert = 0;
                }
                unsigned int phase_bits_2 = gemm_bits;
                int has_hi = 0;
                has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                int global_mini = 0;
                int macro_rows_1 = 0;
                int iterations = hidden / 64;
                if (expert < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_25 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                int _max_1 = ((0) > (_min_25) ? (0) : (_min_25));
                                int mini_rows_3 = _max_1;
                                int required_3 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                            }
                            int ring = 0;
                            #pragma unroll 1
                            for (int idx = 0; idx < iterations; idx++) {
                                mbarrier_wait(gemm_finished_addr + (ring) * 8, phase_bits_2 >> (unsigned int)(16 + ring) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring * 16384)), "l"((&x_shared)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring * 16384)), "l"((&wg_shared)), "r"(0), "r"(y * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)(ring * 16384)), "l"((&wu_shared)), "r"(0), "r"(y * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_2 = phase_bits_2 ^ (unsigned int)(1 << 16 + ring);
                                ring = (ring + 1) % 4;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_1 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_2 >> 22 & 1);
                                phase_bits_2 = phase_bits_2 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_1 = 0; idx_1 < iterations; idx_1++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_1) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_1) * 8, phase_bits_2 >> (unsigned int)ring_1 & 1);
                                    int _mma_a_lo_0 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
                                    int _mma_b_lo_0 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
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
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)));
                                    int _mma_a_lo_1 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
                                    int _mma_b_lo_1 = (((b_hi_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
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
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accumulator + (256))), "r"(((idx_1 == 0) ? 0 : 1)));
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_1) * 8, (uint16_t)(3));
                                    phase_bits_2 = phase_bits_2 ^ (unsigned int)(1 << ring_1);
                                    ring_1 = (ring_1 + 1) % 4;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        mbarrier_wait(output_arrived_addr, phase_bits_2 >> 6 & 1);
                        phase_bits_2 = phase_bits_2 ^ 64;
                        unsigned int gate_packed[32];
                        unsigned int up_packed[32];
                        unsigned int hidden_packed[32];
                        unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 << 16);
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(address));
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(address + 256));
                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        gate_packed[0] = __as_u32(_bf16x2_8);
                        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                        up_packed[0] = __as_u32(_bf16x2_9);
                        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        gate_packed[1] = __as_u32(_bf16x2_10);
                        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                        up_packed[1] = __as_u32(_bf16x2_11);
                        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                        gate_packed[2] = __as_u32(_bf16x2_12);
                        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_tmem_load_1[4], _tmem_load_1[5]));
                        up_packed[2] = __as_u32(_bf16x2_13);
                        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                        gate_packed[3] = __as_u32(_bf16x2_14);
                        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_tmem_load_1[6], _tmem_load_1[7]));
                        up_packed[3] = __as_u32(_bf16x2_15);
                        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                        gate_packed[4] = __as_u32(_bf16x2_16);
                        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_tmem_load_1[8], _tmem_load_1[9]));
                        up_packed[4] = __as_u32(_bf16x2_17);
                        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                        gate_packed[5] = __as_u32(_bf16x2_18);
                        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_tmem_load_1[10], _tmem_load_1[11]));
                        up_packed[5] = __as_u32(_bf16x2_19);
                        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                        gate_packed[6] = __as_u32(_bf16x2_20);
                        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_tmem_load_1[12], _tmem_load_1[13]));
                        up_packed[6] = __as_u32(_bf16x2_21);
                        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                        gate_packed[7] = __as_u32(_bf16x2_22);
                        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_tmem_load_1[14], _tmem_load_1[15]));
                        up_packed[7] = __as_u32(_bf16x2_23);
                        unsigned int address_0 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16);
                        float _tmem_load_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                            : "r"(address_0));
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(address_0 + 256));
                        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_tmem_load_2[0], _tmem_load_2[1]));
                        gate_packed[8] = __as_u32(_bf16x2_24);
                        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_tmem_load_3[0], _tmem_load_3[1]));
                        up_packed[8] = __as_u32(_bf16x2_25);
                        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_tmem_load_2[2], _tmem_load_2[3]));
                        gate_packed[9] = __as_u32(_bf16x2_26);
                        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_tmem_load_3[2], _tmem_load_3[3]));
                        up_packed[9] = __as_u32(_bf16x2_27);
                        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_tmem_load_2[4], _tmem_load_2[5]));
                        gate_packed[10] = __as_u32(_bf16x2_28);
                        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_tmem_load_3[4], _tmem_load_3[5]));
                        up_packed[10] = __as_u32(_bf16x2_29);
                        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_tmem_load_2[6], _tmem_load_2[7]));
                        gate_packed[11] = __as_u32(_bf16x2_30);
                        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_tmem_load_3[6], _tmem_load_3[7]));
                        up_packed[11] = __as_u32(_bf16x2_31);
                        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_tmem_load_2[8], _tmem_load_2[9]));
                        gate_packed[12] = __as_u32(_bf16x2_32);
                        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_tmem_load_3[8], _tmem_load_3[9]));
                        up_packed[12] = __as_u32(_bf16x2_33);
                        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_tmem_load_2[10], _tmem_load_2[11]));
                        gate_packed[13] = __as_u32(_bf16x2_34);
                        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_tmem_load_3[10], _tmem_load_3[11]));
                        up_packed[13] = __as_u32(_bf16x2_35);
                        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(_tmem_load_2[12], _tmem_load_2[13]));
                        gate_packed[14] = __as_u32(_bf16x2_36);
                        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(_tmem_load_3[12], _tmem_load_3[13]));
                        up_packed[14] = __as_u32(_bf16x2_37);
                        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(_tmem_load_2[14], _tmem_load_2[15]));
                        gate_packed[15] = __as_u32(_bf16x2_38);
                        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(_tmem_load_3[14], _tmem_load_3[15]));
                        up_packed[15] = __as_u32(_bf16x2_39);
                        unsigned int address_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 32;
                        float _tmem_load_4[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                            : "r"(address_1));
                        float _tmem_load_5[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                            : "r"(address_1 + 256));
                        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(_tmem_load_4[0], _tmem_load_4[1]));
                        gate_packed[16] = __as_u32(_bf16x2_40);
                        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(_tmem_load_5[0], _tmem_load_5[1]));
                        up_packed[16] = __as_u32(_bf16x2_41);
                        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(_tmem_load_4[2], _tmem_load_4[3]));
                        gate_packed[17] = __as_u32(_bf16x2_42);
                        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(_tmem_load_5[2], _tmem_load_5[3]));
                        up_packed[17] = __as_u32(_bf16x2_43);
                        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_tmem_load_4[4], _tmem_load_4[5]));
                        gate_packed[18] = __as_u32(_bf16x2_44);
                        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_tmem_load_5[4], _tmem_load_5[5]));
                        up_packed[18] = __as_u32(_bf16x2_45);
                        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(_tmem_load_4[6], _tmem_load_4[7]));
                        gate_packed[19] = __as_u32(_bf16x2_46);
                        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(_tmem_load_5[6], _tmem_load_5[7]));
                        up_packed[19] = __as_u32(_bf16x2_47);
                        __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(_tmem_load_4[8], _tmem_load_4[9]));
                        gate_packed[20] = __as_u32(_bf16x2_48);
                        __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(_tmem_load_5[8], _tmem_load_5[9]));
                        up_packed[20] = __as_u32(_bf16x2_49);
                        __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(_tmem_load_4[10], _tmem_load_4[11]));
                        gate_packed[21] = __as_u32(_bf16x2_50);
                        __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(_tmem_load_5[10], _tmem_load_5[11]));
                        up_packed[21] = __as_u32(_bf16x2_51);
                        __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(_tmem_load_4[12], _tmem_load_4[13]));
                        gate_packed[22] = __as_u32(_bf16x2_52);
                        __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(_tmem_load_5[12], _tmem_load_5[13]));
                        up_packed[22] = __as_u32(_bf16x2_53);
                        __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(_tmem_load_4[14], _tmem_load_4[15]));
                        gate_packed[23] = __as_u32(_bf16x2_54);
                        __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(_tmem_load_5[14], _tmem_load_5[15]));
                        up_packed[23] = __as_u32(_bf16x2_55);
                        unsigned int address_2 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 32;
                        float _tmem_load_6[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                            : "r"(address_2));
                        float _tmem_load_7[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                            : "r"(address_2 + 256));
                        __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(_tmem_load_6[0], _tmem_load_6[1]));
                        gate_packed[24] = __as_u32(_bf16x2_56);
                        __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(_tmem_load_7[0], _tmem_load_7[1]));
                        up_packed[24] = __as_u32(_bf16x2_57);
                        __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(_tmem_load_6[2], _tmem_load_6[3]));
                        gate_packed[25] = __as_u32(_bf16x2_58);
                        __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(_tmem_load_7[2], _tmem_load_7[3]));
                        up_packed[25] = __as_u32(_bf16x2_59);
                        __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(_tmem_load_6[4], _tmem_load_6[5]));
                        gate_packed[26] = __as_u32(_bf16x2_60);
                        __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(_tmem_load_7[4], _tmem_load_7[5]));
                        up_packed[26] = __as_u32(_bf16x2_61);
                        __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(_tmem_load_6[6], _tmem_load_6[7]));
                        gate_packed[27] = __as_u32(_bf16x2_62);
                        __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(_tmem_load_7[6], _tmem_load_7[7]));
                        up_packed[27] = __as_u32(_bf16x2_63);
                        __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(_tmem_load_6[8], _tmem_load_6[9]));
                        gate_packed[28] = __as_u32(_bf16x2_64);
                        __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(_tmem_load_7[8], _tmem_load_7[9]));
                        up_packed[28] = __as_u32(_bf16x2_65);
                        __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(_tmem_load_6[10], _tmem_load_6[11]));
                        gate_packed[29] = __as_u32(_bf16x2_66);
                        __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(_tmem_load_7[10], _tmem_load_7[11]));
                        up_packed[29] = __as_u32(_bf16x2_67);
                        __nv_bfloat162 _bf16x2_68 = __float22bfloat162_rn(make_float2(_tmem_load_6[12], _tmem_load_6[13]));
                        gate_packed[30] = __as_u32(_bf16x2_68);
                        __nv_bfloat162 _bf16x2_69 = __float22bfloat162_rn(make_float2(_tmem_load_7[12], _tmem_load_7[13]));
                        up_packed[30] = __as_u32(_bf16x2_69);
                        __nv_bfloat162 _bf16x2_70 = __float22bfloat162_rn(make_float2(_tmem_load_6[14], _tmem_load_6[15]));
                        gate_packed[31] = __as_u32(_bf16x2_70);
                        __nv_bfloat162 _bf16x2_71 = __float22bfloat162_rn(make_float2(_tmem_load_7[14], _tmem_load_7[15]));
                        up_packed[31] = __as_u32(_bf16x2_71);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(gate_packed[0]));
                        float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(up_packed[0]));
                        float gate_x = _cvt_f32_0.x;
                        float gate_y = _cvt_f32_0.y;
                        float up_x = _cvt_f32_1.x;
                        float up_y = _cvt_f32_1.y;
                        float _min_26 = fminf(gate_x, swiglu_limit);
                        gate_x = _min_26;
                        float _min_27 = fminf(gate_y, swiglu_limit);
                        gate_y = _min_27;
                        float _fmax_16 = fmaxf(up_x, -swiglu_limit);
                        float _min_28 = fminf(_fmax_16, swiglu_limit);
                        up_x = _min_28;
                        float _fmax_17 = fmaxf(up_y, -swiglu_limit);
                        float _min_29 = fminf(_fmax_17, swiglu_limit);
                        up_y = _min_29;
                        float _exp_0 = expf(gate_x * -1.0f);
                        float denominator_x = _exp_0 + 1.0f;
                        float _exp_1 = expf(gate_y * -1.0f);
                        float denominator_y = _exp_1 + 1.0f;
                        float hidden_x = gate_x / denominator_x * up_x;
                        float hidden_y = gate_y / denominator_y * up_y;
                        __nv_bfloat162 _bf16x2_72 = __float22bfloat162_rn(make_float2(hidden_x, hidden_y));
                        hidden_packed[0] = __as_u32(_bf16x2_72);
                        float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(gate_packed[1]));
                        float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(up_packed[1]));
                        float gate_x_3 = _cvt_f32_2.x;
                        float gate_y_4 = _cvt_f32_2.y;
                        float up_x_5 = _cvt_f32_3.x;
                        float up_y_6 = _cvt_f32_3.y;
                        float _min_30 = fminf(gate_x_3, swiglu_limit);
                        gate_x_3 = _min_30;
                        float _min_31 = fminf(gate_y_4, swiglu_limit);
                        gate_y_4 = _min_31;
                        float _fmax_18 = fmaxf(up_x_5, -swiglu_limit);
                        float _min_32 = fminf(_fmax_18, swiglu_limit);
                        up_x_5 = _min_32;
                        float _fmax_19 = fmaxf(up_y_6, -swiglu_limit);
                        float _min_33 = fminf(_fmax_19, swiglu_limit);
                        up_y_6 = _min_33;
                        float _exp_2 = expf(gate_x_3 * -1.0f);
                        float denominator_x_7 = _exp_2 + 1.0f;
                        float _exp_3 = expf(gate_y_4 * -1.0f);
                        float denominator_y_8 = _exp_3 + 1.0f;
                        float hidden_x_9 = gate_x_3 / denominator_x_7 * up_x_5;
                        float hidden_y_10 = gate_y_4 / denominator_y_8 * up_y_6;
                        __nv_bfloat162 _bf16x2_73 = __float22bfloat162_rn(make_float2(hidden_x_9, hidden_y_10));
                        hidden_packed[1] = __as_u32(_bf16x2_73);
                        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(gate_packed[2]));
                        float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(up_packed[2]));
                        float gate_x_11 = _cvt_f32_4.x;
                        float gate_y_12 = _cvt_f32_4.y;
                        float up_x_13 = _cvt_f32_5.x;
                        float up_y_14 = _cvt_f32_5.y;
                        float _min_34 = fminf(gate_x_11, swiglu_limit);
                        gate_x_11 = _min_34;
                        float _min_35 = fminf(gate_y_12, swiglu_limit);
                        gate_y_12 = _min_35;
                        float _fmax_20 = fmaxf(up_x_13, -swiglu_limit);
                        float _min_36 = fminf(_fmax_20, swiglu_limit);
                        up_x_13 = _min_36;
                        float _fmax_21 = fmaxf(up_y_14, -swiglu_limit);
                        float _min_37 = fminf(_fmax_21, swiglu_limit);
                        up_y_14 = _min_37;
                        float _exp_4 = expf(gate_x_11 * -1.0f);
                        float denominator_x_15 = _exp_4 + 1.0f;
                        float _exp_5 = expf(gate_y_12 * -1.0f);
                        float denominator_y_16 = _exp_5 + 1.0f;
                        float hidden_x_17 = gate_x_11 / denominator_x_15 * up_x_13;
                        float hidden_y_18 = gate_y_12 / denominator_y_16 * up_y_14;
                        __nv_bfloat162 _bf16x2_74 = __float22bfloat162_rn(make_float2(hidden_x_17, hidden_y_18));
                        hidden_packed[2] = __as_u32(_bf16x2_74);
                        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(gate_packed[3]));
                        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(up_packed[3]));
                        float gate_x_19 = _cvt_f32_6.x;
                        float gate_y_20 = _cvt_f32_6.y;
                        float up_x_21 = _cvt_f32_7.x;
                        float up_y_22 = _cvt_f32_7.y;
                        float _min_38 = fminf(gate_x_19, swiglu_limit);
                        gate_x_19 = _min_38;
                        float _min_39 = fminf(gate_y_20, swiglu_limit);
                        gate_y_20 = _min_39;
                        float _fmax_22 = fmaxf(up_x_21, -swiglu_limit);
                        float _min_40 = fminf(_fmax_22, swiglu_limit);
                        up_x_21 = _min_40;
                        float _fmax_23 = fmaxf(up_y_22, -swiglu_limit);
                        float _min_41 = fminf(_fmax_23, swiglu_limit);
                        up_y_22 = _min_41;
                        float _exp_6 = expf(gate_x_19 * -1.0f);
                        float denominator_x_23 = _exp_6 + 1.0f;
                        float _exp_7 = expf(gate_y_20 * -1.0f);
                        float denominator_y_24 = _exp_7 + 1.0f;
                        float hidden_x_25 = gate_x_19 / denominator_x_23 * up_x_21;
                        float hidden_y_26 = gate_y_20 / denominator_y_24 * up_y_22;
                        __nv_bfloat162 _bf16x2_75 = __float22bfloat162_rn(make_float2(hidden_x_25, hidden_y_26));
                        hidden_packed[3] = __as_u32(_bf16x2_75);
                        float2 _cvt_f32_8 = __bfloat1622float2(__as_bf16x2(gate_packed[4]));
                        float2 _cvt_f32_9 = __bfloat1622float2(__as_bf16x2(up_packed[4]));
                        float gate_x_27 = _cvt_f32_8.x;
                        float gate_y_28 = _cvt_f32_8.y;
                        float up_x_29 = _cvt_f32_9.x;
                        float up_y_30 = _cvt_f32_9.y;
                        float _min_42 = fminf(gate_x_27, swiglu_limit);
                        gate_x_27 = _min_42;
                        float _min_43 = fminf(gate_y_28, swiglu_limit);
                        gate_y_28 = _min_43;
                        float _fmax_24 = fmaxf(up_x_29, -swiglu_limit);
                        float _min_44 = fminf(_fmax_24, swiglu_limit);
                        up_x_29 = _min_44;
                        float _fmax_25 = fmaxf(up_y_30, -swiglu_limit);
                        float _min_45 = fminf(_fmax_25, swiglu_limit);
                        up_y_30 = _min_45;
                        float _exp_8 = expf(gate_x_27 * -1.0f);
                        float denominator_x_31 = _exp_8 + 1.0f;
                        float _exp_9 = expf(gate_y_28 * -1.0f);
                        float denominator_y_32 = _exp_9 + 1.0f;
                        float hidden_x_33 = gate_x_27 / denominator_x_31 * up_x_29;
                        float hidden_y_34 = gate_y_28 / denominator_y_32 * up_y_30;
                        __nv_bfloat162 _bf16x2_76 = __float22bfloat162_rn(make_float2(hidden_x_33, hidden_y_34));
                        hidden_packed[4] = __as_u32(_bf16x2_76);
                        float2 _cvt_f32_10 = __bfloat1622float2(__as_bf16x2(gate_packed[5]));
                        float2 _cvt_f32_11 = __bfloat1622float2(__as_bf16x2(up_packed[5]));
                        float gate_x_35 = _cvt_f32_10.x;
                        float gate_y_36 = _cvt_f32_10.y;
                        float up_x_37 = _cvt_f32_11.x;
                        float up_y_38 = _cvt_f32_11.y;
                        float _min_46 = fminf(gate_x_35, swiglu_limit);
                        gate_x_35 = _min_46;
                        float _min_47 = fminf(gate_y_36, swiglu_limit);
                        gate_y_36 = _min_47;
                        float _fmax_26 = fmaxf(up_x_37, -swiglu_limit);
                        float _min_48 = fminf(_fmax_26, swiglu_limit);
                        up_x_37 = _min_48;
                        float _fmax_27 = fmaxf(up_y_38, -swiglu_limit);
                        float _min_49 = fminf(_fmax_27, swiglu_limit);
                        up_y_38 = _min_49;
                        float _exp_10 = expf(gate_x_35 * -1.0f);
                        float denominator_x_39 = _exp_10 + 1.0f;
                        float _exp_11 = expf(gate_y_36 * -1.0f);
                        float denominator_y_40 = _exp_11 + 1.0f;
                        float hidden_x_41 = gate_x_35 / denominator_x_39 * up_x_37;
                        float hidden_y_42 = gate_y_36 / denominator_y_40 * up_y_38;
                        __nv_bfloat162 _bf16x2_77 = __float22bfloat162_rn(make_float2(hidden_x_41, hidden_y_42));
                        hidden_packed[5] = __as_u32(_bf16x2_77);
                        float2 _cvt_f32_12 = __bfloat1622float2(__as_bf16x2(gate_packed[6]));
                        float2 _cvt_f32_13 = __bfloat1622float2(__as_bf16x2(up_packed[6]));
                        float gate_x_43 = _cvt_f32_12.x;
                        float gate_y_44 = _cvt_f32_12.y;
                        float up_x_45 = _cvt_f32_13.x;
                        float up_y_46 = _cvt_f32_13.y;
                        float _min_50 = fminf(gate_x_43, swiglu_limit);
                        gate_x_43 = _min_50;
                        float _min_51 = fminf(gate_y_44, swiglu_limit);
                        gate_y_44 = _min_51;
                        float _fmax_28 = fmaxf(up_x_45, -swiglu_limit);
                        float _min_52 = fminf(_fmax_28, swiglu_limit);
                        up_x_45 = _min_52;
                        float _fmax_29 = fmaxf(up_y_46, -swiglu_limit);
                        float _min_53 = fminf(_fmax_29, swiglu_limit);
                        up_y_46 = _min_53;
                        float _exp_12 = expf(gate_x_43 * -1.0f);
                        float denominator_x_47 = _exp_12 + 1.0f;
                        float _exp_13 = expf(gate_y_44 * -1.0f);
                        float denominator_y_48 = _exp_13 + 1.0f;
                        float hidden_x_49 = gate_x_43 / denominator_x_47 * up_x_45;
                        float hidden_y_50 = gate_y_44 / denominator_y_48 * up_y_46;
                        __nv_bfloat162 _bf16x2_78 = __float22bfloat162_rn(make_float2(hidden_x_49, hidden_y_50));
                        hidden_packed[6] = __as_u32(_bf16x2_78);
                        float2 _cvt_f32_14 = __bfloat1622float2(__as_bf16x2(gate_packed[7]));
                        float2 _cvt_f32_15 = __bfloat1622float2(__as_bf16x2(up_packed[7]));
                        float gate_x_51 = _cvt_f32_14.x;
                        float gate_y_52 = _cvt_f32_14.y;
                        float up_x_53 = _cvt_f32_15.x;
                        float up_y_54 = _cvt_f32_15.y;
                        float _min_54 = fminf(gate_x_51, swiglu_limit);
                        gate_x_51 = _min_54;
                        float _min_55 = fminf(gate_y_52, swiglu_limit);
                        gate_y_52 = _min_55;
                        float _fmax_30 = fmaxf(up_x_53, -swiglu_limit);
                        float _min_56 = fminf(_fmax_30, swiglu_limit);
                        up_x_53 = _min_56;
                        float _fmax_31 = fmaxf(up_y_54, -swiglu_limit);
                        float _min_57 = fminf(_fmax_31, swiglu_limit);
                        up_y_54 = _min_57;
                        float _exp_14 = expf(gate_x_51 * -1.0f);
                        float denominator_x_55 = _exp_14 + 1.0f;
                        float _exp_15 = expf(gate_y_52 * -1.0f);
                        float denominator_y_56 = _exp_15 + 1.0f;
                        float hidden_x_57 = gate_x_51 / denominator_x_55 * up_x_53;
                        float hidden_y_58 = gate_y_52 / denominator_y_56 * up_y_54;
                        __nv_bfloat162 _bf16x2_79 = __float22bfloat162_rn(make_float2(hidden_x_57, hidden_y_58));
                        hidden_packed[7] = __as_u32(_bf16x2_79);
                        float2 _cvt_f32_16 = __bfloat1622float2(__as_bf16x2(gate_packed[8]));
                        float2 _cvt_f32_17 = __bfloat1622float2(__as_bf16x2(up_packed[8]));
                        float gate_x_59 = _cvt_f32_16.x;
                        float gate_y_60 = _cvt_f32_16.y;
                        float up_x_61 = _cvt_f32_17.x;
                        float up_y_62 = _cvt_f32_17.y;
                        float _min_58 = fminf(gate_x_59, swiglu_limit);
                        gate_x_59 = _min_58;
                        float _min_59 = fminf(gate_y_60, swiglu_limit);
                        gate_y_60 = _min_59;
                        float _fmax_32 = fmaxf(up_x_61, -swiglu_limit);
                        float _min_60 = fminf(_fmax_32, swiglu_limit);
                        up_x_61 = _min_60;
                        float _fmax_33 = fmaxf(up_y_62, -swiglu_limit);
                        float _min_61 = fminf(_fmax_33, swiglu_limit);
                        up_y_62 = _min_61;
                        float _exp_16 = expf(gate_x_59 * -1.0f);
                        float denominator_x_63 = _exp_16 + 1.0f;
                        float _exp_17 = expf(gate_y_60 * -1.0f);
                        float denominator_y_64 = _exp_17 + 1.0f;
                        float hidden_x_65 = gate_x_59 / denominator_x_63 * up_x_61;
                        float hidden_y_66 = gate_y_60 / denominator_y_64 * up_y_62;
                        __nv_bfloat162 _bf16x2_80 = __float22bfloat162_rn(make_float2(hidden_x_65, hidden_y_66));
                        hidden_packed[8] = __as_u32(_bf16x2_80);
                        float2 _cvt_f32_18 = __bfloat1622float2(__as_bf16x2(gate_packed[9]));
                        float2 _cvt_f32_19 = __bfloat1622float2(__as_bf16x2(up_packed[9]));
                        float gate_x_67 = _cvt_f32_18.x;
                        float gate_y_68 = _cvt_f32_18.y;
                        float up_x_69 = _cvt_f32_19.x;
                        float up_y_70 = _cvt_f32_19.y;
                        float _min_62 = fminf(gate_x_67, swiglu_limit);
                        gate_x_67 = _min_62;
                        float _min_63 = fminf(gate_y_68, swiglu_limit);
                        gate_y_68 = _min_63;
                        float _fmax_34 = fmaxf(up_x_69, -swiglu_limit);
                        float _min_64 = fminf(_fmax_34, swiglu_limit);
                        up_x_69 = _min_64;
                        float _fmax_35 = fmaxf(up_y_70, -swiglu_limit);
                        float _min_65 = fminf(_fmax_35, swiglu_limit);
                        up_y_70 = _min_65;
                        float _exp_18 = expf(gate_x_67 * -1.0f);
                        float denominator_x_71 = _exp_18 + 1.0f;
                        float _exp_19 = expf(gate_y_68 * -1.0f);
                        float denominator_y_72 = _exp_19 + 1.0f;
                        float hidden_x_73 = gate_x_67 / denominator_x_71 * up_x_69;
                        float hidden_y_74 = gate_y_68 / denominator_y_72 * up_y_70;
                        __nv_bfloat162 _bf16x2_81 = __float22bfloat162_rn(make_float2(hidden_x_73, hidden_y_74));
                        hidden_packed[9] = __as_u32(_bf16x2_81);
                        float2 _cvt_f32_20 = __bfloat1622float2(__as_bf16x2(gate_packed[10]));
                        float2 _cvt_f32_21 = __bfloat1622float2(__as_bf16x2(up_packed[10]));
                        float gate_x_75 = _cvt_f32_20.x;
                        float gate_y_76 = _cvt_f32_20.y;
                        float up_x_77 = _cvt_f32_21.x;
                        float up_y_78 = _cvt_f32_21.y;
                        float _min_66 = fminf(gate_x_75, swiglu_limit);
                        gate_x_75 = _min_66;
                        float _min_67 = fminf(gate_y_76, swiglu_limit);
                        gate_y_76 = _min_67;
                        float _fmax_36 = fmaxf(up_x_77, -swiglu_limit);
                        float _min_68 = fminf(_fmax_36, swiglu_limit);
                        up_x_77 = _min_68;
                        float _fmax_37 = fmaxf(up_y_78, -swiglu_limit);
                        float _min_69 = fminf(_fmax_37, swiglu_limit);
                        up_y_78 = _min_69;
                        float _exp_20 = expf(gate_x_75 * -1.0f);
                        float denominator_x_79 = _exp_20 + 1.0f;
                        float _exp_21 = expf(gate_y_76 * -1.0f);
                        float denominator_y_80 = _exp_21 + 1.0f;
                        float hidden_x_81 = gate_x_75 / denominator_x_79 * up_x_77;
                        float hidden_y_82 = gate_y_76 / denominator_y_80 * up_y_78;
                        __nv_bfloat162 _bf16x2_82 = __float22bfloat162_rn(make_float2(hidden_x_81, hidden_y_82));
                        hidden_packed[10] = __as_u32(_bf16x2_82);
                        float2 _cvt_f32_22 = __bfloat1622float2(__as_bf16x2(gate_packed[11]));
                        float2 _cvt_f32_23 = __bfloat1622float2(__as_bf16x2(up_packed[11]));
                        float gate_x_83 = _cvt_f32_22.x;
                        float gate_y_84 = _cvt_f32_22.y;
                        float up_x_85 = _cvt_f32_23.x;
                        float up_y_86 = _cvt_f32_23.y;
                        float _min_70 = fminf(gate_x_83, swiglu_limit);
                        gate_x_83 = _min_70;
                        float _min_71 = fminf(gate_y_84, swiglu_limit);
                        gate_y_84 = _min_71;
                        float _fmax_38 = fmaxf(up_x_85, -swiglu_limit);
                        float _min_72 = fminf(_fmax_38, swiglu_limit);
                        up_x_85 = _min_72;
                        float _fmax_39 = fmaxf(up_y_86, -swiglu_limit);
                        float _min_73 = fminf(_fmax_39, swiglu_limit);
                        up_y_86 = _min_73;
                        float _exp_22 = expf(gate_x_83 * -1.0f);
                        float denominator_x_87 = _exp_22 + 1.0f;
                        float _exp_23 = expf(gate_y_84 * -1.0f);
                        float denominator_y_88 = _exp_23 + 1.0f;
                        float hidden_x_89 = gate_x_83 / denominator_x_87 * up_x_85;
                        float hidden_y_90 = gate_y_84 / denominator_y_88 * up_y_86;
                        __nv_bfloat162 _bf16x2_83 = __float22bfloat162_rn(make_float2(hidden_x_89, hidden_y_90));
                        hidden_packed[11] = __as_u32(_bf16x2_83);
                        float2 _cvt_f32_24 = __bfloat1622float2(__as_bf16x2(gate_packed[12]));
                        float2 _cvt_f32_25 = __bfloat1622float2(__as_bf16x2(up_packed[12]));
                        float gate_x_91 = _cvt_f32_24.x;
                        float gate_y_92 = _cvt_f32_24.y;
                        float up_x_93 = _cvt_f32_25.x;
                        float up_y_94 = _cvt_f32_25.y;
                        float _min_74 = fminf(gate_x_91, swiglu_limit);
                        gate_x_91 = _min_74;
                        float _min_75 = fminf(gate_y_92, swiglu_limit);
                        gate_y_92 = _min_75;
                        float _fmax_40 = fmaxf(up_x_93, -swiglu_limit);
                        float _min_76 = fminf(_fmax_40, swiglu_limit);
                        up_x_93 = _min_76;
                        float _fmax_41 = fmaxf(up_y_94, -swiglu_limit);
                        float _min_77 = fminf(_fmax_41, swiglu_limit);
                        up_y_94 = _min_77;
                        float _exp_24 = expf(gate_x_91 * -1.0f);
                        float denominator_x_95 = _exp_24 + 1.0f;
                        float _exp_25 = expf(gate_y_92 * -1.0f);
                        float denominator_y_96 = _exp_25 + 1.0f;
                        float hidden_x_97 = gate_x_91 / denominator_x_95 * up_x_93;
                        float hidden_y_98 = gate_y_92 / denominator_y_96 * up_y_94;
                        __nv_bfloat162 _bf16x2_84 = __float22bfloat162_rn(make_float2(hidden_x_97, hidden_y_98));
                        hidden_packed[12] = __as_u32(_bf16x2_84);
                        float2 _cvt_f32_26 = __bfloat1622float2(__as_bf16x2(gate_packed[13]));
                        float2 _cvt_f32_27 = __bfloat1622float2(__as_bf16x2(up_packed[13]));
                        float gate_x_99 = _cvt_f32_26.x;
                        float gate_y_100 = _cvt_f32_26.y;
                        float up_x_101 = _cvt_f32_27.x;
                        float up_y_102 = _cvt_f32_27.y;
                        float _min_78 = fminf(gate_x_99, swiglu_limit);
                        gate_x_99 = _min_78;
                        float _min_79 = fminf(gate_y_100, swiglu_limit);
                        gate_y_100 = _min_79;
                        float _fmax_42 = fmaxf(up_x_101, -swiglu_limit);
                        float _min_80 = fminf(_fmax_42, swiglu_limit);
                        up_x_101 = _min_80;
                        float _fmax_43 = fmaxf(up_y_102, -swiglu_limit);
                        float _min_81 = fminf(_fmax_43, swiglu_limit);
                        up_y_102 = _min_81;
                        float _exp_26 = expf(gate_x_99 * -1.0f);
                        float denominator_x_103 = _exp_26 + 1.0f;
                        float _exp_27 = expf(gate_y_100 * -1.0f);
                        float denominator_y_104 = _exp_27 + 1.0f;
                        float hidden_x_105 = gate_x_99 / denominator_x_103 * up_x_101;
                        float hidden_y_106 = gate_y_100 / denominator_y_104 * up_y_102;
                        __nv_bfloat162 _bf16x2_85 = __float22bfloat162_rn(make_float2(hidden_x_105, hidden_y_106));
                        hidden_packed[13] = __as_u32(_bf16x2_85);
                        float2 _cvt_f32_28 = __bfloat1622float2(__as_bf16x2(gate_packed[14]));
                        float2 _cvt_f32_29 = __bfloat1622float2(__as_bf16x2(up_packed[14]));
                        float gate_x_107 = _cvt_f32_28.x;
                        float gate_y_108 = _cvt_f32_28.y;
                        float up_x_109 = _cvt_f32_29.x;
                        float up_y_110 = _cvt_f32_29.y;
                        float _min_82 = fminf(gate_x_107, swiglu_limit);
                        gate_x_107 = _min_82;
                        float _min_83 = fminf(gate_y_108, swiglu_limit);
                        gate_y_108 = _min_83;
                        float _fmax_44 = fmaxf(up_x_109, -swiglu_limit);
                        float _min_84 = fminf(_fmax_44, swiglu_limit);
                        up_x_109 = _min_84;
                        float _fmax_45 = fmaxf(up_y_110, -swiglu_limit);
                        float _min_85 = fminf(_fmax_45, swiglu_limit);
                        up_y_110 = _min_85;
                        float _exp_28 = expf(gate_x_107 * -1.0f);
                        float denominator_x_111 = _exp_28 + 1.0f;
                        float _exp_29 = expf(gate_y_108 * -1.0f);
                        float denominator_y_112 = _exp_29 + 1.0f;
                        float hidden_x_113 = gate_x_107 / denominator_x_111 * up_x_109;
                        float hidden_y_114 = gate_y_108 / denominator_y_112 * up_y_110;
                        __nv_bfloat162 _bf16x2_86 = __float22bfloat162_rn(make_float2(hidden_x_113, hidden_y_114));
                        hidden_packed[14] = __as_u32(_bf16x2_86);
                        float2 _cvt_f32_30 = __bfloat1622float2(__as_bf16x2(gate_packed[15]));
                        float2 _cvt_f32_31 = __bfloat1622float2(__as_bf16x2(up_packed[15]));
                        float gate_x_115 = _cvt_f32_30.x;
                        float gate_y_116 = _cvt_f32_30.y;
                        float up_x_117 = _cvt_f32_31.x;
                        float up_y_118 = _cvt_f32_31.y;
                        float _min_86 = fminf(gate_x_115, swiglu_limit);
                        gate_x_115 = _min_86;
                        float _min_87 = fminf(gate_y_116, swiglu_limit);
                        gate_y_116 = _min_87;
                        float _fmax_46 = fmaxf(up_x_117, -swiglu_limit);
                        float _min_88 = fminf(_fmax_46, swiglu_limit);
                        up_x_117 = _min_88;
                        float _fmax_47 = fmaxf(up_y_118, -swiglu_limit);
                        float _min_89 = fminf(_fmax_47, swiglu_limit);
                        up_y_118 = _min_89;
                        float _exp_30 = expf(gate_x_115 * -1.0f);
                        float denominator_x_119 = _exp_30 + 1.0f;
                        float _exp_31 = expf(gate_y_116 * -1.0f);
                        float denominator_y_120 = _exp_31 + 1.0f;
                        float hidden_x_121 = gate_x_115 / denominator_x_119 * up_x_117;
                        float hidden_y_122 = gate_y_116 / denominator_y_120 * up_y_118;
                        __nv_bfloat162 _bf16x2_87 = __float22bfloat162_rn(make_float2(hidden_x_121, hidden_y_122));
                        hidden_packed[15] = __as_u32(_bf16x2_87);
                        float2 _cvt_f32_32 = __bfloat1622float2(__as_bf16x2(gate_packed[16]));
                        float2 _cvt_f32_33 = __bfloat1622float2(__as_bf16x2(up_packed[16]));
                        float gate_x_123 = _cvt_f32_32.x;
                        float gate_y_124 = _cvt_f32_32.y;
                        float up_x_125 = _cvt_f32_33.x;
                        float up_y_126 = _cvt_f32_33.y;
                        float _min_90 = fminf(gate_x_123, swiglu_limit);
                        gate_x_123 = _min_90;
                        float _min_91 = fminf(gate_y_124, swiglu_limit);
                        gate_y_124 = _min_91;
                        float _fmax_48 = fmaxf(up_x_125, -swiglu_limit);
                        float _min_92 = fminf(_fmax_48, swiglu_limit);
                        up_x_125 = _min_92;
                        float _fmax_49 = fmaxf(up_y_126, -swiglu_limit);
                        float _min_93 = fminf(_fmax_49, swiglu_limit);
                        up_y_126 = _min_93;
                        float _exp_32 = expf(gate_x_123 * -1.0f);
                        float denominator_x_127 = _exp_32 + 1.0f;
                        float _exp_33 = expf(gate_y_124 * -1.0f);
                        float denominator_y_128 = _exp_33 + 1.0f;
                        float hidden_x_129 = gate_x_123 / denominator_x_127 * up_x_125;
                        float hidden_y_130 = gate_y_124 / denominator_y_128 * up_y_126;
                        __nv_bfloat162 _bf16x2_88 = __float22bfloat162_rn(make_float2(hidden_x_129, hidden_y_130));
                        hidden_packed[16] = __as_u32(_bf16x2_88);
                        float2 _cvt_f32_34 = __bfloat1622float2(__as_bf16x2(gate_packed[17]));
                        float2 _cvt_f32_35 = __bfloat1622float2(__as_bf16x2(up_packed[17]));
                        float gate_x_131 = _cvt_f32_34.x;
                        float gate_y_132 = _cvt_f32_34.y;
                        float up_x_133 = _cvt_f32_35.x;
                        float up_y_134 = _cvt_f32_35.y;
                        float _min_94 = fminf(gate_x_131, swiglu_limit);
                        gate_x_131 = _min_94;
                        float _min_95 = fminf(gate_y_132, swiglu_limit);
                        gate_y_132 = _min_95;
                        float _fmax_50 = fmaxf(up_x_133, -swiglu_limit);
                        float _min_96 = fminf(_fmax_50, swiglu_limit);
                        up_x_133 = _min_96;
                        float _fmax_51 = fmaxf(up_y_134, -swiglu_limit);
                        float _min_97 = fminf(_fmax_51, swiglu_limit);
                        up_y_134 = _min_97;
                        float _exp_34 = expf(gate_x_131 * -1.0f);
                        float denominator_x_135 = _exp_34 + 1.0f;
                        float _exp_35 = expf(gate_y_132 * -1.0f);
                        float denominator_y_136 = _exp_35 + 1.0f;
                        float hidden_x_137 = gate_x_131 / denominator_x_135 * up_x_133;
                        float hidden_y_138 = gate_y_132 / denominator_y_136 * up_y_134;
                        __nv_bfloat162 _bf16x2_89 = __float22bfloat162_rn(make_float2(hidden_x_137, hidden_y_138));
                        hidden_packed[17] = __as_u32(_bf16x2_89);
                        float2 _cvt_f32_36 = __bfloat1622float2(__as_bf16x2(gate_packed[18]));
                        float2 _cvt_f32_37 = __bfloat1622float2(__as_bf16x2(up_packed[18]));
                        float gate_x_139 = _cvt_f32_36.x;
                        float gate_y_140 = _cvt_f32_36.y;
                        float up_x_141 = _cvt_f32_37.x;
                        float up_y_142 = _cvt_f32_37.y;
                        float _min_98 = fminf(gate_x_139, swiglu_limit);
                        gate_x_139 = _min_98;
                        float _min_99 = fminf(gate_y_140, swiglu_limit);
                        gate_y_140 = _min_99;
                        float _fmax_52 = fmaxf(up_x_141, -swiglu_limit);
                        float _min_100 = fminf(_fmax_52, swiglu_limit);
                        up_x_141 = _min_100;
                        float _fmax_53 = fmaxf(up_y_142, -swiglu_limit);
                        float _min_101 = fminf(_fmax_53, swiglu_limit);
                        up_y_142 = _min_101;
                        float _exp_36 = expf(gate_x_139 * -1.0f);
                        float denominator_x_143 = _exp_36 + 1.0f;
                        float _exp_37 = expf(gate_y_140 * -1.0f);
                        float denominator_y_144 = _exp_37 + 1.0f;
                        float hidden_x_145 = gate_x_139 / denominator_x_143 * up_x_141;
                        float hidden_y_146 = gate_y_140 / denominator_y_144 * up_y_142;
                        __nv_bfloat162 _bf16x2_90 = __float22bfloat162_rn(make_float2(hidden_x_145, hidden_y_146));
                        hidden_packed[18] = __as_u32(_bf16x2_90);
                        float2 _cvt_f32_38 = __bfloat1622float2(__as_bf16x2(gate_packed[19]));
                        float2 _cvt_f32_39 = __bfloat1622float2(__as_bf16x2(up_packed[19]));
                        float gate_x_147 = _cvt_f32_38.x;
                        float gate_y_148 = _cvt_f32_38.y;
                        float up_x_149 = _cvt_f32_39.x;
                        float up_y_150 = _cvt_f32_39.y;
                        float _min_102 = fminf(gate_x_147, swiglu_limit);
                        gate_x_147 = _min_102;
                        float _min_103 = fminf(gate_y_148, swiglu_limit);
                        gate_y_148 = _min_103;
                        float _fmax_54 = fmaxf(up_x_149, -swiglu_limit);
                        float _min_104 = fminf(_fmax_54, swiglu_limit);
                        up_x_149 = _min_104;
                        float _fmax_55 = fmaxf(up_y_150, -swiglu_limit);
                        float _min_105 = fminf(_fmax_55, swiglu_limit);
                        up_y_150 = _min_105;
                        float _exp_38 = expf(gate_x_147 * -1.0f);
                        float denominator_x_151 = _exp_38 + 1.0f;
                        float _exp_39 = expf(gate_y_148 * -1.0f);
                        float denominator_y_152 = _exp_39 + 1.0f;
                        float hidden_x_153 = gate_x_147 / denominator_x_151 * up_x_149;
                        float hidden_y_154 = gate_y_148 / denominator_y_152 * up_y_150;
                        __nv_bfloat162 _bf16x2_91 = __float22bfloat162_rn(make_float2(hidden_x_153, hidden_y_154));
                        hidden_packed[19] = __as_u32(_bf16x2_91);
                        float2 _cvt_f32_40 = __bfloat1622float2(__as_bf16x2(gate_packed[20]));
                        float2 _cvt_f32_41 = __bfloat1622float2(__as_bf16x2(up_packed[20]));
                        float gate_x_155 = _cvt_f32_40.x;
                        float gate_y_156 = _cvt_f32_40.y;
                        float up_x_157 = _cvt_f32_41.x;
                        float up_y_158 = _cvt_f32_41.y;
                        float _min_106 = fminf(gate_x_155, swiglu_limit);
                        gate_x_155 = _min_106;
                        float _min_107 = fminf(gate_y_156, swiglu_limit);
                        gate_y_156 = _min_107;
                        float _fmax_56 = fmaxf(up_x_157, -swiglu_limit);
                        float _min_108 = fminf(_fmax_56, swiglu_limit);
                        up_x_157 = _min_108;
                        float _fmax_57 = fmaxf(up_y_158, -swiglu_limit);
                        float _min_109 = fminf(_fmax_57, swiglu_limit);
                        up_y_158 = _min_109;
                        float _exp_40 = expf(gate_x_155 * -1.0f);
                        float denominator_x_159 = _exp_40 + 1.0f;
                        float _exp_41 = expf(gate_y_156 * -1.0f);
                        float denominator_y_160 = _exp_41 + 1.0f;
                        float hidden_x_161 = gate_x_155 / denominator_x_159 * up_x_157;
                        float hidden_y_162 = gate_y_156 / denominator_y_160 * up_y_158;
                        __nv_bfloat162 _bf16x2_92 = __float22bfloat162_rn(make_float2(hidden_x_161, hidden_y_162));
                        hidden_packed[20] = __as_u32(_bf16x2_92);
                        float2 _cvt_f32_42 = __bfloat1622float2(__as_bf16x2(gate_packed[21]));
                        float2 _cvt_f32_43 = __bfloat1622float2(__as_bf16x2(up_packed[21]));
                        float gate_x_163 = _cvt_f32_42.x;
                        float gate_y_164 = _cvt_f32_42.y;
                        float up_x_165 = _cvt_f32_43.x;
                        float up_y_166 = _cvt_f32_43.y;
                        float _min_110 = fminf(gate_x_163, swiglu_limit);
                        gate_x_163 = _min_110;
                        float _min_111 = fminf(gate_y_164, swiglu_limit);
                        gate_y_164 = _min_111;
                        float _fmax_58 = fmaxf(up_x_165, -swiglu_limit);
                        float _min_112 = fminf(_fmax_58, swiglu_limit);
                        up_x_165 = _min_112;
                        float _fmax_59 = fmaxf(up_y_166, -swiglu_limit);
                        float _min_113 = fminf(_fmax_59, swiglu_limit);
                        up_y_166 = _min_113;
                        float _exp_42 = expf(gate_x_163 * -1.0f);
                        float denominator_x_167 = _exp_42 + 1.0f;
                        float _exp_43 = expf(gate_y_164 * -1.0f);
                        float denominator_y_168 = _exp_43 + 1.0f;
                        float hidden_x_169 = gate_x_163 / denominator_x_167 * up_x_165;
                        float hidden_y_170 = gate_y_164 / denominator_y_168 * up_y_166;
                        __nv_bfloat162 _bf16x2_93 = __float22bfloat162_rn(make_float2(hidden_x_169, hidden_y_170));
                        hidden_packed[21] = __as_u32(_bf16x2_93);
                        float2 _cvt_f32_44 = __bfloat1622float2(__as_bf16x2(gate_packed[22]));
                        float2 _cvt_f32_45 = __bfloat1622float2(__as_bf16x2(up_packed[22]));
                        float gate_x_171 = _cvt_f32_44.x;
                        float gate_y_172 = _cvt_f32_44.y;
                        float up_x_173 = _cvt_f32_45.x;
                        float up_y_174 = _cvt_f32_45.y;
                        float _min_114 = fminf(gate_x_171, swiglu_limit);
                        gate_x_171 = _min_114;
                        float _min_115 = fminf(gate_y_172, swiglu_limit);
                        gate_y_172 = _min_115;
                        float _fmax_60 = fmaxf(up_x_173, -swiglu_limit);
                        float _min_116 = fminf(_fmax_60, swiglu_limit);
                        up_x_173 = _min_116;
                        float _fmax_61 = fmaxf(up_y_174, -swiglu_limit);
                        float _min_117 = fminf(_fmax_61, swiglu_limit);
                        up_y_174 = _min_117;
                        float _exp_44 = expf(gate_x_171 * -1.0f);
                        float denominator_x_175 = _exp_44 + 1.0f;
                        float _exp_45 = expf(gate_y_172 * -1.0f);
                        float denominator_y_176 = _exp_45 + 1.0f;
                        float hidden_x_177 = gate_x_171 / denominator_x_175 * up_x_173;
                        float hidden_y_178 = gate_y_172 / denominator_y_176 * up_y_174;
                        __nv_bfloat162 _bf16x2_94 = __float22bfloat162_rn(make_float2(hidden_x_177, hidden_y_178));
                        hidden_packed[22] = __as_u32(_bf16x2_94);
                        float2 _cvt_f32_46 = __bfloat1622float2(__as_bf16x2(gate_packed[23]));
                        float2 _cvt_f32_47 = __bfloat1622float2(__as_bf16x2(up_packed[23]));
                        float gate_x_179 = _cvt_f32_46.x;
                        float gate_y_180 = _cvt_f32_46.y;
                        float up_x_181 = _cvt_f32_47.x;
                        float up_y_182 = _cvt_f32_47.y;
                        float _min_118 = fminf(gate_x_179, swiglu_limit);
                        gate_x_179 = _min_118;
                        float _min_119 = fminf(gate_y_180, swiglu_limit);
                        gate_y_180 = _min_119;
                        float _fmax_62 = fmaxf(up_x_181, -swiglu_limit);
                        float _min_120 = fminf(_fmax_62, swiglu_limit);
                        up_x_181 = _min_120;
                        float _fmax_63 = fmaxf(up_y_182, -swiglu_limit);
                        float _min_121 = fminf(_fmax_63, swiglu_limit);
                        up_y_182 = _min_121;
                        float _exp_46 = expf(gate_x_179 * -1.0f);
                        float denominator_x_183 = _exp_46 + 1.0f;
                        float _exp_47 = expf(gate_y_180 * -1.0f);
                        float denominator_y_184 = _exp_47 + 1.0f;
                        float hidden_x_185 = gate_x_179 / denominator_x_183 * up_x_181;
                        float hidden_y_186 = gate_y_180 / denominator_y_184 * up_y_182;
                        __nv_bfloat162 _bf16x2_95 = __float22bfloat162_rn(make_float2(hidden_x_185, hidden_y_186));
                        hidden_packed[23] = __as_u32(_bf16x2_95);
                        float2 _cvt_f32_48 = __bfloat1622float2(__as_bf16x2(gate_packed[24]));
                        float2 _cvt_f32_49 = __bfloat1622float2(__as_bf16x2(up_packed[24]));
                        float gate_x_187 = _cvt_f32_48.x;
                        float gate_y_188 = _cvt_f32_48.y;
                        float up_x_189 = _cvt_f32_49.x;
                        float up_y_190 = _cvt_f32_49.y;
                        float _min_122 = fminf(gate_x_187, swiglu_limit);
                        gate_x_187 = _min_122;
                        float _min_123 = fminf(gate_y_188, swiglu_limit);
                        gate_y_188 = _min_123;
                        float _fmax_64 = fmaxf(up_x_189, -swiglu_limit);
                        float _min_124 = fminf(_fmax_64, swiglu_limit);
                        up_x_189 = _min_124;
                        float _fmax_65 = fmaxf(up_y_190, -swiglu_limit);
                        float _min_125 = fminf(_fmax_65, swiglu_limit);
                        up_y_190 = _min_125;
                        float _exp_48 = expf(gate_x_187 * -1.0f);
                        float denominator_x_191 = _exp_48 + 1.0f;
                        float _exp_49 = expf(gate_y_188 * -1.0f);
                        float denominator_y_192 = _exp_49 + 1.0f;
                        float hidden_x_193 = gate_x_187 / denominator_x_191 * up_x_189;
                        float hidden_y_194 = gate_y_188 / denominator_y_192 * up_y_190;
                        __nv_bfloat162 _bf16x2_96 = __float22bfloat162_rn(make_float2(hidden_x_193, hidden_y_194));
                        hidden_packed[24] = __as_u32(_bf16x2_96);
                        float2 _cvt_f32_50 = __bfloat1622float2(__as_bf16x2(gate_packed[25]));
                        float2 _cvt_f32_51 = __bfloat1622float2(__as_bf16x2(up_packed[25]));
                        float gate_x_195 = _cvt_f32_50.x;
                        float gate_y_196 = _cvt_f32_50.y;
                        float up_x_197 = _cvt_f32_51.x;
                        float up_y_198 = _cvt_f32_51.y;
                        float _min_126 = fminf(gate_x_195, swiglu_limit);
                        gate_x_195 = _min_126;
                        float _min_127 = fminf(gate_y_196, swiglu_limit);
                        gate_y_196 = _min_127;
                        float _fmax_66 = fmaxf(up_x_197, -swiglu_limit);
                        float _min_128 = fminf(_fmax_66, swiglu_limit);
                        up_x_197 = _min_128;
                        float _fmax_67 = fmaxf(up_y_198, -swiglu_limit);
                        float _min_129 = fminf(_fmax_67, swiglu_limit);
                        up_y_198 = _min_129;
                        float _exp_50 = expf(gate_x_195 * -1.0f);
                        float denominator_x_199 = _exp_50 + 1.0f;
                        float _exp_51 = expf(gate_y_196 * -1.0f);
                        float denominator_y_200 = _exp_51 + 1.0f;
                        float hidden_x_201 = gate_x_195 / denominator_x_199 * up_x_197;
                        float hidden_y_202 = gate_y_196 / denominator_y_200 * up_y_198;
                        __nv_bfloat162 _bf16x2_97 = __float22bfloat162_rn(make_float2(hidden_x_201, hidden_y_202));
                        hidden_packed[25] = __as_u32(_bf16x2_97);
                        float2 _cvt_f32_52 = __bfloat1622float2(__as_bf16x2(gate_packed[26]));
                        float2 _cvt_f32_53 = __bfloat1622float2(__as_bf16x2(up_packed[26]));
                        float gate_x_203 = _cvt_f32_52.x;
                        float gate_y_204 = _cvt_f32_52.y;
                        float up_x_205 = _cvt_f32_53.x;
                        float up_y_206 = _cvt_f32_53.y;
                        float _min_130 = fminf(gate_x_203, swiglu_limit);
                        gate_x_203 = _min_130;
                        float _min_131 = fminf(gate_y_204, swiglu_limit);
                        gate_y_204 = _min_131;
                        float _fmax_68 = fmaxf(up_x_205, -swiglu_limit);
                        float _min_132 = fminf(_fmax_68, swiglu_limit);
                        up_x_205 = _min_132;
                        float _fmax_69 = fmaxf(up_y_206, -swiglu_limit);
                        float _min_133 = fminf(_fmax_69, swiglu_limit);
                        up_y_206 = _min_133;
                        float _exp_52 = expf(gate_x_203 * -1.0f);
                        float denominator_x_207 = _exp_52 + 1.0f;
                        float _exp_53 = expf(gate_y_204 * -1.0f);
                        float denominator_y_208 = _exp_53 + 1.0f;
                        float hidden_x_209 = gate_x_203 / denominator_x_207 * up_x_205;
                        float hidden_y_210 = gate_y_204 / denominator_y_208 * up_y_206;
                        __nv_bfloat162 _bf16x2_98 = __float22bfloat162_rn(make_float2(hidden_x_209, hidden_y_210));
                        hidden_packed[26] = __as_u32(_bf16x2_98);
                        float2 _cvt_f32_54 = __bfloat1622float2(__as_bf16x2(gate_packed[27]));
                        float2 _cvt_f32_55 = __bfloat1622float2(__as_bf16x2(up_packed[27]));
                        float gate_x_211 = _cvt_f32_54.x;
                        float gate_y_212 = _cvt_f32_54.y;
                        float up_x_213 = _cvt_f32_55.x;
                        float up_y_214 = _cvt_f32_55.y;
                        float _min_134 = fminf(gate_x_211, swiglu_limit);
                        gate_x_211 = _min_134;
                        float _min_135 = fminf(gate_y_212, swiglu_limit);
                        gate_y_212 = _min_135;
                        float _fmax_70 = fmaxf(up_x_213, -swiglu_limit);
                        float _min_136 = fminf(_fmax_70, swiglu_limit);
                        up_x_213 = _min_136;
                        float _fmax_71 = fmaxf(up_y_214, -swiglu_limit);
                        float _min_137 = fminf(_fmax_71, swiglu_limit);
                        up_y_214 = _min_137;
                        float _exp_54 = expf(gate_x_211 * -1.0f);
                        float denominator_x_215 = _exp_54 + 1.0f;
                        float _exp_55 = expf(gate_y_212 * -1.0f);
                        float denominator_y_216 = _exp_55 + 1.0f;
                        float hidden_x_217 = gate_x_211 / denominator_x_215 * up_x_213;
                        float hidden_y_218 = gate_y_212 / denominator_y_216 * up_y_214;
                        __nv_bfloat162 _bf16x2_99 = __float22bfloat162_rn(make_float2(hidden_x_217, hidden_y_218));
                        hidden_packed[27] = __as_u32(_bf16x2_99);
                        float2 _cvt_f32_56 = __bfloat1622float2(__as_bf16x2(gate_packed[28]));
                        float2 _cvt_f32_57 = __bfloat1622float2(__as_bf16x2(up_packed[28]));
                        float gate_x_219 = _cvt_f32_56.x;
                        float gate_y_220 = _cvt_f32_56.y;
                        float up_x_221 = _cvt_f32_57.x;
                        float up_y_222 = _cvt_f32_57.y;
                        float _min_138 = fminf(gate_x_219, swiglu_limit);
                        gate_x_219 = _min_138;
                        float _min_139 = fminf(gate_y_220, swiglu_limit);
                        gate_y_220 = _min_139;
                        float _fmax_72 = fmaxf(up_x_221, -swiglu_limit);
                        float _min_140 = fminf(_fmax_72, swiglu_limit);
                        up_x_221 = _min_140;
                        float _fmax_73 = fmaxf(up_y_222, -swiglu_limit);
                        float _min_141 = fminf(_fmax_73, swiglu_limit);
                        up_y_222 = _min_141;
                        float _exp_56 = expf(gate_x_219 * -1.0f);
                        float denominator_x_223 = _exp_56 + 1.0f;
                        float _exp_57 = expf(gate_y_220 * -1.0f);
                        float denominator_y_224 = _exp_57 + 1.0f;
                        float hidden_x_225 = gate_x_219 / denominator_x_223 * up_x_221;
                        float hidden_y_226 = gate_y_220 / denominator_y_224 * up_y_222;
                        __nv_bfloat162 _bf16x2_100 = __float22bfloat162_rn(make_float2(hidden_x_225, hidden_y_226));
                        hidden_packed[28] = __as_u32(_bf16x2_100);
                        float2 _cvt_f32_58 = __bfloat1622float2(__as_bf16x2(gate_packed[29]));
                        float2 _cvt_f32_59 = __bfloat1622float2(__as_bf16x2(up_packed[29]));
                        float gate_x_227 = _cvt_f32_58.x;
                        float gate_y_228 = _cvt_f32_58.y;
                        float up_x_229 = _cvt_f32_59.x;
                        float up_y_230 = _cvt_f32_59.y;
                        float _min_142 = fminf(gate_x_227, swiglu_limit);
                        gate_x_227 = _min_142;
                        float _min_143 = fminf(gate_y_228, swiglu_limit);
                        gate_y_228 = _min_143;
                        float _fmax_74 = fmaxf(up_x_229, -swiglu_limit);
                        float _min_144 = fminf(_fmax_74, swiglu_limit);
                        up_x_229 = _min_144;
                        float _fmax_75 = fmaxf(up_y_230, -swiglu_limit);
                        float _min_145 = fminf(_fmax_75, swiglu_limit);
                        up_y_230 = _min_145;
                        float _exp_58 = expf(gate_x_227 * -1.0f);
                        float denominator_x_231 = _exp_58 + 1.0f;
                        float _exp_59 = expf(gate_y_228 * -1.0f);
                        float denominator_y_232 = _exp_59 + 1.0f;
                        float hidden_x_233 = gate_x_227 / denominator_x_231 * up_x_229;
                        float hidden_y_234 = gate_y_228 / denominator_y_232 * up_y_230;
                        __nv_bfloat162 _bf16x2_101 = __float22bfloat162_rn(make_float2(hidden_x_233, hidden_y_234));
                        hidden_packed[29] = __as_u32(_bf16x2_101);
                        float2 _cvt_f32_60 = __bfloat1622float2(__as_bf16x2(gate_packed[30]));
                        float2 _cvt_f32_61 = __bfloat1622float2(__as_bf16x2(up_packed[30]));
                        float gate_x_235 = _cvt_f32_60.x;
                        float gate_y_236 = _cvt_f32_60.y;
                        float up_x_237 = _cvt_f32_61.x;
                        float up_y_238 = _cvt_f32_61.y;
                        float _min_146 = fminf(gate_x_235, swiglu_limit);
                        gate_x_235 = _min_146;
                        float _min_147 = fminf(gate_y_236, swiglu_limit);
                        gate_y_236 = _min_147;
                        float _fmax_76 = fmaxf(up_x_237, -swiglu_limit);
                        float _min_148 = fminf(_fmax_76, swiglu_limit);
                        up_x_237 = _min_148;
                        float _fmax_77 = fmaxf(up_y_238, -swiglu_limit);
                        float _min_149 = fminf(_fmax_77, swiglu_limit);
                        up_y_238 = _min_149;
                        float _exp_60 = expf(gate_x_235 * -1.0f);
                        float denominator_x_239 = _exp_60 + 1.0f;
                        float _exp_61 = expf(gate_y_236 * -1.0f);
                        float denominator_y_240 = _exp_61 + 1.0f;
                        float hidden_x_241 = gate_x_235 / denominator_x_239 * up_x_237;
                        float hidden_y_242 = gate_y_236 / denominator_y_240 * up_y_238;
                        __nv_bfloat162 _bf16x2_102 = __float22bfloat162_rn(make_float2(hidden_x_241, hidden_y_242));
                        hidden_packed[30] = __as_u32(_bf16x2_102);
                        float2 _cvt_f32_62 = __bfloat1622float2(__as_bf16x2(gate_packed[31]));
                        float2 _cvt_f32_63 = __bfloat1622float2(__as_bf16x2(up_packed[31]));
                        float gate_x_243 = _cvt_f32_62.x;
                        float gate_y_244 = _cvt_f32_62.y;
                        float up_x_245 = _cvt_f32_63.x;
                        float up_y_246 = _cvt_f32_63.y;
                        float _min_150 = fminf(gate_x_243, swiglu_limit);
                        gate_x_243 = _min_150;
                        float _min_151 = fminf(gate_y_244, swiglu_limit);
                        gate_y_244 = _min_151;
                        float _fmax_78 = fmaxf(up_x_245, -swiglu_limit);
                        float _min_152 = fminf(_fmax_78, swiglu_limit);
                        up_x_245 = _min_152;
                        float _fmax_79 = fmaxf(up_y_246, -swiglu_limit);
                        float _min_153 = fminf(_fmax_79, swiglu_limit);
                        up_y_246 = _min_153;
                        float _exp_62 = expf(gate_x_243 * -1.0f);
                        float denominator_x_247 = _exp_62 + 1.0f;
                        float _exp_63 = expf(gate_y_244 * -1.0f);
                        float denominator_y_248 = _exp_63 + 1.0f;
                        float hidden_x_249 = gate_x_243 / denominator_x_247 * up_x_245;
                        float hidden_y_250 = gate_y_244 / denominator_y_248 * up_y_246;
                        __nv_bfloat162 _bf16x2_103 = __float22bfloat162_rn(make_float2(hidden_x_249, hidden_y_250));
                        hidden_packed[31] = __as_u32(_bf16x2_103);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_251 = tid / 32;
                        int lane_1 = tid % 32;
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            #pragma unroll
                            for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                int row_5 = warp_251 * 32 + half * 16 + lane_1 % 16;
                                int col_33 = col_tile_8 * 16 + lane_1 / 16 * 8;
                                unsigned int address_3 = d_smem_addr + (unsigned int)((row_5 * 32 + col_33) * 2);
                                address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                int offset = half * 8 + col_tile_8 * 4;
                                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(address_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_252 = tid / 32;
                        int lane_253 = tid % 32;
                        #pragma unroll
                        for (int half_1 = 0; half_1 < 2; half_1++) {
                            #pragma unroll
                            for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                int row_6 = warp_252 * 32 + half_1 * 16 + lane_253 % 16;
                                int col_34 = col_tile_9 * 16 + lane_253 / 16 * 8;
                                unsigned int address_3_1 = d_smem_addr + 8192 + (unsigned int)((row_6 * 32 + col_34) * 2);
                                address_3_1 = address_3_1 ^ (address_3_1 & 511) >> 7 << 4;
                                int offset_1 = half_1 * 8 + col_tile_9 * 4;
                                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(address_3_1);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_254 = tid / 32;
                        int lane_255 = tid % 32;
                        #pragma unroll
                        for (int half_2 = 0; half_2 < 2; half_2++) {
                            #pragma unroll
                            for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                int row_7 = warp_254 * 32 + half_2 * 16 + lane_255 % 16;
                                int col_35 = col_tile_10 * 16 + lane_255 / 16 * 8;
                                unsigned int address_3_2 = d_smem_addr + 16384 + (unsigned int)((row_7 * 32 + col_35) * 2);
                                address_3_2 = address_3_2 ^ (address_3_2 & 511) >> 7 << 4;
                                int offset_2 = half_2 * 8 + col_tile_10 * 4;
                                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_3_2);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_256 = tid / 32;
                        int lane_257 = tid % 32;
                        #pragma unroll
                        for (int half_3 = 0; half_3 < 2; half_3++) {
                            #pragma unroll
                            for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                int row_8 = warp_256 * 32 + half_3 * 16 + lane_257 % 16;
                                int col_36 = col_tile_11 * 16 + lane_257 / 16 * 8;
                                unsigned int address_3_3 = d_smem_addr + (unsigned int)((row_8 * 32 + col_36) * 2);
                                address_3_3 = address_3_3 ^ (address_3_3 & 511) >> 7 << 4;
                                int offset_3 = 16 + half_3 * 8 + col_tile_11 * 4;
                                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(address_3_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_258 = tid / 32;
                        int lane_259 = tid % 32;
                        #pragma unroll
                        for (int half_4 = 0; half_4 < 2; half_4++) {
                            #pragma unroll
                            for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                int row_9 = warp_258 * 32 + half_4 * 16 + lane_259 % 16;
                                int col_37 = col_tile_12 * 16 + lane_259 / 16 * 8;
                                unsigned int address_3_4 = d_smem_addr + 8192 + (unsigned int)((row_9 * 32 + col_37) * 2);
                                address_3_4 = address_3_4 ^ (address_3_4 & 511) >> 7 << 4;
                                int offset_4 = 16 + half_4 * 8 + col_tile_12 * 4;
                                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(address_3_4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_260 = tid / 32;
                        int lane_261 = tid % 32;
                        #pragma unroll
                        for (int half_5 = 0; half_5 < 2; half_5++) {
                            #pragma unroll
                            for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                int row_10 = warp_260 * 32 + half_5 * 16 + lane_261 % 16;
                                int col_38 = col_tile_13 * 16 + lane_261 / 16 * 8;
                                unsigned int address_3_5 = d_smem_addr + 16384 + (unsigned int)((row_10 * 32 + col_38) * 2);
                                address_3_5 = address_3_5 ^ (address_3_5 & 511) >> 7 << 4;
                                int offset_5 = 16 + half_5 * 8 + col_tile_13 * 4;
                                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(address_3_5);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        unsigned int gate_packed_262[32];
                        unsigned int up_packed_263[32];
                        unsigned int hidden_packed_264[32];
                        unsigned int address_265 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 64;
                        float _tmem_load_8[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                            : "r"(address_265));
                        float _tmem_load_9[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                            : "r"(address_265 + 256));
                        __nv_bfloat162 _bf16x2_104 = __float22bfloat162_rn(make_float2(_tmem_load_8[0], _tmem_load_8[1]));
                        gate_packed_262[0] = __as_u32(_bf16x2_104);
                        __nv_bfloat162 _bf16x2_105 = __float22bfloat162_rn(make_float2(_tmem_load_9[0], _tmem_load_9[1]));
                        up_packed_263[0] = __as_u32(_bf16x2_105);
                        __nv_bfloat162 _bf16x2_106 = __float22bfloat162_rn(make_float2(_tmem_load_8[2], _tmem_load_8[3]));
                        gate_packed_262[1] = __as_u32(_bf16x2_106);
                        __nv_bfloat162 _bf16x2_107 = __float22bfloat162_rn(make_float2(_tmem_load_9[2], _tmem_load_9[3]));
                        up_packed_263[1] = __as_u32(_bf16x2_107);
                        __nv_bfloat162 _bf16x2_108 = __float22bfloat162_rn(make_float2(_tmem_load_8[4], _tmem_load_8[5]));
                        gate_packed_262[2] = __as_u32(_bf16x2_108);
                        __nv_bfloat162 _bf16x2_109 = __float22bfloat162_rn(make_float2(_tmem_load_9[4], _tmem_load_9[5]));
                        up_packed_263[2] = __as_u32(_bf16x2_109);
                        __nv_bfloat162 _bf16x2_110 = __float22bfloat162_rn(make_float2(_tmem_load_8[6], _tmem_load_8[7]));
                        gate_packed_262[3] = __as_u32(_bf16x2_110);
                        __nv_bfloat162 _bf16x2_111 = __float22bfloat162_rn(make_float2(_tmem_load_9[6], _tmem_load_9[7]));
                        up_packed_263[3] = __as_u32(_bf16x2_111);
                        __nv_bfloat162 _bf16x2_112 = __float22bfloat162_rn(make_float2(_tmem_load_8[8], _tmem_load_8[9]));
                        gate_packed_262[4] = __as_u32(_bf16x2_112);
                        __nv_bfloat162 _bf16x2_113 = __float22bfloat162_rn(make_float2(_tmem_load_9[8], _tmem_load_9[9]));
                        up_packed_263[4] = __as_u32(_bf16x2_113);
                        __nv_bfloat162 _bf16x2_114 = __float22bfloat162_rn(make_float2(_tmem_load_8[10], _tmem_load_8[11]));
                        gate_packed_262[5] = __as_u32(_bf16x2_114);
                        __nv_bfloat162 _bf16x2_115 = __float22bfloat162_rn(make_float2(_tmem_load_9[10], _tmem_load_9[11]));
                        up_packed_263[5] = __as_u32(_bf16x2_115);
                        __nv_bfloat162 _bf16x2_116 = __float22bfloat162_rn(make_float2(_tmem_load_8[12], _tmem_load_8[13]));
                        gate_packed_262[6] = __as_u32(_bf16x2_116);
                        __nv_bfloat162 _bf16x2_117 = __float22bfloat162_rn(make_float2(_tmem_load_9[12], _tmem_load_9[13]));
                        up_packed_263[6] = __as_u32(_bf16x2_117);
                        __nv_bfloat162 _bf16x2_118 = __float22bfloat162_rn(make_float2(_tmem_load_8[14], _tmem_load_8[15]));
                        gate_packed_262[7] = __as_u32(_bf16x2_118);
                        __nv_bfloat162 _bf16x2_119 = __float22bfloat162_rn(make_float2(_tmem_load_9[14], _tmem_load_9[15]));
                        up_packed_263[7] = __as_u32(_bf16x2_119);
                        unsigned int address_266 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 64;
                        float _tmem_load_10[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                            : "r"(address_266));
                        float _tmem_load_11[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                            : "r"(address_266 + 256));
                        __nv_bfloat162 _bf16x2_120 = __float22bfloat162_rn(make_float2(_tmem_load_10[0], _tmem_load_10[1]));
                        gate_packed_262[8] = __as_u32(_bf16x2_120);
                        __nv_bfloat162 _bf16x2_121 = __float22bfloat162_rn(make_float2(_tmem_load_11[0], _tmem_load_11[1]));
                        up_packed_263[8] = __as_u32(_bf16x2_121);
                        __nv_bfloat162 _bf16x2_122 = __float22bfloat162_rn(make_float2(_tmem_load_10[2], _tmem_load_10[3]));
                        gate_packed_262[9] = __as_u32(_bf16x2_122);
                        __nv_bfloat162 _bf16x2_123 = __float22bfloat162_rn(make_float2(_tmem_load_11[2], _tmem_load_11[3]));
                        up_packed_263[9] = __as_u32(_bf16x2_123);
                        __nv_bfloat162 _bf16x2_124 = __float22bfloat162_rn(make_float2(_tmem_load_10[4], _tmem_load_10[5]));
                        gate_packed_262[10] = __as_u32(_bf16x2_124);
                        __nv_bfloat162 _bf16x2_125 = __float22bfloat162_rn(make_float2(_tmem_load_11[4], _tmem_load_11[5]));
                        up_packed_263[10] = __as_u32(_bf16x2_125);
                        __nv_bfloat162 _bf16x2_126 = __float22bfloat162_rn(make_float2(_tmem_load_10[6], _tmem_load_10[7]));
                        gate_packed_262[11] = __as_u32(_bf16x2_126);
                        __nv_bfloat162 _bf16x2_127 = __float22bfloat162_rn(make_float2(_tmem_load_11[6], _tmem_load_11[7]));
                        up_packed_263[11] = __as_u32(_bf16x2_127);
                        __nv_bfloat162 _bf16x2_128 = __float22bfloat162_rn(make_float2(_tmem_load_10[8], _tmem_load_10[9]));
                        gate_packed_262[12] = __as_u32(_bf16x2_128);
                        __nv_bfloat162 _bf16x2_129 = __float22bfloat162_rn(make_float2(_tmem_load_11[8], _tmem_load_11[9]));
                        up_packed_263[12] = __as_u32(_bf16x2_129);
                        __nv_bfloat162 _bf16x2_130 = __float22bfloat162_rn(make_float2(_tmem_load_10[10], _tmem_load_10[11]));
                        gate_packed_262[13] = __as_u32(_bf16x2_130);
                        __nv_bfloat162 _bf16x2_131 = __float22bfloat162_rn(make_float2(_tmem_load_11[10], _tmem_load_11[11]));
                        up_packed_263[13] = __as_u32(_bf16x2_131);
                        __nv_bfloat162 _bf16x2_132 = __float22bfloat162_rn(make_float2(_tmem_load_10[12], _tmem_load_10[13]));
                        gate_packed_262[14] = __as_u32(_bf16x2_132);
                        __nv_bfloat162 _bf16x2_133 = __float22bfloat162_rn(make_float2(_tmem_load_11[12], _tmem_load_11[13]));
                        up_packed_263[14] = __as_u32(_bf16x2_133);
                        __nv_bfloat162 _bf16x2_134 = __float22bfloat162_rn(make_float2(_tmem_load_10[14], _tmem_load_10[15]));
                        gate_packed_262[15] = __as_u32(_bf16x2_134);
                        __nv_bfloat162 _bf16x2_135 = __float22bfloat162_rn(make_float2(_tmem_load_11[14], _tmem_load_11[15]));
                        up_packed_263[15] = __as_u32(_bf16x2_135);
                        unsigned int address_267 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 96;
                        float _tmem_load_12[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                            : "r"(address_267));
                        float _tmem_load_13[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                            : "r"(address_267 + 256));
                        __nv_bfloat162 _bf16x2_136 = __float22bfloat162_rn(make_float2(_tmem_load_12[0], _tmem_load_12[1]));
                        gate_packed_262[16] = __as_u32(_bf16x2_136);
                        __nv_bfloat162 _bf16x2_137 = __float22bfloat162_rn(make_float2(_tmem_load_13[0], _tmem_load_13[1]));
                        up_packed_263[16] = __as_u32(_bf16x2_137);
                        __nv_bfloat162 _bf16x2_138 = __float22bfloat162_rn(make_float2(_tmem_load_12[2], _tmem_load_12[3]));
                        gate_packed_262[17] = __as_u32(_bf16x2_138);
                        __nv_bfloat162 _bf16x2_139 = __float22bfloat162_rn(make_float2(_tmem_load_13[2], _tmem_load_13[3]));
                        up_packed_263[17] = __as_u32(_bf16x2_139);
                        __nv_bfloat162 _bf16x2_140 = __float22bfloat162_rn(make_float2(_tmem_load_12[4], _tmem_load_12[5]));
                        gate_packed_262[18] = __as_u32(_bf16x2_140);
                        __nv_bfloat162 _bf16x2_141 = __float22bfloat162_rn(make_float2(_tmem_load_13[4], _tmem_load_13[5]));
                        up_packed_263[18] = __as_u32(_bf16x2_141);
                        __nv_bfloat162 _bf16x2_142 = __float22bfloat162_rn(make_float2(_tmem_load_12[6], _tmem_load_12[7]));
                        gate_packed_262[19] = __as_u32(_bf16x2_142);
                        __nv_bfloat162 _bf16x2_143 = __float22bfloat162_rn(make_float2(_tmem_load_13[6], _tmem_load_13[7]));
                        up_packed_263[19] = __as_u32(_bf16x2_143);
                        __nv_bfloat162 _bf16x2_144 = __float22bfloat162_rn(make_float2(_tmem_load_12[8], _tmem_load_12[9]));
                        gate_packed_262[20] = __as_u32(_bf16x2_144);
                        __nv_bfloat162 _bf16x2_145 = __float22bfloat162_rn(make_float2(_tmem_load_13[8], _tmem_load_13[9]));
                        up_packed_263[20] = __as_u32(_bf16x2_145);
                        __nv_bfloat162 _bf16x2_146 = __float22bfloat162_rn(make_float2(_tmem_load_12[10], _tmem_load_12[11]));
                        gate_packed_262[21] = __as_u32(_bf16x2_146);
                        __nv_bfloat162 _bf16x2_147 = __float22bfloat162_rn(make_float2(_tmem_load_13[10], _tmem_load_13[11]));
                        up_packed_263[21] = __as_u32(_bf16x2_147);
                        __nv_bfloat162 _bf16x2_148 = __float22bfloat162_rn(make_float2(_tmem_load_12[12], _tmem_load_12[13]));
                        gate_packed_262[22] = __as_u32(_bf16x2_148);
                        __nv_bfloat162 _bf16x2_149 = __float22bfloat162_rn(make_float2(_tmem_load_13[12], _tmem_load_13[13]));
                        up_packed_263[22] = __as_u32(_bf16x2_149);
                        __nv_bfloat162 _bf16x2_150 = __float22bfloat162_rn(make_float2(_tmem_load_12[14], _tmem_load_12[15]));
                        gate_packed_262[23] = __as_u32(_bf16x2_150);
                        __nv_bfloat162 _bf16x2_151 = __float22bfloat162_rn(make_float2(_tmem_load_13[14], _tmem_load_13[15]));
                        up_packed_263[23] = __as_u32(_bf16x2_151);
                        unsigned int address_268 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 96;
                        float _tmem_load_14[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                            : "r"(address_268));
                        float _tmem_load_15[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                            : "r"(address_268 + 256));
                        __nv_bfloat162 _bf16x2_152 = __float22bfloat162_rn(make_float2(_tmem_load_14[0], _tmem_load_14[1]));
                        gate_packed_262[24] = __as_u32(_bf16x2_152);
                        __nv_bfloat162 _bf16x2_153 = __float22bfloat162_rn(make_float2(_tmem_load_15[0], _tmem_load_15[1]));
                        up_packed_263[24] = __as_u32(_bf16x2_153);
                        __nv_bfloat162 _bf16x2_154 = __float22bfloat162_rn(make_float2(_tmem_load_14[2], _tmem_load_14[3]));
                        gate_packed_262[25] = __as_u32(_bf16x2_154);
                        __nv_bfloat162 _bf16x2_155 = __float22bfloat162_rn(make_float2(_tmem_load_15[2], _tmem_load_15[3]));
                        up_packed_263[25] = __as_u32(_bf16x2_155);
                        __nv_bfloat162 _bf16x2_156 = __float22bfloat162_rn(make_float2(_tmem_load_14[4], _tmem_load_14[5]));
                        gate_packed_262[26] = __as_u32(_bf16x2_156);
                        __nv_bfloat162 _bf16x2_157 = __float22bfloat162_rn(make_float2(_tmem_load_15[4], _tmem_load_15[5]));
                        up_packed_263[26] = __as_u32(_bf16x2_157);
                        __nv_bfloat162 _bf16x2_158 = __float22bfloat162_rn(make_float2(_tmem_load_14[6], _tmem_load_14[7]));
                        gate_packed_262[27] = __as_u32(_bf16x2_158);
                        __nv_bfloat162 _bf16x2_159 = __float22bfloat162_rn(make_float2(_tmem_load_15[6], _tmem_load_15[7]));
                        up_packed_263[27] = __as_u32(_bf16x2_159);
                        __nv_bfloat162 _bf16x2_160 = __float22bfloat162_rn(make_float2(_tmem_load_14[8], _tmem_load_14[9]));
                        gate_packed_262[28] = __as_u32(_bf16x2_160);
                        __nv_bfloat162 _bf16x2_161 = __float22bfloat162_rn(make_float2(_tmem_load_15[8], _tmem_load_15[9]));
                        up_packed_263[28] = __as_u32(_bf16x2_161);
                        __nv_bfloat162 _bf16x2_162 = __float22bfloat162_rn(make_float2(_tmem_load_14[10], _tmem_load_14[11]));
                        gate_packed_262[29] = __as_u32(_bf16x2_162);
                        __nv_bfloat162 _bf16x2_163 = __float22bfloat162_rn(make_float2(_tmem_load_15[10], _tmem_load_15[11]));
                        up_packed_263[29] = __as_u32(_bf16x2_163);
                        __nv_bfloat162 _bf16x2_164 = __float22bfloat162_rn(make_float2(_tmem_load_14[12], _tmem_load_14[13]));
                        gate_packed_262[30] = __as_u32(_bf16x2_164);
                        __nv_bfloat162 _bf16x2_165 = __float22bfloat162_rn(make_float2(_tmem_load_15[12], _tmem_load_15[13]));
                        up_packed_263[30] = __as_u32(_bf16x2_165);
                        __nv_bfloat162 _bf16x2_166 = __float22bfloat162_rn(make_float2(_tmem_load_14[14], _tmem_load_14[15]));
                        gate_packed_262[31] = __as_u32(_bf16x2_166);
                        __nv_bfloat162 _bf16x2_167 = __float22bfloat162_rn(make_float2(_tmem_load_15[14], _tmem_load_15[15]));
                        up_packed_263[31] = __as_u32(_bf16x2_167);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_64 = __bfloat1622float2(__as_bf16x2(gate_packed_262[0]));
                        float2 _cvt_f32_65 = __bfloat1622float2(__as_bf16x2(up_packed_263[0]));
                        float gate_x_269 = _cvt_f32_64.x;
                        float gate_y_270 = _cvt_f32_64.y;
                        float up_x_271 = _cvt_f32_65.x;
                        float up_y_272 = _cvt_f32_65.y;
                        float _min_154 = fminf(gate_x_269, swiglu_limit);
                        gate_x_269 = _min_154;
                        float _min_155 = fminf(gate_y_270, swiglu_limit);
                        gate_y_270 = _min_155;
                        float _fmax_80 = fmaxf(up_x_271, -swiglu_limit);
                        float _min_156 = fminf(_fmax_80, swiglu_limit);
                        up_x_271 = _min_156;
                        float _fmax_81 = fmaxf(up_y_272, -swiglu_limit);
                        float _min_157 = fminf(_fmax_81, swiglu_limit);
                        up_y_272 = _min_157;
                        float _exp_64 = expf(gate_x_269 * -1.0f);
                        float denominator_x_273 = _exp_64 + 1.0f;
                        float _exp_65 = expf(gate_y_270 * -1.0f);
                        float denominator_y_274 = _exp_65 + 1.0f;
                        float hidden_x_275 = gate_x_269 / denominator_x_273 * up_x_271;
                        float hidden_y_276 = gate_y_270 / denominator_y_274 * up_y_272;
                        __nv_bfloat162 _bf16x2_168 = __float22bfloat162_rn(make_float2(hidden_x_275, hidden_y_276));
                        hidden_packed_264[0] = __as_u32(_bf16x2_168);
                        float2 _cvt_f32_66 = __bfloat1622float2(__as_bf16x2(gate_packed_262[1]));
                        float2 _cvt_f32_67 = __bfloat1622float2(__as_bf16x2(up_packed_263[1]));
                        float gate_x_277 = _cvt_f32_66.x;
                        float gate_y_278 = _cvt_f32_66.y;
                        float up_x_279 = _cvt_f32_67.x;
                        float up_y_280 = _cvt_f32_67.y;
                        float _min_158 = fminf(gate_x_277, swiglu_limit);
                        gate_x_277 = _min_158;
                        float _min_159 = fminf(gate_y_278, swiglu_limit);
                        gate_y_278 = _min_159;
                        float _fmax_82 = fmaxf(up_x_279, -swiglu_limit);
                        float _min_160 = fminf(_fmax_82, swiglu_limit);
                        up_x_279 = _min_160;
                        float _fmax_83 = fmaxf(up_y_280, -swiglu_limit);
                        float _min_161 = fminf(_fmax_83, swiglu_limit);
                        up_y_280 = _min_161;
                        float _exp_66 = expf(gate_x_277 * -1.0f);
                        float denominator_x_281 = _exp_66 + 1.0f;
                        float _exp_67 = expf(gate_y_278 * -1.0f);
                        float denominator_y_282 = _exp_67 + 1.0f;
                        float hidden_x_283 = gate_x_277 / denominator_x_281 * up_x_279;
                        float hidden_y_284 = gate_y_278 / denominator_y_282 * up_y_280;
                        __nv_bfloat162 _bf16x2_169 = __float22bfloat162_rn(make_float2(hidden_x_283, hidden_y_284));
                        hidden_packed_264[1] = __as_u32(_bf16x2_169);
                        float2 _cvt_f32_68 = __bfloat1622float2(__as_bf16x2(gate_packed_262[2]));
                        float2 _cvt_f32_69 = __bfloat1622float2(__as_bf16x2(up_packed_263[2]));
                        float gate_x_285 = _cvt_f32_68.x;
                        float gate_y_286 = _cvt_f32_68.y;
                        float up_x_287 = _cvt_f32_69.x;
                        float up_y_288 = _cvt_f32_69.y;
                        float _min_162 = fminf(gate_x_285, swiglu_limit);
                        gate_x_285 = _min_162;
                        float _min_163 = fminf(gate_y_286, swiglu_limit);
                        gate_y_286 = _min_163;
                        float _fmax_84 = fmaxf(up_x_287, -swiglu_limit);
                        float _min_164 = fminf(_fmax_84, swiglu_limit);
                        up_x_287 = _min_164;
                        float _fmax_85 = fmaxf(up_y_288, -swiglu_limit);
                        float _min_165 = fminf(_fmax_85, swiglu_limit);
                        up_y_288 = _min_165;
                        float _exp_68 = expf(gate_x_285 * -1.0f);
                        float denominator_x_289 = _exp_68 + 1.0f;
                        float _exp_69 = expf(gate_y_286 * -1.0f);
                        float denominator_y_290 = _exp_69 + 1.0f;
                        float hidden_x_291 = gate_x_285 / denominator_x_289 * up_x_287;
                        float hidden_y_292 = gate_y_286 / denominator_y_290 * up_y_288;
                        __nv_bfloat162 _bf16x2_170 = __float22bfloat162_rn(make_float2(hidden_x_291, hidden_y_292));
                        hidden_packed_264[2] = __as_u32(_bf16x2_170);
                        float2 _cvt_f32_70 = __bfloat1622float2(__as_bf16x2(gate_packed_262[3]));
                        float2 _cvt_f32_71 = __bfloat1622float2(__as_bf16x2(up_packed_263[3]));
                        float gate_x_293 = _cvt_f32_70.x;
                        float gate_y_294 = _cvt_f32_70.y;
                        float up_x_295 = _cvt_f32_71.x;
                        float up_y_296 = _cvt_f32_71.y;
                        float _min_166 = fminf(gate_x_293, swiglu_limit);
                        gate_x_293 = _min_166;
                        float _min_167 = fminf(gate_y_294, swiglu_limit);
                        gate_y_294 = _min_167;
                        float _fmax_86 = fmaxf(up_x_295, -swiglu_limit);
                        float _min_168 = fminf(_fmax_86, swiglu_limit);
                        up_x_295 = _min_168;
                        float _fmax_87 = fmaxf(up_y_296, -swiglu_limit);
                        float _min_169 = fminf(_fmax_87, swiglu_limit);
                        up_y_296 = _min_169;
                        float _exp_70 = expf(gate_x_293 * -1.0f);
                        float denominator_x_297 = _exp_70 + 1.0f;
                        float _exp_71 = expf(gate_y_294 * -1.0f);
                        float denominator_y_298 = _exp_71 + 1.0f;
                        float hidden_x_299 = gate_x_293 / denominator_x_297 * up_x_295;
                        float hidden_y_300 = gate_y_294 / denominator_y_298 * up_y_296;
                        __nv_bfloat162 _bf16x2_171 = __float22bfloat162_rn(make_float2(hidden_x_299, hidden_y_300));
                        hidden_packed_264[3] = __as_u32(_bf16x2_171);
                        float2 _cvt_f32_72 = __bfloat1622float2(__as_bf16x2(gate_packed_262[4]));
                        float2 _cvt_f32_73 = __bfloat1622float2(__as_bf16x2(up_packed_263[4]));
                        float gate_x_301 = _cvt_f32_72.x;
                        float gate_y_302 = _cvt_f32_72.y;
                        float up_x_303 = _cvt_f32_73.x;
                        float up_y_304 = _cvt_f32_73.y;
                        float _min_170 = fminf(gate_x_301, swiglu_limit);
                        gate_x_301 = _min_170;
                        float _min_171 = fminf(gate_y_302, swiglu_limit);
                        gate_y_302 = _min_171;
                        float _fmax_88 = fmaxf(up_x_303, -swiglu_limit);
                        float _min_172 = fminf(_fmax_88, swiglu_limit);
                        up_x_303 = _min_172;
                        float _fmax_89 = fmaxf(up_y_304, -swiglu_limit);
                        float _min_173 = fminf(_fmax_89, swiglu_limit);
                        up_y_304 = _min_173;
                        float _exp_72 = expf(gate_x_301 * -1.0f);
                        float denominator_x_305 = _exp_72 + 1.0f;
                        float _exp_73 = expf(gate_y_302 * -1.0f);
                        float denominator_y_306 = _exp_73 + 1.0f;
                        float hidden_x_307 = gate_x_301 / denominator_x_305 * up_x_303;
                        float hidden_y_308 = gate_y_302 / denominator_y_306 * up_y_304;
                        __nv_bfloat162 _bf16x2_172 = __float22bfloat162_rn(make_float2(hidden_x_307, hidden_y_308));
                        hidden_packed_264[4] = __as_u32(_bf16x2_172);
                        float2 _cvt_f32_74 = __bfloat1622float2(__as_bf16x2(gate_packed_262[5]));
                        float2 _cvt_f32_75 = __bfloat1622float2(__as_bf16x2(up_packed_263[5]));
                        float gate_x_309 = _cvt_f32_74.x;
                        float gate_y_310 = _cvt_f32_74.y;
                        float up_x_311 = _cvt_f32_75.x;
                        float up_y_312 = _cvt_f32_75.y;
                        float _min_174 = fminf(gate_x_309, swiglu_limit);
                        gate_x_309 = _min_174;
                        float _min_175 = fminf(gate_y_310, swiglu_limit);
                        gate_y_310 = _min_175;
                        float _fmax_90 = fmaxf(up_x_311, -swiglu_limit);
                        float _min_176 = fminf(_fmax_90, swiglu_limit);
                        up_x_311 = _min_176;
                        float _fmax_91 = fmaxf(up_y_312, -swiglu_limit);
                        float _min_177 = fminf(_fmax_91, swiglu_limit);
                        up_y_312 = _min_177;
                        float _exp_74 = expf(gate_x_309 * -1.0f);
                        float denominator_x_313 = _exp_74 + 1.0f;
                        float _exp_75 = expf(gate_y_310 * -1.0f);
                        float denominator_y_314 = _exp_75 + 1.0f;
                        float hidden_x_315 = gate_x_309 / denominator_x_313 * up_x_311;
                        float hidden_y_316 = gate_y_310 / denominator_y_314 * up_y_312;
                        __nv_bfloat162 _bf16x2_173 = __float22bfloat162_rn(make_float2(hidden_x_315, hidden_y_316));
                        hidden_packed_264[5] = __as_u32(_bf16x2_173);
                        float2 _cvt_f32_76 = __bfloat1622float2(__as_bf16x2(gate_packed_262[6]));
                        float2 _cvt_f32_77 = __bfloat1622float2(__as_bf16x2(up_packed_263[6]));
                        float gate_x_317 = _cvt_f32_76.x;
                        float gate_y_318 = _cvt_f32_76.y;
                        float up_x_319 = _cvt_f32_77.x;
                        float up_y_320 = _cvt_f32_77.y;
                        float _min_178 = fminf(gate_x_317, swiglu_limit);
                        gate_x_317 = _min_178;
                        float _min_179 = fminf(gate_y_318, swiglu_limit);
                        gate_y_318 = _min_179;
                        float _fmax_92 = fmaxf(up_x_319, -swiglu_limit);
                        float _min_180 = fminf(_fmax_92, swiglu_limit);
                        up_x_319 = _min_180;
                        float _fmax_93 = fmaxf(up_y_320, -swiglu_limit);
                        float _min_181 = fminf(_fmax_93, swiglu_limit);
                        up_y_320 = _min_181;
                        float _exp_76 = expf(gate_x_317 * -1.0f);
                        float denominator_x_321 = _exp_76 + 1.0f;
                        float _exp_77 = expf(gate_y_318 * -1.0f);
                        float denominator_y_322 = _exp_77 + 1.0f;
                        float hidden_x_323 = gate_x_317 / denominator_x_321 * up_x_319;
                        float hidden_y_324 = gate_y_318 / denominator_y_322 * up_y_320;
                        __nv_bfloat162 _bf16x2_174 = __float22bfloat162_rn(make_float2(hidden_x_323, hidden_y_324));
                        hidden_packed_264[6] = __as_u32(_bf16x2_174);
                        float2 _cvt_f32_78 = __bfloat1622float2(__as_bf16x2(gate_packed_262[7]));
                        float2 _cvt_f32_79 = __bfloat1622float2(__as_bf16x2(up_packed_263[7]));
                        float gate_x_325 = _cvt_f32_78.x;
                        float gate_y_326 = _cvt_f32_78.y;
                        float up_x_327 = _cvt_f32_79.x;
                        float up_y_328 = _cvt_f32_79.y;
                        float _min_182 = fminf(gate_x_325, swiglu_limit);
                        gate_x_325 = _min_182;
                        float _min_183 = fminf(gate_y_326, swiglu_limit);
                        gate_y_326 = _min_183;
                        float _fmax_94 = fmaxf(up_x_327, -swiglu_limit);
                        float _min_184 = fminf(_fmax_94, swiglu_limit);
                        up_x_327 = _min_184;
                        float _fmax_95 = fmaxf(up_y_328, -swiglu_limit);
                        float _min_185 = fminf(_fmax_95, swiglu_limit);
                        up_y_328 = _min_185;
                        float _exp_78 = expf(gate_x_325 * -1.0f);
                        float denominator_x_329 = _exp_78 + 1.0f;
                        float _exp_79 = expf(gate_y_326 * -1.0f);
                        float denominator_y_330 = _exp_79 + 1.0f;
                        float hidden_x_331 = gate_x_325 / denominator_x_329 * up_x_327;
                        float hidden_y_332 = gate_y_326 / denominator_y_330 * up_y_328;
                        __nv_bfloat162 _bf16x2_175 = __float22bfloat162_rn(make_float2(hidden_x_331, hidden_y_332));
                        hidden_packed_264[7] = __as_u32(_bf16x2_175);
                        float2 _cvt_f32_80 = __bfloat1622float2(__as_bf16x2(gate_packed_262[8]));
                        float2 _cvt_f32_81 = __bfloat1622float2(__as_bf16x2(up_packed_263[8]));
                        float gate_x_333 = _cvt_f32_80.x;
                        float gate_y_334 = _cvt_f32_80.y;
                        float up_x_335 = _cvt_f32_81.x;
                        float up_y_336 = _cvt_f32_81.y;
                        float _min_186 = fminf(gate_x_333, swiglu_limit);
                        gate_x_333 = _min_186;
                        float _min_187 = fminf(gate_y_334, swiglu_limit);
                        gate_y_334 = _min_187;
                        float _fmax_96 = fmaxf(up_x_335, -swiglu_limit);
                        float _min_188 = fminf(_fmax_96, swiglu_limit);
                        up_x_335 = _min_188;
                        float _fmax_97 = fmaxf(up_y_336, -swiglu_limit);
                        float _min_189 = fminf(_fmax_97, swiglu_limit);
                        up_y_336 = _min_189;
                        float _exp_80 = expf(gate_x_333 * -1.0f);
                        float denominator_x_337 = _exp_80 + 1.0f;
                        float _exp_81 = expf(gate_y_334 * -1.0f);
                        float denominator_y_338 = _exp_81 + 1.0f;
                        float hidden_x_339 = gate_x_333 / denominator_x_337 * up_x_335;
                        float hidden_y_340 = gate_y_334 / denominator_y_338 * up_y_336;
                        __nv_bfloat162 _bf16x2_176 = __float22bfloat162_rn(make_float2(hidden_x_339, hidden_y_340));
                        hidden_packed_264[8] = __as_u32(_bf16x2_176);
                        float2 _cvt_f32_82 = __bfloat1622float2(__as_bf16x2(gate_packed_262[9]));
                        float2 _cvt_f32_83 = __bfloat1622float2(__as_bf16x2(up_packed_263[9]));
                        float gate_x_341 = _cvt_f32_82.x;
                        float gate_y_342 = _cvt_f32_82.y;
                        float up_x_343 = _cvt_f32_83.x;
                        float up_y_344 = _cvt_f32_83.y;
                        float _min_190 = fminf(gate_x_341, swiglu_limit);
                        gate_x_341 = _min_190;
                        float _min_191 = fminf(gate_y_342, swiglu_limit);
                        gate_y_342 = _min_191;
                        float _fmax_98 = fmaxf(up_x_343, -swiglu_limit);
                        float _min_192 = fminf(_fmax_98, swiglu_limit);
                        up_x_343 = _min_192;
                        float _fmax_99 = fmaxf(up_y_344, -swiglu_limit);
                        float _min_193 = fminf(_fmax_99, swiglu_limit);
                        up_y_344 = _min_193;
                        float _exp_82 = expf(gate_x_341 * -1.0f);
                        float denominator_x_345 = _exp_82 + 1.0f;
                        float _exp_83 = expf(gate_y_342 * -1.0f);
                        float denominator_y_346 = _exp_83 + 1.0f;
                        float hidden_x_347 = gate_x_341 / denominator_x_345 * up_x_343;
                        float hidden_y_348 = gate_y_342 / denominator_y_346 * up_y_344;
                        __nv_bfloat162 _bf16x2_177 = __float22bfloat162_rn(make_float2(hidden_x_347, hidden_y_348));
                        hidden_packed_264[9] = __as_u32(_bf16x2_177);
                        float2 _cvt_f32_84 = __bfloat1622float2(__as_bf16x2(gate_packed_262[10]));
                        float2 _cvt_f32_85 = __bfloat1622float2(__as_bf16x2(up_packed_263[10]));
                        float gate_x_349 = _cvt_f32_84.x;
                        float gate_y_350 = _cvt_f32_84.y;
                        float up_x_351 = _cvt_f32_85.x;
                        float up_y_352 = _cvt_f32_85.y;
                        float _min_194 = fminf(gate_x_349, swiglu_limit);
                        gate_x_349 = _min_194;
                        float _min_195 = fminf(gate_y_350, swiglu_limit);
                        gate_y_350 = _min_195;
                        float _fmax_100 = fmaxf(up_x_351, -swiglu_limit);
                        float _min_196 = fminf(_fmax_100, swiglu_limit);
                        up_x_351 = _min_196;
                        float _fmax_101 = fmaxf(up_y_352, -swiglu_limit);
                        float _min_197 = fminf(_fmax_101, swiglu_limit);
                        up_y_352 = _min_197;
                        float _exp_84 = expf(gate_x_349 * -1.0f);
                        float denominator_x_353 = _exp_84 + 1.0f;
                        float _exp_85 = expf(gate_y_350 * -1.0f);
                        float denominator_y_354 = _exp_85 + 1.0f;
                        float hidden_x_355 = gate_x_349 / denominator_x_353 * up_x_351;
                        float hidden_y_356 = gate_y_350 / denominator_y_354 * up_y_352;
                        __nv_bfloat162 _bf16x2_178 = __float22bfloat162_rn(make_float2(hidden_x_355, hidden_y_356));
                        hidden_packed_264[10] = __as_u32(_bf16x2_178);
                        float2 _cvt_f32_86 = __bfloat1622float2(__as_bf16x2(gate_packed_262[11]));
                        float2 _cvt_f32_87 = __bfloat1622float2(__as_bf16x2(up_packed_263[11]));
                        float gate_x_357 = _cvt_f32_86.x;
                        float gate_y_358 = _cvt_f32_86.y;
                        float up_x_359 = _cvt_f32_87.x;
                        float up_y_360 = _cvt_f32_87.y;
                        float _min_198 = fminf(gate_x_357, swiglu_limit);
                        gate_x_357 = _min_198;
                        float _min_199 = fminf(gate_y_358, swiglu_limit);
                        gate_y_358 = _min_199;
                        float _fmax_102 = fmaxf(up_x_359, -swiglu_limit);
                        float _min_200 = fminf(_fmax_102, swiglu_limit);
                        up_x_359 = _min_200;
                        float _fmax_103 = fmaxf(up_y_360, -swiglu_limit);
                        float _min_201 = fminf(_fmax_103, swiglu_limit);
                        up_y_360 = _min_201;
                        float _exp_86 = expf(gate_x_357 * -1.0f);
                        float denominator_x_361 = _exp_86 + 1.0f;
                        float _exp_87 = expf(gate_y_358 * -1.0f);
                        float denominator_y_362 = _exp_87 + 1.0f;
                        float hidden_x_363 = gate_x_357 / denominator_x_361 * up_x_359;
                        float hidden_y_364 = gate_y_358 / denominator_y_362 * up_y_360;
                        __nv_bfloat162 _bf16x2_179 = __float22bfloat162_rn(make_float2(hidden_x_363, hidden_y_364));
                        hidden_packed_264[11] = __as_u32(_bf16x2_179);
                        float2 _cvt_f32_88 = __bfloat1622float2(__as_bf16x2(gate_packed_262[12]));
                        float2 _cvt_f32_89 = __bfloat1622float2(__as_bf16x2(up_packed_263[12]));
                        float gate_x_365 = _cvt_f32_88.x;
                        float gate_y_366 = _cvt_f32_88.y;
                        float up_x_367 = _cvt_f32_89.x;
                        float up_y_368 = _cvt_f32_89.y;
                        float _min_202 = fminf(gate_x_365, swiglu_limit);
                        gate_x_365 = _min_202;
                        float _min_203 = fminf(gate_y_366, swiglu_limit);
                        gate_y_366 = _min_203;
                        float _fmax_104 = fmaxf(up_x_367, -swiglu_limit);
                        float _min_204 = fminf(_fmax_104, swiglu_limit);
                        up_x_367 = _min_204;
                        float _fmax_105 = fmaxf(up_y_368, -swiglu_limit);
                        float _min_205 = fminf(_fmax_105, swiglu_limit);
                        up_y_368 = _min_205;
                        float _exp_88 = expf(gate_x_365 * -1.0f);
                        float denominator_x_369 = _exp_88 + 1.0f;
                        float _exp_89 = expf(gate_y_366 * -1.0f);
                        float denominator_y_370 = _exp_89 + 1.0f;
                        float hidden_x_371 = gate_x_365 / denominator_x_369 * up_x_367;
                        float hidden_y_372 = gate_y_366 / denominator_y_370 * up_y_368;
                        __nv_bfloat162 _bf16x2_180 = __float22bfloat162_rn(make_float2(hidden_x_371, hidden_y_372));
                        hidden_packed_264[12] = __as_u32(_bf16x2_180);
                        float2 _cvt_f32_90 = __bfloat1622float2(__as_bf16x2(gate_packed_262[13]));
                        float2 _cvt_f32_91 = __bfloat1622float2(__as_bf16x2(up_packed_263[13]));
                        float gate_x_373 = _cvt_f32_90.x;
                        float gate_y_374 = _cvt_f32_90.y;
                        float up_x_375 = _cvt_f32_91.x;
                        float up_y_376 = _cvt_f32_91.y;
                        float _min_206 = fminf(gate_x_373, swiglu_limit);
                        gate_x_373 = _min_206;
                        float _min_207 = fminf(gate_y_374, swiglu_limit);
                        gate_y_374 = _min_207;
                        float _fmax_106 = fmaxf(up_x_375, -swiglu_limit);
                        float _min_208 = fminf(_fmax_106, swiglu_limit);
                        up_x_375 = _min_208;
                        float _fmax_107 = fmaxf(up_y_376, -swiglu_limit);
                        float _min_209 = fminf(_fmax_107, swiglu_limit);
                        up_y_376 = _min_209;
                        float _exp_90 = expf(gate_x_373 * -1.0f);
                        float denominator_x_377 = _exp_90 + 1.0f;
                        float _exp_91 = expf(gate_y_374 * -1.0f);
                        float denominator_y_378 = _exp_91 + 1.0f;
                        float hidden_x_379 = gate_x_373 / denominator_x_377 * up_x_375;
                        float hidden_y_380 = gate_y_374 / denominator_y_378 * up_y_376;
                        __nv_bfloat162 _bf16x2_181 = __float22bfloat162_rn(make_float2(hidden_x_379, hidden_y_380));
                        hidden_packed_264[13] = __as_u32(_bf16x2_181);
                        float2 _cvt_f32_92 = __bfloat1622float2(__as_bf16x2(gate_packed_262[14]));
                        float2 _cvt_f32_93 = __bfloat1622float2(__as_bf16x2(up_packed_263[14]));
                        float gate_x_381 = _cvt_f32_92.x;
                        float gate_y_382 = _cvt_f32_92.y;
                        float up_x_383 = _cvt_f32_93.x;
                        float up_y_384 = _cvt_f32_93.y;
                        float _min_210 = fminf(gate_x_381, swiglu_limit);
                        gate_x_381 = _min_210;
                        float _min_211 = fminf(gate_y_382, swiglu_limit);
                        gate_y_382 = _min_211;
                        float _fmax_108 = fmaxf(up_x_383, -swiglu_limit);
                        float _min_212 = fminf(_fmax_108, swiglu_limit);
                        up_x_383 = _min_212;
                        float _fmax_109 = fmaxf(up_y_384, -swiglu_limit);
                        float _min_213 = fminf(_fmax_109, swiglu_limit);
                        up_y_384 = _min_213;
                        float _exp_92 = expf(gate_x_381 * -1.0f);
                        float denominator_x_385 = _exp_92 + 1.0f;
                        float _exp_93 = expf(gate_y_382 * -1.0f);
                        float denominator_y_386 = _exp_93 + 1.0f;
                        float hidden_x_387 = gate_x_381 / denominator_x_385 * up_x_383;
                        float hidden_y_388 = gate_y_382 / denominator_y_386 * up_y_384;
                        __nv_bfloat162 _bf16x2_182 = __float22bfloat162_rn(make_float2(hidden_x_387, hidden_y_388));
                        hidden_packed_264[14] = __as_u32(_bf16x2_182);
                        float2 _cvt_f32_94 = __bfloat1622float2(__as_bf16x2(gate_packed_262[15]));
                        float2 _cvt_f32_95 = __bfloat1622float2(__as_bf16x2(up_packed_263[15]));
                        float gate_x_389 = _cvt_f32_94.x;
                        float gate_y_390 = _cvt_f32_94.y;
                        float up_x_391 = _cvt_f32_95.x;
                        float up_y_392 = _cvt_f32_95.y;
                        float _min_214 = fminf(gate_x_389, swiglu_limit);
                        gate_x_389 = _min_214;
                        float _min_215 = fminf(gate_y_390, swiglu_limit);
                        gate_y_390 = _min_215;
                        float _fmax_110 = fmaxf(up_x_391, -swiglu_limit);
                        float _min_216 = fminf(_fmax_110, swiglu_limit);
                        up_x_391 = _min_216;
                        float _fmax_111 = fmaxf(up_y_392, -swiglu_limit);
                        float _min_217 = fminf(_fmax_111, swiglu_limit);
                        up_y_392 = _min_217;
                        float _exp_94 = expf(gate_x_389 * -1.0f);
                        float denominator_x_393 = _exp_94 + 1.0f;
                        float _exp_95 = expf(gate_y_390 * -1.0f);
                        float denominator_y_394 = _exp_95 + 1.0f;
                        float hidden_x_395 = gate_x_389 / denominator_x_393 * up_x_391;
                        float hidden_y_396 = gate_y_390 / denominator_y_394 * up_y_392;
                        __nv_bfloat162 _bf16x2_183 = __float22bfloat162_rn(make_float2(hidden_x_395, hidden_y_396));
                        hidden_packed_264[15] = __as_u32(_bf16x2_183);
                        float2 _cvt_f32_96 = __bfloat1622float2(__as_bf16x2(gate_packed_262[16]));
                        float2 _cvt_f32_97 = __bfloat1622float2(__as_bf16x2(up_packed_263[16]));
                        float gate_x_397 = _cvt_f32_96.x;
                        float gate_y_398 = _cvt_f32_96.y;
                        float up_x_399 = _cvt_f32_97.x;
                        float up_y_400 = _cvt_f32_97.y;
                        float _min_218 = fminf(gate_x_397, swiglu_limit);
                        gate_x_397 = _min_218;
                        float _min_219 = fminf(gate_y_398, swiglu_limit);
                        gate_y_398 = _min_219;
                        float _fmax_112 = fmaxf(up_x_399, -swiglu_limit);
                        float _min_220 = fminf(_fmax_112, swiglu_limit);
                        up_x_399 = _min_220;
                        float _fmax_113 = fmaxf(up_y_400, -swiglu_limit);
                        float _min_221 = fminf(_fmax_113, swiglu_limit);
                        up_y_400 = _min_221;
                        float _exp_96 = expf(gate_x_397 * -1.0f);
                        float denominator_x_401 = _exp_96 + 1.0f;
                        float _exp_97 = expf(gate_y_398 * -1.0f);
                        float denominator_y_402 = _exp_97 + 1.0f;
                        float hidden_x_403 = gate_x_397 / denominator_x_401 * up_x_399;
                        float hidden_y_404 = gate_y_398 / denominator_y_402 * up_y_400;
                        __nv_bfloat162 _bf16x2_184 = __float22bfloat162_rn(make_float2(hidden_x_403, hidden_y_404));
                        hidden_packed_264[16] = __as_u32(_bf16x2_184);
                        float2 _cvt_f32_98 = __bfloat1622float2(__as_bf16x2(gate_packed_262[17]));
                        float2 _cvt_f32_99 = __bfloat1622float2(__as_bf16x2(up_packed_263[17]));
                        float gate_x_405 = _cvt_f32_98.x;
                        float gate_y_406 = _cvt_f32_98.y;
                        float up_x_407 = _cvt_f32_99.x;
                        float up_y_408 = _cvt_f32_99.y;
                        float _min_222 = fminf(gate_x_405, swiglu_limit);
                        gate_x_405 = _min_222;
                        float _min_223 = fminf(gate_y_406, swiglu_limit);
                        gate_y_406 = _min_223;
                        float _fmax_114 = fmaxf(up_x_407, -swiglu_limit);
                        float _min_224 = fminf(_fmax_114, swiglu_limit);
                        up_x_407 = _min_224;
                        float _fmax_115 = fmaxf(up_y_408, -swiglu_limit);
                        float _min_225 = fminf(_fmax_115, swiglu_limit);
                        up_y_408 = _min_225;
                        float _exp_98 = expf(gate_x_405 * -1.0f);
                        float denominator_x_409 = _exp_98 + 1.0f;
                        float _exp_99 = expf(gate_y_406 * -1.0f);
                        float denominator_y_410 = _exp_99 + 1.0f;
                        float hidden_x_411 = gate_x_405 / denominator_x_409 * up_x_407;
                        float hidden_y_412 = gate_y_406 / denominator_y_410 * up_y_408;
                        __nv_bfloat162 _bf16x2_185 = __float22bfloat162_rn(make_float2(hidden_x_411, hidden_y_412));
                        hidden_packed_264[17] = __as_u32(_bf16x2_185);
                        float2 _cvt_f32_100 = __bfloat1622float2(__as_bf16x2(gate_packed_262[18]));
                        float2 _cvt_f32_101 = __bfloat1622float2(__as_bf16x2(up_packed_263[18]));
                        float gate_x_413 = _cvt_f32_100.x;
                        float gate_y_414 = _cvt_f32_100.y;
                        float up_x_415 = _cvt_f32_101.x;
                        float up_y_416 = _cvt_f32_101.y;
                        float _min_226 = fminf(gate_x_413, swiglu_limit);
                        gate_x_413 = _min_226;
                        float _min_227 = fminf(gate_y_414, swiglu_limit);
                        gate_y_414 = _min_227;
                        float _fmax_116 = fmaxf(up_x_415, -swiglu_limit);
                        float _min_228 = fminf(_fmax_116, swiglu_limit);
                        up_x_415 = _min_228;
                        float _fmax_117 = fmaxf(up_y_416, -swiglu_limit);
                        float _min_229 = fminf(_fmax_117, swiglu_limit);
                        up_y_416 = _min_229;
                        float _exp_100 = expf(gate_x_413 * -1.0f);
                        float denominator_x_417 = _exp_100 + 1.0f;
                        float _exp_101 = expf(gate_y_414 * -1.0f);
                        float denominator_y_418 = _exp_101 + 1.0f;
                        float hidden_x_419 = gate_x_413 / denominator_x_417 * up_x_415;
                        float hidden_y_420 = gate_y_414 / denominator_y_418 * up_y_416;
                        __nv_bfloat162 _bf16x2_186 = __float22bfloat162_rn(make_float2(hidden_x_419, hidden_y_420));
                        hidden_packed_264[18] = __as_u32(_bf16x2_186);
                        float2 _cvt_f32_102 = __bfloat1622float2(__as_bf16x2(gate_packed_262[19]));
                        float2 _cvt_f32_103 = __bfloat1622float2(__as_bf16x2(up_packed_263[19]));
                        float gate_x_421 = _cvt_f32_102.x;
                        float gate_y_422 = _cvt_f32_102.y;
                        float up_x_423 = _cvt_f32_103.x;
                        float up_y_424 = _cvt_f32_103.y;
                        float _min_230 = fminf(gate_x_421, swiglu_limit);
                        gate_x_421 = _min_230;
                        float _min_231 = fminf(gate_y_422, swiglu_limit);
                        gate_y_422 = _min_231;
                        float _fmax_118 = fmaxf(up_x_423, -swiglu_limit);
                        float _min_232 = fminf(_fmax_118, swiglu_limit);
                        up_x_423 = _min_232;
                        float _fmax_119 = fmaxf(up_y_424, -swiglu_limit);
                        float _min_233 = fminf(_fmax_119, swiglu_limit);
                        up_y_424 = _min_233;
                        float _exp_102 = expf(gate_x_421 * -1.0f);
                        float denominator_x_425 = _exp_102 + 1.0f;
                        float _exp_103 = expf(gate_y_422 * -1.0f);
                        float denominator_y_426 = _exp_103 + 1.0f;
                        float hidden_x_427 = gate_x_421 / denominator_x_425 * up_x_423;
                        float hidden_y_428 = gate_y_422 / denominator_y_426 * up_y_424;
                        __nv_bfloat162 _bf16x2_187 = __float22bfloat162_rn(make_float2(hidden_x_427, hidden_y_428));
                        hidden_packed_264[19] = __as_u32(_bf16x2_187);
                        float2 _cvt_f32_104 = __bfloat1622float2(__as_bf16x2(gate_packed_262[20]));
                        float2 _cvt_f32_105 = __bfloat1622float2(__as_bf16x2(up_packed_263[20]));
                        float gate_x_429 = _cvt_f32_104.x;
                        float gate_y_430 = _cvt_f32_104.y;
                        float up_x_431 = _cvt_f32_105.x;
                        float up_y_432 = _cvt_f32_105.y;
                        float _min_234 = fminf(gate_x_429, swiglu_limit);
                        gate_x_429 = _min_234;
                        float _min_235 = fminf(gate_y_430, swiglu_limit);
                        gate_y_430 = _min_235;
                        float _fmax_120 = fmaxf(up_x_431, -swiglu_limit);
                        float _min_236 = fminf(_fmax_120, swiglu_limit);
                        up_x_431 = _min_236;
                        float _fmax_121 = fmaxf(up_y_432, -swiglu_limit);
                        float _min_237 = fminf(_fmax_121, swiglu_limit);
                        up_y_432 = _min_237;
                        float _exp_104 = expf(gate_x_429 * -1.0f);
                        float denominator_x_433 = _exp_104 + 1.0f;
                        float _exp_105 = expf(gate_y_430 * -1.0f);
                        float denominator_y_434 = _exp_105 + 1.0f;
                        float hidden_x_435 = gate_x_429 / denominator_x_433 * up_x_431;
                        float hidden_y_436 = gate_y_430 / denominator_y_434 * up_y_432;
                        __nv_bfloat162 _bf16x2_188 = __float22bfloat162_rn(make_float2(hidden_x_435, hidden_y_436));
                        hidden_packed_264[20] = __as_u32(_bf16x2_188);
                        float2 _cvt_f32_106 = __bfloat1622float2(__as_bf16x2(gate_packed_262[21]));
                        float2 _cvt_f32_107 = __bfloat1622float2(__as_bf16x2(up_packed_263[21]));
                        float gate_x_437 = _cvt_f32_106.x;
                        float gate_y_438 = _cvt_f32_106.y;
                        float up_x_439 = _cvt_f32_107.x;
                        float up_y_440 = _cvt_f32_107.y;
                        float _min_238 = fminf(gate_x_437, swiglu_limit);
                        gate_x_437 = _min_238;
                        float _min_239 = fminf(gate_y_438, swiglu_limit);
                        gate_y_438 = _min_239;
                        float _fmax_122 = fmaxf(up_x_439, -swiglu_limit);
                        float _min_240 = fminf(_fmax_122, swiglu_limit);
                        up_x_439 = _min_240;
                        float _fmax_123 = fmaxf(up_y_440, -swiglu_limit);
                        float _min_241 = fminf(_fmax_123, swiglu_limit);
                        up_y_440 = _min_241;
                        float _exp_106 = expf(gate_x_437 * -1.0f);
                        float denominator_x_441 = _exp_106 + 1.0f;
                        float _exp_107 = expf(gate_y_438 * -1.0f);
                        float denominator_y_442 = _exp_107 + 1.0f;
                        float hidden_x_443 = gate_x_437 / denominator_x_441 * up_x_439;
                        float hidden_y_444 = gate_y_438 / denominator_y_442 * up_y_440;
                        __nv_bfloat162 _bf16x2_189 = __float22bfloat162_rn(make_float2(hidden_x_443, hidden_y_444));
                        hidden_packed_264[21] = __as_u32(_bf16x2_189);
                        float2 _cvt_f32_108 = __bfloat1622float2(__as_bf16x2(gate_packed_262[22]));
                        float2 _cvt_f32_109 = __bfloat1622float2(__as_bf16x2(up_packed_263[22]));
                        float gate_x_445 = _cvt_f32_108.x;
                        float gate_y_446 = _cvt_f32_108.y;
                        float up_x_447 = _cvt_f32_109.x;
                        float up_y_448 = _cvt_f32_109.y;
                        float _min_242 = fminf(gate_x_445, swiglu_limit);
                        gate_x_445 = _min_242;
                        float _min_243 = fminf(gate_y_446, swiglu_limit);
                        gate_y_446 = _min_243;
                        float _fmax_124 = fmaxf(up_x_447, -swiglu_limit);
                        float _min_244 = fminf(_fmax_124, swiglu_limit);
                        up_x_447 = _min_244;
                        float _fmax_125 = fmaxf(up_y_448, -swiglu_limit);
                        float _min_245 = fminf(_fmax_125, swiglu_limit);
                        up_y_448 = _min_245;
                        float _exp_108 = expf(gate_x_445 * -1.0f);
                        float denominator_x_449 = _exp_108 + 1.0f;
                        float _exp_109 = expf(gate_y_446 * -1.0f);
                        float denominator_y_450 = _exp_109 + 1.0f;
                        float hidden_x_451 = gate_x_445 / denominator_x_449 * up_x_447;
                        float hidden_y_452 = gate_y_446 / denominator_y_450 * up_y_448;
                        __nv_bfloat162 _bf16x2_190 = __float22bfloat162_rn(make_float2(hidden_x_451, hidden_y_452));
                        hidden_packed_264[22] = __as_u32(_bf16x2_190);
                        float2 _cvt_f32_110 = __bfloat1622float2(__as_bf16x2(gate_packed_262[23]));
                        float2 _cvt_f32_111 = __bfloat1622float2(__as_bf16x2(up_packed_263[23]));
                        float gate_x_453 = _cvt_f32_110.x;
                        float gate_y_454 = _cvt_f32_110.y;
                        float up_x_455 = _cvt_f32_111.x;
                        float up_y_456 = _cvt_f32_111.y;
                        float _min_246 = fminf(gate_x_453, swiglu_limit);
                        gate_x_453 = _min_246;
                        float _min_247 = fminf(gate_y_454, swiglu_limit);
                        gate_y_454 = _min_247;
                        float _fmax_126 = fmaxf(up_x_455, -swiglu_limit);
                        float _min_248 = fminf(_fmax_126, swiglu_limit);
                        up_x_455 = _min_248;
                        float _fmax_127 = fmaxf(up_y_456, -swiglu_limit);
                        float _min_249 = fminf(_fmax_127, swiglu_limit);
                        up_y_456 = _min_249;
                        float _exp_110 = expf(gate_x_453 * -1.0f);
                        float denominator_x_457 = _exp_110 + 1.0f;
                        float _exp_111 = expf(gate_y_454 * -1.0f);
                        float denominator_y_458 = _exp_111 + 1.0f;
                        float hidden_x_459 = gate_x_453 / denominator_x_457 * up_x_455;
                        float hidden_y_460 = gate_y_454 / denominator_y_458 * up_y_456;
                        __nv_bfloat162 _bf16x2_191 = __float22bfloat162_rn(make_float2(hidden_x_459, hidden_y_460));
                        hidden_packed_264[23] = __as_u32(_bf16x2_191);
                        float2 _cvt_f32_112 = __bfloat1622float2(__as_bf16x2(gate_packed_262[24]));
                        float2 _cvt_f32_113 = __bfloat1622float2(__as_bf16x2(up_packed_263[24]));
                        float gate_x_461 = _cvt_f32_112.x;
                        float gate_y_462 = _cvt_f32_112.y;
                        float up_x_463 = _cvt_f32_113.x;
                        float up_y_464 = _cvt_f32_113.y;
                        float _min_250 = fminf(gate_x_461, swiglu_limit);
                        gate_x_461 = _min_250;
                        float _min_251 = fminf(gate_y_462, swiglu_limit);
                        gate_y_462 = _min_251;
                        float _fmax_128 = fmaxf(up_x_463, -swiglu_limit);
                        float _min_252 = fminf(_fmax_128, swiglu_limit);
                        up_x_463 = _min_252;
                        float _fmax_129 = fmaxf(up_y_464, -swiglu_limit);
                        float _min_253 = fminf(_fmax_129, swiglu_limit);
                        up_y_464 = _min_253;
                        float _exp_112 = expf(gate_x_461 * -1.0f);
                        float denominator_x_465 = _exp_112 + 1.0f;
                        float _exp_113 = expf(gate_y_462 * -1.0f);
                        float denominator_y_466 = _exp_113 + 1.0f;
                        float hidden_x_467 = gate_x_461 / denominator_x_465 * up_x_463;
                        float hidden_y_468 = gate_y_462 / denominator_y_466 * up_y_464;
                        __nv_bfloat162 _bf16x2_192 = __float22bfloat162_rn(make_float2(hidden_x_467, hidden_y_468));
                        hidden_packed_264[24] = __as_u32(_bf16x2_192);
                        float2 _cvt_f32_114 = __bfloat1622float2(__as_bf16x2(gate_packed_262[25]));
                        float2 _cvt_f32_115 = __bfloat1622float2(__as_bf16x2(up_packed_263[25]));
                        float gate_x_469 = _cvt_f32_114.x;
                        float gate_y_470 = _cvt_f32_114.y;
                        float up_x_471 = _cvt_f32_115.x;
                        float up_y_472 = _cvt_f32_115.y;
                        float _min_254 = fminf(gate_x_469, swiglu_limit);
                        gate_x_469 = _min_254;
                        float _min_255 = fminf(gate_y_470, swiglu_limit);
                        gate_y_470 = _min_255;
                        float _fmax_130 = fmaxf(up_x_471, -swiglu_limit);
                        float _min_256 = fminf(_fmax_130, swiglu_limit);
                        up_x_471 = _min_256;
                        float _fmax_131 = fmaxf(up_y_472, -swiglu_limit);
                        float _min_257 = fminf(_fmax_131, swiglu_limit);
                        up_y_472 = _min_257;
                        float _exp_114 = expf(gate_x_469 * -1.0f);
                        float denominator_x_473 = _exp_114 + 1.0f;
                        float _exp_115 = expf(gate_y_470 * -1.0f);
                        float denominator_y_474 = _exp_115 + 1.0f;
                        float hidden_x_475 = gate_x_469 / denominator_x_473 * up_x_471;
                        float hidden_y_476 = gate_y_470 / denominator_y_474 * up_y_472;
                        __nv_bfloat162 _bf16x2_193 = __float22bfloat162_rn(make_float2(hidden_x_475, hidden_y_476));
                        hidden_packed_264[25] = __as_u32(_bf16x2_193);
                        float2 _cvt_f32_116 = __bfloat1622float2(__as_bf16x2(gate_packed_262[26]));
                        float2 _cvt_f32_117 = __bfloat1622float2(__as_bf16x2(up_packed_263[26]));
                        float gate_x_477 = _cvt_f32_116.x;
                        float gate_y_478 = _cvt_f32_116.y;
                        float up_x_479 = _cvt_f32_117.x;
                        float up_y_480 = _cvt_f32_117.y;
                        float _min_258 = fminf(gate_x_477, swiglu_limit);
                        gate_x_477 = _min_258;
                        float _min_259 = fminf(gate_y_478, swiglu_limit);
                        gate_y_478 = _min_259;
                        float _fmax_132 = fmaxf(up_x_479, -swiglu_limit);
                        float _min_260 = fminf(_fmax_132, swiglu_limit);
                        up_x_479 = _min_260;
                        float _fmax_133 = fmaxf(up_y_480, -swiglu_limit);
                        float _min_261 = fminf(_fmax_133, swiglu_limit);
                        up_y_480 = _min_261;
                        float _exp_116 = expf(gate_x_477 * -1.0f);
                        float denominator_x_481 = _exp_116 + 1.0f;
                        float _exp_117 = expf(gate_y_478 * -1.0f);
                        float denominator_y_482 = _exp_117 + 1.0f;
                        float hidden_x_483 = gate_x_477 / denominator_x_481 * up_x_479;
                        float hidden_y_484 = gate_y_478 / denominator_y_482 * up_y_480;
                        __nv_bfloat162 _bf16x2_194 = __float22bfloat162_rn(make_float2(hidden_x_483, hidden_y_484));
                        hidden_packed_264[26] = __as_u32(_bf16x2_194);
                        float2 _cvt_f32_118 = __bfloat1622float2(__as_bf16x2(gate_packed_262[27]));
                        float2 _cvt_f32_119 = __bfloat1622float2(__as_bf16x2(up_packed_263[27]));
                        float gate_x_485 = _cvt_f32_118.x;
                        float gate_y_486 = _cvt_f32_118.y;
                        float up_x_487 = _cvt_f32_119.x;
                        float up_y_488 = _cvt_f32_119.y;
                        float _min_262 = fminf(gate_x_485, swiglu_limit);
                        gate_x_485 = _min_262;
                        float _min_263 = fminf(gate_y_486, swiglu_limit);
                        gate_y_486 = _min_263;
                        float _fmax_134 = fmaxf(up_x_487, -swiglu_limit);
                        float _min_264 = fminf(_fmax_134, swiglu_limit);
                        up_x_487 = _min_264;
                        float _fmax_135 = fmaxf(up_y_488, -swiglu_limit);
                        float _min_265 = fminf(_fmax_135, swiglu_limit);
                        up_y_488 = _min_265;
                        float _exp_118 = expf(gate_x_485 * -1.0f);
                        float denominator_x_489 = _exp_118 + 1.0f;
                        float _exp_119 = expf(gate_y_486 * -1.0f);
                        float denominator_y_490 = _exp_119 + 1.0f;
                        float hidden_x_491 = gate_x_485 / denominator_x_489 * up_x_487;
                        float hidden_y_492 = gate_y_486 / denominator_y_490 * up_y_488;
                        __nv_bfloat162 _bf16x2_195 = __float22bfloat162_rn(make_float2(hidden_x_491, hidden_y_492));
                        hidden_packed_264[27] = __as_u32(_bf16x2_195);
                        float2 _cvt_f32_120 = __bfloat1622float2(__as_bf16x2(gate_packed_262[28]));
                        float2 _cvt_f32_121 = __bfloat1622float2(__as_bf16x2(up_packed_263[28]));
                        float gate_x_493 = _cvt_f32_120.x;
                        float gate_y_494 = _cvt_f32_120.y;
                        float up_x_495 = _cvt_f32_121.x;
                        float up_y_496 = _cvt_f32_121.y;
                        float _min_266 = fminf(gate_x_493, swiglu_limit);
                        gate_x_493 = _min_266;
                        float _min_267 = fminf(gate_y_494, swiglu_limit);
                        gate_y_494 = _min_267;
                        float _fmax_136 = fmaxf(up_x_495, -swiglu_limit);
                        float _min_268 = fminf(_fmax_136, swiglu_limit);
                        up_x_495 = _min_268;
                        float _fmax_137 = fmaxf(up_y_496, -swiglu_limit);
                        float _min_269 = fminf(_fmax_137, swiglu_limit);
                        up_y_496 = _min_269;
                        float _exp_120 = expf(gate_x_493 * -1.0f);
                        float denominator_x_497 = _exp_120 + 1.0f;
                        float _exp_121 = expf(gate_y_494 * -1.0f);
                        float denominator_y_498 = _exp_121 + 1.0f;
                        float hidden_x_499 = gate_x_493 / denominator_x_497 * up_x_495;
                        float hidden_y_500 = gate_y_494 / denominator_y_498 * up_y_496;
                        __nv_bfloat162 _bf16x2_196 = __float22bfloat162_rn(make_float2(hidden_x_499, hidden_y_500));
                        hidden_packed_264[28] = __as_u32(_bf16x2_196);
                        float2 _cvt_f32_122 = __bfloat1622float2(__as_bf16x2(gate_packed_262[29]));
                        float2 _cvt_f32_123 = __bfloat1622float2(__as_bf16x2(up_packed_263[29]));
                        float gate_x_501 = _cvt_f32_122.x;
                        float gate_y_502 = _cvt_f32_122.y;
                        float up_x_503 = _cvt_f32_123.x;
                        float up_y_504 = _cvt_f32_123.y;
                        float _min_270 = fminf(gate_x_501, swiglu_limit);
                        gate_x_501 = _min_270;
                        float _min_271 = fminf(gate_y_502, swiglu_limit);
                        gate_y_502 = _min_271;
                        float _fmax_138 = fmaxf(up_x_503, -swiglu_limit);
                        float _min_272 = fminf(_fmax_138, swiglu_limit);
                        up_x_503 = _min_272;
                        float _fmax_139 = fmaxf(up_y_504, -swiglu_limit);
                        float _min_273 = fminf(_fmax_139, swiglu_limit);
                        up_y_504 = _min_273;
                        float _exp_122 = expf(gate_x_501 * -1.0f);
                        float denominator_x_505 = _exp_122 + 1.0f;
                        float _exp_123 = expf(gate_y_502 * -1.0f);
                        float denominator_y_506 = _exp_123 + 1.0f;
                        float hidden_x_507 = gate_x_501 / denominator_x_505 * up_x_503;
                        float hidden_y_508 = gate_y_502 / denominator_y_506 * up_y_504;
                        __nv_bfloat162 _bf16x2_197 = __float22bfloat162_rn(make_float2(hidden_x_507, hidden_y_508));
                        hidden_packed_264[29] = __as_u32(_bf16x2_197);
                        float2 _cvt_f32_124 = __bfloat1622float2(__as_bf16x2(gate_packed_262[30]));
                        float2 _cvt_f32_125 = __bfloat1622float2(__as_bf16x2(up_packed_263[30]));
                        float gate_x_509 = _cvt_f32_124.x;
                        float gate_y_510 = _cvt_f32_124.y;
                        float up_x_511 = _cvt_f32_125.x;
                        float up_y_512 = _cvt_f32_125.y;
                        float _min_274 = fminf(gate_x_509, swiglu_limit);
                        gate_x_509 = _min_274;
                        float _min_275 = fminf(gate_y_510, swiglu_limit);
                        gate_y_510 = _min_275;
                        float _fmax_140 = fmaxf(up_x_511, -swiglu_limit);
                        float _min_276 = fminf(_fmax_140, swiglu_limit);
                        up_x_511 = _min_276;
                        float _fmax_141 = fmaxf(up_y_512, -swiglu_limit);
                        float _min_277 = fminf(_fmax_141, swiglu_limit);
                        up_y_512 = _min_277;
                        float _exp_124 = expf(gate_x_509 * -1.0f);
                        float denominator_x_513 = _exp_124 + 1.0f;
                        float _exp_125 = expf(gate_y_510 * -1.0f);
                        float denominator_y_514 = _exp_125 + 1.0f;
                        float hidden_x_515 = gate_x_509 / denominator_x_513 * up_x_511;
                        float hidden_y_516 = gate_y_510 / denominator_y_514 * up_y_512;
                        __nv_bfloat162 _bf16x2_198 = __float22bfloat162_rn(make_float2(hidden_x_515, hidden_y_516));
                        hidden_packed_264[30] = __as_u32(_bf16x2_198);
                        float2 _cvt_f32_126 = __bfloat1622float2(__as_bf16x2(gate_packed_262[31]));
                        float2 _cvt_f32_127 = __bfloat1622float2(__as_bf16x2(up_packed_263[31]));
                        float gate_x_517 = _cvt_f32_126.x;
                        float gate_y_518 = _cvt_f32_126.y;
                        float up_x_519 = _cvt_f32_127.x;
                        float up_y_520 = _cvt_f32_127.y;
                        float _min_278 = fminf(gate_x_517, swiglu_limit);
                        gate_x_517 = _min_278;
                        float _min_279 = fminf(gate_y_518, swiglu_limit);
                        gate_y_518 = _min_279;
                        float _fmax_142 = fmaxf(up_x_519, -swiglu_limit);
                        float _min_280 = fminf(_fmax_142, swiglu_limit);
                        up_x_519 = _min_280;
                        float _fmax_143 = fmaxf(up_y_520, -swiglu_limit);
                        float _min_281 = fminf(_fmax_143, swiglu_limit);
                        up_y_520 = _min_281;
                        float _exp_126 = expf(gate_x_517 * -1.0f);
                        float denominator_x_521 = _exp_126 + 1.0f;
                        float _exp_127 = expf(gate_y_518 * -1.0f);
                        float denominator_y_522 = _exp_127 + 1.0f;
                        float hidden_x_523 = gate_x_517 / denominator_x_521 * up_x_519;
                        float hidden_y_524 = gate_y_518 / denominator_y_522 * up_y_520;
                        __nv_bfloat162 _bf16x2_199 = __float22bfloat162_rn(make_float2(hidden_x_523, hidden_y_524));
                        hidden_packed_264[31] = __as_u32(_bf16x2_199);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_525 = tid / 32;
                        int lane_526 = tid % 32;
                        #pragma unroll
                        for (int half_6 = 0; half_6 < 2; half_6++) {
                            #pragma unroll
                            for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                int row_11 = warp_525 * 32 + half_6 * 16 + lane_526 % 16;
                                int col_39 = col_tile_14 * 16 + lane_526 / 16 * 8;
                                unsigned int address_3_6 = d_smem_addr + (unsigned int)((row_11 * 32 + col_39) * 2);
                                address_3_6 = address_3_6 ^ (address_3_6 & 511) >> 7 << 4;
                                int offset_6 = half_6 * 8 + col_tile_14 * 4;
                                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(address_3_6);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_527 = tid / 32;
                        int lane_528 = tid % 32;
                        #pragma unroll
                        for (int half_7 = 0; half_7 < 2; half_7++) {
                            #pragma unroll
                            for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                int row_12 = warp_527 * 32 + half_7 * 16 + lane_528 % 16;
                                int col_40 = col_tile_15 * 16 + lane_528 / 16 * 8;
                                unsigned int address_3_7 = d_smem_addr + 8192 + (unsigned int)((row_12 * 32 + col_40) * 2);
                                address_3_7 = address_3_7 ^ (address_3_7 & 511) >> 7 << 4;
                                int offset_7 = half_7 * 8 + col_tile_15 * 4;
                                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(address_3_7);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_529 = tid / 32;
                        int lane_530 = tid % 32;
                        #pragma unroll
                        for (int half_8 = 0; half_8 < 2; half_8++) {
                            #pragma unroll
                            for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                int row_13 = warp_529 * 32 + half_8 * 16 + lane_530 % 16;
                                int col_41 = col_tile_16 * 16 + lane_530 / 16 * 8;
                                unsigned int address_3_8 = d_smem_addr + 16384 + (unsigned int)((row_13 * 32 + col_41) * 2);
                                address_3_8 = address_3_8 ^ (address_3_8 & 511) >> 7 << 4;
                                int offset_8 = half_8 * 8 + col_tile_16 * 4;
                                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(address_3_8);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_531 = tid / 32;
                        int lane_532 = tid % 32;
                        #pragma unroll
                        for (int half_9 = 0; half_9 < 2; half_9++) {
                            #pragma unroll
                            for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                int row_14 = warp_531 * 32 + half_9 * 16 + lane_532 % 16;
                                int col_42 = col_tile_17 * 16 + lane_532 / 16 * 8;
                                unsigned int address_3_9 = d_smem_addr + (unsigned int)((row_14 * 32 + col_42) * 2);
                                address_3_9 = address_3_9 ^ (address_3_9 & 511) >> 7 << 4;
                                int offset_9 = 16 + half_9 * 8 + col_tile_17 * 4;
                                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(address_3_9);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_533 = tid / 32;
                        int lane_534 = tid % 32;
                        #pragma unroll
                        for (int half_10 = 0; half_10 < 2; half_10++) {
                            #pragma unroll
                            for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                int row_15 = warp_533 * 32 + half_10 * 16 + lane_534 % 16;
                                int col_43 = col_tile_18 * 16 + lane_534 / 16 * 8;
                                unsigned int address_3_10 = d_smem_addr + 8192 + (unsigned int)((row_15 * 32 + col_43) * 2);
                                address_3_10 = address_3_10 ^ (address_3_10 & 511) >> 7 << 4;
                                int offset_10 = 16 + half_10 * 8 + col_tile_18 * 4;
                                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(address_3_10);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_535 = tid / 32;
                        int lane_536 = tid % 32;
                        #pragma unroll
                        for (int half_11 = 0; half_11 < 2; half_11++) {
                            #pragma unroll
                            for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                int row_16 = warp_535 * 32 + half_11 * 16 + lane_536 % 16;
                                int col_44 = col_tile_19 * 16 + lane_536 / 16 * 8;
                                unsigned int address_3_11 = d_smem_addr + 16384 + (unsigned int)((row_16 * 32 + col_44) * 2);
                                address_3_11 = address_3_11 ^ (address_3_11 & 511) >> 7 << 4;
                                int offset_11 = 16 + half_11 * 8 + col_tile_19 * 4;
                                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(address_3_11);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        unsigned int gate_packed_537[32];
                        unsigned int up_packed_538[32];
                        unsigned int hidden_packed_539[32];
                        unsigned int address_540 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 128;
                        float _tmem_load_16[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                            : "r"(address_540));
                        float _tmem_load_17[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15]))
                            : "r"(address_540 + 256));
                        __nv_bfloat162 _bf16x2_200 = __float22bfloat162_rn(make_float2(_tmem_load_16[0], _tmem_load_16[1]));
                        gate_packed_537[0] = __as_u32(_bf16x2_200);
                        __nv_bfloat162 _bf16x2_201 = __float22bfloat162_rn(make_float2(_tmem_load_17[0], _tmem_load_17[1]));
                        up_packed_538[0] = __as_u32(_bf16x2_201);
                        __nv_bfloat162 _bf16x2_202 = __float22bfloat162_rn(make_float2(_tmem_load_16[2], _tmem_load_16[3]));
                        gate_packed_537[1] = __as_u32(_bf16x2_202);
                        __nv_bfloat162 _bf16x2_203 = __float22bfloat162_rn(make_float2(_tmem_load_17[2], _tmem_load_17[3]));
                        up_packed_538[1] = __as_u32(_bf16x2_203);
                        __nv_bfloat162 _bf16x2_204 = __float22bfloat162_rn(make_float2(_tmem_load_16[4], _tmem_load_16[5]));
                        gate_packed_537[2] = __as_u32(_bf16x2_204);
                        __nv_bfloat162 _bf16x2_205 = __float22bfloat162_rn(make_float2(_tmem_load_17[4], _tmem_load_17[5]));
                        up_packed_538[2] = __as_u32(_bf16x2_205);
                        __nv_bfloat162 _bf16x2_206 = __float22bfloat162_rn(make_float2(_tmem_load_16[6], _tmem_load_16[7]));
                        gate_packed_537[3] = __as_u32(_bf16x2_206);
                        __nv_bfloat162 _bf16x2_207 = __float22bfloat162_rn(make_float2(_tmem_load_17[6], _tmem_load_17[7]));
                        up_packed_538[3] = __as_u32(_bf16x2_207);
                        __nv_bfloat162 _bf16x2_208 = __float22bfloat162_rn(make_float2(_tmem_load_16[8], _tmem_load_16[9]));
                        gate_packed_537[4] = __as_u32(_bf16x2_208);
                        __nv_bfloat162 _bf16x2_209 = __float22bfloat162_rn(make_float2(_tmem_load_17[8], _tmem_load_17[9]));
                        up_packed_538[4] = __as_u32(_bf16x2_209);
                        __nv_bfloat162 _bf16x2_210 = __float22bfloat162_rn(make_float2(_tmem_load_16[10], _tmem_load_16[11]));
                        gate_packed_537[5] = __as_u32(_bf16x2_210);
                        __nv_bfloat162 _bf16x2_211 = __float22bfloat162_rn(make_float2(_tmem_load_17[10], _tmem_load_17[11]));
                        up_packed_538[5] = __as_u32(_bf16x2_211);
                        __nv_bfloat162 _bf16x2_212 = __float22bfloat162_rn(make_float2(_tmem_load_16[12], _tmem_load_16[13]));
                        gate_packed_537[6] = __as_u32(_bf16x2_212);
                        __nv_bfloat162 _bf16x2_213 = __float22bfloat162_rn(make_float2(_tmem_load_17[12], _tmem_load_17[13]));
                        up_packed_538[6] = __as_u32(_bf16x2_213);
                        __nv_bfloat162 _bf16x2_214 = __float22bfloat162_rn(make_float2(_tmem_load_16[14], _tmem_load_16[15]));
                        gate_packed_537[7] = __as_u32(_bf16x2_214);
                        __nv_bfloat162 _bf16x2_215 = __float22bfloat162_rn(make_float2(_tmem_load_17[14], _tmem_load_17[15]));
                        up_packed_538[7] = __as_u32(_bf16x2_215);
                        unsigned int address_541 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 128;
                        float _tmem_load_18[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[15]))
                            : "r"(address_541));
                        float _tmem_load_19[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[15]))
                            : "r"(address_541 + 256));
                        __nv_bfloat162 _bf16x2_216 = __float22bfloat162_rn(make_float2(_tmem_load_18[0], _tmem_load_18[1]));
                        gate_packed_537[8] = __as_u32(_bf16x2_216);
                        __nv_bfloat162 _bf16x2_217 = __float22bfloat162_rn(make_float2(_tmem_load_19[0], _tmem_load_19[1]));
                        up_packed_538[8] = __as_u32(_bf16x2_217);
                        __nv_bfloat162 _bf16x2_218 = __float22bfloat162_rn(make_float2(_tmem_load_18[2], _tmem_load_18[3]));
                        gate_packed_537[9] = __as_u32(_bf16x2_218);
                        __nv_bfloat162 _bf16x2_219 = __float22bfloat162_rn(make_float2(_tmem_load_19[2], _tmem_load_19[3]));
                        up_packed_538[9] = __as_u32(_bf16x2_219);
                        __nv_bfloat162 _bf16x2_220 = __float22bfloat162_rn(make_float2(_tmem_load_18[4], _tmem_load_18[5]));
                        gate_packed_537[10] = __as_u32(_bf16x2_220);
                        __nv_bfloat162 _bf16x2_221 = __float22bfloat162_rn(make_float2(_tmem_load_19[4], _tmem_load_19[5]));
                        up_packed_538[10] = __as_u32(_bf16x2_221);
                        __nv_bfloat162 _bf16x2_222 = __float22bfloat162_rn(make_float2(_tmem_load_18[6], _tmem_load_18[7]));
                        gate_packed_537[11] = __as_u32(_bf16x2_222);
                        __nv_bfloat162 _bf16x2_223 = __float22bfloat162_rn(make_float2(_tmem_load_19[6], _tmem_load_19[7]));
                        up_packed_538[11] = __as_u32(_bf16x2_223);
                        __nv_bfloat162 _bf16x2_224 = __float22bfloat162_rn(make_float2(_tmem_load_18[8], _tmem_load_18[9]));
                        gate_packed_537[12] = __as_u32(_bf16x2_224);
                        __nv_bfloat162 _bf16x2_225 = __float22bfloat162_rn(make_float2(_tmem_load_19[8], _tmem_load_19[9]));
                        up_packed_538[12] = __as_u32(_bf16x2_225);
                        __nv_bfloat162 _bf16x2_226 = __float22bfloat162_rn(make_float2(_tmem_load_18[10], _tmem_load_18[11]));
                        gate_packed_537[13] = __as_u32(_bf16x2_226);
                        __nv_bfloat162 _bf16x2_227 = __float22bfloat162_rn(make_float2(_tmem_load_19[10], _tmem_load_19[11]));
                        up_packed_538[13] = __as_u32(_bf16x2_227);
                        __nv_bfloat162 _bf16x2_228 = __float22bfloat162_rn(make_float2(_tmem_load_18[12], _tmem_load_18[13]));
                        gate_packed_537[14] = __as_u32(_bf16x2_228);
                        __nv_bfloat162 _bf16x2_229 = __float22bfloat162_rn(make_float2(_tmem_load_19[12], _tmem_load_19[13]));
                        up_packed_538[14] = __as_u32(_bf16x2_229);
                        __nv_bfloat162 _bf16x2_230 = __float22bfloat162_rn(make_float2(_tmem_load_18[14], _tmem_load_18[15]));
                        gate_packed_537[15] = __as_u32(_bf16x2_230);
                        __nv_bfloat162 _bf16x2_231 = __float22bfloat162_rn(make_float2(_tmem_load_19[14], _tmem_load_19[15]));
                        up_packed_538[15] = __as_u32(_bf16x2_231);
                        unsigned int address_542 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 160;
                        float _tmem_load_20[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[15]))
                            : "r"(address_542));
                        float _tmem_load_21[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[15]))
                            : "r"(address_542 + 256));
                        __nv_bfloat162 _bf16x2_232 = __float22bfloat162_rn(make_float2(_tmem_load_20[0], _tmem_load_20[1]));
                        gate_packed_537[16] = __as_u32(_bf16x2_232);
                        __nv_bfloat162 _bf16x2_233 = __float22bfloat162_rn(make_float2(_tmem_load_21[0], _tmem_load_21[1]));
                        up_packed_538[16] = __as_u32(_bf16x2_233);
                        __nv_bfloat162 _bf16x2_234 = __float22bfloat162_rn(make_float2(_tmem_load_20[2], _tmem_load_20[3]));
                        gate_packed_537[17] = __as_u32(_bf16x2_234);
                        __nv_bfloat162 _bf16x2_235 = __float22bfloat162_rn(make_float2(_tmem_load_21[2], _tmem_load_21[3]));
                        up_packed_538[17] = __as_u32(_bf16x2_235);
                        __nv_bfloat162 _bf16x2_236 = __float22bfloat162_rn(make_float2(_tmem_load_20[4], _tmem_load_20[5]));
                        gate_packed_537[18] = __as_u32(_bf16x2_236);
                        __nv_bfloat162 _bf16x2_237 = __float22bfloat162_rn(make_float2(_tmem_load_21[4], _tmem_load_21[5]));
                        up_packed_538[18] = __as_u32(_bf16x2_237);
                        __nv_bfloat162 _bf16x2_238 = __float22bfloat162_rn(make_float2(_tmem_load_20[6], _tmem_load_20[7]));
                        gate_packed_537[19] = __as_u32(_bf16x2_238);
                        __nv_bfloat162 _bf16x2_239 = __float22bfloat162_rn(make_float2(_tmem_load_21[6], _tmem_load_21[7]));
                        up_packed_538[19] = __as_u32(_bf16x2_239);
                        __nv_bfloat162 _bf16x2_240 = __float22bfloat162_rn(make_float2(_tmem_load_20[8], _tmem_load_20[9]));
                        gate_packed_537[20] = __as_u32(_bf16x2_240);
                        __nv_bfloat162 _bf16x2_241 = __float22bfloat162_rn(make_float2(_tmem_load_21[8], _tmem_load_21[9]));
                        up_packed_538[20] = __as_u32(_bf16x2_241);
                        __nv_bfloat162 _bf16x2_242 = __float22bfloat162_rn(make_float2(_tmem_load_20[10], _tmem_load_20[11]));
                        gate_packed_537[21] = __as_u32(_bf16x2_242);
                        __nv_bfloat162 _bf16x2_243 = __float22bfloat162_rn(make_float2(_tmem_load_21[10], _tmem_load_21[11]));
                        up_packed_538[21] = __as_u32(_bf16x2_243);
                        __nv_bfloat162 _bf16x2_244 = __float22bfloat162_rn(make_float2(_tmem_load_20[12], _tmem_load_20[13]));
                        gate_packed_537[22] = __as_u32(_bf16x2_244);
                        __nv_bfloat162 _bf16x2_245 = __float22bfloat162_rn(make_float2(_tmem_load_21[12], _tmem_load_21[13]));
                        up_packed_538[22] = __as_u32(_bf16x2_245);
                        __nv_bfloat162 _bf16x2_246 = __float22bfloat162_rn(make_float2(_tmem_load_20[14], _tmem_load_20[15]));
                        gate_packed_537[23] = __as_u32(_bf16x2_246);
                        __nv_bfloat162 _bf16x2_247 = __float22bfloat162_rn(make_float2(_tmem_load_21[14], _tmem_load_21[15]));
                        up_packed_538[23] = __as_u32(_bf16x2_247);
                        unsigned int address_543 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 160;
                        float _tmem_load_22[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[15]))
                            : "r"(address_543));
                        float _tmem_load_23[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[15]))
                            : "r"(address_543 + 256));
                        __nv_bfloat162 _bf16x2_248 = __float22bfloat162_rn(make_float2(_tmem_load_22[0], _tmem_load_22[1]));
                        gate_packed_537[24] = __as_u32(_bf16x2_248);
                        __nv_bfloat162 _bf16x2_249 = __float22bfloat162_rn(make_float2(_tmem_load_23[0], _tmem_load_23[1]));
                        up_packed_538[24] = __as_u32(_bf16x2_249);
                        __nv_bfloat162 _bf16x2_250 = __float22bfloat162_rn(make_float2(_tmem_load_22[2], _tmem_load_22[3]));
                        gate_packed_537[25] = __as_u32(_bf16x2_250);
                        __nv_bfloat162 _bf16x2_251 = __float22bfloat162_rn(make_float2(_tmem_load_23[2], _tmem_load_23[3]));
                        up_packed_538[25] = __as_u32(_bf16x2_251);
                        __nv_bfloat162 _bf16x2_252 = __float22bfloat162_rn(make_float2(_tmem_load_22[4], _tmem_load_22[5]));
                        gate_packed_537[26] = __as_u32(_bf16x2_252);
                        __nv_bfloat162 _bf16x2_253 = __float22bfloat162_rn(make_float2(_tmem_load_23[4], _tmem_load_23[5]));
                        up_packed_538[26] = __as_u32(_bf16x2_253);
                        __nv_bfloat162 _bf16x2_254 = __float22bfloat162_rn(make_float2(_tmem_load_22[6], _tmem_load_22[7]));
                        gate_packed_537[27] = __as_u32(_bf16x2_254);
                        __nv_bfloat162 _bf16x2_255 = __float22bfloat162_rn(make_float2(_tmem_load_23[6], _tmem_load_23[7]));
                        up_packed_538[27] = __as_u32(_bf16x2_255);
                        __nv_bfloat162 _bf16x2_256 = __float22bfloat162_rn(make_float2(_tmem_load_22[8], _tmem_load_22[9]));
                        gate_packed_537[28] = __as_u32(_bf16x2_256);
                        __nv_bfloat162 _bf16x2_257 = __float22bfloat162_rn(make_float2(_tmem_load_23[8], _tmem_load_23[9]));
                        up_packed_538[28] = __as_u32(_bf16x2_257);
                        __nv_bfloat162 _bf16x2_258 = __float22bfloat162_rn(make_float2(_tmem_load_22[10], _tmem_load_22[11]));
                        gate_packed_537[29] = __as_u32(_bf16x2_258);
                        __nv_bfloat162 _bf16x2_259 = __float22bfloat162_rn(make_float2(_tmem_load_23[10], _tmem_load_23[11]));
                        up_packed_538[29] = __as_u32(_bf16x2_259);
                        __nv_bfloat162 _bf16x2_260 = __float22bfloat162_rn(make_float2(_tmem_load_22[12], _tmem_load_22[13]));
                        gate_packed_537[30] = __as_u32(_bf16x2_260);
                        __nv_bfloat162 _bf16x2_261 = __float22bfloat162_rn(make_float2(_tmem_load_23[12], _tmem_load_23[13]));
                        up_packed_538[30] = __as_u32(_bf16x2_261);
                        __nv_bfloat162 _bf16x2_262 = __float22bfloat162_rn(make_float2(_tmem_load_22[14], _tmem_load_22[15]));
                        gate_packed_537[31] = __as_u32(_bf16x2_262);
                        __nv_bfloat162 _bf16x2_263 = __float22bfloat162_rn(make_float2(_tmem_load_23[14], _tmem_load_23[15]));
                        up_packed_538[31] = __as_u32(_bf16x2_263);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_128 = __bfloat1622float2(__as_bf16x2(gate_packed_537[0]));
                        float2 _cvt_f32_129 = __bfloat1622float2(__as_bf16x2(up_packed_538[0]));
                        float gate_x_544 = _cvt_f32_128.x;
                        float gate_y_545 = _cvt_f32_128.y;
                        float up_x_546 = _cvt_f32_129.x;
                        float up_y_547 = _cvt_f32_129.y;
                        float _min_282 = fminf(gate_x_544, swiglu_limit);
                        gate_x_544 = _min_282;
                        float _min_283 = fminf(gate_y_545, swiglu_limit);
                        gate_y_545 = _min_283;
                        float _fmax_144 = fmaxf(up_x_546, -swiglu_limit);
                        float _min_284 = fminf(_fmax_144, swiglu_limit);
                        up_x_546 = _min_284;
                        float _fmax_145 = fmaxf(up_y_547, -swiglu_limit);
                        float _min_285 = fminf(_fmax_145, swiglu_limit);
                        up_y_547 = _min_285;
                        float _exp_128 = expf(gate_x_544 * -1.0f);
                        float denominator_x_548 = _exp_128 + 1.0f;
                        float _exp_129 = expf(gate_y_545 * -1.0f);
                        float denominator_y_549 = _exp_129 + 1.0f;
                        float hidden_x_550 = gate_x_544 / denominator_x_548 * up_x_546;
                        float hidden_y_551 = gate_y_545 / denominator_y_549 * up_y_547;
                        __nv_bfloat162 _bf16x2_264 = __float22bfloat162_rn(make_float2(hidden_x_550, hidden_y_551));
                        hidden_packed_539[0] = __as_u32(_bf16x2_264);
                        float2 _cvt_f32_130 = __bfloat1622float2(__as_bf16x2(gate_packed_537[1]));
                        float2 _cvt_f32_131 = __bfloat1622float2(__as_bf16x2(up_packed_538[1]));
                        float gate_x_552 = _cvt_f32_130.x;
                        float gate_y_553 = _cvt_f32_130.y;
                        float up_x_554 = _cvt_f32_131.x;
                        float up_y_555 = _cvt_f32_131.y;
                        float _min_286 = fminf(gate_x_552, swiglu_limit);
                        gate_x_552 = _min_286;
                        float _min_287 = fminf(gate_y_553, swiglu_limit);
                        gate_y_553 = _min_287;
                        float _fmax_146 = fmaxf(up_x_554, -swiglu_limit);
                        float _min_288 = fminf(_fmax_146, swiglu_limit);
                        up_x_554 = _min_288;
                        float _fmax_147 = fmaxf(up_y_555, -swiglu_limit);
                        float _min_289 = fminf(_fmax_147, swiglu_limit);
                        up_y_555 = _min_289;
                        float _exp_130 = expf(gate_x_552 * -1.0f);
                        float denominator_x_556 = _exp_130 + 1.0f;
                        float _exp_131 = expf(gate_y_553 * -1.0f);
                        float denominator_y_557 = _exp_131 + 1.0f;
                        float hidden_x_558 = gate_x_552 / denominator_x_556 * up_x_554;
                        float hidden_y_559 = gate_y_553 / denominator_y_557 * up_y_555;
                        __nv_bfloat162 _bf16x2_265 = __float22bfloat162_rn(make_float2(hidden_x_558, hidden_y_559));
                        hidden_packed_539[1] = __as_u32(_bf16x2_265);
                        float2 _cvt_f32_132 = __bfloat1622float2(__as_bf16x2(gate_packed_537[2]));
                        float2 _cvt_f32_133 = __bfloat1622float2(__as_bf16x2(up_packed_538[2]));
                        float gate_x_560 = _cvt_f32_132.x;
                        float gate_y_561 = _cvt_f32_132.y;
                        float up_x_562 = _cvt_f32_133.x;
                        float up_y_563 = _cvt_f32_133.y;
                        float _min_290 = fminf(gate_x_560, swiglu_limit);
                        gate_x_560 = _min_290;
                        float _min_291 = fminf(gate_y_561, swiglu_limit);
                        gate_y_561 = _min_291;
                        float _fmax_148 = fmaxf(up_x_562, -swiglu_limit);
                        float _min_292 = fminf(_fmax_148, swiglu_limit);
                        up_x_562 = _min_292;
                        float _fmax_149 = fmaxf(up_y_563, -swiglu_limit);
                        float _min_293 = fminf(_fmax_149, swiglu_limit);
                        up_y_563 = _min_293;
                        float _exp_132 = expf(gate_x_560 * -1.0f);
                        float denominator_x_564 = _exp_132 + 1.0f;
                        float _exp_133 = expf(gate_y_561 * -1.0f);
                        float denominator_y_565 = _exp_133 + 1.0f;
                        float hidden_x_566 = gate_x_560 / denominator_x_564 * up_x_562;
                        float hidden_y_567 = gate_y_561 / denominator_y_565 * up_y_563;
                        __nv_bfloat162 _bf16x2_266 = __float22bfloat162_rn(make_float2(hidden_x_566, hidden_y_567));
                        hidden_packed_539[2] = __as_u32(_bf16x2_266);
                        float2 _cvt_f32_134 = __bfloat1622float2(__as_bf16x2(gate_packed_537[3]));
                        float2 _cvt_f32_135 = __bfloat1622float2(__as_bf16x2(up_packed_538[3]));
                        float gate_x_568 = _cvt_f32_134.x;
                        float gate_y_569 = _cvt_f32_134.y;
                        float up_x_570 = _cvt_f32_135.x;
                        float up_y_571 = _cvt_f32_135.y;
                        float _min_294 = fminf(gate_x_568, swiglu_limit);
                        gate_x_568 = _min_294;
                        float _min_295 = fminf(gate_y_569, swiglu_limit);
                        gate_y_569 = _min_295;
                        float _fmax_150 = fmaxf(up_x_570, -swiglu_limit);
                        float _min_296 = fminf(_fmax_150, swiglu_limit);
                        up_x_570 = _min_296;
                        float _fmax_151 = fmaxf(up_y_571, -swiglu_limit);
                        float _min_297 = fminf(_fmax_151, swiglu_limit);
                        up_y_571 = _min_297;
                        float _exp_134 = expf(gate_x_568 * -1.0f);
                        float denominator_x_572 = _exp_134 + 1.0f;
                        float _exp_135 = expf(gate_y_569 * -1.0f);
                        float denominator_y_573 = _exp_135 + 1.0f;
                        float hidden_x_574 = gate_x_568 / denominator_x_572 * up_x_570;
                        float hidden_y_575 = gate_y_569 / denominator_y_573 * up_y_571;
                        __nv_bfloat162 _bf16x2_267 = __float22bfloat162_rn(make_float2(hidden_x_574, hidden_y_575));
                        hidden_packed_539[3] = __as_u32(_bf16x2_267);
                        float2 _cvt_f32_136 = __bfloat1622float2(__as_bf16x2(gate_packed_537[4]));
                        float2 _cvt_f32_137 = __bfloat1622float2(__as_bf16x2(up_packed_538[4]));
                        float gate_x_576 = _cvt_f32_136.x;
                        float gate_y_577 = _cvt_f32_136.y;
                        float up_x_578 = _cvt_f32_137.x;
                        float up_y_579 = _cvt_f32_137.y;
                        float _min_298 = fminf(gate_x_576, swiglu_limit);
                        gate_x_576 = _min_298;
                        float _min_299 = fminf(gate_y_577, swiglu_limit);
                        gate_y_577 = _min_299;
                        float _fmax_152 = fmaxf(up_x_578, -swiglu_limit);
                        float _min_300 = fminf(_fmax_152, swiglu_limit);
                        up_x_578 = _min_300;
                        float _fmax_153 = fmaxf(up_y_579, -swiglu_limit);
                        float _min_301 = fminf(_fmax_153, swiglu_limit);
                        up_y_579 = _min_301;
                        float _exp_136 = expf(gate_x_576 * -1.0f);
                        float denominator_x_580 = _exp_136 + 1.0f;
                        float _exp_137 = expf(gate_y_577 * -1.0f);
                        float denominator_y_581 = _exp_137 + 1.0f;
                        float hidden_x_582 = gate_x_576 / denominator_x_580 * up_x_578;
                        float hidden_y_583 = gate_y_577 / denominator_y_581 * up_y_579;
                        __nv_bfloat162 _bf16x2_268 = __float22bfloat162_rn(make_float2(hidden_x_582, hidden_y_583));
                        hidden_packed_539[4] = __as_u32(_bf16x2_268);
                        float2 _cvt_f32_138 = __bfloat1622float2(__as_bf16x2(gate_packed_537[5]));
                        float2 _cvt_f32_139 = __bfloat1622float2(__as_bf16x2(up_packed_538[5]));
                        float gate_x_584 = _cvt_f32_138.x;
                        float gate_y_585 = _cvt_f32_138.y;
                        float up_x_586 = _cvt_f32_139.x;
                        float up_y_587 = _cvt_f32_139.y;
                        float _min_302 = fminf(gate_x_584, swiglu_limit);
                        gate_x_584 = _min_302;
                        float _min_303 = fminf(gate_y_585, swiglu_limit);
                        gate_y_585 = _min_303;
                        float _fmax_154 = fmaxf(up_x_586, -swiglu_limit);
                        float _min_304 = fminf(_fmax_154, swiglu_limit);
                        up_x_586 = _min_304;
                        float _fmax_155 = fmaxf(up_y_587, -swiglu_limit);
                        float _min_305 = fminf(_fmax_155, swiglu_limit);
                        up_y_587 = _min_305;
                        float _exp_138 = expf(gate_x_584 * -1.0f);
                        float denominator_x_588 = _exp_138 + 1.0f;
                        float _exp_139 = expf(gate_y_585 * -1.0f);
                        float denominator_y_589 = _exp_139 + 1.0f;
                        float hidden_x_590 = gate_x_584 / denominator_x_588 * up_x_586;
                        float hidden_y_591 = gate_y_585 / denominator_y_589 * up_y_587;
                        __nv_bfloat162 _bf16x2_269 = __float22bfloat162_rn(make_float2(hidden_x_590, hidden_y_591));
                        hidden_packed_539[5] = __as_u32(_bf16x2_269);
                        float2 _cvt_f32_140 = __bfloat1622float2(__as_bf16x2(gate_packed_537[6]));
                        float2 _cvt_f32_141 = __bfloat1622float2(__as_bf16x2(up_packed_538[6]));
                        float gate_x_592 = _cvt_f32_140.x;
                        float gate_y_593 = _cvt_f32_140.y;
                        float up_x_594 = _cvt_f32_141.x;
                        float up_y_595 = _cvt_f32_141.y;
                        float _min_306 = fminf(gate_x_592, swiglu_limit);
                        gate_x_592 = _min_306;
                        float _min_307 = fminf(gate_y_593, swiglu_limit);
                        gate_y_593 = _min_307;
                        float _fmax_156 = fmaxf(up_x_594, -swiglu_limit);
                        float _min_308 = fminf(_fmax_156, swiglu_limit);
                        up_x_594 = _min_308;
                        float _fmax_157 = fmaxf(up_y_595, -swiglu_limit);
                        float _min_309 = fminf(_fmax_157, swiglu_limit);
                        up_y_595 = _min_309;
                        float _exp_140 = expf(gate_x_592 * -1.0f);
                        float denominator_x_596 = _exp_140 + 1.0f;
                        float _exp_141 = expf(gate_y_593 * -1.0f);
                        float denominator_y_597 = _exp_141 + 1.0f;
                        float hidden_x_598 = gate_x_592 / denominator_x_596 * up_x_594;
                        float hidden_y_599 = gate_y_593 / denominator_y_597 * up_y_595;
                        __nv_bfloat162 _bf16x2_270 = __float22bfloat162_rn(make_float2(hidden_x_598, hidden_y_599));
                        hidden_packed_539[6] = __as_u32(_bf16x2_270);
                        float2 _cvt_f32_142 = __bfloat1622float2(__as_bf16x2(gate_packed_537[7]));
                        float2 _cvt_f32_143 = __bfloat1622float2(__as_bf16x2(up_packed_538[7]));
                        float gate_x_600 = _cvt_f32_142.x;
                        float gate_y_601 = _cvt_f32_142.y;
                        float up_x_602 = _cvt_f32_143.x;
                        float up_y_603 = _cvt_f32_143.y;
                        float _min_310 = fminf(gate_x_600, swiglu_limit);
                        gate_x_600 = _min_310;
                        float _min_311 = fminf(gate_y_601, swiglu_limit);
                        gate_y_601 = _min_311;
                        float _fmax_158 = fmaxf(up_x_602, -swiglu_limit);
                        float _min_312 = fminf(_fmax_158, swiglu_limit);
                        up_x_602 = _min_312;
                        float _fmax_159 = fmaxf(up_y_603, -swiglu_limit);
                        float _min_313 = fminf(_fmax_159, swiglu_limit);
                        up_y_603 = _min_313;
                        float _exp_142 = expf(gate_x_600 * -1.0f);
                        float denominator_x_604 = _exp_142 + 1.0f;
                        float _exp_143 = expf(gate_y_601 * -1.0f);
                        float denominator_y_605 = _exp_143 + 1.0f;
                        float hidden_x_606 = gate_x_600 / denominator_x_604 * up_x_602;
                        float hidden_y_607 = gate_y_601 / denominator_y_605 * up_y_603;
                        __nv_bfloat162 _bf16x2_271 = __float22bfloat162_rn(make_float2(hidden_x_606, hidden_y_607));
                        hidden_packed_539[7] = __as_u32(_bf16x2_271);
                        float2 _cvt_f32_144 = __bfloat1622float2(__as_bf16x2(gate_packed_537[8]));
                        float2 _cvt_f32_145 = __bfloat1622float2(__as_bf16x2(up_packed_538[8]));
                        float gate_x_608 = _cvt_f32_144.x;
                        float gate_y_609 = _cvt_f32_144.y;
                        float up_x_610 = _cvt_f32_145.x;
                        float up_y_611 = _cvt_f32_145.y;
                        float _min_314 = fminf(gate_x_608, swiglu_limit);
                        gate_x_608 = _min_314;
                        float _min_315 = fminf(gate_y_609, swiglu_limit);
                        gate_y_609 = _min_315;
                        float _fmax_160 = fmaxf(up_x_610, -swiglu_limit);
                        float _min_316 = fminf(_fmax_160, swiglu_limit);
                        up_x_610 = _min_316;
                        float _fmax_161 = fmaxf(up_y_611, -swiglu_limit);
                        float _min_317 = fminf(_fmax_161, swiglu_limit);
                        up_y_611 = _min_317;
                        float _exp_144 = expf(gate_x_608 * -1.0f);
                        float denominator_x_612 = _exp_144 + 1.0f;
                        float _exp_145 = expf(gate_y_609 * -1.0f);
                        float denominator_y_613 = _exp_145 + 1.0f;
                        float hidden_x_614 = gate_x_608 / denominator_x_612 * up_x_610;
                        float hidden_y_615 = gate_y_609 / denominator_y_613 * up_y_611;
                        __nv_bfloat162 _bf16x2_272 = __float22bfloat162_rn(make_float2(hidden_x_614, hidden_y_615));
                        hidden_packed_539[8] = __as_u32(_bf16x2_272);
                        float2 _cvt_f32_146 = __bfloat1622float2(__as_bf16x2(gate_packed_537[9]));
                        float2 _cvt_f32_147 = __bfloat1622float2(__as_bf16x2(up_packed_538[9]));
                        float gate_x_616 = _cvt_f32_146.x;
                        float gate_y_617 = _cvt_f32_146.y;
                        float up_x_618 = _cvt_f32_147.x;
                        float up_y_619 = _cvt_f32_147.y;
                        float _min_318 = fminf(gate_x_616, swiglu_limit);
                        gate_x_616 = _min_318;
                        float _min_319 = fminf(gate_y_617, swiglu_limit);
                        gate_y_617 = _min_319;
                        float _fmax_162 = fmaxf(up_x_618, -swiglu_limit);
                        float _min_320 = fminf(_fmax_162, swiglu_limit);
                        up_x_618 = _min_320;
                        float _fmax_163 = fmaxf(up_y_619, -swiglu_limit);
                        float _min_321 = fminf(_fmax_163, swiglu_limit);
                        up_y_619 = _min_321;
                        float _exp_146 = expf(gate_x_616 * -1.0f);
                        float denominator_x_620 = _exp_146 + 1.0f;
                        float _exp_147 = expf(gate_y_617 * -1.0f);
                        float denominator_y_621 = _exp_147 + 1.0f;
                        float hidden_x_622 = gate_x_616 / denominator_x_620 * up_x_618;
                        float hidden_y_623 = gate_y_617 / denominator_y_621 * up_y_619;
                        __nv_bfloat162 _bf16x2_273 = __float22bfloat162_rn(make_float2(hidden_x_622, hidden_y_623));
                        hidden_packed_539[9] = __as_u32(_bf16x2_273);
                        float2 _cvt_f32_148 = __bfloat1622float2(__as_bf16x2(gate_packed_537[10]));
                        float2 _cvt_f32_149 = __bfloat1622float2(__as_bf16x2(up_packed_538[10]));
                        float gate_x_624 = _cvt_f32_148.x;
                        float gate_y_625 = _cvt_f32_148.y;
                        float up_x_626 = _cvt_f32_149.x;
                        float up_y_627 = _cvt_f32_149.y;
                        float _min_322 = fminf(gate_x_624, swiglu_limit);
                        gate_x_624 = _min_322;
                        float _min_323 = fminf(gate_y_625, swiglu_limit);
                        gate_y_625 = _min_323;
                        float _fmax_164 = fmaxf(up_x_626, -swiglu_limit);
                        float _min_324 = fminf(_fmax_164, swiglu_limit);
                        up_x_626 = _min_324;
                        float _fmax_165 = fmaxf(up_y_627, -swiglu_limit);
                        float _min_325 = fminf(_fmax_165, swiglu_limit);
                        up_y_627 = _min_325;
                        float _exp_148 = expf(gate_x_624 * -1.0f);
                        float denominator_x_628 = _exp_148 + 1.0f;
                        float _exp_149 = expf(gate_y_625 * -1.0f);
                        float denominator_y_629 = _exp_149 + 1.0f;
                        float hidden_x_630 = gate_x_624 / denominator_x_628 * up_x_626;
                        float hidden_y_631 = gate_y_625 / denominator_y_629 * up_y_627;
                        __nv_bfloat162 _bf16x2_274 = __float22bfloat162_rn(make_float2(hidden_x_630, hidden_y_631));
                        hidden_packed_539[10] = __as_u32(_bf16x2_274);
                        float2 _cvt_f32_150 = __bfloat1622float2(__as_bf16x2(gate_packed_537[11]));
                        float2 _cvt_f32_151 = __bfloat1622float2(__as_bf16x2(up_packed_538[11]));
                        float gate_x_632 = _cvt_f32_150.x;
                        float gate_y_633 = _cvt_f32_150.y;
                        float up_x_634 = _cvt_f32_151.x;
                        float up_y_635 = _cvt_f32_151.y;
                        float _min_326 = fminf(gate_x_632, swiglu_limit);
                        gate_x_632 = _min_326;
                        float _min_327 = fminf(gate_y_633, swiglu_limit);
                        gate_y_633 = _min_327;
                        float _fmax_166 = fmaxf(up_x_634, -swiglu_limit);
                        float _min_328 = fminf(_fmax_166, swiglu_limit);
                        up_x_634 = _min_328;
                        float _fmax_167 = fmaxf(up_y_635, -swiglu_limit);
                        float _min_329 = fminf(_fmax_167, swiglu_limit);
                        up_y_635 = _min_329;
                        float _exp_150 = expf(gate_x_632 * -1.0f);
                        float denominator_x_636 = _exp_150 + 1.0f;
                        float _exp_151 = expf(gate_y_633 * -1.0f);
                        float denominator_y_637 = _exp_151 + 1.0f;
                        float hidden_x_638 = gate_x_632 / denominator_x_636 * up_x_634;
                        float hidden_y_639 = gate_y_633 / denominator_y_637 * up_y_635;
                        __nv_bfloat162 _bf16x2_275 = __float22bfloat162_rn(make_float2(hidden_x_638, hidden_y_639));
                        hidden_packed_539[11] = __as_u32(_bf16x2_275);
                        float2 _cvt_f32_152 = __bfloat1622float2(__as_bf16x2(gate_packed_537[12]));
                        float2 _cvt_f32_153 = __bfloat1622float2(__as_bf16x2(up_packed_538[12]));
                        float gate_x_640 = _cvt_f32_152.x;
                        float gate_y_641 = _cvt_f32_152.y;
                        float up_x_642 = _cvt_f32_153.x;
                        float up_y_643 = _cvt_f32_153.y;
                        float _min_330 = fminf(gate_x_640, swiglu_limit);
                        gate_x_640 = _min_330;
                        float _min_331 = fminf(gate_y_641, swiglu_limit);
                        gate_y_641 = _min_331;
                        float _fmax_168 = fmaxf(up_x_642, -swiglu_limit);
                        float _min_332 = fminf(_fmax_168, swiglu_limit);
                        up_x_642 = _min_332;
                        float _fmax_169 = fmaxf(up_y_643, -swiglu_limit);
                        float _min_333 = fminf(_fmax_169, swiglu_limit);
                        up_y_643 = _min_333;
                        float _exp_152 = expf(gate_x_640 * -1.0f);
                        float denominator_x_644 = _exp_152 + 1.0f;
                        float _exp_153 = expf(gate_y_641 * -1.0f);
                        float denominator_y_645 = _exp_153 + 1.0f;
                        float hidden_x_646 = gate_x_640 / denominator_x_644 * up_x_642;
                        float hidden_y_647 = gate_y_641 / denominator_y_645 * up_y_643;
                        __nv_bfloat162 _bf16x2_276 = __float22bfloat162_rn(make_float2(hidden_x_646, hidden_y_647));
                        hidden_packed_539[12] = __as_u32(_bf16x2_276);
                        float2 _cvt_f32_154 = __bfloat1622float2(__as_bf16x2(gate_packed_537[13]));
                        float2 _cvt_f32_155 = __bfloat1622float2(__as_bf16x2(up_packed_538[13]));
                        float gate_x_648 = _cvt_f32_154.x;
                        float gate_y_649 = _cvt_f32_154.y;
                        float up_x_650 = _cvt_f32_155.x;
                        float up_y_651 = _cvt_f32_155.y;
                        float _min_334 = fminf(gate_x_648, swiglu_limit);
                        gate_x_648 = _min_334;
                        float _min_335 = fminf(gate_y_649, swiglu_limit);
                        gate_y_649 = _min_335;
                        float _fmax_170 = fmaxf(up_x_650, -swiglu_limit);
                        float _min_336 = fminf(_fmax_170, swiglu_limit);
                        up_x_650 = _min_336;
                        float _fmax_171 = fmaxf(up_y_651, -swiglu_limit);
                        float _min_337 = fminf(_fmax_171, swiglu_limit);
                        up_y_651 = _min_337;
                        float _exp_154 = expf(gate_x_648 * -1.0f);
                        float denominator_x_652 = _exp_154 + 1.0f;
                        float _exp_155 = expf(gate_y_649 * -1.0f);
                        float denominator_y_653 = _exp_155 + 1.0f;
                        float hidden_x_654 = gate_x_648 / denominator_x_652 * up_x_650;
                        float hidden_y_655 = gate_y_649 / denominator_y_653 * up_y_651;
                        __nv_bfloat162 _bf16x2_277 = __float22bfloat162_rn(make_float2(hidden_x_654, hidden_y_655));
                        hidden_packed_539[13] = __as_u32(_bf16x2_277);
                        float2 _cvt_f32_156 = __bfloat1622float2(__as_bf16x2(gate_packed_537[14]));
                        float2 _cvt_f32_157 = __bfloat1622float2(__as_bf16x2(up_packed_538[14]));
                        float gate_x_656 = _cvt_f32_156.x;
                        float gate_y_657 = _cvt_f32_156.y;
                        float up_x_658 = _cvt_f32_157.x;
                        float up_y_659 = _cvt_f32_157.y;
                        float _min_338 = fminf(gate_x_656, swiglu_limit);
                        gate_x_656 = _min_338;
                        float _min_339 = fminf(gate_y_657, swiglu_limit);
                        gate_y_657 = _min_339;
                        float _fmax_172 = fmaxf(up_x_658, -swiglu_limit);
                        float _min_340 = fminf(_fmax_172, swiglu_limit);
                        up_x_658 = _min_340;
                        float _fmax_173 = fmaxf(up_y_659, -swiglu_limit);
                        float _min_341 = fminf(_fmax_173, swiglu_limit);
                        up_y_659 = _min_341;
                        float _exp_156 = expf(gate_x_656 * -1.0f);
                        float denominator_x_660 = _exp_156 + 1.0f;
                        float _exp_157 = expf(gate_y_657 * -1.0f);
                        float denominator_y_661 = _exp_157 + 1.0f;
                        float hidden_x_662 = gate_x_656 / denominator_x_660 * up_x_658;
                        float hidden_y_663 = gate_y_657 / denominator_y_661 * up_y_659;
                        __nv_bfloat162 _bf16x2_278 = __float22bfloat162_rn(make_float2(hidden_x_662, hidden_y_663));
                        hidden_packed_539[14] = __as_u32(_bf16x2_278);
                        float2 _cvt_f32_158 = __bfloat1622float2(__as_bf16x2(gate_packed_537[15]));
                        float2 _cvt_f32_159 = __bfloat1622float2(__as_bf16x2(up_packed_538[15]));
                        float gate_x_664 = _cvt_f32_158.x;
                        float gate_y_665 = _cvt_f32_158.y;
                        float up_x_666 = _cvt_f32_159.x;
                        float up_y_667 = _cvt_f32_159.y;
                        float _min_342 = fminf(gate_x_664, swiglu_limit);
                        gate_x_664 = _min_342;
                        float _min_343 = fminf(gate_y_665, swiglu_limit);
                        gate_y_665 = _min_343;
                        float _fmax_174 = fmaxf(up_x_666, -swiglu_limit);
                        float _min_344 = fminf(_fmax_174, swiglu_limit);
                        up_x_666 = _min_344;
                        float _fmax_175 = fmaxf(up_y_667, -swiglu_limit);
                        float _min_345 = fminf(_fmax_175, swiglu_limit);
                        up_y_667 = _min_345;
                        float _exp_158 = expf(gate_x_664 * -1.0f);
                        float denominator_x_668 = _exp_158 + 1.0f;
                        float _exp_159 = expf(gate_y_665 * -1.0f);
                        float denominator_y_669 = _exp_159 + 1.0f;
                        float hidden_x_670 = gate_x_664 / denominator_x_668 * up_x_666;
                        float hidden_y_671 = gate_y_665 / denominator_y_669 * up_y_667;
                        __nv_bfloat162 _bf16x2_279 = __float22bfloat162_rn(make_float2(hidden_x_670, hidden_y_671));
                        hidden_packed_539[15] = __as_u32(_bf16x2_279);
                        float2 _cvt_f32_160 = __bfloat1622float2(__as_bf16x2(gate_packed_537[16]));
                        float2 _cvt_f32_161 = __bfloat1622float2(__as_bf16x2(up_packed_538[16]));
                        float gate_x_672 = _cvt_f32_160.x;
                        float gate_y_673 = _cvt_f32_160.y;
                        float up_x_674 = _cvt_f32_161.x;
                        float up_y_675 = _cvt_f32_161.y;
                        float _min_346 = fminf(gate_x_672, swiglu_limit);
                        gate_x_672 = _min_346;
                        float _min_347 = fminf(gate_y_673, swiglu_limit);
                        gate_y_673 = _min_347;
                        float _fmax_176 = fmaxf(up_x_674, -swiglu_limit);
                        float _min_348 = fminf(_fmax_176, swiglu_limit);
                        up_x_674 = _min_348;
                        float _fmax_177 = fmaxf(up_y_675, -swiglu_limit);
                        float _min_349 = fminf(_fmax_177, swiglu_limit);
                        up_y_675 = _min_349;
                        float _exp_160 = expf(gate_x_672 * -1.0f);
                        float denominator_x_676 = _exp_160 + 1.0f;
                        float _exp_161 = expf(gate_y_673 * -1.0f);
                        float denominator_y_677 = _exp_161 + 1.0f;
                        float hidden_x_678 = gate_x_672 / denominator_x_676 * up_x_674;
                        float hidden_y_679 = gate_y_673 / denominator_y_677 * up_y_675;
                        __nv_bfloat162 _bf16x2_280 = __float22bfloat162_rn(make_float2(hidden_x_678, hidden_y_679));
                        hidden_packed_539[16] = __as_u32(_bf16x2_280);
                        float2 _cvt_f32_162 = __bfloat1622float2(__as_bf16x2(gate_packed_537[17]));
                        float2 _cvt_f32_163 = __bfloat1622float2(__as_bf16x2(up_packed_538[17]));
                        float gate_x_680 = _cvt_f32_162.x;
                        float gate_y_681 = _cvt_f32_162.y;
                        float up_x_682 = _cvt_f32_163.x;
                        float up_y_683 = _cvt_f32_163.y;
                        float _min_350 = fminf(gate_x_680, swiglu_limit);
                        gate_x_680 = _min_350;
                        float _min_351 = fminf(gate_y_681, swiglu_limit);
                        gate_y_681 = _min_351;
                        float _fmax_178 = fmaxf(up_x_682, -swiglu_limit);
                        float _min_352 = fminf(_fmax_178, swiglu_limit);
                        up_x_682 = _min_352;
                        float _fmax_179 = fmaxf(up_y_683, -swiglu_limit);
                        float _min_353 = fminf(_fmax_179, swiglu_limit);
                        up_y_683 = _min_353;
                        float _exp_162 = expf(gate_x_680 * -1.0f);
                        float denominator_x_684 = _exp_162 + 1.0f;
                        float _exp_163 = expf(gate_y_681 * -1.0f);
                        float denominator_y_685 = _exp_163 + 1.0f;
                        float hidden_x_686 = gate_x_680 / denominator_x_684 * up_x_682;
                        float hidden_y_687 = gate_y_681 / denominator_y_685 * up_y_683;
                        __nv_bfloat162 _bf16x2_281 = __float22bfloat162_rn(make_float2(hidden_x_686, hidden_y_687));
                        hidden_packed_539[17] = __as_u32(_bf16x2_281);
                        float2 _cvt_f32_164 = __bfloat1622float2(__as_bf16x2(gate_packed_537[18]));
                        float2 _cvt_f32_165 = __bfloat1622float2(__as_bf16x2(up_packed_538[18]));
                        float gate_x_688 = _cvt_f32_164.x;
                        float gate_y_689 = _cvt_f32_164.y;
                        float up_x_690 = _cvt_f32_165.x;
                        float up_y_691 = _cvt_f32_165.y;
                        float _min_354 = fminf(gate_x_688, swiglu_limit);
                        gate_x_688 = _min_354;
                        float _min_355 = fminf(gate_y_689, swiglu_limit);
                        gate_y_689 = _min_355;
                        float _fmax_180 = fmaxf(up_x_690, -swiglu_limit);
                        float _min_356 = fminf(_fmax_180, swiglu_limit);
                        up_x_690 = _min_356;
                        float _fmax_181 = fmaxf(up_y_691, -swiglu_limit);
                        float _min_357 = fminf(_fmax_181, swiglu_limit);
                        up_y_691 = _min_357;
                        float _exp_164 = expf(gate_x_688 * -1.0f);
                        float denominator_x_692 = _exp_164 + 1.0f;
                        float _exp_165 = expf(gate_y_689 * -1.0f);
                        float denominator_y_693 = _exp_165 + 1.0f;
                        float hidden_x_694 = gate_x_688 / denominator_x_692 * up_x_690;
                        float hidden_y_695 = gate_y_689 / denominator_y_693 * up_y_691;
                        __nv_bfloat162 _bf16x2_282 = __float22bfloat162_rn(make_float2(hidden_x_694, hidden_y_695));
                        hidden_packed_539[18] = __as_u32(_bf16x2_282);
                        float2 _cvt_f32_166 = __bfloat1622float2(__as_bf16x2(gate_packed_537[19]));
                        float2 _cvt_f32_167 = __bfloat1622float2(__as_bf16x2(up_packed_538[19]));
                        float gate_x_696 = _cvt_f32_166.x;
                        float gate_y_697 = _cvt_f32_166.y;
                        float up_x_698 = _cvt_f32_167.x;
                        float up_y_699 = _cvt_f32_167.y;
                        float _min_358 = fminf(gate_x_696, swiglu_limit);
                        gate_x_696 = _min_358;
                        float _min_359 = fminf(gate_y_697, swiglu_limit);
                        gate_y_697 = _min_359;
                        float _fmax_182 = fmaxf(up_x_698, -swiglu_limit);
                        float _min_360 = fminf(_fmax_182, swiglu_limit);
                        up_x_698 = _min_360;
                        float _fmax_183 = fmaxf(up_y_699, -swiglu_limit);
                        float _min_361 = fminf(_fmax_183, swiglu_limit);
                        up_y_699 = _min_361;
                        float _exp_166 = expf(gate_x_696 * -1.0f);
                        float denominator_x_700 = _exp_166 + 1.0f;
                        float _exp_167 = expf(gate_y_697 * -1.0f);
                        float denominator_y_701 = _exp_167 + 1.0f;
                        float hidden_x_702 = gate_x_696 / denominator_x_700 * up_x_698;
                        float hidden_y_703 = gate_y_697 / denominator_y_701 * up_y_699;
                        __nv_bfloat162 _bf16x2_283 = __float22bfloat162_rn(make_float2(hidden_x_702, hidden_y_703));
                        hidden_packed_539[19] = __as_u32(_bf16x2_283);
                        float2 _cvt_f32_168 = __bfloat1622float2(__as_bf16x2(gate_packed_537[20]));
                        float2 _cvt_f32_169 = __bfloat1622float2(__as_bf16x2(up_packed_538[20]));
                        float gate_x_704 = _cvt_f32_168.x;
                        float gate_y_705 = _cvt_f32_168.y;
                        float up_x_706 = _cvt_f32_169.x;
                        float up_y_707 = _cvt_f32_169.y;
                        float _min_362 = fminf(gate_x_704, swiglu_limit);
                        gate_x_704 = _min_362;
                        float _min_363 = fminf(gate_y_705, swiglu_limit);
                        gate_y_705 = _min_363;
                        float _fmax_184 = fmaxf(up_x_706, -swiglu_limit);
                        float _min_364 = fminf(_fmax_184, swiglu_limit);
                        up_x_706 = _min_364;
                        float _fmax_185 = fmaxf(up_y_707, -swiglu_limit);
                        float _min_365 = fminf(_fmax_185, swiglu_limit);
                        up_y_707 = _min_365;
                        float _exp_168 = expf(gate_x_704 * -1.0f);
                        float denominator_x_708 = _exp_168 + 1.0f;
                        float _exp_169 = expf(gate_y_705 * -1.0f);
                        float denominator_y_709 = _exp_169 + 1.0f;
                        float hidden_x_710 = gate_x_704 / denominator_x_708 * up_x_706;
                        float hidden_y_711 = gate_y_705 / denominator_y_709 * up_y_707;
                        __nv_bfloat162 _bf16x2_284 = __float22bfloat162_rn(make_float2(hidden_x_710, hidden_y_711));
                        hidden_packed_539[20] = __as_u32(_bf16x2_284);
                        float2 _cvt_f32_170 = __bfloat1622float2(__as_bf16x2(gate_packed_537[21]));
                        float2 _cvt_f32_171 = __bfloat1622float2(__as_bf16x2(up_packed_538[21]));
                        float gate_x_712 = _cvt_f32_170.x;
                        float gate_y_713 = _cvt_f32_170.y;
                        float up_x_714 = _cvt_f32_171.x;
                        float up_y_715 = _cvt_f32_171.y;
                        float _min_366 = fminf(gate_x_712, swiglu_limit);
                        gate_x_712 = _min_366;
                        float _min_367 = fminf(gate_y_713, swiglu_limit);
                        gate_y_713 = _min_367;
                        float _fmax_186 = fmaxf(up_x_714, -swiglu_limit);
                        float _min_368 = fminf(_fmax_186, swiglu_limit);
                        up_x_714 = _min_368;
                        float _fmax_187 = fmaxf(up_y_715, -swiglu_limit);
                        float _min_369 = fminf(_fmax_187, swiglu_limit);
                        up_y_715 = _min_369;
                        float _exp_170 = expf(gate_x_712 * -1.0f);
                        float denominator_x_716 = _exp_170 + 1.0f;
                        float _exp_171 = expf(gate_y_713 * -1.0f);
                        float denominator_y_717 = _exp_171 + 1.0f;
                        float hidden_x_718 = gate_x_712 / denominator_x_716 * up_x_714;
                        float hidden_y_719 = gate_y_713 / denominator_y_717 * up_y_715;
                        __nv_bfloat162 _bf16x2_285 = __float22bfloat162_rn(make_float2(hidden_x_718, hidden_y_719));
                        hidden_packed_539[21] = __as_u32(_bf16x2_285);
                        float2 _cvt_f32_172 = __bfloat1622float2(__as_bf16x2(gate_packed_537[22]));
                        float2 _cvt_f32_173 = __bfloat1622float2(__as_bf16x2(up_packed_538[22]));
                        float gate_x_720 = _cvt_f32_172.x;
                        float gate_y_721 = _cvt_f32_172.y;
                        float up_x_722 = _cvt_f32_173.x;
                        float up_y_723 = _cvt_f32_173.y;
                        float _min_370 = fminf(gate_x_720, swiglu_limit);
                        gate_x_720 = _min_370;
                        float _min_371 = fminf(gate_y_721, swiglu_limit);
                        gate_y_721 = _min_371;
                        float _fmax_188 = fmaxf(up_x_722, -swiglu_limit);
                        float _min_372 = fminf(_fmax_188, swiglu_limit);
                        up_x_722 = _min_372;
                        float _fmax_189 = fmaxf(up_y_723, -swiglu_limit);
                        float _min_373 = fminf(_fmax_189, swiglu_limit);
                        up_y_723 = _min_373;
                        float _exp_172 = expf(gate_x_720 * -1.0f);
                        float denominator_x_724 = _exp_172 + 1.0f;
                        float _exp_173 = expf(gate_y_721 * -1.0f);
                        float denominator_y_725 = _exp_173 + 1.0f;
                        float hidden_x_726 = gate_x_720 / denominator_x_724 * up_x_722;
                        float hidden_y_727 = gate_y_721 / denominator_y_725 * up_y_723;
                        __nv_bfloat162 _bf16x2_286 = __float22bfloat162_rn(make_float2(hidden_x_726, hidden_y_727));
                        hidden_packed_539[22] = __as_u32(_bf16x2_286);
                        float2 _cvt_f32_174 = __bfloat1622float2(__as_bf16x2(gate_packed_537[23]));
                        float2 _cvt_f32_175 = __bfloat1622float2(__as_bf16x2(up_packed_538[23]));
                        float gate_x_728 = _cvt_f32_174.x;
                        float gate_y_729 = _cvt_f32_174.y;
                        float up_x_730 = _cvt_f32_175.x;
                        float up_y_731 = _cvt_f32_175.y;
                        float _min_374 = fminf(gate_x_728, swiglu_limit);
                        gate_x_728 = _min_374;
                        float _min_375 = fminf(gate_y_729, swiglu_limit);
                        gate_y_729 = _min_375;
                        float _fmax_190 = fmaxf(up_x_730, -swiglu_limit);
                        float _min_376 = fminf(_fmax_190, swiglu_limit);
                        up_x_730 = _min_376;
                        float _fmax_191 = fmaxf(up_y_731, -swiglu_limit);
                        float _min_377 = fminf(_fmax_191, swiglu_limit);
                        up_y_731 = _min_377;
                        float _exp_174 = expf(gate_x_728 * -1.0f);
                        float denominator_x_732 = _exp_174 + 1.0f;
                        float _exp_175 = expf(gate_y_729 * -1.0f);
                        float denominator_y_733 = _exp_175 + 1.0f;
                        float hidden_x_734 = gate_x_728 / denominator_x_732 * up_x_730;
                        float hidden_y_735 = gate_y_729 / denominator_y_733 * up_y_731;
                        __nv_bfloat162 _bf16x2_287 = __float22bfloat162_rn(make_float2(hidden_x_734, hidden_y_735));
                        hidden_packed_539[23] = __as_u32(_bf16x2_287);
                        float2 _cvt_f32_176 = __bfloat1622float2(__as_bf16x2(gate_packed_537[24]));
                        float2 _cvt_f32_177 = __bfloat1622float2(__as_bf16x2(up_packed_538[24]));
                        float gate_x_736 = _cvt_f32_176.x;
                        float gate_y_737 = _cvt_f32_176.y;
                        float up_x_738 = _cvt_f32_177.x;
                        float up_y_739 = _cvt_f32_177.y;
                        float _min_378 = fminf(gate_x_736, swiglu_limit);
                        gate_x_736 = _min_378;
                        float _min_379 = fminf(gate_y_737, swiglu_limit);
                        gate_y_737 = _min_379;
                        float _fmax_192 = fmaxf(up_x_738, -swiglu_limit);
                        float _min_380 = fminf(_fmax_192, swiglu_limit);
                        up_x_738 = _min_380;
                        float _fmax_193 = fmaxf(up_y_739, -swiglu_limit);
                        float _min_381 = fminf(_fmax_193, swiglu_limit);
                        up_y_739 = _min_381;
                        float _exp_176 = expf(gate_x_736 * -1.0f);
                        float denominator_x_740 = _exp_176 + 1.0f;
                        float _exp_177 = expf(gate_y_737 * -1.0f);
                        float denominator_y_741 = _exp_177 + 1.0f;
                        float hidden_x_742 = gate_x_736 / denominator_x_740 * up_x_738;
                        float hidden_y_743 = gate_y_737 / denominator_y_741 * up_y_739;
                        __nv_bfloat162 _bf16x2_288 = __float22bfloat162_rn(make_float2(hidden_x_742, hidden_y_743));
                        hidden_packed_539[24] = __as_u32(_bf16x2_288);
                        float2 _cvt_f32_178 = __bfloat1622float2(__as_bf16x2(gate_packed_537[25]));
                        float2 _cvt_f32_179 = __bfloat1622float2(__as_bf16x2(up_packed_538[25]));
                        float gate_x_744 = _cvt_f32_178.x;
                        float gate_y_745 = _cvt_f32_178.y;
                        float up_x_746 = _cvt_f32_179.x;
                        float up_y_747 = _cvt_f32_179.y;
                        float _min_382 = fminf(gate_x_744, swiglu_limit);
                        gate_x_744 = _min_382;
                        float _min_383 = fminf(gate_y_745, swiglu_limit);
                        gate_y_745 = _min_383;
                        float _fmax_194 = fmaxf(up_x_746, -swiglu_limit);
                        float _min_384 = fminf(_fmax_194, swiglu_limit);
                        up_x_746 = _min_384;
                        float _fmax_195 = fmaxf(up_y_747, -swiglu_limit);
                        float _min_385 = fminf(_fmax_195, swiglu_limit);
                        up_y_747 = _min_385;
                        float _exp_178 = expf(gate_x_744 * -1.0f);
                        float denominator_x_748 = _exp_178 + 1.0f;
                        float _exp_179 = expf(gate_y_745 * -1.0f);
                        float denominator_y_749 = _exp_179 + 1.0f;
                        float hidden_x_750 = gate_x_744 / denominator_x_748 * up_x_746;
                        float hidden_y_751 = gate_y_745 / denominator_y_749 * up_y_747;
                        __nv_bfloat162 _bf16x2_289 = __float22bfloat162_rn(make_float2(hidden_x_750, hidden_y_751));
                        hidden_packed_539[25] = __as_u32(_bf16x2_289);
                        float2 _cvt_f32_180 = __bfloat1622float2(__as_bf16x2(gate_packed_537[26]));
                        float2 _cvt_f32_181 = __bfloat1622float2(__as_bf16x2(up_packed_538[26]));
                        float gate_x_752 = _cvt_f32_180.x;
                        float gate_y_753 = _cvt_f32_180.y;
                        float up_x_754 = _cvt_f32_181.x;
                        float up_y_755 = _cvt_f32_181.y;
                        float _min_386 = fminf(gate_x_752, swiglu_limit);
                        gate_x_752 = _min_386;
                        float _min_387 = fminf(gate_y_753, swiglu_limit);
                        gate_y_753 = _min_387;
                        float _fmax_196 = fmaxf(up_x_754, -swiglu_limit);
                        float _min_388 = fminf(_fmax_196, swiglu_limit);
                        up_x_754 = _min_388;
                        float _fmax_197 = fmaxf(up_y_755, -swiglu_limit);
                        float _min_389 = fminf(_fmax_197, swiglu_limit);
                        up_y_755 = _min_389;
                        float _exp_180 = expf(gate_x_752 * -1.0f);
                        float denominator_x_756 = _exp_180 + 1.0f;
                        float _exp_181 = expf(gate_y_753 * -1.0f);
                        float denominator_y_757 = _exp_181 + 1.0f;
                        float hidden_x_758 = gate_x_752 / denominator_x_756 * up_x_754;
                        float hidden_y_759 = gate_y_753 / denominator_y_757 * up_y_755;
                        __nv_bfloat162 _bf16x2_290 = __float22bfloat162_rn(make_float2(hidden_x_758, hidden_y_759));
                        hidden_packed_539[26] = __as_u32(_bf16x2_290);
                        float2 _cvt_f32_182 = __bfloat1622float2(__as_bf16x2(gate_packed_537[27]));
                        float2 _cvt_f32_183 = __bfloat1622float2(__as_bf16x2(up_packed_538[27]));
                        float gate_x_760 = _cvt_f32_182.x;
                        float gate_y_761 = _cvt_f32_182.y;
                        float up_x_762 = _cvt_f32_183.x;
                        float up_y_763 = _cvt_f32_183.y;
                        float _min_390 = fminf(gate_x_760, swiglu_limit);
                        gate_x_760 = _min_390;
                        float _min_391 = fminf(gate_y_761, swiglu_limit);
                        gate_y_761 = _min_391;
                        float _fmax_198 = fmaxf(up_x_762, -swiglu_limit);
                        float _min_392 = fminf(_fmax_198, swiglu_limit);
                        up_x_762 = _min_392;
                        float _fmax_199 = fmaxf(up_y_763, -swiglu_limit);
                        float _min_393 = fminf(_fmax_199, swiglu_limit);
                        up_y_763 = _min_393;
                        float _exp_182 = expf(gate_x_760 * -1.0f);
                        float denominator_x_764 = _exp_182 + 1.0f;
                        float _exp_183 = expf(gate_y_761 * -1.0f);
                        float denominator_y_765 = _exp_183 + 1.0f;
                        float hidden_x_766 = gate_x_760 / denominator_x_764 * up_x_762;
                        float hidden_y_767 = gate_y_761 / denominator_y_765 * up_y_763;
                        __nv_bfloat162 _bf16x2_291 = __float22bfloat162_rn(make_float2(hidden_x_766, hidden_y_767));
                        hidden_packed_539[27] = __as_u32(_bf16x2_291);
                        float2 _cvt_f32_184 = __bfloat1622float2(__as_bf16x2(gate_packed_537[28]));
                        float2 _cvt_f32_185 = __bfloat1622float2(__as_bf16x2(up_packed_538[28]));
                        float gate_x_768 = _cvt_f32_184.x;
                        float gate_y_769 = _cvt_f32_184.y;
                        float up_x_770 = _cvt_f32_185.x;
                        float up_y_771 = _cvt_f32_185.y;
                        float _min_394 = fminf(gate_x_768, swiglu_limit);
                        gate_x_768 = _min_394;
                        float _min_395 = fminf(gate_y_769, swiglu_limit);
                        gate_y_769 = _min_395;
                        float _fmax_200 = fmaxf(up_x_770, -swiglu_limit);
                        float _min_396 = fminf(_fmax_200, swiglu_limit);
                        up_x_770 = _min_396;
                        float _fmax_201 = fmaxf(up_y_771, -swiglu_limit);
                        float _min_397 = fminf(_fmax_201, swiglu_limit);
                        up_y_771 = _min_397;
                        float _exp_184 = expf(gate_x_768 * -1.0f);
                        float denominator_x_772 = _exp_184 + 1.0f;
                        float _exp_185 = expf(gate_y_769 * -1.0f);
                        float denominator_y_773 = _exp_185 + 1.0f;
                        float hidden_x_774 = gate_x_768 / denominator_x_772 * up_x_770;
                        float hidden_y_775 = gate_y_769 / denominator_y_773 * up_y_771;
                        __nv_bfloat162 _bf16x2_292 = __float22bfloat162_rn(make_float2(hidden_x_774, hidden_y_775));
                        hidden_packed_539[28] = __as_u32(_bf16x2_292);
                        float2 _cvt_f32_186 = __bfloat1622float2(__as_bf16x2(gate_packed_537[29]));
                        float2 _cvt_f32_187 = __bfloat1622float2(__as_bf16x2(up_packed_538[29]));
                        float gate_x_776 = _cvt_f32_186.x;
                        float gate_y_777 = _cvt_f32_186.y;
                        float up_x_778 = _cvt_f32_187.x;
                        float up_y_779 = _cvt_f32_187.y;
                        float _min_398 = fminf(gate_x_776, swiglu_limit);
                        gate_x_776 = _min_398;
                        float _min_399 = fminf(gate_y_777, swiglu_limit);
                        gate_y_777 = _min_399;
                        float _fmax_202 = fmaxf(up_x_778, -swiglu_limit);
                        float _min_400 = fminf(_fmax_202, swiglu_limit);
                        up_x_778 = _min_400;
                        float _fmax_203 = fmaxf(up_y_779, -swiglu_limit);
                        float _min_401 = fminf(_fmax_203, swiglu_limit);
                        up_y_779 = _min_401;
                        float _exp_186 = expf(gate_x_776 * -1.0f);
                        float denominator_x_780 = _exp_186 + 1.0f;
                        float _exp_187 = expf(gate_y_777 * -1.0f);
                        float denominator_y_781 = _exp_187 + 1.0f;
                        float hidden_x_782 = gate_x_776 / denominator_x_780 * up_x_778;
                        float hidden_y_783 = gate_y_777 / denominator_y_781 * up_y_779;
                        __nv_bfloat162 _bf16x2_293 = __float22bfloat162_rn(make_float2(hidden_x_782, hidden_y_783));
                        hidden_packed_539[29] = __as_u32(_bf16x2_293);
                        float2 _cvt_f32_188 = __bfloat1622float2(__as_bf16x2(gate_packed_537[30]));
                        float2 _cvt_f32_189 = __bfloat1622float2(__as_bf16x2(up_packed_538[30]));
                        float gate_x_784 = _cvt_f32_188.x;
                        float gate_y_785 = _cvt_f32_188.y;
                        float up_x_786 = _cvt_f32_189.x;
                        float up_y_787 = _cvt_f32_189.y;
                        float _min_402 = fminf(gate_x_784, swiglu_limit);
                        gate_x_784 = _min_402;
                        float _min_403 = fminf(gate_y_785, swiglu_limit);
                        gate_y_785 = _min_403;
                        float _fmax_204 = fmaxf(up_x_786, -swiglu_limit);
                        float _min_404 = fminf(_fmax_204, swiglu_limit);
                        up_x_786 = _min_404;
                        float _fmax_205 = fmaxf(up_y_787, -swiglu_limit);
                        float _min_405 = fminf(_fmax_205, swiglu_limit);
                        up_y_787 = _min_405;
                        float _exp_188 = expf(gate_x_784 * -1.0f);
                        float denominator_x_788 = _exp_188 + 1.0f;
                        float _exp_189 = expf(gate_y_785 * -1.0f);
                        float denominator_y_789 = _exp_189 + 1.0f;
                        float hidden_x_790 = gate_x_784 / denominator_x_788 * up_x_786;
                        float hidden_y_791 = gate_y_785 / denominator_y_789 * up_y_787;
                        __nv_bfloat162 _bf16x2_294 = __float22bfloat162_rn(make_float2(hidden_x_790, hidden_y_791));
                        hidden_packed_539[30] = __as_u32(_bf16x2_294);
                        float2 _cvt_f32_190 = __bfloat1622float2(__as_bf16x2(gate_packed_537[31]));
                        float2 _cvt_f32_191 = __bfloat1622float2(__as_bf16x2(up_packed_538[31]));
                        float gate_x_792 = _cvt_f32_190.x;
                        float gate_y_793 = _cvt_f32_190.y;
                        float up_x_794 = _cvt_f32_191.x;
                        float up_y_795 = _cvt_f32_191.y;
                        float _min_406 = fminf(gate_x_792, swiglu_limit);
                        gate_x_792 = _min_406;
                        float _min_407 = fminf(gate_y_793, swiglu_limit);
                        gate_y_793 = _min_407;
                        float _fmax_206 = fmaxf(up_x_794, -swiglu_limit);
                        float _min_408 = fminf(_fmax_206, swiglu_limit);
                        up_x_794 = _min_408;
                        float _fmax_207 = fmaxf(up_y_795, -swiglu_limit);
                        float _min_409 = fminf(_fmax_207, swiglu_limit);
                        up_y_795 = _min_409;
                        float _exp_190 = expf(gate_x_792 * -1.0f);
                        float denominator_x_796 = _exp_190 + 1.0f;
                        float _exp_191 = expf(gate_y_793 * -1.0f);
                        float denominator_y_797 = _exp_191 + 1.0f;
                        float hidden_x_798 = gate_x_792 / denominator_x_796 * up_x_794;
                        float hidden_y_799 = gate_y_793 / denominator_y_797 * up_y_795;
                        __nv_bfloat162 _bf16x2_295 = __float22bfloat162_rn(make_float2(hidden_x_798, hidden_y_799));
                        hidden_packed_539[31] = __as_u32(_bf16x2_295);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_800 = tid / 32;
                        int lane_801 = tid % 32;
                        #pragma unroll
                        for (int half_12 = 0; half_12 < 2; half_12++) {
                            #pragma unroll
                            for (int col_tile_20 = 0; col_tile_20 < 2; col_tile_20++) {
                                int row_17 = warp_800 * 32 + half_12 * 16 + lane_801 % 16;
                                int col_45 = col_tile_20 * 16 + lane_801 / 16 * 8;
                                unsigned int address_3_12 = d_smem_addr + (unsigned int)((row_17 * 32 + col_45) * 2);
                                address_3_12 = address_3_12 ^ (address_3_12 & 511) >> 7 << 4;
                                int offset_12 = half_12 * 8 + col_tile_20 * 4;
                                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(address_3_12);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_802 = tid / 32;
                        int lane_803 = tid % 32;
                        #pragma unroll
                        for (int half_13 = 0; half_13 < 2; half_13++) {
                            #pragma unroll
                            for (int col_tile_21 = 0; col_tile_21 < 2; col_tile_21++) {
                                int row_18 = warp_802 * 32 + half_13 * 16 + lane_803 % 16;
                                int col_46 = col_tile_21 * 16 + lane_803 / 16 * 8;
                                unsigned int address_3_13 = d_smem_addr + 8192 + (unsigned int)((row_18 * 32 + col_46) * 2);
                                address_3_13 = address_3_13 ^ (address_3_13 & 511) >> 7 << 4;
                                int offset_13 = half_13 * 8 + col_tile_21 * 4;
                                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(address_3_13);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_804 = tid / 32;
                        int lane_805 = tid % 32;
                        #pragma unroll
                        for (int half_14 = 0; half_14 < 2; half_14++) {
                            #pragma unroll
                            for (int col_tile_22 = 0; col_tile_22 < 2; col_tile_22++) {
                                int row_19 = warp_804 * 32 + half_14 * 16 + lane_805 % 16;
                                int col_47 = col_tile_22 * 16 + lane_805 / 16 * 8;
                                unsigned int address_3_14 = d_smem_addr + 16384 + (unsigned int)((row_19 * 32 + col_47) * 2);
                                address_3_14 = address_3_14 ^ (address_3_14 & 511) >> 7 << 4;
                                int offset_14 = half_14 * 8 + col_tile_22 * 4;
                                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(address_3_14);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_806 = tid / 32;
                        int lane_807 = tid % 32;
                        #pragma unroll
                        for (int half_15 = 0; half_15 < 2; half_15++) {
                            #pragma unroll
                            for (int col_tile_23 = 0; col_tile_23 < 2; col_tile_23++) {
                                int row_20 = warp_806 * 32 + half_15 * 16 + lane_807 % 16;
                                int col_48 = col_tile_23 * 16 + lane_807 / 16 * 8;
                                unsigned int address_3_15 = d_smem_addr + (unsigned int)((row_20 * 32 + col_48) * 2);
                                address_3_15 = address_3_15 ^ (address_3_15 & 511) >> 7 << 4;
                                int offset_15 = 16 + half_15 * 8 + col_tile_23 * 4;
                                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(address_3_15);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_808 = tid / 32;
                        int lane_809 = tid % 32;
                        #pragma unroll
                        for (int half_16 = 0; half_16 < 2; half_16++) {
                            #pragma unroll
                            for (int col_tile_24 = 0; col_tile_24 < 2; col_tile_24++) {
                                int row_21 = warp_808 * 32 + half_16 * 16 + lane_809 % 16;
                                int col_49 = col_tile_24 * 16 + lane_809 / 16 * 8;
                                unsigned int address_3_16 = d_smem_addr + 8192 + (unsigned int)((row_21 * 32 + col_49) * 2);
                                address_3_16 = address_3_16 ^ (address_3_16 & 511) >> 7 << 4;
                                int offset_16 = 16 + half_16 * 8 + col_tile_24 * 4;
                                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(address_3_16);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_810 = tid / 32;
                        int lane_811 = tid % 32;
                        #pragma unroll
                        for (int half_17 = 0; half_17 < 2; half_17++) {
                            #pragma unroll
                            for (int col_tile_25 = 0; col_tile_25 < 2; col_tile_25++) {
                                int row_22 = warp_810 * 32 + half_17 * 16 + lane_811 % 16;
                                int col_50 = col_tile_25 * 16 + lane_811 / 16 * 8;
                                unsigned int address_3_17 = d_smem_addr + 16384 + (unsigned int)((row_22 * 32 + col_50) * 2);
                                address_3_17 = address_3_17 ^ (address_3_17 & 511) >> 7 << 4;
                                int offset_17 = 16 + half_17 * 8 + col_tile_25 * 4;
                                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(address_3_17);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        unsigned int gate_packed_812[32];
                        unsigned int up_packed_813[32];
                        unsigned int hidden_packed_814[32];
                        unsigned int address_815 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 192;
                        float _tmem_load_24[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[15]))
                            : "r"(address_815));
                        float _tmem_load_25[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[15]))
                            : "r"(address_815 + 256));
                        __nv_bfloat162 _bf16x2_296 = __float22bfloat162_rn(make_float2(_tmem_load_24[0], _tmem_load_24[1]));
                        gate_packed_812[0] = __as_u32(_bf16x2_296);
                        __nv_bfloat162 _bf16x2_297 = __float22bfloat162_rn(make_float2(_tmem_load_25[0], _tmem_load_25[1]));
                        up_packed_813[0] = __as_u32(_bf16x2_297);
                        __nv_bfloat162 _bf16x2_298 = __float22bfloat162_rn(make_float2(_tmem_load_24[2], _tmem_load_24[3]));
                        gate_packed_812[1] = __as_u32(_bf16x2_298);
                        __nv_bfloat162 _bf16x2_299 = __float22bfloat162_rn(make_float2(_tmem_load_25[2], _tmem_load_25[3]));
                        up_packed_813[1] = __as_u32(_bf16x2_299);
                        __nv_bfloat162 _bf16x2_300 = __float22bfloat162_rn(make_float2(_tmem_load_24[4], _tmem_load_24[5]));
                        gate_packed_812[2] = __as_u32(_bf16x2_300);
                        __nv_bfloat162 _bf16x2_301 = __float22bfloat162_rn(make_float2(_tmem_load_25[4], _tmem_load_25[5]));
                        up_packed_813[2] = __as_u32(_bf16x2_301);
                        __nv_bfloat162 _bf16x2_302 = __float22bfloat162_rn(make_float2(_tmem_load_24[6], _tmem_load_24[7]));
                        gate_packed_812[3] = __as_u32(_bf16x2_302);
                        __nv_bfloat162 _bf16x2_303 = __float22bfloat162_rn(make_float2(_tmem_load_25[6], _tmem_load_25[7]));
                        up_packed_813[3] = __as_u32(_bf16x2_303);
                        __nv_bfloat162 _bf16x2_304 = __float22bfloat162_rn(make_float2(_tmem_load_24[8], _tmem_load_24[9]));
                        gate_packed_812[4] = __as_u32(_bf16x2_304);
                        __nv_bfloat162 _bf16x2_305 = __float22bfloat162_rn(make_float2(_tmem_load_25[8], _tmem_load_25[9]));
                        up_packed_813[4] = __as_u32(_bf16x2_305);
                        __nv_bfloat162 _bf16x2_306 = __float22bfloat162_rn(make_float2(_tmem_load_24[10], _tmem_load_24[11]));
                        gate_packed_812[5] = __as_u32(_bf16x2_306);
                        __nv_bfloat162 _bf16x2_307 = __float22bfloat162_rn(make_float2(_tmem_load_25[10], _tmem_load_25[11]));
                        up_packed_813[5] = __as_u32(_bf16x2_307);
                        __nv_bfloat162 _bf16x2_308 = __float22bfloat162_rn(make_float2(_tmem_load_24[12], _tmem_load_24[13]));
                        gate_packed_812[6] = __as_u32(_bf16x2_308);
                        __nv_bfloat162 _bf16x2_309 = __float22bfloat162_rn(make_float2(_tmem_load_25[12], _tmem_load_25[13]));
                        up_packed_813[6] = __as_u32(_bf16x2_309);
                        __nv_bfloat162 _bf16x2_310 = __float22bfloat162_rn(make_float2(_tmem_load_24[14], _tmem_load_24[15]));
                        gate_packed_812[7] = __as_u32(_bf16x2_310);
                        __nv_bfloat162 _bf16x2_311 = __float22bfloat162_rn(make_float2(_tmem_load_25[14], _tmem_load_25[15]));
                        up_packed_813[7] = __as_u32(_bf16x2_311);
                        unsigned int address_816 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 192;
                        float _tmem_load_26[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[15]))
                            : "r"(address_816));
                        float _tmem_load_27[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[15]))
                            : "r"(address_816 + 256));
                        __nv_bfloat162 _bf16x2_312 = __float22bfloat162_rn(make_float2(_tmem_load_26[0], _tmem_load_26[1]));
                        gate_packed_812[8] = __as_u32(_bf16x2_312);
                        __nv_bfloat162 _bf16x2_313 = __float22bfloat162_rn(make_float2(_tmem_load_27[0], _tmem_load_27[1]));
                        up_packed_813[8] = __as_u32(_bf16x2_313);
                        __nv_bfloat162 _bf16x2_314 = __float22bfloat162_rn(make_float2(_tmem_load_26[2], _tmem_load_26[3]));
                        gate_packed_812[9] = __as_u32(_bf16x2_314);
                        __nv_bfloat162 _bf16x2_315 = __float22bfloat162_rn(make_float2(_tmem_load_27[2], _tmem_load_27[3]));
                        up_packed_813[9] = __as_u32(_bf16x2_315);
                        __nv_bfloat162 _bf16x2_316 = __float22bfloat162_rn(make_float2(_tmem_load_26[4], _tmem_load_26[5]));
                        gate_packed_812[10] = __as_u32(_bf16x2_316);
                        __nv_bfloat162 _bf16x2_317 = __float22bfloat162_rn(make_float2(_tmem_load_27[4], _tmem_load_27[5]));
                        up_packed_813[10] = __as_u32(_bf16x2_317);
                        __nv_bfloat162 _bf16x2_318 = __float22bfloat162_rn(make_float2(_tmem_load_26[6], _tmem_load_26[7]));
                        gate_packed_812[11] = __as_u32(_bf16x2_318);
                        __nv_bfloat162 _bf16x2_319 = __float22bfloat162_rn(make_float2(_tmem_load_27[6], _tmem_load_27[7]));
                        up_packed_813[11] = __as_u32(_bf16x2_319);
                        __nv_bfloat162 _bf16x2_320 = __float22bfloat162_rn(make_float2(_tmem_load_26[8], _tmem_load_26[9]));
                        gate_packed_812[12] = __as_u32(_bf16x2_320);
                        __nv_bfloat162 _bf16x2_321 = __float22bfloat162_rn(make_float2(_tmem_load_27[8], _tmem_load_27[9]));
                        up_packed_813[12] = __as_u32(_bf16x2_321);
                        __nv_bfloat162 _bf16x2_322 = __float22bfloat162_rn(make_float2(_tmem_load_26[10], _tmem_load_26[11]));
                        gate_packed_812[13] = __as_u32(_bf16x2_322);
                        __nv_bfloat162 _bf16x2_323 = __float22bfloat162_rn(make_float2(_tmem_load_27[10], _tmem_load_27[11]));
                        up_packed_813[13] = __as_u32(_bf16x2_323);
                        __nv_bfloat162 _bf16x2_324 = __float22bfloat162_rn(make_float2(_tmem_load_26[12], _tmem_load_26[13]));
                        gate_packed_812[14] = __as_u32(_bf16x2_324);
                        __nv_bfloat162 _bf16x2_325 = __float22bfloat162_rn(make_float2(_tmem_load_27[12], _tmem_load_27[13]));
                        up_packed_813[14] = __as_u32(_bf16x2_325);
                        __nv_bfloat162 _bf16x2_326 = __float22bfloat162_rn(make_float2(_tmem_load_26[14], _tmem_load_26[15]));
                        gate_packed_812[15] = __as_u32(_bf16x2_326);
                        __nv_bfloat162 _bf16x2_327 = __float22bfloat162_rn(make_float2(_tmem_load_27[14], _tmem_load_27[15]));
                        up_packed_813[15] = __as_u32(_bf16x2_327);
                        unsigned int address_817 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 224;
                        float _tmem_load_28[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[15]))
                            : "r"(address_817));
                        float _tmem_load_29[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[15]))
                            : "r"(address_817 + 256));
                        __nv_bfloat162 _bf16x2_328 = __float22bfloat162_rn(make_float2(_tmem_load_28[0], _tmem_load_28[1]));
                        gate_packed_812[16] = __as_u32(_bf16x2_328);
                        __nv_bfloat162 _bf16x2_329 = __float22bfloat162_rn(make_float2(_tmem_load_29[0], _tmem_load_29[1]));
                        up_packed_813[16] = __as_u32(_bf16x2_329);
                        __nv_bfloat162 _bf16x2_330 = __float22bfloat162_rn(make_float2(_tmem_load_28[2], _tmem_load_28[3]));
                        gate_packed_812[17] = __as_u32(_bf16x2_330);
                        __nv_bfloat162 _bf16x2_331 = __float22bfloat162_rn(make_float2(_tmem_load_29[2], _tmem_load_29[3]));
                        up_packed_813[17] = __as_u32(_bf16x2_331);
                        __nv_bfloat162 _bf16x2_332 = __float22bfloat162_rn(make_float2(_tmem_load_28[4], _tmem_load_28[5]));
                        gate_packed_812[18] = __as_u32(_bf16x2_332);
                        __nv_bfloat162 _bf16x2_333 = __float22bfloat162_rn(make_float2(_tmem_load_29[4], _tmem_load_29[5]));
                        up_packed_813[18] = __as_u32(_bf16x2_333);
                        __nv_bfloat162 _bf16x2_334 = __float22bfloat162_rn(make_float2(_tmem_load_28[6], _tmem_load_28[7]));
                        gate_packed_812[19] = __as_u32(_bf16x2_334);
                        __nv_bfloat162 _bf16x2_335 = __float22bfloat162_rn(make_float2(_tmem_load_29[6], _tmem_load_29[7]));
                        up_packed_813[19] = __as_u32(_bf16x2_335);
                        __nv_bfloat162 _bf16x2_336 = __float22bfloat162_rn(make_float2(_tmem_load_28[8], _tmem_load_28[9]));
                        gate_packed_812[20] = __as_u32(_bf16x2_336);
                        __nv_bfloat162 _bf16x2_337 = __float22bfloat162_rn(make_float2(_tmem_load_29[8], _tmem_load_29[9]));
                        up_packed_813[20] = __as_u32(_bf16x2_337);
                        __nv_bfloat162 _bf16x2_338 = __float22bfloat162_rn(make_float2(_tmem_load_28[10], _tmem_load_28[11]));
                        gate_packed_812[21] = __as_u32(_bf16x2_338);
                        __nv_bfloat162 _bf16x2_339 = __float22bfloat162_rn(make_float2(_tmem_load_29[10], _tmem_load_29[11]));
                        up_packed_813[21] = __as_u32(_bf16x2_339);
                        __nv_bfloat162 _bf16x2_340 = __float22bfloat162_rn(make_float2(_tmem_load_28[12], _tmem_load_28[13]));
                        gate_packed_812[22] = __as_u32(_bf16x2_340);
                        __nv_bfloat162 _bf16x2_341 = __float22bfloat162_rn(make_float2(_tmem_load_29[12], _tmem_load_29[13]));
                        up_packed_813[22] = __as_u32(_bf16x2_341);
                        __nv_bfloat162 _bf16x2_342 = __float22bfloat162_rn(make_float2(_tmem_load_28[14], _tmem_load_28[15]));
                        gate_packed_812[23] = __as_u32(_bf16x2_342);
                        __nv_bfloat162 _bf16x2_343 = __float22bfloat162_rn(make_float2(_tmem_load_29[14], _tmem_load_29[15]));
                        up_packed_813[23] = __as_u32(_bf16x2_343);
                        unsigned int address_818 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 224;
                        float _tmem_load_30[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[15]))
                            : "r"(address_818));
                        float _tmem_load_31[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[15]))
                            : "r"(address_818 + 256));
                        __nv_bfloat162 _bf16x2_344 = __float22bfloat162_rn(make_float2(_tmem_load_30[0], _tmem_load_30[1]));
                        gate_packed_812[24] = __as_u32(_bf16x2_344);
                        __nv_bfloat162 _bf16x2_345 = __float22bfloat162_rn(make_float2(_tmem_load_31[0], _tmem_load_31[1]));
                        up_packed_813[24] = __as_u32(_bf16x2_345);
                        __nv_bfloat162 _bf16x2_346 = __float22bfloat162_rn(make_float2(_tmem_load_30[2], _tmem_load_30[3]));
                        gate_packed_812[25] = __as_u32(_bf16x2_346);
                        __nv_bfloat162 _bf16x2_347 = __float22bfloat162_rn(make_float2(_tmem_load_31[2], _tmem_load_31[3]));
                        up_packed_813[25] = __as_u32(_bf16x2_347);
                        __nv_bfloat162 _bf16x2_348 = __float22bfloat162_rn(make_float2(_tmem_load_30[4], _tmem_load_30[5]));
                        gate_packed_812[26] = __as_u32(_bf16x2_348);
                        __nv_bfloat162 _bf16x2_349 = __float22bfloat162_rn(make_float2(_tmem_load_31[4], _tmem_load_31[5]));
                        up_packed_813[26] = __as_u32(_bf16x2_349);
                        __nv_bfloat162 _bf16x2_350 = __float22bfloat162_rn(make_float2(_tmem_load_30[6], _tmem_load_30[7]));
                        gate_packed_812[27] = __as_u32(_bf16x2_350);
                        __nv_bfloat162 _bf16x2_351 = __float22bfloat162_rn(make_float2(_tmem_load_31[6], _tmem_load_31[7]));
                        up_packed_813[27] = __as_u32(_bf16x2_351);
                        __nv_bfloat162 _bf16x2_352 = __float22bfloat162_rn(make_float2(_tmem_load_30[8], _tmem_load_30[9]));
                        gate_packed_812[28] = __as_u32(_bf16x2_352);
                        __nv_bfloat162 _bf16x2_353 = __float22bfloat162_rn(make_float2(_tmem_load_31[8], _tmem_load_31[9]));
                        up_packed_813[28] = __as_u32(_bf16x2_353);
                        __nv_bfloat162 _bf16x2_354 = __float22bfloat162_rn(make_float2(_tmem_load_30[10], _tmem_load_30[11]));
                        gate_packed_812[29] = __as_u32(_bf16x2_354);
                        __nv_bfloat162 _bf16x2_355 = __float22bfloat162_rn(make_float2(_tmem_load_31[10], _tmem_load_31[11]));
                        up_packed_813[29] = __as_u32(_bf16x2_355);
                        __nv_bfloat162 _bf16x2_356 = __float22bfloat162_rn(make_float2(_tmem_load_30[12], _tmem_load_30[13]));
                        gate_packed_812[30] = __as_u32(_bf16x2_356);
                        __nv_bfloat162 _bf16x2_357 = __float22bfloat162_rn(make_float2(_tmem_load_31[12], _tmem_load_31[13]));
                        up_packed_813[30] = __as_u32(_bf16x2_357);
                        __nv_bfloat162 _bf16x2_358 = __float22bfloat162_rn(make_float2(_tmem_load_30[14], _tmem_load_30[15]));
                        gate_packed_812[31] = __as_u32(_bf16x2_358);
                        __nv_bfloat162 _bf16x2_359 = __float22bfloat162_rn(make_float2(_tmem_load_31[14], _tmem_load_31[15]));
                        up_packed_813[31] = __as_u32(_bf16x2_359);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                        }
                        float2 _cvt_f32_192 = __bfloat1622float2(__as_bf16x2(gate_packed_812[0]));
                        float2 _cvt_f32_193 = __bfloat1622float2(__as_bf16x2(up_packed_813[0]));
                        float gate_x_819 = _cvt_f32_192.x;
                        float gate_y_820 = _cvt_f32_192.y;
                        float up_x_821 = _cvt_f32_193.x;
                        float up_y_822 = _cvt_f32_193.y;
                        float _min_410 = fminf(gate_x_819, swiglu_limit);
                        gate_x_819 = _min_410;
                        float _min_411 = fminf(gate_y_820, swiglu_limit);
                        gate_y_820 = _min_411;
                        float _fmax_208 = fmaxf(up_x_821, -swiglu_limit);
                        float _min_412 = fminf(_fmax_208, swiglu_limit);
                        up_x_821 = _min_412;
                        float _fmax_209 = fmaxf(up_y_822, -swiglu_limit);
                        float _min_413 = fminf(_fmax_209, swiglu_limit);
                        up_y_822 = _min_413;
                        float _exp_192 = expf(gate_x_819 * -1.0f);
                        float denominator_x_823 = _exp_192 + 1.0f;
                        float _exp_193 = expf(gate_y_820 * -1.0f);
                        float denominator_y_824 = _exp_193 + 1.0f;
                        float hidden_x_825 = gate_x_819 / denominator_x_823 * up_x_821;
                        float hidden_y_826 = gate_y_820 / denominator_y_824 * up_y_822;
                        __nv_bfloat162 _bf16x2_360 = __float22bfloat162_rn(make_float2(hidden_x_825, hidden_y_826));
                        hidden_packed_814[0] = __as_u32(_bf16x2_360);
                        float2 _cvt_f32_194 = __bfloat1622float2(__as_bf16x2(gate_packed_812[1]));
                        float2 _cvt_f32_195 = __bfloat1622float2(__as_bf16x2(up_packed_813[1]));
                        float gate_x_827 = _cvt_f32_194.x;
                        float gate_y_828 = _cvt_f32_194.y;
                        float up_x_829 = _cvt_f32_195.x;
                        float up_y_830 = _cvt_f32_195.y;
                        float _min_414 = fminf(gate_x_827, swiglu_limit);
                        gate_x_827 = _min_414;
                        float _min_415 = fminf(gate_y_828, swiglu_limit);
                        gate_y_828 = _min_415;
                        float _fmax_210 = fmaxf(up_x_829, -swiglu_limit);
                        float _min_416 = fminf(_fmax_210, swiglu_limit);
                        up_x_829 = _min_416;
                        float _fmax_211 = fmaxf(up_y_830, -swiglu_limit);
                        float _min_417 = fminf(_fmax_211, swiglu_limit);
                        up_y_830 = _min_417;
                        float _exp_194 = expf(gate_x_827 * -1.0f);
                        float denominator_x_831 = _exp_194 + 1.0f;
                        float _exp_195 = expf(gate_y_828 * -1.0f);
                        float denominator_y_832 = _exp_195 + 1.0f;
                        float hidden_x_833 = gate_x_827 / denominator_x_831 * up_x_829;
                        float hidden_y_834 = gate_y_828 / denominator_y_832 * up_y_830;
                        __nv_bfloat162 _bf16x2_361 = __float22bfloat162_rn(make_float2(hidden_x_833, hidden_y_834));
                        hidden_packed_814[1] = __as_u32(_bf16x2_361);
                        float2 _cvt_f32_196 = __bfloat1622float2(__as_bf16x2(gate_packed_812[2]));
                        float2 _cvt_f32_197 = __bfloat1622float2(__as_bf16x2(up_packed_813[2]));
                        float gate_x_835 = _cvt_f32_196.x;
                        float gate_y_836 = _cvt_f32_196.y;
                        float up_x_837 = _cvt_f32_197.x;
                        float up_y_838 = _cvt_f32_197.y;
                        float _min_418 = fminf(gate_x_835, swiglu_limit);
                        gate_x_835 = _min_418;
                        float _min_419 = fminf(gate_y_836, swiglu_limit);
                        gate_y_836 = _min_419;
                        float _fmax_212 = fmaxf(up_x_837, -swiglu_limit);
                        float _min_420 = fminf(_fmax_212, swiglu_limit);
                        up_x_837 = _min_420;
                        float _fmax_213 = fmaxf(up_y_838, -swiglu_limit);
                        float _min_421 = fminf(_fmax_213, swiglu_limit);
                        up_y_838 = _min_421;
                        float _exp_196 = expf(gate_x_835 * -1.0f);
                        float denominator_x_839 = _exp_196 + 1.0f;
                        float _exp_197 = expf(gate_y_836 * -1.0f);
                        float denominator_y_840 = _exp_197 + 1.0f;
                        float hidden_x_841 = gate_x_835 / denominator_x_839 * up_x_837;
                        float hidden_y_842 = gate_y_836 / denominator_y_840 * up_y_838;
                        __nv_bfloat162 _bf16x2_362 = __float22bfloat162_rn(make_float2(hidden_x_841, hidden_y_842));
                        hidden_packed_814[2] = __as_u32(_bf16x2_362);
                        float2 _cvt_f32_198 = __bfloat1622float2(__as_bf16x2(gate_packed_812[3]));
                        float2 _cvt_f32_199 = __bfloat1622float2(__as_bf16x2(up_packed_813[3]));
                        float gate_x_843 = _cvt_f32_198.x;
                        float gate_y_844 = _cvt_f32_198.y;
                        float up_x_845 = _cvt_f32_199.x;
                        float up_y_846 = _cvt_f32_199.y;
                        float _min_422 = fminf(gate_x_843, swiglu_limit);
                        gate_x_843 = _min_422;
                        float _min_423 = fminf(gate_y_844, swiglu_limit);
                        gate_y_844 = _min_423;
                        float _fmax_214 = fmaxf(up_x_845, -swiglu_limit);
                        float _min_424 = fminf(_fmax_214, swiglu_limit);
                        up_x_845 = _min_424;
                        float _fmax_215 = fmaxf(up_y_846, -swiglu_limit);
                        float _min_425 = fminf(_fmax_215, swiglu_limit);
                        up_y_846 = _min_425;
                        float _exp_198 = expf(gate_x_843 * -1.0f);
                        float denominator_x_847 = _exp_198 + 1.0f;
                        float _exp_199 = expf(gate_y_844 * -1.0f);
                        float denominator_y_848 = _exp_199 + 1.0f;
                        float hidden_x_849 = gate_x_843 / denominator_x_847 * up_x_845;
                        float hidden_y_850 = gate_y_844 / denominator_y_848 * up_y_846;
                        __nv_bfloat162 _bf16x2_363 = __float22bfloat162_rn(make_float2(hidden_x_849, hidden_y_850));
                        hidden_packed_814[3] = __as_u32(_bf16x2_363);
                        float2 _cvt_f32_200 = __bfloat1622float2(__as_bf16x2(gate_packed_812[4]));
                        float2 _cvt_f32_201 = __bfloat1622float2(__as_bf16x2(up_packed_813[4]));
                        float gate_x_851 = _cvt_f32_200.x;
                        float gate_y_852 = _cvt_f32_200.y;
                        float up_x_853 = _cvt_f32_201.x;
                        float up_y_854 = _cvt_f32_201.y;
                        float _min_426 = fminf(gate_x_851, swiglu_limit);
                        gate_x_851 = _min_426;
                        float _min_427 = fminf(gate_y_852, swiglu_limit);
                        gate_y_852 = _min_427;
                        float _fmax_216 = fmaxf(up_x_853, -swiglu_limit);
                        float _min_428 = fminf(_fmax_216, swiglu_limit);
                        up_x_853 = _min_428;
                        float _fmax_217 = fmaxf(up_y_854, -swiglu_limit);
                        float _min_429 = fminf(_fmax_217, swiglu_limit);
                        up_y_854 = _min_429;
                        float _exp_200 = expf(gate_x_851 * -1.0f);
                        float denominator_x_855 = _exp_200 + 1.0f;
                        float _exp_201 = expf(gate_y_852 * -1.0f);
                        float denominator_y_856 = _exp_201 + 1.0f;
                        float hidden_x_857 = gate_x_851 / denominator_x_855 * up_x_853;
                        float hidden_y_858 = gate_y_852 / denominator_y_856 * up_y_854;
                        __nv_bfloat162 _bf16x2_364 = __float22bfloat162_rn(make_float2(hidden_x_857, hidden_y_858));
                        hidden_packed_814[4] = __as_u32(_bf16x2_364);
                        float2 _cvt_f32_202 = __bfloat1622float2(__as_bf16x2(gate_packed_812[5]));
                        float2 _cvt_f32_203 = __bfloat1622float2(__as_bf16x2(up_packed_813[5]));
                        float gate_x_859 = _cvt_f32_202.x;
                        float gate_y_860 = _cvt_f32_202.y;
                        float up_x_861 = _cvt_f32_203.x;
                        float up_y_862 = _cvt_f32_203.y;
                        float _min_430 = fminf(gate_x_859, swiglu_limit);
                        gate_x_859 = _min_430;
                        float _min_431 = fminf(gate_y_860, swiglu_limit);
                        gate_y_860 = _min_431;
                        float _fmax_218 = fmaxf(up_x_861, -swiglu_limit);
                        float _min_432 = fminf(_fmax_218, swiglu_limit);
                        up_x_861 = _min_432;
                        float _fmax_219 = fmaxf(up_y_862, -swiglu_limit);
                        float _min_433 = fminf(_fmax_219, swiglu_limit);
                        up_y_862 = _min_433;
                        float _exp_202 = expf(gate_x_859 * -1.0f);
                        float denominator_x_863 = _exp_202 + 1.0f;
                        float _exp_203 = expf(gate_y_860 * -1.0f);
                        float denominator_y_864 = _exp_203 + 1.0f;
                        float hidden_x_865 = gate_x_859 / denominator_x_863 * up_x_861;
                        float hidden_y_866 = gate_y_860 / denominator_y_864 * up_y_862;
                        __nv_bfloat162 _bf16x2_365 = __float22bfloat162_rn(make_float2(hidden_x_865, hidden_y_866));
                        hidden_packed_814[5] = __as_u32(_bf16x2_365);
                        float2 _cvt_f32_204 = __bfloat1622float2(__as_bf16x2(gate_packed_812[6]));
                        float2 _cvt_f32_205 = __bfloat1622float2(__as_bf16x2(up_packed_813[6]));
                        float gate_x_867 = _cvt_f32_204.x;
                        float gate_y_868 = _cvt_f32_204.y;
                        float up_x_869 = _cvt_f32_205.x;
                        float up_y_870 = _cvt_f32_205.y;
                        float _min_434 = fminf(gate_x_867, swiglu_limit);
                        gate_x_867 = _min_434;
                        float _min_435 = fminf(gate_y_868, swiglu_limit);
                        gate_y_868 = _min_435;
                        float _fmax_220 = fmaxf(up_x_869, -swiglu_limit);
                        float _min_436 = fminf(_fmax_220, swiglu_limit);
                        up_x_869 = _min_436;
                        float _fmax_221 = fmaxf(up_y_870, -swiglu_limit);
                        float _min_437 = fminf(_fmax_221, swiglu_limit);
                        up_y_870 = _min_437;
                        float _exp_204 = expf(gate_x_867 * -1.0f);
                        float denominator_x_871 = _exp_204 + 1.0f;
                        float _exp_205 = expf(gate_y_868 * -1.0f);
                        float denominator_y_872 = _exp_205 + 1.0f;
                        float hidden_x_873 = gate_x_867 / denominator_x_871 * up_x_869;
                        float hidden_y_874 = gate_y_868 / denominator_y_872 * up_y_870;
                        __nv_bfloat162 _bf16x2_366 = __float22bfloat162_rn(make_float2(hidden_x_873, hidden_y_874));
                        hidden_packed_814[6] = __as_u32(_bf16x2_366);
                        float2 _cvt_f32_206 = __bfloat1622float2(__as_bf16x2(gate_packed_812[7]));
                        float2 _cvt_f32_207 = __bfloat1622float2(__as_bf16x2(up_packed_813[7]));
                        float gate_x_875 = _cvt_f32_206.x;
                        float gate_y_876 = _cvt_f32_206.y;
                        float up_x_877 = _cvt_f32_207.x;
                        float up_y_878 = _cvt_f32_207.y;
                        float _min_438 = fminf(gate_x_875, swiglu_limit);
                        gate_x_875 = _min_438;
                        float _min_439 = fminf(gate_y_876, swiglu_limit);
                        gate_y_876 = _min_439;
                        float _fmax_222 = fmaxf(up_x_877, -swiglu_limit);
                        float _min_440 = fminf(_fmax_222, swiglu_limit);
                        up_x_877 = _min_440;
                        float _fmax_223 = fmaxf(up_y_878, -swiglu_limit);
                        float _min_441 = fminf(_fmax_223, swiglu_limit);
                        up_y_878 = _min_441;
                        float _exp_206 = expf(gate_x_875 * -1.0f);
                        float denominator_x_879 = _exp_206 + 1.0f;
                        float _exp_207 = expf(gate_y_876 * -1.0f);
                        float denominator_y_880 = _exp_207 + 1.0f;
                        float hidden_x_881 = gate_x_875 / denominator_x_879 * up_x_877;
                        float hidden_y_882 = gate_y_876 / denominator_y_880 * up_y_878;
                        __nv_bfloat162 _bf16x2_367 = __float22bfloat162_rn(make_float2(hidden_x_881, hidden_y_882));
                        hidden_packed_814[7] = __as_u32(_bf16x2_367);
                        float2 _cvt_f32_208 = __bfloat1622float2(__as_bf16x2(gate_packed_812[8]));
                        float2 _cvt_f32_209 = __bfloat1622float2(__as_bf16x2(up_packed_813[8]));
                        float gate_x_883 = _cvt_f32_208.x;
                        float gate_y_884 = _cvt_f32_208.y;
                        float up_x_885 = _cvt_f32_209.x;
                        float up_y_886 = _cvt_f32_209.y;
                        float _min_442 = fminf(gate_x_883, swiglu_limit);
                        gate_x_883 = _min_442;
                        float _min_443 = fminf(gate_y_884, swiglu_limit);
                        gate_y_884 = _min_443;
                        float _fmax_224 = fmaxf(up_x_885, -swiglu_limit);
                        float _min_444 = fminf(_fmax_224, swiglu_limit);
                        up_x_885 = _min_444;
                        float _fmax_225 = fmaxf(up_y_886, -swiglu_limit);
                        float _min_445 = fminf(_fmax_225, swiglu_limit);
                        up_y_886 = _min_445;
                        float _exp_208 = expf(gate_x_883 * -1.0f);
                        float denominator_x_887 = _exp_208 + 1.0f;
                        float _exp_209 = expf(gate_y_884 * -1.0f);
                        float denominator_y_888 = _exp_209 + 1.0f;
                        float hidden_x_889 = gate_x_883 / denominator_x_887 * up_x_885;
                        float hidden_y_890 = gate_y_884 / denominator_y_888 * up_y_886;
                        __nv_bfloat162 _bf16x2_368 = __float22bfloat162_rn(make_float2(hidden_x_889, hidden_y_890));
                        hidden_packed_814[8] = __as_u32(_bf16x2_368);
                        float2 _cvt_f32_210 = __bfloat1622float2(__as_bf16x2(gate_packed_812[9]));
                        float2 _cvt_f32_211 = __bfloat1622float2(__as_bf16x2(up_packed_813[9]));
                        float gate_x_891 = _cvt_f32_210.x;
                        float gate_y_892 = _cvt_f32_210.y;
                        float up_x_893 = _cvt_f32_211.x;
                        float up_y_894 = _cvt_f32_211.y;
                        float _min_446 = fminf(gate_x_891, swiglu_limit);
                        gate_x_891 = _min_446;
                        float _min_447 = fminf(gate_y_892, swiglu_limit);
                        gate_y_892 = _min_447;
                        float _fmax_226 = fmaxf(up_x_893, -swiglu_limit);
                        float _min_448 = fminf(_fmax_226, swiglu_limit);
                        up_x_893 = _min_448;
                        float _fmax_227 = fmaxf(up_y_894, -swiglu_limit);
                        float _min_449 = fminf(_fmax_227, swiglu_limit);
                        up_y_894 = _min_449;
                        float _exp_210 = expf(gate_x_891 * -1.0f);
                        float denominator_x_895 = _exp_210 + 1.0f;
                        float _exp_211 = expf(gate_y_892 * -1.0f);
                        float denominator_y_896 = _exp_211 + 1.0f;
                        float hidden_x_897 = gate_x_891 / denominator_x_895 * up_x_893;
                        float hidden_y_898 = gate_y_892 / denominator_y_896 * up_y_894;
                        __nv_bfloat162 _bf16x2_369 = __float22bfloat162_rn(make_float2(hidden_x_897, hidden_y_898));
                        hidden_packed_814[9] = __as_u32(_bf16x2_369);
                        float2 _cvt_f32_212 = __bfloat1622float2(__as_bf16x2(gate_packed_812[10]));
                        float2 _cvt_f32_213 = __bfloat1622float2(__as_bf16x2(up_packed_813[10]));
                        float gate_x_899 = _cvt_f32_212.x;
                        float gate_y_900 = _cvt_f32_212.y;
                        float up_x_901 = _cvt_f32_213.x;
                        float up_y_902 = _cvt_f32_213.y;
                        float _min_450 = fminf(gate_x_899, swiglu_limit);
                        gate_x_899 = _min_450;
                        float _min_451 = fminf(gate_y_900, swiglu_limit);
                        gate_y_900 = _min_451;
                        float _fmax_228 = fmaxf(up_x_901, -swiglu_limit);
                        float _min_452 = fminf(_fmax_228, swiglu_limit);
                        up_x_901 = _min_452;
                        float _fmax_229 = fmaxf(up_y_902, -swiglu_limit);
                        float _min_453 = fminf(_fmax_229, swiglu_limit);
                        up_y_902 = _min_453;
                        float _exp_212 = expf(gate_x_899 * -1.0f);
                        float denominator_x_903 = _exp_212 + 1.0f;
                        float _exp_213 = expf(gate_y_900 * -1.0f);
                        float denominator_y_904 = _exp_213 + 1.0f;
                        float hidden_x_905 = gate_x_899 / denominator_x_903 * up_x_901;
                        float hidden_y_906 = gate_y_900 / denominator_y_904 * up_y_902;
                        __nv_bfloat162 _bf16x2_370 = __float22bfloat162_rn(make_float2(hidden_x_905, hidden_y_906));
                        hidden_packed_814[10] = __as_u32(_bf16x2_370);
                        float2 _cvt_f32_214 = __bfloat1622float2(__as_bf16x2(gate_packed_812[11]));
                        float2 _cvt_f32_215 = __bfloat1622float2(__as_bf16x2(up_packed_813[11]));
                        float gate_x_907 = _cvt_f32_214.x;
                        float gate_y_908 = _cvt_f32_214.y;
                        float up_x_909 = _cvt_f32_215.x;
                        float up_y_910 = _cvt_f32_215.y;
                        float _min_454 = fminf(gate_x_907, swiglu_limit);
                        gate_x_907 = _min_454;
                        float _min_455 = fminf(gate_y_908, swiglu_limit);
                        gate_y_908 = _min_455;
                        float _fmax_230 = fmaxf(up_x_909, -swiglu_limit);
                        float _min_456 = fminf(_fmax_230, swiglu_limit);
                        up_x_909 = _min_456;
                        float _fmax_231 = fmaxf(up_y_910, -swiglu_limit);
                        float _min_457 = fminf(_fmax_231, swiglu_limit);
                        up_y_910 = _min_457;
                        float _exp_214 = expf(gate_x_907 * -1.0f);
                        float denominator_x_911 = _exp_214 + 1.0f;
                        float _exp_215 = expf(gate_y_908 * -1.0f);
                        float denominator_y_912 = _exp_215 + 1.0f;
                        float hidden_x_913 = gate_x_907 / denominator_x_911 * up_x_909;
                        float hidden_y_914 = gate_y_908 / denominator_y_912 * up_y_910;
                        __nv_bfloat162 _bf16x2_371 = __float22bfloat162_rn(make_float2(hidden_x_913, hidden_y_914));
                        hidden_packed_814[11] = __as_u32(_bf16x2_371);
                        float2 _cvt_f32_216 = __bfloat1622float2(__as_bf16x2(gate_packed_812[12]));
                        float2 _cvt_f32_217 = __bfloat1622float2(__as_bf16x2(up_packed_813[12]));
                        float gate_x_915 = _cvt_f32_216.x;
                        float gate_y_916 = _cvt_f32_216.y;
                        float up_x_917 = _cvt_f32_217.x;
                        float up_y_918 = _cvt_f32_217.y;
                        float _min_458 = fminf(gate_x_915, swiglu_limit);
                        gate_x_915 = _min_458;
                        float _min_459 = fminf(gate_y_916, swiglu_limit);
                        gate_y_916 = _min_459;
                        float _fmax_232 = fmaxf(up_x_917, -swiglu_limit);
                        float _min_460 = fminf(_fmax_232, swiglu_limit);
                        up_x_917 = _min_460;
                        float _fmax_233 = fmaxf(up_y_918, -swiglu_limit);
                        float _min_461 = fminf(_fmax_233, swiglu_limit);
                        up_y_918 = _min_461;
                        float _exp_216 = expf(gate_x_915 * -1.0f);
                        float denominator_x_919 = _exp_216 + 1.0f;
                        float _exp_217 = expf(gate_y_916 * -1.0f);
                        float denominator_y_920 = _exp_217 + 1.0f;
                        float hidden_x_921 = gate_x_915 / denominator_x_919 * up_x_917;
                        float hidden_y_922 = gate_y_916 / denominator_y_920 * up_y_918;
                        __nv_bfloat162 _bf16x2_372 = __float22bfloat162_rn(make_float2(hidden_x_921, hidden_y_922));
                        hidden_packed_814[12] = __as_u32(_bf16x2_372);
                        float2 _cvt_f32_218 = __bfloat1622float2(__as_bf16x2(gate_packed_812[13]));
                        float2 _cvt_f32_219 = __bfloat1622float2(__as_bf16x2(up_packed_813[13]));
                        float gate_x_923 = _cvt_f32_218.x;
                        float gate_y_924 = _cvt_f32_218.y;
                        float up_x_925 = _cvt_f32_219.x;
                        float up_y_926 = _cvt_f32_219.y;
                        float _min_462 = fminf(gate_x_923, swiglu_limit);
                        gate_x_923 = _min_462;
                        float _min_463 = fminf(gate_y_924, swiglu_limit);
                        gate_y_924 = _min_463;
                        float _fmax_234 = fmaxf(up_x_925, -swiglu_limit);
                        float _min_464 = fminf(_fmax_234, swiglu_limit);
                        up_x_925 = _min_464;
                        float _fmax_235 = fmaxf(up_y_926, -swiglu_limit);
                        float _min_465 = fminf(_fmax_235, swiglu_limit);
                        up_y_926 = _min_465;
                        float _exp_218 = expf(gate_x_923 * -1.0f);
                        float denominator_x_927 = _exp_218 + 1.0f;
                        float _exp_219 = expf(gate_y_924 * -1.0f);
                        float denominator_y_928 = _exp_219 + 1.0f;
                        float hidden_x_929 = gate_x_923 / denominator_x_927 * up_x_925;
                        float hidden_y_930 = gate_y_924 / denominator_y_928 * up_y_926;
                        __nv_bfloat162 _bf16x2_373 = __float22bfloat162_rn(make_float2(hidden_x_929, hidden_y_930));
                        hidden_packed_814[13] = __as_u32(_bf16x2_373);
                        float2 _cvt_f32_220 = __bfloat1622float2(__as_bf16x2(gate_packed_812[14]));
                        float2 _cvt_f32_221 = __bfloat1622float2(__as_bf16x2(up_packed_813[14]));
                        float gate_x_931 = _cvt_f32_220.x;
                        float gate_y_932 = _cvt_f32_220.y;
                        float up_x_933 = _cvt_f32_221.x;
                        float up_y_934 = _cvt_f32_221.y;
                        float _min_466 = fminf(gate_x_931, swiglu_limit);
                        gate_x_931 = _min_466;
                        float _min_467 = fminf(gate_y_932, swiglu_limit);
                        gate_y_932 = _min_467;
                        float _fmax_236 = fmaxf(up_x_933, -swiglu_limit);
                        float _min_468 = fminf(_fmax_236, swiglu_limit);
                        up_x_933 = _min_468;
                        float _fmax_237 = fmaxf(up_y_934, -swiglu_limit);
                        float _min_469 = fminf(_fmax_237, swiglu_limit);
                        up_y_934 = _min_469;
                        float _exp_220 = expf(gate_x_931 * -1.0f);
                        float denominator_x_935 = _exp_220 + 1.0f;
                        float _exp_221 = expf(gate_y_932 * -1.0f);
                        float denominator_y_936 = _exp_221 + 1.0f;
                        float hidden_x_937 = gate_x_931 / denominator_x_935 * up_x_933;
                        float hidden_y_938 = gate_y_932 / denominator_y_936 * up_y_934;
                        __nv_bfloat162 _bf16x2_374 = __float22bfloat162_rn(make_float2(hidden_x_937, hidden_y_938));
                        hidden_packed_814[14] = __as_u32(_bf16x2_374);
                        float2 _cvt_f32_222 = __bfloat1622float2(__as_bf16x2(gate_packed_812[15]));
                        float2 _cvt_f32_223 = __bfloat1622float2(__as_bf16x2(up_packed_813[15]));
                        float gate_x_939 = _cvt_f32_222.x;
                        float gate_y_940 = _cvt_f32_222.y;
                        float up_x_941 = _cvt_f32_223.x;
                        float up_y_942 = _cvt_f32_223.y;
                        float _min_470 = fminf(gate_x_939, swiglu_limit);
                        gate_x_939 = _min_470;
                        float _min_471 = fminf(gate_y_940, swiglu_limit);
                        gate_y_940 = _min_471;
                        float _fmax_238 = fmaxf(up_x_941, -swiglu_limit);
                        float _min_472 = fminf(_fmax_238, swiglu_limit);
                        up_x_941 = _min_472;
                        float _fmax_239 = fmaxf(up_y_942, -swiglu_limit);
                        float _min_473 = fminf(_fmax_239, swiglu_limit);
                        up_y_942 = _min_473;
                        float _exp_222 = expf(gate_x_939 * -1.0f);
                        float denominator_x_943 = _exp_222 + 1.0f;
                        float _exp_223 = expf(gate_y_940 * -1.0f);
                        float denominator_y_944 = _exp_223 + 1.0f;
                        float hidden_x_945 = gate_x_939 / denominator_x_943 * up_x_941;
                        float hidden_y_946 = gate_y_940 / denominator_y_944 * up_y_942;
                        __nv_bfloat162 _bf16x2_375 = __float22bfloat162_rn(make_float2(hidden_x_945, hidden_y_946));
                        hidden_packed_814[15] = __as_u32(_bf16x2_375);
                        float2 _cvt_f32_224 = __bfloat1622float2(__as_bf16x2(gate_packed_812[16]));
                        float2 _cvt_f32_225 = __bfloat1622float2(__as_bf16x2(up_packed_813[16]));
                        float gate_x_947 = _cvt_f32_224.x;
                        float gate_y_948 = _cvt_f32_224.y;
                        float up_x_949 = _cvt_f32_225.x;
                        float up_y_950 = _cvt_f32_225.y;
                        float _min_474 = fminf(gate_x_947, swiglu_limit);
                        gate_x_947 = _min_474;
                        float _min_475 = fminf(gate_y_948, swiglu_limit);
                        gate_y_948 = _min_475;
                        float _fmax_240 = fmaxf(up_x_949, -swiglu_limit);
                        float _min_476 = fminf(_fmax_240, swiglu_limit);
                        up_x_949 = _min_476;
                        float _fmax_241 = fmaxf(up_y_950, -swiglu_limit);
                        float _min_477 = fminf(_fmax_241, swiglu_limit);
                        up_y_950 = _min_477;
                        float _exp_224 = expf(gate_x_947 * -1.0f);
                        float denominator_x_951 = _exp_224 + 1.0f;
                        float _exp_225 = expf(gate_y_948 * -1.0f);
                        float denominator_y_952 = _exp_225 + 1.0f;
                        float hidden_x_953 = gate_x_947 / denominator_x_951 * up_x_949;
                        float hidden_y_954 = gate_y_948 / denominator_y_952 * up_y_950;
                        __nv_bfloat162 _bf16x2_376 = __float22bfloat162_rn(make_float2(hidden_x_953, hidden_y_954));
                        hidden_packed_814[16] = __as_u32(_bf16x2_376);
                        float2 _cvt_f32_226 = __bfloat1622float2(__as_bf16x2(gate_packed_812[17]));
                        float2 _cvt_f32_227 = __bfloat1622float2(__as_bf16x2(up_packed_813[17]));
                        float gate_x_955 = _cvt_f32_226.x;
                        float gate_y_956 = _cvt_f32_226.y;
                        float up_x_957 = _cvt_f32_227.x;
                        float up_y_958 = _cvt_f32_227.y;
                        float _min_478 = fminf(gate_x_955, swiglu_limit);
                        gate_x_955 = _min_478;
                        float _min_479 = fminf(gate_y_956, swiglu_limit);
                        gate_y_956 = _min_479;
                        float _fmax_242 = fmaxf(up_x_957, -swiglu_limit);
                        float _min_480 = fminf(_fmax_242, swiglu_limit);
                        up_x_957 = _min_480;
                        float _fmax_243 = fmaxf(up_y_958, -swiglu_limit);
                        float _min_481 = fminf(_fmax_243, swiglu_limit);
                        up_y_958 = _min_481;
                        float _exp_226 = expf(gate_x_955 * -1.0f);
                        float denominator_x_959 = _exp_226 + 1.0f;
                        float _exp_227 = expf(gate_y_956 * -1.0f);
                        float denominator_y_960 = _exp_227 + 1.0f;
                        float hidden_x_961 = gate_x_955 / denominator_x_959 * up_x_957;
                        float hidden_y_962 = gate_y_956 / denominator_y_960 * up_y_958;
                        __nv_bfloat162 _bf16x2_377 = __float22bfloat162_rn(make_float2(hidden_x_961, hidden_y_962));
                        hidden_packed_814[17] = __as_u32(_bf16x2_377);
                        float2 _cvt_f32_228 = __bfloat1622float2(__as_bf16x2(gate_packed_812[18]));
                        float2 _cvt_f32_229 = __bfloat1622float2(__as_bf16x2(up_packed_813[18]));
                        float gate_x_963 = _cvt_f32_228.x;
                        float gate_y_964 = _cvt_f32_228.y;
                        float up_x_965 = _cvt_f32_229.x;
                        float up_y_966 = _cvt_f32_229.y;
                        float _min_482 = fminf(gate_x_963, swiglu_limit);
                        gate_x_963 = _min_482;
                        float _min_483 = fminf(gate_y_964, swiglu_limit);
                        gate_y_964 = _min_483;
                        float _fmax_244 = fmaxf(up_x_965, -swiglu_limit);
                        float _min_484 = fminf(_fmax_244, swiglu_limit);
                        up_x_965 = _min_484;
                        float _fmax_245 = fmaxf(up_y_966, -swiglu_limit);
                        float _min_485 = fminf(_fmax_245, swiglu_limit);
                        up_y_966 = _min_485;
                        float _exp_228 = expf(gate_x_963 * -1.0f);
                        float denominator_x_967 = _exp_228 + 1.0f;
                        float _exp_229 = expf(gate_y_964 * -1.0f);
                        float denominator_y_968 = _exp_229 + 1.0f;
                        float hidden_x_969 = gate_x_963 / denominator_x_967 * up_x_965;
                        float hidden_y_970 = gate_y_964 / denominator_y_968 * up_y_966;
                        __nv_bfloat162 _bf16x2_378 = __float22bfloat162_rn(make_float2(hidden_x_969, hidden_y_970));
                        hidden_packed_814[18] = __as_u32(_bf16x2_378);
                        float2 _cvt_f32_230 = __bfloat1622float2(__as_bf16x2(gate_packed_812[19]));
                        float2 _cvt_f32_231 = __bfloat1622float2(__as_bf16x2(up_packed_813[19]));
                        float gate_x_971 = _cvt_f32_230.x;
                        float gate_y_972 = _cvt_f32_230.y;
                        float up_x_973 = _cvt_f32_231.x;
                        float up_y_974 = _cvt_f32_231.y;
                        float _min_486 = fminf(gate_x_971, swiglu_limit);
                        gate_x_971 = _min_486;
                        float _min_487 = fminf(gate_y_972, swiglu_limit);
                        gate_y_972 = _min_487;
                        float _fmax_246 = fmaxf(up_x_973, -swiglu_limit);
                        float _min_488 = fminf(_fmax_246, swiglu_limit);
                        up_x_973 = _min_488;
                        float _fmax_247 = fmaxf(up_y_974, -swiglu_limit);
                        float _min_489 = fminf(_fmax_247, swiglu_limit);
                        up_y_974 = _min_489;
                        float _exp_230 = expf(gate_x_971 * -1.0f);
                        float denominator_x_975 = _exp_230 + 1.0f;
                        float _exp_231 = expf(gate_y_972 * -1.0f);
                        float denominator_y_976 = _exp_231 + 1.0f;
                        float hidden_x_977 = gate_x_971 / denominator_x_975 * up_x_973;
                        float hidden_y_978 = gate_y_972 / denominator_y_976 * up_y_974;
                        __nv_bfloat162 _bf16x2_379 = __float22bfloat162_rn(make_float2(hidden_x_977, hidden_y_978));
                        hidden_packed_814[19] = __as_u32(_bf16x2_379);
                        float2 _cvt_f32_232 = __bfloat1622float2(__as_bf16x2(gate_packed_812[20]));
                        float2 _cvt_f32_233 = __bfloat1622float2(__as_bf16x2(up_packed_813[20]));
                        float gate_x_979 = _cvt_f32_232.x;
                        float gate_y_980 = _cvt_f32_232.y;
                        float up_x_981 = _cvt_f32_233.x;
                        float up_y_982 = _cvt_f32_233.y;
                        float _min_490 = fminf(gate_x_979, swiglu_limit);
                        gate_x_979 = _min_490;
                        float _min_491 = fminf(gate_y_980, swiglu_limit);
                        gate_y_980 = _min_491;
                        float _fmax_248 = fmaxf(up_x_981, -swiglu_limit);
                        float _min_492 = fminf(_fmax_248, swiglu_limit);
                        up_x_981 = _min_492;
                        float _fmax_249 = fmaxf(up_y_982, -swiglu_limit);
                        float _min_493 = fminf(_fmax_249, swiglu_limit);
                        up_y_982 = _min_493;
                        float _exp_232 = expf(gate_x_979 * -1.0f);
                        float denominator_x_983 = _exp_232 + 1.0f;
                        float _exp_233 = expf(gate_y_980 * -1.0f);
                        float denominator_y_984 = _exp_233 + 1.0f;
                        float hidden_x_985 = gate_x_979 / denominator_x_983 * up_x_981;
                        float hidden_y_986 = gate_y_980 / denominator_y_984 * up_y_982;
                        __nv_bfloat162 _bf16x2_380 = __float22bfloat162_rn(make_float2(hidden_x_985, hidden_y_986));
                        hidden_packed_814[20] = __as_u32(_bf16x2_380);
                        float2 _cvt_f32_234 = __bfloat1622float2(__as_bf16x2(gate_packed_812[21]));
                        float2 _cvt_f32_235 = __bfloat1622float2(__as_bf16x2(up_packed_813[21]));
                        float gate_x_987 = _cvt_f32_234.x;
                        float gate_y_988 = _cvt_f32_234.y;
                        float up_x_989 = _cvt_f32_235.x;
                        float up_y_990 = _cvt_f32_235.y;
                        float _min_494 = fminf(gate_x_987, swiglu_limit);
                        gate_x_987 = _min_494;
                        float _min_495 = fminf(gate_y_988, swiglu_limit);
                        gate_y_988 = _min_495;
                        float _fmax_250 = fmaxf(up_x_989, -swiglu_limit);
                        float _min_496 = fminf(_fmax_250, swiglu_limit);
                        up_x_989 = _min_496;
                        float _fmax_251 = fmaxf(up_y_990, -swiglu_limit);
                        float _min_497 = fminf(_fmax_251, swiglu_limit);
                        up_y_990 = _min_497;
                        float _exp_234 = expf(gate_x_987 * -1.0f);
                        float denominator_x_991 = _exp_234 + 1.0f;
                        float _exp_235 = expf(gate_y_988 * -1.0f);
                        float denominator_y_992 = _exp_235 + 1.0f;
                        float hidden_x_993 = gate_x_987 / denominator_x_991 * up_x_989;
                        float hidden_y_994 = gate_y_988 / denominator_y_992 * up_y_990;
                        __nv_bfloat162 _bf16x2_381 = __float22bfloat162_rn(make_float2(hidden_x_993, hidden_y_994));
                        hidden_packed_814[21] = __as_u32(_bf16x2_381);
                        float2 _cvt_f32_236 = __bfloat1622float2(__as_bf16x2(gate_packed_812[22]));
                        float2 _cvt_f32_237 = __bfloat1622float2(__as_bf16x2(up_packed_813[22]));
                        float gate_x_995 = _cvt_f32_236.x;
                        float gate_y_996 = _cvt_f32_236.y;
                        float up_x_997 = _cvt_f32_237.x;
                        float up_y_998 = _cvt_f32_237.y;
                        float _min_498 = fminf(gate_x_995, swiglu_limit);
                        gate_x_995 = _min_498;
                        float _min_499 = fminf(gate_y_996, swiglu_limit);
                        gate_y_996 = _min_499;
                        float _fmax_252 = fmaxf(up_x_997, -swiglu_limit);
                        float _min_500 = fminf(_fmax_252, swiglu_limit);
                        up_x_997 = _min_500;
                        float _fmax_253 = fmaxf(up_y_998, -swiglu_limit);
                        float _min_501 = fminf(_fmax_253, swiglu_limit);
                        up_y_998 = _min_501;
                        float _exp_236 = expf(gate_x_995 * -1.0f);
                        float denominator_x_999 = _exp_236 + 1.0f;
                        float _exp_237 = expf(gate_y_996 * -1.0f);
                        float denominator_y_1000 = _exp_237 + 1.0f;
                        float hidden_x_1001 = gate_x_995 / denominator_x_999 * up_x_997;
                        float hidden_y_1002 = gate_y_996 / denominator_y_1000 * up_y_998;
                        __nv_bfloat162 _bf16x2_382 = __float22bfloat162_rn(make_float2(hidden_x_1001, hidden_y_1002));
                        hidden_packed_814[22] = __as_u32(_bf16x2_382);
                        float2 _cvt_f32_238 = __bfloat1622float2(__as_bf16x2(gate_packed_812[23]));
                        float2 _cvt_f32_239 = __bfloat1622float2(__as_bf16x2(up_packed_813[23]));
                        float gate_x_1003 = _cvt_f32_238.x;
                        float gate_y_1004 = _cvt_f32_238.y;
                        float up_x_1005 = _cvt_f32_239.x;
                        float up_y_1006 = _cvt_f32_239.y;
                        float _min_502 = fminf(gate_x_1003, swiglu_limit);
                        gate_x_1003 = _min_502;
                        float _min_503 = fminf(gate_y_1004, swiglu_limit);
                        gate_y_1004 = _min_503;
                        float _fmax_254 = fmaxf(up_x_1005, -swiglu_limit);
                        float _min_504 = fminf(_fmax_254, swiglu_limit);
                        up_x_1005 = _min_504;
                        float _fmax_255 = fmaxf(up_y_1006, -swiglu_limit);
                        float _min_505 = fminf(_fmax_255, swiglu_limit);
                        up_y_1006 = _min_505;
                        float _exp_238 = expf(gate_x_1003 * -1.0f);
                        float denominator_x_1007 = _exp_238 + 1.0f;
                        float _exp_239 = expf(gate_y_1004 * -1.0f);
                        float denominator_y_1008 = _exp_239 + 1.0f;
                        float hidden_x_1009 = gate_x_1003 / denominator_x_1007 * up_x_1005;
                        float hidden_y_1010 = gate_y_1004 / denominator_y_1008 * up_y_1006;
                        __nv_bfloat162 _bf16x2_383 = __float22bfloat162_rn(make_float2(hidden_x_1009, hidden_y_1010));
                        hidden_packed_814[23] = __as_u32(_bf16x2_383);
                        float2 _cvt_f32_240 = __bfloat1622float2(__as_bf16x2(gate_packed_812[24]));
                        float2 _cvt_f32_241 = __bfloat1622float2(__as_bf16x2(up_packed_813[24]));
                        float gate_x_1011 = _cvt_f32_240.x;
                        float gate_y_1012 = _cvt_f32_240.y;
                        float up_x_1013 = _cvt_f32_241.x;
                        float up_y_1014 = _cvt_f32_241.y;
                        float _min_506 = fminf(gate_x_1011, swiglu_limit);
                        gate_x_1011 = _min_506;
                        float _min_507 = fminf(gate_y_1012, swiglu_limit);
                        gate_y_1012 = _min_507;
                        float _fmax_256 = fmaxf(up_x_1013, -swiglu_limit);
                        float _min_508 = fminf(_fmax_256, swiglu_limit);
                        up_x_1013 = _min_508;
                        float _fmax_257 = fmaxf(up_y_1014, -swiglu_limit);
                        float _min_509 = fminf(_fmax_257, swiglu_limit);
                        up_y_1014 = _min_509;
                        float _exp_240 = expf(gate_x_1011 * -1.0f);
                        float denominator_x_1015 = _exp_240 + 1.0f;
                        float _exp_241 = expf(gate_y_1012 * -1.0f);
                        float denominator_y_1016 = _exp_241 + 1.0f;
                        float hidden_x_1017 = gate_x_1011 / denominator_x_1015 * up_x_1013;
                        float hidden_y_1018 = gate_y_1012 / denominator_y_1016 * up_y_1014;
                        __nv_bfloat162 _bf16x2_384 = __float22bfloat162_rn(make_float2(hidden_x_1017, hidden_y_1018));
                        hidden_packed_814[24] = __as_u32(_bf16x2_384);
                        float2 _cvt_f32_242 = __bfloat1622float2(__as_bf16x2(gate_packed_812[25]));
                        float2 _cvt_f32_243 = __bfloat1622float2(__as_bf16x2(up_packed_813[25]));
                        float gate_x_1019 = _cvt_f32_242.x;
                        float gate_y_1020 = _cvt_f32_242.y;
                        float up_x_1021 = _cvt_f32_243.x;
                        float up_y_1022 = _cvt_f32_243.y;
                        float _min_510 = fminf(gate_x_1019, swiglu_limit);
                        gate_x_1019 = _min_510;
                        float _min_511 = fminf(gate_y_1020, swiglu_limit);
                        gate_y_1020 = _min_511;
                        float _fmax_258 = fmaxf(up_x_1021, -swiglu_limit);
                        float _min_512 = fminf(_fmax_258, swiglu_limit);
                        up_x_1021 = _min_512;
                        float _fmax_259 = fmaxf(up_y_1022, -swiglu_limit);
                        float _min_513 = fminf(_fmax_259, swiglu_limit);
                        up_y_1022 = _min_513;
                        float _exp_242 = expf(gate_x_1019 * -1.0f);
                        float denominator_x_1023 = _exp_242 + 1.0f;
                        float _exp_243 = expf(gate_y_1020 * -1.0f);
                        float denominator_y_1024 = _exp_243 + 1.0f;
                        float hidden_x_1025 = gate_x_1019 / denominator_x_1023 * up_x_1021;
                        float hidden_y_1026 = gate_y_1020 / denominator_y_1024 * up_y_1022;
                        __nv_bfloat162 _bf16x2_385 = __float22bfloat162_rn(make_float2(hidden_x_1025, hidden_y_1026));
                        hidden_packed_814[25] = __as_u32(_bf16x2_385);
                        float2 _cvt_f32_244 = __bfloat1622float2(__as_bf16x2(gate_packed_812[26]));
                        float2 _cvt_f32_245 = __bfloat1622float2(__as_bf16x2(up_packed_813[26]));
                        float gate_x_1027 = _cvt_f32_244.x;
                        float gate_y_1028 = _cvt_f32_244.y;
                        float up_x_1029 = _cvt_f32_245.x;
                        float up_y_1030 = _cvt_f32_245.y;
                        float _min_514 = fminf(gate_x_1027, swiglu_limit);
                        gate_x_1027 = _min_514;
                        float _min_515 = fminf(gate_y_1028, swiglu_limit);
                        gate_y_1028 = _min_515;
                        float _fmax_260 = fmaxf(up_x_1029, -swiglu_limit);
                        float _min_516 = fminf(_fmax_260, swiglu_limit);
                        up_x_1029 = _min_516;
                        float _fmax_261 = fmaxf(up_y_1030, -swiglu_limit);
                        float _min_517 = fminf(_fmax_261, swiglu_limit);
                        up_y_1030 = _min_517;
                        float _exp_244 = expf(gate_x_1027 * -1.0f);
                        float denominator_x_1031 = _exp_244 + 1.0f;
                        float _exp_245 = expf(gate_y_1028 * -1.0f);
                        float denominator_y_1032 = _exp_245 + 1.0f;
                        float hidden_x_1033 = gate_x_1027 / denominator_x_1031 * up_x_1029;
                        float hidden_y_1034 = gate_y_1028 / denominator_y_1032 * up_y_1030;
                        __nv_bfloat162 _bf16x2_386 = __float22bfloat162_rn(make_float2(hidden_x_1033, hidden_y_1034));
                        hidden_packed_814[26] = __as_u32(_bf16x2_386);
                        float2 _cvt_f32_246 = __bfloat1622float2(__as_bf16x2(gate_packed_812[27]));
                        float2 _cvt_f32_247 = __bfloat1622float2(__as_bf16x2(up_packed_813[27]));
                        float gate_x_1035 = _cvt_f32_246.x;
                        float gate_y_1036 = _cvt_f32_246.y;
                        float up_x_1037 = _cvt_f32_247.x;
                        float up_y_1038 = _cvt_f32_247.y;
                        float _min_518 = fminf(gate_x_1035, swiglu_limit);
                        gate_x_1035 = _min_518;
                        float _min_519 = fminf(gate_y_1036, swiglu_limit);
                        gate_y_1036 = _min_519;
                        float _fmax_262 = fmaxf(up_x_1037, -swiglu_limit);
                        float _min_520 = fminf(_fmax_262, swiglu_limit);
                        up_x_1037 = _min_520;
                        float _fmax_263 = fmaxf(up_y_1038, -swiglu_limit);
                        float _min_521 = fminf(_fmax_263, swiglu_limit);
                        up_y_1038 = _min_521;
                        float _exp_246 = expf(gate_x_1035 * -1.0f);
                        float denominator_x_1039 = _exp_246 + 1.0f;
                        float _exp_247 = expf(gate_y_1036 * -1.0f);
                        float denominator_y_1040 = _exp_247 + 1.0f;
                        float hidden_x_1041 = gate_x_1035 / denominator_x_1039 * up_x_1037;
                        float hidden_y_1042 = gate_y_1036 / denominator_y_1040 * up_y_1038;
                        __nv_bfloat162 _bf16x2_387 = __float22bfloat162_rn(make_float2(hidden_x_1041, hidden_y_1042));
                        hidden_packed_814[27] = __as_u32(_bf16x2_387);
                        float2 _cvt_f32_248 = __bfloat1622float2(__as_bf16x2(gate_packed_812[28]));
                        float2 _cvt_f32_249 = __bfloat1622float2(__as_bf16x2(up_packed_813[28]));
                        float gate_x_1043 = _cvt_f32_248.x;
                        float gate_y_1044 = _cvt_f32_248.y;
                        float up_x_1045 = _cvt_f32_249.x;
                        float up_y_1046 = _cvt_f32_249.y;
                        float _min_522 = fminf(gate_x_1043, swiglu_limit);
                        gate_x_1043 = _min_522;
                        float _min_523 = fminf(gate_y_1044, swiglu_limit);
                        gate_y_1044 = _min_523;
                        float _fmax_264 = fmaxf(up_x_1045, -swiglu_limit);
                        float _min_524 = fminf(_fmax_264, swiglu_limit);
                        up_x_1045 = _min_524;
                        float _fmax_265 = fmaxf(up_y_1046, -swiglu_limit);
                        float _min_525 = fminf(_fmax_265, swiglu_limit);
                        up_y_1046 = _min_525;
                        float _exp_248 = expf(gate_x_1043 * -1.0f);
                        float denominator_x_1047 = _exp_248 + 1.0f;
                        float _exp_249 = expf(gate_y_1044 * -1.0f);
                        float denominator_y_1048 = _exp_249 + 1.0f;
                        float hidden_x_1049 = gate_x_1043 / denominator_x_1047 * up_x_1045;
                        float hidden_y_1050 = gate_y_1044 / denominator_y_1048 * up_y_1046;
                        __nv_bfloat162 _bf16x2_388 = __float22bfloat162_rn(make_float2(hidden_x_1049, hidden_y_1050));
                        hidden_packed_814[28] = __as_u32(_bf16x2_388);
                        float2 _cvt_f32_250 = __bfloat1622float2(__as_bf16x2(gate_packed_812[29]));
                        float2 _cvt_f32_251 = __bfloat1622float2(__as_bf16x2(up_packed_813[29]));
                        float gate_x_1051 = _cvt_f32_250.x;
                        float gate_y_1052 = _cvt_f32_250.y;
                        float up_x_1053 = _cvt_f32_251.x;
                        float up_y_1054 = _cvt_f32_251.y;
                        float _min_526 = fminf(gate_x_1051, swiglu_limit);
                        gate_x_1051 = _min_526;
                        float _min_527 = fminf(gate_y_1052, swiglu_limit);
                        gate_y_1052 = _min_527;
                        float _fmax_266 = fmaxf(up_x_1053, -swiglu_limit);
                        float _min_528 = fminf(_fmax_266, swiglu_limit);
                        up_x_1053 = _min_528;
                        float _fmax_267 = fmaxf(up_y_1054, -swiglu_limit);
                        float _min_529 = fminf(_fmax_267, swiglu_limit);
                        up_y_1054 = _min_529;
                        float _exp_250 = expf(gate_x_1051 * -1.0f);
                        float denominator_x_1055 = _exp_250 + 1.0f;
                        float _exp_251 = expf(gate_y_1052 * -1.0f);
                        float denominator_y_1056 = _exp_251 + 1.0f;
                        float hidden_x_1057 = gate_x_1051 / denominator_x_1055 * up_x_1053;
                        float hidden_y_1058 = gate_y_1052 / denominator_y_1056 * up_y_1054;
                        __nv_bfloat162 _bf16x2_389 = __float22bfloat162_rn(make_float2(hidden_x_1057, hidden_y_1058));
                        hidden_packed_814[29] = __as_u32(_bf16x2_389);
                        float2 _cvt_f32_252 = __bfloat1622float2(__as_bf16x2(gate_packed_812[30]));
                        float2 _cvt_f32_253 = __bfloat1622float2(__as_bf16x2(up_packed_813[30]));
                        float gate_x_1059 = _cvt_f32_252.x;
                        float gate_y_1060 = _cvt_f32_252.y;
                        float up_x_1061 = _cvt_f32_253.x;
                        float up_y_1062 = _cvt_f32_253.y;
                        float _min_530 = fminf(gate_x_1059, swiglu_limit);
                        gate_x_1059 = _min_530;
                        float _min_531 = fminf(gate_y_1060, swiglu_limit);
                        gate_y_1060 = _min_531;
                        float _fmax_268 = fmaxf(up_x_1061, -swiglu_limit);
                        float _min_532 = fminf(_fmax_268, swiglu_limit);
                        up_x_1061 = _min_532;
                        float _fmax_269 = fmaxf(up_y_1062, -swiglu_limit);
                        float _min_533 = fminf(_fmax_269, swiglu_limit);
                        up_y_1062 = _min_533;
                        float _exp_252 = expf(gate_x_1059 * -1.0f);
                        float denominator_x_1063 = _exp_252 + 1.0f;
                        float _exp_253 = expf(gate_y_1060 * -1.0f);
                        float denominator_y_1064 = _exp_253 + 1.0f;
                        float hidden_x_1065 = gate_x_1059 / denominator_x_1063 * up_x_1061;
                        float hidden_y_1066 = gate_y_1060 / denominator_y_1064 * up_y_1062;
                        __nv_bfloat162 _bf16x2_390 = __float22bfloat162_rn(make_float2(hidden_x_1065, hidden_y_1066));
                        hidden_packed_814[30] = __as_u32(_bf16x2_390);
                        float2 _cvt_f32_254 = __bfloat1622float2(__as_bf16x2(gate_packed_812[31]));
                        float2 _cvt_f32_255 = __bfloat1622float2(__as_bf16x2(up_packed_813[31]));
                        float gate_x_1067 = _cvt_f32_254.x;
                        float gate_y_1068 = _cvt_f32_254.y;
                        float up_x_1069 = _cvt_f32_255.x;
                        float up_y_1070 = _cvt_f32_255.y;
                        float _min_534 = fminf(gate_x_1067, swiglu_limit);
                        gate_x_1067 = _min_534;
                        float _min_535 = fminf(gate_y_1068, swiglu_limit);
                        gate_y_1068 = _min_535;
                        float _fmax_270 = fmaxf(up_x_1069, -swiglu_limit);
                        float _min_536 = fminf(_fmax_270, swiglu_limit);
                        up_x_1069 = _min_536;
                        float _fmax_271 = fmaxf(up_y_1070, -swiglu_limit);
                        float _min_537 = fminf(_fmax_271, swiglu_limit);
                        up_y_1070 = _min_537;
                        float _exp_254 = expf(gate_x_1067 * -1.0f);
                        float denominator_x_1071 = _exp_254 + 1.0f;
                        float _exp_255 = expf(gate_y_1068 * -1.0f);
                        float denominator_y_1072 = _exp_255 + 1.0f;
                        float hidden_x_1073 = gate_x_1067 / denominator_x_1071 * up_x_1069;
                        float hidden_y_1074 = gate_y_1068 / denominator_y_1072 * up_y_1070;
                        __nv_bfloat162 _bf16x2_391 = __float22bfloat162_rn(make_float2(hidden_x_1073, hidden_y_1074));
                        hidden_packed_814[31] = __as_u32(_bf16x2_391);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1075 = tid / 32;
                        int lane_1076 = tid % 32;
                        #pragma unroll
                        for (int half_18 = 0; half_18 < 2; half_18++) {
                            #pragma unroll
                            for (int col_tile_26 = 0; col_tile_26 < 2; col_tile_26++) {
                                int row_23 = warp_1075 * 32 + half_18 * 16 + lane_1076 % 16;
                                int col_51 = col_tile_26 * 16 + lane_1076 / 16 * 8;
                                unsigned int address_3_18 = d_smem_addr + (unsigned int)((row_23 * 32 + col_51) * 2);
                                address_3_18 = address_3_18 ^ (address_3_18 & 511) >> 7 << 4;
                                int offset_18 = half_18 * 8 + col_tile_26 * 4;
                                uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(address_3_18);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1077 = tid / 32;
                        int lane_1078 = tid % 32;
                        #pragma unroll
                        for (int half_19 = 0; half_19 < 2; half_19++) {
                            #pragma unroll
                            for (int col_tile_27 = 0; col_tile_27 < 2; col_tile_27++) {
                                int row_24 = warp_1077 * 32 + half_19 * 16 + lane_1078 % 16;
                                int col_52 = col_tile_27 * 16 + lane_1078 / 16 * 8;
                                unsigned int address_3_19 = d_smem_addr + 8192 + (unsigned int)((row_24 * 32 + col_52) * 2);
                                address_3_19 = address_3_19 ^ (address_3_19 & 511) >> 7 << 4;
                                int offset_19 = half_19 * 8 + col_tile_27 * 4;
                                uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(address_3_19);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1079 = tid / 32;
                        int lane_1080 = tid % 32;
                        #pragma unroll
                        for (int half_20 = 0; half_20 < 2; half_20++) {
                            #pragma unroll
                            for (int col_tile_28 = 0; col_tile_28 < 2; col_tile_28++) {
                                int row_25 = warp_1079 * 32 + half_20 * 16 + lane_1080 % 16;
                                int col_53 = col_tile_28 * 16 + lane_1080 / 16 * 8;
                                unsigned int address_3_20 = d_smem_addr + 16384 + (unsigned int)((row_25 * 32 + col_53) * 2);
                                address_3_20 = address_3_20 ^ (address_3_20 & 511) >> 7 << 4;
                                int offset_20 = half_20 * 8 + col_tile_28 * 4;
                                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(address_3_20);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1081 = tid / 32;
                        int lane_1082 = tid % 32;
                        #pragma unroll
                        for (int half_21 = 0; half_21 < 2; half_21++) {
                            #pragma unroll
                            for (int col_tile_29 = 0; col_tile_29 < 2; col_tile_29++) {
                                int row_26 = warp_1081 * 32 + half_21 * 16 + lane_1082 % 16;
                                int col_54 = col_tile_29 * 16 + lane_1082 / 16 * 8;
                                unsigned int address_3_21 = d_smem_addr + (unsigned int)((row_26 * 32 + col_54) * 2);
                                address_3_21 = address_3_21 ^ (address_3_21 & 511) >> 7 << 4;
                                int offset_21 = 16 + half_21 * 8 + col_tile_29 * 4;
                                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(address_3_21);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1083 = tid / 32;
                        int lane_1084 = tid % 32;
                        #pragma unroll
                        for (int half_22 = 0; half_22 < 2; half_22++) {
                            #pragma unroll
                            for (int col_tile_30 = 0; col_tile_30 < 2; col_tile_30++) {
                                int row_27 = warp_1083 * 32 + half_22 * 16 + lane_1084 % 16;
                                int col_55 = col_tile_30 * 16 + lane_1084 / 16 * 8;
                                unsigned int address_3_22 = d_smem_addr + 8192 + (unsigned int)((row_27 * 32 + col_55) * 2);
                                address_3_22 = address_3_22 ^ (address_3_22 & 511) >> 7 << 4;
                                int offset_22 = 16 + half_22 * 8 + col_tile_30 * 4;
                                uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(address_3_22);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1085 = tid / 32;
                        int lane_1086 = tid % 32;
                        #pragma unroll
                        for (int half_23 = 0; half_23 < 2; half_23++) {
                            #pragma unroll
                            for (int col_tile_31 = 0; col_tile_31 < 2; col_tile_31++) {
                                int row_28 = warp_1085 * 32 + half_23 * 16 + lane_1086 % 16;
                                int col_56 = col_tile_31 * 16 + lane_1086 / 16 * 8;
                                unsigned int address_3_23 = d_smem_addr + 16384 + (unsigned int)((row_28 * 32 + col_56) * 2);
                                address_3_23 = address_3_23 ^ (address_3_23 & 511) >> 7 << 4;
                                int offset_23 = 16 + half_23 * 8 + col_tile_31 * 4;
                                uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(address_3_23);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                        if (tid / 32 == 0) {
                            if (warp == 0) {
                                if (elect_sync()) {
                                    asm volatile("cp.async.bulk.wait_group 0;");
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (x))), "r"(static_cast<unsigned int>(2)) : "memory");
                                }
                            }
                        }
                    }
                }
                gemm_bits = phase_bits_2;
            } else if (compute < shared_tasks) {
                int col_blocks_4 = (hidden + 512 - 1) / 512;
                int x_1 = -1;
                int y_1 = -1;
                int expert_1 = -1;
                int k_start_1 = 0;
                int k_end_1 = 0;
                int first_1 = 0;
                int row_blocks_1 = local_tokens / 256;
                if (compute - shared_fused < row_blocks_1 * col_blocks_4) {
                    int supergroup_1 = (compute - shared_fused) / (row_blocks_1 * 8);
                    int full_cols_1 = col_blocks_4 / 8 * 8;
                    int row_29 = 0;
                    int col_57 = 0;
                    if (compute - shared_fused < row_blocks_1 * full_cols_1) {
                        row_29 = (compute - shared_fused) % (row_blocks_1 * 8) / 8;
                        col_57 = supergroup_1 * 8 + (compute - shared_fused) % 8;
                    } else {
                        row_29 = (compute - shared_fused - row_blocks_1 * full_cols_1) / (col_blocks_4 - full_cols_1);
                        col_57 = full_cols_1 + (compute - shared_fused - row_blocks_1 * full_cols_1) % (col_blocks_4 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_29 = row_blocks_1 - row_29 - 1;
                    }
                    x_1 = row_29;
                    y_1 = col_57;
                    expert_1 = 0;
                }
                unsigned int phase_bits_3 = gemm_bits;
                int has_hi_1 = 0;
                has_hi_1 = (int)((y_1 * 2 + 1) * 256 < hidden);
                int global_mini_1 = 0;
                int macro_rows_2 = 0;
                int iterations_1 = intermediate / 64;
                if (expert_1 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                bool enabled_value_2 = 1;
                                if (enabled_value_2 != 0) {
                                    int32_t _relaxed_ld_6;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                    int value_3 = _relaxed_ld_6;
                                    while (value_3 < 2 * (intermediate / 128)) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_7;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                        value_3 = _relaxed_ld_7;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                int _min_538 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                int _max_2 = ((0) > (_min_538) ? (0) : (_min_538));
                                int mini_rows_4 = _max_2;
                                int required_4 = (mini_rows_4 + 127) / 128 * ((intermediate + 511) / 512);
                            }
                            int ring_2 = 0;
                            #pragma unroll 1
                            for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                mbarrier_wait(gemm_finished_addr + (ring_2) * 8, phase_bits_3 >> (unsigned int)(16 + ring_2) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&hidden_shared_in)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&wd_shared)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)(ring_2 * 16384)), "l"((&wd_shared)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << 16 + ring_2);
                                ring_2 = (ring_2 + 1) % 4;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_3 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_3 >> 22 & 1);
                                phase_bits_3 = phase_bits_3 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_3 = 0; idx_3 < iterations_1; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_3 >> (unsigned int)ring_3 & 1);
                                    int _mma_a_lo_2 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                    int _mma_b_lo_2 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
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
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)));
                                    int _mma_a_lo_3 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                    int _mma_b_lo_3 = (((b_hi_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
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
            "mov.b32 id, 272630928;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)));
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_3) * 8, (uint16_t)(3));
                                    phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << ring_3);
                                    ring_3 = (ring_3 + 1) % 4;
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
                                unsigned int address_4 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk * 32);
                                float _tmem_load_32[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[15]))
                                    : "r"(address_4));
                                #pragma unroll
                                for (int pair = 0; pair < 8; pair++) {
                                    __nv_bfloat162 _bf16x2_392 = __float22bfloat162_rn(make_float2(_tmem_load_32[pair * 2], _tmem_load_32[pair * 2 + 1]));
                                    packed[chunk * 16 + sub * 8 + pair] = __as_u32(_bf16x2_392);
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int last = 1;
                        last = 1 - has_hi_1;
                        if (last != 0) {
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        if (tid == 0) {
                            int previous_offset_1 = macro_size;
                            int output_row = x_1 * 256 + cta_rank_0 * 128;
                            int _min_539 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            if (output_row < _min_539) {
                            }
                        }
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < 8; chunk_1++) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_0 = tid / 32;
                            int lane_2 = tid % 32;
                            #pragma unroll
                            for (int half_24 = 0; half_24 < 2; half_24++) {
                                #pragma unroll
                                for (int col_tile_32 = 0; col_tile_32 < 2; col_tile_32++) {
                                    int row_30 = warp_0 * 32 + half_24 * 16 + lane_2 % 16;
                                    int col_58 = col_tile_32 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_30 * 32 + col_58) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_24 = chunk_1 * 16 + half_24 * 8 + col_tile_32 * 4;
                                    uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&y_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 2 * 8 + chunk_1), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (has_hi_1 != 0) {
                            unsigned int packed_0[128];
                            #pragma unroll
                            for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                                #pragma unroll
                                for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                    unsigned int address_6 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_2 * 32);
                                    float _tmem_load_33[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[15]))
                                        : "r"(address_6));
                                    #pragma unroll
                                    for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                        __nv_bfloat162 _bf16x2_393 = __float22bfloat162_rn(make_float2(_tmem_load_33[pair_1 * 2], _tmem_load_33[pair_1 * 2 + 1]));
                                        packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(_bf16x2_393);
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
                                int lane_3 = tid % 32;
                                #pragma unroll
                                for (int half_25 = 0; half_25 < 2; half_25++) {
                                    #pragma unroll
                                    for (int col_tile_33 = 0; col_tile_33 < 2; col_tile_33++) {
                                        int row_31 = warp_0_1 * 32 + half_25 * 16 + lane_3 % 16;
                                        int col_59 = col_tile_33 * 16 + lane_3 / 16 * 8;
                                        unsigned int address_7 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_31 * 32 + col_59) * 2);
                                        address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                        int offset_25 = chunk_3 * 16 + half_25 * 8 + col_tile_33 * 4;
                                        uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(address_7);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&y_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"((y_1 * 2 + 1) * 8 + chunk_3), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                gemm_bits = phase_bits_3;
            } else {
                int ordered_mini = (compute - shared_tasks) / mini_tasks;
                int task_2 = (compute - shared_tasks) % mini_tasks;
                int macro_1 = macros - 1;
                int mini_2 = ordered_mini;
                if (ordered_mini >= last_minis) {
                    macro_1 = macros - 2 - (ordered_mini - last_minis) / minis_per_macro;
                    mini_2 = (ordered_mini - last_minis) % minis_per_macro;
                }
                if (task_2 < mini_gate) {
                    int col_blocks_5 = (intermediate + 256 - 1) / 256;
                    int x_2 = -1;
                    int y_2 = -1;
                    int expert_2 = -1;
                    int k_start_2 = 0;
                    int k_end_2 = 0;
                    int first_2 = 0;
                    int first_block = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int _min_540 = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    int end_block = _min_540;
                    int block = first_block + task_2 / col_blocks_5;
                    if (block < end_block) {
                        int index = counts[3 * experts + block];
                        int offset_26 = counts[experts + index] / 256;
                        int _max_3 = ((first_block) > (offset_26) ? (first_block) : (offset_26));
                        int first_row_2 = _max_3;
                        int _min_541 = ((end_block) < (offset_26 + counts[index] / 256) ? (end_block) : (offset_26 + counts[index] / 256));
                        int rows_2 = _min_541 - first_row_2;
                        int supergroup_2 = (task_2 - (first_row_2 - first_block) * col_blocks_5) / (rows_2 * 8);
                        int full_cols_2 = col_blocks_5 / 8 * 8;
                        int row_32 = 0;
                        int col_60 = 0;
                        if (task_2 - (first_row_2 - first_block) * col_blocks_5 < rows_2 * full_cols_2) {
                            row_32 = (task_2 - (first_row_2 - first_block) * col_blocks_5) % (rows_2 * 8) / 8;
                            col_60 = supergroup_2 * 8 + (task_2 - (first_row_2 - first_block) * col_blocks_5) % 8;
                        } else {
                            row_32 = (task_2 - (first_row_2 - first_block) * col_blocks_5 - rows_2 * full_cols_2) / (col_blocks_5 - full_cols_2);
                            col_60 = full_cols_2 + (task_2 - (first_row_2 - first_block) * col_blocks_5 - rows_2 * full_cols_2) % (col_blocks_5 - full_cols_2);
                        }
                        if ((supergroup_2 & 1) != 0) {
                            row_32 = rows_2 - row_32 - 1;
                        }
                        x_2 = first_row_2 + row_32 - macro_1 * (macro_size / 256);
                        y_2 = col_60;
                        expert_2 = index;
                    }
                    unsigned int phase_bits_4 = gemm_bits;
                    int has_hi_2 = 0;
                    has_hi_2 = (int)((y_2 * 2 + 1) * 256 < intermediate);
                    int global_mini_2 = macro_1 * (macro_size / mini_size) + mini_2;
                    int macro_rows_3 = macro_1 * (macro_size / 256);
                    int iterations_2 = hidden / 128;
                    int macro_k = macro_1 * (macro_size / 128);
                    if (expert_2 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_542 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                    int _max_4 = ((0) > (_min_542) ? (0) : (_min_542));
                                    int mini_rows_5 = _max_4;
                                    int required_5 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_3 = 1;
                                    if (enabled_value_3 != 0) {
                                        int32_t _relaxed_ld_8;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(x_ready + global_mini_2) : "memory");
                                        int value_4 = _relaxed_ld_8;
                                        while (value_4 < required_5) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_9;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(x_ready + global_mini_2) : "memory");
                                            value_4 = _relaxed_ld_9;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                int ring_4 = 0;
                                #pragma unroll 1
                                for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_4) * 8, phase_bits_4 >> (unsigned int)(16 + ring_4) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v40_addr + (unsigned int)(ring_4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v41_addr + (unsigned int)(ring_4 * 16384)), "l"((&wg_q)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v42_addr + (unsigned int)(ring_4 * 16384)), "l"((&wu_q)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_4);
                                    ring_4 = (ring_4 + 1) % 4;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        int _min_543 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                        int _max_5 = ((0) > (_min_543) ? (0) : (_min_543));
                                        int mini_rows_6 = _max_5;
                                        int required_6 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_4 = 1;
                                        if (enabled_value_4 != 0) {
                                            int32_t _relaxed_ld_10;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(x_ready + global_mini_2) : "memory");
                                            int value_5 = _relaxed_ld_10;
                                            while (value_5 < required_6) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_11;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(x_ready + global_mini_2) : "memory");
                                                value_5 = _relaxed_ld_11;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_5 = 0;
                                    #pragma unroll 1
                                    for (int idx_5 = 0; idx_5 < iterations_2; idx_5++) {
                                        mbarrier_wait(scales_finished_addr + (ring_5) * 8, phase_bits_4 >> (unsigned int)(16 + ring_5) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v43_addr + (unsigned int)(ring_5 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + (ring_5) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v44_addr + (unsigned int)(ring_5 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((expert_2 * (intermediate / 128) + y_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + (ring_5) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v45_addr + (unsigned int)(ring_5 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((expert_2 * (intermediate / 128) + y_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + (ring_5) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_5);
                                        ring_5 = (ring_5 + 1) % 4;
                                    }
                                }
                            }
                        } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_6 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_4 >> 22 & 1);
                                    phase_bits_4 = phase_bits_4 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_6 = 0; idx_6 < iterations_2; idx_6++) {
                                        mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_6) * 8, 5120);
                                        mbarrier_wait(scales_arrived_addr + (ring_6) * 8, phase_bits_4 >> (unsigned int)(8 + ring_6) & 1);
                                        int buffer = idx_6 % 3;
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer * 4, make_sf_cp_desc_sbo128(smem_v43_addr + (unsigned int)(ring_6 * 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer * 8, make_sf_cp_desc_sbo128(smem_v44_addr + (unsigned int)(ring_6 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer * 8 + 4), make_sf_cp_desc_sbo128((smem_v44_addr + (unsigned int)(ring_6 * 1024) + 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer * 8, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)(ring_6 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer * 8 + 4), make_sf_cp_desc_sbo128((smem_v45_addr + (unsigned int)(ring_6 * 1024) + 512)));
                                        tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_6) * 8, (uint16_t)(3));
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_6) * 8, 98304);
                                        mbarrier_wait(gemm_arrived_addr + (ring_6) * 8, phase_bits_4 >> (unsigned int)ring_6 & 1);
                                        int _mma_a_lo_4 = (((smem_v40_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                        int _mma_b_lo_4 = (((smem_v41_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb + buffer * 8 + 0)), ((idx_6 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer * 8 + 0) | 0x80000000), 1);
                                        }
                                        int _mma_a_lo_5 = (((smem_v40_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                        int _mma_b_lo_5 = (((smem_v42_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer * 8 + 0)), ((idx_6 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer * 8 + 0) | 0x80000000), 1);
                                        }
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_6) * 8, (uint16_t)(3));
                                        phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << ring_6) ^ (unsigned int)(1 << 8 + ring_6);
                                        ring_6 = (ring_6 + 1) % 4;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else {
                            if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_4 >> 6 & 1);
                                int warp_row = tid / 32 * 32;
                                unsigned int packed_1[128];
                                #pragma unroll
                                for (int i_32 = 0; i_32 < 8; i_32++) {
                                    float _tmem_load_34[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_34[0]), "=f"(_tmem_load_34[1]), "=f"(_tmem_load_34[2]), "=f"(_tmem_load_34[3]), "=f"(_tmem_load_34[4]), "=f"(_tmem_load_34[5]), "=f"(_tmem_load_34[6]), "=f"(_tmem_load_34[7]), "=f"(_tmem_load_34[8]), "=f"(_tmem_load_34[9]), "=f"(_tmem_load_34[10]), "=f"(_tmem_load_34[11]), "=f"(_tmem_load_34[12]), "=f"(_tmem_load_34[13]), "=f"(_tmem_load_34[14]), "=f"(_tmem_load_34[15]), "=f"(_tmem_load_34[16]), "=f"(_tmem_load_34[17]), "=f"(_tmem_load_34[18]), "=f"(_tmem_load_34[19]), "=f"(_tmem_load_34[20]), "=f"(_tmem_load_34[21]), "=f"(_tmem_load_34[22]), "=f"(_tmem_load_34[23]), "=f"(_tmem_load_34[24]), "=f"(_tmem_load_34[25]), "=f"(_tmem_load_34[26]), "=f"(_tmem_load_34[27]), "=f"(_tmem_load_34[28]), "=f"(_tmem_load_34[29]), "=f"(_tmem_load_34[30]), "=f"(_tmem_load_34[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row << 16) + (unsigned int)(i_32 * 32)));
                                    #pragma unroll
                                    for (int j_16 = 0; j_16 < 16; j_16++) {
                                        __nv_bfloat162 _bf16x2_394 = __float22bfloat162_rn(make_float2(_tmem_load_34[2 * j_16], _tmem_load_34[2 * j_16 + 1]));
                                        packed_1[i_32 * 16 + j_16] = __as_u32(_bf16x2_394);
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                unsigned int scale_word_16 = 0;
                                unsigned int block_0[16];
                                #pragma unroll
                                for (int j_17 = 0; j_17 < 16; j_17++) {
                                    block_0[j_17] = packed_1[j_17];
                                }
                                #pragma unroll
                                for (int j_18 = 0; j_18 < 4; j_18++) {
                                    unsigned int address_8 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_18 * 16);
                                    address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_8 - smem_v50_addr)), "r"(packed_1[4 * j_18]), "r"(packed_1[4 * j_18 + 1]), "r"(packed_1[4 * j_18 + 2]), "r"(packed_1[4 * j_18 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_32;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(block_0[0]));
                                unsigned int amax_pair_16 = _bf16x2_abs_32;
                                #pragma unroll
                                for (int i_33 = 1; i_33 < 16; i_33++) {
                                    uint32_t _bf16x2_abs_33;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(block_0[i_33]));
                                    uint32_t _bf16x2_max_16;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(amax_pair_16), "r"(_bf16x2_abs_33));
                                    amax_pair_16 = _bf16x2_max_16;
                                }
                                uint16_t _bf16_max_16;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_16) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                float _cvt_f32_bf16_16;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(_bf16_max_16)));
                                float amax_16 = _cvt_f32_bf16_16;
                                float _fmax_272 = fmaxf(amax_16 * 0.002232142857f, 1e-12f);
                                float scale_16 = _fmax_272;
                                uint16_t _ue8m0x2_f32_16;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_16) : "f"(scale_16), "f"(scale_16));
                                unsigned int scale_byte_16 = (unsigned int)_ue8m0x2_f32_16 & 255;
                                unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                unsigned int words_16[8];
                                #pragma unroll
                                for (int i_34 = 0; i_34 < 8; i_34++) {
                                    uint32_t _bf16x2_mul_32;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(block_0[i_34 * 2]), "r"(inverse_16));
                                    uint16_t _e4m3x2_32;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_32) : "r"(_bf16x2_mul_32));
                                    uint32_t _bf16x2_mul_33;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(block_0[i_34 * 2 + 1]), "r"(inverse_16));
                                    uint16_t _e4m3x2_33;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_33) : "r"(_bf16x2_mul_33));
                                    words_16[i_34] = (unsigned int)_e4m3x2_32 | (unsigned int)_e4m3x2_33 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_16[0]), "r"(words_16[1]), "r"(words_16[2]), "r"(words_16[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16[4]), "r"(words_16[5]), "r"(words_16[6]), "r"(words_16[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_1[16];
                                #pragma unroll
                                for (int j_19 = 0; j_19 < 16; j_19++) {
                                    block_1[j_19] = packed_1[16 + j_19];
                                }
                                #pragma unroll
                                for (int j_20 = 0; j_20 < 4; j_20++) {
                                    unsigned int address_9 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_20 * 16);
                                    address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_9 - smem_v50_addr)), "r"(packed_1[16 + 4 * j_20]), "r"(packed_1[16 + 4 * j_20 + 1]), "r"(packed_1[16 + 4 * j_20 + 2]), "r"(packed_1[16 + 4 * j_20 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_34;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(block_1[0]));
                                unsigned int amax_pair_2_1 = _bf16x2_abs_34;
                                #pragma unroll
                                for (int i_35 = 1; i_35 < 16; i_35++) {
                                    uint32_t _bf16x2_abs_35;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(block_1[i_35]));
                                    uint32_t _bf16x2_max_17;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(amax_pair_2_1), "r"(_bf16x2_abs_35));
                                    amax_pair_2_1 = _bf16x2_max_17;
                                }
                                uint16_t _bf16_max_17;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_17) : "h"((uint16_t)(amax_pair_2_1 & 65535)), "h"((uint16_t)(amax_pair_2_1 >> 16)));
                                float _cvt_f32_bf16_17;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_17) : "h"((uint16_t)(_bf16_max_17)));
                                float amax_3_1 = _cvt_f32_bf16_17;
                                float _fmax_273 = fmaxf(amax_3_1 * 0.002232142857f, 1e-12f);
                                float scale_4_1 = _fmax_273;
                                uint16_t _ue8m0x2_f32_17;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_17) : "f"(scale_4_1), "f"(scale_4_1));
                                unsigned int scale_byte_5_1 = (unsigned int)_ue8m0x2_f32_17 & 255;
                                unsigned int inverse_lane_6_1 = 254 - scale_byte_5_1 << 7;
                                unsigned int inverse_7_1 = inverse_lane_6_1 | inverse_lane_6_1 << 16;
                                unsigned int words_8_1[8];
                                #pragma unroll
                                for (int i_36 = 0; i_36 < 8; i_36++) {
                                    uint32_t _bf16x2_mul_34;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(block_1[i_36 * 2]), "r"(inverse_7_1));
                                    uint16_t _e4m3x2_34;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_34) : "r"(_bf16x2_mul_34));
                                    uint32_t _bf16x2_mul_35;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(block_1[i_36 * 2 + 1]), "r"(inverse_7_1));
                                    uint16_t _e4m3x2_35;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_35) : "r"(_bf16x2_mul_35));
                                    words_8_1[i_36] = (unsigned int)_e4m3x2_34 | (unsigned int)_e4m3x2_35 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_5_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_8_1[0]), "r"(words_8_1[1]), "r"(words_8_1[2]), "r"(words_8_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_1[4]), "r"(words_8_1[5]), "r"(words_8_1[6]), "r"(words_8_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 32), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_9[16];
                                #pragma unroll
                                for (int j_21 = 0; j_21 < 16; j_21++) {
                                    block_9[j_21] = packed_1[32 + j_21];
                                }
                                #pragma unroll
                                for (int j_22 = 0; j_22 < 4; j_22++) {
                                    unsigned int address_10 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_22 * 16);
                                    address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_10 - smem_v50_addr)), "r"(packed_1[32 + 4 * j_22]), "r"(packed_1[32 + 4 * j_22 + 1]), "r"(packed_1[32 + 4 * j_22 + 2]), "r"(packed_1[32 + 4 * j_22 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_36;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(block_9[0]));
                                unsigned int amax_pair_10_1 = _bf16x2_abs_36;
                                #pragma unroll
                                for (int i_37 = 1; i_37 < 16; i_37++) {
                                    uint32_t _bf16x2_abs_37;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(block_9[i_37]));
                                    uint32_t _bf16x2_max_18;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(amax_pair_10_1), "r"(_bf16x2_abs_37));
                                    amax_pair_10_1 = _bf16x2_max_18;
                                }
                                uint16_t _bf16_max_18;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_18) : "h"((uint16_t)(amax_pair_10_1 & 65535)), "h"((uint16_t)(amax_pair_10_1 >> 16)));
                                float _cvt_f32_bf16_18;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_18) : "h"((uint16_t)(_bf16_max_18)));
                                float amax_11_1 = _cvt_f32_bf16_18;
                                float _fmax_274 = fmaxf(amax_11_1 * 0.002232142857f, 1e-12f);
                                float scale_12_1 = _fmax_274;
                                uint16_t _ue8m0x2_f32_18;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_18) : "f"(scale_12_1), "f"(scale_12_1));
                                unsigned int scale_byte_13_1 = (unsigned int)_ue8m0x2_f32_18 & 255;
                                unsigned int inverse_lane_14_1 = 254 - scale_byte_13_1 << 7;
                                unsigned int inverse_15_1 = inverse_lane_14_1 | inverse_lane_14_1 << 16;
                                unsigned int words_16_1[8];
                                #pragma unroll
                                for (int i_38 = 0; i_38 < 8; i_38++) {
                                    uint32_t _bf16x2_mul_36;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(block_9[i_38 * 2]), "r"(inverse_15_1));
                                    uint16_t _e4m3x2_36;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_36) : "r"(_bf16x2_mul_36));
                                    uint32_t _bf16x2_mul_37;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(block_9[i_38 * 2 + 1]), "r"(inverse_15_1));
                                    uint16_t _e4m3x2_37;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_37) : "r"(_bf16x2_mul_37));
                                    words_16_1[i_38] = (unsigned int)_e4m3x2_36 | (unsigned int)_e4m3x2_37 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_13_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_16_1[0]), "r"(words_16_1[1]), "r"(words_16_1[2]), "r"(words_16_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16_1[4]), "r"(words_16_1[5]), "r"(words_16_1[6]), "r"(words_16_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 64), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_17[16];
                                #pragma unroll
                                for (int j_23 = 0; j_23 < 16; j_23++) {
                                    block_17[j_23] = packed_1[48 + j_23];
                                }
                                #pragma unroll
                                for (int j_24 = 0; j_24 < 4; j_24++) {
                                    unsigned int address_11 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_24 * 16);
                                    address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_11 - smem_v50_addr)), "r"(packed_1[48 + 4 * j_24]), "r"(packed_1[48 + 4 * j_24 + 1]), "r"(packed_1[48 + 4 * j_24 + 2]), "r"(packed_1[48 + 4 * j_24 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 3), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_38;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(block_17[0]));
                                unsigned int amax_pair_18 = _bf16x2_abs_38;
                                #pragma unroll
                                for (int i_39 = 1; i_39 < 16; i_39++) {
                                    uint32_t _bf16x2_abs_39;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(block_17[i_39]));
                                    uint32_t _bf16x2_max_19;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(amax_pair_18), "r"(_bf16x2_abs_39));
                                    amax_pair_18 = _bf16x2_max_19;
                                }
                                uint16_t _bf16_max_19;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_19) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                float _cvt_f32_bf16_19;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(_bf16_max_19)));
                                float amax_19 = _cvt_f32_bf16_19;
                                float _fmax_275 = fmaxf(amax_19 * 0.002232142857f, 1e-12f);
                                float scale_20 = _fmax_275;
                                uint16_t _ue8m0x2_f32_19;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_19) : "f"(scale_20), "f"(scale_20));
                                unsigned int scale_byte_21 = (unsigned int)_ue8m0x2_f32_19 & 255;
                                unsigned int inverse_lane_22 = 254 - scale_byte_21 << 7;
                                unsigned int inverse_23 = inverse_lane_22 | inverse_lane_22 << 16;
                                unsigned int words_24[8];
                                #pragma unroll
                                for (int i_40 = 0; i_40 < 8; i_40++) {
                                    uint32_t _bf16x2_mul_38;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(block_17[i_40 * 2]), "r"(inverse_23));
                                    uint16_t _e4m3x2_38;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_38) : "r"(_bf16x2_mul_38));
                                    uint32_t _bf16x2_mul_39;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(block_17[i_40 * 2 + 1]), "r"(inverse_23));
                                    uint16_t _e4m3x2_39;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_39) : "r"(_bf16x2_mul_39));
                                    words_24[i_40] = (unsigned int)_e4m3x2_38 | (unsigned int)_e4m3x2_39 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_21 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_24[0]), "r"(words_24[1]), "r"(words_24[2]), "r"(words_24[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24[4]), "r"(words_24[5]), "r"(words_24[6]), "r"(words_24[7]) : "memory");
                                smem_v52[tid % 32 * 4 + tid / 32] = scale_word_16;
                                scale_word_16 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 96), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2, smem_v52_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_25[16];
                                #pragma unroll
                                for (int j_25 = 0; j_25 < 16; j_25++) {
                                    block_25[j_25] = packed_1[64 + j_25];
                                }
                                #pragma unroll
                                for (int j_26 = 0; j_26 < 4; j_26++) {
                                    unsigned int address_12 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_26 * 16);
                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_12 - smem_v50_addr)), "r"(packed_1[64 + 4 * j_26]), "r"(packed_1[64 + 4 * j_26 + 1]), "r"(packed_1[64 + 4 * j_26 + 2]), "r"(packed_1[64 + 4 * j_26 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_40;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(block_25[0]));
                                unsigned int amax_pair_26 = _bf16x2_abs_40;
                                #pragma unroll
                                for (int i_41 = 1; i_41 < 16; i_41++) {
                                    uint32_t _bf16x2_abs_41;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(block_25[i_41]));
                                    uint32_t _bf16x2_max_20;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(amax_pair_26), "r"(_bf16x2_abs_41));
                                    amax_pair_26 = _bf16x2_max_20;
                                }
                                uint16_t _bf16_max_20;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_20) : "h"((uint16_t)(amax_pair_26 & 65535)), "h"((uint16_t)(amax_pair_26 >> 16)));
                                float _cvt_f32_bf16_20;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(_bf16_max_20)));
                                float amax_27 = _cvt_f32_bf16_20;
                                float _fmax_276 = fmaxf(amax_27 * 0.002232142857f, 1e-12f);
                                float scale_28 = _fmax_276;
                                uint16_t _ue8m0x2_f32_20;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_20) : "f"(scale_28), "f"(scale_28));
                                unsigned int scale_byte_29 = (unsigned int)_ue8m0x2_f32_20 & 255;
                                unsigned int inverse_lane_30 = 254 - scale_byte_29 << 7;
                                unsigned int inverse_31 = inverse_lane_30 | inverse_lane_30 << 16;
                                unsigned int words_32[8];
                                #pragma unroll
                                for (int i_42 = 0; i_42 < 8; i_42++) {
                                    uint32_t _bf16x2_mul_40;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(block_25[i_42 * 2]), "r"(inverse_31));
                                    uint16_t _e4m3x2_40;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_40) : "r"(_bf16x2_mul_40));
                                    uint32_t _bf16x2_mul_41;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(block_25[i_42 * 2 + 1]), "r"(inverse_31));
                                    uint16_t _e4m3x2_41;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_41) : "r"(_bf16x2_mul_41));
                                    words_32[i_42] = (unsigned int)_e4m3x2_40 | (unsigned int)_e4m3x2_41 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_29;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_32[0]), "r"(words_32[1]), "r"(words_32[2]), "r"(words_32[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_32[4]), "r"(words_32[5]), "r"(words_32[6]), "r"(words_32[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 128), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_33[16];
                                #pragma unroll
                                for (int j_27 = 0; j_27 < 16; j_27++) {
                                    block_33[j_27] = packed_1[80 + j_27];
                                }
                                #pragma unroll
                                for (int j_28 = 0; j_28 < 4; j_28++) {
                                    unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_28 * 16);
                                    address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_13 - smem_v50_addr)), "r"(packed_1[80 + 4 * j_28]), "r"(packed_1[80 + 4 * j_28 + 1]), "r"(packed_1[80 + 4 * j_28 + 2]), "r"(packed_1[80 + 4 * j_28 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 5), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_42;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(block_33[0]));
                                unsigned int amax_pair_34 = _bf16x2_abs_42;
                                #pragma unroll
                                for (int i_43 = 1; i_43 < 16; i_43++) {
                                    uint32_t _bf16x2_abs_43;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(block_33[i_43]));
                                    uint32_t _bf16x2_max_21;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(amax_pair_34), "r"(_bf16x2_abs_43));
                                    amax_pair_34 = _bf16x2_max_21;
                                }
                                uint16_t _bf16_max_21;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_21) : "h"((uint16_t)(amax_pair_34 & 65535)), "h"((uint16_t)(amax_pair_34 >> 16)));
                                float _cvt_f32_bf16_21;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_21) : "h"((uint16_t)(_bf16_max_21)));
                                float amax_35 = _cvt_f32_bf16_21;
                                float _fmax_277 = fmaxf(amax_35 * 0.002232142857f, 1e-12f);
                                float scale_36 = _fmax_277;
                                uint16_t _ue8m0x2_f32_21;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_21) : "f"(scale_36), "f"(scale_36));
                                unsigned int scale_byte_37 = (unsigned int)_ue8m0x2_f32_21 & 255;
                                unsigned int inverse_lane_38 = 254 - scale_byte_37 << 7;
                                unsigned int inverse_39 = inverse_lane_38 | inverse_lane_38 << 16;
                                unsigned int words_40[8];
                                #pragma unroll
                                for (int i_44 = 0; i_44 < 8; i_44++) {
                                    uint32_t _bf16x2_mul_42;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(block_33[i_44 * 2]), "r"(inverse_39));
                                    uint16_t _e4m3x2_42;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_42) : "r"(_bf16x2_mul_42));
                                    uint32_t _bf16x2_mul_43;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(block_33[i_44 * 2 + 1]), "r"(inverse_39));
                                    uint16_t _e4m3x2_43;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_43) : "r"(_bf16x2_mul_43));
                                    words_40[i_44] = (unsigned int)_e4m3x2_42 | (unsigned int)_e4m3x2_43 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_37 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_40[0]), "r"(words_40[1]), "r"(words_40[2]), "r"(words_40[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_40[4]), "r"(words_40[5]), "r"(words_40[6]), "r"(words_40[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 160), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_41[16];
                                #pragma unroll
                                for (int j_29 = 0; j_29 < 16; j_29++) {
                                    block_41[j_29] = packed_1[96 + j_29];
                                }
                                #pragma unroll
                                for (int j_30 = 0; j_30 < 4; j_30++) {
                                    unsigned int address_14 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_30 * 16);
                                    address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_14 - smem_v50_addr)), "r"(packed_1[96 + 4 * j_30]), "r"(packed_1[96 + 4 * j_30 + 1]), "r"(packed_1[96 + 4 * j_30 + 2]), "r"(packed_1[96 + 4 * j_30 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_44;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(block_41[0]));
                                unsigned int amax_pair_42 = _bf16x2_abs_44;
                                #pragma unroll
                                for (int i_45 = 1; i_45 < 16; i_45++) {
                                    uint32_t _bf16x2_abs_45;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(block_41[i_45]));
                                    uint32_t _bf16x2_max_22;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(amax_pair_42), "r"(_bf16x2_abs_45));
                                    amax_pair_42 = _bf16x2_max_22;
                                }
                                uint16_t _bf16_max_22;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_22) : "h"((uint16_t)(amax_pair_42 & 65535)), "h"((uint16_t)(amax_pair_42 >> 16)));
                                float _cvt_f32_bf16_22;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(_bf16_max_22)));
                                float amax_43 = _cvt_f32_bf16_22;
                                float _fmax_278 = fmaxf(amax_43 * 0.002232142857f, 1e-12f);
                                float scale_44 = _fmax_278;
                                uint16_t _ue8m0x2_f32_22;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_22) : "f"(scale_44), "f"(scale_44));
                                unsigned int scale_byte_45 = (unsigned int)_ue8m0x2_f32_22 & 255;
                                unsigned int inverse_lane_46 = 254 - scale_byte_45 << 7;
                                unsigned int inverse_47 = inverse_lane_46 | inverse_lane_46 << 16;
                                unsigned int words_48[8];
                                #pragma unroll
                                for (int i_46 = 0; i_46 < 8; i_46++) {
                                    uint32_t _bf16x2_mul_44;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(block_41[i_46 * 2]), "r"(inverse_47));
                                    uint16_t _e4m3x2_44;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_44) : "r"(_bf16x2_mul_44));
                                    uint32_t _bf16x2_mul_45;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(block_41[i_46 * 2 + 1]), "r"(inverse_47));
                                    uint16_t _e4m3x2_45;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_45) : "r"(_bf16x2_mul_45));
                                    words_48[i_46] = (unsigned int)_e4m3x2_44 | (unsigned int)_e4m3x2_45 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_45 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_48[0]), "r"(words_48[1]), "r"(words_48[2]), "r"(words_48[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_48[4]), "r"(words_48[5]), "r"(words_48[6]), "r"(words_48[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 192), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_49[16];
                                #pragma unroll
                                for (int j_31 = 0; j_31 < 16; j_31++) {
                                    block_49[j_31] = packed_1[112 + j_31];
                                }
                                #pragma unroll
                                for (int j_32 = 0; j_32 < 4; j_32++) {
                                    unsigned int address_15 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_32 * 16);
                                    address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_15 - smem_v50_addr)), "r"(packed_1[112 + 4 * j_32]), "r"(packed_1[112 + 4 * j_32 + 1]), "r"(packed_1[112 + 4 * j_32 + 2]), "r"(packed_1[112 + 4 * j_32 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 7), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_46;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(block_49[0]));
                                unsigned int amax_pair_50 = _bf16x2_abs_46;
                                #pragma unroll
                                for (int i_47 = 1; i_47 < 16; i_47++) {
                                    uint32_t _bf16x2_abs_47;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(block_49[i_47]));
                                    uint32_t _bf16x2_max_23;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(amax_pair_50), "r"(_bf16x2_abs_47));
                                    amax_pair_50 = _bf16x2_max_23;
                                }
                                uint16_t _bf16_max_23;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_23) : "h"((uint16_t)(amax_pair_50 & 65535)), "h"((uint16_t)(amax_pair_50 >> 16)));
                                float _cvt_f32_bf16_23;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(_bf16_max_23)));
                                float amax_51 = _cvt_f32_bf16_23;
                                float _fmax_279 = fmaxf(amax_51 * 0.002232142857f, 1e-12f);
                                float scale_52 = _fmax_279;
                                uint16_t _ue8m0x2_f32_23;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_23) : "f"(scale_52), "f"(scale_52));
                                unsigned int scale_byte_53 = (unsigned int)_ue8m0x2_f32_23 & 255;
                                unsigned int inverse_lane_54 = 254 - scale_byte_53 << 7;
                                unsigned int inverse_55 = inverse_lane_54 | inverse_lane_54 << 16;
                                unsigned int words_56[8];
                                #pragma unroll
                                for (int i_48 = 0; i_48 < 8; i_48++) {
                                    uint32_t _bf16x2_mul_46;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(block_49[i_48 * 2]), "r"(inverse_55));
                                    uint16_t _e4m3x2_46;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_46) : "r"(_bf16x2_mul_46));
                                    uint32_t _bf16x2_mul_47;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(block_49[i_48 * 2 + 1]), "r"(inverse_55));
                                    uint16_t _e4m3x2_47;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_47) : "r"(_bf16x2_mul_47));
                                    words_56[i_48] = (unsigned int)_e4m3x2_46 | (unsigned int)_e4m3x2_47 << 16;
                                }
                                scale_word_16 = scale_word_16 | scale_byte_53 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_56[0]), "r"(words_56[1]), "r"(words_56[2]), "r"(words_56[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_56[4]), "r"(words_56[5]), "r"(words_56[6]), "r"(words_56[7]) : "memory");
                                smem_v53[tid % 32 * 4 + tid / 32] = scale_word_16;
                                scale_word_16 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 224), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2 + 1, smem_v53_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                #pragma unroll
                                for (int i_49 = 0; i_49 < 8; i_49++) {
                                    float _tmem_load_35[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_35[0]), "=f"(_tmem_load_35[1]), "=f"(_tmem_load_35[2]), "=f"(_tmem_load_35[3]), "=f"(_tmem_load_35[4]), "=f"(_tmem_load_35[5]), "=f"(_tmem_load_35[6]), "=f"(_tmem_load_35[7]), "=f"(_tmem_load_35[8]), "=f"(_tmem_load_35[9]), "=f"(_tmem_load_35[10]), "=f"(_tmem_load_35[11]), "=f"(_tmem_load_35[12]), "=f"(_tmem_load_35[13]), "=f"(_tmem_load_35[14]), "=f"(_tmem_load_35[15]), "=f"(_tmem_load_35[16]), "=f"(_tmem_load_35[17]), "=f"(_tmem_load_35[18]), "=f"(_tmem_load_35[19]), "=f"(_tmem_load_35[20]), "=f"(_tmem_load_35[21]), "=f"(_tmem_load_35[22]), "=f"(_tmem_load_35[23]), "=f"(_tmem_load_35[24]), "=f"(_tmem_load_35[25]), "=f"(_tmem_load_35[26]), "=f"(_tmem_load_35[27]), "=f"(_tmem_load_35[28]), "=f"(_tmem_load_35[29]), "=f"(_tmem_load_35[30]), "=f"(_tmem_load_35[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row << 16) + 256 + (unsigned int)(i_49 * 32)));
                                    #pragma unroll
                                    for (int j_33 = 0; j_33 < 16; j_33++) {
                                        __nv_bfloat162 _bf16x2_395 = __float22bfloat162_rn(make_float2(_tmem_load_35[2 * j_33], _tmem_load_35[2 * j_33 + 1]));
                                        packed_1[i_49 * 16 + j_33] = __as_u32(_bf16x2_395);
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                                unsigned int scale_word_57 = 0;
                                unsigned int block_58[16];
                                #pragma unroll
                                for (int j_34 = 0; j_34 < 16; j_34++) {
                                    block_58[j_34] = packed_1[j_34];
                                }
                                #pragma unroll
                                for (int j_35 = 0; j_35 < 4; j_35++) {
                                    unsigned int address_16 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_35 * 16);
                                    address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_16 - smem_v50_addr)), "r"(packed_1[4 * j_35]), "r"(packed_1[4 * j_35 + 1]), "r"(packed_1[4 * j_35 + 2]), "r"(packed_1[4 * j_35 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_48;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(block_58[0]));
                                unsigned int amax_pair_59 = _bf16x2_abs_48;
                                #pragma unroll
                                for (int i_50 = 1; i_50 < 16; i_50++) {
                                    uint32_t _bf16x2_abs_49;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(block_58[i_50]));
                                    uint32_t _bf16x2_max_24;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(amax_pair_59), "r"(_bf16x2_abs_49));
                                    amax_pair_59 = _bf16x2_max_24;
                                }
                                uint16_t _bf16_max_24;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_24) : "h"((uint16_t)(amax_pair_59 & 65535)), "h"((uint16_t)(amax_pair_59 >> 16)));
                                float _cvt_f32_bf16_24;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(_bf16_max_24)));
                                float amax_60 = _cvt_f32_bf16_24;
                                float _fmax_280 = fmaxf(amax_60 * 0.002232142857f, 1e-12f);
                                float scale_61 = _fmax_280;
                                uint16_t _ue8m0x2_f32_24;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_24) : "f"(scale_61), "f"(scale_61));
                                unsigned int scale_byte_62 = (unsigned int)_ue8m0x2_f32_24 & 255;
                                unsigned int inverse_lane_63 = 254 - scale_byte_62 << 7;
                                unsigned int inverse_64 = inverse_lane_63 | inverse_lane_63 << 16;
                                unsigned int words_65[8];
                                #pragma unroll
                                for (int i_51 = 0; i_51 < 8; i_51++) {
                                    uint32_t _bf16x2_mul_48;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(block_58[i_51 * 2]), "r"(inverse_64));
                                    uint16_t _e4m3x2_48;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_48) : "r"(_bf16x2_mul_48));
                                    uint32_t _bf16x2_mul_49;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(block_58[i_51 * 2 + 1]), "r"(inverse_64));
                                    uint16_t _e4m3x2_49;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_49) : "r"(_bf16x2_mul_49));
                                    words_65[i_51] = (unsigned int)_e4m3x2_48 | (unsigned int)_e4m3x2_49 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_62;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_65[0]), "r"(words_65[1]), "r"(words_65[2]), "r"(words_65[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_65[4]), "r"(words_65[5]), "r"(words_65[6]), "r"(words_65[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_66[16];
                                #pragma unroll
                                for (int j_36 = 0; j_36 < 16; j_36++) {
                                    block_66[j_36] = packed_1[16 + j_36];
                                }
                                #pragma unroll
                                for (int j_37 = 0; j_37 < 4; j_37++) {
                                    unsigned int address_17 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_37 * 16);
                                    address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_17 - smem_v50_addr)), "r"(packed_1[16 + 4 * j_37]), "r"(packed_1[16 + 4 * j_37 + 1]), "r"(packed_1[16 + 4 * j_37 + 2]), "r"(packed_1[16 + 4 * j_37 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_50;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(block_66[0]));
                                unsigned int amax_pair_67 = _bf16x2_abs_50;
                                #pragma unroll
                                for (int i_52 = 1; i_52 < 16; i_52++) {
                                    uint32_t _bf16x2_abs_51;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(block_66[i_52]));
                                    uint32_t _bf16x2_max_25;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(amax_pair_67), "r"(_bf16x2_abs_51));
                                    amax_pair_67 = _bf16x2_max_25;
                                }
                                uint16_t _bf16_max_25;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_25) : "h"((uint16_t)(amax_pair_67 & 65535)), "h"((uint16_t)(amax_pair_67 >> 16)));
                                float _cvt_f32_bf16_25;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(_bf16_max_25)));
                                float amax_68 = _cvt_f32_bf16_25;
                                float _fmax_281 = fmaxf(amax_68 * 0.002232142857f, 1e-12f);
                                float scale_69 = _fmax_281;
                                uint16_t _ue8m0x2_f32_25;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_25) : "f"(scale_69), "f"(scale_69));
                                unsigned int scale_byte_70 = (unsigned int)_ue8m0x2_f32_25 & 255;
                                unsigned int inverse_lane_71 = 254 - scale_byte_70 << 7;
                                unsigned int inverse_72 = inverse_lane_71 | inverse_lane_71 << 16;
                                unsigned int words_73[8];
                                #pragma unroll
                                for (int i_53 = 0; i_53 < 8; i_53++) {
                                    uint32_t _bf16x2_mul_50;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(block_66[i_53 * 2]), "r"(inverse_72));
                                    uint16_t _e4m3x2_50;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_50) : "r"(_bf16x2_mul_50));
                                    uint32_t _bf16x2_mul_51;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(block_66[i_53 * 2 + 1]), "r"(inverse_72));
                                    uint16_t _e4m3x2_51;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_51) : "r"(_bf16x2_mul_51));
                                    words_73[i_53] = (unsigned int)_e4m3x2_50 | (unsigned int)_e4m3x2_51 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_70 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_73[0]), "r"(words_73[1]), "r"(words_73[2]), "r"(words_73[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_73[4]), "r"(words_73[5]), "r"(words_73[6]), "r"(words_73[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 32), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_74[16];
                                #pragma unroll
                                for (int j_38 = 0; j_38 < 16; j_38++) {
                                    block_74[j_38] = packed_1[32 + j_38];
                                }
                                #pragma unroll
                                for (int j_39 = 0; j_39 < 4; j_39++) {
                                    unsigned int address_18 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_39 * 16);
                                    address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_18 - smem_v50_addr)), "r"(packed_1[32 + 4 * j_39]), "r"(packed_1[32 + 4 * j_39 + 1]), "r"(packed_1[32 + 4 * j_39 + 2]), "r"(packed_1[32 + 4 * j_39 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_52;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(block_74[0]));
                                unsigned int amax_pair_75 = _bf16x2_abs_52;
                                #pragma unroll
                                for (int i_54 = 1; i_54 < 16; i_54++) {
                                    uint32_t _bf16x2_abs_53;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(block_74[i_54]));
                                    uint32_t _bf16x2_max_26;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(amax_pair_75), "r"(_bf16x2_abs_53));
                                    amax_pair_75 = _bf16x2_max_26;
                                }
                                uint16_t _bf16_max_26;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_26) : "h"((uint16_t)(amax_pair_75 & 65535)), "h"((uint16_t)(amax_pair_75 >> 16)));
                                float _cvt_f32_bf16_26;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(_bf16_max_26)));
                                float amax_76 = _cvt_f32_bf16_26;
                                float _fmax_282 = fmaxf(amax_76 * 0.002232142857f, 1e-12f);
                                float scale_77 = _fmax_282;
                                uint16_t _ue8m0x2_f32_26;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_26) : "f"(scale_77), "f"(scale_77));
                                unsigned int scale_byte_78 = (unsigned int)_ue8m0x2_f32_26 & 255;
                                unsigned int inverse_lane_79 = 254 - scale_byte_78 << 7;
                                unsigned int inverse_80 = inverse_lane_79 | inverse_lane_79 << 16;
                                unsigned int words_81[8];
                                #pragma unroll
                                for (int i_55 = 0; i_55 < 8; i_55++) {
                                    uint32_t _bf16x2_mul_52;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(block_74[i_55 * 2]), "r"(inverse_80));
                                    uint16_t _e4m3x2_52;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_52) : "r"(_bf16x2_mul_52));
                                    uint32_t _bf16x2_mul_53;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(block_74[i_55 * 2 + 1]), "r"(inverse_80));
                                    uint16_t _e4m3x2_53;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_53) : "r"(_bf16x2_mul_53));
                                    words_81[i_55] = (unsigned int)_e4m3x2_52 | (unsigned int)_e4m3x2_53 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_78 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_81[0]), "r"(words_81[1]), "r"(words_81[2]), "r"(words_81[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_81[4]), "r"(words_81[5]), "r"(words_81[6]), "r"(words_81[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 64), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_82[16];
                                #pragma unroll
                                for (int j_40 = 0; j_40 < 16; j_40++) {
                                    block_82[j_40] = packed_1[48 + j_40];
                                }
                                #pragma unroll
                                for (int j_41 = 0; j_41 < 4; j_41++) {
                                    unsigned int address_19 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_41 * 16);
                                    address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_19 - smem_v50_addr)), "r"(packed_1[48 + 4 * j_41]), "r"(packed_1[48 + 4 * j_41 + 1]), "r"(packed_1[48 + 4 * j_41 + 2]), "r"(packed_1[48 + 4 * j_41 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 3), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_54;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(block_82[0]));
                                unsigned int amax_pair_83 = _bf16x2_abs_54;
                                #pragma unroll
                                for (int i_56 = 1; i_56 < 16; i_56++) {
                                    uint32_t _bf16x2_abs_55;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(block_82[i_56]));
                                    uint32_t _bf16x2_max_27;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(amax_pair_83), "r"(_bf16x2_abs_55));
                                    amax_pair_83 = _bf16x2_max_27;
                                }
                                uint16_t _bf16_max_27;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_27) : "h"((uint16_t)(amax_pair_83 & 65535)), "h"((uint16_t)(amax_pair_83 >> 16)));
                                float _cvt_f32_bf16_27;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_27) : "h"((uint16_t)(_bf16_max_27)));
                                float amax_84 = _cvt_f32_bf16_27;
                                float _fmax_283 = fmaxf(amax_84 * 0.002232142857f, 1e-12f);
                                float scale_85 = _fmax_283;
                                uint16_t _ue8m0x2_f32_27;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_27) : "f"(scale_85), "f"(scale_85));
                                unsigned int scale_byte_86 = (unsigned int)_ue8m0x2_f32_27 & 255;
                                unsigned int inverse_lane_87 = 254 - scale_byte_86 << 7;
                                unsigned int inverse_88 = inverse_lane_87 | inverse_lane_87 << 16;
                                unsigned int words_89[8];
                                #pragma unroll
                                for (int i_57 = 0; i_57 < 8; i_57++) {
                                    uint32_t _bf16x2_mul_54;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(block_82[i_57 * 2]), "r"(inverse_88));
                                    uint16_t _e4m3x2_54;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_54) : "r"(_bf16x2_mul_54));
                                    uint32_t _bf16x2_mul_55;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(block_82[i_57 * 2 + 1]), "r"(inverse_88));
                                    uint16_t _e4m3x2_55;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_55) : "r"(_bf16x2_mul_55));
                                    words_89[i_57] = (unsigned int)_e4m3x2_54 | (unsigned int)_e4m3x2_55 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_86 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_89[0]), "r"(words_89[1]), "r"(words_89[2]), "r"(words_89[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_89[4]), "r"(words_89[5]), "r"(words_89[6]), "r"(words_89[7]) : "memory");
                                smem_v52[tid % 32 * 4 + tid / 32] = scale_word_57;
                                scale_word_57 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 96), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2, smem_v52_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_90[16];
                                #pragma unroll
                                for (int j_42 = 0; j_42 < 16; j_42++) {
                                    block_90[j_42] = packed_1[64 + j_42];
                                }
                                #pragma unroll
                                for (int j_43 = 0; j_43 < 4; j_43++) {
                                    unsigned int address_20 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_43 * 16);
                                    address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_20 - smem_v50_addr)), "r"(packed_1[64 + 4 * j_43]), "r"(packed_1[64 + 4 * j_43 + 1]), "r"(packed_1[64 + 4 * j_43 + 2]), "r"(packed_1[64 + 4 * j_43 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_56;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(block_90[0]));
                                unsigned int amax_pair_91 = _bf16x2_abs_56;
                                #pragma unroll
                                for (int i_58 = 1; i_58 < 16; i_58++) {
                                    uint32_t _bf16x2_abs_57;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(block_90[i_58]));
                                    uint32_t _bf16x2_max_28;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(amax_pair_91), "r"(_bf16x2_abs_57));
                                    amax_pair_91 = _bf16x2_max_28;
                                }
                                uint16_t _bf16_max_28;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_28) : "h"((uint16_t)(amax_pair_91 & 65535)), "h"((uint16_t)(amax_pair_91 >> 16)));
                                float _cvt_f32_bf16_28;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(_bf16_max_28)));
                                float amax_92 = _cvt_f32_bf16_28;
                                float _fmax_284 = fmaxf(amax_92 * 0.002232142857f, 1e-12f);
                                float scale_93 = _fmax_284;
                                uint16_t _ue8m0x2_f32_28;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_28) : "f"(scale_93), "f"(scale_93));
                                unsigned int scale_byte_94 = (unsigned int)_ue8m0x2_f32_28 & 255;
                                unsigned int inverse_lane_95 = 254 - scale_byte_94 << 7;
                                unsigned int inverse_96 = inverse_lane_95 | inverse_lane_95 << 16;
                                unsigned int words_97[8];
                                #pragma unroll
                                for (int i_59 = 0; i_59 < 8; i_59++) {
                                    uint32_t _bf16x2_mul_56;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(block_90[i_59 * 2]), "r"(inverse_96));
                                    uint16_t _e4m3x2_56;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_56) : "r"(_bf16x2_mul_56));
                                    uint32_t _bf16x2_mul_57;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(block_90[i_59 * 2 + 1]), "r"(inverse_96));
                                    uint16_t _e4m3x2_57;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_57) : "r"(_bf16x2_mul_57));
                                    words_97[i_59] = (unsigned int)_e4m3x2_56 | (unsigned int)_e4m3x2_57 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_94;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_97[0]), "r"(words_97[1]), "r"(words_97[2]), "r"(words_97[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_97[4]), "r"(words_97[5]), "r"(words_97[6]), "r"(words_97[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 128), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_98[16];
                                #pragma unroll
                                for (int j_44 = 0; j_44 < 16; j_44++) {
                                    block_98[j_44] = packed_1[80 + j_44];
                                }
                                #pragma unroll
                                for (int j_45 = 0; j_45 < 4; j_45++) {
                                    unsigned int address_21 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_45 * 16);
                                    address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_21 - smem_v50_addr)), "r"(packed_1[80 + 4 * j_45]), "r"(packed_1[80 + 4 * j_45 + 1]), "r"(packed_1[80 + 4 * j_45 + 2]), "r"(packed_1[80 + 4 * j_45 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 5), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_58;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(block_98[0]));
                                unsigned int amax_pair_99 = _bf16x2_abs_58;
                                #pragma unroll
                                for (int i_60 = 1; i_60 < 16; i_60++) {
                                    uint32_t _bf16x2_abs_59;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(block_98[i_60]));
                                    uint32_t _bf16x2_max_29;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(amax_pair_99), "r"(_bf16x2_abs_59));
                                    amax_pair_99 = _bf16x2_max_29;
                                }
                                uint16_t _bf16_max_29;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_29) : "h"((uint16_t)(amax_pair_99 & 65535)), "h"((uint16_t)(amax_pair_99 >> 16)));
                                float _cvt_f32_bf16_29;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(_bf16_max_29)));
                                float amax_100 = _cvt_f32_bf16_29;
                                float _fmax_285 = fmaxf(amax_100 * 0.002232142857f, 1e-12f);
                                float scale_101 = _fmax_285;
                                uint16_t _ue8m0x2_f32_29;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_29) : "f"(scale_101), "f"(scale_101));
                                unsigned int scale_byte_102 = (unsigned int)_ue8m0x2_f32_29 & 255;
                                unsigned int inverse_lane_103 = 254 - scale_byte_102 << 7;
                                unsigned int inverse_104 = inverse_lane_103 | inverse_lane_103 << 16;
                                unsigned int words_105[8];
                                #pragma unroll
                                for (int i_61 = 0; i_61 < 8; i_61++) {
                                    uint32_t _bf16x2_mul_58;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(block_98[i_61 * 2]), "r"(inverse_104));
                                    uint16_t _e4m3x2_58;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_58) : "r"(_bf16x2_mul_58));
                                    uint32_t _bf16x2_mul_59;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(block_98[i_61 * 2 + 1]), "r"(inverse_104));
                                    uint16_t _e4m3x2_59;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_59) : "r"(_bf16x2_mul_59));
                                    words_105[i_61] = (unsigned int)_e4m3x2_58 | (unsigned int)_e4m3x2_59 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_102 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_105[0]), "r"(words_105[1]), "r"(words_105[2]), "r"(words_105[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_105[4]), "r"(words_105[5]), "r"(words_105[6]), "r"(words_105[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 160), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_106[16];
                                #pragma unroll
                                for (int j_46 = 0; j_46 < 16; j_46++) {
                                    block_106[j_46] = packed_1[96 + j_46];
                                }
                                #pragma unroll
                                for (int j_47 = 0; j_47 < 4; j_47++) {
                                    unsigned int address_22 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_47 * 16);
                                    address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_22 - smem_v50_addr)), "r"(packed_1[96 + 4 * j_47]), "r"(packed_1[96 + 4 * j_47 + 1]), "r"(packed_1[96 + 4 * j_47 + 2]), "r"(packed_1[96 + 4 * j_47 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_60;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_60) : "r"(block_106[0]));
                                unsigned int amax_pair_107 = _bf16x2_abs_60;
                                #pragma unroll
                                for (int i_62 = 1; i_62 < 16; i_62++) {
                                    uint32_t _bf16x2_abs_61;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_61) : "r"(block_106[i_62]));
                                    uint32_t _bf16x2_max_30;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_30) : "r"(amax_pair_107), "r"(_bf16x2_abs_61));
                                    amax_pair_107 = _bf16x2_max_30;
                                }
                                uint16_t _bf16_max_30;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_30) : "h"((uint16_t)(amax_pair_107 & 65535)), "h"((uint16_t)(amax_pair_107 >> 16)));
                                float _cvt_f32_bf16_30;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_30) : "h"((uint16_t)(_bf16_max_30)));
                                float amax_108 = _cvt_f32_bf16_30;
                                float _fmax_286 = fmaxf(amax_108 * 0.002232142857f, 1e-12f);
                                float scale_109 = _fmax_286;
                                uint16_t _ue8m0x2_f32_30;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_30) : "f"(scale_109), "f"(scale_109));
                                unsigned int scale_byte_110 = (unsigned int)_ue8m0x2_f32_30 & 255;
                                unsigned int inverse_lane_111 = 254 - scale_byte_110 << 7;
                                unsigned int inverse_112 = inverse_lane_111 | inverse_lane_111 << 16;
                                unsigned int words_113[8];
                                #pragma unroll
                                for (int i_63 = 0; i_63 < 8; i_63++) {
                                    uint32_t _bf16x2_mul_60;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(block_106[i_63 * 2]), "r"(inverse_112));
                                    uint16_t _e4m3x2_60;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_60) : "r"(_bf16x2_mul_60));
                                    uint32_t _bf16x2_mul_61;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(block_106[i_63 * 2 + 1]), "r"(inverse_112));
                                    uint16_t _e4m3x2_61;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_61) : "r"(_bf16x2_mul_61));
                                    words_113[i_63] = (unsigned int)_e4m3x2_60 | (unsigned int)_e4m3x2_61 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_110 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_113[0]), "r"(words_113[1]), "r"(words_113[2]), "r"(words_113[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_113[4]), "r"(words_113[5]), "r"(words_113[6]), "r"(words_113[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 192), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_114[16];
                                #pragma unroll
                                for (int j_48 = 0; j_48 < 16; j_48++) {
                                    block_114[j_48] = packed_1[112 + j_48];
                                }
                                #pragma unroll
                                for (int j_49 = 0; j_49 < 4; j_49++) {
                                    unsigned int address_23 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_49 * 16);
                                    address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (address_23 - smem_v50_addr)), "r"(packed_1[112 + 4 * j_49]), "r"(packed_1[112 + 4 * j_49 + 1]), "r"(packed_1[112 + 4 * j_49 + 2]), "r"(packed_1[112 + 4 * j_49 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 7), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t _bf16x2_abs_62;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_62) : "r"(block_114[0]));
                                unsigned int amax_pair_115 = _bf16x2_abs_62;
                                #pragma unroll
                                for (int i_64 = 1; i_64 < 16; i_64++) {
                                    uint32_t _bf16x2_abs_63;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_63) : "r"(block_114[i_64]));
                                    uint32_t _bf16x2_max_31;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_31) : "r"(amax_pair_115), "r"(_bf16x2_abs_63));
                                    amax_pair_115 = _bf16x2_max_31;
                                }
                                uint16_t _bf16_max_31;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_31) : "h"((uint16_t)(amax_pair_115 & 65535)), "h"((uint16_t)(amax_pair_115 >> 16)));
                                float _cvt_f32_bf16_31;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_31) : "h"((uint16_t)(_bf16_max_31)));
                                float amax_116 = _cvt_f32_bf16_31;
                                float _fmax_287 = fmaxf(amax_116 * 0.002232142857f, 1e-12f);
                                float scale_117 = _fmax_287;
                                uint16_t _ue8m0x2_f32_31;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_31) : "f"(scale_117), "f"(scale_117));
                                unsigned int scale_byte_118 = (unsigned int)_ue8m0x2_f32_31 & 255;
                                unsigned int inverse_lane_119 = 254 - scale_byte_118 << 7;
                                unsigned int inverse_120 = inverse_lane_119 | inverse_lane_119 << 16;
                                unsigned int words_121[8];
                                #pragma unroll
                                for (int i_65 = 0; i_65 < 8; i_65++) {
                                    uint32_t _bf16x2_mul_62;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(block_114[i_65 * 2]), "r"(inverse_120));
                                    uint16_t _e4m3x2_62;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_62) : "r"(_bf16x2_mul_62));
                                    uint32_t _bf16x2_mul_63;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(block_114[i_65 * 2 + 1]), "r"(inverse_120));
                                    uint16_t _e4m3x2_63;
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_63) : "r"(_bf16x2_mul_63));
                                    words_121[i_65] = (unsigned int)_e4m3x2_62 | (unsigned int)_e4m3x2_63 << 16;
                                }
                                scale_word_57 = scale_word_57 | scale_byte_118 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32)), "r"(words_121[0]), "r"(words_121[1]), "r"(words_121[2]), "r"(words_121[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v51_addr + (unsigned int)(tid * 32 + 16)), "r"(words_121[4]), "r"(words_121[5]), "r"(words_121[6]), "r"(words_121[7]) : "memory");
                                smem_v53[tid % 32 * 4 + tid / 32] = scale_word_57;
                                scale_word_57 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 224), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v51_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2 + 1, smem_v53_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                phase_bits_4 = phase_bits_4 ^ 64;
                                if (tid / 32 == 0) {
                                    if (warp == 0) {
                                        if (elect_sync()) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (gate_base + (macro_rows_3 + x_2) * (intermediate / 256) + y_2))), "r"(static_cast<unsigned int>(2)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_bits = phase_bits_4;
                } else if (task_2 < mini_gate + mini_swiglu) {
                    unsigned int phase_bits_5 = swiglu_bits;
                    int col_blocks_6 = intermediate / 128;
                    int num_tiles = tokens / 128 * col_blocks_6;
                    int macro_row_offset = macro_1 * (macro_size / 128);
                    int first_tile_2 = (task_2 - mini_gate) * 6 + cta_rank_0 * 3;
                    int global_mini_3 = macro_1 * (macro_size / mini_size) + mini_2;
                    int mini_tiles = mini_size / 128 * col_blocks_6;
                    first_tile_2 = first_tile_2 + global_mini_3 * mini_tiles;
                    int _min_544 = ((num_tiles) < ((global_mini_3 + 1) * mini_tiles) ? (num_tiles) : ((global_mini_3 + 1) * mini_tiles));
                    int tile_end = _min_544;
                    int macro_tiles_2 = macro_size / 128;
                    if (first_tile_2 < tile_end) {
                        int first_row_3 = first_tile_2 / col_blocks_6;
                        int first_col_2 = first_tile_2 % col_blocks_6;
                        if (tid == 0) {
                            if (tile_end > first_tile_2) {
                                int row_33 = first_row_3;
                                int col_61 = first_col_2;
                                if (col_61 >= col_blocks_6) {
                                    row_33 = row_33 + 1;
                                    col_61 = col_61 - col_blocks_6;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr, 65536);
                                int parent = row_33 / 2 * (intermediate / 256) + col_61 / 2;
                                int32_t _relaxed_ld_12;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(gate_ready + (gate_base + parent)) : "memory");
                                int value_6 = _relaxed_ld_12;
                                while (value_6 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_13;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(gate_ready + (gate_base + parent)) : "memory");
                                    value_6 = _relaxed_ld_13;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr), "l"((&gate_routed_in)), "r"(0), "r"((row_33 - macro_row_offset) * 128), "r"(col_61 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr), "l"((&up_routed_in)), "r"(0), "r"((row_33 - macro_row_offset) * 128), "r"(col_61 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr) : "memory");
                            }
                            if (tile_end > first_tile_2 + 1) {
                                int row_34 = first_row_3;
                                int col_62 = first_col_2 + 1;
                                if (col_62 >= col_blocks_6) {
                                    row_34 = row_34 + 1;
                                    col_62 = col_62 - col_blocks_6;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + 8, 65536);
                                int parent_1 = row_34 / 2 * (intermediate / 256) + col_62 / 2;
                                int32_t _relaxed_ld_14;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(gate_ready + (gate_base + parent_1)) : "memory");
                                int value_7 = _relaxed_ld_14;
                                while (value_7 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_15;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(gate_ready + (gate_base + parent_1)) : "memory");
                                    value_7 = _relaxed_ld_15;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + 32768), "l"((&gate_routed_in)), "r"(0), "r"((row_34 - macro_row_offset) * 128), "r"(col_62 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + 32768), "l"((&up_routed_in)), "r"(0), "r"((row_34 - macro_row_offset) * 128), "r"(col_62 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 8) : "memory");
                            }
                            if (tile_end > first_tile_2 + 2) {
                                int row_35 = first_row_3;
                                int col_63 = first_col_2 + 2;
                                if (col_63 >= col_blocks_6) {
                                    row_35 = row_35 + 1;
                                    col_63 = col_63 - col_blocks_6;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + 16, 65536);
                                int parent_2 = row_35 / 2 * (intermediate / 256) + col_63 / 2;
                                int32_t _relaxed_ld_16;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_16) : "l"(gate_ready + (gate_base + parent_2)) : "memory");
                                int value_8 = _relaxed_ld_16;
                                while (value_8 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_17;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_17) : "l"(gate_ready + (gate_base + parent_2)) : "memory");
                                    value_8 = _relaxed_ld_17;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + 65536), "l"((&gate_routed_in)), "r"(0), "r"((row_35 - macro_row_offset) * 128), "r"(col_63 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 16) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + 65536), "l"((&up_routed_in)), "r"(0), "r"((row_35 - macro_row_offset) * 128), "r"(col_63 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 16) : "memory");
                            }
                        }
                        if (tile_end > first_tile_2) {
                            mbarrier_wait(swiglu_arrived_addr, phase_bits_5 & 1);
                            phase_bits_5 = phase_bits_5 ^ 1;
                            int row_36 = first_row_3;
                            int col_64 = first_col_2;
                            if (col_64 >= col_blocks_6) {
                                row_36 = row_36 + 1;
                                col_64 = col_64 - col_blocks_6;
                            }
                            float gate[64];
                            float up[64];
                            float denominator[64];
                            int warp_0_2 = tid / 32;
                            int local_warp = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                            int lane_4 = tid % 32;
                            #pragma unroll
                            for (int tile_col = 0; tile_col < 8; tile_col++) {
                                unsigned int packed_2[4];
                                unsigned int address_24 = gate_smem_addr + (unsigned int)(((tile_col * 16 + lane_4 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_4 % 16) * 64 + (tile_col * 16 + lane_4 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                    : "r"(address_24 ^ (address_24 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                    float2 _cvt_f32_256 = __bfloat1622float2(__as_bf16x2(packed_2[pair_2]));
                                    gate[tile_col * 8 + pair_2 * 2] = _cvt_f32_256.x;
                                    gate[tile_col * 8 + pair_2 * 2 + 1] = _cvt_f32_256.y;
                                }
                            }
                            int warp_1_1 = tid / 32;
                            int local_warp_2 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                            int lane_3_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                unsigned int packed_3[4];
                                unsigned int address_25 = up_smem_addr + (unsigned int)(((tile_col_1 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3_1 % 16) * 64 + (tile_col_1 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                    : "r"(address_25 ^ (address_25 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                    float2 _cvt_f32_257 = __bfloat1622float2(__as_bf16x2(packed_3[pair_3]));
                                    up[tile_col_1 * 8 + pair_3 * 2] = _cvt_f32_257.x;
                                    up[tile_col_1 * 8 + pair_3 * 2 + 1] = _cvt_f32_257.y;
                                }
                            }
                            #pragma unroll
                            for (int elem = 0; elem < 64; elem++) {
                                float _min_545 = fminf(gate[elem], swiglu_limit);
                                gate[elem] = _min_545;
                            }
                            #pragma unroll
                            for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                float _fmax_288 = fmaxf(up[elem_1], -swiglu_limit);
                                float _min_546 = fminf(_fmax_288, swiglu_limit);
                                up[elem_1] = _min_546;
                            }
                            #pragma unroll
                            for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                denominator[elem_2] = gate[elem_2] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                float _exp_256 = expf(denominator[elem_3]);
                                denominator[elem_3] = _exp_256;
                            }
                            #pragma unroll
                            for (int elem_4 = 0; elem_4 < 64; elem_4++) {
                                denominator[elem_4] = denominator[elem_4] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                gate[elem_5] = gate[elem_5] / denominator[elem_5];
                            }
                            #pragma unroll
                            for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                gate[elem_6] = gate[elem_6] * up[elem_6];
                            }
                            __syncthreads();
                            int warp_4 = tid / 32;
                            int local_warp_5 = warp_4 / 4 + warp_4 % 4 * 2;
                            int lane_6 = tid % 32;
                            #pragma unroll
                            for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                unsigned int packed_4[4];
                                #pragma unroll
                                for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                    __nv_bfloat162 _bf16x2_396 = __float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_4 * 2], gate[tile_col_2 * 8 + pair_4 * 2 + 1]));
                                    packed_4[pair_4] = __as_u32(_bf16x2_396);
                                }
                                int row_0_16 = local_warp_5 * 16 + lane_6 % 16;
                                int col_1_1 = tile_col_2 * 16 + lane_6 / 16 * 8;
                                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_16 * 136 + col_1_1) * 2));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_17 = tid;
                                row_0_17 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_17 = 0;
                                #pragma unroll 1
                                for (int j_50 = 0; j_50 < 4; j_50++) {
                                    int k_block_16 = (j_50 + tid / 8) % 4;
                                    unsigned int pairs_16[16];
                                    #pragma unroll
                                    for (int k_32 = 0; k_32 < 16; k_32++) {
                                        int col_0 = k_block_16 * 32 + (tid * 4 + k_32 * 2) % 32;
                                        float x0_16 = 0.0f;
                                        float x1_16 = 0.0f;
                                        x0_16 = (float)hidden_staging[col_0 * 136 + row_0_17];
                                        x1_16 = (float)hidden_staging[(col_0 + 1) * 136 + row_0_17];
                                        __nv_bfloat162 _bf16x2_397 = __float22bfloat162_rn(make_float2(x0_16, x1_16));
                                        pairs_16[k_32] = __as_u32(_bf16x2_397);
                                    }
                                    uint32_t _bf16x2_abs_64;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_64) : "r"(pairs_16[0]));
                                    unsigned int amax_pair_17 = _bf16x2_abs_64;
                                    #pragma unroll
                                    for (int i_66 = 1; i_66 < 16; i_66++) {
                                        uint32_t _bf16x2_abs_65;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_65) : "r"(pairs_16[i_66]));
                                        uint32_t _bf16x2_max_32;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_32) : "r"(amax_pair_17), "r"(_bf16x2_abs_65));
                                        amax_pair_17 = _bf16x2_max_32;
                                    }
                                    uint16_t _bf16_max_32;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_32) : "h"((uint16_t)(amax_pair_17 & 65535)), "h"((uint16_t)(amax_pair_17 >> 16)));
                                    float _cvt_f32_bf16_32;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_32) : "h"((uint16_t)(_bf16_max_32)));
                                    float amax_17 = _cvt_f32_bf16_32;
                                    float _fmax_289 = fmaxf(amax_17 * 0.002232142857f, 1e-12f);
                                    float scale_17 = _fmax_289;
                                    uint16_t _ue8m0x2_f32_32;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_32) : "f"(scale_17), "f"(scale_17));
                                    unsigned int scale_byte_17 = (unsigned int)_ue8m0x2_f32_32 & 255;
                                    unsigned int inverse_lane_17 = 254 - scale_byte_17 << 7;
                                    unsigned int inverse_17 = inverse_lane_17 | inverse_lane_17 << 16;
                                    unsigned int words_17[8];
                                    #pragma unroll
                                    for (int i_67 = 0; i_67 < 8; i_67++) {
                                        uint32_t _bf16x2_mul_64;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(pairs_16[i_67 * 2]), "r"(inverse_17));
                                        uint16_t _e4m3x2_64;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_64) : "r"(_bf16x2_mul_64));
                                        uint32_t _bf16x2_mul_65;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(pairs_16[i_67 * 2 + 1]), "r"(inverse_17));
                                        uint16_t _e4m3x2_65;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_65) : "r"(_bf16x2_mul_65));
                                        words_17[i_67] = (unsigned int)_e4m3x2_64 | (unsigned int)_e4m3x2_65 << 16;
                                    }
                                    scale_word_17 = scale_word_17 | scale_byte_17 << (unsigned int)(k_block_16 * 8);
                                    #pragma unroll
                                    for (int k_33 = 0; k_33 < 8; k_33++) {
                                        int col_0_1 = k_block_16 * 32 + (tid * 4 + k_33 * 4) % 32;
                                        smem_v10[(row_0_17 * 128 + col_0_1) / 4] = words_17[k_33];
                                    }
                                }
                                smem_v11[row_0_17 % 32 * 4 + row_0_17 / 32] = scale_word_17;
                            } else {
                                int row_0_18 = tid - 128;
                                unsigned int scale_word_18 = 0;
                                #pragma unroll 1
                                for (int j_51 = 0; j_51 < 4; j_51++) {
                                    int k_block_17 = (j_51 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_17[16];
                                    #pragma unroll
                                    for (int k_34 = 0; k_34 < 16; k_34++) {
                                        int col_0_2 = k_block_17 * 32 + ((tid - 128) * 4 + k_34 * 2) % 32;
                                        float x0_17 = 0.0f;
                                        float x1_17 = 0.0f;
                                        pairs_17[k_34] = hidden_words[(row_0_18 * 136 + col_0_2) / 2];
                                    }
                                    uint32_t _bf16x2_abs_66;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_66) : "r"(pairs_17[0]));
                                    unsigned int amax_pair_19 = _bf16x2_abs_66;
                                    #pragma unroll
                                    for (int i_68 = 1; i_68 < 16; i_68++) {
                                        uint32_t _bf16x2_abs_67;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_67) : "r"(pairs_17[i_68]));
                                        uint32_t _bf16x2_max_33;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_33) : "r"(amax_pair_19), "r"(_bf16x2_abs_67));
                                        amax_pair_19 = _bf16x2_max_33;
                                    }
                                    uint16_t _bf16_max_33;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_33) : "h"((uint16_t)(amax_pair_19 & 65535)), "h"((uint16_t)(amax_pair_19 >> 16)));
                                    float _cvt_f32_bf16_33;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_33) : "h"((uint16_t)(_bf16_max_33)));
                                    float amax_18 = _cvt_f32_bf16_33;
                                    float _fmax_290 = fmaxf(amax_18 * 0.002232142857f, 1e-12f);
                                    float scale_18 = _fmax_290;
                                    uint16_t _ue8m0x2_f32_33;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_33) : "f"(scale_18), "f"(scale_18));
                                    unsigned int scale_byte_18 = (unsigned int)_ue8m0x2_f32_33 & 255;
                                    unsigned int inverse_lane_18 = 254 - scale_byte_18 << 7;
                                    unsigned int inverse_18 = inverse_lane_18 | inverse_lane_18 << 16;
                                    unsigned int words_18[8];
                                    #pragma unroll
                                    for (int i_69 = 0; i_69 < 8; i_69++) {
                                        uint32_t _bf16x2_mul_66;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(pairs_17[i_69 * 2]), "r"(inverse_18));
                                        uint16_t _e4m3x2_66;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_66) : "r"(_bf16x2_mul_66));
                                        uint32_t _bf16x2_mul_67;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(pairs_17[i_69 * 2 + 1]), "r"(inverse_18));
                                        uint16_t _e4m3x2_67;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_67) : "r"(_bf16x2_mul_67));
                                        words_18[i_69] = (unsigned int)_e4m3x2_66 | (unsigned int)_e4m3x2_67 << 16;
                                    }
                                    scale_word_18 = scale_word_18 | scale_byte_18 << (unsigned int)(k_block_17 * 8);
                                    #pragma unroll
                                    for (int k_35 = 0; k_35 < 8; k_35++) {
                                        int col_0_3 = k_block_17 * 32 + ((tid - 128) * 4 + k_35 * 4) % 32;
                                        smem_v8[(row_0_18 * 128 + col_0_3) / 4] = words_18[k_35];
                                    }
                                }
                                smem_v9[row_0_18 % 32 * 4 + row_0_18 / 32] = scale_word_18;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int local_row = row_36 - macro_row_offset;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&hidden_q_store), col_64 * 128, local_row * 128, smem_v8_addr);
                                tma_store_3d((&hidden_sc_store), 0, 0, local_row * col_blocks_6 + col_64, smem_v9_addr);
                                tma_store_2d((&hidden_t_store), local_row * 128, col_64 * 128, smem_v10_addr);
                                tma_store_3d((&hidden_sc_t_store), 0, 0, col_64 * macro_tiles_2 + local_row, smem_v11_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tile_end > first_tile_2 + 1) {
                            mbarrier_wait(swiglu_arrived_addr + 8, phase_bits_5 >> 1 & 1);
                            phase_bits_5 = phase_bits_5 ^ 2;
                            int row_37 = first_row_3;
                            int col_65 = first_col_2 + 1;
                            if (col_65 >= col_blocks_6) {
                                row_37 = row_37 + 1;
                                col_65 = col_65 - col_blocks_6;
                            }
                            float gate_1[64];
                            float up_1[64];
                            float denominator_1[64];
                            int warp_0_3 = tid / 32;
                            int local_warp_1 = warp_0_3 / 4 + warp_0_3 % 4 * 2;
                            int lane_5 = tid % 32;
                            #pragma unroll
                            for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                unsigned int packed_5[4];
                                unsigned int address_26 = gate_smem_addr + 32768 + (unsigned int)(((tile_col_3 * 16 + lane_5 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_5 % 16) * 64 + (tile_col_3 * 16 + lane_5 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_5[0]), "=r"(packed_5[1]), "=r"(packed_5[2]), "=r"(packed_5[3])
                                    : "r"(address_26 ^ (address_26 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                    float2 _cvt_f32_258 = __bfloat1622float2(__as_bf16x2(packed_5[pair_5]));
                                    gate_1[tile_col_3 * 8 + pair_5 * 2] = _cvt_f32_258.x;
                                    gate_1[tile_col_3 * 8 + pair_5 * 2 + 1] = _cvt_f32_258.y;
                                }
                            }
                            int warp_1_2 = tid / 32;
                            int local_warp_2_1 = warp_1_2 / 4 + warp_1_2 % 4 * 2;
                            int lane_3_2 = tid % 32;
                            #pragma unroll
                            for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                unsigned int packed_6[4];
                                unsigned int address_27 = up_smem_addr + 32768 + (unsigned int)(((tile_col_4 * 16 + lane_3_2 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_1 * 16 + lane_3_2 % 16) * 64 + (tile_col_4 * 16 + lane_3_2 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_6[0]), "=r"(packed_6[1]), "=r"(packed_6[2]), "=r"(packed_6[3])
                                    : "r"(address_27 ^ (address_27 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_6 = 0; pair_6 < 4; pair_6++) {
                                    float2 _cvt_f32_259 = __bfloat1622float2(__as_bf16x2(packed_6[pair_6]));
                                    up_1[tile_col_4 * 8 + pair_6 * 2] = _cvt_f32_259.x;
                                    up_1[tile_col_4 * 8 + pair_6 * 2 + 1] = _cvt_f32_259.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_7 = 0; elem_7 < 64; elem_7++) {
                                float _min_547 = fminf(gate_1[elem_7], swiglu_limit);
                                gate_1[elem_7] = _min_547;
                            }
                            #pragma unroll
                            for (int elem_8 = 0; elem_8 < 64; elem_8++) {
                                float _fmax_291 = fmaxf(up_1[elem_8], -swiglu_limit);
                                float _min_548 = fminf(_fmax_291, swiglu_limit);
                                up_1[elem_8] = _min_548;
                            }
                            #pragma unroll
                            for (int elem_9 = 0; elem_9 < 64; elem_9++) {
                                denominator_1[elem_9] = gate_1[elem_9] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_10 = 0; elem_10 < 64; elem_10++) {
                                float _exp_257 = expf(denominator_1[elem_10]);
                                denominator_1[elem_10] = _exp_257;
                            }
                            #pragma unroll
                            for (int elem_11 = 0; elem_11 < 64; elem_11++) {
                                denominator_1[elem_11] = denominator_1[elem_11] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_12 = 0; elem_12 < 64; elem_12++) {
                                gate_1[elem_12] = gate_1[elem_12] / denominator_1[elem_12];
                            }
                            #pragma unroll
                            for (int elem_13 = 0; elem_13 < 64; elem_13++) {
                                gate_1[elem_13] = gate_1[elem_13] * up_1[elem_13];
                            }
                            __syncthreads();
                            int warp_4_1 = tid / 32;
                            int local_warp_5_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                            int lane_6_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                unsigned int packed_7[4];
                                #pragma unroll
                                for (int pair_7 = 0; pair_7 < 4; pair_7++) {
                                    __nv_bfloat162 _bf16x2_398 = __float22bfloat162_rn(make_float2(gate_1[tile_col_5 * 8 + pair_7 * 2], gate_1[tile_col_5 * 8 + pair_7 * 2 + 1]));
                                    packed_7[pair_7] = __as_u32(_bf16x2_398);
                                }
                                int row_0_19 = local_warp_5_1 * 16 + lane_6_1 % 16;
                                int col_1_2 = tile_col_5 * 16 + lane_6_1 / 16 * 8;
                                uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_19 * 136 + col_1_2) * 2));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_20 = tid;
                                row_0_20 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_19 = 0;
                                #pragma unroll 1
                                for (int j_52 = 0; j_52 < 4; j_52++) {
                                    int k_block_18 = (j_52 + tid / 8) % 4;
                                    unsigned int pairs_18[16];
                                    #pragma unroll
                                    for (int k_36 = 0; k_36 < 16; k_36++) {
                                        int col_0_4 = k_block_18 * 32 + (tid * 4 + k_36 * 2) % 32;
                                        float x0_18 = 0.0f;
                                        float x1_18 = 0.0f;
                                        x0_18 = (float)hidden_staging[col_0_4 * 136 + row_0_20];
                                        x1_18 = (float)hidden_staging[(col_0_4 + 1) * 136 + row_0_20];
                                        __nv_bfloat162 _bf16x2_399 = __float22bfloat162_rn(make_float2(x0_18, x1_18));
                                        pairs_18[k_36] = __as_u32(_bf16x2_399);
                                    }
                                    uint32_t _bf16x2_abs_68;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_68) : "r"(pairs_18[0]));
                                    unsigned int amax_pair_20 = _bf16x2_abs_68;
                                    #pragma unroll
                                    for (int i_70 = 1; i_70 < 16; i_70++) {
                                        uint32_t _bf16x2_abs_69;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_69) : "r"(pairs_18[i_70]));
                                        uint32_t _bf16x2_max_34;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_34) : "r"(amax_pair_20), "r"(_bf16x2_abs_69));
                                        amax_pair_20 = _bf16x2_max_34;
                                    }
                                    uint16_t _bf16_max_34;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_34) : "h"((uint16_t)(amax_pair_20 & 65535)), "h"((uint16_t)(amax_pair_20 >> 16)));
                                    float _cvt_f32_bf16_34;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_34) : "h"((uint16_t)(_bf16_max_34)));
                                    float amax_20 = _cvt_f32_bf16_34;
                                    float _fmax_292 = fmaxf(amax_20 * 0.002232142857f, 1e-12f);
                                    float scale_19 = _fmax_292;
                                    uint16_t _ue8m0x2_f32_34;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_34) : "f"(scale_19), "f"(scale_19));
                                    unsigned int scale_byte_19 = (unsigned int)_ue8m0x2_f32_34 & 255;
                                    unsigned int inverse_lane_19 = 254 - scale_byte_19 << 7;
                                    unsigned int inverse_19 = inverse_lane_19 | inverse_lane_19 << 16;
                                    unsigned int words_19[8];
                                    #pragma unroll
                                    for (int i_71 = 0; i_71 < 8; i_71++) {
                                        uint32_t _bf16x2_mul_68;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(pairs_18[i_71 * 2]), "r"(inverse_19));
                                        uint16_t _e4m3x2_68;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_68) : "r"(_bf16x2_mul_68));
                                        uint32_t _bf16x2_mul_69;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(pairs_18[i_71 * 2 + 1]), "r"(inverse_19));
                                        uint16_t _e4m3x2_69;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_69) : "r"(_bf16x2_mul_69));
                                        words_19[i_71] = (unsigned int)_e4m3x2_68 | (unsigned int)_e4m3x2_69 << 16;
                                    }
                                    scale_word_19 = scale_word_19 | scale_byte_19 << (unsigned int)(k_block_18 * 8);
                                    #pragma unroll
                                    for (int k_37 = 0; k_37 < 8; k_37++) {
                                        int col_0_5 = k_block_18 * 32 + (tid * 4 + k_37 * 4) % 32;
                                        smem_v14[(row_0_20 * 128 + col_0_5) / 4] = words_19[k_37];
                                    }
                                }
                                smem_v15[row_0_20 % 32 * 4 + row_0_20 / 32] = scale_word_19;
                            } else {
                                int row_0_21 = tid - 128;
                                unsigned int scale_word_20 = 0;
                                #pragma unroll 1
                                for (int j_53 = 0; j_53 < 4; j_53++) {
                                    int k_block_19 = (j_53 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_19[16];
                                    #pragma unroll
                                    for (int k_38 = 0; k_38 < 16; k_38++) {
                                        int col_0_6 = k_block_19 * 32 + ((tid - 128) * 4 + k_38 * 2) % 32;
                                        float x0_19 = 0.0f;
                                        float x1_19 = 0.0f;
                                        pairs_19[k_38] = hidden_words[(row_0_21 * 136 + col_0_6) / 2];
                                    }
                                    uint32_t _bf16x2_abs_70;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_70) : "r"(pairs_19[0]));
                                    unsigned int amax_pair_21 = _bf16x2_abs_70;
                                    #pragma unroll
                                    for (int i_72 = 1; i_72 < 16; i_72++) {
                                        uint32_t _bf16x2_abs_71;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_71) : "r"(pairs_19[i_72]));
                                        uint32_t _bf16x2_max_35;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_35) : "r"(amax_pair_21), "r"(_bf16x2_abs_71));
                                        amax_pair_21 = _bf16x2_max_35;
                                    }
                                    uint16_t _bf16_max_35;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_35) : "h"((uint16_t)(amax_pair_21 & 65535)), "h"((uint16_t)(amax_pair_21 >> 16)));
                                    float _cvt_f32_bf16_35;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_35) : "h"((uint16_t)(_bf16_max_35)));
                                    float amax_21 = _cvt_f32_bf16_35;
                                    float _fmax_293 = fmaxf(amax_21 * 0.002232142857f, 1e-12f);
                                    float scale_21 = _fmax_293;
                                    uint16_t _ue8m0x2_f32_35;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_35) : "f"(scale_21), "f"(scale_21));
                                    unsigned int scale_byte_20 = (unsigned int)_ue8m0x2_f32_35 & 255;
                                    unsigned int inverse_lane_20 = 254 - scale_byte_20 << 7;
                                    unsigned int inverse_20 = inverse_lane_20 | inverse_lane_20 << 16;
                                    unsigned int words_20[8];
                                    #pragma unroll
                                    for (int i_73 = 0; i_73 < 8; i_73++) {
                                        uint32_t _bf16x2_mul_70;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(pairs_19[i_73 * 2]), "r"(inverse_20));
                                        uint16_t _e4m3x2_70;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_70) : "r"(_bf16x2_mul_70));
                                        uint32_t _bf16x2_mul_71;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(pairs_19[i_73 * 2 + 1]), "r"(inverse_20));
                                        uint16_t _e4m3x2_71;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_71) : "r"(_bf16x2_mul_71));
                                        words_20[i_73] = (unsigned int)_e4m3x2_70 | (unsigned int)_e4m3x2_71 << 16;
                                    }
                                    scale_word_20 = scale_word_20 | scale_byte_20 << (unsigned int)(k_block_19 * 8);
                                    #pragma unroll
                                    for (int k_39 = 0; k_39 < 8; k_39++) {
                                        int col_0_7 = k_block_19 * 32 + ((tid - 128) * 4 + k_39 * 4) % 32;
                                        smem_v12[(row_0_21 * 128 + col_0_7) / 4] = words_20[k_39];
                                    }
                                }
                                smem_v13[row_0_21 % 32 * 4 + row_0_21 / 32] = scale_word_20;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int local_row_1 = row_37 - macro_row_offset;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&hidden_q_store), col_65 * 128, local_row_1 * 128, smem_v12_addr);
                                tma_store_3d((&hidden_sc_store), 0, 0, local_row_1 * col_blocks_6 + col_65, smem_v13_addr);
                                tma_store_2d((&hidden_t_store), local_row_1 * 128, col_65 * 128, smem_v14_addr);
                                tma_store_3d((&hidden_sc_t_store), 0, 0, col_65 * macro_tiles_2 + local_row_1, smem_v15_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tile_end > first_tile_2 + 2) {
                            mbarrier_wait(swiglu_arrived_addr + 16, phase_bits_5 >> 2 & 1);
                            phase_bits_5 = phase_bits_5 ^ 4;
                            int row_38 = first_row_3;
                            int col_66 = first_col_2 + 2;
                            if (col_66 >= col_blocks_6) {
                                row_38 = row_38 + 1;
                                col_66 = col_66 - col_blocks_6;
                            }
                            float gate_2[64];
                            float up_2[64];
                            float denominator_2[64];
                            int warp_0_4 = tid / 32;
                            int local_warp_3 = warp_0_4 / 4 + warp_0_4 % 4 * 2;
                            int lane_7 = tid % 32;
                            #pragma unroll
                            for (int tile_col_6 = 0; tile_col_6 < 8; tile_col_6++) {
                                unsigned int packed_8[4];
                                unsigned int address_28 = gate_smem_addr + 65536 + (unsigned int)(((tile_col_6 * 16 + lane_7 / 16 * 8) / 64 * 128 * 64 + (local_warp_3 * 16 + lane_7 % 16) * 64 + (tile_col_6 * 16 + lane_7 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_8[0]), "=r"(packed_8[1]), "=r"(packed_8[2]), "=r"(packed_8[3])
                                    : "r"(address_28 ^ (address_28 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                    float2 _cvt_f32_260 = __bfloat1622float2(__as_bf16x2(packed_8[pair_8]));
                                    gate_2[tile_col_6 * 8 + pair_8 * 2] = _cvt_f32_260.x;
                                    gate_2[tile_col_6 * 8 + pair_8 * 2 + 1] = _cvt_f32_260.y;
                                }
                            }
                            int warp_1_3 = tid / 32;
                            int local_warp_2_2 = warp_1_3 / 4 + warp_1_3 % 4 * 2;
                            int lane_3_3 = tid % 32;
                            #pragma unroll
                            for (int tile_col_7 = 0; tile_col_7 < 8; tile_col_7++) {
                                unsigned int packed_9[4];
                                unsigned int address_29 = up_smem_addr + 65536 + (unsigned int)(((tile_col_7 * 16 + lane_3_3 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_2 * 16 + lane_3_3 % 16) * 64 + (tile_col_7 * 16 + lane_3_3 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_9[0]), "=r"(packed_9[1]), "=r"(packed_9[2]), "=r"(packed_9[3])
                                    : "r"(address_29 ^ (address_29 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                    float2 _cvt_f32_261 = __bfloat1622float2(__as_bf16x2(packed_9[pair_9]));
                                    up_2[tile_col_7 * 8 + pair_9 * 2] = _cvt_f32_261.x;
                                    up_2[tile_col_7 * 8 + pair_9 * 2 + 1] = _cvt_f32_261.y;
                                }
                            }
                            #pragma unroll
                            for (int elem_14 = 0; elem_14 < 64; elem_14++) {
                                float _min_549 = fminf(gate_2[elem_14], swiglu_limit);
                                gate_2[elem_14] = _min_549;
                            }
                            #pragma unroll
                            for (int elem_15 = 0; elem_15 < 64; elem_15++) {
                                float _fmax_294 = fmaxf(up_2[elem_15], -swiglu_limit);
                                float _min_550 = fminf(_fmax_294, swiglu_limit);
                                up_2[elem_15] = _min_550;
                            }
                            #pragma unroll
                            for (int elem_16 = 0; elem_16 < 64; elem_16++) {
                                denominator_2[elem_16] = gate_2[elem_16] * -1.0f;
                            }
                            #pragma unroll
                            for (int elem_17 = 0; elem_17 < 64; elem_17++) {
                                float _exp_258 = expf(denominator_2[elem_17]);
                                denominator_2[elem_17] = _exp_258;
                            }
                            #pragma unroll
                            for (int elem_18 = 0; elem_18 < 64; elem_18++) {
                                denominator_2[elem_18] = denominator_2[elem_18] + 1.0f;
                            }
                            #pragma unroll
                            for (int elem_19 = 0; elem_19 < 64; elem_19++) {
                                gate_2[elem_19] = gate_2[elem_19] / denominator_2[elem_19];
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
                            for (int tile_col_8 = 0; tile_col_8 < 8; tile_col_8++) {
                                unsigned int packed_10[4];
                                #pragma unroll
                                for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                    __nv_bfloat162 _bf16x2_400 = __float22bfloat162_rn(make_float2(gate_2[tile_col_8 * 8 + pair_10 * 2], gate_2[tile_col_8 * 8 + pair_10 * 2 + 1]));
                                    packed_10[pair_10] = __as_u32(_bf16x2_400);
                                }
                                int row_0_22 = local_warp_5_2 * 16 + lane_6_2 % 16;
                                int col_1_3 = tile_col_8 * 16 + lane_6_2 / 16 * 8;
                                uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_22 * 136 + col_1_3) * 2));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid < 128) {
                                int row_0_23 = tid;
                                row_0_23 = tid % 64 * 2 + tid / 64;
                                unsigned int scale_word_21 = 0;
                                #pragma unroll 1
                                for (int j_54 = 0; j_54 < 4; j_54++) {
                                    int k_block_20 = (j_54 + tid / 8) % 4;
                                    unsigned int pairs_20[16];
                                    #pragma unroll
                                    for (int k_40 = 0; k_40 < 16; k_40++) {
                                        int col_0_8 = k_block_20 * 32 + (tid * 4 + k_40 * 2) % 32;
                                        float x0_20 = 0.0f;
                                        float x1_20 = 0.0f;
                                        x0_20 = (float)hidden_staging[col_0_8 * 136 + row_0_23];
                                        x1_20 = (float)hidden_staging[(col_0_8 + 1) * 136 + row_0_23];
                                        __nv_bfloat162 _bf16x2_401 = __float22bfloat162_rn(make_float2(x0_20, x1_20));
                                        pairs_20[k_40] = __as_u32(_bf16x2_401);
                                    }
                                    uint32_t _bf16x2_abs_72;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_72) : "r"(pairs_20[0]));
                                    unsigned int amax_pair_22 = _bf16x2_abs_72;
                                    #pragma unroll
                                    for (int i_74 = 1; i_74 < 16; i_74++) {
                                        uint32_t _bf16x2_abs_73;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_73) : "r"(pairs_20[i_74]));
                                        uint32_t _bf16x2_max_36;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_36) : "r"(amax_pair_22), "r"(_bf16x2_abs_73));
                                        amax_pair_22 = _bf16x2_max_36;
                                    }
                                    uint16_t _bf16_max_36;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_36) : "h"((uint16_t)(amax_pair_22 & 65535)), "h"((uint16_t)(amax_pair_22 >> 16)));
                                    float _cvt_f32_bf16_36;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_36) : "h"((uint16_t)(_bf16_max_36)));
                                    float amax_22 = _cvt_f32_bf16_36;
                                    float _fmax_295 = fmaxf(amax_22 * 0.002232142857f, 1e-12f);
                                    float scale_22 = _fmax_295;
                                    uint16_t _ue8m0x2_f32_36;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_36) : "f"(scale_22), "f"(scale_22));
                                    unsigned int scale_byte_22 = (unsigned int)_ue8m0x2_f32_36 & 255;
                                    unsigned int inverse_lane_21 = 254 - scale_byte_22 << 7;
                                    unsigned int inverse_21 = inverse_lane_21 | inverse_lane_21 << 16;
                                    unsigned int words_21[8];
                                    #pragma unroll
                                    for (int i_75 = 0; i_75 < 8; i_75++) {
                                        uint32_t _bf16x2_mul_72;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(pairs_20[i_75 * 2]), "r"(inverse_21));
                                        uint16_t _e4m3x2_72;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_72) : "r"(_bf16x2_mul_72));
                                        uint32_t _bf16x2_mul_73;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(pairs_20[i_75 * 2 + 1]), "r"(inverse_21));
                                        uint16_t _e4m3x2_73;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_73) : "r"(_bf16x2_mul_73));
                                        words_21[i_75] = (unsigned int)_e4m3x2_72 | (unsigned int)_e4m3x2_73 << 16;
                                    }
                                    scale_word_21 = scale_word_21 | scale_byte_22 << (unsigned int)(k_block_20 * 8);
                                    #pragma unroll
                                    for (int k_41 = 0; k_41 < 8; k_41++) {
                                        int col_0_9 = k_block_20 * 32 + (tid * 4 + k_41 * 4) % 32;
                                        smem_v18[(row_0_23 * 128 + col_0_9) / 4] = words_21[k_41];
                                    }
                                }
                                smem_v19[row_0_23 % 32 * 4 + row_0_23 / 32] = scale_word_21;
                            } else {
                                int row_0_24 = tid - 128;
                                unsigned int scale_word_22 = 0;
                                #pragma unroll 1
                                for (int j_55 = 0; j_55 < 4; j_55++) {
                                    int k_block_21 = (j_55 + (tid - 128) / 8) % 4;
                                    unsigned int pairs_21[16];
                                    #pragma unroll
                                    for (int k_42 = 0; k_42 < 16; k_42++) {
                                        int col_0_10 = k_block_21 * 32 + ((tid - 128) * 4 + k_42 * 2) % 32;
                                        float x0_21 = 0.0f;
                                        float x1_21 = 0.0f;
                                        pairs_21[k_42] = hidden_words[(row_0_24 * 136 + col_0_10) / 2];
                                    }
                                    uint32_t _bf16x2_abs_74;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_74) : "r"(pairs_21[0]));
                                    unsigned int amax_pair_23 = _bf16x2_abs_74;
                                    #pragma unroll
                                    for (int i_76 = 1; i_76 < 16; i_76++) {
                                        uint32_t _bf16x2_abs_75;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_75) : "r"(pairs_21[i_76]));
                                        uint32_t _bf16x2_max_37;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_37) : "r"(amax_pair_23), "r"(_bf16x2_abs_75));
                                        amax_pair_23 = _bf16x2_max_37;
                                    }
                                    uint16_t _bf16_max_37;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_37) : "h"((uint16_t)(amax_pair_23 & 65535)), "h"((uint16_t)(amax_pair_23 >> 16)));
                                    float _cvt_f32_bf16_37;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_37) : "h"((uint16_t)(_bf16_max_37)));
                                    float amax_23 = _cvt_f32_bf16_37;
                                    float _fmax_296 = fmaxf(amax_23 * 0.002232142857f, 1e-12f);
                                    float scale_23 = _fmax_296;
                                    uint16_t _ue8m0x2_f32_37;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_37) : "f"(scale_23), "f"(scale_23));
                                    unsigned int scale_byte_23 = (unsigned int)_ue8m0x2_f32_37 & 255;
                                    unsigned int inverse_lane_23 = 254 - scale_byte_23 << 7;
                                    unsigned int inverse_22 = inverse_lane_23 | inverse_lane_23 << 16;
                                    unsigned int words_22[8];
                                    #pragma unroll
                                    for (int i_77 = 0; i_77 < 8; i_77++) {
                                        uint32_t _bf16x2_mul_74;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(pairs_21[i_77 * 2]), "r"(inverse_22));
                                        uint16_t _e4m3x2_74;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_74) : "r"(_bf16x2_mul_74));
                                        uint32_t _bf16x2_mul_75;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(pairs_21[i_77 * 2 + 1]), "r"(inverse_22));
                                        uint16_t _e4m3x2_75;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_75) : "r"(_bf16x2_mul_75));
                                        words_22[i_77] = (unsigned int)_e4m3x2_74 | (unsigned int)_e4m3x2_75 << 16;
                                    }
                                    scale_word_22 = scale_word_22 | scale_byte_23 << (unsigned int)(k_block_21 * 8);
                                    #pragma unroll
                                    for (int k_43 = 0; k_43 < 8; k_43++) {
                                        int col_0_11 = k_block_21 * 32 + ((tid - 128) * 4 + k_43 * 4) % 32;
                                        smem_v16[(row_0_24 * 128 + col_0_11) / 4] = words_22[k_43];
                                    }
                                }
                                smem_v17[row_0_24 % 32 * 4 + row_0_24 / 32] = scale_word_22;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int local_row_2 = row_38 - macro_row_offset;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&hidden_q_store), col_66 * 128, local_row_2 * 128, smem_v16_addr);
                                tma_store_3d((&hidden_sc_store), 0, 0, local_row_2 * col_blocks_6 + col_66, smem_v17_addr);
                                tma_store_2d((&hidden_t_store), local_row_2 * 128, col_66 * 128, smem_v18_addr);
                                tma_store_3d((&hidden_sc_t_store), 0, 0, col_66 * macro_tiles_2 + local_row_2, smem_v19_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            if (tile_end > first_tile_2) {
                                int row_39 = first_row_3;
                                if (col_blocks_6 <= first_col_2) {
                                    row_39 = row_39 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_39 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end > first_tile_2 + 1) {
                                int row_40 = first_row_3;
                                if (col_blocks_6 <= first_col_2 + 1) {
                                    row_40 = row_40 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_40 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end > first_tile_2 + 2) {
                                int row_41 = first_row_3;
                                if (col_blocks_6 <= first_col_2 + 2) {
                                    row_41 = row_41 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_41 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    swiglu_bits = phase_bits_5;
                } else {
                    int col_blocks_7 = (hidden + 512 - 1) / 512;
                    int x_3 = -1;
                    int y_3 = -1;
                    int expert_3 = -1;
                    int k_start_3 = 0;
                    int k_end_3 = 0;
                    int first_3 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int _min_551 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int end_block_1 = _min_551;
                    int block_2 = first_block_1 + (task_2 - mini_gate - mini_swiglu) / col_blocks_7;
                    if (block_2 < end_block_1) {
                        int index_1 = counts[3 * experts + block_2];
                        int offset_27 = counts[experts + index_1] / 256;
                        int _max_6 = ((first_block_1) > (offset_27) ? (first_block_1) : (offset_27));
                        int first_row_4 = _max_6;
                        int _min_552 = ((end_block_1) < (offset_27 + counts[index_1] / 256) ? (end_block_1) : (offset_27 + counts[index_1] / 256));
                        int rows_3 = _min_552 - first_row_4;
                        int supergroup_3 = (task_2 - mini_gate - mini_swiglu - (first_row_4 - first_block_1) * col_blocks_7) / (rows_3 * 8);
                        int full_cols_3 = col_blocks_7 / 8 * 8;
                        int row_42 = 0;
                        int col_67 = 0;
                        if (task_2 - mini_gate - mini_swiglu - (first_row_4 - first_block_1) * col_blocks_7 < rows_3 * full_cols_3) {
                            row_42 = (task_2 - mini_gate - mini_swiglu - (first_row_4 - first_block_1) * col_blocks_7) % (rows_3 * 8) / 8;
                            col_67 = supergroup_3 * 8 + (task_2 - mini_gate - mini_swiglu - (first_row_4 - first_block_1) * col_blocks_7) % 8;
                        } else {
                            row_42 = (task_2 - mini_gate - mini_swiglu - (first_row_4 - first_block_1) * col_blocks_7 - rows_3 * full_cols_3) / (col_blocks_7 - full_cols_3);
                            col_67 = full_cols_3 + (task_2 - mini_gate - mini_swiglu - (first_row_4 - first_block_1) * col_blocks_7 - rows_3 * full_cols_3) % (col_blocks_7 - full_cols_3);
                        }
                        if ((supergroup_3 & 1) != 0) {
                            row_42 = rows_3 - row_42 - 1;
                        }
                        x_3 = first_row_4 + row_42 - macro_1 * (macro_size / 256);
                        y_3 = col_67;
                        expert_3 = index_1;
                    }
                    unsigned int phase_bits_6 = gemm_bits;
                    int has_hi_3 = 0;
                    has_hi_3 = (int)((y_3 * 2 + 1) * 256 < hidden);
                    int global_mini_4 = macro_1 * (macro_size / mini_size) + mini_2;
                    int macro_rows_4 = macro_1 * (macro_size / 256);
                    int iterations_3 = intermediate / 128;
                    int macro_k_1 = macro_1 * (macro_size / 128);
                    if (expert_3 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    bool enabled_value_5 = 1;
                                    if (enabled_value_5 != 0) {
                                        int32_t _relaxed_ld_18;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_18) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                        int value_9 = _relaxed_ld_18;
                                        while (value_9 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_19;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_19) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                            value_9 = _relaxed_ld_19;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_553 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                    int _max_7 = ((0) > (_min_553) ? (0) : (_min_553));
                                    int mini_rows_7 = _max_7;
                                    int required_7 = (mini_rows_7 + 127) / 128 * ((intermediate + 511) / 512);
                                }
                                int ring_7 = 0;
                                #pragma unroll 1
                                for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_7) * 8, phase_bits_6 >> (unsigned int)(16 + ring_7) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v40_addr + (unsigned int)(ring_7 * 16384)), "l"((&hidden_q)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v41_addr + (unsigned int)(ring_7 * 16384)), "l"((&wd_q)), "r"(0), "r"(y_3 * 512 + cta_rank_0 * 128), "r"(idx_7), "r"(expert_3), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(smem_v42_addr + (unsigned int)(ring_7 * 16384)), "l"((&wd_q)), "r"(0), "r"(y_3 * 512 + 256 + cta_rank_0 * 128), "r"(idx_7), "r"(expert_3), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << 16 + ring_7);
                                    ring_7 = (ring_7 + 1) % 4;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        bool enabled_value_6 = 1;
                                        if (enabled_value_6 != 0) {
                                            int32_t _relaxed_ld_20;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_20) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                            int value_10 = _relaxed_ld_20;
                                            while (value_10 < 2 * (intermediate / 128)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_21;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_21) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                                value_10 = _relaxed_ld_21;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_554 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                        int _max_8 = ((0) > (_min_554) ? (0) : (_min_554));
                                        int mini_rows_8 = _max_8;
                                        int required_8 = (mini_rows_8 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    int ring_8 = 0;
                                    #pragma unroll 1
                                    for (int idx_8 = 0; idx_8 < iterations_3; idx_8++) {
                                        mbarrier_wait(scales_finished_addr + (ring_8) * 8, phase_bits_6 >> (unsigned int)(16 + ring_8) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v43_addr + (unsigned int)(ring_8 * 512)), "l"((&hidden_sc)), "r"(0), "r"(0), "r"((x_3 * 2 + cta_rank_0) * (intermediate / 128) + idx_8),
                                               "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v44_addr + (unsigned int)(ring_8 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_sc)), "r"(0), "r"(0), "r"((expert_3 * (hidden / 128) + y_3 * 4 + cta_rank_0) * (intermediate / 128) + idx_8),
                                               "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v45_addr + (unsigned int)(ring_8 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_sc)), "r"(0), "r"(0), "r"((expert_3 * (hidden / 128) + y_3 * 4 + 2 + cta_rank_0) * (intermediate / 128) + idx_8),
                                               "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << 16 + ring_8);
                                        ring_8 = (ring_8 + 1) % 4;
                                    }
                                }
                            }
                        } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_9 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_6 >> 22 & 1);
                                    phase_bits_6 = phase_bits_6 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_9 = 0; idx_9 < iterations_3; idx_9++) {
                                        mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_9) * 8, 5120);
                                        mbarrier_wait(scales_arrived_addr + (ring_9) * 8, phase_bits_6 >> (unsigned int)(8 + ring_9) & 1);
                                        int buffer_1 = idx_9 % 3;
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_1 * 4, make_sf_cp_desc_sbo128(smem_v43_addr + (unsigned int)(ring_9 * 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_1 * 8, make_sf_cp_desc_sbo128(smem_v44_addr + (unsigned int)(ring_9 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v44_addr + (unsigned int)(ring_9 * 1024) + 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_1 * 8, make_sf_cp_desc_sbo128(smem_v45_addr + (unsigned int)(ring_9 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v45_addr + (unsigned int)(ring_9 * 1024) + 512)));
                                        tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_9) * 8, 98304);
                                        mbarrier_wait(gemm_arrived_addr + (ring_9) * 8, phase_bits_6 >> (unsigned int)ring_9 & 1);
                                        int _mma_a_lo_6 = (((smem_v40_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                        int _mma_b_lo_6 = (((smem_v41_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0)), ((idx_9 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0) | 0x80000000), 1);
                                        }
                                        int _mma_a_lo_7 = (((smem_v40_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                        int _mma_b_lo_7 = (((smem_v42_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_1 * 8 + 0)), ((idx_9 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_1 * 8 + 0) | 0x80000000), 1);
                                        }
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                        phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << ring_9) ^ (unsigned int)(1 << 8 + ring_9);
                                        ring_9 = (ring_9 + 1) % 4;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else {
                            if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_6 >> 6 & 1);
                                unsigned int packed_11[128];
                                #pragma unroll
                                for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                    #pragma unroll
                                    for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                        unsigned int address_30 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                        float _tmem_load_36[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[15]))
                                            : "r"(address_30));
                                        #pragma unroll
                                        for (int pair_11 = 0; pair_11 < 8; pair_11++) {
                                            __nv_bfloat162 _bf16x2_402 = __float22bfloat162_rn(make_float2(_tmem_load_36[pair_11 * 2], _tmem_load_36[pair_11 * 2 + 1]));
                                            packed_11[chunk_4 * 16 + sub_2 * 8 + pair_11] = __as_u32(_bf16x2_402);
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_2 = 1;
                                last_2 = 1 - has_hi_3;
                                if (last_2 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                if (tid == 0) {
                                    int previous_offset_2 = (macro_1 + 1) * macro_size;
                                    int output_row_1 = x_3 * 256 + cta_rank_0 * 128;
                                    int _min_555 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                    if (output_row_1 < _min_555) {
                                        bool enabled_value_7 = 1;
                                        if (enabled_value_7 != 0) {
                                            int32_t _relaxed_ld_22;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_22) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                            int value_11 = _relaxed_ld_22;
                                            while (value_11 < 8 * ((hidden + 1023) / 1024)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_23;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_23) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                                value_11 = _relaxed_ld_23;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int chunk_5 = 0; chunk_5 < 8; chunk_5++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_5 = tid / 32;
                                    int lane_8 = tid % 32;
                                    #pragma unroll
                                    for (int half_26 = 0; half_26 < 2; half_26++) {
                                        #pragma unroll
                                        for (int col_tile_34 = 0; col_tile_34 < 2; col_tile_34++) {
                                            int row_43 = warp_0_5 * 32 + half_26 * 16 + lane_8 % 16;
                                            int col_68 = col_tile_34 * 16 + lane_8 / 16 * 8;
                                            unsigned int address_31 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_43 * 32 + col_68) * 2);
                                            address_31 = address_31 ^ (address_31 & 511) >> 7 << 4;
                                            int offset_28 = chunk_5 * 16 + half_26 * 8 + col_tile_34 * 4;
                                            uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(address_31);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_28])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_28 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_28 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_28 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&y_routed)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 2 * 8 + chunk_5), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                if (has_hi_3 != 0) {
                                    unsigned int packed_0_1[128];
                                    #pragma unroll
                                    for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                        #pragma unroll
                                        for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                            unsigned int address_32 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                            float _tmem_load_37[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[15]))
                                                : "r"(address_32));
                                            #pragma unroll
                                            for (int pair_12 = 0; pair_12 < 8; pair_12++) {
                                                __nv_bfloat162 _bf16x2_403 = __float22bfloat162_rn(make_float2(_tmem_load_37[pair_12 * 2], _tmem_load_37[pair_12 * 2 + 1]));
                                                packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_12] = __as_u32(_bf16x2_403);
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
                                        int warp_0_6 = tid / 32;
                                        int lane_9 = tid % 32;
                                        #pragma unroll
                                        for (int half_27 = 0; half_27 < 2; half_27++) {
                                            #pragma unroll
                                            for (int col_tile_35 = 0; col_tile_35 < 2; col_tile_35++) {
                                                int row_44 = warp_0_6 * 32 + half_27 * 16 + lane_9 % 16;
                                                int col_69 = col_tile_35 * 16 + lane_9 / 16 * 8;
                                                unsigned int address_33 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_44 * 32 + col_69) * 2);
                                                address_33 = address_33 ^ (address_33 & 511) >> 7 << 4;
                                                int offset_29 = chunk_7 * 16 + half_27 * 8 + col_tile_35 * 4;
                                                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(address_33);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&y_routed)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"((y_3 * 2 + 1) * 8 + chunk_7), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                phase_bits_6 = phase_bits_6 ^ 64;
                                if (tid / 32 == 0) {
                                    if (warp == 0) {
                                        if (elect_sync()) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_8 = 1;
                                            if (enabled_value_8 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_4))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                            if (has_hi_3 != 0) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_0 = 1;
                                                if (enabled_value_0 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_4))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_bits = phase_bits_6;
                }
            }
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
            int result_0 = 0;
            if (cluster - comm_clusters >= 0) {
                if (shared_tasks > cluster - comm_clusters) {
                    if (shared_tasks <= cluster - comm_clusters && cluster - comm_clusters < shared_tasks) {
                        result_0 = 1;
                    }
                } else {
                    int mini_task_1 = (cluster - comm_clusters - shared_tasks) % mini_tasks;
                    if (mini_task_1 >= mini_gate && mini_task_1 < mini_gate + mini_swiglu) {
                        result_0 = 1;
                    }
                }
            }
            if (result != 0 && cluster >= 0 && result_0 == 0) {
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
        if (macros > 0) {
            unsigned int phase_bits_7 = combine_bits;
            int col_blocks_8 = (hidden + 1023) / 1024;
            int macro_offset_1 = 0;
            int _min_556 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
            int macro_tokens_1 = _min_556;
            int num_tasks_1 = (macro_tokens_1 / 16 * col_blocks_8 + 6) / 7;
            if (tid == 0) {
                unsigned int _atomic_old_2 = atomicAdd(&combine_next[0], 1);
                combine_slot[0] = (int)_atomic_old_2;
            }
            __syncthreads();
            int task_3 = combine_slot[0];
            __syncthreads();
            while (task_3 < num_tasks_1) {
                unsigned int phase_bits_0_3 = phase_bits_7;
                int col_blocks_1_3 = (hidden + 1023) / 1024;
                int first_tile_3 = task_3 * 7;
                int macro_offset_2_3 = 0;
                int _min_557 = ((macro_size) < (tokens - macro_offset_2_3) ? (macro_size) : (tokens - macro_offset_2_3));
                int macro_tokens_3_3 = _min_557;
                int _min_558 = ((7) < (macro_tokens_3_3 / 16 * col_blocks_1_3 - first_tile_3) ? (7) : (macro_tokens_3_3 / 16 * col_blocks_1_3 - first_tile_3));
                int valid_tiles_2 = _min_558;
                if (valid_tiles_2 > 0) {
                    int first_row_5 = first_tile_3 / col_blocks_1_3 * 16 + tid;
                    int first_col_3 = first_tile_3 % col_blocks_1_3;
                    int rows_4[7];
                    int columns_2[7];
                    int peers_2[7];
                    int tokens_0_2[7];
                    unsigned int counts_1_2[7];
                    int row_45 = first_row_5;
                    int column_2 = first_col_3;
                    #pragma unroll
                    for (int stage_8 = 0; stage_8 < 7; stage_8++) {
                        rows_4[stage_8] = row_45;
                        columns_2[stage_8] = column_2;
                        peers_2[stage_8] = -1;
                        tokens_0_2[stage_8] = -1;
                        if (valid_tiles_2 > stage_8 && tid < 16) {
                            peers_2[stage_8] = schedule_rank[macro_offset_2_3 + row_45];
                            tokens_0_2[stage_8] = schedule_token[macro_offset_2_3 + row_45];
                        }
                        counts_1_2[stage_8] = 0;
                        if (valid_tiles_2 > stage_8) {
                            if (stage_8 == 0 || column_2 == 0) {
                                uint32_t _cta_count_6 = __syncthreads_count(peers_2[stage_8] >= 0);
                                counts_1_2[stage_8] = _cta_count_6;
                            } else {
                                counts_1_2[stage_8] = counts_1_2[stage_8 - 1];
                            }
                        }
                        column_2 = column_2 + 1;
                        if (column_2 == col_blocks_1_3) {
                            column_2 = 0;
                            row_45 = row_45 + 16;
                        }
                    }
                    if (tid == 0) {
                        int first_mini_2 = (macro_offset_2_3 + first_row_5) / mini_size;
                        int last_mini_2 = (macro_offset_2_3 + (first_tile_3 + valid_tiles_2 - 1) / col_blocks_1_3 * 16) / mini_size;
                        #pragma unroll 1
                        for (int mini_3 = first_mini_2; mini_3 < last_mini_2 + 1; mini_3++) {
                            int _min_559 = ((mini_size) < (tokens - mini_3 * mini_size) ? (mini_size) : (tokens - mini_3 * mini_size));
                            int mini_rows_9 = _min_559;
                            int required_9 = (mini_rows_9 + 255) / 256 * (hidden / 256) * 2;
                            int32_t _relaxed_ld_24;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_24) : "l"(y_ready + mini_3) : "memory");
                            int value_12 = _relaxed_ld_24;
                            while (value_12 < required_9) {
                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                int32_t _relaxed_ld_25;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_25) : "l"(y_ready + mini_3) : "memory");
                                value_12 = _relaxed_ld_25;
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                        }
                        #pragma unroll
                        for (int stage_9 = 0; stage_9 < 7; stage_9++) {
                            if (valid_tiles_2 > stage_9) {
                                int _min_560 = ((1024) < (hidden - columns_2[stage_9] * 1024) ? (1024) : (hidden - columns_2[stage_9] * 1024));
                                unsigned int chunk_bytes_5 = (unsigned int)(_min_560 * 2);
                                mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_9) * 8, counts_1_2[stage_9] * chunk_bytes_5);
                            }
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int stage_10 = 0; stage_10 < 7; stage_10++) {
                        if (peers_2[stage_10] >= 0) {
                            int _min_561 = ((1024) < (hidden - columns_2[stage_10] * 1024) ? (1024) : (hidden - columns_2[stage_10] * 1024));
                            int chunk_cols_4 = _min_561;
                            cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_10 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows_4[stage_10] * (unsigned long long)hidden + (unsigned long long)(columns_2[stage_10] * 1024)) * (unsigned long long)2)), chunk_cols_4 * 2, combine_arrived_addr + (stage_10) * 8);
                        }
                    }
                    #pragma unroll
                    for (int stage_11 = 0; stage_11 < 7; stage_11++) {
                        if (valid_tiles_2 > stage_11) {
                            mbarrier_wait(combine_arrived_addr + (stage_11) * 8, phase_bits_0_3 >> (unsigned int)stage_11 & 1);
                            phase_bits_0_3 = phase_bits_0_3 ^ (unsigned int)(1 << stage_11);
                            if (peers_2[stage_11] >= 0) {
                                int _min_562 = ((1024) < (hidden - columns_2[stage_11] * 1024) ? (1024) : (hidden - columns_2[stage_11] * 1024));
                                unsigned int chunk_bytes_6 = (unsigned int)(_min_562 * 2);
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_33 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[stage_11]]) + ((unsigned long long)tokens_0_2[stage_11] * (unsigned long long)hidden + (unsigned long long)(columns_2[stage_11] * 1024)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_33), "r"(combine_smem_addr + (unsigned int)((stage_11 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_6))
                                        : "memory");
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    int warp_2_1 = tid / 32;
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    __syncthreads();
                }
                phase_bits_7 = phase_bits_0_3;
                if (tid == 0) {
                    unsigned int _atomic_old_3 = atomicAdd(&combine_next[0], 1);
                    combine_slot[0] = (int)_atomic_old_3;
                }
                __syncthreads();
                task_3 = combine_slot[0];
                __syncthreads();
            }
            combine_bits = phase_bits_7;
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
        asm volatile("tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(576));
    }
}

} // extern "C"
