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
#define SMEM_HIDDEN_SMEM_OFF 197632
#define SMEM_HIDDEN_SMEM_STAGE_BYTES 32768
#define SMEM_HIDDEN_SMEM_STRIDE 32768
#define SMEM_HIDDEN_STAGING_OFF 197632
#define SMEM_HIDDEN_STAGING_STAGE_BYTES 34816
#define SMEM_HIDDEN_STAGING_STRIDE 34816
#define SMEM_HIDDEN_WORDS_OFF 197632
#define SMEM_HIDDEN_WORDS_STAGE_BYTES 34816
#define SMEM_HIDDEN_WORDS_STRIDE 34816
#define SMEM_SMEM_V9_OFF 1024
#define SMEM_SMEM_V9_STAGE_BYTES 16384
#define SMEM_SMEM_V9_STRIDE 16384
#define SMEM_SMEM_V10_OFF 17408
#define SMEM_SMEM_V10_STAGE_BYTES 512
#define SMEM_SMEM_V10_STRIDE 512
#define SMEM_SMEM_V11_OFF 99328
#define SMEM_SMEM_V11_STAGE_BYTES 16384
#define SMEM_SMEM_V11_STRIDE 16384
#define SMEM_SMEM_V12_OFF 115712
#define SMEM_SMEM_V12_STAGE_BYTES 512
#define SMEM_SMEM_V12_STRIDE 512
#define SMEM_SMEM_V13_OFF 33792
#define SMEM_SMEM_V13_STAGE_BYTES 16384
#define SMEM_SMEM_V13_STRIDE 16384
#define SMEM_SMEM_V14_OFF 50176
#define SMEM_SMEM_V14_STAGE_BYTES 512
#define SMEM_SMEM_V14_STRIDE 512
#define SMEM_SMEM_V15_OFF 132096
#define SMEM_SMEM_V15_STAGE_BYTES 16384
#define SMEM_SMEM_V15_STRIDE 16384
#define SMEM_SMEM_V16_OFF 148480
#define SMEM_SMEM_V16_STAGE_BYTES 512
#define SMEM_SMEM_V16_STRIDE 512
#define SMEM_SMEM_V17_OFF 66560
#define SMEM_SMEM_V17_STAGE_BYTES 16384
#define SMEM_SMEM_V17_STRIDE 16384
#define SMEM_SMEM_V18_OFF 82944
#define SMEM_SMEM_V18_STAGE_BYTES 512
#define SMEM_SMEM_V18_STRIDE 512
#define SMEM_SMEM_V19_OFF 164864
#define SMEM_SMEM_V19_STAGE_BYTES 16384
#define SMEM_SMEM_V19_STRIDE 16384
#define SMEM_SMEM_V20_OFF 181248
#define SMEM_SMEM_V20_STAGE_BYTES 512
#define SMEM_SMEM_V20_STRIDE 512
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 131072
#define SMEM_DISPATCH_WORDS_STRIDE 131072
#define SMEM_DISPATCH_WEIGHTS_OFF 199680
#define SMEM_DISPATCH_WEIGHTS_STAGE_BYTES 512
#define SMEM_DISPATCH_WEIGHTS_STRIDE 512
#define SMEM_SMEM_V23_OFF 1024
#define SMEM_SMEM_V23_STAGE_BYTES 131072
#define SMEM_SMEM_V23_STRIDE 131072
#define SMEM_SMEM_V24_OFF 1280
#define SMEM_SMEM_V24_STAGE_BYTES 130816
#define SMEM_SMEM_V24_STRIDE 130816
#define SMEM_SMEM_V25_OFF 1536
#define SMEM_SMEM_V25_STAGE_BYTES 130560
#define SMEM_SMEM_V25_STRIDE 130560
#define SMEM_SMEM_V26_OFF 1792
#define SMEM_SMEM_V26_STAGE_BYTES 130304
#define SMEM_SMEM_V26_STRIDE 130304
#define SMEM_SMEM_V27_OFF 132096
#define SMEM_SMEM_V27_STAGE_BYTES 16384
#define SMEM_SMEM_V27_STRIDE 16384
#define SMEM_SMEM_V28_OFF 148480
#define SMEM_SMEM_V28_STAGE_BYTES 16384
#define SMEM_SMEM_V28_STRIDE 16384
#define SMEM_SMEM_V29_OFF 164864
#define SMEM_SMEM_V29_STAGE_BYTES 512
#define SMEM_SMEM_V29_STRIDE 512
#define SMEM_SMEM_V30_OFF 165376
#define SMEM_SMEM_V30_STAGE_BYTES 512
#define SMEM_SMEM_V30_STRIDE 512
#define SMEM_SMEM_V31_OFF 165888
#define SMEM_SMEM_V31_STAGE_BYTES 16384
#define SMEM_SMEM_V31_STRIDE 16384
#define SMEM_SMEM_V32_OFF 182272
#define SMEM_SMEM_V32_STAGE_BYTES 16384
#define SMEM_SMEM_V32_STRIDE 16384
#define SMEM_SMEM_V33_OFF 198656
#define SMEM_SMEM_V33_STAGE_BYTES 512
#define SMEM_SMEM_V33_STRIDE 512
#define SMEM_SMEM_V34_OFF 199168
#define SMEM_SMEM_V34_STAGE_BYTES 512
#define SMEM_SMEM_V34_STRIDE 512
#define SMEM_SMEM_V35_OFF 1024
#define SMEM_SMEM_V35_STAGE_BYTES 65536
#define SMEM_SMEM_V35_STRIDE 65536
#define SMEM_SMEM_V36_OFF 1280
#define SMEM_SMEM_V36_STAGE_BYTES 65280
#define SMEM_SMEM_V36_STRIDE 65280
#define SMEM_SMEM_V37_OFF 66560
#define SMEM_SMEM_V37_STAGE_BYTES 65536
#define SMEM_SMEM_V37_STRIDE 65536
#define SMEM_SMEM_V38_OFF 66816
#define SMEM_SMEM_V38_STAGE_BYTES 65280
#define SMEM_SMEM_V38_STRIDE 65280
#define SMEM_SMEM_V39_OFF 1024
#define SMEM_SMEM_V39_STAGE_BYTES 16384
#define SMEM_SMEM_V39_STRIDE 16384
#define SMEM_SMEM_V40_OFF 66560
#define SMEM_SMEM_V40_STAGE_BYTES 16384
#define SMEM_SMEM_V40_STRIDE 16384
#define SMEM_SMEM_V41_OFF 132096
#define SMEM_SMEM_V41_STAGE_BYTES 16384
#define SMEM_SMEM_V41_STRIDE 16384
#define SMEM_SMEM_V42_OFF 197632
#define SMEM_SMEM_V42_STAGE_BYTES 512
#define SMEM_SMEM_V42_STRIDE 512
#define SMEM_SMEM_V43_OFF 199680
#define SMEM_SMEM_V43_STAGE_BYTES 1024
#define SMEM_SMEM_V43_STRIDE 1024
#define SMEM_SMEM_V44_OFF 203776
#define SMEM_SMEM_V44_STAGE_BYTES 1024
#define SMEM_SMEM_V44_STRIDE 1024
#define SMEM_SMEM_V45_OFF 1024
#define SMEM_SMEM_V45_STAGE_BYTES 16384
#define SMEM_SMEM_V45_STRIDE 16384
#define SMEM_SMEM_V46_OFF 99328
#define SMEM_SMEM_V46_STAGE_BYTES 16384
#define SMEM_SMEM_V46_STRIDE 16384
#define SMEM_SMEM_V47_OFF 197632
#define SMEM_SMEM_V47_STAGE_BYTES 512
#define SMEM_SMEM_V47_STRIDE 512
#define SMEM_SMEM_V48_OFF 200704
#define SMEM_SMEM_V48_STAGE_BYTES 1024
#define SMEM_SMEM_V48_STRIDE 1024
#define SMEM_SMEM_V49_OFF 207872
#define SMEM_SMEM_V49_STAGE_BYTES 16384
#define SMEM_SMEM_V49_STRIDE 16384
#define SMEM_SMEM_V50_OFF 224256
#define SMEM_SMEM_V50_STAGE_BYTES 4096
#define SMEM_SMEM_V50_STRIDE 4096
#define SMEM_SMEM_V51_OFF 228352
#define SMEM_SMEM_V51_STAGE_BYTES 512
#define SMEM_SMEM_V51_STRIDE 512
#define SMEM_SMEM_V52_OFF 228864
#define SMEM_SMEM_V52_STAGE_BYTES 512
#define SMEM_SMEM_V52_STRIDE 512
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 328

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
kernel_cake_mok_recompute_mxfp8(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap gate_shared_in, const __grid_constant__ CUtensorMap up_shared_in, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_q_store, const __grid_constant__ CUtensorMap hidden_sc_store, const __grid_constant__ CUtensorMap hidden_t_store, const __grid_constant__ CUtensorMap hidden_sc_t_store, unsigned long long* __restrict__ x_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit)
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
    __nv_bfloat16* hidden_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_smem_addr = smem + 197632;
    __nv_bfloat16* hidden_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_staging_addr = smem + 197632;
    unsigned int* hidden_words = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int hidden_words_addr = smem + 197632;
    unsigned int* smem_v9 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_v9_addr = smem + 1024;
    unsigned int* smem_v10 = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int smem_v10_addr = smem + 17408;
    unsigned int* smem_v11 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int smem_v11_addr = smem + 99328;
    unsigned int* smem_v12 = reinterpret_cast<unsigned int*>(smem_raw + 115712);
    const int smem_v12_addr = smem + 115712;
    unsigned int* smem_v13 = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int smem_v13_addr = smem + 33792;
    unsigned int* smem_v14 = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int smem_v14_addr = smem + 50176;
    unsigned int* smem_v15 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v15_addr = smem + 132096;
    unsigned int* smem_v16 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v16_addr = smem + 148480;
    unsigned int* smem_v17 = reinterpret_cast<unsigned int*>(smem_raw + 66560);
    const int smem_v17_addr = smem + 66560;
    unsigned int* smem_v18 = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int smem_v18_addr = smem + 82944;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v19_addr = smem + 164864;
    unsigned int* smem_v20 = reinterpret_cast<unsigned int*>(smem_raw + 181248);
    const int smem_v20_addr = smem + 181248;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int dispatch_words_addr = smem + 1024;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + 199680);
    const int dispatch_weights_addr = smem + 199680;
    __nv_bfloat16* smem_v23 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v23_addr = smem + 1024;
    __nv_bfloat16* smem_v24 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v24_addr = smem + 1280;
    __nv_bfloat16* smem_v25 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_v25_addr = smem + 1536;
    __nv_bfloat16* smem_v26 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1792);
    const int smem_v26_addr = smem + 1792;
    unsigned int* smem_v27 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v27_addr = smem + 132096;
    unsigned int* smem_v28 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v28_addr = smem + 148480;
    unsigned int* smem_v29 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v29_addr = smem + 164864;
    unsigned int* smem_v30 = reinterpret_cast<unsigned int*>(smem_raw + 165376);
    const int smem_v30_addr = smem + 165376;
    unsigned int* smem_v31 = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_v31_addr = smem + 165888;
    unsigned int* smem_v32 = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int smem_v32_addr = smem + 182272;
    unsigned int* smem_v33 = reinterpret_cast<unsigned int*>(smem_raw + 198656);
    const int smem_v33_addr = smem + 198656;
    unsigned int* smem_v34 = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int smem_v34_addr = smem + 199168;
    __nv_bfloat16* smem_v35 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v35_addr = smem + 1024;
    __nv_bfloat16* smem_v36 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v36_addr = smem + 1280;
    __nv_bfloat16* smem_v37 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v37_addr = smem + 66560;
    __nv_bfloat16* smem_v38 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66816);
    const int smem_v38_addr = smem + 66816;
    uint8_t* smem_v39 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v39_addr = smem + 1024;
    uint8_t* smem_v40 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_v40_addr = smem + 66560;
    uint8_t* smem_v41 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v41_addr = smem + 132096;
    uint8_t* smem_v42 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v42_addr = smem + 197632;
    uint8_t* smem_v43 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v43_addr = smem + 199680;
    uint8_t* smem_v44 = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_v44_addr = smem + 203776;
    uint8_t* smem_v45 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v45_addr = smem + 1024;
    uint8_t* smem_v46 = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v46_addr = smem + 99328;
    uint8_t* smem_v47 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v47_addr = smem + 197632;
    uint8_t* smem_v48 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_v48_addr = smem + 200704;
    unsigned int* smem_v49 = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int smem_v49_addr = smem + 207872;
    unsigned int* smem_v50 = reinterpret_cast<unsigned int*>(smem_raw + 224256);
    const int smem_v50_addr = smem + 224256;
    unsigned int* smem_v51 = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_v51_addr = smem + 228352;
    unsigned int* smem_v52 = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_v52_addr = smem + 228864;
    int tokens = num_tokens[0];
    int _min_0 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
    int routed_tokens = _min_0;
    int shared_rows = local_tokens / 256;
    int shared_gate = shared_rows * (intermediate / 256);
    int shared_gate_tasks = shared_rows * ((intermediate + 511) / 512);
    int mini_gate = mini_size / 256 * (intermediate / 256);
    int shared_swiglu = (local_tokens / 128 * (intermediate / 128) + 5) / 6;
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int shared_tasks = 2 * shared_gate_tasks + shared_swiglu;
    int mini_tasks = 2 * mini_gate + mini_swiglu;
    int comm_clusters = comm_sms / 2;
    int routed_minis = (routed_tokens + mini_size - 1) / mini_size;
    int true_clusters = comm_clusters + shared_tasks + routed_minis * mini_tasks;
    int i_tiles = intermediate / 128;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (27 pipeline groups, 0 ordered-sequence groups, 41 barriers)
    // Mbarriers at smem_raw[0..328)

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
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (576 columns, 572 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 328);
    if (warp == 0) {
        int _tmem_hold = smem + 328;
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
    const int tmem_accumulator = taddr;
    unsigned int taddr_1 = reinterpret_cast<const volatile unsigned int*>(reinterpret_cast<uint8_t*>(smem_raw) + CAKE_TMEM_HOLD_OFFSET)[0];
    int cluster = bid / 2;
    int cta_rank_0 = cta_rank;
    unsigned int gemm_bits = 4294901760;
    unsigned int swiglu_bits = 4294901760;
    unsigned int dispatch_bits = 4294901760;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        unsigned int phase_bits = dispatch_bits;
        int col_blocks = (hidden + 511) / 512;
        int macro_offset = 0;
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
                cp_async_bulk_gmem2smem(smem_v35_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(task % col_blocks * 512)) * (unsigned long long)2)), _min_2 * 2, dispatch_arrived_addr);
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
                    cp_async_bulk_gmem2smem(smem_v37_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512) + 256) * (unsigned long long)2)), hi_cols * 2, dispatch_arrived_hi_addr);
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
                            x0 = (float)smem_v35[col * 256 + row_0];
                            x1 = (float)smem_v35[(col + 1) * 256 + row_0];
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
                            smem_v31[(row_0 * 128 + col_1) / 4] = words[k_1];
                        }
                    }
                    smem_v33[row_0 % 32 * 4 + row_0 / 32] = scale_word;
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
                            smem_v27[(row_0_1 * 128 + col_3) / 4] = words_1[k_3];
                        }
                    }
                    smem_v29[row_0_1 % 32 * 4 + row_0_1 / 32] = scale_word_1;
                }
                __syncthreads();
                if (tid == 0) {
                    int col_tile = col_block * 4;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d((&x_q_store), col_tile * 128, row, smem_v27_addr);
                    tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile, smem_v29_addr);
                    tma_store_2d((&x_t_store), row, col_tile * 128, smem_v31_addr);
                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile * macro_tiles + row_tile, smem_v33_addr);
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
                            x0_2 = (float)smem_v36[col_4 * 256 + row_0_2];
                            x1_2 = (float)smem_v36[(col_4 + 1) * 256 + row_0_2];
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
                            smem_v32[(row_0_2 * 128 + col_5) / 4] = words_2[k_5];
                        }
                    }
                    smem_v34[row_0_2 % 32 * 4 + row_0_2 / 32] = scale_word_2;
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
                            smem_v28[(row_0_3 * 128 + col_7) / 4] = words_3[k_7];
                        }
                    }
                    smem_v30[row_0_3 % 32 * 4 + row_0_3 / 32] = scale_word_3;
                }
                __syncthreads();
                if (tid == 0) {
                    int col_tile_1 = col_block * 4 + 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d((&x_q_store), col_tile_1 * 128, row, smem_v28_addr);
                    tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile_1, smem_v30_addr);
                    tma_store_2d((&x_t_store), row, col_tile_1 * 128, smem_v32_addr);
                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_1 * macro_tiles + row_tile, smem_v34_addr);
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
                    cp_async_bulk_gmem2smem(smem_v35_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task % col_blocks * 512)) * (unsigned long long)2)), _min_5 * 2, dispatch_arrived_addr);
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
                                x0_4 = (float)smem_v37[col_8 * 256 + row_0_4];
                                x1_4 = (float)smem_v37[(col_8 + 1) * 256 + row_0_4];
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
                                smem_v31[(row_0_4 * 128 + col_9) / 4] = words_4[k_9];
                            }
                        }
                        smem_v33[row_0_4 % 32 * 4 + row_0_4 / 32] = scale_word_4;
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
                                smem_v27[(row_0_5 * 128 + col_11) / 4] = words_5[k_11];
                            }
                        }
                        smem_v29[row_0_5 % 32 * 4 + row_0_5 / 32] = scale_word_5;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_2 = col_block * 4 + 2;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_2 * 128, row, smem_v27_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_2 + col_tile_2, smem_v29_addr);
                        tma_store_2d((&x_t_store), row, col_tile_2 * 128, smem_v31_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_2 * macro_tiles_3 + row_tile_1, smem_v33_addr);
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
                                x0_6 = (float)smem_v38[col_12 * 256 + row_0_6];
                                x1_6 = (float)smem_v38[(col_12 + 1) * 256 + row_0_6];
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
                                smem_v32[(row_0_6 * 128 + col_13) / 4] = words_6[k_13];
                            }
                        }
                        smem_v34[row_0_6 % 32 * 4 + row_0_6 / 32] = scale_word_6;
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
                                smem_v28[(row_0_7 * 128 + col_15) / 4] = words_7[k_15];
                            }
                        }
                        smem_v30[row_0_7 % 32 * 4 + row_0_7 / 32] = scale_word_7;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_3 = col_block * 4 + 2 + 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_3 * 128, row, smem_v28_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_2 + col_tile_3, smem_v30_addr);
                        tma_store_2d((&x_t_store), row, col_tile_3 * 128, smem_v32_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_3 * macro_tiles_3 + row_tile_1, smem_v34_addr);
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
                    if (compute >= 2 * shared_gate_tasks && compute < 2 * shared_gate_tasks + shared_swiglu) {
                        result = 1;
                    }
                } else {
                    int mini_task = (compute - shared_tasks) % mini_tasks;
                    if (mini_task >= 2 * mini_gate && mini_task < 2 * mini_gate + mini_swiglu) {
                        result = 1;
                    }
                }
            }
            if (compute < shared_gate_tasks) {
                int col_blocks_1 = (intermediate + 512 - 1) / 512;
                int x = -1;
                int y = -1;
                int expert = -1;
                int k_start = 0;
                int k_end = 0;
                int first = 0;
                int row_blocks = local_tokens / 256;
                if (compute < row_blocks * col_blocks_1) {
                    int supergroup = compute / (row_blocks * 8);
                    int full_cols = col_blocks_1 / 8 * 8;
                    int row_1 = 0;
                    int col_16 = 0;
                    if (compute < row_blocks * full_cols) {
                        row_1 = compute % (row_blocks * 8) / 8;
                        col_16 = supergroup * 8 + compute % 8;
                    } else {
                        row_1 = (compute - row_blocks * full_cols) / (col_blocks_1 - full_cols);
                        col_16 = full_cols + (compute - row_blocks * full_cols) % (col_blocks_1 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_1 = row_blocks - row_1 - 1;
                    }
                    x = row_1;
                    y = col_16;
                    expert = 0;
                }
                unsigned int phase_bits_1 = gemm_bits;
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
                                int _min_6 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                int _max_0 = ((0) > (_min_6) ? (0) : (_min_6));
                                int mini_rows = _max_0;
                                int required = (mini_rows + 127) / 128 * ((hidden + 511) / 512);
                            }
                            int ring = 0;
                            #pragma unroll 1
                            for (int idx = 0; idx < iterations; idx++) {
                                mbarrier_wait(gemm_finished_addr + (ring) * 8, phase_bits_1 >> (unsigned int)(16 + ring) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring * 16384)), "l"((&x_shared)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring * 16384)), "l"((&wg_shared)), "r"(0), "r"(y * 2 * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)(ring * 16384)), "l"((&wg_shared)), "r"(0), "r"((y * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_1 = phase_bits_1 ^ (unsigned int)(1 << 16 + ring);
                                ring = (ring + 1) % 4;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_1 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_1 >> 22 & 1);
                                phase_bits_1 = phase_bits_1 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_1 = 0; idx_1 < iterations; idx_1++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_1) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_1) * 8, phase_bits_1 >> (unsigned int)ring_1 & 1);
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
                                    phase_bits_1 = phase_bits_1 ^ (unsigned int)(1 << ring_1);
                                    ring_1 = (ring_1 + 1) % 4;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        mbarrier_wait(output_arrived_addr, phase_bits_1 >> 6 & 1);
                        phase_bits_1 = phase_bits_1 ^ 64;
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
                                    __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]));
                                    packed[chunk * 16 + sub * 8 + pair] = __as_u32(_bf16x2_4);
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
                            int _min_7 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                            if (output_row < _min_7) {
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
                                for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                    int row_2 = warp_0 * 32 + half * 16 + lane_1 % 16;
                                    int col_17 = col_tile_4 * 16 + lane_1 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_2 * 32 + col_17) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset = chunk_1 * 16 + half * 8 + col_tile_4 * 4;
                                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 2 * 8 + chunk_1), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_1[pair_1 * 2], _tmem_load_1[pair_1 * 2 + 1]));
                                        packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(_bf16x2_5);
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
                                    for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                        int row_3 = warp_0_1 * 32 + half_1 * 16 + lane_2 % 16;
                                        int col_18 = col_tile_5 * 16 + lane_2 / 16 * 8;
                                        unsigned int address_3 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_3 * 32 + col_18) * 2);
                                        address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                        int offset_1 = chunk_3 * 16 + half_1 * 8 + col_tile_5 * 4;
                                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(address_3);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"((y * 2 + 1) * 8 + chunk_3), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                    bool enabled_value = 1;
                                    if (enabled_value != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows + x) * (intermediate / 256) + y * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                    if (has_hi != 0) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_0 = 1;
                                        if (enabled_value_0 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows + x) * (intermediate / 256) + y * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gemm_bits = phase_bits_1;
            } else if (compute < 2 * shared_gate_tasks) {
                int col_blocks_2 = (intermediate + 512 - 1) / 512;
                int x_1 = -1;
                int y_1 = -1;
                int expert_1 = -1;
                int k_start_1 = 0;
                int k_end_1 = 0;
                int first_1 = 0;
                int row_blocks_1 = local_tokens / 256;
                if (compute - shared_gate_tasks < row_blocks_1 * col_blocks_2) {
                    int supergroup_1 = (compute - shared_gate_tasks) / (row_blocks_1 * 8);
                    int full_cols_1 = col_blocks_2 / 8 * 8;
                    int row_4 = 0;
                    int col_19 = 0;
                    if (compute - shared_gate_tasks < row_blocks_1 * full_cols_1) {
                        row_4 = (compute - shared_gate_tasks) % (row_blocks_1 * 8) / 8;
                        col_19 = supergroup_1 * 8 + (compute - shared_gate_tasks) % 8;
                    } else {
                        row_4 = (compute - shared_gate_tasks - row_blocks_1 * full_cols_1) / (col_blocks_2 - full_cols_1);
                        col_19 = full_cols_1 + (compute - shared_gate_tasks - row_blocks_1 * full_cols_1) % (col_blocks_2 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_4 = row_blocks_1 - row_4 - 1;
                    }
                    x_1 = row_4;
                    y_1 = col_19;
                    expert_1 = 0;
                }
                unsigned int phase_bits_2 = gemm_bits;
                int has_hi_1 = 0;
                has_hi_1 = (int)((y_1 * 2 + 1) * 256 < intermediate);
                int global_mini_1 = 0;
                int macro_rows_1 = 0;
                int iterations_1 = hidden / 64;
                if (expert_1 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_8 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                int _max_1 = ((0) > (_min_8) ? (0) : (_min_8));
                                int mini_rows_1 = _max_1;
                                int required_1 = (mini_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                            }
                            int ring_2 = 0;
                            #pragma unroll 1
                            for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                mbarrier_wait(gemm_finished_addr + (ring_2) * 8, phase_bits_2 >> (unsigned int)(16 + ring_2) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&x_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&wu_shared)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)(ring_2 * 16384)), "l"((&wu_shared)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_2 = phase_bits_2 ^ (unsigned int)(1 << 16 + ring_2);
                                ring_2 = (ring_2 + 1) % 4;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_3 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_2 >> 22 & 1);
                                phase_bits_2 = phase_bits_2 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_3 = 0; idx_3 < iterations_1; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_2 >> (unsigned int)ring_3 & 1);
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
                                    phase_bits_2 = phase_bits_2 ^ (unsigned int)(1 << ring_3);
                                    ring_3 = (ring_3 + 1) % 4;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        mbarrier_wait(output_arrived_addr, phase_bits_2 >> 6 & 1);
                        phase_bits_2 = phase_bits_2 ^ 64;
                        unsigned int packed_1[128];
                        #pragma unroll
                        for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                            #pragma unroll
                            for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                unsigned int address_4 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                float _tmem_load_2[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                    : "r"(address_4));
                                #pragma unroll
                                for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                                    __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_tmem_load_2[pair_2 * 2], _tmem_load_2[pair_2 * 2 + 1]));
                                    packed_1[chunk_4 * 16 + sub_2 * 8 + pair_2] = __as_u32(_bf16x2_6);
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
                            int _min_9 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            if (output_row_1 < _min_9) {
                            }
                        }
                        #pragma unroll
                        for (int chunk_5 = 0; chunk_5 < 8; chunk_5++) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_0_2 = tid / 32;
                            int lane_3 = tid % 32;
                            #pragma unroll
                            for (int half_2 = 0; half_2 < 2; half_2++) {
                                #pragma unroll
                                for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                    int row_5 = warp_0_2 * 32 + half_2 * 16 + lane_3 % 16;
                                    int col_20 = col_tile_6 * 16 + lane_3 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_5 * 32 + col_20) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_2 = chunk_5 * 16 + half_2 * 8 + col_tile_6 * 4;
                                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_shared_out)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 2 * 8 + chunk_5), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (has_hi_1 != 0) {
                            unsigned int packed_0_1[128];
                            #pragma unroll
                            for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                #pragma unroll
                                for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                    unsigned int address_6 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                    float _tmem_load_3[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                        : "r"(address_6));
                                    #pragma unroll
                                    for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_3[pair_3 * 2], _tmem_load_3[pair_3 * 2 + 1]));
                                        packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_3] = __as_u32(_bf16x2_7);
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
                                int warp_0_3 = tid / 32;
                                int lane_4 = tid % 32;
                                #pragma unroll
                                for (int half_3 = 0; half_3 < 2; half_3++) {
                                    #pragma unroll
                                    for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                        int row_6 = warp_0_3 * 32 + half_3 * 16 + lane_4 % 16;
                                        int col_21 = col_tile_7 * 16 + lane_4 / 16 * 8;
                                        unsigned int address_7 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_6 * 32 + col_21) * 2);
                                        address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                        int offset_3 = chunk_7 * 16 + half_3 * 8 + col_tile_7 * 4;
                                        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(address_7);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_shared_out)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"((y_1 * 2 + 1) * 8 + chunk_7), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                    bool enabled_value_1 = 1;
                                    if (enabled_value_1 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows_1 + x_1) * (intermediate / 256) + y_1 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                    if (has_hi_1 != 0) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_0_1 = 1;
                                        if (enabled_value_0_1 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows_1 + x_1) * (intermediate / 256) + y_1 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gemm_bits = phase_bits_2;
            } else {
                if (compute < shared_tasks) {
                    unsigned int phase_bits_3 = swiglu_bits;
                    int col_blocks_3 = intermediate / 128;
                    int num_tiles = local_tokens / 128 * col_blocks_3;
                    int macro_row_offset = 0;
                    int first_tile = (compute - 2 * shared_gate_tasks) * 6 + cta_rank_0 * 3;
                    int tile_end = num_tiles;
                    if (first_tile < tile_end) {
                        int first_row = first_tile / col_blocks_3;
                        int first_col = first_tile % col_blocks_3;
                        if (tid == 0) {
                            #pragma unroll
                            for (int stage = 0; stage < 3; stage++) {
                                if (tile_end > first_tile + stage) {
                                    int row_7 = first_row;
                                    int col_22 = first_col + stage;
                                    if (col_22 >= col_blocks_3) {
                                        row_7 = row_7 + 1;
                                        col_22 = col_22 - col_blocks_3;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage) * 8, 65536);
                                    int parent = row_7 / 2 * (intermediate / 256) + col_22 / 2;
                                    int32_t _relaxed_ld_0;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(gate_ready + parent) : "memory");
                                    int value = _relaxed_ld_0;
                                    while (value < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_1;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_1) : "l"(gate_ready + parent) : "memory");
                                        value = _relaxed_ld_1;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + (unsigned int)(stage * 32768)), "l"((&gate_shared_in)), "r"(0), "r"((row_7 - macro_row_offset) * 128), "r"(col_22 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + (unsigned int)(stage * 32768)), "l"((&up_shared_in)), "r"(0), "r"((row_7 - macro_row_offset) * 128), "r"(col_22 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage) * 8) : "memory");
                                }
                            }
                        }
                        #pragma unroll 3
                        for (int stage_1 = 0; stage_1 < 3; stage_1++) {
                            if (tile_end > first_tile + stage_1) {
                                mbarrier_wait(swiglu_arrived_addr + (stage_1) * 8, phase_bits_3 >> (unsigned int)stage_1 & 1);
                                phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << stage_1);
                                int row_8 = first_row;
                                int col_23 = first_col + stage_1;
                                if (col_23 >= col_blocks_3) {
                                    row_8 = row_8 + 1;
                                    col_23 = col_23 - col_blocks_3;
                                }
                                float gate[64];
                                float up[64];
                                float denominator[64];
                                int warp_0_4 = tid / 32;
                                int local_warp = warp_0_4 / 4 + warp_0_4 % 4 * 2;
                                int lane_5 = tid % 32;
                                #pragma unroll
                                for (int tile_col = 0; tile_col < 8; tile_col++) {
                                    unsigned int packed_2[4];
                                    unsigned int address_8 = gate_smem_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col * 16 + lane_5 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_5 % 16) * 64 + (tile_col * 16 + lane_5 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                        : "r"(address_8 ^ (address_8 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                        float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(packed_2[pair_4]));
                                        gate[tile_col * 8 + pair_4 * 2] = _cvt_f32_0.x;
                                        gate[tile_col * 8 + pair_4 * 2 + 1] = _cvt_f32_0.y;
                                    }
                                }
                                int warp_1 = tid / 32;
                                int local_warp_2 = warp_1 / 4 + warp_1 % 4 * 2;
                                int lane_3_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                    unsigned int packed_3[4];
                                    unsigned int address_9 = up_smem_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3_1 % 16) * 64 + (tile_col_1 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                        : "r"(address_9 ^ (address_9 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                        float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(packed_3[pair_5]));
                                        up[tile_col_1 * 8 + pair_5 * 2] = _cvt_f32_1.x;
                                        up[tile_col_1 * 8 + pair_5 * 2 + 1] = _cvt_f32_1.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem = 0; elem < 64; elem++) {
                                    denominator[elem] = gate[elem] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                    float _exp_0 = expf(denominator[elem_1]);
                                    denominator[elem_1] = _exp_0;
                                }
                                #pragma unroll
                                for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                    denominator[elem_2] = denominator[elem_2] + 1.0f;
                                }
                                #pragma unroll
                                for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                    gate[elem_3] = gate[elem_3] / denominator[elem_3];
                                }
                                #pragma unroll
                                for (int elem_4 = 0; elem_4 < 64; elem_4++) {
                                    gate[elem_4] = gate[elem_4] * up[elem_4];
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                __syncthreads();
                                int warp_4 = tid / 32;
                                int local_warp_5 = warp_4 / 4 + warp_4 % 4 * 2;
                                int lane_6 = tid % 32;
                                #pragma unroll
                                for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                    unsigned int packed_4[4];
                                    #pragma unroll
                                    for (int pair_6 = 0; pair_6 < 4; pair_6++) {
                                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_6 * 2], gate[tile_col_2 * 8 + pair_6 * 2 + 1]));
                                        packed_4[pair_6] = __as_u32(_bf16x2_8);
                                    }
                                    unsigned int address_10 = hidden_smem_addr + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_10 ^ (address_10 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&hidden_shared_out), 0, (row_8 - macro_row_offset) * 128, col_23 * 2, 0, 0, hidden_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 3; stage_2++) {
                                if (tile_end > first_tile + stage_2) {
                                    int row_9 = first_row;
                                    if (col_blocks_3 <= first_col + stage_2) {
                                        row_9 = row_9 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (row_9 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                    }
                    swiglu_bits = phase_bits_3;
                } else {
                    int mini = (compute - shared_tasks) / mini_tasks;
                    int task_1 = (compute - shared_tasks) % mini_tasks;
                    if (task_1 < mini_gate) {
                        int col_blocks_4 = (intermediate + 256 - 1) / 256;
                        int x_2 = -1;
                        int y_2 = -1;
                        int expert_2 = -1;
                        int k_start_2 = 0;
                        int k_end_2 = 0;
                        int first_2 = 0;
                        int first_block = mini * (mini_size / 256);
                        int _min_11 = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                        int end_block = _min_11;
                        int block = first_block + task_1 / col_blocks_4;
                        if (block < end_block) {
                            int index = counts[3 * experts + block];
                            int offset_4 = counts[experts + index] / 256;
                            int _max_2 = ((first_block) > (offset_4) ? (first_block) : (offset_4));
                            int first_row_1 = _max_2;
                            int _min_12 = ((end_block) < (offset_4 + counts[index] / 256) ? (end_block) : (offset_4 + counts[index] / 256));
                            int rows = _min_12 - first_row_1;
                            int supergroup_2 = (task_1 - (first_row_1 - first_block) * col_blocks_4) / (rows * 8);
                            int full_cols_2 = col_blocks_4 / 8 * 8;
                            int row_10 = 0;
                            int col_24 = 0;
                            if (task_1 - (first_row_1 - first_block) * col_blocks_4 < rows * full_cols_2) {
                                row_10 = (task_1 - (first_row_1 - first_block) * col_blocks_4) % (rows * 8) / 8;
                                col_24 = supergroup_2 * 8 + (task_1 - (first_row_1 - first_block) * col_blocks_4) % 8;
                            } else {
                                row_10 = (task_1 - (first_row_1 - first_block) * col_blocks_4 - rows * full_cols_2) / (col_blocks_4 - full_cols_2);
                                col_24 = full_cols_2 + (task_1 - (first_row_1 - first_block) * col_blocks_4 - rows * full_cols_2) % (col_blocks_4 - full_cols_2);
                            }
                            if ((supergroup_2 & 1) != 0) {
                                row_10 = rows - row_10 - 1;
                            }
                            x_2 = first_row_1 + row_10;
                            y_2 = col_24;
                            expert_2 = index;
                        }
                        unsigned int phase_bits_4 = gemm_bits;
                        int has_hi_2 = 0;
                        int global_mini_2 = mini;
                        int macro_rows_2 = 0;
                        int iterations_2 = hidden / 128;
                        int macro_k = 0;
                        if (expert_2 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_13 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                        int _max_3 = ((0) > (_min_13) ? (0) : (_min_13));
                                        int mini_rows_2 = _max_3;
                                        int required_2 = (mini_rows_2 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_2 = 1;
                                        if (enabled_value_2 != 0) {
                                            int32_t _relaxed_ld_2;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_2) : "l"(x_ready + global_mini_2) : "memory");
                                            int value_1 = _relaxed_ld_2;
                                            while (value_1 < required_2) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_3;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_3) : "l"(x_ready + global_mini_2) : "memory");
                                                value_1 = _relaxed_ld_3;
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
                                            :: "r"(smem_v39_addr + (unsigned int)(ring_4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v40_addr + (unsigned int)(ring_4 * 16384)), "l"((&wg_q)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
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
                                            int _min_14 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                            int _max_4 = ((0) > (_min_14) ? (0) : (_min_14));
                                            int mini_rows_3 = _max_4;
                                            int required_3 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_3 = 1;
                                            if (enabled_value_3 != 0) {
                                                int32_t _relaxed_ld_4;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(x_ready + global_mini_2) : "memory");
                                                int value_2 = _relaxed_ld_4;
                                                while (value_2 < required_3) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_5;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(x_ready + global_mini_2) : "memory");
                                                    value_2 = _relaxed_ld_5;
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
                                                :: "r"(smem_v42_addr + (unsigned int)(ring_5 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                                   "r"(((scales_arrived_addr + (ring_5) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v43_addr + (unsigned int)(ring_5 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((expert_2 * (intermediate / 128) + y_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
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
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_6) * 8, 3072);
                                            mbarrier_wait(scales_arrived_addr + (ring_6) * 8, phase_bits_4 >> (unsigned int)(8 + ring_6) & 1);
                                            int buffer = idx_6 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer * 4, make_sf_cp_desc_sbo128(smem_v42_addr + (unsigned int)(ring_6 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer * 8, make_sf_cp_desc_sbo128(smem_v43_addr + (unsigned int)(ring_6 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer * 8 + 4), make_sf_cp_desc_sbo128((smem_v43_addr + (unsigned int)(ring_6 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_6) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_6) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_6) * 8, phase_bits_4 >> (unsigned int)ring_6 & 1);
                                            int _mma_a_lo_4 = (((smem_v39_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                            int _mma_b_lo_4 = (((smem_v40_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb + buffer * 8 + 0)), ((idx_6 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer * 8 + 0) | 0x80000000), 1);
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
                                    unsigned int packed_5[128];
                                    #pragma unroll
                                    for (int i_16 = 0; i_16 < 8; i_16++) {
                                        float _tmem_load_4[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                                            : "r"(taddr_1 + (unsigned int)(warp_row << 16) + (unsigned int)(i_16 * 32)));
                                        #pragma unroll
                                        for (int j_8 = 0; j_8 < 16; j_8++) {
                                            __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_tmem_load_4[2 * j_8], _tmem_load_4[2 * j_8 + 1]));
                                            packed_5[i_16 * 16 + j_8] = __as_u32(_bf16x2_9);
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                    unsigned int scale_word_8 = 0;
                                    unsigned int block_0[16];
                                    #pragma unroll
                                    for (int j_9 = 0; j_9 < 16; j_9++) {
                                        block_0[j_9] = packed_5[j_9];
                                    }
                                    #pragma unroll
                                    for (int j_10 = 0; j_10 < 4; j_10++) {
                                        unsigned int address_11 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_10 * 16);
                                        address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_11 - smem_v49_addr)), "r"(packed_5[4 * j_10]), "r"(packed_5[4 * j_10 + 1]), "r"(packed_5[4 * j_10 + 2]), "r"(packed_5[4 * j_10 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_16;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(block_0[0]));
                                    unsigned int amax_pair_8 = _bf16x2_abs_16;
                                    #pragma unroll
                                    for (int i_17 = 1; i_17 < 16; i_17++) {
                                        uint32_t _bf16x2_abs_17;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(block_0[i_17]));
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
                                    for (int i_18 = 0; i_18 < 8; i_18++) {
                                        uint32_t _bf16x2_mul_16;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(block_0[i_18 * 2]), "r"(inverse_8));
                                        uint16_t _e4m3x2_16;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_16) : "r"(_bf16x2_mul_16));
                                        uint32_t _bf16x2_mul_17;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(block_0[i_18 * 2 + 1]), "r"(inverse_8));
                                        uint16_t _e4m3x2_17;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_17) : "r"(_bf16x2_mul_17));
                                        words_8[i_18] = (unsigned int)_e4m3x2_16 | (unsigned int)_e4m3x2_17 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_8[0]), "r"(words_8[1]), "r"(words_8[2]), "r"(words_8[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8[4]), "r"(words_8[5]), "r"(words_8[6]), "r"(words_8[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_1[16];
                                    #pragma unroll
                                    for (int j_11 = 0; j_11 < 16; j_11++) {
                                        block_1[j_11] = packed_5[16 + j_11];
                                    }
                                    #pragma unroll
                                    for (int j_12 = 0; j_12 < 4; j_12++) {
                                        unsigned int address_12 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_12 * 16);
                                        address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_12 - smem_v49_addr)), "r"(packed_5[16 + 4 * j_12]), "r"(packed_5[16 + 4 * j_12 + 1]), "r"(packed_5[16 + 4 * j_12 + 2]), "r"(packed_5[16 + 4 * j_12 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_18;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(block_1[0]));
                                    unsigned int amax_pair_2_1 = _bf16x2_abs_18;
                                    #pragma unroll
                                    for (int i_19 = 1; i_19 < 16; i_19++) {
                                        uint32_t _bf16x2_abs_19;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(block_1[i_19]));
                                        uint32_t _bf16x2_max_9;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(amax_pair_2_1), "r"(_bf16x2_abs_19));
                                        amax_pair_2_1 = _bf16x2_max_9;
                                    }
                                    uint16_t _bf16_max_9;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_9) : "h"((uint16_t)(amax_pair_2_1 & 65535)), "h"((uint16_t)(amax_pair_2_1 >> 16)));
                                    float _cvt_f32_bf16_9;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_9) : "h"((uint16_t)(_bf16_max_9)));
                                    float amax_3_1 = _cvt_f32_bf16_9;
                                    float _fmax_9 = fmaxf(amax_3_1 * 0.002232142857f, 1e-12f);
                                    float scale_4_1 = _fmax_9;
                                    uint16_t _ue8m0x2_f32_9;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_9) : "f"(scale_4_1), "f"(scale_4_1));
                                    unsigned int scale_byte_5_1 = (unsigned int)_ue8m0x2_f32_9 & 255;
                                    unsigned int inverse_lane_6_1 = 254 - scale_byte_5_1 << 7;
                                    unsigned int inverse_7_1 = inverse_lane_6_1 | inverse_lane_6_1 << 16;
                                    unsigned int words_8_1[8];
                                    #pragma unroll
                                    for (int i_20 = 0; i_20 < 8; i_20++) {
                                        uint32_t _bf16x2_mul_18;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(block_1[i_20 * 2]), "r"(inverse_7_1));
                                        uint16_t _e4m3x2_18;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_18) : "r"(_bf16x2_mul_18));
                                        uint32_t _bf16x2_mul_19;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(block_1[i_20 * 2 + 1]), "r"(inverse_7_1));
                                        uint16_t _e4m3x2_19;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_19) : "r"(_bf16x2_mul_19));
                                        words_8_1[i_20] = (unsigned int)_e4m3x2_18 | (unsigned int)_e4m3x2_19 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_5_1 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_8_1[0]), "r"(words_8_1[1]), "r"(words_8_1[2]), "r"(words_8_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_1[4]), "r"(words_8_1[5]), "r"(words_8_1[6]), "r"(words_8_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 32), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_9[16];
                                    #pragma unroll
                                    for (int j_13 = 0; j_13 < 16; j_13++) {
                                        block_9[j_13] = packed_5[32 + j_13];
                                    }
                                    #pragma unroll
                                    for (int j_14 = 0; j_14 < 4; j_14++) {
                                        unsigned int address_13 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_14 * 16);
                                        address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_13 - smem_v49_addr)), "r"(packed_5[32 + 4 * j_14]), "r"(packed_5[32 + 4 * j_14 + 1]), "r"(packed_5[32 + 4 * j_14 + 2]), "r"(packed_5[32 + 4 * j_14 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_20;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(block_9[0]));
                                    unsigned int amax_pair_10 = _bf16x2_abs_20;
                                    #pragma unroll
                                    for (int i_21 = 1; i_21 < 16; i_21++) {
                                        uint32_t _bf16x2_abs_21;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(block_9[i_21]));
                                        uint32_t _bf16x2_max_10;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(amax_pair_10), "r"(_bf16x2_abs_21));
                                        amax_pair_10 = _bf16x2_max_10;
                                    }
                                    uint16_t _bf16_max_10;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_10) : "h"((uint16_t)(amax_pair_10 & 65535)), "h"((uint16_t)(amax_pair_10 >> 16)));
                                    float _cvt_f32_bf16_10;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_10) : "h"((uint16_t)(_bf16_max_10)));
                                    float amax_11 = _cvt_f32_bf16_10;
                                    float _fmax_10 = fmaxf(amax_11 * 0.002232142857f, 1e-12f);
                                    float scale_12 = _fmax_10;
                                    uint16_t _ue8m0x2_f32_10;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_10) : "f"(scale_12), "f"(scale_12));
                                    unsigned int scale_byte_13 = (unsigned int)_ue8m0x2_f32_10 & 255;
                                    unsigned int inverse_lane_14 = 254 - scale_byte_13 << 7;
                                    unsigned int inverse_15 = inverse_lane_14 | inverse_lane_14 << 16;
                                    unsigned int words_16[8];
                                    #pragma unroll
                                    for (int i_22 = 0; i_22 < 8; i_22++) {
                                        uint32_t _bf16x2_mul_20;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(block_9[i_22 * 2]), "r"(inverse_15));
                                        uint16_t _e4m3x2_20;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_20) : "r"(_bf16x2_mul_20));
                                        uint32_t _bf16x2_mul_21;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(block_9[i_22 * 2 + 1]), "r"(inverse_15));
                                        uint16_t _e4m3x2_21;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_21) : "r"(_bf16x2_mul_21));
                                        words_16[i_22] = (unsigned int)_e4m3x2_20 | (unsigned int)_e4m3x2_21 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_13 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_16[0]), "r"(words_16[1]), "r"(words_16[2]), "r"(words_16[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16[4]), "r"(words_16[5]), "r"(words_16[6]), "r"(words_16[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 64), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_17[16];
                                    #pragma unroll
                                    for (int j_15 = 0; j_15 < 16; j_15++) {
                                        block_17[j_15] = packed_5[48 + j_15];
                                    }
                                    #pragma unroll
                                    for (int j_16 = 0; j_16 < 4; j_16++) {
                                        unsigned int address_14 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_16 * 16);
                                        address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_14 - smem_v49_addr)), "r"(packed_5[48 + 4 * j_16]), "r"(packed_5[48 + 4 * j_16 + 1]), "r"(packed_5[48 + 4 * j_16 + 2]), "r"(packed_5[48 + 4 * j_16 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_22;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(block_17[0]));
                                    unsigned int amax_pair_18 = _bf16x2_abs_22;
                                    #pragma unroll
                                    for (int i_23 = 1; i_23 < 16; i_23++) {
                                        uint32_t _bf16x2_abs_23;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(block_17[i_23]));
                                        uint32_t _bf16x2_max_11;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(amax_pair_18), "r"(_bf16x2_abs_23));
                                        amax_pair_18 = _bf16x2_max_11;
                                    }
                                    uint16_t _bf16_max_11;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_11) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                    float _cvt_f32_bf16_11;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_11) : "h"((uint16_t)(_bf16_max_11)));
                                    float amax_19 = _cvt_f32_bf16_11;
                                    float _fmax_11 = fmaxf(amax_19 * 0.002232142857f, 1e-12f);
                                    float scale_20 = _fmax_11;
                                    uint16_t _ue8m0x2_f32_11;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_11) : "f"(scale_20), "f"(scale_20));
                                    unsigned int scale_byte_21 = (unsigned int)_ue8m0x2_f32_11 & 255;
                                    unsigned int inverse_lane_22 = 254 - scale_byte_21 << 7;
                                    unsigned int inverse_23 = inverse_lane_22 | inverse_lane_22 << 16;
                                    unsigned int words_24[8];
                                    #pragma unroll
                                    for (int i_24 = 0; i_24 < 8; i_24++) {
                                        uint32_t _bf16x2_mul_22;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(block_17[i_24 * 2]), "r"(inverse_23));
                                        uint16_t _e4m3x2_22;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_22) : "r"(_bf16x2_mul_22));
                                        uint32_t _bf16x2_mul_23;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(block_17[i_24 * 2 + 1]), "r"(inverse_23));
                                        uint16_t _e4m3x2_23;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_23) : "r"(_bf16x2_mul_23));
                                        words_24[i_24] = (unsigned int)_e4m3x2_22 | (unsigned int)_e4m3x2_23 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_21 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_24[0]), "r"(words_24[1]), "r"(words_24[2]), "r"(words_24[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24[4]), "r"(words_24[5]), "r"(words_24[6]), "r"(words_24[7]) : "memory");
                                    smem_v51[tid % 32 * 4 + tid / 32] = scale_word_8;
                                    scale_word_8 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 96), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2, smem_v51_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_25[16];
                                    #pragma unroll
                                    for (int j_17 = 0; j_17 < 16; j_17++) {
                                        block_25[j_17] = packed_5[64 + j_17];
                                    }
                                    #pragma unroll
                                    for (int j_18 = 0; j_18 < 4; j_18++) {
                                        unsigned int address_15 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_18 * 16);
                                        address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_15 - smem_v49_addr)), "r"(packed_5[64 + 4 * j_18]), "r"(packed_5[64 + 4 * j_18 + 1]), "r"(packed_5[64 + 4 * j_18 + 2]), "r"(packed_5[64 + 4 * j_18 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_24;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(block_25[0]));
                                    unsigned int amax_pair_26 = _bf16x2_abs_24;
                                    #pragma unroll
                                    for (int i_25 = 1; i_25 < 16; i_25++) {
                                        uint32_t _bf16x2_abs_25;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(block_25[i_25]));
                                        uint32_t _bf16x2_max_12;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(amax_pair_26), "r"(_bf16x2_abs_25));
                                        amax_pair_26 = _bf16x2_max_12;
                                    }
                                    uint16_t _bf16_max_12;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_12) : "h"((uint16_t)(amax_pair_26 & 65535)), "h"((uint16_t)(amax_pair_26 >> 16)));
                                    float _cvt_f32_bf16_12;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_12) : "h"((uint16_t)(_bf16_max_12)));
                                    float amax_27 = _cvt_f32_bf16_12;
                                    float _fmax_12 = fmaxf(amax_27 * 0.002232142857f, 1e-12f);
                                    float scale_28 = _fmax_12;
                                    uint16_t _ue8m0x2_f32_12;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_12) : "f"(scale_28), "f"(scale_28));
                                    unsigned int scale_byte_29 = (unsigned int)_ue8m0x2_f32_12 & 255;
                                    unsigned int inverse_lane_30 = 254 - scale_byte_29 << 7;
                                    unsigned int inverse_31 = inverse_lane_30 | inverse_lane_30 << 16;
                                    unsigned int words_32[8];
                                    #pragma unroll
                                    for (int i_26 = 0; i_26 < 8; i_26++) {
                                        uint32_t _bf16x2_mul_24;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(block_25[i_26 * 2]), "r"(inverse_31));
                                        uint16_t _e4m3x2_24;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_24) : "r"(_bf16x2_mul_24));
                                        uint32_t _bf16x2_mul_25;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(block_25[i_26 * 2 + 1]), "r"(inverse_31));
                                        uint16_t _e4m3x2_25;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_25) : "r"(_bf16x2_mul_25));
                                        words_32[i_26] = (unsigned int)_e4m3x2_24 | (unsigned int)_e4m3x2_25 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_29;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_32[0]), "r"(words_32[1]), "r"(words_32[2]), "r"(words_32[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_32[4]), "r"(words_32[5]), "r"(words_32[6]), "r"(words_32[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 128), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_33[16];
                                    #pragma unroll
                                    for (int j_19 = 0; j_19 < 16; j_19++) {
                                        block_33[j_19] = packed_5[80 + j_19];
                                    }
                                    #pragma unroll
                                    for (int j_20 = 0; j_20 < 4; j_20++) {
                                        unsigned int address_16 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_20 * 16);
                                        address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_16 - smem_v49_addr)), "r"(packed_5[80 + 4 * j_20]), "r"(packed_5[80 + 4 * j_20 + 1]), "r"(packed_5[80 + 4 * j_20 + 2]), "r"(packed_5[80 + 4 * j_20 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_26;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_26) : "r"(block_33[0]));
                                    unsigned int amax_pair_34 = _bf16x2_abs_26;
                                    #pragma unroll
                                    for (int i_27 = 1; i_27 < 16; i_27++) {
                                        uint32_t _bf16x2_abs_27;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_27) : "r"(block_33[i_27]));
                                        uint32_t _bf16x2_max_13;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_13) : "r"(amax_pair_34), "r"(_bf16x2_abs_27));
                                        amax_pair_34 = _bf16x2_max_13;
                                    }
                                    uint16_t _bf16_max_13;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_13) : "h"((uint16_t)(amax_pair_34 & 65535)), "h"((uint16_t)(amax_pair_34 >> 16)));
                                    float _cvt_f32_bf16_13;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_13) : "h"((uint16_t)(_bf16_max_13)));
                                    float amax_35 = _cvt_f32_bf16_13;
                                    float _fmax_13 = fmaxf(amax_35 * 0.002232142857f, 1e-12f);
                                    float scale_36 = _fmax_13;
                                    uint16_t _ue8m0x2_f32_13;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_13) : "f"(scale_36), "f"(scale_36));
                                    unsigned int scale_byte_37 = (unsigned int)_ue8m0x2_f32_13 & 255;
                                    unsigned int inverse_lane_38 = 254 - scale_byte_37 << 7;
                                    unsigned int inverse_39 = inverse_lane_38 | inverse_lane_38 << 16;
                                    unsigned int words_40[8];
                                    #pragma unroll
                                    for (int i_28 = 0; i_28 < 8; i_28++) {
                                        uint32_t _bf16x2_mul_26;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(block_33[i_28 * 2]), "r"(inverse_39));
                                        uint16_t _e4m3x2_26;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_26) : "r"(_bf16x2_mul_26));
                                        uint32_t _bf16x2_mul_27;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(block_33[i_28 * 2 + 1]), "r"(inverse_39));
                                        uint16_t _e4m3x2_27;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_27) : "r"(_bf16x2_mul_27));
                                        words_40[i_28] = (unsigned int)_e4m3x2_26 | (unsigned int)_e4m3x2_27 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_37 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_40[0]), "r"(words_40[1]), "r"(words_40[2]), "r"(words_40[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_40[4]), "r"(words_40[5]), "r"(words_40[6]), "r"(words_40[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 160), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_41[16];
                                    #pragma unroll
                                    for (int j_21 = 0; j_21 < 16; j_21++) {
                                        block_41[j_21] = packed_5[96 + j_21];
                                    }
                                    #pragma unroll
                                    for (int j_22 = 0; j_22 < 4; j_22++) {
                                        unsigned int address_17 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_22 * 16);
                                        address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_17 - smem_v49_addr)), "r"(packed_5[96 + 4 * j_22]), "r"(packed_5[96 + 4 * j_22 + 1]), "r"(packed_5[96 + 4 * j_22 + 2]), "r"(packed_5[96 + 4 * j_22 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_28;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_28) : "r"(block_41[0]));
                                    unsigned int amax_pair_42 = _bf16x2_abs_28;
                                    #pragma unroll
                                    for (int i_29 = 1; i_29 < 16; i_29++) {
                                        uint32_t _bf16x2_abs_29;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_29) : "r"(block_41[i_29]));
                                        uint32_t _bf16x2_max_14;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14) : "r"(amax_pair_42), "r"(_bf16x2_abs_29));
                                        amax_pair_42 = _bf16x2_max_14;
                                    }
                                    uint16_t _bf16_max_14;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_14) : "h"((uint16_t)(amax_pair_42 & 65535)), "h"((uint16_t)(amax_pair_42 >> 16)));
                                    float _cvt_f32_bf16_14;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_14) : "h"((uint16_t)(_bf16_max_14)));
                                    float amax_43 = _cvt_f32_bf16_14;
                                    float _fmax_14 = fmaxf(amax_43 * 0.002232142857f, 1e-12f);
                                    float scale_44 = _fmax_14;
                                    uint16_t _ue8m0x2_f32_14;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_14) : "f"(scale_44), "f"(scale_44));
                                    unsigned int scale_byte_45 = (unsigned int)_ue8m0x2_f32_14 & 255;
                                    unsigned int inverse_lane_46 = 254 - scale_byte_45 << 7;
                                    unsigned int inverse_47 = inverse_lane_46 | inverse_lane_46 << 16;
                                    unsigned int words_48[8];
                                    #pragma unroll
                                    for (int i_30 = 0; i_30 < 8; i_30++) {
                                        uint32_t _bf16x2_mul_28;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(block_41[i_30 * 2]), "r"(inverse_47));
                                        uint16_t _e4m3x2_28;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_28) : "r"(_bf16x2_mul_28));
                                        uint32_t _bf16x2_mul_29;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(block_41[i_30 * 2 + 1]), "r"(inverse_47));
                                        uint16_t _e4m3x2_29;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_29) : "r"(_bf16x2_mul_29));
                                        words_48[i_30] = (unsigned int)_e4m3x2_28 | (unsigned int)_e4m3x2_29 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_45 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_48[0]), "r"(words_48[1]), "r"(words_48[2]), "r"(words_48[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_48[4]), "r"(words_48[5]), "r"(words_48[6]), "r"(words_48[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 192), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_49[16];
                                    #pragma unroll
                                    for (int j_23 = 0; j_23 < 16; j_23++) {
                                        block_49[j_23] = packed_5[112 + j_23];
                                    }
                                    #pragma unroll
                                    for (int j_24 = 0; j_24 < 4; j_24++) {
                                        unsigned int address_18 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_24 * 16);
                                        address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_18 - smem_v49_addr)), "r"(packed_5[112 + 4 * j_24]), "r"(packed_5[112 + 4 * j_24 + 1]), "r"(packed_5[112 + 4 * j_24 + 2]), "r"(packed_5[112 + 4 * j_24 + 3]) : "memory");
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
                                    uint32_t _bf16x2_abs_30;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_30) : "r"(block_49[0]));
                                    unsigned int amax_pair_50 = _bf16x2_abs_30;
                                    #pragma unroll
                                    for (int i_31 = 1; i_31 < 16; i_31++) {
                                        uint32_t _bf16x2_abs_31;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_31) : "r"(block_49[i_31]));
                                        uint32_t _bf16x2_max_15;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15) : "r"(amax_pair_50), "r"(_bf16x2_abs_31));
                                        amax_pair_50 = _bf16x2_max_15;
                                    }
                                    uint16_t _bf16_max_15;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_15) : "h"((uint16_t)(amax_pair_50 & 65535)), "h"((uint16_t)(amax_pair_50 >> 16)));
                                    float _cvt_f32_bf16_15;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_15) : "h"((uint16_t)(_bf16_max_15)));
                                    float amax_51 = _cvt_f32_bf16_15;
                                    float _fmax_15 = fmaxf(amax_51 * 0.002232142857f, 1e-12f);
                                    float scale_52 = _fmax_15;
                                    uint16_t _ue8m0x2_f32_15;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_15) : "f"(scale_52), "f"(scale_52));
                                    unsigned int scale_byte_53 = (unsigned int)_ue8m0x2_f32_15 & 255;
                                    unsigned int inverse_lane_54 = 254 - scale_byte_53 << 7;
                                    unsigned int inverse_55 = inverse_lane_54 | inverse_lane_54 << 16;
                                    unsigned int words_56[8];
                                    #pragma unroll
                                    for (int i_32 = 0; i_32 < 8; i_32++) {
                                        uint32_t _bf16x2_mul_30;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(block_49[i_32 * 2]), "r"(inverse_55));
                                        uint16_t _e4m3x2_30;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_30) : "r"(_bf16x2_mul_30));
                                        uint32_t _bf16x2_mul_31;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(block_49[i_32 * 2 + 1]), "r"(inverse_55));
                                        uint16_t _e4m3x2_31;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_31) : "r"(_bf16x2_mul_31));
                                        words_56[i_32] = (unsigned int)_e4m3x2_30 | (unsigned int)_e4m3x2_31 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_53 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_56[0]), "r"(words_56[1]), "r"(words_56[2]), "r"(words_56[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_56[4]), "r"(words_56[5]), "r"(words_56[6]), "r"(words_56[7]) : "memory");
                                    smem_v52[tid % 32 * 4 + tid / 32] = scale_word_8;
                                    scale_word_8 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(y_2 * 256 + 224), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2 + 1, smem_v52_addr);
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
                                                bool enabled_value_4 = 1;
                                                if (enabled_value_4 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_2 + x_2) * (intermediate / 256) + y_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_bits = phase_bits_4;
                    } else if (task_1 < 2 * mini_gate) {
                        int col_blocks_5 = (intermediate + 256 - 1) / 256;
                        int x_3 = -1;
                        int y_3 = -1;
                        int expert_3 = -1;
                        int k_start_3 = 0;
                        int k_end_3 = 0;
                        int first_3 = 0;
                        int first_block_1 = mini * (mini_size / 256);
                        int _min_15 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                        int end_block_1 = _min_15;
                        int block_2 = first_block_1 + (task_1 - mini_gate) / col_blocks_5;
                        if (block_2 < end_block_1) {
                            int index_1 = counts[3 * experts + block_2];
                            int offset_5 = counts[experts + index_1] / 256;
                            int _max_5 = ((first_block_1) > (offset_5) ? (first_block_1) : (offset_5));
                            int first_row_2 = _max_5;
                            int _min_16 = ((end_block_1) < (offset_5 + counts[index_1] / 256) ? (end_block_1) : (offset_5 + counts[index_1] / 256));
                            int rows_1 = _min_16 - first_row_2;
                            int supergroup_3 = (task_1 - mini_gate - (first_row_2 - first_block_1) * col_blocks_5) / (rows_1 * 8);
                            int full_cols_3 = col_blocks_5 / 8 * 8;
                            int row_11 = 0;
                            int col_25 = 0;
                            if (task_1 - mini_gate - (first_row_2 - first_block_1) * col_blocks_5 < rows_1 * full_cols_3) {
                                row_11 = (task_1 - mini_gate - (first_row_2 - first_block_1) * col_blocks_5) % (rows_1 * 8) / 8;
                                col_25 = supergroup_3 * 8 + (task_1 - mini_gate - (first_row_2 - first_block_1) * col_blocks_5) % 8;
                            } else {
                                row_11 = (task_1 - mini_gate - (first_row_2 - first_block_1) * col_blocks_5 - rows_1 * full_cols_3) / (col_blocks_5 - full_cols_3);
                                col_25 = full_cols_3 + (task_1 - mini_gate - (first_row_2 - first_block_1) * col_blocks_5 - rows_1 * full_cols_3) % (col_blocks_5 - full_cols_3);
                            }
                            if ((supergroup_3 & 1) != 0) {
                                row_11 = rows_1 - row_11 - 1;
                            }
                            x_3 = first_row_2 + row_11;
                            y_3 = col_25;
                            expert_3 = index_1;
                        }
                        unsigned int phase_bits_5 = gemm_bits;
                        int has_hi_3 = 0;
                        int global_mini_3 = mini;
                        int macro_rows_3 = 0;
                        int iterations_3 = hidden / 128;
                        int macro_k_1 = 0;
                        if (expert_3 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_17 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                        int _max_6 = ((0) > (_min_17) ? (0) : (_min_17));
                                        int mini_rows_4 = _max_6;
                                        int required_4 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_5 = 1;
                                        if (enabled_value_5 != 0) {
                                            int32_t _relaxed_ld_6;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(x_ready + global_mini_3) : "memory");
                                            int value_3 = _relaxed_ld_6;
                                            while (value_3 < required_4) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_7;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(x_ready + global_mini_3) : "memory");
                                                value_3 = _relaxed_ld_7;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_7 = 0;
                                    #pragma unroll 1
                                    for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_7) * 8, phase_bits_5 >> (unsigned int)(16 + ring_7) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v39_addr + (unsigned int)(ring_7 * 16384)), "l"((&x_q)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v40_addr + (unsigned int)(ring_7 * 16384)), "l"((&wu_q)), "r"(0), "r"(y_3 * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(expert_3), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_7);
                                        ring_7 = (ring_7 + 1) % 4;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_18 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                            int _max_7 = ((0) > (_min_18) ? (0) : (_min_18));
                                            int mini_rows_5 = _max_7;
                                            int required_5 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_6 = 1;
                                            if (enabled_value_6 != 0) {
                                                int32_t _relaxed_ld_8;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(x_ready + global_mini_3) : "memory");
                                                int value_4 = _relaxed_ld_8;
                                                while (value_4 < required_5) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_9;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(x_ready + global_mini_3) : "memory");
                                                    value_4 = _relaxed_ld_9;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        int ring_8 = 0;
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_3; idx_8++) {
                                            mbarrier_wait(scales_finished_addr + (ring_8) * 8, phase_bits_5 >> (unsigned int)(16 + ring_8) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v42_addr + (unsigned int)(ring_8 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_3 * 2 + cta_rank_0) * (hidden / 128) + idx_8),
                                                   "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v43_addr + (unsigned int)(ring_8 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((expert_3 * (intermediate / 128) + y_3 * 2 + cta_rank_0) * (hidden / 128) + idx_8),
                                                   "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_8);
                                            ring_8 = (ring_8 + 1) % 4;
                                        }
                                    }
                                }
                            } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_9 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_5 >> 22 & 1);
                                        phase_bits_5 = phase_bits_5 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_9 = 0; idx_9 < iterations_3; idx_9++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_9) * 8, 3072);
                                            mbarrier_wait(scales_arrived_addr + (ring_9) * 8, phase_bits_5 >> (unsigned int)(8 + ring_9) & 1);
                                            int buffer_1 = idx_9 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_1 * 4, make_sf_cp_desc_sbo128(smem_v42_addr + (unsigned int)(ring_9 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_1 * 8, make_sf_cp_desc_sbo128(smem_v43_addr + (unsigned int)(ring_9 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v43_addr + (unsigned int)(ring_9 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_9) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_9) * 8, phase_bits_5 >> (unsigned int)ring_9 & 1);
                                            int _mma_a_lo_5 = (((smem_v39_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                            int _mma_b_lo_5 = (((smem_v40_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0)), ((idx_9 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0) | 0x80000000), 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << ring_9) ^ (unsigned int)(1 << 8 + ring_9);
                                            ring_9 = (ring_9 + 1) % 4;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_5 >> 6 & 1);
                                    int warp_row_1 = tid / 32 * 32;
                                    unsigned int packed_6[128];
                                    #pragma unroll
                                    for (int i_33 = 0; i_33 < 8; i_33++) {
                                        float _tmem_load_5[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                                            : "r"(taddr_1 + (unsigned int)(warp_row_1 << 16) + (unsigned int)(i_33 * 32)));
                                        #pragma unroll
                                        for (int j_25 = 0; j_25 < 16; j_25++) {
                                            __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_5[2 * j_25], _tmem_load_5[2 * j_25 + 1]));
                                            packed_6[i_33 * 16 + j_25] = __as_u32(_bf16x2_10);
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                    unsigned int scale_word_9 = 0;
                                    unsigned int block_0_1[16];
                                    #pragma unroll
                                    for (int j_26 = 0; j_26 < 16; j_26++) {
                                        block_0_1[j_26] = packed_6[j_26];
                                    }
                                    #pragma unroll
                                    for (int j_27 = 0; j_27 < 4; j_27++) {
                                        unsigned int address_19 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_27 * 16);
                                        address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_19 - smem_v49_addr)), "r"(packed_6[4 * j_27]), "r"(packed_6[4 * j_27 + 1]), "r"(packed_6[4 * j_27 + 2]), "r"(packed_6[4 * j_27 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_32;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(block_0_1[0]));
                                    unsigned int amax_pair_9 = _bf16x2_abs_32;
                                    #pragma unroll
                                    for (int i_34 = 1; i_34 < 16; i_34++) {
                                        uint32_t _bf16x2_abs_33;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(block_0_1[i_34]));
                                        uint32_t _bf16x2_max_16;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(amax_pair_9), "r"(_bf16x2_abs_33));
                                        amax_pair_9 = _bf16x2_max_16;
                                    }
                                    uint16_t _bf16_max_16;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_16) : "h"((uint16_t)(amax_pair_9 & 65535)), "h"((uint16_t)(amax_pair_9 >> 16)));
                                    float _cvt_f32_bf16_16;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(_bf16_max_16)));
                                    float amax_9 = _cvt_f32_bf16_16;
                                    float _fmax_16 = fmaxf(amax_9 * 0.002232142857f, 1e-12f);
                                    float scale_9 = _fmax_16;
                                    uint16_t _ue8m0x2_f32_16;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_16) : "f"(scale_9), "f"(scale_9));
                                    unsigned int scale_byte_9 = (unsigned int)_ue8m0x2_f32_16 & 255;
                                    unsigned int inverse_lane_9 = 254 - scale_byte_9 << 7;
                                    unsigned int inverse_9 = inverse_lane_9 | inverse_lane_9 << 16;
                                    unsigned int words_9[8];
                                    #pragma unroll
                                    for (int i_35 = 0; i_35 < 8; i_35++) {
                                        uint32_t _bf16x2_mul_32;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(block_0_1[i_35 * 2]), "r"(inverse_9));
                                        uint16_t _e4m3x2_32;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_32) : "r"(_bf16x2_mul_32));
                                        uint32_t _bf16x2_mul_33;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(block_0_1[i_35 * 2 + 1]), "r"(inverse_9));
                                        uint16_t _e4m3x2_33;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_33) : "r"(_bf16x2_mul_33));
                                        words_9[i_35] = (unsigned int)_e4m3x2_32 | (unsigned int)_e4m3x2_33 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_9;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_9[0]), "r"(words_9[1]), "r"(words_9[2]), "r"(words_9[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_9[4]), "r"(words_9[5]), "r"(words_9[6]), "r"(words_9[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_1_1[16];
                                    #pragma unroll
                                    for (int j_28 = 0; j_28 < 16; j_28++) {
                                        block_1_1[j_28] = packed_6[16 + j_28];
                                    }
                                    #pragma unroll
                                    for (int j_29 = 0; j_29 < 4; j_29++) {
                                        unsigned int address_20 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_29 * 16);
                                        address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_20 - smem_v49_addr)), "r"(packed_6[16 + 4 * j_29]), "r"(packed_6[16 + 4 * j_29 + 1]), "r"(packed_6[16 + 4 * j_29 + 2]), "r"(packed_6[16 + 4 * j_29 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_34;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(block_1_1[0]));
                                    unsigned int amax_pair_2_2 = _bf16x2_abs_34;
                                    #pragma unroll
                                    for (int i_36 = 1; i_36 < 16; i_36++) {
                                        uint32_t _bf16x2_abs_35;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(block_1_1[i_36]));
                                        uint32_t _bf16x2_max_17;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(amax_pair_2_2), "r"(_bf16x2_abs_35));
                                        amax_pair_2_2 = _bf16x2_max_17;
                                    }
                                    uint16_t _bf16_max_17;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_17) : "h"((uint16_t)(amax_pair_2_2 & 65535)), "h"((uint16_t)(amax_pair_2_2 >> 16)));
                                    float _cvt_f32_bf16_17;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_17) : "h"((uint16_t)(_bf16_max_17)));
                                    float amax_3_2 = _cvt_f32_bf16_17;
                                    float _fmax_17 = fmaxf(amax_3_2 * 0.002232142857f, 1e-12f);
                                    float scale_4_2 = _fmax_17;
                                    uint16_t _ue8m0x2_f32_17;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_17) : "f"(scale_4_2), "f"(scale_4_2));
                                    unsigned int scale_byte_5_2 = (unsigned int)_ue8m0x2_f32_17 & 255;
                                    unsigned int inverse_lane_6_2 = 254 - scale_byte_5_2 << 7;
                                    unsigned int inverse_7_2 = inverse_lane_6_2 | inverse_lane_6_2 << 16;
                                    unsigned int words_8_2[8];
                                    #pragma unroll
                                    for (int i_37 = 0; i_37 < 8; i_37++) {
                                        uint32_t _bf16x2_mul_34;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(block_1_1[i_37 * 2]), "r"(inverse_7_2));
                                        uint16_t _e4m3x2_34;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_34) : "r"(_bf16x2_mul_34));
                                        uint32_t _bf16x2_mul_35;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(block_1_1[i_37 * 2 + 1]), "r"(inverse_7_2));
                                        uint16_t _e4m3x2_35;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_35) : "r"(_bf16x2_mul_35));
                                        words_8_2[i_37] = (unsigned int)_e4m3x2_34 | (unsigned int)_e4m3x2_35 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_5_2 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_8_2[0]), "r"(words_8_2[1]), "r"(words_8_2[2]), "r"(words_8_2[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_2[4]), "r"(words_8_2[5]), "r"(words_8_2[6]), "r"(words_8_2[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 32), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_9_1[16];
                                    #pragma unroll
                                    for (int j_30 = 0; j_30 < 16; j_30++) {
                                        block_9_1[j_30] = packed_6[32 + j_30];
                                    }
                                    #pragma unroll
                                    for (int j_31 = 0; j_31 < 4; j_31++) {
                                        unsigned int address_21 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_31 * 16);
                                        address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_21 - smem_v49_addr)), "r"(packed_6[32 + 4 * j_31]), "r"(packed_6[32 + 4 * j_31 + 1]), "r"(packed_6[32 + 4 * j_31 + 2]), "r"(packed_6[32 + 4 * j_31 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_36;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(block_9_1[0]));
                                    unsigned int amax_pair_10_1 = _bf16x2_abs_36;
                                    #pragma unroll
                                    for (int i_38 = 1; i_38 < 16; i_38++) {
                                        uint32_t _bf16x2_abs_37;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(block_9_1[i_38]));
                                        uint32_t _bf16x2_max_18;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(amax_pair_10_1), "r"(_bf16x2_abs_37));
                                        amax_pair_10_1 = _bf16x2_max_18;
                                    }
                                    uint16_t _bf16_max_18;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_18) : "h"((uint16_t)(amax_pair_10_1 & 65535)), "h"((uint16_t)(amax_pair_10_1 >> 16)));
                                    float _cvt_f32_bf16_18;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_18) : "h"((uint16_t)(_bf16_max_18)));
                                    float amax_11_1 = _cvt_f32_bf16_18;
                                    float _fmax_18 = fmaxf(amax_11_1 * 0.002232142857f, 1e-12f);
                                    float scale_12_1 = _fmax_18;
                                    uint16_t _ue8m0x2_f32_18;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_18) : "f"(scale_12_1), "f"(scale_12_1));
                                    unsigned int scale_byte_13_1 = (unsigned int)_ue8m0x2_f32_18 & 255;
                                    unsigned int inverse_lane_14_1 = 254 - scale_byte_13_1 << 7;
                                    unsigned int inverse_15_1 = inverse_lane_14_1 | inverse_lane_14_1 << 16;
                                    unsigned int words_16_1[8];
                                    #pragma unroll
                                    for (int i_39 = 0; i_39 < 8; i_39++) {
                                        uint32_t _bf16x2_mul_36;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(block_9_1[i_39 * 2]), "r"(inverse_15_1));
                                        uint16_t _e4m3x2_36;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_36) : "r"(_bf16x2_mul_36));
                                        uint32_t _bf16x2_mul_37;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(block_9_1[i_39 * 2 + 1]), "r"(inverse_15_1));
                                        uint16_t _e4m3x2_37;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_37) : "r"(_bf16x2_mul_37));
                                        words_16_1[i_39] = (unsigned int)_e4m3x2_36 | (unsigned int)_e4m3x2_37 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_13_1 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_16_1[0]), "r"(words_16_1[1]), "r"(words_16_1[2]), "r"(words_16_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16_1[4]), "r"(words_16_1[5]), "r"(words_16_1[6]), "r"(words_16_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 64), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_17_1[16];
                                    #pragma unroll
                                    for (int j_32 = 0; j_32 < 16; j_32++) {
                                        block_17_1[j_32] = packed_6[48 + j_32];
                                    }
                                    #pragma unroll
                                    for (int j_33 = 0; j_33 < 4; j_33++) {
                                        unsigned int address_22 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_33 * 16);
                                        address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_22 - smem_v49_addr)), "r"(packed_6[48 + 4 * j_33]), "r"(packed_6[48 + 4 * j_33 + 1]), "r"(packed_6[48 + 4 * j_33 + 2]), "r"(packed_6[48 + 4 * j_33 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 3), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_38;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(block_17_1[0]));
                                    unsigned int amax_pair_18_1 = _bf16x2_abs_38;
                                    #pragma unroll
                                    for (int i_40 = 1; i_40 < 16; i_40++) {
                                        uint32_t _bf16x2_abs_39;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(block_17_1[i_40]));
                                        uint32_t _bf16x2_max_19;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(amax_pair_18_1), "r"(_bf16x2_abs_39));
                                        amax_pair_18_1 = _bf16x2_max_19;
                                    }
                                    uint16_t _bf16_max_19;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_19) : "h"((uint16_t)(amax_pair_18_1 & 65535)), "h"((uint16_t)(amax_pair_18_1 >> 16)));
                                    float _cvt_f32_bf16_19;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(_bf16_max_19)));
                                    float amax_19_1 = _cvt_f32_bf16_19;
                                    float _fmax_19 = fmaxf(amax_19_1 * 0.002232142857f, 1e-12f);
                                    float scale_20_1 = _fmax_19;
                                    uint16_t _ue8m0x2_f32_19;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_19) : "f"(scale_20_1), "f"(scale_20_1));
                                    unsigned int scale_byte_21_1 = (unsigned int)_ue8m0x2_f32_19 & 255;
                                    unsigned int inverse_lane_22_1 = 254 - scale_byte_21_1 << 7;
                                    unsigned int inverse_23_1 = inverse_lane_22_1 | inverse_lane_22_1 << 16;
                                    unsigned int words_24_1[8];
                                    #pragma unroll
                                    for (int i_41 = 0; i_41 < 8; i_41++) {
                                        uint32_t _bf16x2_mul_38;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(block_17_1[i_41 * 2]), "r"(inverse_23_1));
                                        uint16_t _e4m3x2_38;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_38) : "r"(_bf16x2_mul_38));
                                        uint32_t _bf16x2_mul_39;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(block_17_1[i_41 * 2 + 1]), "r"(inverse_23_1));
                                        uint16_t _e4m3x2_39;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_39) : "r"(_bf16x2_mul_39));
                                        words_24_1[i_41] = (unsigned int)_e4m3x2_38 | (unsigned int)_e4m3x2_39 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_21_1 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_24_1[0]), "r"(words_24_1[1]), "r"(words_24_1[2]), "r"(words_24_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_24_1[4]), "r"(words_24_1[5]), "r"(words_24_1[6]), "r"(words_24_1[7]) : "memory");
                                    smem_v51[tid % 32 * 4 + tid / 32] = scale_word_9;
                                    scale_word_9 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 96), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&up_sc_store), 0, 0, (x_3 * 2 + cta_rank_0) * i_tiles + y_3 * 2, smem_v51_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_25_1[16];
                                    #pragma unroll
                                    for (int j_34 = 0; j_34 < 16; j_34++) {
                                        block_25_1[j_34] = packed_6[64 + j_34];
                                    }
                                    #pragma unroll
                                    for (int j_35 = 0; j_35 < 4; j_35++) {
                                        unsigned int address_23 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_35 * 16);
                                        address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_23 - smem_v49_addr)), "r"(packed_6[64 + 4 * j_35]), "r"(packed_6[64 + 4 * j_35 + 1]), "r"(packed_6[64 + 4 * j_35 + 2]), "r"(packed_6[64 + 4 * j_35 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_40;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(block_25_1[0]));
                                    unsigned int amax_pair_26_1 = _bf16x2_abs_40;
                                    #pragma unroll
                                    for (int i_42 = 1; i_42 < 16; i_42++) {
                                        uint32_t _bf16x2_abs_41;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(block_25_1[i_42]));
                                        uint32_t _bf16x2_max_20;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(amax_pair_26_1), "r"(_bf16x2_abs_41));
                                        amax_pair_26_1 = _bf16x2_max_20;
                                    }
                                    uint16_t _bf16_max_20;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_20) : "h"((uint16_t)(amax_pair_26_1 & 65535)), "h"((uint16_t)(amax_pair_26_1 >> 16)));
                                    float _cvt_f32_bf16_20;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(_bf16_max_20)));
                                    float amax_27_1 = _cvt_f32_bf16_20;
                                    float _fmax_20 = fmaxf(amax_27_1 * 0.002232142857f, 1e-12f);
                                    float scale_28_1 = _fmax_20;
                                    uint16_t _ue8m0x2_f32_20;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_20) : "f"(scale_28_1), "f"(scale_28_1));
                                    unsigned int scale_byte_29_1 = (unsigned int)_ue8m0x2_f32_20 & 255;
                                    unsigned int inverse_lane_30_1 = 254 - scale_byte_29_1 << 7;
                                    unsigned int inverse_31_1 = inverse_lane_30_1 | inverse_lane_30_1 << 16;
                                    unsigned int words_32_1[8];
                                    #pragma unroll
                                    for (int i_43 = 0; i_43 < 8; i_43++) {
                                        uint32_t _bf16x2_mul_40;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(block_25_1[i_43 * 2]), "r"(inverse_31_1));
                                        uint16_t _e4m3x2_40;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_40) : "r"(_bf16x2_mul_40));
                                        uint32_t _bf16x2_mul_41;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(block_25_1[i_43 * 2 + 1]), "r"(inverse_31_1));
                                        uint16_t _e4m3x2_41;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_41) : "r"(_bf16x2_mul_41));
                                        words_32_1[i_43] = (unsigned int)_e4m3x2_40 | (unsigned int)_e4m3x2_41 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_29_1;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_32_1[0]), "r"(words_32_1[1]), "r"(words_32_1[2]), "r"(words_32_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_32_1[4]), "r"(words_32_1[5]), "r"(words_32_1[6]), "r"(words_32_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 128), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_33_1[16];
                                    #pragma unroll
                                    for (int j_36 = 0; j_36 < 16; j_36++) {
                                        block_33_1[j_36] = packed_6[80 + j_36];
                                    }
                                    #pragma unroll
                                    for (int j_37 = 0; j_37 < 4; j_37++) {
                                        unsigned int address_24 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_37 * 16);
                                        address_24 = address_24 ^ (address_24 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_24 - smem_v49_addr)), "r"(packed_6[80 + 4 * j_37]), "r"(packed_6[80 + 4 * j_37 + 1]), "r"(packed_6[80 + 4 * j_37 + 2]), "r"(packed_6[80 + 4 * j_37 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 5), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_42;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(block_33_1[0]));
                                    unsigned int amax_pair_34_1 = _bf16x2_abs_42;
                                    #pragma unroll
                                    for (int i_44 = 1; i_44 < 16; i_44++) {
                                        uint32_t _bf16x2_abs_43;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(block_33_1[i_44]));
                                        uint32_t _bf16x2_max_21;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(amax_pair_34_1), "r"(_bf16x2_abs_43));
                                        amax_pair_34_1 = _bf16x2_max_21;
                                    }
                                    uint16_t _bf16_max_21;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_21) : "h"((uint16_t)(amax_pair_34_1 & 65535)), "h"((uint16_t)(amax_pair_34_1 >> 16)));
                                    float _cvt_f32_bf16_21;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_21) : "h"((uint16_t)(_bf16_max_21)));
                                    float amax_35_1 = _cvt_f32_bf16_21;
                                    float _fmax_21 = fmaxf(amax_35_1 * 0.002232142857f, 1e-12f);
                                    float scale_36_1 = _fmax_21;
                                    uint16_t _ue8m0x2_f32_21;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_21) : "f"(scale_36_1), "f"(scale_36_1));
                                    unsigned int scale_byte_37_1 = (unsigned int)_ue8m0x2_f32_21 & 255;
                                    unsigned int inverse_lane_38_1 = 254 - scale_byte_37_1 << 7;
                                    unsigned int inverse_39_1 = inverse_lane_38_1 | inverse_lane_38_1 << 16;
                                    unsigned int words_40_1[8];
                                    #pragma unroll
                                    for (int i_45 = 0; i_45 < 8; i_45++) {
                                        uint32_t _bf16x2_mul_42;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(block_33_1[i_45 * 2]), "r"(inverse_39_1));
                                        uint16_t _e4m3x2_42;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_42) : "r"(_bf16x2_mul_42));
                                        uint32_t _bf16x2_mul_43;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(block_33_1[i_45 * 2 + 1]), "r"(inverse_39_1));
                                        uint16_t _e4m3x2_43;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_43) : "r"(_bf16x2_mul_43));
                                        words_40_1[i_45] = (unsigned int)_e4m3x2_42 | (unsigned int)_e4m3x2_43 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_37_1 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_40_1[0]), "r"(words_40_1[1]), "r"(words_40_1[2]), "r"(words_40_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_40_1[4]), "r"(words_40_1[5]), "r"(words_40_1[6]), "r"(words_40_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 160), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_41_1[16];
                                    #pragma unroll
                                    for (int j_38 = 0; j_38 < 16; j_38++) {
                                        block_41_1[j_38] = packed_6[96 + j_38];
                                    }
                                    #pragma unroll
                                    for (int j_39 = 0; j_39 < 4; j_39++) {
                                        unsigned int address_25 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_39 * 16);
                                        address_25 = address_25 ^ (address_25 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_25 - smem_v49_addr)), "r"(packed_6[96 + 4 * j_39]), "r"(packed_6[96 + 4 * j_39 + 1]), "r"(packed_6[96 + 4 * j_39 + 2]), "r"(packed_6[96 + 4 * j_39 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_44;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(block_41_1[0]));
                                    unsigned int amax_pair_42_1 = _bf16x2_abs_44;
                                    #pragma unroll
                                    for (int i_46 = 1; i_46 < 16; i_46++) {
                                        uint32_t _bf16x2_abs_45;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(block_41_1[i_46]));
                                        uint32_t _bf16x2_max_22;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(amax_pair_42_1), "r"(_bf16x2_abs_45));
                                        amax_pair_42_1 = _bf16x2_max_22;
                                    }
                                    uint16_t _bf16_max_22;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_22) : "h"((uint16_t)(amax_pair_42_1 & 65535)), "h"((uint16_t)(amax_pair_42_1 >> 16)));
                                    float _cvt_f32_bf16_22;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(_bf16_max_22)));
                                    float amax_43_1 = _cvt_f32_bf16_22;
                                    float _fmax_22 = fmaxf(amax_43_1 * 0.002232142857f, 1e-12f);
                                    float scale_44_1 = _fmax_22;
                                    uint16_t _ue8m0x2_f32_22;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_22) : "f"(scale_44_1), "f"(scale_44_1));
                                    unsigned int scale_byte_45_1 = (unsigned int)_ue8m0x2_f32_22 & 255;
                                    unsigned int inverse_lane_46_1 = 254 - scale_byte_45_1 << 7;
                                    unsigned int inverse_47_1 = inverse_lane_46_1 | inverse_lane_46_1 << 16;
                                    unsigned int words_48_1[8];
                                    #pragma unroll
                                    for (int i_47 = 0; i_47 < 8; i_47++) {
                                        uint32_t _bf16x2_mul_44;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(block_41_1[i_47 * 2]), "r"(inverse_47_1));
                                        uint16_t _e4m3x2_44;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_44) : "r"(_bf16x2_mul_44));
                                        uint32_t _bf16x2_mul_45;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(block_41_1[i_47 * 2 + 1]), "r"(inverse_47_1));
                                        uint16_t _e4m3x2_45;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_45) : "r"(_bf16x2_mul_45));
                                        words_48_1[i_47] = (unsigned int)_e4m3x2_44 | (unsigned int)_e4m3x2_45 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_45_1 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_48_1[0]), "r"(words_48_1[1]), "r"(words_48_1[2]), "r"(words_48_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_48_1[4]), "r"(words_48_1[5]), "r"(words_48_1[6]), "r"(words_48_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 192), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_49_1[16];
                                    #pragma unroll
                                    for (int j_40 = 0; j_40 < 16; j_40++) {
                                        block_49_1[j_40] = packed_6[112 + j_40];
                                    }
                                    #pragma unroll
                                    for (int j_41 = 0; j_41 < 4; j_41++) {
                                        unsigned int address_26 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_41 * 16);
                                        address_26 = address_26 ^ (address_26 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v49_addr + (address_26 - smem_v49_addr)), "r"(packed_6[112 + 4 * j_41]), "r"(packed_6[112 + 4 * j_41 + 1]), "r"(packed_6[112 + 4 * j_41 + 2]), "r"(packed_6[112 + 4 * j_41 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + 7), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_46;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(block_49_1[0]));
                                    unsigned int amax_pair_50_1 = _bf16x2_abs_46;
                                    #pragma unroll
                                    for (int i_48 = 1; i_48 < 16; i_48++) {
                                        uint32_t _bf16x2_abs_47;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(block_49_1[i_48]));
                                        uint32_t _bf16x2_max_23;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(amax_pair_50_1), "r"(_bf16x2_abs_47));
                                        amax_pair_50_1 = _bf16x2_max_23;
                                    }
                                    uint16_t _bf16_max_23;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_23) : "h"((uint16_t)(amax_pair_50_1 & 65535)), "h"((uint16_t)(amax_pair_50_1 >> 16)));
                                    float _cvt_f32_bf16_23;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(_bf16_max_23)));
                                    float amax_51_1 = _cvt_f32_bf16_23;
                                    float _fmax_23 = fmaxf(amax_51_1 * 0.002232142857f, 1e-12f);
                                    float scale_52_1 = _fmax_23;
                                    uint16_t _ue8m0x2_f32_23;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_23) : "f"(scale_52_1), "f"(scale_52_1));
                                    unsigned int scale_byte_53_1 = (unsigned int)_ue8m0x2_f32_23 & 255;
                                    unsigned int inverse_lane_54_1 = 254 - scale_byte_53_1 << 7;
                                    unsigned int inverse_55_1 = inverse_lane_54_1 | inverse_lane_54_1 << 16;
                                    unsigned int words_56_1[8];
                                    #pragma unroll
                                    for (int i_49 = 0; i_49 < 8; i_49++) {
                                        uint32_t _bf16x2_mul_46;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(block_49_1[i_49 * 2]), "r"(inverse_55_1));
                                        uint16_t _e4m3x2_46;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_46) : "r"(_bf16x2_mul_46));
                                        uint32_t _bf16x2_mul_47;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(block_49_1[i_49 * 2 + 1]), "r"(inverse_55_1));
                                        uint16_t _e4m3x2_47;
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_47) : "r"(_bf16x2_mul_47));
                                        words_56_1[i_49] = (unsigned int)_e4m3x2_46 | (unsigned int)_e4m3x2_47 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_53_1 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32)), "r"(words_56_1[0]), "r"(words_56_1[1]), "r"(words_56_1[2]), "r"(words_56_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v50_addr + (unsigned int)(tid * 32 + 16)), "r"(words_56_1[4]), "r"(words_56_1[5]), "r"(words_56_1[6]), "r"(words_56_1[7]) : "memory");
                                    smem_v52[tid % 32 * 4 + tid / 32] = scale_word_9;
                                    scale_word_9 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 224), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v50_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&up_sc_store), 0, 0, (x_3 * 2 + cta_rank_0) * i_tiles + y_3 * 2 + 1, smem_v52_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    phase_bits_5 = phase_bits_5 ^ 64;
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_7 = 1;
                                                if (enabled_value_7 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_3 + x_3) * (intermediate / 256) + y_3))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_bits = phase_bits_5;
                    } else {
                        unsigned int phase_bits_6 = swiglu_bits;
                        int col_blocks_6 = intermediate / 128;
                        int num_tiles_1 = tokens / 128 * col_blocks_6;
                        int macro_row_offset_1 = 0;
                        int first_tile_1 = (task_1 - 2 * mini_gate) * 6 + cta_rank_0 * 3;
                        int global_mini_4 = mini;
                        int mini_tiles = mini_size / 128 * col_blocks_6;
                        first_tile_1 = first_tile_1 + global_mini_4 * mini_tiles;
                        int _min_19 = ((num_tiles_1) < ((global_mini_4 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_4 + 1) * mini_tiles));
                        int tile_end_1 = _min_19;
                        int macro_tiles_1 = macro_size / 128;
                        if (first_tile_1 < tile_end_1) {
                            int first_row_3 = first_tile_1 / col_blocks_6;
                            int first_col_1 = first_tile_1 % col_blocks_6;
                            if (tid == 0) {
                                if (tile_end_1 > first_tile_1) {
                                    int row_12 = first_row_3;
                                    int col_26 = first_col_1;
                                    if (col_26 >= col_blocks_6) {
                                        row_12 = row_12 + 1;
                                        col_26 = col_26 - col_blocks_6;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr, 65536);
                                    int parent_1 = row_12 / 2 * (intermediate / 256) + col_26 / 2;
                                    int32_t _relaxed_ld_10;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                    int value_5 = _relaxed_ld_10;
                                    while (value_5 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_11;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                        value_5 = _relaxed_ld_11;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr), "l"((&gate_routed_in)), "r"(0), "r"((row_12 - macro_row_offset_1) * 128), "r"(col_26 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr), "l"((&up_routed_in)), "r"(0), "r"((row_12 - macro_row_offset_1) * 128), "r"(col_26 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 1) {
                                    int row_13 = first_row_3;
                                    int col_27 = first_col_1 + 1;
                                    if (col_27 >= col_blocks_6) {
                                        row_13 = row_13 + 1;
                                        col_27 = col_27 - col_blocks_6;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + 8, 65536);
                                    int parent_2 = row_13 / 2 * (intermediate / 256) + col_27 / 2;
                                    int32_t _relaxed_ld_12;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(gate_ready + (shared_gate + parent_2)) : "memory");
                                    int value_6 = _relaxed_ld_12;
                                    while (value_6 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_13;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(gate_ready + (shared_gate + parent_2)) : "memory");
                                        value_6 = _relaxed_ld_13;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + 32768), "l"((&gate_routed_in)), "r"(0), "r"((row_13 - macro_row_offset_1) * 128), "r"(col_27 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + 32768), "l"((&up_routed_in)), "r"(0), "r"((row_13 - macro_row_offset_1) * 128), "r"(col_27 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 8) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 2) {
                                    int row_14 = first_row_3;
                                    int col_28 = first_col_1 + 2;
                                    if (col_28 >= col_blocks_6) {
                                        row_14 = row_14 + 1;
                                        col_28 = col_28 - col_blocks_6;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + 16, 65536);
                                    int parent_3 = row_14 / 2 * (intermediate / 256) + col_28 / 2;
                                    int32_t _relaxed_ld_14;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(gate_ready + (shared_gate + parent_3)) : "memory");
                                    int value_7 = _relaxed_ld_14;
                                    while (value_7 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_15;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(gate_ready + (shared_gate + parent_3)) : "memory");
                                        value_7 = _relaxed_ld_15;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + 65536), "l"((&gate_routed_in)), "r"(0), "r"((row_14 - macro_row_offset_1) * 128), "r"(col_28 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 16) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + 65536), "l"((&up_routed_in)), "r"(0), "r"((row_14 - macro_row_offset_1) * 128), "r"(col_28 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 16) : "memory");
                                }
                            }
                            if (tile_end_1 > first_tile_1) {
                                mbarrier_wait(swiglu_arrived_addr, phase_bits_6 & 1);
                                phase_bits_6 = phase_bits_6 ^ 1;
                                int row_15 = first_row_3;
                                int col_29 = first_col_1;
                                if (col_29 >= col_blocks_6) {
                                    row_15 = row_15 + 1;
                                    col_29 = col_29 - col_blocks_6;
                                }
                                float gate_1[64];
                                float up_1[64];
                                float denominator_1[64];
                                int warp_0_5 = tid / 32;
                                int local_warp_1 = warp_0_5 / 4 + warp_0_5 % 4 * 2;
                                int lane_7 = tid % 32;
                                #pragma unroll
                                for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                    unsigned int packed_7[4];
                                    unsigned int address_27 = gate_smem_addr + (unsigned int)(((tile_col_3 * 16 + lane_7 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_7 % 16) * 64 + (tile_col_3 * 16 + lane_7 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_7[0]), "=r"(packed_7[1]), "=r"(packed_7[2]), "=r"(packed_7[3])
                                        : "r"(address_27 ^ (address_27 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_7 = 0; pair_7 < 4; pair_7++) {
                                        float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(packed_7[pair_7]));
                                        gate_1[tile_col_3 * 8 + pair_7 * 2] = _cvt_f32_2.x;
                                        gate_1[tile_col_3 * 8 + pair_7 * 2 + 1] = _cvt_f32_2.y;
                                    }
                                }
                                int warp_1_1 = tid / 32;
                                int local_warp_2_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                int lane_3_2 = tid % 32;
                                #pragma unroll
                                for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                    unsigned int packed_8[4];
                                    unsigned int address_28 = up_smem_addr + (unsigned int)(((tile_col_4 * 16 + lane_3_2 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_1 * 16 + lane_3_2 % 16) * 64 + (tile_col_4 * 16 + lane_3_2 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_8[0]), "=r"(packed_8[1]), "=r"(packed_8[2]), "=r"(packed_8[3])
                                        : "r"(address_28 ^ (address_28 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                        float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(packed_8[pair_8]));
                                        up_1[tile_col_4 * 8 + pair_8 * 2] = _cvt_f32_3.x;
                                        up_1[tile_col_4 * 8 + pair_8 * 2 + 1] = _cvt_f32_3.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                    denominator_1[elem_5] = gate_1[elem_5] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                    float _exp_1 = expf(denominator_1[elem_6]);
                                    denominator_1[elem_6] = _exp_1;
                                }
                                #pragma unroll
                                for (int elem_7 = 0; elem_7 < 64; elem_7++) {
                                    denominator_1[elem_7] = denominator_1[elem_7] + 1.0f;
                                }
                                #pragma unroll
                                for (int elem_8 = 0; elem_8 < 64; elem_8++) {
                                    gate_1[elem_8] = gate_1[elem_8] / denominator_1[elem_8];
                                }
                                #pragma unroll
                                for (int elem_9 = 0; elem_9 < 64; elem_9++) {
                                    gate_1[elem_9] = gate_1[elem_9] * up_1[elem_9];
                                }
                                __syncthreads();
                                int warp_4_1 = tid / 32;
                                int local_warp_5_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                                int lane_6_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                    unsigned int packed_9[4];
                                    #pragma unroll
                                    for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(gate_1[tile_col_5 * 8 + pair_9 * 2], gate_1[tile_col_5 * 8 + pair_9 * 2 + 1]));
                                        packed_9[pair_9] = __as_u32(_bf16x2_11);
                                    }
                                    int row_0_8 = local_warp_5_1 * 16 + lane_6_1 % 16;
                                    int col_1_1 = tile_col_5 * 16 + lane_6_1 / 16 * 8;
                                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_8 * 136 + col_1_1) * 2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int row_0_9 = tid;
                                    row_0_9 = tid % 64 * 2 + tid / 64;
                                    unsigned int scale_word_10 = 0;
                                    #pragma unroll 1
                                    for (int j_42 = 0; j_42 < 4; j_42++) {
                                        int k_block_8 = (j_42 + tid / 8) % 4;
                                        unsigned int pairs_8[16];
                                        #pragma unroll
                                        for (int k_16 = 0; k_16 < 16; k_16++) {
                                            int col_0 = k_block_8 * 32 + (tid * 4 + k_16 * 2) % 32;
                                            float x0_8 = 0.0f;
                                            float x1_8 = 0.0f;
                                            x0_8 = (float)hidden_staging[col_0 * 136 + row_0_9];
                                            x1_8 = (float)hidden_staging[(col_0 + 1) * 136 + row_0_9];
                                            __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(x0_8, x1_8));
                                            pairs_8[k_16] = __as_u32(_bf16x2_12);
                                        }
                                        uint32_t _bf16x2_abs_48;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(pairs_8[0]));
                                        unsigned int amax_pair_11 = _bf16x2_abs_48;
                                        #pragma unroll
                                        for (int i_50 = 1; i_50 < 16; i_50++) {
                                            uint32_t _bf16x2_abs_49;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(pairs_8[i_50]));
                                            uint32_t _bf16x2_max_24;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(amax_pair_11), "r"(_bf16x2_abs_49));
                                            amax_pair_11 = _bf16x2_max_24;
                                        }
                                        uint16_t _bf16_max_24;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_24) : "h"((uint16_t)(amax_pair_11 & 65535)), "h"((uint16_t)(amax_pair_11 >> 16)));
                                        float _cvt_f32_bf16_24;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(_bf16_max_24)));
                                        float amax_10 = _cvt_f32_bf16_24;
                                        float _fmax_24 = fmaxf(amax_10 * 0.002232142857f, 1e-12f);
                                        float scale_10 = _fmax_24;
                                        uint16_t _ue8m0x2_f32_24;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_24) : "f"(scale_10), "f"(scale_10));
                                        unsigned int scale_byte_10 = (unsigned int)_ue8m0x2_f32_24 & 255;
                                        unsigned int inverse_lane_10 = 254 - scale_byte_10 << 7;
                                        unsigned int inverse_10 = inverse_lane_10 | inverse_lane_10 << 16;
                                        unsigned int words_10[8];
                                        #pragma unroll
                                        for (int i_51 = 0; i_51 < 8; i_51++) {
                                            uint32_t _bf16x2_mul_48;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(pairs_8[i_51 * 2]), "r"(inverse_10));
                                            uint16_t _e4m3x2_48;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_48) : "r"(_bf16x2_mul_48));
                                            uint32_t _bf16x2_mul_49;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(pairs_8[i_51 * 2 + 1]), "r"(inverse_10));
                                            uint16_t _e4m3x2_49;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_49) : "r"(_bf16x2_mul_49));
                                            words_10[i_51] = (unsigned int)_e4m3x2_48 | (unsigned int)_e4m3x2_49 << 16;
                                        }
                                        scale_word_10 = scale_word_10 | scale_byte_10 << (unsigned int)(k_block_8 * 8);
                                        #pragma unroll
                                        for (int k_17 = 0; k_17 < 8; k_17++) {
                                            int col_0_1 = k_block_8 * 32 + (tid * 4 + k_17 * 4) % 32;
                                            smem_v11[(row_0_9 * 128 + col_0_1) / 4] = words_10[k_17];
                                        }
                                    }
                                    smem_v12[row_0_9 % 32 * 4 + row_0_9 / 32] = scale_word_10;
                                } else {
                                    int row_0_10 = tid - 128;
                                    unsigned int scale_word_11 = 0;
                                    #pragma unroll 1
                                    for (int j_43 = 0; j_43 < 4; j_43++) {
                                        int k_block_9 = (j_43 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_9[16];
                                        #pragma unroll
                                        for (int k_18 = 0; k_18 < 16; k_18++) {
                                            int col_0_2 = k_block_9 * 32 + ((tid - 128) * 4 + k_18 * 2) % 32;
                                            float x0_9 = 0.0f;
                                            float x1_9 = 0.0f;
                                            pairs_9[k_18] = hidden_words[(row_0_10 * 136 + col_0_2) / 2];
                                        }
                                        uint32_t _bf16x2_abs_50;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(pairs_9[0]));
                                        unsigned int amax_pair_12 = _bf16x2_abs_50;
                                        #pragma unroll
                                        for (int i_52 = 1; i_52 < 16; i_52++) {
                                            uint32_t _bf16x2_abs_51;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(pairs_9[i_52]));
                                            uint32_t _bf16x2_max_25;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(amax_pair_12), "r"(_bf16x2_abs_51));
                                            amax_pair_12 = _bf16x2_max_25;
                                        }
                                        uint16_t _bf16_max_25;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_25) : "h"((uint16_t)(amax_pair_12 & 65535)), "h"((uint16_t)(amax_pair_12 >> 16)));
                                        float _cvt_f32_bf16_25;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(_bf16_max_25)));
                                        float amax_12 = _cvt_f32_bf16_25;
                                        float _fmax_25 = fmaxf(amax_12 * 0.002232142857f, 1e-12f);
                                        float scale_11 = _fmax_25;
                                        uint16_t _ue8m0x2_f32_25;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_25) : "f"(scale_11), "f"(scale_11));
                                        unsigned int scale_byte_11 = (unsigned int)_ue8m0x2_f32_25 & 255;
                                        unsigned int inverse_lane_11 = 254 - scale_byte_11 << 7;
                                        unsigned int inverse_11 = inverse_lane_11 | inverse_lane_11 << 16;
                                        unsigned int words_11[8];
                                        #pragma unroll
                                        for (int i_53 = 0; i_53 < 8; i_53++) {
                                            uint32_t _bf16x2_mul_50;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(pairs_9[i_53 * 2]), "r"(inverse_11));
                                            uint16_t _e4m3x2_50;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_50) : "r"(_bf16x2_mul_50));
                                            uint32_t _bf16x2_mul_51;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(pairs_9[i_53 * 2 + 1]), "r"(inverse_11));
                                            uint16_t _e4m3x2_51;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_51) : "r"(_bf16x2_mul_51));
                                            words_11[i_53] = (unsigned int)_e4m3x2_50 | (unsigned int)_e4m3x2_51 << 16;
                                        }
                                        scale_word_11 = scale_word_11 | scale_byte_11 << (unsigned int)(k_block_9 * 8);
                                        #pragma unroll
                                        for (int k_19 = 0; k_19 < 8; k_19++) {
                                            int col_0_3 = k_block_9 * 32 + ((tid - 128) * 4 + k_19 * 4) % 32;
                                            smem_v9[(row_0_10 * 128 + col_0_3) / 4] = words_11[k_19];
                                        }
                                    }
                                    smem_v10[row_0_10 % 32 * 4 + row_0_10 / 32] = scale_word_11;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row = row_15 - macro_row_offset_1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&hidden_q_store), col_29 * 128, local_row * 128, smem_v9_addr);
                                    tma_store_3d((&hidden_sc_store), 0, 0, local_row * col_blocks_6 + col_29, smem_v10_addr);
                                    tma_store_2d((&hidden_t_store), local_row * 128, col_29 * 128, smem_v11_addr);
                                    tma_store_3d((&hidden_sc_t_store), 0, 0, col_29 * macro_tiles_1 + local_row, smem_v12_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (tile_end_1 > first_tile_1 + 1) {
                                mbarrier_wait(swiglu_arrived_addr + 8, phase_bits_6 >> 1 & 1);
                                phase_bits_6 = phase_bits_6 ^ 2;
                                int row_16 = first_row_3;
                                int col_30 = first_col_1 + 1;
                                if (col_30 >= col_blocks_6) {
                                    row_16 = row_16 + 1;
                                    col_30 = col_30 - col_blocks_6;
                                }
                                float gate_2[64];
                                float up_2[64];
                                float denominator_2[64];
                                int warp_0_6 = tid / 32;
                                int local_warp_3 = warp_0_6 / 4 + warp_0_6 % 4 * 2;
                                int lane_8 = tid % 32;
                                #pragma unroll
                                for (int tile_col_6 = 0; tile_col_6 < 8; tile_col_6++) {
                                    unsigned int packed_10[4];
                                    unsigned int address_29 = gate_smem_addr + 32768 + (unsigned int)(((tile_col_6 * 16 + lane_8 / 16 * 8) / 64 * 128 * 64 + (local_warp_3 * 16 + lane_8 % 16) * 64 + (tile_col_6 * 16 + lane_8 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_10[0]), "=r"(packed_10[1]), "=r"(packed_10[2]), "=r"(packed_10[3])
                                        : "r"(address_29 ^ (address_29 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(packed_10[pair_10]));
                                        gate_2[tile_col_6 * 8 + pair_10 * 2] = _cvt_f32_4.x;
                                        gate_2[tile_col_6 * 8 + pair_10 * 2 + 1] = _cvt_f32_4.y;
                                    }
                                }
                                int warp_1_2 = tid / 32;
                                int local_warp_2_2 = warp_1_2 / 4 + warp_1_2 % 4 * 2;
                                int lane_3_3 = tid % 32;
                                #pragma unroll
                                for (int tile_col_7 = 0; tile_col_7 < 8; tile_col_7++) {
                                    unsigned int packed_11[4];
                                    unsigned int address_30 = up_smem_addr + 32768 + (unsigned int)(((tile_col_7 * 16 + lane_3_3 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_2 * 16 + lane_3_3 % 16) * 64 + (tile_col_7 * 16 + lane_3_3 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_11[0]), "=r"(packed_11[1]), "=r"(packed_11[2]), "=r"(packed_11[3])
                                        : "r"(address_30 ^ (address_30 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_11 = 0; pair_11 < 4; pair_11++) {
                                        float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(packed_11[pair_11]));
                                        up_2[tile_col_7 * 8 + pair_11 * 2] = _cvt_f32_5.x;
                                        up_2[tile_col_7 * 8 + pair_11 * 2 + 1] = _cvt_f32_5.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem_10 = 0; elem_10 < 64; elem_10++) {
                                    denominator_2[elem_10] = gate_2[elem_10] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_11 = 0; elem_11 < 64; elem_11++) {
                                    float _exp_2 = expf(denominator_2[elem_11]);
                                    denominator_2[elem_11] = _exp_2;
                                }
                                #pragma unroll
                                for (int elem_12 = 0; elem_12 < 64; elem_12++) {
                                    denominator_2[elem_12] = denominator_2[elem_12] + 1.0f;
                                }
                                #pragma unroll
                                for (int elem_13 = 0; elem_13 < 64; elem_13++) {
                                    gate_2[elem_13] = gate_2[elem_13] / denominator_2[elem_13];
                                }
                                #pragma unroll
                                for (int elem_14 = 0; elem_14 < 64; elem_14++) {
                                    gate_2[elem_14] = gate_2[elem_14] * up_2[elem_14];
                                }
                                __syncthreads();
                                int warp_4_2 = tid / 32;
                                int local_warp_5_2 = warp_4_2 / 4 + warp_4_2 % 4 * 2;
                                int lane_6_2 = tid % 32;
                                #pragma unroll
                                for (int tile_col_8 = 0; tile_col_8 < 8; tile_col_8++) {
                                    unsigned int packed_12[4];
                                    #pragma unroll
                                    for (int pair_12 = 0; pair_12 < 4; pair_12++) {
                                        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(gate_2[tile_col_8 * 8 + pair_12 * 2], gate_2[tile_col_8 * 8 + pair_12 * 2 + 1]));
                                        packed_12[pair_12] = __as_u32(_bf16x2_13);
                                    }
                                    int row_0_11 = local_warp_5_2 * 16 + lane_6_2 % 16;
                                    int col_1_2 = tile_col_8 * 16 + lane_6_2 / 16 * 8;
                                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_11 * 136 + col_1_2) * 2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int row_0_12 = tid;
                                    row_0_12 = tid % 64 * 2 + tid / 64;
                                    unsigned int scale_word_12 = 0;
                                    #pragma unroll 1
                                    for (int j_44 = 0; j_44 < 4; j_44++) {
                                        int k_block_10 = (j_44 + tid / 8) % 4;
                                        unsigned int pairs_10[16];
                                        #pragma unroll
                                        for (int k_20 = 0; k_20 < 16; k_20++) {
                                            int col_0_4 = k_block_10 * 32 + (tid * 4 + k_20 * 2) % 32;
                                            float x0_10 = 0.0f;
                                            float x1_10 = 0.0f;
                                            x0_10 = (float)hidden_staging[col_0_4 * 136 + row_0_12];
                                            x1_10 = (float)hidden_staging[(col_0_4 + 1) * 136 + row_0_12];
                                            __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(x0_10, x1_10));
                                            pairs_10[k_20] = __as_u32(_bf16x2_14);
                                        }
                                        uint32_t _bf16x2_abs_52;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(pairs_10[0]));
                                        unsigned int amax_pair_13 = _bf16x2_abs_52;
                                        #pragma unroll
                                        for (int i_54 = 1; i_54 < 16; i_54++) {
                                            uint32_t _bf16x2_abs_53;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(pairs_10[i_54]));
                                            uint32_t _bf16x2_max_26;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(amax_pair_13), "r"(_bf16x2_abs_53));
                                            amax_pair_13 = _bf16x2_max_26;
                                        }
                                        uint16_t _bf16_max_26;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_26) : "h"((uint16_t)(amax_pair_13 & 65535)), "h"((uint16_t)(amax_pair_13 >> 16)));
                                        float _cvt_f32_bf16_26;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(_bf16_max_26)));
                                        float amax_13 = _cvt_f32_bf16_26;
                                        float _fmax_26 = fmaxf(amax_13 * 0.002232142857f, 1e-12f);
                                        float scale_13 = _fmax_26;
                                        uint16_t _ue8m0x2_f32_26;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_26) : "f"(scale_13), "f"(scale_13));
                                        unsigned int scale_byte_12 = (unsigned int)_ue8m0x2_f32_26 & 255;
                                        unsigned int inverse_lane_12 = 254 - scale_byte_12 << 7;
                                        unsigned int inverse_12 = inverse_lane_12 | inverse_lane_12 << 16;
                                        unsigned int words_12[8];
                                        #pragma unroll
                                        for (int i_55 = 0; i_55 < 8; i_55++) {
                                            uint32_t _bf16x2_mul_52;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(pairs_10[i_55 * 2]), "r"(inverse_12));
                                            uint16_t _e4m3x2_52;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_52) : "r"(_bf16x2_mul_52));
                                            uint32_t _bf16x2_mul_53;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(pairs_10[i_55 * 2 + 1]), "r"(inverse_12));
                                            uint16_t _e4m3x2_53;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_53) : "r"(_bf16x2_mul_53));
                                            words_12[i_55] = (unsigned int)_e4m3x2_52 | (unsigned int)_e4m3x2_53 << 16;
                                        }
                                        scale_word_12 = scale_word_12 | scale_byte_12 << (unsigned int)(k_block_10 * 8);
                                        #pragma unroll
                                        for (int k_21 = 0; k_21 < 8; k_21++) {
                                            int col_0_5 = k_block_10 * 32 + (tid * 4 + k_21 * 4) % 32;
                                            smem_v15[(row_0_12 * 128 + col_0_5) / 4] = words_12[k_21];
                                        }
                                    }
                                    smem_v16[row_0_12 % 32 * 4 + row_0_12 / 32] = scale_word_12;
                                } else {
                                    int row_0_13 = tid - 128;
                                    unsigned int scale_word_13 = 0;
                                    #pragma unroll 1
                                    for (int j_45 = 0; j_45 < 4; j_45++) {
                                        int k_block_11 = (j_45 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_11[16];
                                        #pragma unroll
                                        for (int k_22 = 0; k_22 < 16; k_22++) {
                                            int col_0_6 = k_block_11 * 32 + ((tid - 128) * 4 + k_22 * 2) % 32;
                                            float x0_11 = 0.0f;
                                            float x1_11 = 0.0f;
                                            pairs_11[k_22] = hidden_words[(row_0_13 * 136 + col_0_6) / 2];
                                        }
                                        uint32_t _bf16x2_abs_54;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(pairs_11[0]));
                                        unsigned int amax_pair_14 = _bf16x2_abs_54;
                                        #pragma unroll
                                        for (int i_56 = 1; i_56 < 16; i_56++) {
                                            uint32_t _bf16x2_abs_55;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(pairs_11[i_56]));
                                            uint32_t _bf16x2_max_27;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(amax_pair_14), "r"(_bf16x2_abs_55));
                                            amax_pair_14 = _bf16x2_max_27;
                                        }
                                        uint16_t _bf16_max_27;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_27) : "h"((uint16_t)(amax_pair_14 & 65535)), "h"((uint16_t)(amax_pair_14 >> 16)));
                                        float _cvt_f32_bf16_27;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_27) : "h"((uint16_t)(_bf16_max_27)));
                                        float amax_14 = _cvt_f32_bf16_27;
                                        float _fmax_27 = fmaxf(amax_14 * 0.002232142857f, 1e-12f);
                                        float scale_14 = _fmax_27;
                                        uint16_t _ue8m0x2_f32_27;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_27) : "f"(scale_14), "f"(scale_14));
                                        unsigned int scale_byte_14 = (unsigned int)_ue8m0x2_f32_27 & 255;
                                        unsigned int inverse_lane_13 = 254 - scale_byte_14 << 7;
                                        unsigned int inverse_13 = inverse_lane_13 | inverse_lane_13 << 16;
                                        unsigned int words_13[8];
                                        #pragma unroll
                                        for (int i_57 = 0; i_57 < 8; i_57++) {
                                            uint32_t _bf16x2_mul_54;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(pairs_11[i_57 * 2]), "r"(inverse_13));
                                            uint16_t _e4m3x2_54;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_54) : "r"(_bf16x2_mul_54));
                                            uint32_t _bf16x2_mul_55;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(pairs_11[i_57 * 2 + 1]), "r"(inverse_13));
                                            uint16_t _e4m3x2_55;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_55) : "r"(_bf16x2_mul_55));
                                            words_13[i_57] = (unsigned int)_e4m3x2_54 | (unsigned int)_e4m3x2_55 << 16;
                                        }
                                        scale_word_13 = scale_word_13 | scale_byte_14 << (unsigned int)(k_block_11 * 8);
                                        #pragma unroll
                                        for (int k_23 = 0; k_23 < 8; k_23++) {
                                            int col_0_7 = k_block_11 * 32 + ((tid - 128) * 4 + k_23 * 4) % 32;
                                            smem_v13[(row_0_13 * 128 + col_0_7) / 4] = words_13[k_23];
                                        }
                                    }
                                    smem_v14[row_0_13 % 32 * 4 + row_0_13 / 32] = scale_word_13;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_1 = row_16 - macro_row_offset_1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&hidden_q_store), col_30 * 128, local_row_1 * 128, smem_v13_addr);
                                    tma_store_3d((&hidden_sc_store), 0, 0, local_row_1 * col_blocks_6 + col_30, smem_v14_addr);
                                    tma_store_2d((&hidden_t_store), local_row_1 * 128, col_30 * 128, smem_v15_addr);
                                    tma_store_3d((&hidden_sc_t_store), 0, 0, col_30 * macro_tiles_1 + local_row_1, smem_v16_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (tile_end_1 > first_tile_1 + 2) {
                                mbarrier_wait(swiglu_arrived_addr + 16, phase_bits_6 >> 2 & 1);
                                phase_bits_6 = phase_bits_6 ^ 4;
                                int row_17 = first_row_3;
                                int col_31 = first_col_1 + 2;
                                if (col_31 >= col_blocks_6) {
                                    row_17 = row_17 + 1;
                                    col_31 = col_31 - col_blocks_6;
                                }
                                float gate_3[64];
                                float up_3[64];
                                float denominator_3[64];
                                int warp_0_7 = tid / 32;
                                int local_warp_4 = warp_0_7 / 4 + warp_0_7 % 4 * 2;
                                int lane_9 = tid % 32;
                                #pragma unroll
                                for (int tile_col_9 = 0; tile_col_9 < 8; tile_col_9++) {
                                    unsigned int packed_13[4];
                                    unsigned int address_31 = gate_smem_addr + 65536 + (unsigned int)(((tile_col_9 * 16 + lane_9 / 16 * 8) / 64 * 128 * 64 + (local_warp_4 * 16 + lane_9 % 16) * 64 + (tile_col_9 * 16 + lane_9 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_13[0]), "=r"(packed_13[1]), "=r"(packed_13[2]), "=r"(packed_13[3])
                                        : "r"(address_31 ^ (address_31 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_13 = 0; pair_13 < 4; pair_13++) {
                                        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(packed_13[pair_13]));
                                        gate_3[tile_col_9 * 8 + pair_13 * 2] = _cvt_f32_6.x;
                                        gate_3[tile_col_9 * 8 + pair_13 * 2 + 1] = _cvt_f32_6.y;
                                    }
                                }
                                int warp_1_3 = tid / 32;
                                int local_warp_2_3 = warp_1_3 / 4 + warp_1_3 % 4 * 2;
                                int lane_3_4 = tid % 32;
                                #pragma unroll
                                for (int tile_col_10 = 0; tile_col_10 < 8; tile_col_10++) {
                                    unsigned int packed_14[4];
                                    unsigned int address_32 = up_smem_addr + 65536 + (unsigned int)(((tile_col_10 * 16 + lane_3_4 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_3 * 16 + lane_3_4 % 16) * 64 + (tile_col_10 * 16 + lane_3_4 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_14[0]), "=r"(packed_14[1]), "=r"(packed_14[2]), "=r"(packed_14[3])
                                        : "r"(address_32 ^ (address_32 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_14 = 0; pair_14 < 4; pair_14++) {
                                        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(packed_14[pair_14]));
                                        up_3[tile_col_10 * 8 + pair_14 * 2] = _cvt_f32_7.x;
                                        up_3[tile_col_10 * 8 + pair_14 * 2 + 1] = _cvt_f32_7.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem_15 = 0; elem_15 < 64; elem_15++) {
                                    denominator_3[elem_15] = gate_3[elem_15] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_16 = 0; elem_16 < 64; elem_16++) {
                                    float _exp_3 = expf(denominator_3[elem_16]);
                                    denominator_3[elem_16] = _exp_3;
                                }
                                #pragma unroll
                                for (int elem_17 = 0; elem_17 < 64; elem_17++) {
                                    denominator_3[elem_17] = denominator_3[elem_17] + 1.0f;
                                }
                                #pragma unroll
                                for (int elem_18 = 0; elem_18 < 64; elem_18++) {
                                    gate_3[elem_18] = gate_3[elem_18] / denominator_3[elem_18];
                                }
                                #pragma unroll
                                for (int elem_19 = 0; elem_19 < 64; elem_19++) {
                                    gate_3[elem_19] = gate_3[elem_19] * up_3[elem_19];
                                }
                                __syncthreads();
                                int warp_4_3 = tid / 32;
                                int local_warp_5_3 = warp_4_3 / 4 + warp_4_3 % 4 * 2;
                                int lane_6_3 = tid % 32;
                                #pragma unroll
                                for (int tile_col_11 = 0; tile_col_11 < 8; tile_col_11++) {
                                    unsigned int packed_15[4];
                                    #pragma unroll
                                    for (int pair_15 = 0; pair_15 < 4; pair_15++) {
                                        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(gate_3[tile_col_11 * 8 + pair_15 * 2], gate_3[tile_col_11 * 8 + pair_15 * 2 + 1]));
                                        packed_15[pair_15] = __as_u32(_bf16x2_15);
                                    }
                                    int row_0_14 = local_warp_5_3 * 16 + lane_6_3 % 16;
                                    int col_1_3 = tile_col_11 * 16 + lane_6_3 / 16 * 8;
                                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_14 * 136 + col_1_3) * 2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&packed_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_15[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_15[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_15[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int row_0_15 = tid;
                                    row_0_15 = tid % 64 * 2 + tid / 64;
                                    unsigned int scale_word_14 = 0;
                                    #pragma unroll 1
                                    for (int j_46 = 0; j_46 < 4; j_46++) {
                                        int k_block_12 = (j_46 + tid / 8) % 4;
                                        unsigned int pairs_12[16];
                                        #pragma unroll
                                        for (int k_24 = 0; k_24 < 16; k_24++) {
                                            int col_0_8 = k_block_12 * 32 + (tid * 4 + k_24 * 2) % 32;
                                            float x0_12 = 0.0f;
                                            float x1_12 = 0.0f;
                                            x0_12 = (float)hidden_staging[col_0_8 * 136 + row_0_15];
                                            x1_12 = (float)hidden_staging[(col_0_8 + 1) * 136 + row_0_15];
                                            __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(x0_12, x1_12));
                                            pairs_12[k_24] = __as_u32(_bf16x2_16);
                                        }
                                        uint32_t _bf16x2_abs_56;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(pairs_12[0]));
                                        unsigned int amax_pair_15 = _bf16x2_abs_56;
                                        #pragma unroll
                                        for (int i_58 = 1; i_58 < 16; i_58++) {
                                            uint32_t _bf16x2_abs_57;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(pairs_12[i_58]));
                                            uint32_t _bf16x2_max_28;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(amax_pair_15), "r"(_bf16x2_abs_57));
                                            amax_pair_15 = _bf16x2_max_28;
                                        }
                                        uint16_t _bf16_max_28;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_28) : "h"((uint16_t)(amax_pair_15 & 65535)), "h"((uint16_t)(amax_pair_15 >> 16)));
                                        float _cvt_f32_bf16_28;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(_bf16_max_28)));
                                        float amax_15 = _cvt_f32_bf16_28;
                                        float _fmax_28 = fmaxf(amax_15 * 0.002232142857f, 1e-12f);
                                        float scale_15 = _fmax_28;
                                        uint16_t _ue8m0x2_f32_28;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_28) : "f"(scale_15), "f"(scale_15));
                                        unsigned int scale_byte_15 = (unsigned int)_ue8m0x2_f32_28 & 255;
                                        unsigned int inverse_lane_15 = 254 - scale_byte_15 << 7;
                                        unsigned int inverse_14 = inverse_lane_15 | inverse_lane_15 << 16;
                                        unsigned int words_14[8];
                                        #pragma unroll
                                        for (int i_59 = 0; i_59 < 8; i_59++) {
                                            uint32_t _bf16x2_mul_56;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(pairs_12[i_59 * 2]), "r"(inverse_14));
                                            uint16_t _e4m3x2_56;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_56) : "r"(_bf16x2_mul_56));
                                            uint32_t _bf16x2_mul_57;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(pairs_12[i_59 * 2 + 1]), "r"(inverse_14));
                                            uint16_t _e4m3x2_57;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_57) : "r"(_bf16x2_mul_57));
                                            words_14[i_59] = (unsigned int)_e4m3x2_56 | (unsigned int)_e4m3x2_57 << 16;
                                        }
                                        scale_word_14 = scale_word_14 | scale_byte_15 << (unsigned int)(k_block_12 * 8);
                                        #pragma unroll
                                        for (int k_25 = 0; k_25 < 8; k_25++) {
                                            int col_0_9 = k_block_12 * 32 + (tid * 4 + k_25 * 4) % 32;
                                            smem_v19[(row_0_15 * 128 + col_0_9) / 4] = words_14[k_25];
                                        }
                                    }
                                    smem_v20[row_0_15 % 32 * 4 + row_0_15 / 32] = scale_word_14;
                                } else {
                                    int row_0_16 = tid - 128;
                                    unsigned int scale_word_15 = 0;
                                    #pragma unroll 1
                                    for (int j_47 = 0; j_47 < 4; j_47++) {
                                        int k_block_13 = (j_47 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_13[16];
                                        #pragma unroll
                                        for (int k_26 = 0; k_26 < 16; k_26++) {
                                            int col_0_10 = k_block_13 * 32 + ((tid - 128) * 4 + k_26 * 2) % 32;
                                            float x0_13 = 0.0f;
                                            float x1_13 = 0.0f;
                                            pairs_13[k_26] = hidden_words[(row_0_16 * 136 + col_0_10) / 2];
                                        }
                                        uint32_t _bf16x2_abs_58;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(pairs_13[0]));
                                        unsigned int amax_pair_16 = _bf16x2_abs_58;
                                        #pragma unroll
                                        for (int i_60 = 1; i_60 < 16; i_60++) {
                                            uint32_t _bf16x2_abs_59;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(pairs_13[i_60]));
                                            uint32_t _bf16x2_max_29;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(amax_pair_16), "r"(_bf16x2_abs_59));
                                            amax_pair_16 = _bf16x2_max_29;
                                        }
                                        uint16_t _bf16_max_29;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_29) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                        float _cvt_f32_bf16_29;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(_bf16_max_29)));
                                        float amax_16 = _cvt_f32_bf16_29;
                                        float _fmax_29 = fmaxf(amax_16 * 0.002232142857f, 1e-12f);
                                        float scale_16 = _fmax_29;
                                        uint16_t _ue8m0x2_f32_29;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_29) : "f"(scale_16), "f"(scale_16));
                                        unsigned int scale_byte_16 = (unsigned int)_ue8m0x2_f32_29 & 255;
                                        unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                        unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                        unsigned int words_15[8];
                                        #pragma unroll
                                        for (int i_61 = 0; i_61 < 8; i_61++) {
                                            uint32_t _bf16x2_mul_58;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(pairs_13[i_61 * 2]), "r"(inverse_16));
                                            uint16_t _e4m3x2_58;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_58) : "r"(_bf16x2_mul_58));
                                            uint32_t _bf16x2_mul_59;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(pairs_13[i_61 * 2 + 1]), "r"(inverse_16));
                                            uint16_t _e4m3x2_59;
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_59) : "r"(_bf16x2_mul_59));
                                            words_15[i_61] = (unsigned int)_e4m3x2_58 | (unsigned int)_e4m3x2_59 << 16;
                                        }
                                        scale_word_15 = scale_word_15 | scale_byte_16 << (unsigned int)(k_block_13 * 8);
                                        #pragma unroll
                                        for (int k_27 = 0; k_27 < 8; k_27++) {
                                            int col_0_11 = k_block_13 * 32 + ((tid - 128) * 4 + k_27 * 4) % 32;
                                            smem_v17[(row_0_16 * 128 + col_0_11) / 4] = words_15[k_27];
                                        }
                                    }
                                    smem_v18[row_0_16 % 32 * 4 + row_0_16 / 32] = scale_word_15;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_2 = row_17 - macro_row_offset_1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&hidden_q_store), col_31 * 128, local_row_2 * 128, smem_v17_addr);
                                    tma_store_3d((&hidden_sc_store), 0, 0, local_row_2 * col_blocks_6 + col_31, smem_v18_addr);
                                    tma_store_2d((&hidden_t_store), local_row_2 * 128, col_31 * 128, smem_v19_addr);
                                    tma_store_3d((&hidden_sc_t_store), 0, 0, col_31 * macro_tiles_1 + local_row_2, smem_v20_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                if (tile_end_1 > first_tile_1) {
                                    int row_18 = first_row_3;
                                    if (col_blocks_6 <= first_col_1) {
                                        row_18 = row_18 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_18 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 1) {
                                    int row_19 = first_row_3;
                                    if (col_blocks_6 <= first_col_1 + 1) {
                                        row_19 = row_19 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_19 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 2) {
                                    int row_20 = first_row_3;
                                    if (col_blocks_6 <= first_col_1 + 2) {
                                        row_20 = row_20 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_20 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        swiglu_bits = phase_bits_6;
                    }
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
                    if (cluster - comm_clusters >= 2 * shared_gate_tasks && cluster - comm_clusters < 2 * shared_gate_tasks + shared_swiglu) {
                        result_0 = 1;
                    }
                } else {
                    int mini_task_1 = (cluster - comm_clusters - shared_tasks) % mini_tasks;
                    if (mini_task_1 >= 2 * mini_gate && mini_task_1 < 2 * mini_gate + mini_swiglu) {
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
