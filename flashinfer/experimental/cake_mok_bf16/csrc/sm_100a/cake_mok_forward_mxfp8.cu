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
#define TMEM_NCOLS 512
#define TMEM_TMEM_SFA_OFFSET 0
#define TMEM_TMEM_SFB_OFFSET 12
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
#define SMEM_SMEM_V8_STAGE_BYTES 81920
#define SMEM_SMEM_V8_STRIDE 81920
#define SMEM_SMEM_V9_OFF 17408
#define SMEM_SMEM_V9_STAGE_BYTES 66048
#define SMEM_SMEM_V9_STRIDE 66048
#define SMEM_SMEM_V10_OFF 99328
#define SMEM_SMEM_V10_STAGE_BYTES 81920
#define SMEM_SMEM_V10_STRIDE 81920
#define SMEM_SMEM_V11_OFF 115712
#define SMEM_SMEM_V11_STAGE_BYTES 66048
#define SMEM_SMEM_V11_STRIDE 66048
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 131072
#define SMEM_DISPATCH_WORDS_STRIDE 131072
#define SMEM_DISPATCH_WEIGHTS_OFF 199680
#define SMEM_DISPATCH_WEIGHTS_STAGE_BYTES 512
#define SMEM_DISPATCH_WEIGHTS_STRIDE 512
#define SMEM_SMEM_V14_OFF 1024
#define SMEM_SMEM_V14_STAGE_BYTES 131072
#define SMEM_SMEM_V14_STRIDE 131072
#define SMEM_SMEM_V15_OFF 1280
#define SMEM_SMEM_V15_STAGE_BYTES 130816
#define SMEM_SMEM_V15_STRIDE 130816
#define SMEM_SMEM_V16_OFF 1536
#define SMEM_SMEM_V16_STAGE_BYTES 130560
#define SMEM_SMEM_V16_STRIDE 130560
#define SMEM_SMEM_V17_OFF 1792
#define SMEM_SMEM_V17_STAGE_BYTES 130304
#define SMEM_SMEM_V17_STRIDE 130304
#define SMEM_SMEM_V18_OFF 132096
#define SMEM_SMEM_V18_STAGE_BYTES 16384
#define SMEM_SMEM_V18_STRIDE 16384
#define SMEM_SMEM_V19_OFF 148480
#define SMEM_SMEM_V19_STAGE_BYTES 16384
#define SMEM_SMEM_V19_STRIDE 16384
#define SMEM_SMEM_V20_OFF 164864
#define SMEM_SMEM_V20_STAGE_BYTES 512
#define SMEM_SMEM_V20_STRIDE 512
#define SMEM_SMEM_V21_OFF 165376
#define SMEM_SMEM_V21_STAGE_BYTES 512
#define SMEM_SMEM_V21_STRIDE 512
#define SMEM_SMEM_V22_OFF 165888
#define SMEM_SMEM_V22_STAGE_BYTES 16384
#define SMEM_SMEM_V22_STRIDE 16384
#define SMEM_SMEM_V23_OFF 182272
#define SMEM_SMEM_V23_STAGE_BYTES 16384
#define SMEM_SMEM_V23_STRIDE 16384
#define SMEM_SMEM_V24_OFF 198656
#define SMEM_SMEM_V24_STAGE_BYTES 512
#define SMEM_SMEM_V24_STRIDE 512
#define SMEM_SMEM_V25_OFF 199168
#define SMEM_SMEM_V25_STAGE_BYTES 512
#define SMEM_SMEM_V25_STRIDE 512
#define SMEM_SMEM_V26_OFF 1024
#define SMEM_SMEM_V26_STAGE_BYTES 65536
#define SMEM_SMEM_V26_STRIDE 65536
#define SMEM_SMEM_V27_OFF 1280
#define SMEM_SMEM_V27_STAGE_BYTES 65280
#define SMEM_SMEM_V27_STRIDE 65280
#define SMEM_SMEM_V28_OFF 66560
#define SMEM_SMEM_V28_STAGE_BYTES 65536
#define SMEM_SMEM_V28_STRIDE 65536
#define SMEM_SMEM_V29_OFF 66816
#define SMEM_SMEM_V29_STAGE_BYTES 65280
#define SMEM_SMEM_V29_STRIDE 65280
#define SMEM_COMBINE_SMEM_OFF 1024
#define SMEM_COMBINE_SMEM_STAGE_BYTES 229376
#define SMEM_COMBINE_SMEM_STRIDE 229376
#define SMEM_COMBINE_SLOT_OFF 232440
#define SMEM_COMBINE_SLOT_STAGE_BYTES 8
#define SMEM_COMBINE_SLOT_STRIDE 8
#define SMEM_SMEM_V32_OFF 1024
#define SMEM_SMEM_V32_STAGE_BYTES 16384
#define SMEM_SMEM_V32_STRIDE 16384
#define SMEM_SMEM_V33_OFF 66560
#define SMEM_SMEM_V33_STAGE_BYTES 16384
#define SMEM_SMEM_V33_STRIDE 16384
#define SMEM_SMEM_V34_OFF 132096
#define SMEM_SMEM_V34_STAGE_BYTES 16384
#define SMEM_SMEM_V34_STRIDE 16384
#define SMEM_SMEM_V35_OFF 197632
#define SMEM_SMEM_V35_STAGE_BYTES 512
#define SMEM_SMEM_V35_STRIDE 512
#define SMEM_SMEM_V36_OFF 199680
#define SMEM_SMEM_V36_STAGE_BYTES 1024
#define SMEM_SMEM_V36_STRIDE 1024
#define SMEM_SMEM_V37_OFF 203776
#define SMEM_SMEM_V37_STAGE_BYTES 1024
#define SMEM_SMEM_V37_STRIDE 1024
#define SMEM_SMEM_V38_OFF 1024
#define SMEM_SMEM_V38_STAGE_BYTES 16384
#define SMEM_SMEM_V38_STRIDE 16384
#define SMEM_SMEM_V39_OFF 99328
#define SMEM_SMEM_V39_STAGE_BYTES 16384
#define SMEM_SMEM_V39_STRIDE 16384
#define SMEM_SMEM_V40_OFF 197632
#define SMEM_SMEM_V40_STAGE_BYTES 512
#define SMEM_SMEM_V40_STRIDE 512
#define SMEM_SMEM_V41_OFF 200704
#define SMEM_SMEM_V41_STAGE_BYTES 1024
#define SMEM_SMEM_V41_STRIDE 1024
#define SMEM_SMEM_V42_OFF 207872
#define SMEM_SMEM_V42_STAGE_BYTES 16384
#define SMEM_SMEM_V42_STRIDE 16384
#define SMEM_SMEM_V43_OFF 224256
#define SMEM_SMEM_V43_STAGE_BYTES 4096
#define SMEM_SMEM_V43_STRIDE 4096
#define SMEM_SMEM_V44_OFF 228352
#define SMEM_SMEM_V44_STAGE_BYTES 512
#define SMEM_SMEM_V44_STRIDE 512
#define SMEM_SMEM_V45_OFF 228864
#define SMEM_SMEM_V45_STAGE_BYTES 512
#define SMEM_SMEM_V45_STRIDE 512
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 752
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

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mok_forward_mxfp8(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap wd_q, const __grid_constant__ CUtensorMap wd_sc, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_q_store, const __grid_constant__ CUtensorMap hidden_sc_store, const __grid_constant__ CUtensorMap hidden_t_store, const __grid_constant__ CUtensorMap hidden_sc_t_store, const __grid_constant__ CUtensorMap hidden_q, const __grid_constant__ CUtensorMap hidden_sc, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, unsigned int* __restrict__ combine_next, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, CakeFastDivmod mini_tasks_div, CakeFastDivmod minis_per_macro_div, CakeFastDivmod gate_col_blocks_div, CakeFastDivmod down_col_blocks_div)
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
    #define scales_arrived_addr (mbar_base + 72)
    #define gemm_finished_addr (mbar_base + 120)
    #define scales_finished_addr (mbar_base + 168)
    #define output_arrived_addr (mbar_base + 216)
    #define output_finished_addr (mbar_base + 224)
    #define prefix_released_addr (mbar_base + 240)
    #define schedule_arrived_addr (mbar_base + 256)
    #define schedule_finished_addr (mbar_base + 264)
    #define drain_arrived_0_addr (mbar_base + 272)
    #define drain_arrived_1_addr (mbar_base + 280)
    #define drain_arrived_2_addr (mbar_base + 288)
    #define drain_arrived_3_addr (mbar_base + 296)
    #define drain_arrived_4_addr (mbar_base + 304)
    #define drain_arrived_5_addr (mbar_base + 312)
    #define drain_arrived_6_addr (mbar_base + 320)
    #define drain_arrived_7_addr (mbar_base + 328)
    #define drain_finished_0_addr (mbar_base + 336)
    #define drain_finished_1_addr (mbar_base + 344)
    #define drain_finished_2_addr (mbar_base + 352)
    #define drain_finished_3_addr (mbar_base + 360)
    #define drain_finished_4_addr (mbar_base + 368)
    #define drain_finished_5_addr (mbar_base + 376)
    #define drain_finished_6_addr (mbar_base + 384)
    #define drain_finished_7_addr (mbar_base + 392)
    #define dispatch_arrived_addr (mbar_base + 400)
    #define dispatch_arrived_hi_addr (mbar_base + 408)
    #define combine_arrived_addr (mbar_base + 416)
    #define decode_ready_addr (mbar_base + 472)
    #define decode_words_0_addr (mbar_base + 656)
    #define decode_words_1_addr (mbar_base + 672)
    #define decode_words_2_addr (mbar_base + 688)
    #define decode_words_3_addr (mbar_base + 704)
    #define decode_words_4_addr (mbar_base + 720)
    #define decode_words_5_addr (mbar_base + 736)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* a_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_A_SMEM_OFF);
    const int a_smem_addr = smem + SMEM_A_SMEM_OFF;
    __nv_bfloat16* b_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_B_SMEM_OFF);
    const int b_smem_addr = smem + SMEM_B_SMEM_OFF;
    __nv_bfloat16* b_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_B_HI_OFF);
    const int b_hi_addr = smem + SMEM_B_HI_OFF;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_D_SMEM_OFF);
    const int d_smem_addr = smem + SMEM_D_SMEM_OFF;
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_GATE_SMEM_OFF);
    const int gate_smem_addr = smem + SMEM_GATE_SMEM_OFF;
    __nv_bfloat16* up_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_UP_SMEM_OFF);
    const int up_smem_addr = smem + SMEM_UP_SMEM_OFF;
    __nv_bfloat16* hidden_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_HIDDEN_STAGING_OFF);
    const int hidden_staging_addr = smem + SMEM_HIDDEN_STAGING_OFF;
    unsigned int* hidden_words = reinterpret_cast<unsigned int*>(smem_raw + SMEM_HIDDEN_WORDS_OFF);
    const int hidden_words_addr = smem + SMEM_HIDDEN_WORDS_OFF;
    unsigned int* smem_v8 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V8_OFF);
    const int smem_v8_addr = smem + SMEM_SMEM_V8_OFF;
    unsigned int* smem_v9 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V9_OFF);
    const int smem_v9_addr = smem + SMEM_SMEM_V9_OFF;
    unsigned int* smem_v10 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V10_OFF);
    const int smem_v10_addr = smem + SMEM_SMEM_V10_OFF;
    unsigned int* smem_v11 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V11_OFF);
    const int smem_v11_addr = smem + SMEM_SMEM_V11_OFF;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + SMEM_DISPATCH_WORDS_OFF);
    const int dispatch_words_addr = smem + SMEM_DISPATCH_WORDS_OFF;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + SMEM_DISPATCH_WEIGHTS_OFF);
    const int dispatch_weights_addr = smem + SMEM_DISPATCH_WEIGHTS_OFF;
    __nv_bfloat16* smem_v14 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V14_OFF);
    const int smem_v14_addr = smem + SMEM_SMEM_V14_OFF;
    __nv_bfloat16* smem_v15 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V15_OFF);
    const int smem_v15_addr = smem + SMEM_SMEM_V15_OFF;
    __nv_bfloat16* smem_v16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V16_OFF);
    const int smem_v16_addr = smem + SMEM_SMEM_V16_OFF;
    __nv_bfloat16* smem_v17 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V17_OFF);
    const int smem_v17_addr = smem + SMEM_SMEM_V17_OFF;
    unsigned int* smem_v18 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V18_OFF);
    const int smem_v18_addr = smem + SMEM_SMEM_V18_OFF;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V19_OFF);
    const int smem_v19_addr = smem + SMEM_SMEM_V19_OFF;
    unsigned int* smem_v20 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V20_OFF);
    const int smem_v20_addr = smem + SMEM_SMEM_V20_OFF;
    unsigned int* smem_v21 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V21_OFF);
    const int smem_v21_addr = smem + SMEM_SMEM_V21_OFF;
    unsigned int* smem_v22 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V22_OFF);
    const int smem_v22_addr = smem + SMEM_SMEM_V22_OFF;
    unsigned int* smem_v23 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V23_OFF);
    const int smem_v23_addr = smem + SMEM_SMEM_V23_OFF;
    unsigned int* smem_v24 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V24_OFF);
    const int smem_v24_addr = smem + SMEM_SMEM_V24_OFF;
    unsigned int* smem_v25 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V25_OFF);
    const int smem_v25_addr = smem + SMEM_SMEM_V25_OFF;
    __nv_bfloat16* smem_v26 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V26_OFF);
    const int smem_v26_addr = smem + SMEM_SMEM_V26_OFF;
    __nv_bfloat16* smem_v27 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V27_OFF);
    const int smem_v27_addr = smem + SMEM_SMEM_V27_OFF;
    __nv_bfloat16* smem_v28 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V28_OFF);
    const int smem_v28_addr = smem + SMEM_SMEM_V28_OFF;
    __nv_bfloat16* smem_v29 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V29_OFF);
    const int smem_v29_addr = smem + SMEM_SMEM_V29_OFF;
    __nv_bfloat16* combine_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_COMBINE_SMEM_OFF);
    const int combine_smem_addr = smem + SMEM_COMBINE_SMEM_OFF;
    int* combine_slot = reinterpret_cast<int*>(smem_raw + SMEM_COMBINE_SLOT_OFF);
    const int combine_slot_addr = smem + SMEM_COMBINE_SLOT_OFF;
    uint8_t* smem_v32 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V32_OFF);
    const int smem_v32_addr = smem + SMEM_SMEM_V32_OFF;
    uint8_t* smem_v33 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V33_OFF);
    const int smem_v33_addr = smem + SMEM_SMEM_V33_OFF;
    uint8_t* smem_v34 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V34_OFF);
    const int smem_v34_addr = smem + SMEM_SMEM_V34_OFF;
    uint8_t* smem_v35 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V35_OFF);
    const int smem_v35_addr = smem + SMEM_SMEM_V35_OFF;
    uint8_t* smem_v36 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V36_OFF);
    const int smem_v36_addr = smem + SMEM_SMEM_V36_OFF;
    uint8_t* smem_v37 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V37_OFF);
    const int smem_v37_addr = smem + SMEM_SMEM_V37_OFF;
    uint8_t* smem_v38 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V38_OFF);
    const int smem_v38_addr = smem + SMEM_SMEM_V38_OFF;
    uint8_t* smem_v39 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V39_OFF);
    const int smem_v39_addr = smem + SMEM_SMEM_V39_OFF;
    uint8_t* smem_v40 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V40_OFF);
    const int smem_v40_addr = smem + SMEM_SMEM_V40_OFF;
    uint8_t* smem_v41 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V41_OFF);
    const int smem_v41_addr = smem + SMEM_SMEM_V41_OFF;
    unsigned int* smem_v42 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V42_OFF);
    const int smem_v42_addr = smem + SMEM_SMEM_V42_OFF;
    unsigned int* smem_v43 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V43_OFF);
    const int smem_v43_addr = smem + SMEM_SMEM_V43_OFF;
    unsigned int* smem_v44 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V44_OFF);
    const int smem_v44_addr = smem + SMEM_SMEM_V44_OFF;
    unsigned int* smem_v45 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_V45_OFF);
    const int smem_v45_addr = smem + SMEM_SMEM_V45_OFF;
    int tokens = num_tokens[0];
    int shared_rows = local_tokens / 256;
    int gate_base = shared_rows * (intermediate / 256);
    int shared_fused = shared_rows * (intermediate / 256);
    int gate_tiles = mini_size / 256 * (intermediate / 256);
    int mini_gate = 2 * gate_tiles;
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int shared_tasks = shared_fused + shared_rows * ((hidden + 511) / 512);
    int mini_tasks = mini_gate + mini_swiglu + mini_size / 256 * ((hidden + 256 - 1) / 256);
    int comm_clusters = comm_sms / 2;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis_per_macro = macro_size / mini_size;
    int true_minis = (tokens + mini_size - 1) / mini_size;
    int last_minis = true_minis - (macros - 1) * minis_per_macro;
    int true_clusters = comm_clusters + shared_tasks + true_minis * mini_tasks;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (36 pipeline groups, 0 ordered-sequence groups, 72 barriers)
    // Mbarriers at smem_raw[0..752)

    if (threadIdx.x == 0) {
        // swiglu_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        // gemm_arrived: 6 barriers, init_count=1
        mbarrier_init(smem + 24, 1);
        mbarrier_init(smem + 32, 1);
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        mbarrier_init(smem + 56, 1);
        mbarrier_init(smem + 64, 1);
        // scales_arrived: 6 barriers, init_count=1
        mbarrier_init(smem + 72, 1);
        mbarrier_init(smem + 80, 1);
        mbarrier_init(smem + 88, 1);
        mbarrier_init(smem + 96, 1);
        mbarrier_init(smem + 104, 1);
        mbarrier_init(smem + 112, 1);
        // gemm_finished: 6 barriers, init_count=1
        mbarrier_init(smem + 120, 1);
        mbarrier_init(smem + 128, 1);
        mbarrier_init(smem + 136, 1);
        mbarrier_init(smem + 144, 1);
        mbarrier_init(smem + 152, 1);
        mbarrier_init(smem + 160, 1);
        // scales_finished: 6 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        mbarrier_init(smem + 176, 1);
        mbarrier_init(smem + 184, 1);
        mbarrier_init(smem + 192, 1);
        mbarrier_init(smem + 200, 1);
        mbarrier_init(smem + 208, 1);
        // output_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 216, 1);
        // output_finished: 2 barriers, init_count=2
        mbarrier_init(smem + 224, 2);
        mbarrier_init(smem + 232, 2);
        // prefix_released: 2 barriers, init_count=2
        mbarrier_init(smem + 240, 2);
        mbarrier_init(smem + 248, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 256, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 264, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 272, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 280, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 288, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 296, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 304, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 312, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 320, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 328, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 336, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 344, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 352, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 360, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 368, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 376, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 384, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 392, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 400, 1);
        // dispatch_arrived_hi: 1 barriers, init_count=1
        mbarrier_init(smem + 408, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 416, 1);
        mbarrier_init(smem + 424, 1);
        mbarrier_init(smem + 432, 1);
        mbarrier_init(smem + 440, 1);
        mbarrier_init(smem + 448, 1);
        mbarrier_init(smem + 456, 1);
        mbarrier_init(smem + 464, 1);
        // decode_ready: 1 barriers, init_count=1
        mbarrier_init(smem + 472, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 752);
    if (warp == 0) {
        int _tmem_hold = smem + 752;
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
    const int tmem_tmem_sfa = taddr;
    const int tmem_tmem_sfb = taddr + 12;
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
            int last_rows = ((macro_size) < (tokens - (macros - 1) * macro_size) ? (macro_size) : (tokens - (macros - 1) * macro_size));
            int last_dispatch = last_rows / 128 * ((hidden + 511) / 512);
            unsigned int phase_bits = dispatch_bits;
            int col_blocks = (hidden + 511) / 512;
            int macro_offset = (macros - 1) * macro_size;
            int macro_tokens = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
            int tasks = macro_tokens / 128 * col_blocks;
            int task = comm_cta;
            if (task < tasks) {
                int _min_0 = ((256) < (hidden - task % col_blocks * 512) ? (256) : (hidden - task % col_blocks * 512));
                int peer = -1;
                int peer_token = -1;
                if (tid < 128) {
                    peer = schedule_rank[macro_offset + task / col_blocks * 128 + tid];
                    peer_token = schedule_token[macro_offset + task / col_blocks * 128 + tid];
                }
                uint32_t _cta_count_0 = __syncthreads_count(peer >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_0 * (unsigned int)(_min_0 * 2));
                }
                __syncthreads();
                if (peer >= 0) {
                    cp_async_bulk_gmem2smem(smem_v26_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(task % col_blocks * 512)) * (unsigned long long)2)), _min_0 * 2, dispatch_arrived_addr);
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
                int chunk_cols = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                int lo_cols = ((256) < (chunk_cols) ? (256) : (chunk_cols));
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
                        cp_async_bulk_gmem2smem(smem_v28_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512) + 256) * (unsigned long long)2)), hi_cols * 2, dispatch_arrived_hi_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_1 = 0; vec_1 < 32; vec_1++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (65536 + tid * 512 + vec_1 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                unsigned int bits = phase_bits;
                unsigned int _parity_dispatch_arrived = bits % 2;
                mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived);
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
                                unsigned int w0_0 = dispatch_words[col * word_stride + r];
                                unsigned int w1_1 = dispatch_words[(col + 1) * word_stride + r];
                                uint32_t lo;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo) : "r"(w0_0), "r"(w1_1));
                                uint32_t hi;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi) : "r"(w0_0), "r"(w1_1));
                                pairs0[k] = lo;
                                pairs1[k] = hi;
                            }
                            uint32_t _bf16x2_abs_0;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(pairs0[0]));
                            unsigned int amax_pair = _bf16x2_abs_0;
                            #pragma unroll
                            for (int i = 1; i < 16; i++) {
                                uint32_t _bf16x2_abs_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(pairs0[i]));
                                uint32_t _bf16x2_max_0;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(amax_pair), "r"(_bf16x2_abs_1));
                                amax_pair = _bf16x2_max_0;
                            }
                            uint16_t _bf16_max_0;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(amax_pair & 65535)), "h"((uint16_t)(amax_pair >> 16)));
                            float amax;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax) : "h"((uint16_t)(_bf16_max_0)));
                            float scale = fmaxf(amax * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_0;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_0) : "f"(scale), "f"(scale));
                            unsigned int scale_byte = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_0 & 255;
                            unsigned int inverse_lane = 254 - scale_byte << 7;
                            unsigned int inverse = inverse_lane | inverse_lane << 16;
                            unsigned int words[8];
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 8; i_1++) {
                                uint32_t _bf16x2_mul_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(pairs0[i_1 * 2]), "r"(inverse));
                                uint16_t _e4m3x2_bf16x2_0;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_0) : "r"(_bf16x2_mul_0));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_0);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_0) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t _bf16x2_mul_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(pairs0[i_1 * 2 + 1]), "r"(inverse));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_1;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_1) : "r"(_bf16x2_mul_1));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_1);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words[i_1] = (unsigned int)_e4m3x2_bf16x2_0 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_1 << 16;
                            }
                            uint32_t _bf16x2_abs_2;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(pairs1[0]));
                            unsigned int amax_pair_0 = _bf16x2_abs_2;
                            #pragma unroll
                            for (int i_2 = 1; i_2 < 16; i_2++) {
                                uint32_t _bf16x2_abs_3;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(pairs1[i_2]));
                                uint32_t _bf16x2_max_1;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(amax_pair_0), "r"(_bf16x2_abs_3));
                                amax_pair_0 = _bf16x2_max_1;
                            }
                            uint16_t _bf16_max_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_1) : "h"((uint16_t)(amax_pair_0 & 65535)), "h"((uint16_t)(amax_pair_0 >> 16)));
                            float amax_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1) : "h"((uint16_t)(_bf16_max_1)));
                            float scale_2 = fmaxf(amax_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_2;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_2) : "f"(scale_2), "f"(scale_2));
                            unsigned int scale_byte_3 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_2 & 255;
                            unsigned int inverse_lane_4 = 254 - scale_byte_3 << 7;
                            unsigned int inverse_5 = inverse_lane_4 | inverse_lane_4 << 16;
                            unsigned int words_6[8];
                            #pragma unroll
                            for (int i_3 = 0; i_3 < 8; i_3++) {
                                uint32_t _bf16x2_mul_2;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(pairs1[i_3 * 2]), "r"(inverse_5));
                                uint16_t _e4m3x2_bf16x2_1;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_1) : "r"(_bf16x2_mul_2));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_2);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t _bf16x2_mul_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(pairs1[i_3 * 2 + 1]), "r"(inverse_5));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_3;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "r"(_bf16x2_mul_3));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_3);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_6[i_3] = (unsigned int)_e4m3x2_bf16x2_1 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_3 << 16;
                            }
                            #pragma unroll
                            for (int k_1 = 0; k_1 < 8; k_1++) {
                                int col_1 = k_block * 32 + (tid * 4 + k_1 * 4) % 32;
                                smem_v22[(row0 * 128 + col_1) / 4] = words[k_1];
                                smem_v22[(row1 * 128 + col_1) / 4] = words_6[k_1];
                            }
                            {
                                uint32_t _byte_4 = static_cast<uint32_t>(scale_byte) & 0xFFu;
                                uint32_t _addr_4 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0 + k_block));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_4), "r"(_byte_4) : "memory");
                            }
                            {
                                uint32_t _byte_5 = static_cast<uint32_t>(scale_byte_3) & 0xFFu;
                                uint32_t _addr_5 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1 + k_block));
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
                                pairs[k_2] = dispatch_words[(row_0 * 256 + col_2) / 2];
                            }
                            uint32_t _bf16x2_abs_4;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(pairs[0]));
                            unsigned int amax_pair_1 = _bf16x2_abs_4;
                            #pragma unroll
                            for (int i_4 = 1; i_4 < 16; i_4++) {
                                uint32_t _bf16x2_abs_5;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(pairs[i_4]));
                                uint32_t _bf16x2_max_2;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(amax_pair_1), "r"(_bf16x2_abs_5));
                                amax_pair_1 = _bf16x2_max_2;
                            }
                            uint16_t _bf16_max_2;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_2) : "h"((uint16_t)(amax_pair_1 & 65535)), "h"((uint16_t)(amax_pair_1 >> 16)));
                            float amax_2;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2) : "h"((uint16_t)(_bf16_max_2)));
                            float scale_1 = fmaxf(amax_2 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_6;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_6) : "f"(scale_1), "f"(scale_1));
                            unsigned int scale_byte_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_6 & 255;
                            unsigned int inverse_lane_1 = 254 - scale_byte_1 << 7;
                            unsigned int inverse_1 = inverse_lane_1 | inverse_lane_1 << 16;
                            unsigned int words_1[8];
                            #pragma unroll
                            for (int i_5 = 0; i_5 < 8; i_5++) {
                                uint32_t _bf16x2_mul_4;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(pairs[i_5 * 2]), "r"(inverse_1));
                                uint16_t _e4m3x2_bf16x2_2;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_2) : "r"(_bf16x2_mul_4));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_4);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t _bf16x2_mul_5;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(pairs[i_5 * 2 + 1]), "r"(inverse_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_7;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_7) : "r"(_bf16x2_mul_5));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_5);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_1[i_5] = (unsigned int)_e4m3x2_bf16x2_2 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_7 << 16;
                            }
                            scale_word = scale_word | scale_byte_1 << (unsigned int)(k_block_1 * 8);
                            #pragma unroll
                            for (int k_3 = 0; k_3 < 8; k_3++) {
                                int col_3 = k_block_1 * 32 + ((tid - 128) * 4 + k_3 * 4) % 32;
                                smem_v18[(row_0 * 128 + col_3) / 4] = words_1[k_3];
                            }
                        }
                        smem_v20[row_0 % 32 * 4 + row_0 / 32] = scale_word;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile = col_block * 4;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile * 128, row, smem_v18_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile, smem_v20_addr);
                        tma_store_2d((&x_t_store), row, col_tile * 128, smem_v22_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile * macro_tiles + row_tile, smem_v24_addr);
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
                                unsigned int w0_0_1 = dispatch_words[64 + col_4 * word_stride_1 + r_1];
                                unsigned int w1_1_1 = dispatch_words[64 + (col_4 + 1) * word_stride_1 + r_1];
                                uint32_t lo_1;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_1) : "r"(w0_0_1), "r"(w1_1_1));
                                uint32_t hi_1;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_1) : "r"(w0_0_1), "r"(w1_1_1));
                                pairs0_1[k_4] = lo_1;
                                pairs1_1[k_4] = hi_1;
                            }
                            uint32_t _bf16x2_abs_6;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(pairs0_1[0]));
                            unsigned int amax_pair_2 = _bf16x2_abs_6;
                            #pragma unroll
                            for (int i_6 = 1; i_6 < 16; i_6++) {
                                uint32_t _bf16x2_abs_7;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(pairs0_1[i_6]));
                                uint32_t _bf16x2_max_3;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(amax_pair_2), "r"(_bf16x2_abs_7));
                                amax_pair_2 = _bf16x2_max_3;
                            }
                            uint16_t _bf16_max_3;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_3) : "h"((uint16_t)(amax_pair_2 & 65535)), "h"((uint16_t)(amax_pair_2 >> 16)));
                            float amax_3;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_3) : "h"((uint16_t)(_bf16_max_3)));
                            float scale_3 = fmaxf(amax_3 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_8;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_8) : "f"(scale_3), "f"(scale_3));
                            unsigned int scale_byte_2 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_8 & 255;
                            unsigned int inverse_lane_2 = 254 - scale_byte_2 << 7;
                            unsigned int inverse_2 = inverse_lane_2 | inverse_lane_2 << 16;
                            unsigned int words_2[8];
                            #pragma unroll
                            for (int i_7 = 0; i_7 < 8; i_7++) {
                                uint32_t _bf16x2_mul_6;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(pairs0_1[i_7 * 2]), "r"(inverse_2));
                                uint16_t _e4m3x2_bf16x2_3;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_3) : "r"(_bf16x2_mul_6));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_6);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t _bf16x2_mul_7;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(pairs0_1[i_7 * 2 + 1]), "r"(inverse_2));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_9;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_9) : "r"(_bf16x2_mul_7));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_7);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_2[i_7] = (unsigned int)_e4m3x2_bf16x2_3 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_9 << 16;
                            }
                            uint32_t _bf16x2_abs_8;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(pairs1_1[0]));
                            unsigned int amax_pair_0_1 = _bf16x2_abs_8;
                            #pragma unroll
                            for (int i_8 = 1; i_8 < 16; i_8++) {
                                uint32_t _bf16x2_abs_9;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(pairs1_1[i_8]));
                                uint32_t _bf16x2_max_4;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(amax_pair_0_1), "r"(_bf16x2_abs_9));
                                amax_pair_0_1 = _bf16x2_max_4;
                            }
                            uint16_t _bf16_max_4;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_4) : "h"((uint16_t)(amax_pair_0_1 & 65535)), "h"((uint16_t)(amax_pair_0_1 >> 16)));
                            float amax_1_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_1) : "h"((uint16_t)(_bf16_max_4)));
                            float scale_2_1 = fmaxf(amax_1_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_10;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_10) : "f"(scale_2_1), "f"(scale_2_1));
                            unsigned int scale_byte_3_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_10 & 255;
                            unsigned int inverse_lane_4_1 = 254 - scale_byte_3_1 << 7;
                            unsigned int inverse_5_1 = inverse_lane_4_1 | inverse_lane_4_1 << 16;
                            unsigned int words_6_1[8];
                            #pragma unroll
                            for (int i_9 = 0; i_9 < 8; i_9++) {
                                uint32_t _bf16x2_mul_8;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(pairs1_1[i_9 * 2]), "r"(inverse_5_1));
                                uint16_t _e4m3x2_bf16x2_4;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_4) : "r"(_bf16x2_mul_8));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_8);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t _bf16x2_mul_9;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(pairs1_1[i_9 * 2 + 1]), "r"(inverse_5_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_11;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_11) : "r"(_bf16x2_mul_9));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_9);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_6_1[i_9] = (unsigned int)_e4m3x2_bf16x2_4 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_11 << 16;
                            }
                            #pragma unroll
                            for (int k_5 = 0; k_5 < 8; k_5++) {
                                int col_5 = k_block_2 * 32 + (tid * 4 + k_5 * 4) % 32;
                                smem_v23[(row0_1 * 128 + col_5) / 4] = words_2[k_5];
                                smem_v23[(row1_1 * 128 + col_5) / 4] = words_6_1[k_5];
                            }
                            {
                                uint32_t _byte_12 = static_cast<uint32_t>(scale_byte_2) & 0xFFu;
                                uint32_t _addr_12 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_1 + k_block_2));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_12), "r"(_byte_12) : "memory");
                            }
                            {
                                uint32_t _byte_13 = static_cast<uint32_t>(scale_byte_3_1) & 0xFFu;
                                uint32_t _addr_13 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_1 + k_block_2));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_13), "r"(_byte_13) : "memory");
                            }
                        }
                    } else {
                        int row_0_1 = tid - 128;
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
                                pairs_1[k_6] = dispatch_words[64 + (row_0_1 * 256 + col_6) / 2];
                            }
                            uint32_t _bf16x2_abs_10;
                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(pairs_1[0]));
                            unsigned int amax_pair_3 = _bf16x2_abs_10;
                            #pragma unroll
                            for (int i_10 = 1; i_10 < 16; i_10++) {
                                uint32_t _bf16x2_abs_11;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(pairs_1[i_10]));
                                uint32_t _bf16x2_max_5;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(amax_pair_3), "r"(_bf16x2_abs_11));
                                amax_pair_3 = _bf16x2_max_5;
                            }
                            uint16_t _bf16_max_5;
                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_5) : "h"((uint16_t)(amax_pair_3 & 65535)), "h"((uint16_t)(amax_pair_3 >> 16)));
                            float amax_4;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_4) : "h"((uint16_t)(_bf16_max_5)));
                            float scale_4 = fmaxf(amax_4 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_14;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_14) : "f"(scale_4), "f"(scale_4));
                            unsigned int scale_byte_4 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_14 & 255;
                            unsigned int inverse_lane_3 = 254 - scale_byte_4 << 7;
                            unsigned int inverse_3 = inverse_lane_3 | inverse_lane_3 << 16;
                            unsigned int words_3[8];
                            #pragma unroll
                            for (int i_11 = 0; i_11 < 8; i_11++) {
                                uint32_t _bf16x2_mul_10;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(pairs_1[i_11 * 2]), "r"(inverse_3));
                                uint16_t _e4m3x2_bf16x2_5;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_5) : "r"(_bf16x2_mul_10));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_10);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                uint32_t _bf16x2_mul_11;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(pairs_1[i_11 * 2 + 1]), "r"(inverse_3));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_15;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_15) : "r"(_bf16x2_mul_11));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_11);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_3[i_11] = (unsigned int)_e4m3x2_bf16x2_5 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_15 << 16;
                            }
                            scale_word_1 = scale_word_1 | scale_byte_4 << (unsigned int)(k_block_3 * 8);
                            #pragma unroll
                            for (int k_7 = 0; k_7 < 8; k_7++) {
                                int col_7 = k_block_3 * 32 + ((tid - 128) * 4 + k_7 * 4) % 32;
                                smem_v19[(row_0_1 * 128 + col_7) / 4] = words_3[k_7];
                            }
                        }
                        smem_v21[row_0_1 % 32 * 4 + row_0_1 / 32] = scale_word_1;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_1 = col_block * 4 + 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_1 * 128, row, smem_v19_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile_1, smem_v21_addr);
                        tma_store_2d((&x_t_store), row, col_tile_1 * 128, smem_v23_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_1 * macro_tiles + row_tile, smem_v25_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                phase_bits = bits;
                int next_task = task + comm_sms;
                if (next_task < tasks) {
                    int _min_1 = ((256) < (hidden - next_task % col_blocks * 512) ? (256) : (hidden - next_task % col_blocks * 512));
                    int peer_2 = -1;
                    int peer_token_2 = -1;
                    if (tid < 128) {
                        peer_2 = schedule_rank[macro_offset + next_task / col_blocks * 128 + tid];
                        peer_token_2 = schedule_token[macro_offset + next_task / col_blocks * 128 + tid];
                    }
                    uint32_t _cta_count_2 = __syncthreads_count(peer_2 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * (unsigned int)(_min_1 * 2));
                    }
                    __syncthreads();
                    if (peer_2 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v26_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task % col_blocks * 512)) * (unsigned long long)2)), _min_1 * 2, dispatch_arrived_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_2 = 0; vec_2 < 32; vec_2++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 512 + vec_2 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                if (hi_cols > 0) {
                    unsigned int bits_0 = phase_bits;
                    unsigned int _parity_dispatch_arrived_hi = (bits_0 >> 1) % 2;
                    mbarrier_wait(dispatch_arrived_hi_addr, _parity_dispatch_arrived_hi);
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
                                    unsigned int w0_0_2 = dispatch_words[16384 + col_8 * word_stride_2 + r_2];
                                    unsigned int w1_1_2 = dispatch_words[16384 + (col_8 + 1) * word_stride_2 + r_2];
                                    uint32_t lo_2;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_2) : "r"(w0_0_2), "r"(w1_1_2));
                                    uint32_t hi_2;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_2) : "r"(w0_0_2), "r"(w1_1_2));
                                    pairs0_2[k_8] = lo_2;
                                    pairs1_2[k_8] = hi_2;
                                }
                                uint32_t _bf16x2_abs_12;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(pairs0_2[0]));
                                unsigned int amax_pair_4 = _bf16x2_abs_12;
                                #pragma unroll
                                for (int i_12 = 1; i_12 < 16; i_12++) {
                                    uint32_t _bf16x2_abs_13;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(pairs0_2[i_12]));
                                    uint32_t _bf16x2_max_6;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(amax_pair_4), "r"(_bf16x2_abs_13));
                                    amax_pair_4 = _bf16x2_max_6;
                                }
                                uint16_t _bf16_max_6;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_6) : "h"((uint16_t)(amax_pair_4 & 65535)), "h"((uint16_t)(amax_pair_4 >> 16)));
                                float amax_5;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_5) : "h"((uint16_t)(_bf16_max_6)));
                                float scale_5 = fmaxf(amax_5 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_16;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_16) : "f"(scale_5), "f"(scale_5));
                                unsigned int scale_byte_5 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_16 & 255;
                                unsigned int inverse_lane_5 = 254 - scale_byte_5 << 7;
                                unsigned int inverse_4 = inverse_lane_5 | inverse_lane_5 << 16;
                                unsigned int words_4[8];
                                #pragma unroll
                                for (int i_13 = 0; i_13 < 8; i_13++) {
                                    uint32_t _bf16x2_mul_12;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(pairs0_2[i_13 * 2]), "r"(inverse_4));
                                    uint16_t _e4m3x2_bf16x2_6;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_6) : "r"(_bf16x2_mul_12));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_12);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t _bf16x2_mul_13;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(pairs0_2[i_13 * 2 + 1]), "r"(inverse_4));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_17;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_17) : "r"(_bf16x2_mul_13));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_13);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_4[i_13] = (unsigned int)_e4m3x2_bf16x2_6 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_17 << 16;
                                }
                                uint32_t _bf16x2_abs_14;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(pairs1_2[0]));
                                unsigned int amax_pair_0_2 = _bf16x2_abs_14;
                                #pragma unroll
                                for (int i_14 = 1; i_14 < 16; i_14++) {
                                    uint32_t _bf16x2_abs_15;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(pairs1_2[i_14]));
                                    uint32_t _bf16x2_max_7;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(amax_pair_0_2), "r"(_bf16x2_abs_15));
                                    amax_pair_0_2 = _bf16x2_max_7;
                                }
                                uint16_t _bf16_max_7;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_7) : "h"((uint16_t)(amax_pair_0_2 & 65535)), "h"((uint16_t)(amax_pair_0_2 >> 16)));
                                float amax_1_2;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_2) : "h"((uint16_t)(_bf16_max_7)));
                                float scale_2_2 = fmaxf(amax_1_2 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_18;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_18) : "f"(scale_2_2), "f"(scale_2_2));
                                unsigned int scale_byte_3_2 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_18 & 255;
                                unsigned int inverse_lane_4_2 = 254 - scale_byte_3_2 << 7;
                                unsigned int inverse_5_2 = inverse_lane_4_2 | inverse_lane_4_2 << 16;
                                unsigned int words_6_2[8];
                                #pragma unroll
                                for (int i_15 = 0; i_15 < 8; i_15++) {
                                    uint32_t _bf16x2_mul_14;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(pairs1_2[i_15 * 2]), "r"(inverse_5_2));
                                    uint16_t _e4m3x2_bf16x2_7;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_7) : "r"(_bf16x2_mul_14));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_14);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t _bf16x2_mul_15;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(pairs1_2[i_15 * 2 + 1]), "r"(inverse_5_2));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_19;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "r"(_bf16x2_mul_15));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_15);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_6_2[i_15] = (unsigned int)_e4m3x2_bf16x2_7 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_19 << 16;
                                }
                                #pragma unroll
                                for (int k_9 = 0; k_9 < 8; k_9++) {
                                    int col_9 = k_block_4 * 32 + (tid * 4 + k_9 * 4) % 32;
                                    smem_v22[(row0_2 * 128 + col_9) / 4] = words_4[k_9];
                                    smem_v22[(row1_2 * 128 + col_9) / 4] = words_6_2[k_9];
                                }
                                {
                                    uint32_t _byte_20 = static_cast<uint32_t>(scale_byte_5) & 0xFFu;
                                    uint32_t _addr_20 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0_2 + k_block_4));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_20), "r"(_byte_20) : "memory");
                                }
                                {
                                    uint32_t _byte_21 = static_cast<uint32_t>(scale_byte_3_2) & 0xFFu;
                                    uint32_t _addr_21 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1_2 + k_block_4));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_21), "r"(_byte_21) : "memory");
                                }
                            }
                        } else {
                            int row_0_2 = tid - 128;
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
                                    pairs_2[k_10] = dispatch_words[16384 + (row_0_2 * 256 + col_10) / 2];
                                }
                                uint32_t _bf16x2_abs_16;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(pairs_2[0]));
                                unsigned int amax_pair_5 = _bf16x2_abs_16;
                                #pragma unroll
                                for (int i_16 = 1; i_16 < 16; i_16++) {
                                    uint32_t _bf16x2_abs_17;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(pairs_2[i_16]));
                                    uint32_t _bf16x2_max_8;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_8) : "r"(amax_pair_5), "r"(_bf16x2_abs_17));
                                    amax_pair_5 = _bf16x2_max_8;
                                }
                                uint16_t _bf16_max_8;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_8) : "h"((uint16_t)(amax_pair_5 & 65535)), "h"((uint16_t)(amax_pair_5 >> 16)));
                                float amax_6;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_6) : "h"((uint16_t)(_bf16_max_8)));
                                float scale_6 = fmaxf(amax_6 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_22;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_22) : "f"(scale_6), "f"(scale_6));
                                unsigned int scale_byte_6 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_22 & 255;
                                unsigned int inverse_lane_6 = 254 - scale_byte_6 << 7;
                                unsigned int inverse_6 = inverse_lane_6 | inverse_lane_6 << 16;
                                unsigned int words_5[8];
                                #pragma unroll
                                for (int i_17 = 0; i_17 < 8; i_17++) {
                                    uint32_t _bf16x2_mul_16;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(pairs_2[i_17 * 2]), "r"(inverse_6));
                                    uint16_t _e4m3x2_bf16x2_8;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_8) : "r"(_bf16x2_mul_16));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_16);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_8) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t _bf16x2_mul_17;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(pairs_2[i_17 * 2 + 1]), "r"(inverse_6));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_23;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_23) : "r"(_bf16x2_mul_17));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_17);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_5[i_17] = (unsigned int)_e4m3x2_bf16x2_8 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_23 << 16;
                                }
                                scale_word_2 = scale_word_2 | scale_byte_6 << (unsigned int)(k_block_5 * 8);
                                #pragma unroll
                                for (int k_11 = 0; k_11 < 8; k_11++) {
                                    int col_11 = k_block_5 * 32 + ((tid - 128) * 4 + k_11 * 4) % 32;
                                    smem_v18[(row_0_2 * 128 + col_11) / 4] = words_5[k_11];
                                }
                            }
                            smem_v20[row_0_2 % 32 * 4 + row_0_2 / 32] = scale_word_2;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_2 = col_block * 4 + 2;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_2 * 128, row, smem_v18_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_2 + col_tile_2, smem_v20_addr);
                            tma_store_2d((&x_t_store), row, col_tile_2 * 128, smem_v22_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_2 * macro_tiles_3 + row_tile_1, smem_v24_addr);
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
                                    unsigned int w0_0_3 = dispatch_words[16448 + col_12 * word_stride_3 + r_3];
                                    unsigned int w1_1_3 = dispatch_words[16448 + (col_12 + 1) * word_stride_3 + r_3];
                                    uint32_t lo_3;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_3) : "r"(w0_0_3), "r"(w1_1_3));
                                    uint32_t hi_3;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_3) : "r"(w0_0_3), "r"(w1_1_3));
                                    pairs0_3[k_12] = lo_3;
                                    pairs1_3[k_12] = hi_3;
                                }
                                uint32_t _bf16x2_abs_18;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(pairs0_3[0]));
                                unsigned int amax_pair_6 = _bf16x2_abs_18;
                                #pragma unroll
                                for (int i_18 = 1; i_18 < 16; i_18++) {
                                    uint32_t _bf16x2_abs_19;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(pairs0_3[i_18]));
                                    uint32_t _bf16x2_max_9;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(amax_pair_6), "r"(_bf16x2_abs_19));
                                    amax_pair_6 = _bf16x2_max_9;
                                }
                                uint16_t _bf16_max_9;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_9) : "h"((uint16_t)(amax_pair_6 & 65535)), "h"((uint16_t)(amax_pair_6 >> 16)));
                                float amax_7;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_7) : "h"((uint16_t)(_bf16_max_9)));
                                float scale_7 = fmaxf(amax_7 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_24;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_24) : "f"(scale_7), "f"(scale_7));
                                unsigned int scale_byte_7 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_24 & 255;
                                unsigned int inverse_lane_7 = 254 - scale_byte_7 << 7;
                                unsigned int inverse_7 = inverse_lane_7 | inverse_lane_7 << 16;
                                unsigned int words_7[8];
                                #pragma unroll
                                for (int i_19 = 0; i_19 < 8; i_19++) {
                                    uint32_t _bf16x2_mul_18;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(pairs0_3[i_19 * 2]), "r"(inverse_7));
                                    uint16_t _e4m3x2_bf16x2_9;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_9) : "r"(_bf16x2_mul_18));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_18);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t _bf16x2_mul_19;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(pairs0_3[i_19 * 2 + 1]), "r"(inverse_7));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_25;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_25) : "r"(_bf16x2_mul_19));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_19);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_7[i_19] = (unsigned int)_e4m3x2_bf16x2_9 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_25 << 16;
                                }
                                uint32_t _bf16x2_abs_20;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(pairs1_3[0]));
                                unsigned int amax_pair_0_3 = _bf16x2_abs_20;
                                #pragma unroll
                                for (int i_20 = 1; i_20 < 16; i_20++) {
                                    uint32_t _bf16x2_abs_21;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(pairs1_3[i_20]));
                                    uint32_t _bf16x2_max_10;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(amax_pair_0_3), "r"(_bf16x2_abs_21));
                                    amax_pair_0_3 = _bf16x2_max_10;
                                }
                                uint16_t _bf16_max_10;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_10) : "h"((uint16_t)(amax_pair_0_3 & 65535)), "h"((uint16_t)(amax_pair_0_3 >> 16)));
                                float amax_1_3;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_3) : "h"((uint16_t)(_bf16_max_10)));
                                float scale_2_3 = fmaxf(amax_1_3 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_26;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_26) : "f"(scale_2_3), "f"(scale_2_3));
                                unsigned int scale_byte_3_3 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_26 & 255;
                                unsigned int inverse_lane_4_3 = 254 - scale_byte_3_3 << 7;
                                unsigned int inverse_5_3 = inverse_lane_4_3 | inverse_lane_4_3 << 16;
                                unsigned int words_6_3[8];
                                #pragma unroll
                                for (int i_21 = 0; i_21 < 8; i_21++) {
                                    uint32_t _bf16x2_mul_20;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(pairs1_3[i_21 * 2]), "r"(inverse_5_3));
                                    uint16_t _e4m3x2_bf16x2_10;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_10) : "r"(_bf16x2_mul_20));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_20);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t _bf16x2_mul_21;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(pairs1_3[i_21 * 2 + 1]), "r"(inverse_5_3));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_27;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_27) : "r"(_bf16x2_mul_21));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_21);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_6_3[i_21] = (unsigned int)_e4m3x2_bf16x2_10 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_27 << 16;
                                }
                                #pragma unroll
                                for (int k_13 = 0; k_13 < 8; k_13++) {
                                    int col_13 = k_block_6 * 32 + (tid * 4 + k_13 * 4) % 32;
                                    smem_v23[(row0_3 * 128 + col_13) / 4] = words_7[k_13];
                                    smem_v23[(row1_3 * 128 + col_13) / 4] = words_6_3[k_13];
                                }
                                {
                                    uint32_t _byte_28 = static_cast<uint32_t>(scale_byte_7) & 0xFFu;
                                    uint32_t _addr_28 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_3 + k_block_6));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_28), "r"(_byte_28) : "memory");
                                }
                                {
                                    uint32_t _byte_29 = static_cast<uint32_t>(scale_byte_3_3) & 0xFFu;
                                    uint32_t _addr_29 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_3 + k_block_6));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_29), "r"(_byte_29) : "memory");
                                }
                            }
                        } else {
                            int row_0_3 = tid - 128;
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
                                    pairs_3[k_14] = dispatch_words[16448 + (row_0_3 * 256 + col_14) / 2];
                                }
                                uint32_t _bf16x2_abs_22;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(pairs_3[0]));
                                unsigned int amax_pair_7 = _bf16x2_abs_22;
                                #pragma unroll
                                for (int i_22 = 1; i_22 < 16; i_22++) {
                                    uint32_t _bf16x2_abs_23;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(pairs_3[i_22]));
                                    uint32_t _bf16x2_max_11;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(amax_pair_7), "r"(_bf16x2_abs_23));
                                    amax_pair_7 = _bf16x2_max_11;
                                }
                                uint16_t _bf16_max_11;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_11) : "h"((uint16_t)(amax_pair_7 & 65535)), "h"((uint16_t)(amax_pair_7 >> 16)));
                                float amax_8;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8) : "h"((uint16_t)(_bf16_max_11)));
                                float scale_8 = fmaxf(amax_8 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_30;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_30) : "f"(scale_8), "f"(scale_8));
                                unsigned int scale_byte_8 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_30 & 255;
                                unsigned int inverse_lane_8 = 254 - scale_byte_8 << 7;
                                unsigned int inverse_8 = inverse_lane_8 | inverse_lane_8 << 16;
                                unsigned int words_8[8];
                                #pragma unroll
                                for (int i_23 = 0; i_23 < 8; i_23++) {
                                    uint32_t _bf16x2_mul_22;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(pairs_3[i_23 * 2]), "r"(inverse_8));
                                    uint16_t _e4m3x2_bf16x2_11;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_11) : "r"(_bf16x2_mul_22));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_22);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    uint32_t _bf16x2_mul_23;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(pairs_3[i_23 * 2 + 1]), "r"(inverse_8));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_31;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_31) : "r"(_bf16x2_mul_23));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_23);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_8[i_23] = (unsigned int)_e4m3x2_bf16x2_11 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_31 << 16;
                                }
                                scale_word_3 = scale_word_3 | scale_byte_8 << (unsigned int)(k_block_7 * 8);
                                #pragma unroll
                                for (int k_15 = 0; k_15 < 8; k_15++) {
                                    int col_15 = k_block_7 * 32 + ((tid - 128) * 4 + k_15 * 4) % 32;
                                    smem_v19[(row_0_3 * 128 + col_15) / 4] = words_8[k_15];
                                }
                            }
                            smem_v21[row_0_3 % 32 * 4 + row_0_3 / 32] = scale_word_3;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_3 = col_block * 4 + 2 + 1;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_3 * 128, row, smem_v19_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_2 + col_tile_3, smem_v21_addr);
                            tma_store_2d((&x_t_store), row, col_tile_3 * 128, smem_v23_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_3 * macro_tiles_3 + row_tile_1, smem_v25_addr);
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
                int macro_rows = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                int dispatch_tasks = 0;
                if (macro > 0) {
                    int previous_rows = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                    dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                }
                if (macro == 0) {
                    unsigned int phase_bits_0 = combine_bits;
                    int col_blocks_1 = (hidden + 1023) / 1024;
                    int macro_offset_2 = 0;
                    int macro_tokens_3 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
                    int num_tasks = (macro_tokens_3 / 16 * col_blocks_1 + 6) / 7;
                    if (tid == 0) {
                        unsigned int _atomic_fetch_add_32 = atomicAdd(&combine_next[0], 1);
                        combine_slot[0] = (int)_atomic_fetch_add_32;
                    }
                    __syncthreads();
                    int task_4 = combine_slot[0];
                    __syncthreads();
                    while (task_4 < num_tasks) {
                        unsigned int phase_bits_1 = phase_bits_0;
                        int col_blocks_2 = (hidden + 1023) / 1024;
                        int first_tile = task_4 * 7;
                        int macro_offset_3 = 0;
                        int macro_tokens_4 = ((macro_size) < (tokens - macro_offset_3) ? (macro_size) : (tokens - macro_offset_3));
                        int valid_tiles = ((7) < (macro_tokens_4 / 16 * col_blocks_2 - first_tile) ? (7) : (macro_tokens_4 / 16 * col_blocks_2 - first_tile));
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
                                        uint32_t _barrier_vote_33 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _barrier_vote_33;
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
                                    int mini_rows = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                                    int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                                    int _poll_0_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_0_first) : "l"(y_ready + mini) : "memory");
                                    int _poll_0 = _poll_0_first;
                                    while (_poll_0 < required) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int _poll_0_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_0_next) : "l"(y_ready + mini) : "memory");
                                        _poll_0 = _poll_0_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                #pragma unroll
                                for (int stage_1 = 0; stage_1 < 7; stage_1++) {
                                    if (valid_tiles > stage_1) {
                                        int _min_34 = ((1024) < (hidden - columns[stage_1] * 1024) ? (1024) : (hidden - columns[stage_1] * 1024));
                                        unsigned int chunk_bytes = (unsigned int)(_min_34 * 2);
                                        mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_1) * 8, counts_1[stage_1] * chunk_bytes);
                                    }
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 7; stage_2++) {
                                if (peers[stage_2] >= 0) {
                                    int chunk_cols_1 = ((1024) < (hidden - columns[stage_2] * 1024) ? (1024) : (hidden - columns[stage_2] * 1024));
                                    cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_2 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows[stage_2] * (unsigned long long)hidden + (unsigned long long)(columns[stage_2] * 1024)) * (unsigned long long)2)), chunk_cols_1 * 2, combine_arrived_addr + (stage_2) * 8);
                                }
                            }
                            #pragma unroll
                            for (int stage_3 = 0; stage_3 < 7; stage_3++) {
                                if (valid_tiles > stage_3) {
                                    mbarrier_wait(combine_arrived_addr + (stage_3) * 8, phase_bits_1 >> (unsigned int)stage_3 & 1);
                                    phase_bits_1 = phase_bits_1 ^ (unsigned int)(1 << stage_3);
                                    if (peers[stage_3] >= 0) {
                                        int _min_35 = ((1024) < (hidden - columns[stage_3] * 1024) ? (1024) : (hidden - columns[stage_3] * 1024));
                                        unsigned int chunk_bytes_1 = (unsigned int)(_min_35 * 2);
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        {
                                            void* _cpbulk_dst_36 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[stage_3]]) + ((unsigned long long)tokens_0[stage_3] * (unsigned long long)hidden + (unsigned long long)(columns[stage_3] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_36), "r"(combine_smem_addr + (unsigned int)((stage_3 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_1))
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
                            unsigned int _atomic_fetch_add_37 = atomicAdd(&combine_next[0], 1);
                            combine_slot[0] = (int)_atomic_fetch_add_37;
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
                        int macro_tokens_3_1 = ((macro_size) < (tokens - macro_offset_2_1) ? (macro_size) : (tokens - macro_offset_2_1));
                        int valid_tiles_1 = ((7) < (macro_tokens_3_1 / 16 * col_blocks_1_1 - first_tile_1) ? (7) : (macro_tokens_3_1 / 16 * col_blocks_1_1 - first_tile_1));
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
                                        uint32_t _barrier_vote_38 = __syncthreads_count(peers_1[stage_4] >= 0);
                                        counts_1_1[stage_4] = _barrier_vote_38;
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
                                    int mini_rows_1 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    int _poll_1_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_1_first) : "l"(y_ready + mini_1) : "memory");
                                    int _poll_1 = _poll_1_first;
                                    while (_poll_1 < required_1) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int _poll_1_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_1_next) : "l"(y_ready + mini_1) : "memory");
                                        _poll_1 = _poll_1_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                #pragma unroll
                                for (int stage_5 = 0; stage_5 < 7; stage_5++) {
                                    if (valid_tiles_1 > stage_5) {
                                        int _min_39 = ((1024) < (hidden - columns_1[stage_5] * 1024) ? (1024) : (hidden - columns_1[stage_5] * 1024));
                                        unsigned int chunk_bytes_2 = (unsigned int)(_min_39 * 2);
                                        mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_5) * 8, counts_1_1[stage_5] * chunk_bytes_2);
                                    }
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int stage_6 = 0; stage_6 < 7; stage_6++) {
                                if (peers_1[stage_6] >= 0) {
                                    int chunk_cols_2 = ((1024) < (hidden - columns_1[stage_6] * 1024) ? (1024) : (hidden - columns_1[stage_6] * 1024));
                                    cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_6 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows_1[stage_6] * (unsigned long long)hidden + (unsigned long long)(columns_1[stage_6] * 1024)) * (unsigned long long)2)), chunk_cols_2 * 2, combine_arrived_addr + (stage_6) * 8);
                                }
                            }
                            #pragma unroll
                            for (int stage_7 = 0; stage_7 < 7; stage_7++) {
                                if (valid_tiles_1 > stage_7) {
                                    mbarrier_wait(combine_arrived_addr + (stage_7) * 8, phase_bits_0_1 >> (unsigned int)stage_7 & 1);
                                    phase_bits_0_1 = phase_bits_0_1 ^ (unsigned int)(1 << stage_7);
                                    if (peers_1[stage_7] >= 0) {
                                        int _min_40 = ((1024) < (hidden - columns_1[stage_7] * 1024) ? (1024) : (hidden - columns_1[stage_7] * 1024));
                                        unsigned int chunk_bytes_3 = (unsigned int)(_min_40 * 2);
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        {
                                            void* _cpbulk_dst_41 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[stage_7]]) + ((unsigned long long)tokens_0_1[stage_7] * (unsigned long long)hidden + (unsigned long long)(columns_1[stage_7] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_41), "r"(combine_smem_addr + (unsigned int)((stage_7 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_3))
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
                        int macro_tokens_3_2 = ((macro_size) < (tokens - macro_offset_2_2) ? (macro_size) : (tokens - macro_offset_2_2));
                        if (task_1 < macro_tokens_3_2 / 128 * col_blocks_1_2) {
                            int row_3 = task_1 / col_blocks_1_2 * 128;
                            int col_block_1 = task_1 % col_blocks_1_2;
                            int chunk_cols_3 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                            unsigned int chunk_bytes_4 = (unsigned int)(chunk_cols_3 * 2);
                            int peer_3 = -1;
                            int peer_token_3 = -1;
                            if (tid < 128) {
                                peer_3 = schedule_rank[macro_offset_2_2 + row_3 + tid];
                                peer_token_3 = schedule_token[macro_offset_2_2 + row_3 + tid];
                            }
                            uint32_t _cta_count_3 = __syncthreads_count(peer_3 >= 0);
                            if (tid == 0) {
                                int previous_offset = macro * macro_size;
                                int previous_tokens = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                if (row_3 < previous_tokens) {
                                    int previous_mini = (previous_offset + row_3) / mini_size;
                                    int mini_rows_2 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                    int required_2 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                                    bool enabled_value_1 = 1;
                                    if (enabled_value_1 != 0) {
                                        int _poll_2_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_2_first) : "l"(y_ready + previous_mini) : "memory");
                                        int _poll_2 = _poll_2_first;
                                        while (_poll_2 < required_2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int _poll_2_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_2_next) : "l"(y_ready + previous_mini) : "memory");
                                            _poll_2 = _poll_2_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_3 * chunk_bytes_4);
                            }
                            __syncthreads();
                            if (peer_3 >= 0) {
                                cp_async_bulk_gmem2smem(smem_v14_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_3])) + ((unsigned long long)((unsigned long long)(peer_token_3 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2)), chunk_cols_3 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_3 = 0; vec_3 < 64; vec_3++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (tid * 1024 + vec_3 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                            unsigned int _parity_dispatch_arrived_1 = phase_bits_0_2 % 2;
                            mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived_1);
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
                                    int r_4 = tid % 64;
                                    int row0_4 = r_4 * 2;
                                    int row1_4 = row0_4 + 1;
                                    int half_4 = tid / 64;
                                    int word_stride_4 = 256;
                                    int scale_base0_4 = (row0_4 % 32 * 4 + row0_4 / 32) * 4;
                                    int scale_base1_4 = (row1_4 % 32 * 4 + row1_4 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_8 = 0; j_8 < 2; j_8++) {
                                        int k_block_8 = (half_4 * 2 + j_8 + tid / 8) % 4;
                                        unsigned int pairs0_4[16];
                                        unsigned int pairs1_4[16];
                                        #pragma unroll
                                        for (int k_16 = 0; k_16 < 16; k_16++) {
                                            int col_16 = k_block_8 * 32 + (tid * 4 + k_16 * 2) % 32;
                                            unsigned int w0_0_4 = dispatch_words[col_16 * word_stride_4 + r_4];
                                            unsigned int w1_1_4 = dispatch_words[(col_16 + 1) * word_stride_4 + r_4];
                                            uint32_t lo_4;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_4) : "r"(w0_0_4), "r"(w1_1_4));
                                            uint32_t hi_4;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_4) : "r"(w0_0_4), "r"(w1_1_4));
                                            pairs0_4[k_16] = lo_4;
                                            pairs1_4[k_16] = hi_4;
                                        }
                                        uint32_t _bf16x2_abs_24;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(pairs0_4[0]));
                                        unsigned int amax_pair_8 = _bf16x2_abs_24;
                                        #pragma unroll
                                        for (int i_24 = 1; i_24 < 16; i_24++) {
                                            uint32_t _bf16x2_abs_25;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(pairs0_4[i_24]));
                                            uint32_t _bf16x2_max_12;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(amax_pair_8), "r"(_bf16x2_abs_25));
                                            amax_pair_8 = _bf16x2_max_12;
                                        }
                                        uint16_t _bf16_max_12;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_12) : "h"((uint16_t)(amax_pair_8 & 65535)), "h"((uint16_t)(amax_pair_8 >> 16)));
                                        float amax_9;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_9) : "h"((uint16_t)(_bf16_max_12)));
                                        float scale_9 = fmaxf(amax_9 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_42;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_42) : "f"(scale_9), "f"(scale_9));
                                        unsigned int scale_byte_9 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_42 & 255;
                                        unsigned int inverse_lane_9 = 254 - scale_byte_9 << 7;
                                        unsigned int inverse_9 = inverse_lane_9 | inverse_lane_9 << 16;
                                        unsigned int words_9[8];
                                        #pragma unroll
                                        for (int i_25 = 0; i_25 < 8; i_25++) {
                                            uint32_t _bf16x2_mul_24;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(pairs0_4[i_25 * 2]), "r"(inverse_9));
                                            uint16_t _e4m3x2_bf16x2_12;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_12) : "r"(_bf16x2_mul_24));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_24);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_25;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(pairs0_4[i_25 * 2 + 1]), "r"(inverse_9));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_43;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_43) : "r"(_bf16x2_mul_25));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_25);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_43) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_9[i_25] = (unsigned int)_e4m3x2_bf16x2_12 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_43 << 16;
                                        }
                                        uint32_t _bf16x2_abs_26;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_26) : "r"(pairs1_4[0]));
                                        unsigned int amax_pair_0_4 = _bf16x2_abs_26;
                                        #pragma unroll
                                        for (int i_26 = 1; i_26 < 16; i_26++) {
                                            uint32_t _bf16x2_abs_27;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_27) : "r"(pairs1_4[i_26]));
                                            uint32_t _bf16x2_max_13;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_13) : "r"(amax_pair_0_4), "r"(_bf16x2_abs_27));
                                            amax_pair_0_4 = _bf16x2_max_13;
                                        }
                                        uint16_t _bf16_max_13;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_13) : "h"((uint16_t)(amax_pair_0_4 & 65535)), "h"((uint16_t)(amax_pair_0_4 >> 16)));
                                        float amax_1_4;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_4) : "h"((uint16_t)(_bf16_max_13)));
                                        float scale_2_4 = fmaxf(amax_1_4 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_44;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_44) : "f"(scale_2_4), "f"(scale_2_4));
                                        unsigned int scale_byte_3_4 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_44 & 255;
                                        unsigned int inverse_lane_4_4 = 254 - scale_byte_3_4 << 7;
                                        unsigned int inverse_5_4 = inverse_lane_4_4 | inverse_lane_4_4 << 16;
                                        unsigned int words_6_4[8];
                                        #pragma unroll
                                        for (int i_27 = 0; i_27 < 8; i_27++) {
                                            uint32_t _bf16x2_mul_26;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(pairs1_4[i_27 * 2]), "r"(inverse_5_4));
                                            uint16_t _e4m3x2_bf16x2_13;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_13) : "r"(_bf16x2_mul_26));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_26);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_27;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(pairs1_4[i_27 * 2 + 1]), "r"(inverse_5_4));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_45;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_45) : "r"(_bf16x2_mul_27));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_27);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_45) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_6_4[i_27] = (unsigned int)_e4m3x2_bf16x2_13 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_45 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_17 = 0; k_17 < 8; k_17++) {
                                            int col_17 = k_block_8 * 32 + (tid * 4 + k_17 * 4) % 32;
                                            smem_v22[(row0_4 * 128 + col_17) / 4] = words_9[k_17];
                                            smem_v22[(row1_4 * 128 + col_17) / 4] = words_6_4[k_17];
                                        }
                                        {
                                            uint32_t _byte_46 = static_cast<uint32_t>(scale_byte_9) & 0xFFu;
                                            uint32_t _addr_46 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0_4 + k_block_8));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_46), "r"(_byte_46) : "memory");
                                        }
                                        {
                                            uint32_t _byte_47 = static_cast<uint32_t>(scale_byte_3_4) & 0xFFu;
                                            uint32_t _addr_47 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1_4 + k_block_8));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_47), "r"(_byte_47) : "memory");
                                        }
                                    }
                                } else {
                                    int row_0_4 = tid - 128;
                                    unsigned int scale_word_4 = 0;
                                    #pragma unroll 1
                                    for (int j_9 = 0; j_9 < 4; j_9++) {
                                        int k_block_9 = (j_9 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_4[16];
                                        #pragma unroll
                                        for (int k_18 = 0; k_18 < 16; k_18++) {
                                            int col_18 = k_block_9 * 32 + ((tid - 128) * 4 + k_18 * 2) % 32;
                                            float x0_4 = 0.0f;
                                            float x1_4 = 0.0f;
                                            pairs_4[k_18] = dispatch_words[(row_0_4 * 512 + col_18) / 2];
                                        }
                                        uint32_t _bf16x2_abs_28;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_28) : "r"(pairs_4[0]));
                                        unsigned int amax_pair_9 = _bf16x2_abs_28;
                                        #pragma unroll
                                        for (int i_28 = 1; i_28 < 16; i_28++) {
                                            uint32_t _bf16x2_abs_29;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_29) : "r"(pairs_4[i_28]));
                                            uint32_t _bf16x2_max_14;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_14) : "r"(amax_pair_9), "r"(_bf16x2_abs_29));
                                            amax_pair_9 = _bf16x2_max_14;
                                        }
                                        uint16_t _bf16_max_14;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_14) : "h"((uint16_t)(amax_pair_9 & 65535)), "h"((uint16_t)(amax_pair_9 >> 16)));
                                        float amax_10;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10) : "h"((uint16_t)(_bf16_max_14)));
                                        float scale_10 = fmaxf(amax_10 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_48;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_48) : "f"(scale_10), "f"(scale_10));
                                        unsigned int scale_byte_10 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_48 & 255;
                                        unsigned int inverse_lane_10 = 254 - scale_byte_10 << 7;
                                        unsigned int inverse_10 = inverse_lane_10 | inverse_lane_10 << 16;
                                        unsigned int words_10[8];
                                        #pragma unroll
                                        for (int i_29 = 0; i_29 < 8; i_29++) {
                                            uint32_t _bf16x2_mul_28;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(pairs_4[i_29 * 2]), "r"(inverse_10));
                                            uint16_t _e4m3x2_bf16x2_14;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_14) : "r"(_bf16x2_mul_28));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_28);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_29;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(pairs_4[i_29 * 2 + 1]), "r"(inverse_10));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_49;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_49) : "r"(_bf16x2_mul_29));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_29);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_49) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_10[i_29] = (unsigned int)_e4m3x2_bf16x2_14 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_49 << 16;
                                        }
                                        scale_word_4 = scale_word_4 | scale_byte_10 << (unsigned int)(k_block_9 * 8);
                                        #pragma unroll
                                        for (int k_19 = 0; k_19 < 8; k_19++) {
                                            int col_19 = k_block_9 * 32 + ((tid - 128) * 4 + k_19 * 4) % 32;
                                            smem_v18[(row_0_4 * 128 + col_19) / 4] = words_10[k_19];
                                        }
                                    }
                                    smem_v20[row_0_4 % 32 * 4 + row_0_4 / 32] = scale_word_4;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_4 = col_block_1 * 4;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_4 * 128, row_3, smem_v18_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_4, smem_v20_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_4 * 128, smem_v22_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_4 * macro_tiles_1 + row_tile_2, smem_v24_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (chunk_cols_3 / 128 > 1) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int r_5 = tid % 64;
                                    int row0_5 = r_5 * 2;
                                    int row1_5 = row0_5 + 1;
                                    int half_5 = tid / 64;
                                    int word_stride_5 = 256;
                                    int scale_base0_5 = (row0_5 % 32 * 4 + row0_5 / 32) * 4;
                                    int scale_base1_5 = (row1_5 % 32 * 4 + row1_5 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_10 = 0; j_10 < 2; j_10++) {
                                        int k_block_10 = (half_5 * 2 + j_10 + tid / 8) % 4;
                                        unsigned int pairs0_5[16];
                                        unsigned int pairs1_5[16];
                                        #pragma unroll
                                        for (int k_20 = 0; k_20 < 16; k_20++) {
                                            int col_20 = k_block_10 * 32 + (tid * 4 + k_20 * 2) % 32;
                                            unsigned int w0_0_5 = dispatch_words[64 + col_20 * word_stride_5 + r_5];
                                            unsigned int w1_1_5 = dispatch_words[64 + (col_20 + 1) * word_stride_5 + r_5];
                                            uint32_t lo_5;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_5) : "r"(w0_0_5), "r"(w1_1_5));
                                            uint32_t hi_5;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_5) : "r"(w0_0_5), "r"(w1_1_5));
                                            pairs0_5[k_20] = lo_5;
                                            pairs1_5[k_20] = hi_5;
                                        }
                                        uint32_t _bf16x2_abs_30;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_30) : "r"(pairs0_5[0]));
                                        unsigned int amax_pair_10 = _bf16x2_abs_30;
                                        #pragma unroll
                                        for (int i_30 = 1; i_30 < 16; i_30++) {
                                            uint32_t _bf16x2_abs_31;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_31) : "r"(pairs0_5[i_30]));
                                            uint32_t _bf16x2_max_15;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_15) : "r"(amax_pair_10), "r"(_bf16x2_abs_31));
                                            amax_pair_10 = _bf16x2_max_15;
                                        }
                                        uint16_t _bf16_max_15;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_15) : "h"((uint16_t)(amax_pair_10 & 65535)), "h"((uint16_t)(amax_pair_10 >> 16)));
                                        float amax_11;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11) : "h"((uint16_t)(_bf16_max_15)));
                                        float scale_11 = fmaxf(amax_11 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_50;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_50) : "f"(scale_11), "f"(scale_11));
                                        unsigned int scale_byte_11 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_50 & 255;
                                        unsigned int inverse_lane_11 = 254 - scale_byte_11 << 7;
                                        unsigned int inverse_11 = inverse_lane_11 | inverse_lane_11 << 16;
                                        unsigned int words_11[8];
                                        #pragma unroll
                                        for (int i_31 = 0; i_31 < 8; i_31++) {
                                            uint32_t _bf16x2_mul_30;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(pairs0_5[i_31 * 2]), "r"(inverse_11));
                                            uint16_t _e4m3x2_bf16x2_15;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_15) : "r"(_bf16x2_mul_30));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_30);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_31;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(pairs0_5[i_31 * 2 + 1]), "r"(inverse_11));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_51;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_51) : "r"(_bf16x2_mul_31));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_31);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_51) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_11[i_31] = (unsigned int)_e4m3x2_bf16x2_15 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_51 << 16;
                                        }
                                        uint32_t _bf16x2_abs_32;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(pairs1_5[0]));
                                        unsigned int amax_pair_0_5 = _bf16x2_abs_32;
                                        #pragma unroll
                                        for (int i_32 = 1; i_32 < 16; i_32++) {
                                            uint32_t _bf16x2_abs_33;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(pairs1_5[i_32]));
                                            uint32_t _bf16x2_max_16;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(amax_pair_0_5), "r"(_bf16x2_abs_33));
                                            amax_pair_0_5 = _bf16x2_max_16;
                                        }
                                        uint16_t _bf16_max_16;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_16) : "h"((uint16_t)(amax_pair_0_5 & 65535)), "h"((uint16_t)(amax_pair_0_5 >> 16)));
                                        float amax_1_5;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_5) : "h"((uint16_t)(_bf16_max_16)));
                                        float scale_2_5 = fmaxf(amax_1_5 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_52;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_52) : "f"(scale_2_5), "f"(scale_2_5));
                                        unsigned int scale_byte_3_5 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_52 & 255;
                                        unsigned int inverse_lane_4_5 = 254 - scale_byte_3_5 << 7;
                                        unsigned int inverse_5_5 = inverse_lane_4_5 | inverse_lane_4_5 << 16;
                                        unsigned int words_6_5[8];
                                        #pragma unroll
                                        for (int i_33 = 0; i_33 < 8; i_33++) {
                                            uint32_t _bf16x2_mul_32;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(pairs1_5[i_33 * 2]), "r"(inverse_5_5));
                                            uint16_t _e4m3x2_bf16x2_16;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_16) : "r"(_bf16x2_mul_32));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_32);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_33;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(pairs1_5[i_33 * 2 + 1]), "r"(inverse_5_5));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_53;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_53) : "r"(_bf16x2_mul_33));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_33);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_53) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_6_5[i_33] = (unsigned int)_e4m3x2_bf16x2_16 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_53 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_21 = 0; k_21 < 8; k_21++) {
                                            int col_21 = k_block_10 * 32 + (tid * 4 + k_21 * 4) % 32;
                                            smem_v23[(row0_5 * 128 + col_21) / 4] = words_11[k_21];
                                            smem_v23[(row1_5 * 128 + col_21) / 4] = words_6_5[k_21];
                                        }
                                        {
                                            uint32_t _byte_54 = static_cast<uint32_t>(scale_byte_11) & 0xFFu;
                                            uint32_t _addr_54 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_5 + k_block_10));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_54), "r"(_byte_54) : "memory");
                                        }
                                        {
                                            uint32_t _byte_55 = static_cast<uint32_t>(scale_byte_3_5) & 0xFFu;
                                            uint32_t _addr_55 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_5 + k_block_10));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_55), "r"(_byte_55) : "memory");
                                        }
                                    }
                                } else {
                                    int row_0_5 = tid - 128;
                                    unsigned int scale_word_5 = 0;
                                    #pragma unroll 1
                                    for (int j_11 = 0; j_11 < 4; j_11++) {
                                        int k_block_11 = (j_11 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_5[16];
                                        #pragma unroll
                                        for (int k_22 = 0; k_22 < 16; k_22++) {
                                            int col_22 = k_block_11 * 32 + ((tid - 128) * 4 + k_22 * 2) % 32;
                                            float x0_5 = 0.0f;
                                            float x1_5 = 0.0f;
                                            pairs_5[k_22] = dispatch_words[64 + (row_0_5 * 512 + col_22) / 2];
                                        }
                                        uint32_t _bf16x2_abs_34;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(pairs_5[0]));
                                        unsigned int amax_pair_11 = _bf16x2_abs_34;
                                        #pragma unroll
                                        for (int i_34 = 1; i_34 < 16; i_34++) {
                                            uint32_t _bf16x2_abs_35;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(pairs_5[i_34]));
                                            uint32_t _bf16x2_max_17;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(amax_pair_11), "r"(_bf16x2_abs_35));
                                            amax_pair_11 = _bf16x2_max_17;
                                        }
                                        uint16_t _bf16_max_17;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_17) : "h"((uint16_t)(amax_pair_11 & 65535)), "h"((uint16_t)(amax_pair_11 >> 16)));
                                        float amax_12;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_12) : "h"((uint16_t)(_bf16_max_17)));
                                        float scale_12 = fmaxf(amax_12 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_56;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_56) : "f"(scale_12), "f"(scale_12));
                                        unsigned int scale_byte_12 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_56 & 255;
                                        unsigned int inverse_lane_12 = 254 - scale_byte_12 << 7;
                                        unsigned int inverse_12 = inverse_lane_12 | inverse_lane_12 << 16;
                                        unsigned int words_12[8];
                                        #pragma unroll
                                        for (int i_35 = 0; i_35 < 8; i_35++) {
                                            uint32_t _bf16x2_mul_34;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(pairs_5[i_35 * 2]), "r"(inverse_12));
                                            uint16_t _e4m3x2_bf16x2_17;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_17) : "r"(_bf16x2_mul_34));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_34);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_35;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(pairs_5[i_35 * 2 + 1]), "r"(inverse_12));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_57;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_57) : "r"(_bf16x2_mul_35));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_35);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_57) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_12[i_35] = (unsigned int)_e4m3x2_bf16x2_17 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_57 << 16;
                                        }
                                        scale_word_5 = scale_word_5 | scale_byte_12 << (unsigned int)(k_block_11 * 8);
                                        #pragma unroll
                                        for (int k_23 = 0; k_23 < 8; k_23++) {
                                            int col_23 = k_block_11 * 32 + ((tid - 128) * 4 + k_23 * 4) % 32;
                                            smem_v19[(row_0_5 * 128 + col_23) / 4] = words_12[k_23];
                                        }
                                    }
                                    smem_v21[row_0_5 % 32 * 4 + row_0_5 / 32] = scale_word_5;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_5 = col_block_1 * 4 + 1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_5 * 128, row_3, smem_v19_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_5, smem_v21_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_5 * 128, smem_v23_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_5 * macro_tiles_1 + row_tile_2, smem_v25_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (chunk_cols_3 / 128 > 2) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int r_6 = tid % 64;
                                    int row0_6 = r_6 * 2;
                                    int row1_6 = row0_6 + 1;
                                    int half_6 = tid / 64;
                                    int word_stride_6 = 256;
                                    int scale_base0_6 = (row0_6 % 32 * 4 + row0_6 / 32) * 4;
                                    int scale_base1_6 = (row1_6 % 32 * 4 + row1_6 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_12 = 0; j_12 < 2; j_12++) {
                                        int k_block_12 = (half_6 * 2 + j_12 + tid / 8) % 4;
                                        unsigned int pairs0_6[16];
                                        unsigned int pairs1_6[16];
                                        #pragma unroll
                                        for (int k_24 = 0; k_24 < 16; k_24++) {
                                            int col_24 = k_block_12 * 32 + (tid * 4 + k_24 * 2) % 32;
                                            unsigned int w0_0_6 = dispatch_words[128 + col_24 * word_stride_6 + r_6];
                                            unsigned int w1_1_6 = dispatch_words[128 + (col_24 + 1) * word_stride_6 + r_6];
                                            uint32_t lo_6;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_6) : "r"(w0_0_6), "r"(w1_1_6));
                                            uint32_t hi_6;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_6) : "r"(w0_0_6), "r"(w1_1_6));
                                            pairs0_6[k_24] = lo_6;
                                            pairs1_6[k_24] = hi_6;
                                        }
                                        uint32_t _bf16x2_abs_36;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(pairs0_6[0]));
                                        unsigned int amax_pair_12 = _bf16x2_abs_36;
                                        #pragma unroll
                                        for (int i_36 = 1; i_36 < 16; i_36++) {
                                            uint32_t _bf16x2_abs_37;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(pairs0_6[i_36]));
                                            uint32_t _bf16x2_max_18;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(amax_pair_12), "r"(_bf16x2_abs_37));
                                            amax_pair_12 = _bf16x2_max_18;
                                        }
                                        uint16_t _bf16_max_18;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_18) : "h"((uint16_t)(amax_pair_12 & 65535)), "h"((uint16_t)(amax_pair_12 >> 16)));
                                        float amax_13;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_13) : "h"((uint16_t)(_bf16_max_18)));
                                        float scale_13 = fmaxf(amax_13 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_58;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_58) : "f"(scale_13), "f"(scale_13));
                                        unsigned int scale_byte_13 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_58 & 255;
                                        unsigned int inverse_lane_13 = 254 - scale_byte_13 << 7;
                                        unsigned int inverse_13 = inverse_lane_13 | inverse_lane_13 << 16;
                                        unsigned int words_13[8];
                                        #pragma unroll
                                        for (int i_37 = 0; i_37 < 8; i_37++) {
                                            uint32_t _bf16x2_mul_36;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(pairs0_6[i_37 * 2]), "r"(inverse_13));
                                            uint16_t _e4m3x2_bf16x2_18;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_18) : "r"(_bf16x2_mul_36));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_36);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_18) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_37;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(pairs0_6[i_37 * 2 + 1]), "r"(inverse_13));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_59;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_59) : "r"(_bf16x2_mul_37));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_37);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_59) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_13[i_37] = (unsigned int)_e4m3x2_bf16x2_18 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_59 << 16;
                                        }
                                        uint32_t _bf16x2_abs_38;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(pairs1_6[0]));
                                        unsigned int amax_pair_0_6 = _bf16x2_abs_38;
                                        #pragma unroll
                                        for (int i_38 = 1; i_38 < 16; i_38++) {
                                            uint32_t _bf16x2_abs_39;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(pairs1_6[i_38]));
                                            uint32_t _bf16x2_max_19;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(amax_pair_0_6), "r"(_bf16x2_abs_39));
                                            amax_pair_0_6 = _bf16x2_max_19;
                                        }
                                        uint16_t _bf16_max_19;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_19) : "h"((uint16_t)(amax_pair_0_6 & 65535)), "h"((uint16_t)(amax_pair_0_6 >> 16)));
                                        float amax_1_6;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_6) : "h"((uint16_t)(_bf16_max_19)));
                                        float scale_2_6 = fmaxf(amax_1_6 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_60;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_60) : "f"(scale_2_6), "f"(scale_2_6));
                                        unsigned int scale_byte_3_6 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_60 & 255;
                                        unsigned int inverse_lane_4_6 = 254 - scale_byte_3_6 << 7;
                                        unsigned int inverse_5_6 = inverse_lane_4_6 | inverse_lane_4_6 << 16;
                                        unsigned int words_6_6[8];
                                        #pragma unroll
                                        for (int i_39 = 0; i_39 < 8; i_39++) {
                                            uint32_t _bf16x2_mul_38;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(pairs1_6[i_39 * 2]), "r"(inverse_5_6));
                                            uint16_t _e4m3x2_bf16x2_19;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_19) : "r"(_bf16x2_mul_38));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_38);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_39;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(pairs1_6[i_39 * 2 + 1]), "r"(inverse_5_6));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_61;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_61) : "r"(_bf16x2_mul_39));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_39);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_61) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_6_6[i_39] = (unsigned int)_e4m3x2_bf16x2_19 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_61 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_25 = 0; k_25 < 8; k_25++) {
                                            int col_25 = k_block_12 * 32 + (tid * 4 + k_25 * 4) % 32;
                                            smem_v22[(row0_6 * 128 + col_25) / 4] = words_13[k_25];
                                            smem_v22[(row1_6 * 128 + col_25) / 4] = words_6_6[k_25];
                                        }
                                        {
                                            uint32_t _byte_62 = static_cast<uint32_t>(scale_byte_13) & 0xFFu;
                                            uint32_t _addr_62 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0_6 + k_block_12));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_62), "r"(_byte_62) : "memory");
                                        }
                                        {
                                            uint32_t _byte_63 = static_cast<uint32_t>(scale_byte_3_6) & 0xFFu;
                                            uint32_t _addr_63 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1_6 + k_block_12));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_63), "r"(_byte_63) : "memory");
                                        }
                                    }
                                } else {
                                    int row_0_6 = tid - 128;
                                    unsigned int scale_word_6 = 0;
                                    #pragma unroll 1
                                    for (int j_13 = 0; j_13 < 4; j_13++) {
                                        int k_block_13 = (j_13 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_6[16];
                                        #pragma unroll
                                        for (int k_26 = 0; k_26 < 16; k_26++) {
                                            int col_26 = k_block_13 * 32 + ((tid - 128) * 4 + k_26 * 2) % 32;
                                            float x0_6 = 0.0f;
                                            float x1_6 = 0.0f;
                                            pairs_6[k_26] = dispatch_words[128 + (row_0_6 * 512 + col_26) / 2];
                                        }
                                        uint32_t _bf16x2_abs_40;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(pairs_6[0]));
                                        unsigned int amax_pair_13 = _bf16x2_abs_40;
                                        #pragma unroll
                                        for (int i_40 = 1; i_40 < 16; i_40++) {
                                            uint32_t _bf16x2_abs_41;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(pairs_6[i_40]));
                                            uint32_t _bf16x2_max_20;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(amax_pair_13), "r"(_bf16x2_abs_41));
                                            amax_pair_13 = _bf16x2_max_20;
                                        }
                                        uint16_t _bf16_max_20;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_20) : "h"((uint16_t)(amax_pair_13 & 65535)), "h"((uint16_t)(amax_pair_13 >> 16)));
                                        float amax_14;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_14) : "h"((uint16_t)(_bf16_max_20)));
                                        float scale_14 = fmaxf(amax_14 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_64;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_64) : "f"(scale_14), "f"(scale_14));
                                        unsigned int scale_byte_14 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_64 & 255;
                                        unsigned int inverse_lane_14 = 254 - scale_byte_14 << 7;
                                        unsigned int inverse_14 = inverse_lane_14 | inverse_lane_14 << 16;
                                        unsigned int words_14[8];
                                        #pragma unroll
                                        for (int i_41 = 0; i_41 < 8; i_41++) {
                                            uint32_t _bf16x2_mul_40;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(pairs_6[i_41 * 2]), "r"(inverse_14));
                                            uint16_t _e4m3x2_bf16x2_20;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_20) : "r"(_bf16x2_mul_40));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_40);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_41;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(pairs_6[i_41 * 2 + 1]), "r"(inverse_14));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_65;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_65) : "r"(_bf16x2_mul_41));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_41);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_65) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_14[i_41] = (unsigned int)_e4m3x2_bf16x2_20 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_65 << 16;
                                        }
                                        scale_word_6 = scale_word_6 | scale_byte_14 << (unsigned int)(k_block_13 * 8);
                                        #pragma unroll
                                        for (int k_27 = 0; k_27 < 8; k_27++) {
                                            int col_27 = k_block_13 * 32 + ((tid - 128) * 4 + k_27 * 4) % 32;
                                            smem_v18[(row_0_6 * 128 + col_27) / 4] = words_14[k_27];
                                        }
                                    }
                                    smem_v20[row_0_6 % 32 * 4 + row_0_6 / 32] = scale_word_6;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_6 = col_block_1 * 4 + 2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_6 * 128, row_3, smem_v18_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_6, smem_v20_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_6 * 128, smem_v22_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_6 * macro_tiles_1 + row_tile_2, smem_v24_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (chunk_cols_3 / 128 > 3) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int r_7 = tid % 64;
                                    int row0_7 = r_7 * 2;
                                    int row1_7 = row0_7 + 1;
                                    int half_7 = tid / 64;
                                    int word_stride_7 = 256;
                                    int scale_base0_7 = (row0_7 % 32 * 4 + row0_7 / 32) * 4;
                                    int scale_base1_7 = (row1_7 % 32 * 4 + row1_7 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_14 = 0; j_14 < 2; j_14++) {
                                        int k_block_14 = (half_7 * 2 + j_14 + tid / 8) % 4;
                                        unsigned int pairs0_7[16];
                                        unsigned int pairs1_7[16];
                                        #pragma unroll
                                        for (int k_28 = 0; k_28 < 16; k_28++) {
                                            int col_28 = k_block_14 * 32 + (tid * 4 + k_28 * 2) % 32;
                                            unsigned int w0_0_7 = dispatch_words[192 + col_28 * word_stride_7 + r_7];
                                            unsigned int w1_1_7 = dispatch_words[192 + (col_28 + 1) * word_stride_7 + r_7];
                                            uint32_t lo_7;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_7) : "r"(w0_0_7), "r"(w1_1_7));
                                            uint32_t hi_7;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_7) : "r"(w0_0_7), "r"(w1_1_7));
                                            pairs0_7[k_28] = lo_7;
                                            pairs1_7[k_28] = hi_7;
                                        }
                                        uint32_t _bf16x2_abs_42;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(pairs0_7[0]));
                                        unsigned int amax_pair_14 = _bf16x2_abs_42;
                                        #pragma unroll
                                        for (int i_42 = 1; i_42 < 16; i_42++) {
                                            uint32_t _bf16x2_abs_43;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(pairs0_7[i_42]));
                                            uint32_t _bf16x2_max_21;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(amax_pair_14), "r"(_bf16x2_abs_43));
                                            amax_pair_14 = _bf16x2_max_21;
                                        }
                                        uint16_t _bf16_max_21;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_21) : "h"((uint16_t)(amax_pair_14 & 65535)), "h"((uint16_t)(amax_pair_14 >> 16)));
                                        float amax_15;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_15) : "h"((uint16_t)(_bf16_max_21)));
                                        float scale_15 = fmaxf(amax_15 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_66;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_66) : "f"(scale_15), "f"(scale_15));
                                        unsigned int scale_byte_15 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_66 & 255;
                                        unsigned int inverse_lane_15 = 254 - scale_byte_15 << 7;
                                        unsigned int inverse_15 = inverse_lane_15 | inverse_lane_15 << 16;
                                        unsigned int words_15[8];
                                        #pragma unroll
                                        for (int i_43 = 0; i_43 < 8; i_43++) {
                                            uint32_t _bf16x2_mul_42;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(pairs0_7[i_43 * 2]), "r"(inverse_15));
                                            uint16_t _e4m3x2_bf16x2_21;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_21) : "r"(_bf16x2_mul_42));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_42);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_43;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(pairs0_7[i_43 * 2 + 1]), "r"(inverse_15));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_67;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_67) : "r"(_bf16x2_mul_43));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_43);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_67) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_15[i_43] = (unsigned int)_e4m3x2_bf16x2_21 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_67 << 16;
                                        }
                                        uint32_t _bf16x2_abs_44;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(pairs1_7[0]));
                                        unsigned int amax_pair_0_7 = _bf16x2_abs_44;
                                        #pragma unroll
                                        for (int i_44 = 1; i_44 < 16; i_44++) {
                                            uint32_t _bf16x2_abs_45;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(pairs1_7[i_44]));
                                            uint32_t _bf16x2_max_22;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(amax_pair_0_7), "r"(_bf16x2_abs_45));
                                            amax_pair_0_7 = _bf16x2_max_22;
                                        }
                                        uint16_t _bf16_max_22;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_22) : "h"((uint16_t)(amax_pair_0_7 & 65535)), "h"((uint16_t)(amax_pair_0_7 >> 16)));
                                        float amax_1_7;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_7) : "h"((uint16_t)(_bf16_max_22)));
                                        float scale_2_7 = fmaxf(amax_1_7 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_68;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_68) : "f"(scale_2_7), "f"(scale_2_7));
                                        unsigned int scale_byte_3_7 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_68 & 255;
                                        unsigned int inverse_lane_4_7 = 254 - scale_byte_3_7 << 7;
                                        unsigned int inverse_5_7 = inverse_lane_4_7 | inverse_lane_4_7 << 16;
                                        unsigned int words_6_7[8];
                                        #pragma unroll
                                        for (int i_45 = 0; i_45 < 8; i_45++) {
                                            uint32_t _bf16x2_mul_44;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(pairs1_7[i_45 * 2]), "r"(inverse_5_7));
                                            uint16_t _e4m3x2_bf16x2_22;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_22) : "r"(_bf16x2_mul_44));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_44);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_45;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(pairs1_7[i_45 * 2 + 1]), "r"(inverse_5_7));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_69;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_69) : "r"(_bf16x2_mul_45));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_45);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_69) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_6_7[i_45] = (unsigned int)_e4m3x2_bf16x2_22 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_69 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_29 = 0; k_29 < 8; k_29++) {
                                            int col_29 = k_block_14 * 32 + (tid * 4 + k_29 * 4) % 32;
                                            smem_v23[(row0_7 * 128 + col_29) / 4] = words_15[k_29];
                                            smem_v23[(row1_7 * 128 + col_29) / 4] = words_6_7[k_29];
                                        }
                                        {
                                            uint32_t _byte_70 = static_cast<uint32_t>(scale_byte_15) & 0xFFu;
                                            uint32_t _addr_70 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_7 + k_block_14));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_70), "r"(_byte_70) : "memory");
                                        }
                                        {
                                            uint32_t _byte_71 = static_cast<uint32_t>(scale_byte_3_7) & 0xFFu;
                                            uint32_t _addr_71 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_7 + k_block_14));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_71), "r"(_byte_71) : "memory");
                                        }
                                    }
                                } else {
                                    int row_0_7 = tid - 128;
                                    unsigned int scale_word_7 = 0;
                                    #pragma unroll 1
                                    for (int j_15 = 0; j_15 < 4; j_15++) {
                                        int k_block_15 = (j_15 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_7[16];
                                        #pragma unroll
                                        for (int k_30 = 0; k_30 < 16; k_30++) {
                                            int col_30 = k_block_15 * 32 + ((tid - 128) * 4 + k_30 * 2) % 32;
                                            float x0_7 = 0.0f;
                                            float x1_7 = 0.0f;
                                            pairs_7[k_30] = dispatch_words[192 + (row_0_7 * 512 + col_30) / 2];
                                        }
                                        uint32_t _bf16x2_abs_46;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(pairs_7[0]));
                                        unsigned int amax_pair_15 = _bf16x2_abs_46;
                                        #pragma unroll
                                        for (int i_46 = 1; i_46 < 16; i_46++) {
                                            uint32_t _bf16x2_abs_47;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(pairs_7[i_46]));
                                            uint32_t _bf16x2_max_23;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(amax_pair_15), "r"(_bf16x2_abs_47));
                                            amax_pair_15 = _bf16x2_max_23;
                                        }
                                        uint16_t _bf16_max_23;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_23) : "h"((uint16_t)(amax_pair_15 & 65535)), "h"((uint16_t)(amax_pair_15 >> 16)));
                                        float amax_16;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_16) : "h"((uint16_t)(_bf16_max_23)));
                                        float scale_16 = fmaxf(amax_16 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_72;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_72) : "f"(scale_16), "f"(scale_16));
                                        unsigned int scale_byte_16 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_72 & 255;
                                        unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                        unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                        unsigned int words_16[8];
                                        #pragma unroll
                                        for (int i_47 = 0; i_47 < 8; i_47++) {
                                            uint32_t _bf16x2_mul_46;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(pairs_7[i_47 * 2]), "r"(inverse_16));
                                            uint16_t _e4m3x2_bf16x2_23;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_23) : "r"(_bf16x2_mul_46));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_46);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_47;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(pairs_7[i_47 * 2 + 1]), "r"(inverse_16));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_73;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_73) : "r"(_bf16x2_mul_47));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_47);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_73) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_16[i_47] = (unsigned int)_e4m3x2_bf16x2_23 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_73 << 16;
                                        }
                                        scale_word_7 = scale_word_7 | scale_byte_16 << (unsigned int)(k_block_15 * 8);
                                        #pragma unroll
                                        for (int k_31 = 0; k_31 < 8; k_31++) {
                                            int col_31 = k_block_15 * 32 + ((tid - 128) * 4 + k_31 * 4) % 32;
                                            smem_v19[(row_0_7 * 128 + col_31) / 4] = words_16[k_31];
                                        }
                                    }
                                    smem_v21[row_0_7 % 32 * 4 + row_0_7 / 32] = scale_word_7;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_7 = col_block_1 * 4 + 3;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_7 * 128, row_3, smem_v19_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_2 * k_tiles_1 + col_tile_7, smem_v21_addr);
                                    tma_store_2d((&x_t_store), row_3, col_tile_7 * 128, smem_v23_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_7 * macro_tiles_1 + row_tile_2, smem_v25_addr);
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
        int task_2 = 0;
        int macro_1 = 0;
        int mini_2 = 0;
        int swiglu = 0;
        if (shared_tasks <= cluster - comm_clusters) {
            uint32_t _fast_div_q_0 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks));
            uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks)) - _fast_div_q_0 * (uint32_t)(mini_tasks_div.divisor);
            int ordered_mini = (int)_fast_div_q_0;
            task_2 = (int)_fast_div_r_0;
            macro_1 = macros - 1;
            mini_2 = ordered_mini;
            if (ordered_mini >= last_minis) {
                uint32_t _fast_div_q_1 = (minis_per_macro_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(ordered_mini - last_minis)), minis_per_macro_div.multiplier) >> minis_per_macro_div.shift_right) : (uint32_t)((unsigned int)(ordered_mini - last_minis));
                uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)(ordered_mini - last_minis)) - _fast_div_q_1 * (uint32_t)(minis_per_macro_div.divisor);
                macro_1 = macros - 2 - (int)_fast_div_q_1;
                mini_2 = (int)_fast_div_r_1;
            }
            if (task_2 >= mini_gate && task_2 < mini_gate + mini_swiglu) {
                swiglu = 1;
            }
        }
        int task_0 = task_2;
        int macro_1_1 = macro_1;
        int mini_2_1 = mini_2;
        int current_local = swiglu;
        int td_x = 0;
        int td_y = 0;
        int td_expert = 0;
        int td_k0 = 0;
        int td_k1 = 0;
        int td_first = 0;
        int td_blocks = 0;
        int td_iters = 0;
        int td_gmini = 0;
        int td_mrows = 0;
        int td_mk = 0;
        if (cluster < true_clusters) {
            int x = 0;
            int y = 0;
            int expert = 0;
            int k_start = 0;
            int k_end = 0;
            int first = 0;
            int col_blocks_3 = 0;
            int iterations = 0;
            int global_mini = 0;
            int macro_rows_1 = 0;
            int macro_k = 0;
            if (shared_tasks <= cluster - comm_clusters) {
                if (task_2 < mini_gate) {
                    int gate_task = task_2;
                    if (task_2 >= gate_tiles) {
                        gate_task = task_2 - gate_tiles;
                    }
                    int col_blocks_0 = (intermediate + 256 - 1) / 256;
                    int x_1 = -1;
                    int y_2 = -1;
                    int expert_3 = -1;
                    int k_start_4 = 0;
                    int k_end_5 = 0;
                    int first_6 = 0;
                    int first_block = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    int block = first_block;
                    uint32_t _fast_div_q_2 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)gate_task), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)gate_task);
                    uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)gate_task) - _fast_div_q_2 * (uint32_t)(gate_col_blocks_div.divisor);
                    block = first_block + (int)_fast_div_q_2;
                    if (block < end_block) {
                        int index = counts[3 * experts + block];
                        int offset = counts[experts + index] / 256;
                        int first_row_2 = ((first_block) > (offset) ? (first_block) : (offset));
                        int _min_74 = ((end_block) < (offset + counts[index] / 256) ? (end_block) : (offset + counts[index] / 256));
                        int rows_2 = _min_74 - first_row_2;
                        int supergroup = (gate_task - (first_row_2 - first_block) * col_blocks_0) / (rows_2 * 8);
                        int full_cols = col_blocks_0 / 8 * 8;
                        int row_4 = 0;
                        int col_32 = 0;
                        if (gate_task - (first_row_2 - first_block) * col_blocks_0 < rows_2 * full_cols) {
                            row_4 = (gate_task - (first_row_2 - first_block) * col_blocks_0) % (rows_2 * 8) / 8;
                            col_32 = supergroup * 8 + (gate_task - (first_row_2 - first_block) * col_blocks_0) % 8;
                        } else {
                            row_4 = (gate_task - (first_row_2 - first_block) * col_blocks_0 - rows_2 * full_cols) / (col_blocks_0 - full_cols);
                            col_32 = full_cols + (gate_task - (first_row_2 - first_block) * col_blocks_0 - rows_2 * full_cols) % (col_blocks_0 - full_cols);
                        }
                        if ((supergroup & 1) != 0) {
                            row_4 = rows_2 - row_4 - 1;
                        }
                        x_1 = first_row_2 + row_4 - macro_1 * (macro_size / 256);
                        y_2 = col_32;
                        expert_3 = index;
                    }
                    int iterations_7 = hidden / 128;
                    int global_mini_8 = macro_1 * (macro_size / mini_size) + mini_2;
                    int macro_rows_9 = macro_1 * (macro_size / 256);
                    int macro_k_10 = macro_1 * (macro_size / 128);
                    x = x_1;
                    y = y_2;
                    expert = expert_3;
                    k_start = k_start_4;
                    k_end = k_end_5;
                    first = first_6;
                    col_blocks_3 = col_blocks_0;
                    iterations = iterations_7;
                    global_mini = global_mini_8;
                    macro_rows_1 = macro_rows_9;
                    macro_k = macro_k_10;
                } else if (task_2 >= mini_gate + mini_swiglu) {
                    int col_blocks_0_1 = (hidden + 256 - 1) / 256;
                    int x_1_1 = -1;
                    int y_2_1 = -1;
                    int expert_3_1 = -1;
                    int k_start_4_1 = 0;
                    int k_end_5_1 = 0;
                    int first_6_1 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int block_1 = first_block_1;
                    uint32_t _fast_div_q_3 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(task_2 - mini_gate - mini_swiglu)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(task_2 - mini_gate - mini_swiglu));
                    uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)(task_2 - mini_gate - mini_swiglu)) - _fast_div_q_3 * (uint32_t)(down_col_blocks_div.divisor);
                    block_1 = first_block_1 + (int)_fast_div_q_3;
                    if (block_1 < end_block_1) {
                        int index_1 = counts[3 * experts + block_1];
                        int offset_1 = counts[experts + index_1] / 256;
                        int first_row_3 = ((first_block_1) > (offset_1) ? (first_block_1) : (offset_1));
                        int _min_75 = ((end_block_1) < (offset_1 + counts[index_1] / 256) ? (end_block_1) : (offset_1 + counts[index_1] / 256));
                        int rows_3 = _min_75 - first_row_3;
                        int supergroup_1 = (task_2 - mini_gate - mini_swiglu - (first_row_3 - first_block_1) * col_blocks_0_1) / (rows_3 * 8);
                        int full_cols_1 = col_blocks_0_1 / 8 * 8;
                        int row_5 = 0;
                        int col_33 = 0;
                        if (task_2 - mini_gate - mini_swiglu - (first_row_3 - first_block_1) * col_blocks_0_1 < rows_3 * full_cols_1) {
                            row_5 = (task_2 - mini_gate - mini_swiglu - (first_row_3 - first_block_1) * col_blocks_0_1) % (rows_3 * 8) / 8;
                            col_33 = supergroup_1 * 8 + (task_2 - mini_gate - mini_swiglu - (first_row_3 - first_block_1) * col_blocks_0_1) % 8;
                        } else {
                            row_5 = (task_2 - mini_gate - mini_swiglu - (first_row_3 - first_block_1) * col_blocks_0_1 - rows_3 * full_cols_1) / (col_blocks_0_1 - full_cols_1);
                            col_33 = full_cols_1 + (task_2 - mini_gate - mini_swiglu - (first_row_3 - first_block_1) * col_blocks_0_1 - rows_3 * full_cols_1) % (col_blocks_0_1 - full_cols_1);
                        }
                        if ((supergroup_1 & 1) != 0) {
                            row_5 = rows_3 - row_5 - 1;
                        }
                        x_1_1 = first_row_3 + row_5 - macro_1 * (macro_size / 256);
                        y_2_1 = col_33;
                        expert_3_1 = index_1;
                    }
                    int iterations_7_1 = intermediate / 128;
                    int global_mini_8_1 = macro_1 * (macro_size / mini_size) + mini_2;
                    int macro_rows_9_1 = macro_1 * (macro_size / 256);
                    int macro_k_10_1 = macro_1 * (macro_size / 128);
                    x = x_1_1;
                    y = y_2_1;
                    expert = expert_3_1;
                    k_start = k_start_4_1;
                    k_end = k_end_5_1;
                    first = first_6_1;
                    col_blocks_3 = col_blocks_0_1;
                    iterations = iterations_7_1;
                    global_mini = global_mini_8_1;
                    macro_rows_1 = macro_rows_9_1;
                    macro_k = macro_k_10_1;
                }
            }
            td_x = x;
            td_y = y;
            td_expert = expert;
            td_k0 = k_start;
            td_k1 = k_end;
            td_first = first;
            td_blocks = col_blocks_3;
            td_iters = iterations;
            td_gmini = global_mini;
            td_mrows = macro_rows_1;
            td_mk = macro_k;
        }
        while (cluster >= 0 && cluster < true_clusters) {
            if (tid / 32 == 5) {
                if (warp == 5) {
                    if (elect_sync()) {
                        if (cta_rank_0 == 0) {
                            int _parity_schedule_finished = (int)((unsigned int)(iteration + 1) % 2);
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
            int compute = cluster - comm_clusters;
            if (compute < shared_fused) {
                int col_blocks_4 = (intermediate + 256 - 1) / 256;
                int x_2 = -1;
                int y_1 = -1;
                int expert_1 = -1;
                int k_start_1 = 0;
                int k_end_1 = 0;
                int first_1 = 0;
                int row_blocks = local_tokens / 256;
                if (compute < row_blocks * col_blocks_4) {
                    int supergroup_2 = compute / (row_blocks * 8);
                    int full_cols_2 = col_blocks_4 / 8 * 8;
                    int row_6 = 0;
                    int col_34 = 0;
                    if (compute < row_blocks * full_cols_2) {
                        row_6 = compute % (row_blocks * 8) / 8;
                        col_34 = supergroup_2 * 8 + compute % 8;
                    } else {
                        row_6 = (compute - row_blocks * full_cols_2) / (col_blocks_4 - full_cols_2);
                        col_34 = full_cols_2 + (compute - row_blocks * full_cols_2) % (col_blocks_4 - full_cols_2);
                    }
                    if ((supergroup_2 & 1) != 0) {
                        row_6 = row_blocks - row_6 - 1;
                    }
                    x_2 = row_6;
                    y_1 = col_34;
                    expert_1 = 0;
                }
                unsigned int phase_bits_2 = gemm_bits;
                int has_hi = 0;
                has_hi = (int)((y_1 * 2 + 1) * 256 < intermediate);
                int global_mini_1 = 0;
                int macro_rows_2 = 0;
                int iterations_1 = hidden / 64;
                if (expert_1 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_76 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                int mini_rows_3 = ((0) > (_min_76) ? (0) : (_min_76));
                                int required_3 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                            }
                            unsigned int previous = phase_bits_2 >> 7 & 1;
                            unsigned int bits_1 = phase_bits_2;
                            if (previous != 0) {
                                mbarrier_wait(gemm_finished_addr, bits_1 >> 16 & 1);
                                mbarrier_wait(gemm_finished_addr + 8, bits_1 >> 17 & 1);
                                mbarrier_wait(gemm_finished_addr + 16, bits_1 >> 18 & 1);
                                mbarrier_wait(gemm_finished_addr + 24, bits_1 >> 19 & 1);
                                mbarrier_wait(gemm_finished_addr + 32, bits_1 >> 20 & 1);
                                mbarrier_wait(gemm_finished_addr + 40, bits_1 >> 21 & 1);
                                bits_1 = bits_1 ^ 128;
                            }
                            phase_bits_2 = bits_1;
                            int ring = 0;
                            #pragma unroll 1
                            for (int idx = 0; idx < iterations_1; idx++) {
                                mbarrier_wait(gemm_finished_addr + (ring) * 8, phase_bits_2 >> (unsigned int)(16 + ring) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring * 16384)), "l"((&x_shared)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring * 16384)), "l"((&wg_shared)), "r"(0), "r"(y_1 * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)(ring * 16384)), "l"((&wu_shared)), "r"(0), "r"(y_1 * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert_1), "r"(0),
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
                                mbarrier_wait(output_finished_addr + 8, phase_bits_2 >> 23 & 1);
                                mbarrier_wait(prefix_released_addr, phase_bits_2 >> 22 & 1);
                                mbarrier_wait(prefix_released_addr + 8, phase_bits_2 >> 23 & 1);
                                phase_bits_2 = phase_bits_2 ^ 12582912;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_1 = 0; idx_1 < iterations_1; idx_1++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_1) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_1) * 8, phase_bits_2 >> (unsigned int)ring_1 & 1);
                                    int _smem_desc_lo_77 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
                                    int _smem_desc_lo_78 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
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
            :: "r"(_smem_desc_lo_77), "r"(_smem_desc_lo_78), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)));
                                    int _smem_desc_lo_79 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
                                    int _smem_desc_lo_80 = (((b_hi_addr) >> 4) & 0x3FFF) + (ring_1) * 1024;
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
            :: "r"(_smem_desc_lo_79), "r"(_smem_desc_lo_80), "r"((tmem_accumulator + (256))), "r"(((idx_1 == 0) ? 0 : 1)));
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_1) * 8, (uint16_t)(3));
                                    phase_bits_2 = phase_bits_2 ^ (unsigned int)(1 << ring_1);
                                    ring_1 = (ring_1 + 1) % 4;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        unsigned int _parity_output_arrived = (phase_bits_2 >> 6) % 2;
                        mbarrier_wait(output_arrived_addr, _parity_output_arrived);
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
                        gate_packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1])));
                        up_packed[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1])));
                        gate_packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3])));
                        up_packed[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3])));
                        gate_packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[4], _tmem_load_0[5])));
                        up_packed[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[4], _tmem_load_1[5])));
                        gate_packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[6], _tmem_load_0[7])));
                        up_packed[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[6], _tmem_load_1[7])));
                        gate_packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[8], _tmem_load_0[9])));
                        up_packed[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[8], _tmem_load_1[9])));
                        gate_packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[10], _tmem_load_0[11])));
                        up_packed[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[10], _tmem_load_1[11])));
                        gate_packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[12], _tmem_load_0[13])));
                        up_packed[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[12], _tmem_load_1[13])));
                        gate_packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[14], _tmem_load_0[15])));
                        up_packed[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[14], _tmem_load_1[15])));
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
                        gate_packed[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[0], _tmem_load_2[1])));
                        up_packed[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[0], _tmem_load_3[1])));
                        gate_packed[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[2], _tmem_load_2[3])));
                        up_packed[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[2], _tmem_load_3[3])));
                        gate_packed[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[4], _tmem_load_2[5])));
                        up_packed[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[4], _tmem_load_3[5])));
                        gate_packed[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[6], _tmem_load_2[7])));
                        up_packed[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[6], _tmem_load_3[7])));
                        gate_packed[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[8], _tmem_load_2[9])));
                        up_packed[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[8], _tmem_load_3[9])));
                        gate_packed[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[10], _tmem_load_2[11])));
                        up_packed[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[10], _tmem_load_3[11])));
                        gate_packed[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[12], _tmem_load_2[13])));
                        up_packed[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[12], _tmem_load_3[13])));
                        gate_packed[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[14], _tmem_load_2[15])));
                        up_packed[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[14], _tmem_load_3[15])));
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
                        gate_packed[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[0], _tmem_load_4[1])));
                        up_packed[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[0], _tmem_load_5[1])));
                        gate_packed[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[2], _tmem_load_4[3])));
                        up_packed[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[2], _tmem_load_5[3])));
                        gate_packed[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[4], _tmem_load_4[5])));
                        up_packed[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[4], _tmem_load_5[5])));
                        gate_packed[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[6], _tmem_load_4[7])));
                        up_packed[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[6], _tmem_load_5[7])));
                        gate_packed[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[8], _tmem_load_4[9])));
                        up_packed[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[8], _tmem_load_5[9])));
                        gate_packed[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[10], _tmem_load_4[11])));
                        up_packed[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[10], _tmem_load_5[11])));
                        gate_packed[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[12], _tmem_load_4[13])));
                        up_packed[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[12], _tmem_load_5[13])));
                        gate_packed[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[14], _tmem_load_4[15])));
                        up_packed[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[14], _tmem_load_5[15])));
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
                        gate_packed[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[0], _tmem_load_6[1])));
                        up_packed[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[0], _tmem_load_7[1])));
                        gate_packed[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[2], _tmem_load_6[3])));
                        up_packed[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[2], _tmem_load_7[3])));
                        gate_packed[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[4], _tmem_load_6[5])));
                        up_packed[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[4], _tmem_load_7[5])));
                        gate_packed[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[6], _tmem_load_6[7])));
                        up_packed[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[6], _tmem_load_7[7])));
                        gate_packed[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[8], _tmem_load_6[9])));
                        up_packed[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[8], _tmem_load_7[9])));
                        gate_packed[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[10], _tmem_load_6[11])));
                        up_packed[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[10], _tmem_load_7[11])));
                        gate_packed[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[12], _tmem_load_6[13])));
                        up_packed[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[12], _tmem_load_7[13])));
                        gate_packed[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[14], _tmem_load_6[15])));
                        up_packed[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[14], _tmem_load_7[15])));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[0]));
                        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[0]));
                        float gate_x = _cvt_f32_0.x;
                        float gate_y = _cvt_f32_0.y;
                        float up_x = _cvt_f32_1.x;
                        float up_y = _cvt_f32_1.y;
                        float _exp_81 = expf(gate_x * -1.0f);
                        float denominator_x = _exp_81 + 1.0f;
                        float _exp_82 = expf(gate_y * -1.0f);
                        float denominator_y = _exp_82 + 1.0f;
                        float hidden_x = gate_x / denominator_x * up_x;
                        float hidden_y = gate_y / denominator_y * up_y;
                        hidden_packed[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x, hidden_y)));
                        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[1]));
                        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[1]));
                        float gate_x_3 = _cvt_f32_2.x;
                        float gate_y_4 = _cvt_f32_2.y;
                        float up_x_5 = _cvt_f32_3.x;
                        float up_y_6 = _cvt_f32_3.y;
                        float _exp_83 = expf(gate_x_3 * -1.0f);
                        float denominator_x_7 = _exp_83 + 1.0f;
                        float _exp_84 = expf(gate_y_4 * -1.0f);
                        float denominator_y_8 = _exp_84 + 1.0f;
                        float hidden_x_9 = gate_x_3 / denominator_x_7 * up_x_5;
                        float hidden_y_10 = gate_y_4 / denominator_y_8 * up_y_6;
                        hidden_packed[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_9, hidden_y_10)));
                        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[2]));
                        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[2]));
                        float gate_x_11 = _cvt_f32_4.x;
                        float gate_y_12 = _cvt_f32_4.y;
                        float up_x_13 = _cvt_f32_5.x;
                        float up_y_14 = _cvt_f32_5.y;
                        float _exp_85 = expf(gate_x_11 * -1.0f);
                        float denominator_x_15 = _exp_85 + 1.0f;
                        float _exp_86 = expf(gate_y_12 * -1.0f);
                        float denominator_y_16 = _exp_86 + 1.0f;
                        float hidden_x_17 = gate_x_11 / denominator_x_15 * up_x_13;
                        float hidden_y_18 = gate_y_12 / denominator_y_16 * up_y_14;
                        hidden_packed[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_17, hidden_y_18)));
                        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[3]));
                        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[3]));
                        float gate_x_19 = _cvt_f32_6.x;
                        float gate_y_20 = _cvt_f32_6.y;
                        float up_x_21 = _cvt_f32_7.x;
                        float up_y_22 = _cvt_f32_7.y;
                        float _exp_87 = expf(gate_x_19 * -1.0f);
                        float denominator_x_23 = _exp_87 + 1.0f;
                        float _exp_88 = expf(gate_y_20 * -1.0f);
                        float denominator_y_24 = _exp_88 + 1.0f;
                        float hidden_x_25 = gate_x_19 / denominator_x_23 * up_x_21;
                        float hidden_y_26 = gate_y_20 / denominator_y_24 * up_y_22;
                        hidden_packed[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_25, hidden_y_26)));
                        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[4]));
                        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[4]));
                        float gate_x_27 = _cvt_f32_8.x;
                        float gate_y_28 = _cvt_f32_8.y;
                        float up_x_29 = _cvt_f32_9.x;
                        float up_y_30 = _cvt_f32_9.y;
                        float _exp_89 = expf(gate_x_27 * -1.0f);
                        float denominator_x_31 = _exp_89 + 1.0f;
                        float _exp_90 = expf(gate_y_28 * -1.0f);
                        float denominator_y_32 = _exp_90 + 1.0f;
                        float hidden_x_33 = gate_x_27 / denominator_x_31 * up_x_29;
                        float hidden_y_34 = gate_y_28 / denominator_y_32 * up_y_30;
                        hidden_packed[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_33, hidden_y_34)));
                        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[5]));
                        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[5]));
                        float gate_x_35 = _cvt_f32_10.x;
                        float gate_y_36 = _cvt_f32_10.y;
                        float up_x_37 = _cvt_f32_11.x;
                        float up_y_38 = _cvt_f32_11.y;
                        float _exp_91 = expf(gate_x_35 * -1.0f);
                        float denominator_x_39 = _exp_91 + 1.0f;
                        float _exp_92 = expf(gate_y_36 * -1.0f);
                        float denominator_y_40 = _exp_92 + 1.0f;
                        float hidden_x_41 = gate_x_35 / denominator_x_39 * up_x_37;
                        float hidden_y_42 = gate_y_36 / denominator_y_40 * up_y_38;
                        hidden_packed[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_41, hidden_y_42)));
                        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[6]));
                        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[6]));
                        float gate_x_43 = _cvt_f32_12.x;
                        float gate_y_44 = _cvt_f32_12.y;
                        float up_x_45 = _cvt_f32_13.x;
                        float up_y_46 = _cvt_f32_13.y;
                        float _exp_93 = expf(gate_x_43 * -1.0f);
                        float denominator_x_47 = _exp_93 + 1.0f;
                        float _exp_94 = expf(gate_y_44 * -1.0f);
                        float denominator_y_48 = _exp_94 + 1.0f;
                        float hidden_x_49 = gate_x_43 / denominator_x_47 * up_x_45;
                        float hidden_y_50 = gate_y_44 / denominator_y_48 * up_y_46;
                        hidden_packed[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_49, hidden_y_50)));
                        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[7]));
                        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[7]));
                        float gate_x_51 = _cvt_f32_14.x;
                        float gate_y_52 = _cvt_f32_14.y;
                        float up_x_53 = _cvt_f32_15.x;
                        float up_y_54 = _cvt_f32_15.y;
                        float _exp_95 = expf(gate_x_51 * -1.0f);
                        float denominator_x_55 = _exp_95 + 1.0f;
                        float _exp_96 = expf(gate_y_52 * -1.0f);
                        float denominator_y_56 = _exp_96 + 1.0f;
                        float hidden_x_57 = gate_x_51 / denominator_x_55 * up_x_53;
                        float hidden_y_58 = gate_y_52 / denominator_y_56 * up_y_54;
                        hidden_packed[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_57, hidden_y_58)));
                        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[8]));
                        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[8]));
                        float gate_x_59 = _cvt_f32_16.x;
                        float gate_y_60 = _cvt_f32_16.y;
                        float up_x_61 = _cvt_f32_17.x;
                        float up_y_62 = _cvt_f32_17.y;
                        float _exp_97 = expf(gate_x_59 * -1.0f);
                        float denominator_x_63 = _exp_97 + 1.0f;
                        float _exp_98 = expf(gate_y_60 * -1.0f);
                        float denominator_y_64 = _exp_98 + 1.0f;
                        float hidden_x_65 = gate_x_59 / denominator_x_63 * up_x_61;
                        float hidden_y_66 = gate_y_60 / denominator_y_64 * up_y_62;
                        hidden_packed[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_65, hidden_y_66)));
                        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[9]));
                        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[9]));
                        float gate_x_67 = _cvt_f32_18.x;
                        float gate_y_68 = _cvt_f32_18.y;
                        float up_x_69 = _cvt_f32_19.x;
                        float up_y_70 = _cvt_f32_19.y;
                        float _exp_99 = expf(gate_x_67 * -1.0f);
                        float denominator_x_71 = _exp_99 + 1.0f;
                        float _exp_100 = expf(gate_y_68 * -1.0f);
                        float denominator_y_72 = _exp_100 + 1.0f;
                        float hidden_x_73 = gate_x_67 / denominator_x_71 * up_x_69;
                        float hidden_y_74 = gate_y_68 / denominator_y_72 * up_y_70;
                        hidden_packed[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_73, hidden_y_74)));
                        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[10]));
                        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[10]));
                        float gate_x_75 = _cvt_f32_20.x;
                        float gate_y_76 = _cvt_f32_20.y;
                        float up_x_77 = _cvt_f32_21.x;
                        float up_y_78 = _cvt_f32_21.y;
                        float _exp_101 = expf(gate_x_75 * -1.0f);
                        float denominator_x_79 = _exp_101 + 1.0f;
                        float _exp_102 = expf(gate_y_76 * -1.0f);
                        float denominator_y_80 = _exp_102 + 1.0f;
                        float hidden_x_81 = gate_x_75 / denominator_x_79 * up_x_77;
                        float hidden_y_82 = gate_y_76 / denominator_y_80 * up_y_78;
                        hidden_packed[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_81, hidden_y_82)));
                        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[11]));
                        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[11]));
                        float gate_x_83 = _cvt_f32_22.x;
                        float gate_y_84 = _cvt_f32_22.y;
                        float up_x_85 = _cvt_f32_23.x;
                        float up_y_86 = _cvt_f32_23.y;
                        float _exp_103 = expf(gate_x_83 * -1.0f);
                        float denominator_x_87 = _exp_103 + 1.0f;
                        float _exp_104 = expf(gate_y_84 * -1.0f);
                        float denominator_y_88 = _exp_104 + 1.0f;
                        float hidden_x_89 = gate_x_83 / denominator_x_87 * up_x_85;
                        float hidden_y_90 = gate_y_84 / denominator_y_88 * up_y_86;
                        hidden_packed[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_89, hidden_y_90)));
                        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[12]));
                        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[12]));
                        float gate_x_91 = _cvt_f32_24.x;
                        float gate_y_92 = _cvt_f32_24.y;
                        float up_x_93 = _cvt_f32_25.x;
                        float up_y_94 = _cvt_f32_25.y;
                        float _exp_105 = expf(gate_x_91 * -1.0f);
                        float denominator_x_95 = _exp_105 + 1.0f;
                        float _exp_106 = expf(gate_y_92 * -1.0f);
                        float denominator_y_96 = _exp_106 + 1.0f;
                        float hidden_x_97 = gate_x_91 / denominator_x_95 * up_x_93;
                        float hidden_y_98 = gate_y_92 / denominator_y_96 * up_y_94;
                        hidden_packed[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_97, hidden_y_98)));
                        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[13]));
                        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[13]));
                        float gate_x_99 = _cvt_f32_26.x;
                        float gate_y_100 = _cvt_f32_26.y;
                        float up_x_101 = _cvt_f32_27.x;
                        float up_y_102 = _cvt_f32_27.y;
                        float _exp_107 = expf(gate_x_99 * -1.0f);
                        float denominator_x_103 = _exp_107 + 1.0f;
                        float _exp_108 = expf(gate_y_100 * -1.0f);
                        float denominator_y_104 = _exp_108 + 1.0f;
                        float hidden_x_105 = gate_x_99 / denominator_x_103 * up_x_101;
                        float hidden_y_106 = gate_y_100 / denominator_y_104 * up_y_102;
                        hidden_packed[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_105, hidden_y_106)));
                        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[14]));
                        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[14]));
                        float gate_x_107 = _cvt_f32_28.x;
                        float gate_y_108 = _cvt_f32_28.y;
                        float up_x_109 = _cvt_f32_29.x;
                        float up_y_110 = _cvt_f32_29.y;
                        float _exp_109 = expf(gate_x_107 * -1.0f);
                        float denominator_x_111 = _exp_109 + 1.0f;
                        float _exp_110 = expf(gate_y_108 * -1.0f);
                        float denominator_y_112 = _exp_110 + 1.0f;
                        float hidden_x_113 = gate_x_107 / denominator_x_111 * up_x_109;
                        float hidden_y_114 = gate_y_108 / denominator_y_112 * up_y_110;
                        hidden_packed[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_113, hidden_y_114)));
                        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[15]));
                        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[15]));
                        float gate_x_115 = _cvt_f32_30.x;
                        float gate_y_116 = _cvt_f32_30.y;
                        float up_x_117 = _cvt_f32_31.x;
                        float up_y_118 = _cvt_f32_31.y;
                        float _exp_111 = expf(gate_x_115 * -1.0f);
                        float denominator_x_119 = _exp_111 + 1.0f;
                        float _exp_112 = expf(gate_y_116 * -1.0f);
                        float denominator_y_120 = _exp_112 + 1.0f;
                        float hidden_x_121 = gate_x_115 / denominator_x_119 * up_x_117;
                        float hidden_y_122 = gate_y_116 / denominator_y_120 * up_y_118;
                        hidden_packed[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_121, hidden_y_122)));
                        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[16]));
                        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[16]));
                        float gate_x_123 = _cvt_f32_32.x;
                        float gate_y_124 = _cvt_f32_32.y;
                        float up_x_125 = _cvt_f32_33.x;
                        float up_y_126 = _cvt_f32_33.y;
                        float _exp_113 = expf(gate_x_123 * -1.0f);
                        float denominator_x_127 = _exp_113 + 1.0f;
                        float _exp_114 = expf(gate_y_124 * -1.0f);
                        float denominator_y_128 = _exp_114 + 1.0f;
                        float hidden_x_129 = gate_x_123 / denominator_x_127 * up_x_125;
                        float hidden_y_130 = gate_y_124 / denominator_y_128 * up_y_126;
                        hidden_packed[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_129, hidden_y_130)));
                        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[17]));
                        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[17]));
                        float gate_x_131 = _cvt_f32_34.x;
                        float gate_y_132 = _cvt_f32_34.y;
                        float up_x_133 = _cvt_f32_35.x;
                        float up_y_134 = _cvt_f32_35.y;
                        float _exp_115 = expf(gate_x_131 * -1.0f);
                        float denominator_x_135 = _exp_115 + 1.0f;
                        float _exp_116 = expf(gate_y_132 * -1.0f);
                        float denominator_y_136 = _exp_116 + 1.0f;
                        float hidden_x_137 = gate_x_131 / denominator_x_135 * up_x_133;
                        float hidden_y_138 = gate_y_132 / denominator_y_136 * up_y_134;
                        hidden_packed[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_137, hidden_y_138)));
                        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[18]));
                        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[18]));
                        float gate_x_139 = _cvt_f32_36.x;
                        float gate_y_140 = _cvt_f32_36.y;
                        float up_x_141 = _cvt_f32_37.x;
                        float up_y_142 = _cvt_f32_37.y;
                        float _exp_117 = expf(gate_x_139 * -1.0f);
                        float denominator_x_143 = _exp_117 + 1.0f;
                        float _exp_118 = expf(gate_y_140 * -1.0f);
                        float denominator_y_144 = _exp_118 + 1.0f;
                        float hidden_x_145 = gate_x_139 / denominator_x_143 * up_x_141;
                        float hidden_y_146 = gate_y_140 / denominator_y_144 * up_y_142;
                        hidden_packed[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_145, hidden_y_146)));
                        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[19]));
                        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[19]));
                        float gate_x_147 = _cvt_f32_38.x;
                        float gate_y_148 = _cvt_f32_38.y;
                        float up_x_149 = _cvt_f32_39.x;
                        float up_y_150 = _cvt_f32_39.y;
                        float _exp_119 = expf(gate_x_147 * -1.0f);
                        float denominator_x_151 = _exp_119 + 1.0f;
                        float _exp_120 = expf(gate_y_148 * -1.0f);
                        float denominator_y_152 = _exp_120 + 1.0f;
                        float hidden_x_153 = gate_x_147 / denominator_x_151 * up_x_149;
                        float hidden_y_154 = gate_y_148 / denominator_y_152 * up_y_150;
                        hidden_packed[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_153, hidden_y_154)));
                        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[20]));
                        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[20]));
                        float gate_x_155 = _cvt_f32_40.x;
                        float gate_y_156 = _cvt_f32_40.y;
                        float up_x_157 = _cvt_f32_41.x;
                        float up_y_158 = _cvt_f32_41.y;
                        float _exp_121 = expf(gate_x_155 * -1.0f);
                        float denominator_x_159 = _exp_121 + 1.0f;
                        float _exp_122 = expf(gate_y_156 * -1.0f);
                        float denominator_y_160 = _exp_122 + 1.0f;
                        float hidden_x_161 = gate_x_155 / denominator_x_159 * up_x_157;
                        float hidden_y_162 = gate_y_156 / denominator_y_160 * up_y_158;
                        hidden_packed[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_161, hidden_y_162)));
                        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[21]));
                        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[21]));
                        float gate_x_163 = _cvt_f32_42.x;
                        float gate_y_164 = _cvt_f32_42.y;
                        float up_x_165 = _cvt_f32_43.x;
                        float up_y_166 = _cvt_f32_43.y;
                        float _exp_123 = expf(gate_x_163 * -1.0f);
                        float denominator_x_167 = _exp_123 + 1.0f;
                        float _exp_124 = expf(gate_y_164 * -1.0f);
                        float denominator_y_168 = _exp_124 + 1.0f;
                        float hidden_x_169 = gate_x_163 / denominator_x_167 * up_x_165;
                        float hidden_y_170 = gate_y_164 / denominator_y_168 * up_y_166;
                        hidden_packed[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_169, hidden_y_170)));
                        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[22]));
                        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[22]));
                        float gate_x_171 = _cvt_f32_44.x;
                        float gate_y_172 = _cvt_f32_44.y;
                        float up_x_173 = _cvt_f32_45.x;
                        float up_y_174 = _cvt_f32_45.y;
                        float _exp_125 = expf(gate_x_171 * -1.0f);
                        float denominator_x_175 = _exp_125 + 1.0f;
                        float _exp_126 = expf(gate_y_172 * -1.0f);
                        float denominator_y_176 = _exp_126 + 1.0f;
                        float hidden_x_177 = gate_x_171 / denominator_x_175 * up_x_173;
                        float hidden_y_178 = gate_y_172 / denominator_y_176 * up_y_174;
                        hidden_packed[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_177, hidden_y_178)));
                        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[23]));
                        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[23]));
                        float gate_x_179 = _cvt_f32_46.x;
                        float gate_y_180 = _cvt_f32_46.y;
                        float up_x_181 = _cvt_f32_47.x;
                        float up_y_182 = _cvt_f32_47.y;
                        float _exp_127 = expf(gate_x_179 * -1.0f);
                        float denominator_x_183 = _exp_127 + 1.0f;
                        float _exp_128 = expf(gate_y_180 * -1.0f);
                        float denominator_y_184 = _exp_128 + 1.0f;
                        float hidden_x_185 = gate_x_179 / denominator_x_183 * up_x_181;
                        float hidden_y_186 = gate_y_180 / denominator_y_184 * up_y_182;
                        hidden_packed[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_185, hidden_y_186)));
                        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[24]));
                        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[24]));
                        float gate_x_187 = _cvt_f32_48.x;
                        float gate_y_188 = _cvt_f32_48.y;
                        float up_x_189 = _cvt_f32_49.x;
                        float up_y_190 = _cvt_f32_49.y;
                        float _exp_129 = expf(gate_x_187 * -1.0f);
                        float denominator_x_191 = _exp_129 + 1.0f;
                        float _exp_130 = expf(gate_y_188 * -1.0f);
                        float denominator_y_192 = _exp_130 + 1.0f;
                        float hidden_x_193 = gate_x_187 / denominator_x_191 * up_x_189;
                        float hidden_y_194 = gate_y_188 / denominator_y_192 * up_y_190;
                        hidden_packed[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_193, hidden_y_194)));
                        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[25]));
                        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[25]));
                        float gate_x_195 = _cvt_f32_50.x;
                        float gate_y_196 = _cvt_f32_50.y;
                        float up_x_197 = _cvt_f32_51.x;
                        float up_y_198 = _cvt_f32_51.y;
                        float _exp_131 = expf(gate_x_195 * -1.0f);
                        float denominator_x_199 = _exp_131 + 1.0f;
                        float _exp_132 = expf(gate_y_196 * -1.0f);
                        float denominator_y_200 = _exp_132 + 1.0f;
                        float hidden_x_201 = gate_x_195 / denominator_x_199 * up_x_197;
                        float hidden_y_202 = gate_y_196 / denominator_y_200 * up_y_198;
                        hidden_packed[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_201, hidden_y_202)));
                        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[26]));
                        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[26]));
                        float gate_x_203 = _cvt_f32_52.x;
                        float gate_y_204 = _cvt_f32_52.y;
                        float up_x_205 = _cvt_f32_53.x;
                        float up_y_206 = _cvt_f32_53.y;
                        float _exp_133 = expf(gate_x_203 * -1.0f);
                        float denominator_x_207 = _exp_133 + 1.0f;
                        float _exp_134 = expf(gate_y_204 * -1.0f);
                        float denominator_y_208 = _exp_134 + 1.0f;
                        float hidden_x_209 = gate_x_203 / denominator_x_207 * up_x_205;
                        float hidden_y_210 = gate_y_204 / denominator_y_208 * up_y_206;
                        hidden_packed[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_209, hidden_y_210)));
                        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[27]));
                        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[27]));
                        float gate_x_211 = _cvt_f32_54.x;
                        float gate_y_212 = _cvt_f32_54.y;
                        float up_x_213 = _cvt_f32_55.x;
                        float up_y_214 = _cvt_f32_55.y;
                        float _exp_135 = expf(gate_x_211 * -1.0f);
                        float denominator_x_215 = _exp_135 + 1.0f;
                        float _exp_136 = expf(gate_y_212 * -1.0f);
                        float denominator_y_216 = _exp_136 + 1.0f;
                        float hidden_x_217 = gate_x_211 / denominator_x_215 * up_x_213;
                        float hidden_y_218 = gate_y_212 / denominator_y_216 * up_y_214;
                        hidden_packed[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_217, hidden_y_218)));
                        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[28]));
                        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[28]));
                        float gate_x_219 = _cvt_f32_56.x;
                        float gate_y_220 = _cvt_f32_56.y;
                        float up_x_221 = _cvt_f32_57.x;
                        float up_y_222 = _cvt_f32_57.y;
                        float _exp_137 = expf(gate_x_219 * -1.0f);
                        float denominator_x_223 = _exp_137 + 1.0f;
                        float _exp_138 = expf(gate_y_220 * -1.0f);
                        float denominator_y_224 = _exp_138 + 1.0f;
                        float hidden_x_225 = gate_x_219 / denominator_x_223 * up_x_221;
                        float hidden_y_226 = gate_y_220 / denominator_y_224 * up_y_222;
                        hidden_packed[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_225, hidden_y_226)));
                        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[29]));
                        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[29]));
                        float gate_x_227 = _cvt_f32_58.x;
                        float gate_y_228 = _cvt_f32_58.y;
                        float up_x_229 = _cvt_f32_59.x;
                        float up_y_230 = _cvt_f32_59.y;
                        float _exp_139 = expf(gate_x_227 * -1.0f);
                        float denominator_x_231 = _exp_139 + 1.0f;
                        float _exp_140 = expf(gate_y_228 * -1.0f);
                        float denominator_y_232 = _exp_140 + 1.0f;
                        float hidden_x_233 = gate_x_227 / denominator_x_231 * up_x_229;
                        float hidden_y_234 = gate_y_228 / denominator_y_232 * up_y_230;
                        hidden_packed[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_233, hidden_y_234)));
                        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[30]));
                        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[30]));
                        float gate_x_235 = _cvt_f32_60.x;
                        float gate_y_236 = _cvt_f32_60.y;
                        float up_x_237 = _cvt_f32_61.x;
                        float up_y_238 = _cvt_f32_61.y;
                        float _exp_141 = expf(gate_x_235 * -1.0f);
                        float denominator_x_239 = _exp_141 + 1.0f;
                        float _exp_142 = expf(gate_y_236 * -1.0f);
                        float denominator_y_240 = _exp_142 + 1.0f;
                        float hidden_x_241 = gate_x_235 / denominator_x_239 * up_x_237;
                        float hidden_y_242 = gate_y_236 / denominator_y_240 * up_y_238;
                        hidden_packed[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_241, hidden_y_242)));
                        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed[31]));
                        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed[31]));
                        float gate_x_243 = _cvt_f32_62.x;
                        float gate_y_244 = _cvt_f32_62.y;
                        float up_x_245 = _cvt_f32_63.x;
                        float up_y_246 = _cvt_f32_63.y;
                        float _exp_143 = expf(gate_x_243 * -1.0f);
                        float denominator_x_247 = _exp_143 + 1.0f;
                        float _exp_144 = expf(gate_y_244 * -1.0f);
                        float denominator_y_248 = _exp_144 + 1.0f;
                        float hidden_x_249 = gate_x_243 / denominator_x_247 * up_x_245;
                        float hidden_y_250 = gate_y_244 / denominator_y_248 * up_y_246;
                        hidden_packed[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_249, hidden_y_250)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_251 = tid / 32;
                        int lane_1 = tid % 32;
                        #pragma unroll
                        for (int half_8 = 0; half_8 < 2; half_8++) {
                            #pragma unroll
                            for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                int row_7 = warp_251 * 32 + half_8 * 16 + lane_1 % 16;
                                int col_35 = col_tile_8 * 16 + lane_1 / 16 * 8;
                                unsigned int address_3 = d_smem_addr + (unsigned int)((row_7 * 32 + col_35) * 2);
                                address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                int offset_2 = half_8 * 8 + col_tile_8 * 4;
                                uint32_t _stmatrix_addr_145 = static_cast<uint32_t>(address_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_145), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_2 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_252 = tid / 32;
                        int lane_253 = tid % 32;
                        #pragma unroll
                        for (int half_9 = 0; half_9 < 2; half_9++) {
                            #pragma unroll
                            for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                int row_8 = warp_252 * 32 + half_9 * 16 + lane_253 % 16;
                                int col_36 = col_tile_9 * 16 + lane_253 / 16 * 8;
                                unsigned int address_3_1 = d_smem_addr + 8192 + (unsigned int)((row_8 * 32 + col_36) * 2);
                                address_3_1 = address_3_1 ^ (address_3_1 & 511) >> 7 << 4;
                                int offset_3 = half_9 * 8 + col_tile_9 * 4;
                                uint32_t _stmatrix_addr_146 = static_cast<uint32_t>(address_3_1);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_146), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_3 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_254 = tid / 32;
                        int lane_255 = tid % 32;
                        #pragma unroll
                        for (int half_10 = 0; half_10 < 2; half_10++) {
                            #pragma unroll
                            for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                int row_9 = warp_254 * 32 + half_10 * 16 + lane_255 % 16;
                                int col_37 = col_tile_10 * 16 + lane_255 / 16 * 8;
                                unsigned int address_3_2 = d_smem_addr + 16384 + (unsigned int)((row_9 * 32 + col_37) * 2);
                                address_3_2 = address_3_2 ^ (address_3_2 & 511) >> 7 << 4;
                                int offset_4 = half_10 * 8 + col_tile_10 * 4;
                                uint32_t _stmatrix_addr_147 = static_cast<uint32_t>(address_3_2);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_147), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_4 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_256 = tid / 32;
                        int lane_257 = tid % 32;
                        #pragma unroll
                        for (int half_11 = 0; half_11 < 2; half_11++) {
                            #pragma unroll
                            for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                int row_10 = warp_256 * 32 + half_11 * 16 + lane_257 % 16;
                                int col_38 = col_tile_11 * 16 + lane_257 / 16 * 8;
                                unsigned int address_3_3 = d_smem_addr + (unsigned int)((row_10 * 32 + col_38) * 2);
                                address_3_3 = address_3_3 ^ (address_3_3 & 511) >> 7 << 4;
                                int offset_5 = 16 + half_11 * 8 + col_tile_11 * 4;
                                uint32_t _stmatrix_addr_148 = static_cast<uint32_t>(address_3_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_148), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_5 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_258 = tid / 32;
                        int lane_259 = tid % 32;
                        #pragma unroll
                        for (int half_12 = 0; half_12 < 2; half_12++) {
                            #pragma unroll
                            for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                int row_11 = warp_258 * 32 + half_12 * 16 + lane_259 % 16;
                                int col_39 = col_tile_12 * 16 + lane_259 / 16 * 8;
                                unsigned int address_3_4 = d_smem_addr + 8192 + (unsigned int)((row_11 * 32 + col_39) * 2);
                                address_3_4 = address_3_4 ^ (address_3_4 & 511) >> 7 << 4;
                                int offset_6 = 16 + half_12 * 8 + col_tile_12 * 4;
                                uint32_t _stmatrix_addr_149 = static_cast<uint32_t>(address_3_4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_149), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_6 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_260 = tid / 32;
                        int lane_261 = tid % 32;
                        #pragma unroll
                        for (int half_13 = 0; half_13 < 2; half_13++) {
                            #pragma unroll
                            for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                int row_12 = warp_260 * 32 + half_13 * 16 + lane_261 % 16;
                                int col_40 = col_tile_13 * 16 + lane_261 / 16 * 8;
                                unsigned int address_3_5 = d_smem_addr + 16384 + (unsigned int)((row_12 * 32 + col_40) * 2);
                                address_3_5 = address_3_5 ^ (address_3_5 & 511) >> 7 << 4;
                                int offset_7 = 16 + half_13 * 8 + col_tile_13 * 4;
                                uint32_t _stmatrix_addr_150 = static_cast<uint32_t>(address_3_5);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_150), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_7 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
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
                        gate_packed_262[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[0], _tmem_load_8[1])));
                        up_packed_263[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[0], _tmem_load_9[1])));
                        gate_packed_262[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[2], _tmem_load_8[3])));
                        up_packed_263[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[2], _tmem_load_9[3])));
                        gate_packed_262[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[4], _tmem_load_8[5])));
                        up_packed_263[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[4], _tmem_load_9[5])));
                        gate_packed_262[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[6], _tmem_load_8[7])));
                        up_packed_263[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[6], _tmem_load_9[7])));
                        gate_packed_262[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[8], _tmem_load_8[9])));
                        up_packed_263[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[8], _tmem_load_9[9])));
                        gate_packed_262[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[10], _tmem_load_8[11])));
                        up_packed_263[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[10], _tmem_load_9[11])));
                        gate_packed_262[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[12], _tmem_load_8[13])));
                        up_packed_263[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[12], _tmem_load_9[13])));
                        gate_packed_262[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[14], _tmem_load_8[15])));
                        up_packed_263[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[14], _tmem_load_9[15])));
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
                        gate_packed_262[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[0], _tmem_load_10[1])));
                        up_packed_263[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[0], _tmem_load_11[1])));
                        gate_packed_262[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[2], _tmem_load_10[3])));
                        up_packed_263[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[2], _tmem_load_11[3])));
                        gate_packed_262[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[4], _tmem_load_10[5])));
                        up_packed_263[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[4], _tmem_load_11[5])));
                        gate_packed_262[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[6], _tmem_load_10[7])));
                        up_packed_263[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[6], _tmem_load_11[7])));
                        gate_packed_262[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[8], _tmem_load_10[9])));
                        up_packed_263[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[8], _tmem_load_11[9])));
                        gate_packed_262[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[10], _tmem_load_10[11])));
                        up_packed_263[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[10], _tmem_load_11[11])));
                        gate_packed_262[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[12], _tmem_load_10[13])));
                        up_packed_263[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[12], _tmem_load_11[13])));
                        gate_packed_262[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[14], _tmem_load_10[15])));
                        up_packed_263[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[14], _tmem_load_11[15])));
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
                        gate_packed_262[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[0], _tmem_load_12[1])));
                        up_packed_263[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[0], _tmem_load_13[1])));
                        gate_packed_262[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[2], _tmem_load_12[3])));
                        up_packed_263[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[2], _tmem_load_13[3])));
                        gate_packed_262[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[4], _tmem_load_12[5])));
                        up_packed_263[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[4], _tmem_load_13[5])));
                        gate_packed_262[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[6], _tmem_load_12[7])));
                        up_packed_263[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[6], _tmem_load_13[7])));
                        gate_packed_262[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[8], _tmem_load_12[9])));
                        up_packed_263[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[8], _tmem_load_13[9])));
                        gate_packed_262[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[10], _tmem_load_12[11])));
                        up_packed_263[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[10], _tmem_load_13[11])));
                        gate_packed_262[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[12], _tmem_load_12[13])));
                        up_packed_263[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[12], _tmem_load_13[13])));
                        gate_packed_262[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[14], _tmem_load_12[15])));
                        up_packed_263[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[14], _tmem_load_13[15])));
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
                        gate_packed_262[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[0], _tmem_load_14[1])));
                        up_packed_263[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[0], _tmem_load_15[1])));
                        gate_packed_262[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[2], _tmem_load_14[3])));
                        up_packed_263[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[2], _tmem_load_15[3])));
                        gate_packed_262[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[4], _tmem_load_14[5])));
                        up_packed_263[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[4], _tmem_load_15[5])));
                        gate_packed_262[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[6], _tmem_load_14[7])));
                        up_packed_263[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[6], _tmem_load_15[7])));
                        gate_packed_262[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[8], _tmem_load_14[9])));
                        up_packed_263[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[8], _tmem_load_15[9])));
                        gate_packed_262[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[10], _tmem_load_14[11])));
                        up_packed_263[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[10], _tmem_load_15[11])));
                        gate_packed_262[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[12], _tmem_load_14[13])));
                        up_packed_263[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[12], _tmem_load_15[13])));
                        gate_packed_262[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[14], _tmem_load_14[15])));
                        up_packed_263[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[14], _tmem_load_15[15])));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[0]));
                        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[0]));
                        float gate_x_269 = _cvt_f32_64.x;
                        float gate_y_270 = _cvt_f32_64.y;
                        float up_x_271 = _cvt_f32_65.x;
                        float up_y_272 = _cvt_f32_65.y;
                        float _exp_151 = expf(gate_x_269 * -1.0f);
                        float denominator_x_273 = _exp_151 + 1.0f;
                        float _exp_152 = expf(gate_y_270 * -1.0f);
                        float denominator_y_274 = _exp_152 + 1.0f;
                        float hidden_x_275 = gate_x_269 / denominator_x_273 * up_x_271;
                        float hidden_y_276 = gate_y_270 / denominator_y_274 * up_y_272;
                        hidden_packed_264[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_275, hidden_y_276)));
                        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[1]));
                        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[1]));
                        float gate_x_277 = _cvt_f32_66.x;
                        float gate_y_278 = _cvt_f32_66.y;
                        float up_x_279 = _cvt_f32_67.x;
                        float up_y_280 = _cvt_f32_67.y;
                        float _exp_153 = expf(gate_x_277 * -1.0f);
                        float denominator_x_281 = _exp_153 + 1.0f;
                        float _exp_154 = expf(gate_y_278 * -1.0f);
                        float denominator_y_282 = _exp_154 + 1.0f;
                        float hidden_x_283 = gate_x_277 / denominator_x_281 * up_x_279;
                        float hidden_y_284 = gate_y_278 / denominator_y_282 * up_y_280;
                        hidden_packed_264[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_283, hidden_y_284)));
                        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[2]));
                        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[2]));
                        float gate_x_285 = _cvt_f32_68.x;
                        float gate_y_286 = _cvt_f32_68.y;
                        float up_x_287 = _cvt_f32_69.x;
                        float up_y_288 = _cvt_f32_69.y;
                        float _exp_155 = expf(gate_x_285 * -1.0f);
                        float denominator_x_289 = _exp_155 + 1.0f;
                        float _exp_156 = expf(gate_y_286 * -1.0f);
                        float denominator_y_290 = _exp_156 + 1.0f;
                        float hidden_x_291 = gate_x_285 / denominator_x_289 * up_x_287;
                        float hidden_y_292 = gate_y_286 / denominator_y_290 * up_y_288;
                        hidden_packed_264[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_291, hidden_y_292)));
                        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[3]));
                        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[3]));
                        float gate_x_293 = _cvt_f32_70.x;
                        float gate_y_294 = _cvt_f32_70.y;
                        float up_x_295 = _cvt_f32_71.x;
                        float up_y_296 = _cvt_f32_71.y;
                        float _exp_157 = expf(gate_x_293 * -1.0f);
                        float denominator_x_297 = _exp_157 + 1.0f;
                        float _exp_158 = expf(gate_y_294 * -1.0f);
                        float denominator_y_298 = _exp_158 + 1.0f;
                        float hidden_x_299 = gate_x_293 / denominator_x_297 * up_x_295;
                        float hidden_y_300 = gate_y_294 / denominator_y_298 * up_y_296;
                        hidden_packed_264[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_299, hidden_y_300)));
                        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[4]));
                        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[4]));
                        float gate_x_301 = _cvt_f32_72.x;
                        float gate_y_302 = _cvt_f32_72.y;
                        float up_x_303 = _cvt_f32_73.x;
                        float up_y_304 = _cvt_f32_73.y;
                        float _exp_159 = expf(gate_x_301 * -1.0f);
                        float denominator_x_305 = _exp_159 + 1.0f;
                        float _exp_160 = expf(gate_y_302 * -1.0f);
                        float denominator_y_306 = _exp_160 + 1.0f;
                        float hidden_x_307 = gate_x_301 / denominator_x_305 * up_x_303;
                        float hidden_y_308 = gate_y_302 / denominator_y_306 * up_y_304;
                        hidden_packed_264[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_307, hidden_y_308)));
                        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[5]));
                        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[5]));
                        float gate_x_309 = _cvt_f32_74.x;
                        float gate_y_310 = _cvt_f32_74.y;
                        float up_x_311 = _cvt_f32_75.x;
                        float up_y_312 = _cvt_f32_75.y;
                        float _exp_161 = expf(gate_x_309 * -1.0f);
                        float denominator_x_313 = _exp_161 + 1.0f;
                        float _exp_162 = expf(gate_y_310 * -1.0f);
                        float denominator_y_314 = _exp_162 + 1.0f;
                        float hidden_x_315 = gate_x_309 / denominator_x_313 * up_x_311;
                        float hidden_y_316 = gate_y_310 / denominator_y_314 * up_y_312;
                        hidden_packed_264[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_315, hidden_y_316)));
                        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[6]));
                        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[6]));
                        float gate_x_317 = _cvt_f32_76.x;
                        float gate_y_318 = _cvt_f32_76.y;
                        float up_x_319 = _cvt_f32_77.x;
                        float up_y_320 = _cvt_f32_77.y;
                        float _exp_163 = expf(gate_x_317 * -1.0f);
                        float denominator_x_321 = _exp_163 + 1.0f;
                        float _exp_164 = expf(gate_y_318 * -1.0f);
                        float denominator_y_322 = _exp_164 + 1.0f;
                        float hidden_x_323 = gate_x_317 / denominator_x_321 * up_x_319;
                        float hidden_y_324 = gate_y_318 / denominator_y_322 * up_y_320;
                        hidden_packed_264[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_323, hidden_y_324)));
                        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[7]));
                        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[7]));
                        float gate_x_325 = _cvt_f32_78.x;
                        float gate_y_326 = _cvt_f32_78.y;
                        float up_x_327 = _cvt_f32_79.x;
                        float up_y_328 = _cvt_f32_79.y;
                        float _exp_165 = expf(gate_x_325 * -1.0f);
                        float denominator_x_329 = _exp_165 + 1.0f;
                        float _exp_166 = expf(gate_y_326 * -1.0f);
                        float denominator_y_330 = _exp_166 + 1.0f;
                        float hidden_x_331 = gate_x_325 / denominator_x_329 * up_x_327;
                        float hidden_y_332 = gate_y_326 / denominator_y_330 * up_y_328;
                        hidden_packed_264[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_331, hidden_y_332)));
                        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[8]));
                        float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[8]));
                        float gate_x_333 = _cvt_f32_80.x;
                        float gate_y_334 = _cvt_f32_80.y;
                        float up_x_335 = _cvt_f32_81.x;
                        float up_y_336 = _cvt_f32_81.y;
                        float _exp_167 = expf(gate_x_333 * -1.0f);
                        float denominator_x_337 = _exp_167 + 1.0f;
                        float _exp_168 = expf(gate_y_334 * -1.0f);
                        float denominator_y_338 = _exp_168 + 1.0f;
                        float hidden_x_339 = gate_x_333 / denominator_x_337 * up_x_335;
                        float hidden_y_340 = gate_y_334 / denominator_y_338 * up_y_336;
                        hidden_packed_264[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_339, hidden_y_340)));
                        float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[9]));
                        float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[9]));
                        float gate_x_341 = _cvt_f32_82.x;
                        float gate_y_342 = _cvt_f32_82.y;
                        float up_x_343 = _cvt_f32_83.x;
                        float up_y_344 = _cvt_f32_83.y;
                        float _exp_169 = expf(gate_x_341 * -1.0f);
                        float denominator_x_345 = _exp_169 + 1.0f;
                        float _exp_170 = expf(gate_y_342 * -1.0f);
                        float denominator_y_346 = _exp_170 + 1.0f;
                        float hidden_x_347 = gate_x_341 / denominator_x_345 * up_x_343;
                        float hidden_y_348 = gate_y_342 / denominator_y_346 * up_y_344;
                        hidden_packed_264[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_347, hidden_y_348)));
                        float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[10]));
                        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[10]));
                        float gate_x_349 = _cvt_f32_84.x;
                        float gate_y_350 = _cvt_f32_84.y;
                        float up_x_351 = _cvt_f32_85.x;
                        float up_y_352 = _cvt_f32_85.y;
                        float _exp_171 = expf(gate_x_349 * -1.0f);
                        float denominator_x_353 = _exp_171 + 1.0f;
                        float _exp_172 = expf(gate_y_350 * -1.0f);
                        float denominator_y_354 = _exp_172 + 1.0f;
                        float hidden_x_355 = gate_x_349 / denominator_x_353 * up_x_351;
                        float hidden_y_356 = gate_y_350 / denominator_y_354 * up_y_352;
                        hidden_packed_264[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_355, hidden_y_356)));
                        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[11]));
                        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[11]));
                        float gate_x_357 = _cvt_f32_86.x;
                        float gate_y_358 = _cvt_f32_86.y;
                        float up_x_359 = _cvt_f32_87.x;
                        float up_y_360 = _cvt_f32_87.y;
                        float _exp_173 = expf(gate_x_357 * -1.0f);
                        float denominator_x_361 = _exp_173 + 1.0f;
                        float _exp_174 = expf(gate_y_358 * -1.0f);
                        float denominator_y_362 = _exp_174 + 1.0f;
                        float hidden_x_363 = gate_x_357 / denominator_x_361 * up_x_359;
                        float hidden_y_364 = gate_y_358 / denominator_y_362 * up_y_360;
                        hidden_packed_264[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_363, hidden_y_364)));
                        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[12]));
                        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[12]));
                        float gate_x_365 = _cvt_f32_88.x;
                        float gate_y_366 = _cvt_f32_88.y;
                        float up_x_367 = _cvt_f32_89.x;
                        float up_y_368 = _cvt_f32_89.y;
                        float _exp_175 = expf(gate_x_365 * -1.0f);
                        float denominator_x_369 = _exp_175 + 1.0f;
                        float _exp_176 = expf(gate_y_366 * -1.0f);
                        float denominator_y_370 = _exp_176 + 1.0f;
                        float hidden_x_371 = gate_x_365 / denominator_x_369 * up_x_367;
                        float hidden_y_372 = gate_y_366 / denominator_y_370 * up_y_368;
                        hidden_packed_264[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_371, hidden_y_372)));
                        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[13]));
                        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[13]));
                        float gate_x_373 = _cvt_f32_90.x;
                        float gate_y_374 = _cvt_f32_90.y;
                        float up_x_375 = _cvt_f32_91.x;
                        float up_y_376 = _cvt_f32_91.y;
                        float _exp_177 = expf(gate_x_373 * -1.0f);
                        float denominator_x_377 = _exp_177 + 1.0f;
                        float _exp_178 = expf(gate_y_374 * -1.0f);
                        float denominator_y_378 = _exp_178 + 1.0f;
                        float hidden_x_379 = gate_x_373 / denominator_x_377 * up_x_375;
                        float hidden_y_380 = gate_y_374 / denominator_y_378 * up_y_376;
                        hidden_packed_264[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_379, hidden_y_380)));
                        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[14]));
                        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[14]));
                        float gate_x_381 = _cvt_f32_92.x;
                        float gate_y_382 = _cvt_f32_92.y;
                        float up_x_383 = _cvt_f32_93.x;
                        float up_y_384 = _cvt_f32_93.y;
                        float _exp_179 = expf(gate_x_381 * -1.0f);
                        float denominator_x_385 = _exp_179 + 1.0f;
                        float _exp_180 = expf(gate_y_382 * -1.0f);
                        float denominator_y_386 = _exp_180 + 1.0f;
                        float hidden_x_387 = gate_x_381 / denominator_x_385 * up_x_383;
                        float hidden_y_388 = gate_y_382 / denominator_y_386 * up_y_384;
                        hidden_packed_264[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_387, hidden_y_388)));
                        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[15]));
                        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[15]));
                        float gate_x_389 = _cvt_f32_94.x;
                        float gate_y_390 = _cvt_f32_94.y;
                        float up_x_391 = _cvt_f32_95.x;
                        float up_y_392 = _cvt_f32_95.y;
                        float _exp_181 = expf(gate_x_389 * -1.0f);
                        float denominator_x_393 = _exp_181 + 1.0f;
                        float _exp_182 = expf(gate_y_390 * -1.0f);
                        float denominator_y_394 = _exp_182 + 1.0f;
                        float hidden_x_395 = gate_x_389 / denominator_x_393 * up_x_391;
                        float hidden_y_396 = gate_y_390 / denominator_y_394 * up_y_392;
                        hidden_packed_264[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_395, hidden_y_396)));
                        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[16]));
                        float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[16]));
                        float gate_x_397 = _cvt_f32_96.x;
                        float gate_y_398 = _cvt_f32_96.y;
                        float up_x_399 = _cvt_f32_97.x;
                        float up_y_400 = _cvt_f32_97.y;
                        float _exp_183 = expf(gate_x_397 * -1.0f);
                        float denominator_x_401 = _exp_183 + 1.0f;
                        float _exp_184 = expf(gate_y_398 * -1.0f);
                        float denominator_y_402 = _exp_184 + 1.0f;
                        float hidden_x_403 = gate_x_397 / denominator_x_401 * up_x_399;
                        float hidden_y_404 = gate_y_398 / denominator_y_402 * up_y_400;
                        hidden_packed_264[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_403, hidden_y_404)));
                        float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[17]));
                        float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[17]));
                        float gate_x_405 = _cvt_f32_98.x;
                        float gate_y_406 = _cvt_f32_98.y;
                        float up_x_407 = _cvt_f32_99.x;
                        float up_y_408 = _cvt_f32_99.y;
                        float _exp_185 = expf(gate_x_405 * -1.0f);
                        float denominator_x_409 = _exp_185 + 1.0f;
                        float _exp_186 = expf(gate_y_406 * -1.0f);
                        float denominator_y_410 = _exp_186 + 1.0f;
                        float hidden_x_411 = gate_x_405 / denominator_x_409 * up_x_407;
                        float hidden_y_412 = gate_y_406 / denominator_y_410 * up_y_408;
                        hidden_packed_264[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_411, hidden_y_412)));
                        float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[18]));
                        float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[18]));
                        float gate_x_413 = _cvt_f32_100.x;
                        float gate_y_414 = _cvt_f32_100.y;
                        float up_x_415 = _cvt_f32_101.x;
                        float up_y_416 = _cvt_f32_101.y;
                        float _exp_187 = expf(gate_x_413 * -1.0f);
                        float denominator_x_417 = _exp_187 + 1.0f;
                        float _exp_188 = expf(gate_y_414 * -1.0f);
                        float denominator_y_418 = _exp_188 + 1.0f;
                        float hidden_x_419 = gate_x_413 / denominator_x_417 * up_x_415;
                        float hidden_y_420 = gate_y_414 / denominator_y_418 * up_y_416;
                        hidden_packed_264[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_419, hidden_y_420)));
                        float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[19]));
                        float2 _cvt_f32_103 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[19]));
                        float gate_x_421 = _cvt_f32_102.x;
                        float gate_y_422 = _cvt_f32_102.y;
                        float up_x_423 = _cvt_f32_103.x;
                        float up_y_424 = _cvt_f32_103.y;
                        float _exp_189 = expf(gate_x_421 * -1.0f);
                        float denominator_x_425 = _exp_189 + 1.0f;
                        float _exp_190 = expf(gate_y_422 * -1.0f);
                        float denominator_y_426 = _exp_190 + 1.0f;
                        float hidden_x_427 = gate_x_421 / denominator_x_425 * up_x_423;
                        float hidden_y_428 = gate_y_422 / denominator_y_426 * up_y_424;
                        hidden_packed_264[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_427, hidden_y_428)));
                        float2 _cvt_f32_104 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[20]));
                        float2 _cvt_f32_105 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[20]));
                        float gate_x_429 = _cvt_f32_104.x;
                        float gate_y_430 = _cvt_f32_104.y;
                        float up_x_431 = _cvt_f32_105.x;
                        float up_y_432 = _cvt_f32_105.y;
                        float _exp_191 = expf(gate_x_429 * -1.0f);
                        float denominator_x_433 = _exp_191 + 1.0f;
                        float _exp_192 = expf(gate_y_430 * -1.0f);
                        float denominator_y_434 = _exp_192 + 1.0f;
                        float hidden_x_435 = gate_x_429 / denominator_x_433 * up_x_431;
                        float hidden_y_436 = gate_y_430 / denominator_y_434 * up_y_432;
                        hidden_packed_264[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_435, hidden_y_436)));
                        float2 _cvt_f32_106 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[21]));
                        float2 _cvt_f32_107 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[21]));
                        float gate_x_437 = _cvt_f32_106.x;
                        float gate_y_438 = _cvt_f32_106.y;
                        float up_x_439 = _cvt_f32_107.x;
                        float up_y_440 = _cvt_f32_107.y;
                        float _exp_193 = expf(gate_x_437 * -1.0f);
                        float denominator_x_441 = _exp_193 + 1.0f;
                        float _exp_194 = expf(gate_y_438 * -1.0f);
                        float denominator_y_442 = _exp_194 + 1.0f;
                        float hidden_x_443 = gate_x_437 / denominator_x_441 * up_x_439;
                        float hidden_y_444 = gate_y_438 / denominator_y_442 * up_y_440;
                        hidden_packed_264[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_443, hidden_y_444)));
                        float2 _cvt_f32_108 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[22]));
                        float2 _cvt_f32_109 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[22]));
                        float gate_x_445 = _cvt_f32_108.x;
                        float gate_y_446 = _cvt_f32_108.y;
                        float up_x_447 = _cvt_f32_109.x;
                        float up_y_448 = _cvt_f32_109.y;
                        float _exp_195 = expf(gate_x_445 * -1.0f);
                        float denominator_x_449 = _exp_195 + 1.0f;
                        float _exp_196 = expf(gate_y_446 * -1.0f);
                        float denominator_y_450 = _exp_196 + 1.0f;
                        float hidden_x_451 = gate_x_445 / denominator_x_449 * up_x_447;
                        float hidden_y_452 = gate_y_446 / denominator_y_450 * up_y_448;
                        hidden_packed_264[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_451, hidden_y_452)));
                        float2 _cvt_f32_110 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[23]));
                        float2 _cvt_f32_111 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[23]));
                        float gate_x_453 = _cvt_f32_110.x;
                        float gate_y_454 = _cvt_f32_110.y;
                        float up_x_455 = _cvt_f32_111.x;
                        float up_y_456 = _cvt_f32_111.y;
                        float _exp_197 = expf(gate_x_453 * -1.0f);
                        float denominator_x_457 = _exp_197 + 1.0f;
                        float _exp_198 = expf(gate_y_454 * -1.0f);
                        float denominator_y_458 = _exp_198 + 1.0f;
                        float hidden_x_459 = gate_x_453 / denominator_x_457 * up_x_455;
                        float hidden_y_460 = gate_y_454 / denominator_y_458 * up_y_456;
                        hidden_packed_264[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_459, hidden_y_460)));
                        float2 _cvt_f32_112 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[24]));
                        float2 _cvt_f32_113 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[24]));
                        float gate_x_461 = _cvt_f32_112.x;
                        float gate_y_462 = _cvt_f32_112.y;
                        float up_x_463 = _cvt_f32_113.x;
                        float up_y_464 = _cvt_f32_113.y;
                        float _exp_199 = expf(gate_x_461 * -1.0f);
                        float denominator_x_465 = _exp_199 + 1.0f;
                        float _exp_200 = expf(gate_y_462 * -1.0f);
                        float denominator_y_466 = _exp_200 + 1.0f;
                        float hidden_x_467 = gate_x_461 / denominator_x_465 * up_x_463;
                        float hidden_y_468 = gate_y_462 / denominator_y_466 * up_y_464;
                        hidden_packed_264[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_467, hidden_y_468)));
                        float2 _cvt_f32_114 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[25]));
                        float2 _cvt_f32_115 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[25]));
                        float gate_x_469 = _cvt_f32_114.x;
                        float gate_y_470 = _cvt_f32_114.y;
                        float up_x_471 = _cvt_f32_115.x;
                        float up_y_472 = _cvt_f32_115.y;
                        float _exp_201 = expf(gate_x_469 * -1.0f);
                        float denominator_x_473 = _exp_201 + 1.0f;
                        float _exp_202 = expf(gate_y_470 * -1.0f);
                        float denominator_y_474 = _exp_202 + 1.0f;
                        float hidden_x_475 = gate_x_469 / denominator_x_473 * up_x_471;
                        float hidden_y_476 = gate_y_470 / denominator_y_474 * up_y_472;
                        hidden_packed_264[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_475, hidden_y_476)));
                        float2 _cvt_f32_116 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[26]));
                        float2 _cvt_f32_117 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[26]));
                        float gate_x_477 = _cvt_f32_116.x;
                        float gate_y_478 = _cvt_f32_116.y;
                        float up_x_479 = _cvt_f32_117.x;
                        float up_y_480 = _cvt_f32_117.y;
                        float _exp_203 = expf(gate_x_477 * -1.0f);
                        float denominator_x_481 = _exp_203 + 1.0f;
                        float _exp_204 = expf(gate_y_478 * -1.0f);
                        float denominator_y_482 = _exp_204 + 1.0f;
                        float hidden_x_483 = gate_x_477 / denominator_x_481 * up_x_479;
                        float hidden_y_484 = gate_y_478 / denominator_y_482 * up_y_480;
                        hidden_packed_264[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_483, hidden_y_484)));
                        float2 _cvt_f32_118 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[27]));
                        float2 _cvt_f32_119 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[27]));
                        float gate_x_485 = _cvt_f32_118.x;
                        float gate_y_486 = _cvt_f32_118.y;
                        float up_x_487 = _cvt_f32_119.x;
                        float up_y_488 = _cvt_f32_119.y;
                        float _exp_205 = expf(gate_x_485 * -1.0f);
                        float denominator_x_489 = _exp_205 + 1.0f;
                        float _exp_206 = expf(gate_y_486 * -1.0f);
                        float denominator_y_490 = _exp_206 + 1.0f;
                        float hidden_x_491 = gate_x_485 / denominator_x_489 * up_x_487;
                        float hidden_y_492 = gate_y_486 / denominator_y_490 * up_y_488;
                        hidden_packed_264[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_491, hidden_y_492)));
                        float2 _cvt_f32_120 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[28]));
                        float2 _cvt_f32_121 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[28]));
                        float gate_x_493 = _cvt_f32_120.x;
                        float gate_y_494 = _cvt_f32_120.y;
                        float up_x_495 = _cvt_f32_121.x;
                        float up_y_496 = _cvt_f32_121.y;
                        float _exp_207 = expf(gate_x_493 * -1.0f);
                        float denominator_x_497 = _exp_207 + 1.0f;
                        float _exp_208 = expf(gate_y_494 * -1.0f);
                        float denominator_y_498 = _exp_208 + 1.0f;
                        float hidden_x_499 = gate_x_493 / denominator_x_497 * up_x_495;
                        float hidden_y_500 = gate_y_494 / denominator_y_498 * up_y_496;
                        hidden_packed_264[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_499, hidden_y_500)));
                        float2 _cvt_f32_122 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[29]));
                        float2 _cvt_f32_123 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[29]));
                        float gate_x_501 = _cvt_f32_122.x;
                        float gate_y_502 = _cvt_f32_122.y;
                        float up_x_503 = _cvt_f32_123.x;
                        float up_y_504 = _cvt_f32_123.y;
                        float _exp_209 = expf(gate_x_501 * -1.0f);
                        float denominator_x_505 = _exp_209 + 1.0f;
                        float _exp_210 = expf(gate_y_502 * -1.0f);
                        float denominator_y_506 = _exp_210 + 1.0f;
                        float hidden_x_507 = gate_x_501 / denominator_x_505 * up_x_503;
                        float hidden_y_508 = gate_y_502 / denominator_y_506 * up_y_504;
                        hidden_packed_264[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_507, hidden_y_508)));
                        float2 _cvt_f32_124 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[30]));
                        float2 _cvt_f32_125 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[30]));
                        float gate_x_509 = _cvt_f32_124.x;
                        float gate_y_510 = _cvt_f32_124.y;
                        float up_x_511 = _cvt_f32_125.x;
                        float up_y_512 = _cvt_f32_125.y;
                        float _exp_211 = expf(gate_x_509 * -1.0f);
                        float denominator_x_513 = _exp_211 + 1.0f;
                        float _exp_212 = expf(gate_y_510 * -1.0f);
                        float denominator_y_514 = _exp_212 + 1.0f;
                        float hidden_x_515 = gate_x_509 / denominator_x_513 * up_x_511;
                        float hidden_y_516 = gate_y_510 / denominator_y_514 * up_y_512;
                        hidden_packed_264[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_515, hidden_y_516)));
                        float2 _cvt_f32_126 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_262[31]));
                        float2 _cvt_f32_127 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_263[31]));
                        float gate_x_517 = _cvt_f32_126.x;
                        float gate_y_518 = _cvt_f32_126.y;
                        float up_x_519 = _cvt_f32_127.x;
                        float up_y_520 = _cvt_f32_127.y;
                        float _exp_213 = expf(gate_x_517 * -1.0f);
                        float denominator_x_521 = _exp_213 + 1.0f;
                        float _exp_214 = expf(gate_y_518 * -1.0f);
                        float denominator_y_522 = _exp_214 + 1.0f;
                        float hidden_x_523 = gate_x_517 / denominator_x_521 * up_x_519;
                        float hidden_y_524 = gate_y_518 / denominator_y_522 * up_y_520;
                        hidden_packed_264[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_523, hidden_y_524)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_525 = tid / 32;
                        int lane_526 = tid % 32;
                        #pragma unroll
                        for (int half_14 = 0; half_14 < 2; half_14++) {
                            #pragma unroll
                            for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                int row_13 = warp_525 * 32 + half_14 * 16 + lane_526 % 16;
                                int col_41 = col_tile_14 * 16 + lane_526 / 16 * 8;
                                unsigned int address_3_6 = d_smem_addr + (unsigned int)((row_13 * 32 + col_41) * 2);
                                address_3_6 = address_3_6 ^ (address_3_6 & 511) >> 7 << 4;
                                int offset_8 = half_14 * 8 + col_tile_14 * 4;
                                uint32_t _stmatrix_addr_215 = static_cast<uint32_t>(address_3_6);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_215), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_8 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_527 = tid / 32;
                        int lane_528 = tid % 32;
                        #pragma unroll
                        for (int half_15 = 0; half_15 < 2; half_15++) {
                            #pragma unroll
                            for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                int row_14 = warp_527 * 32 + half_15 * 16 + lane_528 % 16;
                                int col_42 = col_tile_15 * 16 + lane_528 / 16 * 8;
                                unsigned int address_3_7 = d_smem_addr + 8192 + (unsigned int)((row_14 * 32 + col_42) * 2);
                                address_3_7 = address_3_7 ^ (address_3_7 & 511) >> 7 << 4;
                                int offset_9 = half_15 * 8 + col_tile_15 * 4;
                                uint32_t _stmatrix_addr_216 = static_cast<uint32_t>(address_3_7);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_216), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_9 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_529 = tid / 32;
                        int lane_530 = tid % 32;
                        #pragma unroll
                        for (int half_16 = 0; half_16 < 2; half_16++) {
                            #pragma unroll
                            for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                int row_15 = warp_529 * 32 + half_16 * 16 + lane_530 % 16;
                                int col_43 = col_tile_16 * 16 + lane_530 / 16 * 8;
                                unsigned int address_3_8 = d_smem_addr + 16384 + (unsigned int)((row_15 * 32 + col_43) * 2);
                                address_3_8 = address_3_8 ^ (address_3_8 & 511) >> 7 << 4;
                                int offset_10 = half_16 * 8 + col_tile_16 * 4;
                                uint32_t _stmatrix_addr_217 = static_cast<uint32_t>(address_3_8);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_217), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_10 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_531 = tid / 32;
                        int lane_532 = tid % 32;
                        #pragma unroll
                        for (int half_17 = 0; half_17 < 2; half_17++) {
                            #pragma unroll
                            for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                int row_16 = warp_531 * 32 + half_17 * 16 + lane_532 % 16;
                                int col_44 = col_tile_17 * 16 + lane_532 / 16 * 8;
                                unsigned int address_3_9 = d_smem_addr + (unsigned int)((row_16 * 32 + col_44) * 2);
                                address_3_9 = address_3_9 ^ (address_3_9 & 511) >> 7 << 4;
                                int offset_11 = 16 + half_17 * 8 + col_tile_17 * 4;
                                uint32_t _stmatrix_addr_218 = static_cast<uint32_t>(address_3_9);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_218), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_11 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_533 = tid / 32;
                        int lane_534 = tid % 32;
                        #pragma unroll
                        for (int half_18 = 0; half_18 < 2; half_18++) {
                            #pragma unroll
                            for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                int row_17 = warp_533 * 32 + half_18 * 16 + lane_534 % 16;
                                int col_45 = col_tile_18 * 16 + lane_534 / 16 * 8;
                                unsigned int address_3_10 = d_smem_addr + 8192 + (unsigned int)((row_17 * 32 + col_45) * 2);
                                address_3_10 = address_3_10 ^ (address_3_10 & 511) >> 7 << 4;
                                int offset_12 = 16 + half_18 * 8 + col_tile_18 * 4;
                                uint32_t _stmatrix_addr_219 = static_cast<uint32_t>(address_3_10);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_219), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_12 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_535 = tid / 32;
                        int lane_536 = tid % 32;
                        #pragma unroll
                        for (int half_19 = 0; half_19 < 2; half_19++) {
                            #pragma unroll
                            for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                int row_18 = warp_535 * 32 + half_19 * 16 + lane_536 % 16;
                                int col_46 = col_tile_19 * 16 + lane_536 / 16 * 8;
                                unsigned int address_3_11 = d_smem_addr + 16384 + (unsigned int)((row_18 * 32 + col_46) * 2);
                                address_3_11 = address_3_11 ^ (address_3_11 & 511) >> 7 << 4;
                                int offset_13 = 16 + half_19 * 8 + col_tile_19 * 4;
                                uint32_t _stmatrix_addr_220 = static_cast<uint32_t>(address_3_11);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_220), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_13 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
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
                        gate_packed_537[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[0], _tmem_load_16[1])));
                        up_packed_538[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[0], _tmem_load_17[1])));
                        gate_packed_537[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[2], _tmem_load_16[3])));
                        up_packed_538[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[2], _tmem_load_17[3])));
                        gate_packed_537[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[4], _tmem_load_16[5])));
                        up_packed_538[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[4], _tmem_load_17[5])));
                        gate_packed_537[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[6], _tmem_load_16[7])));
                        up_packed_538[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[6], _tmem_load_17[7])));
                        gate_packed_537[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[8], _tmem_load_16[9])));
                        up_packed_538[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[8], _tmem_load_17[9])));
                        gate_packed_537[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[10], _tmem_load_16[11])));
                        up_packed_538[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[10], _tmem_load_17[11])));
                        gate_packed_537[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[12], _tmem_load_16[13])));
                        up_packed_538[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[12], _tmem_load_17[13])));
                        gate_packed_537[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[14], _tmem_load_16[15])));
                        up_packed_538[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[14], _tmem_load_17[15])));
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
                        gate_packed_537[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[0], _tmem_load_18[1])));
                        up_packed_538[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[0], _tmem_load_19[1])));
                        gate_packed_537[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[2], _tmem_load_18[3])));
                        up_packed_538[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[2], _tmem_load_19[3])));
                        gate_packed_537[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[4], _tmem_load_18[5])));
                        up_packed_538[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[4], _tmem_load_19[5])));
                        gate_packed_537[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[6], _tmem_load_18[7])));
                        up_packed_538[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[6], _tmem_load_19[7])));
                        gate_packed_537[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[8], _tmem_load_18[9])));
                        up_packed_538[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[8], _tmem_load_19[9])));
                        gate_packed_537[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[10], _tmem_load_18[11])));
                        up_packed_538[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[10], _tmem_load_19[11])));
                        gate_packed_537[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[12], _tmem_load_18[13])));
                        up_packed_538[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[12], _tmem_load_19[13])));
                        gate_packed_537[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[14], _tmem_load_18[15])));
                        up_packed_538[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[14], _tmem_load_19[15])));
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
                        gate_packed_537[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[0], _tmem_load_20[1])));
                        up_packed_538[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[0], _tmem_load_21[1])));
                        gate_packed_537[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[2], _tmem_load_20[3])));
                        up_packed_538[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[2], _tmem_load_21[3])));
                        gate_packed_537[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[4], _tmem_load_20[5])));
                        up_packed_538[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[4], _tmem_load_21[5])));
                        gate_packed_537[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[6], _tmem_load_20[7])));
                        up_packed_538[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[6], _tmem_load_21[7])));
                        gate_packed_537[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[8], _tmem_load_20[9])));
                        up_packed_538[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[8], _tmem_load_21[9])));
                        gate_packed_537[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[10], _tmem_load_20[11])));
                        up_packed_538[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[10], _tmem_load_21[11])));
                        gate_packed_537[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[12], _tmem_load_20[13])));
                        up_packed_538[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[12], _tmem_load_21[13])));
                        gate_packed_537[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[14], _tmem_load_20[15])));
                        up_packed_538[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[14], _tmem_load_21[15])));
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
                        gate_packed_537[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[0], _tmem_load_22[1])));
                        up_packed_538[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[0], _tmem_load_23[1])));
                        gate_packed_537[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[2], _tmem_load_22[3])));
                        up_packed_538[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[2], _tmem_load_23[3])));
                        gate_packed_537[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[4], _tmem_load_22[5])));
                        up_packed_538[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[4], _tmem_load_23[5])));
                        gate_packed_537[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[6], _tmem_load_22[7])));
                        up_packed_538[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[6], _tmem_load_23[7])));
                        gate_packed_537[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[8], _tmem_load_22[9])));
                        up_packed_538[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[8], _tmem_load_23[9])));
                        gate_packed_537[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[10], _tmem_load_22[11])));
                        up_packed_538[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[10], _tmem_load_23[11])));
                        gate_packed_537[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[12], _tmem_load_22[13])));
                        up_packed_538[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[12], _tmem_load_23[13])));
                        gate_packed_537[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[14], _tmem_load_22[15])));
                        up_packed_538[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[14], _tmem_load_23[15])));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_128 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[0]));
                        float2 _cvt_f32_129 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[0]));
                        float gate_x_544 = _cvt_f32_128.x;
                        float gate_y_545 = _cvt_f32_128.y;
                        float up_x_546 = _cvt_f32_129.x;
                        float up_y_547 = _cvt_f32_129.y;
                        float _exp_221 = expf(gate_x_544 * -1.0f);
                        float denominator_x_548 = _exp_221 + 1.0f;
                        float _exp_222 = expf(gate_y_545 * -1.0f);
                        float denominator_y_549 = _exp_222 + 1.0f;
                        float hidden_x_550 = gate_x_544 / denominator_x_548 * up_x_546;
                        float hidden_y_551 = gate_y_545 / denominator_y_549 * up_y_547;
                        hidden_packed_539[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_550, hidden_y_551)));
                        float2 _cvt_f32_130 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[1]));
                        float2 _cvt_f32_131 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[1]));
                        float gate_x_552 = _cvt_f32_130.x;
                        float gate_y_553 = _cvt_f32_130.y;
                        float up_x_554 = _cvt_f32_131.x;
                        float up_y_555 = _cvt_f32_131.y;
                        float _exp_223 = expf(gate_x_552 * -1.0f);
                        float denominator_x_556 = _exp_223 + 1.0f;
                        float _exp_224 = expf(gate_y_553 * -1.0f);
                        float denominator_y_557 = _exp_224 + 1.0f;
                        float hidden_x_558 = gate_x_552 / denominator_x_556 * up_x_554;
                        float hidden_y_559 = gate_y_553 / denominator_y_557 * up_y_555;
                        hidden_packed_539[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_558, hidden_y_559)));
                        float2 _cvt_f32_132 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[2]));
                        float2 _cvt_f32_133 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[2]));
                        float gate_x_560 = _cvt_f32_132.x;
                        float gate_y_561 = _cvt_f32_132.y;
                        float up_x_562 = _cvt_f32_133.x;
                        float up_y_563 = _cvt_f32_133.y;
                        float _exp_225 = expf(gate_x_560 * -1.0f);
                        float denominator_x_564 = _exp_225 + 1.0f;
                        float _exp_226 = expf(gate_y_561 * -1.0f);
                        float denominator_y_565 = _exp_226 + 1.0f;
                        float hidden_x_566 = gate_x_560 / denominator_x_564 * up_x_562;
                        float hidden_y_567 = gate_y_561 / denominator_y_565 * up_y_563;
                        hidden_packed_539[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_566, hidden_y_567)));
                        float2 _cvt_f32_134 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[3]));
                        float2 _cvt_f32_135 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[3]));
                        float gate_x_568 = _cvt_f32_134.x;
                        float gate_y_569 = _cvt_f32_134.y;
                        float up_x_570 = _cvt_f32_135.x;
                        float up_y_571 = _cvt_f32_135.y;
                        float _exp_227 = expf(gate_x_568 * -1.0f);
                        float denominator_x_572 = _exp_227 + 1.0f;
                        float _exp_228 = expf(gate_y_569 * -1.0f);
                        float denominator_y_573 = _exp_228 + 1.0f;
                        float hidden_x_574 = gate_x_568 / denominator_x_572 * up_x_570;
                        float hidden_y_575 = gate_y_569 / denominator_y_573 * up_y_571;
                        hidden_packed_539[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_574, hidden_y_575)));
                        float2 _cvt_f32_136 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[4]));
                        float2 _cvt_f32_137 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[4]));
                        float gate_x_576 = _cvt_f32_136.x;
                        float gate_y_577 = _cvt_f32_136.y;
                        float up_x_578 = _cvt_f32_137.x;
                        float up_y_579 = _cvt_f32_137.y;
                        float _exp_229 = expf(gate_x_576 * -1.0f);
                        float denominator_x_580 = _exp_229 + 1.0f;
                        float _exp_230 = expf(gate_y_577 * -1.0f);
                        float denominator_y_581 = _exp_230 + 1.0f;
                        float hidden_x_582 = gate_x_576 / denominator_x_580 * up_x_578;
                        float hidden_y_583 = gate_y_577 / denominator_y_581 * up_y_579;
                        hidden_packed_539[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_582, hidden_y_583)));
                        float2 _cvt_f32_138 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[5]));
                        float2 _cvt_f32_139 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[5]));
                        float gate_x_584 = _cvt_f32_138.x;
                        float gate_y_585 = _cvt_f32_138.y;
                        float up_x_586 = _cvt_f32_139.x;
                        float up_y_587 = _cvt_f32_139.y;
                        float _exp_231 = expf(gate_x_584 * -1.0f);
                        float denominator_x_588 = _exp_231 + 1.0f;
                        float _exp_232 = expf(gate_y_585 * -1.0f);
                        float denominator_y_589 = _exp_232 + 1.0f;
                        float hidden_x_590 = gate_x_584 / denominator_x_588 * up_x_586;
                        float hidden_y_591 = gate_y_585 / denominator_y_589 * up_y_587;
                        hidden_packed_539[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_590, hidden_y_591)));
                        float2 _cvt_f32_140 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[6]));
                        float2 _cvt_f32_141 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[6]));
                        float gate_x_592 = _cvt_f32_140.x;
                        float gate_y_593 = _cvt_f32_140.y;
                        float up_x_594 = _cvt_f32_141.x;
                        float up_y_595 = _cvt_f32_141.y;
                        float _exp_233 = expf(gate_x_592 * -1.0f);
                        float denominator_x_596 = _exp_233 + 1.0f;
                        float _exp_234 = expf(gate_y_593 * -1.0f);
                        float denominator_y_597 = _exp_234 + 1.0f;
                        float hidden_x_598 = gate_x_592 / denominator_x_596 * up_x_594;
                        float hidden_y_599 = gate_y_593 / denominator_y_597 * up_y_595;
                        hidden_packed_539[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_598, hidden_y_599)));
                        float2 _cvt_f32_142 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[7]));
                        float2 _cvt_f32_143 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[7]));
                        float gate_x_600 = _cvt_f32_142.x;
                        float gate_y_601 = _cvt_f32_142.y;
                        float up_x_602 = _cvt_f32_143.x;
                        float up_y_603 = _cvt_f32_143.y;
                        float _exp_235 = expf(gate_x_600 * -1.0f);
                        float denominator_x_604 = _exp_235 + 1.0f;
                        float _exp_236 = expf(gate_y_601 * -1.0f);
                        float denominator_y_605 = _exp_236 + 1.0f;
                        float hidden_x_606 = gate_x_600 / denominator_x_604 * up_x_602;
                        float hidden_y_607 = gate_y_601 / denominator_y_605 * up_y_603;
                        hidden_packed_539[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_606, hidden_y_607)));
                        float2 _cvt_f32_144 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[8]));
                        float2 _cvt_f32_145 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[8]));
                        float gate_x_608 = _cvt_f32_144.x;
                        float gate_y_609 = _cvt_f32_144.y;
                        float up_x_610 = _cvt_f32_145.x;
                        float up_y_611 = _cvt_f32_145.y;
                        float _exp_237 = expf(gate_x_608 * -1.0f);
                        float denominator_x_612 = _exp_237 + 1.0f;
                        float _exp_238 = expf(gate_y_609 * -1.0f);
                        float denominator_y_613 = _exp_238 + 1.0f;
                        float hidden_x_614 = gate_x_608 / denominator_x_612 * up_x_610;
                        float hidden_y_615 = gate_y_609 / denominator_y_613 * up_y_611;
                        hidden_packed_539[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_614, hidden_y_615)));
                        float2 _cvt_f32_146 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[9]));
                        float2 _cvt_f32_147 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[9]));
                        float gate_x_616 = _cvt_f32_146.x;
                        float gate_y_617 = _cvt_f32_146.y;
                        float up_x_618 = _cvt_f32_147.x;
                        float up_y_619 = _cvt_f32_147.y;
                        float _exp_239 = expf(gate_x_616 * -1.0f);
                        float denominator_x_620 = _exp_239 + 1.0f;
                        float _exp_240 = expf(gate_y_617 * -1.0f);
                        float denominator_y_621 = _exp_240 + 1.0f;
                        float hidden_x_622 = gate_x_616 / denominator_x_620 * up_x_618;
                        float hidden_y_623 = gate_y_617 / denominator_y_621 * up_y_619;
                        hidden_packed_539[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_622, hidden_y_623)));
                        float2 _cvt_f32_148 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[10]));
                        float2 _cvt_f32_149 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[10]));
                        float gate_x_624 = _cvt_f32_148.x;
                        float gate_y_625 = _cvt_f32_148.y;
                        float up_x_626 = _cvt_f32_149.x;
                        float up_y_627 = _cvt_f32_149.y;
                        float _exp_241 = expf(gate_x_624 * -1.0f);
                        float denominator_x_628 = _exp_241 + 1.0f;
                        float _exp_242 = expf(gate_y_625 * -1.0f);
                        float denominator_y_629 = _exp_242 + 1.0f;
                        float hidden_x_630 = gate_x_624 / denominator_x_628 * up_x_626;
                        float hidden_y_631 = gate_y_625 / denominator_y_629 * up_y_627;
                        hidden_packed_539[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_630, hidden_y_631)));
                        float2 _cvt_f32_150 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[11]));
                        float2 _cvt_f32_151 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[11]));
                        float gate_x_632 = _cvt_f32_150.x;
                        float gate_y_633 = _cvt_f32_150.y;
                        float up_x_634 = _cvt_f32_151.x;
                        float up_y_635 = _cvt_f32_151.y;
                        float _exp_243 = expf(gate_x_632 * -1.0f);
                        float denominator_x_636 = _exp_243 + 1.0f;
                        float _exp_244 = expf(gate_y_633 * -1.0f);
                        float denominator_y_637 = _exp_244 + 1.0f;
                        float hidden_x_638 = gate_x_632 / denominator_x_636 * up_x_634;
                        float hidden_y_639 = gate_y_633 / denominator_y_637 * up_y_635;
                        hidden_packed_539[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_638, hidden_y_639)));
                        float2 _cvt_f32_152 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[12]));
                        float2 _cvt_f32_153 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[12]));
                        float gate_x_640 = _cvt_f32_152.x;
                        float gate_y_641 = _cvt_f32_152.y;
                        float up_x_642 = _cvt_f32_153.x;
                        float up_y_643 = _cvt_f32_153.y;
                        float _exp_245 = expf(gate_x_640 * -1.0f);
                        float denominator_x_644 = _exp_245 + 1.0f;
                        float _exp_246 = expf(gate_y_641 * -1.0f);
                        float denominator_y_645 = _exp_246 + 1.0f;
                        float hidden_x_646 = gate_x_640 / denominator_x_644 * up_x_642;
                        float hidden_y_647 = gate_y_641 / denominator_y_645 * up_y_643;
                        hidden_packed_539[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_646, hidden_y_647)));
                        float2 _cvt_f32_154 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[13]));
                        float2 _cvt_f32_155 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[13]));
                        float gate_x_648 = _cvt_f32_154.x;
                        float gate_y_649 = _cvt_f32_154.y;
                        float up_x_650 = _cvt_f32_155.x;
                        float up_y_651 = _cvt_f32_155.y;
                        float _exp_247 = expf(gate_x_648 * -1.0f);
                        float denominator_x_652 = _exp_247 + 1.0f;
                        float _exp_248 = expf(gate_y_649 * -1.0f);
                        float denominator_y_653 = _exp_248 + 1.0f;
                        float hidden_x_654 = gate_x_648 / denominator_x_652 * up_x_650;
                        float hidden_y_655 = gate_y_649 / denominator_y_653 * up_y_651;
                        hidden_packed_539[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_654, hidden_y_655)));
                        float2 _cvt_f32_156 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[14]));
                        float2 _cvt_f32_157 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[14]));
                        float gate_x_656 = _cvt_f32_156.x;
                        float gate_y_657 = _cvt_f32_156.y;
                        float up_x_658 = _cvt_f32_157.x;
                        float up_y_659 = _cvt_f32_157.y;
                        float _exp_249 = expf(gate_x_656 * -1.0f);
                        float denominator_x_660 = _exp_249 + 1.0f;
                        float _exp_250 = expf(gate_y_657 * -1.0f);
                        float denominator_y_661 = _exp_250 + 1.0f;
                        float hidden_x_662 = gate_x_656 / denominator_x_660 * up_x_658;
                        float hidden_y_663 = gate_y_657 / denominator_y_661 * up_y_659;
                        hidden_packed_539[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_662, hidden_y_663)));
                        float2 _cvt_f32_158 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[15]));
                        float2 _cvt_f32_159 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[15]));
                        float gate_x_664 = _cvt_f32_158.x;
                        float gate_y_665 = _cvt_f32_158.y;
                        float up_x_666 = _cvt_f32_159.x;
                        float up_y_667 = _cvt_f32_159.y;
                        float _exp_251 = expf(gate_x_664 * -1.0f);
                        float denominator_x_668 = _exp_251 + 1.0f;
                        float _exp_252 = expf(gate_y_665 * -1.0f);
                        float denominator_y_669 = _exp_252 + 1.0f;
                        float hidden_x_670 = gate_x_664 / denominator_x_668 * up_x_666;
                        float hidden_y_671 = gate_y_665 / denominator_y_669 * up_y_667;
                        hidden_packed_539[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_670, hidden_y_671)));
                        float2 _cvt_f32_160 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[16]));
                        float2 _cvt_f32_161 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[16]));
                        float gate_x_672 = _cvt_f32_160.x;
                        float gate_y_673 = _cvt_f32_160.y;
                        float up_x_674 = _cvt_f32_161.x;
                        float up_y_675 = _cvt_f32_161.y;
                        float _exp_253 = expf(gate_x_672 * -1.0f);
                        float denominator_x_676 = _exp_253 + 1.0f;
                        float _exp_254 = expf(gate_y_673 * -1.0f);
                        float denominator_y_677 = _exp_254 + 1.0f;
                        float hidden_x_678 = gate_x_672 / denominator_x_676 * up_x_674;
                        float hidden_y_679 = gate_y_673 / denominator_y_677 * up_y_675;
                        hidden_packed_539[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_678, hidden_y_679)));
                        float2 _cvt_f32_162 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[17]));
                        float2 _cvt_f32_163 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[17]));
                        float gate_x_680 = _cvt_f32_162.x;
                        float gate_y_681 = _cvt_f32_162.y;
                        float up_x_682 = _cvt_f32_163.x;
                        float up_y_683 = _cvt_f32_163.y;
                        float _exp_255 = expf(gate_x_680 * -1.0f);
                        float denominator_x_684 = _exp_255 + 1.0f;
                        float _exp_256 = expf(gate_y_681 * -1.0f);
                        float denominator_y_685 = _exp_256 + 1.0f;
                        float hidden_x_686 = gate_x_680 / denominator_x_684 * up_x_682;
                        float hidden_y_687 = gate_y_681 / denominator_y_685 * up_y_683;
                        hidden_packed_539[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_686, hidden_y_687)));
                        float2 _cvt_f32_164 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[18]));
                        float2 _cvt_f32_165 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[18]));
                        float gate_x_688 = _cvt_f32_164.x;
                        float gate_y_689 = _cvt_f32_164.y;
                        float up_x_690 = _cvt_f32_165.x;
                        float up_y_691 = _cvt_f32_165.y;
                        float _exp_257 = expf(gate_x_688 * -1.0f);
                        float denominator_x_692 = _exp_257 + 1.0f;
                        float _exp_258 = expf(gate_y_689 * -1.0f);
                        float denominator_y_693 = _exp_258 + 1.0f;
                        float hidden_x_694 = gate_x_688 / denominator_x_692 * up_x_690;
                        float hidden_y_695 = gate_y_689 / denominator_y_693 * up_y_691;
                        hidden_packed_539[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_694, hidden_y_695)));
                        float2 _cvt_f32_166 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[19]));
                        float2 _cvt_f32_167 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[19]));
                        float gate_x_696 = _cvt_f32_166.x;
                        float gate_y_697 = _cvt_f32_166.y;
                        float up_x_698 = _cvt_f32_167.x;
                        float up_y_699 = _cvt_f32_167.y;
                        float _exp_259 = expf(gate_x_696 * -1.0f);
                        float denominator_x_700 = _exp_259 + 1.0f;
                        float _exp_260 = expf(gate_y_697 * -1.0f);
                        float denominator_y_701 = _exp_260 + 1.0f;
                        float hidden_x_702 = gate_x_696 / denominator_x_700 * up_x_698;
                        float hidden_y_703 = gate_y_697 / denominator_y_701 * up_y_699;
                        hidden_packed_539[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_702, hidden_y_703)));
                        float2 _cvt_f32_168 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[20]));
                        float2 _cvt_f32_169 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[20]));
                        float gate_x_704 = _cvt_f32_168.x;
                        float gate_y_705 = _cvt_f32_168.y;
                        float up_x_706 = _cvt_f32_169.x;
                        float up_y_707 = _cvt_f32_169.y;
                        float _exp_261 = expf(gate_x_704 * -1.0f);
                        float denominator_x_708 = _exp_261 + 1.0f;
                        float _exp_262 = expf(gate_y_705 * -1.0f);
                        float denominator_y_709 = _exp_262 + 1.0f;
                        float hidden_x_710 = gate_x_704 / denominator_x_708 * up_x_706;
                        float hidden_y_711 = gate_y_705 / denominator_y_709 * up_y_707;
                        hidden_packed_539[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_710, hidden_y_711)));
                        float2 _cvt_f32_170 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[21]));
                        float2 _cvt_f32_171 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[21]));
                        float gate_x_712 = _cvt_f32_170.x;
                        float gate_y_713 = _cvt_f32_170.y;
                        float up_x_714 = _cvt_f32_171.x;
                        float up_y_715 = _cvt_f32_171.y;
                        float _exp_263 = expf(gate_x_712 * -1.0f);
                        float denominator_x_716 = _exp_263 + 1.0f;
                        float _exp_264 = expf(gate_y_713 * -1.0f);
                        float denominator_y_717 = _exp_264 + 1.0f;
                        float hidden_x_718 = gate_x_712 / denominator_x_716 * up_x_714;
                        float hidden_y_719 = gate_y_713 / denominator_y_717 * up_y_715;
                        hidden_packed_539[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_718, hidden_y_719)));
                        float2 _cvt_f32_172 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[22]));
                        float2 _cvt_f32_173 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[22]));
                        float gate_x_720 = _cvt_f32_172.x;
                        float gate_y_721 = _cvt_f32_172.y;
                        float up_x_722 = _cvt_f32_173.x;
                        float up_y_723 = _cvt_f32_173.y;
                        float _exp_265 = expf(gate_x_720 * -1.0f);
                        float denominator_x_724 = _exp_265 + 1.0f;
                        float _exp_266 = expf(gate_y_721 * -1.0f);
                        float denominator_y_725 = _exp_266 + 1.0f;
                        float hidden_x_726 = gate_x_720 / denominator_x_724 * up_x_722;
                        float hidden_y_727 = gate_y_721 / denominator_y_725 * up_y_723;
                        hidden_packed_539[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_726, hidden_y_727)));
                        float2 _cvt_f32_174 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[23]));
                        float2 _cvt_f32_175 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[23]));
                        float gate_x_728 = _cvt_f32_174.x;
                        float gate_y_729 = _cvt_f32_174.y;
                        float up_x_730 = _cvt_f32_175.x;
                        float up_y_731 = _cvt_f32_175.y;
                        float _exp_267 = expf(gate_x_728 * -1.0f);
                        float denominator_x_732 = _exp_267 + 1.0f;
                        float _exp_268 = expf(gate_y_729 * -1.0f);
                        float denominator_y_733 = _exp_268 + 1.0f;
                        float hidden_x_734 = gate_x_728 / denominator_x_732 * up_x_730;
                        float hidden_y_735 = gate_y_729 / denominator_y_733 * up_y_731;
                        hidden_packed_539[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_734, hidden_y_735)));
                        float2 _cvt_f32_176 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[24]));
                        float2 _cvt_f32_177 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[24]));
                        float gate_x_736 = _cvt_f32_176.x;
                        float gate_y_737 = _cvt_f32_176.y;
                        float up_x_738 = _cvt_f32_177.x;
                        float up_y_739 = _cvt_f32_177.y;
                        float _exp_269 = expf(gate_x_736 * -1.0f);
                        float denominator_x_740 = _exp_269 + 1.0f;
                        float _exp_270 = expf(gate_y_737 * -1.0f);
                        float denominator_y_741 = _exp_270 + 1.0f;
                        float hidden_x_742 = gate_x_736 / denominator_x_740 * up_x_738;
                        float hidden_y_743 = gate_y_737 / denominator_y_741 * up_y_739;
                        hidden_packed_539[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_742, hidden_y_743)));
                        float2 _cvt_f32_178 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[25]));
                        float2 _cvt_f32_179 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[25]));
                        float gate_x_744 = _cvt_f32_178.x;
                        float gate_y_745 = _cvt_f32_178.y;
                        float up_x_746 = _cvt_f32_179.x;
                        float up_y_747 = _cvt_f32_179.y;
                        float _exp_271 = expf(gate_x_744 * -1.0f);
                        float denominator_x_748 = _exp_271 + 1.0f;
                        float _exp_272 = expf(gate_y_745 * -1.0f);
                        float denominator_y_749 = _exp_272 + 1.0f;
                        float hidden_x_750 = gate_x_744 / denominator_x_748 * up_x_746;
                        float hidden_y_751 = gate_y_745 / denominator_y_749 * up_y_747;
                        hidden_packed_539[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_750, hidden_y_751)));
                        float2 _cvt_f32_180 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[26]));
                        float2 _cvt_f32_181 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[26]));
                        float gate_x_752 = _cvt_f32_180.x;
                        float gate_y_753 = _cvt_f32_180.y;
                        float up_x_754 = _cvt_f32_181.x;
                        float up_y_755 = _cvt_f32_181.y;
                        float _exp_273 = expf(gate_x_752 * -1.0f);
                        float denominator_x_756 = _exp_273 + 1.0f;
                        float _exp_274 = expf(gate_y_753 * -1.0f);
                        float denominator_y_757 = _exp_274 + 1.0f;
                        float hidden_x_758 = gate_x_752 / denominator_x_756 * up_x_754;
                        float hidden_y_759 = gate_y_753 / denominator_y_757 * up_y_755;
                        hidden_packed_539[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_758, hidden_y_759)));
                        float2 _cvt_f32_182 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[27]));
                        float2 _cvt_f32_183 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[27]));
                        float gate_x_760 = _cvt_f32_182.x;
                        float gate_y_761 = _cvt_f32_182.y;
                        float up_x_762 = _cvt_f32_183.x;
                        float up_y_763 = _cvt_f32_183.y;
                        float _exp_275 = expf(gate_x_760 * -1.0f);
                        float denominator_x_764 = _exp_275 + 1.0f;
                        float _exp_276 = expf(gate_y_761 * -1.0f);
                        float denominator_y_765 = _exp_276 + 1.0f;
                        float hidden_x_766 = gate_x_760 / denominator_x_764 * up_x_762;
                        float hidden_y_767 = gate_y_761 / denominator_y_765 * up_y_763;
                        hidden_packed_539[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_766, hidden_y_767)));
                        float2 _cvt_f32_184 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[28]));
                        float2 _cvt_f32_185 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[28]));
                        float gate_x_768 = _cvt_f32_184.x;
                        float gate_y_769 = _cvt_f32_184.y;
                        float up_x_770 = _cvt_f32_185.x;
                        float up_y_771 = _cvt_f32_185.y;
                        float _exp_277 = expf(gate_x_768 * -1.0f);
                        float denominator_x_772 = _exp_277 + 1.0f;
                        float _exp_278 = expf(gate_y_769 * -1.0f);
                        float denominator_y_773 = _exp_278 + 1.0f;
                        float hidden_x_774 = gate_x_768 / denominator_x_772 * up_x_770;
                        float hidden_y_775 = gate_y_769 / denominator_y_773 * up_y_771;
                        hidden_packed_539[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_774, hidden_y_775)));
                        float2 _cvt_f32_186 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[29]));
                        float2 _cvt_f32_187 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[29]));
                        float gate_x_776 = _cvt_f32_186.x;
                        float gate_y_777 = _cvt_f32_186.y;
                        float up_x_778 = _cvt_f32_187.x;
                        float up_y_779 = _cvt_f32_187.y;
                        float _exp_279 = expf(gate_x_776 * -1.0f);
                        float denominator_x_780 = _exp_279 + 1.0f;
                        float _exp_280 = expf(gate_y_777 * -1.0f);
                        float denominator_y_781 = _exp_280 + 1.0f;
                        float hidden_x_782 = gate_x_776 / denominator_x_780 * up_x_778;
                        float hidden_y_783 = gate_y_777 / denominator_y_781 * up_y_779;
                        hidden_packed_539[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_782, hidden_y_783)));
                        float2 _cvt_f32_188 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[30]));
                        float2 _cvt_f32_189 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[30]));
                        float gate_x_784 = _cvt_f32_188.x;
                        float gate_y_785 = _cvt_f32_188.y;
                        float up_x_786 = _cvt_f32_189.x;
                        float up_y_787 = _cvt_f32_189.y;
                        float _exp_281 = expf(gate_x_784 * -1.0f);
                        float denominator_x_788 = _exp_281 + 1.0f;
                        float _exp_282 = expf(gate_y_785 * -1.0f);
                        float denominator_y_789 = _exp_282 + 1.0f;
                        float hidden_x_790 = gate_x_784 / denominator_x_788 * up_x_786;
                        float hidden_y_791 = gate_y_785 / denominator_y_789 * up_y_787;
                        hidden_packed_539[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_790, hidden_y_791)));
                        float2 _cvt_f32_190 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_537[31]));
                        float2 _cvt_f32_191 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_538[31]));
                        float gate_x_792 = _cvt_f32_190.x;
                        float gate_y_793 = _cvt_f32_190.y;
                        float up_x_794 = _cvt_f32_191.x;
                        float up_y_795 = _cvt_f32_191.y;
                        float _exp_283 = expf(gate_x_792 * -1.0f);
                        float denominator_x_796 = _exp_283 + 1.0f;
                        float _exp_284 = expf(gate_y_793 * -1.0f);
                        float denominator_y_797 = _exp_284 + 1.0f;
                        float hidden_x_798 = gate_x_792 / denominator_x_796 * up_x_794;
                        float hidden_y_799 = gate_y_793 / denominator_y_797 * up_y_795;
                        hidden_packed_539[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_798, hidden_y_799)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_800 = tid / 32;
                        int lane_801 = tid % 32;
                        #pragma unroll
                        for (int half_20 = 0; half_20 < 2; half_20++) {
                            #pragma unroll
                            for (int col_tile_20 = 0; col_tile_20 < 2; col_tile_20++) {
                                int row_19 = warp_800 * 32 + half_20 * 16 + lane_801 % 16;
                                int col_47 = col_tile_20 * 16 + lane_801 / 16 * 8;
                                unsigned int address_3_12 = d_smem_addr + (unsigned int)((row_19 * 32 + col_47) * 2);
                                address_3_12 = address_3_12 ^ (address_3_12 & 511) >> 7 << 4;
                                int offset_14 = half_20 * 8 + col_tile_20 * 4;
                                uint32_t _stmatrix_addr_285 = static_cast<uint32_t>(address_3_12);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_285), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_14 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_802 = tid / 32;
                        int lane_803 = tid % 32;
                        #pragma unroll
                        for (int half_21 = 0; half_21 < 2; half_21++) {
                            #pragma unroll
                            for (int col_tile_21 = 0; col_tile_21 < 2; col_tile_21++) {
                                int row_20 = warp_802 * 32 + half_21 * 16 + lane_803 % 16;
                                int col_48 = col_tile_21 * 16 + lane_803 / 16 * 8;
                                unsigned int address_3_13 = d_smem_addr + 8192 + (unsigned int)((row_20 * 32 + col_48) * 2);
                                address_3_13 = address_3_13 ^ (address_3_13 & 511) >> 7 << 4;
                                int offset_15 = half_21 * 8 + col_tile_21 * 4;
                                uint32_t _stmatrix_addr_286 = static_cast<uint32_t>(address_3_13);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_286), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_15 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_804 = tid / 32;
                        int lane_805 = tid % 32;
                        #pragma unroll
                        for (int half_22 = 0; half_22 < 2; half_22++) {
                            #pragma unroll
                            for (int col_tile_22 = 0; col_tile_22 < 2; col_tile_22++) {
                                int row_21 = warp_804 * 32 + half_22 * 16 + lane_805 % 16;
                                int col_49 = col_tile_22 * 16 + lane_805 / 16 * 8;
                                unsigned int address_3_14 = d_smem_addr + 16384 + (unsigned int)((row_21 * 32 + col_49) * 2);
                                address_3_14 = address_3_14 ^ (address_3_14 & 511) >> 7 << 4;
                                int offset_16 = half_22 * 8 + col_tile_22 * 4;
                                uint32_t _stmatrix_addr_287 = static_cast<uint32_t>(address_3_14);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_287), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_16 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_806 = tid / 32;
                        int lane_807 = tid % 32;
                        #pragma unroll
                        for (int half_23 = 0; half_23 < 2; half_23++) {
                            #pragma unroll
                            for (int col_tile_23 = 0; col_tile_23 < 2; col_tile_23++) {
                                int row_22 = warp_806 * 32 + half_23 * 16 + lane_807 % 16;
                                int col_50 = col_tile_23 * 16 + lane_807 / 16 * 8;
                                unsigned int address_3_15 = d_smem_addr + (unsigned int)((row_22 * 32 + col_50) * 2);
                                address_3_15 = address_3_15 ^ (address_3_15 & 511) >> 7 << 4;
                                int offset_17 = 16 + half_23 * 8 + col_tile_23 * 4;
                                uint32_t _stmatrix_addr_288 = static_cast<uint32_t>(address_3_15);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_288), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_17 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_808 = tid / 32;
                        int lane_809 = tid % 32;
                        #pragma unroll
                        for (int half_24 = 0; half_24 < 2; half_24++) {
                            #pragma unroll
                            for (int col_tile_24 = 0; col_tile_24 < 2; col_tile_24++) {
                                int row_23 = warp_808 * 32 + half_24 * 16 + lane_809 % 16;
                                int col_51 = col_tile_24 * 16 + lane_809 / 16 * 8;
                                unsigned int address_3_16 = d_smem_addr + 8192 + (unsigned int)((row_23 * 32 + col_51) * 2);
                                address_3_16 = address_3_16 ^ (address_3_16 & 511) >> 7 << 4;
                                int offset_18 = 16 + half_24 * 8 + col_tile_24 * 4;
                                uint32_t _stmatrix_addr_289 = static_cast<uint32_t>(address_3_16);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_289), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_18 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_810 = tid / 32;
                        int lane_811 = tid % 32;
                        #pragma unroll
                        for (int half_25 = 0; half_25 < 2; half_25++) {
                            #pragma unroll
                            for (int col_tile_25 = 0; col_tile_25 < 2; col_tile_25++) {
                                int row_24 = warp_810 * 32 + half_25 * 16 + lane_811 % 16;
                                int col_52 = col_tile_25 * 16 + lane_811 / 16 * 8;
                                unsigned int address_3_17 = d_smem_addr + 16384 + (unsigned int)((row_24 * 32 + col_52) * 2);
                                address_3_17 = address_3_17 ^ (address_3_17 & 511) >> 7 << 4;
                                int offset_19 = 16 + half_25 * 8 + col_tile_25 * 4;
                                uint32_t _stmatrix_addr_290 = static_cast<uint32_t>(address_3_17);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_290), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_19])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_19 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_19 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_19 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
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
                        gate_packed_812[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[0], _tmem_load_24[1])));
                        up_packed_813[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[0], _tmem_load_25[1])));
                        gate_packed_812[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[2], _tmem_load_24[3])));
                        up_packed_813[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[2], _tmem_load_25[3])));
                        gate_packed_812[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[4], _tmem_load_24[5])));
                        up_packed_813[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[4], _tmem_load_25[5])));
                        gate_packed_812[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[6], _tmem_load_24[7])));
                        up_packed_813[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[6], _tmem_load_25[7])));
                        gate_packed_812[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[8], _tmem_load_24[9])));
                        up_packed_813[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[8], _tmem_load_25[9])));
                        gate_packed_812[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[10], _tmem_load_24[11])));
                        up_packed_813[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[10], _tmem_load_25[11])));
                        gate_packed_812[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[12], _tmem_load_24[13])));
                        up_packed_813[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[12], _tmem_load_25[13])));
                        gate_packed_812[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[14], _tmem_load_24[15])));
                        up_packed_813[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[14], _tmem_load_25[15])));
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
                        gate_packed_812[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[0], _tmem_load_26[1])));
                        up_packed_813[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[0], _tmem_load_27[1])));
                        gate_packed_812[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[2], _tmem_load_26[3])));
                        up_packed_813[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[2], _tmem_load_27[3])));
                        gate_packed_812[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[4], _tmem_load_26[5])));
                        up_packed_813[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[4], _tmem_load_27[5])));
                        gate_packed_812[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[6], _tmem_load_26[7])));
                        up_packed_813[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[6], _tmem_load_27[7])));
                        gate_packed_812[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[8], _tmem_load_26[9])));
                        up_packed_813[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[8], _tmem_load_27[9])));
                        gate_packed_812[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[10], _tmem_load_26[11])));
                        up_packed_813[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[10], _tmem_load_27[11])));
                        gate_packed_812[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[12], _tmem_load_26[13])));
                        up_packed_813[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[12], _tmem_load_27[13])));
                        gate_packed_812[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[14], _tmem_load_26[15])));
                        up_packed_813[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[14], _tmem_load_27[15])));
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
                        gate_packed_812[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[0], _tmem_load_28[1])));
                        up_packed_813[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[0], _tmem_load_29[1])));
                        gate_packed_812[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[2], _tmem_load_28[3])));
                        up_packed_813[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[2], _tmem_load_29[3])));
                        gate_packed_812[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[4], _tmem_load_28[5])));
                        up_packed_813[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[4], _tmem_load_29[5])));
                        gate_packed_812[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[6], _tmem_load_28[7])));
                        up_packed_813[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[6], _tmem_load_29[7])));
                        gate_packed_812[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[8], _tmem_load_28[9])));
                        up_packed_813[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[8], _tmem_load_29[9])));
                        gate_packed_812[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[10], _tmem_load_28[11])));
                        up_packed_813[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[10], _tmem_load_29[11])));
                        gate_packed_812[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[12], _tmem_load_28[13])));
                        up_packed_813[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[12], _tmem_load_29[13])));
                        gate_packed_812[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[14], _tmem_load_28[15])));
                        up_packed_813[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[14], _tmem_load_29[15])));
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
                        gate_packed_812[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[0], _tmem_load_30[1])));
                        up_packed_813[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[0], _tmem_load_31[1])));
                        gate_packed_812[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[2], _tmem_load_30[3])));
                        up_packed_813[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[2], _tmem_load_31[3])));
                        gate_packed_812[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[4], _tmem_load_30[5])));
                        up_packed_813[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[4], _tmem_load_31[5])));
                        gate_packed_812[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[6], _tmem_load_30[7])));
                        up_packed_813[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[6], _tmem_load_31[7])));
                        gate_packed_812[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[8], _tmem_load_30[9])));
                        up_packed_813[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[8], _tmem_load_31[9])));
                        gate_packed_812[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[10], _tmem_load_30[11])));
                        up_packed_813[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[10], _tmem_load_31[11])));
                        gate_packed_812[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[12], _tmem_load_30[13])));
                        up_packed_813[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[12], _tmem_load_31[13])));
                        gate_packed_812[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[14], _tmem_load_30[15])));
                        up_packed_813[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[14], _tmem_load_31[15])));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((prefix_released_addr) & 0xFEFFFFFF) : "memory");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((output_finished_addr + 8) & 0xFEFFFFFF) : "memory");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((prefix_released_addr + 8) & 0xFEFFFFFF) : "memory");
                        }
                        float2 _cvt_f32_192 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[0]));
                        float2 _cvt_f32_193 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[0]));
                        float gate_x_819 = _cvt_f32_192.x;
                        float gate_y_820 = _cvt_f32_192.y;
                        float up_x_821 = _cvt_f32_193.x;
                        float up_y_822 = _cvt_f32_193.y;
                        float _exp_291 = expf(gate_x_819 * -1.0f);
                        float denominator_x_823 = _exp_291 + 1.0f;
                        float _exp_292 = expf(gate_y_820 * -1.0f);
                        float denominator_y_824 = _exp_292 + 1.0f;
                        float hidden_x_825 = gate_x_819 / denominator_x_823 * up_x_821;
                        float hidden_y_826 = gate_y_820 / denominator_y_824 * up_y_822;
                        hidden_packed_814[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_825, hidden_y_826)));
                        float2 _cvt_f32_194 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[1]));
                        float2 _cvt_f32_195 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[1]));
                        float gate_x_827 = _cvt_f32_194.x;
                        float gate_y_828 = _cvt_f32_194.y;
                        float up_x_829 = _cvt_f32_195.x;
                        float up_y_830 = _cvt_f32_195.y;
                        float _exp_293 = expf(gate_x_827 * -1.0f);
                        float denominator_x_831 = _exp_293 + 1.0f;
                        float _exp_294 = expf(gate_y_828 * -1.0f);
                        float denominator_y_832 = _exp_294 + 1.0f;
                        float hidden_x_833 = gate_x_827 / denominator_x_831 * up_x_829;
                        float hidden_y_834 = gate_y_828 / denominator_y_832 * up_y_830;
                        hidden_packed_814[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_833, hidden_y_834)));
                        float2 _cvt_f32_196 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[2]));
                        float2 _cvt_f32_197 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[2]));
                        float gate_x_835 = _cvt_f32_196.x;
                        float gate_y_836 = _cvt_f32_196.y;
                        float up_x_837 = _cvt_f32_197.x;
                        float up_y_838 = _cvt_f32_197.y;
                        float _exp_295 = expf(gate_x_835 * -1.0f);
                        float denominator_x_839 = _exp_295 + 1.0f;
                        float _exp_296 = expf(gate_y_836 * -1.0f);
                        float denominator_y_840 = _exp_296 + 1.0f;
                        float hidden_x_841 = gate_x_835 / denominator_x_839 * up_x_837;
                        float hidden_y_842 = gate_y_836 / denominator_y_840 * up_y_838;
                        hidden_packed_814[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_841, hidden_y_842)));
                        float2 _cvt_f32_198 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[3]));
                        float2 _cvt_f32_199 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[3]));
                        float gate_x_843 = _cvt_f32_198.x;
                        float gate_y_844 = _cvt_f32_198.y;
                        float up_x_845 = _cvt_f32_199.x;
                        float up_y_846 = _cvt_f32_199.y;
                        float _exp_297 = expf(gate_x_843 * -1.0f);
                        float denominator_x_847 = _exp_297 + 1.0f;
                        float _exp_298 = expf(gate_y_844 * -1.0f);
                        float denominator_y_848 = _exp_298 + 1.0f;
                        float hidden_x_849 = gate_x_843 / denominator_x_847 * up_x_845;
                        float hidden_y_850 = gate_y_844 / denominator_y_848 * up_y_846;
                        hidden_packed_814[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_849, hidden_y_850)));
                        float2 _cvt_f32_200 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[4]));
                        float2 _cvt_f32_201 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[4]));
                        float gate_x_851 = _cvt_f32_200.x;
                        float gate_y_852 = _cvt_f32_200.y;
                        float up_x_853 = _cvt_f32_201.x;
                        float up_y_854 = _cvt_f32_201.y;
                        float _exp_299 = expf(gate_x_851 * -1.0f);
                        float denominator_x_855 = _exp_299 + 1.0f;
                        float _exp_300 = expf(gate_y_852 * -1.0f);
                        float denominator_y_856 = _exp_300 + 1.0f;
                        float hidden_x_857 = gate_x_851 / denominator_x_855 * up_x_853;
                        float hidden_y_858 = gate_y_852 / denominator_y_856 * up_y_854;
                        hidden_packed_814[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_857, hidden_y_858)));
                        float2 _cvt_f32_202 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[5]));
                        float2 _cvt_f32_203 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[5]));
                        float gate_x_859 = _cvt_f32_202.x;
                        float gate_y_860 = _cvt_f32_202.y;
                        float up_x_861 = _cvt_f32_203.x;
                        float up_y_862 = _cvt_f32_203.y;
                        float _exp_301 = expf(gate_x_859 * -1.0f);
                        float denominator_x_863 = _exp_301 + 1.0f;
                        float _exp_302 = expf(gate_y_860 * -1.0f);
                        float denominator_y_864 = _exp_302 + 1.0f;
                        float hidden_x_865 = gate_x_859 / denominator_x_863 * up_x_861;
                        float hidden_y_866 = gate_y_860 / denominator_y_864 * up_y_862;
                        hidden_packed_814[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_865, hidden_y_866)));
                        float2 _cvt_f32_204 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[6]));
                        float2 _cvt_f32_205 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[6]));
                        float gate_x_867 = _cvt_f32_204.x;
                        float gate_y_868 = _cvt_f32_204.y;
                        float up_x_869 = _cvt_f32_205.x;
                        float up_y_870 = _cvt_f32_205.y;
                        float _exp_303 = expf(gate_x_867 * -1.0f);
                        float denominator_x_871 = _exp_303 + 1.0f;
                        float _exp_304 = expf(gate_y_868 * -1.0f);
                        float denominator_y_872 = _exp_304 + 1.0f;
                        float hidden_x_873 = gate_x_867 / denominator_x_871 * up_x_869;
                        float hidden_y_874 = gate_y_868 / denominator_y_872 * up_y_870;
                        hidden_packed_814[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_873, hidden_y_874)));
                        float2 _cvt_f32_206 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[7]));
                        float2 _cvt_f32_207 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[7]));
                        float gate_x_875 = _cvt_f32_206.x;
                        float gate_y_876 = _cvt_f32_206.y;
                        float up_x_877 = _cvt_f32_207.x;
                        float up_y_878 = _cvt_f32_207.y;
                        float _exp_305 = expf(gate_x_875 * -1.0f);
                        float denominator_x_879 = _exp_305 + 1.0f;
                        float _exp_306 = expf(gate_y_876 * -1.0f);
                        float denominator_y_880 = _exp_306 + 1.0f;
                        float hidden_x_881 = gate_x_875 / denominator_x_879 * up_x_877;
                        float hidden_y_882 = gate_y_876 / denominator_y_880 * up_y_878;
                        hidden_packed_814[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_881, hidden_y_882)));
                        float2 _cvt_f32_208 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[8]));
                        float2 _cvt_f32_209 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[8]));
                        float gate_x_883 = _cvt_f32_208.x;
                        float gate_y_884 = _cvt_f32_208.y;
                        float up_x_885 = _cvt_f32_209.x;
                        float up_y_886 = _cvt_f32_209.y;
                        float _exp_307 = expf(gate_x_883 * -1.0f);
                        float denominator_x_887 = _exp_307 + 1.0f;
                        float _exp_308 = expf(gate_y_884 * -1.0f);
                        float denominator_y_888 = _exp_308 + 1.0f;
                        float hidden_x_889 = gate_x_883 / denominator_x_887 * up_x_885;
                        float hidden_y_890 = gate_y_884 / denominator_y_888 * up_y_886;
                        hidden_packed_814[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_889, hidden_y_890)));
                        float2 _cvt_f32_210 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[9]));
                        float2 _cvt_f32_211 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[9]));
                        float gate_x_891 = _cvt_f32_210.x;
                        float gate_y_892 = _cvt_f32_210.y;
                        float up_x_893 = _cvt_f32_211.x;
                        float up_y_894 = _cvt_f32_211.y;
                        float _exp_309 = expf(gate_x_891 * -1.0f);
                        float denominator_x_895 = _exp_309 + 1.0f;
                        float _exp_310 = expf(gate_y_892 * -1.0f);
                        float denominator_y_896 = _exp_310 + 1.0f;
                        float hidden_x_897 = gate_x_891 / denominator_x_895 * up_x_893;
                        float hidden_y_898 = gate_y_892 / denominator_y_896 * up_y_894;
                        hidden_packed_814[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_897, hidden_y_898)));
                        float2 _cvt_f32_212 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[10]));
                        float2 _cvt_f32_213 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[10]));
                        float gate_x_899 = _cvt_f32_212.x;
                        float gate_y_900 = _cvt_f32_212.y;
                        float up_x_901 = _cvt_f32_213.x;
                        float up_y_902 = _cvt_f32_213.y;
                        float _exp_311 = expf(gate_x_899 * -1.0f);
                        float denominator_x_903 = _exp_311 + 1.0f;
                        float _exp_312 = expf(gate_y_900 * -1.0f);
                        float denominator_y_904 = _exp_312 + 1.0f;
                        float hidden_x_905 = gate_x_899 / denominator_x_903 * up_x_901;
                        float hidden_y_906 = gate_y_900 / denominator_y_904 * up_y_902;
                        hidden_packed_814[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_905, hidden_y_906)));
                        float2 _cvt_f32_214 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[11]));
                        float2 _cvt_f32_215 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[11]));
                        float gate_x_907 = _cvt_f32_214.x;
                        float gate_y_908 = _cvt_f32_214.y;
                        float up_x_909 = _cvt_f32_215.x;
                        float up_y_910 = _cvt_f32_215.y;
                        float _exp_313 = expf(gate_x_907 * -1.0f);
                        float denominator_x_911 = _exp_313 + 1.0f;
                        float _exp_314 = expf(gate_y_908 * -1.0f);
                        float denominator_y_912 = _exp_314 + 1.0f;
                        float hidden_x_913 = gate_x_907 / denominator_x_911 * up_x_909;
                        float hidden_y_914 = gate_y_908 / denominator_y_912 * up_y_910;
                        hidden_packed_814[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_913, hidden_y_914)));
                        float2 _cvt_f32_216 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[12]));
                        float2 _cvt_f32_217 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[12]));
                        float gate_x_915 = _cvt_f32_216.x;
                        float gate_y_916 = _cvt_f32_216.y;
                        float up_x_917 = _cvt_f32_217.x;
                        float up_y_918 = _cvt_f32_217.y;
                        float _exp_315 = expf(gate_x_915 * -1.0f);
                        float denominator_x_919 = _exp_315 + 1.0f;
                        float _exp_316 = expf(gate_y_916 * -1.0f);
                        float denominator_y_920 = _exp_316 + 1.0f;
                        float hidden_x_921 = gate_x_915 / denominator_x_919 * up_x_917;
                        float hidden_y_922 = gate_y_916 / denominator_y_920 * up_y_918;
                        hidden_packed_814[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_921, hidden_y_922)));
                        float2 _cvt_f32_218 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[13]));
                        float2 _cvt_f32_219 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[13]));
                        float gate_x_923 = _cvt_f32_218.x;
                        float gate_y_924 = _cvt_f32_218.y;
                        float up_x_925 = _cvt_f32_219.x;
                        float up_y_926 = _cvt_f32_219.y;
                        float _exp_317 = expf(gate_x_923 * -1.0f);
                        float denominator_x_927 = _exp_317 + 1.0f;
                        float _exp_318 = expf(gate_y_924 * -1.0f);
                        float denominator_y_928 = _exp_318 + 1.0f;
                        float hidden_x_929 = gate_x_923 / denominator_x_927 * up_x_925;
                        float hidden_y_930 = gate_y_924 / denominator_y_928 * up_y_926;
                        hidden_packed_814[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_929, hidden_y_930)));
                        float2 _cvt_f32_220 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[14]));
                        float2 _cvt_f32_221 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[14]));
                        float gate_x_931 = _cvt_f32_220.x;
                        float gate_y_932 = _cvt_f32_220.y;
                        float up_x_933 = _cvt_f32_221.x;
                        float up_y_934 = _cvt_f32_221.y;
                        float _exp_319 = expf(gate_x_931 * -1.0f);
                        float denominator_x_935 = _exp_319 + 1.0f;
                        float _exp_320 = expf(gate_y_932 * -1.0f);
                        float denominator_y_936 = _exp_320 + 1.0f;
                        float hidden_x_937 = gate_x_931 / denominator_x_935 * up_x_933;
                        float hidden_y_938 = gate_y_932 / denominator_y_936 * up_y_934;
                        hidden_packed_814[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_937, hidden_y_938)));
                        float2 _cvt_f32_222 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[15]));
                        float2 _cvt_f32_223 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[15]));
                        float gate_x_939 = _cvt_f32_222.x;
                        float gate_y_940 = _cvt_f32_222.y;
                        float up_x_941 = _cvt_f32_223.x;
                        float up_y_942 = _cvt_f32_223.y;
                        float _exp_321 = expf(gate_x_939 * -1.0f);
                        float denominator_x_943 = _exp_321 + 1.0f;
                        float _exp_322 = expf(gate_y_940 * -1.0f);
                        float denominator_y_944 = _exp_322 + 1.0f;
                        float hidden_x_945 = gate_x_939 / denominator_x_943 * up_x_941;
                        float hidden_y_946 = gate_y_940 / denominator_y_944 * up_y_942;
                        hidden_packed_814[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_945, hidden_y_946)));
                        float2 _cvt_f32_224 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[16]));
                        float2 _cvt_f32_225 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[16]));
                        float gate_x_947 = _cvt_f32_224.x;
                        float gate_y_948 = _cvt_f32_224.y;
                        float up_x_949 = _cvt_f32_225.x;
                        float up_y_950 = _cvt_f32_225.y;
                        float _exp_323 = expf(gate_x_947 * -1.0f);
                        float denominator_x_951 = _exp_323 + 1.0f;
                        float _exp_324 = expf(gate_y_948 * -1.0f);
                        float denominator_y_952 = _exp_324 + 1.0f;
                        float hidden_x_953 = gate_x_947 / denominator_x_951 * up_x_949;
                        float hidden_y_954 = gate_y_948 / denominator_y_952 * up_y_950;
                        hidden_packed_814[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_953, hidden_y_954)));
                        float2 _cvt_f32_226 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[17]));
                        float2 _cvt_f32_227 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[17]));
                        float gate_x_955 = _cvt_f32_226.x;
                        float gate_y_956 = _cvt_f32_226.y;
                        float up_x_957 = _cvt_f32_227.x;
                        float up_y_958 = _cvt_f32_227.y;
                        float _exp_325 = expf(gate_x_955 * -1.0f);
                        float denominator_x_959 = _exp_325 + 1.0f;
                        float _exp_326 = expf(gate_y_956 * -1.0f);
                        float denominator_y_960 = _exp_326 + 1.0f;
                        float hidden_x_961 = gate_x_955 / denominator_x_959 * up_x_957;
                        float hidden_y_962 = gate_y_956 / denominator_y_960 * up_y_958;
                        hidden_packed_814[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_961, hidden_y_962)));
                        float2 _cvt_f32_228 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[18]));
                        float2 _cvt_f32_229 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[18]));
                        float gate_x_963 = _cvt_f32_228.x;
                        float gate_y_964 = _cvt_f32_228.y;
                        float up_x_965 = _cvt_f32_229.x;
                        float up_y_966 = _cvt_f32_229.y;
                        float _exp_327 = expf(gate_x_963 * -1.0f);
                        float denominator_x_967 = _exp_327 + 1.0f;
                        float _exp_328 = expf(gate_y_964 * -1.0f);
                        float denominator_y_968 = _exp_328 + 1.0f;
                        float hidden_x_969 = gate_x_963 / denominator_x_967 * up_x_965;
                        float hidden_y_970 = gate_y_964 / denominator_y_968 * up_y_966;
                        hidden_packed_814[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_969, hidden_y_970)));
                        float2 _cvt_f32_230 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[19]));
                        float2 _cvt_f32_231 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[19]));
                        float gate_x_971 = _cvt_f32_230.x;
                        float gate_y_972 = _cvt_f32_230.y;
                        float up_x_973 = _cvt_f32_231.x;
                        float up_y_974 = _cvt_f32_231.y;
                        float _exp_329 = expf(gate_x_971 * -1.0f);
                        float denominator_x_975 = _exp_329 + 1.0f;
                        float _exp_330 = expf(gate_y_972 * -1.0f);
                        float denominator_y_976 = _exp_330 + 1.0f;
                        float hidden_x_977 = gate_x_971 / denominator_x_975 * up_x_973;
                        float hidden_y_978 = gate_y_972 / denominator_y_976 * up_y_974;
                        hidden_packed_814[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_977, hidden_y_978)));
                        float2 _cvt_f32_232 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[20]));
                        float2 _cvt_f32_233 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[20]));
                        float gate_x_979 = _cvt_f32_232.x;
                        float gate_y_980 = _cvt_f32_232.y;
                        float up_x_981 = _cvt_f32_233.x;
                        float up_y_982 = _cvt_f32_233.y;
                        float _exp_331 = expf(gate_x_979 * -1.0f);
                        float denominator_x_983 = _exp_331 + 1.0f;
                        float _exp_332 = expf(gate_y_980 * -1.0f);
                        float denominator_y_984 = _exp_332 + 1.0f;
                        float hidden_x_985 = gate_x_979 / denominator_x_983 * up_x_981;
                        float hidden_y_986 = gate_y_980 / denominator_y_984 * up_y_982;
                        hidden_packed_814[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_985, hidden_y_986)));
                        float2 _cvt_f32_234 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[21]));
                        float2 _cvt_f32_235 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[21]));
                        float gate_x_987 = _cvt_f32_234.x;
                        float gate_y_988 = _cvt_f32_234.y;
                        float up_x_989 = _cvt_f32_235.x;
                        float up_y_990 = _cvt_f32_235.y;
                        float _exp_333 = expf(gate_x_987 * -1.0f);
                        float denominator_x_991 = _exp_333 + 1.0f;
                        float _exp_334 = expf(gate_y_988 * -1.0f);
                        float denominator_y_992 = _exp_334 + 1.0f;
                        float hidden_x_993 = gate_x_987 / denominator_x_991 * up_x_989;
                        float hidden_y_994 = gate_y_988 / denominator_y_992 * up_y_990;
                        hidden_packed_814[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_993, hidden_y_994)));
                        float2 _cvt_f32_236 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[22]));
                        float2 _cvt_f32_237 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[22]));
                        float gate_x_995 = _cvt_f32_236.x;
                        float gate_y_996 = _cvt_f32_236.y;
                        float up_x_997 = _cvt_f32_237.x;
                        float up_y_998 = _cvt_f32_237.y;
                        float _exp_335 = expf(gate_x_995 * -1.0f);
                        float denominator_x_999 = _exp_335 + 1.0f;
                        float _exp_336 = expf(gate_y_996 * -1.0f);
                        float denominator_y_1000 = _exp_336 + 1.0f;
                        float hidden_x_1001 = gate_x_995 / denominator_x_999 * up_x_997;
                        float hidden_y_1002 = gate_y_996 / denominator_y_1000 * up_y_998;
                        hidden_packed_814[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1001, hidden_y_1002)));
                        float2 _cvt_f32_238 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[23]));
                        float2 _cvt_f32_239 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[23]));
                        float gate_x_1003 = _cvt_f32_238.x;
                        float gate_y_1004 = _cvt_f32_238.y;
                        float up_x_1005 = _cvt_f32_239.x;
                        float up_y_1006 = _cvt_f32_239.y;
                        float _exp_337 = expf(gate_x_1003 * -1.0f);
                        float denominator_x_1007 = _exp_337 + 1.0f;
                        float _exp_338 = expf(gate_y_1004 * -1.0f);
                        float denominator_y_1008 = _exp_338 + 1.0f;
                        float hidden_x_1009 = gate_x_1003 / denominator_x_1007 * up_x_1005;
                        float hidden_y_1010 = gate_y_1004 / denominator_y_1008 * up_y_1006;
                        hidden_packed_814[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1009, hidden_y_1010)));
                        float2 _cvt_f32_240 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[24]));
                        float2 _cvt_f32_241 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[24]));
                        float gate_x_1011 = _cvt_f32_240.x;
                        float gate_y_1012 = _cvt_f32_240.y;
                        float up_x_1013 = _cvt_f32_241.x;
                        float up_y_1014 = _cvt_f32_241.y;
                        float _exp_339 = expf(gate_x_1011 * -1.0f);
                        float denominator_x_1015 = _exp_339 + 1.0f;
                        float _exp_340 = expf(gate_y_1012 * -1.0f);
                        float denominator_y_1016 = _exp_340 + 1.0f;
                        float hidden_x_1017 = gate_x_1011 / denominator_x_1015 * up_x_1013;
                        float hidden_y_1018 = gate_y_1012 / denominator_y_1016 * up_y_1014;
                        hidden_packed_814[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1017, hidden_y_1018)));
                        float2 _cvt_f32_242 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[25]));
                        float2 _cvt_f32_243 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[25]));
                        float gate_x_1019 = _cvt_f32_242.x;
                        float gate_y_1020 = _cvt_f32_242.y;
                        float up_x_1021 = _cvt_f32_243.x;
                        float up_y_1022 = _cvt_f32_243.y;
                        float _exp_341 = expf(gate_x_1019 * -1.0f);
                        float denominator_x_1023 = _exp_341 + 1.0f;
                        float _exp_342 = expf(gate_y_1020 * -1.0f);
                        float denominator_y_1024 = _exp_342 + 1.0f;
                        float hidden_x_1025 = gate_x_1019 / denominator_x_1023 * up_x_1021;
                        float hidden_y_1026 = gate_y_1020 / denominator_y_1024 * up_y_1022;
                        hidden_packed_814[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1025, hidden_y_1026)));
                        float2 _cvt_f32_244 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[26]));
                        float2 _cvt_f32_245 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[26]));
                        float gate_x_1027 = _cvt_f32_244.x;
                        float gate_y_1028 = _cvt_f32_244.y;
                        float up_x_1029 = _cvt_f32_245.x;
                        float up_y_1030 = _cvt_f32_245.y;
                        float _exp_343 = expf(gate_x_1027 * -1.0f);
                        float denominator_x_1031 = _exp_343 + 1.0f;
                        float _exp_344 = expf(gate_y_1028 * -1.0f);
                        float denominator_y_1032 = _exp_344 + 1.0f;
                        float hidden_x_1033 = gate_x_1027 / denominator_x_1031 * up_x_1029;
                        float hidden_y_1034 = gate_y_1028 / denominator_y_1032 * up_y_1030;
                        hidden_packed_814[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1033, hidden_y_1034)));
                        float2 _cvt_f32_246 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[27]));
                        float2 _cvt_f32_247 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[27]));
                        float gate_x_1035 = _cvt_f32_246.x;
                        float gate_y_1036 = _cvt_f32_246.y;
                        float up_x_1037 = _cvt_f32_247.x;
                        float up_y_1038 = _cvt_f32_247.y;
                        float _exp_345 = expf(gate_x_1035 * -1.0f);
                        float denominator_x_1039 = _exp_345 + 1.0f;
                        float _exp_346 = expf(gate_y_1036 * -1.0f);
                        float denominator_y_1040 = _exp_346 + 1.0f;
                        float hidden_x_1041 = gate_x_1035 / denominator_x_1039 * up_x_1037;
                        float hidden_y_1042 = gate_y_1036 / denominator_y_1040 * up_y_1038;
                        hidden_packed_814[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1041, hidden_y_1042)));
                        float2 _cvt_f32_248 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[28]));
                        float2 _cvt_f32_249 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[28]));
                        float gate_x_1043 = _cvt_f32_248.x;
                        float gate_y_1044 = _cvt_f32_248.y;
                        float up_x_1045 = _cvt_f32_249.x;
                        float up_y_1046 = _cvt_f32_249.y;
                        float _exp_347 = expf(gate_x_1043 * -1.0f);
                        float denominator_x_1047 = _exp_347 + 1.0f;
                        float _exp_348 = expf(gate_y_1044 * -1.0f);
                        float denominator_y_1048 = _exp_348 + 1.0f;
                        float hidden_x_1049 = gate_x_1043 / denominator_x_1047 * up_x_1045;
                        float hidden_y_1050 = gate_y_1044 / denominator_y_1048 * up_y_1046;
                        hidden_packed_814[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1049, hidden_y_1050)));
                        float2 _cvt_f32_250 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[29]));
                        float2 _cvt_f32_251 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[29]));
                        float gate_x_1051 = _cvt_f32_250.x;
                        float gate_y_1052 = _cvt_f32_250.y;
                        float up_x_1053 = _cvt_f32_251.x;
                        float up_y_1054 = _cvt_f32_251.y;
                        float _exp_349 = expf(gate_x_1051 * -1.0f);
                        float denominator_x_1055 = _exp_349 + 1.0f;
                        float _exp_350 = expf(gate_y_1052 * -1.0f);
                        float denominator_y_1056 = _exp_350 + 1.0f;
                        float hidden_x_1057 = gate_x_1051 / denominator_x_1055 * up_x_1053;
                        float hidden_y_1058 = gate_y_1052 / denominator_y_1056 * up_y_1054;
                        hidden_packed_814[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1057, hidden_y_1058)));
                        float2 _cvt_f32_252 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[30]));
                        float2 _cvt_f32_253 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[30]));
                        float gate_x_1059 = _cvt_f32_252.x;
                        float gate_y_1060 = _cvt_f32_252.y;
                        float up_x_1061 = _cvt_f32_253.x;
                        float up_y_1062 = _cvt_f32_253.y;
                        float _exp_351 = expf(gate_x_1059 * -1.0f);
                        float denominator_x_1063 = _exp_351 + 1.0f;
                        float _exp_352 = expf(gate_y_1060 * -1.0f);
                        float denominator_y_1064 = _exp_352 + 1.0f;
                        float hidden_x_1065 = gate_x_1059 / denominator_x_1063 * up_x_1061;
                        float hidden_y_1066 = gate_y_1060 / denominator_y_1064 * up_y_1062;
                        hidden_packed_814[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1065, hidden_y_1066)));
                        float2 _cvt_f32_254 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_812[31]));
                        float2 _cvt_f32_255 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_813[31]));
                        float gate_x_1067 = _cvt_f32_254.x;
                        float gate_y_1068 = _cvt_f32_254.y;
                        float up_x_1069 = _cvt_f32_255.x;
                        float up_y_1070 = _cvt_f32_255.y;
                        float _exp_353 = expf(gate_x_1067 * -1.0f);
                        float denominator_x_1071 = _exp_353 + 1.0f;
                        float _exp_354 = expf(gate_y_1068 * -1.0f);
                        float denominator_y_1072 = _exp_354 + 1.0f;
                        float hidden_x_1073 = gate_x_1067 / denominator_x_1071 * up_x_1069;
                        float hidden_y_1074 = gate_y_1068 / denominator_y_1072 * up_y_1070;
                        hidden_packed_814[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1073, hidden_y_1074)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1075 = tid / 32;
                        int lane_1076 = tid % 32;
                        #pragma unroll
                        for (int half_26 = 0; half_26 < 2; half_26++) {
                            #pragma unroll
                            for (int col_tile_26 = 0; col_tile_26 < 2; col_tile_26++) {
                                int row_25 = warp_1075 * 32 + half_26 * 16 + lane_1076 % 16;
                                int col_53 = col_tile_26 * 16 + lane_1076 / 16 * 8;
                                unsigned int address_3_18 = d_smem_addr + (unsigned int)((row_25 * 32 + col_53) * 2);
                                address_3_18 = address_3_18 ^ (address_3_18 & 511) >> 7 << 4;
                                int offset_20 = half_26 * 8 + col_tile_26 * 4;
                                uint32_t _stmatrix_addr_355 = static_cast<uint32_t>(address_3_18);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_355), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_20 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1077 = tid / 32;
                        int lane_1078 = tid % 32;
                        #pragma unroll
                        for (int half_27 = 0; half_27 < 2; half_27++) {
                            #pragma unroll
                            for (int col_tile_27 = 0; col_tile_27 < 2; col_tile_27++) {
                                int row_26 = warp_1077 * 32 + half_27 * 16 + lane_1078 % 16;
                                int col_54 = col_tile_27 * 16 + lane_1078 / 16 * 8;
                                unsigned int address_3_19 = d_smem_addr + 8192 + (unsigned int)((row_26 * 32 + col_54) * 2);
                                address_3_19 = address_3_19 ^ (address_3_19 & 511) >> 7 << 4;
                                int offset_21 = half_27 * 8 + col_tile_27 * 4;
                                uint32_t _stmatrix_addr_356 = static_cast<uint32_t>(address_3_19);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_356), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_21 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1079 = tid / 32;
                        int lane_1080 = tid % 32;
                        #pragma unroll
                        for (int half_28 = 0; half_28 < 2; half_28++) {
                            #pragma unroll
                            for (int col_tile_28 = 0; col_tile_28 < 2; col_tile_28++) {
                                int row_27 = warp_1079 * 32 + half_28 * 16 + lane_1080 % 16;
                                int col_55 = col_tile_28 * 16 + lane_1080 / 16 * 8;
                                unsigned int address_3_20 = d_smem_addr + 16384 + (unsigned int)((row_27 * 32 + col_55) * 2);
                                address_3_20 = address_3_20 ^ (address_3_20 & 511) >> 7 << 4;
                                int offset_22 = half_28 * 8 + col_tile_28 * 4;
                                uint32_t _stmatrix_addr_357 = static_cast<uint32_t>(address_3_20);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_357), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_22])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_22 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_22 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_22 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1081 = tid / 32;
                        int lane_1082 = tid % 32;
                        #pragma unroll
                        for (int half_29 = 0; half_29 < 2; half_29++) {
                            #pragma unroll
                            for (int col_tile_29 = 0; col_tile_29 < 2; col_tile_29++) {
                                int row_28 = warp_1081 * 32 + half_29 * 16 + lane_1082 % 16;
                                int col_56 = col_tile_29 * 16 + lane_1082 / 16 * 8;
                                unsigned int address_3_21 = d_smem_addr + (unsigned int)((row_28 * 32 + col_56) * 2);
                                address_3_21 = address_3_21 ^ (address_3_21 & 511) >> 7 << 4;
                                int offset_23 = 16 + half_29 * 8 + col_tile_29 * 4;
                                uint32_t _stmatrix_addr_358 = static_cast<uint32_t>(address_3_21);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_358), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_23])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_23 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_23 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_23 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&gate_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1083 = tid / 32;
                        int lane_1084 = tid % 32;
                        #pragma unroll
                        for (int half_30 = 0; half_30 < 2; half_30++) {
                            #pragma unroll
                            for (int col_tile_30 = 0; col_tile_30 < 2; col_tile_30++) {
                                int row_29 = warp_1083 * 32 + half_30 * 16 + lane_1084 % 16;
                                int col_57 = col_tile_30 * 16 + lane_1084 / 16 * 8;
                                unsigned int address_3_22 = d_smem_addr + 8192 + (unsigned int)((row_29 * 32 + col_57) * 2);
                                address_3_22 = address_3_22 ^ (address_3_22 & 511) >> 7 << 4;
                                int offset_24 = 16 + half_30 * 8 + col_tile_30 * 4;
                                uint32_t _stmatrix_addr_359 = static_cast<uint32_t>(address_3_22);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_359), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_24])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_24 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_24 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_24 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&up_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1085 = tid / 32;
                        int lane_1086 = tid % 32;
                        #pragma unroll
                        for (int half_31 = 0; half_31 < 2; half_31++) {
                            #pragma unroll
                            for (int col_tile_31 = 0; col_tile_31 < 2; col_tile_31++) {
                                int row_30 = warp_1085 * 32 + half_31 * 16 + lane_1086 % 16;
                                int col_58 = col_tile_31 * 16 + lane_1086 / 16 * 8;
                                unsigned int address_3_23 = d_smem_addr + 16384 + (unsigned int)((row_30 * 32 + col_58) * 2);
                                address_3_23 = address_3_23 ^ (address_3_23 & 511) >> 7 << 4;
                                int offset_25 = 16 + half_31 * 8 + col_tile_31 * 4;
                                uint32_t _stmatrix_addr_360 = static_cast<uint32_t>(address_3_23);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_360), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_25])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_25 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_25 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_25 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
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
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (x_2))), "r"(static_cast<unsigned int>(2)) : "memory");
                                }
                            }
                        }
                    }
                }
                gemm_bits = phase_bits_2;
            } else if (compute < shared_tasks) {
                int col_blocks_5 = (hidden + 512 - 1) / 512;
                int x_3 = -1;
                int y_3 = -1;
                int expert_2 = -1;
                int k_start_2 = 0;
                int k_end_2 = 0;
                int first_2 = 0;
                int row_blocks_1 = local_tokens / 256;
                if (compute - shared_fused < row_blocks_1 * col_blocks_5) {
                    int supergroup_3 = (compute - shared_fused) / (row_blocks_1 * 8);
                    int full_cols_3 = col_blocks_5 / 8 * 8;
                    int row_31 = 0;
                    int col_59 = 0;
                    if (compute - shared_fused < row_blocks_1 * full_cols_3) {
                        row_31 = (compute - shared_fused) % (row_blocks_1 * 8) / 8;
                        col_59 = supergroup_3 * 8 + (compute - shared_fused) % 8;
                    } else {
                        row_31 = (compute - shared_fused - row_blocks_1 * full_cols_3) / (col_blocks_5 - full_cols_3);
                        col_59 = full_cols_3 + (compute - shared_fused - row_blocks_1 * full_cols_3) % (col_blocks_5 - full_cols_3);
                    }
                    if ((supergroup_3 & 1) != 0) {
                        row_31 = row_blocks_1 - row_31 - 1;
                    }
                    x_3 = row_31;
                    y_3 = col_59;
                    expert_2 = 0;
                }
                unsigned int phase_bits_3 = gemm_bits;
                int has_hi_1 = 0;
                has_hi_1 = (int)((y_3 * 2 + 1) * 256 < hidden);
                int global_mini_2 = 0;
                int macro_rows_3 = 0;
                int iterations_2 = intermediate / 64;
                if (expert_2 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                bool enabled_value_2 = 1;
                                if (enabled_value_2 != 0) {
                                    int _poll_3_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_3_first) : "l"(hidden_ready + (macro_rows_3 + x_3)) : "memory");
                                    int _poll_3 = _poll_3_first;
                                    while (_poll_3 < 2 * (intermediate / 128)) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int _poll_3_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_3_next) : "l"(hidden_ready + (macro_rows_3 + x_3)) : "memory");
                                        _poll_3 = _poll_3_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                int _min_361 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                int mini_rows_4 = ((0) > (_min_361) ? (0) : (_min_361));
                                int required_4 = (mini_rows_4 + 127) / 128 * ((intermediate + 511) / 512);
                            }
                            unsigned int previous_1 = phase_bits_3 >> 7 & 1;
                            unsigned int bits_2 = phase_bits_3;
                            if (previous_1 != 0) {
                                mbarrier_wait(gemm_finished_addr, bits_2 >> 16 & 1);
                                mbarrier_wait(gemm_finished_addr + 8, bits_2 >> 17 & 1);
                                mbarrier_wait(gemm_finished_addr + 16, bits_2 >> 18 & 1);
                                mbarrier_wait(gemm_finished_addr + 24, bits_2 >> 19 & 1);
                                mbarrier_wait(gemm_finished_addr + 32, bits_2 >> 20 & 1);
                                mbarrier_wait(gemm_finished_addr + 40, bits_2 >> 21 & 1);
                                bits_2 = bits_2 ^ 128;
                            }
                            phase_bits_3 = bits_2;
                            int ring_2 = 0;
                            #pragma unroll 1
                            for (int idx_2 = 0; idx_2 < iterations_2; idx_2++) {
                                mbarrier_wait(gemm_finished_addr + (ring_2) * 8, phase_bits_3 >> (unsigned int)(16 + ring_2) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&hidden_shared_in)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&wd_shared)), "r"(0), "r"(y_3 * 2 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_2), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)(ring_2 * 16384)), "l"((&wd_shared)), "r"(0), "r"((y_3 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_2), "r"(0),
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
                                mbarrier_wait(output_finished_addr + 8, phase_bits_3 >> 23 & 1);
                                mbarrier_wait(prefix_released_addr, phase_bits_3 >> 22 & 1);
                                mbarrier_wait(prefix_released_addr + 8, phase_bits_3 >> 23 & 1);
                                phase_bits_3 = phase_bits_3 ^ 12582912;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_3 = 0; idx_3 < iterations_2; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_3 >> (unsigned int)ring_3 & 1);
                                    int _smem_desc_lo_362 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                    int _smem_desc_lo_363 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
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
            :: "r"(_smem_desc_lo_362), "r"(_smem_desc_lo_363), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)));
                                    int _smem_desc_lo_364 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                    int _smem_desc_lo_365 = (((b_hi_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
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
            :: "r"(_smem_desc_lo_364), "r"(_smem_desc_lo_365), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)));
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_3) * 8, (uint16_t)(3));
                                    phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << ring_3);
                                    ring_3 = (ring_3 + 1) % 4;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        unsigned int _parity_output_arrived_1 = (phase_bits_3 >> 6) % 2;
                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_1);
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
                                    packed[chunk * 16 + sub * 8 + pair] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32[pair * 2], _tmem_load_32[pair * 2 + 1])));
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
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((prefix_released_addr) & 0xFEFFFFFF) : "memory");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + 8) & 0xFEFFFFFF) : "memory");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((prefix_released_addr + 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        if (tid == 0) {
                            int previous_offset_1 = macro_size;
                            int output_row = x_3 * 256 + cta_rank_0 * 128;
                            int _min_366 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            if (output_row < _min_366) {
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
                            for (int half_32 = 0; half_32 < 2; half_32++) {
                                #pragma unroll
                                for (int col_tile_32 = 0; col_tile_32 < 2; col_tile_32++) {
                                    int row_32 = warp_0 * 32 + half_32 * 16 + lane_2 % 16;
                                    int col_60 = col_tile_32 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_32 * 32 + col_60) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_26 = chunk_1 * 16 + half_32 * 8 + col_tile_32 * 4;
                                    uint32_t _stmatrix_addr_367 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_367), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_26 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&y_shared)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 2 * 8 + chunk_1), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                        packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_33[pair_1 * 2], _tmem_load_33[pair_1 * 2 + 1])));
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
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((prefix_released_addr) & 0xFEFFFFFF) : "memory");
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr + 8) & 0xFEFFFFFF) : "memory");
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((prefix_released_addr + 8) & 0xFEFFFFFF) : "memory");
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
                                for (int half_33 = 0; half_33 < 2; half_33++) {
                                    #pragma unroll
                                    for (int col_tile_33 = 0; col_tile_33 < 2; col_tile_33++) {
                                        int row_33 = warp_0_1 * 32 + half_33 * 16 + lane_3 % 16;
                                        int col_61 = col_tile_33 * 16 + lane_3 / 16 * 8;
                                        unsigned int address_7 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_33 * 32 + col_61) * 2);
                                        address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                        int offset_27 = chunk_3 * 16 + half_33 * 8 + col_tile_33 * 4;
                                        uint32_t _stmatrix_addr_368 = static_cast<uint32_t>(address_7);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_368), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_27 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&y_shared)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"((y_3 * 2 + 1) * 8 + chunk_3), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                if (task_0 < mini_gate) {
                    if (task_0 < gate_tiles) {
                        unsigned int phase_bits_4 = gemm_bits;
                        int acc_buffer = (int)(phase_bits_4 >> 14 & 1);
                        int other_buffer = acc_buffer ^ 1;
                        unsigned int acc_taddr = taddr_1 + (unsigned int)(acc_buffer * 256);
                        int has_hi_2 = 0;
                        int iterations_3 = td_iters;
                        int global_mini_3 = td_gmini;
                        int macro_rows_4 = td_mrows;
                        int macro_k_1 = td_mk;
                        if (td_expert < 0) {
                            if (tid == 0) {
                            }
                        } else {
                            if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            int _min_369 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                            int mini_rows_5 = ((0) > (_min_369) ? (0) : (_min_369));
                                            int required_5 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_3 = 1;
                                            if (enabled_value_3 != 0) {
                                                int _poll_4_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_4_first) : "l"(x_ready + global_mini_3) : "memory");
                                                int _poll_4 = _poll_4_first;
                                                while (_poll_4 < required_5) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int _poll_4_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_4_next) : "l"(x_ready + global_mini_3) : "memory");
                                                    _poll_4 = _poll_4_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        unsigned int previous_2 = phase_bits_4 >> 7 & 1;
                                        unsigned int bits_3 = phase_bits_4;
                                        if (previous_2 != 1) {
                                            mbarrier_wait(gemm_finished_addr, bits_3 >> 16 & 1);
                                            mbarrier_wait(gemm_finished_addr + 8, bits_3 >> 17 & 1);
                                            mbarrier_wait(gemm_finished_addr + 16, bits_3 >> 18 & 1);
                                            mbarrier_wait(gemm_finished_addr + 24, bits_3 >> 19 & 1);
                                            mbarrier_wait(gemm_finished_addr + 32, bits_3 >> 20 & 1);
                                            mbarrier_wait(gemm_finished_addr + 40, bits_3 >> 21 & 1);
                                            bits_3 = bits_3 ^ 128;
                                        }
                                        phase_bits_4 = bits_3;
                                        int ring_4 = 0;
                                        #pragma unroll 1
                                        for (int idx_4 = 0; idx_4 < iterations_3; idx_4++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_4) * 8, phase_bits_4 >> (unsigned int)(16 + ring_4) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v38_addr + (unsigned int)(ring_4 * 16384)), "l"((&x_q)), "r"(0), "r"(td_x * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v39_addr + (unsigned int)(ring_4 * 16384)), "l"((&wg_q)), "r"(0), "r"(td_y * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(td_expert), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_4);
                                            ring_4 = (ring_4 + 1) % 6;
                                        }
                                    }
                                }
                            } else if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_370 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                            int mini_rows_6 = ((0) > (_min_370) ? (0) : (_min_370));
                                            int required_6 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_4 = 1;
                                            if (enabled_value_4 != 0) {
                                                int _poll_5_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_5_first) : "l"(x_ready + global_mini_3) : "memory");
                                                int _poll_5 = _poll_5_first;
                                                while (_poll_5 < required_6) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int _poll_5_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_5_next) : "l"(x_ready + global_mini_3) : "memory");
                                                    _poll_5 = _poll_5_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        unsigned int previous_3 = phase_bits_4 >> 7 & 1;
                                        unsigned int bits_4 = phase_bits_4;
                                        if (previous_3 != 1) {
                                            mbarrier_wait(scales_finished_addr, bits_4 >> 16 & 1);
                                            mbarrier_wait(scales_finished_addr + 8, bits_4 >> 17 & 1);
                                            mbarrier_wait(scales_finished_addr + 16, bits_4 >> 18 & 1);
                                            mbarrier_wait(scales_finished_addr + 24, bits_4 >> 19 & 1);
                                            mbarrier_wait(scales_finished_addr + 32, bits_4 >> 20 & 1);
                                            mbarrier_wait(scales_finished_addr + 40, bits_4 >> 21 & 1);
                                            bits_4 = bits_4 ^ 128;
                                        }
                                        phase_bits_4 = bits_4;
                                        int ring_5 = 0;
                                        #pragma unroll 1
                                        for (int idx_5 = 0; idx_5 < iterations_3; idx_5++) {
                                            mbarrier_wait(scales_finished_addr + (ring_5) * 8, phase_bits_4 >> (unsigned int)(16 + ring_5) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v40_addr + (unsigned int)(ring_5 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((td_x * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                                   "r"(((scales_arrived_addr + (ring_5) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v41_addr + (unsigned int)(ring_5 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((td_expert * (intermediate / 128) + td_y * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                                   "r"(((scales_arrived_addr + (ring_5) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_5);
                                            ring_5 = (ring_5 + 1) % 6;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_6 = 0;
                                            mbarrier_wait(output_finished_addr + (acc_buffer) * 8, phase_bits_4 >> (unsigned int)(22 + acc_buffer) & 1);
                                            mbarrier_wait(prefix_released_addr + (acc_buffer) * 8, phase_bits_4 >> (unsigned int)(22 + acc_buffer) & 1);
                                            mbarrier_wait(prefix_released_addr + (other_buffer) * 8, phase_bits_4 >> (unsigned int)(22 + other_buffer) & 1);
                                            phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 22 + acc_buffer);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_6) * 8, 3072);
                                                mbarrier_wait(scales_arrived_addr + (ring_6) * 8, phase_bits_4 >> (unsigned int)(8 + ring_6) & 1);
                                                int buffer = idx_6 % 3;
                                                int sfa_slot = buffer + other_buffer * 64;
                                                int sfb_slot = buffer + other_buffer * 32;
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + sfa_slot * 4, make_sf_cp_desc_sbo128(smem_v40_addr + (unsigned int)(ring_6 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + sfb_slot * 8, make_sf_cp_desc_sbo128(smem_v41_addr + (unsigned int)(ring_6 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + sfb_slot * 8 + 4), make_sf_cp_desc_sbo128((smem_v41_addr + (unsigned int)(ring_6 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_6) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_6) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_6) * 8, phase_bits_4 >> (unsigned int)ring_6 & 1);
                                                int _smem_desc_lo_371 = (((smem_v38_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                                int _smem_desc_lo_372 = (((smem_v39_addr) >> 4) & 0x3FFF) + (ring_6) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_371) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_372) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer * 256)), a_desc + 0, b_desc + 0,
                                                        0x10c00000U, tmem_tmem_sfa + sfa_slot * 4, tmem_tmem_sfb + sfb_slot * 8, ((idx_6 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer * 256)), a_desc + 2, b_desc + 2,
                                                        0x30c00010U, tmem_tmem_sfa + sfa_slot * 4, tmem_tmem_sfb + sfb_slot * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer * 256)), a_desc + 4, b_desc + 4,
                                                        0x50c00020U, tmem_tmem_sfa + sfa_slot * 4, tmem_tmem_sfb + sfb_slot * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer * 256)), a_desc + 6, b_desc + 6,
                                                        0x70c00030U, tmem_tmem_sfa + sfa_slot * 4, tmem_tmem_sfb + sfb_slot * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_6) * 8, (uint16_t)(3));
                                                phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << ring_6) ^ (unsigned int)(1 << 8 + ring_6);
                                                ring_6 = (ring_6 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    unsigned int _parity_output_arrived_2 = (phase_bits_4 >> 6) % 2;
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_2);
                                    int warp_row = tid / 32 * 32;
                                    unsigned int packed_1[128];
                                    #pragma unroll
                                    for (int i_48 = 0; i_48 < 8; i_48++) {
                                        float _tmem_load_34[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(_tmem_load_34[0]), "=f"(_tmem_load_34[1]), "=f"(_tmem_load_34[2]), "=f"(_tmem_load_34[3]), "=f"(_tmem_load_34[4]), "=f"(_tmem_load_34[5]), "=f"(_tmem_load_34[6]), "=f"(_tmem_load_34[7]), "=f"(_tmem_load_34[8]), "=f"(_tmem_load_34[9]), "=f"(_tmem_load_34[10]), "=f"(_tmem_load_34[11]), "=f"(_tmem_load_34[12]), "=f"(_tmem_load_34[13]), "=f"(_tmem_load_34[14]), "=f"(_tmem_load_34[15]), "=f"(_tmem_load_34[16]), "=f"(_tmem_load_34[17]), "=f"(_tmem_load_34[18]), "=f"(_tmem_load_34[19]), "=f"(_tmem_load_34[20]), "=f"(_tmem_load_34[21]), "=f"(_tmem_load_34[22]), "=f"(_tmem_load_34[23]), "=f"(_tmem_load_34[24]), "=f"(_tmem_load_34[25]), "=f"(_tmem_load_34[26]), "=f"(_tmem_load_34[27]), "=f"(_tmem_load_34[28]), "=f"(_tmem_load_34[29]), "=f"(_tmem_load_34[30]), "=f"(_tmem_load_34[31])
                                            : "r"(acc_taddr + (unsigned int)(warp_row << 16) + (unsigned int)(i_48 * 32)));
                                        #pragma unroll
                                        for (int j_16 = 0; j_16 < 16; j_16++) {
                                            packed_1[i_48 * 16 + j_16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_34[2 * j_16], _tmem_load_34[2 * j_16 + 1])));
                                        }
                                        if (i_48 == 1) {
                                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((prefix_released_addr + (acc_buffer) * 8) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + (acc_buffer) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                    unsigned int scale_word_8 = 0;
                                    unsigned int block_2[16];
                                    #pragma unroll
                                    for (int j_17 = 0; j_17 < 16; j_17++) {
                                        block_2[j_17] = packed_1[j_17];
                                    }
                                    #pragma unroll
                                    for (int j_18 = 0; j_18 < 4; j_18++) {
                                        unsigned int address_8 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_18 * 16);
                                        address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_8 - smem_v42_addr)), "r"(packed_1[4 * j_18]), "r"(packed_1[4 * j_18 + 1]), "r"(packed_1[4 * j_18 + 2]), "r"(packed_1[4 * j_18 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_48;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(block_2[0]));
                                    unsigned int amax_pair_16 = _bf16x2_abs_48;
                                    #pragma unroll
                                    for (int i_49 = 1; i_49 < 16; i_49++) {
                                        uint32_t _bf16x2_abs_49;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(block_2[i_49]));
                                        uint32_t _bf16x2_max_24;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(amax_pair_16), "r"(_bf16x2_abs_49));
                                        amax_pair_16 = _bf16x2_max_24;
                                    }
                                    uint16_t _bf16_max_24;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_24) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                    float amax_17;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_17) : "h"((uint16_t)(_bf16_max_24)));
                                    float scale_17 = fmaxf(amax_17 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_373;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_373) : "f"(scale_17), "f"(scale_17));
                                    unsigned int scale_byte_17 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_373 & 255;
                                    unsigned int inverse_lane_17 = 254 - scale_byte_17 << 7;
                                    unsigned int inverse_17 = inverse_lane_17 | inverse_lane_17 << 16;
                                    unsigned int words_17[8];
                                    #pragma unroll
                                    for (int i_50 = 0; i_50 < 8; i_50++) {
                                        uint32_t _bf16x2_mul_48;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(block_2[i_50 * 2]), "r"(inverse_17));
                                        uint16_t _e4m3x2_bf16x2_24;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_24) : "r"(_bf16x2_mul_48));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_48);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_24) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_49;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(block_2[i_50 * 2 + 1]), "r"(inverse_17));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_374;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_374) : "r"(_bf16x2_mul_49));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_49);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_374) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_17[i_50] = (unsigned int)_e4m3x2_bf16x2_24 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_374 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_17;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_17[0]), "r"(words_17[1]), "r"(words_17[2]), "r"(words_17[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_17[4]), "r"(words_17[5]), "r"(words_17[6]), "r"(words_17[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_0[16];
                                    #pragma unroll
                                    for (int j_19 = 0; j_19 < 16; j_19++) {
                                        block_0[j_19] = packed_1[16 + j_19];
                                    }
                                    #pragma unroll
                                    for (int j_20 = 0; j_20 < 4; j_20++) {
                                        unsigned int address_9 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_20 * 16);
                                        address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_9 - smem_v42_addr)), "r"(packed_1[16 + 4 * j_20]), "r"(packed_1[16 + 4 * j_20 + 1]), "r"(packed_1[16 + 4 * j_20 + 2]), "r"(packed_1[16 + 4 * j_20 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 1, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_50;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(block_0[0]));
                                    unsigned int amax_pair_1_1 = _bf16x2_abs_50;
                                    #pragma unroll
                                    for (int i_51 = 1; i_51 < 16; i_51++) {
                                        uint32_t _bf16x2_abs_51;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(block_0[i_51]));
                                        uint32_t _bf16x2_max_25;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(amax_pair_1_1), "r"(_bf16x2_abs_51));
                                        amax_pair_1_1 = _bf16x2_max_25;
                                    }
                                    uint16_t _bf16_max_25;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_25) : "h"((uint16_t)(amax_pair_1_1 & 65535)), "h"((uint16_t)(amax_pair_1_1 >> 16)));
                                    float amax_2_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2_1) : "h"((uint16_t)(_bf16_max_25)));
                                    float scale_3_1 = fmaxf(amax_2_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_375;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_375) : "f"(scale_3_1), "f"(scale_3_1));
                                    unsigned int scale_byte_4_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_375 & 255;
                                    unsigned int inverse_lane_5_1 = 254 - scale_byte_4_1 << 7;
                                    unsigned int inverse_6_1 = inverse_lane_5_1 | inverse_lane_5_1 << 16;
                                    unsigned int words_7_1[8];
                                    #pragma unroll
                                    for (int i_52 = 0; i_52 < 8; i_52++) {
                                        uint32_t _bf16x2_mul_50;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(block_0[i_52 * 2]), "r"(inverse_6_1));
                                        uint16_t _e4m3x2_bf16x2_25;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_25) : "r"(_bf16x2_mul_50));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_50);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_51;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(block_0[i_52 * 2 + 1]), "r"(inverse_6_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_376;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_376) : "r"(_bf16x2_mul_51));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_51);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_376) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_7_1[i_52] = (unsigned int)_e4m3x2_bf16x2_25 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_376 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_4_1 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_7_1[0]), "r"(words_7_1[1]), "r"(words_7_1[2]), "r"(words_7_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_7_1[4]), "r"(words_7_1[5]), "r"(words_7_1[6]), "r"(words_7_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 32), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_8[16];
                                    #pragma unroll
                                    for (int j_21 = 0; j_21 < 16; j_21++) {
                                        block_8[j_21] = packed_1[32 + j_21];
                                    }
                                    #pragma unroll
                                    for (int j_22 = 0; j_22 < 4; j_22++) {
                                        unsigned int address_10 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_22 * 16);
                                        address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_10 - smem_v42_addr)), "r"(packed_1[32 + 4 * j_22]), "r"(packed_1[32 + 4 * j_22 + 1]), "r"(packed_1[32 + 4 * j_22 + 2]), "r"(packed_1[32 + 4 * j_22 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 2, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_52;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(block_8[0]));
                                    unsigned int amax_pair_9_1 = _bf16x2_abs_52;
                                    #pragma unroll
                                    for (int i_53 = 1; i_53 < 16; i_53++) {
                                        uint32_t _bf16x2_abs_53;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(block_8[i_53]));
                                        uint32_t _bf16x2_max_26;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(amax_pair_9_1), "r"(_bf16x2_abs_53));
                                        amax_pair_9_1 = _bf16x2_max_26;
                                    }
                                    uint16_t _bf16_max_26;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_26) : "h"((uint16_t)(amax_pair_9_1 & 65535)), "h"((uint16_t)(amax_pair_9_1 >> 16)));
                                    float amax_10_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10_1) : "h"((uint16_t)(_bf16_max_26)));
                                    float scale_11_1 = fmaxf(amax_10_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_377;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_377) : "f"(scale_11_1), "f"(scale_11_1));
                                    unsigned int scale_byte_12_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_377 & 255;
                                    unsigned int inverse_lane_13_1 = 254 - scale_byte_12_1 << 7;
                                    unsigned int inverse_14_1 = inverse_lane_13_1 | inverse_lane_13_1 << 16;
                                    unsigned int words_15_1[8];
                                    #pragma unroll
                                    for (int i_54 = 0; i_54 < 8; i_54++) {
                                        uint32_t _bf16x2_mul_52;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(block_8[i_54 * 2]), "r"(inverse_14_1));
                                        uint16_t _e4m3x2_bf16x2_26;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_26) : "r"(_bf16x2_mul_52));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_52);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_53;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(block_8[i_54 * 2 + 1]), "r"(inverse_14_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_378;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_378) : "r"(_bf16x2_mul_53));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_53);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_378) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_15_1[i_54] = (unsigned int)_e4m3x2_bf16x2_26 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_378 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_12_1 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_15_1[0]), "r"(words_15_1[1]), "r"(words_15_1[2]), "r"(words_15_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_15_1[4]), "r"(words_15_1[5]), "r"(words_15_1[6]), "r"(words_15_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 64), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_16[16];
                                    #pragma unroll
                                    for (int j_23 = 0; j_23 < 16; j_23++) {
                                        block_16[j_23] = packed_1[48 + j_23];
                                    }
                                    #pragma unroll
                                    for (int j_24 = 0; j_24 < 4; j_24++) {
                                        unsigned int address_11 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_24 * 16);
                                        address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_11 - smem_v42_addr)), "r"(packed_1[48 + 4 * j_24]), "r"(packed_1[48 + 4 * j_24 + 1]), "r"(packed_1[48 + 4 * j_24 + 2]), "r"(packed_1[48 + 4 * j_24 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 3, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_54;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(block_16[0]));
                                    unsigned int amax_pair_17 = _bf16x2_abs_54;
                                    #pragma unroll
                                    for (int i_55 = 1; i_55 < 16; i_55++) {
                                        uint32_t _bf16x2_abs_55;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(block_16[i_55]));
                                        uint32_t _bf16x2_max_27;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(amax_pair_17), "r"(_bf16x2_abs_55));
                                        amax_pair_17 = _bf16x2_max_27;
                                    }
                                    uint16_t _bf16_max_27;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_27) : "h"((uint16_t)(amax_pair_17 & 65535)), "h"((uint16_t)(amax_pair_17 >> 16)));
                                    float amax_18;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18) : "h"((uint16_t)(_bf16_max_27)));
                                    float scale_19 = fmaxf(amax_18 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_379;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_379) : "f"(scale_19), "f"(scale_19));
                                    unsigned int scale_byte_20 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_379 & 255;
                                    unsigned int inverse_lane_21 = 254 - scale_byte_20 << 7;
                                    unsigned int inverse_22 = inverse_lane_21 | inverse_lane_21 << 16;
                                    unsigned int words_23[8];
                                    #pragma unroll
                                    for (int i_56 = 0; i_56 < 8; i_56++) {
                                        uint32_t _bf16x2_mul_54;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(block_16[i_56 * 2]), "r"(inverse_22));
                                        uint16_t _e4m3x2_bf16x2_27;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_27) : "r"(_bf16x2_mul_54));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_54);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_55;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(block_16[i_56 * 2 + 1]), "r"(inverse_22));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_380;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_380) : "r"(_bf16x2_mul_55));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_55);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_380) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_23[i_56] = (unsigned int)_e4m3x2_bf16x2_27 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_380 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_20 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_23[0]), "r"(words_23[1]), "r"(words_23[2]), "r"(words_23[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_23[4]), "r"(words_23[5]), "r"(words_23[6]), "r"(words_23[7]) : "memory");
                                    smem_v44[tid % 32 * 4 + tid / 32] = scale_word_8;
                                    scale_word_8 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 96), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&gate_sc_store), 0, 0, (td_x * 2 + cta_rank_0) * i_tiles + td_y * 2, smem_v44_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_24[16];
                                    #pragma unroll
                                    for (int j_25 = 0; j_25 < 16; j_25++) {
                                        block_24[j_25] = packed_1[64 + j_25];
                                    }
                                    #pragma unroll
                                    for (int j_26 = 0; j_26 < 4; j_26++) {
                                        unsigned int address_12 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_26 * 16);
                                        address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_12 - smem_v42_addr)), "r"(packed_1[64 + 4 * j_26]), "r"(packed_1[64 + 4 * j_26 + 1]), "r"(packed_1[64 + 4 * j_26 + 2]), "r"(packed_1[64 + 4 * j_26 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 4, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_56;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(block_24[0]));
                                    unsigned int amax_pair_25 = _bf16x2_abs_56;
                                    #pragma unroll
                                    for (int i_57 = 1; i_57 < 16; i_57++) {
                                        uint32_t _bf16x2_abs_57;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(block_24[i_57]));
                                        uint32_t _bf16x2_max_28;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(amax_pair_25), "r"(_bf16x2_abs_57));
                                        amax_pair_25 = _bf16x2_max_28;
                                    }
                                    uint16_t _bf16_max_28;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_28) : "h"((uint16_t)(amax_pair_25 & 65535)), "h"((uint16_t)(amax_pair_25 >> 16)));
                                    float amax_26;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_26) : "h"((uint16_t)(_bf16_max_28)));
                                    float scale_27 = fmaxf(amax_26 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_381;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_381) : "f"(scale_27), "f"(scale_27));
                                    unsigned int scale_byte_28 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_381 & 255;
                                    unsigned int inverse_lane_29 = 254 - scale_byte_28 << 7;
                                    unsigned int inverse_30 = inverse_lane_29 | inverse_lane_29 << 16;
                                    unsigned int words_31[8];
                                    #pragma unroll
                                    for (int i_58 = 0; i_58 < 8; i_58++) {
                                        uint32_t _bf16x2_mul_56;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(block_24[i_58 * 2]), "r"(inverse_30));
                                        uint16_t _e4m3x2_bf16x2_28;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_28) : "r"(_bf16x2_mul_56));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_56);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_57;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(block_24[i_58 * 2 + 1]), "r"(inverse_30));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_382;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_382) : "r"(_bf16x2_mul_57));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_57);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_382) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_31[i_58] = (unsigned int)_e4m3x2_bf16x2_28 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_382 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_28;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_31[0]), "r"(words_31[1]), "r"(words_31[2]), "r"(words_31[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_31[4]), "r"(words_31[5]), "r"(words_31[6]), "r"(words_31[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 128), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_32[16];
                                    #pragma unroll
                                    for (int j_27 = 0; j_27 < 16; j_27++) {
                                        block_32[j_27] = packed_1[80 + j_27];
                                    }
                                    #pragma unroll
                                    for (int j_28 = 0; j_28 < 4; j_28++) {
                                        unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_28 * 16);
                                        address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_13 - smem_v42_addr)), "r"(packed_1[80 + 4 * j_28]), "r"(packed_1[80 + 4 * j_28 + 1]), "r"(packed_1[80 + 4 * j_28 + 2]), "r"(packed_1[80 + 4 * j_28 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 5, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_58;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(block_32[0]));
                                    unsigned int amax_pair_33 = _bf16x2_abs_58;
                                    #pragma unroll
                                    for (int i_59 = 1; i_59 < 16; i_59++) {
                                        uint32_t _bf16x2_abs_59;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(block_32[i_59]));
                                        uint32_t _bf16x2_max_29;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(amax_pair_33), "r"(_bf16x2_abs_59));
                                        amax_pair_33 = _bf16x2_max_29;
                                    }
                                    uint16_t _bf16_max_29;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_29) : "h"((uint16_t)(amax_pair_33 & 65535)), "h"((uint16_t)(amax_pair_33 >> 16)));
                                    float amax_34;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_34) : "h"((uint16_t)(_bf16_max_29)));
                                    float scale_35 = fmaxf(amax_34 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_383;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_383) : "f"(scale_35), "f"(scale_35));
                                    unsigned int scale_byte_36 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_383 & 255;
                                    unsigned int inverse_lane_37 = 254 - scale_byte_36 << 7;
                                    unsigned int inverse_38 = inverse_lane_37 | inverse_lane_37 << 16;
                                    unsigned int words_39[8];
                                    #pragma unroll
                                    for (int i_60 = 0; i_60 < 8; i_60++) {
                                        uint32_t _bf16x2_mul_58;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(block_32[i_60 * 2]), "r"(inverse_38));
                                        uint16_t _e4m3x2_bf16x2_29;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_29) : "r"(_bf16x2_mul_58));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_58);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_59;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(block_32[i_60 * 2 + 1]), "r"(inverse_38));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_384;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_384) : "r"(_bf16x2_mul_59));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_59);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_384) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_39[i_60] = (unsigned int)_e4m3x2_bf16x2_29 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_384 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_36 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_39[0]), "r"(words_39[1]), "r"(words_39[2]), "r"(words_39[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_39[4]), "r"(words_39[5]), "r"(words_39[6]), "r"(words_39[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 160), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_40[16];
                                    #pragma unroll
                                    for (int j_29 = 0; j_29 < 16; j_29++) {
                                        block_40[j_29] = packed_1[96 + j_29];
                                    }
                                    #pragma unroll
                                    for (int j_30 = 0; j_30 < 4; j_30++) {
                                        unsigned int address_14 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_30 * 16);
                                        address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_14 - smem_v42_addr)), "r"(packed_1[96 + 4 * j_30]), "r"(packed_1[96 + 4 * j_30 + 1]), "r"(packed_1[96 + 4 * j_30 + 2]), "r"(packed_1[96 + 4 * j_30 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 6, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_60;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_60) : "r"(block_40[0]));
                                    unsigned int amax_pair_41 = _bf16x2_abs_60;
                                    #pragma unroll
                                    for (int i_61 = 1; i_61 < 16; i_61++) {
                                        uint32_t _bf16x2_abs_61;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_61) : "r"(block_40[i_61]));
                                        uint32_t _bf16x2_max_30;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_30) : "r"(amax_pair_41), "r"(_bf16x2_abs_61));
                                        amax_pair_41 = _bf16x2_max_30;
                                    }
                                    uint16_t _bf16_max_30;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_30) : "h"((uint16_t)(amax_pair_41 & 65535)), "h"((uint16_t)(amax_pair_41 >> 16)));
                                    float amax_42;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_42) : "h"((uint16_t)(_bf16_max_30)));
                                    float scale_43 = fmaxf(amax_42 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_385;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_385) : "f"(scale_43), "f"(scale_43));
                                    unsigned int scale_byte_44 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_385 & 255;
                                    unsigned int inverse_lane_45 = 254 - scale_byte_44 << 7;
                                    unsigned int inverse_46 = inverse_lane_45 | inverse_lane_45 << 16;
                                    unsigned int words_47[8];
                                    #pragma unroll
                                    for (int i_62 = 0; i_62 < 8; i_62++) {
                                        uint32_t _bf16x2_mul_60;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(block_40[i_62 * 2]), "r"(inverse_46));
                                        uint16_t _e4m3x2_bf16x2_30;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_30) : "r"(_bf16x2_mul_60));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_60);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_61;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(block_40[i_62 * 2 + 1]), "r"(inverse_46));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_386;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_386) : "r"(_bf16x2_mul_61));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_61);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_386) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_47[i_62] = (unsigned int)_e4m3x2_bf16x2_30 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_386 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_44 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_47[0]), "r"(words_47[1]), "r"(words_47[2]), "r"(words_47[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_47[4]), "r"(words_47[5]), "r"(words_47[6]), "r"(words_47[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 192), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_48[16];
                                    #pragma unroll
                                    for (int j_31 = 0; j_31 < 16; j_31++) {
                                        block_48[j_31] = packed_1[112 + j_31];
                                    }
                                    #pragma unroll
                                    for (int j_32 = 0; j_32 < 4; j_32++) {
                                        unsigned int address_15 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_32 * 16);
                                        address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_15 - smem_v42_addr)), "r"(packed_1[112 + 4 * j_32]), "r"(packed_1[112 + 4 * j_32 + 1]), "r"(packed_1[112 + 4 * j_32 + 2]), "r"(packed_1[112 + 4 * j_32 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&gate_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 7, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_62;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_62) : "r"(block_48[0]));
                                    unsigned int amax_pair_49 = _bf16x2_abs_62;
                                    #pragma unroll
                                    for (int i_63 = 1; i_63 < 16; i_63++) {
                                        uint32_t _bf16x2_abs_63;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_63) : "r"(block_48[i_63]));
                                        uint32_t _bf16x2_max_31;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_31) : "r"(amax_pair_49), "r"(_bf16x2_abs_63));
                                        amax_pair_49 = _bf16x2_max_31;
                                    }
                                    uint16_t _bf16_max_31;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_31) : "h"((uint16_t)(amax_pair_49 & 65535)), "h"((uint16_t)(amax_pair_49 >> 16)));
                                    float amax_50;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_50) : "h"((uint16_t)(_bf16_max_31)));
                                    float scale_51 = fmaxf(amax_50 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_387;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_387) : "f"(scale_51), "f"(scale_51));
                                    unsigned int scale_byte_52 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_387 & 255;
                                    unsigned int inverse_lane_53 = 254 - scale_byte_52 << 7;
                                    unsigned int inverse_54 = inverse_lane_53 | inverse_lane_53 << 16;
                                    unsigned int words_55[8];
                                    #pragma unroll
                                    for (int i_64 = 0; i_64 < 8; i_64++) {
                                        uint32_t _bf16x2_mul_62;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(block_48[i_64 * 2]), "r"(inverse_54));
                                        uint16_t _e4m3x2_bf16x2_31;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_31) : "r"(_bf16x2_mul_62));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_62);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_63;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(block_48[i_64 * 2 + 1]), "r"(inverse_54));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_388;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_388) : "r"(_bf16x2_mul_63));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_63);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_388) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_55[i_64] = (unsigned int)_e4m3x2_bf16x2_31 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_388 << 16;
                                    }
                                    scale_word_8 = scale_word_8 | scale_byte_52 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_55[0]), "r"(words_55[1]), "r"(words_55[2]), "r"(words_55[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_55[4]), "r"(words_55[5]), "r"(words_55[6]), "r"(words_55[7]) : "memory");
                                    smem_v45[tid % 32 * 4 + tid / 32] = scale_word_8;
                                    scale_word_8 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&gate_q_store)), "r"(td_y * 256 + 224), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&gate_sc_store), 0, 0, (td_x * 2 + cta_rank_0) * i_tiles + td_y * 2 + 1, smem_v45_addr);
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
                                                bool enabled_value_5 = 1;
                                                if (enabled_value_5 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (gate_base + (macro_rows_4 + td_x) * (intermediate / 256) + td_y))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            phase_bits_4 = phase_bits_4 ^ 16384;
                        }
                        gemm_bits = phase_bits_4;
                    } else {
                        unsigned int phase_bits_5 = gemm_bits;
                        int acc_buffer_1 = (int)(phase_bits_5 >> 14 & 1);
                        int other_buffer_1 = acc_buffer_1 ^ 1;
                        unsigned int acc_taddr_1 = taddr_1 + (unsigned int)(acc_buffer_1 * 256);
                        int has_hi_3 = 0;
                        int iterations_4 = td_iters;
                        int global_mini_4 = td_gmini;
                        int macro_rows_5 = td_mrows;
                        int macro_k_2 = td_mk;
                        if (td_expert < 0) {
                            if (tid == 0) {
                            }
                        } else {
                            if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            int _min_389 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                            int mini_rows_7 = ((0) > (_min_389) ? (0) : (_min_389));
                                            int required_7 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_6 = 1;
                                            if (enabled_value_6 != 0) {
                                                int _poll_6_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_6_first) : "l"(x_ready + global_mini_4) : "memory");
                                                int _poll_6 = _poll_6_first;
                                                while (_poll_6 < required_7) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int _poll_6_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_6_next) : "l"(x_ready + global_mini_4) : "memory");
                                                    _poll_6 = _poll_6_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        unsigned int previous_4 = phase_bits_5 >> 7 & 1;
                                        unsigned int bits_5 = phase_bits_5;
                                        if (previous_4 != 1) {
                                            mbarrier_wait(gemm_finished_addr, bits_5 >> 16 & 1);
                                            mbarrier_wait(gemm_finished_addr + 8, bits_5 >> 17 & 1);
                                            mbarrier_wait(gemm_finished_addr + 16, bits_5 >> 18 & 1);
                                            mbarrier_wait(gemm_finished_addr + 24, bits_5 >> 19 & 1);
                                            mbarrier_wait(gemm_finished_addr + 32, bits_5 >> 20 & 1);
                                            mbarrier_wait(gemm_finished_addr + 40, bits_5 >> 21 & 1);
                                            bits_5 = bits_5 ^ 128;
                                        }
                                        phase_bits_5 = bits_5;
                                        int ring_7 = 0;
                                        #pragma unroll 1
                                        for (int idx_7 = 0; idx_7 < iterations_4; idx_7++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_7) * 8, phase_bits_5 >> (unsigned int)(16 + ring_7) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v38_addr + (unsigned int)(ring_7 * 16384)), "l"((&x_q)), "r"(0), "r"(td_x * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v39_addr + (unsigned int)(ring_7 * 16384)), "l"((&wu_q)), "r"(0), "r"(td_y * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(td_expert), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_7);
                                            ring_7 = (ring_7 + 1) % 6;
                                        }
                                    }
                                }
                            } else if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_390 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                            int mini_rows_8 = ((0) > (_min_390) ? (0) : (_min_390));
                                            int required_8 = (mini_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_7 = 1;
                                            if (enabled_value_7 != 0) {
                                                int _poll_7_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_7_first) : "l"(x_ready + global_mini_4) : "memory");
                                                int _poll_7 = _poll_7_first;
                                                while (_poll_7 < required_8) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int _poll_7_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_7_next) : "l"(x_ready + global_mini_4) : "memory");
                                                    _poll_7 = _poll_7_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        unsigned int previous_5 = phase_bits_5 >> 7 & 1;
                                        unsigned int bits_6 = phase_bits_5;
                                        if (previous_5 != 1) {
                                            mbarrier_wait(scales_finished_addr, bits_6 >> 16 & 1);
                                            mbarrier_wait(scales_finished_addr + 8, bits_6 >> 17 & 1);
                                            mbarrier_wait(scales_finished_addr + 16, bits_6 >> 18 & 1);
                                            mbarrier_wait(scales_finished_addr + 24, bits_6 >> 19 & 1);
                                            mbarrier_wait(scales_finished_addr + 32, bits_6 >> 20 & 1);
                                            mbarrier_wait(scales_finished_addr + 40, bits_6 >> 21 & 1);
                                            bits_6 = bits_6 ^ 128;
                                        }
                                        phase_bits_5 = bits_6;
                                        int ring_8 = 0;
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_4; idx_8++) {
                                            mbarrier_wait(scales_finished_addr + (ring_8) * 8, phase_bits_5 >> (unsigned int)(16 + ring_8) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v40_addr + (unsigned int)(ring_8 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((td_x * 2 + cta_rank_0) * (hidden / 128) + idx_8),
                                                   "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v41_addr + (unsigned int)(ring_8 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((td_expert * (intermediate / 128) + td_y * 2 + cta_rank_0) * (hidden / 128) + idx_8),
                                                   "r"(((scales_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_8);
                                            ring_8 = (ring_8 + 1) % 6;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_9 = 0;
                                            mbarrier_wait(output_finished_addr + (acc_buffer_1) * 8, phase_bits_5 >> (unsigned int)(22 + acc_buffer_1) & 1);
                                            mbarrier_wait(prefix_released_addr + (acc_buffer_1) * 8, phase_bits_5 >> (unsigned int)(22 + acc_buffer_1) & 1);
                                            mbarrier_wait(prefix_released_addr + (other_buffer_1) * 8, phase_bits_5 >> (unsigned int)(22 + other_buffer_1) & 1);
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 22 + acc_buffer_1);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_9 = 0; idx_9 < iterations_4; idx_9++) {
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_9) * 8, 3072);
                                                mbarrier_wait(scales_arrived_addr + (ring_9) * 8, phase_bits_5 >> (unsigned int)(8 + ring_9) & 1);
                                                int buffer_1 = idx_9 % 3;
                                                int sfa_slot_1 = buffer_1 + other_buffer_1 * 64;
                                                int sfb_slot_1 = buffer_1 + other_buffer_1 * 32;
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + sfa_slot_1 * 4, make_sf_cp_desc_sbo128(smem_v40_addr + (unsigned int)(ring_9 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + sfb_slot_1 * 8, make_sf_cp_desc_sbo128(smem_v41_addr + (unsigned int)(ring_9 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + sfb_slot_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v41_addr + (unsigned int)(ring_9 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_9) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_9) * 8, phase_bits_5 >> (unsigned int)ring_9 & 1);
                                                int _smem_desc_lo_391 = (((smem_v38_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                                int _smem_desc_lo_392 = (((smem_v39_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_391) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_392) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_1 * 256)), a_desc + 0, b_desc + 0,
                                                        0x10c00000U, tmem_tmem_sfa + sfa_slot_1 * 4, tmem_tmem_sfb + sfb_slot_1 * 8, ((idx_9 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_1 * 256)), a_desc + 2, b_desc + 2,
                                                        0x30c00010U, tmem_tmem_sfa + sfa_slot_1 * 4, tmem_tmem_sfb + sfb_slot_1 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_1 * 256)), a_desc + 4, b_desc + 4,
                                                        0x50c00020U, tmem_tmem_sfa + sfa_slot_1 * 4, tmem_tmem_sfb + sfb_slot_1 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_1 * 256)), a_desc + 6, b_desc + 6,
                                                        0x70c00030U, tmem_tmem_sfa + sfa_slot_1 * 4, tmem_tmem_sfb + sfb_slot_1 * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                                phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << ring_9) ^ (unsigned int)(1 << 8 + ring_9);
                                                ring_9 = (ring_9 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    unsigned int _parity_output_arrived_3 = (phase_bits_5 >> 6) % 2;
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_3);
                                    int warp_row_1 = tid / 32 * 32;
                                    unsigned int packed_2[128];
                                    #pragma unroll
                                    for (int i_65 = 0; i_65 < 8; i_65++) {
                                        float _tmem_load_35[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(_tmem_load_35[0]), "=f"(_tmem_load_35[1]), "=f"(_tmem_load_35[2]), "=f"(_tmem_load_35[3]), "=f"(_tmem_load_35[4]), "=f"(_tmem_load_35[5]), "=f"(_tmem_load_35[6]), "=f"(_tmem_load_35[7]), "=f"(_tmem_load_35[8]), "=f"(_tmem_load_35[9]), "=f"(_tmem_load_35[10]), "=f"(_tmem_load_35[11]), "=f"(_tmem_load_35[12]), "=f"(_tmem_load_35[13]), "=f"(_tmem_load_35[14]), "=f"(_tmem_load_35[15]), "=f"(_tmem_load_35[16]), "=f"(_tmem_load_35[17]), "=f"(_tmem_load_35[18]), "=f"(_tmem_load_35[19]), "=f"(_tmem_load_35[20]), "=f"(_tmem_load_35[21]), "=f"(_tmem_load_35[22]), "=f"(_tmem_load_35[23]), "=f"(_tmem_load_35[24]), "=f"(_tmem_load_35[25]), "=f"(_tmem_load_35[26]), "=f"(_tmem_load_35[27]), "=f"(_tmem_load_35[28]), "=f"(_tmem_load_35[29]), "=f"(_tmem_load_35[30]), "=f"(_tmem_load_35[31])
                                            : "r"(acc_taddr_1 + (unsigned int)(warp_row_1 << 16) + (unsigned int)(i_65 * 32)));
                                        #pragma unroll
                                        for (int j_33 = 0; j_33 < 16; j_33++) {
                                            packed_2[i_65 * 16 + j_33] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_35[2 * j_33], _tmem_load_35[2 * j_33 + 1])));
                                        }
                                        if (i_65 == 1) {
                                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((prefix_released_addr + (acc_buffer_1) * 8) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + (acc_buffer_1) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                    unsigned int scale_word_9 = 0;
                                    unsigned int block_3[16];
                                    #pragma unroll
                                    for (int j_34 = 0; j_34 < 16; j_34++) {
                                        block_3[j_34] = packed_2[j_34];
                                    }
                                    #pragma unroll
                                    for (int j_35 = 0; j_35 < 4; j_35++) {
                                        unsigned int address_16 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_35 * 16);
                                        address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_16 - smem_v42_addr)), "r"(packed_2[4 * j_35]), "r"(packed_2[4 * j_35 + 1]), "r"(packed_2[4 * j_35 + 2]), "r"(packed_2[4 * j_35 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_64;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_64) : "r"(block_3[0]));
                                    unsigned int amax_pair_18 = _bf16x2_abs_64;
                                    #pragma unroll
                                    for (int i_66 = 1; i_66 < 16; i_66++) {
                                        uint32_t _bf16x2_abs_65;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_65) : "r"(block_3[i_66]));
                                        uint32_t _bf16x2_max_32;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_32) : "r"(amax_pair_18), "r"(_bf16x2_abs_65));
                                        amax_pair_18 = _bf16x2_max_32;
                                    }
                                    uint16_t _bf16_max_32;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_32) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                    float amax_19;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_19) : "h"((uint16_t)(_bf16_max_32)));
                                    float scale_18 = fmaxf(amax_19 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_393;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_393) : "f"(scale_18), "f"(scale_18));
                                    unsigned int scale_byte_18 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_393 & 255;
                                    unsigned int inverse_lane_18 = 254 - scale_byte_18 << 7;
                                    unsigned int inverse_18 = inverse_lane_18 | inverse_lane_18 << 16;
                                    unsigned int words_18[8];
                                    #pragma unroll
                                    for (int i_67 = 0; i_67 < 8; i_67++) {
                                        uint32_t _bf16x2_mul_64;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(block_3[i_67 * 2]), "r"(inverse_18));
                                        uint16_t _e4m3x2_bf16x2_32;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_32) : "r"(_bf16x2_mul_64));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_64);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_32) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_65;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(block_3[i_67 * 2 + 1]), "r"(inverse_18));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_394;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_394) : "r"(_bf16x2_mul_65));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_65);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_394) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_18[i_67] = (unsigned int)_e4m3x2_bf16x2_32 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_394 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_18;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_18[0]), "r"(words_18[1]), "r"(words_18[2]), "r"(words_18[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_18[4]), "r"(words_18[5]), "r"(words_18[6]), "r"(words_18[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_0_1[16];
                                    #pragma unroll
                                    for (int j_36 = 0; j_36 < 16; j_36++) {
                                        block_0_1[j_36] = packed_2[16 + j_36];
                                    }
                                    #pragma unroll
                                    for (int j_37 = 0; j_37 < 4; j_37++) {
                                        unsigned int address_17 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_37 * 16);
                                        address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_17 - smem_v42_addr)), "r"(packed_2[16 + 4 * j_37]), "r"(packed_2[16 + 4 * j_37 + 1]), "r"(packed_2[16 + 4 * j_37 + 2]), "r"(packed_2[16 + 4 * j_37 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 1, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_66;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_66) : "r"(block_0_1[0]));
                                    unsigned int amax_pair_1_2 = _bf16x2_abs_66;
                                    #pragma unroll
                                    for (int i_68 = 1; i_68 < 16; i_68++) {
                                        uint32_t _bf16x2_abs_67;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_67) : "r"(block_0_1[i_68]));
                                        uint32_t _bf16x2_max_33;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_33) : "r"(amax_pair_1_2), "r"(_bf16x2_abs_67));
                                        amax_pair_1_2 = _bf16x2_max_33;
                                    }
                                    uint16_t _bf16_max_33;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_33) : "h"((uint16_t)(amax_pair_1_2 & 65535)), "h"((uint16_t)(amax_pair_1_2 >> 16)));
                                    float amax_2_2;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2_2) : "h"((uint16_t)(_bf16_max_33)));
                                    float scale_3_2 = fmaxf(amax_2_2 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_395;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_395) : "f"(scale_3_2), "f"(scale_3_2));
                                    unsigned int scale_byte_4_2 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_395 & 255;
                                    unsigned int inverse_lane_5_2 = 254 - scale_byte_4_2 << 7;
                                    unsigned int inverse_6_2 = inverse_lane_5_2 | inverse_lane_5_2 << 16;
                                    unsigned int words_7_2[8];
                                    #pragma unroll
                                    for (int i_69 = 0; i_69 < 8; i_69++) {
                                        uint32_t _bf16x2_mul_66;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(block_0_1[i_69 * 2]), "r"(inverse_6_2));
                                        uint16_t _e4m3x2_bf16x2_33;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_33) : "r"(_bf16x2_mul_66));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_66);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_33) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_67;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(block_0_1[i_69 * 2 + 1]), "r"(inverse_6_2));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_396;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_396) : "r"(_bf16x2_mul_67));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_67);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_396) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_7_2[i_69] = (unsigned int)_e4m3x2_bf16x2_33 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_396 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_4_2 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_7_2[0]), "r"(words_7_2[1]), "r"(words_7_2[2]), "r"(words_7_2[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_7_2[4]), "r"(words_7_2[5]), "r"(words_7_2[6]), "r"(words_7_2[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 32), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_8_1[16];
                                    #pragma unroll
                                    for (int j_38 = 0; j_38 < 16; j_38++) {
                                        block_8_1[j_38] = packed_2[32 + j_38];
                                    }
                                    #pragma unroll
                                    for (int j_39 = 0; j_39 < 4; j_39++) {
                                        unsigned int address_18 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_39 * 16);
                                        address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_18 - smem_v42_addr)), "r"(packed_2[32 + 4 * j_39]), "r"(packed_2[32 + 4 * j_39 + 1]), "r"(packed_2[32 + 4 * j_39 + 2]), "r"(packed_2[32 + 4 * j_39 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 2, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_68;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_68) : "r"(block_8_1[0]));
                                    unsigned int amax_pair_9_2 = _bf16x2_abs_68;
                                    #pragma unroll
                                    for (int i_70 = 1; i_70 < 16; i_70++) {
                                        uint32_t _bf16x2_abs_69;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_69) : "r"(block_8_1[i_70]));
                                        uint32_t _bf16x2_max_34;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_34) : "r"(amax_pair_9_2), "r"(_bf16x2_abs_69));
                                        amax_pair_9_2 = _bf16x2_max_34;
                                    }
                                    uint16_t _bf16_max_34;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_34) : "h"((uint16_t)(amax_pair_9_2 & 65535)), "h"((uint16_t)(amax_pair_9_2 >> 16)));
                                    float amax_10_2;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10_2) : "h"((uint16_t)(_bf16_max_34)));
                                    float scale_11_2 = fmaxf(amax_10_2 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_397;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_397) : "f"(scale_11_2), "f"(scale_11_2));
                                    unsigned int scale_byte_12_2 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_397 & 255;
                                    unsigned int inverse_lane_13_2 = 254 - scale_byte_12_2 << 7;
                                    unsigned int inverse_14_2 = inverse_lane_13_2 | inverse_lane_13_2 << 16;
                                    unsigned int words_15_2[8];
                                    #pragma unroll
                                    for (int i_71 = 0; i_71 < 8; i_71++) {
                                        uint32_t _bf16x2_mul_68;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(block_8_1[i_71 * 2]), "r"(inverse_14_2));
                                        uint16_t _e4m3x2_bf16x2_34;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_34) : "r"(_bf16x2_mul_68));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_68);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_34) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_69;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(block_8_1[i_71 * 2 + 1]), "r"(inverse_14_2));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_398;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_398) : "r"(_bf16x2_mul_69));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_69);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_398) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_15_2[i_71] = (unsigned int)_e4m3x2_bf16x2_34 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_398 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_12_2 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_15_2[0]), "r"(words_15_2[1]), "r"(words_15_2[2]), "r"(words_15_2[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_15_2[4]), "r"(words_15_2[5]), "r"(words_15_2[6]), "r"(words_15_2[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 64), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_16_1[16];
                                    #pragma unroll
                                    for (int j_40 = 0; j_40 < 16; j_40++) {
                                        block_16_1[j_40] = packed_2[48 + j_40];
                                    }
                                    #pragma unroll
                                    for (int j_41 = 0; j_41 < 4; j_41++) {
                                        unsigned int address_19 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_41 * 16);
                                        address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_19 - smem_v42_addr)), "r"(packed_2[48 + 4 * j_41]), "r"(packed_2[48 + 4 * j_41 + 1]), "r"(packed_2[48 + 4 * j_41 + 2]), "r"(packed_2[48 + 4 * j_41 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 3, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_70;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_70) : "r"(block_16_1[0]));
                                    unsigned int amax_pair_17_1 = _bf16x2_abs_70;
                                    #pragma unroll
                                    for (int i_72 = 1; i_72 < 16; i_72++) {
                                        uint32_t _bf16x2_abs_71;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_71) : "r"(block_16_1[i_72]));
                                        uint32_t _bf16x2_max_35;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_35) : "r"(amax_pair_17_1), "r"(_bf16x2_abs_71));
                                        amax_pair_17_1 = _bf16x2_max_35;
                                    }
                                    uint16_t _bf16_max_35;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_35) : "h"((uint16_t)(amax_pair_17_1 & 65535)), "h"((uint16_t)(amax_pair_17_1 >> 16)));
                                    float amax_18_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18_1) : "h"((uint16_t)(_bf16_max_35)));
                                    float scale_19_1 = fmaxf(amax_18_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_399;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_399) : "f"(scale_19_1), "f"(scale_19_1));
                                    unsigned int scale_byte_20_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_399 & 255;
                                    unsigned int inverse_lane_21_1 = 254 - scale_byte_20_1 << 7;
                                    unsigned int inverse_22_1 = inverse_lane_21_1 | inverse_lane_21_1 << 16;
                                    unsigned int words_23_1[8];
                                    #pragma unroll
                                    for (int i_73 = 0; i_73 < 8; i_73++) {
                                        uint32_t _bf16x2_mul_70;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(block_16_1[i_73 * 2]), "r"(inverse_22_1));
                                        uint16_t _e4m3x2_bf16x2_35;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_35) : "r"(_bf16x2_mul_70));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_70);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_35) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_71;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(block_16_1[i_73 * 2 + 1]), "r"(inverse_22_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_400;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_400) : "r"(_bf16x2_mul_71));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_71);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_400) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_23_1[i_73] = (unsigned int)_e4m3x2_bf16x2_35 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_400 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_20_1 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_23_1[0]), "r"(words_23_1[1]), "r"(words_23_1[2]), "r"(words_23_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_23_1[4]), "r"(words_23_1[5]), "r"(words_23_1[6]), "r"(words_23_1[7]) : "memory");
                                    smem_v44[tid % 32 * 4 + tid / 32] = scale_word_9;
                                    scale_word_9 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 96), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&up_sc_store), 0, 0, (td_x * 2 + cta_rank_0) * i_tiles + td_y * 2, smem_v44_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_24_1[16];
                                    #pragma unroll
                                    for (int j_42 = 0; j_42 < 16; j_42++) {
                                        block_24_1[j_42] = packed_2[64 + j_42];
                                    }
                                    #pragma unroll
                                    for (int j_43 = 0; j_43 < 4; j_43++) {
                                        unsigned int address_20 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_43 * 16);
                                        address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_20 - smem_v42_addr)), "r"(packed_2[64 + 4 * j_43]), "r"(packed_2[64 + 4 * j_43 + 1]), "r"(packed_2[64 + 4 * j_43 + 2]), "r"(packed_2[64 + 4 * j_43 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 4, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_72;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_72) : "r"(block_24_1[0]));
                                    unsigned int amax_pair_25_1 = _bf16x2_abs_72;
                                    #pragma unroll
                                    for (int i_74 = 1; i_74 < 16; i_74++) {
                                        uint32_t _bf16x2_abs_73;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_73) : "r"(block_24_1[i_74]));
                                        uint32_t _bf16x2_max_36;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_36) : "r"(amax_pair_25_1), "r"(_bf16x2_abs_73));
                                        amax_pair_25_1 = _bf16x2_max_36;
                                    }
                                    uint16_t _bf16_max_36;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_36) : "h"((uint16_t)(amax_pair_25_1 & 65535)), "h"((uint16_t)(amax_pair_25_1 >> 16)));
                                    float amax_26_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_26_1) : "h"((uint16_t)(_bf16_max_36)));
                                    float scale_27_1 = fmaxf(amax_26_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_401;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_401) : "f"(scale_27_1), "f"(scale_27_1));
                                    unsigned int scale_byte_28_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_401 & 255;
                                    unsigned int inverse_lane_29_1 = 254 - scale_byte_28_1 << 7;
                                    unsigned int inverse_30_1 = inverse_lane_29_1 | inverse_lane_29_1 << 16;
                                    unsigned int words_31_1[8];
                                    #pragma unroll
                                    for (int i_75 = 0; i_75 < 8; i_75++) {
                                        uint32_t _bf16x2_mul_72;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(block_24_1[i_75 * 2]), "r"(inverse_30_1));
                                        uint16_t _e4m3x2_bf16x2_36;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_36) : "r"(_bf16x2_mul_72));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_72);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_36) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_73;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(block_24_1[i_75 * 2 + 1]), "r"(inverse_30_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_402;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_402) : "r"(_bf16x2_mul_73));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_73);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_402) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_31_1[i_75] = (unsigned int)_e4m3x2_bf16x2_36 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_402 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_28_1;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_31_1[0]), "r"(words_31_1[1]), "r"(words_31_1[2]), "r"(words_31_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_31_1[4]), "r"(words_31_1[5]), "r"(words_31_1[6]), "r"(words_31_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 128), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_32_1[16];
                                    #pragma unroll
                                    for (int j_44 = 0; j_44 < 16; j_44++) {
                                        block_32_1[j_44] = packed_2[80 + j_44];
                                    }
                                    #pragma unroll
                                    for (int j_45 = 0; j_45 < 4; j_45++) {
                                        unsigned int address_21 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_45 * 16);
                                        address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_21 - smem_v42_addr)), "r"(packed_2[80 + 4 * j_45]), "r"(packed_2[80 + 4 * j_45 + 1]), "r"(packed_2[80 + 4 * j_45 + 2]), "r"(packed_2[80 + 4 * j_45 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 5, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_74;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_74) : "r"(block_32_1[0]));
                                    unsigned int amax_pair_33_1 = _bf16x2_abs_74;
                                    #pragma unroll
                                    for (int i_76 = 1; i_76 < 16; i_76++) {
                                        uint32_t _bf16x2_abs_75;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_75) : "r"(block_32_1[i_76]));
                                        uint32_t _bf16x2_max_37;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_37) : "r"(amax_pair_33_1), "r"(_bf16x2_abs_75));
                                        amax_pair_33_1 = _bf16x2_max_37;
                                    }
                                    uint16_t _bf16_max_37;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_37) : "h"((uint16_t)(amax_pair_33_1 & 65535)), "h"((uint16_t)(amax_pair_33_1 >> 16)));
                                    float amax_34_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_34_1) : "h"((uint16_t)(_bf16_max_37)));
                                    float scale_35_1 = fmaxf(amax_34_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_403;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_403) : "f"(scale_35_1), "f"(scale_35_1));
                                    unsigned int scale_byte_36_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_403 & 255;
                                    unsigned int inverse_lane_37_1 = 254 - scale_byte_36_1 << 7;
                                    unsigned int inverse_38_1 = inverse_lane_37_1 | inverse_lane_37_1 << 16;
                                    unsigned int words_39_1[8];
                                    #pragma unroll
                                    for (int i_77 = 0; i_77 < 8; i_77++) {
                                        uint32_t _bf16x2_mul_74;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(block_32_1[i_77 * 2]), "r"(inverse_38_1));
                                        uint16_t _e4m3x2_bf16x2_37;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_37) : "r"(_bf16x2_mul_74));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_74);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_37) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_75;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(block_32_1[i_77 * 2 + 1]), "r"(inverse_38_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_404;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_404) : "r"(_bf16x2_mul_75));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_75);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_404) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_39_1[i_77] = (unsigned int)_e4m3x2_bf16x2_37 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_404 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_36_1 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_39_1[0]), "r"(words_39_1[1]), "r"(words_39_1[2]), "r"(words_39_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_39_1[4]), "r"(words_39_1[5]), "r"(words_39_1[6]), "r"(words_39_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 160), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_40_1[16];
                                    #pragma unroll
                                    for (int j_46 = 0; j_46 < 16; j_46++) {
                                        block_40_1[j_46] = packed_2[96 + j_46];
                                    }
                                    #pragma unroll
                                    for (int j_47 = 0; j_47 < 4; j_47++) {
                                        unsigned int address_22 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_47 * 16);
                                        address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_22 - smem_v42_addr)), "r"(packed_2[96 + 4 * j_47]), "r"(packed_2[96 + 4 * j_47 + 1]), "r"(packed_2[96 + 4 * j_47 + 2]), "r"(packed_2[96 + 4 * j_47 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 6, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_76;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_76) : "r"(block_40_1[0]));
                                    unsigned int amax_pair_41_1 = _bf16x2_abs_76;
                                    #pragma unroll
                                    for (int i_78 = 1; i_78 < 16; i_78++) {
                                        uint32_t _bf16x2_abs_77;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_77) : "r"(block_40_1[i_78]));
                                        uint32_t _bf16x2_max_38;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_38) : "r"(amax_pair_41_1), "r"(_bf16x2_abs_77));
                                        amax_pair_41_1 = _bf16x2_max_38;
                                    }
                                    uint16_t _bf16_max_38;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_38) : "h"((uint16_t)(amax_pair_41_1 & 65535)), "h"((uint16_t)(amax_pair_41_1 >> 16)));
                                    float amax_42_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_42_1) : "h"((uint16_t)(_bf16_max_38)));
                                    float scale_43_1 = fmaxf(amax_42_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_405;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_405) : "f"(scale_43_1), "f"(scale_43_1));
                                    unsigned int scale_byte_44_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_405 & 255;
                                    unsigned int inverse_lane_45_1 = 254 - scale_byte_44_1 << 7;
                                    unsigned int inverse_46_1 = inverse_lane_45_1 | inverse_lane_45_1 << 16;
                                    unsigned int words_47_1[8];
                                    #pragma unroll
                                    for (int i_79 = 0; i_79 < 8; i_79++) {
                                        uint32_t _bf16x2_mul_76;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(block_40_1[i_79 * 2]), "r"(inverse_46_1));
                                        uint16_t _e4m3x2_bf16x2_38;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_38) : "r"(_bf16x2_mul_76));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_76);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_38) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_77;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(block_40_1[i_79 * 2 + 1]), "r"(inverse_46_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_406;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_406) : "r"(_bf16x2_mul_77));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_77);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_406) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_47_1[i_79] = (unsigned int)_e4m3x2_bf16x2_38 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_406 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_44_1 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_47_1[0]), "r"(words_47_1[1]), "r"(words_47_1[2]), "r"(words_47_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_47_1[4]), "r"(words_47_1[5]), "r"(words_47_1[6]), "r"(words_47_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 192), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_48_1[16];
                                    #pragma unroll
                                    for (int j_48 = 0; j_48 < 16; j_48++) {
                                        block_48_1[j_48] = packed_2[112 + j_48];
                                    }
                                    #pragma unroll
                                    for (int j_49 = 0; j_49 < 4; j_49++) {
                                        unsigned int address_23 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_49 * 16);
                                        address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_23 - smem_v42_addr)), "r"(packed_2[112 + 4 * j_49]), "r"(packed_2[112 + 4 * j_49 + 1]), "r"(packed_2[112 + 4 * j_49 + 2]), "r"(packed_2[112 + 4 * j_49 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + 7, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t _bf16x2_abs_78;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_78) : "r"(block_48_1[0]));
                                    unsigned int amax_pair_49_1 = _bf16x2_abs_78;
                                    #pragma unroll
                                    for (int i_80 = 1; i_80 < 16; i_80++) {
                                        uint32_t _bf16x2_abs_79;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_79) : "r"(block_48_1[i_80]));
                                        uint32_t _bf16x2_max_39;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_39) : "r"(amax_pair_49_1), "r"(_bf16x2_abs_79));
                                        amax_pair_49_1 = _bf16x2_max_39;
                                    }
                                    uint16_t _bf16_max_39;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_39) : "h"((uint16_t)(amax_pair_49_1 & 65535)), "h"((uint16_t)(amax_pair_49_1 >> 16)));
                                    float amax_50_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_50_1) : "h"((uint16_t)(_bf16_max_39)));
                                    float scale_51_1 = fmaxf(amax_50_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_407;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_407) : "f"(scale_51_1), "f"(scale_51_1));
                                    unsigned int scale_byte_52_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_407 & 255;
                                    unsigned int inverse_lane_53_1 = 254 - scale_byte_52_1 << 7;
                                    unsigned int inverse_54_1 = inverse_lane_53_1 | inverse_lane_53_1 << 16;
                                    unsigned int words_55_1[8];
                                    #pragma unroll
                                    for (int i_81 = 0; i_81 < 8; i_81++) {
                                        uint32_t _bf16x2_mul_78;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(block_48_1[i_81 * 2]), "r"(inverse_54_1));
                                        uint16_t _e4m3x2_bf16x2_39;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_39) : "r"(_bf16x2_mul_78));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_78);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_39) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _bf16x2_mul_79;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(block_48_1[i_81 * 2 + 1]), "r"(inverse_54_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_408;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_408) : "r"(_bf16x2_mul_79));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_79);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_408) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_55_1[i_81] = (unsigned int)_e4m3x2_bf16x2_39 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_408 << 16;
                                    }
                                    scale_word_9 = scale_word_9 | scale_byte_52_1 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_55_1[0]), "r"(words_55_1[1]), "r"(words_55_1[2]), "r"(words_55_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_55_1[4]), "r"(words_55_1[5]), "r"(words_55_1[6]), "r"(words_55_1[7]) : "memory");
                                    smem_v45[tid % 32 * 4 + tid / 32] = scale_word_9;
                                    scale_word_9 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(td_y * 256 + 224), "r"(td_x * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&up_sc_store), 0, 0, (td_x * 2 + cta_rank_0) * i_tiles + td_y * 2 + 1, smem_v45_addr);
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
                                                bool enabled_value_8 = 1;
                                                if (enabled_value_8 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (gate_base + (macro_rows_5 + td_x) * (intermediate / 256) + td_y))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            phase_bits_5 = phase_bits_5 ^ 16384;
                        }
                        gemm_bits = phase_bits_5;
                    }
                } else if (task_0 < mini_gate + mini_swiglu) {
                    unsigned int phase_bits_6 = swiglu_bits;
                    int col_blocks_6 = intermediate / 128;
                    int num_tiles = tokens / 128 * col_blocks_6;
                    int macro_row_offset = macro_1_1 * (macro_size / 128);
                    int first_tile_2 = (task_0 - mini_gate) * 6 + cta_rank_0 * 3;
                    int global_mini_5 = macro_1_1 * (macro_size / mini_size) + mini_2_1;
                    int mini_tiles = mini_size / 128 * col_blocks_6;
                    first_tile_2 = first_tile_2 + global_mini_5 * mini_tiles;
                    int tile_end = ((num_tiles) < ((global_mini_5 + 1) * mini_tiles) ? (num_tiles) : ((global_mini_5 + 1) * mini_tiles));
                    int macro_tiles_2 = macro_size / 128;
                    if (first_tile_2 < tile_end) {
                        int first_row_4 = first_tile_2 / col_blocks_6;
                        int first_col_2 = first_tile_2 % col_blocks_6;
                        if (tid == 0) {
                            if (tile_end > first_tile_2) {
                                int row_34 = first_row_4;
                                int col_62 = first_col_2;
                                if (col_62 >= col_blocks_6) {
                                    row_34 = row_34 + 1;
                                    col_62 = col_62 - col_blocks_6;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr, 65536);
                                int parent = row_34 / 2 * (intermediate / 256) + col_62 / 2;
                                int _poll_8_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_8_first) : "l"(gate_ready + (gate_base + parent)) : "memory");
                                int _poll_8 = _poll_8_first;
                                while (_poll_8 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int _poll_8_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_8_next) : "l"(gate_ready + (gate_base + parent)) : "memory");
                                    _poll_8 = _poll_8_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr), "l"((&gate_routed_in)), "r"(0), "r"((row_34 - macro_row_offset) * 128), "r"(col_62 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr), "l"((&up_routed_in)), "r"(0), "r"((row_34 - macro_row_offset) * 128), "r"(col_62 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr) : "memory");
                            }
                            if (tile_end > first_tile_2 + 1) {
                                int row_35 = first_row_4;
                                int col_63 = first_col_2 + 1;
                                if (col_63 >= col_blocks_6) {
                                    row_35 = row_35 + 1;
                                    col_63 = col_63 - col_blocks_6;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + 8, 65536);
                                int parent_1 = row_35 / 2 * (intermediate / 256) + col_63 / 2;
                                int _poll_9_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_9_first) : "l"(gate_ready + (gate_base + parent_1)) : "memory");
                                int _poll_9 = _poll_9_first;
                                while (_poll_9 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int _poll_9_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_9_next) : "l"(gate_ready + (gate_base + parent_1)) : "memory");
                                    _poll_9 = _poll_9_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + 32768), "l"((&gate_routed_in)), "r"(0), "r"((row_35 - macro_row_offset) * 128), "r"(col_63 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + 32768), "l"((&up_routed_in)), "r"(0), "r"((row_35 - macro_row_offset) * 128), "r"(col_63 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 8) : "memory");
                            }
                            if (tile_end > first_tile_2 + 2) {
                                int row_36 = first_row_4;
                                int col_64 = first_col_2 + 2;
                                if (col_64 >= col_blocks_6) {
                                    row_36 = row_36 + 1;
                                    col_64 = col_64 - col_blocks_6;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + 16, 65536);
                                int parent_2 = row_36 / 2 * (intermediate / 256) + col_64 / 2;
                                int _poll_10_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_10_first) : "l"(gate_ready + (gate_base + parent_2)) : "memory");
                                int _poll_10 = _poll_10_first;
                                while (_poll_10 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int _poll_10_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_10_next) : "l"(gate_ready + (gate_base + parent_2)) : "memory");
                                    _poll_10 = _poll_10_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + 65536), "l"((&gate_routed_in)), "r"(0), "r"((row_36 - macro_row_offset) * 128), "r"(col_64 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 16) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + 65536), "l"((&up_routed_in)), "r"(0), "r"((row_36 - macro_row_offset) * 128), "r"(col_64 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + 16) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int stage_8 = 0; stage_8 < 3; stage_8++) {
                            if (tile_end > first_tile_2 + stage_8) {
                                mbarrier_wait(swiglu_arrived_addr + (stage_8) * 8, phase_bits_6 >> (unsigned int)stage_8 & 1);
                                phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << stage_8);
                                int row_37 = first_row_4;
                                int col_65 = first_col_2 + stage_8;
                                if (col_65 >= col_blocks_6) {
                                    row_37 = row_37 + 1;
                                    col_65 = col_65 - col_blocks_6;
                                }
                                float gate[64];
                                float up[64];
                                float denominator[64];
                                int warp_0_2 = tid / 32;
                                int local_warp = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                                int lane_4 = tid % 32;
                                #pragma unroll
                                for (int tile_col = 0; tile_col < 8; tile_col++) {
                                    unsigned int packed_3[4];
                                    unsigned int address_24 = gate_smem_addr + (unsigned int)(stage_8 * 32768) + (unsigned int)(((tile_col * 16 + lane_4 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_4 % 16) * 64 + (tile_col * 16 + lane_4 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                        : "r"(address_24 ^ (address_24 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                        float2 _cvt_f32_256 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_3[pair_2]));
                                        gate[tile_col * 8 + pair_2 * 2] = _cvt_f32_256.x;
                                        gate[tile_col * 8 + pair_2 * 2 + 1] = _cvt_f32_256.y;
                                    }
                                }
                                int warp_1_1 = tid / 32;
                                int local_warp_2 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                int lane_3_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                    unsigned int packed_4[4];
                                    unsigned int address_25 = up_smem_addr + (unsigned int)(stage_8 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3_1 % 16) * 64 + (tile_col_1 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_4[0]), "=r"(packed_4[1]), "=r"(packed_4[2]), "=r"(packed_4[3])
                                        : "r"(address_25 ^ (address_25 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                        float2 _cvt_f32_257 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_4[pair_3]));
                                        up[tile_col_1 * 8 + pair_3 * 2] = _cvt_f32_257.x;
                                        up[tile_col_1 * 8 + pair_3 * 2 + 1] = _cvt_f32_257.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem = 0; elem < 64; elem++) {
                                    denominator[elem] = gate[elem] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                    float _exp_409 = expf(denominator[elem_1]);
                                    denominator[elem_1] = _exp_409;
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
                                __syncthreads();
                                int warp_4 = tid / 32;
                                int local_warp_5 = warp_4 / 4 + warp_4 % 4 * 2;
                                int lane_6 = tid % 32;
                                #pragma unroll
                                for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                    unsigned int packed_5[4];
                                    #pragma unroll
                                    for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                        packed_5[pair_4] = __as_u32(__float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_4 * 2], gate[tile_col_2 * 8 + pair_4 * 2 + 1])));
                                    }
                                    int row_0_8 = local_warp_5 * 16 + lane_6 % 16;
                                    int col_1_1 = tile_col_2 * 16 + lane_6 / 16 * 8;
                                    uint32_t _stmatrix_addr_410 = static_cast<uint32_t>(hidden_staging_addr + (unsigned int)((row_0_8 * 136 + col_1_1) * 2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_410), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                int out_base = stage_8 * 8192;
                                if (tid < 128) {
                                    int r_8 = tid % 64;
                                    int row0_8 = r_8 * 2;
                                    int row1_8 = row0_8 + 1;
                                    int half_34 = tid / 64;
                                    int word_stride_8 = 68;
                                    int scale_base0_8 = (row0_8 % 32 * 4 + row0_8 / 32) * 4;
                                    int scale_base1_8 = (row1_8 % 32 * 4 + row1_8 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_50 = 0; j_50 < 2; j_50++) {
                                        int k_block_16 = (half_34 * 2 + j_50 + tid / 8) % 4;
                                        unsigned int pairs0_8[16];
                                        unsigned int pairs1_8[16];
                                        #pragma unroll
                                        for (int k_32 = 0; k_32 < 16; k_32++) {
                                            int col_0 = k_block_16 * 32 + (tid * 4 + k_32 * 2) % 32;
                                            unsigned int w0_1 = hidden_words[col_0 * word_stride_8 + r_8];
                                            unsigned int w1_2 = hidden_words[(col_0 + 1) * word_stride_8 + r_8];
                                            uint32_t lo_8;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo_8) : "r"(w0_1), "r"(w1_2));
                                            uint32_t hi_8;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi_8) : "r"(w0_1), "r"(w1_2));
                                            pairs0_8[k_32] = lo_8;
                                            pairs1_8[k_32] = hi_8;
                                        }
                                        uint32_t _bf16x2_abs_80;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_80) : "r"(pairs0_8[0]));
                                        unsigned int amax_pair_19 = _bf16x2_abs_80;
                                        #pragma unroll
                                        for (int i_82 = 1; i_82 < 16; i_82++) {
                                            uint32_t _bf16x2_abs_81;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_81) : "r"(pairs0_8[i_82]));
                                            uint32_t _bf16x2_max_40;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_40) : "r"(amax_pair_19), "r"(_bf16x2_abs_81));
                                            amax_pair_19 = _bf16x2_max_40;
                                        }
                                        uint16_t _bf16_max_40;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_40) : "h"((uint16_t)(amax_pair_19 & 65535)), "h"((uint16_t)(amax_pair_19 >> 16)));
                                        float amax_20;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_20) : "h"((uint16_t)(_bf16_max_40)));
                                        float scale_20 = fmaxf(amax_20 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_411;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_411) : "f"(scale_20), "f"(scale_20));
                                        unsigned int scale_byte_19 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_411 & 255;
                                        unsigned int inverse_lane_19 = 254 - scale_byte_19 << 7;
                                        unsigned int inverse_19 = inverse_lane_19 | inverse_lane_19 << 16;
                                        unsigned int words_19[8];
                                        #pragma unroll
                                        for (int i_83 = 0; i_83 < 8; i_83++) {
                                            uint32_t _bf16x2_mul_80;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(pairs0_8[i_83 * 2]), "r"(inverse_19));
                                            uint16_t _e4m3x2_bf16x2_40;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_40) : "r"(_bf16x2_mul_80));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_80);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_40) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_81;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(pairs0_8[i_83 * 2 + 1]), "r"(inverse_19));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_412;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_412) : "r"(_bf16x2_mul_81));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_81);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_412) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_19[i_83] = (unsigned int)_e4m3x2_bf16x2_40 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_412 << 16;
                                        }
                                        uint32_t _bf16x2_abs_82;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_82) : "r"(pairs1_8[0]));
                                        unsigned int amax_pair_0_8 = _bf16x2_abs_82;
                                        #pragma unroll
                                        for (int i_84 = 1; i_84 < 16; i_84++) {
                                            uint32_t _bf16x2_abs_83;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_83) : "r"(pairs1_8[i_84]));
                                            uint32_t _bf16x2_max_41;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_41) : "r"(amax_pair_0_8), "r"(_bf16x2_abs_83));
                                            amax_pair_0_8 = _bf16x2_max_41;
                                        }
                                        uint16_t _bf16_max_41;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_41) : "h"((uint16_t)(amax_pair_0_8 & 65535)), "h"((uint16_t)(amax_pair_0_8 >> 16)));
                                        float amax_1_8;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_8) : "h"((uint16_t)(_bf16_max_41)));
                                        float scale_2_8 = fmaxf(amax_1_8 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_413;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_413) : "f"(scale_2_8), "f"(scale_2_8));
                                        unsigned int scale_byte_3_8 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_413 & 255;
                                        unsigned int inverse_lane_4_8 = 254 - scale_byte_3_8 << 7;
                                        unsigned int inverse_5_8 = inverse_lane_4_8 | inverse_lane_4_8 << 16;
                                        unsigned int words_6_8[8];
                                        #pragma unroll
                                        for (int i_85 = 0; i_85 < 8; i_85++) {
                                            uint32_t _bf16x2_mul_82;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(pairs1_8[i_85 * 2]), "r"(inverse_5_8));
                                            uint16_t _e4m3x2_bf16x2_41;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_41) : "r"(_bf16x2_mul_82));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_82);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_41) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_83;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(pairs1_8[i_85 * 2 + 1]), "r"(inverse_5_8));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_414;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_414) : "r"(_bf16x2_mul_83));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_83);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_414) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_6_8[i_85] = (unsigned int)_e4m3x2_bf16x2_41 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_414 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_33 = 0; k_33 < 8; k_33++) {
                                            int col_0_1 = k_block_16 * 32 + (tid * 4 + k_33 * 4) % 32;
                                            smem_v10[out_base + (row0_8 * 128 + col_0_1) / 4] = words_19[k_33];
                                            smem_v10[out_base + (row1_8 * 128 + col_0_1) / 4] = words_6_8[k_33];
                                        }
                                        {
                                            uint32_t _byte_415 = static_cast<uint32_t>(scale_byte_19) & 0xFFu;
                                            uint32_t _addr_415 = static_cast<uint32_t>(smem_v11_addr + (unsigned int)(out_base * 4 + scale_base0_8 + k_block_16));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_415), "r"(_byte_415) : "memory");
                                        }
                                        {
                                            uint32_t _byte_416 = static_cast<uint32_t>(scale_byte_3_8) & 0xFFu;
                                            uint32_t _addr_416 = static_cast<uint32_t>(smem_v11_addr + (unsigned int)(out_base * 4 + scale_base1_8 + k_block_16));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_416), "r"(_byte_416) : "memory");
                                        }
                                    }
                                } else {
                                    int row_0_9 = tid - 128;
                                    unsigned int scale_word_10 = 0;
                                    #pragma unroll 1
                                    for (int j_51 = 0; j_51 < 4; j_51++) {
                                        int k_block_17 = (j_51 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_8[16];
                                        #pragma unroll
                                        for (int k_34 = 0; k_34 < 16; k_34++) {
                                            int col_0_2 = k_block_17 * 32 + ((tid - 128) * 4 + k_34 * 2) % 32;
                                            float x0_8 = 0.0f;
                                            float x1_8 = 0.0f;
                                            pairs_8[k_34] = hidden_words[(row_0_9 * 136 + col_0_2) / 2];
                                        }
                                        uint32_t _bf16x2_abs_84;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_84) : "r"(pairs_8[0]));
                                        unsigned int amax_pair_20 = _bf16x2_abs_84;
                                        #pragma unroll
                                        for (int i_86 = 1; i_86 < 16; i_86++) {
                                            uint32_t _bf16x2_abs_85;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_85) : "r"(pairs_8[i_86]));
                                            uint32_t _bf16x2_max_42;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_42) : "r"(amax_pair_20), "r"(_bf16x2_abs_85));
                                            amax_pair_20 = _bf16x2_max_42;
                                        }
                                        uint16_t _bf16_max_42;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_42) : "h"((uint16_t)(amax_pair_20 & 65535)), "h"((uint16_t)(amax_pair_20 >> 16)));
                                        float amax_21;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_21) : "h"((uint16_t)(_bf16_max_42)));
                                        float scale_21 = fmaxf(amax_21 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_417;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_417) : "f"(scale_21), "f"(scale_21));
                                        unsigned int scale_byte_21 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_417 & 255;
                                        unsigned int inverse_lane_20 = 254 - scale_byte_21 << 7;
                                        unsigned int inverse_20 = inverse_lane_20 | inverse_lane_20 << 16;
                                        unsigned int words_20[8];
                                        #pragma unroll
                                        for (int i_87 = 0; i_87 < 8; i_87++) {
                                            uint32_t _bf16x2_mul_84;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(pairs_8[i_87 * 2]), "r"(inverse_20));
                                            uint16_t _e4m3x2_bf16x2_42;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_e4m3x2_bf16x2_42) : "r"(_bf16x2_mul_84));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_84);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_bf16x2_42) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            uint32_t _bf16x2_mul_85;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(pairs_8[i_87 * 2 + 1]), "r"(inverse_20));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_418;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_418) : "r"(_bf16x2_mul_85));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(_bf16x2_mul_85);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_418) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_20[i_87] = (unsigned int)_e4m3x2_bf16x2_42 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_418 << 16;
                                        }
                                        scale_word_10 = scale_word_10 | scale_byte_21 << (unsigned int)(k_block_17 * 8);
                                        #pragma unroll
                                        for (int k_35 = 0; k_35 < 8; k_35++) {
                                            int col_0_3 = k_block_17 * 32 + ((tid - 128) * 4 + k_35 * 4) % 32;
                                            smem_v8[out_base + (row_0_9 * 128 + col_0_3) / 4] = words_20[k_35];
                                        }
                                    }
                                    smem_v9[out_base + row_0_9 % 32 * 4 + row_0_9 / 32] = scale_word_10;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row = row_37 - macro_row_offset;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&hidden_q_store), col_65 * 128, local_row * 128, smem_v8_addr + (unsigned int)(stage_8 * 32768));
                                    tma_store_3d((&hidden_sc_store), 0, 0, local_row * col_blocks_6 + col_65, smem_v9_addr + (unsigned int)(stage_8 * 32768));
                                    tma_store_2d((&hidden_t_store), local_row * 128, col_65 * 128, smem_v10_addr + (unsigned int)(stage_8 * 32768));
                                    tma_store_3d((&hidden_sc_t_store), 0, 0, col_65 * macro_tiles_2 + local_row, smem_v11_addr + (unsigned int)(stage_8 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            if (tile_end > first_tile_2) {
                                int row_38 = first_row_4;
                                if (col_blocks_6 <= first_col_2) {
                                    row_38 = row_38 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_38 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end > first_tile_2 + 1) {
                                int row_39 = first_row_4;
                                if (col_blocks_6 <= first_col_2 + 1) {
                                    row_39 = row_39 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_39 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end > first_tile_2 + 2) {
                                int row_40 = first_row_4;
                                if (col_blocks_6 <= first_col_2 + 2) {
                                    row_40 = row_40 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_40 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    swiglu_bits = phase_bits_6;
                } else {
                    unsigned int phase_bits_7 = gemm_bits;
                    int acc_buffer_2 = (int)(phase_bits_7 >> 14 & 1);
                    int other_buffer_2 = acc_buffer_2 ^ 1;
                    unsigned int acc_taddr_2 = taddr_1 + (unsigned int)(acc_buffer_2 * 256);
                    int has_hi_4 = 0;
                    int iterations_5 = td_iters;
                    int global_mini_6 = td_gmini;
                    int macro_rows_6 = td_mrows;
                    int macro_k_3 = td_mk;
                    if (td_expert < 0) {
                        if (tid == 0) {
                        }
                    } else {
                        if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        bool enabled_value_9 = 1;
                                        if (enabled_value_9 != 0) {
                                            int _poll_11_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_11_first) : "l"(hidden_ready + (shared_rows + macro_rows_6 + td_x)) : "memory");
                                            int _poll_11 = _poll_11_first;
                                            while (_poll_11 < 2 * (intermediate / 128)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int _poll_11_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_11_next) : "l"(hidden_ready + (shared_rows + macro_rows_6 + td_x)) : "memory");
                                                _poll_11 = _poll_11_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_419 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                        int mini_rows_9 = ((0) > (_min_419) ? (0) : (_min_419));
                                        int required_9 = (mini_rows_9 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    unsigned int previous_6 = phase_bits_7 >> 7 & 1;
                                    unsigned int bits_7 = phase_bits_7;
                                    if (previous_6 != 1) {
                                        mbarrier_wait(gemm_finished_addr, bits_7 >> 16 & 1);
                                        mbarrier_wait(gemm_finished_addr + 8, bits_7 >> 17 & 1);
                                        mbarrier_wait(gemm_finished_addr + 16, bits_7 >> 18 & 1);
                                        mbarrier_wait(gemm_finished_addr + 24, bits_7 >> 19 & 1);
                                        mbarrier_wait(gemm_finished_addr + 32, bits_7 >> 20 & 1);
                                        mbarrier_wait(gemm_finished_addr + 40, bits_7 >> 21 & 1);
                                        bits_7 = bits_7 ^ 128;
                                    }
                                    phase_bits_7 = bits_7;
                                    int ring_10 = 0;
                                    #pragma unroll 1
                                    for (int idx_10 = 0; idx_10 < iterations_5; idx_10++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_10) * 8, phase_bits_7 >> (unsigned int)(16 + ring_10) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v38_addr + (unsigned int)(ring_10 * 16384)), "l"((&hidden_q)), "r"(0), "r"(td_x * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v39_addr + (unsigned int)(ring_10 * 16384)), "l"((&wd_q)), "r"(0), "r"(td_y * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(td_expert), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 16 + ring_10);
                                        ring_10 = (ring_10 + 1) % 6;
                                    }
                                }
                            }
                        } else if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        bool enabled_value_10 = 1;
                                        if (enabled_value_10 != 0) {
                                            int _poll_12_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_12_first) : "l"(hidden_ready + (shared_rows + macro_rows_6 + td_x)) : "memory");
                                            int _poll_12 = _poll_12_first;
                                            while (_poll_12 < 2 * (intermediate / 128)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int _poll_12_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_12_next) : "l"(hidden_ready + (shared_rows + macro_rows_6 + td_x)) : "memory");
                                                _poll_12 = _poll_12_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_420 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                        int mini_rows_10 = ((0) > (_min_420) ? (0) : (_min_420));
                                        int required_10 = (mini_rows_10 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    unsigned int previous_7 = phase_bits_7 >> 7 & 1;
                                    unsigned int bits_8 = phase_bits_7;
                                    if (previous_7 != 1) {
                                        mbarrier_wait(scales_finished_addr, bits_8 >> 16 & 1);
                                        mbarrier_wait(scales_finished_addr + 8, bits_8 >> 17 & 1);
                                        mbarrier_wait(scales_finished_addr + 16, bits_8 >> 18 & 1);
                                        mbarrier_wait(scales_finished_addr + 24, bits_8 >> 19 & 1);
                                        mbarrier_wait(scales_finished_addr + 32, bits_8 >> 20 & 1);
                                        mbarrier_wait(scales_finished_addr + 40, bits_8 >> 21 & 1);
                                        bits_8 = bits_8 ^ 128;
                                    }
                                    phase_bits_7 = bits_8;
                                    int ring_11 = 0;
                                    #pragma unroll 1
                                    for (int idx_11 = 0; idx_11 < iterations_5; idx_11++) {
                                        mbarrier_wait(scales_finished_addr + (ring_11) * 8, phase_bits_7 >> (unsigned int)(16 + ring_11) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v40_addr + (unsigned int)(ring_11 * 512)), "l"((&hidden_sc)), "r"(0), "r"(0), "r"((td_x * 2 + cta_rank_0) * (intermediate / 128) + idx_11),
                                               "r"(((scales_arrived_addr + (ring_11) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v41_addr + (unsigned int)(ring_11 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_sc)), "r"(0), "r"(0), "r"((td_expert * (hidden / 128) + td_y * 2 + cta_rank_0) * (intermediate / 128) + idx_11),
                                               "r"(((scales_arrived_addr + (ring_11) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 16 + ring_11);
                                        ring_11 = (ring_11 + 1) % 6;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_12 = 0;
                                        mbarrier_wait(output_finished_addr + (acc_buffer_2) * 8, phase_bits_7 >> (unsigned int)(22 + acc_buffer_2) & 1);
                                        mbarrier_wait(prefix_released_addr + (acc_buffer_2) * 8, phase_bits_7 >> (unsigned int)(22 + acc_buffer_2) & 1);
                                        mbarrier_wait(prefix_released_addr + (other_buffer_2) * 8, phase_bits_7 >> (unsigned int)(22 + other_buffer_2) & 1);
                                        phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 22 + acc_buffer_2);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_12 = 0; idx_12 < iterations_5; idx_12++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_12) * 8, 3072);
                                            mbarrier_wait(scales_arrived_addr + (ring_12) * 8, phase_bits_7 >> (unsigned int)(8 + ring_12) & 1);
                                            int buffer_2 = idx_12 % 3;
                                            int sfa_slot_2 = buffer_2 + other_buffer_2 * 64;
                                            int sfb_slot_2 = buffer_2 + other_buffer_2 * 32;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + sfa_slot_2 * 4, make_sf_cp_desc_sbo128(smem_v40_addr + (unsigned int)(ring_12 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + sfb_slot_2 * 8, make_sf_cp_desc_sbo128(smem_v41_addr + (unsigned int)(ring_12 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + sfb_slot_2 * 8 + 4), make_sf_cp_desc_sbo128((smem_v41_addr + (unsigned int)(ring_12 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_12) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_12) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_12) * 8, phase_bits_7 >> (unsigned int)ring_12 & 1);
                                            int _smem_desc_lo_421 = (((smem_v38_addr) >> 4) & 0x3FFF) + (ring_12) * 1024;
                                            int _smem_desc_lo_422 = (((smem_v39_addr) >> 4) & 0x3FFF) + (ring_12) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_421) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_422) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_2 * 256)), a_desc + 0, b_desc + 0,
                                                    0x10c00000U, tmem_tmem_sfa + sfa_slot_2 * 4, tmem_tmem_sfb + sfb_slot_2 * 8, ((idx_12 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_2 * 256)), a_desc + 2, b_desc + 2,
                                                    0x30c00010U, tmem_tmem_sfa + sfa_slot_2 * 4, tmem_tmem_sfb + sfb_slot_2 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_2 * 256)), a_desc + 4, b_desc + 4,
                                                    0x50c00020U, tmem_tmem_sfa + sfa_slot_2 * 4, tmem_tmem_sfb + sfb_slot_2 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2((tmem_accumulator + (acc_buffer_2 * 256)), a_desc + 6, b_desc + 6,
                                                    0x70c00030U, tmem_tmem_sfa + sfa_slot_2 * 4, tmem_tmem_sfb + sfb_slot_2 * 8, 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_12) * 8, (uint16_t)(3));
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << ring_12) ^ (unsigned int)(1 << 8 + ring_12);
                                            ring_12 = (ring_12 + 1) % 6;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                unsigned int _parity_output_arrived_4 = (phase_bits_7 >> 6) % 2;
                                mbarrier_wait(output_arrived_addr, _parity_output_arrived_4);
                                unsigned int packed_6[128];
                                #pragma unroll
                                for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                    #pragma unroll
                                    for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                        unsigned int address_26 = acc_taddr_2 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                        float _tmem_load_36[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[15]))
                                            : "r"(address_26));
                                        #pragma unroll
                                        for (int pair_5 = 0; pair_5 < 8; pair_5++) {
                                            packed_6[chunk_4 * 16 + sub_2 * 8 + pair_5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_36[pair_5 * 2], _tmem_load_36[pair_5 * 2 + 1])));
                                        }
                                    }
                                    if (chunk_4 == 1) {
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((prefix_released_addr + (acc_buffer_2) * 8) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_2 = 1;
                                if (last_2 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr + (acc_buffer_2) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                if (tid == 0) {
                                    int previous_offset_2 = (macro_1_1 + 1) * macro_size;
                                    int output_row_1 = td_x * 256 + cta_rank_0 * 128;
                                    int _min_423 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                    if (output_row_1 < _min_423) {
                                        bool enabled_value_11 = 1;
                                        if (enabled_value_11 != 0) {
                                            int _poll_13_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_13_first) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                            int _poll_13 = _poll_13_first;
                                            while (_poll_13 < 8 * ((hidden + 1023) / 1024)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int _poll_13_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_13_next) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                                _poll_13 = _poll_13_next;
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
                                    int warp_0_3 = tid / 32;
                                    int lane_5 = tid % 32;
                                    #pragma unroll
                                    for (int half_35 = 0; half_35 < 2; half_35++) {
                                        #pragma unroll
                                        for (int col_tile_34 = 0; col_tile_34 < 2; col_tile_34++) {
                                            int row_41 = warp_0_3 * 32 + half_35 * 16 + lane_5 % 16;
                                            int col_66 = col_tile_34 * 16 + lane_5 / 16 * 8;
                                            unsigned int address_27 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_41 * 32 + col_66) * 2);
                                            address_27 = address_27 ^ (address_27 & 511) >> 7 << 4;
                                            int offset_28 = chunk_5 * 16 + half_35 * 8 + col_tile_34 * 4;
                                            uint32_t _stmatrix_addr_424 = static_cast<uint32_t>(address_27);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_424), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_28])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_28 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_28 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_28 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&y_routed), 0, td_x * 256 + cta_rank_0 * 128, td_y * 8 + chunk_5, 0, 0, d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                if (has_hi_4 != 0) {
                                    unsigned int packed_0_1[128];
                                    #pragma unroll
                                    for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                        #pragma unroll
                                        for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                            unsigned int address_28 = acc_taddr_2 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                            float _tmem_load_37[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[15]))
                                                : "r"(address_28));
                                            #pragma unroll
                                            for (int pair_6 = 0; pair_6 < 8; pair_6++) {
                                                packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_37[pair_6 * 2], _tmem_load_37[pair_6 * 2 + 1])));
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
                                                :: "r"((output_finished_addr + (acc_buffer_2) * 8) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_7 = 0; chunk_7 < 8; chunk_7++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_4 = tid / 32;
                                        int lane_7 = tid % 32;
                                        #pragma unroll
                                        for (int half_36 = 0; half_36 < 2; half_36++) {
                                            #pragma unroll
                                            for (int col_tile_35 = 0; col_tile_35 < 2; col_tile_35++) {
                                                int row_42 = warp_0_4 * 32 + half_36 * 16 + lane_7 % 16;
                                                int col_67 = col_tile_35 * 16 + lane_7 / 16 * 8;
                                                unsigned int address_29 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_42 * 32 + col_67) * 2);
                                                address_29 = address_29 ^ (address_29 & 511) >> 7 << 4;
                                                int offset_29 = chunk_7 * 16 + half_36 * 8 + col_tile_35 * 4;
                                                uint32_t _stmatrix_addr_425 = static_cast<uint32_t>(address_29);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_425), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_29 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_5d((&y_routed), 0, td_x * 256 + cta_rank_0 * 128, (td_y + 1) * 8 + chunk_7, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192));
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                phase_bits_7 = phase_bits_7 ^ 64;
                                if (tid / 32 == 0) {
                                    if (warp == 0) {
                                        if (elect_sync()) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_12 = 1;
                                            if (enabled_value_12 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_6))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        phase_bits_7 = phase_bits_7 ^ 16384;
                    }
                    gemm_bits = phase_bits_7;
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
                        int task_1_1 = 0;
                        int macro_2 = 0;
                        int mini_3 = 0;
                        int swiglu_4 = 0;
                        if (shared_tasks <= decoded_cluster - comm_clusters) {
                            uint32_t _fast_div_q_4 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks));
                            uint32_t _fast_div_r_4 = (uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks)) - _fast_div_q_4 * (uint32_t)(mini_tasks_div.divisor);
                            int ordered_mini_1 = (int)_fast_div_q_4;
                            task_1_1 = (int)_fast_div_r_4;
                            macro_2 = macros - 1;
                            mini_3 = ordered_mini_1;
                            if (ordered_mini_1 >= last_minis) {
                                uint32_t _fast_div_q_5 = (minis_per_macro_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(ordered_mini_1 - last_minis)), minis_per_macro_div.multiplier) >> minis_per_macro_div.shift_right) : (uint32_t)((unsigned int)(ordered_mini_1 - last_minis));
                                uint32_t _fast_div_r_5 = (uint32_t)((unsigned int)(ordered_mini_1 - last_minis)) - _fast_div_q_5 * (uint32_t)(minis_per_macro_div.divisor);
                                macro_2 = macros - 2 - (int)_fast_div_q_5;
                                mini_3 = (int)_fast_div_r_5;
                            }
                            if (task_1_1 >= mini_gate && task_1_1 < mini_gate + mini_swiglu) {
                                swiglu_4 = 1;
                            }
                        }
                        int nt_x = 0;
                        int nt_y = 0;
                        int nt_expert = 0;
                        int nt_k0 = 0;
                        int nt_k1 = 0;
                        int nt_first = 0;
                        int nt_blocks = 0;
                        int nt_iters = 0;
                        int nt_gmini = 0;
                        int nt_mrows = 0;
                        int nt_mk = 0;
                        if (decoded_cluster < true_clusters) {
                            int x_4 = 0;
                            int y_4 = 0;
                            int expert_4 = 0;
                            int k_start_3 = 0;
                            int k_end_3 = 0;
                            int first_3 = 0;
                            int col_blocks_7 = 0;
                            int iterations_6 = 0;
                            int global_mini_7 = 0;
                            int macro_rows_7 = 0;
                            int macro_k_4 = 0;
                            if (shared_tasks <= decoded_cluster - comm_clusters) {
                                if (task_1_1 < mini_gate) {
                                    int gate_task_1 = task_1_1;
                                    if (task_1_1 >= gate_tiles) {
                                        gate_task_1 = task_1_1 - gate_tiles;
                                    }
                                    int col_blocks_0_2 = (intermediate + 256 - 1) / 256;
                                    int x_1_2 = -1;
                                    int y_2_2 = -1;
                                    int expert_3_2 = -1;
                                    int k_start_4_2 = 0;
                                    int k_end_5_2 = 0;
                                    int first_6_2 = 0;
                                    int first_block_2 = (macro_2 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                                    int end_block_2 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                                    int block_4 = first_block_2;
                                    uint32_t _fast_div_q_6 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)gate_task_1), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)gate_task_1);
                                    uint32_t _fast_div_r_6 = (uint32_t)((unsigned int)gate_task_1) - _fast_div_q_6 * (uint32_t)(gate_col_blocks_div.divisor);
                                    block_4 = first_block_2 + (int)_fast_div_q_6;
                                    if (block_4 < end_block_2) {
                                        int index_2 = counts[3 * experts + block_4];
                                        int offset_30 = counts[experts + index_2] / 256;
                                        int first_row_5 = ((first_block_2) > (offset_30) ? (first_block_2) : (offset_30));
                                        int _min_426 = ((end_block_2) < (offset_30 + counts[index_2] / 256) ? (end_block_2) : (offset_30 + counts[index_2] / 256));
                                        int rows_4 = _min_426 - first_row_5;
                                        int supergroup_4 = (gate_task_1 - (first_row_5 - first_block_2) * col_blocks_0_2) / (rows_4 * 8);
                                        int full_cols_4 = col_blocks_0_2 / 8 * 8;
                                        int row_43 = 0;
                                        int col_68 = 0;
                                        if (gate_task_1 - (first_row_5 - first_block_2) * col_blocks_0_2 < rows_4 * full_cols_4) {
                                            row_43 = (gate_task_1 - (first_row_5 - first_block_2) * col_blocks_0_2) % (rows_4 * 8) / 8;
                                            col_68 = supergroup_4 * 8 + (gate_task_1 - (first_row_5 - first_block_2) * col_blocks_0_2) % 8;
                                        } else {
                                            row_43 = (gate_task_1 - (first_row_5 - first_block_2) * col_blocks_0_2 - rows_4 * full_cols_4) / (col_blocks_0_2 - full_cols_4);
                                            col_68 = full_cols_4 + (gate_task_1 - (first_row_5 - first_block_2) * col_blocks_0_2 - rows_4 * full_cols_4) % (col_blocks_0_2 - full_cols_4);
                                        }
                                        if ((supergroup_4 & 1) != 0) {
                                            row_43 = rows_4 - row_43 - 1;
                                        }
                                        x_1_2 = first_row_5 + row_43 - macro_2 * (macro_size / 256);
                                        y_2_2 = col_68;
                                        expert_3_2 = index_2;
                                    }
                                    int iterations_7_2 = hidden / 128;
                                    int global_mini_8_2 = macro_2 * (macro_size / mini_size) + mini_3;
                                    int macro_rows_9_2 = macro_2 * (macro_size / 256);
                                    int macro_k_10_2 = macro_2 * (macro_size / 128);
                                    x_4 = x_1_2;
                                    y_4 = y_2_2;
                                    expert_4 = expert_3_2;
                                    k_start_3 = k_start_4_2;
                                    k_end_3 = k_end_5_2;
                                    first_3 = first_6_2;
                                    col_blocks_7 = col_blocks_0_2;
                                    iterations_6 = iterations_7_2;
                                    global_mini_7 = global_mini_8_2;
                                    macro_rows_7 = macro_rows_9_2;
                                    macro_k_4 = macro_k_10_2;
                                } else if (task_1_1 >= mini_gate + mini_swiglu) {
                                    int col_blocks_0_3 = (hidden + 256 - 1) / 256;
                                    int x_1_3 = -1;
                                    int y_2_3 = -1;
                                    int expert_3_3 = -1;
                                    int k_start_4_3 = 0;
                                    int k_end_5_3 = 0;
                                    int first_6_3 = 0;
                                    int first_block_3 = (macro_2 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                                    int end_block_3 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                                    int block_5 = first_block_3;
                                    uint32_t _fast_div_q_7 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(task_1_1 - mini_gate - mini_swiglu)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(task_1_1 - mini_gate - mini_swiglu));
                                    uint32_t _fast_div_r_7 = (uint32_t)((unsigned int)(task_1_1 - mini_gate - mini_swiglu)) - _fast_div_q_7 * (uint32_t)(down_col_blocks_div.divisor);
                                    block_5 = first_block_3 + (int)_fast_div_q_7;
                                    if (block_5 < end_block_3) {
                                        int index_3 = counts[3 * experts + block_5];
                                        int offset_31 = counts[experts + index_3] / 256;
                                        int first_row_6 = ((first_block_3) > (offset_31) ? (first_block_3) : (offset_31));
                                        int _min_427 = ((end_block_3) < (offset_31 + counts[index_3] / 256) ? (end_block_3) : (offset_31 + counts[index_3] / 256));
                                        int rows_5 = _min_427 - first_row_6;
                                        int supergroup_5 = (task_1_1 - mini_gate - mini_swiglu - (first_row_6 - first_block_3) * col_blocks_0_3) / (rows_5 * 8);
                                        int full_cols_5 = col_blocks_0_3 / 8 * 8;
                                        int row_44 = 0;
                                        int col_69 = 0;
                                        if (task_1_1 - mini_gate - mini_swiglu - (first_row_6 - first_block_3) * col_blocks_0_3 < rows_5 * full_cols_5) {
                                            row_44 = (task_1_1 - mini_gate - mini_swiglu - (first_row_6 - first_block_3) * col_blocks_0_3) % (rows_5 * 8) / 8;
                                            col_69 = supergroup_5 * 8 + (task_1_1 - mini_gate - mini_swiglu - (first_row_6 - first_block_3) * col_blocks_0_3) % 8;
                                        } else {
                                            row_44 = (task_1_1 - mini_gate - mini_swiglu - (first_row_6 - first_block_3) * col_blocks_0_3 - rows_5 * full_cols_5) / (col_blocks_0_3 - full_cols_5);
                                            col_69 = full_cols_5 + (task_1_1 - mini_gate - mini_swiglu - (first_row_6 - first_block_3) * col_blocks_0_3 - rows_5 * full_cols_5) % (col_blocks_0_3 - full_cols_5);
                                        }
                                        if ((supergroup_5 & 1) != 0) {
                                            row_44 = rows_5 - row_44 - 1;
                                        }
                                        x_1_3 = first_row_6 + row_44 - macro_2 * (macro_size / 256);
                                        y_2_3 = col_69;
                                        expert_3_3 = index_3;
                                    }
                                    int iterations_7_3 = intermediate / 128;
                                    int global_mini_8_3 = macro_2 * (macro_size / mini_size) + mini_3;
                                    int macro_rows_9_3 = macro_2 * (macro_size / 256);
                                    int macro_k_10_3 = macro_2 * (macro_size / 128);
                                    x_4 = x_1_3;
                                    y_4 = y_2_3;
                                    expert_4 = expert_3_3;
                                    k_start_3 = k_start_4_3;
                                    k_end_3 = k_end_5_3;
                                    first_3 = first_6_3;
                                    col_blocks_7 = col_blocks_0_3;
                                    iterations_6 = iterations_7_3;
                                    global_mini_7 = global_mini_8_3;
                                    macro_rows_7 = macro_rows_9_3;
                                    macro_k_4 = macro_k_10_3;
                                }
                            }
                            nt_x = x_4;
                            nt_y = y_4;
                            nt_expert = expert_4;
                            nt_k0 = k_start_3;
                            nt_k1 = k_end_3;
                            nt_first = first_3;
                            nt_blocks = col_blocks_7;
                            nt_iters = iterations_6;
                            nt_gmini = global_mini_7;
                            nt_mrows = macro_rows_7;
                            nt_mk = macro_k_4;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_0_addr), "r"((unsigned int)decoded_cluster), "r"((unsigned int)swiglu_4), "r"((unsigned int)task_1_1), "r"((unsigned int)macro_2) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_1_addr), "r"((unsigned int)mini_3), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_2_addr), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_3_addr), "r"((unsigned int)nt_x), "r"((unsigned int)nt_y), "r"((unsigned int)nt_expert), "r"((unsigned int)nt_k0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_4_addr), "r"((unsigned int)nt_k1), "r"((unsigned int)nt_first), "r"((unsigned int)nt_blocks), "r"((unsigned int)nt_iters) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_5_addr), "r"((unsigned int)nt_gmini), "r"((unsigned int)nt_mrows), "r"((unsigned int)nt_mk), "r"(0) : "memory");
                        mbarrier_arrive(decode_ready_addr);
                    }
                }
            }
            int _parity_decode_ready = (int)((unsigned int)iteration % 2);
            mbarrier_wait(decode_ready_addr, _parity_decode_ready);
            unsigned int decode_quad[4];
            __int128_t _smem_raw_428;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_428) : "r"(decode_words_0_addr) : "memory");
            decode_quad[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_428)[0];
            decode_quad[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_428)[1];
            decode_quad[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_428)[2];
            decode_quad[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_428)[3];
            unsigned int decode_quad_0[4];
            __int128_t _smem_raw_429;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_429) : "r"(decode_words_1_addr) : "memory");
            decode_quad_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_429)[0];
            decode_quad_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_429)[1];
            decode_quad_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_429)[2];
            decode_quad_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_429)[3];
            unsigned int decode_quad_1[4];
            __int128_t _smem_raw_430;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_430) : "r"(decode_words_2_addr) : "memory");
            decode_quad_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_430)[0];
            decode_quad_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_430)[1];
            decode_quad_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_430)[2];
            decode_quad_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_430)[3];
            unsigned int decode_quad_2[4];
            __int128_t _smem_raw_431;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_431) : "r"(decode_words_3_addr) : "memory");
            decode_quad_2[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_431)[0];
            decode_quad_2[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_431)[1];
            decode_quad_2[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_431)[2];
            decode_quad_2[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_431)[3];
            unsigned int decode_quad_3[4];
            __int128_t _smem_raw_432;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_432) : "r"(decode_words_4_addr) : "memory");
            decode_quad_3[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_432)[0];
            decode_quad_3[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_432)[1];
            decode_quad_3[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_432)[2];
            decode_quad_3[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_432)[3];
            unsigned int decode_quad_4[4];
            __int128_t _smem_raw_433;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_433) : "r"(decode_words_5_addr) : "memory");
            decode_quad_4[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_433)[0];
            decode_quad_4[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_433)[1];
            decode_quad_4[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_433)[2];
            decode_quad_4[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_433)[3];
            cluster = (int)decode_quad[0];
            int next_local = (int)decode_quad[1];
            int next_task_1 = (int)decode_quad[2];
            int next_macro = (int)decode_quad[3];
            int next_mini = (int)decode_quad_0[0];
            __syncwarp();
            bool _elect_sync_0 = elect_sync();
            if (_elect_sync_0) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((schedule_finished_addr) & 0xFEFFFFFF) : "memory");
            }
            if (current_local != 0 && cluster >= 0 && next_local == 0) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            task_0 = next_task_1;
            macro_1_1 = next_macro;
            mini_2_1 = next_mini;
            current_local = next_local;
            td_x = (int)decode_quad_2[0];
            td_y = (int)decode_quad_2[1];
            td_expert = (int)decode_quad_2[2];
            td_k0 = (int)decode_quad_2[3];
            td_k1 = (int)decode_quad_3[0];
            td_first = (int)decode_quad_3[1];
            td_blocks = (int)decode_quad_3[2];
            td_iters = (int)decode_quad_3[3];
            td_gmini = (int)decode_quad_4[0];
            td_mrows = (int)decode_quad_4[1];
            td_mk = (int)decode_quad_4[2];
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
                int _parity_drain_arrived_0 = (int)((unsigned int)iteration_0 % 2);
                mbarrier_wait(drain_arrived_0_addr, _parity_drain_arrived_0);
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
                    int _parity_drain_finished_0 = (int)((unsigned int)iteration_0 % 2);
                    mbarrier_wait(drain_finished_0_addr, _parity_drain_finished_0);
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
                int _parity_drain_arrived_1 = (int)((unsigned int)iteration_0_1 % 2);
                mbarrier_wait(drain_arrived_1_addr, _parity_drain_arrived_1);
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
                    int _parity_drain_finished_1 = (int)((unsigned int)iteration_0_1 % 2);
                    mbarrier_wait(drain_finished_1_addr, _parity_drain_finished_1);
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
                int _parity_drain_arrived_2 = (int)((unsigned int)iteration_0_2 % 2);
                mbarrier_wait(drain_arrived_2_addr, _parity_drain_arrived_2);
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
                    int _parity_drain_finished_2 = (int)((unsigned int)iteration_0_2 % 2);
                    mbarrier_wait(drain_finished_2_addr, _parity_drain_finished_2);
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
                int _parity_drain_arrived_3 = (int)((unsigned int)iteration_0_3 % 2);
                mbarrier_wait(drain_arrived_3_addr, _parity_drain_arrived_3);
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
                    int _parity_drain_finished_3 = (int)((unsigned int)iteration_0_3 % 2);
                    mbarrier_wait(drain_finished_3_addr, _parity_drain_finished_3);
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
                int _parity_drain_arrived_4 = (int)((unsigned int)iteration_0_4 % 2);
                mbarrier_wait(drain_arrived_4_addr, _parity_drain_arrived_4);
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
                    int _parity_drain_finished_4 = (int)((unsigned int)iteration_0_4 % 2);
                    mbarrier_wait(drain_finished_4_addr, _parity_drain_finished_4);
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
                int _parity_drain_arrived_5 = (int)((unsigned int)iteration_0_5 % 2);
                mbarrier_wait(drain_arrived_5_addr, _parity_drain_arrived_5);
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
                    int _parity_drain_finished_5 = (int)((unsigned int)iteration_0_5 % 2);
                    mbarrier_wait(drain_finished_5_addr, _parity_drain_finished_5);
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
                int _parity_drain_arrived_6 = (int)((unsigned int)iteration_0_6 % 2);
                mbarrier_wait(drain_arrived_6_addr, _parity_drain_arrived_6);
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
                    int _parity_drain_finished_6 = (int)((unsigned int)iteration_0_6 % 2);
                    mbarrier_wait(drain_finished_6_addr, _parity_drain_finished_6);
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
                int _parity_drain_arrived_7 = (int)((unsigned int)iteration_0_7 % 2);
                mbarrier_wait(drain_arrived_7_addr, _parity_drain_arrived_7);
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
                    int _parity_drain_finished_7 = (int)((unsigned int)iteration_0_7 % 2);
                    mbarrier_wait(drain_finished_7_addr, _parity_drain_finished_7);
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
            unsigned int phase_bits_8 = combine_bits;
            int col_blocks_8 = (hidden + 1023) / 1024;
            int macro_offset_1 = 0;
            int macro_tokens_1 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
            int num_tasks_1 = (macro_tokens_1 / 16 * col_blocks_8 + 6) / 7;
            if (tid == 0) {
                unsigned int _atomic_fetch_add_434 = atomicAdd(&combine_next[0], 1);
                combine_slot[0] = (int)_atomic_fetch_add_434;
            }
            __syncthreads();
            int task_1_2 = combine_slot[0];
            __syncthreads();
            while (task_1_2 < num_tasks_1) {
                unsigned int phase_bits_0_3 = phase_bits_8;
                int col_blocks_1_3 = (hidden + 1023) / 1024;
                int first_tile_3 = task_1_2 * 7;
                int macro_offset_2_3 = 0;
                int macro_tokens_3_3 = ((macro_size) < (tokens - macro_offset_2_3) ? (macro_size) : (tokens - macro_offset_2_3));
                int valid_tiles_2 = ((7) < (macro_tokens_3_3 / 16 * col_blocks_1_3 - first_tile_3) ? (7) : (macro_tokens_3_3 / 16 * col_blocks_1_3 - first_tile_3));
                if (valid_tiles_2 > 0) {
                    int first_row_7 = first_tile_3 / col_blocks_1_3 * 16 + tid;
                    int first_col_3 = first_tile_3 % col_blocks_1_3;
                    int rows_6[7];
                    int columns_2[7];
                    int peers_2[7];
                    int tokens_0_2[7];
                    unsigned int counts_1_2[7];
                    int row_45 = first_row_7;
                    int column_2 = first_col_3;
                    #pragma unroll
                    for (int stage_9 = 0; stage_9 < 7; stage_9++) {
                        rows_6[stage_9] = row_45;
                        columns_2[stage_9] = column_2;
                        peers_2[stage_9] = -1;
                        tokens_0_2[stage_9] = -1;
                        if (valid_tiles_2 > stage_9 && tid < 16) {
                            peers_2[stage_9] = schedule_rank[macro_offset_2_3 + row_45];
                            tokens_0_2[stage_9] = schedule_token[macro_offset_2_3 + row_45];
                        }
                        counts_1_2[stage_9] = 0;
                        if (valid_tiles_2 > stage_9) {
                            if (stage_9 == 0 || column_2 == 0) {
                                uint32_t _barrier_vote_435 = __syncthreads_count(peers_2[stage_9] >= 0);
                                counts_1_2[stage_9] = _barrier_vote_435;
                            } else {
                                counts_1_2[stage_9] = counts_1_2[stage_9 - 1];
                            }
                        }
                        column_2 = column_2 + 1;
                        if (column_2 == col_blocks_1_3) {
                            column_2 = 0;
                            row_45 = row_45 + 16;
                        }
                    }
                    if (tid == 0) {
                        int first_mini_2 = (macro_offset_2_3 + first_row_7) / mini_size;
                        int last_mini_2 = (macro_offset_2_3 + (first_tile_3 + valid_tiles_2 - 1) / col_blocks_1_3 * 16) / mini_size;
                        #pragma unroll 1
                        for (int mini_1_1 = first_mini_2; mini_1_1 < last_mini_2 + 1; mini_1_1++) {
                            int mini_rows_11 = ((mini_size) < (tokens - mini_1_1 * mini_size) ? (mini_size) : (tokens - mini_1_1 * mini_size));
                            int required_11 = (mini_rows_11 + 255) / 256 * (hidden / 256) * 2;
                            int _poll_14_first;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_14_first) : "l"(y_ready + mini_1_1) : "memory");
                            int _poll_14 = _poll_14_first;
                            while (_poll_14 < required_11) {
                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                int _poll_14_next;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_poll_14_next) : "l"(y_ready + mini_1_1) : "memory");
                                _poll_14 = _poll_14_next;
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                        }
                        #pragma unroll
                        for (int stage_10 = 0; stage_10 < 7; stage_10++) {
                            if (valid_tiles_2 > stage_10) {
                                int _min_436 = ((1024) < (hidden - columns_2[stage_10] * 1024) ? (1024) : (hidden - columns_2[stage_10] * 1024));
                                unsigned int chunk_bytes_5 = (unsigned int)(_min_436 * 2);
                                mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_10) * 8, counts_1_2[stage_10] * chunk_bytes_5);
                            }
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int stage_11 = 0; stage_11 < 7; stage_11++) {
                        if (peers_2[stage_11] >= 0) {
                            int chunk_cols_4 = ((1024) < (hidden - columns_2[stage_11] * 1024) ? (1024) : (hidden - columns_2[stage_11] * 1024));
                            cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_11 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows_6[stage_11] * (unsigned long long)hidden + (unsigned long long)(columns_2[stage_11] * 1024)) * (unsigned long long)2)), chunk_cols_4 * 2, combine_arrived_addr + (stage_11) * 8);
                        }
                    }
                    #pragma unroll
                    for (int stage_12 = 0; stage_12 < 7; stage_12++) {
                        if (valid_tiles_2 > stage_12) {
                            mbarrier_wait(combine_arrived_addr + (stage_12) * 8, phase_bits_0_3 >> (unsigned int)stage_12 & 1);
                            phase_bits_0_3 = phase_bits_0_3 ^ (unsigned int)(1 << stage_12);
                            if (peers_2[stage_12] >= 0) {
                                int _min_437 = ((1024) < (hidden - columns_2[stage_12] * 1024) ? (1024) : (hidden - columns_2[stage_12] * 1024));
                                unsigned int chunk_bytes_6 = (unsigned int)(_min_437 * 2);
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_438 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[stage_12]]) + ((unsigned long long)tokens_0_2[stage_12] * (unsigned long long)hidden + (unsigned long long)(columns_2[stage_12] * 1024)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_438), "r"(combine_smem_addr + (unsigned int)((stage_12 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_6))
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
                phase_bits_8 = phase_bits_0_3;
                if (tid == 0) {
                    unsigned int _atomic_fetch_add_439 = atomicAdd(&combine_next[0], 1);
                    combine_slot[0] = (int)_atomic_fetch_add_439;
                }
                __syncthreads();
                task_1_2 = combine_slot[0];
                __syncthreads();
            }
            combine_bits = phase_bits_8;
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
