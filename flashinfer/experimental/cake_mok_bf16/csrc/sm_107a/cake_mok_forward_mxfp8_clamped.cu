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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mok_forward_mxfp8_clamped(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap wd_q, const __grid_constant__ CUtensorMap wd_sc, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_q_store, const __grid_constant__ CUtensorMap hidden_sc_store, const __grid_constant__ CUtensorMap hidden_t_store, const __grid_constant__ CUtensorMap hidden_sc_t_store, const __grid_constant__ CUtensorMap hidden_q, const __grid_constant__ CUtensorMap hidden_sc, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, unsigned int* __restrict__ combine_next, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, CakeFastDivmod mini_tasks_div, CakeFastDivmod minis_per_macro_div, CakeFastDivmod gate_col_blocks_div, CakeFastDivmod down_col_blocks_div)
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
    #define decode_ready_addr (mbar_base + 384)
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

    // Prologue
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
    __nv_bfloat16* flat_view = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int flat_view_addr = smem + 197632;
    unsigned int* flat_view_0 = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int flat_view_addr_0 = smem + 197632;
    unsigned int* flat_view_1 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_1 = smem + 1024;
    unsigned int* flat_view_2 = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int flat_view_addr_2 = smem + 17408;
    unsigned int* flat_view_3 = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int flat_view_addr_3 = smem + 99328;
    unsigned int* flat_view_4 = reinterpret_cast<unsigned int*>(smem_raw + 115712);
    const int flat_view_addr_4 = smem + 115712;
    unsigned int* flat_view_5 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_5 = smem + 1024;
    float* flat_view_6 = reinterpret_cast<float*>(smem_raw + 199680);
    const int flat_view_addr_6 = smem + 199680;
    __nv_bfloat16* smem_v14 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v14_addr = smem + 1024;
    __nv_bfloat16* smem_v_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v_addr_0 = smem + 1280;
    __nv_bfloat16* smem_v_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_v_addr_1 = smem + 1536;
    __nv_bfloat16* smem_v_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1792);
    const int smem_v_addr_2 = smem + 1792;
    unsigned int* smem_v18 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v18_addr = smem + 132096;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v19_addr = smem + 148480;
    unsigned int* smem_v20 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v20_addr = smem + 164864;
    unsigned int* smem_v21 = reinterpret_cast<unsigned int*>(smem_raw + 165376);
    const int smem_v21_addr = smem + 165376;
    unsigned int* smem_v22 = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_v22_addr = smem + 165888;
    unsigned int* smem_v23 = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int smem_v23_addr = smem + 182272;
    unsigned int* smem_v24 = reinterpret_cast<unsigned int*>(smem_raw + 198656);
    const int smem_v24_addr = smem + 198656;
    unsigned int* smem_v25 = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int smem_v25_addr = smem + 199168;
    __nv_bfloat16* smem_v26 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v26_addr = smem + 1024;
    __nv_bfloat16* smem_v27 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v27_addr = smem + 1280;
    __nv_bfloat16* smem_v28 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v28_addr = smem + 66560;
    __nv_bfloat16* smem_v29 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66816);
    const int smem_v29_addr = smem + 66816;
    __nv_bfloat16* flat_view_7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr_7 = smem + 1024;
    int* flat_view_8 = reinterpret_cast<int*>(smem_raw + 232440);
    const int flat_view_addr_8 = smem + 232440;
    uint8_t* a4 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a4_addr = smem + 1024;
    uint8_t* b4 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int b4_addr = smem + 66560;
    uint8_t* b_hi4 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int b_hi4_addr = smem + 132096;
    uint8_t* sfa4 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int sfa4_addr = smem + 197632;
    uint8_t* sfb4 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int sfb4_addr = smem + 199680;
    uint8_t* sfb_hi4 = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int sfb_hi4_addr = smem + 203776;
    uint8_t* a6 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a6_addr = smem + 1024;
    uint8_t* b6 = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int b6_addr = smem + 99328;
    uint8_t* sfa6 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int sfa6_addr = smem + 197632;
    uint8_t* sfb6 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int sfb6_addr = smem + 200704;
    unsigned int* smem_v42 = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int smem_v42_addr = smem + 207872;
    unsigned int* smem_v43 = reinterpret_cast<unsigned int*>(smem_raw + 224256);
    const int smem_v43_addr = smem + 224256;
    unsigned int* smem_v44 = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_v44_addr = smem + 228352;
    unsigned int* smem_v45 = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_v45_addr = smem + 228864;

    // Kernel setup ops
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
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (55 barriers at smem_raw[0..704))

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
        // decode_ready: 1 barriers, init_count=1
        mbarrier_init(smem + 384, 1);
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

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem_sfa = taddr + 512;
    const unsigned int tmem_tmem_sfb = taddr + 524;
    const unsigned int tmem_tmem_sfb_hi = taddr + 548;
    const unsigned int tmem_accumulator = taddr;

    // Kernel post-init ops
    unsigned int taddr_1 = taddr;
    int cluster = bid / 2;
    int cta_rank_0 = cta_rank;
    int gemm_deep = 0;
    int gemm_wide = 0;
    int gemm_layout = 0;
    int gemm_out0 = 0;
    int gemm_out1 = 0;
    int gemm_acc_tiles = 0;
    int scales_wide = 0;
    int swiglu_seq = 0;
    int dispatch_seq = 0;
    int dispatch_seq_hi = 0;
    int combine_seq_0 = 0;
    int combine_seq_1 = 0;
    int combine_seq_2 = 0;
    int combine_seq_3 = 0;
    int combine_seq_4 = 0;
    int combine_seq_5 = 0;
    int combine_seq_6 = 0;
    int i_tiles = intermediate / 128;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        if (macros > 0) {
            int last_rows = ((macro_size) < (tokens - (macros - 1) * macro_size) ? (macro_size) : (tokens - (macros - 1) * macro_size));
            int last_dispatch = last_rows / 128 * ((hidden + 511) / 512);
            int col_blocks = (hidden + 511) / 512;
            int macro_offset_0 = (macros - 1) * macro_size;
            int macro_tokens = ((macro_size) < (tokens - macro_offset_0) ? (macro_size) : (tokens - macro_offset_0));
            int tasks = macro_tokens / 128 * col_blocks;
            int task = comm_cta;
            if (task < tasks) {
                int _min_0 = ((256) < (hidden - task % col_blocks * 512) ? (256) : (hidden - task % col_blocks * 512));
                int peer = -1;
                int peer_token = -1;
                if (tid < 128) {
                    peer = schedule_rank[macro_offset_0 + task / col_blocks * 128 + tid];
                    peer_token = schedule_token[macro_offset_0 + task / col_blocks * 128 + tid];
                }
                uint32_t cta_count = __syncthreads_count(peer >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count * (unsigned int)(_min_0 * 2));
                }
                __syncthreads();
                if (peer >= 0) {
                    cp_async_bulk_gmem2smem(smem_v26_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + (unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(task % col_blocks * 512)) * (unsigned long long)2), _min_0 * 2, dispatch_arrived_addr);
                } else if (tid < 128) {
                    #pragma unroll
                    for (int vec = 0; vec < 32; vec++) {
                        asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_5) + (tid * 512 + vec * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
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
                        peer_1 = schedule_rank[macro_offset_0 + row + tid];
                        peer_token_1 = schedule_token[macro_offset_0 + row + tid];
                    }
                    uint32_t cta_count_1 = __syncthreads_count(peer_1 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_hi_addr, cta_count_1 * (unsigned int)(hi_cols * 2));
                    }
                    __syncthreads();
                    if (peer_1 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v28_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + (unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512) + 256ull) * (unsigned long long)2), hi_cols * 2, dispatch_arrived_hi_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_1 = 0; vec_1 < 32; vec_1++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_5) + (65536 + tid * 512 + vec_1 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                int _parity_dispatch_arrived = (int)((unsigned int)dispatch_seq % 2);
                mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived);
                dispatch_seq += 1;
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
                        int group = tid % 2;
                        #pragma unroll 1
                        for (int j = 0; j < 2; j++) {
                            int k_block = (half * 2 + j + tid / 2) % 4;
                            unsigned int pairs0[16];
                            unsigned int pairs1[16];
                            #pragma unroll
                            for (int k = 0; k < 16; k++) {
                                int col = k_block * 32 + (k * 2 + group * 16) % 32;
                                unsigned int w0 = flat_view_5[col * word_stride + r];
                                unsigned int w1 = flat_view_5[(col + 1) * word_stride + r];
                                uint32_t _prmt_0;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_0) : "r"(w0), "r"(w1));
                                pairs0[k] = _prmt_0;
                                uint32_t _prmt_1;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_1) : "r"(w0), "r"(w1));
                                pairs1[k] = _prmt_1;
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
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_2;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_2) : "f"(scale), "f"(scale));
                            unsigned int scale_byte = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_2 & 255;
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
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_3;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "r"(bf16x2_mul_1));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words[i_1] = (unsigned int)e4m3x2_bf16x2 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_3 << 16;
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
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_4;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_4) : "f"(scale_0), "f"(scale_0));
                            unsigned int scale_byte_0 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_4 & 255;
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
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_5;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_5) : "r"(bf16x2_mul_1_1));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_0[i_3] = (unsigned int)e4m3x2_bf16x2_1 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_5 << 16;
                            }
                            int base0 = row0 * 32 + k_block * 8;
                            int base1 = row1 * 32 + k_block * 8;
                            const void* _smem_vector_store_ptr_6 = reinterpret_cast<const void*>(smem_v22 + (base0 + group * 4));
                            uint32_t _smem_vector_store_addr_6 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_6));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_6), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]));
                            const void* _smem_vector_store_ptr_7 = reinterpret_cast<const void*>(smem_v22 + (base0 + (1 - group) * 4));
                            uint32_t _smem_vector_store_addr_7 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_7));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_7), "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]));
                            const void* _smem_vector_store_ptr_8 = reinterpret_cast<const void*>(smem_v22 + (base1 + group * 4));
                            uint32_t _smem_vector_store_addr_8 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_8));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_8), "r"(words_0[0]), "r"(words_0[1]), "r"(words_0[2]), "r"(words_0[3]));
                            const void* _smem_vector_store_ptr_9 = reinterpret_cast<const void*>(smem_v22 + (base1 + (1 - group) * 4));
                            uint32_t _smem_vector_store_addr_9 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_9));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_9), "r"(words_0[4]), "r"(words_0[5]), "r"(words_0[6]), "r"(words_0[7]));
                            {
                                uint32_t _byte_10 = static_cast<uint32_t>(scale_byte) & 0xFFu;
                                uint32_t _addr_10 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0 + k_block));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_10), "r"(_byte_10) : "memory");
                            }
                            {
                                uint32_t _byte_11 = static_cast<uint32_t>(scale_byte_0) & 0xFFu;
                                uint32_t _addr_11 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1 + k_block));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_11), "r"(_byte_11) : "memory");
                            }
                        }
                    } else {
                        int row_0 = tid - 128;
                        int rot = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                        unsigned int scale_word = 0;
                        #pragma unroll 1
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            int k_block_1 = (j_1 + (tid - 128) / 2) % 4;
                            unsigned int pairs[16];
                            #pragma unroll
                            for (int i_4 = 0; i_4 < 4; i_4++) {
                                int chunk = row_0 * 128 + k_block_1 * 16 + (i_4 + rot) % 4 * 4;
                                const void* _smem_load_ptr_12 = reinterpret_cast<const void*>(flat_view_5 + chunk);
                                uint32_t _smem_load_addr_12 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_12));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&pairs[i_4 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs[(i_4 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs[(i_4 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs[(i_4 * 4) + 3]))
                                    : "r"(_smem_load_addr_12));
                            }
                            uint32_t bf16x2_abs_2;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_2) : "r"(pairs[0]));
                            unsigned int amax_pair_1 = bf16x2_abs_2;
                            #pragma unroll
                            for (int i_5 = 1; i_5 < 16; i_5++) {
                                uint32_t bf16x2_abs_1_2;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_2) : "r"(pairs[i_5]));
                                uint32_t bf16x2_max_2;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_2) : "r"(amax_pair_1), "r"(bf16x2_abs_1_2));
                                amax_pair_1 = bf16x2_max_2;
                            }
                            uint16_t bf16_max_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_1) : "h"((uint16_t)(amax_pair_1 & 65535)), "h"((uint16_t)(amax_pair_1 >> 16)));
                            float amax_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1) : "h"((uint16_t)(bf16_max_1)));
                            float scale_1 = fmaxf(amax_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_13;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_13) : "f"(scale_1), "f"(scale_1));
                            unsigned int scale_byte_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_13 & 255;
                            unsigned int inverse_lane_1 = 254 - scale_byte_1 << 7;
                            unsigned int inverse_1 = inverse_lane_1 | inverse_lane_1 << 16;
                            unsigned int words_1[8];
                            #pragma unroll
                            for (int i_6 = 0; i_6 < 8; i_6++) {
                                uint32_t bf16x2_mul_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(pairs[i_6 * 2]), "r"(inverse_1));
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
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(pairs[i_6 * 2 + 1]), "r"(inverse_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_14;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_14) : "r"(bf16x2_mul_1_2));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_1[i_6] = (unsigned int)e4m3x2_bf16x2_2 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_14 << 16;
                            }
                            scale_word = scale_word | scale_byte_1 << (unsigned int)(k_block_1 * 8);
                            #pragma unroll
                            for (int i_7 = 0; i_7 < 4; i_7++) {
                                const void* _smem_vector_store_ptr_15 = reinterpret_cast<const void*>(smem_v18 + (row_0 * 32 + k_block_1 * 8 + (i_7 + rot) % 4 * 2));
                                uint32_t _smem_vector_store_addr_15 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_15));
                                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_15), "r"(words_1[i_7 * 2]), "r"(words_1[i_7 * 2 + 1]));
                            }
                        }
                        smem_v20[row_0 % 32 * 4 + row_0 / 32] = scale_word;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_0 = col_block * 4;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_0 * 128, row, smem_v18_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile_0, smem_v20_addr);
                        tma_store_2d((&x_t_store), row, col_tile_0 * 128, smem_v22_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_0 * macro_tiles + row_tile, smem_v24_addr);
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
                        int group_1 = tid % 2;
                        #pragma unroll 1
                        for (int j_2 = 0; j_2 < 2; j_2++) {
                            int k_block_2 = (half_1 * 2 + j_2 + tid / 2) % 4;
                            unsigned int pairs0_1[16];
                            unsigned int pairs1_1[16];
                            #pragma unroll
                            for (int k_1 = 0; k_1 < 16; k_1++) {
                                int col_1 = k_block_2 * 32 + (k_1 * 2 + group_1 * 16) % 32;
                                unsigned int w0_1 = flat_view_5[64 + col_1 * word_stride_1 + r_1];
                                unsigned int w1_1 = flat_view_5[64 + (col_1 + 1) * word_stride_1 + r_1];
                                uint32_t _prmt_16;
                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_16) : "r"(w0_1), "r"(w1_1));
                                pairs0_1[k_1] = _prmt_16;
                                uint32_t _prmt_17;
                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_17) : "r"(w0_1), "r"(w1_1));
                                pairs1_1[k_1] = _prmt_17;
                            }
                            uint32_t bf16x2_abs_3;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_3) : "r"(pairs0_1[0]));
                            unsigned int amax_pair_2 = bf16x2_abs_3;
                            #pragma unroll
                            for (int i_8 = 1; i_8 < 16; i_8++) {
                                uint32_t bf16x2_abs_1_3;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_3) : "r"(pairs0_1[i_8]));
                                uint32_t bf16x2_max_3;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_3) : "r"(amax_pair_2), "r"(bf16x2_abs_1_3));
                                amax_pair_2 = bf16x2_max_3;
                            }
                            uint16_t bf16_max_2;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_2) : "h"((uint16_t)(amax_pair_2 & 65535)), "h"((uint16_t)(amax_pair_2 >> 16)));
                            float amax_2;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2) : "h"((uint16_t)(bf16_max_2)));
                            float scale_2 = fmaxf(amax_2 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_18;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_18) : "f"(scale_2), "f"(scale_2));
                            unsigned int scale_byte_2 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_18 & 255;
                            unsigned int inverse_lane_2 = 254 - scale_byte_2 << 7;
                            unsigned int inverse_2 = inverse_lane_2 | inverse_lane_2 << 16;
                            unsigned int words_2[8];
                            #pragma unroll
                            for (int i_9 = 0; i_9 < 8; i_9++) {
                                uint32_t bf16x2_mul_4;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(pairs0_1[i_9 * 2]), "r"(inverse_2));
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
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(pairs0_1[i_9 * 2 + 1]), "r"(inverse_2));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_19;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "r"(bf16x2_mul_1_3));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_2[i_9] = (unsigned int)e4m3x2_bf16x2_3 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_19 << 16;
                            }
                            uint32_t bf16x2_abs_1_4;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_4) : "r"(pairs1_1[0]));
                            unsigned int amax_pair_1_1 = bf16x2_abs_1_4;
                            #pragma unroll
                            for (int i_10 = 1; i_10 < 16; i_10++) {
                                uint32_t bf16x2_abs_1_0;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_0) : "r"(pairs1_1[i_10]));
                                uint32_t bf16x2_max_4;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_4) : "r"(amax_pair_1_1), "r"(bf16x2_abs_1_0));
                                amax_pair_1_1 = bf16x2_max_4;
                            }
                            uint16_t bf16_max_1_1;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_1_1) : "h"((uint16_t)(amax_pair_1_1 & 65535)), "h"((uint16_t)(amax_pair_1_1 >> 16)));
                            float amax_1_1;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_1_1) : "h"((uint16_t)(bf16_max_1_1)));
                            float scale_1_1 = fmaxf(amax_1_1 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_20;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_20) : "f"(scale_1_1), "f"(scale_1_1));
                            unsigned int scale_byte_1_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_20 & 255;
                            unsigned int inverse_lane_1_1 = 254 - scale_byte_1_1 << 7;
                            unsigned int inverse_1_1 = inverse_lane_1_1 | inverse_lane_1_1 << 16;
                            unsigned int words_1_1[8];
                            #pragma unroll
                            for (int i_11 = 0; i_11 < 8; i_11++) {
                                uint32_t bf16x2_mul_5;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(pairs1_1[i_11 * 2]), "r"(inverse_1_1));
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
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(pairs1_1[i_11 * 2 + 1]), "r"(inverse_1_1));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_21;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_21) : "r"(bf16x2_mul_1_4));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_1_1[i_11] = (unsigned int)e4m3x2_bf16x2_4 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_21 << 16;
                            }
                            int base0_1 = row0_1 * 32 + k_block_2 * 8;
                            int base1_1 = row1_1 * 32 + k_block_2 * 8;
                            const void* _smem_vector_store_ptr_22 = reinterpret_cast<const void*>(smem_v23 + (base0_1 + group_1 * 4));
                            uint32_t _smem_vector_store_addr_22 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_22));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_22), "r"(words_2[0]), "r"(words_2[1]), "r"(words_2[2]), "r"(words_2[3]));
                            const void* _smem_vector_store_ptr_23 = reinterpret_cast<const void*>(smem_v23 + (base0_1 + (1 - group_1) * 4));
                            uint32_t _smem_vector_store_addr_23 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_23));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_23), "r"(words_2[4]), "r"(words_2[5]), "r"(words_2[6]), "r"(words_2[7]));
                            const void* _smem_vector_store_ptr_24 = reinterpret_cast<const void*>(smem_v23 + (base1_1 + group_1 * 4));
                            uint32_t _smem_vector_store_addr_24 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_24));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_24), "r"(words_1_1[0]), "r"(words_1_1[1]), "r"(words_1_1[2]), "r"(words_1_1[3]));
                            const void* _smem_vector_store_ptr_25 = reinterpret_cast<const void*>(smem_v23 + (base1_1 + (1 - group_1) * 4));
                            uint32_t _smem_vector_store_addr_25 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_25));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_25), "r"(words_1_1[4]), "r"(words_1_1[5]), "r"(words_1_1[6]), "r"(words_1_1[7]));
                            {
                                uint32_t _byte_26 = static_cast<uint32_t>(scale_byte_2) & 0xFFu;
                                uint32_t _addr_26 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_1 + k_block_2));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_26), "r"(_byte_26) : "memory");
                            }
                            {
                                uint32_t _byte_27 = static_cast<uint32_t>(scale_byte_1_1) & 0xFFu;
                                uint32_t _addr_27 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_1 + k_block_2));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_27), "r"(_byte_27) : "memory");
                            }
                        }
                    } else {
                        int row_1 = tid - 128;
                        int rot_1 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                        unsigned int scale_word_1 = 0;
                        #pragma unroll 1
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            int k_block_3 = (j_3 + (tid - 128) / 2) % 4;
                            unsigned int pairs_1[16];
                            #pragma unroll
                            for (int i_12 = 0; i_12 < 4; i_12++) {
                                int chunk_1 = row_1 * 128 + k_block_3 * 16 + (i_12 + rot_1) % 4 * 4;
                                const void* _smem_load_ptr_28 = reinterpret_cast<const void*>(flat_view_5 + (64 + chunk_1));
                                uint32_t _smem_load_addr_28 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_28));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&pairs_1[i_12 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_1[(i_12 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_1[(i_12 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_1[(i_12 * 4) + 3]))
                                    : "r"(_smem_load_addr_28));
                            }
                            uint32_t bf16x2_abs_4;
                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_4) : "r"(pairs_1[0]));
                            unsigned int amax_pair_3 = bf16x2_abs_4;
                            #pragma unroll
                            for (int i_13 = 1; i_13 < 16; i_13++) {
                                uint32_t bf16x2_abs_1_5;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_5) : "r"(pairs_1[i_13]));
                                uint32_t bf16x2_max_5;
                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_5) : "r"(amax_pair_3), "r"(bf16x2_abs_1_5));
                                amax_pair_3 = bf16x2_max_5;
                            }
                            uint16_t bf16_max_3;
                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_3) : "h"((uint16_t)(amax_pair_3 & 65535)), "h"((uint16_t)(amax_pair_3 >> 16)));
                            float amax_3;
                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_3) : "h"((uint16_t)(bf16_max_3)));
                            float scale_3 = fmaxf(amax_3 * 0.002232143f, 1e-12f);
                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_29;
                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_29) : "f"(scale_3), "f"(scale_3));
                            unsigned int scale_byte_3 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_29 & 255;
                            unsigned int inverse_lane_3 = 254 - scale_byte_3 << 7;
                            unsigned int inverse_3 = inverse_lane_3 | inverse_lane_3 << 16;
                            unsigned int words_3[8];
                            #pragma unroll
                            for (int i_14 = 0; i_14 < 8; i_14++) {
                                uint32_t bf16x2_mul_6;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(pairs_1[i_14 * 2]), "r"(inverse_3));
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
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(pairs_1[i_14 * 2 + 1]), "r"(inverse_3));
                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_30;
                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_30) : "r"(bf16x2_mul_1_5));
                                #else
                                {
                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
                                }
                                #endif
                                words_3[i_14] = (unsigned int)e4m3x2_bf16x2_5 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_30 << 16;
                            }
                            scale_word_1 = scale_word_1 | scale_byte_3 << (unsigned int)(k_block_3 * 8);
                            #pragma unroll
                            for (int i_15 = 0; i_15 < 4; i_15++) {
                                const void* _smem_vector_store_ptr_31 = reinterpret_cast<const void*>(smem_v19 + (row_1 * 32 + k_block_3 * 8 + (i_15 + rot_1) % 4 * 2));
                                uint32_t _smem_vector_store_addr_31 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_31));
                                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_31), "r"(words_3[i_15 * 2]), "r"(words_3[i_15 * 2 + 1]));
                            }
                        }
                        smem_v21[row_1 % 32 * 4 + row_1 / 32] = scale_word_1;
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
                int next_task = task + comm_sms;
                if (next_task < tasks) {
                    int _min_1 = ((256) < (hidden - next_task % col_blocks * 512) ? (256) : (hidden - next_task % col_blocks * 512));
                    int peer_2 = -1;
                    int peer_token_2 = -1;
                    if (tid < 128) {
                        peer_2 = schedule_rank[macro_offset_0 + next_task / col_blocks * 128 + tid];
                        peer_token_2 = schedule_token[macro_offset_0 + next_task / col_blocks * 128 + tid];
                    }
                    uint32_t cta_count_2 = __syncthreads_count(peer_2 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_2 * (unsigned int)(_min_1 * 2));
                    }
                    __syncthreads();
                    if (peer_2 >= 0) {
                        cp_async_bulk_gmem2smem(smem_v26_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + (unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task % col_blocks * 512)) * (unsigned long long)2), _min_1 * 2, dispatch_arrived_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec_2 = 0; vec_2 < 32; vec_2++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_5) + (tid * 512 + vec_2 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
                if (hi_cols > 0) {
                    int _parity_dispatch_arrived_hi = (int)((unsigned int)dispatch_seq_hi % 2);
                    mbarrier_wait(dispatch_arrived_hi_addr, _parity_dispatch_arrived_hi);
                    dispatch_seq_hi += 1;
                    int row_tile_0 = row / 128;
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
                            int group_2 = tid % 2;
                            #pragma unroll 1
                            for (int j_4 = 0; j_4 < 2; j_4++) {
                                int k_block_4 = (half_2 * 2 + j_4 + tid / 2) % 4;
                                unsigned int pairs0_2[16];
                                unsigned int pairs1_2[16];
                                #pragma unroll
                                for (int k_2 = 0; k_2 < 16; k_2++) {
                                    int col_2 = k_block_4 * 32 + (k_2 * 2 + group_2 * 16) % 32;
                                    unsigned int w0_2 = flat_view_5[16384 + col_2 * word_stride_2 + r_2];
                                    unsigned int w1_2 = flat_view_5[16384 + (col_2 + 1) * word_stride_2 + r_2];
                                    uint32_t _prmt_32;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_32) : "r"(w0_2), "r"(w1_2));
                                    pairs0_2[k_2] = _prmt_32;
                                    uint32_t _prmt_33;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_33) : "r"(w0_2), "r"(w1_2));
                                    pairs1_2[k_2] = _prmt_33;
                                }
                                uint32_t bf16x2_abs_5;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_5) : "r"(pairs0_2[0]));
                                unsigned int amax_pair_4 = bf16x2_abs_5;
                                #pragma unroll
                                for (int i_16 = 1; i_16 < 16; i_16++) {
                                    uint32_t bf16x2_abs_1_6;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_6) : "r"(pairs0_2[i_16]));
                                    uint32_t bf16x2_max_6;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_6) : "r"(amax_pair_4), "r"(bf16x2_abs_1_6));
                                    amax_pair_4 = bf16x2_max_6;
                                }
                                uint16_t bf16_max_4;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_4) : "h"((uint16_t)(amax_pair_4 & 65535)), "h"((uint16_t)(amax_pair_4 >> 16)));
                                float amax_4;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_4) : "h"((uint16_t)(bf16_max_4)));
                                float scale_4 = fmaxf(amax_4 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_34;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_34) : "f"(scale_4), "f"(scale_4));
                                unsigned int scale_byte_4 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_34 & 255;
                                unsigned int inverse_lane_4 = 254 - scale_byte_4 << 7;
                                unsigned int inverse_4 = inverse_lane_4 | inverse_lane_4 << 16;
                                unsigned int words_4[8];
                                #pragma unroll
                                for (int i_17 = 0; i_17 < 8; i_17++) {
                                    uint32_t bf16x2_mul_7;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(pairs0_2[i_17 * 2]), "r"(inverse_4));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(pairs0_2[i_17 * 2 + 1]), "r"(inverse_4));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_35;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_35) : "r"(bf16x2_mul_1_6));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_35) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_4[i_17] = (unsigned int)e4m3x2_bf16x2_6 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_35 << 16;
                                }
                                uint32_t bf16x2_abs_2_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_2_1) : "r"(pairs1_2[0]));
                                unsigned int amax_pair_2_1 = bf16x2_abs_2_1;
                                #pragma unroll
                                for (int i_18 = 1; i_18 < 16; i_18++) {
                                    uint32_t bf16x2_abs_1_7;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_7) : "r"(pairs1_2[i_18]));
                                    uint32_t bf16x2_max_7;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_7) : "r"(amax_pair_2_1), "r"(bf16x2_abs_1_7));
                                    amax_pair_2_1 = bf16x2_max_7;
                                }
                                uint16_t bf16_max_2_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_2_1) : "h"((uint16_t)(amax_pair_2_1 & 65535)), "h"((uint16_t)(amax_pair_2_1 >> 16)));
                                float amax_2_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_2_1) : "h"((uint16_t)(bf16_max_2_1)));
                                float scale_2_1 = fmaxf(amax_2_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_36;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_36) : "f"(scale_2_1), "f"(scale_2_1));
                                unsigned int scale_byte_2_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_36 & 255;
                                unsigned int inverse_lane_2_1 = 254 - scale_byte_2_1 << 7;
                                unsigned int inverse_2_1 = inverse_lane_2_1 | inverse_lane_2_1 << 16;
                                unsigned int words_2_1[8];
                                #pragma unroll
                                for (int i_19 = 0; i_19 < 8; i_19++) {
                                    uint32_t bf16x2_mul_8;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(pairs1_2[i_19 * 2]), "r"(inverse_2_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(pairs1_2[i_19 * 2 + 1]), "r"(inverse_2_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_37;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_37) : "r"(bf16x2_mul_1_7));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_37) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_2_1[i_19] = (unsigned int)e4m3x2_bf16x2_7 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_37 << 16;
                                }
                                int base0_2 = row0_2 * 32 + k_block_4 * 8;
                                int base1_2 = row1_2 * 32 + k_block_4 * 8;
                                const void* _smem_vector_store_ptr_38 = reinterpret_cast<const void*>(smem_v22 + (base0_2 + group_2 * 4));
                                uint32_t _smem_vector_store_addr_38 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_38));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_38), "r"(words_4[0]), "r"(words_4[1]), "r"(words_4[2]), "r"(words_4[3]));
                                const void* _smem_vector_store_ptr_39 = reinterpret_cast<const void*>(smem_v22 + (base0_2 + (1 - group_2) * 4));
                                uint32_t _smem_vector_store_addr_39 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_39));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_39), "r"(words_4[4]), "r"(words_4[5]), "r"(words_4[6]), "r"(words_4[7]));
                                const void* _smem_vector_store_ptr_40 = reinterpret_cast<const void*>(smem_v22 + (base1_2 + group_2 * 4));
                                uint32_t _smem_vector_store_addr_40 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_40));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_40), "r"(words_2_1[0]), "r"(words_2_1[1]), "r"(words_2_1[2]), "r"(words_2_1[3]));
                                const void* _smem_vector_store_ptr_41 = reinterpret_cast<const void*>(smem_v22 + (base1_2 + (1 - group_2) * 4));
                                uint32_t _smem_vector_store_addr_41 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_41));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_41), "r"(words_2_1[4]), "r"(words_2_1[5]), "r"(words_2_1[6]), "r"(words_2_1[7]));
                                {
                                    uint32_t _byte_42 = static_cast<uint32_t>(scale_byte_4) & 0xFFu;
                                    uint32_t _addr_42 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0_2 + k_block_4));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_42), "r"(_byte_42) : "memory");
                                }
                                {
                                    uint32_t _byte_43 = static_cast<uint32_t>(scale_byte_2_1) & 0xFFu;
                                    uint32_t _addr_43 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1_2 + k_block_4));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_43), "r"(_byte_43) : "memory");
                                }
                            }
                        } else {
                            int row_2 = tid - 128;
                            int rot_2 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                            unsigned int scale_word_2 = 0;
                            #pragma unroll 1
                            for (int j_5 = 0; j_5 < 4; j_5++) {
                                int k_block_5 = (j_5 + (tid - 128) / 2) % 4;
                                unsigned int pairs_2[16];
                                #pragma unroll
                                for (int i_20 = 0; i_20 < 4; i_20++) {
                                    int chunk_2 = row_2 * 128 + k_block_5 * 16 + (i_20 + rot_2) % 4 * 4;
                                    const void* _smem_load_ptr_44 = reinterpret_cast<const void*>(flat_view_5 + (16384 + chunk_2));
                                    uint32_t _smem_load_addr_44 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_44));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&pairs_2[i_20 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_2[(i_20 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_2[(i_20 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_2[(i_20 * 4) + 3]))
                                        : "r"(_smem_load_addr_44));
                                }
                                uint32_t bf16x2_abs_6;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_6) : "r"(pairs_2[0]));
                                unsigned int amax_pair_5 = bf16x2_abs_6;
                                #pragma unroll
                                for (int i_21 = 1; i_21 < 16; i_21++) {
                                    uint32_t bf16x2_abs_1_8;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_8) : "r"(pairs_2[i_21]));
                                    uint32_t bf16x2_max_8;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_8) : "r"(amax_pair_5), "r"(bf16x2_abs_1_8));
                                    amax_pair_5 = bf16x2_max_8;
                                }
                                uint16_t bf16_max_5;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_5) : "h"((uint16_t)(amax_pair_5 & 65535)), "h"((uint16_t)(amax_pair_5 >> 16)));
                                float amax_5;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_5) : "h"((uint16_t)(bf16_max_5)));
                                float scale_5 = fmaxf(amax_5 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_45;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_45) : "f"(scale_5), "f"(scale_5));
                                unsigned int scale_byte_5 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_45 & 255;
                                unsigned int inverse_lane_5 = 254 - scale_byte_5 << 7;
                                unsigned int inverse_5 = inverse_lane_5 | inverse_lane_5 << 16;
                                unsigned int words_5[8];
                                #pragma unroll
                                for (int i_22 = 0; i_22 < 8; i_22++) {
                                    uint32_t bf16x2_mul_9;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(pairs_2[i_22 * 2]), "r"(inverse_5));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(pairs_2[i_22 * 2 + 1]), "r"(inverse_5));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_46;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_46) : "r"(bf16x2_mul_1_8));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_46) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_5[i_22] = (unsigned int)e4m3x2_bf16x2_8 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_46 << 16;
                                }
                                scale_word_2 = scale_word_2 | scale_byte_5 << (unsigned int)(k_block_5 * 8);
                                #pragma unroll
                                for (int i_23 = 0; i_23 < 4; i_23++) {
                                    const void* _smem_vector_store_ptr_47 = reinterpret_cast<const void*>(smem_v18 + (row_2 * 32 + k_block_5 * 8 + (i_23 + rot_2) % 4 * 2));
                                    uint32_t _smem_vector_store_addr_47 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_47));
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_47), "r"(words_5[i_23 * 2]), "r"(words_5[i_23 * 2 + 1]));
                                }
                            }
                            smem_v20[row_2 % 32 * 4 + row_2 / 32] = scale_word_2;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_2 = col_block * 4 + 2;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_2 * 128, row, smem_v18_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_0 * k_tiles_0 + col_tile_2, smem_v20_addr);
                            tma_store_2d((&x_t_store), row, col_tile_2 * 128, smem_v22_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_2 * macro_tiles_0 + row_tile_0, smem_v24_addr);
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
                            int group_3 = tid % 2;
                            #pragma unroll 1
                            for (int j_6 = 0; j_6 < 2; j_6++) {
                                int k_block_6 = (half_3 * 2 + j_6 + tid / 2) % 4;
                                unsigned int pairs0_3[16];
                                unsigned int pairs1_3[16];
                                #pragma unroll
                                for (int k_3 = 0; k_3 < 16; k_3++) {
                                    int col_3 = k_block_6 * 32 + (k_3 * 2 + group_3 * 16) % 32;
                                    unsigned int w0_3 = flat_view_5[16448 + col_3 * word_stride_3 + r_3];
                                    unsigned int w1_3 = flat_view_5[16448 + (col_3 + 1) * word_stride_3 + r_3];
                                    uint32_t _prmt_48;
                                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_48) : "r"(w0_3), "r"(w1_3));
                                    pairs0_3[k_3] = _prmt_48;
                                    uint32_t _prmt_49;
                                    asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_49) : "r"(w0_3), "r"(w1_3));
                                    pairs1_3[k_3] = _prmt_49;
                                }
                                uint32_t bf16x2_abs_7;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_7) : "r"(pairs0_3[0]));
                                unsigned int amax_pair_6 = bf16x2_abs_7;
                                #pragma unroll
                                for (int i_24 = 1; i_24 < 16; i_24++) {
                                    uint32_t bf16x2_abs_1_9;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_9) : "r"(pairs0_3[i_24]));
                                    uint32_t bf16x2_max_9;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_9) : "r"(amax_pair_6), "r"(bf16x2_abs_1_9));
                                    amax_pair_6 = bf16x2_max_9;
                                }
                                uint16_t bf16_max_6;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_6) : "h"((uint16_t)(amax_pair_6 & 65535)), "h"((uint16_t)(amax_pair_6 >> 16)));
                                float amax_6;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_6) : "h"((uint16_t)(bf16_max_6)));
                                float scale_6 = fmaxf(amax_6 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_50;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_50) : "f"(scale_6), "f"(scale_6));
                                unsigned int scale_byte_6 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_50 & 255;
                                unsigned int inverse_lane_6 = 254 - scale_byte_6 << 7;
                                unsigned int inverse_6 = inverse_lane_6 | inverse_lane_6 << 16;
                                unsigned int words_6[8];
                                #pragma unroll
                                for (int i_25 = 0; i_25 < 8; i_25++) {
                                    uint32_t bf16x2_mul_10;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(pairs0_3[i_25 * 2]), "r"(inverse_6));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(pairs0_3[i_25 * 2 + 1]), "r"(inverse_6));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_51;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_51) : "r"(bf16x2_mul_1_9));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_51) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_6[i_25] = (unsigned int)e4m3x2_bf16x2_9 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_51 << 16;
                                }
                                uint32_t bf16x2_abs_3_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_3_1) : "r"(pairs1_3[0]));
                                unsigned int amax_pair_3_1 = bf16x2_abs_3_1;
                                #pragma unroll
                                for (int i_26 = 1; i_26 < 16; i_26++) {
                                    uint32_t bf16x2_abs_1_10;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_10) : "r"(pairs1_3[i_26]));
                                    uint32_t bf16x2_max_10;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_10) : "r"(amax_pair_3_1), "r"(bf16x2_abs_1_10));
                                    amax_pair_3_1 = bf16x2_max_10;
                                }
                                uint16_t bf16_max_3_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_3_1) : "h"((uint16_t)(amax_pair_3_1 & 65535)), "h"((uint16_t)(amax_pair_3_1 >> 16)));
                                float amax_3_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_3_1) : "h"((uint16_t)(bf16_max_3_1)));
                                float scale_3_1 = fmaxf(amax_3_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_52;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_52) : "f"(scale_3_1), "f"(scale_3_1));
                                unsigned int scale_byte_3_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_52 & 255;
                                unsigned int inverse_lane_3_1 = 254 - scale_byte_3_1 << 7;
                                unsigned int inverse_3_1 = inverse_lane_3_1 | inverse_lane_3_1 << 16;
                                unsigned int words_3_1[8];
                                #pragma unroll
                                for (int i_27 = 0; i_27 < 8; i_27++) {
                                    uint32_t bf16x2_mul_11;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(pairs1_3[i_27 * 2]), "r"(inverse_3_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(pairs1_3[i_27 * 2 + 1]), "r"(inverse_3_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_53;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_53) : "r"(bf16x2_mul_1_10));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_53) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_3_1[i_27] = (unsigned int)e4m3x2_bf16x2_10 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_53 << 16;
                                }
                                int base0_3 = row0_3 * 32 + k_block_6 * 8;
                                int base1_3 = row1_3 * 32 + k_block_6 * 8;
                                const void* _smem_vector_store_ptr_54 = reinterpret_cast<const void*>(smem_v23 + (base0_3 + group_3 * 4));
                                uint32_t _smem_vector_store_addr_54 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_54));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_54), "r"(words_6[0]), "r"(words_6[1]), "r"(words_6[2]), "r"(words_6[3]));
                                const void* _smem_vector_store_ptr_55 = reinterpret_cast<const void*>(smem_v23 + (base0_3 + (1 - group_3) * 4));
                                uint32_t _smem_vector_store_addr_55 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_55));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_55), "r"(words_6[4]), "r"(words_6[5]), "r"(words_6[6]), "r"(words_6[7]));
                                const void* _smem_vector_store_ptr_56 = reinterpret_cast<const void*>(smem_v23 + (base1_3 + group_3 * 4));
                                uint32_t _smem_vector_store_addr_56 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_56));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_56), "r"(words_3_1[0]), "r"(words_3_1[1]), "r"(words_3_1[2]), "r"(words_3_1[3]));
                                const void* _smem_vector_store_ptr_57 = reinterpret_cast<const void*>(smem_v23 + (base1_3 + (1 - group_3) * 4));
                                uint32_t _smem_vector_store_addr_57 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_57));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_57), "r"(words_3_1[4]), "r"(words_3_1[5]), "r"(words_3_1[6]), "r"(words_3_1[7]));
                                {
                                    uint32_t _byte_58 = static_cast<uint32_t>(scale_byte_6) & 0xFFu;
                                    uint32_t _addr_58 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_3 + k_block_6));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                                }
                                {
                                    uint32_t _byte_59 = static_cast<uint32_t>(scale_byte_3_1) & 0xFFu;
                                    uint32_t _addr_59 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_3 + k_block_6));
                                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_59), "r"(_byte_59) : "memory");
                                }
                            }
                        } else {
                            int row_3 = tid - 128;
                            int rot_3 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                            unsigned int scale_word_3 = 0;
                            #pragma unroll 1
                            for (int j_7 = 0; j_7 < 4; j_7++) {
                                int k_block_7 = (j_7 + (tid - 128) / 2) % 4;
                                unsigned int pairs_3[16];
                                #pragma unroll
                                for (int i_28 = 0; i_28 < 4; i_28++) {
                                    int chunk_3 = row_3 * 128 + k_block_7 * 16 + (i_28 + rot_3) % 4 * 4;
                                    const void* _smem_load_ptr_60 = reinterpret_cast<const void*>(flat_view_5 + (16448 + chunk_3));
                                    uint32_t _smem_load_addr_60 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_60));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&pairs_3[i_28 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_3[(i_28 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_3[(i_28 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_3[(i_28 * 4) + 3]))
                                        : "r"(_smem_load_addr_60));
                                }
                                uint32_t bf16x2_abs_8;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_8) : "r"(pairs_3[0]));
                                unsigned int amax_pair_7 = bf16x2_abs_8;
                                #pragma unroll
                                for (int i_29 = 1; i_29 < 16; i_29++) {
                                    uint32_t bf16x2_abs_1_11;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_11) : "r"(pairs_3[i_29]));
                                    uint32_t bf16x2_max_11;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_11) : "r"(amax_pair_7), "r"(bf16x2_abs_1_11));
                                    amax_pair_7 = bf16x2_max_11;
                                }
                                uint16_t bf16_max_7;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_7) : "h"((uint16_t)(amax_pair_7 & 65535)), "h"((uint16_t)(amax_pair_7 >> 16)));
                                float amax_7;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_7) : "h"((uint16_t)(bf16_max_7)));
                                float scale_7 = fmaxf(amax_7 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_61;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_61) : "f"(scale_7), "f"(scale_7));
                                unsigned int scale_byte_7 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_61 & 255;
                                unsigned int inverse_lane_7 = 254 - scale_byte_7 << 7;
                                unsigned int inverse_7 = inverse_lane_7 | inverse_lane_7 << 16;
                                unsigned int words_7[8];
                                #pragma unroll
                                for (int i_30 = 0; i_30 < 8; i_30++) {
                                    uint32_t bf16x2_mul_12;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(pairs_3[i_30 * 2]), "r"(inverse_7));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(pairs_3[i_30 * 2 + 1]), "r"(inverse_7));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_62;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_62) : "r"(bf16x2_mul_1_11));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_62) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_7[i_30] = (unsigned int)e4m3x2_bf16x2_11 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_62 << 16;
                                }
                                scale_word_3 = scale_word_3 | scale_byte_7 << (unsigned int)(k_block_7 * 8);
                                #pragma unroll
                                for (int i_31 = 0; i_31 < 4; i_31++) {
                                    const void* _smem_vector_store_ptr_63 = reinterpret_cast<const void*>(smem_v19 + (row_3 * 32 + k_block_7 * 8 + (i_31 + rot_3) % 4 * 2));
                                    uint32_t _smem_vector_store_addr_63 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_63));
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_63), "r"(words_7[i_31 * 2]), "r"(words_7[i_31 * 2 + 1]));
                                }
                            }
                            smem_v21[row_3 % 32 * 4 + row_3 / 32] = scale_word_3;
                        }
                        __syncthreads();
                        if (tid == 0) {
                            int col_tile_3 = col_block * 4 + 2 + 1;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            tma_store_2d((&x_q_store), col_tile_3 * 128, row, smem_v19_addr);
                            tma_store_3d((&x_sc_store), 0, 0, row_tile_0 * k_tiles_0 + col_tile_3, smem_v21_addr);
                            tma_store_2d((&x_t_store), row, col_tile_3 * 128, smem_v23_addr);
                            tma_store_3d((&x_sc_t_store), 0, 0, col_tile_3 * macro_tiles_0 + row_tile_0, smem_v25_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                }
                if (tid == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_0 + row) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
                __syncthreads();
                task = next_task;
            }
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
                    int col_blocks_1 = (hidden + 1023) / 1024;
                    int macro_offset = 0;
                    int macro_tokens_0 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
                    int num_tasks = (macro_tokens_0 / 16 * col_blocks_1 + 6) / 7;
                    if (tid == 0) {
                        unsigned int _atomic_fetch_add_64 = atomicAdd(&combine_next[0], 1);
                        flat_view_8[0] = (int)_atomic_fetch_add_64;
                    }
                    __syncthreads();
                    int task_2 = flat_view_8[0];
                    __syncthreads();
                    while (task_2 < num_tasks) {
                        int col_blocks_2 = (hidden + 1023) / 1024;
                        int first_tile = task_2 * 7;
                        int macro_offset_2 = 0;
                        int macro_tokens_1 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
                        int valid_tiles = ((7) < (macro_tokens_1 / 16 * col_blocks_2 - first_tile) ? (7) : (macro_tokens_1 / 16 * col_blocks_2 - first_tile));
                        if (valid_tiles > 0) {
                            int first_row = first_tile / col_blocks_2 * 16 + tid;
                            int first_col = first_tile % col_blocks_2;
                            int rows[7];
                            int columns[7];
                            int peers[7];
                            int tokens_0[7];
                            unsigned int counts_1[7];
                            int row_4 = first_row;
                            int column = first_col;
                            #pragma unroll
                            for (int stage = 0; stage < 7; stage++) {
                                rows[stage] = row_4;
                                columns[stage] = column;
                                peers[stage] = -1;
                                tokens_0[stage] = -1;
                                if (valid_tiles > stage && tid < 16) {
                                    peers[stage] = schedule_rank[macro_offset_2 + row_4];
                                    tokens_0[stage] = schedule_token[macro_offset_2 + row_4];
                                }
                                counts_1[stage] = 0;
                                if (valid_tiles > stage) {
                                    if (stage == 0 || column == 0) {
                                        uint32_t _barrier_vote_65 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _barrier_vote_65;
                                    } else {
                                        counts_1[stage] = counts_1[stage - 1];
                                    }
                                }
                                column = column + 1;
                                if (column == col_blocks_2) {
                                    column = 0;
                                    row_4 = row_4 + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini = (macro_offset_2 + first_row) / mini_size;
                                int last_mini = (macro_offset_2 + (first_tile + valid_tiles - 1) / col_blocks_2 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini = first_mini; mini < last_mini + 1; mini++) {
                                    int mini_rows = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                                    int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                                    int poll_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_first) : "l"(y_ready + mini) : "memory");
                                    int poll = poll_first;
                                    while (poll < required) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_next) : "l"(y_ready + mini) : "memory");
                                        poll = poll_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                if (valid_tiles > 0) {
                                    int _min_66 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                    unsigned int chunk_bytes = (unsigned int)(_min_66 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr, counts_1[0] * chunk_bytes);
                                }
                                if (valid_tiles > 1) {
                                    int _min_67 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                    unsigned int chunk_bytes_1 = (unsigned int)(_min_67 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 8, counts_1[1] * chunk_bytes_1);
                                }
                                if (valid_tiles > 2) {
                                    int _min_68 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                    unsigned int chunk_bytes_2 = (unsigned int)(_min_68 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 16, counts_1[2] * chunk_bytes_2);
                                }
                                if (valid_tiles > 3) {
                                    int _min_69 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                    unsigned int chunk_bytes_3 = (unsigned int)(_min_69 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 24, counts_1[3] * chunk_bytes_3);
                                }
                                if (valid_tiles > 4) {
                                    int _min_70 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                    unsigned int chunk_bytes_4 = (unsigned int)(_min_70 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 32, counts_1[4] * chunk_bytes_4);
                                }
                                if (valid_tiles > 5) {
                                    int _min_71 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                    unsigned int chunk_bytes_5 = (unsigned int)(_min_71 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 40, counts_1[5] * chunk_bytes_5);
                                }
                                if (valid_tiles > 6) {
                                    int _min_72 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                    unsigned int chunk_bytes_6 = (unsigned int)(_min_72 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 48, counts_1[6] * chunk_bytes_6);
                                }
                            }
                            __syncthreads();
                            if (peers[0] >= 0) {
                                int chunk_cols_1 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)(tid * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)) * (unsigned long long)2), chunk_cols_1 * 2, combine_arrived_addr);
                            }
                            if (peers[1] >= 0) {
                                int chunk_cols_2 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)) * (unsigned long long)2), chunk_cols_2 * 2, combine_arrived_addr + 8);
                            }
                            if (peers[2] >= 0) {
                                int chunk_cols_3 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((32 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)) * (unsigned long long)2), chunk_cols_3 * 2, combine_arrived_addr + 16);
                            }
                            if (peers[3] >= 0) {
                                int chunk_cols_4 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((48 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)) * (unsigned long long)2), chunk_cols_4 * 2, combine_arrived_addr + 24);
                            }
                            if (peers[4] >= 0) {
                                int chunk_cols_5 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((64 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)) * (unsigned long long)2), chunk_cols_5 * 2, combine_arrived_addr + 32);
                            }
                            if (peers[5] >= 0) {
                                int chunk_cols_6 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((80 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)) * (unsigned long long)2), chunk_cols_6 * 2, combine_arrived_addr + 40);
                            }
                            if (peers[6] >= 0) {
                                int chunk_cols_7 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((96 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)) * (unsigned long long)2), chunk_cols_7 * 2, combine_arrived_addr + 48);
                            }
                            if (valid_tiles > 0) {
                                int _parity_combine_arrived = (int)((unsigned int)combine_seq_0 % 2);
                                mbarrier_wait(combine_arrived_addr, _parity_combine_arrived);
                                combine_seq_0 += 1;
                                if (peers[0] >= 0) {
                                    int _min_73 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                    unsigned int chunk_bytes_7 = (unsigned int)(_min_73 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_74 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[0]]) + ((unsigned long long)tokens_0[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_74), "r"(flat_view_addr_7 + (unsigned int)(tid * 2048)), "r"((uint32_t)(chunk_bytes_7))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 1) {
                                int _parity_combine_arrived_1 = (int)((unsigned int)combine_seq_1 % 2);
                                mbarrier_wait(combine_arrived_addr + 8, _parity_combine_arrived_1);
                                combine_seq_1 += 1;
                                if (peers[1] >= 0) {
                                    int _min_75 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                    unsigned int chunk_bytes_8 = (unsigned int)(_min_75 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_76 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[1]]) + ((unsigned long long)tokens_0[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_76), "r"(flat_view_addr_7 + (unsigned int)((16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_8))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 2) {
                                int _parity_combine_arrived_2 = (int)((unsigned int)combine_seq_2 % 2);
                                mbarrier_wait(combine_arrived_addr + 16, _parity_combine_arrived_2);
                                combine_seq_2 += 1;
                                if (peers[2] >= 0) {
                                    int _min_77 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                    unsigned int chunk_bytes_9 = (unsigned int)(_min_77 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_78 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[2]]) + ((unsigned long long)tokens_0[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_78), "r"(flat_view_addr_7 + (unsigned int)((32 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_9))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 3) {
                                int _parity_combine_arrived_3 = (int)((unsigned int)combine_seq_3 % 2);
                                mbarrier_wait(combine_arrived_addr + 24, _parity_combine_arrived_3);
                                combine_seq_3 += 1;
                                if (peers[3] >= 0) {
                                    int _min_79 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                    unsigned int chunk_bytes_10 = (unsigned int)(_min_79 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_80 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[3]]) + ((unsigned long long)tokens_0[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_80), "r"(flat_view_addr_7 + (unsigned int)((48 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_10))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 4) {
                                int _parity_combine_arrived_4 = (int)((unsigned int)combine_seq_4 % 2);
                                mbarrier_wait(combine_arrived_addr + 32, _parity_combine_arrived_4);
                                combine_seq_4 += 1;
                                if (peers[4] >= 0) {
                                    int _min_81 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                    unsigned int chunk_bytes_11 = (unsigned int)(_min_81 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_82 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[4]]) + ((unsigned long long)tokens_0[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_82), "r"(flat_view_addr_7 + (unsigned int)((64 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_11))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 5) {
                                int _parity_combine_arrived_5 = (int)((unsigned int)combine_seq_5 % 2);
                                mbarrier_wait(combine_arrived_addr + 40, _parity_combine_arrived_5);
                                combine_seq_5 += 1;
                                if (peers[5] >= 0) {
                                    int _min_83 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                    unsigned int chunk_bytes_12 = (unsigned int)(_min_83 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_84 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[5]]) + ((unsigned long long)tokens_0[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_84), "r"(flat_view_addr_7 + (unsigned int)((80 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_12))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles > 6) {
                                int _parity_combine_arrived_6 = (int)((unsigned int)combine_seq_6 % 2);
                                mbarrier_wait(combine_arrived_addr + 48, _parity_combine_arrived_6);
                                combine_seq_6 += 1;
                                if (peers[6] >= 0) {
                                    int _min_85 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                    unsigned int chunk_bytes_13 = (unsigned int)(_min_85 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_86 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[6]]) + ((unsigned long long)tokens_0[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_86), "r"(flat_view_addr_7 + (unsigned int)((96 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_13))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            int warp_1 = tid / 32;
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                        if (tid == 0) {
                            unsigned int _atomic_fetch_add_87 = atomicAdd(&combine_next[0], 1);
                            flat_view_8[0] = (int)_atomic_fetch_add_87;
                        }
                        __syncthreads();
                        task_2 = flat_view_8[0];
                        __syncthreads();
                    }
                }
                int _max_0 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task_1 = comm_cta; task_1 < ((macro > 0) ? _max_0 : 0); task_1 += comm_sms) {
                    if (combine_tasks > task_1) {
                        int col_blocks_3 = (hidden + 1023) / 1024;
                        int first_tile_1 = task_1 * 7;
                        int macro_offset_1 = macro * macro_size;
                        int macro_tokens_3 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                        int valid_tiles_1 = ((7) < (macro_tokens_3 / 16 * col_blocks_3 - first_tile_1) ? (7) : (macro_tokens_3 / 16 * col_blocks_3 - first_tile_1));
                        if (valid_tiles_1 > 0) {
                            int first_row_1 = first_tile_1 / col_blocks_3 * 16 + tid;
                            int first_col_1 = first_tile_1 % col_blocks_3;
                            int rows_1[7];
                            int columns_1[7];
                            int peers_1[7];
                            int tokens_0_1[7];
                            unsigned int counts_1_1[7];
                            int row_5 = first_row_1;
                            int column_1 = first_col_1;
                            #pragma unroll
                            for (int stage_1 = 0; stage_1 < 7; stage_1++) {
                                rows_1[stage_1] = row_5;
                                columns_1[stage_1] = column_1;
                                peers_1[stage_1] = -1;
                                tokens_0_1[stage_1] = -1;
                                if (valid_tiles_1 > stage_1 && tid < 16) {
                                    peers_1[stage_1] = schedule_rank[macro_offset_1 + row_5];
                                    tokens_0_1[stage_1] = schedule_token[macro_offset_1 + row_5];
                                }
                                counts_1_1[stage_1] = 0;
                                if (valid_tiles_1 > stage_1) {
                                    if (stage_1 == 0 || column_1 == 0) {
                                        uint32_t _barrier_vote_88 = __syncthreads_count(peers_1[stage_1] >= 0);
                                        counts_1_1[stage_1] = _barrier_vote_88;
                                    } else {
                                        counts_1_1[stage_1] = counts_1_1[stage_1 - 1];
                                    }
                                }
                                column_1 = column_1 + 1;
                                if (column_1 == col_blocks_3) {
                                    column_1 = 0;
                                    row_5 = row_5 + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini_1 = (macro_offset_1 + first_row_1) / mini_size;
                                int last_mini_1 = (macro_offset_1 + (first_tile_1 + valid_tiles_1 - 1) / col_blocks_3 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini_1 = first_mini_1; mini_1 < last_mini_1 + 1; mini_1++) {
                                    int mini_rows_1 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    int poll_1_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_first) : "l"(y_ready + mini_1) : "memory");
                                    int poll_1 = poll_1_first;
                                    while (poll_1 < required_1) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_1_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_next) : "l"(y_ready + mini_1) : "memory");
                                        poll_1 = poll_1_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                if (valid_tiles_1 > 0) {
                                    int _min_89 = ((1024) < (hidden - columns_1[0] * 1024) ? (1024) : (hidden - columns_1[0] * 1024));
                                    unsigned int chunk_bytes_14 = (unsigned int)(_min_89 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr, counts_1_1[0] * chunk_bytes_14);
                                }
                                if (valid_tiles_1 > 1) {
                                    int _min_90 = ((1024) < (hidden - columns_1[1] * 1024) ? (1024) : (hidden - columns_1[1] * 1024));
                                    unsigned int chunk_bytes_15 = (unsigned int)(_min_90 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 8, counts_1_1[1] * chunk_bytes_15);
                                }
                                if (valid_tiles_1 > 2) {
                                    int _min_91 = ((1024) < (hidden - columns_1[2] * 1024) ? (1024) : (hidden - columns_1[2] * 1024));
                                    unsigned int chunk_bytes_16 = (unsigned int)(_min_91 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 16, counts_1_1[2] * chunk_bytes_16);
                                }
                                if (valid_tiles_1 > 3) {
                                    int _min_92 = ((1024) < (hidden - columns_1[3] * 1024) ? (1024) : (hidden - columns_1[3] * 1024));
                                    unsigned int chunk_bytes_17 = (unsigned int)(_min_92 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 24, counts_1_1[3] * chunk_bytes_17);
                                }
                                if (valid_tiles_1 > 4) {
                                    int _min_93 = ((1024) < (hidden - columns_1[4] * 1024) ? (1024) : (hidden - columns_1[4] * 1024));
                                    unsigned int chunk_bytes_18 = (unsigned int)(_min_93 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 32, counts_1_1[4] * chunk_bytes_18);
                                }
                                if (valid_tiles_1 > 5) {
                                    int _min_94 = ((1024) < (hidden - columns_1[5] * 1024) ? (1024) : (hidden - columns_1[5] * 1024));
                                    unsigned int chunk_bytes_19 = (unsigned int)(_min_94 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 40, counts_1_1[5] * chunk_bytes_19);
                                }
                                if (valid_tiles_1 > 6) {
                                    int _min_95 = ((1024) < (hidden - columns_1[6] * 1024) ? (1024) : (hidden - columns_1[6] * 1024));
                                    unsigned int chunk_bytes_20 = (unsigned int)(_min_95 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 48, counts_1_1[6] * chunk_bytes_20);
                                }
                            }
                            __syncthreads();
                            if (peers_1[0] >= 0) {
                                int chunk_cols_8 = ((1024) < (hidden - columns_1[0] * 1024) ? (1024) : (hidden - columns_1[0] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)(tid * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[0] * (unsigned long long)hidden + (unsigned long long)(columns_1[0] * 1024)) * (unsigned long long)2), chunk_cols_8 * 2, combine_arrived_addr);
                            }
                            if (peers_1[1] >= 0) {
                                int chunk_cols_9 = ((1024) < (hidden - columns_1[1] * 1024) ? (1024) : (hidden - columns_1[1] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[1] * (unsigned long long)hidden + (unsigned long long)(columns_1[1] * 1024)) * (unsigned long long)2), chunk_cols_9 * 2, combine_arrived_addr + 8);
                            }
                            if (peers_1[2] >= 0) {
                                int chunk_cols_10 = ((1024) < (hidden - columns_1[2] * 1024) ? (1024) : (hidden - columns_1[2] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((32 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[2] * (unsigned long long)hidden + (unsigned long long)(columns_1[2] * 1024)) * (unsigned long long)2), chunk_cols_10 * 2, combine_arrived_addr + 16);
                            }
                            if (peers_1[3] >= 0) {
                                int chunk_cols_11 = ((1024) < (hidden - columns_1[3] * 1024) ? (1024) : (hidden - columns_1[3] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((48 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[3] * (unsigned long long)hidden + (unsigned long long)(columns_1[3] * 1024)) * (unsigned long long)2), chunk_cols_11 * 2, combine_arrived_addr + 24);
                            }
                            if (peers_1[4] >= 0) {
                                int chunk_cols_12 = ((1024) < (hidden - columns_1[4] * 1024) ? (1024) : (hidden - columns_1[4] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((64 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[4] * (unsigned long long)hidden + (unsigned long long)(columns_1[4] * 1024)) * (unsigned long long)2), chunk_cols_12 * 2, combine_arrived_addr + 32);
                            }
                            if (peers_1[5] >= 0) {
                                int chunk_cols_13 = ((1024) < (hidden - columns_1[5] * 1024) ? (1024) : (hidden - columns_1[5] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((80 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[5] * (unsigned long long)hidden + (unsigned long long)(columns_1[5] * 1024)) * (unsigned long long)2), chunk_cols_13 * 2, combine_arrived_addr + 40);
                            }
                            if (peers_1[6] >= 0) {
                                int chunk_cols_14 = ((1024) < (hidden - columns_1[6] * 1024) ? (1024) : (hidden - columns_1[6] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((96 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_1[6] * (unsigned long long)hidden + (unsigned long long)(columns_1[6] * 1024)) * (unsigned long long)2), chunk_cols_14 * 2, combine_arrived_addr + 48);
                            }
                            if (valid_tiles_1 > 0) {
                                int _parity_combine_arrived_7 = (int)((unsigned int)combine_seq_0 % 2);
                                mbarrier_wait(combine_arrived_addr, _parity_combine_arrived_7);
                                combine_seq_0 += 1;
                                if (peers_1[0] >= 0) {
                                    int _min_96 = ((1024) < (hidden - columns_1[0] * 1024) ? (1024) : (hidden - columns_1[0] * 1024));
                                    unsigned int chunk_bytes_21 = (unsigned int)(_min_96 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_97 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[0]]) + ((unsigned long long)tokens_0_1[0] * (unsigned long long)hidden + (unsigned long long)(columns_1[0] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_97), "r"(flat_view_addr_7 + (unsigned int)(tid * 2048)), "r"((uint32_t)(chunk_bytes_21))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles_1 > 1) {
                                int _parity_combine_arrived_8 = (int)((unsigned int)combine_seq_1 % 2);
                                mbarrier_wait(combine_arrived_addr + 8, _parity_combine_arrived_8);
                                combine_seq_1 += 1;
                                if (peers_1[1] >= 0) {
                                    int _min_98 = ((1024) < (hidden - columns_1[1] * 1024) ? (1024) : (hidden - columns_1[1] * 1024));
                                    unsigned int chunk_bytes_22 = (unsigned int)(_min_98 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_99 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[1]]) + ((unsigned long long)tokens_0_1[1] * (unsigned long long)hidden + (unsigned long long)(columns_1[1] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_99), "r"(flat_view_addr_7 + (unsigned int)((16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_22))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles_1 > 2) {
                                int _parity_combine_arrived_9 = (int)((unsigned int)combine_seq_2 % 2);
                                mbarrier_wait(combine_arrived_addr + 16, _parity_combine_arrived_9);
                                combine_seq_2 += 1;
                                if (peers_1[2] >= 0) {
                                    int _min_100 = ((1024) < (hidden - columns_1[2] * 1024) ? (1024) : (hidden - columns_1[2] * 1024));
                                    unsigned int chunk_bytes_23 = (unsigned int)(_min_100 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_101 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[2]]) + ((unsigned long long)tokens_0_1[2] * (unsigned long long)hidden + (unsigned long long)(columns_1[2] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_101), "r"(flat_view_addr_7 + (unsigned int)((32 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_23))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles_1 > 3) {
                                int _parity_combine_arrived_10 = (int)((unsigned int)combine_seq_3 % 2);
                                mbarrier_wait(combine_arrived_addr + 24, _parity_combine_arrived_10);
                                combine_seq_3 += 1;
                                if (peers_1[3] >= 0) {
                                    int _min_102 = ((1024) < (hidden - columns_1[3] * 1024) ? (1024) : (hidden - columns_1[3] * 1024));
                                    unsigned int chunk_bytes_24 = (unsigned int)(_min_102 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_103 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[3]]) + ((unsigned long long)tokens_0_1[3] * (unsigned long long)hidden + (unsigned long long)(columns_1[3] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_103), "r"(flat_view_addr_7 + (unsigned int)((48 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_24))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles_1 > 4) {
                                int _parity_combine_arrived_11 = (int)((unsigned int)combine_seq_4 % 2);
                                mbarrier_wait(combine_arrived_addr + 32, _parity_combine_arrived_11);
                                combine_seq_4 += 1;
                                if (peers_1[4] >= 0) {
                                    int _min_104 = ((1024) < (hidden - columns_1[4] * 1024) ? (1024) : (hidden - columns_1[4] * 1024));
                                    unsigned int chunk_bytes_25 = (unsigned int)(_min_104 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_105 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[4]]) + ((unsigned long long)tokens_0_1[4] * (unsigned long long)hidden + (unsigned long long)(columns_1[4] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_105), "r"(flat_view_addr_7 + (unsigned int)((64 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_25))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles_1 > 5) {
                                int _parity_combine_arrived_12 = (int)((unsigned int)combine_seq_5 % 2);
                                mbarrier_wait(combine_arrived_addr + 40, _parity_combine_arrived_12);
                                combine_seq_5 += 1;
                                if (peers_1[5] >= 0) {
                                    int _min_106 = ((1024) < (hidden - columns_1[5] * 1024) ? (1024) : (hidden - columns_1[5] * 1024));
                                    unsigned int chunk_bytes_26 = (unsigned int)(_min_106 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_107 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[5]]) + ((unsigned long long)tokens_0_1[5] * (unsigned long long)hidden + (unsigned long long)(columns_1[5] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_107), "r"(flat_view_addr_7 + (unsigned int)((80 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_26))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (valid_tiles_1 > 6) {
                                int _parity_combine_arrived_13 = (int)((unsigned int)combine_seq_6 % 2);
                                mbarrier_wait(combine_arrived_addr + 48, _parity_combine_arrived_13);
                                combine_seq_6 += 1;
                                if (peers_1[6] >= 0) {
                                    int _min_108 = ((1024) < (hidden - columns_1[6] * 1024) ? (1024) : (hidden - columns_1[6] * 1024));
                                    unsigned int chunk_bytes_27 = (unsigned int)(_min_108 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_109 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_1[6]]) + ((unsigned long long)tokens_0_1[6] * (unsigned long long)hidden + (unsigned long long)(columns_1[6] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_109), "r"(flat_view_addr_7 + (unsigned int)((96 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_27))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            int warp_2 = tid / 32;
                            if (tid % 32 == 0 && warp_2 < valid_tiles_1 && macro > 0) {
                                int row_done = (first_tile_1 + warp_2) / col_blocks_3 * 16;
                                bool enabled_value = 1;
                                if (enabled_value != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_1 + row_done) / 128))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                    }
                    if (dispatch_tasks > task_1) {
                        int col_blocks_4 = (hidden + 511) / 512;
                        int macro_offset_3 = (macro - 1) * macro_size;
                        int row_6 = task_1 / col_blocks_4 * 128;
                        int col_block_1 = task_1 % col_blocks_4;
                        int chunk_cols_15 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                        int lo_cols_1 = ((256) < (chunk_cols_15) ? (256) : (chunk_cols_15));
                        int hi_cols_1 = chunk_cols_15 - lo_cols_1;
                        if (task_1 < comm_sms || task_1 < combine_tasks) {
                            int peer_3 = -1;
                            int peer_token_3 = -1;
                            if (tid < 128) {
                                peer_3 = schedule_rank[macro_offset_3 + row_6 + tid];
                                peer_token_3 = schedule_token[macro_offset_3 + row_6 + tid];
                            }
                            uint32_t cta_count_3 = __syncthreads_count(peer_3 >= 0);
                            if (tid == 0) {
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_3 * (unsigned int)(lo_cols_1 * 2));
                            }
                            __syncthreads();
                            if (peer_3 >= 0) {
                                cp_async_bulk_gmem2smem(smem_v26_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_3])) + (unsigned long long)((unsigned long long)(peer_token_3 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2), lo_cols_1 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_3 = 0; vec_3 < 32; vec_3++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_5) + (tid * 512 + vec_3 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                        }
                        if (hi_cols_1 > 0) {
                            int peer_4 = -1;
                            int peer_token_4 = -1;
                            if (tid < 128) {
                                peer_4 = schedule_rank[macro_offset_3 + row_6 + tid];
                                peer_token_4 = schedule_token[macro_offset_3 + row_6 + tid];
                            }
                            uint32_t cta_count_4 = __syncthreads_count(peer_4 >= 0);
                            if (tid == 0) {
                                mbarrier_arrive_expect_tx(dispatch_arrived_hi_addr, cta_count_4 * (unsigned int)(hi_cols_1 * 2));
                            }
                            __syncthreads();
                            if (peer_4 >= 0) {
                                cp_async_bulk_gmem2smem(smem_v28_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_4])) + (unsigned long long)((unsigned long long)(peer_token_4 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512) + 256ull) * (unsigned long long)2), hi_cols_1 * 2, dispatch_arrived_hi_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_4 = 0; vec_4 < 32; vec_4++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_5) + (65536 + tid * 512 + vec_4 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                        }
                        if (tid == 0) {
                            int previous_offset = macro * macro_size;
                            int previous_tokens = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                            if (row_6 < previous_tokens) {
                                int previous_mini = (previous_offset + row_6) / mini_size;
                                int mini_rows_2 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                int required_2 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                                bool enabled_value_1 = 1;
                                if (enabled_value_1 != 0) {
                                    int poll_2_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_first) : "l"(y_ready + previous_mini) : "memory");
                                    int poll_2 = poll_2_first;
                                    while (poll_2 < required_2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_2_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_next) : "l"(y_ready + previous_mini) : "memory");
                                        poll_2 = poll_2_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
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
                                int group_4 = tid % 2;
                                #pragma unroll 1
                                for (int j_8 = 0; j_8 < 2; j_8++) {
                                    int k_block_8 = (half_4 * 2 + j_8 + tid / 2) % 4;
                                    unsigned int pairs0_4[16];
                                    unsigned int pairs1_4[16];
                                    #pragma unroll
                                    for (int k_4 = 0; k_4 < 16; k_4++) {
                                        int col_4 = k_block_8 * 32 + (k_4 * 2 + group_4 * 16) % 32;
                                        unsigned int w0_4 = flat_view_5[col_4 * word_stride_4 + r_4];
                                        unsigned int w1_4 = flat_view_5[(col_4 + 1) * word_stride_4 + r_4];
                                        uint32_t _prmt_110;
                                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_110) : "r"(w0_4), "r"(w1_4));
                                        pairs0_4[k_4] = _prmt_110;
                                        uint32_t _prmt_111;
                                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_111) : "r"(w0_4), "r"(w1_4));
                                        pairs1_4[k_4] = _prmt_111;
                                    }
                                    uint32_t bf16x2_abs_9;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_9) : "r"(pairs0_4[0]));
                                    unsigned int amax_pair_8 = bf16x2_abs_9;
                                    #pragma unroll
                                    for (int i_32 = 1; i_32 < 16; i_32++) {
                                        uint32_t bf16x2_abs_1_12;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_12) : "r"(pairs0_4[i_32]));
                                        uint32_t bf16x2_max_12;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_12) : "r"(amax_pair_8), "r"(bf16x2_abs_1_12));
                                        amax_pair_8 = bf16x2_max_12;
                                    }
                                    uint16_t bf16_max_8;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_8) : "h"((uint16_t)(amax_pair_8 & 65535)), "h"((uint16_t)(amax_pair_8 >> 16)));
                                    float amax_8;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8) : "h"((uint16_t)(bf16_max_8)));
                                    float scale_8 = fmaxf(amax_8 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_112;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_112) : "f"(scale_8), "f"(scale_8));
                                    unsigned int scale_byte_8 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_112 & 255;
                                    unsigned int inverse_lane_8 = 254 - scale_byte_8 << 7;
                                    unsigned int inverse_8 = inverse_lane_8 | inverse_lane_8 << 16;
                                    unsigned int words_8[8];
                                    #pragma unroll
                                    for (int i_33 = 0; i_33 < 8; i_33++) {
                                        uint32_t bf16x2_mul_13;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(pairs0_4[i_33 * 2]), "r"(inverse_8));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(pairs0_4[i_33 * 2 + 1]), "r"(inverse_8));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_113;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_113) : "r"(bf16x2_mul_1_12));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_113) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_8[i_33] = (unsigned int)e4m3x2_bf16x2_12 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_113 << 16;
                                    }
                                    uint32_t bf16x2_abs_4_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_4_1) : "r"(pairs1_4[0]));
                                    unsigned int amax_pair_4_1 = bf16x2_abs_4_1;
                                    #pragma unroll
                                    for (int i_34 = 1; i_34 < 16; i_34++) {
                                        uint32_t bf16x2_abs_1_13;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_13) : "r"(pairs1_4[i_34]));
                                        uint32_t bf16x2_max_13;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_13) : "r"(amax_pair_4_1), "r"(bf16x2_abs_1_13));
                                        amax_pair_4_1 = bf16x2_max_13;
                                    }
                                    uint16_t bf16_max_4_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_4_1) : "h"((uint16_t)(amax_pair_4_1 & 65535)), "h"((uint16_t)(amax_pair_4_1 >> 16)));
                                    float amax_4_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_4_1) : "h"((uint16_t)(bf16_max_4_1)));
                                    float scale_4_1 = fmaxf(amax_4_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_114;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_114) : "f"(scale_4_1), "f"(scale_4_1));
                                    unsigned int scale_byte_4_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_114 & 255;
                                    unsigned int inverse_lane_4_1 = 254 - scale_byte_4_1 << 7;
                                    unsigned int inverse_4_1 = inverse_lane_4_1 | inverse_lane_4_1 << 16;
                                    unsigned int words_4_1[8];
                                    #pragma unroll
                                    for (int i_35 = 0; i_35 < 8; i_35++) {
                                        uint32_t bf16x2_mul_14;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(pairs1_4[i_35 * 2]), "r"(inverse_4_1));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(pairs1_4[i_35 * 2 + 1]), "r"(inverse_4_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_115;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_115) : "r"(bf16x2_mul_1_13));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_115) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_4_1[i_35] = (unsigned int)e4m3x2_bf16x2_13 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_115 << 16;
                                    }
                                    int base0_4 = row0_4 * 32 + k_block_8 * 8;
                                    int base1_4 = row1_4 * 32 + k_block_8 * 8;
                                    const void* _smem_vector_store_ptr_116 = reinterpret_cast<const void*>(smem_v22 + (base0_4 + group_4 * 4));
                                    uint32_t _smem_vector_store_addr_116 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_116));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_116), "r"(words_8[0]), "r"(words_8[1]), "r"(words_8[2]), "r"(words_8[3]));
                                    const void* _smem_vector_store_ptr_117 = reinterpret_cast<const void*>(smem_v22 + (base0_4 + (1 - group_4) * 4));
                                    uint32_t _smem_vector_store_addr_117 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_117));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_117), "r"(words_8[4]), "r"(words_8[5]), "r"(words_8[6]), "r"(words_8[7]));
                                    const void* _smem_vector_store_ptr_118 = reinterpret_cast<const void*>(smem_v22 + (base1_4 + group_4 * 4));
                                    uint32_t _smem_vector_store_addr_118 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_118));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_118), "r"(words_4_1[0]), "r"(words_4_1[1]), "r"(words_4_1[2]), "r"(words_4_1[3]));
                                    const void* _smem_vector_store_ptr_119 = reinterpret_cast<const void*>(smem_v22 + (base1_4 + (1 - group_4) * 4));
                                    uint32_t _smem_vector_store_addr_119 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_119));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_119), "r"(words_4_1[4]), "r"(words_4_1[5]), "r"(words_4_1[6]), "r"(words_4_1[7]));
                                    {
                                        uint32_t _byte_120 = static_cast<uint32_t>(scale_byte_8) & 0xFFu;
                                        uint32_t _addr_120 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0_4 + k_block_8));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_120), "r"(_byte_120) : "memory");
                                    }
                                    {
                                        uint32_t _byte_121 = static_cast<uint32_t>(scale_byte_4_1) & 0xFFu;
                                        uint32_t _addr_121 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1_4 + k_block_8));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_121), "r"(_byte_121) : "memory");
                                    }
                                }
                            } else {
                                int row_4_1 = tid - 128;
                                int rot_4 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                                unsigned int scale_word_4 = 0;
                                #pragma unroll 1
                                for (int j_9 = 0; j_9 < 4; j_9++) {
                                    int k_block_9 = (j_9 + (tid - 128) / 2) % 4;
                                    unsigned int pairs_4[16];
                                    #pragma unroll
                                    for (int i_36 = 0; i_36 < 4; i_36++) {
                                        int chunk_4 = row_4_1 * 128 + k_block_9 * 16 + (i_36 + rot_4) % 4 * 4;
                                        const void* _smem_load_ptr_122 = reinterpret_cast<const void*>(flat_view_5 + chunk_4);
                                        uint32_t _smem_load_addr_122 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_122));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&pairs_4[i_36 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_4[(i_36 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_4[(i_36 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_4[(i_36 * 4) + 3]))
                                            : "r"(_smem_load_addr_122));
                                    }
                                    uint32_t bf16x2_abs_10;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_10) : "r"(pairs_4[0]));
                                    unsigned int amax_pair_9 = bf16x2_abs_10;
                                    #pragma unroll
                                    for (int i_37 = 1; i_37 < 16; i_37++) {
                                        uint32_t bf16x2_abs_1_14;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_14) : "r"(pairs_4[i_37]));
                                        uint32_t bf16x2_max_14;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_14) : "r"(amax_pair_9), "r"(bf16x2_abs_1_14));
                                        amax_pair_9 = bf16x2_max_14;
                                    }
                                    uint16_t bf16_max_9;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_9) : "h"((uint16_t)(amax_pair_9 & 65535)), "h"((uint16_t)(amax_pair_9 >> 16)));
                                    float amax_9;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_9) : "h"((uint16_t)(bf16_max_9)));
                                    float scale_9 = fmaxf(amax_9 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_123;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_123) : "f"(scale_9), "f"(scale_9));
                                    unsigned int scale_byte_9 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_123 & 255;
                                    unsigned int inverse_lane_9 = 254 - scale_byte_9 << 7;
                                    unsigned int inverse_9 = inverse_lane_9 | inverse_lane_9 << 16;
                                    unsigned int words_9[8];
                                    #pragma unroll
                                    for (int i_38 = 0; i_38 < 8; i_38++) {
                                        uint32_t bf16x2_mul_15;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(pairs_4[i_38 * 2]), "r"(inverse_9));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(pairs_4[i_38 * 2 + 1]), "r"(inverse_9));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_124;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_124) : "r"(bf16x2_mul_1_14));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_124) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_9[i_38] = (unsigned int)e4m3x2_bf16x2_14 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_124 << 16;
                                    }
                                    scale_word_4 = scale_word_4 | scale_byte_9 << (unsigned int)(k_block_9 * 8);
                                    #pragma unroll
                                    for (int i_39 = 0; i_39 < 4; i_39++) {
                                        const void* _smem_vector_store_ptr_125 = reinterpret_cast<const void*>(smem_v18 + (row_4_1 * 32 + k_block_9 * 8 + (i_39 + rot_4) % 4 * 2));
                                        uint32_t _smem_vector_store_addr_125 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_125));
                                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_125), "r"(words_9[i_39 * 2]), "r"(words_9[i_39 * 2 + 1]));
                                    }
                                }
                                smem_v20[row_4_1 % 32 * 4 + row_4_1 / 32] = scale_word_4;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_4 = col_block_1 * 4;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_4 * 128, row_6, smem_v18_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_4, smem_v20_addr);
                                tma_store_2d((&x_t_store), row_6, col_tile_4 * 128, smem_v22_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_4 * macro_tiles_1 + row_tile_1, smem_v24_addr);
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
                                int group_5 = tid % 2;
                                #pragma unroll 1
                                for (int j_10 = 0; j_10 < 2; j_10++) {
                                    int k_block_10 = (half_5 * 2 + j_10 + tid / 2) % 4;
                                    unsigned int pairs0_5[16];
                                    unsigned int pairs1_5[16];
                                    #pragma unroll
                                    for (int k_5 = 0; k_5 < 16; k_5++) {
                                        int col_5 = k_block_10 * 32 + (k_5 * 2 + group_5 * 16) % 32;
                                        unsigned int w0_5 = flat_view_5[64 + col_5 * word_stride_5 + r_5];
                                        unsigned int w1_5 = flat_view_5[64 + (col_5 + 1) * word_stride_5 + r_5];
                                        uint32_t _prmt_126;
                                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_126) : "r"(w0_5), "r"(w1_5));
                                        pairs0_5[k_5] = _prmt_126;
                                        uint32_t _prmt_127;
                                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_127) : "r"(w0_5), "r"(w1_5));
                                        pairs1_5[k_5] = _prmt_127;
                                    }
                                    uint32_t bf16x2_abs_11;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_11) : "r"(pairs0_5[0]));
                                    unsigned int amax_pair_10 = bf16x2_abs_11;
                                    #pragma unroll
                                    for (int i_40 = 1; i_40 < 16; i_40++) {
                                        uint32_t bf16x2_abs_1_15;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_15) : "r"(pairs0_5[i_40]));
                                        uint32_t bf16x2_max_15;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_15) : "r"(amax_pair_10), "r"(bf16x2_abs_1_15));
                                        amax_pair_10 = bf16x2_max_15;
                                    }
                                    uint16_t bf16_max_10;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_10) : "h"((uint16_t)(amax_pair_10 & 65535)), "h"((uint16_t)(amax_pair_10 >> 16)));
                                    float amax_10;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10) : "h"((uint16_t)(bf16_max_10)));
                                    float scale_10 = fmaxf(amax_10 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_128;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_128) : "f"(scale_10), "f"(scale_10));
                                    unsigned int scale_byte_10 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_128 & 255;
                                    unsigned int inverse_lane_10 = 254 - scale_byte_10 << 7;
                                    unsigned int inverse_10 = inverse_lane_10 | inverse_lane_10 << 16;
                                    unsigned int words_10[8];
                                    #pragma unroll
                                    for (int i_41 = 0; i_41 < 8; i_41++) {
                                        uint32_t bf16x2_mul_16;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(pairs0_5[i_41 * 2]), "r"(inverse_10));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(pairs0_5[i_41 * 2 + 1]), "r"(inverse_10));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_129;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "r"(bf16x2_mul_1_15));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_129) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_10[i_41] = (unsigned int)e4m3x2_bf16x2_15 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_129 << 16;
                                    }
                                    uint32_t bf16x2_abs_5_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_5_1) : "r"(pairs1_5[0]));
                                    unsigned int amax_pair_5_1 = bf16x2_abs_5_1;
                                    #pragma unroll
                                    for (int i_42 = 1; i_42 < 16; i_42++) {
                                        uint32_t bf16x2_abs_1_16;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_16) : "r"(pairs1_5[i_42]));
                                        uint32_t bf16x2_max_16;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_16) : "r"(amax_pair_5_1), "r"(bf16x2_abs_1_16));
                                        amax_pair_5_1 = bf16x2_max_16;
                                    }
                                    uint16_t bf16_max_5_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_5_1) : "h"((uint16_t)(amax_pair_5_1 & 65535)), "h"((uint16_t)(amax_pair_5_1 >> 16)));
                                    float amax_5_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_5_1) : "h"((uint16_t)(bf16_max_5_1)));
                                    float scale_5_1 = fmaxf(amax_5_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_130;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_130) : "f"(scale_5_1), "f"(scale_5_1));
                                    unsigned int scale_byte_5_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_130 & 255;
                                    unsigned int inverse_lane_5_1 = 254 - scale_byte_5_1 << 7;
                                    unsigned int inverse_5_1 = inverse_lane_5_1 | inverse_lane_5_1 << 16;
                                    unsigned int words_5_1[8];
                                    #pragma unroll
                                    for (int i_43 = 0; i_43 < 8; i_43++) {
                                        uint32_t bf16x2_mul_17;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(pairs1_5[i_43 * 2]), "r"(inverse_5_1));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(pairs1_5[i_43 * 2 + 1]), "r"(inverse_5_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_131;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_131) : "r"(bf16x2_mul_1_16));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_131) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_5_1[i_43] = (unsigned int)e4m3x2_bf16x2_16 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_131 << 16;
                                    }
                                    int base0_5 = row0_5 * 32 + k_block_10 * 8;
                                    int base1_5 = row1_5 * 32 + k_block_10 * 8;
                                    const void* _smem_vector_store_ptr_132 = reinterpret_cast<const void*>(smem_v23 + (base0_5 + group_5 * 4));
                                    uint32_t _smem_vector_store_addr_132 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_132));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_132), "r"(words_10[0]), "r"(words_10[1]), "r"(words_10[2]), "r"(words_10[3]));
                                    const void* _smem_vector_store_ptr_133 = reinterpret_cast<const void*>(smem_v23 + (base0_5 + (1 - group_5) * 4));
                                    uint32_t _smem_vector_store_addr_133 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_133));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_133), "r"(words_10[4]), "r"(words_10[5]), "r"(words_10[6]), "r"(words_10[7]));
                                    const void* _smem_vector_store_ptr_134 = reinterpret_cast<const void*>(smem_v23 + (base1_5 + group_5 * 4));
                                    uint32_t _smem_vector_store_addr_134 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_134));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_134), "r"(words_5_1[0]), "r"(words_5_1[1]), "r"(words_5_1[2]), "r"(words_5_1[3]));
                                    const void* _smem_vector_store_ptr_135 = reinterpret_cast<const void*>(smem_v23 + (base1_5 + (1 - group_5) * 4));
                                    uint32_t _smem_vector_store_addr_135 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_135));
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_135), "r"(words_5_1[4]), "r"(words_5_1[5]), "r"(words_5_1[6]), "r"(words_5_1[7]));
                                    {
                                        uint32_t _byte_136 = static_cast<uint32_t>(scale_byte_10) & 0xFFu;
                                        uint32_t _addr_136 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_5 + k_block_10));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_136), "r"(_byte_136) : "memory");
                                    }
                                    {
                                        uint32_t _byte_137 = static_cast<uint32_t>(scale_byte_5_1) & 0xFFu;
                                        uint32_t _addr_137 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_5 + k_block_10));
                                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_137), "r"(_byte_137) : "memory");
                                    }
                                }
                            } else {
                                int row_5_1 = tid - 128;
                                int rot_5 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                                unsigned int scale_word_5 = 0;
                                #pragma unroll 1
                                for (int j_11 = 0; j_11 < 4; j_11++) {
                                    int k_block_11 = (j_11 + (tid - 128) / 2) % 4;
                                    unsigned int pairs_5[16];
                                    #pragma unroll
                                    for (int i_44 = 0; i_44 < 4; i_44++) {
                                        int chunk_5 = row_5_1 * 128 + k_block_11 * 16 + (i_44 + rot_5) % 4 * 4;
                                        const void* _smem_load_ptr_138 = reinterpret_cast<const void*>(flat_view_5 + (64 + chunk_5));
                                        uint32_t _smem_load_addr_138 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_138));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&pairs_5[i_44 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_5[(i_44 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_5[(i_44 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_5[(i_44 * 4) + 3]))
                                            : "r"(_smem_load_addr_138));
                                    }
                                    uint32_t bf16x2_abs_12;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_12) : "r"(pairs_5[0]));
                                    unsigned int amax_pair_11 = bf16x2_abs_12;
                                    #pragma unroll
                                    for (int i_45 = 1; i_45 < 16; i_45++) {
                                        uint32_t bf16x2_abs_1_17;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_17) : "r"(pairs_5[i_45]));
                                        uint32_t bf16x2_max_17;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_17) : "r"(amax_pair_11), "r"(bf16x2_abs_1_17));
                                        amax_pair_11 = bf16x2_max_17;
                                    }
                                    uint16_t bf16_max_11;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_11) : "h"((uint16_t)(amax_pair_11 & 65535)), "h"((uint16_t)(amax_pair_11 >> 16)));
                                    float amax_11;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11) : "h"((uint16_t)(bf16_max_11)));
                                    float scale_11 = fmaxf(amax_11 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_139;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_139) : "f"(scale_11), "f"(scale_11));
                                    unsigned int scale_byte_11 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_139 & 255;
                                    unsigned int inverse_lane_11 = 254 - scale_byte_11 << 7;
                                    unsigned int inverse_11 = inverse_lane_11 | inverse_lane_11 << 16;
                                    unsigned int words_11[8];
                                    #pragma unroll
                                    for (int i_46 = 0; i_46 < 8; i_46++) {
                                        uint32_t bf16x2_mul_18;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(pairs_5[i_46 * 2]), "r"(inverse_11));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(pairs_5[i_46 * 2 + 1]), "r"(inverse_11));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_140;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_140) : "r"(bf16x2_mul_1_17));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_140) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_11[i_46] = (unsigned int)e4m3x2_bf16x2_17 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_140 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_11 << (unsigned int)(k_block_11 * 8);
                                    #pragma unroll
                                    for (int i_47 = 0; i_47 < 4; i_47++) {
                                        const void* _smem_vector_store_ptr_141 = reinterpret_cast<const void*>(smem_v19 + (row_5_1 * 32 + k_block_11 * 8 + (i_47 + rot_5) % 4 * 2));
                                        uint32_t _smem_vector_store_addr_141 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_141));
                                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_141), "r"(words_11[i_47 * 2]), "r"(words_11[i_47 * 2 + 1]));
                                    }
                                }
                                smem_v21[row_5_1 % 32 * 4 + row_5_1 / 32] = scale_word_5;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col_tile_5 = col_block_1 * 4 + 1;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_q_store), col_tile_5 * 128, row_6, smem_v19_addr);
                                tma_store_3d((&x_sc_store), 0, 0, row_tile_1 * k_tiles_1 + col_tile_5, smem_v21_addr);
                                tma_store_2d((&x_t_store), row_6, col_tile_5 * 128, smem_v23_addr);
                                tma_store_3d((&x_sc_t_store), 0, 0, col_tile_5 * macro_tiles_1 + row_tile_1, smem_v25_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        int next_task_1 = task_1 + comm_sms;
                        if (next_task_1 < dispatch_tasks && next_task_1 >= combine_tasks) {
                            int peer_5 = -1;
                            int peer_token_5 = -1;
                            if (tid < 128) {
                                peer_5 = schedule_rank[macro_offset_3 + next_task_1 / col_blocks_4 * 128 + tid];
                                peer_token_5 = schedule_token[macro_offset_3 + next_task_1 / col_blocks_4 * 128 + tid];
                            }
                            uint32_t cta_count_5 = __syncthreads_count(peer_5 >= 0);
                            if (tid == 0) {
                                int _min_142 = ((256) < (hidden - next_task_1 % col_blocks_4 * 512) ? (256) : (hidden - next_task_1 % col_blocks_4 * 512));
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, cta_count_5 * (unsigned int)(_min_142 * 2));
                            }
                            __syncthreads();
                            if (peer_5 >= 0) {
                                int _min_143 = ((256) < (hidden - next_task_1 % col_blocks_4 * 512) ? (256) : (hidden - next_task_1 % col_blocks_4 * 512));
                                cp_async_bulk_gmem2smem(smem_v26_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_5])) + (unsigned long long)((unsigned long long)(peer_token_5 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task_1 % col_blocks_4 * 512)) * (unsigned long long)2), _min_143 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_5 = 0; vec_5 < 32; vec_5++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_5) + (tid * 512 + vec_5 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
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
                                    int group_6 = tid % 2;
                                    #pragma unroll 1
                                    for (int j_12 = 0; j_12 < 2; j_12++) {
                                        int k_block_12 = (half_6 * 2 + j_12 + tid / 2) % 4;
                                        unsigned int pairs0_6[16];
                                        unsigned int pairs1_6[16];
                                        #pragma unroll
                                        for (int k_6 = 0; k_6 < 16; k_6++) {
                                            int col_6 = k_block_12 * 32 + (k_6 * 2 + group_6 * 16) % 32;
                                            unsigned int w0_6 = flat_view_5[16384 + col_6 * word_stride_6 + r_6];
                                            unsigned int w1_6 = flat_view_5[16384 + (col_6 + 1) * word_stride_6 + r_6];
                                            uint32_t _prmt_144;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_144) : "r"(w0_6), "r"(w1_6));
                                            pairs0_6[k_6] = _prmt_144;
                                            uint32_t _prmt_145;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_145) : "r"(w0_6), "r"(w1_6));
                                            pairs1_6[k_6] = _prmt_145;
                                        }
                                        uint32_t bf16x2_abs_13;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_13) : "r"(pairs0_6[0]));
                                        unsigned int amax_pair_12 = bf16x2_abs_13;
                                        #pragma unroll
                                        for (int i_48 = 1; i_48 < 16; i_48++) {
                                            uint32_t bf16x2_abs_1_18;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_18) : "r"(pairs0_6[i_48]));
                                            uint32_t bf16x2_max_18;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_18) : "r"(amax_pair_12), "r"(bf16x2_abs_1_18));
                                            amax_pair_12 = bf16x2_max_18;
                                        }
                                        uint16_t bf16_max_12;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_12) : "h"((uint16_t)(amax_pair_12 & 65535)), "h"((uint16_t)(amax_pair_12 >> 16)));
                                        float amax_12;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_12) : "h"((uint16_t)(bf16_max_12)));
                                        float scale_12 = fmaxf(amax_12 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_146;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_146) : "f"(scale_12), "f"(scale_12));
                                        unsigned int scale_byte_12 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_146 & 255;
                                        unsigned int inverse_lane_12 = 254 - scale_byte_12 << 7;
                                        unsigned int inverse_12 = inverse_lane_12 | inverse_lane_12 << 16;
                                        unsigned int words_12[8];
                                        #pragma unroll
                                        for (int i_49 = 0; i_49 < 8; i_49++) {
                                            uint32_t bf16x2_mul_19;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(pairs0_6[i_49 * 2]), "r"(inverse_12));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(pairs0_6[i_49 * 2 + 1]), "r"(inverse_12));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_147;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "r"(bf16x2_mul_1_18));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_12[i_49] = (unsigned int)e4m3x2_bf16x2_18 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_147 << 16;
                                        }
                                        uint32_t bf16x2_abs_6_1;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_6_1) : "r"(pairs1_6[0]));
                                        unsigned int amax_pair_6_1 = bf16x2_abs_6_1;
                                        #pragma unroll
                                        for (int i_50 = 1; i_50 < 16; i_50++) {
                                            uint32_t bf16x2_abs_1_19;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_19) : "r"(pairs1_6[i_50]));
                                            uint32_t bf16x2_max_19;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_19) : "r"(amax_pair_6_1), "r"(bf16x2_abs_1_19));
                                            amax_pair_6_1 = bf16x2_max_19;
                                        }
                                        uint16_t bf16_max_6_1;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_6_1) : "h"((uint16_t)(amax_pair_6_1 & 65535)), "h"((uint16_t)(amax_pair_6_1 >> 16)));
                                        float amax_6_1;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_6_1) : "h"((uint16_t)(bf16_max_6_1)));
                                        float scale_6_1 = fmaxf(amax_6_1 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_148;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_148) : "f"(scale_6_1), "f"(scale_6_1));
                                        unsigned int scale_byte_6_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_148 & 255;
                                        unsigned int inverse_lane_6_1 = 254 - scale_byte_6_1 << 7;
                                        unsigned int inverse_6_1 = inverse_lane_6_1 | inverse_lane_6_1 << 16;
                                        unsigned int words_6_1[8];
                                        #pragma unroll
                                        for (int i_51 = 0; i_51 < 8; i_51++) {
                                            uint32_t bf16x2_mul_20;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(pairs1_6[i_51 * 2]), "r"(inverse_6_1));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(pairs1_6[i_51 * 2 + 1]), "r"(inverse_6_1));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_149;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "r"(bf16x2_mul_1_19));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_149) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_6_1[i_51] = (unsigned int)e4m3x2_bf16x2_19 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_149 << 16;
                                        }
                                        int base0_6 = row0_6 * 32 + k_block_12 * 8;
                                        int base1_6 = row1_6 * 32 + k_block_12 * 8;
                                        const void* _smem_vector_store_ptr_150 = reinterpret_cast<const void*>(smem_v22 + (base0_6 + group_6 * 4));
                                        uint32_t _smem_vector_store_addr_150 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_150));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_150), "r"(words_12[0]), "r"(words_12[1]), "r"(words_12[2]), "r"(words_12[3]));
                                        const void* _smem_vector_store_ptr_151 = reinterpret_cast<const void*>(smem_v22 + (base0_6 + (1 - group_6) * 4));
                                        uint32_t _smem_vector_store_addr_151 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_151));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_151), "r"(words_12[4]), "r"(words_12[5]), "r"(words_12[6]), "r"(words_12[7]));
                                        const void* _smem_vector_store_ptr_152 = reinterpret_cast<const void*>(smem_v22 + (base1_6 + group_6 * 4));
                                        uint32_t _smem_vector_store_addr_152 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_152));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_152), "r"(words_6_1[0]), "r"(words_6_1[1]), "r"(words_6_1[2]), "r"(words_6_1[3]));
                                        const void* _smem_vector_store_ptr_153 = reinterpret_cast<const void*>(smem_v22 + (base1_6 + (1 - group_6) * 4));
                                        uint32_t _smem_vector_store_addr_153 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_153));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_153), "r"(words_6_1[4]), "r"(words_6_1[5]), "r"(words_6_1[6]), "r"(words_6_1[7]));
                                        {
                                            uint32_t _byte_154 = static_cast<uint32_t>(scale_byte_12) & 0xFFu;
                                            uint32_t _addr_154 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base0_6 + k_block_12));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_154), "r"(_byte_154) : "memory");
                                        }
                                        {
                                            uint32_t _byte_155 = static_cast<uint32_t>(scale_byte_6_1) & 0xFFu;
                                            uint32_t _addr_155 = static_cast<uint32_t>(smem_v24_addr + (unsigned int)(scale_base1_6 + k_block_12));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_155), "r"(_byte_155) : "memory");
                                        }
                                    }
                                } else {
                                    int row_6_1 = tid - 128;
                                    int rot_6 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                                    unsigned int scale_word_6 = 0;
                                    #pragma unroll 1
                                    for (int j_13 = 0; j_13 < 4; j_13++) {
                                        int k_block_13 = (j_13 + (tid - 128) / 2) % 4;
                                        unsigned int pairs_6[16];
                                        #pragma unroll
                                        for (int i_52 = 0; i_52 < 4; i_52++) {
                                            int chunk_6 = row_6_1 * 128 + k_block_13 * 16 + (i_52 + rot_6) % 4 * 4;
                                            const void* _smem_load_ptr_156 = reinterpret_cast<const void*>(flat_view_5 + (16384 + chunk_6));
                                            uint32_t _smem_load_addr_156 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_156));
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&pairs_6[i_52 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_6[(i_52 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_6[(i_52 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_6[(i_52 * 4) + 3]))
                                                : "r"(_smem_load_addr_156));
                                        }
                                        uint32_t bf16x2_abs_14;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_14) : "r"(pairs_6[0]));
                                        unsigned int amax_pair_13 = bf16x2_abs_14;
                                        #pragma unroll
                                        for (int i_53 = 1; i_53 < 16; i_53++) {
                                            uint32_t bf16x2_abs_1_20;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_20) : "r"(pairs_6[i_53]));
                                            uint32_t bf16x2_max_20;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_20) : "r"(amax_pair_13), "r"(bf16x2_abs_1_20));
                                            amax_pair_13 = bf16x2_max_20;
                                        }
                                        uint16_t bf16_max_13;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_13) : "h"((uint16_t)(amax_pair_13 & 65535)), "h"((uint16_t)(amax_pair_13 >> 16)));
                                        float amax_13;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_13) : "h"((uint16_t)(bf16_max_13)));
                                        float scale_13 = fmaxf(amax_13 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_157;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_157) : "f"(scale_13), "f"(scale_13));
                                        unsigned int scale_byte_13 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_157 & 255;
                                        unsigned int inverse_lane_13 = 254 - scale_byte_13 << 7;
                                        unsigned int inverse_13 = inverse_lane_13 | inverse_lane_13 << 16;
                                        unsigned int words_13[8];
                                        #pragma unroll
                                        for (int i_54 = 0; i_54 < 8; i_54++) {
                                            uint32_t bf16x2_mul_21;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(pairs_6[i_54 * 2]), "r"(inverse_13));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(pairs_6[i_54 * 2 + 1]), "r"(inverse_13));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_158;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_158) : "r"(bf16x2_mul_1_20));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_158) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_13[i_54] = (unsigned int)e4m3x2_bf16x2_20 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_158 << 16;
                                        }
                                        scale_word_6 = scale_word_6 | scale_byte_13 << (unsigned int)(k_block_13 * 8);
                                        #pragma unroll
                                        for (int i_55 = 0; i_55 < 4; i_55++) {
                                            const void* _smem_vector_store_ptr_159 = reinterpret_cast<const void*>(smem_v18 + (row_6_1 * 32 + k_block_13 * 8 + (i_55 + rot_6) % 4 * 2));
                                            uint32_t _smem_vector_store_addr_159 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_159));
                                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_159), "r"(words_13[i_55 * 2]), "r"(words_13[i_55 * 2 + 1]));
                                        }
                                    }
                                    smem_v20[row_6_1 % 32 * 4 + row_6_1 / 32] = scale_word_6;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_6 = col_block_1 * 4 + 2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_6 * 128, row_6, smem_v18_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_1_1 * k_tiles_1_1 + col_tile_6, smem_v20_addr);
                                    tma_store_2d((&x_t_store), row_6, col_tile_6 * 128, smem_v22_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_6 * macro_tiles_1_1 + row_tile_1_1, smem_v24_addr);
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
                                    int group_7 = tid % 2;
                                    #pragma unroll 1
                                    for (int j_14 = 0; j_14 < 2; j_14++) {
                                        int k_block_14 = (half_7 * 2 + j_14 + tid / 2) % 4;
                                        unsigned int pairs0_7[16];
                                        unsigned int pairs1_7[16];
                                        #pragma unroll
                                        for (int k_7 = 0; k_7 < 16; k_7++) {
                                            int col_7 = k_block_14 * 32 + (k_7 * 2 + group_7 * 16) % 32;
                                            unsigned int w0_7 = flat_view_5[16448 + col_7 * word_stride_7 + r_7];
                                            unsigned int w1_7 = flat_view_5[16448 + (col_7 + 1) * word_stride_7 + r_7];
                                            uint32_t _prmt_160;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_160) : "r"(w0_7), "r"(w1_7));
                                            pairs0_7[k_7] = _prmt_160;
                                            uint32_t _prmt_161;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_161) : "r"(w0_7), "r"(w1_7));
                                            pairs1_7[k_7] = _prmt_161;
                                        }
                                        uint32_t bf16x2_abs_15;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_15) : "r"(pairs0_7[0]));
                                        unsigned int amax_pair_14 = bf16x2_abs_15;
                                        #pragma unroll
                                        for (int i_56 = 1; i_56 < 16; i_56++) {
                                            uint32_t bf16x2_abs_1_21;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_21) : "r"(pairs0_7[i_56]));
                                            uint32_t bf16x2_max_21;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_21) : "r"(amax_pair_14), "r"(bf16x2_abs_1_21));
                                            amax_pair_14 = bf16x2_max_21;
                                        }
                                        uint16_t bf16_max_14;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_14) : "h"((uint16_t)(amax_pair_14 & 65535)), "h"((uint16_t)(amax_pair_14 >> 16)));
                                        float amax_14;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_14) : "h"((uint16_t)(bf16_max_14)));
                                        float scale_14 = fmaxf(amax_14 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_162;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_162) : "f"(scale_14), "f"(scale_14));
                                        unsigned int scale_byte_14 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_162 & 255;
                                        unsigned int inverse_lane_14 = 254 - scale_byte_14 << 7;
                                        unsigned int inverse_14 = inverse_lane_14 | inverse_lane_14 << 16;
                                        unsigned int words_14[8];
                                        #pragma unroll
                                        for (int i_57 = 0; i_57 < 8; i_57++) {
                                            uint32_t bf16x2_mul_22;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(pairs0_7[i_57 * 2]), "r"(inverse_14));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(pairs0_7[i_57 * 2 + 1]), "r"(inverse_14));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_163;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "r"(bf16x2_mul_1_21));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_14[i_57] = (unsigned int)e4m3x2_bf16x2_21 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_163 << 16;
                                        }
                                        uint32_t bf16x2_abs_7_1;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_7_1) : "r"(pairs1_7[0]));
                                        unsigned int amax_pair_7_1 = bf16x2_abs_7_1;
                                        #pragma unroll
                                        for (int i_58 = 1; i_58 < 16; i_58++) {
                                            uint32_t bf16x2_abs_1_22;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_22) : "r"(pairs1_7[i_58]));
                                            uint32_t bf16x2_max_22;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_22) : "r"(amax_pair_7_1), "r"(bf16x2_abs_1_22));
                                            amax_pair_7_1 = bf16x2_max_22;
                                        }
                                        uint16_t bf16_max_7_1;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_7_1) : "h"((uint16_t)(amax_pair_7_1 & 65535)), "h"((uint16_t)(amax_pair_7_1 >> 16)));
                                        float amax_7_1;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_7_1) : "h"((uint16_t)(bf16_max_7_1)));
                                        float scale_7_1 = fmaxf(amax_7_1 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_164;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_164) : "f"(scale_7_1), "f"(scale_7_1));
                                        unsigned int scale_byte_7_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_164 & 255;
                                        unsigned int inverse_lane_7_1 = 254 - scale_byte_7_1 << 7;
                                        unsigned int inverse_7_1 = inverse_lane_7_1 | inverse_lane_7_1 << 16;
                                        unsigned int words_7_1[8];
                                        #pragma unroll
                                        for (int i_59 = 0; i_59 < 8; i_59++) {
                                            uint32_t bf16x2_mul_23;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(pairs1_7[i_59 * 2]), "r"(inverse_7_1));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(pairs1_7[i_59 * 2 + 1]), "r"(inverse_7_1));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_165;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "r"(bf16x2_mul_1_22));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_7_1[i_59] = (unsigned int)e4m3x2_bf16x2_22 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_165 << 16;
                                        }
                                        int base0_7 = row0_7 * 32 + k_block_14 * 8;
                                        int base1_7 = row1_7 * 32 + k_block_14 * 8;
                                        const void* _smem_vector_store_ptr_166 = reinterpret_cast<const void*>(smem_v23 + (base0_7 + group_7 * 4));
                                        uint32_t _smem_vector_store_addr_166 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_166));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_166), "r"(words_14[0]), "r"(words_14[1]), "r"(words_14[2]), "r"(words_14[3]));
                                        const void* _smem_vector_store_ptr_167 = reinterpret_cast<const void*>(smem_v23 + (base0_7 + (1 - group_7) * 4));
                                        uint32_t _smem_vector_store_addr_167 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_167));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_167), "r"(words_14[4]), "r"(words_14[5]), "r"(words_14[6]), "r"(words_14[7]));
                                        const void* _smem_vector_store_ptr_168 = reinterpret_cast<const void*>(smem_v23 + (base1_7 + group_7 * 4));
                                        uint32_t _smem_vector_store_addr_168 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_168));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_168), "r"(words_7_1[0]), "r"(words_7_1[1]), "r"(words_7_1[2]), "r"(words_7_1[3]));
                                        const void* _smem_vector_store_ptr_169 = reinterpret_cast<const void*>(smem_v23 + (base1_7 + (1 - group_7) * 4));
                                        uint32_t _smem_vector_store_addr_169 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_169));
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_169), "r"(words_7_1[4]), "r"(words_7_1[5]), "r"(words_7_1[6]), "r"(words_7_1[7]));
                                        {
                                            uint32_t _byte_170 = static_cast<uint32_t>(scale_byte_14) & 0xFFu;
                                            uint32_t _addr_170 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_7 + k_block_14));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_170), "r"(_byte_170) : "memory");
                                        }
                                        {
                                            uint32_t _byte_171 = static_cast<uint32_t>(scale_byte_7_1) & 0xFFu;
                                            uint32_t _addr_171 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_7 + k_block_14));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_171), "r"(_byte_171) : "memory");
                                        }
                                    }
                                } else {
                                    int row_7 = tid - 128;
                                    int rot_7 = (tid - 128) % 2 + 2 * (((tid - 128) / 4 + (tid - 128) / 8) % 2);
                                    unsigned int scale_word_7 = 0;
                                    #pragma unroll 1
                                    for (int j_15 = 0; j_15 < 4; j_15++) {
                                        int k_block_15 = (j_15 + (tid - 128) / 2) % 4;
                                        unsigned int pairs_7[16];
                                        #pragma unroll
                                        for (int i_60 = 0; i_60 < 4; i_60++) {
                                            int chunk_7 = row_7 * 128 + k_block_15 * 16 + (i_60 + rot_7) % 4 * 4;
                                            const void* _smem_load_ptr_172 = reinterpret_cast<const void*>(flat_view_5 + (16448 + chunk_7));
                                            uint32_t _smem_load_addr_172 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_load_ptr_172));
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&pairs_7[i_60 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_7[(i_60 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_7[(i_60 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pairs_7[(i_60 * 4) + 3]))
                                                : "r"(_smem_load_addr_172));
                                        }
                                        uint32_t bf16x2_abs_16;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_16) : "r"(pairs_7[0]));
                                        unsigned int amax_pair_15 = bf16x2_abs_16;
                                        #pragma unroll
                                        for (int i_61 = 1; i_61 < 16; i_61++) {
                                            uint32_t bf16x2_abs_1_23;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_23) : "r"(pairs_7[i_61]));
                                            uint32_t bf16x2_max_23;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_23) : "r"(amax_pair_15), "r"(bf16x2_abs_1_23));
                                            amax_pair_15 = bf16x2_max_23;
                                        }
                                        uint16_t bf16_max_15;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_15) : "h"((uint16_t)(amax_pair_15 & 65535)), "h"((uint16_t)(amax_pair_15 >> 16)));
                                        float amax_15;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_15) : "h"((uint16_t)(bf16_max_15)));
                                        float scale_15 = fmaxf(amax_15 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_173;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_173) : "f"(scale_15), "f"(scale_15));
                                        unsigned int scale_byte_15 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_173 & 255;
                                        unsigned int inverse_lane_15 = 254 - scale_byte_15 << 7;
                                        unsigned int inverse_15 = inverse_lane_15 | inverse_lane_15 << 16;
                                        unsigned int words_15[8];
                                        #pragma unroll
                                        for (int i_62 = 0; i_62 < 8; i_62++) {
                                            uint32_t bf16x2_mul_24;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(pairs_7[i_62 * 2]), "r"(inverse_15));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(pairs_7[i_62 * 2 + 1]), "r"(inverse_15));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_174;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_174) : "r"(bf16x2_mul_1_23));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_174) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_15[i_62] = (unsigned int)e4m3x2_bf16x2_23 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_174 << 16;
                                        }
                                        scale_word_7 = scale_word_7 | scale_byte_15 << (unsigned int)(k_block_15 * 8);
                                        #pragma unroll
                                        for (int i_63 = 0; i_63 < 4; i_63++) {
                                            const void* _smem_vector_store_ptr_175 = reinterpret_cast<const void*>(smem_v19 + (row_7 * 32 + k_block_15 * 8 + (i_63 + rot_7) % 4 * 2));
                                            uint32_t _smem_vector_store_addr_175 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_175));
                                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_175), "r"(words_15[i_63 * 2]), "r"(words_15[i_63 * 2 + 1]));
                                        }
                                    }
                                    smem_v21[row_7 % 32 * 4 + row_7 / 32] = scale_word_7;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col_tile_7 = col_block_1 * 4 + 2 + 1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_q_store), col_tile_7 * 128, row_6, smem_v19_addr);
                                    tma_store_3d((&x_sc_store), 0, 0, row_tile_1_1 * k_tiles_1_1 + col_tile_7, smem_v21_addr);
                                    tma_store_2d((&x_t_store), row_6, col_tile_7 * 128, smem_v23_addr);
                                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_7 * macro_tiles_1_1 + row_tile_1_1, smem_v25_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_3 + row_6) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                        __syncthreads();
                    }
                }
                macro = macro - 1;
            }
        }
    } else {
        int iteration = 0;
        int task_3 = 0;
        int macro_1 = 0;
        int mini_2 = 0;
        int swiglu = 0;
        if (shared_tasks <= cluster - comm_clusters) {
            uint32_t fast_div_q = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks));
            uint32_t fast_div_r = (uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks)) - fast_div_q * (uint32_t)(mini_tasks_div.divisor);
            int ordered_mini = (int)fast_div_q;
            task_3 = (int)fast_div_r;
            macro_1 = macros - 1;
            mini_2 = ordered_mini;
            if (ordered_mini >= last_minis) {
                uint32_t fast_div_q_1 = (minis_per_macro_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(ordered_mini - last_minis)), minis_per_macro_div.multiplier) >> minis_per_macro_div.shift_right) : (uint32_t)((unsigned int)(ordered_mini - last_minis));
                uint32_t fast_div_r_1 = (uint32_t)((unsigned int)(ordered_mini - last_minis)) - fast_div_q_1 * (uint32_t)(minis_per_macro_div.divisor);
                macro_1 = macros - 2 - (int)fast_div_q_1;
                mini_2 = (int)fast_div_r_1;
            }
            if (task_3 >= mini_gate && task_3 < mini_gate + mini_swiglu) {
                swiglu = 1;
            }
        }
        int tile_word_0 = task_3;
        int tile_word_1 = macro_1;
        int tile_word_2 = mini_2;
        int tile_word_3 = swiglu;
        int tile_word_4 = 0;
        int tile_word_5 = 0;
        int tile_word_6 = 0;
        if (cluster < true_clusters) {
            int word0 = 0;
            int word1 = 0;
            int word2 = 0;
            if (shared_tasks <= cluster - comm_clusters) {
                if (task_3 < mini_gate) {
                    int gate_task = task_3;
                    int word0_2 = 0;
                    int word1_0 = 0;
                    int word2_0 = 0;
                    int first_block = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    uint32_t fast_div_q_2 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)gate_task), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)gate_task);
                    uint32_t fast_div_r_2 = (uint32_t)((unsigned int)gate_task) - fast_div_q_2 * (uint32_t)(gate_col_blocks_div.divisor);
                    int block = first_block + (int)fast_div_q_2;
                    if (block < end_block) {
                        word0_2 = counts[3 * experts + block];
                        word1_0 = counts[experts + word0_2];
                        word2_0 = counts[word0_2];
                    }
                    word0 = word0_2;
                    word1 = word1_0;
                    word2 = word2_0;
                } else if (task_3 >= mini_gate + mini_swiglu) {
                    int word0_3 = 0;
                    int word1_2 = 0;
                    int word2_3 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    uint32_t fast_div_q_3 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(task_3 - mini_gate - mini_swiglu)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(task_3 - mini_gate - mini_swiglu));
                    uint32_t fast_div_r_3 = (uint32_t)((unsigned int)(task_3 - mini_gate - mini_swiglu)) - fast_div_q_3 * (uint32_t)(down_col_blocks_div.divisor);
                    int block_1 = first_block_1 + (int)fast_div_q_3;
                    if (block_1 < end_block_1) {
                        word0_3 = counts[3 * experts + block_1];
                        word1_2 = counts[experts + word0_3];
                        word2_3 = counts[word0_3];
                    }
                    word0 = word0_3;
                    word1 = word1_2;
                    word2 = word2_3;
                }
            }
            tile_word_4 = word0;
            tile_word_5 = word1;
            tile_word_6 = word2;
        }
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
            int compute = cluster - comm_clusters;
            if (compute < shared_fused) {
                int col_blocks_5 = (intermediate + 256 - 1) / 256;
                int x = -1;
                int y = -1;
                int expert = -1;
                int k_start = 0;
                int k_end = 0;
                int first = 0;
                int row_blocks = local_tokens / 256;
                if (compute < row_blocks * col_blocks_5) {
                    int supergroup = compute / (row_blocks * 8);
                    int full_cols = col_blocks_5 / 8 * 8;
                    int row_8 = 0;
                    int col_8 = 0;
                    if (compute < row_blocks * full_cols) {
                        row_8 = compute % (row_blocks * 8) / 8;
                        col_8 = supergroup * 8 + compute % 8;
                    } else {
                        row_8 = (compute - row_blocks * full_cols) / (col_blocks_5 - full_cols);
                        col_8 = full_cols + (compute - row_blocks * full_cols) % (col_blocks_5 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_8 = row_blocks - row_8 - 1;
                    }
                    x = row_8;
                    y = col_8;
                    expert = 0;
                }
                int has_hi = 0;
                has_hi = (int)((y * 2 + 1) * 256 < intermediate);
                int global_mini = 0;
                int macro_rows_1 = 0;
                int iterations = hidden / 64;
                if (expert < 0) {
                    if (tid == 0) {
                    }
                } else {
                    if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_176 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int mini_rows_3 = ((0) > (_min_176) ? (0) : (_min_176));
                                    int required_3 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                }
                                #pragma unroll 1
                                for (int idx = 0; idx < iterations; idx++) {
                                    int _stage_gemm_finished = (int)((unsigned int)(gemm_wide + idx) % 4);
                                    int _parity_gemm_finished = (int)((unsigned int)(gemm_wide + idx) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished) * 8, _parity_gemm_finished);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&x_shared)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(idx), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_shared)), "r"(0), "r"(y * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wu_shared)), "r"(0), "r"(y * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
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
                                    int _smem_desc_lo_177 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_178 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_177), "r"(_smem_desc_lo_178), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_179 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_180 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_179), "r"(_smem_desc_lo_180), "r"((tmem_accumulator + (256))), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_1) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else {
                        if (tid < 128) {
                            int _parity_output_arrived = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived);
                            unsigned int gate_packed_0[32];
                            unsigned int up_packed_0[32];
                            unsigned int hidden_packed_0[32];
                            unsigned int address_3_0 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16);
                            float _tmem_load_0[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                : "r"(address_3_0));
                            float _tmem_load_32_0[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_0[15]))
                                : "r"(address_3_0 + 256));
                            gate_packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1])));
                            up_packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[0], _tmem_load_32_0[1])));
                            gate_packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3])));
                            up_packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[2], _tmem_load_32_0[3])));
                            gate_packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[4], _tmem_load_0[5])));
                            up_packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[4], _tmem_load_32_0[5])));
                            gate_packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[6], _tmem_load_0[7])));
                            up_packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[6], _tmem_load_32_0[7])));
                            gate_packed_0[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[8], _tmem_load_0[9])));
                            up_packed_0[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[8], _tmem_load_32_0[9])));
                            gate_packed_0[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[10], _tmem_load_0[11])));
                            up_packed_0[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[10], _tmem_load_32_0[11])));
                            gate_packed_0[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[12], _tmem_load_0[13])));
                            up_packed_0[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[12], _tmem_load_32_0[13])));
                            gate_packed_0[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[14], _tmem_load_0[15])));
                            up_packed_0[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_0[14], _tmem_load_32_0[15])));
                            unsigned int address_3_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16);
                            float _tmem_load_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                : "r"(address_3_1));
                            float _tmem_load_32_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_1[15]))
                                : "r"(address_3_1 + 256));
                            gate_packed_0[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1])));
                            up_packed_0[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[0], _tmem_load_32_1[1])));
                            gate_packed_0[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3])));
                            up_packed_0[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[2], _tmem_load_32_1[3])));
                            gate_packed_0[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[4], _tmem_load_1[5])));
                            up_packed_0[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[4], _tmem_load_32_1[5])));
                            gate_packed_0[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[6], _tmem_load_1[7])));
                            up_packed_0[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[6], _tmem_load_32_1[7])));
                            gate_packed_0[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[8], _tmem_load_1[9])));
                            up_packed_0[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[8], _tmem_load_32_1[9])));
                            gate_packed_0[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[10], _tmem_load_1[11])));
                            up_packed_0[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[10], _tmem_load_32_1[11])));
                            gate_packed_0[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[12], _tmem_load_1[13])));
                            up_packed_0[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[12], _tmem_load_32_1[13])));
                            gate_packed_0[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[14], _tmem_load_1[15])));
                            up_packed_0[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_1[14], _tmem_load_32_1[15])));
                            unsigned int address_3_2 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 32;
                            float _tmem_load_2[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                : "r"(address_3_2));
                            float _tmem_load_32_2[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_2[15]))
                                : "r"(address_3_2 + 256));
                            gate_packed_0[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[0], _tmem_load_2[1])));
                            up_packed_0[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[0], _tmem_load_32_2[1])));
                            gate_packed_0[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[2], _tmem_load_2[3])));
                            up_packed_0[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[2], _tmem_load_32_2[3])));
                            gate_packed_0[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[4], _tmem_load_2[5])));
                            up_packed_0[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[4], _tmem_load_32_2[5])));
                            gate_packed_0[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[6], _tmem_load_2[7])));
                            up_packed_0[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[6], _tmem_load_32_2[7])));
                            gate_packed_0[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[8], _tmem_load_2[9])));
                            up_packed_0[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[8], _tmem_load_32_2[9])));
                            gate_packed_0[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[10], _tmem_load_2[11])));
                            up_packed_0[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[10], _tmem_load_32_2[11])));
                            gate_packed_0[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[12], _tmem_load_2[13])));
                            up_packed_0[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[12], _tmem_load_32_2[13])));
                            gate_packed_0[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[14], _tmem_load_2[15])));
                            up_packed_0[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_2[14], _tmem_load_32_2[15])));
                            unsigned int address_3_3 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 32;
                            float _tmem_load_3[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                : "r"(address_3_3));
                            float _tmem_load_32_3[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_3[15]))
                                : "r"(address_3_3 + 256));
                            gate_packed_0[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[0], _tmem_load_3[1])));
                            up_packed_0[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[0], _tmem_load_32_3[1])));
                            gate_packed_0[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[2], _tmem_load_3[3])));
                            up_packed_0[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[2], _tmem_load_32_3[3])));
                            gate_packed_0[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[4], _tmem_load_3[5])));
                            up_packed_0[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[4], _tmem_load_32_3[5])));
                            gate_packed_0[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[6], _tmem_load_3[7])));
                            up_packed_0[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[6], _tmem_load_32_3[7])));
                            gate_packed_0[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[8], _tmem_load_3[9])));
                            up_packed_0[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[8], _tmem_load_32_3[9])));
                            gate_packed_0[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[10], _tmem_load_3[11])));
                            up_packed_0[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[10], _tmem_load_32_3[11])));
                            gate_packed_0[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[12], _tmem_load_3[13])));
                            up_packed_0[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[12], _tmem_load_32_3[13])));
                            gate_packed_0[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[14], _tmem_load_3[15])));
                            up_packed_0[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_3[14], _tmem_load_32_3[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[0]));
                            float2 _cvt_f32_256_0 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[0]));
                            float gate_x_1_0 = _cvt_f32_0.x;
                            float gate_y_1_0 = _cvt_f32_0.y;
                            float up_x_1_0 = _cvt_f32_256_0.x;
                            float up_y_1_0 = _cvt_f32_256_0.y;
                            gate_x_1_0 = fminf(gate_x_1_0, swiglu_limit);
                            gate_y_1_0 = fminf(gate_y_1_0, swiglu_limit);
                            up_x_1_0 = fminf(fmaxf(up_x_1_0, -swiglu_limit), swiglu_limit);
                            up_y_1_0 = fminf(fmaxf(up_y_1_0, -swiglu_limit), swiglu_limit);
                            float _exp_181 = expf(gate_x_1_0 * -1.0f);
                            float denominator_x_1_0 = _exp_181 + 1.0f;
                            float _exp_182 = expf(gate_y_1_0 * -1.0f);
                            float denominator_y_1_0 = _exp_182 + 1.0f;
                            float hidden_x_1_0 = gate_x_1_0 / denominator_x_1_0 * up_x_1_0;
                            float hidden_y_1_0 = gate_y_1_0 / denominator_y_1_0 * up_y_1_0;
                            hidden_packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_0, hidden_y_1_0)));
                            float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[1]));
                            float2 _cvt_f32_256_1 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[1]));
                            float gate_x_1_1 = _cvt_f32_1.x;
                            float gate_y_1_1 = _cvt_f32_1.y;
                            float up_x_1_1 = _cvt_f32_256_1.x;
                            float up_y_1_1 = _cvt_f32_256_1.y;
                            gate_x_1_1 = fminf(gate_x_1_1, swiglu_limit);
                            gate_y_1_1 = fminf(gate_y_1_1, swiglu_limit);
                            up_x_1_1 = fminf(fmaxf(up_x_1_1, -swiglu_limit), swiglu_limit);
                            up_y_1_1 = fminf(fmaxf(up_y_1_1, -swiglu_limit), swiglu_limit);
                            float _exp_183 = expf(gate_x_1_1 * -1.0f);
                            float denominator_x_1_1 = _exp_183 + 1.0f;
                            float _exp_184 = expf(gate_y_1_1 * -1.0f);
                            float denominator_y_1_1 = _exp_184 + 1.0f;
                            float hidden_x_1_1 = gate_x_1_1 / denominator_x_1_1 * up_x_1_1;
                            float hidden_y_1_1 = gate_y_1_1 / denominator_y_1_1 * up_y_1_1;
                            hidden_packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_1, hidden_y_1_1)));
                            float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[2]));
                            float2 _cvt_f32_256_2 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[2]));
                            float gate_x_1_2 = _cvt_f32_2.x;
                            float gate_y_1_2 = _cvt_f32_2.y;
                            float up_x_1_2 = _cvt_f32_256_2.x;
                            float up_y_1_2 = _cvt_f32_256_2.y;
                            gate_x_1_2 = fminf(gate_x_1_2, swiglu_limit);
                            gate_y_1_2 = fminf(gate_y_1_2, swiglu_limit);
                            up_x_1_2 = fminf(fmaxf(up_x_1_2, -swiglu_limit), swiglu_limit);
                            up_y_1_2 = fminf(fmaxf(up_y_1_2, -swiglu_limit), swiglu_limit);
                            float _exp_185 = expf(gate_x_1_2 * -1.0f);
                            float denominator_x_1_2 = _exp_185 + 1.0f;
                            float _exp_186 = expf(gate_y_1_2 * -1.0f);
                            float denominator_y_1_2 = _exp_186 + 1.0f;
                            float hidden_x_1_2 = gate_x_1_2 / denominator_x_1_2 * up_x_1_2;
                            float hidden_y_1_2 = gate_y_1_2 / denominator_y_1_2 * up_y_1_2;
                            hidden_packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_2, hidden_y_1_2)));
                            float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[3]));
                            float2 _cvt_f32_256_3 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[3]));
                            float gate_x_1_3 = _cvt_f32_3.x;
                            float gate_y_1_3 = _cvt_f32_3.y;
                            float up_x_1_3 = _cvt_f32_256_3.x;
                            float up_y_1_3 = _cvt_f32_256_3.y;
                            gate_x_1_3 = fminf(gate_x_1_3, swiglu_limit);
                            gate_y_1_3 = fminf(gate_y_1_3, swiglu_limit);
                            up_x_1_3 = fminf(fmaxf(up_x_1_3, -swiglu_limit), swiglu_limit);
                            up_y_1_3 = fminf(fmaxf(up_y_1_3, -swiglu_limit), swiglu_limit);
                            float _exp_187 = expf(gate_x_1_3 * -1.0f);
                            float denominator_x_1_3 = _exp_187 + 1.0f;
                            float _exp_188 = expf(gate_y_1_3 * -1.0f);
                            float denominator_y_1_3 = _exp_188 + 1.0f;
                            float hidden_x_1_3 = gate_x_1_3 / denominator_x_1_3 * up_x_1_3;
                            float hidden_y_1_3 = gate_y_1_3 / denominator_y_1_3 * up_y_1_3;
                            hidden_packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_3, hidden_y_1_3)));
                            float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[4]));
                            float2 _cvt_f32_256_4 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[4]));
                            float gate_x_1_4 = _cvt_f32_4.x;
                            float gate_y_1_4 = _cvt_f32_4.y;
                            float up_x_1_4 = _cvt_f32_256_4.x;
                            float up_y_1_4 = _cvt_f32_256_4.y;
                            gate_x_1_4 = fminf(gate_x_1_4, swiglu_limit);
                            gate_y_1_4 = fminf(gate_y_1_4, swiglu_limit);
                            up_x_1_4 = fminf(fmaxf(up_x_1_4, -swiglu_limit), swiglu_limit);
                            up_y_1_4 = fminf(fmaxf(up_y_1_4, -swiglu_limit), swiglu_limit);
                            float _exp_189 = expf(gate_x_1_4 * -1.0f);
                            float denominator_x_1_4 = _exp_189 + 1.0f;
                            float _exp_190 = expf(gate_y_1_4 * -1.0f);
                            float denominator_y_1_4 = _exp_190 + 1.0f;
                            float hidden_x_1_4 = gate_x_1_4 / denominator_x_1_4 * up_x_1_4;
                            float hidden_y_1_4 = gate_y_1_4 / denominator_y_1_4 * up_y_1_4;
                            hidden_packed_0[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_4, hidden_y_1_4)));
                            float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[5]));
                            float2 _cvt_f32_256_5 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[5]));
                            float gate_x_1_5 = _cvt_f32_5.x;
                            float gate_y_1_5 = _cvt_f32_5.y;
                            float up_x_1_5 = _cvt_f32_256_5.x;
                            float up_y_1_5 = _cvt_f32_256_5.y;
                            gate_x_1_5 = fminf(gate_x_1_5, swiglu_limit);
                            gate_y_1_5 = fminf(gate_y_1_5, swiglu_limit);
                            up_x_1_5 = fminf(fmaxf(up_x_1_5, -swiglu_limit), swiglu_limit);
                            up_y_1_5 = fminf(fmaxf(up_y_1_5, -swiglu_limit), swiglu_limit);
                            float _exp_191 = expf(gate_x_1_5 * -1.0f);
                            float denominator_x_1_5 = _exp_191 + 1.0f;
                            float _exp_192 = expf(gate_y_1_5 * -1.0f);
                            float denominator_y_1_5 = _exp_192 + 1.0f;
                            float hidden_x_1_5 = gate_x_1_5 / denominator_x_1_5 * up_x_1_5;
                            float hidden_y_1_5 = gate_y_1_5 / denominator_y_1_5 * up_y_1_5;
                            hidden_packed_0[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_5, hidden_y_1_5)));
                            float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[6]));
                            float2 _cvt_f32_256_6 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[6]));
                            float gate_x_1_6 = _cvt_f32_6.x;
                            float gate_y_1_6 = _cvt_f32_6.y;
                            float up_x_1_6 = _cvt_f32_256_6.x;
                            float up_y_1_6 = _cvt_f32_256_6.y;
                            gate_x_1_6 = fminf(gate_x_1_6, swiglu_limit);
                            gate_y_1_6 = fminf(gate_y_1_6, swiglu_limit);
                            up_x_1_6 = fminf(fmaxf(up_x_1_6, -swiglu_limit), swiglu_limit);
                            up_y_1_6 = fminf(fmaxf(up_y_1_6, -swiglu_limit), swiglu_limit);
                            float _exp_193 = expf(gate_x_1_6 * -1.0f);
                            float denominator_x_1_6 = _exp_193 + 1.0f;
                            float _exp_194 = expf(gate_y_1_6 * -1.0f);
                            float denominator_y_1_6 = _exp_194 + 1.0f;
                            float hidden_x_1_6 = gate_x_1_6 / denominator_x_1_6 * up_x_1_6;
                            float hidden_y_1_6 = gate_y_1_6 / denominator_y_1_6 * up_y_1_6;
                            hidden_packed_0[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_6, hidden_y_1_6)));
                            float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[7]));
                            float2 _cvt_f32_256_7 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[7]));
                            float gate_x_1_7 = _cvt_f32_7.x;
                            float gate_y_1_7 = _cvt_f32_7.y;
                            float up_x_1_7 = _cvt_f32_256_7.x;
                            float up_y_1_7 = _cvt_f32_256_7.y;
                            gate_x_1_7 = fminf(gate_x_1_7, swiglu_limit);
                            gate_y_1_7 = fminf(gate_y_1_7, swiglu_limit);
                            up_x_1_7 = fminf(fmaxf(up_x_1_7, -swiglu_limit), swiglu_limit);
                            up_y_1_7 = fminf(fmaxf(up_y_1_7, -swiglu_limit), swiglu_limit);
                            float _exp_195 = expf(gate_x_1_7 * -1.0f);
                            float denominator_x_1_7 = _exp_195 + 1.0f;
                            float _exp_196 = expf(gate_y_1_7 * -1.0f);
                            float denominator_y_1_7 = _exp_196 + 1.0f;
                            float hidden_x_1_7 = gate_x_1_7 / denominator_x_1_7 * up_x_1_7;
                            float hidden_y_1_7 = gate_y_1_7 / denominator_y_1_7 * up_y_1_7;
                            hidden_packed_0[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_7, hidden_y_1_7)));
                            float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[8]));
                            float2 _cvt_f32_256_8 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[8]));
                            float gate_x_1_8 = _cvt_f32_8.x;
                            float gate_y_1_8 = _cvt_f32_8.y;
                            float up_x_1_8 = _cvt_f32_256_8.x;
                            float up_y_1_8 = _cvt_f32_256_8.y;
                            gate_x_1_8 = fminf(gate_x_1_8, swiglu_limit);
                            gate_y_1_8 = fminf(gate_y_1_8, swiglu_limit);
                            up_x_1_8 = fminf(fmaxf(up_x_1_8, -swiglu_limit), swiglu_limit);
                            up_y_1_8 = fminf(fmaxf(up_y_1_8, -swiglu_limit), swiglu_limit);
                            float _exp_197 = expf(gate_x_1_8 * -1.0f);
                            float denominator_x_1_8 = _exp_197 + 1.0f;
                            float _exp_198 = expf(gate_y_1_8 * -1.0f);
                            float denominator_y_1_8 = _exp_198 + 1.0f;
                            float hidden_x_1_8 = gate_x_1_8 / denominator_x_1_8 * up_x_1_8;
                            float hidden_y_1_8 = gate_y_1_8 / denominator_y_1_8 * up_y_1_8;
                            hidden_packed_0[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_8, hidden_y_1_8)));
                            float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[9]));
                            float2 _cvt_f32_256_9 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[9]));
                            float gate_x_1_9 = _cvt_f32_9.x;
                            float gate_y_1_9 = _cvt_f32_9.y;
                            float up_x_1_9 = _cvt_f32_256_9.x;
                            float up_y_1_9 = _cvt_f32_256_9.y;
                            gate_x_1_9 = fminf(gate_x_1_9, swiglu_limit);
                            gate_y_1_9 = fminf(gate_y_1_9, swiglu_limit);
                            up_x_1_9 = fminf(fmaxf(up_x_1_9, -swiglu_limit), swiglu_limit);
                            up_y_1_9 = fminf(fmaxf(up_y_1_9, -swiglu_limit), swiglu_limit);
                            float _exp_199 = expf(gate_x_1_9 * -1.0f);
                            float denominator_x_1_9 = _exp_199 + 1.0f;
                            float _exp_200 = expf(gate_y_1_9 * -1.0f);
                            float denominator_y_1_9 = _exp_200 + 1.0f;
                            float hidden_x_1_9 = gate_x_1_9 / denominator_x_1_9 * up_x_1_9;
                            float hidden_y_1_9 = gate_y_1_9 / denominator_y_1_9 * up_y_1_9;
                            hidden_packed_0[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_9, hidden_y_1_9)));
                            float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[10]));
                            float2 _cvt_f32_256_10 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[10]));
                            float gate_x_1_10 = _cvt_f32_10.x;
                            float gate_y_1_10 = _cvt_f32_10.y;
                            float up_x_1_10 = _cvt_f32_256_10.x;
                            float up_y_1_10 = _cvt_f32_256_10.y;
                            gate_x_1_10 = fminf(gate_x_1_10, swiglu_limit);
                            gate_y_1_10 = fminf(gate_y_1_10, swiglu_limit);
                            up_x_1_10 = fminf(fmaxf(up_x_1_10, -swiglu_limit), swiglu_limit);
                            up_y_1_10 = fminf(fmaxf(up_y_1_10, -swiglu_limit), swiglu_limit);
                            float _exp_201 = expf(gate_x_1_10 * -1.0f);
                            float denominator_x_1_10 = _exp_201 + 1.0f;
                            float _exp_202 = expf(gate_y_1_10 * -1.0f);
                            float denominator_y_1_10 = _exp_202 + 1.0f;
                            float hidden_x_1_10 = gate_x_1_10 / denominator_x_1_10 * up_x_1_10;
                            float hidden_y_1_10 = gate_y_1_10 / denominator_y_1_10 * up_y_1_10;
                            hidden_packed_0[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_10, hidden_y_1_10)));
                            float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[11]));
                            float2 _cvt_f32_256_11 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[11]));
                            float gate_x_1_11 = _cvt_f32_11.x;
                            float gate_y_1_11 = _cvt_f32_11.y;
                            float up_x_1_11 = _cvt_f32_256_11.x;
                            float up_y_1_11 = _cvt_f32_256_11.y;
                            gate_x_1_11 = fminf(gate_x_1_11, swiglu_limit);
                            gate_y_1_11 = fminf(gate_y_1_11, swiglu_limit);
                            up_x_1_11 = fminf(fmaxf(up_x_1_11, -swiglu_limit), swiglu_limit);
                            up_y_1_11 = fminf(fmaxf(up_y_1_11, -swiglu_limit), swiglu_limit);
                            float _exp_203 = expf(gate_x_1_11 * -1.0f);
                            float denominator_x_1_11 = _exp_203 + 1.0f;
                            float _exp_204 = expf(gate_y_1_11 * -1.0f);
                            float denominator_y_1_11 = _exp_204 + 1.0f;
                            float hidden_x_1_11 = gate_x_1_11 / denominator_x_1_11 * up_x_1_11;
                            float hidden_y_1_11 = gate_y_1_11 / denominator_y_1_11 * up_y_1_11;
                            hidden_packed_0[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_11, hidden_y_1_11)));
                            float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[12]));
                            float2 _cvt_f32_256_12 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[12]));
                            float gate_x_1_12 = _cvt_f32_12.x;
                            float gate_y_1_12 = _cvt_f32_12.y;
                            float up_x_1_12 = _cvt_f32_256_12.x;
                            float up_y_1_12 = _cvt_f32_256_12.y;
                            gate_x_1_12 = fminf(gate_x_1_12, swiglu_limit);
                            gate_y_1_12 = fminf(gate_y_1_12, swiglu_limit);
                            up_x_1_12 = fminf(fmaxf(up_x_1_12, -swiglu_limit), swiglu_limit);
                            up_y_1_12 = fminf(fmaxf(up_y_1_12, -swiglu_limit), swiglu_limit);
                            float _exp_205 = expf(gate_x_1_12 * -1.0f);
                            float denominator_x_1_12 = _exp_205 + 1.0f;
                            float _exp_206 = expf(gate_y_1_12 * -1.0f);
                            float denominator_y_1_12 = _exp_206 + 1.0f;
                            float hidden_x_1_12 = gate_x_1_12 / denominator_x_1_12 * up_x_1_12;
                            float hidden_y_1_12 = gate_y_1_12 / denominator_y_1_12 * up_y_1_12;
                            hidden_packed_0[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_12, hidden_y_1_12)));
                            float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[13]));
                            float2 _cvt_f32_256_13 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[13]));
                            float gate_x_1_13 = _cvt_f32_13.x;
                            float gate_y_1_13 = _cvt_f32_13.y;
                            float up_x_1_13 = _cvt_f32_256_13.x;
                            float up_y_1_13 = _cvt_f32_256_13.y;
                            gate_x_1_13 = fminf(gate_x_1_13, swiglu_limit);
                            gate_y_1_13 = fminf(gate_y_1_13, swiglu_limit);
                            up_x_1_13 = fminf(fmaxf(up_x_1_13, -swiglu_limit), swiglu_limit);
                            up_y_1_13 = fminf(fmaxf(up_y_1_13, -swiglu_limit), swiglu_limit);
                            float _exp_207 = expf(gate_x_1_13 * -1.0f);
                            float denominator_x_1_13 = _exp_207 + 1.0f;
                            float _exp_208 = expf(gate_y_1_13 * -1.0f);
                            float denominator_y_1_13 = _exp_208 + 1.0f;
                            float hidden_x_1_13 = gate_x_1_13 / denominator_x_1_13 * up_x_1_13;
                            float hidden_y_1_13 = gate_y_1_13 / denominator_y_1_13 * up_y_1_13;
                            hidden_packed_0[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_13, hidden_y_1_13)));
                            float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[14]));
                            float2 _cvt_f32_256_14 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[14]));
                            float gate_x_1_14 = _cvt_f32_14.x;
                            float gate_y_1_14 = _cvt_f32_14.y;
                            float up_x_1_14 = _cvt_f32_256_14.x;
                            float up_y_1_14 = _cvt_f32_256_14.y;
                            gate_x_1_14 = fminf(gate_x_1_14, swiglu_limit);
                            gate_y_1_14 = fminf(gate_y_1_14, swiglu_limit);
                            up_x_1_14 = fminf(fmaxf(up_x_1_14, -swiglu_limit), swiglu_limit);
                            up_y_1_14 = fminf(fmaxf(up_y_1_14, -swiglu_limit), swiglu_limit);
                            float _exp_209 = expf(gate_x_1_14 * -1.0f);
                            float denominator_x_1_14 = _exp_209 + 1.0f;
                            float _exp_210 = expf(gate_y_1_14 * -1.0f);
                            float denominator_y_1_14 = _exp_210 + 1.0f;
                            float hidden_x_1_14 = gate_x_1_14 / denominator_x_1_14 * up_x_1_14;
                            float hidden_y_1_14 = gate_y_1_14 / denominator_y_1_14 * up_y_1_14;
                            hidden_packed_0[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_14, hidden_y_1_14)));
                            float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[15]));
                            float2 _cvt_f32_256_15 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[15]));
                            float gate_x_1_15 = _cvt_f32_15.x;
                            float gate_y_1_15 = _cvt_f32_15.y;
                            float up_x_1_15 = _cvt_f32_256_15.x;
                            float up_y_1_15 = _cvt_f32_256_15.y;
                            gate_x_1_15 = fminf(gate_x_1_15, swiglu_limit);
                            gate_y_1_15 = fminf(gate_y_1_15, swiglu_limit);
                            up_x_1_15 = fminf(fmaxf(up_x_1_15, -swiglu_limit), swiglu_limit);
                            up_y_1_15 = fminf(fmaxf(up_y_1_15, -swiglu_limit), swiglu_limit);
                            float _exp_211 = expf(gate_x_1_15 * -1.0f);
                            float denominator_x_1_15 = _exp_211 + 1.0f;
                            float _exp_212 = expf(gate_y_1_15 * -1.0f);
                            float denominator_y_1_15 = _exp_212 + 1.0f;
                            float hidden_x_1_15 = gate_x_1_15 / denominator_x_1_15 * up_x_1_15;
                            float hidden_y_1_15 = gate_y_1_15 / denominator_y_1_15 * up_y_1_15;
                            hidden_packed_0[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_15, hidden_y_1_15)));
                            float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[16]));
                            float2 _cvt_f32_256_16 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[16]));
                            float gate_x_1_16 = _cvt_f32_16.x;
                            float gate_y_1_16 = _cvt_f32_16.y;
                            float up_x_1_16 = _cvt_f32_256_16.x;
                            float up_y_1_16 = _cvt_f32_256_16.y;
                            gate_x_1_16 = fminf(gate_x_1_16, swiglu_limit);
                            gate_y_1_16 = fminf(gate_y_1_16, swiglu_limit);
                            up_x_1_16 = fminf(fmaxf(up_x_1_16, -swiglu_limit), swiglu_limit);
                            up_y_1_16 = fminf(fmaxf(up_y_1_16, -swiglu_limit), swiglu_limit);
                            float _exp_213 = expf(gate_x_1_16 * -1.0f);
                            float denominator_x_1_16 = _exp_213 + 1.0f;
                            float _exp_214 = expf(gate_y_1_16 * -1.0f);
                            float denominator_y_1_16 = _exp_214 + 1.0f;
                            float hidden_x_1_16 = gate_x_1_16 / denominator_x_1_16 * up_x_1_16;
                            float hidden_y_1_16 = gate_y_1_16 / denominator_y_1_16 * up_y_1_16;
                            hidden_packed_0[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_16, hidden_y_1_16)));
                            float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[17]));
                            float2 _cvt_f32_256_17 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[17]));
                            float gate_x_1_17 = _cvt_f32_17.x;
                            float gate_y_1_17 = _cvt_f32_17.y;
                            float up_x_1_17 = _cvt_f32_256_17.x;
                            float up_y_1_17 = _cvt_f32_256_17.y;
                            gate_x_1_17 = fminf(gate_x_1_17, swiglu_limit);
                            gate_y_1_17 = fminf(gate_y_1_17, swiglu_limit);
                            up_x_1_17 = fminf(fmaxf(up_x_1_17, -swiglu_limit), swiglu_limit);
                            up_y_1_17 = fminf(fmaxf(up_y_1_17, -swiglu_limit), swiglu_limit);
                            float _exp_215 = expf(gate_x_1_17 * -1.0f);
                            float denominator_x_1_17 = _exp_215 + 1.0f;
                            float _exp_216 = expf(gate_y_1_17 * -1.0f);
                            float denominator_y_1_17 = _exp_216 + 1.0f;
                            float hidden_x_1_17 = gate_x_1_17 / denominator_x_1_17 * up_x_1_17;
                            float hidden_y_1_17 = gate_y_1_17 / denominator_y_1_17 * up_y_1_17;
                            hidden_packed_0[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_17, hidden_y_1_17)));
                            float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[18]));
                            float2 _cvt_f32_256_18 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[18]));
                            float gate_x_1_18 = _cvt_f32_18.x;
                            float gate_y_1_18 = _cvt_f32_18.y;
                            float up_x_1_18 = _cvt_f32_256_18.x;
                            float up_y_1_18 = _cvt_f32_256_18.y;
                            gate_x_1_18 = fminf(gate_x_1_18, swiglu_limit);
                            gate_y_1_18 = fminf(gate_y_1_18, swiglu_limit);
                            up_x_1_18 = fminf(fmaxf(up_x_1_18, -swiglu_limit), swiglu_limit);
                            up_y_1_18 = fminf(fmaxf(up_y_1_18, -swiglu_limit), swiglu_limit);
                            float _exp_217 = expf(gate_x_1_18 * -1.0f);
                            float denominator_x_1_18 = _exp_217 + 1.0f;
                            float _exp_218 = expf(gate_y_1_18 * -1.0f);
                            float denominator_y_1_18 = _exp_218 + 1.0f;
                            float hidden_x_1_18 = gate_x_1_18 / denominator_x_1_18 * up_x_1_18;
                            float hidden_y_1_18 = gate_y_1_18 / denominator_y_1_18 * up_y_1_18;
                            hidden_packed_0[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_18, hidden_y_1_18)));
                            float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[19]));
                            float2 _cvt_f32_256_19 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[19]));
                            float gate_x_1_19 = _cvt_f32_19.x;
                            float gate_y_1_19 = _cvt_f32_19.y;
                            float up_x_1_19 = _cvt_f32_256_19.x;
                            float up_y_1_19 = _cvt_f32_256_19.y;
                            gate_x_1_19 = fminf(gate_x_1_19, swiglu_limit);
                            gate_y_1_19 = fminf(gate_y_1_19, swiglu_limit);
                            up_x_1_19 = fminf(fmaxf(up_x_1_19, -swiglu_limit), swiglu_limit);
                            up_y_1_19 = fminf(fmaxf(up_y_1_19, -swiglu_limit), swiglu_limit);
                            float _exp_219 = expf(gate_x_1_19 * -1.0f);
                            float denominator_x_1_19 = _exp_219 + 1.0f;
                            float _exp_220 = expf(gate_y_1_19 * -1.0f);
                            float denominator_y_1_19 = _exp_220 + 1.0f;
                            float hidden_x_1_19 = gate_x_1_19 / denominator_x_1_19 * up_x_1_19;
                            float hidden_y_1_19 = gate_y_1_19 / denominator_y_1_19 * up_y_1_19;
                            hidden_packed_0[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_19, hidden_y_1_19)));
                            float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[20]));
                            float2 _cvt_f32_256_20 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[20]));
                            float gate_x_1_20 = _cvt_f32_20.x;
                            float gate_y_1_20 = _cvt_f32_20.y;
                            float up_x_1_20 = _cvt_f32_256_20.x;
                            float up_y_1_20 = _cvt_f32_256_20.y;
                            gate_x_1_20 = fminf(gate_x_1_20, swiglu_limit);
                            gate_y_1_20 = fminf(gate_y_1_20, swiglu_limit);
                            up_x_1_20 = fminf(fmaxf(up_x_1_20, -swiglu_limit), swiglu_limit);
                            up_y_1_20 = fminf(fmaxf(up_y_1_20, -swiglu_limit), swiglu_limit);
                            float _exp_221 = expf(gate_x_1_20 * -1.0f);
                            float denominator_x_1_20 = _exp_221 + 1.0f;
                            float _exp_222 = expf(gate_y_1_20 * -1.0f);
                            float denominator_y_1_20 = _exp_222 + 1.0f;
                            float hidden_x_1_20 = gate_x_1_20 / denominator_x_1_20 * up_x_1_20;
                            float hidden_y_1_20 = gate_y_1_20 / denominator_y_1_20 * up_y_1_20;
                            hidden_packed_0[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_20, hidden_y_1_20)));
                            float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[21]));
                            float2 _cvt_f32_256_21 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[21]));
                            float gate_x_1_21 = _cvt_f32_21.x;
                            float gate_y_1_21 = _cvt_f32_21.y;
                            float up_x_1_21 = _cvt_f32_256_21.x;
                            float up_y_1_21 = _cvt_f32_256_21.y;
                            gate_x_1_21 = fminf(gate_x_1_21, swiglu_limit);
                            gate_y_1_21 = fminf(gate_y_1_21, swiglu_limit);
                            up_x_1_21 = fminf(fmaxf(up_x_1_21, -swiglu_limit), swiglu_limit);
                            up_y_1_21 = fminf(fmaxf(up_y_1_21, -swiglu_limit), swiglu_limit);
                            float _exp_223 = expf(gate_x_1_21 * -1.0f);
                            float denominator_x_1_21 = _exp_223 + 1.0f;
                            float _exp_224 = expf(gate_y_1_21 * -1.0f);
                            float denominator_y_1_21 = _exp_224 + 1.0f;
                            float hidden_x_1_21 = gate_x_1_21 / denominator_x_1_21 * up_x_1_21;
                            float hidden_y_1_21 = gate_y_1_21 / denominator_y_1_21 * up_y_1_21;
                            hidden_packed_0[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_21, hidden_y_1_21)));
                            float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[22]));
                            float2 _cvt_f32_256_22 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[22]));
                            float gate_x_1_22 = _cvt_f32_22.x;
                            float gate_y_1_22 = _cvt_f32_22.y;
                            float up_x_1_22 = _cvt_f32_256_22.x;
                            float up_y_1_22 = _cvt_f32_256_22.y;
                            gate_x_1_22 = fminf(gate_x_1_22, swiglu_limit);
                            gate_y_1_22 = fminf(gate_y_1_22, swiglu_limit);
                            up_x_1_22 = fminf(fmaxf(up_x_1_22, -swiglu_limit), swiglu_limit);
                            up_y_1_22 = fminf(fmaxf(up_y_1_22, -swiglu_limit), swiglu_limit);
                            float _exp_225 = expf(gate_x_1_22 * -1.0f);
                            float denominator_x_1_22 = _exp_225 + 1.0f;
                            float _exp_226 = expf(gate_y_1_22 * -1.0f);
                            float denominator_y_1_22 = _exp_226 + 1.0f;
                            float hidden_x_1_22 = gate_x_1_22 / denominator_x_1_22 * up_x_1_22;
                            float hidden_y_1_22 = gate_y_1_22 / denominator_y_1_22 * up_y_1_22;
                            hidden_packed_0[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_22, hidden_y_1_22)));
                            float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[23]));
                            float2 _cvt_f32_256_23 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[23]));
                            float gate_x_1_23 = _cvt_f32_23.x;
                            float gate_y_1_23 = _cvt_f32_23.y;
                            float up_x_1_23 = _cvt_f32_256_23.x;
                            float up_y_1_23 = _cvt_f32_256_23.y;
                            gate_x_1_23 = fminf(gate_x_1_23, swiglu_limit);
                            gate_y_1_23 = fminf(gate_y_1_23, swiglu_limit);
                            up_x_1_23 = fminf(fmaxf(up_x_1_23, -swiglu_limit), swiglu_limit);
                            up_y_1_23 = fminf(fmaxf(up_y_1_23, -swiglu_limit), swiglu_limit);
                            float _exp_227 = expf(gate_x_1_23 * -1.0f);
                            float denominator_x_1_23 = _exp_227 + 1.0f;
                            float _exp_228 = expf(gate_y_1_23 * -1.0f);
                            float denominator_y_1_23 = _exp_228 + 1.0f;
                            float hidden_x_1_23 = gate_x_1_23 / denominator_x_1_23 * up_x_1_23;
                            float hidden_y_1_23 = gate_y_1_23 / denominator_y_1_23 * up_y_1_23;
                            hidden_packed_0[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_23, hidden_y_1_23)));
                            float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[24]));
                            float2 _cvt_f32_256_24 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[24]));
                            float gate_x_1_24 = _cvt_f32_24.x;
                            float gate_y_1_24 = _cvt_f32_24.y;
                            float up_x_1_24 = _cvt_f32_256_24.x;
                            float up_y_1_24 = _cvt_f32_256_24.y;
                            gate_x_1_24 = fminf(gate_x_1_24, swiglu_limit);
                            gate_y_1_24 = fminf(gate_y_1_24, swiglu_limit);
                            up_x_1_24 = fminf(fmaxf(up_x_1_24, -swiglu_limit), swiglu_limit);
                            up_y_1_24 = fminf(fmaxf(up_y_1_24, -swiglu_limit), swiglu_limit);
                            float _exp_229 = expf(gate_x_1_24 * -1.0f);
                            float denominator_x_1_24 = _exp_229 + 1.0f;
                            float _exp_230 = expf(gate_y_1_24 * -1.0f);
                            float denominator_y_1_24 = _exp_230 + 1.0f;
                            float hidden_x_1_24 = gate_x_1_24 / denominator_x_1_24 * up_x_1_24;
                            float hidden_y_1_24 = gate_y_1_24 / denominator_y_1_24 * up_y_1_24;
                            hidden_packed_0[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_24, hidden_y_1_24)));
                            float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[25]));
                            float2 _cvt_f32_256_25 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[25]));
                            float gate_x_1_25 = _cvt_f32_25.x;
                            float gate_y_1_25 = _cvt_f32_25.y;
                            float up_x_1_25 = _cvt_f32_256_25.x;
                            float up_y_1_25 = _cvt_f32_256_25.y;
                            gate_x_1_25 = fminf(gate_x_1_25, swiglu_limit);
                            gate_y_1_25 = fminf(gate_y_1_25, swiglu_limit);
                            up_x_1_25 = fminf(fmaxf(up_x_1_25, -swiglu_limit), swiglu_limit);
                            up_y_1_25 = fminf(fmaxf(up_y_1_25, -swiglu_limit), swiglu_limit);
                            float _exp_231 = expf(gate_x_1_25 * -1.0f);
                            float denominator_x_1_25 = _exp_231 + 1.0f;
                            float _exp_232 = expf(gate_y_1_25 * -1.0f);
                            float denominator_y_1_25 = _exp_232 + 1.0f;
                            float hidden_x_1_25 = gate_x_1_25 / denominator_x_1_25 * up_x_1_25;
                            float hidden_y_1_25 = gate_y_1_25 / denominator_y_1_25 * up_y_1_25;
                            hidden_packed_0[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_25, hidden_y_1_25)));
                            float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[26]));
                            float2 _cvt_f32_256_26 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[26]));
                            float gate_x_1_26 = _cvt_f32_26.x;
                            float gate_y_1_26 = _cvt_f32_26.y;
                            float up_x_1_26 = _cvt_f32_256_26.x;
                            float up_y_1_26 = _cvt_f32_256_26.y;
                            gate_x_1_26 = fminf(gate_x_1_26, swiglu_limit);
                            gate_y_1_26 = fminf(gate_y_1_26, swiglu_limit);
                            up_x_1_26 = fminf(fmaxf(up_x_1_26, -swiglu_limit), swiglu_limit);
                            up_y_1_26 = fminf(fmaxf(up_y_1_26, -swiglu_limit), swiglu_limit);
                            float _exp_233 = expf(gate_x_1_26 * -1.0f);
                            float denominator_x_1_26 = _exp_233 + 1.0f;
                            float _exp_234 = expf(gate_y_1_26 * -1.0f);
                            float denominator_y_1_26 = _exp_234 + 1.0f;
                            float hidden_x_1_26 = gate_x_1_26 / denominator_x_1_26 * up_x_1_26;
                            float hidden_y_1_26 = gate_y_1_26 / denominator_y_1_26 * up_y_1_26;
                            hidden_packed_0[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_26, hidden_y_1_26)));
                            float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[27]));
                            float2 _cvt_f32_256_27 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[27]));
                            float gate_x_1_27 = _cvt_f32_27.x;
                            float gate_y_1_27 = _cvt_f32_27.y;
                            float up_x_1_27 = _cvt_f32_256_27.x;
                            float up_y_1_27 = _cvt_f32_256_27.y;
                            gate_x_1_27 = fminf(gate_x_1_27, swiglu_limit);
                            gate_y_1_27 = fminf(gate_y_1_27, swiglu_limit);
                            up_x_1_27 = fminf(fmaxf(up_x_1_27, -swiglu_limit), swiglu_limit);
                            up_y_1_27 = fminf(fmaxf(up_y_1_27, -swiglu_limit), swiglu_limit);
                            float _exp_235 = expf(gate_x_1_27 * -1.0f);
                            float denominator_x_1_27 = _exp_235 + 1.0f;
                            float _exp_236 = expf(gate_y_1_27 * -1.0f);
                            float denominator_y_1_27 = _exp_236 + 1.0f;
                            float hidden_x_1_27 = gate_x_1_27 / denominator_x_1_27 * up_x_1_27;
                            float hidden_y_1_27 = gate_y_1_27 / denominator_y_1_27 * up_y_1_27;
                            hidden_packed_0[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_27, hidden_y_1_27)));
                            float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[28]));
                            float2 _cvt_f32_256_28 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[28]));
                            float gate_x_1_28 = _cvt_f32_28.x;
                            float gate_y_1_28 = _cvt_f32_28.y;
                            float up_x_1_28 = _cvt_f32_256_28.x;
                            float up_y_1_28 = _cvt_f32_256_28.y;
                            gate_x_1_28 = fminf(gate_x_1_28, swiglu_limit);
                            gate_y_1_28 = fminf(gate_y_1_28, swiglu_limit);
                            up_x_1_28 = fminf(fmaxf(up_x_1_28, -swiglu_limit), swiglu_limit);
                            up_y_1_28 = fminf(fmaxf(up_y_1_28, -swiglu_limit), swiglu_limit);
                            float _exp_237 = expf(gate_x_1_28 * -1.0f);
                            float denominator_x_1_28 = _exp_237 + 1.0f;
                            float _exp_238 = expf(gate_y_1_28 * -1.0f);
                            float denominator_y_1_28 = _exp_238 + 1.0f;
                            float hidden_x_1_28 = gate_x_1_28 / denominator_x_1_28 * up_x_1_28;
                            float hidden_y_1_28 = gate_y_1_28 / denominator_y_1_28 * up_y_1_28;
                            hidden_packed_0[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_28, hidden_y_1_28)));
                            float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[29]));
                            float2 _cvt_f32_256_29 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[29]));
                            float gate_x_1_29 = _cvt_f32_29.x;
                            float gate_y_1_29 = _cvt_f32_29.y;
                            float up_x_1_29 = _cvt_f32_256_29.x;
                            float up_y_1_29 = _cvt_f32_256_29.y;
                            gate_x_1_29 = fminf(gate_x_1_29, swiglu_limit);
                            gate_y_1_29 = fminf(gate_y_1_29, swiglu_limit);
                            up_x_1_29 = fminf(fmaxf(up_x_1_29, -swiglu_limit), swiglu_limit);
                            up_y_1_29 = fminf(fmaxf(up_y_1_29, -swiglu_limit), swiglu_limit);
                            float _exp_239 = expf(gate_x_1_29 * -1.0f);
                            float denominator_x_1_29 = _exp_239 + 1.0f;
                            float _exp_240 = expf(gate_y_1_29 * -1.0f);
                            float denominator_y_1_29 = _exp_240 + 1.0f;
                            float hidden_x_1_29 = gate_x_1_29 / denominator_x_1_29 * up_x_1_29;
                            float hidden_y_1_29 = gate_y_1_29 / denominator_y_1_29 * up_y_1_29;
                            hidden_packed_0[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_29, hidden_y_1_29)));
                            float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[30]));
                            float2 _cvt_f32_256_30 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[30]));
                            float gate_x_1_30 = _cvt_f32_30.x;
                            float gate_y_1_30 = _cvt_f32_30.y;
                            float up_x_1_30 = _cvt_f32_256_30.x;
                            float up_y_1_30 = _cvt_f32_256_30.y;
                            gate_x_1_30 = fminf(gate_x_1_30, swiglu_limit);
                            gate_y_1_30 = fminf(gate_y_1_30, swiglu_limit);
                            up_x_1_30 = fminf(fmaxf(up_x_1_30, -swiglu_limit), swiglu_limit);
                            up_y_1_30 = fminf(fmaxf(up_y_1_30, -swiglu_limit), swiglu_limit);
                            float _exp_241 = expf(gate_x_1_30 * -1.0f);
                            float denominator_x_1_30 = _exp_241 + 1.0f;
                            float _exp_242 = expf(gate_y_1_30 * -1.0f);
                            float denominator_y_1_30 = _exp_242 + 1.0f;
                            float hidden_x_1_30 = gate_x_1_30 / denominator_x_1_30 * up_x_1_30;
                            float hidden_y_1_30 = gate_y_1_30 / denominator_y_1_30 * up_y_1_30;
                            hidden_packed_0[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_30, hidden_y_1_30)));
                            float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[31]));
                            float2 _cvt_f32_256_31 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[31]));
                            float gate_x_1_31 = _cvt_f32_31.x;
                            float gate_y_1_31 = _cvt_f32_31.y;
                            float up_x_1_31 = _cvt_f32_256_31.x;
                            float up_y_1_31 = _cvt_f32_256_31.y;
                            gate_x_1_31 = fminf(gate_x_1_31, swiglu_limit);
                            gate_y_1_31 = fminf(gate_y_1_31, swiglu_limit);
                            up_x_1_31 = fminf(fmaxf(up_x_1_31, -swiglu_limit), swiglu_limit);
                            up_y_1_31 = fminf(fmaxf(up_y_1_31, -swiglu_limit), swiglu_limit);
                            float _exp_243 = expf(gate_x_1_31 * -1.0f);
                            float denominator_x_1_31 = _exp_243 + 1.0f;
                            float _exp_244 = expf(gate_y_1_31 * -1.0f);
                            float denominator_y_1_31 = _exp_244 + 1.0f;
                            float hidden_x_1_31 = gate_x_1_31 / denominator_x_1_31 * up_x_1_31;
                            float hidden_y_1_31 = gate_y_1_31 / denominator_y_1_31 * up_y_1_31;
                            hidden_packed_0[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_31, hidden_y_1_31)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_3 = tid / 32;
                            int lane_1 = tid % 32;
                            #pragma unroll
                            for (int half_8 = 0; half_8 < 2; half_8++) {
                                #pragma unroll
                                for (int col_tile = 0; col_tile < 2; col_tile++) {
                                    int row_9 = warp_3 * 32 + half_8 * 16 + lane_1 % 16;
                                    int col_9 = col_tile * 16 + lane_1 / 16 * 8;
                                    unsigned int address = d_smem_addr + (unsigned int)((row_9 * 32 + col_9) * 2);
                                    address = address ^ (address & 511) >> 7 << 4;
                                    int offset = half_8 * 8 + col_tile * 4;
                                    uint32_t _stmatrix_addr_245 = static_cast<uint32_t>(address);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_245), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset + 3]))
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
                            int warp_0 = tid / 32;
                            int lane_0 = tid % 32;
                            #pragma unroll
                            for (int half_9 = 0; half_9 < 2; half_9++) {
                                #pragma unroll
                                for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                    int row_10 = warp_0 * 32 + half_9 * 16 + lane_0 % 16;
                                    int col_10 = col_tile_8 * 16 + lane_0 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + 8192 + (unsigned int)((row_10 * 32 + col_10) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset_1 = half_9 * 8 + col_tile_8 * 4;
                                    uint32_t _stmatrix_addr_246 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_246), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_1 + 3]))
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
                            int warp_1_1 = tid / 32;
                            int lane_1_1 = tid % 32;
                            #pragma unroll
                            for (int half_10 = 0; half_10 < 2; half_10++) {
                                #pragma unroll
                                for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                    int row_11 = warp_1_1 * 32 + half_10 * 16 + lane_1_1 % 16;
                                    int col_11 = col_tile_9 * 16 + lane_1_1 / 16 * 8;
                                    unsigned int address_2 = d_smem_addr + 16384 + (unsigned int)((row_11 * 32 + col_11) * 2);
                                    address_2 = address_2 ^ (address_2 & 511) >> 7 << 4;
                                    int offset_2 = half_10 * 8 + col_tile_9 * 4;
                                    uint32_t _stmatrix_addr_247 = static_cast<uint32_t>(address_2);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_247), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2 + 3]))
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
                            int warp_2_1 = tid / 32;
                            int lane_2 = tid % 32;
                            #pragma unroll
                            for (int half_11 = 0; half_11 < 2; half_11++) {
                                #pragma unroll
                                for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                    int row_12 = warp_2_1 * 32 + half_11 * 16 + lane_2 % 16;
                                    int col_12 = col_tile_10 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_3 = d_smem_addr + (unsigned int)((row_12 * 32 + col_12) * 2);
                                    address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                    int offset_3 = 16 + half_11 * 8 + col_tile_10 * 4;
                                    uint32_t _stmatrix_addr_248 = static_cast<uint32_t>(address_3);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_248), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_0[offset_3 + 3]))
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
                            int warp_3_1 = tid / 32;
                            int lane_3 = tid % 32;
                            #pragma unroll
                            for (int half_12 = 0; half_12 < 2; half_12++) {
                                #pragma unroll
                                for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                    int row_13 = warp_3_1 * 32 + half_12 * 16 + lane_3 % 16;
                                    int col_13 = col_tile_11 * 16 + lane_3 / 16 * 8;
                                    unsigned int address_4 = d_smem_addr + 8192 + (unsigned int)((row_13 * 32 + col_13) * 2);
                                    address_4 = address_4 ^ (address_4 & 511) >> 7 << 4;
                                    int offset_4 = 16 + half_12 * 8 + col_tile_11 * 4;
                                    uint32_t _stmatrix_addr_249 = static_cast<uint32_t>(address_4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_249), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_0[offset_4 + 3]))
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
                            int warp_4 = tid / 32;
                            int lane_4 = tid % 32;
                            #pragma unroll
                            for (int half_13 = 0; half_13 < 2; half_13++) {
                                #pragma unroll
                                for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                    int row_14 = warp_4 * 32 + half_13 * 16 + lane_4 % 16;
                                    int col_14 = col_tile_12 * 16 + lane_4 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + 16384 + (unsigned int)((row_14 * 32 + col_14) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_5 = 16 + half_13 * 8 + col_tile_12 * 4;
                                    uint32_t _stmatrix_addr_250 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_250), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_5 + 3]))
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
                            unsigned int gate_packed_1[32];
                            unsigned int up_packed_1[32];
                            unsigned int hidden_packed_1[32];
                            unsigned int address_3_4 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 64;
                            float _tmem_load_4[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                                : "r"(address_3_4));
                            float _tmem_load_32_4[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_4[15]))
                                : "r"(address_3_4 + 256));
                            gate_packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[0], _tmem_load_4[1])));
                            up_packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[0], _tmem_load_32_4[1])));
                            gate_packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[2], _tmem_load_4[3])));
                            up_packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[2], _tmem_load_32_4[3])));
                            gate_packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[4], _tmem_load_4[5])));
                            up_packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[4], _tmem_load_32_4[5])));
                            gate_packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[6], _tmem_load_4[7])));
                            up_packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[6], _tmem_load_32_4[7])));
                            gate_packed_1[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[8], _tmem_load_4[9])));
                            up_packed_1[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[8], _tmem_load_32_4[9])));
                            gate_packed_1[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[10], _tmem_load_4[11])));
                            up_packed_1[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[10], _tmem_load_32_4[11])));
                            gate_packed_1[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[12], _tmem_load_4[13])));
                            up_packed_1[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[12], _tmem_load_32_4[13])));
                            gate_packed_1[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_4[14], _tmem_load_4[15])));
                            up_packed_1[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_4[14], _tmem_load_32_4[15])));
                            unsigned int address_3_5 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 64;
                            float _tmem_load_5[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                                : "r"(address_3_5));
                            float _tmem_load_32_5[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_5[15]))
                                : "r"(address_3_5 + 256));
                            gate_packed_1[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[0], _tmem_load_5[1])));
                            up_packed_1[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[0], _tmem_load_32_5[1])));
                            gate_packed_1[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[2], _tmem_load_5[3])));
                            up_packed_1[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[2], _tmem_load_32_5[3])));
                            gate_packed_1[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[4], _tmem_load_5[5])));
                            up_packed_1[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[4], _tmem_load_32_5[5])));
                            gate_packed_1[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[6], _tmem_load_5[7])));
                            up_packed_1[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[6], _tmem_load_32_5[7])));
                            gate_packed_1[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[8], _tmem_load_5[9])));
                            up_packed_1[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[8], _tmem_load_32_5[9])));
                            gate_packed_1[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[10], _tmem_load_5[11])));
                            up_packed_1[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[10], _tmem_load_32_5[11])));
                            gate_packed_1[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[12], _tmem_load_5[13])));
                            up_packed_1[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[12], _tmem_load_32_5[13])));
                            gate_packed_1[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_5[14], _tmem_load_5[15])));
                            up_packed_1[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_5[14], _tmem_load_32_5[15])));
                            unsigned int address_3_6 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 96;
                            float _tmem_load_6[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                                : "r"(address_3_6));
                            float _tmem_load_32_6[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_6[15]))
                                : "r"(address_3_6 + 256));
                            gate_packed_1[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[0], _tmem_load_6[1])));
                            up_packed_1[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[0], _tmem_load_32_6[1])));
                            gate_packed_1[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[2], _tmem_load_6[3])));
                            up_packed_1[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[2], _tmem_load_32_6[3])));
                            gate_packed_1[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[4], _tmem_load_6[5])));
                            up_packed_1[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[4], _tmem_load_32_6[5])));
                            gate_packed_1[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[6], _tmem_load_6[7])));
                            up_packed_1[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[6], _tmem_load_32_6[7])));
                            gate_packed_1[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[8], _tmem_load_6[9])));
                            up_packed_1[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[8], _tmem_load_32_6[9])));
                            gate_packed_1[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[10], _tmem_load_6[11])));
                            up_packed_1[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[10], _tmem_load_32_6[11])));
                            gate_packed_1[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[12], _tmem_load_6[13])));
                            up_packed_1[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[12], _tmem_load_32_6[13])));
                            gate_packed_1[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_6[14], _tmem_load_6[15])));
                            up_packed_1[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_6[14], _tmem_load_32_6[15])));
                            unsigned int address_3_7 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 96;
                            float _tmem_load_7[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                                : "r"(address_3_7));
                            float _tmem_load_32_7[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_7[15]))
                                : "r"(address_3_7 + 256));
                            gate_packed_1[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[0], _tmem_load_7[1])));
                            up_packed_1[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[0], _tmem_load_32_7[1])));
                            gate_packed_1[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[2], _tmem_load_7[3])));
                            up_packed_1[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[2], _tmem_load_32_7[3])));
                            gate_packed_1[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[4], _tmem_load_7[5])));
                            up_packed_1[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[4], _tmem_load_32_7[5])));
                            gate_packed_1[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[6], _tmem_load_7[7])));
                            up_packed_1[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[6], _tmem_load_32_7[7])));
                            gate_packed_1[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[8], _tmem_load_7[9])));
                            up_packed_1[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[8], _tmem_load_32_7[9])));
                            gate_packed_1[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[10], _tmem_load_7[11])));
                            up_packed_1[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[10], _tmem_load_32_7[11])));
                            gate_packed_1[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[12], _tmem_load_7[13])));
                            up_packed_1[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[12], _tmem_load_32_7[13])));
                            gate_packed_1[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_7[14], _tmem_load_7[15])));
                            up_packed_1[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_7[14], _tmem_load_32_7[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[0]));
                            float2 _cvt_f32_256_32 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[0]));
                            float gate_x_1_32 = _cvt_f32_32.x;
                            float gate_y_1_32 = _cvt_f32_32.y;
                            float up_x_1_32 = _cvt_f32_256_32.x;
                            float up_y_1_32 = _cvt_f32_256_32.y;
                            gate_x_1_32 = fminf(gate_x_1_32, swiglu_limit);
                            gate_y_1_32 = fminf(gate_y_1_32, swiglu_limit);
                            up_x_1_32 = fminf(fmaxf(up_x_1_32, -swiglu_limit), swiglu_limit);
                            up_y_1_32 = fminf(fmaxf(up_y_1_32, -swiglu_limit), swiglu_limit);
                            float _exp_251 = expf(gate_x_1_32 * -1.0f);
                            float denominator_x_1_32 = _exp_251 + 1.0f;
                            float _exp_252 = expf(gate_y_1_32 * -1.0f);
                            float denominator_y_1_32 = _exp_252 + 1.0f;
                            float hidden_x_1_32 = gate_x_1_32 / denominator_x_1_32 * up_x_1_32;
                            float hidden_y_1_32 = gate_y_1_32 / denominator_y_1_32 * up_y_1_32;
                            hidden_packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_32, hidden_y_1_32)));
                            float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[1]));
                            float2 _cvt_f32_256_33 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[1]));
                            float gate_x_1_33 = _cvt_f32_33.x;
                            float gate_y_1_33 = _cvt_f32_33.y;
                            float up_x_1_33 = _cvt_f32_256_33.x;
                            float up_y_1_33 = _cvt_f32_256_33.y;
                            gate_x_1_33 = fminf(gate_x_1_33, swiglu_limit);
                            gate_y_1_33 = fminf(gate_y_1_33, swiglu_limit);
                            up_x_1_33 = fminf(fmaxf(up_x_1_33, -swiglu_limit), swiglu_limit);
                            up_y_1_33 = fminf(fmaxf(up_y_1_33, -swiglu_limit), swiglu_limit);
                            float _exp_253 = expf(gate_x_1_33 * -1.0f);
                            float denominator_x_1_33 = _exp_253 + 1.0f;
                            float _exp_254 = expf(gate_y_1_33 * -1.0f);
                            float denominator_y_1_33 = _exp_254 + 1.0f;
                            float hidden_x_1_33 = gate_x_1_33 / denominator_x_1_33 * up_x_1_33;
                            float hidden_y_1_33 = gate_y_1_33 / denominator_y_1_33 * up_y_1_33;
                            hidden_packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_33, hidden_y_1_33)));
                            float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[2]));
                            float2 _cvt_f32_256_34 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[2]));
                            float gate_x_1_34 = _cvt_f32_34.x;
                            float gate_y_1_34 = _cvt_f32_34.y;
                            float up_x_1_34 = _cvt_f32_256_34.x;
                            float up_y_1_34 = _cvt_f32_256_34.y;
                            gate_x_1_34 = fminf(gate_x_1_34, swiglu_limit);
                            gate_y_1_34 = fminf(gate_y_1_34, swiglu_limit);
                            up_x_1_34 = fminf(fmaxf(up_x_1_34, -swiglu_limit), swiglu_limit);
                            up_y_1_34 = fminf(fmaxf(up_y_1_34, -swiglu_limit), swiglu_limit);
                            float _exp_255 = expf(gate_x_1_34 * -1.0f);
                            float denominator_x_1_34 = _exp_255 + 1.0f;
                            float _exp_256 = expf(gate_y_1_34 * -1.0f);
                            float denominator_y_1_34 = _exp_256 + 1.0f;
                            float hidden_x_1_34 = gate_x_1_34 / denominator_x_1_34 * up_x_1_34;
                            float hidden_y_1_34 = gate_y_1_34 / denominator_y_1_34 * up_y_1_34;
                            hidden_packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_34, hidden_y_1_34)));
                            float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[3]));
                            float2 _cvt_f32_256_35 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[3]));
                            float gate_x_1_35 = _cvt_f32_35.x;
                            float gate_y_1_35 = _cvt_f32_35.y;
                            float up_x_1_35 = _cvt_f32_256_35.x;
                            float up_y_1_35 = _cvt_f32_256_35.y;
                            gate_x_1_35 = fminf(gate_x_1_35, swiglu_limit);
                            gate_y_1_35 = fminf(gate_y_1_35, swiglu_limit);
                            up_x_1_35 = fminf(fmaxf(up_x_1_35, -swiglu_limit), swiglu_limit);
                            up_y_1_35 = fminf(fmaxf(up_y_1_35, -swiglu_limit), swiglu_limit);
                            float _exp_257 = expf(gate_x_1_35 * -1.0f);
                            float denominator_x_1_35 = _exp_257 + 1.0f;
                            float _exp_258 = expf(gate_y_1_35 * -1.0f);
                            float denominator_y_1_35 = _exp_258 + 1.0f;
                            float hidden_x_1_35 = gate_x_1_35 / denominator_x_1_35 * up_x_1_35;
                            float hidden_y_1_35 = gate_y_1_35 / denominator_y_1_35 * up_y_1_35;
                            hidden_packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_35, hidden_y_1_35)));
                            float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[4]));
                            float2 _cvt_f32_256_36 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[4]));
                            float gate_x_1_36 = _cvt_f32_36.x;
                            float gate_y_1_36 = _cvt_f32_36.y;
                            float up_x_1_36 = _cvt_f32_256_36.x;
                            float up_y_1_36 = _cvt_f32_256_36.y;
                            gate_x_1_36 = fminf(gate_x_1_36, swiglu_limit);
                            gate_y_1_36 = fminf(gate_y_1_36, swiglu_limit);
                            up_x_1_36 = fminf(fmaxf(up_x_1_36, -swiglu_limit), swiglu_limit);
                            up_y_1_36 = fminf(fmaxf(up_y_1_36, -swiglu_limit), swiglu_limit);
                            float _exp_259 = expf(gate_x_1_36 * -1.0f);
                            float denominator_x_1_36 = _exp_259 + 1.0f;
                            float _exp_260 = expf(gate_y_1_36 * -1.0f);
                            float denominator_y_1_36 = _exp_260 + 1.0f;
                            float hidden_x_1_36 = gate_x_1_36 / denominator_x_1_36 * up_x_1_36;
                            float hidden_y_1_36 = gate_y_1_36 / denominator_y_1_36 * up_y_1_36;
                            hidden_packed_1[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_36, hidden_y_1_36)));
                            float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[5]));
                            float2 _cvt_f32_256_37 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[5]));
                            float gate_x_1_37 = _cvt_f32_37.x;
                            float gate_y_1_37 = _cvt_f32_37.y;
                            float up_x_1_37 = _cvt_f32_256_37.x;
                            float up_y_1_37 = _cvt_f32_256_37.y;
                            gate_x_1_37 = fminf(gate_x_1_37, swiglu_limit);
                            gate_y_1_37 = fminf(gate_y_1_37, swiglu_limit);
                            up_x_1_37 = fminf(fmaxf(up_x_1_37, -swiglu_limit), swiglu_limit);
                            up_y_1_37 = fminf(fmaxf(up_y_1_37, -swiglu_limit), swiglu_limit);
                            float _exp_261 = expf(gate_x_1_37 * -1.0f);
                            float denominator_x_1_37 = _exp_261 + 1.0f;
                            float _exp_262 = expf(gate_y_1_37 * -1.0f);
                            float denominator_y_1_37 = _exp_262 + 1.0f;
                            float hidden_x_1_37 = gate_x_1_37 / denominator_x_1_37 * up_x_1_37;
                            float hidden_y_1_37 = gate_y_1_37 / denominator_y_1_37 * up_y_1_37;
                            hidden_packed_1[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_37, hidden_y_1_37)));
                            float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[6]));
                            float2 _cvt_f32_256_38 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[6]));
                            float gate_x_1_38 = _cvt_f32_38.x;
                            float gate_y_1_38 = _cvt_f32_38.y;
                            float up_x_1_38 = _cvt_f32_256_38.x;
                            float up_y_1_38 = _cvt_f32_256_38.y;
                            gate_x_1_38 = fminf(gate_x_1_38, swiglu_limit);
                            gate_y_1_38 = fminf(gate_y_1_38, swiglu_limit);
                            up_x_1_38 = fminf(fmaxf(up_x_1_38, -swiglu_limit), swiglu_limit);
                            up_y_1_38 = fminf(fmaxf(up_y_1_38, -swiglu_limit), swiglu_limit);
                            float _exp_263 = expf(gate_x_1_38 * -1.0f);
                            float denominator_x_1_38 = _exp_263 + 1.0f;
                            float _exp_264 = expf(gate_y_1_38 * -1.0f);
                            float denominator_y_1_38 = _exp_264 + 1.0f;
                            float hidden_x_1_38 = gate_x_1_38 / denominator_x_1_38 * up_x_1_38;
                            float hidden_y_1_38 = gate_y_1_38 / denominator_y_1_38 * up_y_1_38;
                            hidden_packed_1[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_38, hidden_y_1_38)));
                            float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[7]));
                            float2 _cvt_f32_256_39 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[7]));
                            float gate_x_1_39 = _cvt_f32_39.x;
                            float gate_y_1_39 = _cvt_f32_39.y;
                            float up_x_1_39 = _cvt_f32_256_39.x;
                            float up_y_1_39 = _cvt_f32_256_39.y;
                            gate_x_1_39 = fminf(gate_x_1_39, swiglu_limit);
                            gate_y_1_39 = fminf(gate_y_1_39, swiglu_limit);
                            up_x_1_39 = fminf(fmaxf(up_x_1_39, -swiglu_limit), swiglu_limit);
                            up_y_1_39 = fminf(fmaxf(up_y_1_39, -swiglu_limit), swiglu_limit);
                            float _exp_265 = expf(gate_x_1_39 * -1.0f);
                            float denominator_x_1_39 = _exp_265 + 1.0f;
                            float _exp_266 = expf(gate_y_1_39 * -1.0f);
                            float denominator_y_1_39 = _exp_266 + 1.0f;
                            float hidden_x_1_39 = gate_x_1_39 / denominator_x_1_39 * up_x_1_39;
                            float hidden_y_1_39 = gate_y_1_39 / denominator_y_1_39 * up_y_1_39;
                            hidden_packed_1[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_39, hidden_y_1_39)));
                            float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[8]));
                            float2 _cvt_f32_256_40 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[8]));
                            float gate_x_1_40 = _cvt_f32_40.x;
                            float gate_y_1_40 = _cvt_f32_40.y;
                            float up_x_1_40 = _cvt_f32_256_40.x;
                            float up_y_1_40 = _cvt_f32_256_40.y;
                            gate_x_1_40 = fminf(gate_x_1_40, swiglu_limit);
                            gate_y_1_40 = fminf(gate_y_1_40, swiglu_limit);
                            up_x_1_40 = fminf(fmaxf(up_x_1_40, -swiglu_limit), swiglu_limit);
                            up_y_1_40 = fminf(fmaxf(up_y_1_40, -swiglu_limit), swiglu_limit);
                            float _exp_267 = expf(gate_x_1_40 * -1.0f);
                            float denominator_x_1_40 = _exp_267 + 1.0f;
                            float _exp_268 = expf(gate_y_1_40 * -1.0f);
                            float denominator_y_1_40 = _exp_268 + 1.0f;
                            float hidden_x_1_40 = gate_x_1_40 / denominator_x_1_40 * up_x_1_40;
                            float hidden_y_1_40 = gate_y_1_40 / denominator_y_1_40 * up_y_1_40;
                            hidden_packed_1[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_40, hidden_y_1_40)));
                            float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[9]));
                            float2 _cvt_f32_256_41 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[9]));
                            float gate_x_1_41 = _cvt_f32_41.x;
                            float gate_y_1_41 = _cvt_f32_41.y;
                            float up_x_1_41 = _cvt_f32_256_41.x;
                            float up_y_1_41 = _cvt_f32_256_41.y;
                            gate_x_1_41 = fminf(gate_x_1_41, swiglu_limit);
                            gate_y_1_41 = fminf(gate_y_1_41, swiglu_limit);
                            up_x_1_41 = fminf(fmaxf(up_x_1_41, -swiglu_limit), swiglu_limit);
                            up_y_1_41 = fminf(fmaxf(up_y_1_41, -swiglu_limit), swiglu_limit);
                            float _exp_269 = expf(gate_x_1_41 * -1.0f);
                            float denominator_x_1_41 = _exp_269 + 1.0f;
                            float _exp_270 = expf(gate_y_1_41 * -1.0f);
                            float denominator_y_1_41 = _exp_270 + 1.0f;
                            float hidden_x_1_41 = gate_x_1_41 / denominator_x_1_41 * up_x_1_41;
                            float hidden_y_1_41 = gate_y_1_41 / denominator_y_1_41 * up_y_1_41;
                            hidden_packed_1[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_41, hidden_y_1_41)));
                            float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[10]));
                            float2 _cvt_f32_256_42 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[10]));
                            float gate_x_1_42 = _cvt_f32_42.x;
                            float gate_y_1_42 = _cvt_f32_42.y;
                            float up_x_1_42 = _cvt_f32_256_42.x;
                            float up_y_1_42 = _cvt_f32_256_42.y;
                            gate_x_1_42 = fminf(gate_x_1_42, swiglu_limit);
                            gate_y_1_42 = fminf(gate_y_1_42, swiglu_limit);
                            up_x_1_42 = fminf(fmaxf(up_x_1_42, -swiglu_limit), swiglu_limit);
                            up_y_1_42 = fminf(fmaxf(up_y_1_42, -swiglu_limit), swiglu_limit);
                            float _exp_271 = expf(gate_x_1_42 * -1.0f);
                            float denominator_x_1_42 = _exp_271 + 1.0f;
                            float _exp_272 = expf(gate_y_1_42 * -1.0f);
                            float denominator_y_1_42 = _exp_272 + 1.0f;
                            float hidden_x_1_42 = gate_x_1_42 / denominator_x_1_42 * up_x_1_42;
                            float hidden_y_1_42 = gate_y_1_42 / denominator_y_1_42 * up_y_1_42;
                            hidden_packed_1[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_42, hidden_y_1_42)));
                            float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[11]));
                            float2 _cvt_f32_256_43 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[11]));
                            float gate_x_1_43 = _cvt_f32_43.x;
                            float gate_y_1_43 = _cvt_f32_43.y;
                            float up_x_1_43 = _cvt_f32_256_43.x;
                            float up_y_1_43 = _cvt_f32_256_43.y;
                            gate_x_1_43 = fminf(gate_x_1_43, swiglu_limit);
                            gate_y_1_43 = fminf(gate_y_1_43, swiglu_limit);
                            up_x_1_43 = fminf(fmaxf(up_x_1_43, -swiglu_limit), swiglu_limit);
                            up_y_1_43 = fminf(fmaxf(up_y_1_43, -swiglu_limit), swiglu_limit);
                            float _exp_273 = expf(gate_x_1_43 * -1.0f);
                            float denominator_x_1_43 = _exp_273 + 1.0f;
                            float _exp_274 = expf(gate_y_1_43 * -1.0f);
                            float denominator_y_1_43 = _exp_274 + 1.0f;
                            float hidden_x_1_43 = gate_x_1_43 / denominator_x_1_43 * up_x_1_43;
                            float hidden_y_1_43 = gate_y_1_43 / denominator_y_1_43 * up_y_1_43;
                            hidden_packed_1[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_43, hidden_y_1_43)));
                            float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[12]));
                            float2 _cvt_f32_256_44 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[12]));
                            float gate_x_1_44 = _cvt_f32_44.x;
                            float gate_y_1_44 = _cvt_f32_44.y;
                            float up_x_1_44 = _cvt_f32_256_44.x;
                            float up_y_1_44 = _cvt_f32_256_44.y;
                            gate_x_1_44 = fminf(gate_x_1_44, swiglu_limit);
                            gate_y_1_44 = fminf(gate_y_1_44, swiglu_limit);
                            up_x_1_44 = fminf(fmaxf(up_x_1_44, -swiglu_limit), swiglu_limit);
                            up_y_1_44 = fminf(fmaxf(up_y_1_44, -swiglu_limit), swiglu_limit);
                            float _exp_275 = expf(gate_x_1_44 * -1.0f);
                            float denominator_x_1_44 = _exp_275 + 1.0f;
                            float _exp_276 = expf(gate_y_1_44 * -1.0f);
                            float denominator_y_1_44 = _exp_276 + 1.0f;
                            float hidden_x_1_44 = gate_x_1_44 / denominator_x_1_44 * up_x_1_44;
                            float hidden_y_1_44 = gate_y_1_44 / denominator_y_1_44 * up_y_1_44;
                            hidden_packed_1[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_44, hidden_y_1_44)));
                            float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[13]));
                            float2 _cvt_f32_256_45 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[13]));
                            float gate_x_1_45 = _cvt_f32_45.x;
                            float gate_y_1_45 = _cvt_f32_45.y;
                            float up_x_1_45 = _cvt_f32_256_45.x;
                            float up_y_1_45 = _cvt_f32_256_45.y;
                            gate_x_1_45 = fminf(gate_x_1_45, swiglu_limit);
                            gate_y_1_45 = fminf(gate_y_1_45, swiglu_limit);
                            up_x_1_45 = fminf(fmaxf(up_x_1_45, -swiglu_limit), swiglu_limit);
                            up_y_1_45 = fminf(fmaxf(up_y_1_45, -swiglu_limit), swiglu_limit);
                            float _exp_277 = expf(gate_x_1_45 * -1.0f);
                            float denominator_x_1_45 = _exp_277 + 1.0f;
                            float _exp_278 = expf(gate_y_1_45 * -1.0f);
                            float denominator_y_1_45 = _exp_278 + 1.0f;
                            float hidden_x_1_45 = gate_x_1_45 / denominator_x_1_45 * up_x_1_45;
                            float hidden_y_1_45 = gate_y_1_45 / denominator_y_1_45 * up_y_1_45;
                            hidden_packed_1[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_45, hidden_y_1_45)));
                            float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[14]));
                            float2 _cvt_f32_256_46 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[14]));
                            float gate_x_1_46 = _cvt_f32_46.x;
                            float gate_y_1_46 = _cvt_f32_46.y;
                            float up_x_1_46 = _cvt_f32_256_46.x;
                            float up_y_1_46 = _cvt_f32_256_46.y;
                            gate_x_1_46 = fminf(gate_x_1_46, swiglu_limit);
                            gate_y_1_46 = fminf(gate_y_1_46, swiglu_limit);
                            up_x_1_46 = fminf(fmaxf(up_x_1_46, -swiglu_limit), swiglu_limit);
                            up_y_1_46 = fminf(fmaxf(up_y_1_46, -swiglu_limit), swiglu_limit);
                            float _exp_279 = expf(gate_x_1_46 * -1.0f);
                            float denominator_x_1_46 = _exp_279 + 1.0f;
                            float _exp_280 = expf(gate_y_1_46 * -1.0f);
                            float denominator_y_1_46 = _exp_280 + 1.0f;
                            float hidden_x_1_46 = gate_x_1_46 / denominator_x_1_46 * up_x_1_46;
                            float hidden_y_1_46 = gate_y_1_46 / denominator_y_1_46 * up_y_1_46;
                            hidden_packed_1[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_46, hidden_y_1_46)));
                            float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[15]));
                            float2 _cvt_f32_256_47 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[15]));
                            float gate_x_1_47 = _cvt_f32_47.x;
                            float gate_y_1_47 = _cvt_f32_47.y;
                            float up_x_1_47 = _cvt_f32_256_47.x;
                            float up_y_1_47 = _cvt_f32_256_47.y;
                            gate_x_1_47 = fminf(gate_x_1_47, swiglu_limit);
                            gate_y_1_47 = fminf(gate_y_1_47, swiglu_limit);
                            up_x_1_47 = fminf(fmaxf(up_x_1_47, -swiglu_limit), swiglu_limit);
                            up_y_1_47 = fminf(fmaxf(up_y_1_47, -swiglu_limit), swiglu_limit);
                            float _exp_281 = expf(gate_x_1_47 * -1.0f);
                            float denominator_x_1_47 = _exp_281 + 1.0f;
                            float _exp_282 = expf(gate_y_1_47 * -1.0f);
                            float denominator_y_1_47 = _exp_282 + 1.0f;
                            float hidden_x_1_47 = gate_x_1_47 / denominator_x_1_47 * up_x_1_47;
                            float hidden_y_1_47 = gate_y_1_47 / denominator_y_1_47 * up_y_1_47;
                            hidden_packed_1[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_47, hidden_y_1_47)));
                            float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[16]));
                            float2 _cvt_f32_256_48 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[16]));
                            float gate_x_1_48 = _cvt_f32_48.x;
                            float gate_y_1_48 = _cvt_f32_48.y;
                            float up_x_1_48 = _cvt_f32_256_48.x;
                            float up_y_1_48 = _cvt_f32_256_48.y;
                            gate_x_1_48 = fminf(gate_x_1_48, swiglu_limit);
                            gate_y_1_48 = fminf(gate_y_1_48, swiglu_limit);
                            up_x_1_48 = fminf(fmaxf(up_x_1_48, -swiglu_limit), swiglu_limit);
                            up_y_1_48 = fminf(fmaxf(up_y_1_48, -swiglu_limit), swiglu_limit);
                            float _exp_283 = expf(gate_x_1_48 * -1.0f);
                            float denominator_x_1_48 = _exp_283 + 1.0f;
                            float _exp_284 = expf(gate_y_1_48 * -1.0f);
                            float denominator_y_1_48 = _exp_284 + 1.0f;
                            float hidden_x_1_48 = gate_x_1_48 / denominator_x_1_48 * up_x_1_48;
                            float hidden_y_1_48 = gate_y_1_48 / denominator_y_1_48 * up_y_1_48;
                            hidden_packed_1[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_48, hidden_y_1_48)));
                            float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[17]));
                            float2 _cvt_f32_256_49 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[17]));
                            float gate_x_1_49 = _cvt_f32_49.x;
                            float gate_y_1_49 = _cvt_f32_49.y;
                            float up_x_1_49 = _cvt_f32_256_49.x;
                            float up_y_1_49 = _cvt_f32_256_49.y;
                            gate_x_1_49 = fminf(gate_x_1_49, swiglu_limit);
                            gate_y_1_49 = fminf(gate_y_1_49, swiglu_limit);
                            up_x_1_49 = fminf(fmaxf(up_x_1_49, -swiglu_limit), swiglu_limit);
                            up_y_1_49 = fminf(fmaxf(up_y_1_49, -swiglu_limit), swiglu_limit);
                            float _exp_285 = expf(gate_x_1_49 * -1.0f);
                            float denominator_x_1_49 = _exp_285 + 1.0f;
                            float _exp_286 = expf(gate_y_1_49 * -1.0f);
                            float denominator_y_1_49 = _exp_286 + 1.0f;
                            float hidden_x_1_49 = gate_x_1_49 / denominator_x_1_49 * up_x_1_49;
                            float hidden_y_1_49 = gate_y_1_49 / denominator_y_1_49 * up_y_1_49;
                            hidden_packed_1[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_49, hidden_y_1_49)));
                            float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[18]));
                            float2 _cvt_f32_256_50 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[18]));
                            float gate_x_1_50 = _cvt_f32_50.x;
                            float gate_y_1_50 = _cvt_f32_50.y;
                            float up_x_1_50 = _cvt_f32_256_50.x;
                            float up_y_1_50 = _cvt_f32_256_50.y;
                            gate_x_1_50 = fminf(gate_x_1_50, swiglu_limit);
                            gate_y_1_50 = fminf(gate_y_1_50, swiglu_limit);
                            up_x_1_50 = fminf(fmaxf(up_x_1_50, -swiglu_limit), swiglu_limit);
                            up_y_1_50 = fminf(fmaxf(up_y_1_50, -swiglu_limit), swiglu_limit);
                            float _exp_287 = expf(gate_x_1_50 * -1.0f);
                            float denominator_x_1_50 = _exp_287 + 1.0f;
                            float _exp_288 = expf(gate_y_1_50 * -1.0f);
                            float denominator_y_1_50 = _exp_288 + 1.0f;
                            float hidden_x_1_50 = gate_x_1_50 / denominator_x_1_50 * up_x_1_50;
                            float hidden_y_1_50 = gate_y_1_50 / denominator_y_1_50 * up_y_1_50;
                            hidden_packed_1[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_50, hidden_y_1_50)));
                            float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[19]));
                            float2 _cvt_f32_256_51 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[19]));
                            float gate_x_1_51 = _cvt_f32_51.x;
                            float gate_y_1_51 = _cvt_f32_51.y;
                            float up_x_1_51 = _cvt_f32_256_51.x;
                            float up_y_1_51 = _cvt_f32_256_51.y;
                            gate_x_1_51 = fminf(gate_x_1_51, swiglu_limit);
                            gate_y_1_51 = fminf(gate_y_1_51, swiglu_limit);
                            up_x_1_51 = fminf(fmaxf(up_x_1_51, -swiglu_limit), swiglu_limit);
                            up_y_1_51 = fminf(fmaxf(up_y_1_51, -swiglu_limit), swiglu_limit);
                            float _exp_289 = expf(gate_x_1_51 * -1.0f);
                            float denominator_x_1_51 = _exp_289 + 1.0f;
                            float _exp_290 = expf(gate_y_1_51 * -1.0f);
                            float denominator_y_1_51 = _exp_290 + 1.0f;
                            float hidden_x_1_51 = gate_x_1_51 / denominator_x_1_51 * up_x_1_51;
                            float hidden_y_1_51 = gate_y_1_51 / denominator_y_1_51 * up_y_1_51;
                            hidden_packed_1[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_51, hidden_y_1_51)));
                            float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[20]));
                            float2 _cvt_f32_256_52 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[20]));
                            float gate_x_1_52 = _cvt_f32_52.x;
                            float gate_y_1_52 = _cvt_f32_52.y;
                            float up_x_1_52 = _cvt_f32_256_52.x;
                            float up_y_1_52 = _cvt_f32_256_52.y;
                            gate_x_1_52 = fminf(gate_x_1_52, swiglu_limit);
                            gate_y_1_52 = fminf(gate_y_1_52, swiglu_limit);
                            up_x_1_52 = fminf(fmaxf(up_x_1_52, -swiglu_limit), swiglu_limit);
                            up_y_1_52 = fminf(fmaxf(up_y_1_52, -swiglu_limit), swiglu_limit);
                            float _exp_291 = expf(gate_x_1_52 * -1.0f);
                            float denominator_x_1_52 = _exp_291 + 1.0f;
                            float _exp_292 = expf(gate_y_1_52 * -1.0f);
                            float denominator_y_1_52 = _exp_292 + 1.0f;
                            float hidden_x_1_52 = gate_x_1_52 / denominator_x_1_52 * up_x_1_52;
                            float hidden_y_1_52 = gate_y_1_52 / denominator_y_1_52 * up_y_1_52;
                            hidden_packed_1[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_52, hidden_y_1_52)));
                            float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[21]));
                            float2 _cvt_f32_256_53 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[21]));
                            float gate_x_1_53 = _cvt_f32_53.x;
                            float gate_y_1_53 = _cvt_f32_53.y;
                            float up_x_1_53 = _cvt_f32_256_53.x;
                            float up_y_1_53 = _cvt_f32_256_53.y;
                            gate_x_1_53 = fminf(gate_x_1_53, swiglu_limit);
                            gate_y_1_53 = fminf(gate_y_1_53, swiglu_limit);
                            up_x_1_53 = fminf(fmaxf(up_x_1_53, -swiglu_limit), swiglu_limit);
                            up_y_1_53 = fminf(fmaxf(up_y_1_53, -swiglu_limit), swiglu_limit);
                            float _exp_293 = expf(gate_x_1_53 * -1.0f);
                            float denominator_x_1_53 = _exp_293 + 1.0f;
                            float _exp_294 = expf(gate_y_1_53 * -1.0f);
                            float denominator_y_1_53 = _exp_294 + 1.0f;
                            float hidden_x_1_53 = gate_x_1_53 / denominator_x_1_53 * up_x_1_53;
                            float hidden_y_1_53 = gate_y_1_53 / denominator_y_1_53 * up_y_1_53;
                            hidden_packed_1[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_53, hidden_y_1_53)));
                            float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[22]));
                            float2 _cvt_f32_256_54 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[22]));
                            float gate_x_1_54 = _cvt_f32_54.x;
                            float gate_y_1_54 = _cvt_f32_54.y;
                            float up_x_1_54 = _cvt_f32_256_54.x;
                            float up_y_1_54 = _cvt_f32_256_54.y;
                            gate_x_1_54 = fminf(gate_x_1_54, swiglu_limit);
                            gate_y_1_54 = fminf(gate_y_1_54, swiglu_limit);
                            up_x_1_54 = fminf(fmaxf(up_x_1_54, -swiglu_limit), swiglu_limit);
                            up_y_1_54 = fminf(fmaxf(up_y_1_54, -swiglu_limit), swiglu_limit);
                            float _exp_295 = expf(gate_x_1_54 * -1.0f);
                            float denominator_x_1_54 = _exp_295 + 1.0f;
                            float _exp_296 = expf(gate_y_1_54 * -1.0f);
                            float denominator_y_1_54 = _exp_296 + 1.0f;
                            float hidden_x_1_54 = gate_x_1_54 / denominator_x_1_54 * up_x_1_54;
                            float hidden_y_1_54 = gate_y_1_54 / denominator_y_1_54 * up_y_1_54;
                            hidden_packed_1[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_54, hidden_y_1_54)));
                            float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[23]));
                            float2 _cvt_f32_256_55 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[23]));
                            float gate_x_1_55 = _cvt_f32_55.x;
                            float gate_y_1_55 = _cvt_f32_55.y;
                            float up_x_1_55 = _cvt_f32_256_55.x;
                            float up_y_1_55 = _cvt_f32_256_55.y;
                            gate_x_1_55 = fminf(gate_x_1_55, swiglu_limit);
                            gate_y_1_55 = fminf(gate_y_1_55, swiglu_limit);
                            up_x_1_55 = fminf(fmaxf(up_x_1_55, -swiglu_limit), swiglu_limit);
                            up_y_1_55 = fminf(fmaxf(up_y_1_55, -swiglu_limit), swiglu_limit);
                            float _exp_297 = expf(gate_x_1_55 * -1.0f);
                            float denominator_x_1_55 = _exp_297 + 1.0f;
                            float _exp_298 = expf(gate_y_1_55 * -1.0f);
                            float denominator_y_1_55 = _exp_298 + 1.0f;
                            float hidden_x_1_55 = gate_x_1_55 / denominator_x_1_55 * up_x_1_55;
                            float hidden_y_1_55 = gate_y_1_55 / denominator_y_1_55 * up_y_1_55;
                            hidden_packed_1[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_55, hidden_y_1_55)));
                            float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[24]));
                            float2 _cvt_f32_256_56 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[24]));
                            float gate_x_1_56 = _cvt_f32_56.x;
                            float gate_y_1_56 = _cvt_f32_56.y;
                            float up_x_1_56 = _cvt_f32_256_56.x;
                            float up_y_1_56 = _cvt_f32_256_56.y;
                            gate_x_1_56 = fminf(gate_x_1_56, swiglu_limit);
                            gate_y_1_56 = fminf(gate_y_1_56, swiglu_limit);
                            up_x_1_56 = fminf(fmaxf(up_x_1_56, -swiglu_limit), swiglu_limit);
                            up_y_1_56 = fminf(fmaxf(up_y_1_56, -swiglu_limit), swiglu_limit);
                            float _exp_299 = expf(gate_x_1_56 * -1.0f);
                            float denominator_x_1_56 = _exp_299 + 1.0f;
                            float _exp_300 = expf(gate_y_1_56 * -1.0f);
                            float denominator_y_1_56 = _exp_300 + 1.0f;
                            float hidden_x_1_56 = gate_x_1_56 / denominator_x_1_56 * up_x_1_56;
                            float hidden_y_1_56 = gate_y_1_56 / denominator_y_1_56 * up_y_1_56;
                            hidden_packed_1[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_56, hidden_y_1_56)));
                            float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[25]));
                            float2 _cvt_f32_256_57 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[25]));
                            float gate_x_1_57 = _cvt_f32_57.x;
                            float gate_y_1_57 = _cvt_f32_57.y;
                            float up_x_1_57 = _cvt_f32_256_57.x;
                            float up_y_1_57 = _cvt_f32_256_57.y;
                            gate_x_1_57 = fminf(gate_x_1_57, swiglu_limit);
                            gate_y_1_57 = fminf(gate_y_1_57, swiglu_limit);
                            up_x_1_57 = fminf(fmaxf(up_x_1_57, -swiglu_limit), swiglu_limit);
                            up_y_1_57 = fminf(fmaxf(up_y_1_57, -swiglu_limit), swiglu_limit);
                            float _exp_301 = expf(gate_x_1_57 * -1.0f);
                            float denominator_x_1_57 = _exp_301 + 1.0f;
                            float _exp_302 = expf(gate_y_1_57 * -1.0f);
                            float denominator_y_1_57 = _exp_302 + 1.0f;
                            float hidden_x_1_57 = gate_x_1_57 / denominator_x_1_57 * up_x_1_57;
                            float hidden_y_1_57 = gate_y_1_57 / denominator_y_1_57 * up_y_1_57;
                            hidden_packed_1[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_57, hidden_y_1_57)));
                            float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[26]));
                            float2 _cvt_f32_256_58 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[26]));
                            float gate_x_1_58 = _cvt_f32_58.x;
                            float gate_y_1_58 = _cvt_f32_58.y;
                            float up_x_1_58 = _cvt_f32_256_58.x;
                            float up_y_1_58 = _cvt_f32_256_58.y;
                            gate_x_1_58 = fminf(gate_x_1_58, swiglu_limit);
                            gate_y_1_58 = fminf(gate_y_1_58, swiglu_limit);
                            up_x_1_58 = fminf(fmaxf(up_x_1_58, -swiglu_limit), swiglu_limit);
                            up_y_1_58 = fminf(fmaxf(up_y_1_58, -swiglu_limit), swiglu_limit);
                            float _exp_303 = expf(gate_x_1_58 * -1.0f);
                            float denominator_x_1_58 = _exp_303 + 1.0f;
                            float _exp_304 = expf(gate_y_1_58 * -1.0f);
                            float denominator_y_1_58 = _exp_304 + 1.0f;
                            float hidden_x_1_58 = gate_x_1_58 / denominator_x_1_58 * up_x_1_58;
                            float hidden_y_1_58 = gate_y_1_58 / denominator_y_1_58 * up_y_1_58;
                            hidden_packed_1[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_58, hidden_y_1_58)));
                            float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[27]));
                            float2 _cvt_f32_256_59 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[27]));
                            float gate_x_1_59 = _cvt_f32_59.x;
                            float gate_y_1_59 = _cvt_f32_59.y;
                            float up_x_1_59 = _cvt_f32_256_59.x;
                            float up_y_1_59 = _cvt_f32_256_59.y;
                            gate_x_1_59 = fminf(gate_x_1_59, swiglu_limit);
                            gate_y_1_59 = fminf(gate_y_1_59, swiglu_limit);
                            up_x_1_59 = fminf(fmaxf(up_x_1_59, -swiglu_limit), swiglu_limit);
                            up_y_1_59 = fminf(fmaxf(up_y_1_59, -swiglu_limit), swiglu_limit);
                            float _exp_305 = expf(gate_x_1_59 * -1.0f);
                            float denominator_x_1_59 = _exp_305 + 1.0f;
                            float _exp_306 = expf(gate_y_1_59 * -1.0f);
                            float denominator_y_1_59 = _exp_306 + 1.0f;
                            float hidden_x_1_59 = gate_x_1_59 / denominator_x_1_59 * up_x_1_59;
                            float hidden_y_1_59 = gate_y_1_59 / denominator_y_1_59 * up_y_1_59;
                            hidden_packed_1[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_59, hidden_y_1_59)));
                            float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[28]));
                            float2 _cvt_f32_256_60 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[28]));
                            float gate_x_1_60 = _cvt_f32_60.x;
                            float gate_y_1_60 = _cvt_f32_60.y;
                            float up_x_1_60 = _cvt_f32_256_60.x;
                            float up_y_1_60 = _cvt_f32_256_60.y;
                            gate_x_1_60 = fminf(gate_x_1_60, swiglu_limit);
                            gate_y_1_60 = fminf(gate_y_1_60, swiglu_limit);
                            up_x_1_60 = fminf(fmaxf(up_x_1_60, -swiglu_limit), swiglu_limit);
                            up_y_1_60 = fminf(fmaxf(up_y_1_60, -swiglu_limit), swiglu_limit);
                            float _exp_307 = expf(gate_x_1_60 * -1.0f);
                            float denominator_x_1_60 = _exp_307 + 1.0f;
                            float _exp_308 = expf(gate_y_1_60 * -1.0f);
                            float denominator_y_1_60 = _exp_308 + 1.0f;
                            float hidden_x_1_60 = gate_x_1_60 / denominator_x_1_60 * up_x_1_60;
                            float hidden_y_1_60 = gate_y_1_60 / denominator_y_1_60 * up_y_1_60;
                            hidden_packed_1[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_60, hidden_y_1_60)));
                            float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[29]));
                            float2 _cvt_f32_256_61 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[29]));
                            float gate_x_1_61 = _cvt_f32_61.x;
                            float gate_y_1_61 = _cvt_f32_61.y;
                            float up_x_1_61 = _cvt_f32_256_61.x;
                            float up_y_1_61 = _cvt_f32_256_61.y;
                            gate_x_1_61 = fminf(gate_x_1_61, swiglu_limit);
                            gate_y_1_61 = fminf(gate_y_1_61, swiglu_limit);
                            up_x_1_61 = fminf(fmaxf(up_x_1_61, -swiglu_limit), swiglu_limit);
                            up_y_1_61 = fminf(fmaxf(up_y_1_61, -swiglu_limit), swiglu_limit);
                            float _exp_309 = expf(gate_x_1_61 * -1.0f);
                            float denominator_x_1_61 = _exp_309 + 1.0f;
                            float _exp_310 = expf(gate_y_1_61 * -1.0f);
                            float denominator_y_1_61 = _exp_310 + 1.0f;
                            float hidden_x_1_61 = gate_x_1_61 / denominator_x_1_61 * up_x_1_61;
                            float hidden_y_1_61 = gate_y_1_61 / denominator_y_1_61 * up_y_1_61;
                            hidden_packed_1[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_61, hidden_y_1_61)));
                            float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[30]));
                            float2 _cvt_f32_256_62 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[30]));
                            float gate_x_1_62 = _cvt_f32_62.x;
                            float gate_y_1_62 = _cvt_f32_62.y;
                            float up_x_1_62 = _cvt_f32_256_62.x;
                            float up_y_1_62 = _cvt_f32_256_62.y;
                            gate_x_1_62 = fminf(gate_x_1_62, swiglu_limit);
                            gate_y_1_62 = fminf(gate_y_1_62, swiglu_limit);
                            up_x_1_62 = fminf(fmaxf(up_x_1_62, -swiglu_limit), swiglu_limit);
                            up_y_1_62 = fminf(fmaxf(up_y_1_62, -swiglu_limit), swiglu_limit);
                            float _exp_311 = expf(gate_x_1_62 * -1.0f);
                            float denominator_x_1_62 = _exp_311 + 1.0f;
                            float _exp_312 = expf(gate_y_1_62 * -1.0f);
                            float denominator_y_1_62 = _exp_312 + 1.0f;
                            float hidden_x_1_62 = gate_x_1_62 / denominator_x_1_62 * up_x_1_62;
                            float hidden_y_1_62 = gate_y_1_62 / denominator_y_1_62 * up_y_1_62;
                            hidden_packed_1[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_62, hidden_y_1_62)));
                            float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[31]));
                            float2 _cvt_f32_256_63 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[31]));
                            float gate_x_1_63 = _cvt_f32_63.x;
                            float gate_y_1_63 = _cvt_f32_63.y;
                            float up_x_1_63 = _cvt_f32_256_63.x;
                            float up_y_1_63 = _cvt_f32_256_63.y;
                            gate_x_1_63 = fminf(gate_x_1_63, swiglu_limit);
                            gate_y_1_63 = fminf(gate_y_1_63, swiglu_limit);
                            up_x_1_63 = fminf(fmaxf(up_x_1_63, -swiglu_limit), swiglu_limit);
                            up_y_1_63 = fminf(fmaxf(up_y_1_63, -swiglu_limit), swiglu_limit);
                            float _exp_313 = expf(gate_x_1_63 * -1.0f);
                            float denominator_x_1_63 = _exp_313 + 1.0f;
                            float _exp_314 = expf(gate_y_1_63 * -1.0f);
                            float denominator_y_1_63 = _exp_314 + 1.0f;
                            float hidden_x_1_63 = gate_x_1_63 / denominator_x_1_63 * up_x_1_63;
                            float hidden_y_1_63 = gate_y_1_63 / denominator_y_1_63 * up_y_1_63;
                            hidden_packed_1[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_63, hidden_y_1_63)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_5 = tid / 32;
                            int lane_5 = tid % 32;
                            #pragma unroll
                            for (int half_14 = 0; half_14 < 2; half_14++) {
                                #pragma unroll
                                for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                    int row_15 = warp_5 * 32 + half_14 * 16 + lane_5 % 16;
                                    int col_15 = col_tile_13 * 16 + lane_5 / 16 * 8;
                                    unsigned int address_6 = d_smem_addr + (unsigned int)((row_15 * 32 + col_15) * 2);
                                    address_6 = address_6 ^ (address_6 & 511) >> 7 << 4;
                                    int offset_6 = half_14 * 8 + col_tile_13 * 4;
                                    uint32_t _stmatrix_addr_315 = static_cast<uint32_t>(address_6);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_315), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_6 + 3]))
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
                            int warp_6 = tid / 32;
                            int lane_6 = tid % 32;
                            #pragma unroll
                            for (int half_15 = 0; half_15 < 2; half_15++) {
                                #pragma unroll
                                for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                    int row_16 = warp_6 * 32 + half_15 * 16 + lane_6 % 16;
                                    int col_16 = col_tile_14 * 16 + lane_6 / 16 * 8;
                                    unsigned int address_7 = d_smem_addr + 8192 + (unsigned int)((row_16 * 32 + col_16) * 2);
                                    address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                    int offset_7 = half_15 * 8 + col_tile_14 * 4;
                                    uint32_t _stmatrix_addr_316 = static_cast<uint32_t>(address_7);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_316), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_7 + 3]))
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
                            int warp_7 = tid / 32;
                            int lane_7 = tid % 32;
                            #pragma unroll
                            for (int half_16 = 0; half_16 < 2; half_16++) {
                                #pragma unroll
                                for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                    int row_17 = warp_7 * 32 + half_16 * 16 + lane_7 % 16;
                                    int col_17 = col_tile_15 * 16 + lane_7 / 16 * 8;
                                    unsigned int address_8 = d_smem_addr + 16384 + (unsigned int)((row_17 * 32 + col_17) * 2);
                                    address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                    int offset_8 = half_16 * 8 + col_tile_15 * 4;
                                    uint32_t _stmatrix_addr_317 = static_cast<uint32_t>(address_8);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_317), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_8 + 3]))
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
                            int warp_8 = tid / 32;
                            int lane_8 = tid % 32;
                            #pragma unroll
                            for (int half_17 = 0; half_17 < 2; half_17++) {
                                #pragma unroll
                                for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                    int row_18 = warp_8 * 32 + half_17 * 16 + lane_8 % 16;
                                    int col_18 = col_tile_16 * 16 + lane_8 / 16 * 8;
                                    unsigned int address_9 = d_smem_addr + (unsigned int)((row_18 * 32 + col_18) * 2);
                                    address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                    int offset_9 = 16 + half_17 * 8 + col_tile_16 * 4;
                                    uint32_t _stmatrix_addr_318 = static_cast<uint32_t>(address_9);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_318), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_9 + 3]))
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
                            int warp_9 = tid / 32;
                            int lane_9 = tid % 32;
                            #pragma unroll
                            for (int half_18 = 0; half_18 < 2; half_18++) {
                                #pragma unroll
                                for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                    int row_19 = warp_9 * 32 + half_18 * 16 + lane_9 % 16;
                                    int col_19 = col_tile_17 * 16 + lane_9 / 16 * 8;
                                    unsigned int address_10 = d_smem_addr + 8192 + (unsigned int)((row_19 * 32 + col_19) * 2);
                                    address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                    int offset_10 = 16 + half_18 * 8 + col_tile_17 * 4;
                                    uint32_t _stmatrix_addr_319 = static_cast<uint32_t>(address_10);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_319), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_10 + 3]))
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
                            int warp_10 = tid / 32;
                            int lane_10 = tid % 32;
                            #pragma unroll
                            for (int half_19 = 0; half_19 < 2; half_19++) {
                                #pragma unroll
                                for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                    int row_20 = warp_10 * 32 + half_19 * 16 + lane_10 % 16;
                                    int col_20 = col_tile_18 * 16 + lane_10 / 16 * 8;
                                    unsigned int address_11 = d_smem_addr + 16384 + (unsigned int)((row_20 * 32 + col_20) * 2);
                                    address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                    int offset_11 = 16 + half_19 * 8 + col_tile_18 * 4;
                                    uint32_t _stmatrix_addr_320 = static_cast<uint32_t>(address_11);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_320), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_11 + 3]))
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
                            unsigned int gate_packed_2[32];
                            unsigned int up_packed_2[32];
                            unsigned int hidden_packed_2[32];
                            unsigned int address_3_8 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 128;
                            float _tmem_load_8[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                                : "r"(address_3_8));
                            float _tmem_load_32_8[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_8[15]))
                                : "r"(address_3_8 + 256));
                            gate_packed_2[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[0], _tmem_load_8[1])));
                            up_packed_2[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[0], _tmem_load_32_8[1])));
                            gate_packed_2[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[2], _tmem_load_8[3])));
                            up_packed_2[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[2], _tmem_load_32_8[3])));
                            gate_packed_2[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[4], _tmem_load_8[5])));
                            up_packed_2[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[4], _tmem_load_32_8[5])));
                            gate_packed_2[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[6], _tmem_load_8[7])));
                            up_packed_2[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[6], _tmem_load_32_8[7])));
                            gate_packed_2[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[8], _tmem_load_8[9])));
                            up_packed_2[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[8], _tmem_load_32_8[9])));
                            gate_packed_2[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[10], _tmem_load_8[11])));
                            up_packed_2[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[10], _tmem_load_32_8[11])));
                            gate_packed_2[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[12], _tmem_load_8[13])));
                            up_packed_2[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[12], _tmem_load_32_8[13])));
                            gate_packed_2[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_8[14], _tmem_load_8[15])));
                            up_packed_2[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_8[14], _tmem_load_32_8[15])));
                            unsigned int address_3_9 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 128;
                            float _tmem_load_9[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                                : "r"(address_3_9));
                            float _tmem_load_32_9[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_9[15]))
                                : "r"(address_3_9 + 256));
                            gate_packed_2[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[0], _tmem_load_9[1])));
                            up_packed_2[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[0], _tmem_load_32_9[1])));
                            gate_packed_2[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[2], _tmem_load_9[3])));
                            up_packed_2[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[2], _tmem_load_32_9[3])));
                            gate_packed_2[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[4], _tmem_load_9[5])));
                            up_packed_2[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[4], _tmem_load_32_9[5])));
                            gate_packed_2[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[6], _tmem_load_9[7])));
                            up_packed_2[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[6], _tmem_load_32_9[7])));
                            gate_packed_2[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[8], _tmem_load_9[9])));
                            up_packed_2[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[8], _tmem_load_32_9[9])));
                            gate_packed_2[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[10], _tmem_load_9[11])));
                            up_packed_2[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[10], _tmem_load_32_9[11])));
                            gate_packed_2[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[12], _tmem_load_9[13])));
                            up_packed_2[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[12], _tmem_load_32_9[13])));
                            gate_packed_2[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_9[14], _tmem_load_9[15])));
                            up_packed_2[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_9[14], _tmem_load_32_9[15])));
                            unsigned int address_3_10 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 160;
                            float _tmem_load_10[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                                : "r"(address_3_10));
                            float _tmem_load_32_10[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_10[15]))
                                : "r"(address_3_10 + 256));
                            gate_packed_2[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[0], _tmem_load_10[1])));
                            up_packed_2[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[0], _tmem_load_32_10[1])));
                            gate_packed_2[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[2], _tmem_load_10[3])));
                            up_packed_2[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[2], _tmem_load_32_10[3])));
                            gate_packed_2[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[4], _tmem_load_10[5])));
                            up_packed_2[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[4], _tmem_load_32_10[5])));
                            gate_packed_2[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[6], _tmem_load_10[7])));
                            up_packed_2[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[6], _tmem_load_32_10[7])));
                            gate_packed_2[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[8], _tmem_load_10[9])));
                            up_packed_2[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[8], _tmem_load_32_10[9])));
                            gate_packed_2[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[10], _tmem_load_10[11])));
                            up_packed_2[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[10], _tmem_load_32_10[11])));
                            gate_packed_2[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[12], _tmem_load_10[13])));
                            up_packed_2[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[12], _tmem_load_32_10[13])));
                            gate_packed_2[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_10[14], _tmem_load_10[15])));
                            up_packed_2[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_10[14], _tmem_load_32_10[15])));
                            unsigned int address_3_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 160;
                            float _tmem_load_11[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                                : "r"(address_3_11));
                            float _tmem_load_32_11[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_11[15]))
                                : "r"(address_3_11 + 256));
                            gate_packed_2[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[0], _tmem_load_11[1])));
                            up_packed_2[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[0], _tmem_load_32_11[1])));
                            gate_packed_2[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[2], _tmem_load_11[3])));
                            up_packed_2[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[2], _tmem_load_32_11[3])));
                            gate_packed_2[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[4], _tmem_load_11[5])));
                            up_packed_2[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[4], _tmem_load_32_11[5])));
                            gate_packed_2[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[6], _tmem_load_11[7])));
                            up_packed_2[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[6], _tmem_load_32_11[7])));
                            gate_packed_2[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[8], _tmem_load_11[9])));
                            up_packed_2[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[8], _tmem_load_32_11[9])));
                            gate_packed_2[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[10], _tmem_load_11[11])));
                            up_packed_2[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[10], _tmem_load_32_11[11])));
                            gate_packed_2[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[12], _tmem_load_11[13])));
                            up_packed_2[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[12], _tmem_load_32_11[13])));
                            gate_packed_2[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_11[14], _tmem_load_11[15])));
                            up_packed_2[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_11[14], _tmem_load_32_11[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[0]));
                            float2 _cvt_f32_256_64 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[0]));
                            float gate_x_1_64 = _cvt_f32_64.x;
                            float gate_y_1_64 = _cvt_f32_64.y;
                            float up_x_1_64 = _cvt_f32_256_64.x;
                            float up_y_1_64 = _cvt_f32_256_64.y;
                            gate_x_1_64 = fminf(gate_x_1_64, swiglu_limit);
                            gate_y_1_64 = fminf(gate_y_1_64, swiglu_limit);
                            up_x_1_64 = fminf(fmaxf(up_x_1_64, -swiglu_limit), swiglu_limit);
                            up_y_1_64 = fminf(fmaxf(up_y_1_64, -swiglu_limit), swiglu_limit);
                            float _exp_321 = expf(gate_x_1_64 * -1.0f);
                            float denominator_x_1_64 = _exp_321 + 1.0f;
                            float _exp_322 = expf(gate_y_1_64 * -1.0f);
                            float denominator_y_1_64 = _exp_322 + 1.0f;
                            float hidden_x_1_64 = gate_x_1_64 / denominator_x_1_64 * up_x_1_64;
                            float hidden_y_1_64 = gate_y_1_64 / denominator_y_1_64 * up_y_1_64;
                            hidden_packed_2[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_64, hidden_y_1_64)));
                            float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[1]));
                            float2 _cvt_f32_256_65 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[1]));
                            float gate_x_1_65 = _cvt_f32_65.x;
                            float gate_y_1_65 = _cvt_f32_65.y;
                            float up_x_1_65 = _cvt_f32_256_65.x;
                            float up_y_1_65 = _cvt_f32_256_65.y;
                            gate_x_1_65 = fminf(gate_x_1_65, swiglu_limit);
                            gate_y_1_65 = fminf(gate_y_1_65, swiglu_limit);
                            up_x_1_65 = fminf(fmaxf(up_x_1_65, -swiglu_limit), swiglu_limit);
                            up_y_1_65 = fminf(fmaxf(up_y_1_65, -swiglu_limit), swiglu_limit);
                            float _exp_323 = expf(gate_x_1_65 * -1.0f);
                            float denominator_x_1_65 = _exp_323 + 1.0f;
                            float _exp_324 = expf(gate_y_1_65 * -1.0f);
                            float denominator_y_1_65 = _exp_324 + 1.0f;
                            float hidden_x_1_65 = gate_x_1_65 / denominator_x_1_65 * up_x_1_65;
                            float hidden_y_1_65 = gate_y_1_65 / denominator_y_1_65 * up_y_1_65;
                            hidden_packed_2[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_65, hidden_y_1_65)));
                            float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[2]));
                            float2 _cvt_f32_256_66 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[2]));
                            float gate_x_1_66 = _cvt_f32_66.x;
                            float gate_y_1_66 = _cvt_f32_66.y;
                            float up_x_1_66 = _cvt_f32_256_66.x;
                            float up_y_1_66 = _cvt_f32_256_66.y;
                            gate_x_1_66 = fminf(gate_x_1_66, swiglu_limit);
                            gate_y_1_66 = fminf(gate_y_1_66, swiglu_limit);
                            up_x_1_66 = fminf(fmaxf(up_x_1_66, -swiglu_limit), swiglu_limit);
                            up_y_1_66 = fminf(fmaxf(up_y_1_66, -swiglu_limit), swiglu_limit);
                            float _exp_325 = expf(gate_x_1_66 * -1.0f);
                            float denominator_x_1_66 = _exp_325 + 1.0f;
                            float _exp_326 = expf(gate_y_1_66 * -1.0f);
                            float denominator_y_1_66 = _exp_326 + 1.0f;
                            float hidden_x_1_66 = gate_x_1_66 / denominator_x_1_66 * up_x_1_66;
                            float hidden_y_1_66 = gate_y_1_66 / denominator_y_1_66 * up_y_1_66;
                            hidden_packed_2[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_66, hidden_y_1_66)));
                            float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[3]));
                            float2 _cvt_f32_256_67 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[3]));
                            float gate_x_1_67 = _cvt_f32_67.x;
                            float gate_y_1_67 = _cvt_f32_67.y;
                            float up_x_1_67 = _cvt_f32_256_67.x;
                            float up_y_1_67 = _cvt_f32_256_67.y;
                            gate_x_1_67 = fminf(gate_x_1_67, swiglu_limit);
                            gate_y_1_67 = fminf(gate_y_1_67, swiglu_limit);
                            up_x_1_67 = fminf(fmaxf(up_x_1_67, -swiglu_limit), swiglu_limit);
                            up_y_1_67 = fminf(fmaxf(up_y_1_67, -swiglu_limit), swiglu_limit);
                            float _exp_327 = expf(gate_x_1_67 * -1.0f);
                            float denominator_x_1_67 = _exp_327 + 1.0f;
                            float _exp_328 = expf(gate_y_1_67 * -1.0f);
                            float denominator_y_1_67 = _exp_328 + 1.0f;
                            float hidden_x_1_67 = gate_x_1_67 / denominator_x_1_67 * up_x_1_67;
                            float hidden_y_1_67 = gate_y_1_67 / denominator_y_1_67 * up_y_1_67;
                            hidden_packed_2[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_67, hidden_y_1_67)));
                            float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[4]));
                            float2 _cvt_f32_256_68 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[4]));
                            float gate_x_1_68 = _cvt_f32_68.x;
                            float gate_y_1_68 = _cvt_f32_68.y;
                            float up_x_1_68 = _cvt_f32_256_68.x;
                            float up_y_1_68 = _cvt_f32_256_68.y;
                            gate_x_1_68 = fminf(gate_x_1_68, swiglu_limit);
                            gate_y_1_68 = fminf(gate_y_1_68, swiglu_limit);
                            up_x_1_68 = fminf(fmaxf(up_x_1_68, -swiglu_limit), swiglu_limit);
                            up_y_1_68 = fminf(fmaxf(up_y_1_68, -swiglu_limit), swiglu_limit);
                            float _exp_329 = expf(gate_x_1_68 * -1.0f);
                            float denominator_x_1_68 = _exp_329 + 1.0f;
                            float _exp_330 = expf(gate_y_1_68 * -1.0f);
                            float denominator_y_1_68 = _exp_330 + 1.0f;
                            float hidden_x_1_68 = gate_x_1_68 / denominator_x_1_68 * up_x_1_68;
                            float hidden_y_1_68 = gate_y_1_68 / denominator_y_1_68 * up_y_1_68;
                            hidden_packed_2[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_68, hidden_y_1_68)));
                            float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[5]));
                            float2 _cvt_f32_256_69 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[5]));
                            float gate_x_1_69 = _cvt_f32_69.x;
                            float gate_y_1_69 = _cvt_f32_69.y;
                            float up_x_1_69 = _cvt_f32_256_69.x;
                            float up_y_1_69 = _cvt_f32_256_69.y;
                            gate_x_1_69 = fminf(gate_x_1_69, swiglu_limit);
                            gate_y_1_69 = fminf(gate_y_1_69, swiglu_limit);
                            up_x_1_69 = fminf(fmaxf(up_x_1_69, -swiglu_limit), swiglu_limit);
                            up_y_1_69 = fminf(fmaxf(up_y_1_69, -swiglu_limit), swiglu_limit);
                            float _exp_331 = expf(gate_x_1_69 * -1.0f);
                            float denominator_x_1_69 = _exp_331 + 1.0f;
                            float _exp_332 = expf(gate_y_1_69 * -1.0f);
                            float denominator_y_1_69 = _exp_332 + 1.0f;
                            float hidden_x_1_69 = gate_x_1_69 / denominator_x_1_69 * up_x_1_69;
                            float hidden_y_1_69 = gate_y_1_69 / denominator_y_1_69 * up_y_1_69;
                            hidden_packed_2[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_69, hidden_y_1_69)));
                            float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[6]));
                            float2 _cvt_f32_256_70 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[6]));
                            float gate_x_1_70 = _cvt_f32_70.x;
                            float gate_y_1_70 = _cvt_f32_70.y;
                            float up_x_1_70 = _cvt_f32_256_70.x;
                            float up_y_1_70 = _cvt_f32_256_70.y;
                            gate_x_1_70 = fminf(gate_x_1_70, swiglu_limit);
                            gate_y_1_70 = fminf(gate_y_1_70, swiglu_limit);
                            up_x_1_70 = fminf(fmaxf(up_x_1_70, -swiglu_limit), swiglu_limit);
                            up_y_1_70 = fminf(fmaxf(up_y_1_70, -swiglu_limit), swiglu_limit);
                            float _exp_333 = expf(gate_x_1_70 * -1.0f);
                            float denominator_x_1_70 = _exp_333 + 1.0f;
                            float _exp_334 = expf(gate_y_1_70 * -1.0f);
                            float denominator_y_1_70 = _exp_334 + 1.0f;
                            float hidden_x_1_70 = gate_x_1_70 / denominator_x_1_70 * up_x_1_70;
                            float hidden_y_1_70 = gate_y_1_70 / denominator_y_1_70 * up_y_1_70;
                            hidden_packed_2[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_70, hidden_y_1_70)));
                            float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[7]));
                            float2 _cvt_f32_256_71 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[7]));
                            float gate_x_1_71 = _cvt_f32_71.x;
                            float gate_y_1_71 = _cvt_f32_71.y;
                            float up_x_1_71 = _cvt_f32_256_71.x;
                            float up_y_1_71 = _cvt_f32_256_71.y;
                            gate_x_1_71 = fminf(gate_x_1_71, swiglu_limit);
                            gate_y_1_71 = fminf(gate_y_1_71, swiglu_limit);
                            up_x_1_71 = fminf(fmaxf(up_x_1_71, -swiglu_limit), swiglu_limit);
                            up_y_1_71 = fminf(fmaxf(up_y_1_71, -swiglu_limit), swiglu_limit);
                            float _exp_335 = expf(gate_x_1_71 * -1.0f);
                            float denominator_x_1_71 = _exp_335 + 1.0f;
                            float _exp_336 = expf(gate_y_1_71 * -1.0f);
                            float denominator_y_1_71 = _exp_336 + 1.0f;
                            float hidden_x_1_71 = gate_x_1_71 / denominator_x_1_71 * up_x_1_71;
                            float hidden_y_1_71 = gate_y_1_71 / denominator_y_1_71 * up_y_1_71;
                            hidden_packed_2[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_71, hidden_y_1_71)));
                            float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[8]));
                            float2 _cvt_f32_256_72 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[8]));
                            float gate_x_1_72 = _cvt_f32_72.x;
                            float gate_y_1_72 = _cvt_f32_72.y;
                            float up_x_1_72 = _cvt_f32_256_72.x;
                            float up_y_1_72 = _cvt_f32_256_72.y;
                            gate_x_1_72 = fminf(gate_x_1_72, swiglu_limit);
                            gate_y_1_72 = fminf(gate_y_1_72, swiglu_limit);
                            up_x_1_72 = fminf(fmaxf(up_x_1_72, -swiglu_limit), swiglu_limit);
                            up_y_1_72 = fminf(fmaxf(up_y_1_72, -swiglu_limit), swiglu_limit);
                            float _exp_337 = expf(gate_x_1_72 * -1.0f);
                            float denominator_x_1_72 = _exp_337 + 1.0f;
                            float _exp_338 = expf(gate_y_1_72 * -1.0f);
                            float denominator_y_1_72 = _exp_338 + 1.0f;
                            float hidden_x_1_72 = gate_x_1_72 / denominator_x_1_72 * up_x_1_72;
                            float hidden_y_1_72 = gate_y_1_72 / denominator_y_1_72 * up_y_1_72;
                            hidden_packed_2[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_72, hidden_y_1_72)));
                            float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[9]));
                            float2 _cvt_f32_256_73 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[9]));
                            float gate_x_1_73 = _cvt_f32_73.x;
                            float gate_y_1_73 = _cvt_f32_73.y;
                            float up_x_1_73 = _cvt_f32_256_73.x;
                            float up_y_1_73 = _cvt_f32_256_73.y;
                            gate_x_1_73 = fminf(gate_x_1_73, swiglu_limit);
                            gate_y_1_73 = fminf(gate_y_1_73, swiglu_limit);
                            up_x_1_73 = fminf(fmaxf(up_x_1_73, -swiglu_limit), swiglu_limit);
                            up_y_1_73 = fminf(fmaxf(up_y_1_73, -swiglu_limit), swiglu_limit);
                            float _exp_339 = expf(gate_x_1_73 * -1.0f);
                            float denominator_x_1_73 = _exp_339 + 1.0f;
                            float _exp_340 = expf(gate_y_1_73 * -1.0f);
                            float denominator_y_1_73 = _exp_340 + 1.0f;
                            float hidden_x_1_73 = gate_x_1_73 / denominator_x_1_73 * up_x_1_73;
                            float hidden_y_1_73 = gate_y_1_73 / denominator_y_1_73 * up_y_1_73;
                            hidden_packed_2[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_73, hidden_y_1_73)));
                            float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[10]));
                            float2 _cvt_f32_256_74 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[10]));
                            float gate_x_1_74 = _cvt_f32_74.x;
                            float gate_y_1_74 = _cvt_f32_74.y;
                            float up_x_1_74 = _cvt_f32_256_74.x;
                            float up_y_1_74 = _cvt_f32_256_74.y;
                            gate_x_1_74 = fminf(gate_x_1_74, swiglu_limit);
                            gate_y_1_74 = fminf(gate_y_1_74, swiglu_limit);
                            up_x_1_74 = fminf(fmaxf(up_x_1_74, -swiglu_limit), swiglu_limit);
                            up_y_1_74 = fminf(fmaxf(up_y_1_74, -swiglu_limit), swiglu_limit);
                            float _exp_341 = expf(gate_x_1_74 * -1.0f);
                            float denominator_x_1_74 = _exp_341 + 1.0f;
                            float _exp_342 = expf(gate_y_1_74 * -1.0f);
                            float denominator_y_1_74 = _exp_342 + 1.0f;
                            float hidden_x_1_74 = gate_x_1_74 / denominator_x_1_74 * up_x_1_74;
                            float hidden_y_1_74 = gate_y_1_74 / denominator_y_1_74 * up_y_1_74;
                            hidden_packed_2[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_74, hidden_y_1_74)));
                            float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[11]));
                            float2 _cvt_f32_256_75 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[11]));
                            float gate_x_1_75 = _cvt_f32_75.x;
                            float gate_y_1_75 = _cvt_f32_75.y;
                            float up_x_1_75 = _cvt_f32_256_75.x;
                            float up_y_1_75 = _cvt_f32_256_75.y;
                            gate_x_1_75 = fminf(gate_x_1_75, swiglu_limit);
                            gate_y_1_75 = fminf(gate_y_1_75, swiglu_limit);
                            up_x_1_75 = fminf(fmaxf(up_x_1_75, -swiglu_limit), swiglu_limit);
                            up_y_1_75 = fminf(fmaxf(up_y_1_75, -swiglu_limit), swiglu_limit);
                            float _exp_343 = expf(gate_x_1_75 * -1.0f);
                            float denominator_x_1_75 = _exp_343 + 1.0f;
                            float _exp_344 = expf(gate_y_1_75 * -1.0f);
                            float denominator_y_1_75 = _exp_344 + 1.0f;
                            float hidden_x_1_75 = gate_x_1_75 / denominator_x_1_75 * up_x_1_75;
                            float hidden_y_1_75 = gate_y_1_75 / denominator_y_1_75 * up_y_1_75;
                            hidden_packed_2[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_75, hidden_y_1_75)));
                            float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[12]));
                            float2 _cvt_f32_256_76 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[12]));
                            float gate_x_1_76 = _cvt_f32_76.x;
                            float gate_y_1_76 = _cvt_f32_76.y;
                            float up_x_1_76 = _cvt_f32_256_76.x;
                            float up_y_1_76 = _cvt_f32_256_76.y;
                            gate_x_1_76 = fminf(gate_x_1_76, swiglu_limit);
                            gate_y_1_76 = fminf(gate_y_1_76, swiglu_limit);
                            up_x_1_76 = fminf(fmaxf(up_x_1_76, -swiglu_limit), swiglu_limit);
                            up_y_1_76 = fminf(fmaxf(up_y_1_76, -swiglu_limit), swiglu_limit);
                            float _exp_345 = expf(gate_x_1_76 * -1.0f);
                            float denominator_x_1_76 = _exp_345 + 1.0f;
                            float _exp_346 = expf(gate_y_1_76 * -1.0f);
                            float denominator_y_1_76 = _exp_346 + 1.0f;
                            float hidden_x_1_76 = gate_x_1_76 / denominator_x_1_76 * up_x_1_76;
                            float hidden_y_1_76 = gate_y_1_76 / denominator_y_1_76 * up_y_1_76;
                            hidden_packed_2[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_76, hidden_y_1_76)));
                            float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[13]));
                            float2 _cvt_f32_256_77 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[13]));
                            float gate_x_1_77 = _cvt_f32_77.x;
                            float gate_y_1_77 = _cvt_f32_77.y;
                            float up_x_1_77 = _cvt_f32_256_77.x;
                            float up_y_1_77 = _cvt_f32_256_77.y;
                            gate_x_1_77 = fminf(gate_x_1_77, swiglu_limit);
                            gate_y_1_77 = fminf(gate_y_1_77, swiglu_limit);
                            up_x_1_77 = fminf(fmaxf(up_x_1_77, -swiglu_limit), swiglu_limit);
                            up_y_1_77 = fminf(fmaxf(up_y_1_77, -swiglu_limit), swiglu_limit);
                            float _exp_347 = expf(gate_x_1_77 * -1.0f);
                            float denominator_x_1_77 = _exp_347 + 1.0f;
                            float _exp_348 = expf(gate_y_1_77 * -1.0f);
                            float denominator_y_1_77 = _exp_348 + 1.0f;
                            float hidden_x_1_77 = gate_x_1_77 / denominator_x_1_77 * up_x_1_77;
                            float hidden_y_1_77 = gate_y_1_77 / denominator_y_1_77 * up_y_1_77;
                            hidden_packed_2[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_77, hidden_y_1_77)));
                            float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[14]));
                            float2 _cvt_f32_256_78 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[14]));
                            float gate_x_1_78 = _cvt_f32_78.x;
                            float gate_y_1_78 = _cvt_f32_78.y;
                            float up_x_1_78 = _cvt_f32_256_78.x;
                            float up_y_1_78 = _cvt_f32_256_78.y;
                            gate_x_1_78 = fminf(gate_x_1_78, swiglu_limit);
                            gate_y_1_78 = fminf(gate_y_1_78, swiglu_limit);
                            up_x_1_78 = fminf(fmaxf(up_x_1_78, -swiglu_limit), swiglu_limit);
                            up_y_1_78 = fminf(fmaxf(up_y_1_78, -swiglu_limit), swiglu_limit);
                            float _exp_349 = expf(gate_x_1_78 * -1.0f);
                            float denominator_x_1_78 = _exp_349 + 1.0f;
                            float _exp_350 = expf(gate_y_1_78 * -1.0f);
                            float denominator_y_1_78 = _exp_350 + 1.0f;
                            float hidden_x_1_78 = gate_x_1_78 / denominator_x_1_78 * up_x_1_78;
                            float hidden_y_1_78 = gate_y_1_78 / denominator_y_1_78 * up_y_1_78;
                            hidden_packed_2[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_78, hidden_y_1_78)));
                            float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[15]));
                            float2 _cvt_f32_256_79 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[15]));
                            float gate_x_1_79 = _cvt_f32_79.x;
                            float gate_y_1_79 = _cvt_f32_79.y;
                            float up_x_1_79 = _cvt_f32_256_79.x;
                            float up_y_1_79 = _cvt_f32_256_79.y;
                            gate_x_1_79 = fminf(gate_x_1_79, swiglu_limit);
                            gate_y_1_79 = fminf(gate_y_1_79, swiglu_limit);
                            up_x_1_79 = fminf(fmaxf(up_x_1_79, -swiglu_limit), swiglu_limit);
                            up_y_1_79 = fminf(fmaxf(up_y_1_79, -swiglu_limit), swiglu_limit);
                            float _exp_351 = expf(gate_x_1_79 * -1.0f);
                            float denominator_x_1_79 = _exp_351 + 1.0f;
                            float _exp_352 = expf(gate_y_1_79 * -1.0f);
                            float denominator_y_1_79 = _exp_352 + 1.0f;
                            float hidden_x_1_79 = gate_x_1_79 / denominator_x_1_79 * up_x_1_79;
                            float hidden_y_1_79 = gate_y_1_79 / denominator_y_1_79 * up_y_1_79;
                            hidden_packed_2[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_79, hidden_y_1_79)));
                            float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[16]));
                            float2 _cvt_f32_256_80 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[16]));
                            float gate_x_1_80 = _cvt_f32_80.x;
                            float gate_y_1_80 = _cvt_f32_80.y;
                            float up_x_1_80 = _cvt_f32_256_80.x;
                            float up_y_1_80 = _cvt_f32_256_80.y;
                            gate_x_1_80 = fminf(gate_x_1_80, swiglu_limit);
                            gate_y_1_80 = fminf(gate_y_1_80, swiglu_limit);
                            up_x_1_80 = fminf(fmaxf(up_x_1_80, -swiglu_limit), swiglu_limit);
                            up_y_1_80 = fminf(fmaxf(up_y_1_80, -swiglu_limit), swiglu_limit);
                            float _exp_353 = expf(gate_x_1_80 * -1.0f);
                            float denominator_x_1_80 = _exp_353 + 1.0f;
                            float _exp_354 = expf(gate_y_1_80 * -1.0f);
                            float denominator_y_1_80 = _exp_354 + 1.0f;
                            float hidden_x_1_80 = gate_x_1_80 / denominator_x_1_80 * up_x_1_80;
                            float hidden_y_1_80 = gate_y_1_80 / denominator_y_1_80 * up_y_1_80;
                            hidden_packed_2[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_80, hidden_y_1_80)));
                            float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[17]));
                            float2 _cvt_f32_256_81 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[17]));
                            float gate_x_1_81 = _cvt_f32_81.x;
                            float gate_y_1_81 = _cvt_f32_81.y;
                            float up_x_1_81 = _cvt_f32_256_81.x;
                            float up_y_1_81 = _cvt_f32_256_81.y;
                            gate_x_1_81 = fminf(gate_x_1_81, swiglu_limit);
                            gate_y_1_81 = fminf(gate_y_1_81, swiglu_limit);
                            up_x_1_81 = fminf(fmaxf(up_x_1_81, -swiglu_limit), swiglu_limit);
                            up_y_1_81 = fminf(fmaxf(up_y_1_81, -swiglu_limit), swiglu_limit);
                            float _exp_355 = expf(gate_x_1_81 * -1.0f);
                            float denominator_x_1_81 = _exp_355 + 1.0f;
                            float _exp_356 = expf(gate_y_1_81 * -1.0f);
                            float denominator_y_1_81 = _exp_356 + 1.0f;
                            float hidden_x_1_81 = gate_x_1_81 / denominator_x_1_81 * up_x_1_81;
                            float hidden_y_1_81 = gate_y_1_81 / denominator_y_1_81 * up_y_1_81;
                            hidden_packed_2[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_81, hidden_y_1_81)));
                            float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[18]));
                            float2 _cvt_f32_256_82 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[18]));
                            float gate_x_1_82 = _cvt_f32_82.x;
                            float gate_y_1_82 = _cvt_f32_82.y;
                            float up_x_1_82 = _cvt_f32_256_82.x;
                            float up_y_1_82 = _cvt_f32_256_82.y;
                            gate_x_1_82 = fminf(gate_x_1_82, swiglu_limit);
                            gate_y_1_82 = fminf(gate_y_1_82, swiglu_limit);
                            up_x_1_82 = fminf(fmaxf(up_x_1_82, -swiglu_limit), swiglu_limit);
                            up_y_1_82 = fminf(fmaxf(up_y_1_82, -swiglu_limit), swiglu_limit);
                            float _exp_357 = expf(gate_x_1_82 * -1.0f);
                            float denominator_x_1_82 = _exp_357 + 1.0f;
                            float _exp_358 = expf(gate_y_1_82 * -1.0f);
                            float denominator_y_1_82 = _exp_358 + 1.0f;
                            float hidden_x_1_82 = gate_x_1_82 / denominator_x_1_82 * up_x_1_82;
                            float hidden_y_1_82 = gate_y_1_82 / denominator_y_1_82 * up_y_1_82;
                            hidden_packed_2[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_82, hidden_y_1_82)));
                            float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[19]));
                            float2 _cvt_f32_256_83 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[19]));
                            float gate_x_1_83 = _cvt_f32_83.x;
                            float gate_y_1_83 = _cvt_f32_83.y;
                            float up_x_1_83 = _cvt_f32_256_83.x;
                            float up_y_1_83 = _cvt_f32_256_83.y;
                            gate_x_1_83 = fminf(gate_x_1_83, swiglu_limit);
                            gate_y_1_83 = fminf(gate_y_1_83, swiglu_limit);
                            up_x_1_83 = fminf(fmaxf(up_x_1_83, -swiglu_limit), swiglu_limit);
                            up_y_1_83 = fminf(fmaxf(up_y_1_83, -swiglu_limit), swiglu_limit);
                            float _exp_359 = expf(gate_x_1_83 * -1.0f);
                            float denominator_x_1_83 = _exp_359 + 1.0f;
                            float _exp_360 = expf(gate_y_1_83 * -1.0f);
                            float denominator_y_1_83 = _exp_360 + 1.0f;
                            float hidden_x_1_83 = gate_x_1_83 / denominator_x_1_83 * up_x_1_83;
                            float hidden_y_1_83 = gate_y_1_83 / denominator_y_1_83 * up_y_1_83;
                            hidden_packed_2[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_83, hidden_y_1_83)));
                            float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[20]));
                            float2 _cvt_f32_256_84 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[20]));
                            float gate_x_1_84 = _cvt_f32_84.x;
                            float gate_y_1_84 = _cvt_f32_84.y;
                            float up_x_1_84 = _cvt_f32_256_84.x;
                            float up_y_1_84 = _cvt_f32_256_84.y;
                            gate_x_1_84 = fminf(gate_x_1_84, swiglu_limit);
                            gate_y_1_84 = fminf(gate_y_1_84, swiglu_limit);
                            up_x_1_84 = fminf(fmaxf(up_x_1_84, -swiglu_limit), swiglu_limit);
                            up_y_1_84 = fminf(fmaxf(up_y_1_84, -swiglu_limit), swiglu_limit);
                            float _exp_361 = expf(gate_x_1_84 * -1.0f);
                            float denominator_x_1_84 = _exp_361 + 1.0f;
                            float _exp_362 = expf(gate_y_1_84 * -1.0f);
                            float denominator_y_1_84 = _exp_362 + 1.0f;
                            float hidden_x_1_84 = gate_x_1_84 / denominator_x_1_84 * up_x_1_84;
                            float hidden_y_1_84 = gate_y_1_84 / denominator_y_1_84 * up_y_1_84;
                            hidden_packed_2[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_84, hidden_y_1_84)));
                            float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[21]));
                            float2 _cvt_f32_256_85 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[21]));
                            float gate_x_1_85 = _cvt_f32_85.x;
                            float gate_y_1_85 = _cvt_f32_85.y;
                            float up_x_1_85 = _cvt_f32_256_85.x;
                            float up_y_1_85 = _cvt_f32_256_85.y;
                            gate_x_1_85 = fminf(gate_x_1_85, swiglu_limit);
                            gate_y_1_85 = fminf(gate_y_1_85, swiglu_limit);
                            up_x_1_85 = fminf(fmaxf(up_x_1_85, -swiglu_limit), swiglu_limit);
                            up_y_1_85 = fminf(fmaxf(up_y_1_85, -swiglu_limit), swiglu_limit);
                            float _exp_363 = expf(gate_x_1_85 * -1.0f);
                            float denominator_x_1_85 = _exp_363 + 1.0f;
                            float _exp_364 = expf(gate_y_1_85 * -1.0f);
                            float denominator_y_1_85 = _exp_364 + 1.0f;
                            float hidden_x_1_85 = gate_x_1_85 / denominator_x_1_85 * up_x_1_85;
                            float hidden_y_1_85 = gate_y_1_85 / denominator_y_1_85 * up_y_1_85;
                            hidden_packed_2[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_85, hidden_y_1_85)));
                            float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[22]));
                            float2 _cvt_f32_256_86 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[22]));
                            float gate_x_1_86 = _cvt_f32_86.x;
                            float gate_y_1_86 = _cvt_f32_86.y;
                            float up_x_1_86 = _cvt_f32_256_86.x;
                            float up_y_1_86 = _cvt_f32_256_86.y;
                            gate_x_1_86 = fminf(gate_x_1_86, swiglu_limit);
                            gate_y_1_86 = fminf(gate_y_1_86, swiglu_limit);
                            up_x_1_86 = fminf(fmaxf(up_x_1_86, -swiglu_limit), swiglu_limit);
                            up_y_1_86 = fminf(fmaxf(up_y_1_86, -swiglu_limit), swiglu_limit);
                            float _exp_365 = expf(gate_x_1_86 * -1.0f);
                            float denominator_x_1_86 = _exp_365 + 1.0f;
                            float _exp_366 = expf(gate_y_1_86 * -1.0f);
                            float denominator_y_1_86 = _exp_366 + 1.0f;
                            float hidden_x_1_86 = gate_x_1_86 / denominator_x_1_86 * up_x_1_86;
                            float hidden_y_1_86 = gate_y_1_86 / denominator_y_1_86 * up_y_1_86;
                            hidden_packed_2[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_86, hidden_y_1_86)));
                            float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[23]));
                            float2 _cvt_f32_256_87 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[23]));
                            float gate_x_1_87 = _cvt_f32_87.x;
                            float gate_y_1_87 = _cvt_f32_87.y;
                            float up_x_1_87 = _cvt_f32_256_87.x;
                            float up_y_1_87 = _cvt_f32_256_87.y;
                            gate_x_1_87 = fminf(gate_x_1_87, swiglu_limit);
                            gate_y_1_87 = fminf(gate_y_1_87, swiglu_limit);
                            up_x_1_87 = fminf(fmaxf(up_x_1_87, -swiglu_limit), swiglu_limit);
                            up_y_1_87 = fminf(fmaxf(up_y_1_87, -swiglu_limit), swiglu_limit);
                            float _exp_367 = expf(gate_x_1_87 * -1.0f);
                            float denominator_x_1_87 = _exp_367 + 1.0f;
                            float _exp_368 = expf(gate_y_1_87 * -1.0f);
                            float denominator_y_1_87 = _exp_368 + 1.0f;
                            float hidden_x_1_87 = gate_x_1_87 / denominator_x_1_87 * up_x_1_87;
                            float hidden_y_1_87 = gate_y_1_87 / denominator_y_1_87 * up_y_1_87;
                            hidden_packed_2[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_87, hidden_y_1_87)));
                            float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[24]));
                            float2 _cvt_f32_256_88 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[24]));
                            float gate_x_1_88 = _cvt_f32_88.x;
                            float gate_y_1_88 = _cvt_f32_88.y;
                            float up_x_1_88 = _cvt_f32_256_88.x;
                            float up_y_1_88 = _cvt_f32_256_88.y;
                            gate_x_1_88 = fminf(gate_x_1_88, swiglu_limit);
                            gate_y_1_88 = fminf(gate_y_1_88, swiglu_limit);
                            up_x_1_88 = fminf(fmaxf(up_x_1_88, -swiglu_limit), swiglu_limit);
                            up_y_1_88 = fminf(fmaxf(up_y_1_88, -swiglu_limit), swiglu_limit);
                            float _exp_369 = expf(gate_x_1_88 * -1.0f);
                            float denominator_x_1_88 = _exp_369 + 1.0f;
                            float _exp_370 = expf(gate_y_1_88 * -1.0f);
                            float denominator_y_1_88 = _exp_370 + 1.0f;
                            float hidden_x_1_88 = gate_x_1_88 / denominator_x_1_88 * up_x_1_88;
                            float hidden_y_1_88 = gate_y_1_88 / denominator_y_1_88 * up_y_1_88;
                            hidden_packed_2[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_88, hidden_y_1_88)));
                            float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[25]));
                            float2 _cvt_f32_256_89 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[25]));
                            float gate_x_1_89 = _cvt_f32_89.x;
                            float gate_y_1_89 = _cvt_f32_89.y;
                            float up_x_1_89 = _cvt_f32_256_89.x;
                            float up_y_1_89 = _cvt_f32_256_89.y;
                            gate_x_1_89 = fminf(gate_x_1_89, swiglu_limit);
                            gate_y_1_89 = fminf(gate_y_1_89, swiglu_limit);
                            up_x_1_89 = fminf(fmaxf(up_x_1_89, -swiglu_limit), swiglu_limit);
                            up_y_1_89 = fminf(fmaxf(up_y_1_89, -swiglu_limit), swiglu_limit);
                            float _exp_371 = expf(gate_x_1_89 * -1.0f);
                            float denominator_x_1_89 = _exp_371 + 1.0f;
                            float _exp_372 = expf(gate_y_1_89 * -1.0f);
                            float denominator_y_1_89 = _exp_372 + 1.0f;
                            float hidden_x_1_89 = gate_x_1_89 / denominator_x_1_89 * up_x_1_89;
                            float hidden_y_1_89 = gate_y_1_89 / denominator_y_1_89 * up_y_1_89;
                            hidden_packed_2[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_89, hidden_y_1_89)));
                            float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[26]));
                            float2 _cvt_f32_256_90 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[26]));
                            float gate_x_1_90 = _cvt_f32_90.x;
                            float gate_y_1_90 = _cvt_f32_90.y;
                            float up_x_1_90 = _cvt_f32_256_90.x;
                            float up_y_1_90 = _cvt_f32_256_90.y;
                            gate_x_1_90 = fminf(gate_x_1_90, swiglu_limit);
                            gate_y_1_90 = fminf(gate_y_1_90, swiglu_limit);
                            up_x_1_90 = fminf(fmaxf(up_x_1_90, -swiglu_limit), swiglu_limit);
                            up_y_1_90 = fminf(fmaxf(up_y_1_90, -swiglu_limit), swiglu_limit);
                            float _exp_373 = expf(gate_x_1_90 * -1.0f);
                            float denominator_x_1_90 = _exp_373 + 1.0f;
                            float _exp_374 = expf(gate_y_1_90 * -1.0f);
                            float denominator_y_1_90 = _exp_374 + 1.0f;
                            float hidden_x_1_90 = gate_x_1_90 / denominator_x_1_90 * up_x_1_90;
                            float hidden_y_1_90 = gate_y_1_90 / denominator_y_1_90 * up_y_1_90;
                            hidden_packed_2[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_90, hidden_y_1_90)));
                            float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[27]));
                            float2 _cvt_f32_256_91 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[27]));
                            float gate_x_1_91 = _cvt_f32_91.x;
                            float gate_y_1_91 = _cvt_f32_91.y;
                            float up_x_1_91 = _cvt_f32_256_91.x;
                            float up_y_1_91 = _cvt_f32_256_91.y;
                            gate_x_1_91 = fminf(gate_x_1_91, swiglu_limit);
                            gate_y_1_91 = fminf(gate_y_1_91, swiglu_limit);
                            up_x_1_91 = fminf(fmaxf(up_x_1_91, -swiglu_limit), swiglu_limit);
                            up_y_1_91 = fminf(fmaxf(up_y_1_91, -swiglu_limit), swiglu_limit);
                            float _exp_375 = expf(gate_x_1_91 * -1.0f);
                            float denominator_x_1_91 = _exp_375 + 1.0f;
                            float _exp_376 = expf(gate_y_1_91 * -1.0f);
                            float denominator_y_1_91 = _exp_376 + 1.0f;
                            float hidden_x_1_91 = gate_x_1_91 / denominator_x_1_91 * up_x_1_91;
                            float hidden_y_1_91 = gate_y_1_91 / denominator_y_1_91 * up_y_1_91;
                            hidden_packed_2[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_91, hidden_y_1_91)));
                            float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[28]));
                            float2 _cvt_f32_256_92 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[28]));
                            float gate_x_1_92 = _cvt_f32_92.x;
                            float gate_y_1_92 = _cvt_f32_92.y;
                            float up_x_1_92 = _cvt_f32_256_92.x;
                            float up_y_1_92 = _cvt_f32_256_92.y;
                            gate_x_1_92 = fminf(gate_x_1_92, swiglu_limit);
                            gate_y_1_92 = fminf(gate_y_1_92, swiglu_limit);
                            up_x_1_92 = fminf(fmaxf(up_x_1_92, -swiglu_limit), swiglu_limit);
                            up_y_1_92 = fminf(fmaxf(up_y_1_92, -swiglu_limit), swiglu_limit);
                            float _exp_377 = expf(gate_x_1_92 * -1.0f);
                            float denominator_x_1_92 = _exp_377 + 1.0f;
                            float _exp_378 = expf(gate_y_1_92 * -1.0f);
                            float denominator_y_1_92 = _exp_378 + 1.0f;
                            float hidden_x_1_92 = gate_x_1_92 / denominator_x_1_92 * up_x_1_92;
                            float hidden_y_1_92 = gate_y_1_92 / denominator_y_1_92 * up_y_1_92;
                            hidden_packed_2[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_92, hidden_y_1_92)));
                            float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[29]));
                            float2 _cvt_f32_256_93 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[29]));
                            float gate_x_1_93 = _cvt_f32_93.x;
                            float gate_y_1_93 = _cvt_f32_93.y;
                            float up_x_1_93 = _cvt_f32_256_93.x;
                            float up_y_1_93 = _cvt_f32_256_93.y;
                            gate_x_1_93 = fminf(gate_x_1_93, swiglu_limit);
                            gate_y_1_93 = fminf(gate_y_1_93, swiglu_limit);
                            up_x_1_93 = fminf(fmaxf(up_x_1_93, -swiglu_limit), swiglu_limit);
                            up_y_1_93 = fminf(fmaxf(up_y_1_93, -swiglu_limit), swiglu_limit);
                            float _exp_379 = expf(gate_x_1_93 * -1.0f);
                            float denominator_x_1_93 = _exp_379 + 1.0f;
                            float _exp_380 = expf(gate_y_1_93 * -1.0f);
                            float denominator_y_1_93 = _exp_380 + 1.0f;
                            float hidden_x_1_93 = gate_x_1_93 / denominator_x_1_93 * up_x_1_93;
                            float hidden_y_1_93 = gate_y_1_93 / denominator_y_1_93 * up_y_1_93;
                            hidden_packed_2[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_93, hidden_y_1_93)));
                            float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[30]));
                            float2 _cvt_f32_256_94 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[30]));
                            float gate_x_1_94 = _cvt_f32_94.x;
                            float gate_y_1_94 = _cvt_f32_94.y;
                            float up_x_1_94 = _cvt_f32_256_94.x;
                            float up_y_1_94 = _cvt_f32_256_94.y;
                            gate_x_1_94 = fminf(gate_x_1_94, swiglu_limit);
                            gate_y_1_94 = fminf(gate_y_1_94, swiglu_limit);
                            up_x_1_94 = fminf(fmaxf(up_x_1_94, -swiglu_limit), swiglu_limit);
                            up_y_1_94 = fminf(fmaxf(up_y_1_94, -swiglu_limit), swiglu_limit);
                            float _exp_381 = expf(gate_x_1_94 * -1.0f);
                            float denominator_x_1_94 = _exp_381 + 1.0f;
                            float _exp_382 = expf(gate_y_1_94 * -1.0f);
                            float denominator_y_1_94 = _exp_382 + 1.0f;
                            float hidden_x_1_94 = gate_x_1_94 / denominator_x_1_94 * up_x_1_94;
                            float hidden_y_1_94 = gate_y_1_94 / denominator_y_1_94 * up_y_1_94;
                            hidden_packed_2[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_94, hidden_y_1_94)));
                            float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[31]));
                            float2 _cvt_f32_256_95 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[31]));
                            float gate_x_1_95 = _cvt_f32_95.x;
                            float gate_y_1_95 = _cvt_f32_95.y;
                            float up_x_1_95 = _cvt_f32_256_95.x;
                            float up_y_1_95 = _cvt_f32_256_95.y;
                            gate_x_1_95 = fminf(gate_x_1_95, swiglu_limit);
                            gate_y_1_95 = fminf(gate_y_1_95, swiglu_limit);
                            up_x_1_95 = fminf(fmaxf(up_x_1_95, -swiglu_limit), swiglu_limit);
                            up_y_1_95 = fminf(fmaxf(up_y_1_95, -swiglu_limit), swiglu_limit);
                            float _exp_383 = expf(gate_x_1_95 * -1.0f);
                            float denominator_x_1_95 = _exp_383 + 1.0f;
                            float _exp_384 = expf(gate_y_1_95 * -1.0f);
                            float denominator_y_1_95 = _exp_384 + 1.0f;
                            float hidden_x_1_95 = gate_x_1_95 / denominator_x_1_95 * up_x_1_95;
                            float hidden_y_1_95 = gate_y_1_95 / denominator_y_1_95 * up_y_1_95;
                            hidden_packed_2[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_95, hidden_y_1_95)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_11 = tid / 32;
                            int lane_11 = tid % 32;
                            #pragma unroll
                            for (int half_20 = 0; half_20 < 2; half_20++) {
                                #pragma unroll
                                for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                    int row_21 = warp_11 * 32 + half_20 * 16 + lane_11 % 16;
                                    int col_21 = col_tile_19 * 16 + lane_11 / 16 * 8;
                                    unsigned int address_12 = d_smem_addr + (unsigned int)((row_21 * 32 + col_21) * 2);
                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                    int offset_12 = half_20 * 8 + col_tile_19 * 4;
                                    uint32_t _stmatrix_addr_385 = static_cast<uint32_t>(address_12);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_385), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_12 + 3]))
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
                            int warp_12 = tid / 32;
                            int lane_12 = tid % 32;
                            #pragma unroll
                            for (int half_21 = 0; half_21 < 2; half_21++) {
                                #pragma unroll
                                for (int col_tile_20 = 0; col_tile_20 < 2; col_tile_20++) {
                                    int row_22 = warp_12 * 32 + half_21 * 16 + lane_12 % 16;
                                    int col_22 = col_tile_20 * 16 + lane_12 / 16 * 8;
                                    unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)((row_22 * 32 + col_22) * 2);
                                    address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                    int offset_13 = half_21 * 8 + col_tile_20 * 4;
                                    uint32_t _stmatrix_addr_386 = static_cast<uint32_t>(address_13);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_386), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_13 + 3]))
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
                            int warp_13 = tid / 32;
                            int lane_13 = tid % 32;
                            #pragma unroll
                            for (int half_22 = 0; half_22 < 2; half_22++) {
                                #pragma unroll
                                for (int col_tile_21 = 0; col_tile_21 < 2; col_tile_21++) {
                                    int row_23 = warp_13 * 32 + half_22 * 16 + lane_13 % 16;
                                    int col_23 = col_tile_21 * 16 + lane_13 / 16 * 8;
                                    unsigned int address_14 = d_smem_addr + 16384 + (unsigned int)((row_23 * 32 + col_23) * 2);
                                    address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                    int offset_14 = half_22 * 8 + col_tile_21 * 4;
                                    uint32_t _stmatrix_addr_387 = static_cast<uint32_t>(address_14);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_387), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_14 + 3]))
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
                            int warp_14 = tid / 32;
                            int lane_14 = tid % 32;
                            #pragma unroll
                            for (int half_23 = 0; half_23 < 2; half_23++) {
                                #pragma unroll
                                for (int col_tile_22 = 0; col_tile_22 < 2; col_tile_22++) {
                                    int row_24 = warp_14 * 32 + half_23 * 16 + lane_14 % 16;
                                    int col_24 = col_tile_22 * 16 + lane_14 / 16 * 8;
                                    unsigned int address_15 = d_smem_addr + (unsigned int)((row_24 * 32 + col_24) * 2);
                                    address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                    int offset_15 = 16 + half_23 * 8 + col_tile_22 * 4;
                                    uint32_t _stmatrix_addr_388 = static_cast<uint32_t>(address_15);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_388), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_2[offset_15 + 3]))
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
                            int warp_15 = tid / 32;
                            int lane_15 = tid % 32;
                            #pragma unroll
                            for (int half_24 = 0; half_24 < 2; half_24++) {
                                #pragma unroll
                                for (int col_tile_23 = 0; col_tile_23 < 2; col_tile_23++) {
                                    int row_25 = warp_15 * 32 + half_24 * 16 + lane_15 % 16;
                                    int col_25 = col_tile_23 * 16 + lane_15 / 16 * 8;
                                    unsigned int address_16 = d_smem_addr + 8192 + (unsigned int)((row_25 * 32 + col_25) * 2);
                                    address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                    int offset_16 = 16 + half_24 * 8 + col_tile_23 * 4;
                                    uint32_t _stmatrix_addr_389 = static_cast<uint32_t>(address_16);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_389), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_2[offset_16 + 3]))
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
                            int warp_16 = tid / 32;
                            int lane_16 = tid % 32;
                            #pragma unroll
                            for (int half_25 = 0; half_25 < 2; half_25++) {
                                #pragma unroll
                                for (int col_tile_24 = 0; col_tile_24 < 2; col_tile_24++) {
                                    int row_26 = warp_16 * 32 + half_25 * 16 + lane_16 % 16;
                                    int col_26 = col_tile_24 * 16 + lane_16 / 16 * 8;
                                    unsigned int address_17 = d_smem_addr + 16384 + (unsigned int)((row_26 * 32 + col_26) * 2);
                                    address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                    int offset_17 = 16 + half_25 * 8 + col_tile_24 * 4;
                                    uint32_t _stmatrix_addr_390 = static_cast<uint32_t>(address_17);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_390), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_17 + 3]))
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
                            unsigned int gate_packed_3[32];
                            unsigned int up_packed_3[32];
                            unsigned int hidden_packed_3[32];
                            unsigned int address_3_12 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 192;
                            float _tmem_load_12[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                                : "r"(address_3_12));
                            float _tmem_load_32_12[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_12[15]))
                                : "r"(address_3_12 + 256));
                            gate_packed_3[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[0], _tmem_load_12[1])));
                            up_packed_3[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[0], _tmem_load_32_12[1])));
                            gate_packed_3[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[2], _tmem_load_12[3])));
                            up_packed_3[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[2], _tmem_load_32_12[3])));
                            gate_packed_3[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[4], _tmem_load_12[5])));
                            up_packed_3[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[4], _tmem_load_32_12[5])));
                            gate_packed_3[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[6], _tmem_load_12[7])));
                            up_packed_3[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[6], _tmem_load_32_12[7])));
                            gate_packed_3[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[8], _tmem_load_12[9])));
                            up_packed_3[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[8], _tmem_load_32_12[9])));
                            gate_packed_3[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[10], _tmem_load_12[11])));
                            up_packed_3[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[10], _tmem_load_32_12[11])));
                            gate_packed_3[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[12], _tmem_load_12[13])));
                            up_packed_3[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[12], _tmem_load_32_12[13])));
                            gate_packed_3[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_12[14], _tmem_load_12[15])));
                            up_packed_3[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_12[14], _tmem_load_32_12[15])));
                            unsigned int address_3_13 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 192;
                            float _tmem_load_13[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                                : "r"(address_3_13));
                            float _tmem_load_32_13[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_13[15]))
                                : "r"(address_3_13 + 256));
                            gate_packed_3[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[0], _tmem_load_13[1])));
                            up_packed_3[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[0], _tmem_load_32_13[1])));
                            gate_packed_3[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[2], _tmem_load_13[3])));
                            up_packed_3[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[2], _tmem_load_32_13[3])));
                            gate_packed_3[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[4], _tmem_load_13[5])));
                            up_packed_3[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[4], _tmem_load_32_13[5])));
                            gate_packed_3[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[6], _tmem_load_13[7])));
                            up_packed_3[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[6], _tmem_load_32_13[7])));
                            gate_packed_3[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[8], _tmem_load_13[9])));
                            up_packed_3[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[8], _tmem_load_32_13[9])));
                            gate_packed_3[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[10], _tmem_load_13[11])));
                            up_packed_3[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[10], _tmem_load_32_13[11])));
                            gate_packed_3[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[12], _tmem_load_13[13])));
                            up_packed_3[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[12], _tmem_load_32_13[13])));
                            gate_packed_3[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_13[14], _tmem_load_13[15])));
                            up_packed_3[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_13[14], _tmem_load_32_13[15])));
                            unsigned int address_3_14 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 224;
                            float _tmem_load_14[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                                : "r"(address_3_14));
                            float _tmem_load_32_14[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_14[15]))
                                : "r"(address_3_14 + 256));
                            gate_packed_3[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[0], _tmem_load_14[1])));
                            up_packed_3[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[0], _tmem_load_32_14[1])));
                            gate_packed_3[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[2], _tmem_load_14[3])));
                            up_packed_3[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[2], _tmem_load_32_14[3])));
                            gate_packed_3[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[4], _tmem_load_14[5])));
                            up_packed_3[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[4], _tmem_load_32_14[5])));
                            gate_packed_3[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[6], _tmem_load_14[7])));
                            up_packed_3[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[6], _tmem_load_32_14[7])));
                            gate_packed_3[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[8], _tmem_load_14[9])));
                            up_packed_3[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[8], _tmem_load_32_14[9])));
                            gate_packed_3[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[10], _tmem_load_14[11])));
                            up_packed_3[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[10], _tmem_load_32_14[11])));
                            gate_packed_3[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[12], _tmem_load_14[13])));
                            up_packed_3[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[12], _tmem_load_32_14[13])));
                            gate_packed_3[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_14[14], _tmem_load_14[15])));
                            up_packed_3[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_14[14], _tmem_load_32_14[15])));
                            unsigned int address_3_15 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 224;
                            float _tmem_load_15[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                                : "r"(address_3_15));
                            float _tmem_load_32_15[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_15[15]))
                                : "r"(address_3_15 + 256));
                            gate_packed_3[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[0], _tmem_load_15[1])));
                            up_packed_3[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[0], _tmem_load_32_15[1])));
                            gate_packed_3[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[2], _tmem_load_15[3])));
                            up_packed_3[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[2], _tmem_load_32_15[3])));
                            gate_packed_3[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[4], _tmem_load_15[5])));
                            up_packed_3[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[4], _tmem_load_32_15[5])));
                            gate_packed_3[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[6], _tmem_load_15[7])));
                            up_packed_3[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[6], _tmem_load_32_15[7])));
                            gate_packed_3[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[8], _tmem_load_15[9])));
                            up_packed_3[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[8], _tmem_load_32_15[9])));
                            gate_packed_3[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[10], _tmem_load_15[11])));
                            up_packed_3[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[10], _tmem_load_32_15[11])));
                            gate_packed_3[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[12], _tmem_load_15[13])));
                            up_packed_3[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[12], _tmem_load_32_15[13])));
                            gate_packed_3[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_15[14], _tmem_load_15[15])));
                            up_packed_3[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_15[14], _tmem_load_32_15[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            }
                            float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[0]));
                            float2 _cvt_f32_256_96 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[0]));
                            float gate_x_1_96 = _cvt_f32_96.x;
                            float gate_y_1_96 = _cvt_f32_96.y;
                            float up_x_1_96 = _cvt_f32_256_96.x;
                            float up_y_1_96 = _cvt_f32_256_96.y;
                            gate_x_1_96 = fminf(gate_x_1_96, swiglu_limit);
                            gate_y_1_96 = fminf(gate_y_1_96, swiglu_limit);
                            up_x_1_96 = fminf(fmaxf(up_x_1_96, -swiglu_limit), swiglu_limit);
                            up_y_1_96 = fminf(fmaxf(up_y_1_96, -swiglu_limit), swiglu_limit);
                            float _exp_391 = expf(gate_x_1_96 * -1.0f);
                            float denominator_x_1_96 = _exp_391 + 1.0f;
                            float _exp_392 = expf(gate_y_1_96 * -1.0f);
                            float denominator_y_1_96 = _exp_392 + 1.0f;
                            float hidden_x_1_96 = gate_x_1_96 / denominator_x_1_96 * up_x_1_96;
                            float hidden_y_1_96 = gate_y_1_96 / denominator_y_1_96 * up_y_1_96;
                            hidden_packed_3[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_96, hidden_y_1_96)));
                            float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[1]));
                            float2 _cvt_f32_256_97 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[1]));
                            float gate_x_1_97 = _cvt_f32_97.x;
                            float gate_y_1_97 = _cvt_f32_97.y;
                            float up_x_1_97 = _cvt_f32_256_97.x;
                            float up_y_1_97 = _cvt_f32_256_97.y;
                            gate_x_1_97 = fminf(gate_x_1_97, swiglu_limit);
                            gate_y_1_97 = fminf(gate_y_1_97, swiglu_limit);
                            up_x_1_97 = fminf(fmaxf(up_x_1_97, -swiglu_limit), swiglu_limit);
                            up_y_1_97 = fminf(fmaxf(up_y_1_97, -swiglu_limit), swiglu_limit);
                            float _exp_393 = expf(gate_x_1_97 * -1.0f);
                            float denominator_x_1_97 = _exp_393 + 1.0f;
                            float _exp_394 = expf(gate_y_1_97 * -1.0f);
                            float denominator_y_1_97 = _exp_394 + 1.0f;
                            float hidden_x_1_97 = gate_x_1_97 / denominator_x_1_97 * up_x_1_97;
                            float hidden_y_1_97 = gate_y_1_97 / denominator_y_1_97 * up_y_1_97;
                            hidden_packed_3[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_97, hidden_y_1_97)));
                            float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[2]));
                            float2 _cvt_f32_256_98 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[2]));
                            float gate_x_1_98 = _cvt_f32_98.x;
                            float gate_y_1_98 = _cvt_f32_98.y;
                            float up_x_1_98 = _cvt_f32_256_98.x;
                            float up_y_1_98 = _cvt_f32_256_98.y;
                            gate_x_1_98 = fminf(gate_x_1_98, swiglu_limit);
                            gate_y_1_98 = fminf(gate_y_1_98, swiglu_limit);
                            up_x_1_98 = fminf(fmaxf(up_x_1_98, -swiglu_limit), swiglu_limit);
                            up_y_1_98 = fminf(fmaxf(up_y_1_98, -swiglu_limit), swiglu_limit);
                            float _exp_395 = expf(gate_x_1_98 * -1.0f);
                            float denominator_x_1_98 = _exp_395 + 1.0f;
                            float _exp_396 = expf(gate_y_1_98 * -1.0f);
                            float denominator_y_1_98 = _exp_396 + 1.0f;
                            float hidden_x_1_98 = gate_x_1_98 / denominator_x_1_98 * up_x_1_98;
                            float hidden_y_1_98 = gate_y_1_98 / denominator_y_1_98 * up_y_1_98;
                            hidden_packed_3[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_98, hidden_y_1_98)));
                            float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[3]));
                            float2 _cvt_f32_256_99 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[3]));
                            float gate_x_1_99 = _cvt_f32_99.x;
                            float gate_y_1_99 = _cvt_f32_99.y;
                            float up_x_1_99 = _cvt_f32_256_99.x;
                            float up_y_1_99 = _cvt_f32_256_99.y;
                            gate_x_1_99 = fminf(gate_x_1_99, swiglu_limit);
                            gate_y_1_99 = fminf(gate_y_1_99, swiglu_limit);
                            up_x_1_99 = fminf(fmaxf(up_x_1_99, -swiglu_limit), swiglu_limit);
                            up_y_1_99 = fminf(fmaxf(up_y_1_99, -swiglu_limit), swiglu_limit);
                            float _exp_397 = expf(gate_x_1_99 * -1.0f);
                            float denominator_x_1_99 = _exp_397 + 1.0f;
                            float _exp_398 = expf(gate_y_1_99 * -1.0f);
                            float denominator_y_1_99 = _exp_398 + 1.0f;
                            float hidden_x_1_99 = gate_x_1_99 / denominator_x_1_99 * up_x_1_99;
                            float hidden_y_1_99 = gate_y_1_99 / denominator_y_1_99 * up_y_1_99;
                            hidden_packed_3[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_99, hidden_y_1_99)));
                            float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[4]));
                            float2 _cvt_f32_256_100 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[4]));
                            float gate_x_1_100 = _cvt_f32_100.x;
                            float gate_y_1_100 = _cvt_f32_100.y;
                            float up_x_1_100 = _cvt_f32_256_100.x;
                            float up_y_1_100 = _cvt_f32_256_100.y;
                            gate_x_1_100 = fminf(gate_x_1_100, swiglu_limit);
                            gate_y_1_100 = fminf(gate_y_1_100, swiglu_limit);
                            up_x_1_100 = fminf(fmaxf(up_x_1_100, -swiglu_limit), swiglu_limit);
                            up_y_1_100 = fminf(fmaxf(up_y_1_100, -swiglu_limit), swiglu_limit);
                            float _exp_399 = expf(gate_x_1_100 * -1.0f);
                            float denominator_x_1_100 = _exp_399 + 1.0f;
                            float _exp_400 = expf(gate_y_1_100 * -1.0f);
                            float denominator_y_1_100 = _exp_400 + 1.0f;
                            float hidden_x_1_100 = gate_x_1_100 / denominator_x_1_100 * up_x_1_100;
                            float hidden_y_1_100 = gate_y_1_100 / denominator_y_1_100 * up_y_1_100;
                            hidden_packed_3[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_100, hidden_y_1_100)));
                            float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[5]));
                            float2 _cvt_f32_256_101 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[5]));
                            float gate_x_1_101 = _cvt_f32_101.x;
                            float gate_y_1_101 = _cvt_f32_101.y;
                            float up_x_1_101 = _cvt_f32_256_101.x;
                            float up_y_1_101 = _cvt_f32_256_101.y;
                            gate_x_1_101 = fminf(gate_x_1_101, swiglu_limit);
                            gate_y_1_101 = fminf(gate_y_1_101, swiglu_limit);
                            up_x_1_101 = fminf(fmaxf(up_x_1_101, -swiglu_limit), swiglu_limit);
                            up_y_1_101 = fminf(fmaxf(up_y_1_101, -swiglu_limit), swiglu_limit);
                            float _exp_401 = expf(gate_x_1_101 * -1.0f);
                            float denominator_x_1_101 = _exp_401 + 1.0f;
                            float _exp_402 = expf(gate_y_1_101 * -1.0f);
                            float denominator_y_1_101 = _exp_402 + 1.0f;
                            float hidden_x_1_101 = gate_x_1_101 / denominator_x_1_101 * up_x_1_101;
                            float hidden_y_1_101 = gate_y_1_101 / denominator_y_1_101 * up_y_1_101;
                            hidden_packed_3[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_101, hidden_y_1_101)));
                            float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[6]));
                            float2 _cvt_f32_256_102 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[6]));
                            float gate_x_1_102 = _cvt_f32_102.x;
                            float gate_y_1_102 = _cvt_f32_102.y;
                            float up_x_1_102 = _cvt_f32_256_102.x;
                            float up_y_1_102 = _cvt_f32_256_102.y;
                            gate_x_1_102 = fminf(gate_x_1_102, swiglu_limit);
                            gate_y_1_102 = fminf(gate_y_1_102, swiglu_limit);
                            up_x_1_102 = fminf(fmaxf(up_x_1_102, -swiglu_limit), swiglu_limit);
                            up_y_1_102 = fminf(fmaxf(up_y_1_102, -swiglu_limit), swiglu_limit);
                            float _exp_403 = expf(gate_x_1_102 * -1.0f);
                            float denominator_x_1_102 = _exp_403 + 1.0f;
                            float _exp_404 = expf(gate_y_1_102 * -1.0f);
                            float denominator_y_1_102 = _exp_404 + 1.0f;
                            float hidden_x_1_102 = gate_x_1_102 / denominator_x_1_102 * up_x_1_102;
                            float hidden_y_1_102 = gate_y_1_102 / denominator_y_1_102 * up_y_1_102;
                            hidden_packed_3[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_102, hidden_y_1_102)));
                            float2 _cvt_f32_103 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[7]));
                            float2 _cvt_f32_256_103 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[7]));
                            float gate_x_1_103 = _cvt_f32_103.x;
                            float gate_y_1_103 = _cvt_f32_103.y;
                            float up_x_1_103 = _cvt_f32_256_103.x;
                            float up_y_1_103 = _cvt_f32_256_103.y;
                            gate_x_1_103 = fminf(gate_x_1_103, swiglu_limit);
                            gate_y_1_103 = fminf(gate_y_1_103, swiglu_limit);
                            up_x_1_103 = fminf(fmaxf(up_x_1_103, -swiglu_limit), swiglu_limit);
                            up_y_1_103 = fminf(fmaxf(up_y_1_103, -swiglu_limit), swiglu_limit);
                            float _exp_405 = expf(gate_x_1_103 * -1.0f);
                            float denominator_x_1_103 = _exp_405 + 1.0f;
                            float _exp_406 = expf(gate_y_1_103 * -1.0f);
                            float denominator_y_1_103 = _exp_406 + 1.0f;
                            float hidden_x_1_103 = gate_x_1_103 / denominator_x_1_103 * up_x_1_103;
                            float hidden_y_1_103 = gate_y_1_103 / denominator_y_1_103 * up_y_1_103;
                            hidden_packed_3[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_103, hidden_y_1_103)));
                            float2 _cvt_f32_104 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[8]));
                            float2 _cvt_f32_256_104 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[8]));
                            float gate_x_1_104 = _cvt_f32_104.x;
                            float gate_y_1_104 = _cvt_f32_104.y;
                            float up_x_1_104 = _cvt_f32_256_104.x;
                            float up_y_1_104 = _cvt_f32_256_104.y;
                            gate_x_1_104 = fminf(gate_x_1_104, swiglu_limit);
                            gate_y_1_104 = fminf(gate_y_1_104, swiglu_limit);
                            up_x_1_104 = fminf(fmaxf(up_x_1_104, -swiglu_limit), swiglu_limit);
                            up_y_1_104 = fminf(fmaxf(up_y_1_104, -swiglu_limit), swiglu_limit);
                            float _exp_407 = expf(gate_x_1_104 * -1.0f);
                            float denominator_x_1_104 = _exp_407 + 1.0f;
                            float _exp_408 = expf(gate_y_1_104 * -1.0f);
                            float denominator_y_1_104 = _exp_408 + 1.0f;
                            float hidden_x_1_104 = gate_x_1_104 / denominator_x_1_104 * up_x_1_104;
                            float hidden_y_1_104 = gate_y_1_104 / denominator_y_1_104 * up_y_1_104;
                            hidden_packed_3[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_104, hidden_y_1_104)));
                            float2 _cvt_f32_105 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[9]));
                            float2 _cvt_f32_256_105 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[9]));
                            float gate_x_1_105 = _cvt_f32_105.x;
                            float gate_y_1_105 = _cvt_f32_105.y;
                            float up_x_1_105 = _cvt_f32_256_105.x;
                            float up_y_1_105 = _cvt_f32_256_105.y;
                            gate_x_1_105 = fminf(gate_x_1_105, swiglu_limit);
                            gate_y_1_105 = fminf(gate_y_1_105, swiglu_limit);
                            up_x_1_105 = fminf(fmaxf(up_x_1_105, -swiglu_limit), swiglu_limit);
                            up_y_1_105 = fminf(fmaxf(up_y_1_105, -swiglu_limit), swiglu_limit);
                            float _exp_409 = expf(gate_x_1_105 * -1.0f);
                            float denominator_x_1_105 = _exp_409 + 1.0f;
                            float _exp_410 = expf(gate_y_1_105 * -1.0f);
                            float denominator_y_1_105 = _exp_410 + 1.0f;
                            float hidden_x_1_105 = gate_x_1_105 / denominator_x_1_105 * up_x_1_105;
                            float hidden_y_1_105 = gate_y_1_105 / denominator_y_1_105 * up_y_1_105;
                            hidden_packed_3[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_105, hidden_y_1_105)));
                            float2 _cvt_f32_106 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[10]));
                            float2 _cvt_f32_256_106 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[10]));
                            float gate_x_1_106 = _cvt_f32_106.x;
                            float gate_y_1_106 = _cvt_f32_106.y;
                            float up_x_1_106 = _cvt_f32_256_106.x;
                            float up_y_1_106 = _cvt_f32_256_106.y;
                            gate_x_1_106 = fminf(gate_x_1_106, swiglu_limit);
                            gate_y_1_106 = fminf(gate_y_1_106, swiglu_limit);
                            up_x_1_106 = fminf(fmaxf(up_x_1_106, -swiglu_limit), swiglu_limit);
                            up_y_1_106 = fminf(fmaxf(up_y_1_106, -swiglu_limit), swiglu_limit);
                            float _exp_411 = expf(gate_x_1_106 * -1.0f);
                            float denominator_x_1_106 = _exp_411 + 1.0f;
                            float _exp_412 = expf(gate_y_1_106 * -1.0f);
                            float denominator_y_1_106 = _exp_412 + 1.0f;
                            float hidden_x_1_106 = gate_x_1_106 / denominator_x_1_106 * up_x_1_106;
                            float hidden_y_1_106 = gate_y_1_106 / denominator_y_1_106 * up_y_1_106;
                            hidden_packed_3[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_106, hidden_y_1_106)));
                            float2 _cvt_f32_107 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[11]));
                            float2 _cvt_f32_256_107 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[11]));
                            float gate_x_1_107 = _cvt_f32_107.x;
                            float gate_y_1_107 = _cvt_f32_107.y;
                            float up_x_1_107 = _cvt_f32_256_107.x;
                            float up_y_1_107 = _cvt_f32_256_107.y;
                            gate_x_1_107 = fminf(gate_x_1_107, swiglu_limit);
                            gate_y_1_107 = fminf(gate_y_1_107, swiglu_limit);
                            up_x_1_107 = fminf(fmaxf(up_x_1_107, -swiglu_limit), swiglu_limit);
                            up_y_1_107 = fminf(fmaxf(up_y_1_107, -swiglu_limit), swiglu_limit);
                            float _exp_413 = expf(gate_x_1_107 * -1.0f);
                            float denominator_x_1_107 = _exp_413 + 1.0f;
                            float _exp_414 = expf(gate_y_1_107 * -1.0f);
                            float denominator_y_1_107 = _exp_414 + 1.0f;
                            float hidden_x_1_107 = gate_x_1_107 / denominator_x_1_107 * up_x_1_107;
                            float hidden_y_1_107 = gate_y_1_107 / denominator_y_1_107 * up_y_1_107;
                            hidden_packed_3[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_107, hidden_y_1_107)));
                            float2 _cvt_f32_108 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[12]));
                            float2 _cvt_f32_256_108 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[12]));
                            float gate_x_1_108 = _cvt_f32_108.x;
                            float gate_y_1_108 = _cvt_f32_108.y;
                            float up_x_1_108 = _cvt_f32_256_108.x;
                            float up_y_1_108 = _cvt_f32_256_108.y;
                            gate_x_1_108 = fminf(gate_x_1_108, swiglu_limit);
                            gate_y_1_108 = fminf(gate_y_1_108, swiglu_limit);
                            up_x_1_108 = fminf(fmaxf(up_x_1_108, -swiglu_limit), swiglu_limit);
                            up_y_1_108 = fminf(fmaxf(up_y_1_108, -swiglu_limit), swiglu_limit);
                            float _exp_415 = expf(gate_x_1_108 * -1.0f);
                            float denominator_x_1_108 = _exp_415 + 1.0f;
                            float _exp_416 = expf(gate_y_1_108 * -1.0f);
                            float denominator_y_1_108 = _exp_416 + 1.0f;
                            float hidden_x_1_108 = gate_x_1_108 / denominator_x_1_108 * up_x_1_108;
                            float hidden_y_1_108 = gate_y_1_108 / denominator_y_1_108 * up_y_1_108;
                            hidden_packed_3[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_108, hidden_y_1_108)));
                            float2 _cvt_f32_109 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[13]));
                            float2 _cvt_f32_256_109 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[13]));
                            float gate_x_1_109 = _cvt_f32_109.x;
                            float gate_y_1_109 = _cvt_f32_109.y;
                            float up_x_1_109 = _cvt_f32_256_109.x;
                            float up_y_1_109 = _cvt_f32_256_109.y;
                            gate_x_1_109 = fminf(gate_x_1_109, swiglu_limit);
                            gate_y_1_109 = fminf(gate_y_1_109, swiglu_limit);
                            up_x_1_109 = fminf(fmaxf(up_x_1_109, -swiglu_limit), swiglu_limit);
                            up_y_1_109 = fminf(fmaxf(up_y_1_109, -swiglu_limit), swiglu_limit);
                            float _exp_417 = expf(gate_x_1_109 * -1.0f);
                            float denominator_x_1_109 = _exp_417 + 1.0f;
                            float _exp_418 = expf(gate_y_1_109 * -1.0f);
                            float denominator_y_1_109 = _exp_418 + 1.0f;
                            float hidden_x_1_109 = gate_x_1_109 / denominator_x_1_109 * up_x_1_109;
                            float hidden_y_1_109 = gate_y_1_109 / denominator_y_1_109 * up_y_1_109;
                            hidden_packed_3[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_109, hidden_y_1_109)));
                            float2 _cvt_f32_110 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[14]));
                            float2 _cvt_f32_256_110 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[14]));
                            float gate_x_1_110 = _cvt_f32_110.x;
                            float gate_y_1_110 = _cvt_f32_110.y;
                            float up_x_1_110 = _cvt_f32_256_110.x;
                            float up_y_1_110 = _cvt_f32_256_110.y;
                            gate_x_1_110 = fminf(gate_x_1_110, swiglu_limit);
                            gate_y_1_110 = fminf(gate_y_1_110, swiglu_limit);
                            up_x_1_110 = fminf(fmaxf(up_x_1_110, -swiglu_limit), swiglu_limit);
                            up_y_1_110 = fminf(fmaxf(up_y_1_110, -swiglu_limit), swiglu_limit);
                            float _exp_419 = expf(gate_x_1_110 * -1.0f);
                            float denominator_x_1_110 = _exp_419 + 1.0f;
                            float _exp_420 = expf(gate_y_1_110 * -1.0f);
                            float denominator_y_1_110 = _exp_420 + 1.0f;
                            float hidden_x_1_110 = gate_x_1_110 / denominator_x_1_110 * up_x_1_110;
                            float hidden_y_1_110 = gate_y_1_110 / denominator_y_1_110 * up_y_1_110;
                            hidden_packed_3[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_110, hidden_y_1_110)));
                            float2 _cvt_f32_111 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[15]));
                            float2 _cvt_f32_256_111 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[15]));
                            float gate_x_1_111 = _cvt_f32_111.x;
                            float gate_y_1_111 = _cvt_f32_111.y;
                            float up_x_1_111 = _cvt_f32_256_111.x;
                            float up_y_1_111 = _cvt_f32_256_111.y;
                            gate_x_1_111 = fminf(gate_x_1_111, swiglu_limit);
                            gate_y_1_111 = fminf(gate_y_1_111, swiglu_limit);
                            up_x_1_111 = fminf(fmaxf(up_x_1_111, -swiglu_limit), swiglu_limit);
                            up_y_1_111 = fminf(fmaxf(up_y_1_111, -swiglu_limit), swiglu_limit);
                            float _exp_421 = expf(gate_x_1_111 * -1.0f);
                            float denominator_x_1_111 = _exp_421 + 1.0f;
                            float _exp_422 = expf(gate_y_1_111 * -1.0f);
                            float denominator_y_1_111 = _exp_422 + 1.0f;
                            float hidden_x_1_111 = gate_x_1_111 / denominator_x_1_111 * up_x_1_111;
                            float hidden_y_1_111 = gate_y_1_111 / denominator_y_1_111 * up_y_1_111;
                            hidden_packed_3[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_111, hidden_y_1_111)));
                            float2 _cvt_f32_112 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[16]));
                            float2 _cvt_f32_256_112 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[16]));
                            float gate_x_1_112 = _cvt_f32_112.x;
                            float gate_y_1_112 = _cvt_f32_112.y;
                            float up_x_1_112 = _cvt_f32_256_112.x;
                            float up_y_1_112 = _cvt_f32_256_112.y;
                            gate_x_1_112 = fminf(gate_x_1_112, swiglu_limit);
                            gate_y_1_112 = fminf(gate_y_1_112, swiglu_limit);
                            up_x_1_112 = fminf(fmaxf(up_x_1_112, -swiglu_limit), swiglu_limit);
                            up_y_1_112 = fminf(fmaxf(up_y_1_112, -swiglu_limit), swiglu_limit);
                            float _exp_423 = expf(gate_x_1_112 * -1.0f);
                            float denominator_x_1_112 = _exp_423 + 1.0f;
                            float _exp_424 = expf(gate_y_1_112 * -1.0f);
                            float denominator_y_1_112 = _exp_424 + 1.0f;
                            float hidden_x_1_112 = gate_x_1_112 / denominator_x_1_112 * up_x_1_112;
                            float hidden_y_1_112 = gate_y_1_112 / denominator_y_1_112 * up_y_1_112;
                            hidden_packed_3[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_112, hidden_y_1_112)));
                            float2 _cvt_f32_113 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[17]));
                            float2 _cvt_f32_256_113 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[17]));
                            float gate_x_1_113 = _cvt_f32_113.x;
                            float gate_y_1_113 = _cvt_f32_113.y;
                            float up_x_1_113 = _cvt_f32_256_113.x;
                            float up_y_1_113 = _cvt_f32_256_113.y;
                            gate_x_1_113 = fminf(gate_x_1_113, swiglu_limit);
                            gate_y_1_113 = fminf(gate_y_1_113, swiglu_limit);
                            up_x_1_113 = fminf(fmaxf(up_x_1_113, -swiglu_limit), swiglu_limit);
                            up_y_1_113 = fminf(fmaxf(up_y_1_113, -swiglu_limit), swiglu_limit);
                            float _exp_425 = expf(gate_x_1_113 * -1.0f);
                            float denominator_x_1_113 = _exp_425 + 1.0f;
                            float _exp_426 = expf(gate_y_1_113 * -1.0f);
                            float denominator_y_1_113 = _exp_426 + 1.0f;
                            float hidden_x_1_113 = gate_x_1_113 / denominator_x_1_113 * up_x_1_113;
                            float hidden_y_1_113 = gate_y_1_113 / denominator_y_1_113 * up_y_1_113;
                            hidden_packed_3[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_113, hidden_y_1_113)));
                            float2 _cvt_f32_114 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[18]));
                            float2 _cvt_f32_256_114 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[18]));
                            float gate_x_1_114 = _cvt_f32_114.x;
                            float gate_y_1_114 = _cvt_f32_114.y;
                            float up_x_1_114 = _cvt_f32_256_114.x;
                            float up_y_1_114 = _cvt_f32_256_114.y;
                            gate_x_1_114 = fminf(gate_x_1_114, swiglu_limit);
                            gate_y_1_114 = fminf(gate_y_1_114, swiglu_limit);
                            up_x_1_114 = fminf(fmaxf(up_x_1_114, -swiglu_limit), swiglu_limit);
                            up_y_1_114 = fminf(fmaxf(up_y_1_114, -swiglu_limit), swiglu_limit);
                            float _exp_427 = expf(gate_x_1_114 * -1.0f);
                            float denominator_x_1_114 = _exp_427 + 1.0f;
                            float _exp_428 = expf(gate_y_1_114 * -1.0f);
                            float denominator_y_1_114 = _exp_428 + 1.0f;
                            float hidden_x_1_114 = gate_x_1_114 / denominator_x_1_114 * up_x_1_114;
                            float hidden_y_1_114 = gate_y_1_114 / denominator_y_1_114 * up_y_1_114;
                            hidden_packed_3[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_114, hidden_y_1_114)));
                            float2 _cvt_f32_115 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[19]));
                            float2 _cvt_f32_256_115 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[19]));
                            float gate_x_1_115 = _cvt_f32_115.x;
                            float gate_y_1_115 = _cvt_f32_115.y;
                            float up_x_1_115 = _cvt_f32_256_115.x;
                            float up_y_1_115 = _cvt_f32_256_115.y;
                            gate_x_1_115 = fminf(gate_x_1_115, swiglu_limit);
                            gate_y_1_115 = fminf(gate_y_1_115, swiglu_limit);
                            up_x_1_115 = fminf(fmaxf(up_x_1_115, -swiglu_limit), swiglu_limit);
                            up_y_1_115 = fminf(fmaxf(up_y_1_115, -swiglu_limit), swiglu_limit);
                            float _exp_429 = expf(gate_x_1_115 * -1.0f);
                            float denominator_x_1_115 = _exp_429 + 1.0f;
                            float _exp_430 = expf(gate_y_1_115 * -1.0f);
                            float denominator_y_1_115 = _exp_430 + 1.0f;
                            float hidden_x_1_115 = gate_x_1_115 / denominator_x_1_115 * up_x_1_115;
                            float hidden_y_1_115 = gate_y_1_115 / denominator_y_1_115 * up_y_1_115;
                            hidden_packed_3[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_115, hidden_y_1_115)));
                            float2 _cvt_f32_116 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[20]));
                            float2 _cvt_f32_256_116 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[20]));
                            float gate_x_1_116 = _cvt_f32_116.x;
                            float gate_y_1_116 = _cvt_f32_116.y;
                            float up_x_1_116 = _cvt_f32_256_116.x;
                            float up_y_1_116 = _cvt_f32_256_116.y;
                            gate_x_1_116 = fminf(gate_x_1_116, swiglu_limit);
                            gate_y_1_116 = fminf(gate_y_1_116, swiglu_limit);
                            up_x_1_116 = fminf(fmaxf(up_x_1_116, -swiglu_limit), swiglu_limit);
                            up_y_1_116 = fminf(fmaxf(up_y_1_116, -swiglu_limit), swiglu_limit);
                            float _exp_431 = expf(gate_x_1_116 * -1.0f);
                            float denominator_x_1_116 = _exp_431 + 1.0f;
                            float _exp_432 = expf(gate_y_1_116 * -1.0f);
                            float denominator_y_1_116 = _exp_432 + 1.0f;
                            float hidden_x_1_116 = gate_x_1_116 / denominator_x_1_116 * up_x_1_116;
                            float hidden_y_1_116 = gate_y_1_116 / denominator_y_1_116 * up_y_1_116;
                            hidden_packed_3[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_116, hidden_y_1_116)));
                            float2 _cvt_f32_117 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[21]));
                            float2 _cvt_f32_256_117 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[21]));
                            float gate_x_1_117 = _cvt_f32_117.x;
                            float gate_y_1_117 = _cvt_f32_117.y;
                            float up_x_1_117 = _cvt_f32_256_117.x;
                            float up_y_1_117 = _cvt_f32_256_117.y;
                            gate_x_1_117 = fminf(gate_x_1_117, swiglu_limit);
                            gate_y_1_117 = fminf(gate_y_1_117, swiglu_limit);
                            up_x_1_117 = fminf(fmaxf(up_x_1_117, -swiglu_limit), swiglu_limit);
                            up_y_1_117 = fminf(fmaxf(up_y_1_117, -swiglu_limit), swiglu_limit);
                            float _exp_433 = expf(gate_x_1_117 * -1.0f);
                            float denominator_x_1_117 = _exp_433 + 1.0f;
                            float _exp_434 = expf(gate_y_1_117 * -1.0f);
                            float denominator_y_1_117 = _exp_434 + 1.0f;
                            float hidden_x_1_117 = gate_x_1_117 / denominator_x_1_117 * up_x_1_117;
                            float hidden_y_1_117 = gate_y_1_117 / denominator_y_1_117 * up_y_1_117;
                            hidden_packed_3[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_117, hidden_y_1_117)));
                            float2 _cvt_f32_118 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[22]));
                            float2 _cvt_f32_256_118 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[22]));
                            float gate_x_1_118 = _cvt_f32_118.x;
                            float gate_y_1_118 = _cvt_f32_118.y;
                            float up_x_1_118 = _cvt_f32_256_118.x;
                            float up_y_1_118 = _cvt_f32_256_118.y;
                            gate_x_1_118 = fminf(gate_x_1_118, swiglu_limit);
                            gate_y_1_118 = fminf(gate_y_1_118, swiglu_limit);
                            up_x_1_118 = fminf(fmaxf(up_x_1_118, -swiglu_limit), swiglu_limit);
                            up_y_1_118 = fminf(fmaxf(up_y_1_118, -swiglu_limit), swiglu_limit);
                            float _exp_435 = expf(gate_x_1_118 * -1.0f);
                            float denominator_x_1_118 = _exp_435 + 1.0f;
                            float _exp_436 = expf(gate_y_1_118 * -1.0f);
                            float denominator_y_1_118 = _exp_436 + 1.0f;
                            float hidden_x_1_118 = gate_x_1_118 / denominator_x_1_118 * up_x_1_118;
                            float hidden_y_1_118 = gate_y_1_118 / denominator_y_1_118 * up_y_1_118;
                            hidden_packed_3[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_118, hidden_y_1_118)));
                            float2 _cvt_f32_119 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[23]));
                            float2 _cvt_f32_256_119 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[23]));
                            float gate_x_1_119 = _cvt_f32_119.x;
                            float gate_y_1_119 = _cvt_f32_119.y;
                            float up_x_1_119 = _cvt_f32_256_119.x;
                            float up_y_1_119 = _cvt_f32_256_119.y;
                            gate_x_1_119 = fminf(gate_x_1_119, swiglu_limit);
                            gate_y_1_119 = fminf(gate_y_1_119, swiglu_limit);
                            up_x_1_119 = fminf(fmaxf(up_x_1_119, -swiglu_limit), swiglu_limit);
                            up_y_1_119 = fminf(fmaxf(up_y_1_119, -swiglu_limit), swiglu_limit);
                            float _exp_437 = expf(gate_x_1_119 * -1.0f);
                            float denominator_x_1_119 = _exp_437 + 1.0f;
                            float _exp_438 = expf(gate_y_1_119 * -1.0f);
                            float denominator_y_1_119 = _exp_438 + 1.0f;
                            float hidden_x_1_119 = gate_x_1_119 / denominator_x_1_119 * up_x_1_119;
                            float hidden_y_1_119 = gate_y_1_119 / denominator_y_1_119 * up_y_1_119;
                            hidden_packed_3[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_119, hidden_y_1_119)));
                            float2 _cvt_f32_120 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[24]));
                            float2 _cvt_f32_256_120 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[24]));
                            float gate_x_1_120 = _cvt_f32_120.x;
                            float gate_y_1_120 = _cvt_f32_120.y;
                            float up_x_1_120 = _cvt_f32_256_120.x;
                            float up_y_1_120 = _cvt_f32_256_120.y;
                            gate_x_1_120 = fminf(gate_x_1_120, swiglu_limit);
                            gate_y_1_120 = fminf(gate_y_1_120, swiglu_limit);
                            up_x_1_120 = fminf(fmaxf(up_x_1_120, -swiglu_limit), swiglu_limit);
                            up_y_1_120 = fminf(fmaxf(up_y_1_120, -swiglu_limit), swiglu_limit);
                            float _exp_439 = expf(gate_x_1_120 * -1.0f);
                            float denominator_x_1_120 = _exp_439 + 1.0f;
                            float _exp_440 = expf(gate_y_1_120 * -1.0f);
                            float denominator_y_1_120 = _exp_440 + 1.0f;
                            float hidden_x_1_120 = gate_x_1_120 / denominator_x_1_120 * up_x_1_120;
                            float hidden_y_1_120 = gate_y_1_120 / denominator_y_1_120 * up_y_1_120;
                            hidden_packed_3[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_120, hidden_y_1_120)));
                            float2 _cvt_f32_121 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[25]));
                            float2 _cvt_f32_256_121 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[25]));
                            float gate_x_1_121 = _cvt_f32_121.x;
                            float gate_y_1_121 = _cvt_f32_121.y;
                            float up_x_1_121 = _cvt_f32_256_121.x;
                            float up_y_1_121 = _cvt_f32_256_121.y;
                            gate_x_1_121 = fminf(gate_x_1_121, swiglu_limit);
                            gate_y_1_121 = fminf(gate_y_1_121, swiglu_limit);
                            up_x_1_121 = fminf(fmaxf(up_x_1_121, -swiglu_limit), swiglu_limit);
                            up_y_1_121 = fminf(fmaxf(up_y_1_121, -swiglu_limit), swiglu_limit);
                            float _exp_441 = expf(gate_x_1_121 * -1.0f);
                            float denominator_x_1_121 = _exp_441 + 1.0f;
                            float _exp_442 = expf(gate_y_1_121 * -1.0f);
                            float denominator_y_1_121 = _exp_442 + 1.0f;
                            float hidden_x_1_121 = gate_x_1_121 / denominator_x_1_121 * up_x_1_121;
                            float hidden_y_1_121 = gate_y_1_121 / denominator_y_1_121 * up_y_1_121;
                            hidden_packed_3[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_121, hidden_y_1_121)));
                            float2 _cvt_f32_122 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[26]));
                            float2 _cvt_f32_256_122 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[26]));
                            float gate_x_1_122 = _cvt_f32_122.x;
                            float gate_y_1_122 = _cvt_f32_122.y;
                            float up_x_1_122 = _cvt_f32_256_122.x;
                            float up_y_1_122 = _cvt_f32_256_122.y;
                            gate_x_1_122 = fminf(gate_x_1_122, swiglu_limit);
                            gate_y_1_122 = fminf(gate_y_1_122, swiglu_limit);
                            up_x_1_122 = fminf(fmaxf(up_x_1_122, -swiglu_limit), swiglu_limit);
                            up_y_1_122 = fminf(fmaxf(up_y_1_122, -swiglu_limit), swiglu_limit);
                            float _exp_443 = expf(gate_x_1_122 * -1.0f);
                            float denominator_x_1_122 = _exp_443 + 1.0f;
                            float _exp_444 = expf(gate_y_1_122 * -1.0f);
                            float denominator_y_1_122 = _exp_444 + 1.0f;
                            float hidden_x_1_122 = gate_x_1_122 / denominator_x_1_122 * up_x_1_122;
                            float hidden_y_1_122 = gate_y_1_122 / denominator_y_1_122 * up_y_1_122;
                            hidden_packed_3[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_122, hidden_y_1_122)));
                            float2 _cvt_f32_123 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[27]));
                            float2 _cvt_f32_256_123 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[27]));
                            float gate_x_1_123 = _cvt_f32_123.x;
                            float gate_y_1_123 = _cvt_f32_123.y;
                            float up_x_1_123 = _cvt_f32_256_123.x;
                            float up_y_1_123 = _cvt_f32_256_123.y;
                            gate_x_1_123 = fminf(gate_x_1_123, swiglu_limit);
                            gate_y_1_123 = fminf(gate_y_1_123, swiglu_limit);
                            up_x_1_123 = fminf(fmaxf(up_x_1_123, -swiglu_limit), swiglu_limit);
                            up_y_1_123 = fminf(fmaxf(up_y_1_123, -swiglu_limit), swiglu_limit);
                            float _exp_445 = expf(gate_x_1_123 * -1.0f);
                            float denominator_x_1_123 = _exp_445 + 1.0f;
                            float _exp_446 = expf(gate_y_1_123 * -1.0f);
                            float denominator_y_1_123 = _exp_446 + 1.0f;
                            float hidden_x_1_123 = gate_x_1_123 / denominator_x_1_123 * up_x_1_123;
                            float hidden_y_1_123 = gate_y_1_123 / denominator_y_1_123 * up_y_1_123;
                            hidden_packed_3[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_123, hidden_y_1_123)));
                            float2 _cvt_f32_124 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[28]));
                            float2 _cvt_f32_256_124 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[28]));
                            float gate_x_1_124 = _cvt_f32_124.x;
                            float gate_y_1_124 = _cvt_f32_124.y;
                            float up_x_1_124 = _cvt_f32_256_124.x;
                            float up_y_1_124 = _cvt_f32_256_124.y;
                            gate_x_1_124 = fminf(gate_x_1_124, swiglu_limit);
                            gate_y_1_124 = fminf(gate_y_1_124, swiglu_limit);
                            up_x_1_124 = fminf(fmaxf(up_x_1_124, -swiglu_limit), swiglu_limit);
                            up_y_1_124 = fminf(fmaxf(up_y_1_124, -swiglu_limit), swiglu_limit);
                            float _exp_447 = expf(gate_x_1_124 * -1.0f);
                            float denominator_x_1_124 = _exp_447 + 1.0f;
                            float _exp_448 = expf(gate_y_1_124 * -1.0f);
                            float denominator_y_1_124 = _exp_448 + 1.0f;
                            float hidden_x_1_124 = gate_x_1_124 / denominator_x_1_124 * up_x_1_124;
                            float hidden_y_1_124 = gate_y_1_124 / denominator_y_1_124 * up_y_1_124;
                            hidden_packed_3[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_124, hidden_y_1_124)));
                            float2 _cvt_f32_125 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[29]));
                            float2 _cvt_f32_256_125 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[29]));
                            float gate_x_1_125 = _cvt_f32_125.x;
                            float gate_y_1_125 = _cvt_f32_125.y;
                            float up_x_1_125 = _cvt_f32_256_125.x;
                            float up_y_1_125 = _cvt_f32_256_125.y;
                            gate_x_1_125 = fminf(gate_x_1_125, swiglu_limit);
                            gate_y_1_125 = fminf(gate_y_1_125, swiglu_limit);
                            up_x_1_125 = fminf(fmaxf(up_x_1_125, -swiglu_limit), swiglu_limit);
                            up_y_1_125 = fminf(fmaxf(up_y_1_125, -swiglu_limit), swiglu_limit);
                            float _exp_449 = expf(gate_x_1_125 * -1.0f);
                            float denominator_x_1_125 = _exp_449 + 1.0f;
                            float _exp_450 = expf(gate_y_1_125 * -1.0f);
                            float denominator_y_1_125 = _exp_450 + 1.0f;
                            float hidden_x_1_125 = gate_x_1_125 / denominator_x_1_125 * up_x_1_125;
                            float hidden_y_1_125 = gate_y_1_125 / denominator_y_1_125 * up_y_1_125;
                            hidden_packed_3[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_125, hidden_y_1_125)));
                            float2 _cvt_f32_126 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[30]));
                            float2 _cvt_f32_256_126 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[30]));
                            float gate_x_1_126 = _cvt_f32_126.x;
                            float gate_y_1_126 = _cvt_f32_126.y;
                            float up_x_1_126 = _cvt_f32_256_126.x;
                            float up_y_1_126 = _cvt_f32_256_126.y;
                            gate_x_1_126 = fminf(gate_x_1_126, swiglu_limit);
                            gate_y_1_126 = fminf(gate_y_1_126, swiglu_limit);
                            up_x_1_126 = fminf(fmaxf(up_x_1_126, -swiglu_limit), swiglu_limit);
                            up_y_1_126 = fminf(fmaxf(up_y_1_126, -swiglu_limit), swiglu_limit);
                            float _exp_451 = expf(gate_x_1_126 * -1.0f);
                            float denominator_x_1_126 = _exp_451 + 1.0f;
                            float _exp_452 = expf(gate_y_1_126 * -1.0f);
                            float denominator_y_1_126 = _exp_452 + 1.0f;
                            float hidden_x_1_126 = gate_x_1_126 / denominator_x_1_126 * up_x_1_126;
                            float hidden_y_1_126 = gate_y_1_126 / denominator_y_1_126 * up_y_1_126;
                            hidden_packed_3[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_126, hidden_y_1_126)));
                            float2 _cvt_f32_127 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[31]));
                            float2 _cvt_f32_256_127 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[31]));
                            float gate_x_1_127 = _cvt_f32_127.x;
                            float gate_y_1_127 = _cvt_f32_127.y;
                            float up_x_1_127 = _cvt_f32_256_127.x;
                            float up_y_1_127 = _cvt_f32_256_127.y;
                            gate_x_1_127 = fminf(gate_x_1_127, swiglu_limit);
                            gate_y_1_127 = fminf(gate_y_1_127, swiglu_limit);
                            up_x_1_127 = fminf(fmaxf(up_x_1_127, -swiglu_limit), swiglu_limit);
                            up_y_1_127 = fminf(fmaxf(up_y_1_127, -swiglu_limit), swiglu_limit);
                            float _exp_453 = expf(gate_x_1_127 * -1.0f);
                            float denominator_x_1_127 = _exp_453 + 1.0f;
                            float _exp_454 = expf(gate_y_1_127 * -1.0f);
                            float denominator_y_1_127 = _exp_454 + 1.0f;
                            float hidden_x_1_127 = gate_x_1_127 / denominator_x_1_127 * up_x_1_127;
                            float hidden_y_1_127 = gate_y_1_127 / denominator_y_1_127 * up_y_1_127;
                            hidden_packed_3[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_127, hidden_y_1_127)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_17 = tid / 32;
                            int lane_17 = tid % 32;
                            #pragma unroll
                            for (int half_26 = 0; half_26 < 2; half_26++) {
                                #pragma unroll
                                for (int col_tile_25 = 0; col_tile_25 < 2; col_tile_25++) {
                                    int row_27 = warp_17 * 32 + half_26 * 16 + lane_17 % 16;
                                    int col_27 = col_tile_25 * 16 + lane_17 / 16 * 8;
                                    unsigned int address_18 = d_smem_addr + (unsigned int)((row_27 * 32 + col_27) * 2);
                                    address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                    int offset_18 = half_26 * 8 + col_tile_25 * 4;
                                    uint32_t _stmatrix_addr_455 = static_cast<uint32_t>(address_18);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_455), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_18 + 3]))
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
                            int warp_18 = tid / 32;
                            int lane_18 = tid % 32;
                            #pragma unroll
                            for (int half_27 = 0; half_27 < 2; half_27++) {
                                #pragma unroll
                                for (int col_tile_26 = 0; col_tile_26 < 2; col_tile_26++) {
                                    int row_28 = warp_18 * 32 + half_27 * 16 + lane_18 % 16;
                                    int col_28 = col_tile_26 * 16 + lane_18 / 16 * 8;
                                    unsigned int address_19 = d_smem_addr + 8192 + (unsigned int)((row_28 * 32 + col_28) * 2);
                                    address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                    int offset_19 = half_27 * 8 + col_tile_26 * 4;
                                    uint32_t _stmatrix_addr_456 = static_cast<uint32_t>(address_19);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_456), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_19])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_19 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_19 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_19 + 3]))
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
                            int warp_19 = tid / 32;
                            int lane_19 = tid % 32;
                            #pragma unroll
                            for (int half_28 = 0; half_28 < 2; half_28++) {
                                #pragma unroll
                                for (int col_tile_27 = 0; col_tile_27 < 2; col_tile_27++) {
                                    int row_29 = warp_19 * 32 + half_28 * 16 + lane_19 % 16;
                                    int col_29 = col_tile_27 * 16 + lane_19 / 16 * 8;
                                    unsigned int address_20 = d_smem_addr + 16384 + (unsigned int)((row_29 * 32 + col_29) * 2);
                                    address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                    int offset_20 = half_28 * 8 + col_tile_27 * 4;
                                    uint32_t _stmatrix_addr_457 = static_cast<uint32_t>(address_20);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_457), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_20 + 3]))
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
                            int warp_20 = tid / 32;
                            int lane_20 = tid % 32;
                            #pragma unroll
                            for (int half_29 = 0; half_29 < 2; half_29++) {
                                #pragma unroll
                                for (int col_tile_28 = 0; col_tile_28 < 2; col_tile_28++) {
                                    int row_30 = warp_20 * 32 + half_29 * 16 + lane_20 % 16;
                                    int col_30 = col_tile_28 * 16 + lane_20 / 16 * 8;
                                    unsigned int address_21 = d_smem_addr + (unsigned int)((row_30 * 32 + col_30) * 2);
                                    address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                    int offset_21 = 16 + half_29 * 8 + col_tile_28 * 4;
                                    uint32_t _stmatrix_addr_458 = static_cast<uint32_t>(address_21);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_458), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_3[offset_21 + 3]))
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
                            int warp_21 = tid / 32;
                            int lane_21 = tid % 32;
                            #pragma unroll
                            for (int half_30 = 0; half_30 < 2; half_30++) {
                                #pragma unroll
                                for (int col_tile_29 = 0; col_tile_29 < 2; col_tile_29++) {
                                    int row_31 = warp_21 * 32 + half_30 * 16 + lane_21 % 16;
                                    int col_31 = col_tile_29 * 16 + lane_21 / 16 * 8;
                                    unsigned int address_22 = d_smem_addr + 8192 + (unsigned int)((row_31 * 32 + col_31) * 2);
                                    address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                    int offset_22 = 16 + half_30 * 8 + col_tile_29 * 4;
                                    uint32_t _stmatrix_addr_459 = static_cast<uint32_t>(address_22);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_459), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_3[offset_22 + 3]))
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
                            int warp_22 = tid / 32;
                            int lane_22 = tid % 32;
                            #pragma unroll
                            for (int half_31 = 0; half_31 < 2; half_31++) {
                                #pragma unroll
                                for (int col_tile_30 = 0; col_tile_30 < 2; col_tile_30++) {
                                    int row_32 = warp_22 * 32 + half_31 * 16 + lane_22 % 16;
                                    int col_32 = col_tile_30 * 16 + lane_22 / 16 * 8;
                                    unsigned int address_23 = d_smem_addr + 16384 + (unsigned int)((row_32 * 32 + col_32) * 2);
                                    address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                    int offset_23 = 16 + half_31 * 8 + col_tile_30 * 4;
                                    uint32_t _stmatrix_addr_460 = static_cast<uint32_t>(address_23);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_460), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_23])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_23 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_23 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_23 + 3]))
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
                    gemm_wide += iterations;
                    gemm_out0 += 1;
                    gemm_acc_tiles += 1;
                }
            } else if (compute < shared_tasks) {
                int col_blocks_6 = (hidden + 512 - 1) / 512;
                int x_1 = -1;
                int y_1 = -1;
                int expert_1 = -1;
                int k_start_1 = 0;
                int k_end_1 = 0;
                int first_1 = 0;
                int row_blocks_1 = local_tokens / 256;
                if (compute - shared_fused < row_blocks_1 * col_blocks_6) {
                    int supergroup_1 = (compute - shared_fused) / (row_blocks_1 * 8);
                    int full_cols_1 = col_blocks_6 / 8 * 8;
                    int row_33 = 0;
                    int col_33 = 0;
                    if (compute - shared_fused < row_blocks_1 * full_cols_1) {
                        row_33 = (compute - shared_fused) % (row_blocks_1 * 8) / 8;
                        col_33 = supergroup_1 * 8 + (compute - shared_fused) % 8;
                    } else {
                        row_33 = (compute - shared_fused - row_blocks_1 * full_cols_1) / (col_blocks_6 - full_cols_1);
                        col_33 = full_cols_1 + (compute - shared_fused - row_blocks_1 * full_cols_1) % (col_blocks_6 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_33 = row_blocks_1 - row_33 - 1;
                    }
                    x_1 = row_33;
                    y_1 = col_33;
                    expert_1 = 0;
                }
                int has_hi_1 = 0;
                has_hi_1 = (int)((y_1 * 2 + 1) * 256 < hidden);
                int global_mini_1 = 0;
                int macro_rows_2 = 0;
                int iterations_1 = intermediate / 64;
                if (expert_1 < 0) {
                    if (tid == 0) {
                    }
                } else {
                    if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    bool enabled_value_2 = 1;
                                    if (enabled_value_2 != 0) {
                                        int poll_3_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_first) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                        int poll_3 = poll_3_first;
                                        while (poll_3 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_3_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_next) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                            poll_3 = poll_3_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_461 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                    int mini_rows_4 = ((0) > (_min_461) ? (0) : (_min_461));
                                    int required_4 = (mini_rows_4 + 127) / 128 * ((intermediate + 511) / 512);
                                }
                                #pragma unroll 1
                                for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                    int _stage_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_2) % 4);
                                    int _parity_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_2) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_1) * 8, _parity_gemm_finished_1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&hidden_shared_in)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wd_shared)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wd_shared)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
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
                                    int _smem_desc_lo_462 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_463 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_462), "r"(_smem_desc_lo_463), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_464 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_465 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_464), "r"(_smem_desc_lo_465), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else {
                        if (tid < 128) {
                            int _parity_output_arrived_1 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_1);
                            unsigned int packed[128];
                            #pragma unroll
                            for (int chunk_8 = 0; chunk_8 < 8; chunk_8++) {
                                #pragma unroll
                                for (int sub = 0; sub < 2; sub++) {
                                    unsigned int address_24 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk_8 * 32);
                                    float _tmem_load_32[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[15]))
                                        : "r"(address_24));
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; pair++) {
                                        packed[chunk_8 * 16 + sub * 8 + pair] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32[pair * 2], _tmem_load_32[pair * 2 + 1])));
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
                                int _min_466 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                                if (output_row < _min_466) {
                                }
                            }
                            #pragma unroll
                            for (int chunk_9 = 0; chunk_9 < 8; chunk_9++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_23 = tid / 32;
                                int lane_23 = tid % 32;
                                #pragma unroll
                                for (int half_32 = 0; half_32 < 2; half_32++) {
                                    #pragma unroll
                                    for (int col_tile_31 = 0; col_tile_31 < 2; col_tile_31++) {
                                        int row_34 = warp_23 * 32 + half_32 * 16 + lane_23 % 16;
                                        int col_34 = col_tile_31 * 16 + lane_23 / 16 * 8;
                                        unsigned int address_25 = d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192) + (unsigned int)((row_34 * 32 + col_34) * 2);
                                        address_25 = address_25 ^ (address_25 & 511) >> 7 << 4;
                                        int offset_24 = chunk_9 * 16 + half_32 * 8 + col_tile_31 * 4;
                                        uint32_t _stmatrix_addr_467 = static_cast<uint32_t>(address_25);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_467), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&y_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 2 * 8 + chunk_9), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (has_hi_1 != 0) {
                                unsigned int packed_0[128];
                                #pragma unroll
                                for (int chunk_10 = 0; chunk_10 < 8; chunk_10++) {
                                    #pragma unroll
                                    for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                        unsigned int address_26 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_10 * 32);
                                        float _tmem_load_32_16[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[15]))
                                            : "r"(address_26));
                                        #pragma unroll
                                        for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                            packed_0[chunk_10 * 16 + sub_1 * 8 + pair_1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16[pair_1 * 2], _tmem_load_32_16[pair_1 * 2 + 1])));
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_0 = 1;
                                if (last_0 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                #pragma unroll
                                for (int chunk_11 = 0; chunk_11 < 8; chunk_11++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_24 = tid / 32;
                                    int lane_24 = tid % 32;
                                    #pragma unroll
                                    for (int half_33 = 0; half_33 < 2; half_33++) {
                                        #pragma unroll
                                        for (int col_tile_32 = 0; col_tile_32 < 2; col_tile_32++) {
                                            int row_35 = warp_24 * 32 + half_33 * 16 + lane_24 % 16;
                                            int col_35 = col_tile_32 * 16 + lane_24 / 16 * 8;
                                            unsigned int address_27 = d_smem_addr + (unsigned int)((8 + chunk_11) % 3 * 8192) + (unsigned int)((row_35 * 32 + col_35) * 2);
                                            address_27 = address_27 ^ (address_27 & 511) >> 7 << 4;
                                            int offset_25 = chunk_11 * 16 + half_33 * 8 + col_tile_32 * 4;
                                            uint32_t _stmatrix_addr_468 = static_cast<uint32_t>(address_27);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_468), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&y_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"((y_1 * 2 + 1) * 8 + chunk_11), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_11) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
            } else {
                if (tile_word_0 < mini_gate) {
                    int col_blocks_7 = (intermediate + 256 - 1) / 256;
                    int x_2 = -1;
                    int y_2 = -1;
                    int expert_2 = -1;
                    int k_start_2 = 0;
                    int k_end_2 = 0;
                    int first_2 = 0;
                    int first_block_2 = (tile_word_1 * (macro_size / mini_size) + tile_word_2) * (mini_size / 256);
                    int end_block_2 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                    int block_2 = first_block_2;
                    uint32_t fast_div_q_4 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_0), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_0);
                    uint32_t fast_div_r_4 = (uint32_t)((unsigned int)tile_word_0) - fast_div_q_4 * (uint32_t)(gate_col_blocks_div.divisor);
                    block_2 = first_block_2 + (int)fast_div_q_4;
                    if (block_2 < end_block_2) {
                        int index = tile_word_4;
                        int offset_26 = tile_word_5 / 256;
                        int first_row_2 = ((first_block_2) > (offset_26) ? (first_block_2) : (offset_26));
                        int _min_469 = ((end_block_2) < (offset_26 + tile_word_6 / 256) ? (end_block_2) : (offset_26 + tile_word_6 / 256));
                        int rows_2 = _min_469 - first_row_2;
                        int supergroup_2 = (tile_word_0 - (first_row_2 - first_block_2) * col_blocks_7) / (rows_2 * 8);
                        int full_cols_2 = col_blocks_7 / 8 * 8;
                        int row_36 = 0;
                        int col_36 = 0;
                        if (tile_word_0 - (first_row_2 - first_block_2) * col_blocks_7 < rows_2 * full_cols_2) {
                            row_36 = (tile_word_0 - (first_row_2 - first_block_2) * col_blocks_7) % (rows_2 * 8) / 8;
                            col_36 = supergroup_2 * 8 + (tile_word_0 - (first_row_2 - first_block_2) * col_blocks_7) % 8;
                        } else {
                            row_36 = (tile_word_0 - (first_row_2 - first_block_2) * col_blocks_7 - rows_2 * full_cols_2) / (col_blocks_7 - full_cols_2);
                            col_36 = full_cols_2 + (tile_word_0 - (first_row_2 - first_block_2) * col_blocks_7 - rows_2 * full_cols_2) % (col_blocks_7 - full_cols_2);
                        }
                        if ((supergroup_2 & 1) != 0) {
                            row_36 = rows_2 - row_36 - 1;
                        }
                        x_2 = first_row_2 + row_36 - tile_word_1 * (macro_size / 256);
                        y_2 = col_36;
                        expert_2 = index;
                    }
                    int has_hi_2 = 0;
                    has_hi_2 = (int)((y_2 * 2 + 1) * 256 < intermediate);
                    int global_mini_2 = tile_word_1 * (macro_size / mini_size) + tile_word_2;
                    int macro_rows_3 = tile_word_1 * (macro_size / 256);
                    int iterations_2 = hidden / 128;
                    int macro_k = tile_word_1 * (macro_size / 128);
                    if (expert_2 < 0) {
                        if (tid == 0) {
                        }
                    } else {
                        if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_470 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                        int mini_rows_5 = ((0) > (_min_470) ? (0) : (_min_470));
                                        int required_5 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_3 = 1;
                                        if (enabled_value_3 != 0) {
                                            int poll_4_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_first) : "l"(x_ready + global_mini_2) : "memory");
                                            int poll_4 = poll_4_first;
                                            while (poll_4 < required_5) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_4_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_next) : "l"(x_ready + global_mini_2) : "memory");
                                                poll_4 = poll_4_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    #pragma unroll 1
                                    for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                        int _stage_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_4) % 4);
                                        int _parity_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_4) / 4 % 2) ^ 1;
                                        mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_2) * 8, _parity_gemm_finished_2);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a4_addr + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b4_addr + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&wg_q)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_hi4_addr + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&wu_q)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        } else if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        int _min_471 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                        int mini_rows_6 = ((0) > (_min_471) ? (0) : (_min_471));
                                        int required_6 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_4 = 1;
                                        if (enabled_value_4 != 0) {
                                            int poll_5_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_first) : "l"(x_ready + global_mini_2) : "memory");
                                            int poll_5 = poll_5_first;
                                            while (poll_5 < required_6) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_5_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_next) : "l"(x_ready + global_mini_2) : "memory");
                                                poll_5 = poll_5_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    #pragma unroll 1
                                    for (int idx_5 = 0; idx_5 < iterations_2; idx_5++) {
                                        int _stage_scales_finished = (int)((unsigned int)(scales_wide + idx_5) % 4);
                                        int _parity_scales_finished = (int)((unsigned int)(scales_wide + idx_5) / 4 % 2) ^ 1;
                                        mbarrier_wait(scales_finished_addr + (_stage_scales_finished) * 8, _parity_scales_finished);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sfa4_addr + (unsigned int)((scales_wide + idx_5) % 4 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_5) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sfb4_addr + (unsigned int)((scales_wide + idx_5) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((expert_2 * (intermediate / 128) + y_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_5) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sfb_hi4_addr + (unsigned int)((scales_wide + idx_5) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((expert_2 * (intermediate / 128) + y_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_5) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int _parity_output_finished_2 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr, _parity_output_finished_2);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_6 = 0; idx_6 < iterations_2; idx_6++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_6) % 4)) * 8, 5120);
                                            int _stage_scales_arrived = (int)((unsigned int)(scales_wide + idx_6) % 4);
                                            int _parity_scales_arrived = (int)((unsigned int)(scales_wide + idx_6) / 4 % 2);
                                            mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived) * 8, _parity_scales_arrived);
                                            int buffer = idx_6 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer * 4, make_sf_cp_desc_sbo128(sfa4_addr + (unsigned int)((scales_wide + idx_6) % 4 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer * 8, make_sf_cp_desc_sbo128(sfb4_addr + (unsigned int)((scales_wide + idx_6) % 4 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer * 8 + 4), make_sf_cp_desc_sbo128((sfb4_addr + (unsigned int)((scales_wide + idx_6) % 4 * 1024) + 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer * 8, make_sf_cp_desc_sbo128(sfb_hi4_addr + (unsigned int)((scales_wide + idx_6) % 4 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer * 8 + 4), make_sf_cp_desc_sbo128((sfb_hi4_addr + (unsigned int)((scales_wide + idx_6) % 4 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_6) % 4)) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_6) % 4);
                                            int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_6) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
                                            int _smem_desc_lo_472 = (((a4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_6) % 4) * 1024;
                                            int _smem_desc_lo_473 = (((b4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_6) % 4) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_smem_desc_lo_472) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_smem_desc_lo_473) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb + buffer * 8 + 0)), ((idx_6 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer * 8 + 0) | 0x80000000), 1);
                                            }
                                            int _smem_desc_lo_474 = (((a4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_6) % 4) * 1024;
                                            int _smem_desc_lo_475 = (((b_hi4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_6) % 4) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_smem_desc_lo_474) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_smem_desc_lo_475) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer * 8 + 0)), ((idx_6 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer * 8 + 0) | 0x80000000), 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8, (uint16_t)(3));
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                int _parity_output_arrived_2 = (int)((unsigned int)gemm_out0 % 2);
                                mbarrier_wait(output_arrived_addr, _parity_output_arrived_2);
                                int warp_row = tid / 32 * 32;
                                unsigned int packed_1[128];
                                #pragma unroll
                                for (int i_64 = 0; i_64 < 8; i_64++) {
                                    float _tmem_load_34[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_34[0]), "=f"(_tmem_load_34[1]), "=f"(_tmem_load_34[2]), "=f"(_tmem_load_34[3]), "=f"(_tmem_load_34[4]), "=f"(_tmem_load_34[5]), "=f"(_tmem_load_34[6]), "=f"(_tmem_load_34[7]), "=f"(_tmem_load_34[8]), "=f"(_tmem_load_34[9]), "=f"(_tmem_load_34[10]), "=f"(_tmem_load_34[11]), "=f"(_tmem_load_34[12]), "=f"(_tmem_load_34[13]), "=f"(_tmem_load_34[14]), "=f"(_tmem_load_34[15]), "=f"(_tmem_load_34[16]), "=f"(_tmem_load_34[17]), "=f"(_tmem_load_34[18]), "=f"(_tmem_load_34[19]), "=f"(_tmem_load_34[20]), "=f"(_tmem_load_34[21]), "=f"(_tmem_load_34[22]), "=f"(_tmem_load_34[23]), "=f"(_tmem_load_34[24]), "=f"(_tmem_load_34[25]), "=f"(_tmem_load_34[26]), "=f"(_tmem_load_34[27]), "=f"(_tmem_load_34[28]), "=f"(_tmem_load_34[29]), "=f"(_tmem_load_34[30]), "=f"(_tmem_load_34[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row << 16) + (unsigned int)(i_64 * 32)));
                                    #pragma unroll
                                    for (int j_16 = 0; j_16 < 16; j_16++) {
                                        packed_1[i_64 * 16 + j_16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_34[2 * j_16], _tmem_load_34[2 * j_16 + 1])));
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                unsigned int scale_word_8 = 0;
                                unsigned int block_0[16];
                                #pragma unroll
                                for (int j_0 = 0; j_0 < 16; j_0++) {
                                    block_0[j_0] = packed_1[j_0];
                                }
                                #pragma unroll
                                for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                                    unsigned int address_0 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_1_1 * 16);
                                    address_0 = address_0 ^ (address_0 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_0 - smem_v42_addr)), "r"(packed_1[4 * j_1_1]), "r"(packed_1[4 * j_1_1 + 1]), "r"(packed_1[4 * j_1_1 + 2]), "r"(packed_1[4 * j_1_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_17;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_17) : "r"(block_0[0]));
                                unsigned int amax_pair_16 = bf16x2_abs_17;
                                #pragma unroll
                                for (int i_65 = 1; i_65 < 16; i_65++) {
                                    uint32_t bf16x2_abs_1_24;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_24) : "r"(block_0[i_65]));
                                    uint32_t bf16x2_max_24;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_24) : "r"(amax_pair_16), "r"(bf16x2_abs_1_24));
                                    amax_pair_16 = bf16x2_max_24;
                                }
                                uint16_t bf16_max_16;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_16) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                float amax_16;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_16) : "h"((uint16_t)(bf16_max_16)));
                                float scale_16 = fmaxf(amax_16 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_476;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_476) : "f"(scale_16), "f"(scale_16));
                                unsigned int scale_byte_16 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_476 & 255;
                                unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                unsigned int words_16[8];
                                #pragma unroll
                                for (int i_66 = 0; i_66 < 8; i_66++) {
                                    uint32_t bf16x2_mul_25;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(block_0[i_66 * 2]), "r"(inverse_16));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(block_0[i_66 * 2 + 1]), "r"(inverse_16));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_477;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_477) : "r"(bf16x2_mul_1_24));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_477) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_16[i_66] = (unsigned int)e4m3x2_bf16x2_24 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_477 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_16[0]), "r"(words_16[1]), "r"(words_16[2]), "r"(words_16[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16[4]), "r"(words_16[5]), "r"(words_16[6]), "r"(words_16[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_1_1[16];
                                #pragma unroll
                                for (int j_2_1 = 0; j_2_1 < 16; j_2_1++) {
                                    block_1_1[j_2_1] = packed_1[16 + j_2_1];
                                }
                                #pragma unroll
                                for (int j_3_1 = 0; j_3_1 < 4; j_3_1++) {
                                    unsigned int address_1_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_3_1 * 16);
                                    address_1_1 = address_1_1 ^ (address_1_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_1_1 - smem_v42_addr)), "r"(packed_1[16 + 4 * j_3_1]), "r"(packed_1[16 + 4 * j_3_1 + 1]), "r"(packed_1[16 + 4 * j_3_1 + 2]), "r"(packed_1[16 + 4 * j_3_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 1, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_8_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_8_1) : "r"(block_1_1[0]));
                                unsigned int amax_pair_8_1 = bf16x2_abs_8_1;
                                #pragma unroll
                                for (int i_67 = 1; i_67 < 16; i_67++) {
                                    uint32_t bf16x2_abs_1_25;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_25) : "r"(block_1_1[i_67]));
                                    uint32_t bf16x2_max_25;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_25) : "r"(amax_pair_8_1), "r"(bf16x2_abs_1_25));
                                    amax_pair_8_1 = bf16x2_max_25;
                                }
                                uint16_t bf16_max_8_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_8_1) : "h"((uint16_t)(amax_pair_8_1 & 65535)), "h"((uint16_t)(amax_pair_8_1 >> 16)));
                                float amax_8_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8_1) : "h"((uint16_t)(bf16_max_8_1)));
                                float scale_8_1 = fmaxf(amax_8_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_478;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_478) : "f"(scale_8_1), "f"(scale_8_1));
                                unsigned int scale_byte_8_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_478 & 255;
                                unsigned int inverse_lane_8_1 = 254 - scale_byte_8_1 << 7;
                                unsigned int inverse_8_1 = inverse_lane_8_1 | inverse_lane_8_1 << 16;
                                unsigned int words_8_1[8];
                                #pragma unroll
                                for (int i_68 = 0; i_68 < 8; i_68++) {
                                    uint32_t bf16x2_mul_26;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(block_1_1[i_68 * 2]), "r"(inverse_8_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(block_1_1[i_68 * 2 + 1]), "r"(inverse_8_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_479;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_479) : "r"(bf16x2_mul_1_25));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_479) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_8_1[i_68] = (unsigned int)e4m3x2_bf16x2_25 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_479 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_8_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_8_1[0]), "r"(words_8_1[1]), "r"(words_8_1[2]), "r"(words_8_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_1[4]), "r"(words_8_1[5]), "r"(words_8_1[6]), "r"(words_8_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 32), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_2_1[16];
                                #pragma unroll
                                for (int j_4_1 = 0; j_4_1 < 16; j_4_1++) {
                                    block_2_1[j_4_1] = packed_1[32 + j_4_1];
                                }
                                #pragma unroll
                                for (int j_5_1 = 0; j_5_1 < 4; j_5_1++) {
                                    unsigned int address_2_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_5_1 * 16);
                                    address_2_1 = address_2_1 ^ (address_2_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_2_1 - smem_v42_addr)), "r"(packed_1[32 + 4 * j_5_1]), "r"(packed_1[32 + 4 * j_5_1 + 1]), "r"(packed_1[32 + 4 * j_5_1 + 2]), "r"(packed_1[32 + 4 * j_5_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 2, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_9_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_9_1) : "r"(block_2_1[0]));
                                unsigned int amax_pair_9_1 = bf16x2_abs_9_1;
                                #pragma unroll
                                for (int i_69 = 1; i_69 < 16; i_69++) {
                                    uint32_t bf16x2_abs_1_26;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_26) : "r"(block_2_1[i_69]));
                                    uint32_t bf16x2_max_26;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_26) : "r"(amax_pair_9_1), "r"(bf16x2_abs_1_26));
                                    amax_pair_9_1 = bf16x2_max_26;
                                }
                                uint16_t bf16_max_9_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_9_1) : "h"((uint16_t)(amax_pair_9_1 & 65535)), "h"((uint16_t)(amax_pair_9_1 >> 16)));
                                float amax_9_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_9_1) : "h"((uint16_t)(bf16_max_9_1)));
                                float scale_9_1 = fmaxf(amax_9_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_480;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_480) : "f"(scale_9_1), "f"(scale_9_1));
                                unsigned int scale_byte_9_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_480 & 255;
                                unsigned int inverse_lane_9_1 = 254 - scale_byte_9_1 << 7;
                                unsigned int inverse_9_1 = inverse_lane_9_1 | inverse_lane_9_1 << 16;
                                unsigned int words_9_1[8];
                                #pragma unroll
                                for (int i_70 = 0; i_70 < 8; i_70++) {
                                    uint32_t bf16x2_mul_27;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(block_2_1[i_70 * 2]), "r"(inverse_9_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(block_2_1[i_70 * 2 + 1]), "r"(inverse_9_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_481;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_481) : "r"(bf16x2_mul_1_26));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_481) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_9_1[i_70] = (unsigned int)e4m3x2_bf16x2_26 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_481 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_9_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_9_1[0]), "r"(words_9_1[1]), "r"(words_9_1[2]), "r"(words_9_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_9_1[4]), "r"(words_9_1[5]), "r"(words_9_1[6]), "r"(words_9_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 64), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_3[16];
                                #pragma unroll
                                for (int j_6_1 = 0; j_6_1 < 16; j_6_1++) {
                                    block_3[j_6_1] = packed_1[48 + j_6_1];
                                }
                                #pragma unroll
                                for (int j_7_1 = 0; j_7_1 < 4; j_7_1++) {
                                    unsigned int address_3_16 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_7_1 * 16);
                                    address_3_16 = address_3_16 ^ (address_3_16 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_3_16 - smem_v42_addr)), "r"(packed_1[48 + 4 * j_7_1]), "r"(packed_1[48 + 4 * j_7_1 + 1]), "r"(packed_1[48 + 4 * j_7_1 + 2]), "r"(packed_1[48 + 4 * j_7_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 3, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_10_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_10_1) : "r"(block_3[0]));
                                unsigned int amax_pair_10_1 = bf16x2_abs_10_1;
                                #pragma unroll
                                for (int i_71 = 1; i_71 < 16; i_71++) {
                                    uint32_t bf16x2_abs_1_27;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_27) : "r"(block_3[i_71]));
                                    uint32_t bf16x2_max_27;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_27) : "r"(amax_pair_10_1), "r"(bf16x2_abs_1_27));
                                    amax_pair_10_1 = bf16x2_max_27;
                                }
                                uint16_t bf16_max_10_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_10_1) : "h"((uint16_t)(amax_pair_10_1 & 65535)), "h"((uint16_t)(amax_pair_10_1 >> 16)));
                                float amax_10_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10_1) : "h"((uint16_t)(bf16_max_10_1)));
                                float scale_10_1 = fmaxf(amax_10_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_482;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_482) : "f"(scale_10_1), "f"(scale_10_1));
                                unsigned int scale_byte_10_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_482 & 255;
                                unsigned int inverse_lane_10_1 = 254 - scale_byte_10_1 << 7;
                                unsigned int inverse_10_1 = inverse_lane_10_1 | inverse_lane_10_1 << 16;
                                unsigned int words_10_1[8];
                                #pragma unroll
                                for (int i_72 = 0; i_72 < 8; i_72++) {
                                    uint32_t bf16x2_mul_28;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(block_3[i_72 * 2]), "r"(inverse_10_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(block_3[i_72 * 2 + 1]), "r"(inverse_10_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_483;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_483) : "r"(bf16x2_mul_1_27));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_483) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_10_1[i_72] = (unsigned int)e4m3x2_bf16x2_27 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_483 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_10_1 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_10_1[0]), "r"(words_10_1[1]), "r"(words_10_1[2]), "r"(words_10_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_10_1[4]), "r"(words_10_1[5]), "r"(words_10_1[6]), "r"(words_10_1[7]) : "memory");
                                smem_v44[tid % 32 * 4 + tid / 32] = scale_word_8;
                                scale_word_8 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 96), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2, smem_v44_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_4[16];
                                #pragma unroll
                                for (int j_8_1 = 0; j_8_1 < 16; j_8_1++) {
                                    block_4[j_8_1] = packed_1[64 + j_8_1];
                                }
                                #pragma unroll
                                for (int j_9_1 = 0; j_9_1 < 4; j_9_1++) {
                                    unsigned int address_4_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_9_1 * 16);
                                    address_4_1 = address_4_1 ^ (address_4_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_4_1 - smem_v42_addr)), "r"(packed_1[64 + 4 * j_9_1]), "r"(packed_1[64 + 4 * j_9_1 + 1]), "r"(packed_1[64 + 4 * j_9_1 + 2]), "r"(packed_1[64 + 4 * j_9_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 4, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_11_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_11_1) : "r"(block_4[0]));
                                unsigned int amax_pair_11_1 = bf16x2_abs_11_1;
                                #pragma unroll
                                for (int i_73 = 1; i_73 < 16; i_73++) {
                                    uint32_t bf16x2_abs_1_28;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_28) : "r"(block_4[i_73]));
                                    uint32_t bf16x2_max_28;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_28) : "r"(amax_pair_11_1), "r"(bf16x2_abs_1_28));
                                    amax_pair_11_1 = bf16x2_max_28;
                                }
                                uint16_t bf16_max_11_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_11_1) : "h"((uint16_t)(amax_pair_11_1 & 65535)), "h"((uint16_t)(amax_pair_11_1 >> 16)));
                                float amax_11_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11_1) : "h"((uint16_t)(bf16_max_11_1)));
                                float scale_11_1 = fmaxf(amax_11_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_484;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_484) : "f"(scale_11_1), "f"(scale_11_1));
                                unsigned int scale_byte_11_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_484 & 255;
                                unsigned int inverse_lane_11_1 = 254 - scale_byte_11_1 << 7;
                                unsigned int inverse_11_1 = inverse_lane_11_1 | inverse_lane_11_1 << 16;
                                unsigned int words_11_1[8];
                                #pragma unroll
                                for (int i_74 = 0; i_74 < 8; i_74++) {
                                    uint32_t bf16x2_mul_29;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(block_4[i_74 * 2]), "r"(inverse_11_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(block_4[i_74 * 2 + 1]), "r"(inverse_11_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_485;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_485) : "r"(bf16x2_mul_1_28));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_485) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_11_1[i_74] = (unsigned int)e4m3x2_bf16x2_28 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_485 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_11_1;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_11_1[0]), "r"(words_11_1[1]), "r"(words_11_1[2]), "r"(words_11_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_11_1[4]), "r"(words_11_1[5]), "r"(words_11_1[6]), "r"(words_11_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 128), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_5[16];
                                #pragma unroll
                                for (int j_10_1 = 0; j_10_1 < 16; j_10_1++) {
                                    block_5[j_10_1] = packed_1[80 + j_10_1];
                                }
                                #pragma unroll
                                for (int j_11_1 = 0; j_11_1 < 4; j_11_1++) {
                                    unsigned int address_5_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_11_1 * 16);
                                    address_5_1 = address_5_1 ^ (address_5_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_5_1 - smem_v42_addr)), "r"(packed_1[80 + 4 * j_11_1]), "r"(packed_1[80 + 4 * j_11_1 + 1]), "r"(packed_1[80 + 4 * j_11_1 + 2]), "r"(packed_1[80 + 4 * j_11_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 5, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_12_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_12_1) : "r"(block_5[0]));
                                unsigned int amax_pair_12_1 = bf16x2_abs_12_1;
                                #pragma unroll
                                for (int i_75 = 1; i_75 < 16; i_75++) {
                                    uint32_t bf16x2_abs_1_29;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_29) : "r"(block_5[i_75]));
                                    uint32_t bf16x2_max_29;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_29) : "r"(amax_pair_12_1), "r"(bf16x2_abs_1_29));
                                    amax_pair_12_1 = bf16x2_max_29;
                                }
                                uint16_t bf16_max_12_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_12_1) : "h"((uint16_t)(amax_pair_12_1 & 65535)), "h"((uint16_t)(amax_pair_12_1 >> 16)));
                                float amax_12_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_12_1) : "h"((uint16_t)(bf16_max_12_1)));
                                float scale_12_1 = fmaxf(amax_12_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_486;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_486) : "f"(scale_12_1), "f"(scale_12_1));
                                unsigned int scale_byte_12_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_486 & 255;
                                unsigned int inverse_lane_12_1 = 254 - scale_byte_12_1 << 7;
                                unsigned int inverse_12_1 = inverse_lane_12_1 | inverse_lane_12_1 << 16;
                                unsigned int words_12_1[8];
                                #pragma unroll
                                for (int i_76 = 0; i_76 < 8; i_76++) {
                                    uint32_t bf16x2_mul_30;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(block_5[i_76 * 2]), "r"(inverse_12_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(block_5[i_76 * 2 + 1]), "r"(inverse_12_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_487;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_487) : "r"(bf16x2_mul_1_29));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_487) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_12_1[i_76] = (unsigned int)e4m3x2_bf16x2_29 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_487 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_12_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_12_1[0]), "r"(words_12_1[1]), "r"(words_12_1[2]), "r"(words_12_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_12_1[4]), "r"(words_12_1[5]), "r"(words_12_1[6]), "r"(words_12_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 160), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_6[16];
                                #pragma unroll
                                for (int j_12_1 = 0; j_12_1 < 16; j_12_1++) {
                                    block_6[j_12_1] = packed_1[96 + j_12_1];
                                }
                                #pragma unroll
                                for (int j_13_1 = 0; j_13_1 < 4; j_13_1++) {
                                    unsigned int address_6_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_13_1 * 16);
                                    address_6_1 = address_6_1 ^ (address_6_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_6_1 - smem_v42_addr)), "r"(packed_1[96 + 4 * j_13_1]), "r"(packed_1[96 + 4 * j_13_1 + 1]), "r"(packed_1[96 + 4 * j_13_1 + 2]), "r"(packed_1[96 + 4 * j_13_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 6, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_13_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_13_1) : "r"(block_6[0]));
                                unsigned int amax_pair_13_1 = bf16x2_abs_13_1;
                                #pragma unroll
                                for (int i_77 = 1; i_77 < 16; i_77++) {
                                    uint32_t bf16x2_abs_1_30;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_30) : "r"(block_6[i_77]));
                                    uint32_t bf16x2_max_30;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_30) : "r"(amax_pair_13_1), "r"(bf16x2_abs_1_30));
                                    amax_pair_13_1 = bf16x2_max_30;
                                }
                                uint16_t bf16_max_13_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_13_1) : "h"((uint16_t)(amax_pair_13_1 & 65535)), "h"((uint16_t)(amax_pair_13_1 >> 16)));
                                float amax_13_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_13_1) : "h"((uint16_t)(bf16_max_13_1)));
                                float scale_13_1 = fmaxf(amax_13_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_488;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_488) : "f"(scale_13_1), "f"(scale_13_1));
                                unsigned int scale_byte_13_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_488 & 255;
                                unsigned int inverse_lane_13_1 = 254 - scale_byte_13_1 << 7;
                                unsigned int inverse_13_1 = inverse_lane_13_1 | inverse_lane_13_1 << 16;
                                unsigned int words_13_1[8];
                                #pragma unroll
                                for (int i_78 = 0; i_78 < 8; i_78++) {
                                    uint32_t bf16x2_mul_31;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(block_6[i_78 * 2]), "r"(inverse_13_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(block_6[i_78 * 2 + 1]), "r"(inverse_13_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_489;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_489) : "r"(bf16x2_mul_1_30));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_489) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_13_1[i_78] = (unsigned int)e4m3x2_bf16x2_30 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_489 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_13_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_13_1[0]), "r"(words_13_1[1]), "r"(words_13_1[2]), "r"(words_13_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_13_1[4]), "r"(words_13_1[5]), "r"(words_13_1[6]), "r"(words_13_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 192), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_7[16];
                                #pragma unroll
                                for (int j_14_1 = 0; j_14_1 < 16; j_14_1++) {
                                    block_7[j_14_1] = packed_1[112 + j_14_1];
                                }
                                #pragma unroll
                                for (int j_15_1 = 0; j_15_1 < 4; j_15_1++) {
                                    unsigned int address_7_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_15_1 * 16);
                                    address_7_1 = address_7_1 ^ (address_7_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_7_1 - smem_v42_addr)), "r"(packed_1[112 + 4 * j_15_1]), "r"(packed_1[112 + 4 * j_15_1 + 1]), "r"(packed_1[112 + 4 * j_15_1 + 2]), "r"(packed_1[112 + 4 * j_15_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 7, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_14_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_14_1) : "r"(block_7[0]));
                                unsigned int amax_pair_14_1 = bf16x2_abs_14_1;
                                #pragma unroll
                                for (int i_79 = 1; i_79 < 16; i_79++) {
                                    uint32_t bf16x2_abs_1_31;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_31) : "r"(block_7[i_79]));
                                    uint32_t bf16x2_max_31;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_31) : "r"(amax_pair_14_1), "r"(bf16x2_abs_1_31));
                                    amax_pair_14_1 = bf16x2_max_31;
                                }
                                uint16_t bf16_max_14_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_14_1) : "h"((uint16_t)(amax_pair_14_1 & 65535)), "h"((uint16_t)(amax_pair_14_1 >> 16)));
                                float amax_14_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_14_1) : "h"((uint16_t)(bf16_max_14_1)));
                                float scale_14_1 = fmaxf(amax_14_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_490;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_490) : "f"(scale_14_1), "f"(scale_14_1));
                                unsigned int scale_byte_14_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_490 & 255;
                                unsigned int inverse_lane_14_1 = 254 - scale_byte_14_1 << 7;
                                unsigned int inverse_14_1 = inverse_lane_14_1 | inverse_lane_14_1 << 16;
                                unsigned int words_14_1[8];
                                #pragma unroll
                                for (int i_80 = 0; i_80 < 8; i_80++) {
                                    uint32_t bf16x2_mul_32;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_32) : "r"(block_7[i_80 * 2]), "r"(inverse_14_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_31) : "r"(block_7[i_80 * 2 + 1]), "r"(inverse_14_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_491;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_491) : "r"(bf16x2_mul_1_31));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_31);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_491) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_14_1[i_80] = (unsigned int)e4m3x2_bf16x2_31 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_491 << 16;
                                }
                                scale_word_8 = scale_word_8 | scale_byte_14_1 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_14_1[0]), "r"(words_14_1[1]), "r"(words_14_1[2]), "r"(words_14_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_14_1[4]), "r"(words_14_1[5]), "r"(words_14_1[6]), "r"(words_14_1[7]) : "memory");
                                smem_v45[tid % 32 * 4 + tid / 32] = scale_word_8;
                                scale_word_8 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 224), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2 + 1, smem_v45_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                #pragma unroll
                                for (int i_81 = 0; i_81 < 8; i_81++) {
                                    float _tmem_load_35[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_35[0]), "=f"(_tmem_load_35[1]), "=f"(_tmem_load_35[2]), "=f"(_tmem_load_35[3]), "=f"(_tmem_load_35[4]), "=f"(_tmem_load_35[5]), "=f"(_tmem_load_35[6]), "=f"(_tmem_load_35[7]), "=f"(_tmem_load_35[8]), "=f"(_tmem_load_35[9]), "=f"(_tmem_load_35[10]), "=f"(_tmem_load_35[11]), "=f"(_tmem_load_35[12]), "=f"(_tmem_load_35[13]), "=f"(_tmem_load_35[14]), "=f"(_tmem_load_35[15]), "=f"(_tmem_load_35[16]), "=f"(_tmem_load_35[17]), "=f"(_tmem_load_35[18]), "=f"(_tmem_load_35[19]), "=f"(_tmem_load_35[20]), "=f"(_tmem_load_35[21]), "=f"(_tmem_load_35[22]), "=f"(_tmem_load_35[23]), "=f"(_tmem_load_35[24]), "=f"(_tmem_load_35[25]), "=f"(_tmem_load_35[26]), "=f"(_tmem_load_35[27]), "=f"(_tmem_load_35[28]), "=f"(_tmem_load_35[29]), "=f"(_tmem_load_35[30]), "=f"(_tmem_load_35[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row << 16) + 256 + (unsigned int)(i_81 * 32)));
                                    #pragma unroll
                                    for (int j_17 = 0; j_17 < 16; j_17++) {
                                        packed_1[i_81 * 16 + j_17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_35[2 * j_17], _tmem_load_35[2 * j_17 + 1])));
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                                unsigned int scale_word_0 = 0;
                                unsigned int block_8[16];
                                #pragma unroll
                                for (int j_16_1 = 0; j_16_1 < 16; j_16_1++) {
                                    block_8[j_16_1] = packed_1[j_16_1];
                                }
                                #pragma unroll
                                for (int j_17_1 = 0; j_17_1 < 4; j_17_1++) {
                                    unsigned int address_8_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_17_1 * 16);
                                    address_8_1 = address_8_1 ^ (address_8_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_8_1 - smem_v42_addr)), "r"(packed_1[4 * j_17_1]), "r"(packed_1[4 * j_17_1 + 1]), "r"(packed_1[4 * j_17_1 + 2]), "r"(packed_1[4 * j_17_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_15_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_15_1) : "r"(block_8[0]));
                                unsigned int amax_pair_15_1 = bf16x2_abs_15_1;
                                #pragma unroll
                                for (int i_82 = 1; i_82 < 16; i_82++) {
                                    uint32_t bf16x2_abs_1_32;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_32) : "r"(block_8[i_82]));
                                    uint32_t bf16x2_max_32;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_32) : "r"(amax_pair_15_1), "r"(bf16x2_abs_1_32));
                                    amax_pair_15_1 = bf16x2_max_32;
                                }
                                uint16_t bf16_max_15_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_15_1) : "h"((uint16_t)(amax_pair_15_1 & 65535)), "h"((uint16_t)(amax_pair_15_1 >> 16)));
                                float amax_15_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_15_1) : "h"((uint16_t)(bf16_max_15_1)));
                                float scale_15_1 = fmaxf(amax_15_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_492;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_492) : "f"(scale_15_1), "f"(scale_15_1));
                                unsigned int scale_byte_15_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_492 & 255;
                                unsigned int inverse_lane_15_1 = 254 - scale_byte_15_1 << 7;
                                unsigned int inverse_15_1 = inverse_lane_15_1 | inverse_lane_15_1 << 16;
                                unsigned int words_15_1[8];
                                #pragma unroll
                                for (int i_83 = 0; i_83 < 8; i_83++) {
                                    uint32_t bf16x2_mul_33;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_33) : "r"(block_8[i_83 * 2]), "r"(inverse_15_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_32) : "r"(block_8[i_83 * 2 + 1]), "r"(inverse_15_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_493;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_493) : "r"(bf16x2_mul_1_32));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_32);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_493) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_15_1[i_83] = (unsigned int)e4m3x2_bf16x2_32 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_493 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_15_1;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_15_1[0]), "r"(words_15_1[1]), "r"(words_15_1[2]), "r"(words_15_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_15_1[4]), "r"(words_15_1[5]), "r"(words_15_1[6]), "r"(words_15_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_9[16];
                                #pragma unroll
                                for (int j_18 = 0; j_18 < 16; j_18++) {
                                    block_9[j_18] = packed_1[16 + j_18];
                                }
                                #pragma unroll
                                for (int j_19 = 0; j_19 < 4; j_19++) {
                                    unsigned int address_9_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_19 * 16);
                                    address_9_1 = address_9_1 ^ (address_9_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_9_1 - smem_v42_addr)), "r"(packed_1[16 + 4 * j_19]), "r"(packed_1[16 + 4 * j_19 + 1]), "r"(packed_1[16 + 4 * j_19 + 2]), "r"(packed_1[16 + 4 * j_19 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 1, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_16_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_16_1) : "r"(block_9[0]));
                                unsigned int amax_pair_16_1 = bf16x2_abs_16_1;
                                #pragma unroll
                                for (int i_84 = 1; i_84 < 16; i_84++) {
                                    uint32_t bf16x2_abs_1_33;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_33) : "r"(block_9[i_84]));
                                    uint32_t bf16x2_max_33;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_33) : "r"(amax_pair_16_1), "r"(bf16x2_abs_1_33));
                                    amax_pair_16_1 = bf16x2_max_33;
                                }
                                uint16_t bf16_max_16_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_16_1) : "h"((uint16_t)(amax_pair_16_1 & 65535)), "h"((uint16_t)(amax_pair_16_1 >> 16)));
                                float amax_16_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_16_1) : "h"((uint16_t)(bf16_max_16_1)));
                                float scale_16_1 = fmaxf(amax_16_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_494;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_494) : "f"(scale_16_1), "f"(scale_16_1));
                                unsigned int scale_byte_16_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_494 & 255;
                                unsigned int inverse_lane_16_1 = 254 - scale_byte_16_1 << 7;
                                unsigned int inverse_16_1 = inverse_lane_16_1 | inverse_lane_16_1 << 16;
                                unsigned int words_16_1[8];
                                #pragma unroll
                                for (int i_85 = 0; i_85 < 8; i_85++) {
                                    uint32_t bf16x2_mul_34;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_34) : "r"(block_9[i_85 * 2]), "r"(inverse_16_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_33) : "r"(block_9[i_85 * 2 + 1]), "r"(inverse_16_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_495;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_495) : "r"(bf16x2_mul_1_33));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_33);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_495) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_16_1[i_85] = (unsigned int)e4m3x2_bf16x2_33 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_495 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_16_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_16_1[0]), "r"(words_16_1[1]), "r"(words_16_1[2]), "r"(words_16_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16_1[4]), "r"(words_16_1[5]), "r"(words_16_1[6]), "r"(words_16_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 32), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_10[16];
                                #pragma unroll
                                for (int j_20 = 0; j_20 < 16; j_20++) {
                                    block_10[j_20] = packed_1[32 + j_20];
                                }
                                #pragma unroll
                                for (int j_21 = 0; j_21 < 4; j_21++) {
                                    unsigned int address_10_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_21 * 16);
                                    address_10_1 = address_10_1 ^ (address_10_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_10_1 - smem_v42_addr)), "r"(packed_1[32 + 4 * j_21]), "r"(packed_1[32 + 4 * j_21 + 1]), "r"(packed_1[32 + 4 * j_21 + 2]), "r"(packed_1[32 + 4 * j_21 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 2, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_17_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_17_1) : "r"(block_10[0]));
                                unsigned int amax_pair_17 = bf16x2_abs_17_1;
                                #pragma unroll
                                for (int i_86 = 1; i_86 < 16; i_86++) {
                                    uint32_t bf16x2_abs_1_34;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_34) : "r"(block_10[i_86]));
                                    uint32_t bf16x2_max_34;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_34) : "r"(amax_pair_17), "r"(bf16x2_abs_1_34));
                                    amax_pair_17 = bf16x2_max_34;
                                }
                                uint16_t bf16_max_17;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_17) : "h"((uint16_t)(amax_pair_17 & 65535)), "h"((uint16_t)(amax_pair_17 >> 16)));
                                float amax_17;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_17) : "h"((uint16_t)(bf16_max_17)));
                                float scale_17 = fmaxf(amax_17 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_496;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_496) : "f"(scale_17), "f"(scale_17));
                                unsigned int scale_byte_17 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_496 & 255;
                                unsigned int inverse_lane_17 = 254 - scale_byte_17 << 7;
                                unsigned int inverse_17 = inverse_lane_17 | inverse_lane_17 << 16;
                                unsigned int words_17[8];
                                #pragma unroll
                                for (int i_87 = 0; i_87 < 8; i_87++) {
                                    uint32_t bf16x2_mul_35;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_35) : "r"(block_10[i_87 * 2]), "r"(inverse_17));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_34) : "r"(block_10[i_87 * 2 + 1]), "r"(inverse_17));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_497;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_497) : "r"(bf16x2_mul_1_34));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_34);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_497) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_17[i_87] = (unsigned int)e4m3x2_bf16x2_34 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_497 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_17 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_17[0]), "r"(words_17[1]), "r"(words_17[2]), "r"(words_17[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_17[4]), "r"(words_17[5]), "r"(words_17[6]), "r"(words_17[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 64), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_11[16];
                                #pragma unroll
                                for (int j_22 = 0; j_22 < 16; j_22++) {
                                    block_11[j_22] = packed_1[48 + j_22];
                                }
                                #pragma unroll
                                for (int j_23 = 0; j_23 < 4; j_23++) {
                                    unsigned int address_11_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_23 * 16);
                                    address_11_1 = address_11_1 ^ (address_11_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_11_1 - smem_v42_addr)), "r"(packed_1[48 + 4 * j_23]), "r"(packed_1[48 + 4 * j_23 + 1]), "r"(packed_1[48 + 4 * j_23 + 2]), "r"(packed_1[48 + 4 * j_23 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 3, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_18;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_18) : "r"(block_11[0]));
                                unsigned int amax_pair_18 = bf16x2_abs_18;
                                #pragma unroll
                                for (int i_88 = 1; i_88 < 16; i_88++) {
                                    uint32_t bf16x2_abs_1_35;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_35) : "r"(block_11[i_88]));
                                    uint32_t bf16x2_max_35;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_35) : "r"(amax_pair_18), "r"(bf16x2_abs_1_35));
                                    amax_pair_18 = bf16x2_max_35;
                                }
                                uint16_t bf16_max_18;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_18) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                float amax_18;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18) : "h"((uint16_t)(bf16_max_18)));
                                float scale_18 = fmaxf(amax_18 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_498;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_498) : "f"(scale_18), "f"(scale_18));
                                unsigned int scale_byte_18 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_498 & 255;
                                unsigned int inverse_lane_18 = 254 - scale_byte_18 << 7;
                                unsigned int inverse_18 = inverse_lane_18 | inverse_lane_18 << 16;
                                unsigned int words_18[8];
                                #pragma unroll
                                for (int i_89 = 0; i_89 < 8; i_89++) {
                                    uint32_t bf16x2_mul_36;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_36) : "r"(block_11[i_89 * 2]), "r"(inverse_18));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_35) : "r"(block_11[i_89 * 2 + 1]), "r"(inverse_18));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_499;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_499) : "r"(bf16x2_mul_1_35));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_35);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_499) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_18[i_89] = (unsigned int)e4m3x2_bf16x2_35 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_499 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_18 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_18[0]), "r"(words_18[1]), "r"(words_18[2]), "r"(words_18[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_18[4]), "r"(words_18[5]), "r"(words_18[6]), "r"(words_18[7]) : "memory");
                                smem_v44[tid % 32 * 4 + tid / 32] = scale_word_0;
                                scale_word_0 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 96), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2, smem_v44_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_12[16];
                                #pragma unroll
                                for (int j_24 = 0; j_24 < 16; j_24++) {
                                    block_12[j_24] = packed_1[64 + j_24];
                                }
                                #pragma unroll
                                for (int j_25 = 0; j_25 < 4; j_25++) {
                                    unsigned int address_12_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_25 * 16);
                                    address_12_1 = address_12_1 ^ (address_12_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_12_1 - smem_v42_addr)), "r"(packed_1[64 + 4 * j_25]), "r"(packed_1[64 + 4 * j_25 + 1]), "r"(packed_1[64 + 4 * j_25 + 2]), "r"(packed_1[64 + 4 * j_25 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 4, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_19;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_19) : "r"(block_12[0]));
                                unsigned int amax_pair_19 = bf16x2_abs_19;
                                #pragma unroll
                                for (int i_90 = 1; i_90 < 16; i_90++) {
                                    uint32_t bf16x2_abs_1_36;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_36) : "r"(block_12[i_90]));
                                    uint32_t bf16x2_max_36;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_36) : "r"(amax_pair_19), "r"(bf16x2_abs_1_36));
                                    amax_pair_19 = bf16x2_max_36;
                                }
                                uint16_t bf16_max_19;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_19) : "h"((uint16_t)(amax_pair_19 & 65535)), "h"((uint16_t)(amax_pair_19 >> 16)));
                                float amax_19;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_19) : "h"((uint16_t)(bf16_max_19)));
                                float scale_19 = fmaxf(amax_19 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_500;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_500) : "f"(scale_19), "f"(scale_19));
                                unsigned int scale_byte_19 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_500 & 255;
                                unsigned int inverse_lane_19 = 254 - scale_byte_19 << 7;
                                unsigned int inverse_19 = inverse_lane_19 | inverse_lane_19 << 16;
                                unsigned int words_19[8];
                                #pragma unroll
                                for (int i_91 = 0; i_91 < 8; i_91++) {
                                    uint32_t bf16x2_mul_37;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_37) : "r"(block_12[i_91 * 2]), "r"(inverse_19));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_36) : "r"(block_12[i_91 * 2 + 1]), "r"(inverse_19));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_501;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "r"(bf16x2_mul_1_36));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_36);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_501) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_19[i_91] = (unsigned int)e4m3x2_bf16x2_36 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_501 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_19;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_19[0]), "r"(words_19[1]), "r"(words_19[2]), "r"(words_19[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_19[4]), "r"(words_19[5]), "r"(words_19[6]), "r"(words_19[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 128), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_13[16];
                                #pragma unroll
                                for (int j_26 = 0; j_26 < 16; j_26++) {
                                    block_13[j_26] = packed_1[80 + j_26];
                                }
                                #pragma unroll
                                for (int j_27 = 0; j_27 < 4; j_27++) {
                                    unsigned int address_13_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_27 * 16);
                                    address_13_1 = address_13_1 ^ (address_13_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_13_1 - smem_v42_addr)), "r"(packed_1[80 + 4 * j_27]), "r"(packed_1[80 + 4 * j_27 + 1]), "r"(packed_1[80 + 4 * j_27 + 2]), "r"(packed_1[80 + 4 * j_27 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 5, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_20;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_20) : "r"(block_13[0]));
                                unsigned int amax_pair_20 = bf16x2_abs_20;
                                #pragma unroll
                                for (int i_92 = 1; i_92 < 16; i_92++) {
                                    uint32_t bf16x2_abs_1_37;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_37) : "r"(block_13[i_92]));
                                    uint32_t bf16x2_max_37;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_37) : "r"(amax_pair_20), "r"(bf16x2_abs_1_37));
                                    amax_pair_20 = bf16x2_max_37;
                                }
                                uint16_t bf16_max_20;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_20) : "h"((uint16_t)(amax_pair_20 & 65535)), "h"((uint16_t)(amax_pair_20 >> 16)));
                                float amax_20;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_20) : "h"((uint16_t)(bf16_max_20)));
                                float scale_20 = fmaxf(amax_20 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_502;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_502) : "f"(scale_20), "f"(scale_20));
                                unsigned int scale_byte_20 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_502 & 255;
                                unsigned int inverse_lane_20 = 254 - scale_byte_20 << 7;
                                unsigned int inverse_20 = inverse_lane_20 | inverse_lane_20 << 16;
                                unsigned int words_20[8];
                                #pragma unroll
                                for (int i_93 = 0; i_93 < 8; i_93++) {
                                    uint32_t bf16x2_mul_38;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_38) : "r"(block_13[i_93 * 2]), "r"(inverse_20));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_37) : "r"(block_13[i_93 * 2 + 1]), "r"(inverse_20));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_503;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "r"(bf16x2_mul_1_37));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_37);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_503) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_20[i_93] = (unsigned int)e4m3x2_bf16x2_37 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_503 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_20 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_20[0]), "r"(words_20[1]), "r"(words_20[2]), "r"(words_20[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_20[4]), "r"(words_20[5]), "r"(words_20[6]), "r"(words_20[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 160), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_14[16];
                                #pragma unroll
                                for (int j_28 = 0; j_28 < 16; j_28++) {
                                    block_14[j_28] = packed_1[96 + j_28];
                                }
                                #pragma unroll
                                for (int j_29 = 0; j_29 < 4; j_29++) {
                                    unsigned int address_14_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_29 * 16);
                                    address_14_1 = address_14_1 ^ (address_14_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_14_1 - smem_v42_addr)), "r"(packed_1[96 + 4 * j_29]), "r"(packed_1[96 + 4 * j_29 + 1]), "r"(packed_1[96 + 4 * j_29 + 2]), "r"(packed_1[96 + 4 * j_29 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 6, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_21;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_21) : "r"(block_14[0]));
                                unsigned int amax_pair_21 = bf16x2_abs_21;
                                #pragma unroll
                                for (int i_94 = 1; i_94 < 16; i_94++) {
                                    uint32_t bf16x2_abs_1_38;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_38) : "r"(block_14[i_94]));
                                    uint32_t bf16x2_max_38;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_38) : "r"(amax_pair_21), "r"(bf16x2_abs_1_38));
                                    amax_pair_21 = bf16x2_max_38;
                                }
                                uint16_t bf16_max_21;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_21) : "h"((uint16_t)(amax_pair_21 & 65535)), "h"((uint16_t)(amax_pair_21 >> 16)));
                                float amax_21;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_21) : "h"((uint16_t)(bf16_max_21)));
                                float scale_21 = fmaxf(amax_21 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_504;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_504) : "f"(scale_21), "f"(scale_21));
                                unsigned int scale_byte_21 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_504 & 255;
                                unsigned int inverse_lane_21 = 254 - scale_byte_21 << 7;
                                unsigned int inverse_21 = inverse_lane_21 | inverse_lane_21 << 16;
                                unsigned int words_21[8];
                                #pragma unroll
                                for (int i_95 = 0; i_95 < 8; i_95++) {
                                    uint32_t bf16x2_mul_39;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_39) : "r"(block_14[i_95 * 2]), "r"(inverse_21));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_38) : "r"(block_14[i_95 * 2 + 1]), "r"(inverse_21));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_505;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "r"(bf16x2_mul_1_38));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_38);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_505) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_21[i_95] = (unsigned int)e4m3x2_bf16x2_38 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_505 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_21 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_21[0]), "r"(words_21[1]), "r"(words_21[2]), "r"(words_21[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_21[4]), "r"(words_21[5]), "r"(words_21[6]), "r"(words_21[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 192), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_15[16];
                                #pragma unroll
                                for (int j_30 = 0; j_30 < 16; j_30++) {
                                    block_15[j_30] = packed_1[112 + j_30];
                                }
                                #pragma unroll
                                for (int j_31 = 0; j_31 < 4; j_31++) {
                                    unsigned int address_15_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_31 * 16);
                                    address_15_1 = address_15_1 ^ (address_15_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (address_15_1 - smem_v42_addr)), "r"(packed_1[112 + 4 * j_31]), "r"(packed_1[112 + 4 * j_31 + 1]), "r"(packed_1[112 + 4 * j_31 + 2]), "r"(packed_1[112 + 4 * j_31 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&up_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 7, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_22;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_22) : "r"(block_15[0]));
                                unsigned int amax_pair_22 = bf16x2_abs_22;
                                #pragma unroll
                                for (int i_96 = 1; i_96 < 16; i_96++) {
                                    uint32_t bf16x2_abs_1_39;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_39) : "r"(block_15[i_96]));
                                    uint32_t bf16x2_max_39;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_39) : "r"(amax_pair_22), "r"(bf16x2_abs_1_39));
                                    amax_pair_22 = bf16x2_max_39;
                                }
                                uint16_t bf16_max_22;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_22) : "h"((uint16_t)(amax_pair_22 & 65535)), "h"((uint16_t)(amax_pair_22 >> 16)));
                                float amax_22;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_22) : "h"((uint16_t)(bf16_max_22)));
                                float scale_22 = fmaxf(amax_22 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_506;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_506) : "f"(scale_22), "f"(scale_22));
                                unsigned int scale_byte_22 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_506 & 255;
                                unsigned int inverse_lane_22 = 254 - scale_byte_22 << 7;
                                unsigned int inverse_22 = inverse_lane_22 | inverse_lane_22 << 16;
                                unsigned int words_22[8];
                                #pragma unroll
                                for (int i_97 = 0; i_97 < 8; i_97++) {
                                    uint32_t bf16x2_mul_40;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_40) : "r"(block_15[i_97 * 2]), "r"(inverse_22));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_39) : "r"(block_15[i_97 * 2 + 1]), "r"(inverse_22));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_507;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "r"(bf16x2_mul_1_39));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_39);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_507) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_22[i_97] = (unsigned int)e4m3x2_bf16x2_39 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_507 << 16;
                                }
                                scale_word_0 = scale_word_0 | scale_byte_22 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32)), "r"(words_22[0]), "r"(words_22[1]), "r"(words_22[2]), "r"(words_22[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v43_addr + (unsigned int)(tid * 32 + 16)), "r"(words_22[4]), "r"(words_22[5]), "r"(words_22[6]), "r"(words_22[7]) : "memory");
                                smem_v45[tid % 32 * 4 + tid / 32] = scale_word_0;
                                scale_word_0 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&up_q_store)), "r"(y_2 * 256 + 224), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v43_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&up_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2 + 1, smem_v45_addr);
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
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (gate_base + (macro_rows_3 + x_2) * (intermediate / 256) + y_2))), "r"(static_cast<unsigned int>(2)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                        gemm_wide += iterations_2;
                        scales_wide += iterations_2;
                        gemm_out0 += 1;
                        gemm_acc_tiles += 1;
                    }
                } else if (tile_word_0 < mini_gate + mini_swiglu) {
                    int col_blocks_8 = intermediate / 128;
                    int num_tiles = tokens / 128 * col_blocks_8;
                    int macro_row_offset = tile_word_1 * (macro_size / 128);
                    int global_mini_3 = tile_word_1 * (macro_size / mini_size) + tile_word_2;
                    int mini_tiles = mini_size / 128 * col_blocks_8;
                    int first_tile_2 = (tile_word_0 - mini_gate) * 6 + cta_rank_0 * 3 + global_mini_3 * mini_tiles;
                    int tile_end = ((num_tiles) < ((global_mini_3 + 1) * mini_tiles) ? (num_tiles) : ((global_mini_3 + 1) * mini_tiles));
                    int macro_tiles_2 = macro_size / 128;
                    if (first_tile_2 < tile_end) {
                        int first_row_3 = first_tile_2 / col_blocks_8;
                        int first_col_2 = first_tile_2 % col_blocks_8;
                        if (tid == 0) {
                            if (tile_end > first_tile_2) {
                                int row_8_1 = first_row_3;
                                int col_0 = first_col_2;
                                if (col_0 >= col_blocks_8) {
                                    row_8_1 = row_8_1 + 1;
                                    col_0 = col_0 - col_blocks_8;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 3)) * 8, 65536);
                                int parent = row_8_1 / 2 * (intermediate / 256) + col_0 / 2;
                                int poll_6_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_first) : "l"(gate_ready + (gate_base + parent)) : "memory");
                                int poll_6 = poll_6_first;
                                while (poll_6 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_6_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_next) : "l"(gate_ready + (gate_base + parent)) : "memory");
                                    poll_6 = poll_6_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + (unsigned int)(swiglu_seq % 3 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_8_1 - macro_row_offset) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 3)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + (unsigned int)(swiglu_seq % 3 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_8_1 - macro_row_offset) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 3)) * 8) : "memory");
                            }
                            if (tile_end > first_tile_2 + 1) {
                                int row_9_1 = first_row_3;
                                int col_1_1 = first_col_2 + 1;
                                if (col_1_1 >= col_blocks_8) {
                                    row_9_1 = row_9_1 + 1;
                                    col_1_1 = col_1_1 - col_blocks_8;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 3)) * 8, 65536);
                                int parent_1 = row_9_1 / 2 * (intermediate / 256) + col_1_1 / 2;
                                int poll_7_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_first) : "l"(gate_ready + (gate_base + parent_1)) : "memory");
                                int poll_7 = poll_7_first;
                                while (poll_7 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_7_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_next) : "l"(gate_ready + (gate_base + parent_1)) : "memory");
                                    poll_7 = poll_7_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + (unsigned int)((swiglu_seq + 1) % 3 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_9_1 - macro_row_offset) * 128), "r"(col_1_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 3)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + (unsigned int)((swiglu_seq + 1) % 3 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_9_1 - macro_row_offset) * 128), "r"(col_1_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 3)) * 8) : "memory");
                            }
                            if (tile_end > first_tile_2 + 2) {
                                int row_10_1 = first_row_3;
                                int col_2_1 = first_col_2 + 2;
                                if (col_2_1 >= col_blocks_8) {
                                    row_10_1 = row_10_1 + 1;
                                    col_2_1 = col_2_1 - col_blocks_8;
                                }
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 2) % 3)) * 8, 65536);
                                int parent_2 = row_10_1 / 2 * (intermediate / 256) + col_2_1 / 2;
                                int poll_8_first;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_8_first) : "l"(gate_ready + (gate_base + parent_2)) : "memory");
                                int poll_8 = poll_8_first;
                                while (poll_8 < 4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int poll_8_next;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_8_next) : "l"(gate_ready + (gate_base + parent_2)) : "memory");
                                    poll_8 = poll_8_next;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(gate_smem_addr + (unsigned int)((swiglu_seq + 2) % 3 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_10_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 2) % 3)) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(up_smem_addr + (unsigned int)((swiglu_seq + 2) % 3 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_10_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 2) % 3)) * 8) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int stage_2 = 0; stage_2 < 3; stage_2++) {
                            if (tile_end > first_tile_2 + stage_2) {
                                int _stage_swiglu_arrived = (int)((unsigned int)(swiglu_seq + stage_2) % 3);
                                int _parity_swiglu_arrived = (int)((unsigned int)(swiglu_seq + stage_2) / 3 % 2);
                                mbarrier_wait(swiglu_arrived_addr + (_stage_swiglu_arrived) * 8, _parity_swiglu_arrived);
                                int slot = (swiglu_seq + stage_2) % 3;
                                int row_37 = first_row_3;
                                int col_37 = first_col_2 + stage_2;
                                if (col_37 >= col_blocks_8) {
                                    row_37 = row_37 + 1;
                                    col_37 = col_37 - col_blocks_8;
                                }
                                float gate[64];
                                float up[64];
                                float denominator[64];
                                int warp_25 = tid / 32;
                                int local_warp = warp_25 / 4 + warp_25 % 4 * 2;
                                int lane_25 = tid % 32;
                                #pragma unroll
                                for (int tile_col = 0; tile_col < 8; tile_col++) {
                                    unsigned int packed_2[4];
                                    unsigned int address_28 = gate_smem_addr + (unsigned int)(slot * 32768) + (unsigned int)(((tile_col * 16 + lane_25 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_25 % 16) * 64 + (tile_col * 16 + lane_25 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                        : "r"(address_28 ^ (address_28 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_2[pair_2]));
                                        gate[tile_col * 8 + pair_2 * 2] = cvt_f32.x;
                                        gate[tile_col * 8 + pair_2 * 2 + 1] = cvt_f32.y;
                                    }
                                }
                                int warp_23_1 = tid / 32;
                                int local_warp_0 = warp_23_1 / 4 + warp_23_1 % 4 * 2;
                                int lane_23_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                    unsigned int packed_3[4];
                                    unsigned int address_29 = up_smem_addr + (unsigned int)(slot * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_23_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_0 * 16 + lane_23_1 % 16) * 64 + (tile_col_1 * 16 + lane_23_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                        : "r"(address_29 ^ (address_29 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_3[pair_3]));
                                        up[tile_col_1 * 8 + pair_3 * 2] = cvt_f32_1.x;
                                        up[tile_col_1 * 8 + pair_3 * 2 + 1] = cvt_f32_1.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem = 0; elem < 64; elem++) {
                                    gate[elem] = fminf(gate[elem], swiglu_limit);
                                }
                                #pragma unroll
                                for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                    up[elem_1] = fminf(fmaxf(up[elem_1], -swiglu_limit), swiglu_limit);
                                }
                                #pragma unroll
                                for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                    denominator[elem_2] = gate[elem_2] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                    float _exp_508 = expf(denominator[elem_3]);
                                    denominator[elem_3] = _exp_508;
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
                                int warp_4_1 = tid / 32;
                                int local_warp_5 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                                int lane_6_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                    unsigned int packed_4[4];
                                    #pragma unroll
                                    for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                        packed_4[pair_4] = __as_u32(__float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_4 * 2], gate[tile_col_2 * 8 + pair_4 * 2 + 1])));
                                    }
                                    int row_0_1 = local_warp_5 * 16 + lane_6_1 % 16;
                                    int col_1_2 = tile_col_2 * 16 + lane_6_1 / 16 * 8;
                                    uint32_t _stmatrix_addr_509 = static_cast<uint32_t>(flat_view_addr + (unsigned int)((row_0_1 * 136 + col_1_2) * 2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_509), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                int out_base = slot * 8192;
                                if (tid < 128) {
                                    int r_8 = tid % 64;
                                    int row0_8 = r_8 * 2;
                                    int row1_8 = row0_8 + 1;
                                    int half_34 = tid / 64;
                                    int word_stride_8 = 68;
                                    int scale_base0_8 = (row0_8 % 32 * 4 + row0_8 / 32) * 4;
                                    int scale_base1_8 = (row1_8 % 32 * 4 + row1_8 / 32) * 4;
                                    #pragma unroll 1
                                    for (int j_32 = 0; j_32 < 2; j_32++) {
                                        int k_block_16 = (half_34 * 2 + j_32 + tid / 8) % 4;
                                        unsigned int pairs0_8[16];
                                        unsigned int pairs1_8[16];
                                        #pragma unroll
                                        for (int k_8 = 0; k_8 < 16; k_8++) {
                                            int col_3_1 = k_block_16 * 32 + (tid * 4 + k_8 * 2) % 32;
                                            unsigned int w0_8 = flat_view_0[col_3_1 * word_stride_8 + r_8];
                                            unsigned int w1_8 = flat_view_0[(col_3_1 + 1) * word_stride_8 + r_8];
                                            uint32_t lo;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo) : "r"(w0_8), "r"(w1_8));
                                            uint32_t hi;
                                            asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi) : "r"(w0_8), "r"(w1_8));
                                            pairs0_8[k_8] = lo;
                                            pairs1_8[k_8] = hi;
                                        }
                                        uint32_t bf16x2_abs_23;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_23) : "r"(pairs0_8[0]));
                                        unsigned int amax_pair_23 = bf16x2_abs_23;
                                        #pragma unroll
                                        for (int i_98 = 1; i_98 < 16; i_98++) {
                                            uint32_t bf16x2_abs_1_40;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_40) : "r"(pairs0_8[i_98]));
                                            uint32_t bf16x2_max_40;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_40) : "r"(amax_pair_23), "r"(bf16x2_abs_1_40));
                                            amax_pair_23 = bf16x2_max_40;
                                        }
                                        uint16_t bf16_max_23;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_23) : "h"((uint16_t)(amax_pair_23 & 65535)), "h"((uint16_t)(amax_pair_23 >> 16)));
                                        float amax_23;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_23) : "h"((uint16_t)(bf16_max_23)));
                                        float scale_23 = fmaxf(amax_23 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_510;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_510) : "f"(scale_23), "f"(scale_23));
                                        unsigned int scale_byte_23 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_510 & 255;
                                        unsigned int inverse_lane_23 = 254 - scale_byte_23 << 7;
                                        unsigned int inverse_23 = inverse_lane_23 | inverse_lane_23 << 16;
                                        unsigned int words_23[8];
                                        #pragma unroll
                                        for (int i_99 = 0; i_99 < 8; i_99++) {
                                            uint32_t bf16x2_mul_41;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_41) : "r"(pairs0_8[i_99 * 2]), "r"(inverse_23));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_40) : "r"(pairs0_8[i_99 * 2 + 1]), "r"(inverse_23));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_511;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "r"(bf16x2_mul_1_40));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_40);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_511) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_23[i_99] = (unsigned int)e4m3x2_bf16x2_40 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_511 << 16;
                                        }
                                        uint32_t bf16x2_abs_23_1;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_23_1) : "r"(pairs1_8[0]));
                                        unsigned int amax_pair_23_1 = bf16x2_abs_23_1;
                                        #pragma unroll
                                        for (int i_100 = 1; i_100 < 16; i_100++) {
                                            uint32_t bf16x2_abs_1_41;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_41) : "r"(pairs1_8[i_100]));
                                            uint32_t bf16x2_max_41;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_41) : "r"(amax_pair_23_1), "r"(bf16x2_abs_1_41));
                                            amax_pair_23_1 = bf16x2_max_41;
                                        }
                                        uint16_t bf16_max_23_1;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_23_1) : "h"((uint16_t)(amax_pair_23_1 & 65535)), "h"((uint16_t)(amax_pair_23_1 >> 16)));
                                        float amax_23_1;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_23_1) : "h"((uint16_t)(bf16_max_23_1)));
                                        float scale_23_1 = fmaxf(amax_23_1 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_512;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_512) : "f"(scale_23_1), "f"(scale_23_1));
                                        unsigned int scale_byte_23_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_512 & 255;
                                        unsigned int inverse_lane_23_1 = 254 - scale_byte_23_1 << 7;
                                        unsigned int inverse_23_1 = inverse_lane_23_1 | inverse_lane_23_1 << 16;
                                        unsigned int words_23_1[8];
                                        #pragma unroll
                                        for (int i_101 = 0; i_101 < 8; i_101++) {
                                            uint32_t bf16x2_mul_42;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_42) : "r"(pairs1_8[i_101 * 2]), "r"(inverse_23_1));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_41) : "r"(pairs1_8[i_101 * 2 + 1]), "r"(inverse_23_1));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_513;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "r"(bf16x2_mul_1_41));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_41);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_513) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_23_1[i_101] = (unsigned int)e4m3x2_bf16x2_41 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_513 << 16;
                                        }
                                        #pragma unroll
                                        for (int k_9 = 0; k_9 < 8; k_9++) {
                                            int col_4_1 = k_block_16 * 32 + (tid * 4 + k_9 * 4) % 32;
                                            flat_view_3[out_base + (row0_8 * 128 + col_4_1) / 4] = words_23[k_9];
                                            flat_view_3[out_base + (row1_8 * 128 + col_4_1) / 4] = words_23_1[k_9];
                                        }
                                        {
                                            uint32_t _byte_514 = static_cast<uint32_t>(scale_byte_23) & 0xFFu;
                                            uint32_t _addr_514 = static_cast<uint32_t>(flat_view_addr_4 + (unsigned int)(out_base * 4 + scale_base0_8 + k_block_16));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_514), "r"(_byte_514) : "memory");
                                        }
                                        {
                                            uint32_t _byte_515 = static_cast<uint32_t>(scale_byte_23_1) & 0xFFu;
                                            uint32_t _addr_515 = static_cast<uint32_t>(flat_view_addr_4 + (unsigned int)(out_base * 4 + scale_base1_8 + k_block_16));
                                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_515), "r"(_byte_515) : "memory");
                                        }
                                    }
                                } else {
                                    int row_11_1 = tid - 128;
                                    unsigned int scale_word_9 = 0;
                                    #pragma unroll 1
                                    for (int j_33 = 0; j_33 < 4; j_33++) {
                                        int k_block_17 = (j_33 + (tid - 128) / 8) % 4;
                                        unsigned int pairs_8[16];
                                        #pragma unroll
                                        for (int k_10 = 0; k_10 < 16; k_10++) {
                                            int col_5_1 = k_block_17 * 32 + ((tid - 128) * 4 + k_10 * 2) % 32;
                                            float x0 = 0.0f;
                                            float x1 = 0.0f;
                                            pairs_8[k_10] = flat_view_0[(row_11_1 * 136 + col_5_1) / 2];
                                        }
                                        uint32_t bf16x2_abs_24;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_24) : "r"(pairs_8[0]));
                                        unsigned int amax_pair_24 = bf16x2_abs_24;
                                        #pragma unroll
                                        for (int i_102 = 1; i_102 < 16; i_102++) {
                                            uint32_t bf16x2_abs_1_42;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_42) : "r"(pairs_8[i_102]));
                                            uint32_t bf16x2_max_42;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_42) : "r"(amax_pair_24), "r"(bf16x2_abs_1_42));
                                            amax_pair_24 = bf16x2_max_42;
                                        }
                                        uint16_t bf16_max_24;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_24) : "h"((uint16_t)(amax_pair_24 & 65535)), "h"((uint16_t)(amax_pair_24 >> 16)));
                                        float amax_24;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_24) : "h"((uint16_t)(bf16_max_24)));
                                        float scale_24 = fmaxf(amax_24 * 0.002232143f, 1e-12f);
                                        uint16_t _cvt_rp_satfinite_ue8m0x2_f32_516;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_516) : "f"(scale_24), "f"(scale_24));
                                        unsigned int scale_byte_24 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_516 & 255;
                                        unsigned int inverse_lane_24 = 254 - scale_byte_24 << 7;
                                        unsigned int inverse_24 = inverse_lane_24 | inverse_lane_24 << 16;
                                        unsigned int words_24[8];
                                        #pragma unroll
                                        for (int i_103 = 0; i_103 < 8; i_103++) {
                                            uint32_t bf16x2_mul_43;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_43) : "r"(pairs_8[i_103 * 2]), "r"(inverse_24));
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
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_42) : "r"(pairs_8[i_103 * 2 + 1]), "r"(inverse_24));
                                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_517;
                                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "r"(bf16x2_mul_1_42));
                                            #else
                                            {
                                            const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_42);
                                            const float _lo = __uint_as_float(_bf16x2 << 16);
                                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_517) : "f"(_hi), "f"(_lo));
                                            }
                                            #endif
                                            words_24[i_103] = (unsigned int)e4m3x2_bf16x2_42 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_517 << 16;
                                        }
                                        scale_word_9 = scale_word_9 | scale_byte_24 << (unsigned int)(k_block_17 * 8);
                                        #pragma unroll
                                        for (int k_11 = 0; k_11 < 8; k_11++) {
                                            int col_6_1 = k_block_17 * 32 + ((tid - 128) * 4 + k_11 * 4) % 32;
                                            flat_view_1[out_base + (row_11_1 * 128 + col_6_1) / 4] = words_24[k_11];
                                        }
                                    }
                                    flat_view_2[out_base + row_11_1 % 32 * 4 + row_11_1 / 32] = scale_word_9;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row = row_37 - macro_row_offset;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&hidden_q_store), col_37 * 128, local_row * 128, flat_view_addr_1 + (unsigned int)(slot * 32768));
                                    tma_store_3d((&hidden_sc_store), 0, 0, local_row * col_blocks_8 + col_37, flat_view_addr_2 + (unsigned int)(slot * 32768));
                                    tma_store_2d((&hidden_t_store), local_row * 128, col_37 * 128, flat_view_addr_3 + (unsigned int)(slot * 32768));
                                    tma_store_3d((&hidden_sc_t_store), 0, 0, col_37 * macro_tiles_2 + local_row, flat_view_addr_4 + (unsigned int)(slot * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            if (tile_end > first_tile_2) {
                                int row_12_1 = first_row_3;
                                if (col_blocks_8 <= first_col_2) {
                                    row_12_1 = row_12_1 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_12_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end > first_tile_2 + 1) {
                                int row_13_1 = first_row_3;
                                if (col_blocks_8 <= first_col_2 + 1) {
                                    row_13_1 = row_13_1 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_13_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            if (tile_end > first_tile_2 + 2) {
                                int row_14_1 = first_row_3;
                                if (col_blocks_8 <= first_col_2 + 2) {
                                    row_14_1 = row_14_1 + 1;
                                }
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_14_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    int _min_519 = ((3) < (tile_end - first_tile_2) ? (3) : (tile_end - first_tile_2));
                    int _max_518 = ((0) > (_min_519) ? (0) : (_min_519));
                    swiglu_seq += _max_518;
                } else {
                    int col_blocks_9 = (hidden + 512 - 1) / 512;
                    int x_3 = -1;
                    int y_3 = -1;
                    int expert_3 = -1;
                    int k_start_3 = 0;
                    int k_end_3 = 0;
                    int first_3 = 0;
                    int first_block_3 = (tile_word_1 * (macro_size / mini_size) + tile_word_2) * (mini_size / 256);
                    int end_block_3 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                    int block_16 = first_block_3;
                    uint32_t fast_div_q_5 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(tile_word_0 - mini_gate - mini_swiglu)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(tile_word_0 - mini_gate - mini_swiglu));
                    uint32_t fast_div_r_5 = (uint32_t)((unsigned int)(tile_word_0 - mini_gate - mini_swiglu)) - fast_div_q_5 * (uint32_t)(down_col_blocks_div.divisor);
                    block_16 = first_block_3 + (int)fast_div_q_5;
                    if (block_16 < end_block_3) {
                        int index_1 = tile_word_4;
                        int offset_27 = tile_word_5 / 256;
                        int first_row_4 = ((first_block_3) > (offset_27) ? (first_block_3) : (offset_27));
                        int _min_520 = ((end_block_3) < (offset_27 + tile_word_6 / 256) ? (end_block_3) : (offset_27 + tile_word_6 / 256));
                        int rows_3 = _min_520 - first_row_4;
                        int supergroup_3 = (tile_word_0 - mini_gate - mini_swiglu - (first_row_4 - first_block_3) * col_blocks_9) / (rows_3 * 8);
                        int full_cols_3 = col_blocks_9 / 8 * 8;
                        int row_38 = 0;
                        int col_38 = 0;
                        if (tile_word_0 - mini_gate - mini_swiglu - (first_row_4 - first_block_3) * col_blocks_9 < rows_3 * full_cols_3) {
                            row_38 = (tile_word_0 - mini_gate - mini_swiglu - (first_row_4 - first_block_3) * col_blocks_9) % (rows_3 * 8) / 8;
                            col_38 = supergroup_3 * 8 + (tile_word_0 - mini_gate - mini_swiglu - (first_row_4 - first_block_3) * col_blocks_9) % 8;
                        } else {
                            row_38 = (tile_word_0 - mini_gate - mini_swiglu - (first_row_4 - first_block_3) * col_blocks_9 - rows_3 * full_cols_3) / (col_blocks_9 - full_cols_3);
                            col_38 = full_cols_3 + (tile_word_0 - mini_gate - mini_swiglu - (first_row_4 - first_block_3) * col_blocks_9 - rows_3 * full_cols_3) % (col_blocks_9 - full_cols_3);
                        }
                        if ((supergroup_3 & 1) != 0) {
                            row_38 = rows_3 - row_38 - 1;
                        }
                        x_3 = first_row_4 + row_38 - tile_word_1 * (macro_size / 256);
                        y_3 = col_38;
                        expert_3 = index_1;
                    }
                    int has_hi_3 = 0;
                    has_hi_3 = (int)((y_3 * 2 + 1) * 256 < hidden);
                    int global_mini_4 = tile_word_1 * (macro_size / mini_size) + tile_word_2;
                    int macro_rows_4 = tile_word_1 * (macro_size / 256);
                    int iterations_3 = intermediate / 128;
                    int macro_k_1 = tile_word_1 * (macro_size / 128);
                    if (expert_3 < 0) {
                        if (tid == 0) {
                        }
                    } else {
                        if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        bool enabled_value_5 = 1;
                                        if (enabled_value_5 != 0) {
                                            int poll_9_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_9_first) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                            int poll_9 = poll_9_first;
                                            while (poll_9 < 2 * (intermediate / 128)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_9_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_9_next) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                                poll_9 = poll_9_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_521 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                        int mini_rows_7 = ((0) > (_min_521) ? (0) : (_min_521));
                                        int required_7 = (mini_rows_7 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    #pragma unroll 1
                                    for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                        int _stage_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_7) % 4);
                                        int _parity_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_7) / 4 % 2) ^ 1;
                                        mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_3) * 8, _parity_gemm_finished_3);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a4_addr + (unsigned int)((gemm_wide + idx_7) % 4 * 16384)), "l"((&hidden_q)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b4_addr + (unsigned int)((gemm_wide + idx_7) % 4 * 16384)), "l"((&wd_q)), "r"(0), "r"(y_3 * 512 + cta_rank_0 * 128), "r"(idx_7), "r"(expert_3), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_hi4_addr + (unsigned int)((gemm_wide + idx_7) % 4 * 16384)), "l"((&wd_q)), "r"(0), "r"(y_3 * 512 + 256 + cta_rank_0 * 128), "r"(idx_7), "r"(expert_3), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        } else if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        bool enabled_value_6 = 1;
                                        if (enabled_value_6 != 0) {
                                            int poll_10_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_10_first) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                            int poll_10 = poll_10_first;
                                            while (poll_10 < 2 * (intermediate / 128)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_10_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_10_next) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                                poll_10 = poll_10_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_522 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                        int mini_rows_8 = ((0) > (_min_522) ? (0) : (_min_522));
                                        int required_8 = (mini_rows_8 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    #pragma unroll 1
                                    for (int idx_8 = 0; idx_8 < iterations_3; idx_8++) {
                                        int _stage_scales_finished_1 = (int)((unsigned int)(scales_wide + idx_8) % 4);
                                        int _parity_scales_finished_1 = (int)((unsigned int)(scales_wide + idx_8) / 4 % 2) ^ 1;
                                        mbarrier_wait(scales_finished_addr + (_stage_scales_finished_1) * 8, _parity_scales_finished_1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sfa4_addr + (unsigned int)((scales_wide + idx_8) % 4 * 512)), "l"((&hidden_sc)), "r"(0), "r"(0), "r"((x_3 * 2 + cta_rank_0) * (intermediate / 128) + idx_8),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sfb4_addr + (unsigned int)((scales_wide + idx_8) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_sc)), "r"(0), "r"(0), "r"((expert_3 * (hidden / 128) + y_3 * 4 + cta_rank_0) * (intermediate / 128) + idx_8),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sfb_hi4_addr + (unsigned int)((scales_wide + idx_8) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_sc)), "r"(0), "r"(0), "r"((expert_3 * (hidden / 128) + y_3 * 4 + 2 + cta_rank_0) * (intermediate / 128) + idx_8),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int _parity_output_finished_3 = (int)((unsigned int)gemm_acc_tiles % 2) ^ 1;
                                        mbarrier_wait(output_finished_addr, _parity_output_finished_3);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_9 = 0; idx_9 < iterations_3; idx_9++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_9) % 4)) * 8, 5120);
                                            int _stage_scales_arrived_1 = (int)((unsigned int)(scales_wide + idx_9) % 4);
                                            int _parity_scales_arrived_1 = (int)((unsigned int)(scales_wide + idx_9) / 4 % 2);
                                            mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_1) * 8, _parity_scales_arrived_1);
                                            int buffer_1 = idx_9 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_1 * 4, make_sf_cp_desc_sbo128(sfa4_addr + (unsigned int)((scales_wide + idx_9) % 4 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_1 * 8, make_sf_cp_desc_sbo128(sfb4_addr + (unsigned int)((scales_wide + idx_9) % 4 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((sfb4_addr + (unsigned int)((scales_wide + idx_9) % 4 * 1024) + 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_hi + buffer_1 * 8, make_sf_cp_desc_sbo128(sfb_hi4_addr + (unsigned int)((scales_wide + idx_9) % 4 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_hi + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((sfb_hi4_addr + (unsigned int)((scales_wide + idx_9) % 4 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, 98304);
                                            int _stage_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_9) % 4);
                                            int _parity_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_9) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_3) * 8, _parity_gemm_arrived_3);
                                            int _smem_desc_lo_523 = (((a4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                            int _smem_desc_lo_524 = (((b4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_smem_desc_lo_523) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_smem_desc_lo_524) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0)), ((idx_9 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0) | 0x80000000), 1);
                                            }
                                            int _smem_desc_lo_525 = (((a4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                            int _smem_desc_lo_526 = (((b_hi4_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_smem_desc_lo_525) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_smem_desc_lo_526) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb_hi + buffer_1 * 8 + 0)), ((idx_9 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2((tmem_accumulator + (256)), a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb_hi + buffer_1 * 8 + 0) | 0x80000000), 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                int _parity_output_arrived_3 = (int)((unsigned int)gemm_out0 % 2);
                                mbarrier_wait(output_arrived_addr, _parity_output_arrived_3);
                                unsigned int packed_5[128];
                                #pragma unroll
                                for (int chunk_12 = 0; chunk_12 < 8; chunk_12++) {
                                    #pragma unroll
                                    for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                        unsigned int address_30 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_12 * 32);
                                        float _tmem_load_32_17[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[15]))
                                            : "r"(address_30));
                                        #pragma unroll
                                        for (int pair_5 = 0; pair_5 < 8; pair_5++) {
                                            packed_5[chunk_12 * 16 + sub_2 * 8 + pair_5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[pair_5 * 2], _tmem_load_32_17[pair_5 * 2 + 1])));
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_1 = 1;
                                last_1 = 1 - has_hi_3;
                                if (last_1 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                if (tid == 0) {
                                    int previous_offset_2 = (tile_word_1 + 1) * macro_size;
                                    int output_row_1 = x_3 * 256 + cta_rank_0 * 128;
                                    int _min_527 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                    if (output_row_1 < _min_527) {
                                        bool enabled_value_7 = 1;
                                        if (enabled_value_7 != 0) {
                                            int poll_11_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_11_first) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                            int poll_11 = poll_11_first;
                                            while (poll_11 < 8 * ((hidden + 1023) / 1024)) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_11_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_11_next) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                                poll_11 = poll_11_next;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int chunk_13 = 0; chunk_13 < 8; chunk_13++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_26 = tid / 32;
                                    int lane_26 = tid % 32;
                                    #pragma unroll
                                    for (int half_35 = 0; half_35 < 2; half_35++) {
                                        #pragma unroll
                                        for (int col_tile_33 = 0; col_tile_33 < 2; col_tile_33++) {
                                            int row_39 = warp_26 * 32 + half_35 * 16 + lane_26 % 16;
                                            int col_39 = col_tile_33 * 16 + lane_26 / 16 * 8;
                                            unsigned int address_31 = d_smem_addr + (unsigned int)(chunk_13 % 3 * 8192) + (unsigned int)((row_39 * 32 + col_39) * 2);
                                            address_31 = address_31 ^ (address_31 & 511) >> 7 << 4;
                                            int offset_28 = chunk_13 * 16 + half_35 * 8 + col_tile_33 * 4;
                                            uint32_t _stmatrix_addr_528 = static_cast<uint32_t>(address_31);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_528), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_28])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_28 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_28 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_28 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&y_routed), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 2 * 8 + chunk_13, 0, 0, d_smem_addr + (unsigned int)(chunk_13 % 3 * 8192));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                if (has_hi_3 != 0) {
                                    unsigned int packed_1_1[128];
                                    #pragma unroll
                                    for (int chunk_14 = 0; chunk_14 < 8; chunk_14++) {
                                        #pragma unroll
                                        for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                            unsigned int address_32 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_14 * 32);
                                            float _tmem_load_32_18[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[15]))
                                                : "r"(address_32));
                                            #pragma unroll
                                            for (int pair_6 = 0; pair_6 < 8; pair_6++) {
                                                packed_1_1[chunk_14 * 16 + sub_3 * 8 + pair_6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[pair_6 * 2], _tmem_load_32_18[pair_6 * 2 + 1])));
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
                                    for (int chunk_15 = 0; chunk_15 < 8; chunk_15++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_27 = tid / 32;
                                        int lane_27 = tid % 32;
                                        #pragma unroll
                                        for (int half_36 = 0; half_36 < 2; half_36++) {
                                            #pragma unroll
                                            for (int col_tile_34 = 0; col_tile_34 < 2; col_tile_34++) {
                                                int row_40 = warp_27 * 32 + half_36 * 16 + lane_27 % 16;
                                                int col_40 = col_tile_34 * 16 + lane_27 / 16 * 8;
                                                unsigned int address_33 = d_smem_addr + (unsigned int)((8 + chunk_15) % 3 * 8192) + (unsigned int)((row_40 * 32 + col_40) * 2);
                                                address_33 = address_33 ^ (address_33 & 511) >> 7 << 4;
                                                int offset_29 = chunk_15 * 16 + half_36 * 8 + col_tile_34 * 4;
                                                uint32_t _stmatrix_addr_529 = static_cast<uint32_t>(address_33);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_529), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_29])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_29 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_29 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_29 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_5d((&y_routed), 0, x_3 * 256 + cta_rank_0 * 128, (y_3 * 2 + 1) * 8 + chunk_15, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_15) % 3 * 8192));
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
                        gemm_wide += iterations_3;
                        scales_wide += iterations_3;
                        gemm_out0 += 1;
                        gemm_acc_tiles += 1;
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
                        int task_3_1 = 0;
                        int macro_0 = 0;
                        int mini_0 = 0;
                        int swiglu_0 = 0;
                        if (shared_tasks <= decoded_cluster - comm_clusters) {
                            uint32_t fast_div_q_6 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks));
                            uint32_t fast_div_r_6 = (uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks)) - fast_div_q_6 * (uint32_t)(mini_tasks_div.divisor);
                            int ordered_mini_1 = (int)fast_div_q_6;
                            task_3_1 = (int)fast_div_r_6;
                            macro_0 = macros - 1;
                            mini_0 = ordered_mini_1;
                            if (ordered_mini_1 >= last_minis) {
                                uint32_t fast_div_q_1_1 = (minis_per_macro_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(ordered_mini_1 - last_minis)), minis_per_macro_div.multiplier) >> minis_per_macro_div.shift_right) : (uint32_t)((unsigned int)(ordered_mini_1 - last_minis));
                                uint32_t fast_div_r_1_1 = (uint32_t)((unsigned int)(ordered_mini_1 - last_minis)) - fast_div_q_1_1 * (uint32_t)(minis_per_macro_div.divisor);
                                macro_0 = macros - 2 - (int)fast_div_q_1_1;
                                mini_0 = (int)fast_div_r_1_1;
                            }
                            if (task_3_1 >= mini_gate && task_3_1 < mini_gate + mini_swiglu) {
                                swiglu_0 = 1;
                            }
                        }
                        int tile_word_7 = 0;
                        int tile_word_8 = 0;
                        int tile_word_9 = 0;
                        if (decoded_cluster < true_clusters) {
                            int word0_1 = 0;
                            int word1_1 = 0;
                            int word2_1 = 0;
                            if (shared_tasks <= decoded_cluster - comm_clusters) {
                                if (task_3_1 < mini_gate) {
                                    int gate_task_1 = task_3_1;
                                    int word0_4 = 0;
                                    int word1_3 = 0;
                                    int word2_4 = 0;
                                    int first_block_4 = (macro_0 * (macro_size / mini_size) + mini_0) * (mini_size / 256);
                                    int end_block_4 = ((first_block_4 + mini_size / 256) < (tokens / 256) ? (first_block_4 + mini_size / 256) : (tokens / 256));
                                    uint32_t fast_div_q_7 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)gate_task_1), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)gate_task_1);
                                    uint32_t fast_div_r_7 = (uint32_t)((unsigned int)gate_task_1) - fast_div_q_7 * (uint32_t)(gate_col_blocks_div.divisor);
                                    int block_17 = first_block_4 + (int)fast_div_q_7;
                                    if (block_17 < end_block_4) {
                                        word0_4 = counts[3 * experts + block_17];
                                        word1_3 = counts[experts + word0_4];
                                        word2_4 = counts[word0_4];
                                    }
                                    word0_1 = word0_4;
                                    word1_1 = word1_3;
                                    word2_1 = word2_4;
                                } else if (task_3_1 >= mini_gate + mini_swiglu) {
                                    int word0_5 = 0;
                                    int word1_4 = 0;
                                    int word2_5 = 0;
                                    int first_block_5 = (macro_0 * (macro_size / mini_size) + mini_0) * (mini_size / 256);
                                    int end_block_5 = ((first_block_5 + mini_size / 256) < (tokens / 256) ? (first_block_5 + mini_size / 256) : (tokens / 256));
                                    uint32_t fast_div_q_8 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(task_3_1 - mini_gate - mini_swiglu)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(task_3_1 - mini_gate - mini_swiglu));
                                    uint32_t fast_div_r_8 = (uint32_t)((unsigned int)(task_3_1 - mini_gate - mini_swiglu)) - fast_div_q_8 * (uint32_t)(down_col_blocks_div.divisor);
                                    int block_18 = first_block_5 + (int)fast_div_q_8;
                                    if (block_18 < end_block_5) {
                                        word0_5 = counts[3 * experts + block_18];
                                        word1_4 = counts[experts + word0_5];
                                        word2_5 = counts[word0_5];
                                    }
                                    word0_1 = word0_5;
                                    word1_1 = word1_4;
                                    word2_1 = word2_5;
                                }
                            }
                            tile_word_7 = word0_1;
                            tile_word_8 = word1_1;
                            tile_word_9 = word2_1;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_0_addr), "r"((unsigned int)decoded_cluster), "r"((unsigned int)swiglu_0), "r"((unsigned int)task_3_1), "r"((unsigned int)macro_0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_1_addr), "r"((unsigned int)mini_0), "r"(0), "r"((unsigned int)tile_word_7), "r"((unsigned int)tile_word_8) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_2_addr), "r"((unsigned int)tile_word_9), "r"(0), "r"(0), "r"(0) : "memory");
                        mbarrier_arrive(decode_ready_addr);
                    }
                }
            }
            int _parity_decode_ready = (int)((unsigned int)iteration % 2);
            mbarrier_wait(decode_ready_addr, _parity_decode_ready);
            unsigned int decode_quad[4];
            __int128_t _smem_raw_530;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_530) : "r"(decode_words_0_addr) : "memory");
            decode_quad[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_530)[0];
            decode_quad[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_530)[1];
            decode_quad[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_530)[2];
            decode_quad[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_530)[3];
            unsigned int decode_quad_0[4];
            __int128_t _smem_raw_531;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_531) : "r"(decode_words_1_addr) : "memory");
            decode_quad_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_531)[0];
            decode_quad_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_531)[1];
            decode_quad_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_531)[2];
            decode_quad_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_531)[3];
            unsigned int decode_quad_1[4];
            __int128_t _smem_raw_532;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_532) : "r"(decode_words_2_addr) : "memory");
            decode_quad_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_532)[0];
            decode_quad_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_532)[1];
            decode_quad_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_532)[2];
            decode_quad_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_532)[3];
            cluster = (int)decode_quad[0];
            int tile_word_10 = (int)decode_quad[1];
            int tile_word_11 = (int)decode_quad[2];
            int tile_word_12 = (int)decode_quad[3];
            int tile_word_13 = (int)decode_quad_0[0];
            __syncwarp();
            bool _elect_sync_0 = elect_sync();
            if (_elect_sync_0) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((schedule_finished_addr) & 0xFEFFFFFF) : "memory");
            }
            if (tile_word_3 != 0 && cluster >= 0 && tile_word_10 == 0) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            tile_word_0 = tile_word_11;
            tile_word_1 = tile_word_12;
            tile_word_2 = tile_word_13;
            tile_word_3 = tile_word_10;
            tile_word_4 = (int)decode_quad_0[2];
            tile_word_5 = (int)decode_quad_0[3];
            tile_word_6 = (int)decode_quad_1[0];
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
        if (macros > 0) {
            int col_blocks_10 = (hidden + 1023) / 1024;
            int macro_offset_4 = 0;
            int macro_tokens_2 = ((macro_size) < (tokens - macro_offset_4) ? (macro_size) : (tokens - macro_offset_4));
            int num_tasks_1 = (macro_tokens_2 / 16 * col_blocks_10 + 6) / 7;
            if (tid == 0) {
                unsigned int _atomic_fetch_add_533 = atomicAdd(&combine_next[0], 1);
                flat_view_8[0] = (int)_atomic_fetch_add_533;
            }
            __syncthreads();
            int task_4 = flat_view_8[0];
            __syncthreads();
            while (task_4 < num_tasks_1) {
                int col_blocks_5_1 = (hidden + 1023) / 1024;
                int first_tile_3 = task_4 * 7;
                int macro_offset_3_1 = 0;
                int macro_tokens_4 = ((macro_size) < (tokens - macro_offset_3_1) ? (macro_size) : (tokens - macro_offset_3_1));
                int valid_tiles_2 = ((7) < (macro_tokens_4 / 16 * col_blocks_5_1 - first_tile_3) ? (7) : (macro_tokens_4 / 16 * col_blocks_5_1 - first_tile_3));
                if (valid_tiles_2 > 0) {
                    int first_row_5 = first_tile_3 / col_blocks_5_1 * 16 + tid;
                    int first_col_3 = first_tile_3 % col_blocks_5_1;
                    int rows_4[7];
                    int columns_2[7];
                    int peers_2[7];
                    int tokens_0_2[7];
                    unsigned int counts_1_2[7];
                    int row_41 = first_row_5;
                    int column_2 = first_col_3;
                    #pragma unroll
                    for (int stage_3 = 0; stage_3 < 7; stage_3++) {
                        rows_4[stage_3] = row_41;
                        columns_2[stage_3] = column_2;
                        peers_2[stage_3] = -1;
                        tokens_0_2[stage_3] = -1;
                        if (valid_tiles_2 > stage_3 && tid < 16) {
                            peers_2[stage_3] = schedule_rank[macro_offset_3_1 + row_41];
                            tokens_0_2[stage_3] = schedule_token[macro_offset_3_1 + row_41];
                        }
                        counts_1_2[stage_3] = 0;
                        if (valid_tiles_2 > stage_3) {
                            if (stage_3 == 0 || column_2 == 0) {
                                uint32_t _barrier_vote_534 = __syncthreads_count(peers_2[stage_3] >= 0);
                                counts_1_2[stage_3] = _barrier_vote_534;
                            } else {
                                counts_1_2[stage_3] = counts_1_2[stage_3 - 1];
                            }
                        }
                        column_2 = column_2 + 1;
                        if (column_2 == col_blocks_5_1) {
                            column_2 = 0;
                            row_41 = row_41 + 16;
                        }
                    }
                    if (tid == 0) {
                        int first_mini_2 = (macro_offset_3_1 + first_row_5) / mini_size;
                        int last_mini_2 = (macro_offset_3_1 + (first_tile_3 + valid_tiles_2 - 1) / col_blocks_5_1 * 16) / mini_size;
                        #pragma unroll 1
                        for (int mini_1_1 = first_mini_2; mini_1_1 < last_mini_2 + 1; mini_1_1++) {
                            int mini_rows_9 = ((mini_size) < (tokens - mini_1_1 * mini_size) ? (mini_size) : (tokens - mini_1_1 * mini_size));
                            int required_9 = (mini_rows_9 + 255) / 256 * (hidden / 256) * 2;
                            int poll_12_first;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_12_first) : "l"(y_ready + mini_1_1) : "memory");
                            int poll_12 = poll_12_first;
                            while (poll_12 < required_9) {
                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                int poll_12_next;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_12_next) : "l"(y_ready + mini_1_1) : "memory");
                                poll_12 = poll_12_next;
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                        }
                        if (valid_tiles_2 > 0) {
                            int _min_535 = ((1024) < (hidden - columns_2[0] * 1024) ? (1024) : (hidden - columns_2[0] * 1024));
                            unsigned int chunk_bytes_28 = (unsigned int)(_min_535 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr, counts_1_2[0] * chunk_bytes_28);
                        }
                        if (valid_tiles_2 > 1) {
                            int _min_536 = ((1024) < (hidden - columns_2[1] * 1024) ? (1024) : (hidden - columns_2[1] * 1024));
                            unsigned int chunk_bytes_29 = (unsigned int)(_min_536 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 8, counts_1_2[1] * chunk_bytes_29);
                        }
                        if (valid_tiles_2 > 2) {
                            int _min_537 = ((1024) < (hidden - columns_2[2] * 1024) ? (1024) : (hidden - columns_2[2] * 1024));
                            unsigned int chunk_bytes_30 = (unsigned int)(_min_537 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 16, counts_1_2[2] * chunk_bytes_30);
                        }
                        if (valid_tiles_2 > 3) {
                            int _min_538 = ((1024) < (hidden - columns_2[3] * 1024) ? (1024) : (hidden - columns_2[3] * 1024));
                            unsigned int chunk_bytes_31 = (unsigned int)(_min_538 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 24, counts_1_2[3] * chunk_bytes_31);
                        }
                        if (valid_tiles_2 > 4) {
                            int _min_539 = ((1024) < (hidden - columns_2[4] * 1024) ? (1024) : (hidden - columns_2[4] * 1024));
                            unsigned int chunk_bytes_32 = (unsigned int)(_min_539 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 32, counts_1_2[4] * chunk_bytes_32);
                        }
                        if (valid_tiles_2 > 5) {
                            int _min_540 = ((1024) < (hidden - columns_2[5] * 1024) ? (1024) : (hidden - columns_2[5] * 1024));
                            unsigned int chunk_bytes_33 = (unsigned int)(_min_540 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 40, counts_1_2[5] * chunk_bytes_33);
                        }
                        if (valid_tiles_2 > 6) {
                            int _min_541 = ((1024) < (hidden - columns_2[6] * 1024) ? (1024) : (hidden - columns_2[6] * 1024));
                            unsigned int chunk_bytes_34 = (unsigned int)(_min_541 * 2);
                            mbarrier_arrive_expect_tx(combine_arrived_addr + 48, counts_1_2[6] * chunk_bytes_34);
                        }
                    }
                    __syncthreads();
                    if (peers_2[0] >= 0) {
                        int chunk_cols_16 = ((1024) < (hidden - columns_2[0] * 1024) ? (1024) : (hidden - columns_2[0] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)(tid * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[0] * (unsigned long long)hidden + (unsigned long long)(columns_2[0] * 1024)) * (unsigned long long)2), chunk_cols_16 * 2, combine_arrived_addr);
                    }
                    if (peers_2[1] >= 0) {
                        int chunk_cols_17 = ((1024) < (hidden - columns_2[1] * 1024) ? (1024) : (hidden - columns_2[1] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[1] * (unsigned long long)hidden + (unsigned long long)(columns_2[1] * 1024)) * (unsigned long long)2), chunk_cols_17 * 2, combine_arrived_addr + 8);
                    }
                    if (peers_2[2] >= 0) {
                        int chunk_cols_18 = ((1024) < (hidden - columns_2[2] * 1024) ? (1024) : (hidden - columns_2[2] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((32 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[2] * (unsigned long long)hidden + (unsigned long long)(columns_2[2] * 1024)) * (unsigned long long)2), chunk_cols_18 * 2, combine_arrived_addr + 16);
                    }
                    if (peers_2[3] >= 0) {
                        int chunk_cols_19 = ((1024) < (hidden - columns_2[3] * 1024) ? (1024) : (hidden - columns_2[3] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((48 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[3] * (unsigned long long)hidden + (unsigned long long)(columns_2[3] * 1024)) * (unsigned long long)2), chunk_cols_19 * 2, combine_arrived_addr + 24);
                    }
                    if (peers_2[4] >= 0) {
                        int chunk_cols_20 = ((1024) < (hidden - columns_2[4] * 1024) ? (1024) : (hidden - columns_2[4] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((64 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[4] * (unsigned long long)hidden + (unsigned long long)(columns_2[4] * 1024)) * (unsigned long long)2), chunk_cols_20 * 2, combine_arrived_addr + 32);
                    }
                    if (peers_2[5] >= 0) {
                        int chunk_cols_21 = ((1024) < (hidden - columns_2[5] * 1024) ? (1024) : (hidden - columns_2[5] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((80 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[5] * (unsigned long long)hidden + (unsigned long long)(columns_2[5] * 1024)) * (unsigned long long)2), chunk_cols_21 * 2, combine_arrived_addr + 40);
                    }
                    if (peers_2[6] >= 0) {
                        int chunk_cols_22 = ((1024) < (hidden - columns_2[6] * 1024) ? (1024) : (hidden - columns_2[6] * 1024));
                        cp_async_bulk_gmem2smem(flat_view_addr_7 + (unsigned int)((96 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_4[6] * (unsigned long long)hidden + (unsigned long long)(columns_2[6] * 1024)) * (unsigned long long)2), chunk_cols_22 * 2, combine_arrived_addr + 48);
                    }
                    if (valid_tiles_2 > 0) {
                        int _parity_combine_arrived_14 = (int)((unsigned int)combine_seq_0 % 2);
                        mbarrier_wait(combine_arrived_addr, _parity_combine_arrived_14);
                        combine_seq_0 += 1;
                        if (peers_2[0] >= 0) {
                            int _min_542 = ((1024) < (hidden - columns_2[0] * 1024) ? (1024) : (hidden - columns_2[0] * 1024));
                            unsigned int chunk_bytes_35 = (unsigned int)(_min_542 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_543 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[0]]) + ((unsigned long long)tokens_0_2[0] * (unsigned long long)hidden + (unsigned long long)(columns_2[0] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_543), "r"(flat_view_addr_7 + (unsigned int)(tid * 2048)), "r"((uint32_t)(chunk_bytes_35))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (valid_tiles_2 > 1) {
                        int _parity_combine_arrived_15 = (int)((unsigned int)combine_seq_1 % 2);
                        mbarrier_wait(combine_arrived_addr + 8, _parity_combine_arrived_15);
                        combine_seq_1 += 1;
                        if (peers_2[1] >= 0) {
                            int _min_544 = ((1024) < (hidden - columns_2[1] * 1024) ? (1024) : (hidden - columns_2[1] * 1024));
                            unsigned int chunk_bytes_36 = (unsigned int)(_min_544 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_545 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[1]]) + ((unsigned long long)tokens_0_2[1] * (unsigned long long)hidden + (unsigned long long)(columns_2[1] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_545), "r"(flat_view_addr_7 + (unsigned int)((16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_36))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (valid_tiles_2 > 2) {
                        int _parity_combine_arrived_16 = (int)((unsigned int)combine_seq_2 % 2);
                        mbarrier_wait(combine_arrived_addr + 16, _parity_combine_arrived_16);
                        combine_seq_2 += 1;
                        if (peers_2[2] >= 0) {
                            int _min_546 = ((1024) < (hidden - columns_2[2] * 1024) ? (1024) : (hidden - columns_2[2] * 1024));
                            unsigned int chunk_bytes_37 = (unsigned int)(_min_546 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_547 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[2]]) + ((unsigned long long)tokens_0_2[2] * (unsigned long long)hidden + (unsigned long long)(columns_2[2] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_547), "r"(flat_view_addr_7 + (unsigned int)((32 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_37))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (valid_tiles_2 > 3) {
                        int _parity_combine_arrived_17 = (int)((unsigned int)combine_seq_3 % 2);
                        mbarrier_wait(combine_arrived_addr + 24, _parity_combine_arrived_17);
                        combine_seq_3 += 1;
                        if (peers_2[3] >= 0) {
                            int _min_548 = ((1024) < (hidden - columns_2[3] * 1024) ? (1024) : (hidden - columns_2[3] * 1024));
                            unsigned int chunk_bytes_38 = (unsigned int)(_min_548 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_549 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[3]]) + ((unsigned long long)tokens_0_2[3] * (unsigned long long)hidden + (unsigned long long)(columns_2[3] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_549), "r"(flat_view_addr_7 + (unsigned int)((48 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_38))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (valid_tiles_2 > 4) {
                        int _parity_combine_arrived_18 = (int)((unsigned int)combine_seq_4 % 2);
                        mbarrier_wait(combine_arrived_addr + 32, _parity_combine_arrived_18);
                        combine_seq_4 += 1;
                        if (peers_2[4] >= 0) {
                            int _min_550 = ((1024) < (hidden - columns_2[4] * 1024) ? (1024) : (hidden - columns_2[4] * 1024));
                            unsigned int chunk_bytes_39 = (unsigned int)(_min_550 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_551 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[4]]) + ((unsigned long long)tokens_0_2[4] * (unsigned long long)hidden + (unsigned long long)(columns_2[4] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_551), "r"(flat_view_addr_7 + (unsigned int)((64 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_39))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (valid_tiles_2 > 5) {
                        int _parity_combine_arrived_19 = (int)((unsigned int)combine_seq_5 % 2);
                        mbarrier_wait(combine_arrived_addr + 40, _parity_combine_arrived_19);
                        combine_seq_5 += 1;
                        if (peers_2[5] >= 0) {
                            int _min_552 = ((1024) < (hidden - columns_2[5] * 1024) ? (1024) : (hidden - columns_2[5] * 1024));
                            unsigned int chunk_bytes_40 = (unsigned int)(_min_552 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_553 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[5]]) + ((unsigned long long)tokens_0_2[5] * (unsigned long long)hidden + (unsigned long long)(columns_2[5] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_553), "r"(flat_view_addr_7 + (unsigned int)((80 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_40))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (valid_tiles_2 > 6) {
                        int _parity_combine_arrived_20 = (int)((unsigned int)combine_seq_6 % 2);
                        mbarrier_wait(combine_arrived_addr + 48, _parity_combine_arrived_20);
                        combine_seq_6 += 1;
                        if (peers_2[6] >= 0) {
                            int _min_554 = ((1024) < (hidden - columns_2[6] * 1024) ? (1024) : (hidden - columns_2[6] * 1024));
                            unsigned int chunk_bytes_41 = (unsigned int)(_min_554 * 2);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_555 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers_2[6]]) + ((unsigned long long)tokens_0_2[6] * (unsigned long long)hidden + (unsigned long long)(columns_2[6] * 1024)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_555), "r"(flat_view_addr_7 + (unsigned int)((96 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_41))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    int warp_28 = tid / 32;
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    __syncthreads();
                }
                if (tid == 0) {
                    unsigned int _atomic_fetch_add_556 = atomicAdd(&combine_next[0], 1);
                    flat_view_8[0] = (int)_atomic_fetch_add_556;
                }
                __syncthreads();
                task_4 = flat_view_8[0];
                __syncthreads();
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
