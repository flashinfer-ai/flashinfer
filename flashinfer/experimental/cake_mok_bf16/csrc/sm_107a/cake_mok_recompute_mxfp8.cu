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
kernel_cake_mok_recompute_mxfp8(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap gate_shared_in, const __grid_constant__ CUtensorMap up_shared_in, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap x_q_store, const __grid_constant__ CUtensorMap x_sc_store, const __grid_constant__ CUtensorMap x_t_store, const __grid_constant__ CUtensorMap x_sc_t_store, const __grid_constant__ CUtensorMap x_q, const __grid_constant__ CUtensorMap x_sc, const __grid_constant__ CUtensorMap wg_q, const __grid_constant__ CUtensorMap wg_sc, const __grid_constant__ CUtensorMap wu_q, const __grid_constant__ CUtensorMap wu_sc, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_q_store, const __grid_constant__ CUtensorMap gate_sc_store, const __grid_constant__ CUtensorMap up_q_store, const __grid_constant__ CUtensorMap up_sc_store, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_q_store, const __grid_constant__ CUtensorMap hidden_sc_store, const __grid_constant__ CUtensorMap hidden_t_store, const __grid_constant__ CUtensorMap hidden_sc_t_store, unsigned long long* __restrict__ x_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, CakeFastDivmod mini_tasks_div, CakeFastDivmod gate_col_blocks_div)
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
    #define decode_ready_addr (mbar_base + 328)
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
    __nv_bfloat16* hidden_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_smem_addr = smem + 197632;
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
    __nv_bfloat16* smem_v15 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v15_addr = smem + 1024;
    __nv_bfloat16* smem_v16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v16_addr = smem + 1280;
    __nv_bfloat16* smem_v17 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_v17_addr = smem + 1536;
    __nv_bfloat16* smem_v18 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1792);
    const int smem_v18_addr = smem + 1792;
    unsigned int* smem_v19 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int smem_v19_addr = smem + 132096;
    unsigned int* smem_v20 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int smem_v20_addr = smem + 148480;
    unsigned int* smem_v21 = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int smem_v21_addr = smem + 164864;
    unsigned int* smem_v22 = reinterpret_cast<unsigned int*>(smem_raw + 165376);
    const int smem_v22_addr = smem + 165376;
    unsigned int* smem_v23 = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_v23_addr = smem + 165888;
    unsigned int* smem_v24 = reinterpret_cast<unsigned int*>(smem_raw + 182272);
    const int smem_v24_addr = smem + 182272;
    unsigned int* smem_v25 = reinterpret_cast<unsigned int*>(smem_raw + 198656);
    const int smem_v25_addr = smem + 198656;
    unsigned int* smem_v26 = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int smem_v26_addr = smem + 199168;
    __nv_bfloat16* smem_v27 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v27_addr = smem + 1024;
    __nv_bfloat16* smem_v28 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1280);
    const int smem_v28_addr = smem + 1280;
    __nv_bfloat16* smem_v29 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v29_addr = smem + 66560;
    __nv_bfloat16* smem_v30 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66816);
    const int smem_v30_addr = smem + 66816;
    uint8_t* smem_v31 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v31_addr = smem + 1024;
    uint8_t* smem_v32 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_v32_addr = smem + 66560;
    uint8_t* smem_v33 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v33_addr = smem + 132096;
    uint8_t* smem_v34 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v34_addr = smem + 197632;
    uint8_t* smem_v35 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v35_addr = smem + 199680;
    uint8_t* smem_v36 = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_v36_addr = smem + 203776;
    uint8_t* smem_v37 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v37_addr = smem + 1024;
    uint8_t* smem_v38 = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v38_addr = smem + 99328;
    uint8_t* smem_v39 = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_v39_addr = smem + 197632;
    uint8_t* smem_v40 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_v40_addr = smem + 200704;
    unsigned int* smem_v41 = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int smem_v41_addr = smem + 207872;
    unsigned int* smem_v42 = reinterpret_cast<unsigned int*>(smem_raw + 224256);
    const int smem_v42_addr = smem + 224256;
    unsigned int* smem_v43 = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_v43_addr = smem + 228352;
    unsigned int* smem_v44 = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_v44_addr = smem + 228864;

    // Kernel setup ops
    int tokens = num_tokens[0];
    int routed_tokens = ((tokens) < (macro_size) ? (tokens) : (macro_size));
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
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (48 barriers at smem_raw[0..704))

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
        // decode_ready: 1 barriers, init_count=1
        mbarrier_init(smem + 328, 1);
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
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        int col_blocks = (hidden + 511) / 512;
        int macro_offset_0 = 0;
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
                cp_async_bulk_gmem2smem(smem_v27_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + (unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(task % col_blocks * 512)) * (unsigned long long)2), _min_0 * 2, dispatch_arrived_addr);
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
                    cp_async_bulk_gmem2smem(smem_v29_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + (unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512) + 256ull) * (unsigned long long)2), hi_cols * 2, dispatch_arrived_hi_addr);
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
                        const void* _smem_vector_store_ptr_6 = reinterpret_cast<const void*>(smem_v23 + (base0 + group * 4));
                        uint32_t _smem_vector_store_addr_6 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_6));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_6), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]));
                        const void* _smem_vector_store_ptr_7 = reinterpret_cast<const void*>(smem_v23 + (base0 + (1 - group) * 4));
                        uint32_t _smem_vector_store_addr_7 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_7));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_7), "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]));
                        const void* _smem_vector_store_ptr_8 = reinterpret_cast<const void*>(smem_v23 + (base1 + group * 4));
                        uint32_t _smem_vector_store_addr_8 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_8));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_8), "r"(words_0[0]), "r"(words_0[1]), "r"(words_0[2]), "r"(words_0[3]));
                        const void* _smem_vector_store_ptr_9 = reinterpret_cast<const void*>(smem_v23 + (base1 + (1 - group) * 4));
                        uint32_t _smem_vector_store_addr_9 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_9));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_9), "r"(words_0[4]), "r"(words_0[5]), "r"(words_0[6]), "r"(words_0[7]));
                        {
                            uint32_t _byte_10 = static_cast<uint32_t>(scale_byte) & 0xFFu;
                            uint32_t _addr_10 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0 + k_block));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_10), "r"(_byte_10) : "memory");
                        }
                        {
                            uint32_t _byte_11 = static_cast<uint32_t>(scale_byte_0) & 0xFFu;
                            uint32_t _addr_11 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1 + k_block));
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
                            const void* _smem_vector_store_ptr_15 = reinterpret_cast<const void*>(smem_v19 + (row_0 * 32 + k_block_1 * 8 + (i_7 + rot) % 4 * 2));
                            uint32_t _smem_vector_store_addr_15 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_15));
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_15), "r"(words_1[i_7 * 2]), "r"(words_1[i_7 * 2 + 1]));
                        }
                    }
                    smem_v21[row_0 % 32 * 4 + row_0 / 32] = scale_word;
                }
                __syncthreads();
                if (tid == 0) {
                    int col_tile_0 = col_block * 4;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d((&x_q_store), col_tile_0 * 128, row, smem_v19_addr);
                    tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile_0, smem_v21_addr);
                    tma_store_2d((&x_t_store), row, col_tile_0 * 128, smem_v23_addr);
                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_0 * macro_tiles + row_tile, smem_v25_addr);
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
                        const void* _smem_vector_store_ptr_22 = reinterpret_cast<const void*>(smem_v24 + (base0_1 + group_1 * 4));
                        uint32_t _smem_vector_store_addr_22 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_22));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_22), "r"(words_2[0]), "r"(words_2[1]), "r"(words_2[2]), "r"(words_2[3]));
                        const void* _smem_vector_store_ptr_23 = reinterpret_cast<const void*>(smem_v24 + (base0_1 + (1 - group_1) * 4));
                        uint32_t _smem_vector_store_addr_23 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_23));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_23), "r"(words_2[4]), "r"(words_2[5]), "r"(words_2[6]), "r"(words_2[7]));
                        const void* _smem_vector_store_ptr_24 = reinterpret_cast<const void*>(smem_v24 + (base1_1 + group_1 * 4));
                        uint32_t _smem_vector_store_addr_24 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_24));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_24), "r"(words_1_1[0]), "r"(words_1_1[1]), "r"(words_1_1[2]), "r"(words_1_1[3]));
                        const void* _smem_vector_store_ptr_25 = reinterpret_cast<const void*>(smem_v24 + (base1_1 + (1 - group_1) * 4));
                        uint32_t _smem_vector_store_addr_25 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_25));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_25), "r"(words_1_1[4]), "r"(words_1_1[5]), "r"(words_1_1[6]), "r"(words_1_1[7]));
                        {
                            uint32_t _byte_26 = static_cast<uint32_t>(scale_byte_2) & 0xFFu;
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_v26_addr + (unsigned int)(scale_base0_1 + k_block_2));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_26), "r"(_byte_26) : "memory");
                        }
                        {
                            uint32_t _byte_27 = static_cast<uint32_t>(scale_byte_1_1) & 0xFFu;
                            uint32_t _addr_27 = static_cast<uint32_t>(smem_v26_addr + (unsigned int)(scale_base1_1 + k_block_2));
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
                            const void* _smem_vector_store_ptr_31 = reinterpret_cast<const void*>(smem_v20 + (row_1 * 32 + k_block_3 * 8 + (i_15 + rot_1) % 4 * 2));
                            uint32_t _smem_vector_store_addr_31 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_31));
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_31), "r"(words_3[i_15 * 2]), "r"(words_3[i_15 * 2 + 1]));
                        }
                    }
                    smem_v22[row_1 % 32 * 4 + row_1 / 32] = scale_word_1;
                }
                __syncthreads();
                if (tid == 0) {
                    int col_tile_1 = col_block * 4 + 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    tma_store_2d((&x_q_store), col_tile_1 * 128, row, smem_v20_addr);
                    tma_store_3d((&x_sc_store), 0, 0, row_tile * k_tiles + col_tile_1, smem_v22_addr);
                    tma_store_2d((&x_t_store), row, col_tile_1 * 128, smem_v24_addr);
                    tma_store_3d((&x_sc_t_store), 0, 0, col_tile_1 * macro_tiles + row_tile, smem_v26_addr);
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
                    cp_async_bulk_gmem2smem(smem_v27_addr + (unsigned int)(tid * 256 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + (unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(next_task % col_blocks * 512)) * (unsigned long long)2), _min_1 * 2, dispatch_arrived_addr);
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
                            const void* _smem_vector_store_ptr_38 = reinterpret_cast<const void*>(smem_v23 + (base0_2 + group_2 * 4));
                            uint32_t _smem_vector_store_addr_38 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_38));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_38), "r"(words_4[0]), "r"(words_4[1]), "r"(words_4[2]), "r"(words_4[3]));
                            const void* _smem_vector_store_ptr_39 = reinterpret_cast<const void*>(smem_v23 + (base0_2 + (1 - group_2) * 4));
                            uint32_t _smem_vector_store_addr_39 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_39));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_39), "r"(words_4[4]), "r"(words_4[5]), "r"(words_4[6]), "r"(words_4[7]));
                            const void* _smem_vector_store_ptr_40 = reinterpret_cast<const void*>(smem_v23 + (base1_2 + group_2 * 4));
                            uint32_t _smem_vector_store_addr_40 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_40));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_40), "r"(words_2_1[0]), "r"(words_2_1[1]), "r"(words_2_1[2]), "r"(words_2_1[3]));
                            const void* _smem_vector_store_ptr_41 = reinterpret_cast<const void*>(smem_v23 + (base1_2 + (1 - group_2) * 4));
                            uint32_t _smem_vector_store_addr_41 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_41));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_41), "r"(words_2_1[4]), "r"(words_2_1[5]), "r"(words_2_1[6]), "r"(words_2_1[7]));
                            {
                                uint32_t _byte_42 = static_cast<uint32_t>(scale_byte_4) & 0xFFu;
                                uint32_t _addr_42 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base0_2 + k_block_4));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_42), "r"(_byte_42) : "memory");
                            }
                            {
                                uint32_t _byte_43 = static_cast<uint32_t>(scale_byte_2_1) & 0xFFu;
                                uint32_t _addr_43 = static_cast<uint32_t>(smem_v25_addr + (unsigned int)(scale_base1_2 + k_block_4));
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
                                const void* _smem_vector_store_ptr_47 = reinterpret_cast<const void*>(smem_v19 + (row_2 * 32 + k_block_5 * 8 + (i_23 + rot_2) % 4 * 2));
                                uint32_t _smem_vector_store_addr_47 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_47));
                                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_47), "r"(words_5[i_23 * 2]), "r"(words_5[i_23 * 2 + 1]));
                            }
                        }
                        smem_v21[row_2 % 32 * 4 + row_2 / 32] = scale_word_2;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_2 = col_block * 4 + 2;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_2 * 128, row, smem_v19_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile_0 * k_tiles_0 + col_tile_2, smem_v21_addr);
                        tma_store_2d((&x_t_store), row, col_tile_2 * 128, smem_v23_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_2 * macro_tiles_0 + row_tile_0, smem_v25_addr);
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
                            const void* _smem_vector_store_ptr_54 = reinterpret_cast<const void*>(smem_v24 + (base0_3 + group_3 * 4));
                            uint32_t _smem_vector_store_addr_54 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_54));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_54), "r"(words_6[0]), "r"(words_6[1]), "r"(words_6[2]), "r"(words_6[3]));
                            const void* _smem_vector_store_ptr_55 = reinterpret_cast<const void*>(smem_v24 + (base0_3 + (1 - group_3) * 4));
                            uint32_t _smem_vector_store_addr_55 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_55));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_55), "r"(words_6[4]), "r"(words_6[5]), "r"(words_6[6]), "r"(words_6[7]));
                            const void* _smem_vector_store_ptr_56 = reinterpret_cast<const void*>(smem_v24 + (base1_3 + group_3 * 4));
                            uint32_t _smem_vector_store_addr_56 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_56));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_56), "r"(words_3_1[0]), "r"(words_3_1[1]), "r"(words_3_1[2]), "r"(words_3_1[3]));
                            const void* _smem_vector_store_ptr_57 = reinterpret_cast<const void*>(smem_v24 + (base1_3 + (1 - group_3) * 4));
                            uint32_t _smem_vector_store_addr_57 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_57));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(_smem_vector_store_addr_57), "r"(words_3_1[4]), "r"(words_3_1[5]), "r"(words_3_1[6]), "r"(words_3_1[7]));
                            {
                                uint32_t _byte_58 = static_cast<uint32_t>(scale_byte_6) & 0xFFu;
                                uint32_t _addr_58 = static_cast<uint32_t>(smem_v26_addr + (unsigned int)(scale_base0_3 + k_block_6));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                            }
                            {
                                uint32_t _byte_59 = static_cast<uint32_t>(scale_byte_3_1) & 0xFFu;
                                uint32_t _addr_59 = static_cast<uint32_t>(smem_v26_addr + (unsigned int)(scale_base1_3 + k_block_6));
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
                                const void* _smem_vector_store_ptr_63 = reinterpret_cast<const void*>(smem_v20 + (row_3 * 32 + k_block_7 * 8 + (i_31 + rot_3) % 4 * 2));
                                uint32_t _smem_vector_store_addr_63 = static_cast<uint32_t>(__cvta_generic_to_shared(_smem_vector_store_ptr_63));
                                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(_smem_vector_store_addr_63), "r"(words_7[i_31 * 2]), "r"(words_7[i_31 * 2 + 1]));
                            }
                        }
                        smem_v22[row_3 % 32 * 4 + row_3 / 32] = scale_word_3;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        int col_tile_3 = col_block * 4 + 2 + 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_2d((&x_q_store), col_tile_3 * 128, row, smem_v20_addr);
                        tma_store_3d((&x_sc_store), 0, 0, row_tile_0 * k_tiles_0 + col_tile_3, smem_v22_addr);
                        tma_store_2d((&x_t_store), row, col_tile_3 * 128, smem_v24_addr);
                        tma_store_3d((&x_sc_t_store), 0, 0, col_tile_3 * macro_tiles_0 + row_tile_0, smem_v26_addr);
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
    } else {
        int iteration = 0;
        int mini = 0;
        int task_1 = 0;
        int swiglu = 0;
        if (shared_tasks <= cluster - comm_clusters) {
            uint32_t fast_div_q = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks));
            uint32_t fast_div_r = (uint32_t)((unsigned int)(cluster - comm_clusters - shared_tasks)) - fast_div_q * (uint32_t)(mini_tasks_div.divisor);
            mini = (int)fast_div_q;
            task_1 = (int)fast_div_r;
            if (task_1 >= 2 * mini_gate && task_1 < 2 * mini_gate + mini_swiglu) {
                swiglu = 1;
            }
        }
        if (shared_tasks > cluster - comm_clusters && cluster - comm_clusters >= 2 * shared_gate_tasks && cluster - comm_clusters < 2 * shared_gate_tasks + shared_swiglu) {
            swiglu = 1;
        }
        int tile_word_0 = mini;
        int tile_word_1 = task_1;
        int tile_word_2 = swiglu;
        int tile_word_3 = 0;
        int tile_word_4 = 0;
        int tile_word_5 = 0;
        if (cluster < true_clusters) {
            int word0 = 0;
            int word1 = 0;
            int word2 = 0;
            if (shared_tasks <= cluster - comm_clusters) {
                if (task_1 < 2 * mini_gate) {
                    int gate_task = task_1;
                    if (task_1 >= mini_gate) {
                        gate_task = task_1 - mini_gate;
                    }
                    int word0_2 = 0;
                    int word1_0 = 0;
                    int word2_0 = 0;
                    int first_block = mini * (mini_size / 256);
                    int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    uint32_t fast_div_q_1 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)gate_task), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)gate_task);
                    uint32_t fast_div_r_1 = (uint32_t)((unsigned int)gate_task) - fast_div_q_1 * (uint32_t)(gate_col_blocks_div.divisor);
                    int block = first_block + (int)fast_div_q_1;
                    if (block < end_block) {
                        word0_2 = counts[3 * experts + block];
                        word1_0 = counts[experts + word0_2];
                        word2_0 = counts[word0_2];
                    }
                    word0 = word0_2;
                    word1 = word1_0;
                    word2 = word2_0;
                }
            }
            tile_word_3 = word0;
            tile_word_4 = word1;
            tile_word_5 = word2;
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
                    int row_4 = 0;
                    int col_4 = 0;
                    if (compute < row_blocks * full_cols) {
                        row_4 = compute % (row_blocks * 8) / 8;
                        col_4 = supergroup * 8 + compute % 8;
                    } else {
                        row_4 = (compute - row_blocks * full_cols) / (col_blocks_1 - full_cols);
                        col_4 = full_cols + (compute - row_blocks * full_cols) % (col_blocks_1 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_4 = row_blocks - row_4 - 1;
                    }
                    x = row_4;
                    y = col_4;
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
                                    int _min_64 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int mini_rows = ((0) > (_min_64) ? (0) : (_min_64));
                                    int required = (mini_rows + 127) / 128 * ((hidden + 511) / 512);
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
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_shared)), "r"(0), "r"(y * 2 * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_shared)), "r"(0), "r"((y * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx), "r"(expert), "r"(0),
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
                                    int _smem_desc_lo_65 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_66 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_65), "r"(_smem_desc_lo_66), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_67 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_68 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_67), "r"(_smem_desc_lo_68), "r"((tmem_accumulator + (256))), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
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
                            for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                #pragma unroll
                                for (int sub = 0; sub < 2; sub++) {
                                    unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                    float tmem_load[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15]))
                                        : "r"(address));
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; pair++) {
                                        packed[chunk_4 * 16 + sub * 8 + pair] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load[pair * 2], tmem_load[pair * 2 + 1])));
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
                                int _min_69 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                if (output_row < _min_69) {
                                }
                            }
                            #pragma unroll
                            for (int chunk_5 = 0; chunk_5 < 8; chunk_5++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_0 = tid / 32;
                                int lane_1 = tid % 32;
                                #pragma unroll
                                for (int half_4 = 0; half_4 < 2; half_4++) {
                                    #pragma unroll
                                    for (int col_tile = 0; col_tile < 2; col_tile++) {
                                        int row_5 = warp_0 * 32 + half_4 * 16 + lane_1 % 16;
                                        int col_5 = col_tile * 16 + lane_1 / 16 * 8;
                                        unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_5 * 32 + col_5) * 2);
                                        address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                        int offset = chunk_5 * 16 + half_4 * 8 + col_tile * 4;
                                        uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(address_1);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 2 * 8 + chunk_5), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (has_hi != 0) {
                                unsigned int packed_0[128];
                                #pragma unroll
                                for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                    #pragma unroll
                                    for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                        unsigned int address_2 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                        float tmem_load_1[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15]))
                                            : "r"(address_2));
                                        #pragma unroll
                                        for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                            packed_0[chunk_6 * 16 + sub_1 * 8 + pair_1] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_1[pair_1 * 2], tmem_load_1[pair_1 * 2 + 1])));
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
                                for (int chunk_7 = 0; chunk_7 < 8; chunk_7++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_1 = tid / 32;
                                    int lane_2 = tid % 32;
                                    #pragma unroll
                                    for (int half_5 = 0; half_5 < 2; half_5++) {
                                        #pragma unroll
                                        for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                            int row_6 = warp_0_1 * 32 + half_5 * 16 + lane_2 % 16;
                                            int col_6 = col_tile_4 * 16 + lane_2 / 16 * 8;
                                            unsigned int address_3 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_6 * 32 + col_6) * 2);
                                            address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                            int offset_1 = chunk_7 * 16 + half_5 * 8 + col_tile_4 * 4;
                                            uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(address_3);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_1 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"((y * 2 + 1) * 8 + chunk_7), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                    gemm_wide += iterations;
                    gemm_out0 += 1;
                    gemm_acc_tiles += 1;
                }
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
                    int row_7 = 0;
                    int col_7 = 0;
                    if (compute - shared_gate_tasks < row_blocks_1 * full_cols_1) {
                        row_7 = (compute - shared_gate_tasks) % (row_blocks_1 * 8) / 8;
                        col_7 = supergroup_1 * 8 + (compute - shared_gate_tasks) % 8;
                    } else {
                        row_7 = (compute - shared_gate_tasks - row_blocks_1 * full_cols_1) / (col_blocks_2 - full_cols_1);
                        col_7 = full_cols_1 + (compute - shared_gate_tasks - row_blocks_1 * full_cols_1) % (col_blocks_2 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_7 = row_blocks_1 - row_7 - 1;
                    }
                    x_1 = row_7;
                    y_1 = col_7;
                    expert_1 = 0;
                }
                int has_hi_1 = 0;
                has_hi_1 = (int)((y_1 * 2 + 1) * 256 < intermediate);
                int global_mini_1 = 0;
                int macro_rows_1 = 0;
                int iterations_1 = hidden / 64;
                if (expert_1 < 0) {
                    if (tid == 0) {
                    }
                } else {
                    if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_72 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                    int mini_rows_1 = ((0) > (_min_72) ? (0) : (_min_72));
                                    int required_1 = (mini_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                                }
                                #pragma unroll 1
                                for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                    int _stage_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_2) % 4);
                                    int _parity_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_2) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_1) * 8, _parity_gemm_finished_1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&x_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wu_shared)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_2) % 4 * 16384)), "l"((&wu_shared)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
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
                                    int _smem_desc_lo_73 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_74 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_73), "r"(_smem_desc_lo_74), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_75 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_76 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
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
            :: "r"(_smem_desc_lo_75), "r"(_smem_desc_lo_76), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else {
                        if (tid < 128) {
                            int _parity_output_arrived_1 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_1);
                            unsigned int packed_1[128];
                            #pragma unroll
                            for (int chunk_8 = 0; chunk_8 < 8; chunk_8++) {
                                #pragma unroll
                                for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                    unsigned int address_4 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_8 * 32);
                                    float tmem_load_2[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15]))
                                        : "r"(address_4));
                                    #pragma unroll
                                    for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                                        packed_1[chunk_8 * 16 + sub_2 * 8 + pair_2] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_2[pair_2 * 2], tmem_load_2[pair_2 * 2 + 1])));
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
                                int _min_77 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                                if (output_row_1 < _min_77) {
                                }
                            }
                            #pragma unroll
                            for (int chunk_9 = 0; chunk_9 < 8; chunk_9++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_0_2 = tid / 32;
                                int lane_3 = tid % 32;
                                #pragma unroll
                                for (int half_6 = 0; half_6 < 2; half_6++) {
                                    #pragma unroll
                                    for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                        int row_8 = warp_0_2 * 32 + half_6 * 16 + lane_3 % 16;
                                        int col_8 = col_tile_5 * 16 + lane_3 / 16 * 8;
                                        unsigned int address_5 = d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192) + (unsigned int)((row_8 * 32 + col_8) * 2);
                                        address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                        int offset_2 = chunk_9 * 16 + half_6 * 8 + col_tile_5 * 4;
                                        uint32_t _stmatrix_addr_78 = static_cast<uint32_t>(address_5);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_2 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_shared_out)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 2 * 8 + chunk_9), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (has_hi_1 != 0) {
                                unsigned int packed_0_1[128];
                                #pragma unroll
                                for (int chunk_10 = 0; chunk_10 < 8; chunk_10++) {
                                    #pragma unroll
                                    for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                        unsigned int address_6 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_10 * 32);
                                        float tmem_load_1_1[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_1[15]))
                                            : "r"(address_6));
                                        #pragma unroll
                                        for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                            packed_0_1[chunk_10 * 16 + sub_3 * 8 + pair_3] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_1_1[pair_3 * 2], tmem_load_1_1[pair_3 * 2 + 1])));
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
                                for (int chunk_11 = 0; chunk_11 < 8; chunk_11++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_3 = tid / 32;
                                    int lane_4 = tid % 32;
                                    #pragma unroll
                                    for (int half_7 = 0; half_7 < 2; half_7++) {
                                        #pragma unroll
                                        for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                            int row_9 = warp_0_3 * 32 + half_7 * 16 + lane_4 % 16;
                                            int col_9 = col_tile_6 * 16 + lane_4 / 16 * 8;
                                            unsigned int address_7 = d_smem_addr + (unsigned int)((8 + chunk_11) % 3 * 8192) + (unsigned int)((row_9 * 32 + col_9) * 2);
                                            address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                            int offset_3 = chunk_11 * 16 + half_7 * 8 + col_tile_6 * 4;
                                            uint32_t _stmatrix_addr_79 = static_cast<uint32_t>(address_7);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_79), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_3 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_shared_out)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"((y_1 * 2 + 1) * 8 + chunk_11), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_11) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                            bool enabled_value_1_1 = 1;
                                            if (enabled_value_1_1 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows_1 + x_1) * (intermediate / 256) + y_1 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
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
                if (compute < shared_tasks) {
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
                                    int slot = (swiglu_seq + stage) % 3;
                                    int row_10 = first_row;
                                    int col_10 = first_col + stage;
                                    if (col_10 >= col_blocks_3) {
                                        row_10 = row_10 + 1;
                                        col_10 = col_10 - col_blocks_3;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage) % 3)) * 8, 65536);
                                    int parent = row_10 / 2 * (intermediate / 256) + col_10 / 2;
                                    int poll_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_first) : "l"(gate_ready + parent) : "memory");
                                    int poll = poll_first;
                                    while (poll < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_next) : "l"(gate_ready + parent) : "memory");
                                        poll = poll_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + (unsigned int)(slot * 32768)), "l"((&gate_shared_in)), "r"(0), "r"((row_10 - macro_row_offset) * 128), "r"(col_10 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage) % 3)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + (unsigned int)(slot * 32768)), "l"((&up_shared_in)), "r"(0), "r"((row_10 - macro_row_offset) * 128), "r"(col_10 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + stage) % 3)) * 8) : "memory");
                                }
                            }
                        }
                        #pragma unroll 3
                        for (int stage_1 = 0; stage_1 < 3; stage_1++) {
                            if (tile_end > first_tile + stage_1) {
                                int slot_1 = (swiglu_seq + stage_1) % 3;
                                int _stage_swiglu_arrived = (int)((unsigned int)(swiglu_seq + stage_1) % 3);
                                int _parity_swiglu_arrived = (int)((unsigned int)(swiglu_seq + stage_1) / 3 % 2);
                                mbarrier_wait(swiglu_arrived_addr + (_stage_swiglu_arrived) * 8, _parity_swiglu_arrived);
                                int row_11 = first_row;
                                int col_11 = first_col + stage_1;
                                if (col_11 >= col_blocks_3) {
                                    row_11 = row_11 + 1;
                                    col_11 = col_11 - col_blocks_3;
                                }
                                float gate[64];
                                float up[64];
                                float denominator[64];
                                int warp_1 = tid / 32;
                                int local_warp = warp_1 / 4 + warp_1 % 4 * 2;
                                int lane_5 = tid % 32;
                                #pragma unroll
                                for (int tile_col = 0; tile_col < 8; tile_col++) {
                                    unsigned int packed_2[4];
                                    unsigned int address_8 = gate_smem_addr + (unsigned int)(slot_1 * 32768) + (unsigned int)(((tile_col * 16 + lane_5 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_5 % 16) * 64 + (tile_col * 16 + lane_5 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                        : "r"(address_8 ^ (address_8 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_2[pair_4]));
                                        gate[tile_col * 8 + pair_4 * 2] = cvt_f32.x;
                                        gate[tile_col * 8 + pair_4 * 2 + 1] = cvt_f32.y;
                                    }
                                }
                                int warp_0_4 = tid / 32;
                                int local_warp_0 = warp_0_4 / 4 + warp_0_4 % 4 * 2;
                                int lane_0 = tid % 32;
                                #pragma unroll
                                for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                    unsigned int packed_3[4];
                                    unsigned int address_9 = up_smem_addr + (unsigned int)(slot_1 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_0 / 16 * 8) / 64 * 128 * 64 + (local_warp_0 * 16 + lane_0 % 16) * 64 + (tile_col_1 * 16 + lane_0 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                        : "r"(address_9 ^ (address_9 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_3[pair_5]));
                                        up[tile_col_1 * 8 + pair_5 * 2] = cvt_f32_1.x;
                                        up[tile_col_1 * 8 + pair_5 * 2 + 1] = cvt_f32_1.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem = 0; elem < 64; elem++) {
                                    denominator[elem] = gate[elem] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                    float _exp_80 = expf(denominator[elem_1]);
                                    denominator[elem_1] = _exp_80;
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
                                        packed_4[pair_6] = __as_u32(__float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_6 * 2], gate[tile_col_2 * 8 + pair_6 * 2 + 1])));
                                    }
                                    unsigned int address_10 = hidden_smem_addr + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(address_10 ^ (address_10 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&hidden_shared_out), 0, (row_11 - macro_row_offset) * 128, col_11 * 2, 0, 0, hidden_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 3; stage_2++) {
                                if (tile_end > first_tile + stage_2) {
                                    int row_12 = first_row;
                                    if (col_blocks_3 <= first_col + stage_2) {
                                        row_12 = row_12 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (row_12 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        int _min_82 = ((3) < (tile_end - first_tile) ? (3) : (tile_end - first_tile));
                        swiglu_seq += _min_82;
                    }
                } else if (tile_word_1 < mini_gate) {
                    int col_blocks_4 = (intermediate + 256 - 1) / 256;
                    int x_2 = -1;
                    int y_2 = -1;
                    int expert_2 = -1;
                    int k_start_2 = 0;
                    int k_end_2 = 0;
                    int first_2 = 0;
                    int first_block_1 = tile_word_0 * (mini_size / 256);
                    int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int block_1 = first_block_1;
                    uint32_t fast_div_q_2 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)tile_word_1), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)tile_word_1);
                    uint32_t fast_div_r_2 = (uint32_t)((unsigned int)tile_word_1) - fast_div_q_2 * (uint32_t)(gate_col_blocks_div.divisor);
                    block_1 = first_block_1 + (int)fast_div_q_2;
                    if (block_1 < end_block_1) {
                        int index = tile_word_3;
                        int offset_4 = tile_word_4 / 256;
                        int first_row_1 = ((first_block_1) > (offset_4) ? (first_block_1) : (offset_4));
                        int _min_83 = ((end_block_1) < (offset_4 + tile_word_5 / 256) ? (end_block_1) : (offset_4 + tile_word_5 / 256));
                        int rows = _min_83 - first_row_1;
                        int supergroup_2 = (tile_word_1 - (first_row_1 - first_block_1) * col_blocks_4) / (rows * 8);
                        int full_cols_2 = col_blocks_4 / 8 * 8;
                        int row_13 = 0;
                        int col_12 = 0;
                        if (tile_word_1 - (first_row_1 - first_block_1) * col_blocks_4 < rows * full_cols_2) {
                            row_13 = (tile_word_1 - (first_row_1 - first_block_1) * col_blocks_4) % (rows * 8) / 8;
                            col_12 = supergroup_2 * 8 + (tile_word_1 - (first_row_1 - first_block_1) * col_blocks_4) % 8;
                        } else {
                            row_13 = (tile_word_1 - (first_row_1 - first_block_1) * col_blocks_4 - rows * full_cols_2) / (col_blocks_4 - full_cols_2);
                            col_12 = full_cols_2 + (tile_word_1 - (first_row_1 - first_block_1) * col_blocks_4 - rows * full_cols_2) % (col_blocks_4 - full_cols_2);
                        }
                        if ((supergroup_2 & 1) != 0) {
                            row_13 = rows - row_13 - 1;
                        }
                        x_2 = first_row_1 + row_13;
                        y_2 = col_12;
                        expert_2 = index;
                    }
                    int has_hi_2 = 0;
                    int global_mini_2 = tile_word_0;
                    int macro_rows_2 = 0;
                    int iterations_2 = hidden / 128;
                    int macro_k = 0;
                    if (expert_2 < 0) {
                        if (tid == 0) {
                        }
                    } else {
                        if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_84 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                        int mini_rows_2 = ((0) > (_min_84) ? (0) : (_min_84));
                                        int required_2 = (mini_rows_2 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_2 = 1;
                                        if (enabled_value_2 != 0) {
                                            int poll_1_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_first) : "l"(x_ready + global_mini_2) : "memory");
                                            int poll_1 = poll_1_first;
                                            while (poll_1 < required_2) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_1_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_next) : "l"(x_ready + global_mini_2) : "memory");
                                                poll_1 = poll_1_next;
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
                                            :: "r"(smem_v31_addr + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(smem_v32_addr + (unsigned int)((gemm_wide + idx_4) % 4 * 16384)), "l"((&wg_q)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        } else if (tid / 32 == 6) {
                            if (warp == 6) {
                                if (elect_sync()) {
                                    {
                                        int _min_85 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                        int mini_rows_3 = ((0) > (_min_85) ? (0) : (_min_85));
                                        int required_3 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_3 = 1;
                                        if (enabled_value_3 != 0) {
                                            int poll_2_first;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_first) : "l"(x_ready + global_mini_2) : "memory");
                                            int poll_2 = poll_2_first;
                                            while (poll_2 < required_3) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int poll_2_next;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_next) : "l"(x_ready + global_mini_2) : "memory");
                                                poll_2 = poll_2_next;
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
                                            :: "r"(smem_v34_addr + (unsigned int)((scales_wide + idx_5) % 4 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
                                               "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_5) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(smem_v35_addr + (unsigned int)((scales_wide + idx_5) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc)), "r"(0), "r"(0), "r"((expert_2 * (intermediate / 128) + y_2 * 2 + cta_rank_0) * (hidden / 128) + idx_5),
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
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_6) % 4)) * 8, 3072);
                                            int _stage_scales_arrived = (int)((unsigned int)(scales_wide + idx_6) % 4);
                                            int _parity_scales_arrived = (int)((unsigned int)(scales_wide + idx_6) / 4 % 2);
                                            mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived) * 8, _parity_scales_arrived);
                                            int buffer = idx_6 % 3;
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer * 4, make_sf_cp_desc_sbo128(smem_v34_addr + (unsigned int)((scales_wide + idx_6) % 4 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer * 8, make_sf_cp_desc_sbo128(smem_v35_addr + (unsigned int)((scales_wide + idx_6) % 4 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer * 8 + 4), make_sf_cp_desc_sbo128((smem_v35_addr + (unsigned int)((scales_wide + idx_6) % 4 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_6) % 4)) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8, 65536);
                                            int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_6) % 4);
                                            int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_6) / 4 % 2);
                                            mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
                                            int _smem_desc_lo_86 = (((smem_v31_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_6) % 4) * 1024;
                                            int _smem_desc_lo_87 = (((smem_v32_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_6) % 4) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)_smem_desc_lo_86) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)_smem_desc_lo_87) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    0x90c00000U, (int)((tmem_tmem_sfa + buffer * 4 + 0)), (int)((tmem_tmem_sfb + buffer * 8 + 0)), ((idx_6 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    0xd0c00020U, (int)((tmem_tmem_sfa + buffer * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer * 8 + 0) | 0x80000000), 1);
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
                                unsigned int packed_5[128];
                                #pragma unroll
                                for (int i_32 = 0; i_32 < 8; i_32++) {
                                    float tmem_load_3[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(tmem_load_3[0]), "=f"(tmem_load_3[1]), "=f"(tmem_load_3[2]), "=f"(tmem_load_3[3]), "=f"(tmem_load_3[4]), "=f"(tmem_load_3[5]), "=f"(tmem_load_3[6]), "=f"(tmem_load_3[7]), "=f"(tmem_load_3[8]), "=f"(tmem_load_3[9]), "=f"(tmem_load_3[10]), "=f"(tmem_load_3[11]), "=f"(tmem_load_3[12]), "=f"(tmem_load_3[13]), "=f"(tmem_load_3[14]), "=f"(tmem_load_3[15]), "=f"(tmem_load_3[16]), "=f"(tmem_load_3[17]), "=f"(tmem_load_3[18]), "=f"(tmem_load_3[19]), "=f"(tmem_load_3[20]), "=f"(tmem_load_3[21]), "=f"(tmem_load_3[22]), "=f"(tmem_load_3[23]), "=f"(tmem_load_3[24]), "=f"(tmem_load_3[25]), "=f"(tmem_load_3[26]), "=f"(tmem_load_3[27]), "=f"(tmem_load_3[28]), "=f"(tmem_load_3[29]), "=f"(tmem_load_3[30]), "=f"(tmem_load_3[31])
                                        : "r"(taddr_1 + (unsigned int)(warp_row << 16) + (unsigned int)(i_32 * 32)));
                                    #pragma unroll
                                    for (int j_8 = 0; j_8 < 16; j_8++) {
                                        packed_5[i_32 * 16 + j_8] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_3[2 * j_8], tmem_load_3[2 * j_8 + 1])));
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                                unsigned int scale_word_4 = 0;
                                unsigned int block_0[16];
                                #pragma unroll
                                for (int j_0 = 0; j_0 < 16; j_0++) {
                                    block_0[j_0] = packed_5[j_0];
                                }
                                #pragma unroll
                                for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                                    unsigned int address_0 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_1_1 * 16);
                                    address_0 = address_0 ^ (address_0 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_0 - smem_v41_addr)), "r"(packed_5[4 * j_1_1]), "r"(packed_5[4 * j_1_1 + 1]), "r"(packed_5[4 * j_1_1 + 2]), "r"(packed_5[4 * j_1_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_9;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_9) : "r"(block_0[0]));
                                unsigned int amax_pair_8 = bf16x2_abs_9;
                                #pragma unroll
                                for (int i_33 = 1; i_33 < 16; i_33++) {
                                    uint32_t bf16x2_abs_1_12;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_12) : "r"(block_0[i_33]));
                                    uint32_t bf16x2_max_12;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_12) : "r"(amax_pair_8), "r"(bf16x2_abs_1_12));
                                    amax_pair_8 = bf16x2_max_12;
                                }
                                uint16_t bf16_max_8;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_8) : "h"((uint16_t)(amax_pair_8 & 65535)), "h"((uint16_t)(amax_pair_8 >> 16)));
                                float amax_8;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8) : "h"((uint16_t)(bf16_max_8)));
                                float scale_8 = fmaxf(amax_8 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_88;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_88) : "f"(scale_8), "f"(scale_8));
                                unsigned int scale_byte_8 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_88 & 255;
                                unsigned int inverse_lane_8 = 254 - scale_byte_8 << 7;
                                unsigned int inverse_8 = inverse_lane_8 | inverse_lane_8 << 16;
                                unsigned int words_8[8];
                                #pragma unroll
                                for (int i_34 = 0; i_34 < 8; i_34++) {
                                    uint32_t bf16x2_mul_13;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(block_0[i_34 * 2]), "r"(inverse_8));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(block_0[i_34 * 2 + 1]), "r"(inverse_8));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_89;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_89) : "r"(bf16x2_mul_1_12));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_89) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_8[i_34] = (unsigned int)e4m3x2_bf16x2_12 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_89 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_8[0]), "r"(words_8[1]), "r"(words_8[2]), "r"(words_8[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8[4]), "r"(words_8[5]), "r"(words_8[6]), "r"(words_8[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_1_1[16];
                                #pragma unroll
                                for (int j_2_1 = 0; j_2_1 < 16; j_2_1++) {
                                    block_1_1[j_2_1] = packed_5[16 + j_2_1];
                                }
                                #pragma unroll
                                for (int j_3_1 = 0; j_3_1 < 4; j_3_1++) {
                                    unsigned int address_1_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_3_1 * 16);
                                    address_1_1 = address_1_1 ^ (address_1_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_1_1 - smem_v41_addr)), "r"(packed_5[16 + 4 * j_3_1]), "r"(packed_5[16 + 4 * j_3_1 + 1]), "r"(packed_5[16 + 4 * j_3_1 + 2]), "r"(packed_5[16 + 4 * j_3_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 1, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_4_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_4_1) : "r"(block_1_1[0]));
                                unsigned int amax_pair_4_1 = bf16x2_abs_4_1;
                                #pragma unroll
                                for (int i_35 = 1; i_35 < 16; i_35++) {
                                    uint32_t bf16x2_abs_1_13;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_13) : "r"(block_1_1[i_35]));
                                    uint32_t bf16x2_max_13;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_13) : "r"(amax_pair_4_1), "r"(bf16x2_abs_1_13));
                                    amax_pair_4_1 = bf16x2_max_13;
                                }
                                uint16_t bf16_max_4_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_4_1) : "h"((uint16_t)(amax_pair_4_1 & 65535)), "h"((uint16_t)(amax_pair_4_1 >> 16)));
                                float amax_4_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_4_1) : "h"((uint16_t)(bf16_max_4_1)));
                                float scale_4_1 = fmaxf(amax_4_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_90;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_90) : "f"(scale_4_1), "f"(scale_4_1));
                                unsigned int scale_byte_4_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_90 & 255;
                                unsigned int inverse_lane_4_1 = 254 - scale_byte_4_1 << 7;
                                unsigned int inverse_4_1 = inverse_lane_4_1 | inverse_lane_4_1 << 16;
                                unsigned int words_4_1[8];
                                #pragma unroll
                                for (int i_36 = 0; i_36 < 8; i_36++) {
                                    uint32_t bf16x2_mul_14;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(block_1_1[i_36 * 2]), "r"(inverse_4_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(block_1_1[i_36 * 2 + 1]), "r"(inverse_4_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_91;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_91) : "r"(bf16x2_mul_1_13));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_91) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_4_1[i_36] = (unsigned int)e4m3x2_bf16x2_13 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_91 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_4_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_4_1[0]), "r"(words_4_1[1]), "r"(words_4_1[2]), "r"(words_4_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_4_1[4]), "r"(words_4_1[5]), "r"(words_4_1[6]), "r"(words_4_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 32), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_2[16];
                                #pragma unroll
                                for (int j_4_1 = 0; j_4_1 < 16; j_4_1++) {
                                    block_2[j_4_1] = packed_5[32 + j_4_1];
                                }
                                #pragma unroll
                                for (int j_5_1 = 0; j_5_1 < 4; j_5_1++) {
                                    unsigned int address_2_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_5_1 * 16);
                                    address_2_1 = address_2_1 ^ (address_2_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_2_1 - smem_v41_addr)), "r"(packed_5[32 + 4 * j_5_1]), "r"(packed_5[32 + 4 * j_5_1 + 1]), "r"(packed_5[32 + 4 * j_5_1 + 2]), "r"(packed_5[32 + 4 * j_5_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 2, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_5_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_5_1) : "r"(block_2[0]));
                                unsigned int amax_pair_5_1 = bf16x2_abs_5_1;
                                #pragma unroll
                                for (int i_37 = 1; i_37 < 16; i_37++) {
                                    uint32_t bf16x2_abs_1_14;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_14) : "r"(block_2[i_37]));
                                    uint32_t bf16x2_max_14;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_14) : "r"(amax_pair_5_1), "r"(bf16x2_abs_1_14));
                                    amax_pair_5_1 = bf16x2_max_14;
                                }
                                uint16_t bf16_max_5_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_5_1) : "h"((uint16_t)(amax_pair_5_1 & 65535)), "h"((uint16_t)(amax_pair_5_1 >> 16)));
                                float amax_5_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_5_1) : "h"((uint16_t)(bf16_max_5_1)));
                                float scale_5_1 = fmaxf(amax_5_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_92;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_92) : "f"(scale_5_1), "f"(scale_5_1));
                                unsigned int scale_byte_5_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_92 & 255;
                                unsigned int inverse_lane_5_1 = 254 - scale_byte_5_1 << 7;
                                unsigned int inverse_5_1 = inverse_lane_5_1 | inverse_lane_5_1 << 16;
                                unsigned int words_5_1[8];
                                #pragma unroll
                                for (int i_38 = 0; i_38 < 8; i_38++) {
                                    uint32_t bf16x2_mul_15;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(block_2[i_38 * 2]), "r"(inverse_5_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(block_2[i_38 * 2 + 1]), "r"(inverse_5_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_93;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_93) : "r"(bf16x2_mul_1_14));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_93) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_5_1[i_38] = (unsigned int)e4m3x2_bf16x2_14 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_93 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_5_1 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_5_1[0]), "r"(words_5_1[1]), "r"(words_5_1[2]), "r"(words_5_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_5_1[4]), "r"(words_5_1[5]), "r"(words_5_1[6]), "r"(words_5_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 64), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_3[16];
                                #pragma unroll
                                for (int j_6_1 = 0; j_6_1 < 16; j_6_1++) {
                                    block_3[j_6_1] = packed_5[48 + j_6_1];
                                }
                                #pragma unroll
                                for (int j_7_1 = 0; j_7_1 < 4; j_7_1++) {
                                    unsigned int address_3_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_7_1 * 16);
                                    address_3_1 = address_3_1 ^ (address_3_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_3_1 - smem_v41_addr)), "r"(packed_5[48 + 4 * j_7_1]), "r"(packed_5[48 + 4 * j_7_1 + 1]), "r"(packed_5[48 + 4 * j_7_1 + 2]), "r"(packed_5[48 + 4 * j_7_1 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 3, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_6_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_6_1) : "r"(block_3[0]));
                                unsigned int amax_pair_6_1 = bf16x2_abs_6_1;
                                #pragma unroll
                                for (int i_39 = 1; i_39 < 16; i_39++) {
                                    uint32_t bf16x2_abs_1_15;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_15) : "r"(block_3[i_39]));
                                    uint32_t bf16x2_max_15;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_15) : "r"(amax_pair_6_1), "r"(bf16x2_abs_1_15));
                                    amax_pair_6_1 = bf16x2_max_15;
                                }
                                uint16_t bf16_max_6_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_6_1) : "h"((uint16_t)(amax_pair_6_1 & 65535)), "h"((uint16_t)(amax_pair_6_1 >> 16)));
                                float amax_6_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_6_1) : "h"((uint16_t)(bf16_max_6_1)));
                                float scale_6_1 = fmaxf(amax_6_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_94;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_94) : "f"(scale_6_1), "f"(scale_6_1));
                                unsigned int scale_byte_6_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_94 & 255;
                                unsigned int inverse_lane_6_1 = 254 - scale_byte_6_1 << 7;
                                unsigned int inverse_6_1 = inverse_lane_6_1 | inverse_lane_6_1 << 16;
                                unsigned int words_6_1[8];
                                #pragma unroll
                                for (int i_40 = 0; i_40 < 8; i_40++) {
                                    uint32_t bf16x2_mul_16;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(block_3[i_40 * 2]), "r"(inverse_6_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(block_3[i_40 * 2 + 1]), "r"(inverse_6_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_95;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_95) : "r"(bf16x2_mul_1_15));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_95) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_6_1[i_40] = (unsigned int)e4m3x2_bf16x2_15 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_95 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_6_1 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_6_1[0]), "r"(words_6_1[1]), "r"(words_6_1[2]), "r"(words_6_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_6_1[4]), "r"(words_6_1[5]), "r"(words_6_1[6]), "r"(words_6_1[7]) : "memory");
                                smem_v43[tid % 32 * 4 + tid / 32] = scale_word_4;
                                scale_word_4 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 96), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2, smem_v43_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_4[16];
                                #pragma unroll
                                for (int j_8_1 = 0; j_8_1 < 16; j_8_1++) {
                                    block_4[j_8_1] = packed_5[64 + j_8_1];
                                }
                                #pragma unroll
                                for (int j_9 = 0; j_9 < 4; j_9++) {
                                    unsigned int address_4_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_9 * 16);
                                    address_4_1 = address_4_1 ^ (address_4_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_4_1 - smem_v41_addr)), "r"(packed_5[64 + 4 * j_9]), "r"(packed_5[64 + 4 * j_9 + 1]), "r"(packed_5[64 + 4 * j_9 + 2]), "r"(packed_5[64 + 4 * j_9 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 4, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_7_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_7_1) : "r"(block_4[0]));
                                unsigned int amax_pair_7_1 = bf16x2_abs_7_1;
                                #pragma unroll
                                for (int i_41 = 1; i_41 < 16; i_41++) {
                                    uint32_t bf16x2_abs_1_16;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_16) : "r"(block_4[i_41]));
                                    uint32_t bf16x2_max_16;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_16) : "r"(amax_pair_7_1), "r"(bf16x2_abs_1_16));
                                    amax_pair_7_1 = bf16x2_max_16;
                                }
                                uint16_t bf16_max_7_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_7_1) : "h"((uint16_t)(amax_pair_7_1 & 65535)), "h"((uint16_t)(amax_pair_7_1 >> 16)));
                                float amax_7_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_7_1) : "h"((uint16_t)(bf16_max_7_1)));
                                float scale_7_1 = fmaxf(amax_7_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_96;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_96) : "f"(scale_7_1), "f"(scale_7_1));
                                unsigned int scale_byte_7_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_96 & 255;
                                unsigned int inverse_lane_7_1 = 254 - scale_byte_7_1 << 7;
                                unsigned int inverse_7_1 = inverse_lane_7_1 | inverse_lane_7_1 << 16;
                                unsigned int words_7_1[8];
                                #pragma unroll
                                for (int i_42 = 0; i_42 < 8; i_42++) {
                                    uint32_t bf16x2_mul_17;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(block_4[i_42 * 2]), "r"(inverse_7_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(block_4[i_42 * 2 + 1]), "r"(inverse_7_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_97;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_97) : "r"(bf16x2_mul_1_16));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_97) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_7_1[i_42] = (unsigned int)e4m3x2_bf16x2_16 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_97 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_7_1;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_7_1[0]), "r"(words_7_1[1]), "r"(words_7_1[2]), "r"(words_7_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_7_1[4]), "r"(words_7_1[5]), "r"(words_7_1[6]), "r"(words_7_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 128), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_5[16];
                                #pragma unroll
                                for (int j_10 = 0; j_10 < 16; j_10++) {
                                    block_5[j_10] = packed_5[80 + j_10];
                                }
                                #pragma unroll
                                for (int j_11 = 0; j_11 < 4; j_11++) {
                                    unsigned int address_5_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_11 * 16);
                                    address_5_1 = address_5_1 ^ (address_5_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_5_1 - smem_v41_addr)), "r"(packed_5[80 + 4 * j_11]), "r"(packed_5[80 + 4 * j_11 + 1]), "r"(packed_5[80 + 4 * j_11 + 2]), "r"(packed_5[80 + 4 * j_11 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 5, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_8_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_8_1) : "r"(block_5[0]));
                                unsigned int amax_pair_8_1 = bf16x2_abs_8_1;
                                #pragma unroll
                                for (int i_43 = 1; i_43 < 16; i_43++) {
                                    uint32_t bf16x2_abs_1_17;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_17) : "r"(block_5[i_43]));
                                    uint32_t bf16x2_max_17;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_17) : "r"(amax_pair_8_1), "r"(bf16x2_abs_1_17));
                                    amax_pair_8_1 = bf16x2_max_17;
                                }
                                uint16_t bf16_max_8_1;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_8_1) : "h"((uint16_t)(amax_pair_8_1 & 65535)), "h"((uint16_t)(amax_pair_8_1 >> 16)));
                                float amax_8_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_8_1) : "h"((uint16_t)(bf16_max_8_1)));
                                float scale_8_1 = fmaxf(amax_8_1 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_98;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_98) : "f"(scale_8_1), "f"(scale_8_1));
                                unsigned int scale_byte_8_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_98 & 255;
                                unsigned int inverse_lane_8_1 = 254 - scale_byte_8_1 << 7;
                                unsigned int inverse_8_1 = inverse_lane_8_1 | inverse_lane_8_1 << 16;
                                unsigned int words_8_1[8];
                                #pragma unroll
                                for (int i_44 = 0; i_44 < 8; i_44++) {
                                    uint32_t bf16x2_mul_18;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(block_5[i_44 * 2]), "r"(inverse_8_1));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(block_5[i_44 * 2 + 1]), "r"(inverse_8_1));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_99;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_99) : "r"(bf16x2_mul_1_17));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_99) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_8_1[i_44] = (unsigned int)e4m3x2_bf16x2_17 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_99 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_8_1 << 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_8_1[0]), "r"(words_8_1[1]), "r"(words_8_1[2]), "r"(words_8_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_8_1[4]), "r"(words_8_1[5]), "r"(words_8_1[6]), "r"(words_8_1[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 160), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_6[16];
                                #pragma unroll
                                for (int j_12 = 0; j_12 < 16; j_12++) {
                                    block_6[j_12] = packed_5[96 + j_12];
                                }
                                #pragma unroll
                                for (int j_13 = 0; j_13 < 4; j_13++) {
                                    unsigned int address_6_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_13 * 16);
                                    address_6_1 = address_6_1 ^ (address_6_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_6_1 - smem_v41_addr)), "r"(packed_5[96 + 4 * j_13]), "r"(packed_5[96 + 4 * j_13 + 1]), "r"(packed_5[96 + 4 * j_13 + 2]), "r"(packed_5[96 + 4 * j_13 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 6, 0, 0, d_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_9_1;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_9_1) : "r"(block_6[0]));
                                unsigned int amax_pair_9 = bf16x2_abs_9_1;
                                #pragma unroll
                                for (int i_45 = 1; i_45 < 16; i_45++) {
                                    uint32_t bf16x2_abs_1_18;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_18) : "r"(block_6[i_45]));
                                    uint32_t bf16x2_max_18;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_18) : "r"(amax_pair_9), "r"(bf16x2_abs_1_18));
                                    amax_pair_9 = bf16x2_max_18;
                                }
                                uint16_t bf16_max_9;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_9) : "h"((uint16_t)(amax_pair_9 & 65535)), "h"((uint16_t)(amax_pair_9 >> 16)));
                                float amax_9;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_9) : "h"((uint16_t)(bf16_max_9)));
                                float scale_9 = fmaxf(amax_9 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_100;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_100) : "f"(scale_9), "f"(scale_9));
                                unsigned int scale_byte_9 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_100 & 255;
                                unsigned int inverse_lane_9 = 254 - scale_byte_9 << 7;
                                unsigned int inverse_9 = inverse_lane_9 | inverse_lane_9 << 16;
                                unsigned int words_9[8];
                                #pragma unroll
                                for (int i_46 = 0; i_46 < 8; i_46++) {
                                    uint32_t bf16x2_mul_19;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(block_6[i_46 * 2]), "r"(inverse_9));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(block_6[i_46 * 2 + 1]), "r"(inverse_9));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_101;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_101) : "r"(bf16x2_mul_1_18));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_101) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_9[i_46] = (unsigned int)e4m3x2_bf16x2_18 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_101 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_9 << 16;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_9[0]), "r"(words_9[1]), "r"(words_9[2]), "r"(words_9[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_9[4]), "r"(words_9[5]), "r"(words_9[6]), "r"(words_9[7]) : "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 192), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                                unsigned int block_7[16];
                                #pragma unroll
                                for (int j_14 = 0; j_14 < 16; j_14++) {
                                    block_7[j_14] = packed_5[112 + j_14];
                                }
                                #pragma unroll
                                for (int j_15 = 0; j_15 < 4; j_15++) {
                                    unsigned int address_7_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_15 * 16);
                                    address_7_1 = address_7_1 ^ (address_7_1 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_7_1 - smem_v41_addr)), "r"(packed_5[112 + 4 * j_15]), "r"(packed_5[112 + 4 * j_15 + 1]), "r"(packed_5[112 + 4 * j_15 + 2]), "r"(packed_5[112 + 4 * j_15 + 3]) : "memory");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&gate_routed_out), 0, x_2 * 256 + cta_rank_0 * 128, y_2 * 8 + 7, 0, 0, d_smem_addr + 8192);
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                uint32_t bf16x2_abs_10;
                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_10) : "r"(block_7[0]));
                                unsigned int amax_pair_10 = bf16x2_abs_10;
                                #pragma unroll
                                for (int i_47 = 1; i_47 < 16; i_47++) {
                                    uint32_t bf16x2_abs_1_19;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_19) : "r"(block_7[i_47]));
                                    uint32_t bf16x2_max_19;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_19) : "r"(amax_pair_10), "r"(bf16x2_abs_1_19));
                                    amax_pair_10 = bf16x2_max_19;
                                }
                                uint16_t bf16_max_10;
                                asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_10) : "h"((uint16_t)(amax_pair_10 & 65535)), "h"((uint16_t)(amax_pair_10 >> 16)));
                                float amax_10;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_10) : "h"((uint16_t)(bf16_max_10)));
                                float scale_10 = fmaxf(amax_10 * 0.002232143f, 1e-12f);
                                uint16_t _cvt_rp_satfinite_ue8m0x2_f32_102;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_102) : "f"(scale_10), "f"(scale_10));
                                unsigned int scale_byte_10 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_102 & 255;
                                unsigned int inverse_lane_10 = 254 - scale_byte_10 << 7;
                                unsigned int inverse_10 = inverse_lane_10 | inverse_lane_10 << 16;
                                unsigned int words_10[8];
                                #pragma unroll
                                for (int i_48 = 0; i_48 < 8; i_48++) {
                                    uint32_t bf16x2_mul_20;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(block_7[i_48 * 2]), "r"(inverse_10));
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
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(block_7[i_48 * 2 + 1]), "r"(inverse_10));
                                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_103;
                                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "r"(bf16x2_mul_1_19));
                                    #else
                                    {
                                    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
                                    const float _lo = __uint_as_float(_bf16x2 << 16);
                                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_103) : "f"(_hi), "f"(_lo));
                                    }
                                    #endif
                                    words_10[i_48] = (unsigned int)e4m3x2_bf16x2_19 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_103 << 16;
                                }
                                scale_word_4 = scale_word_4 | scale_byte_10 << 24;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_10[0]), "r"(words_10[1]), "r"(words_10[2]), "r"(words_10[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_10[4]), "r"(words_10[5]), "r"(words_10[6]), "r"(words_10[7]) : "memory");
                                smem_v44[tid % 32 * 4 + tid / 32] = scale_word_4;
                                scale_word_4 = 0;
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_q_store)), "r"(y_2 * 256 + 224), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    tma_store_3d((&gate_sc_store), 0, 0, (x_2 * 2 + cta_rank_0) * i_tiles + y_2 * 2 + 1, smem_v44_addr);
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
                                            bool enabled_value_4 = 1;
                                            if (enabled_value_4 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_2 + x_2) * (intermediate / 256) + y_2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
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
                } else {
                    if (tile_word_1 < 2 * mini_gate) {
                        int col_blocks_5 = (intermediate + 256 - 1) / 256;
                        int x_3 = -1;
                        int y_3 = -1;
                        int expert_3 = -1;
                        int k_start_3 = 0;
                        int k_end_3 = 0;
                        int first_3 = 0;
                        int first_block_2 = tile_word_0 * (mini_size / 256);
                        int end_block_2 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                        int block_8 = first_block_2;
                        uint32_t fast_div_q_3 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(tile_word_1 - mini_gate)), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(tile_word_1 - mini_gate));
                        uint32_t fast_div_r_3 = (uint32_t)((unsigned int)(tile_word_1 - mini_gate)) - fast_div_q_3 * (uint32_t)(gate_col_blocks_div.divisor);
                        block_8 = first_block_2 + (int)fast_div_q_3;
                        if (block_8 < end_block_2) {
                            int index_1 = tile_word_3;
                            int offset_5 = tile_word_4 / 256;
                            int first_row_2 = ((first_block_2) > (offset_5) ? (first_block_2) : (offset_5));
                            int _min_104 = ((end_block_2) < (offset_5 + tile_word_5 / 256) ? (end_block_2) : (offset_5 + tile_word_5 / 256));
                            int rows_1 = _min_104 - first_row_2;
                            int supergroup_3 = (tile_word_1 - mini_gate - (first_row_2 - first_block_2) * col_blocks_5) / (rows_1 * 8);
                            int full_cols_3 = col_blocks_5 / 8 * 8;
                            int row_14 = 0;
                            int col_13 = 0;
                            if (tile_word_1 - mini_gate - (first_row_2 - first_block_2) * col_blocks_5 < rows_1 * full_cols_3) {
                                row_14 = (tile_word_1 - mini_gate - (first_row_2 - first_block_2) * col_blocks_5) % (rows_1 * 8) / 8;
                                col_13 = supergroup_3 * 8 + (tile_word_1 - mini_gate - (first_row_2 - first_block_2) * col_blocks_5) % 8;
                            } else {
                                row_14 = (tile_word_1 - mini_gate - (first_row_2 - first_block_2) * col_blocks_5 - rows_1 * full_cols_3) / (col_blocks_5 - full_cols_3);
                                col_13 = full_cols_3 + (tile_word_1 - mini_gate - (first_row_2 - first_block_2) * col_blocks_5 - rows_1 * full_cols_3) % (col_blocks_5 - full_cols_3);
                            }
                            if ((supergroup_3 & 1) != 0) {
                                row_14 = rows_1 - row_14 - 1;
                            }
                            x_3 = first_row_2 + row_14;
                            y_3 = col_13;
                            expert_3 = index_1;
                        }
                        int has_hi_3 = 0;
                        int global_mini_3 = tile_word_0;
                        int macro_rows_3 = 0;
                        int iterations_3 = hidden / 128;
                        int macro_k_1 = 0;
                        if (expert_3 < 0) {
                            if (tid == 0) {
                            }
                        } else {
                            if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            int _min_105 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                            int mini_rows_4 = ((0) > (_min_105) ? (0) : (_min_105));
                                            int required_4 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_5 = 1;
                                            if (enabled_value_5 != 0) {
                                                int poll_3_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_first) : "l"(x_ready + global_mini_3) : "memory");
                                                int poll_3 = poll_3_first;
                                                while (poll_3 < required_4) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int poll_3_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_next) : "l"(x_ready + global_mini_3) : "memory");
                                                    poll_3 = poll_3_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                            int _stage_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_7) % 4);
                                            int _parity_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_7) / 4 % 2) ^ 1;
                                            mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_3) * 8, _parity_gemm_finished_3);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v31_addr + (unsigned int)((gemm_wide + idx_7) % 4 * 16384)), "l"((&x_q)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(smem_v32_addr + (unsigned int)((gemm_wide + idx_7) % 4 * 16384)), "l"((&wu_q)), "r"(0), "r"(y_3 * 256 + cta_rank_0 * 128), "r"(idx_7), "r"(expert_3), "r"(0),
                                                   "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        }
                                    }
                                }
                            } else if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_106 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                            int mini_rows_5 = ((0) > (_min_106) ? (0) : (_min_106));
                                            int required_5 = (mini_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_6 = 1;
                                            if (enabled_value_6 != 0) {
                                                int poll_4_first;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_first) : "l"(x_ready + global_mini_3) : "memory");
                                                int poll_4 = poll_4_first;
                                                while (poll_4 < required_5) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int poll_4_next;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_next) : "l"(x_ready + global_mini_3) : "memory");
                                                    poll_4 = poll_4_next;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_3; idx_8++) {
                                            int _stage_scales_finished_1 = (int)((unsigned int)(scales_wide + idx_8) % 4);
                                            int _parity_scales_finished_1 = (int)((unsigned int)(scales_wide + idx_8) / 4 % 2) ^ 1;
                                            mbarrier_wait(scales_finished_addr + (_stage_scales_finished_1) * 8, _parity_scales_finished_1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v34_addr + (unsigned int)((scales_wide + idx_8) % 4 * 512)), "l"((&x_sc)), "r"(0), "r"(0), "r"((x_3 * 2 + cta_rank_0) * (hidden / 128) + idx_8),
                                                   "r"(((scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_8) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(smem_v35_addr + (unsigned int)((scales_wide + idx_8) % 4 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc)), "r"(0), "r"(0), "r"((expert_3 * (intermediate / 128) + y_3 * 2 + cta_rank_0) * (hidden / 128) + idx_8),
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
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + ((int)((unsigned int)(scales_wide + idx_9) % 4)) * 8, 3072);
                                                int _stage_scales_arrived_1 = (int)((unsigned int)(scales_wide + idx_9) % 4);
                                                int _parity_scales_arrived_1 = (int)((unsigned int)(scales_wide + idx_9) / 4 % 2);
                                                mbarrier_wait(scales_arrived_addr + (_stage_scales_arrived_1) * 8, _parity_scales_arrived_1);
                                                int buffer_1 = idx_9 % 3;
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + buffer_1 * 4, make_sf_cp_desc_sbo128(smem_v34_addr + (unsigned int)((scales_wide + idx_9) % 4 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + buffer_1 * 8, make_sf_cp_desc_sbo128(smem_v35_addr + (unsigned int)((scales_wide + idx_9) % 4 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + buffer_1 * 8 + 4), make_sf_cp_desc_sbo128((smem_v35_addr + (unsigned int)((scales_wide + idx_9) % 4 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + ((int)((unsigned int)(scales_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, 65536);
                                                int _stage_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_9) % 4);
                                                int _parity_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_9) / 4 % 2);
                                                mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_3) * 8, _parity_gemm_arrived_3);
                                                int _smem_desc_lo_107 = (((smem_v31_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                                int _smem_desc_lo_108 = (((smem_v32_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)_smem_desc_lo_107) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)_smem_desc_lo_108) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        0x90c00000U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0)), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0)), ((idx_9 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8f6f4_bs_k64_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        0xd0c00020U, (int)((tmem_tmem_sfa + buffer_1 * 4 + 0) | 0x80000000), (int)((tmem_tmem_sfb + buffer_1 * 8 + 0) | 0x80000000), 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    int _parity_output_arrived_3 = (int)((unsigned int)gemm_out0 % 2);
                                    mbarrier_wait(output_arrived_addr, _parity_output_arrived_3);
                                    int warp_row_1 = tid / 32 * 32;
                                    unsigned int packed_6[128];
                                    #pragma unroll
                                    for (int i_49 = 0; i_49 < 8; i_49++) {
                                        float tmem_load_4[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(tmem_load_4[0]), "=f"(tmem_load_4[1]), "=f"(tmem_load_4[2]), "=f"(tmem_load_4[3]), "=f"(tmem_load_4[4]), "=f"(tmem_load_4[5]), "=f"(tmem_load_4[6]), "=f"(tmem_load_4[7]), "=f"(tmem_load_4[8]), "=f"(tmem_load_4[9]), "=f"(tmem_load_4[10]), "=f"(tmem_load_4[11]), "=f"(tmem_load_4[12]), "=f"(tmem_load_4[13]), "=f"(tmem_load_4[14]), "=f"(tmem_load_4[15]), "=f"(tmem_load_4[16]), "=f"(tmem_load_4[17]), "=f"(tmem_load_4[18]), "=f"(tmem_load_4[19]), "=f"(tmem_load_4[20]), "=f"(tmem_load_4[21]), "=f"(tmem_load_4[22]), "=f"(tmem_load_4[23]), "=f"(tmem_load_4[24]), "=f"(tmem_load_4[25]), "=f"(tmem_load_4[26]), "=f"(tmem_load_4[27]), "=f"(tmem_load_4[28]), "=f"(tmem_load_4[29]), "=f"(tmem_load_4[30]), "=f"(tmem_load_4[31])
                                            : "r"(taddr_1 + (unsigned int)(warp_row_1 << 16) + (unsigned int)(i_49 * 32)));
                                        #pragma unroll
                                        for (int j_16 = 0; j_16 < 16; j_16++) {
                                            packed_6[i_49 * 16 + j_16] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_4[2 * j_16], tmem_load_4[2 * j_16 + 1])));
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                    unsigned int scale_word_5 = 0;
                                    unsigned int block_8_1[16];
                                    #pragma unroll
                                    for (int j_16_1 = 0; j_16_1 < 16; j_16_1++) {
                                        block_8_1[j_16_1] = packed_6[j_16_1];
                                    }
                                    #pragma unroll
                                    for (int j_17 = 0; j_17 < 4; j_17++) {
                                        unsigned int address_8_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_17 * 16);
                                        address_8_1 = address_8_1 ^ (address_8_1 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_8_1 - smem_v41_addr)), "r"(packed_6[4 * j_17]), "r"(packed_6[4 * j_17 + 1]), "r"(packed_6[4 * j_17 + 2]), "r"(packed_6[4 * j_17 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_11;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_11) : "r"(block_8_1[0]));
                                    unsigned int amax_pair_11 = bf16x2_abs_11;
                                    #pragma unroll
                                    for (int i_50 = 1; i_50 < 16; i_50++) {
                                        uint32_t bf16x2_abs_1_20;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_20) : "r"(block_8_1[i_50]));
                                        uint32_t bf16x2_max_20;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_20) : "r"(amax_pair_11), "r"(bf16x2_abs_1_20));
                                        amax_pair_11 = bf16x2_max_20;
                                    }
                                    uint16_t bf16_max_11;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_11) : "h"((uint16_t)(amax_pair_11 & 65535)), "h"((uint16_t)(amax_pair_11 >> 16)));
                                    float amax_11;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11) : "h"((uint16_t)(bf16_max_11)));
                                    float scale_11 = fmaxf(amax_11 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_109;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_109) : "f"(scale_11), "f"(scale_11));
                                    unsigned int scale_byte_11 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_109 & 255;
                                    unsigned int inverse_lane_11 = 254 - scale_byte_11 << 7;
                                    unsigned int inverse_11 = inverse_lane_11 | inverse_lane_11 << 16;
                                    unsigned int words_11[8];
                                    #pragma unroll
                                    for (int i_51 = 0; i_51 < 8; i_51++) {
                                        uint32_t bf16x2_mul_21;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(block_8_1[i_51 * 2]), "r"(inverse_11));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(block_8_1[i_51 * 2 + 1]), "r"(inverse_11));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_110;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_110) : "r"(bf16x2_mul_1_20));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_110) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_11[i_51] = (unsigned int)e4m3x2_bf16x2_20 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_110 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_11;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_11[0]), "r"(words_11[1]), "r"(words_11[2]), "r"(words_11[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_11[4]), "r"(words_11[5]), "r"(words_11[6]), "r"(words_11[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_9[16];
                                    #pragma unroll
                                    for (int j_18 = 0; j_18 < 16; j_18++) {
                                        block_9[j_18] = packed_6[16 + j_18];
                                    }
                                    #pragma unroll
                                    for (int j_19 = 0; j_19 < 4; j_19++) {
                                        unsigned int address_9_1 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_19 * 16);
                                        address_9_1 = address_9_1 ^ (address_9_1 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_9_1 - smem_v41_addr)), "r"(packed_6[16 + 4 * j_19]), "r"(packed_6[16 + 4 * j_19 + 1]), "r"(packed_6[16 + 4 * j_19 + 2]), "r"(packed_6[16 + 4 * j_19 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 1, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_11_1;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_11_1) : "r"(block_9[0]));
                                    unsigned int amax_pair_11_1 = bf16x2_abs_11_1;
                                    #pragma unroll
                                    for (int i_52 = 1; i_52 < 16; i_52++) {
                                        uint32_t bf16x2_abs_1_21;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_21) : "r"(block_9[i_52]));
                                        uint32_t bf16x2_max_21;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_21) : "r"(amax_pair_11_1), "r"(bf16x2_abs_1_21));
                                        amax_pair_11_1 = bf16x2_max_21;
                                    }
                                    uint16_t bf16_max_11_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_11_1) : "h"((uint16_t)(amax_pair_11_1 & 65535)), "h"((uint16_t)(amax_pair_11_1 >> 16)));
                                    float amax_11_1;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_11_1) : "h"((uint16_t)(bf16_max_11_1)));
                                    float scale_11_1 = fmaxf(amax_11_1 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_111;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_111) : "f"(scale_11_1), "f"(scale_11_1));
                                    unsigned int scale_byte_11_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_111 & 255;
                                    unsigned int inverse_lane_11_1 = 254 - scale_byte_11_1 << 7;
                                    unsigned int inverse_11_1 = inverse_lane_11_1 | inverse_lane_11_1 << 16;
                                    unsigned int words_11_1[8];
                                    #pragma unroll
                                    for (int i_53 = 0; i_53 < 8; i_53++) {
                                        uint32_t bf16x2_mul_22;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(block_9[i_53 * 2]), "r"(inverse_11_1));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(block_9[i_53 * 2 + 1]), "r"(inverse_11_1));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_112;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_112) : "r"(bf16x2_mul_1_21));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_112) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_11_1[i_53] = (unsigned int)e4m3x2_bf16x2_21 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_112 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_11_1 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_11_1[0]), "r"(words_11_1[1]), "r"(words_11_1[2]), "r"(words_11_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_11_1[4]), "r"(words_11_1[5]), "r"(words_11_1[6]), "r"(words_11_1[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 32), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_10[16];
                                    #pragma unroll
                                    for (int j_20 = 0; j_20 < 16; j_20++) {
                                        block_10[j_20] = packed_6[32 + j_20];
                                    }
                                    #pragma unroll
                                    for (int j_21 = 0; j_21 < 4; j_21++) {
                                        unsigned int address_10_1 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_21 * 16);
                                        address_10_1 = address_10_1 ^ (address_10_1 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_10_1 - smem_v41_addr)), "r"(packed_6[32 + 4 * j_21]), "r"(packed_6[32 + 4 * j_21 + 1]), "r"(packed_6[32 + 4 * j_21 + 2]), "r"(packed_6[32 + 4 * j_21 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 2, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_12;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_12) : "r"(block_10[0]));
                                    unsigned int amax_pair_12 = bf16x2_abs_12;
                                    #pragma unroll
                                    for (int i_54 = 1; i_54 < 16; i_54++) {
                                        uint32_t bf16x2_abs_1_22;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_22) : "r"(block_10[i_54]));
                                        uint32_t bf16x2_max_22;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_22) : "r"(amax_pair_12), "r"(bf16x2_abs_1_22));
                                        amax_pair_12 = bf16x2_max_22;
                                    }
                                    uint16_t bf16_max_12;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_12) : "h"((uint16_t)(amax_pair_12 & 65535)), "h"((uint16_t)(amax_pair_12 >> 16)));
                                    float amax_12;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_12) : "h"((uint16_t)(bf16_max_12)));
                                    float scale_12 = fmaxf(amax_12 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_113;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_113) : "f"(scale_12), "f"(scale_12));
                                    unsigned int scale_byte_12 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_113 & 255;
                                    unsigned int inverse_lane_12 = 254 - scale_byte_12 << 7;
                                    unsigned int inverse_12 = inverse_lane_12 | inverse_lane_12 << 16;
                                    unsigned int words_12[8];
                                    #pragma unroll
                                    for (int i_55 = 0; i_55 < 8; i_55++) {
                                        uint32_t bf16x2_mul_23;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(block_10[i_55 * 2]), "r"(inverse_12));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(block_10[i_55 * 2 + 1]), "r"(inverse_12));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_114;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_114) : "r"(bf16x2_mul_1_22));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_114) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_12[i_55] = (unsigned int)e4m3x2_bf16x2_22 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_114 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_12 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_12[0]), "r"(words_12[1]), "r"(words_12[2]), "r"(words_12[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_12[4]), "r"(words_12[5]), "r"(words_12[6]), "r"(words_12[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 64), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_11[16];
                                    #pragma unroll
                                    for (int j_22 = 0; j_22 < 16; j_22++) {
                                        block_11[j_22] = packed_6[48 + j_22];
                                    }
                                    #pragma unroll
                                    for (int j_23 = 0; j_23 < 4; j_23++) {
                                        unsigned int address_11 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_23 * 16);
                                        address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_11 - smem_v41_addr)), "r"(packed_6[48 + 4 * j_23]), "r"(packed_6[48 + 4 * j_23 + 1]), "r"(packed_6[48 + 4 * j_23 + 2]), "r"(packed_6[48 + 4 * j_23 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 3, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_13;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_13) : "r"(block_11[0]));
                                    unsigned int amax_pair_13 = bf16x2_abs_13;
                                    #pragma unroll
                                    for (int i_56 = 1; i_56 < 16; i_56++) {
                                        uint32_t bf16x2_abs_1_23;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_23) : "r"(block_11[i_56]));
                                        uint32_t bf16x2_max_23;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_23) : "r"(amax_pair_13), "r"(bf16x2_abs_1_23));
                                        amax_pair_13 = bf16x2_max_23;
                                    }
                                    uint16_t bf16_max_13;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_13) : "h"((uint16_t)(amax_pair_13 & 65535)), "h"((uint16_t)(amax_pair_13 >> 16)));
                                    float amax_13;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_13) : "h"((uint16_t)(bf16_max_13)));
                                    float scale_13 = fmaxf(amax_13 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_115;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_115) : "f"(scale_13), "f"(scale_13));
                                    unsigned int scale_byte_13 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_115 & 255;
                                    unsigned int inverse_lane_13 = 254 - scale_byte_13 << 7;
                                    unsigned int inverse_13 = inverse_lane_13 | inverse_lane_13 << 16;
                                    unsigned int words_13[8];
                                    #pragma unroll
                                    for (int i_57 = 0; i_57 < 8; i_57++) {
                                        uint32_t bf16x2_mul_24;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(block_11[i_57 * 2]), "r"(inverse_13));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(block_11[i_57 * 2 + 1]), "r"(inverse_13));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_116;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_116) : "r"(bf16x2_mul_1_23));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_116) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_13[i_57] = (unsigned int)e4m3x2_bf16x2_23 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_116 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_13 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_13[0]), "r"(words_13[1]), "r"(words_13[2]), "r"(words_13[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_13[4]), "r"(words_13[5]), "r"(words_13[6]), "r"(words_13[7]) : "memory");
                                    smem_v43[tid % 32 * 4 + tid / 32] = scale_word_5;
                                    scale_word_5 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 96), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&up_sc_store), 0, 0, (x_3 * 2 + cta_rank_0) * i_tiles + y_3 * 2, smem_v43_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_12[16];
                                    #pragma unroll
                                    for (int j_24 = 0; j_24 < 16; j_24++) {
                                        block_12[j_24] = packed_6[64 + j_24];
                                    }
                                    #pragma unroll
                                    for (int j_25 = 0; j_25 < 4; j_25++) {
                                        unsigned int address_12 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_25 * 16);
                                        address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_12 - smem_v41_addr)), "r"(packed_6[64 + 4 * j_25]), "r"(packed_6[64 + 4 * j_25 + 1]), "r"(packed_6[64 + 4 * j_25 + 2]), "r"(packed_6[64 + 4 * j_25 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 4, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_14;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_14) : "r"(block_12[0]));
                                    unsigned int amax_pair_14 = bf16x2_abs_14;
                                    #pragma unroll
                                    for (int i_58 = 1; i_58 < 16; i_58++) {
                                        uint32_t bf16x2_abs_1_24;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_24) : "r"(block_12[i_58]));
                                        uint32_t bf16x2_max_24;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_24) : "r"(amax_pair_14), "r"(bf16x2_abs_1_24));
                                        amax_pair_14 = bf16x2_max_24;
                                    }
                                    uint16_t bf16_max_14;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_14) : "h"((uint16_t)(amax_pair_14 & 65535)), "h"((uint16_t)(amax_pair_14 >> 16)));
                                    float amax_14;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_14) : "h"((uint16_t)(bf16_max_14)));
                                    float scale_14 = fmaxf(amax_14 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_117;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_117) : "f"(scale_14), "f"(scale_14));
                                    unsigned int scale_byte_14 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_117 & 255;
                                    unsigned int inverse_lane_14 = 254 - scale_byte_14 << 7;
                                    unsigned int inverse_14 = inverse_lane_14 | inverse_lane_14 << 16;
                                    unsigned int words_14[8];
                                    #pragma unroll
                                    for (int i_59 = 0; i_59 < 8; i_59++) {
                                        uint32_t bf16x2_mul_25;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(block_12[i_59 * 2]), "r"(inverse_14));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(block_12[i_59 * 2 + 1]), "r"(inverse_14));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_118;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_118) : "r"(bf16x2_mul_1_24));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_118) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_14[i_59] = (unsigned int)e4m3x2_bf16x2_24 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_118 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_14;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_14[0]), "r"(words_14[1]), "r"(words_14[2]), "r"(words_14[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_14[4]), "r"(words_14[5]), "r"(words_14[6]), "r"(words_14[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 128), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_13[16];
                                    #pragma unroll
                                    for (int j_26 = 0; j_26 < 16; j_26++) {
                                        block_13[j_26] = packed_6[80 + j_26];
                                    }
                                    #pragma unroll
                                    for (int j_27 = 0; j_27 < 4; j_27++) {
                                        unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_27 * 16);
                                        address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_13 - smem_v41_addr)), "r"(packed_6[80 + 4 * j_27]), "r"(packed_6[80 + 4 * j_27 + 1]), "r"(packed_6[80 + 4 * j_27 + 2]), "r"(packed_6[80 + 4 * j_27 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 5, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_15;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_15) : "r"(block_13[0]));
                                    unsigned int amax_pair_15 = bf16x2_abs_15;
                                    #pragma unroll
                                    for (int i_60 = 1; i_60 < 16; i_60++) {
                                        uint32_t bf16x2_abs_1_25;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_25) : "r"(block_13[i_60]));
                                        uint32_t bf16x2_max_25;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_25) : "r"(amax_pair_15), "r"(bf16x2_abs_1_25));
                                        amax_pair_15 = bf16x2_max_25;
                                    }
                                    uint16_t bf16_max_15;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_15) : "h"((uint16_t)(amax_pair_15 & 65535)), "h"((uint16_t)(amax_pair_15 >> 16)));
                                    float amax_15;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_15) : "h"((uint16_t)(bf16_max_15)));
                                    float scale_15 = fmaxf(amax_15 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_119;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_119) : "f"(scale_15), "f"(scale_15));
                                    unsigned int scale_byte_15 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_119 & 255;
                                    unsigned int inverse_lane_15 = 254 - scale_byte_15 << 7;
                                    unsigned int inverse_15 = inverse_lane_15 | inverse_lane_15 << 16;
                                    unsigned int words_15[8];
                                    #pragma unroll
                                    for (int i_61 = 0; i_61 < 8; i_61++) {
                                        uint32_t bf16x2_mul_26;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(block_13[i_61 * 2]), "r"(inverse_15));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(block_13[i_61 * 2 + 1]), "r"(inverse_15));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_120;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_120) : "r"(bf16x2_mul_1_25));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_120) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_15[i_61] = (unsigned int)e4m3x2_bf16x2_25 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_120 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_15 << 8;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_15[0]), "r"(words_15[1]), "r"(words_15[2]), "r"(words_15[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_15[4]), "r"(words_15[5]), "r"(words_15[6]), "r"(words_15[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 160), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_14[16];
                                    #pragma unroll
                                    for (int j_28 = 0; j_28 < 16; j_28++) {
                                        block_14[j_28] = packed_6[96 + j_28];
                                    }
                                    #pragma unroll
                                    for (int j_29 = 0; j_29 < 4; j_29++) {
                                        unsigned int address_14 = d_smem_addr + (unsigned int)(tid * 64) + (unsigned int)(j_29 * 16);
                                        address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_14 - smem_v41_addr)), "r"(packed_6[96 + 4 * j_29]), "r"(packed_6[96 + 4 * j_29 + 1]), "r"(packed_6[96 + 4 * j_29 + 2]), "r"(packed_6[96 + 4 * j_29 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 6, 0, 0, d_smem_addr);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_16;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_16) : "r"(block_14[0]));
                                    unsigned int amax_pair_16 = bf16x2_abs_16;
                                    #pragma unroll
                                    for (int i_62 = 1; i_62 < 16; i_62++) {
                                        uint32_t bf16x2_abs_1_26;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_26) : "r"(block_14[i_62]));
                                        uint32_t bf16x2_max_26;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_26) : "r"(amax_pair_16), "r"(bf16x2_abs_1_26));
                                        amax_pair_16 = bf16x2_max_26;
                                    }
                                    uint16_t bf16_max_16;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_16) : "h"((uint16_t)(amax_pair_16 & 65535)), "h"((uint16_t)(amax_pair_16 >> 16)));
                                    float amax_16;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_16) : "h"((uint16_t)(bf16_max_16)));
                                    float scale_16 = fmaxf(amax_16 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_121;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_121) : "f"(scale_16), "f"(scale_16));
                                    unsigned int scale_byte_16 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_121 & 255;
                                    unsigned int inverse_lane_16 = 254 - scale_byte_16 << 7;
                                    unsigned int inverse_16 = inverse_lane_16 | inverse_lane_16 << 16;
                                    unsigned int words_16[8];
                                    #pragma unroll
                                    for (int i_63 = 0; i_63 < 8; i_63++) {
                                        uint32_t bf16x2_mul_27;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(block_14[i_63 * 2]), "r"(inverse_16));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(block_14[i_63 * 2 + 1]), "r"(inverse_16));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_122;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_122) : "r"(bf16x2_mul_1_26));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_122) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_16[i_63] = (unsigned int)e4m3x2_bf16x2_26 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_122 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_16 << 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_16[0]), "r"(words_16[1]), "r"(words_16[2]), "r"(words_16[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_16[4]), "r"(words_16[5]), "r"(words_16[6]), "r"(words_16[7]) : "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 192), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    unsigned int block_15[16];
                                    #pragma unroll
                                    for (int j_30 = 0; j_30 < 16; j_30++) {
                                        block_15[j_30] = packed_6[112 + j_30];
                                    }
                                    #pragma unroll
                                    for (int j_31 = 0; j_31 < 4; j_31++) {
                                        unsigned int address_15 = d_smem_addr + 8192 + (unsigned int)(tid * 64) + (unsigned int)(j_31 * 16);
                                        address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v41_addr + (address_15 - smem_v41_addr)), "r"(packed_6[112 + 4 * j_31]), "r"(packed_6[112 + 4 * j_31 + 1]), "r"(packed_6[112 + 4 * j_31 + 2]), "r"(packed_6[112 + 4 * j_31 + 3]) : "memory");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&up_routed_out), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 8 + 7, 0, 0, d_smem_addr + 8192);
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    uint32_t bf16x2_abs_17;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_17) : "r"(block_15[0]));
                                    unsigned int amax_pair_17 = bf16x2_abs_17;
                                    #pragma unroll
                                    for (int i_64 = 1; i_64 < 16; i_64++) {
                                        uint32_t bf16x2_abs_1_27;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_27) : "r"(block_15[i_64]));
                                        uint32_t bf16x2_max_27;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_27) : "r"(amax_pair_17), "r"(bf16x2_abs_1_27));
                                        amax_pair_17 = bf16x2_max_27;
                                    }
                                    uint16_t bf16_max_17;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_17) : "h"((uint16_t)(amax_pair_17 & 65535)), "h"((uint16_t)(amax_pair_17 >> 16)));
                                    float amax_17;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_17) : "h"((uint16_t)(bf16_max_17)));
                                    float scale_17 = fmaxf(amax_17 * 0.002232143f, 1e-12f);
                                    uint16_t _cvt_rp_satfinite_ue8m0x2_f32_123;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_123) : "f"(scale_17), "f"(scale_17));
                                    unsigned int scale_byte_17 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_123 & 255;
                                    unsigned int inverse_lane_17 = 254 - scale_byte_17 << 7;
                                    unsigned int inverse_17 = inverse_lane_17 | inverse_lane_17 << 16;
                                    unsigned int words_17[8];
                                    #pragma unroll
                                    for (int i_65 = 0; i_65 < 8; i_65++) {
                                        uint32_t bf16x2_mul_28;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(block_15[i_65 * 2]), "r"(inverse_17));
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
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(block_15[i_65 * 2 + 1]), "r"(inverse_17));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_124;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_124) : "r"(bf16x2_mul_1_27));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_124) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        words_17[i_65] = (unsigned int)e4m3x2_bf16x2_27 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_124 << 16;
                                    }
                                    scale_word_5 = scale_word_5 | scale_byte_17 << 24;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32)), "r"(words_17[0]), "r"(words_17[1]), "r"(words_17[2]), "r"(words_17[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v42_addr + (unsigned int)(tid * 32 + 16)), "r"(words_17[4]), "r"(words_17[5]), "r"(words_17[6]), "r"(words_17[7]) : "memory");
                                    smem_v44[tid % 32 * 4 + tid / 32] = scale_word_5;
                                    scale_word_5 = 0;
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_q_store)), "r"(y_3 * 256 + 224), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(smem_v42_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        tma_store_3d((&up_sc_store), 0, 0, (x_3 * 2 + cta_rank_0) * i_tiles + y_3 * 2 + 1, smem_v44_addr);
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
                                                bool enabled_value_7 = 1;
                                                if (enabled_value_7 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_3 + x_3) * (intermediate / 256) + y_3))), "r"(static_cast<unsigned int>(1)) : "memory");
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
                    } else {
                        int col_blocks_6 = intermediate / 128;
                        int num_tiles_1 = tokens / 128 * col_blocks_6;
                        int macro_row_offset_1 = 0;
                        int global_mini_4 = tile_word_0;
                        int mini_tiles = mini_size / 128 * col_blocks_6;
                        int first_tile_1 = (tile_word_1 - 2 * mini_gate) * 6 + cta_rank_0 * 3 + global_mini_4 * mini_tiles;
                        int tile_end_1 = ((num_tiles_1) < ((global_mini_4 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_4 + 1) * mini_tiles));
                        int macro_tiles_1 = macro_size / 128;
                        if (first_tile_1 < tile_end_1) {
                            int first_row_3 = first_tile_1 / col_blocks_6;
                            int first_col_1 = first_tile_1 % col_blocks_6;
                            if (tid == 0) {
                                if (tile_end_1 > first_tile_1) {
                                    int row_4_1 = first_row_3;
                                    int col_0 = first_col_1;
                                    if (col_0 >= col_blocks_6) {
                                        row_4_1 = row_4_1 + 1;
                                        col_0 = col_0 - col_blocks_6;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 3)) * 8, 65536);
                                    int parent_1 = row_4_1 / 2 * (intermediate / 256) + col_0 / 2;
                                    int poll_5_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_first) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                    int poll_5 = poll_5_first;
                                    while (poll_5 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_5_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_next) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                        poll_5 = poll_5_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + (unsigned int)(swiglu_seq % 3 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_4_1 - macro_row_offset_1) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 3)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + (unsigned int)(swiglu_seq % 3 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_4_1 - macro_row_offset_1) * 128), "r"(col_0 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)swiglu_seq % 3)) * 8) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 1) {
                                    int row_5_1 = first_row_3;
                                    int col_1_1 = first_col_1 + 1;
                                    if (col_1_1 >= col_blocks_6) {
                                        row_5_1 = row_5_1 + 1;
                                        col_1_1 = col_1_1 - col_blocks_6;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 3)) * 8, 65536);
                                    int parent_2 = row_5_1 / 2 * (intermediate / 256) + col_1_1 / 2;
                                    int poll_6_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_first) : "l"(gate_ready + (shared_gate + parent_2)) : "memory");
                                    int poll_6 = poll_6_first;
                                    while (poll_6 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_6_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_next) : "l"(gate_ready + (shared_gate + parent_2)) : "memory");
                                        poll_6 = poll_6_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + (unsigned int)((swiglu_seq + 1) % 3 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_5_1 - macro_row_offset_1) * 128), "r"(col_1_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 3)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + (unsigned int)((swiglu_seq + 1) % 3 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_5_1 - macro_row_offset_1) * 128), "r"(col_1_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 1) % 3)) * 8) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 2) {
                                    int row_6_1 = first_row_3;
                                    int col_2_1 = first_col_1 + 2;
                                    if (col_2_1 >= col_blocks_6) {
                                        row_6_1 = row_6_1 + 1;
                                        col_2_1 = col_2_1 - col_blocks_6;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 2) % 3)) * 8, 65536);
                                    int parent_3 = row_6_1 / 2 * (intermediate / 256) + col_2_1 / 2;
                                    int poll_7_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_first) : "l"(gate_ready + (shared_gate + parent_3)) : "memory");
                                    int poll_7 = poll_7_first;
                                    while (poll_7 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_7_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_next) : "l"(gate_ready + (shared_gate + parent_3)) : "memory");
                                        poll_7 = poll_7_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + (unsigned int)((swiglu_seq + 2) % 3 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_6_1 - macro_row_offset_1) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 2) % 3)) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + (unsigned int)((swiglu_seq + 2) % 3 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_6_1 - macro_row_offset_1) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + ((int)((unsigned int)(swiglu_seq + 2) % 3)) * 8) : "memory");
                                }
                            }
                            #pragma unroll 1
                            for (int stage_3 = 0; stage_3 < 3; stage_3++) {
                                if (tile_end_1 > first_tile_1 + stage_3) {
                                    int _stage_swiglu_arrived_1 = (int)((unsigned int)(swiglu_seq + stage_3) % 3);
                                    int _parity_swiglu_arrived_1 = (int)((unsigned int)(swiglu_seq + stage_3) / 3 % 2);
                                    mbarrier_wait(swiglu_arrived_addr + (_stage_swiglu_arrived_1) * 8, _parity_swiglu_arrived_1);
                                    int slot_2 = (swiglu_seq + stage_3) % 3;
                                    int row_15 = first_row_3;
                                    int col_14 = first_col_1 + stage_3;
                                    if (col_14 >= col_blocks_6) {
                                        row_15 = row_15 + 1;
                                        col_14 = col_14 - col_blocks_6;
                                    }
                                    float gate_1[64];
                                    float up_1[64];
                                    float denominator_1[64];
                                    int warp_2 = tid / 32;
                                    int local_warp_1 = warp_2 / 4 + warp_2 % 4 * 2;
                                    int lane_7 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                        unsigned int packed_7[4];
                                        unsigned int address_16 = gate_smem_addr + (unsigned int)(slot_2 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_7 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_7 % 16) * 64 + (tile_col_3 * 16 + lane_7 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_7[0]), "=r"(packed_7[1]), "=r"(packed_7[2]), "=r"(packed_7[3])
                                            : "r"(address_16 ^ (address_16 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_7 = 0; pair_7 < 4; pair_7++) {
                                            float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_7[pair_7]));
                                            gate_1[tile_col_3 * 8 + pair_7 * 2] = cvt_f32_2.x;
                                            gate_1[tile_col_3 * 8 + pair_7 * 2 + 1] = cvt_f32_2.y;
                                        }
                                    }
                                    int warp_1_1 = tid / 32;
                                    int local_warp_1_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                    int lane_1_1 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                        unsigned int packed_8[4];
                                        unsigned int address_17 = up_smem_addr + (unsigned int)(slot_2 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_1_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_1_1 * 16 + lane_1_1 % 16) * 64 + (tile_col_4 * 16 + lane_1_1 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_8[0]), "=r"(packed_8[1]), "=r"(packed_8[2]), "=r"(packed_8[3])
                                            : "r"(address_17 ^ (address_17 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                            float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(packed_8[pair_8]));
                                            up_1[tile_col_4 * 8 + pair_8 * 2] = cvt_f32_3.x;
                                            up_1[tile_col_4 * 8 + pair_8 * 2 + 1] = cvt_f32_3.y;
                                        }
                                    }
                                    #pragma unroll
                                    for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                        denominator_1[elem_5] = gate_1[elem_5] * -1.0f;
                                    }
                                    #pragma unroll
                                    for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                        float _exp_125 = expf(denominator_1[elem_6]);
                                        denominator_1[elem_6] = _exp_125;
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
                                            packed_9[pair_9] = __as_u32(__float22bfloat162_rn(make_float2(gate_1[tile_col_5 * 8 + pair_9 * 2], gate_1[tile_col_5 * 8 + pair_9 * 2 + 1])));
                                        }
                                        int row_0_1 = local_warp_5_1 * 16 + lane_6_1 % 16;
                                        int col_1_2 = tile_col_5 * 16 + lane_6_1 / 16 * 8;
                                        uint32_t _stmatrix_addr_126 = static_cast<uint32_t>(flat_view_addr + (unsigned int)((row_0_1 * 136 + col_1_2) * 2));
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_126), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[3]))
                                            : "memory");
                                    }
                                    __syncthreads();
                                    int out_base = slot_2 * 8192;
                                    if (tid < 128) {
                                        int r_4 = tid % 64;
                                        int row0_4 = r_4 * 2;
                                        int row1_4 = row0_4 + 1;
                                        int half_8 = tid / 64;
                                        int word_stride_4 = 68;
                                        int scale_base0_4 = (row0_4 % 32 * 4 + row0_4 / 32) * 4;
                                        int scale_base1_4 = (row1_4 % 32 * 4 + row1_4 / 32) * 4;
                                        #pragma unroll 1
                                        for (int j_32 = 0; j_32 < 2; j_32++) {
                                            int k_block_8 = (half_8 * 2 + j_32 + tid / 8) % 4;
                                            unsigned int pairs0_4[16];
                                            unsigned int pairs1_4[16];
                                            #pragma unroll
                                            for (int k_4 = 0; k_4 < 16; k_4++) {
                                                int col_3_1 = k_block_8 * 32 + (tid * 4 + k_4 * 2) % 32;
                                                unsigned int w0_4 = flat_view_0[col_3_1 * word_stride_4 + r_4];
                                                unsigned int w1_4 = flat_view_0[(col_3_1 + 1) * word_stride_4 + r_4];
                                                uint32_t lo;
                                                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(lo) : "r"(w0_4), "r"(w1_4));
                                                uint32_t hi;
                                                asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(hi) : "r"(w0_4), "r"(w1_4));
                                                pairs0_4[k_4] = lo;
                                                pairs1_4[k_4] = hi;
                                            }
                                            uint32_t bf16x2_abs_18;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_18) : "r"(pairs0_4[0]));
                                            unsigned int amax_pair_18 = bf16x2_abs_18;
                                            #pragma unroll
                                            for (int i_66 = 1; i_66 < 16; i_66++) {
                                                uint32_t bf16x2_abs_1_28;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_28) : "r"(pairs0_4[i_66]));
                                                uint32_t bf16x2_max_28;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_28) : "r"(amax_pair_18), "r"(bf16x2_abs_1_28));
                                                amax_pair_18 = bf16x2_max_28;
                                            }
                                            uint16_t bf16_max_18;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_18) : "h"((uint16_t)(amax_pair_18 & 65535)), "h"((uint16_t)(amax_pair_18 >> 16)));
                                            float amax_18;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18) : "h"((uint16_t)(bf16_max_18)));
                                            float scale_18 = fmaxf(amax_18 * 0.002232143f, 1e-12f);
                                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_127;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_127) : "f"(scale_18), "f"(scale_18));
                                            unsigned int scale_byte_18 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_127 & 255;
                                            unsigned int inverse_lane_18 = 254 - scale_byte_18 << 7;
                                            unsigned int inverse_18 = inverse_lane_18 | inverse_lane_18 << 16;
                                            unsigned int words_18[8];
                                            #pragma unroll
                                            for (int i_67 = 0; i_67 < 8; i_67++) {
                                                uint32_t bf16x2_mul_29;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(pairs0_4[i_67 * 2]), "r"(inverse_18));
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
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(pairs0_4[i_67 * 2 + 1]), "r"(inverse_18));
                                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_128;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_128) : "r"(bf16x2_mul_1_28));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_128) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                words_18[i_67] = (unsigned int)e4m3x2_bf16x2_28 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_128 << 16;
                                            }
                                            uint32_t bf16x2_abs_18_1;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_18_1) : "r"(pairs1_4[0]));
                                            unsigned int amax_pair_18_1 = bf16x2_abs_18_1;
                                            #pragma unroll
                                            for (int i_68 = 1; i_68 < 16; i_68++) {
                                                uint32_t bf16x2_abs_1_29;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_29) : "r"(pairs1_4[i_68]));
                                                uint32_t bf16x2_max_29;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_29) : "r"(amax_pair_18_1), "r"(bf16x2_abs_1_29));
                                                amax_pair_18_1 = bf16x2_max_29;
                                            }
                                            uint16_t bf16_max_18_1;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_18_1) : "h"((uint16_t)(amax_pair_18_1 & 65535)), "h"((uint16_t)(amax_pair_18_1 >> 16)));
                                            float amax_18_1;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_18_1) : "h"((uint16_t)(bf16_max_18_1)));
                                            float scale_18_1 = fmaxf(amax_18_1 * 0.002232143f, 1e-12f);
                                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_129;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_129) : "f"(scale_18_1), "f"(scale_18_1));
                                            unsigned int scale_byte_18_1 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_129 & 255;
                                            unsigned int inverse_lane_18_1 = 254 - scale_byte_18_1 << 7;
                                            unsigned int inverse_18_1 = inverse_lane_18_1 | inverse_lane_18_1 << 16;
                                            unsigned int words_18_1[8];
                                            #pragma unroll
                                            for (int i_69 = 0; i_69 < 8; i_69++) {
                                                uint32_t bf16x2_mul_30;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(pairs1_4[i_69 * 2]), "r"(inverse_18_1));
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
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(pairs1_4[i_69 * 2 + 1]), "r"(inverse_18_1));
                                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_130;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_130) : "r"(bf16x2_mul_1_29));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_130) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                words_18_1[i_69] = (unsigned int)e4m3x2_bf16x2_29 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_130 << 16;
                                            }
                                            #pragma unroll
                                            for (int k_5 = 0; k_5 < 8; k_5++) {
                                                int col_4_1 = k_block_8 * 32 + (tid * 4 + k_5 * 4) % 32;
                                                flat_view_3[out_base + (row0_4 * 128 + col_4_1) / 4] = words_18[k_5];
                                                flat_view_3[out_base + (row1_4 * 128 + col_4_1) / 4] = words_18_1[k_5];
                                            }
                                            {
                                                uint32_t _byte_131 = static_cast<uint32_t>(scale_byte_18) & 0xFFu;
                                                uint32_t _addr_131 = static_cast<uint32_t>(flat_view_addr_4 + (unsigned int)(out_base * 4 + scale_base0_4 + k_block_8));
                                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_131), "r"(_byte_131) : "memory");
                                            }
                                            {
                                                uint32_t _byte_132 = static_cast<uint32_t>(scale_byte_18_1) & 0xFFu;
                                                uint32_t _addr_132 = static_cast<uint32_t>(flat_view_addr_4 + (unsigned int)(out_base * 4 + scale_base1_4 + k_block_8));
                                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_132), "r"(_byte_132) : "memory");
                                            }
                                        }
                                    } else {
                                        int row_7_1 = tid - 128;
                                        unsigned int scale_word_6 = 0;
                                        #pragma unroll 1
                                        for (int j_33 = 0; j_33 < 4; j_33++) {
                                            int k_block_9 = (j_33 + (tid - 128) / 8) % 4;
                                            unsigned int pairs_4[16];
                                            #pragma unroll
                                            for (int k_6 = 0; k_6 < 16; k_6++) {
                                                int col_5_1 = k_block_9 * 32 + ((tid - 128) * 4 + k_6 * 2) % 32;
                                                float x0 = 0.0f;
                                                float x1 = 0.0f;
                                                pairs_4[k_6] = flat_view_0[(row_7_1 * 136 + col_5_1) / 2];
                                            }
                                            uint32_t bf16x2_abs_19;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_19) : "r"(pairs_4[0]));
                                            unsigned int amax_pair_19 = bf16x2_abs_19;
                                            #pragma unroll
                                            for (int i_70 = 1; i_70 < 16; i_70++) {
                                                uint32_t bf16x2_abs_1_30;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(bf16x2_abs_1_30) : "r"(pairs_4[i_70]));
                                                uint32_t bf16x2_max_30;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(bf16x2_max_30) : "r"(amax_pair_19), "r"(bf16x2_abs_1_30));
                                                amax_pair_19 = bf16x2_max_30;
                                            }
                                            uint16_t bf16_max_19;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(bf16_max_19) : "h"((uint16_t)(amax_pair_19 & 65535)), "h"((uint16_t)(amax_pair_19 >> 16)));
                                            float amax_19;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(amax_19) : "h"((uint16_t)(bf16_max_19)));
                                            float scale_19 = fmaxf(amax_19 * 0.002232143f, 1e-12f);
                                            uint16_t _cvt_rp_satfinite_ue8m0x2_f32_133;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_cvt_rp_satfinite_ue8m0x2_f32_133) : "f"(scale_19), "f"(scale_19));
                                            unsigned int scale_byte_19 = (unsigned int)_cvt_rp_satfinite_ue8m0x2_f32_133 & 255;
                                            unsigned int inverse_lane_19 = 254 - scale_byte_19 << 7;
                                            unsigned int inverse_19 = inverse_lane_19 | inverse_lane_19 << 16;
                                            unsigned int words_19[8];
                                            #pragma unroll
                                            for (int i_71 = 0; i_71 < 8; i_71++) {
                                                uint32_t bf16x2_mul_31;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(pairs_4[i_71 * 2]), "r"(inverse_19));
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
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(pairs_4[i_71 * 2 + 1]), "r"(inverse_19));
                                                uint16_t _cvt_satfinite_e4m3x2_bf16x2_134;
                                                #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                                asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_134) : "r"(bf16x2_mul_1_30));
                                                #else
                                                {
                                                const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
                                                const float _lo = __uint_as_float(_bf16x2 << 16);
                                                const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_134) : "f"(_hi), "f"(_lo));
                                                }
                                                #endif
                                                words_19[i_71] = (unsigned int)e4m3x2_bf16x2_30 | (unsigned int)_cvt_satfinite_e4m3x2_bf16x2_134 << 16;
                                            }
                                            scale_word_6 = scale_word_6 | scale_byte_19 << (unsigned int)(k_block_9 * 8);
                                            #pragma unroll
                                            for (int k_7 = 0; k_7 < 8; k_7++) {
                                                int col_6_1 = k_block_9 * 32 + ((tid - 128) * 4 + k_7 * 4) % 32;
                                                flat_view_1[out_base + (row_7_1 * 128 + col_6_1) / 4] = words_19[k_7];
                                            }
                                        }
                                        flat_view_2[out_base + row_7_1 % 32 * 4 + row_7_1 / 32] = scale_word_6;
                                    }
                                    __syncthreads();
                                    if (tid == 0) {
                                        int local_row = row_15 - macro_row_offset_1;
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_2d((&hidden_q_store), col_14 * 128, local_row * 128, flat_view_addr_1 + (unsigned int)(slot_2 * 32768));
                                        tma_store_3d((&hidden_sc_store), 0, 0, local_row * col_blocks_6 + col_14, flat_view_addr_2 + (unsigned int)(slot_2 * 32768));
                                        tma_store_2d((&hidden_t_store), local_row * 128, col_14 * 128, flat_view_addr_3 + (unsigned int)(slot_2 * 32768));
                                        tma_store_3d((&hidden_sc_t_store), 0, 0, col_14 * macro_tiles_1 + local_row, flat_view_addr_4 + (unsigned int)(slot_2 * 32768));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                if (tile_end_1 > first_tile_1) {
                                    int row_8_1 = first_row_3;
                                    if (col_blocks_6 <= first_col_1) {
                                        row_8_1 = row_8_1 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_8_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 1) {
                                    int row_9_1 = first_row_3;
                                    if (col_blocks_6 <= first_col_1 + 1) {
                                        row_9_1 = row_9_1 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_9_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                                if (tile_end_1 > first_tile_1 + 2) {
                                    int row_10_1 = first_row_3;
                                    if (col_blocks_6 <= first_col_1 + 2) {
                                        row_10_1 = row_10_1 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_10_1 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        int _min_136 = ((3) < (tile_end_1 - first_tile_1) ? (3) : (tile_end_1 - first_tile_1));
                        int _max_135 = ((0) > (_min_136) ? (0) : (_min_136));
                        swiglu_seq += _max_135;
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
                        int mini_2 = 0;
                        int task_0 = 0;
                        int swiglu_0 = 0;
                        if (shared_tasks <= decoded_cluster - comm_clusters) {
                            uint32_t fast_div_q_4 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks));
                            uint32_t fast_div_r_4 = (uint32_t)((unsigned int)(decoded_cluster - comm_clusters - shared_tasks)) - fast_div_q_4 * (uint32_t)(mini_tasks_div.divisor);
                            mini_2 = (int)fast_div_q_4;
                            task_0 = (int)fast_div_r_4;
                            if (task_0 >= 2 * mini_gate && task_0 < 2 * mini_gate + mini_swiglu) {
                                swiglu_0 = 1;
                            }
                        }
                        if (shared_tasks > decoded_cluster - comm_clusters && decoded_cluster - comm_clusters >= 2 * shared_gate_tasks && decoded_cluster - comm_clusters < 2 * shared_gate_tasks + shared_swiglu) {
                            swiglu_0 = 1;
                        }
                        int tile_word_6 = 0;
                        int tile_word_7 = 0;
                        int tile_word_8 = 0;
                        if (decoded_cluster < true_clusters) {
                            int word0_1 = 0;
                            int word1_1 = 0;
                            int word2_1 = 0;
                            if (shared_tasks <= decoded_cluster - comm_clusters) {
                                if (task_0 < 2 * mini_gate) {
                                    int gate_task_1 = task_0;
                                    if (task_0 >= mini_gate) {
                                        gate_task_1 = task_0 - mini_gate;
                                    }
                                    int word0_3 = 0;
                                    int word1_2 = 0;
                                    int word2_3 = 0;
                                    int first_block_3 = mini_2 * (mini_size / 256);
                                    int end_block_3 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                                    uint32_t fast_div_q_5 = (gate_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)gate_task_1), gate_col_blocks_div.multiplier) >> gate_col_blocks_div.shift_right) : (uint32_t)((unsigned int)gate_task_1);
                                    uint32_t fast_div_r_5 = (uint32_t)((unsigned int)gate_task_1) - fast_div_q_5 * (uint32_t)(gate_col_blocks_div.divisor);
                                    int block_16 = first_block_3 + (int)fast_div_q_5;
                                    if (block_16 < end_block_3) {
                                        word0_3 = counts[3 * experts + block_16];
                                        word1_2 = counts[experts + word0_3];
                                        word2_3 = counts[word0_3];
                                    }
                                    word0_1 = word0_3;
                                    word1_1 = word1_2;
                                    word2_1 = word2_3;
                                }
                            }
                            tile_word_6 = word0_1;
                            tile_word_7 = word1_1;
                            tile_word_8 = word2_1;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_0_addr), "r"((unsigned int)decoded_cluster), "r"((unsigned int)swiglu_0), "r"((unsigned int)task_0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_1_addr), "r"((unsigned int)mini_2), "r"(0), "r"((unsigned int)tile_word_6), "r"((unsigned int)tile_word_7) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(decode_words_2_addr), "r"((unsigned int)tile_word_8), "r"(0), "r"(0), "r"(0) : "memory");
                        mbarrier_arrive(decode_ready_addr);
                    }
                }
            }
            int _parity_decode_ready = (int)((unsigned int)iteration % 2);
            mbarrier_wait(decode_ready_addr, _parity_decode_ready);
            unsigned int decode_quad[4];
            __int128_t _smem_raw_137;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_137) : "r"(decode_words_0_addr) : "memory");
            decode_quad[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_137)[0];
            decode_quad[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_137)[1];
            decode_quad[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_137)[2];
            decode_quad[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_137)[3];
            unsigned int decode_quad_0[4];
            __int128_t _smem_raw_138;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_138) : "r"(decode_words_1_addr) : "memory");
            decode_quad_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_138)[0];
            decode_quad_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_138)[1];
            decode_quad_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_138)[2];
            decode_quad_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_138)[3];
            unsigned int decode_quad_1[4];
            __int128_t _smem_raw_139;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_139) : "r"(decode_words_2_addr) : "memory");
            decode_quad_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_139)[0];
            decode_quad_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_139)[1];
            decode_quad_1[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_139)[2];
            decode_quad_1[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_139)[3];
            cluster = (int)decode_quad[0];
            int tile_word_9 = (int)decode_quad[1];
            int tile_word_10 = (int)decode_quad[2];
            int tile_word_11 = (int)decode_quad_0[0];
            __syncwarp();
            bool _elect_sync_0 = elect_sync();
            if (_elect_sync_0) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((schedule_finished_addr) & 0xFEFFFFFF) : "memory");
            }
            if (tile_word_2 != 0 && cluster >= 0 && tile_word_9 == 0) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            tile_word_0 = tile_word_11;
            tile_word_1 = tile_word_10;
            tile_word_2 = tile_word_9;
            tile_word_3 = (int)decode_quad_0[2];
            tile_word_4 = (int)decode_quad_0[3];
            tile_word_5 = (int)decode_quad_1[0];
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
