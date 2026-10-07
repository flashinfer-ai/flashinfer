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
#define TMEM_ACCUMULATOR_OFFSET 0
#define TMEM_SF_A_OFFSET 256
#define TMEM_SF_B_OFFSET 280
#define TMEM_RESERVED_OFFSET 328
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
#define SMEM_B_SMEM_OFF 99328
#define SMEM_B_SMEM_STAGE_BYTES 16384
#define SMEM_B_SMEM_STRIDE 16384
#define SMEM_A_FP8_SMEM_OFF 1024
#define SMEM_A_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_A_FP8_SMEM_STRIDE 16384
#define SMEM_B_FP8_SMEM_OFF 99328
#define SMEM_B_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_B_FP8_SMEM_STRIDE 16384
#define SMEM_A_SC_SMEM_OFF 197632
#define SMEM_A_SC_SMEM_STAGE_BYTES 512
#define SMEM_A_SC_SMEM_STRIDE 512
#define SMEM_B_SC_SMEM_OFF 200704
#define SMEM_B_SC_SMEM_STAGE_BYTES 1024
#define SMEM_B_SC_SMEM_STRIDE 1024
#define SMEM_D_SMEM_OFF 206848
#define SMEM_D_SMEM_STAGE_BYTES 8192
#define SMEM_D_SMEM_STRIDE 8192
#define SMEM_D_FP8_SMEM_OFF 223232
#define SMEM_D_FP8_SMEM_STAGE_BYTES 4096
#define SMEM_D_FP8_SMEM_STRIDE 4096
#define SMEM_D_SC_SMEM_OFF 227328
#define SMEM_D_SC_SMEM_STAGE_BYTES 1024
#define SMEM_D_SC_SMEM_STRIDE 1024
#define SMEM_GATE_SMEM_OFF 1024
#define SMEM_GATE_SMEM_STAGE_BYTES 32768
#define SMEM_GATE_SMEM_STRIDE 32768
#define SMEM_UP_SMEM_OFF 99328
#define SMEM_UP_SMEM_STAGE_BYTES 32768
#define SMEM_UP_SMEM_STRIDE 32768
#define SMEM_HIDDEN_SMEM_OFF 197632
#define SMEM_HIDDEN_SMEM_STAGE_BYTES 32768
#define SMEM_HIDDEN_SMEM_STRIDE 32768
#define SMEM_GATE_FLAT_OFF 1024
#define SMEM_GATE_FLAT_STAGE_BYTES 98304
#define SMEM_GATE_FLAT_STRIDE 98304
#define SMEM_UP_FLAT_OFF 99328
#define SMEM_UP_FLAT_STAGE_BYTES 98304
#define SMEM_UP_FLAT_STRIDE 98304
#define SMEM_HIDDEN_FLAT_OFF 197632
#define SMEM_HIDDEN_FLAT_STAGE_BYTES 32768
#define SMEM_HIDDEN_FLAT_STRIDE 32768
#define SMEM_GATE_WORDS_OFF 1024
#define SMEM_GATE_WORDS_STAGE_BYTES 98304
#define SMEM_GATE_WORDS_STRIDE 98304
#define SMEM_UP_WORDS_OFF 99328
#define SMEM_UP_WORDS_STAGE_BYTES 98304
#define SMEM_UP_WORDS_STRIDE 98304
#define SMEM_HIDDEN_WORDS_OFF 197632
#define SMEM_HIDDEN_WORDS_STAGE_BYTES 32768
#define SMEM_HIDDEN_WORDS_STRIDE 32768
#define SMEM_DISPATCH_SMEM_OFF 1024
#define SMEM_DISPATCH_SMEM_STAGE_BYTES 131072
#define SMEM_DISPATCH_SMEM_STRIDE 131072
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 131072
#define SMEM_DISPATCH_WORDS_STRIDE 131072
#define SMEM_DISPATCH_QUANT_OFF 132096
#define SMEM_DISPATCH_QUANT_STAGE_BYTES 67584
#define SMEM_DISPATCH_QUANT_STRIDE 67584
#define SMEM_DISPATCH_WEIGHTS_OFF 199680
#define SMEM_DISPATCH_WEIGHTS_STAGE_BYTES 512
#define SMEM_DISPATCH_WEIGHTS_STRIDE 512
#define SMEM_COMBINE_SMEM_OFF 1024
#define SMEM_COMBINE_SMEM_STAGE_BYTES 229376
#define SMEM_COMBINE_SMEM_STRIDE 229376
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 440

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




__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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
kernel_cake_mok_forward_mxfp8(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap x_routed, const __grid_constant__ CUtensorMap x_routed_sc, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap wg_routed, const __grid_constant__ CUtensorMap wu_routed, const __grid_constant__ CUtensorMap wd_routed, const __grid_constant__ CUtensorMap wg_routed_sc, const __grid_constant__ CUtensorMap wu_routed_sc, const __grid_constant__ CUtensorMap wd_routed_sc, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_routed_fp8, const __grid_constant__ CUtensorMap up_routed_fp8, const __grid_constant__ CUtensorMap gate_routed_sc, const __grid_constant__ CUtensorMap up_routed_sc, const __grid_constant__ CUtensorMap gate_shared_in, const __grid_constant__ CUtensorMap up_shared_in, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_routed_fp8, const __grid_constant__ CUtensorMap hidden_routed_sc, const __grid_constant__ CUtensorMap hidden_routed_fp8_t, const __grid_constant__ CUtensorMap hidden_routed_sc_t, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap hidden_routed_in, const __grid_constant__ CUtensorMap hidden_routed_in_sc, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap y_routed, const __grid_constant__ CUtensorMap x_dispatch_fp8, const __grid_constant__ CUtensorMap x_dispatch_sc, const __grid_constant__ CUtensorMap x_dispatch_fp8_t, const __grid_constant__ CUtensorMap x_dispatch_sc_t, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, int swiglu_clamped, int recompute_only)
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
    #define schedule_arrived_addr (mbar_base + 232)
    #define schedule_finished_addr (mbar_base + 240)
    #define drain_arrived_0_addr (mbar_base + 248)
    #define drain_arrived_1_addr (mbar_base + 256)
    #define drain_arrived_2_addr (mbar_base + 264)
    #define drain_arrived_3_addr (mbar_base + 272)
    #define drain_arrived_4_addr (mbar_base + 280)
    #define drain_arrived_5_addr (mbar_base + 288)
    #define drain_arrived_6_addr (mbar_base + 296)
    #define drain_arrived_7_addr (mbar_base + 304)
    #define drain_finished_0_addr (mbar_base + 312)
    #define drain_finished_1_addr (mbar_base + 320)
    #define drain_finished_2_addr (mbar_base + 328)
    #define drain_finished_3_addr (mbar_base + 336)
    #define drain_finished_4_addr (mbar_base + 344)
    #define drain_finished_5_addr (mbar_base + 352)
    #define drain_finished_6_addr (mbar_base + 360)
    #define drain_finished_7_addr (mbar_base + 368)
    #define dispatch_arrived_addr (mbar_base + 376)
    #define combine_arrived_addr (mbar_base + 384)

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
    __nv_bfloat16* b_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int b_smem_addr = smem + 99328;
    uint8_t* a_fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a_fp8_smem_addr = smem + 1024;
    uint8_t* b_fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int b_fp8_smem_addr = smem + 99328;
    unsigned int* a_sc_smem = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int a_sc_smem_addr = smem + 197632;
    unsigned int* b_sc_smem = reinterpret_cast<unsigned int*>(smem_raw + 200704);
    const int b_sc_smem_addr = smem + 200704;
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
    uint8_t* d_fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 223232);
    const int d_fp8_smem_addr = smem + 223232;
    unsigned int* d_sc_smem = reinterpret_cast<unsigned int*>(smem_raw + 227328);
    const int d_sc_smem_addr = smem + 227328;
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int gate_smem_addr = smem + 1024;
    __nv_bfloat16* up_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int up_smem_addr = smem + 99328;
    __nv_bfloat16* hidden_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_smem_addr = smem + 197632;
    __nv_bfloat16* gate_flat = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int gate_flat_addr = smem + 1024;
    __nv_bfloat16* up_flat = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int up_flat_addr = smem + 99328;
    __nv_bfloat16* hidden_flat = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_flat_addr = smem + 197632;
    unsigned int* gate_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int gate_words_addr = smem + 1024;
    unsigned int* up_words = reinterpret_cast<unsigned int*>(smem_raw + 99328);
    const int up_words_addr = smem + 99328;
    unsigned int* hidden_words = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int hidden_words_addr = smem + 197632;
    __nv_bfloat16* dispatch_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int dispatch_smem_addr = smem + 1024;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int dispatch_words_addr = smem + 1024;
    unsigned int* dispatch_quant = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int dispatch_quant_addr = smem + 132096;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + 199680);
    const int dispatch_weights_addr = smem + 199680;
    __nv_bfloat16* combine_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int combine_smem_addr = smem + 1024;
    int tokens = num_tokens[0];
    int shared_rows = (local_tokens + 255) / 256;
    int shared_gate = shared_rows * (intermediate / 256);
    int mini_gate = mini_size / 256 * (intermediate / 256);
    int shared_swiglu = (shared_rows * 2 * (intermediate / 128) + 5) / 6;
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int shared_down = shared_rows * (hidden / 256);
    int mini_down = mini_size / 256 * (hidden / 256);
    if (recompute_only != 0) {
        shared_down = 0;
        mini_down = 0;
    }
    int shared_tasks = 2 * shared_gate + shared_swiglu + shared_down;
    int mini_tasks = 2 * mini_gate + mini_swiglu + mini_down;
    int comm_clusters = comm_sms / 2;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis_per_macro = macro_size / mini_size;
    int true_minis = (tokens + mini_size - 1) / mini_size;
    int last_minis = true_minis - (macros - 1) * minis_per_macro;
    int true_clusters = comm_clusters + shared_tasks + true_minis * mini_tasks;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (27 pipeline groups, 0 ordered-sequence groups, 55 barriers)
    // Mbarriers at smem_raw[0..440)

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
        // output_finished: 1 barriers, init_count=2
        mbarrier_init(smem + 224, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 232, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 240, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 248, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 256, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 264, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 272, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 280, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 288, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 296, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 304, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 312, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 320, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 328, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 336, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 344, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 352, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 360, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 368, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 376, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 384, 1);
        mbarrier_init(smem + 392, 1);
        mbarrier_init(smem + 400, 1);
        mbarrier_init(smem + 408, 1);
        mbarrier_init(smem + 416, 1);
        mbarrier_init(smem + 424, 1);
        mbarrier_init(smem + 432, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 440);
    if (warp == 0) {
        int _tmem_hold = smem + 440;
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
    const int tmem_accumulator = taddr;
    const int tmem_sf_a = taddr + 256;
    const int tmem_sf_b = taddr + 280;
    unsigned int taddr_1 = reinterpret_cast<const volatile unsigned int*>(reinterpret_cast<uint8_t*>(smem_raw) + CAKE_TMEM_HOLD_OFFSET)[0];
    int cluster = bid / 2;
    int cta_rank_0 = cta_rank;
    unsigned int gemm_bits = 4294901760;
    unsigned int swiglu_bits = 4294901760;
    unsigned int dispatch_bits = 4294901760;
    unsigned int combine_bits = 4294901760;
    int macro_row_blocks = macro_size / 128;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        if (macros > 0) {
            int _min_0 = ((macro_size) < (tokens - (macros - 1) * macro_size) ? (macro_size) : (tokens - (macros - 1) * macro_size));
            int last_rows = _min_0;
            int last_dispatch = last_rows / 128 * ((hidden + 511) / 512);
            #pragma unroll 1
            for (int task = comm_cta; task < last_dispatch; task += comm_sms) {
                unsigned int phase_bits = dispatch_bits;
                int col_blocks = (hidden + 511) / 512;
                int macro_offset = (macros - 1) * macro_size;
                int _min_1 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
                int macro_tokens = _min_1;
                if (task < macro_tokens / 128 * col_blocks) {
                    int row = task / col_blocks * 128;
                    int col_block = task % col_blocks;
                    int _min_2 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                    int chunk_cols = _min_2;
                    unsigned int chunk_bytes = (unsigned int)(chunk_cols * 2);
                    int peer = -1;
                    int peer_token = -1;
                    if (tid < 128) {
                        peer = schedule_rank[macro_offset + row + tid];
                        peer_token = schedule_token[macro_offset + row + tid];
                    }
                    uint32_t _cta_count_0 = __syncthreads_count(peer >= 0);
                    if (tid == 0) {
                        int previous_offset = macros * macro_size;
                        int _min_3 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                        int previous_tokens = _min_3;
                        if (row < previous_tokens) {
                            int previous_mini = (previous_offset + row) / mini_size;
                            int _min_4 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                            int mini_rows = _min_4;
                            int required = (mini_rows + 255) / 256 * (hidden / 256) * 2;
                        }
                        mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_0 * chunk_bytes);
                    }
                    __syncthreads();
                    if (peer >= 0) {
                        cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2)), chunk_cols * 2, dispatch_arrived_addr);
                    } else if (tid < 128) {
                        #pragma unroll
                        for (int vec = 0; vec < 64; vec++) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_smem) + (tid * 1024 + vec * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                    mbarrier_wait(dispatch_arrived_addr, phase_bits & 1);
                    phase_bits = phase_bits ^ 1;
                    float inv_e4m3_max = 0.002232142857f;
                    float scale_floor = 1e-12f;
                    int row_block = row / 128;
                    int num_subtiles = chunk_cols / 128;
                    #pragma unroll
                    for (int subtile = 0; subtile < 4; subtile++) {
                        if (num_subtiles > subtile) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            __syncthreads();
                            int half = tid % 128;
                            if (tid < 128) {
                                int t_row = half % 64 * 2 + half / 64;
                                int rotation = half / 8;
                                unsigned int t_scale_word = 0;
                                #pragma unroll 1
                                for (int j = 0; j < 4; j++) {
                                    int k_block = (j + rotation) % 4;
                                    unsigned int t_words[16];
                                    #pragma unroll
                                    for (int k = 0; k < 16; k++) {
                                        int src_row = k_block * 32 + (half * 4 + k * 2) % 32;
                                        float v0 = (float)dispatch_smem[src_row * 512 + subtile * 128 + t_row];
                                        float v1 = (float)dispatch_smem[(src_row + 1) * 512 + subtile * 128 + t_row];
                                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(v0, v1));
                                        t_words[k] = __as_u32(_bf16x2_0);
                                    }
                                    unsigned int t_packed[8];
                                    uint32_t _bf16x2_abs_0;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(t_words[0]));
                                    unsigned int amax2 = _bf16x2_abs_0;
                                    #pragma unroll
                                    for (int k_1 = 1; k_1 < 16; k_1++) {
                                        uint32_t _bf16x2_abs_1;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(t_words[k_1]));
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
                                        unsigned int w0 = t_words[2 * i];
                                        unsigned int w1 = t_words[2 * i + 1];
                                        float _cvt_f32_bf16_1;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(w0 & 65535)));
                                        float v0_1 = _cvt_f32_bf16_1;
                                        float _cvt_f32_bf16_2;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(w0 >> 16)));
                                        float v1_1 = _cvt_f32_bf16_2;
                                        float _cvt_f32_bf16_3;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_3) : "h"((uint16_t)(w1 & 65535)));
                                        float v2 = _cvt_f32_bf16_3;
                                        float _cvt_f32_bf16_4;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(w1 >> 16)));
                                        float v3 = _cvt_f32_bf16_4;
                                        uint16_t _e4m3x2_f32_0;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(v1_1 * inv), "f"(v0_1 * inv));
                                        uint16_t lo = _e4m3x2_f32_0;
                                        uint16_t _e4m3x2_f32_1;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(v3 * inv), "f"(v2 * inv));
                                        uint16_t hi = _e4m3x2_f32_1;
                                        t_packed[i] = (unsigned int)lo | (unsigned int)hi << 16;
                                    }
                                    unsigned int t_scale_byte = scale_byte;
                                    #pragma unroll
                                    for (int i_1 = 0; i_1 < 8; i_1++) {
                                        int t_col = k_block * 32 + (half * 4 + i_1 * 4) % 32;
                                        dispatch_quant[8448 + subtile % 2 * 4096 + t_row * 32 + t_col / 4] = t_packed[i_1];
                                    }
                                    t_scale_word = t_scale_word | t_scale_byte << (unsigned int)(k_block * 8);
                                }
                                dispatch_quant[16640 + subtile % 2 * 128 + t_row % 32 * 4 + t_row / 32] = t_scale_word;
                            } else {
                                int n_row = half;
                                int rotation_1 = half / 8;
                                unsigned int words[64];
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 4; j_1++) {
                                    int k_block_j = (j_1 + rotation_1) % 4;
                                    #pragma unroll
                                    for (int k_2 = 0; k_2 < 16; k_2++) {
                                        int src_col = k_block_j * 32 + (half * 4 + k_2 * 2) % 32;
                                        words[j_1 * 16 + k_2] = dispatch_words[n_row * 256 + (subtile * 128 + src_col) / 2];
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                unsigned int n_scale_word = 0;
                                #pragma unroll
                                for (int j_2 = 0; j_2 < 4; j_2++) {
                                    int k_block_n = (j_2 + rotation_1) % 4;
                                    unsigned int n_packed[8];
                                    uint32_t _bf16x2_abs_2;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(words[j_2 * 16]));
                                    unsigned int amax2_1 = _bf16x2_abs_2;
                                    #pragma unroll
                                    for (int k_3 = 1; k_3 < 16; k_3++) {
                                        uint32_t _bf16x2_abs_3;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(words[j_2 * 16 + k_3]));
                                        uint32_t _bf16x2_max_1;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(amax2_1), "r"(_bf16x2_abs_3));
                                        amax2_1 = _bf16x2_max_1;
                                    }
                                    uint16_t _bf16_max_1;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_1) : "h"((uint16_t)(amax2_1 & 65535)), "h"((uint16_t)(amax2_1 >> 16)));
                                    float _cvt_f32_bf16_5;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_5) : "h"((uint16_t)(_bf16_max_1)));
                                    float amax_1 = _cvt_f32_bf16_5;
                                    float _max_1 = max_noftz(amax_1 * inv_e4m3_max, scale_floor);
                                    float scale_1 = _max_1;
                                    uint16_t _ue8m0x2_f32_1;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_1) : "f"(scale_1), "f"(scale_1));
                                    uint16_t codes_1 = _ue8m0x2_f32_1;
                                    unsigned int scale_byte_1 = (unsigned int)codes_1 & 255;
                                    unsigned int inv_bits_1 = 254 - scale_byte_1 << 23;
                                    float inv_1 = 0.0f;
                                    inv_1 = __uint_as_float(inv_bits_1);
                                    #pragma unroll
                                    for (int i_2 = 0; i_2 < 8; i_2++) {
                                        unsigned int w0_1 = words[j_2 * 16 + 2 * i_2];
                                        unsigned int w1_1 = words[j_2 * 16 + 2 * i_2 + 1];
                                        float _cvt_f32_bf16_6;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_6) : "h"((uint16_t)(w0_1 & 65535)));
                                        float v0_2 = _cvt_f32_bf16_6;
                                        float _cvt_f32_bf16_7;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_7) : "h"((uint16_t)(w0_1 >> 16)));
                                        float v1_2 = _cvt_f32_bf16_7;
                                        float _cvt_f32_bf16_8;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_8) : "h"((uint16_t)(w1_1 & 65535)));
                                        float v2_1 = _cvt_f32_bf16_8;
                                        float _cvt_f32_bf16_9;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_9) : "h"((uint16_t)(w1_1 >> 16)));
                                        float v3_1 = _cvt_f32_bf16_9;
                                        uint16_t _e4m3x2_f32_2;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(v1_2 * inv_1), "f"(v0_2 * inv_1));
                                        uint16_t lo_1 = _e4m3x2_f32_2;
                                        uint16_t _e4m3x2_f32_3;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(v3_1 * inv_1), "f"(v2_1 * inv_1));
                                        uint16_t hi_1 = _e4m3x2_f32_3;
                                        n_packed[i_2] = (unsigned int)lo_1 | (unsigned int)hi_1 << 16;
                                    }
                                    unsigned int n_scale_byte = scale_byte_1;
                                    #pragma unroll
                                    for (int i_3 = 0; i_3 < 8; i_3++) {
                                        int n_col = k_block_n * 32 + (half * 4 + i_3 * 4) % 32;
                                        dispatch_quant[subtile % 2 * 4096 + n_row * 32 + n_col / 4] = n_packed[i_3];
                                    }
                                    n_scale_word = n_scale_word | n_scale_byte << (unsigned int)(k_block_n * 8);
                                }
                                dispatch_quant[8192 + subtile % 2 * 128 + n_row % 32 * 4 + n_row / 32] = n_scale_word;
                            }
                            __syncthreads();
                            if (tid == 0) {
                                int col128 = col_block * 4 + subtile;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d((&x_dispatch_fp8), col128 * 128, row, dispatch_quant_addr + (unsigned int)(subtile % 2 * 4096 * 4));
                                tma_store_3d((&x_dispatch_sc), 0, (row_block * (hidden / 128) + col128) * 32, 0, dispatch_quant_addr + (unsigned int)((8192 + subtile % 2 * 128) * 4));
                                tma_store_2d((&x_dispatch_fp8_t), row, col128 * 128, dispatch_quant_addr + (unsigned int)((8448 + subtile % 2 * 4096) * 4));
                                tma_store_3d((&x_dispatch_sc_t), 0, (col128 * macro_row_blocks + row_block) * 32, 0, dispatch_quant_addr + (unsigned int)((16640 + subtile % 2 * 128) * 4));
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    if (tid == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + row) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                    __syncthreads();
                }
                dispatch_bits = phase_bits;
            }
            int macro = macros - 1;
            while (macro >= 0) {
                int _min_5 = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int macro_rows = _min_5;
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                if (recompute_only != 0) {
                    combine_tasks = 0;
                }
                int dispatch_tasks = 0;
                if (macro > 0) {
                    int _min_6 = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                    int previous_rows = _min_6;
                    dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                }
                int _max_2 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task_1 = comm_cta; task_1 < _max_2; task_1 += comm_sms) {
                    if (combine_tasks > task_1) {
                        unsigned int phase_bits_1 = combine_bits;
                        int col_blocks_1 = (hidden + 1023) / 1024;
                        int first_tile = task_1 * 7;
                        int macro_offset_1 = macro * macro_size;
                        int _min_7 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                        int macro_tokens_1 = _min_7;
                        int _min_8 = ((7) < (macro_tokens_1 / 16 * col_blocks_1 - first_tile) ? (7) : (macro_tokens_1 / 16 * col_blocks_1 - first_tile));
                        int valid_tiles = _min_8;
                        if (valid_tiles > 0) {
                            int first_row = first_tile / col_blocks_1 * 16 + tid;
                            int first_col = first_tile % col_blocks_1;
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
                                    peers[stage] = schedule_rank[macro_offset_1 + row_1];
                                    tokens_0[stage] = schedule_token[macro_offset_1 + row_1];
                                }
                                counts_1[stage] = 0;
                                if (valid_tiles > stage) {
                                    if (stage == 0 || column == 0) {
                                        uint32_t _cta_count_1 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _cta_count_1;
                                    } else {
                                        counts_1[stage] = counts_1[stage - 1];
                                    }
                                }
                                column = column + 1;
                                if (column == col_blocks_1) {
                                    column = 0;
                                    row_1 = row_1 + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini = (macro_offset_1 + first_row) / mini_size;
                                int last_mini = (macro_offset_1 + (first_tile + valid_tiles - 1) / col_blocks_1 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini = first_mini; mini < last_mini + 1; mini++) {
                                    int _min_9 = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                                    int mini_rows_1 = _min_9;
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    int32_t _relaxed_ld_0;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(y_ready + mini) : "memory");
                                    int value = _relaxed_ld_0;
                                    while (value < required_1) {
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
                                        int _min_10 = ((1024) < (hidden - columns[stage_1] * 1024) ? (1024) : (hidden - columns[stage_1] * 1024));
                                        unsigned int chunk_bytes_1 = (unsigned int)(_min_10 * 2);
                                        mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_1) * 8, counts_1[stage_1] * chunk_bytes_1);
                                    }
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 7; stage_2++) {
                                if (peers[stage_2] >= 0) {
                                    int _min_11 = ((1024) < (hidden - columns[stage_2] * 1024) ? (1024) : (hidden - columns[stage_2] * 1024));
                                    int chunk_cols_1 = _min_11;
                                    cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_2 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows[stage_2] * (unsigned long long)hidden + (unsigned long long)(columns[stage_2] * 1024)) * (unsigned long long)2)), chunk_cols_1 * 2, combine_arrived_addr + (stage_2) * 8);
                                }
                            }
                            #pragma unroll
                            for (int stage_3 = 0; stage_3 < 7; stage_3++) {
                                if (valid_tiles > stage_3) {
                                    mbarrier_wait(combine_arrived_addr + (stage_3) * 8, phase_bits_1 >> (unsigned int)stage_3 & 1);
                                    phase_bits_1 = phase_bits_1 ^ (unsigned int)(1 << stage_3);
                                    if (peers[stage_3] >= 0) {
                                        int _min_12 = ((1024) < (hidden - columns[stage_3] * 1024) ? (1024) : (hidden - columns[stage_3] * 1024));
                                        unsigned int chunk_bytes_2 = (unsigned int)(_min_12 * 2);
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        {
                                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[stage_3]]) + ((unsigned long long)tokens_0[stage_3] * (unsigned long long)hidden + (unsigned long long)(columns[stage_3] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_0), "r"(combine_smem_addr + (unsigned int)((stage_3 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_2))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            int warp_1 = tid / 32;
                            if (tid % 32 == 0 && warp_1 < valid_tiles && macro > 0) {
                                int row_done = (first_tile + warp_1) / col_blocks_1 * 16;
                                bool enabled_value = 1;
                                if (enabled_value != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_1 + row_done) / 128))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                        combine_bits = phase_bits_1;
                    }
                    if (dispatch_tasks > task_1) {
                        unsigned int phase_bits_2 = dispatch_bits;
                        int col_blocks_2 = (hidden + 511) / 512;
                        int macro_offset_2 = (macro - 1) * macro_size;
                        int _min_13 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
                        int macro_tokens_2 = _min_13;
                        if (task_1 < macro_tokens_2 / 128 * col_blocks_2) {
                            int row_2 = task_1 / col_blocks_2 * 128;
                            int col_block_1 = task_1 % col_blocks_2;
                            int _min_14 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                            int chunk_cols_2 = _min_14;
                            unsigned int chunk_bytes_3 = (unsigned int)(chunk_cols_2 * 2);
                            int peer_1 = -1;
                            int peer_token_1 = -1;
                            if (tid < 128) {
                                peer_1 = schedule_rank[macro_offset_2 + row_2 + tid];
                                peer_token_1 = schedule_token[macro_offset_2 + row_2 + tid];
                            }
                            uint32_t _cta_count_2 = __syncthreads_count(peer_1 >= 0);
                            if (tid == 0) {
                                int previous_offset_1 = macro * macro_size;
                                int _min_15 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                                int previous_tokens_1 = _min_15;
                                if (row_2 < previous_tokens_1) {
                                    int previous_mini_1 = (previous_offset_1 + row_2) / mini_size;
                                    int _min_16 = ((mini_size) < (tokens - previous_mini_1 * mini_size) ? (mini_size) : (tokens - previous_mini_1 * mini_size));
                                    int mini_rows_2 = _min_16;
                                    int required_2 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                                    bool enabled_value_1 = recompute_only == 0;
                                    if (enabled_value_1 != 0) {
                                        int32_t _relaxed_ld_2;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_2) : "l"(y_ready + previous_mini_1) : "memory");
                                        int value_1 = _relaxed_ld_2;
                                        while (value_1 < required_2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_3;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_3) : "l"(y_ready + previous_mini_1) : "memory");
                                            value_1 = _relaxed_ld_3;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    bool enabled_value_0 = recompute_only != 0;
                                    if (enabled_value_0 != 0) {
                                        int32_t _relaxed_ld_4;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(hidden_ready + (shared_rows + (previous_offset_1 + row_2) / 256)) : "memory");
                                        int value_2 = _relaxed_ld_4;
                                        while (value_2 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_5;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(hidden_ready + (shared_rows + (previous_offset_1 + row_2) / 256)) : "memory");
                                            value_2 = _relaxed_ld_5;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * chunk_bytes_3);
                            }
                            __syncthreads();
                            if (peer_1 >= 0) {
                                cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2)), chunk_cols_2 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_1 = 0; vec_1 < 64; vec_1++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_smem) + (tid * 1024 + vec_1 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                            mbarrier_wait(dispatch_arrived_addr, phase_bits_2 & 1);
                            phase_bits_2 = phase_bits_2 ^ 1;
                            float inv_e4m3_max_1 = 0.002232142857f;
                            float scale_floor_1 = 1e-12f;
                            int row_block_1 = row_2 / 128;
                            int num_subtiles_1 = chunk_cols_2 / 128;
                            #pragma unroll
                            for (int subtile_1 = 0; subtile_1 < 4; subtile_1++) {
                                if (num_subtiles_1 > subtile_1) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    __syncthreads();
                                    int half_1 = tid % 128;
                                    if (tid < 128) {
                                        int t_row_1 = half_1 % 64 * 2 + half_1 / 64;
                                        int rotation_2 = half_1 / 8;
                                        unsigned int t_scale_word_1 = 0;
                                        #pragma unroll 1
                                        for (int j_3 = 0; j_3 < 4; j_3++) {
                                            int k_block_1 = (j_3 + rotation_2) % 4;
                                            unsigned int t_words_1[16];
                                            #pragma unroll
                                            for (int k_4 = 0; k_4 < 16; k_4++) {
                                                int src_row_1 = k_block_1 * 32 + (half_1 * 4 + k_4 * 2) % 32;
                                                float v0_3 = (float)dispatch_smem[src_row_1 * 512 + subtile_1 * 128 + t_row_1];
                                                float v1_3 = (float)dispatch_smem[(src_row_1 + 1) * 512 + subtile_1 * 128 + t_row_1];
                                                __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(v0_3, v1_3));
                                                t_words_1[k_4] = __as_u32(_bf16x2_1);
                                            }
                                            unsigned int t_packed_1[8];
                                            uint32_t _bf16x2_abs_4;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(t_words_1[0]));
                                            unsigned int amax2_2 = _bf16x2_abs_4;
                                            #pragma unroll
                                            for (int k_5 = 1; k_5 < 16; k_5++) {
                                                uint32_t _bf16x2_abs_5;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(t_words_1[k_5]));
                                                uint32_t _bf16x2_max_2;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(amax2_2), "r"(_bf16x2_abs_5));
                                                amax2_2 = _bf16x2_max_2;
                                            }
                                            uint16_t _bf16_max_2;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_2) : "h"((uint16_t)(amax2_2 & 65535)), "h"((uint16_t)(amax2_2 >> 16)));
                                            float _cvt_f32_bf16_10;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_10) : "h"((uint16_t)(_bf16_max_2)));
                                            float amax_2 = _cvt_f32_bf16_10;
                                            float _max_3 = max_noftz(amax_2 * inv_e4m3_max_1, scale_floor_1);
                                            float scale_2 = _max_3;
                                            uint16_t _ue8m0x2_f32_2;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_2) : "f"(scale_2), "f"(scale_2));
                                            uint16_t codes_2 = _ue8m0x2_f32_2;
                                            unsigned int scale_byte_2 = (unsigned int)codes_2 & 255;
                                            unsigned int inv_bits_2 = 254 - scale_byte_2 << 23;
                                            float inv_2 = 0.0f;
                                            inv_2 = __uint_as_float(inv_bits_2);
                                            #pragma unroll
                                            for (int i_4 = 0; i_4 < 8; i_4++) {
                                                unsigned int w0_2 = t_words_1[2 * i_4];
                                                unsigned int w1_2 = t_words_1[2 * i_4 + 1];
                                                float _cvt_f32_bf16_11;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_11) : "h"((uint16_t)(w0_2 & 65535)));
                                                float v0_4 = _cvt_f32_bf16_11;
                                                float _cvt_f32_bf16_12;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_12) : "h"((uint16_t)(w0_2 >> 16)));
                                                float v1_4 = _cvt_f32_bf16_12;
                                                float _cvt_f32_bf16_13;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_13) : "h"((uint16_t)(w1_2 & 65535)));
                                                float v2_2 = _cvt_f32_bf16_13;
                                                float _cvt_f32_bf16_14;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_14) : "h"((uint16_t)(w1_2 >> 16)));
                                                float v3_2 = _cvt_f32_bf16_14;
                                                uint16_t _e4m3x2_f32_4;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(v1_4 * inv_2), "f"(v0_4 * inv_2));
                                                uint16_t lo_2 = _e4m3x2_f32_4;
                                                uint16_t _e4m3x2_f32_5;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(v3_2 * inv_2), "f"(v2_2 * inv_2));
                                                uint16_t hi_2 = _e4m3x2_f32_5;
                                                t_packed_1[i_4] = (unsigned int)lo_2 | (unsigned int)hi_2 << 16;
                                            }
                                            unsigned int t_scale_byte_1 = scale_byte_2;
                                            #pragma unroll
                                            for (int i_5 = 0; i_5 < 8; i_5++) {
                                                int t_col_1 = k_block_1 * 32 + (half_1 * 4 + i_5 * 4) % 32;
                                                dispatch_quant[8448 + subtile_1 % 2 * 4096 + t_row_1 * 32 + t_col_1 / 4] = t_packed_1[i_5];
                                            }
                                            t_scale_word_1 = t_scale_word_1 | t_scale_byte_1 << (unsigned int)(k_block_1 * 8);
                                        }
                                        dispatch_quant[16640 + subtile_1 % 2 * 128 + t_row_1 % 32 * 4 + t_row_1 / 32] = t_scale_word_1;
                                    } else {
                                        int n_row_1 = half_1;
                                        int rotation_3 = half_1 / 8;
                                        unsigned int words_1[64];
                                        #pragma unroll
                                        for (int j_4 = 0; j_4 < 4; j_4++) {
                                            int k_block_j_1 = (j_4 + rotation_3) % 4;
                                            #pragma unroll
                                            for (int k_6 = 0; k_6 < 16; k_6++) {
                                                int src_col_1 = k_block_j_1 * 32 + (half_1 * 4 + k_6 * 2) % 32;
                                                words_1[j_4 * 16 + k_6] = dispatch_words[n_row_1 * 256 + (subtile_1 * 128 + src_col_1) / 2];
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        unsigned int n_scale_word_1 = 0;
                                        #pragma unroll
                                        for (int j_5 = 0; j_5 < 4; j_5++) {
                                            int k_block_n_1 = (j_5 + rotation_3) % 4;
                                            unsigned int n_packed_1[8];
                                            uint32_t _bf16x2_abs_6;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(words_1[j_5 * 16]));
                                            unsigned int amax2_3 = _bf16x2_abs_6;
                                            #pragma unroll
                                            for (int k_7 = 1; k_7 < 16; k_7++) {
                                                uint32_t _bf16x2_abs_7;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(words_1[j_5 * 16 + k_7]));
                                                uint32_t _bf16x2_max_3;
                                                asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(amax2_3), "r"(_bf16x2_abs_7));
                                                amax2_3 = _bf16x2_max_3;
                                            }
                                            uint16_t _bf16_max_3;
                                            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_3) : "h"((uint16_t)(amax2_3 & 65535)), "h"((uint16_t)(amax2_3 >> 16)));
                                            float _cvt_f32_bf16_15;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_15) : "h"((uint16_t)(_bf16_max_3)));
                                            float amax_3 = _cvt_f32_bf16_15;
                                            float _max_4 = max_noftz(amax_3 * inv_e4m3_max_1, scale_floor_1);
                                            float scale_3 = _max_4;
                                            uint16_t _ue8m0x2_f32_3;
                                            asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_3) : "f"(scale_3), "f"(scale_3));
                                            uint16_t codes_3 = _ue8m0x2_f32_3;
                                            unsigned int scale_byte_3 = (unsigned int)codes_3 & 255;
                                            unsigned int inv_bits_3 = 254 - scale_byte_3 << 23;
                                            float inv_3 = 0.0f;
                                            inv_3 = __uint_as_float(inv_bits_3);
                                            #pragma unroll
                                            for (int i_6 = 0; i_6 < 8; i_6++) {
                                                unsigned int w0_3 = words_1[j_5 * 16 + 2 * i_6];
                                                unsigned int w1_3 = words_1[j_5 * 16 + 2 * i_6 + 1];
                                                float _cvt_f32_bf16_16;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(w0_3 & 65535)));
                                                float v0_5 = _cvt_f32_bf16_16;
                                                float _cvt_f32_bf16_17;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_17) : "h"((uint16_t)(w0_3 >> 16)));
                                                float v1_5 = _cvt_f32_bf16_17;
                                                float _cvt_f32_bf16_18;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_18) : "h"((uint16_t)(w1_3 & 65535)));
                                                float v2_3 = _cvt_f32_bf16_18;
                                                float _cvt_f32_bf16_19;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(w1_3 >> 16)));
                                                float v3_3 = _cvt_f32_bf16_19;
                                                uint16_t _e4m3x2_f32_6;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(v1_5 * inv_3), "f"(v0_5 * inv_3));
                                                uint16_t lo_3 = _e4m3x2_f32_6;
                                                uint16_t _e4m3x2_f32_7;
                                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(v3_3 * inv_3), "f"(v2_3 * inv_3));
                                                uint16_t hi_3 = _e4m3x2_f32_7;
                                                n_packed_1[i_6] = (unsigned int)lo_3 | (unsigned int)hi_3 << 16;
                                            }
                                            unsigned int n_scale_byte_1 = scale_byte_3;
                                            #pragma unroll
                                            for (int i_7 = 0; i_7 < 8; i_7++) {
                                                int n_col_1 = k_block_n_1 * 32 + (half_1 * 4 + i_7 * 4) % 32;
                                                dispatch_quant[subtile_1 % 2 * 4096 + n_row_1 * 32 + n_col_1 / 4] = n_packed_1[i_7];
                                            }
                                            n_scale_word_1 = n_scale_word_1 | n_scale_byte_1 << (unsigned int)(k_block_n_1 * 8);
                                        }
                                        dispatch_quant[8192 + subtile_1 % 2 * 128 + n_row_1 % 32 * 4 + n_row_1 / 32] = n_scale_word_1;
                                    }
                                    __syncthreads();
                                    if (tid == 0) {
                                        int col128_1 = col_block_1 * 4 + subtile_1;
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_2d((&x_dispatch_fp8), col128_1 * 128, row_2, dispatch_quant_addr + (unsigned int)(subtile_1 % 2 * 4096 * 4));
                                        tma_store_3d((&x_dispatch_sc), 0, (row_block_1 * (hidden / 128) + col128_1) * 32, 0, dispatch_quant_addr + (unsigned int)((8192 + subtile_1 % 2 * 128) * 4));
                                        tma_store_2d((&x_dispatch_fp8_t), row_2, col128_1 * 128, dispatch_quant_addr + (unsigned int)((8448 + subtile_1 % 2 * 4096) * 4));
                                        tma_store_3d((&x_dispatch_sc_t), 0, (col128_1 * macro_row_blocks + row_block_1) * 32, 0, dispatch_quant_addr + (unsigned int)((16640 + subtile_1 % 2 * 128) * 4));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_2 + row_2) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                            __syncthreads();
                        }
                        dispatch_bits = phase_bits_2;
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
                    if (compute >= 2 * shared_gate && compute < 2 * shared_gate + shared_swiglu) {
                        result = 1;
                    }
                } else {
                    int mini_task = (compute - shared_tasks) % mini_tasks;
                    if (mini_task >= 2 * mini_gate && mini_task < 2 * mini_gate + mini_swiglu) {
                        result = 1;
                    }
                }
            }
            if (compute < shared_gate) {
                int col_blocks_3 = intermediate / 256;
                int x = -1;
                int y = -1;
                int expert = -1;
                int k_start = 0;
                int k_end = 0;
                int first = 0;
                int row_blocks = (local_tokens + 255) / 256;
                if (compute < row_blocks * col_blocks_3) {
                    int supergroup = compute / (row_blocks * 8);
                    int full_cols = col_blocks_3 / 8 * 8;
                    int row_3 = 0;
                    int col = 0;
                    if (compute < row_blocks * full_cols) {
                        row_3 = compute % (row_blocks * 8) / 8;
                        col = supergroup * 8 + compute % 8;
                    } else {
                        row_3 = (compute - row_blocks * full_cols) / (col_blocks_3 - full_cols);
                        col = full_cols + (compute - row_blocks * full_cols) % (col_blocks_3 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_3 = row_blocks - row_3 - 1;
                    }
                    x = row_3;
                    y = col;
                    expert = 0;
                }
                unsigned int phase_bits_3 = gemm_bits;
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
                                int _min_17 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                int _max_5 = ((0) > (_min_17) ? (0) : (_min_17));
                                int mini_rows_3 = _max_5;
                                int required_3 = (mini_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                            }
                            int ring = 0;
                            #pragma unroll 1
                            for (int idx = 0; idx < iterations; idx++) {
                                mbarrier_wait(gemm_finished_addr + (ring) * 8, phase_bits_3 >> (unsigned int)(16 + ring) & 1);
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
                                phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << 16 + ring);
                                ring = (ring + 1) % 6;
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
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_1) * 8, 65536);
                                    mbarrier_wait(gemm_arrived_addr + (ring_1) * 8, phase_bits_3 >> (unsigned int)ring_1 & 1);
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
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_1) * 8, (uint16_t)(3));
                                    phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << ring_1);
                                    ring_1 = (ring_1 + 1) % 6;
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
                            for (int half_2 = 0; half_2 < 2; half_2++) {
                                unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 + half_2 * 16 << 16) + (unsigned int)(chunk * 32);
                                float _tmem_load_0[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                    : "r"(address));
                                #pragma unroll
                                for (int pair = 0; pair < 8; pair++) {
                                    __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]));
                                    packed[chunk * 16 + half_2 * 8 + pair] = __as_u32(_bf16x2_2);
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            int previous_offset_2 = macro_size;
                            int output_row = x * 256 + cta_rank_0 * 128;
                            int _min_18 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                            if (output_row < _min_18) {
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
                            for (int half_3 = 0; half_3 < 2; half_3++) {
                                #pragma unroll
                                for (int col_tile = 0; col_tile < 2; col_tile++) {
                                    int row_4 = warp_0 * 32 + half_3 * 16 + lane_1 % 16;
                                    int col_1 = col_tile * 16 + lane_1 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_4 * 32 + col_1) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset = chunk_1 * 16 + half_3 * 8 + col_tile * 4;
                                    uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + chunk_1), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
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
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows_1 + x) * (intermediate / 256) + y))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
                gemm_bits = phase_bits_3;
            } else if (compute < 2 * shared_gate) {
                int col_blocks_4 = intermediate / 256;
                int x_1 = -1;
                int y_1 = -1;
                int expert_1 = -1;
                int k_start_1 = 0;
                int k_end_1 = 0;
                int first_1 = 0;
                int row_blocks_1 = (local_tokens + 255) / 256;
                if (compute - shared_gate < row_blocks_1 * col_blocks_4) {
                    int supergroup_1 = (compute - shared_gate) / (row_blocks_1 * 8);
                    int full_cols_1 = col_blocks_4 / 8 * 8;
                    int row_5 = 0;
                    int col_2 = 0;
                    if (compute - shared_gate < row_blocks_1 * full_cols_1) {
                        row_5 = (compute - shared_gate) % (row_blocks_1 * 8) / 8;
                        col_2 = supergroup_1 * 8 + (compute - shared_gate) % 8;
                    } else {
                        row_5 = (compute - shared_gate - row_blocks_1 * full_cols_1) / (col_blocks_4 - full_cols_1);
                        col_2 = full_cols_1 + (compute - shared_gate - row_blocks_1 * full_cols_1) % (col_blocks_4 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_5 = row_blocks_1 - row_5 - 1;
                    }
                    x_1 = row_5;
                    y_1 = col_2;
                    expert_1 = 0;
                }
                unsigned int phase_bits_4 = gemm_bits;
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
                                int _min_19 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                int _max_6 = ((0) > (_min_19) ? (0) : (_min_19));
                                int mini_rows_4 = _max_6;
                                int required_4 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                            }
                            int ring_2 = 0;
                            #pragma unroll 1
                            for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                mbarrier_wait(gemm_finished_addr + (ring_2) * 8, phase_bits_4 >> (unsigned int)(16 + ring_2) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&x_shared)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)(ring_2 * 16384)), "l"((&wu_shared)), "r"(0), "r"(y_1 * 256 + cta_rank_0 * 128), "r"(idx_2), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_2);
                                ring_2 = (ring_2 + 1) % 6;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_3 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_4 >> 22 & 1);
                                phase_bits_4 = phase_bits_4 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_3 = 0; idx_3 < iterations_1; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 65536);
                                    mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_4 >> (unsigned int)ring_3 & 1);
                                    int _mma_a_lo_1 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                    int _mma_b_lo_1 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)));
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_3) * 8, (uint16_t)(3));
                                    phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << ring_3);
                                    ring_3 = (ring_3 + 1) % 6;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        mbarrier_wait(output_arrived_addr, phase_bits_4 >> 6 & 1);
                        phase_bits_4 = phase_bits_4 ^ 64;
                        unsigned int packed_1[128];
                        #pragma unroll
                        for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                            #pragma unroll
                            for (int half_4 = 0; half_4 < 2; half_4++) {
                                unsigned int address_2 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_4 * 16 << 16) + (unsigned int)(chunk_2 * 32);
                                float _tmem_load_1[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                    : "r"(address_2));
                                #pragma unroll
                                for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                    __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_1[pair_1 * 2], _tmem_load_1[pair_1 * 2 + 1]));
                                    packed_1[chunk_2 * 16 + half_4 * 8 + pair_1] = __as_u32(_bf16x2_3);
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            int previous_offset_3 = macro_size;
                            int output_row_1 = x_1 * 256 + cta_rank_0 * 128;
                            int _min_20 = ((macro_size) < (tokens - previous_offset_3) ? (macro_size) : (tokens - previous_offset_3));
                            if (output_row_1 < _min_20) {
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
                            for (int half_5 = 0; half_5 < 2; half_5++) {
                                #pragma unroll
                                for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                    int row_6 = warp_0_1 * 32 + half_5 * 16 + lane_2 % 16;
                                    int col_3 = col_tile_1 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_3 = d_smem_addr + (unsigned int)(chunk_3 % 3 * 8192) + (unsigned int)((row_6 * 32 + col_3) * 2);
                                    address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                    int offset_1 = chunk_3 * 16 + half_5 * 8 + col_tile_1 * 4;
                                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(address_3);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_shared_out)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + chunk_3), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_3 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
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
                                    bool enabled_value_3 = 1;
                                    if (enabled_value_3 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + ((macro_rows_2 + x_1) * (intermediate / 256) + y_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
                gemm_bits = phase_bits_4;
            } else {
                if (compute < 2 * shared_gate + shared_swiglu) {
                    unsigned int phase_bits_5 = swiglu_bits;
                    int col_blocks_5 = intermediate / 128;
                    int num_tiles = (shared_rows * 256 + 127) / 128 * col_blocks_5;
                    int macro_row_offset = 0;
                    int first_tile_1 = (compute - 2 * shared_gate) * 6 + cta_rank_0 * 3;
                    int tile_end = num_tiles;
                    if (first_tile_1 < tile_end) {
                        int first_row_1 = first_tile_1 / col_blocks_5;
                        int first_col_1 = first_tile_1 % col_blocks_5;
                        if (tid == 0) {
                            #pragma unroll
                            for (int stage_4 = 0; stage_4 < 3; stage_4++) {
                                if (tile_end > first_tile_1 + stage_4) {
                                    int row_7 = first_row_1;
                                    int col_4 = first_col_1 + stage_4;
                                    if (col_4 >= col_blocks_5) {
                                        row_7 = row_7 + 1;
                                        col_4 = col_4 - col_blocks_5;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_4) * 8, 65536);
                                    int parent = row_7 / 2 * (intermediate / 256) + col_4 / 2;
                                    int32_t _relaxed_ld_6;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(gate_ready + parent) : "memory");
                                    int value_3 = _relaxed_ld_6;
                                    while (value_3 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_7;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(gate_ready + parent) : "memory");
                                        value_3 = _relaxed_ld_7;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(gate_smem_addr + (unsigned int)(stage_4 * 32768)), "l"((&gate_shared_in)), "r"(0), "r"((row_7 - macro_row_offset) * 128), "r"(col_4 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(up_smem_addr + (unsigned int)(stage_4 * 32768)), "l"((&up_shared_in)), "r"(0), "r"((row_7 - macro_row_offset) * 128), "r"(col_4 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                }
                            }
                        }
                        #pragma unroll 3
                        for (int stage_5 = 0; stage_5 < 3; stage_5++) {
                            if (tile_end > first_tile_1 + stage_5) {
                                mbarrier_wait(swiglu_arrived_addr + (stage_5) * 8, phase_bits_5 >> (unsigned int)stage_5 & 1);
                                phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << stage_5);
                                int row_8 = first_row_1;
                                int col_5 = first_col_1 + stage_5;
                                if (col_5 >= col_blocks_5) {
                                    row_8 = row_8 + 1;
                                    col_5 = col_5 - col_blocks_5;
                                }
                                float gate[64];
                                float up[64];
                                float denominator[64];
                                int warp_0_2 = tid / 32;
                                int local_warp = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                                int lane_3 = tid % 32;
                                #pragma unroll
                                for (int tile_col = 0; tile_col < 8; tile_col++) {
                                    unsigned int packed_2[4];
                                    unsigned int address_4 = gate_smem_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col * 16 + lane_3 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_3 % 16) * 64 + (tile_col * 16 + lane_3 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                        : "r"(address_4 ^ (address_4 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                        float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(packed_2[pair_2]));
                                        gate[tile_col * 8 + pair_2 * 2] = _cvt_f32_0.x;
                                        gate[tile_col * 8 + pair_2 * 2 + 1] = _cvt_f32_0.y;
                                    }
                                }
                                int warp_1_1 = tid / 32;
                                int local_warp_2 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                int lane_3_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                    unsigned int packed_3[4];
                                    unsigned int address_5 = up_smem_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3_1 % 16) * 64 + (tile_col_1 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                        : "r"(address_5 ^ (address_5 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                        float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(packed_3[pair_3]));
                                        up[tile_col_1 * 8 + pair_3 * 2] = _cvt_f32_1.x;
                                        up[tile_col_1 * 8 + pair_3 * 2 + 1] = _cvt_f32_1.y;
                                    }
                                }
                                if (swiglu_clamped != 0) {
                                    #pragma unroll
                                    for (int elem = 0; elem < 64; elem++) {
                                        float _min_22 = fminf(gate[elem], swiglu_limit);
                                        gate[elem] = _min_22;
                                    }
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                        float _max_7 = max_noftz(up[elem_1], -swiglu_limit);
                                        up[elem_1] = _max_7;
                                    }
                                    #pragma unroll
                                    for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                        float _min_23 = fminf(up[elem_2], swiglu_limit);
                                        up[elem_2] = _min_23;
                                    }
                                }
                                #pragma unroll
                                for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                    denominator[elem_3] = gate[elem_3] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_4 = 0; elem_4 < 64; elem_4++) {
                                    float _exp_0 = expf(denominator[elem_4]);
                                    denominator[elem_4] = _exp_0;
                                }
                                #pragma unroll
                                for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                    denominator[elem_5] = denominator[elem_5] + 1.0f;
                                }
                                #pragma unroll
                                for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                    gate[elem_6] = gate[elem_6] / denominator[elem_6];
                                }
                                #pragma unroll
                                for (int elem_7 = 0; elem_7 < 64; elem_7++) {
                                    gate[elem_7] = gate[elem_7] * up[elem_7];
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
                                    for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_4 * 2], gate[tile_col_2 * 8 + pair_4 * 2 + 1]));
                                        packed_4[pair_4] = __as_u32(_bf16x2_4);
                                    }
                                    unsigned int address_6 = hidden_smem_addr + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(address_6 ^ (address_6 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&hidden_shared_out), 0, (row_8 - macro_row_offset) * 128, col_5 * 2, 0, 0, hidden_smem_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_6 = 0; stage_6 < 3; stage_6++) {
                                if (tile_end > first_tile_1 + stage_6) {
                                    int row_9 = first_row_1;
                                    if (col_blocks_5 <= first_col_1 + stage_6) {
                                        row_9 = row_9 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (row_9 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                    }
                    swiglu_bits = phase_bits_5;
                } else if (compute < shared_tasks) {
                    int col_blocks_6 = hidden / 256;
                    int x_2 = -1;
                    int y_2 = -1;
                    int expert_2 = -1;
                    int k_start_2 = 0;
                    int k_end_2 = 0;
                    int first_2 = 0;
                    int row_blocks_2 = (local_tokens + 255) / 256;
                    if (compute - 2 * shared_gate - shared_swiglu < row_blocks_2 * col_blocks_6) {
                        int supergroup_2 = (compute - 2 * shared_gate - shared_swiglu) / (row_blocks_2 * 8);
                        int full_cols_2 = col_blocks_6 / 8 * 8;
                        int row_10 = 0;
                        int col_6 = 0;
                        if (compute - 2 * shared_gate - shared_swiglu < row_blocks_2 * full_cols_2) {
                            row_10 = (compute - 2 * shared_gate - shared_swiglu) % (row_blocks_2 * 8) / 8;
                            col_6 = supergroup_2 * 8 + (compute - 2 * shared_gate - shared_swiglu) % 8;
                        } else {
                            row_10 = (compute - 2 * shared_gate - shared_swiglu - row_blocks_2 * full_cols_2) / (col_blocks_6 - full_cols_2);
                            col_6 = full_cols_2 + (compute - 2 * shared_gate - shared_swiglu - row_blocks_2 * full_cols_2) % (col_blocks_6 - full_cols_2);
                        }
                        if ((supergroup_2 & 1) != 0) {
                            row_10 = row_blocks_2 - row_10 - 1;
                        }
                        x_2 = row_10;
                        y_2 = col_6;
                        expert_2 = 0;
                    }
                    unsigned int phase_bits_6 = gemm_bits;
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
                                    bool enabled_value_4 = 1;
                                    if (enabled_value_4 != 0) {
                                        int32_t _relaxed_ld_8;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(hidden_ready + (macro_rows_3 + x_2)) : "memory");
                                        int value_4 = _relaxed_ld_8;
                                        while (value_4 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_9;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(hidden_ready + (macro_rows_3 + x_2)) : "memory");
                                            value_4 = _relaxed_ld_9;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_24 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                    int _max_8 = ((0) > (_min_24) ? (0) : (_min_24));
                                    int mini_rows_5 = _max_8;
                                    int required_5 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
                                }
                                int ring_4 = 0;
                                #pragma unroll 1
                                for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_4) * 8, phase_bits_6 >> (unsigned int)(16 + ring_4) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)(ring_4 * 16384)), "l"((&hidden_shared_in)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)(ring_4 * 16384)), "l"((&wd_shared)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << 16 + ring_4);
                                    ring_4 = (ring_4 + 1) % 6;
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
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_5) * 8, 65536);
                                        mbarrier_wait(gemm_arrived_addr + (ring_5) * 8, phase_bits_6 >> (unsigned int)ring_5 & 1);
                                        int _mma_a_lo_2 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_5) * 1024;
                                        int _mma_b_lo_2 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_5) * 1024;
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
            :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_accumulator), "r"(((idx_5 == 0) ? 0 : 1)));
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_5) * 8, (uint16_t)(3));
                                        phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << ring_5);
                                        ring_5 = (ring_5 + 1) % 6;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_6 >> 6 & 1);
                            phase_bits_6 = phase_bits_6 ^ 64;
                            unsigned int packed_5[128];
                            #pragma unroll
                            for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                #pragma unroll
                                for (int half_6 = 0; half_6 < 2; half_6++) {
                                    unsigned int address_7 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_6 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                    float _tmem_load_2[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                        : "r"(address_7));
                                    #pragma unroll
                                    for (int pair_5 = 0; pair_5 < 8; pair_5++) {
                                        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_2[pair_5 * 2], _tmem_load_2[pair_5 * 2 + 1]));
                                        packed_5[chunk_4 * 16 + half_6 * 8 + pair_5] = __as_u32(_bf16x2_5);
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                int previous_offset_4 = macro_size;
                                int output_row_2 = x_2 * 256 + cta_rank_0 * 128;
                                int _min_25 = ((macro_size) < (tokens - previous_offset_4) ? (macro_size) : (tokens - previous_offset_4));
                                if (output_row_2 < _min_25) {
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
                                for (int half_7 = 0; half_7 < 2; half_7++) {
                                    #pragma unroll
                                    for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                        int row_11 = warp_0_3 * 32 + half_7 * 16 + lane_4 % 16;
                                        int col_7 = col_tile_2 * 16 + lane_4 / 16 * 8;
                                        unsigned int address_8 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_11 * 32 + col_7) * 2);
                                        address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                        int offset_2 = chunk_5 * 16 + half_7 * 8 + col_tile_2 * 4;
                                        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_8);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&y_shared)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + chunk_5), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (tid / 32 == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                    }
                                }
                            }
                        }
                    }
                    gemm_bits = phase_bits_6;
                } else {
                    int ordered_mini = (compute - shared_tasks) / mini_tasks;
                    int task_2 = (compute - shared_tasks) % mini_tasks;
                    int macro_1 = macros - 1;
                    int mini_1 = ordered_mini;
                    if (ordered_mini >= last_minis) {
                        macro_1 = macros - 2 - (ordered_mini - last_minis) / minis_per_macro;
                        mini_1 = (ordered_mini - last_minis) % minis_per_macro;
                    }
                    if (task_2 < mini_gate) {
                        int col_blocks_7 = intermediate / 256;
                        int x_3 = -1;
                        int y_3 = -1;
                        int expert_3 = -1;
                        int k_start_3 = 0;
                        int k_end_3 = 0;
                        int first_3 = 0;
                        int first_block = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                        int offset_3 = 0;
                        int remaining = task_2;
                        #pragma unroll 1
                        for (int index = 0; index < experts; index++) {
                            int blocks = counts[index] / 256;
                            int _max_9 = ((first_block) > (offset_3) ? (first_block) : (offset_3));
                            int first_row_2 = _max_9;
                            int _min_26 = ((first_block + mini_size / 256) < (offset_3 + blocks) ? (first_block + mini_size / 256) : (offset_3 + blocks));
                            int _max_10 = ((0) > (_min_26 - first_row_2) ? (0) : (_min_26 - first_row_2));
                            int rows_1 = _max_10;
                            int tasks = rows_1 * col_blocks_7;
                            if (remaining < tasks) {
                                int supergroup_3 = remaining / (rows_1 * 8);
                                int full_cols_3 = col_blocks_7 / 8 * 8;
                                int row_12 = 0;
                                int col_8 = 0;
                                if (remaining < rows_1 * full_cols_3) {
                                    row_12 = remaining % (rows_1 * 8) / 8;
                                    col_8 = supergroup_3 * 8 + remaining % 8;
                                } else {
                                    row_12 = (remaining - rows_1 * full_cols_3) / (col_blocks_7 - full_cols_3);
                                    col_8 = full_cols_3 + (remaining - rows_1 * full_cols_3) % (col_blocks_7 - full_cols_3);
                                }
                                if ((supergroup_3 & 1) != 0) {
                                    row_12 = rows_1 - row_12 - 1;
                                }
                                x_3 = first_row_2 + row_12 - macro_1 * (macro_size / 256);
                                y_3 = col_8;
                                expert_3 = index;
                                break;
                            }
                            remaining = remaining - tasks;
                            offset_3 = offset_3 + blocks;
                        }
                        unsigned int phase_bits_7 = gemm_bits;
                        int global_mini_3 = macro_1 * (macro_size / mini_size) + mini_1;
                        int macro_rows_4 = macro_1 * (macro_size / 256);
                        int iterations_3 = hidden / 128;
                        int k_blocks = hidden / 128;
                        int n_blocks = intermediate / 128;
                        if (expert_3 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_27 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                        int _max_11 = ((0) > (_min_27) ? (0) : (_min_27));
                                        int mini_rows_6 = _max_11;
                                        int required_6 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_5 = 1;
                                        if (enabled_value_5 != 0) {
                                            int32_t _relaxed_ld_10;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(x_ready + global_mini_3) : "memory");
                                            int value_5 = _relaxed_ld_10;
                                            while (value_5 < required_6) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_11;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(x_ready + global_mini_3) : "memory");
                                                value_5 = _relaxed_ld_11;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_6 = 0;
                                    #pragma unroll 1
                                    for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_6) * 8, phase_bits_7 >> (unsigned int)(16 + ring_6) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_fp8_smem_addr + (unsigned int)(ring_6 * 16384)), "l"((&x_routed)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_fp8_smem_addr + (unsigned int)(ring_6 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y_3 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_3), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 16 + ring_6);
                                        ring_6 = (ring_6 + 1) % 6;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_28 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                            int _max_12 = ((0) > (_min_28) ? (0) : (_min_28));
                                            int mini_rows_7 = _max_12;
                                            int required_7 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_6 = 1;
                                            if (enabled_value_6 != 0) {
                                                int32_t _relaxed_ld_12;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(x_ready + global_mini_3) : "memory");
                                                int value_6 = _relaxed_ld_12;
                                                while (value_6 < required_7) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_13;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(x_ready + global_mini_3) : "memory");
                                                    value_6 = _relaxed_ld_13;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        int ring_7 = 0;
                                        #pragma unroll 1
                                        for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                            mbarrier_wait(scales_finished_addr + (ring_7) * 8, phase_bits_7 >> (unsigned int)(23 + ring_7) & 1);
                                            int a_tile = (x_3 * 2 + cta_rank_0) * k_blocks + idx_7;
                                            int b_tile = (expert_3 * n_blocks + y_3 * 2 + cta_rank_0) * k_blocks + idx_7;
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(a_sc_smem_addr + (unsigned int)(ring_7 * 512)), "l"((&x_routed_sc)), "r"(0), "r"(a_tile * 32), "r"(0),
                                                   "r"(((scales_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(b_sc_smem_addr + (unsigned int)(ring_7 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_routed_sc)), "r"(0), "r"(b_tile * 32), "r"(0),
                                                   "r"(((scales_arrived_addr + (ring_7) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 23 + ring_7);
                                            ring_7 = (ring_7 + 1) % 6;
                                        }
                                    }
                                }
                            } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_8 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_7 >> 22 & 1);
                                        phase_bits_7 = phase_bits_7 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_3; idx_8++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_8) * 8, 3072);
                                            mbarrier_wait(scales_arrived_addr + (ring_8) * 8, phase_bits_7 >> (unsigned int)(7 + ring_8) & 1);
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_8 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_8 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_8 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_8 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_8 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_8 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_8) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_8) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_8) * 8, phase_bits_7 >> (unsigned int)ring_8 & 1);
                                            int _mma_a_lo_3 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_8) * 1024;
                                            int _mma_b_lo_3 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_8) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_8 * 4, tmem_sf_b + ring_8 * 8, ((idx_8 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                    (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_8 * 4, tmem_sf_b + ring_8 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_8 * 4, tmem_sf_b + ring_8 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                    (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_8 * 4, tmem_sf_b + ring_8 * 8, 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_8) * 8, (uint16_t)(3));
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << ring_8);
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 7 + ring_8);
                                            ring_8 = (ring_8 + 1) % 6;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_7 >> 6 & 1);
                                    phase_bits_7 = phase_bits_7 ^ 64;
                                    int tile_row = tid;
                                    float inv_e4m3_max_2 = 0.002232142857f;
                                    float scale_floor_2 = 1e-12f;
                                    unsigned int scale_word = 0;
                                    #pragma unroll 1
                                    for (int i_8 = 0; i_8 < 8; i_8++) {
                                        float _tmem_load_3[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                                            : "r"(taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + (unsigned int)(i_8 * 32)));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        unsigned int words_2[16];
                                        #pragma unroll
                                        for (int j_6 = 0; j_6 < 16; j_6++) {
                                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_tmem_load_3[2 * j_6], _tmem_load_3[2 * j_6 + 1]));
                                            words_2[j_6] = __as_u32(_bf16x2_6);
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int j_7 = 0; j_7 < 4; j_7++) {
                                            unsigned int address_9 = d_smem_addr + (unsigned int)(i_8 % 2 * 8192) + (unsigned int)((tile_row * 32 + j_7 * 8) * 2);
                                            address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"(address_9), "r"(*reinterpret_cast<uint32_t*>(&(words_2 + 4 * j_7)[0])), "r"(*reinterpret_cast<uint32_t*>(&(words_2 + 4 * j_7)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(words_2 + 4 * j_7)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(words_2 + 4 * j_7)[(0) + 3])));
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&gate_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + i_8), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(i_8 % 2 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                            asm volatile("cp.async.bulk.wait_group.read 1;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        unsigned int packed_6[8];
                                        uint32_t _bf16x2_abs_8;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(words_2[0]));
                                        unsigned int amax2_4 = _bf16x2_abs_8;
                                        #pragma unroll
                                        for (int k_8 = 1; k_8 < 16; k_8++) {
                                            uint32_t _bf16x2_abs_9;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(words_2[k_8]));
                                            uint32_t _bf16x2_max_4;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(amax2_4), "r"(_bf16x2_abs_9));
                                            amax2_4 = _bf16x2_max_4;
                                        }
                                        uint16_t _bf16_max_4;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_4) : "h"((uint16_t)(amax2_4 & 65535)), "h"((uint16_t)(amax2_4 >> 16)));
                                        float _cvt_f32_bf16_20;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(_bf16_max_4)));
                                        float amax_4 = _cvt_f32_bf16_20;
                                        float _max_13 = max_noftz(amax_4 * inv_e4m3_max_2, scale_floor_2);
                                        float scale_4 = _max_13;
                                        uint16_t _ue8m0x2_f32_4;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_4) : "f"(scale_4), "f"(scale_4));
                                        uint16_t codes_4 = _ue8m0x2_f32_4;
                                        unsigned int scale_byte_4 = (unsigned int)codes_4 & 255;
                                        unsigned int inv_bits_4 = 254 - scale_byte_4 << 23;
                                        float inv_4 = 0.0f;
                                        inv_4 = __uint_as_float(inv_bits_4);
                                        #pragma unroll
                                        for (int i_9 = 0; i_9 < 8; i_9++) {
                                            unsigned int w0_4 = words_2[2 * i_9];
                                            unsigned int w1_4 = words_2[2 * i_9 + 1];
                                            float _cvt_f32_bf16_21;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_21) : "h"((uint16_t)(w0_4 & 65535)));
                                            float v0_6 = _cvt_f32_bf16_21;
                                            float _cvt_f32_bf16_22;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(w0_4 >> 16)));
                                            float v1_6 = _cvt_f32_bf16_22;
                                            float _cvt_f32_bf16_23;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(w1_4 & 65535)));
                                            float v2_4 = _cvt_f32_bf16_23;
                                            float _cvt_f32_bf16_24;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(w1_4 >> 16)));
                                            float v3_4 = _cvt_f32_bf16_24;
                                            uint16_t _e4m3x2_f32_8;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(v1_6 * inv_4), "f"(v0_6 * inv_4));
                                            uint16_t lo_4 = _e4m3x2_f32_8;
                                            uint16_t _e4m3x2_f32_9;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_9) : "f"(v3_4 * inv_4), "f"(v2_4 * inv_4));
                                            uint16_t hi_4 = _e4m3x2_f32_9;
                                            packed_6[i_9] = (unsigned int)lo_4 | (unsigned int)hi_4 << 16;
                                        }
                                        unsigned int scale_byte_0 = scale_byte_4;
                                        #pragma unroll
                                        for (int m = 0; m < 2; m++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"(d_fp8_smem_addr + (unsigned int)(tile_row * 32) + (unsigned int)(m * 16)), "r"(*reinterpret_cast<uint32_t*>(&(packed_6 + 4 * m)[0])), "r"(*reinterpret_cast<uint32_t*>(&(packed_6 + 4 * m)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(packed_6 + 4 * m)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(packed_6 + 4 * m)[(0) + 3])));
                                        }
                                        scale_word = scale_word | scale_byte_0 << (unsigned int)(i_8 % 4 * 8);
                                        if (i_8 % 4 == 3) {
                                            d_sc_smem[i_8 / 4 * 128 + tile_row % 32 * 4 + tile_row / 32] = scale_word;
                                            scale_word = 0;
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2}], [%3], %4;"
                                                :: "l"((&gate_routed_fp8)), "r"(y_3 * 256 + i_8 * 32), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(d_fp8_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                            if (i_8 % 4 == 3) {
                                                tma_store_3d((&gate_routed_sc), 0, ((x_3 * 2 + cta_rank_0) * n_blocks + y_3 * 2 + i_8 / 4) * 32, 0, d_sc_smem_addr + (unsigned int)(i_8 / 4 * 512));
                                            }
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_7 = 1;
                                                if (enabled_value_7 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_4 + x_3) * (intermediate / 256) + y_3))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_bits = phase_bits_7;
                    } else if (task_2 < 2 * mini_gate) {
                        int col_blocks_8 = intermediate / 256;
                        int x_4 = -1;
                        int y_4 = -1;
                        int expert_4 = -1;
                        int k_start_4 = 0;
                        int k_end_4 = 0;
                        int first_4 = 0;
                        int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                        int offset_4 = 0;
                        int remaining_1 = task_2 - mini_gate;
                        #pragma unroll 1
                        for (int index_1 = 0; index_1 < experts; index_1++) {
                            int blocks_1 = counts[index_1] / 256;
                            int _max_14 = ((first_block_1) > (offset_4) ? (first_block_1) : (offset_4));
                            int first_row_3 = _max_14;
                            int _min_29 = ((first_block_1 + mini_size / 256) < (offset_4 + blocks_1) ? (first_block_1 + mini_size / 256) : (offset_4 + blocks_1));
                            int _max_15 = ((0) > (_min_29 - first_row_3) ? (0) : (_min_29 - first_row_3));
                            int rows_2 = _max_15;
                            int tasks_1 = rows_2 * col_blocks_8;
                            if (remaining_1 < tasks_1) {
                                int supergroup_4 = remaining_1 / (rows_2 * 8);
                                int full_cols_4 = col_blocks_8 / 8 * 8;
                                int row_13 = 0;
                                int col_9 = 0;
                                if (remaining_1 < rows_2 * full_cols_4) {
                                    row_13 = remaining_1 % (rows_2 * 8) / 8;
                                    col_9 = supergroup_4 * 8 + remaining_1 % 8;
                                } else {
                                    row_13 = (remaining_1 - rows_2 * full_cols_4) / (col_blocks_8 - full_cols_4);
                                    col_9 = full_cols_4 + (remaining_1 - rows_2 * full_cols_4) % (col_blocks_8 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_13 = rows_2 - row_13 - 1;
                                }
                                x_4 = first_row_3 + row_13 - macro_1 * (macro_size / 256);
                                y_4 = col_9;
                                expert_4 = index_1;
                                break;
                            }
                            remaining_1 = remaining_1 - tasks_1;
                            offset_4 = offset_4 + blocks_1;
                        }
                        unsigned int phase_bits_8 = gemm_bits;
                        int global_mini_4 = macro_1 * (macro_size / mini_size) + mini_1;
                        int macro_rows_5 = macro_1 * (macro_size / 256);
                        int iterations_4 = hidden / 128;
                        int k_blocks_1 = hidden / 128;
                        int n_blocks_1 = intermediate / 128;
                        if (expert_4 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_30 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                        int _max_16 = ((0) > (_min_30) ? (0) : (_min_30));
                                        int mini_rows_8 = _max_16;
                                        int required_8 = (mini_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_8 = 1;
                                        if (enabled_value_8 != 0) {
                                            int32_t _relaxed_ld_14;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(x_ready + global_mini_4) : "memory");
                                            int value_7 = _relaxed_ld_14;
                                            while (value_7 < required_8) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_15;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(x_ready + global_mini_4) : "memory");
                                                value_7 = _relaxed_ld_15;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_9 = 0;
                                    #pragma unroll 1
                                    for (int idx_9 = 0; idx_9 < iterations_4; idx_9++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_9) * 8, phase_bits_8 >> (unsigned int)(16 + ring_9) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_fp8_smem_addr + (unsigned int)(ring_9 * 16384)), "l"((&x_routed)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(idx_9), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_9) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_fp8_smem_addr + (unsigned int)(ring_9 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y_4 * 256 + cta_rank_0 * 128), "r"(idx_9), "r"(expert_4), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_9) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << 16 + ring_9);
                                        ring_9 = (ring_9 + 1) % 6;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 6) {
                                if (warp == 6) {
                                    if (elect_sync()) {
                                        {
                                            int _min_31 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                            int _max_17 = ((0) > (_min_31) ? (0) : (_min_31));
                                            int mini_rows_9 = _max_17;
                                            int required_9 = (mini_rows_9 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_9 = 1;
                                            if (enabled_value_9 != 0) {
                                                int32_t _relaxed_ld_16;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_16) : "l"(x_ready + global_mini_4) : "memory");
                                                int value_8 = _relaxed_ld_16;
                                                while (value_8 < required_9) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_17;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_17) : "l"(x_ready + global_mini_4) : "memory");
                                                    value_8 = _relaxed_ld_17;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        int ring_10 = 0;
                                        #pragma unroll 1
                                        for (int idx_10 = 0; idx_10 < iterations_4; idx_10++) {
                                            mbarrier_wait(scales_finished_addr + (ring_10) * 8, phase_bits_8 >> (unsigned int)(23 + ring_10) & 1);
                                            int a_tile_1 = (x_4 * 2 + cta_rank_0) * k_blocks_1 + idx_10;
                                            int b_tile_1 = (expert_4 * n_blocks_1 + y_4 * 2 + cta_rank_0) * k_blocks_1 + idx_10;
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(a_sc_smem_addr + (unsigned int)(ring_10 * 512)), "l"((&x_routed_sc)), "r"(0), "r"(a_tile_1 * 32), "r"(0),
                                                   "r"(((scales_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(b_sc_smem_addr + (unsigned int)(ring_10 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_routed_sc)), "r"(0), "r"(b_tile_1 * 32), "r"(0),
                                                   "r"(((scales_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << 23 + ring_10);
                                            ring_10 = (ring_10 + 1) % 6;
                                        }
                                    }
                                }
                            } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_11 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_8 >> 22 & 1);
                                        phase_bits_8 = phase_bits_8 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_11 = 0; idx_11 < iterations_4; idx_11++) {
                                            mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_11) * 8, 3072);
                                            mbarrier_wait(scales_arrived_addr + (ring_11) * 8, phase_bits_8 >> (unsigned int)(7 + ring_11) & 1);
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_11 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_11 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_11 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_11 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_11 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_11 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_11) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_11) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_11) * 8, phase_bits_8 >> (unsigned int)ring_11 & 1);
                                            int _mma_a_lo_4 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
                                            int _mma_b_lo_4 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_11 * 4, tmem_sf_b + ring_11 * 8, ((idx_11 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                    (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_11 * 4, tmem_sf_b + ring_11 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_11 * 4, tmem_sf_b + ring_11 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                    (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_11 * 4, tmem_sf_b + ring_11 * 8, 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_11) * 8, (uint16_t)(3));
                                            phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << ring_11);
                                            phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << 7 + ring_11);
                                            ring_11 = (ring_11 + 1) % 6;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_8 >> 6 & 1);
                                    phase_bits_8 = phase_bits_8 ^ 64;
                                    int tile_row_1 = tid;
                                    float inv_e4m3_max_3 = 0.002232142857f;
                                    float scale_floor_3 = 1e-12f;
                                    unsigned int scale_word_1 = 0;
                                    #pragma unroll 1
                                    for (int i_10 = 0; i_10 < 8; i_10++) {
                                        float _tmem_load_4[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                                            : "r"(taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + (unsigned int)(i_10 * 32)));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        unsigned int words_3[16];
                                        #pragma unroll
                                        for (int j_8 = 0; j_8 < 16; j_8++) {
                                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_4[2 * j_8], _tmem_load_4[2 * j_8 + 1]));
                                            words_3[j_8] = __as_u32(_bf16x2_7);
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int j_9 = 0; j_9 < 4; j_9++) {
                                            unsigned int address_10 = d_smem_addr + (unsigned int)(i_10 % 2 * 8192) + (unsigned int)((tile_row_1 * 32 + j_9 * 8) * 2);
                                            address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"(address_10), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_9)[0])), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_9)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_9)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_9)[(0) + 3])));
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&up_routed_out)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(y_4 * 8 + i_10), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(i_10 % 2 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                            asm volatile("cp.async.bulk.wait_group.read 1;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        unsigned int packed_7[8];
                                        uint32_t _bf16x2_abs_10;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(words_3[0]));
                                        unsigned int amax2_5 = _bf16x2_abs_10;
                                        #pragma unroll
                                        for (int k_9 = 1; k_9 < 16; k_9++) {
                                            uint32_t _bf16x2_abs_11;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(words_3[k_9]));
                                            uint32_t _bf16x2_max_5;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(amax2_5), "r"(_bf16x2_abs_11));
                                            amax2_5 = _bf16x2_max_5;
                                        }
                                        uint16_t _bf16_max_5;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_5) : "h"((uint16_t)(amax2_5 & 65535)), "h"((uint16_t)(amax2_5 >> 16)));
                                        float _cvt_f32_bf16_25;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(_bf16_max_5)));
                                        float amax_5 = _cvt_f32_bf16_25;
                                        float _max_18 = max_noftz(amax_5 * inv_e4m3_max_3, scale_floor_3);
                                        float scale_5 = _max_18;
                                        uint16_t _ue8m0x2_f32_5;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_5) : "f"(scale_5), "f"(scale_5));
                                        uint16_t codes_5 = _ue8m0x2_f32_5;
                                        unsigned int scale_byte_5 = (unsigned int)codes_5 & 255;
                                        unsigned int inv_bits_5 = 254 - scale_byte_5 << 23;
                                        float inv_5 = 0.0f;
                                        inv_5 = __uint_as_float(inv_bits_5);
                                        #pragma unroll
                                        for (int i_11 = 0; i_11 < 8; i_11++) {
                                            unsigned int w0_5 = words_3[2 * i_11];
                                            unsigned int w1_5 = words_3[2 * i_11 + 1];
                                            float _cvt_f32_bf16_26;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(w0_5 & 65535)));
                                            float v0_7 = _cvt_f32_bf16_26;
                                            float _cvt_f32_bf16_27;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_27) : "h"((uint16_t)(w0_5 >> 16)));
                                            float v1_7 = _cvt_f32_bf16_27;
                                            float _cvt_f32_bf16_28;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(w1_5 & 65535)));
                                            float v2_5 = _cvt_f32_bf16_28;
                                            float _cvt_f32_bf16_29;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(w1_5 >> 16)));
                                            float v3_5 = _cvt_f32_bf16_29;
                                            uint16_t _e4m3x2_f32_10;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_10) : "f"(v1_7 * inv_5), "f"(v0_7 * inv_5));
                                            uint16_t lo_5 = _e4m3x2_f32_10;
                                            uint16_t _e4m3x2_f32_11;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_11) : "f"(v3_5 * inv_5), "f"(v2_5 * inv_5));
                                            uint16_t hi_5 = _e4m3x2_f32_11;
                                            packed_7[i_11] = (unsigned int)lo_5 | (unsigned int)hi_5 << 16;
                                        }
                                        unsigned int scale_byte_0_1 = scale_byte_5;
                                        #pragma unroll
                                        for (int m_1 = 0; m_1 < 2; m_1++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"(d_fp8_smem_addr + (unsigned int)(tile_row_1 * 32) + (unsigned int)(m_1 * 16)), "r"(*reinterpret_cast<uint32_t*>(&(packed_7 + 4 * m_1)[0])), "r"(*reinterpret_cast<uint32_t*>(&(packed_7 + 4 * m_1)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(packed_7 + 4 * m_1)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(packed_7 + 4 * m_1)[(0) + 3])));
                                        }
                                        scale_word_1 = scale_word_1 | scale_byte_0_1 << (unsigned int)(i_10 % 4 * 8);
                                        if (i_10 % 4 == 3) {
                                            d_sc_smem[i_10 / 4 * 128 + tile_row_1 % 32 * 4 + tile_row_1 / 32] = scale_word_1;
                                            scale_word_1 = 0;
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2}], [%3], %4;"
                                                :: "l"((&up_routed_fp8)), "r"(y_4 * 256 + i_10 * 32), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(d_fp8_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                            if (i_10 % 4 == 3) {
                                                tma_store_3d((&up_routed_sc), 0, ((x_4 * 2 + cta_rank_0) * n_blocks_1 + y_4 * 2 + i_10 / 4) * 32, 0, d_sc_smem_addr + (unsigned int)(i_10 / 4 * 512));
                                            }
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                    }
                                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                    if (tid / 32 == 0) {
                                        if (warp == 0) {
                                            if (elect_sync()) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_10 = 1;
                                                if (enabled_value_10 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_5 + x_4) * (intermediate / 256) + y_4))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_bits = phase_bits_8;
                    } else {
                        if (task_2 < 2 * mini_gate + mini_swiglu) {
                            unsigned int phase_bits_9 = swiglu_bits;
                            int col_blocks_9 = intermediate / 128;
                            int num_tiles_1 = (tokens + 127) / 128 * col_blocks_9;
                            int macro_row_offset_1 = macro_1 * (macro_size / 128);
                            int macro_row_blocks_0 = macro_size / 128;
                            int global_mini_5 = macro_1 * (macro_size / mini_size) + mini_1;
                            int mini_tiles = mini_size / 128 * col_blocks_9;
                            int first_tile_2 = (task_2 - 2 * mini_gate) * 6 + cta_rank_0 * 3 + global_mini_5 * mini_tiles;
                            int _min_32 = ((num_tiles_1) < ((global_mini_5 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_5 + 1) * mini_tiles));
                            int tile_end_1 = _min_32;
                            if (first_tile_2 < tile_end_1) {
                                int first_row_4 = first_tile_2 / col_blocks_9;
                                int first_col_2 = first_tile_2 % col_blocks_9;
                                if (tid == 0) {
                                    #pragma unroll
                                    for (int stage_7 = 0; stage_7 < 3; stage_7++) {
                                        if (tile_end_1 > first_tile_2 + stage_7) {
                                            int row_14 = first_row_4;
                                            int col_10 = first_col_2 + stage_7;
                                            if (col_10 >= col_blocks_9) {
                                                row_14 = row_14 + 1;
                                                col_10 = col_10 - col_blocks_9;
                                            }
                                            mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_7) * 8, 65536);
                                            int parent_1 = row_14 / 2 * (intermediate / 256) + col_10 / 2;
                                            int32_t _relaxed_ld_18;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_18) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                            int value_9 = _relaxed_ld_18;
                                            while (value_9 < 4) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_19;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_19) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                                value_9 = _relaxed_ld_19;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                                " [%0], [%1, {%2, %3}], [%4];"
                                                :: "r"(gate_flat_addr + (unsigned int)(stage_7 * 32768)), "l"((&gate_routed_in)), "r"(col_10 * 128), "r"((row_14 - macro_row_offset_1) * 128), "r"(swiglu_arrived_addr + (stage_7) * 8) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                                " [%0], [%1, {%2, %3}], [%4];"
                                                :: "r"(up_flat_addr + (unsigned int)(stage_7 * 32768)), "l"((&up_routed_in)), "r"(col_10 * 128), "r"((row_14 - macro_row_offset_1) * 128), "r"(swiglu_arrived_addr + (stage_7) * 8) : "memory");
                                        }
                                    }
                                }
                                float inv_e4m3_max_4 = 0.002232142857f;
                                float scale_floor_4 = 1e-12f;
                                #pragma unroll 1
                                for (int stage_8 = 0; stage_8 < 3; stage_8++) {
                                    if (tile_end_1 > first_tile_2 + stage_8) {
                                        mbarrier_wait(swiglu_arrived_addr + (stage_8) * 8, phase_bits_9 >> (unsigned int)stage_8 & 1);
                                        phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << stage_8);
                                        int row_15 = first_row_4;
                                        int col_11 = first_col_2 + stage_8;
                                        if (col_11 >= col_blocks_9) {
                                            row_15 = row_15 + 1;
                                            col_11 = col_11 - col_blocks_9;
                                        }
                                        #pragma unroll
                                        for (int step = 0; step < 32; step++) {
                                            int pair_6 = step * 256 + tid;
                                            float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(gate_words[stage_8 * 8192 + pair_6]));
                                            float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(up_words[stage_8 * 8192 + pair_6]));
                                            float gx = _cvt_f32_2.x;
                                            float gy = _cvt_f32_2.y;
                                            float ux = _cvt_f32_3.x;
                                            float uy = _cvt_f32_3.y;
                                            if (swiglu_clamped != 0) {
                                                float _min_33 = fminf(gx, swiglu_limit);
                                                gx = _min_33;
                                                float _min_34 = fminf(gy, swiglu_limit);
                                                gy = _min_34;
                                                float _max_19 = max_noftz(ux, -swiglu_limit);
                                                float _min_35 = fminf(_max_19, swiglu_limit);
                                                ux = _min_35;
                                                float _max_20 = max_noftz(uy, -swiglu_limit);
                                                float _min_36 = fminf(_max_20, swiglu_limit);
                                                uy = _min_36;
                                            }
                                            float _exp_1 = expf(gx * -1.0f);
                                            float hx = gx / (_exp_1 + 1.0f) * ux;
                                            float _exp_2 = expf(gy * -1.0f);
                                            float hy = gy / (_exp_2 + 1.0f) * uy;
                                            __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(hx, hy));
                                            hidden_words[pair_6] = __as_u32(_bf16x2_8);
                                        }
                                        __syncthreads();
                                        if (tid < 128) {
                                            int t_row_2 = tid % 64 * 2 + tid / 64;
                                            int rotation_4 = tid / 8;
                                            unsigned int t_scale_word_2 = 0;
                                            #pragma unroll 1
                                            for (int j_10 = 0; j_10 < 4; j_10++) {
                                                int k_block_2 = (j_10 + rotation_4) % 4;
                                                unsigned int t_words_2[16];
                                                #pragma unroll
                                                for (int k_10 = 0; k_10 < 16; k_10++) {
                                                    int src_row_2 = k_block_2 * 32 + (tid * 4 + k_10 * 2) % 32;
                                                    float v0_8 = (float)hidden_flat[src_row_2 * 128 + t_row_2];
                                                    float v1_8 = (float)hidden_flat[(src_row_2 + 1) * 128 + t_row_2];
                                                    __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(v0_8, v1_8));
                                                    t_words_2[k_10] = __as_u32(_bf16x2_9);
                                                }
                                                unsigned int t_packed_2[8];
                                                uint32_t _bf16x2_abs_12;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(t_words_2[0]));
                                                unsigned int amax2_6 = _bf16x2_abs_12;
                                                #pragma unroll
                                                for (int k_11 = 1; k_11 < 16; k_11++) {
                                                    uint32_t _bf16x2_abs_13;
                                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(t_words_2[k_11]));
                                                    uint32_t _bf16x2_max_6;
                                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(amax2_6), "r"(_bf16x2_abs_13));
                                                    amax2_6 = _bf16x2_max_6;
                                                }
                                                uint16_t _bf16_max_6;
                                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_6) : "h"((uint16_t)(amax2_6 & 65535)), "h"((uint16_t)(amax2_6 >> 16)));
                                                float _cvt_f32_bf16_30;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_30) : "h"((uint16_t)(_bf16_max_6)));
                                                float amax_6 = _cvt_f32_bf16_30;
                                                float _max_21 = max_noftz(amax_6 * inv_e4m3_max_4, scale_floor_4);
                                                float scale_6 = _max_21;
                                                uint16_t _ue8m0x2_f32_6;
                                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_6) : "f"(scale_6), "f"(scale_6));
                                                uint16_t codes_6 = _ue8m0x2_f32_6;
                                                unsigned int scale_byte_6 = (unsigned int)codes_6 & 255;
                                                unsigned int inv_bits_6 = 254 - scale_byte_6 << 23;
                                                float inv_6 = 0.0f;
                                                inv_6 = __uint_as_float(inv_bits_6);
                                                #pragma unroll
                                                for (int i_12 = 0; i_12 < 8; i_12++) {
                                                    unsigned int w0_6 = t_words_2[2 * i_12];
                                                    unsigned int w1_6 = t_words_2[2 * i_12 + 1];
                                                    float _cvt_f32_bf16_31;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_31) : "h"((uint16_t)(w0_6 & 65535)));
                                                    float v0_9 = _cvt_f32_bf16_31;
                                                    float _cvt_f32_bf16_32;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_32) : "h"((uint16_t)(w0_6 >> 16)));
                                                    float v1_9 = _cvt_f32_bf16_32;
                                                    float _cvt_f32_bf16_33;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_33) : "h"((uint16_t)(w1_6 & 65535)));
                                                    float v2_6 = _cvt_f32_bf16_33;
                                                    float _cvt_f32_bf16_34;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_34) : "h"((uint16_t)(w1_6 >> 16)));
                                                    float v3_6 = _cvt_f32_bf16_34;
                                                    uint16_t _e4m3x2_f32_12;
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_12) : "f"(v1_9 * inv_6), "f"(v0_9 * inv_6));
                                                    uint16_t lo_6 = _e4m3x2_f32_12;
                                                    uint16_t _e4m3x2_f32_13;
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_13) : "f"(v3_6 * inv_6), "f"(v2_6 * inv_6));
                                                    uint16_t hi_6 = _e4m3x2_f32_13;
                                                    t_packed_2[i_12] = (unsigned int)lo_6 | (unsigned int)hi_6 << 16;
                                                }
                                                unsigned int t_scale_byte_2 = scale_byte_6;
                                                #pragma unroll
                                                for (int i_13 = 0; i_13 < 8; i_13++) {
                                                    int t_col_2 = k_block_2 * 32 + (tid * 4 + i_13 * 4) % 32;
                                                    up_words[stage_8 * 8192 + t_row_2 * 32 + t_col_2 / 4] = t_packed_2[i_13];
                                                }
                                                t_scale_word_2 = t_scale_word_2 | t_scale_byte_2 << (unsigned int)(k_block_2 * 8);
                                            }
                                            up_words[stage_8 * 8192 + 4096 + t_row_2 % 32 * 4 + t_row_2 / 32] = t_scale_word_2;
                                            int n_row_2 = tid;
                                            int rotation_0 = tid / 8;
                                            unsigned int words_4[64];
                                            #pragma unroll
                                            for (int j_11 = 0; j_11 < 4; j_11++) {
                                                int k_block_j_2 = (j_11 + rotation_0) % 4;
                                                #pragma unroll
                                                for (int k_12 = 0; k_12 < 16; k_12++) {
                                                    int src_col_2 = k_block_j_2 * 32 + (tid * 4 + k_12 * 2) % 32;
                                                    words_4[j_11 * 16 + k_12] = hidden_words[n_row_2 * 64 + src_col_2 / 2];
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            unsigned int n_scale_word_2 = 0;
                                            #pragma unroll
                                            for (int j_12 = 0; j_12 < 4; j_12++) {
                                                int k_block_n_2 = (j_12 + rotation_0) % 4;
                                                unsigned int n_packed_2[8];
                                                uint32_t _bf16x2_abs_14;
                                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(words_4[j_12 * 16]));
                                                unsigned int amax2_7 = _bf16x2_abs_14;
                                                #pragma unroll
                                                for (int k_13 = 1; k_13 < 16; k_13++) {
                                                    uint32_t _bf16x2_abs_15;
                                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(words_4[j_12 * 16 + k_13]));
                                                    uint32_t _bf16x2_max_7;
                                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(amax2_7), "r"(_bf16x2_abs_15));
                                                    amax2_7 = _bf16x2_max_7;
                                                }
                                                uint16_t _bf16_max_7;
                                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_7) : "h"((uint16_t)(amax2_7 & 65535)), "h"((uint16_t)(amax2_7 >> 16)));
                                                float _cvt_f32_bf16_35;
                                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_35) : "h"((uint16_t)(_bf16_max_7)));
                                                float amax_7 = _cvt_f32_bf16_35;
                                                float _max_22 = max_noftz(amax_7 * inv_e4m3_max_4, scale_floor_4);
                                                float scale_7 = _max_22;
                                                uint16_t _ue8m0x2_f32_7;
                                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_7) : "f"(scale_7), "f"(scale_7));
                                                uint16_t codes_7 = _ue8m0x2_f32_7;
                                                unsigned int scale_byte_7 = (unsigned int)codes_7 & 255;
                                                unsigned int inv_bits_7 = 254 - scale_byte_7 << 23;
                                                float inv_7 = 0.0f;
                                                inv_7 = __uint_as_float(inv_bits_7);
                                                #pragma unroll
                                                for (int i_14 = 0; i_14 < 8; i_14++) {
                                                    unsigned int w0_7 = words_4[j_12 * 16 + 2 * i_14];
                                                    unsigned int w1_7 = words_4[j_12 * 16 + 2 * i_14 + 1];
                                                    float _cvt_f32_bf16_36;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_36) : "h"((uint16_t)(w0_7 & 65535)));
                                                    float v0_10 = _cvt_f32_bf16_36;
                                                    float _cvt_f32_bf16_37;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_37) : "h"((uint16_t)(w0_7 >> 16)));
                                                    float v1_10 = _cvt_f32_bf16_37;
                                                    float _cvt_f32_bf16_38;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_38) : "h"((uint16_t)(w1_7 & 65535)));
                                                    float v2_7 = _cvt_f32_bf16_38;
                                                    float _cvt_f32_bf16_39;
                                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_39) : "h"((uint16_t)(w1_7 >> 16)));
                                                    float v3_7 = _cvt_f32_bf16_39;
                                                    uint16_t _e4m3x2_f32_14;
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_14) : "f"(v1_10 * inv_7), "f"(v0_10 * inv_7));
                                                    uint16_t lo_7 = _e4m3x2_f32_14;
                                                    uint16_t _e4m3x2_f32_15;
                                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_15) : "f"(v3_7 * inv_7), "f"(v2_7 * inv_7));
                                                    uint16_t hi_7 = _e4m3x2_f32_15;
                                                    n_packed_2[i_14] = (unsigned int)lo_7 | (unsigned int)hi_7 << 16;
                                                }
                                                unsigned int n_scale_byte_2 = scale_byte_7;
                                                #pragma unroll
                                                for (int i_15 = 0; i_15 < 8; i_15++) {
                                                    int n_col_2 = k_block_n_2 * 32 + (tid * 4 + i_15 * 4) % 32;
                                                    gate_words[stage_8 * 8192 + n_row_2 * 32 + n_col_2 / 4] = n_packed_2[i_15];
                                                }
                                                n_scale_word_2 = n_scale_word_2 | n_scale_byte_2 << (unsigned int)(k_block_n_2 * 8);
                                            }
                                            gate_words[stage_8 * 8192 + 4096 + n_row_2 % 32 * 4 + n_row_2 / 32] = n_scale_word_2;
                                        }
                                        __syncthreads();
                                        if (tid == 0) {
                                            int local_row = row_15 - macro_row_offset_1;
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_2d((&hidden_routed_fp8), col_11 * 128, local_row * 128, gate_flat_addr + (unsigned int)(stage_8 * 32768));
                                            tma_store_3d((&hidden_routed_sc), 0, (local_row * col_blocks_9 + col_11) * 32, 0, gate_flat_addr + (unsigned int)(stage_8 * 32768) + 16384);
                                            tma_store_2d((&hidden_routed_fp8_t), local_row * 128, col_11 * 128, up_flat_addr + (unsigned int)(stage_8 * 32768));
                                            tma_store_3d((&hidden_routed_sc_t), 0, (col_11 * macro_row_blocks_0 + local_row) * 32, 0, up_flat_addr + (unsigned int)(stage_8 * 32768) + 16384);
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group 0;");
                                    #pragma unroll
                                    for (int stage_9 = 0; stage_9 < 3; stage_9++) {
                                        if (tile_end_1 > first_tile_2 + stage_9) {
                                            int row_16 = first_row_4;
                                            if (col_blocks_9 <= first_col_2 + stage_9) {
                                                row_16 = row_16 + 1;
                                            }
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_16 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                }
                            }
                            swiglu_bits = phase_bits_9;
                        } else {
                            int col_blocks_10 = hidden / 256;
                            int x_5 = -1;
                            int y_5 = -1;
                            int expert_5 = -1;
                            int k_start_5 = 0;
                            int k_end_5 = 0;
                            int first_5 = 0;
                            int first_block_2 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                            int offset_5 = 0;
                            int remaining_2 = task_2 - 2 * mini_gate - mini_swiglu;
                            #pragma unroll 1
                            for (int index_2 = 0; index_2 < experts; index_2++) {
                                int blocks_2 = counts[index_2] / 256;
                                int _max_23 = ((first_block_2) > (offset_5) ? (first_block_2) : (offset_5));
                                int first_row_5 = _max_23;
                                int _min_37 = ((first_block_2 + mini_size / 256) < (offset_5 + blocks_2) ? (first_block_2 + mini_size / 256) : (offset_5 + blocks_2));
                                int _max_24 = ((0) > (_min_37 - first_row_5) ? (0) : (_min_37 - first_row_5));
                                int rows_3 = _max_24;
                                int tasks_2 = rows_3 * col_blocks_10;
                                if (remaining_2 < tasks_2) {
                                    int supergroup_5 = remaining_2 / (rows_3 * 8);
                                    int full_cols_5 = col_blocks_10 / 8 * 8;
                                    int row_17 = 0;
                                    int col_12 = 0;
                                    if (remaining_2 < rows_3 * full_cols_5) {
                                        row_17 = remaining_2 % (rows_3 * 8) / 8;
                                        col_12 = supergroup_5 * 8 + remaining_2 % 8;
                                    } else {
                                        row_17 = (remaining_2 - rows_3 * full_cols_5) / (col_blocks_10 - full_cols_5);
                                        col_12 = full_cols_5 + (remaining_2 - rows_3 * full_cols_5) % (col_blocks_10 - full_cols_5);
                                    }
                                    if ((supergroup_5 & 1) != 0) {
                                        row_17 = rows_3 - row_17 - 1;
                                    }
                                    x_5 = first_row_5 + row_17 - macro_1 * (macro_size / 256);
                                    y_5 = col_12;
                                    expert_5 = index_2;
                                    break;
                                }
                                remaining_2 = remaining_2 - tasks_2;
                                offset_5 = offset_5 + blocks_2;
                            }
                            unsigned int phase_bits_10 = gemm_bits;
                            int global_mini_6 = macro_1 * (macro_size / mini_size) + mini_1;
                            int macro_rows_6 = macro_1 * (macro_size / 256);
                            int iterations_5 = intermediate / 128;
                            int k_blocks_2 = intermediate / 128;
                            int n_blocks_2 = hidden / 128;
                            if (expert_5 < 0) {
                                if (tid == 0) {
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_11 = 1;
                                            if (enabled_value_11 != 0) {
                                                int32_t _relaxed_ld_20;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_20) : "l"(hidden_ready + (shared_rows + macro_rows_6 + x_5)) : "memory");
                                                int value_10 = _relaxed_ld_20;
                                                while (value_10 < 2 * (intermediate / 128)) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_21;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_21) : "l"(hidden_ready + (shared_rows + macro_rows_6 + x_5)) : "memory");
                                                    value_10 = _relaxed_ld_21;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_38 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                            int _max_25 = ((0) > (_min_38) ? (0) : (_min_38));
                                            int mini_rows_10 = _max_25;
                                            int required_10 = (mini_rows_10 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        int ring_12 = 0;
                                        #pragma unroll 1
                                        for (int idx_12 = 0; idx_12 < iterations_5; idx_12++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_12) * 8, phase_bits_10 >> (unsigned int)(16 + ring_12) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_fp8_smem_addr + (unsigned int)(ring_12 * 16384)), "l"((&hidden_routed_in)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_12) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_fp8_smem_addr + (unsigned int)(ring_12 * 16384)), "l"((&wd_routed)), "r"(0), "r"(y_5 * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(expert_5), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_12) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 16 + ring_12);
                                            ring_12 = (ring_12 + 1) % 6;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 6) {
                                    if (warp == 6) {
                                        if (elect_sync()) {
                                            {
                                                bool enabled_value_12 = 1;
                                                if (enabled_value_12 != 0) {
                                                    int32_t _relaxed_ld_22;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_22) : "l"(hidden_ready + (shared_rows + macro_rows_6 + x_5)) : "memory");
                                                    int value_11 = _relaxed_ld_22;
                                                    while (value_11 < 2 * (intermediate / 128)) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_23;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_23) : "l"(hidden_ready + (shared_rows + macro_rows_6 + x_5)) : "memory");
                                                        value_11 = _relaxed_ld_23;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                                int _min_39 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                                int _max_26 = ((0) > (_min_39) ? (0) : (_min_39));
                                                int mini_rows_11 = _max_26;
                                                int required_11 = (mini_rows_11 + 127) / 128 * ((intermediate + 511) / 512);
                                            }
                                            int ring_13 = 0;
                                            #pragma unroll 1
                                            for (int idx_13 = 0; idx_13 < iterations_5; idx_13++) {
                                                mbarrier_wait(scales_finished_addr + (ring_13) * 8, phase_bits_10 >> (unsigned int)(23 + ring_13) & 1);
                                                int a_tile_2 = (x_5 * 2 + cta_rank_0) * k_blocks_2 + idx_13;
                                                int b_tile_2 = (expert_5 * n_blocks_2 + y_5 * 2 + cta_rank_0) * k_blocks_2 + idx_13;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(a_sc_smem_addr + (unsigned int)(ring_13 * 512)), "l"((&hidden_routed_in_sc)), "r"(0), "r"(a_tile_2 * 32), "r"(0),
                                                       "r"(((scales_arrived_addr + (ring_13) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(b_sc_smem_addr + (unsigned int)(ring_13 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_routed_sc)), "r"(0), "r"(b_tile_2 * 32), "r"(0),
                                                       "r"(((scales_arrived_addr + (ring_13) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 23 + ring_13);
                                                ring_13 = (ring_13 + 1) % 6;
                                            }
                                        }
                                    }
                                } else if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_14 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_10 >> 22 & 1);
                                            phase_bits_10 = phase_bits_10 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_14 = 0; idx_14 < iterations_5; idx_14++) {
                                                mbarrier_arrive_expect_tx(scales_arrived_addr + (ring_14) * 8, 3072);
                                                mbarrier_wait(scales_arrived_addr + (ring_14) * 8, phase_bits_10 >> (unsigned int)(7 + ring_14) & 1);
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_14 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_14 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_14 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_14 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_14 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_14 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_14) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_14) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_14) * 8, phase_bits_10 >> (unsigned int)ring_14 & 1);
                                                int _mma_a_lo_5 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_14) * 1024;
                                                int _mma_b_lo_5 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_14) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_14 * 4, tmem_sf_b + ring_14 * 8, ((idx_14 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                        (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_14 * 4, tmem_sf_b + ring_14 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_14 * 4, tmem_sf_b + ring_14 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                        (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_14 * 4, tmem_sf_b + ring_14 * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_14) * 8, (uint16_t)(3));
                                                phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << ring_14);
                                                phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 7 + ring_14);
                                                ring_14 = (ring_14 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_10 >> 6 & 1);
                                        phase_bits_10 = phase_bits_10 ^ 64;
                                        unsigned int packed_8[128];
                                        #pragma unroll
                                        for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                            #pragma unroll
                                            for (int half_8 = 0; half_8 < 2; half_8++) {
                                                unsigned int address_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_8 * 16 << 16) + (unsigned int)(chunk_6 * 32);
                                                float _tmem_load_5[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                                                    : "r"(address_11));
                                                #pragma unroll
                                                for (int pair_7 = 0; pair_7 < 8; pair_7++) {
                                                    __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_5[pair_7 * 2], _tmem_load_5[pair_7 * 2 + 1]));
                                                    packed_8[chunk_6 * 16 + half_8 * 8 + pair_7] = __as_u32(_bf16x2_10);
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            int previous_offset_5 = (macro_1 + 1) * macro_size;
                                            int output_row_3 = x_5 * 256 + cta_rank_0 * 128;
                                            int _min_40 = ((macro_size) < (tokens - previous_offset_5) ? (macro_size) : (tokens - previous_offset_5));
                                            if (output_row_3 < _min_40) {
                                                bool enabled_value_13 = 1;
                                                if (enabled_value_13 != 0) {
                                                    int32_t _relaxed_ld_24;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_24) : "l"(y_done + ((previous_offset_5 + output_row_3) / 128)) : "memory");
                                                    int value_12 = _relaxed_ld_24;
                                                    while (value_12 < 8 * ((hidden + 1023) / 1024)) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_25;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_25) : "l"(y_done + ((previous_offset_5 + output_row_3) / 128)) : "memory");
                                                        value_12 = _relaxed_ld_25;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
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
                                            for (int half_9 = 0; half_9 < 2; half_9++) {
                                                #pragma unroll
                                                for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                                    int row_18 = warp_0_4 * 32 + half_9 * 16 + lane_5 % 16;
                                                    int col_13 = col_tile_3 * 16 + lane_5 / 16 * 8;
                                                    unsigned int address_12 = d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192) + (unsigned int)((row_18 * 32 + col_13) * 2);
                                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                                    int offset_0 = chunk_7 * 16 + half_9 * 8 + col_tile_3 * 4;
                                                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(address_12);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_0 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_0 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_0 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&y_routed)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + chunk_7), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
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
                                                    bool enabled_value_14 = 1;
                                                    if (enabled_value_14 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_6))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_bits = phase_bits_10;
                        }
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
                    if (cluster - comm_clusters >= 2 * shared_gate && cluster - comm_clusters < 2 * shared_gate + shared_swiglu) {
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
