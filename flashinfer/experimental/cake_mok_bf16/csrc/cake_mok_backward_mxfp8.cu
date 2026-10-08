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
#define SMEM_A_NT_OFF 1024
#define SMEM_A_NT_STAGE_BYTES 16384
#define SMEM_A_NT_STRIDE 16384
#define SMEM_A_ATB_OFF 1024
#define SMEM_A_ATB_STAGE_BYTES 16384
#define SMEM_A_ATB_STRIDE 16384
#define SMEM_B_NT_OFF 99328
#define SMEM_B_NT_STAGE_BYTES 16384
#define SMEM_B_NT_STRIDE 16384
#define SMEM_B_AB_OFF 99328
#define SMEM_B_AB_STAGE_BYTES 16384
#define SMEM_B_AB_STRIDE 16384
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
#define SMEM_Q_FLAT_OFF 1024
#define SMEM_Q_FLAT_STAGE_BYTES 200704
#define SMEM_Q_FLAT_STRIDE 200704
#define SMEM_Q_WORDS_OFF 1024
#define SMEM_Q_WORDS_STAGE_BYTES 200704
#define SMEM_Q_WORDS_STRIDE 200704
#define SMEM_Q_HALVES_OFF 1024
#define SMEM_Q_HALVES_STAGE_BYTES 200704
#define SMEM_Q_HALVES_STRIDE 200704
#define SMEM_Q_ROUTER_OFF 200704
#define SMEM_Q_ROUTER_STAGE_BYTES 1024
#define SMEM_Q_ROUTER_STRIDE 1024
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
#define SMEM_HIDDEN_HALVES_OFF 197632
#define SMEM_HIDDEN_HALVES_STAGE_BYTES 32768
#define SMEM_HIDDEN_HALVES_STRIDE 32768
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
kernel_cake_mok_backward_mxfp8(const __grid_constant__ CUtensorMap dy_s, const __grid_constant__ CUtensorMap dg_s, const __grid_constant__ CUtensorMap du_s, const __grid_constant__ CUtensorMap dy_atb_s, const __grid_constant__ CUtensorMap dg_atb_s, const __grid_constant__ CUtensorMap du_atb_s, const __grid_constant__ CUtensorMap x_atb_s, const __grid_constant__ CUtensorMap h_atb_s, const __grid_constant__ CUtensorMap wg_s, const __grid_constant__ CUtensorMap wu_s, const __grid_constant__ CUtensorMap wd_s, const __grid_constant__ CUtensorMap dh_s, const __grid_constant__ CUtensorMap dx_s, const __grid_constant__ CUtensorMap dwg_s, const __grid_constant__ CUtensorMap dwu_s, const __grid_constant__ CUtensorMap dwd_s, const __grid_constant__ CUtensorMap dh_sw_s, const __grid_constant__ CUtensorMap gate_sw_s, const __grid_constant__ CUtensorMap up_sw_s, const __grid_constant__ CUtensorMap dg_sw_s, const __grid_constant__ CUtensorMap du_sw_s, const __grid_constant__ CUtensorMap dy_r, const __grid_constant__ CUtensorMap dy_sc_r, const __grid_constant__ CUtensorMap wd_t_r, const __grid_constant__ CUtensorMap wd_t_sc_r, const __grid_constant__ CUtensorMap dh_r, const __grid_constant__ CUtensorMap dg_r, const __grid_constant__ CUtensorMap dg_sc_r, const __grid_constant__ CUtensorMap du_r, const __grid_constant__ CUtensorMap du_sc_r, const __grid_constant__ CUtensorMap wg_t_r, const __grid_constant__ CUtensorMap wg_t_sc_r, const __grid_constant__ CUtensorMap wu_t_r, const __grid_constant__ CUtensorMap wu_t_sc_r, const __grid_constant__ CUtensorMap dx_r, const __grid_constant__ CUtensorMap dy_t_r, const __grid_constant__ CUtensorMap dy_sc_t_r, const __grid_constant__ CUtensorMap h_t_r, const __grid_constant__ CUtensorMap h_sc_t_r, const __grid_constant__ CUtensorMap dwd_r, const __grid_constant__ CUtensorMap dg_t_r, const __grid_constant__ CUtensorMap dg_sc_t_r, const __grid_constant__ CUtensorMap du_t_r, const __grid_constant__ CUtensorMap du_sc_t_r, const __grid_constant__ CUtensorMap x_t_r, const __grid_constant__ CUtensorMap x_sc_t_r, const __grid_constant__ CUtensorMap dwg_r, const __grid_constant__ CUtensorMap dwu_r, const __grid_constant__ CUtensorMap x_r, const __grid_constant__ CUtensorMap x_sc_r, const __grid_constant__ CUtensorMap wg_r, const __grid_constant__ CUtensorMap wg_sc_r, const __grid_constant__ CUtensorMap wu_r, const __grid_constant__ CUtensorMap wu_sc_r, const __grid_constant__ CUtensorMap gate_out_r, const __grid_constant__ CUtensorMap up_out_r, const __grid_constant__ CUtensorMap gate_fp8_out_r, const __grid_constant__ CUtensorMap up_fp8_out_r, const __grid_constant__ CUtensorMap gate_sc_r, const __grid_constant__ CUtensorMap up_sc_r, const __grid_constant__ CUtensorMap dh_rows_r, const __grid_constant__ CUtensorMap gate_fp8_r, const __grid_constant__ CUtensorMap up_fp8_r, const __grid_constant__ CUtensorMap dg_fp8_r, const __grid_constant__ CUtensorMap du_fp8_r, const __grid_constant__ CUtensorMap dg_fp8_t_r, const __grid_constant__ CUtensorMap du_fp8_t_r, const __grid_constant__ CUtensorMap gate_rows_r, const __grid_constant__ CUtensorMap up_rows_r, const __grid_constant__ CUtensorMap h_fp8_r, const __grid_constant__ CUtensorMap h_sc_r, const __grid_constant__ CUtensorMap h_fp8_t_r, const __grid_constant__ CUtensorMap dy_disp_fp8, const __grid_constant__ CUtensorMap dy_disp_fp8_t, const __grid_constant__ CUtensorMap x_disp_fp8, const __grid_constant__ CUtensorMap x_disp_fp8_t, __nv_bfloat16* __restrict__ dx_routed_ptr, float* __restrict__ weights, float* __restrict__ partials, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ dy_peers, unsigned long long* __restrict__ dx_peers, unsigned long long* __restrict__ weight_peers, unsigned long long* __restrict__ dweight_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ dh_ready, int* __restrict__ dg_ready, int* __restrict__ dy_ready, int* __restrict__ dx_ready, int* __restrict__ replay_x, int* __restrict__ replay_gu, int* __restrict__ replay_h, int* __restrict__ buffers_done, int* __restrict__ weight_ready, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, int swiglu_clamped)
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
    __nv_bfloat16* b_nt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int b_nt_addr = smem + 99328;
    __nv_bfloat16* b_ab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int b_ab_addr = smem + 99328;
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
    __nv_bfloat16* q_flat = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_flat_addr = smem + 1024;
    unsigned int* q_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int q_words_addr = smem + 1024;
    uint16_t* q_halves = reinterpret_cast<uint16_t*>(smem_raw + 1024);
    const int q_halves_addr = smem + 1024;
    float* q_router = reinterpret_cast<float*>(smem_raw + 200704);
    const int q_router_addr = smem + 200704;
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
    uint16_t* hidden_halves = reinterpret_cast<uint16_t*>(smem_raw + 197632);
    const int hidden_halves_addr = smem + 197632;
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
    int shared_row_blocks = (local_tokens + 255) / 256;
    int shared_down = shared_row_blocks * (intermediate / 256);
    int shared_swiglu = (shared_row_blocks * 2 * (intermediate / 128) + 3) / 4;
    int shared_dx = shared_row_blocks * (hidden / 256);
    int shared_wgrad = intermediate / 256 * (hidden / 256);
    int shared_tasks = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
    int mini_down = mini_size / 256 * (intermediate / 256);
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 3) / 4;
    int mini_dx = mini_size / 256 * (hidden / 256);
    int mini_replay_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int mini_bwd = mini_down + mini_swiglu + mini_dx;
    int mini_replay = 2 * mini_down + mini_replay_swiglu;
    int wgrad_tasks = 3 * experts * shared_wgrad;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis = (tokens + mini_size - 1) / mini_size;
    int _min_0 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
    int saved_minis = (_min_0 + mini_size - 1) / mini_size;
    int weight_tasks = experts * shared_wgrad;
    int saved_tasks = saved_minis * mini_bwd + 3 * weight_tasks;
    int minis_per_macro = macro_size / mini_size;
    int replay_macro_tasks = minis_per_macro * (mini_replay + mini_bwd) + 3 * weight_tasks;
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
    const int tmem_accumulator = taddr;
    const int tmem_sf_a = taddr + 256;
    const int tmem_sf_b = taddr + 280;
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
        int macro_row_blocks = macro_size / 128;
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
                    cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2)), chunk_cols * 2, dispatch_arrived_addr);
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
                int row_block = row_1 / 128;
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
                                    v0 = v0 * dispatch_weights[src_row];
                                    v1 = v1 * dispatch_weights[src_row + 1];
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
                            tma_store_2d((&dy_disp_fp8), col128 * 128, row_1, dispatch_quant_addr + (unsigned int)(subtile % 2 * 4096 * 4));
                            tma_store_3d((&dy_sc_r), 0, (row_block * (hidden / 128) + col128) * 32, 0, dispatch_quant_addr + (unsigned int)((8192 + subtile % 2 * 128) * 4));
                            tma_store_2d((&dy_disp_fp8_t), row_1, col128 * 128, dispatch_quant_addr + (unsigned int)((8448 + subtile % 2 * 4096) * 4));
                            tma_store_3d((&dy_sc_t_r), 0, (col128 * macro_row_blocks + row_block) * 32, 0, dispatch_quant_addr + (unsigned int)((16640 + subtile % 2 * 128) * 4));
                            asm volatile("cp.async.bulk.commit_group;");
                        }
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
                                    for (int col = 0; col < intermediate / 128; col++) {
                                        gradient = gradient + partials[rows_1[stage_3] * (intermediate / 128) + col];
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
                            cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_3])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2)), chunk_cols_2 * 2, dispatch_arrived_addr);
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
                        int row_block_1 = row_4 / 128;
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
                                            v0_3 = v0_3 * dispatch_weights[src_row_1];
                                            v1_3 = v1_3 * dispatch_weights[src_row_1 + 1];
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
                                        float _max_2 = max_noftz(amax_2 * inv_e4m3_max_1, scale_floor_1);
                                        float scale_2 = _max_2;
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
                                        float _max_3 = max_noftz(amax_3 * inv_e4m3_max_1, scale_floor_1);
                                        float scale_3 = _max_3;
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
                                    tma_store_2d((&dy_disp_fp8), col128_1 * 128, row_4, dispatch_quant_addr + (unsigned int)(subtile_1 % 2 * 4096 * 4));
                                    tma_store_3d((&dy_sc_r), 0, (row_block_1 * (hidden / 128) + col128_1) * 32, 0, dispatch_quant_addr + (unsigned int)((8192 + subtile_1 % 2 * 128) * 4));
                                    tma_store_2d((&dy_disp_fp8_t), row_4, col128_1 * 128, dispatch_quant_addr + (unsigned int)((8448 + subtile_1 % 2 * 4096) * 4));
                                    tma_store_3d((&dy_sc_t_r), 0, (col128_1 * macro_row_blocks + row_block_1) * 32, 0, dispatch_quant_addr + (unsigned int)((16640 + subtile_1 % 2 * 128) * 4));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
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
                            cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_4])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_2 * 512)) * (unsigned long long)2)), chunk_cols_3 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_smem) + (tid * 1024 + vec_2 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        mbarrier_wait(dispatch_arrived_addr, phase_bits_0 & 1);
                        phase_bits_0 = phase_bits_0 ^ 1;
                        float inv_e4m3_max_2 = 0.002232142857f;
                        float scale_floor_2 = 1e-12f;
                        int row_block_2 = row_5 / 128;
                        int num_subtiles_2 = chunk_cols_3 / 128;
                        #pragma unroll
                        for (int subtile_2 = 0; subtile_2 < 4; subtile_2++) {
                            if (num_subtiles_2 > subtile_2) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                __syncthreads();
                                int half_2 = tid % 128;
                                if (tid < 128) {
                                    int t_row_2 = half_2 % 64 * 2 + half_2 / 64;
                                    int rotation_4 = half_2 / 8;
                                    unsigned int t_scale_word_2 = 0;
                                    #pragma unroll 1
                                    for (int j_6 = 0; j_6 < 4; j_6++) {
                                        int k_block_2 = (j_6 + rotation_4) % 4;
                                        unsigned int t_words_2[16];
                                        #pragma unroll
                                        for (int k_8 = 0; k_8 < 16; k_8++) {
                                            int src_row_2 = k_block_2 * 32 + (half_2 * 4 + k_8 * 2) % 32;
                                            float v0_6 = (float)dispatch_smem[src_row_2 * 512 + subtile_2 * 128 + t_row_2];
                                            float v1_6 = (float)dispatch_smem[(src_row_2 + 1) * 512 + subtile_2 * 128 + t_row_2];
                                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(v0_6, v1_6));
                                            t_words_2[k_8] = __as_u32(_bf16x2_2);
                                        }
                                        unsigned int t_packed_2[8];
                                        uint32_t _bf16x2_abs_8;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_8) : "r"(t_words_2[0]));
                                        unsigned int amax2_4 = _bf16x2_abs_8;
                                        #pragma unroll
                                        for (int k_9 = 1; k_9 < 16; k_9++) {
                                            uint32_t _bf16x2_abs_9;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_9) : "r"(t_words_2[k_9]));
                                            uint32_t _bf16x2_max_4;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(amax2_4), "r"(_bf16x2_abs_9));
                                            amax2_4 = _bf16x2_max_4;
                                        }
                                        uint16_t _bf16_max_4;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_4) : "h"((uint16_t)(amax2_4 & 65535)), "h"((uint16_t)(amax2_4 >> 16)));
                                        float _cvt_f32_bf16_20;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(_bf16_max_4)));
                                        float amax_4 = _cvt_f32_bf16_20;
                                        float _max_4 = max_noftz(amax_4 * inv_e4m3_max_2, scale_floor_2);
                                        float scale_4 = _max_4;
                                        uint16_t _ue8m0x2_f32_4;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_4) : "f"(scale_4), "f"(scale_4));
                                        uint16_t codes_4 = _ue8m0x2_f32_4;
                                        unsigned int scale_byte_4 = (unsigned int)codes_4 & 255;
                                        unsigned int inv_bits_4 = 254 - scale_byte_4 << 23;
                                        float inv_4 = 0.0f;
                                        inv_4 = __uint_as_float(inv_bits_4);
                                        #pragma unroll
                                        for (int i_8 = 0; i_8 < 8; i_8++) {
                                            unsigned int w0_4 = t_words_2[2 * i_8];
                                            unsigned int w1_4 = t_words_2[2 * i_8 + 1];
                                            float _cvt_f32_bf16_21;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_21) : "h"((uint16_t)(w0_4 & 65535)));
                                            float v0_7 = _cvt_f32_bf16_21;
                                            float _cvt_f32_bf16_22;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(w0_4 >> 16)));
                                            float v1_7 = _cvt_f32_bf16_22;
                                            float _cvt_f32_bf16_23;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(w1_4 & 65535)));
                                            float v2_4 = _cvt_f32_bf16_23;
                                            float _cvt_f32_bf16_24;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(w1_4 >> 16)));
                                            float v3_4 = _cvt_f32_bf16_24;
                                            uint16_t _e4m3x2_f32_8;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(v1_7 * inv_4), "f"(v0_7 * inv_4));
                                            uint16_t lo_4 = _e4m3x2_f32_8;
                                            uint16_t _e4m3x2_f32_9;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_9) : "f"(v3_4 * inv_4), "f"(v2_4 * inv_4));
                                            uint16_t hi_4 = _e4m3x2_f32_9;
                                            t_packed_2[i_8] = (unsigned int)lo_4 | (unsigned int)hi_4 << 16;
                                        }
                                        unsigned int t_scale_byte_2 = scale_byte_4;
                                        #pragma unroll
                                        for (int i_9 = 0; i_9 < 8; i_9++) {
                                            int t_col_2 = k_block_2 * 32 + (half_2 * 4 + i_9 * 4) % 32;
                                            dispatch_quant[8448 + subtile_2 % 2 * 4096 + t_row_2 * 32 + t_col_2 / 4] = t_packed_2[i_9];
                                        }
                                        t_scale_word_2 = t_scale_word_2 | t_scale_byte_2 << (unsigned int)(k_block_2 * 8);
                                    }
                                    dispatch_quant[16640 + subtile_2 % 2 * 128 + t_row_2 % 32 * 4 + t_row_2 / 32] = t_scale_word_2;
                                } else {
                                    int n_row_2 = half_2;
                                    int rotation_5 = half_2 / 8;
                                    unsigned int words_2[64];
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 4; j_7++) {
                                        int k_block_j_2 = (j_7 + rotation_5) % 4;
                                        #pragma unroll
                                        for (int k_10 = 0; k_10 < 16; k_10++) {
                                            int src_col_2 = k_block_j_2 * 32 + (half_2 * 4 + k_10 * 2) % 32;
                                            words_2[j_7 * 16 + k_10] = dispatch_words[n_row_2 * 256 + (subtile_2 * 128 + src_col_2) / 2];
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    unsigned int n_scale_word_2 = 0;
                                    #pragma unroll
                                    for (int j_8 = 0; j_8 < 4; j_8++) {
                                        int k_block_n_2 = (j_8 + rotation_5) % 4;
                                        unsigned int n_packed_2[8];
                                        uint32_t _bf16x2_abs_10;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_10) : "r"(words_2[j_8 * 16]));
                                        unsigned int amax2_5 = _bf16x2_abs_10;
                                        #pragma unroll
                                        for (int k_11 = 1; k_11 < 16; k_11++) {
                                            uint32_t _bf16x2_abs_11;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_11) : "r"(words_2[j_8 * 16 + k_11]));
                                            uint32_t _bf16x2_max_5;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(amax2_5), "r"(_bf16x2_abs_11));
                                            amax2_5 = _bf16x2_max_5;
                                        }
                                        uint16_t _bf16_max_5;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_5) : "h"((uint16_t)(amax2_5 & 65535)), "h"((uint16_t)(amax2_5 >> 16)));
                                        float _cvt_f32_bf16_25;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(_bf16_max_5)));
                                        float amax_5 = _cvt_f32_bf16_25;
                                        float _max_5 = max_noftz(amax_5 * inv_e4m3_max_2, scale_floor_2);
                                        float scale_5 = _max_5;
                                        uint16_t _ue8m0x2_f32_5;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_5) : "f"(scale_5), "f"(scale_5));
                                        uint16_t codes_5 = _ue8m0x2_f32_5;
                                        unsigned int scale_byte_5 = (unsigned int)codes_5 & 255;
                                        unsigned int inv_bits_5 = 254 - scale_byte_5 << 23;
                                        float inv_5 = 0.0f;
                                        inv_5 = __uint_as_float(inv_bits_5);
                                        #pragma unroll
                                        for (int i_10 = 0; i_10 < 8; i_10++) {
                                            unsigned int w0_5 = words_2[j_8 * 16 + 2 * i_10];
                                            unsigned int w1_5 = words_2[j_8 * 16 + 2 * i_10 + 1];
                                            float _cvt_f32_bf16_26;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(w0_5 & 65535)));
                                            float v0_8 = _cvt_f32_bf16_26;
                                            float _cvt_f32_bf16_27;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_27) : "h"((uint16_t)(w0_5 >> 16)));
                                            float v1_8 = _cvt_f32_bf16_27;
                                            float _cvt_f32_bf16_28;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(w1_5 & 65535)));
                                            float v2_5 = _cvt_f32_bf16_28;
                                            float _cvt_f32_bf16_29;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(w1_5 >> 16)));
                                            float v3_5 = _cvt_f32_bf16_29;
                                            uint16_t _e4m3x2_f32_10;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_10) : "f"(v1_8 * inv_5), "f"(v0_8 * inv_5));
                                            uint16_t lo_5 = _e4m3x2_f32_10;
                                            uint16_t _e4m3x2_f32_11;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_11) : "f"(v3_5 * inv_5), "f"(v2_5 * inv_5));
                                            uint16_t hi_5 = _e4m3x2_f32_11;
                                            n_packed_2[i_10] = (unsigned int)lo_5 | (unsigned int)hi_5 << 16;
                                        }
                                        unsigned int n_scale_byte_2 = scale_byte_5;
                                        #pragma unroll
                                        for (int i_11 = 0; i_11 < 8; i_11++) {
                                            int n_col_2 = k_block_n_2 * 32 + (half_2 * 4 + i_11 * 4) % 32;
                                            dispatch_quant[subtile_2 % 2 * 4096 + n_row_2 * 32 + n_col_2 / 4] = n_packed_2[i_11];
                                        }
                                        n_scale_word_2 = n_scale_word_2 | n_scale_byte_2 << (unsigned int)(k_block_n_2 * 8);
                                    }
                                    dispatch_quant[8192 + subtile_2 % 2 * 128 + n_row_2 % 32 * 4 + n_row_2 / 32] = n_scale_word_2;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int col128_2 = col_block_2 * 4 + subtile_2;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&x_disp_fp8), col128_2 * 128, row_5, dispatch_quant_addr + (unsigned int)(subtile_2 % 2 * 4096 * 4));
                                    tma_store_3d((&x_sc_r), 0, (row_block_2 * (hidden / 128) + col128_2) * 32, 0, dispatch_quant_addr + (unsigned int)((8192 + subtile_2 % 2 * 128) * 4));
                                    tma_store_2d((&x_disp_fp8_t), row_5, col128_2 * 128, dispatch_quant_addr + (unsigned int)((8448 + subtile_2 % 2 * 4096) * 4));
                                    tma_store_3d((&x_sc_t_r), 0, (col128_2 * macro_row_blocks + row_block_2) * 32, 0, dispatch_quant_addr + (unsigned int)((16640 + subtile_2 % 2 * 128) * 4));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
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
        int kind = -1;
        int task_3 = 0;
        int macro_1 = 0;
        int mini_1 = 0;
        int shared = 0;
        int kind_0 = -1;
        int task_1_1 = 0;
        int macro_2 = 0;
        int mini_3 = 0;
        int shared_4 = 0;
        if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
            if (shared_tasks > cluster - comm_clusters) {
                shared_4 = 1;
                if (shared_down > cluster - comm_clusters) {
                    kind_0 = 0;
                    task_1_1 = cluster - comm_clusters;
                } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                    kind_0 = 1;
                    task_1_1 = cluster - comm_clusters - shared_down;
                } else {
                    if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                        kind_0 = 2;
                        task_1_1 = cluster - comm_clusters - shared_down - shared_swiglu;
                    } else {
                        int weight_task = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                        kind_0 = 3 + weight_task / shared_wgrad;
                        task_1_1 = weight_task % shared_wgrad;
                    }
                }
            } else {
                int routed = cluster - comm_clusters - shared_tasks;
                int macro_task = routed;
                int macro_minis = saved_minis;
                int replay_tasks = 0;
                if (routed >= saved_tasks) {
                    macro_2 = 1 + (routed - saved_tasks) / replay_macro_tasks;
                    macro_task = (routed - saved_tasks) % replay_macro_tasks;
                    int _min_24 = ((tokens - macro_2 * macro_size) < (macro_size) ? (tokens - macro_2 * macro_size) : (macro_size));
                    macro_minis = (_min_24 + mini_size - 1) / mini_size;
                    replay_tasks = macro_minis * mini_replay;
                }
                if (macro_task < replay_tasks) {
                    mini_3 = macro_task / mini_replay;
                    int mini_task = macro_task % mini_replay;
                    if (mini_task < mini_down) {
                        kind_0 = 6;
                        task_1_1 = mini_task;
                    } else if (mini_task < 2 * mini_down) {
                        kind_0 = 7;
                        task_1_1 = mini_task - mini_down;
                    } else {
                        kind_0 = 8;
                        task_1_1 = mini_task - 2 * mini_down;
                    }
                } else {
                    int bwd_task = macro_task - replay_tasks;
                    if (bwd_task < macro_minis * mini_bwd) {
                        mini_3 = bwd_task / mini_bwd;
                        int mini_task_1 = bwd_task % mini_bwd;
                        if (mini_task_1 < mini_down) {
                            kind_0 = 0;
                            task_1_1 = mini_task_1;
                        } else if (mini_task_1 < mini_down + mini_swiglu) {
                            kind_0 = 1;
                            task_1_1 = mini_task_1 - mini_down;
                        } else {
                            kind_0 = 2;
                            task_1_1 = mini_task_1 - mini_down - mini_swiglu;
                        }
                    } else {
                        int weight_task_1 = bwd_task - macro_minis * mini_bwd;
                        kind_0 = 3 + weight_task_1 / weight_tasks;
                        task_1_1 = weight_task_1 % weight_tasks;
                    }
                }
            }
        }
        kind = kind_0;
        task_3 = task_1_1;
        macro_1 = macro_2;
        mini_1 = mini_3;
        shared = shared_4;
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
            unsigned int gemm_phase = gemm_bits;
            unsigned int swiglu_phase = swiglu_bits;
            unsigned int replay_phase = replay_bits;
            int row_count = 2 * (intermediate / 128);
            int shared_rows = (local_tokens + 255) / 256;
            int shared_down_0 = shared_rows * (intermediate / 256);
            if (shared != 0) {
                if (kind == 0) {
                    int col_blocks_3 = hidden / 256;
                    {
                        col_blocks_3 = intermediate / 256;
                    }
                    int x = -1;
                    int y = -1;
                    int expert = -1;
                    int k_start = 0;
                    int k_end = 0;
                    int first = 0;
                    int row_blocks = (local_tokens + 255) / 256;
                    if (task_3 < row_blocks * col_blocks_3) {
                        int supergroup = task_3 / (row_blocks * 8);
                        int full_cols = col_blocks_3 / 8 * 8;
                        int row_6 = 0;
                        int col_1 = 0;
                        if (task_3 < row_blocks * full_cols) {
                            row_6 = task_3 % (row_blocks * 8) / 8;
                            col_1 = supergroup * 8 + task_3 % 8;
                        } else {
                            row_6 = (task_3 - row_blocks * full_cols) / (col_blocks_3 - full_cols);
                            col_1 = full_cols + (task_3 - row_blocks * full_cols) % (col_blocks_3 - full_cols);
                        }
                        if ((supergroup & 1) != 0) {
                            row_6 = row_blocks - row_6 - 1;
                        }
                        x = row_6;
                        y = col_1;
                        expert = 0;
                    }
                    unsigned int phase_bits_3 = gemm_phase;
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
                                    int _min_25 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int _max_6 = ((0) > (_min_25) ? (0) : (_min_25));
                                    int mini_rows_4 = _max_6;
                                    int required_3 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                }
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
                                        :: "r"(b_ab_addr + (unsigned int)(ring * 16384)), "l"((&wd_s)), "r"(0), "r"(idx * 64), "r"(y * 4 + cta_rank_0 * 2), "r"(expert), "r"(0),
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
                                for (int half_3 = 0; half_3 < 2; half_3++) {
                                    unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 + half_3 * 16 << 16) + (unsigned int)(chunk * 32);
                                    float _tmem_load_0[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                        : "r"(address));
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; pair++) {
                                        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]));
                                        packed[chunk * 16 + half_3 * 8 + pair] = __as_u32(_bf16x2_3);
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
                                int output_row = x * 256 + cta_rank_0 * 128;
                                int _min_26 = ((macro_size) < (tokens - previous_offset_3) ? (macro_size) : (tokens - previous_offset_3));
                                if (output_row < _min_26) {
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
                                for (int half_4 = 0; half_4 < 2; half_4++) {
                                    #pragma unroll
                                    for (int col_tile = 0; col_tile < 2; col_tile++) {
                                        int row_7 = warp_0 * 32 + half_4 * 16 + lane_1 % 16;
                                        int col_2 = col_tile * 16 + lane_1 / 16 * 8;
                                        unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_7 * 32 + col_2) * 2);
                                        address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                        int offset_2 = chunk_1 * 16 + half_4 * 8 + col_tile * 4;
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
                                        :: "l"((&dh_s)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + chunk_1), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + ((macro_rows + x) * (intermediate / 256) + y))), "r"(static_cast<unsigned int>(1)) : "memory");
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
                    int num_tiles = ((local_tokens + 255) / 256 * 256 + 127) / 128 * col_blocks_4;
                    int macro_row_offset = 0;
                    int first_tile_1 = task_3 * 4 + cta_rank_0 * 2;
                    int tile_end = num_tiles;
                    if (first_tile_1 < tile_end) {
                        int first_row_1 = first_tile_1 / col_blocks_4;
                        int first_col_1 = first_tile_1 % col_blocks_4;
                        if (tid == 0) {
                            #pragma unroll
                            for (int stage_4 = 0; stage_4 < 2; stage_4++) {
                                if (tile_end > first_tile_1 + stage_4) {
                                    int row_8 = first_row_1;
                                    int col_3 = first_col_1 + stage_4;
                                    if (col_3 >= col_blocks_4) {
                                        row_8 = row_8 + 1;
                                        col_3 = col_3 - col_blocks_4;
                                    }
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_4) * 8, 98304);
                                    int parent = row_8 / 2 * (intermediate / 256) + col_3 / 2;
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
                                        :: "r"(sw_dh_addr + (unsigned int)(stage_4 * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_8 - macro_row_offset) * 128), "r"(col_3 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_gate_addr + (unsigned int)(stage_4 * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_8 - macro_row_offset) * 128), "r"(col_3 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_up_addr + (unsigned int)(stage_4 * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_8 - macro_row_offset) * 128), "r"(col_3 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_4) * 8) : "memory");
                                }
                            }
                        }
                        #pragma unroll
                        for (int stage_5 = 0; stage_5 < 2; stage_5++) {
                            if (tile_end > first_tile_1 + stage_5) {
                                mbarrier_wait(swiglu_arrived_addr + (stage_5) * 8, phase_bits_4 >> (unsigned int)stage_5 & 1);
                                phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << stage_5);
                                int row_9 = first_row_1;
                                int col_4 = first_col_1 + stage_5;
                                if (col_4 >= col_blocks_4) {
                                    row_9 = row_9 + 1;
                                    col_4 = col_4 - col_blocks_4;
                                }
                                float gate[64];
                                float up[64];
                                float dhidden[64];
                                if (swiglu_clamped != 0) {
                                    int warp_0_1 = tid / 32;
                                    int local_warp = warp_0_1 / 4 + warp_0_1 % 4 * 2;
                                    int lane_2 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col = 0; tile_col < 8; tile_col++) {
                                        unsigned int packed_1[4];
                                        unsigned int address_2 = sw_gate_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col * 16 + lane_2 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_2 % 16) * 64 + (tile_col * 16 + lane_2 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_1[0]), "=r"(packed_1[1]), "=r"(packed_1[2]), "=r"(packed_1[3])
                                            : "r"(address_2 ^ (address_2 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                                            float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(packed_1[pair_1]));
                                            gate[tile_col * 8 + pair_1 * 2] = _cvt_f32_0.x;
                                            gate[tile_col * 8 + pair_1 * 2 + 1] = _cvt_f32_0.y;
                                        }
                                    }
                                    int warp_1 = tid / 32;
                                    int local_warp_2 = warp_1 / 4 + warp_1 % 4 * 2;
                                    int lane_3 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                        unsigned int packed_2[4];
                                        unsigned int address_3 = sw_up_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_3 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3 % 16) * 64 + (tile_col_1 * 16 + lane_3 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                            : "r"(address_3 ^ (address_3 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                                            float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(packed_2[pair_2]));
                                            up[tile_col_1 * 8 + pair_2 * 2] = _cvt_f32_1.x;
                                            up[tile_col_1 * 8 + pair_2 * 2 + 1] = _cvt_f32_1.y;
                                        }
                                    }
                                    int warp_4 = tid / 32;
                                    int local_warp_5 = warp_4 / 4 + warp_4 % 4 * 2;
                                    int lane_6 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                        unsigned int packed_3[4];
                                        unsigned int address_4 = sw_dh_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                            : "r"(address_4 ^ (address_4 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                                            float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(packed_3[pair_3]));
                                            dhidden[tile_col_2 * 8 + pair_3 * 2] = _cvt_f32_2.x;
                                            dhidden[tile_col_2 * 8 + pair_3 * 2 + 1] = _cvt_f32_2.y;
                                        }
                                    }
                                    #pragma unroll
                                    for (int elem = 0; elem < 64; elem++) {
                                        float g = gate[elem];
                                        float u = up[elem];
                                        float dh = dhidden[elem];
                                        bool gate_mask = g <= swiglu_limit;
                                        bool up_mask = u >= -swiglu_limit && u <= swiglu_limit;
                                        float _min_28 = fminf(g, swiglu_limit);
                                        float g_c = _min_28;
                                        float _max_7 = max_noftz(u, -swiglu_limit);
                                        float _min_29 = fminf(_max_7, swiglu_limit);
                                        float u_c = _min_29;
                                        float _exp_0 = expf(-g_c);
                                        float sigmoid = 1.0f / (1.0f + _exp_0);
                                        float silu = g_c * sigmoid;
                                        float dsilu = (1.0f - silu) * sigmoid + silu;
                                        gate[elem] = ((gate_mask) ? dsilu * u_c * dh : 0.0f);
                                        up[elem] = ((up_mask) ? silu * dh : 0.0f);
                                    }
                                    int warp_7 = tid / 32;
                                    int local_warp_8 = warp_7 / 4 + warp_7 % 4 * 2;
                                    int lane_9 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                        unsigned int packed_4[4];
                                        #pragma unroll
                                        for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(gate[tile_col_3 * 8 + pair_4 * 2], gate[tile_col_3 * 8 + pair_4 * 2 + 1]));
                                            packed_4[pair_4] = __as_u32(_bf16x2_4);
                                        }
                                        unsigned int address_5 = sw_gate_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_9 / 16 * 8) / 64 * 128 * 64 + (local_warp_8 * 16 + lane_9 % 16) * 64 + (tile_col_3 * 16 + lane_9 / 16 * 8) % 64) * 2);
                                        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(address_5 ^ (address_5 & 1023) >> 7 << 4);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                            : "memory");
                                    }
                                    int warp_10 = tid / 32;
                                    int local_warp_11 = warp_10 / 4 + warp_10 % 4 * 2;
                                    int lane_12 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                        unsigned int packed_5[4];
                                        #pragma unroll
                                        for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(up[tile_col_4 * 8 + pair_5 * 2], up[tile_col_4 * 8 + pair_5 * 2 + 1]));
                                            packed_5[pair_5] = __as_u32(_bf16x2_5);
                                        }
                                        unsigned int address_6 = sw_up_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_12 / 16 * 8) / 64 * 128 * 64 + (local_warp_11 * 16 + lane_12 % 16) * 64 + (tile_col_4 * 16 + lane_12 / 16 * 8) % 64) * 2);
                                        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(address_6 ^ (address_6 & 1023) >> 7 << 4);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[3]))
                                            : "memory");
                                    }
                                    __syncthreads();
                                } else {
                                    int warp_0_2 = tid / 32;
                                    int local_warp_1 = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                                    int lane_4 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                        unsigned int packed_6[4];
                                        unsigned int address_7 = sw_gate_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_5 * 16 + lane_4 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_4 % 16) * 64 + (tile_col_5 * 16 + lane_4 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_6[0]), "=r"(packed_6[1]), "=r"(packed_6[2]), "=r"(packed_6[3])
                                            : "r"(address_7 ^ (address_7 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_6 = 0; pair_6 < 4; pair_6++) {
                                            float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(packed_6[pair_6]));
                                            gate[tile_col_5 * 8 + pair_6 * 2] = _cvt_f32_3.x;
                                            gate[tile_col_5 * 8 + pair_6 * 2 + 1] = _cvt_f32_3.y;
                                        }
                                    }
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                        dhidden[elem_1] = gate[elem_1] * -1.0f;
                                    }
                                    #pragma unroll
                                    for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                        float _exp_1 = expf(dhidden[elem_2]);
                                        dhidden[elem_2] = _exp_1;
                                    }
                                    #pragma unroll
                                    for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                        dhidden[elem_3] = dhidden[elem_3] + 1.0f;
                                    }
                                    #pragma unroll
                                    for (int elem_4 = 0; elem_4 < 64; elem_4++) {
                                        gate[elem_4] = gate[elem_4] / dhidden[elem_4];
                                    }
                                    #pragma unroll
                                    for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                        up[elem_5] = gate[elem_5] * -1.0f;
                                    }
                                    #pragma unroll
                                    for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                        up[elem_6] = up[elem_6] + 1.0f;
                                    }
                                    #pragma unroll
                                    for (int elem_7 = 0; elem_7 < 64; elem_7++) {
                                        up[elem_7] = up[elem_7] / dhidden[elem_7];
                                    }
                                    #pragma unroll
                                    for (int elem_8 = 0; elem_8 < 64; elem_8++) {
                                        up[elem_8] = up[elem_8] + gate[elem_8];
                                    }
                                    int warp_1_1 = tid / 32;
                                    int local_warp_2_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                    int lane_3_1 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_6 = 0; tile_col_6 < 8; tile_col_6++) {
                                        unsigned int packed_7[4];
                                        unsigned int address_8 = sw_dh_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_6 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_1 * 16 + lane_3_1 % 16) * 64 + (tile_col_6 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_7[0]), "=r"(packed_7[1]), "=r"(packed_7[2]), "=r"(packed_7[3])
                                            : "r"(address_8 ^ (address_8 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_7 = 0; pair_7 < 4; pair_7++) {
                                            float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(packed_7[pair_7]));
                                            dhidden[tile_col_6 * 8 + pair_7 * 2] = _cvt_f32_4.x;
                                            dhidden[tile_col_6 * 8 + pair_7 * 2 + 1] = _cvt_f32_4.y;
                                        }
                                    }
                                    #pragma unroll
                                    for (int elem_9 = 0; elem_9 < 64; elem_9++) {
                                        gate[elem_9] = gate[elem_9] * dhidden[elem_9];
                                    }
                                    #pragma unroll
                                    for (int elem_10 = 0; elem_10 < 64; elem_10++) {
                                        dhidden[elem_10] = dhidden[elem_10] * up[elem_10];
                                    }
                                    int warp_4_1 = tid / 32;
                                    int local_warp_5_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                                    int lane_6_1 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_7 = 0; tile_col_7 < 8; tile_col_7++) {
                                        unsigned int packed_8[4];
                                        unsigned int address_9 = sw_up_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_7 * 16 + lane_6_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_5_1 * 16 + lane_6_1 % 16) * 64 + (tile_col_7 * 16 + lane_6_1 / 16 * 8) % 64) * 2);
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(packed_8[0]), "=r"(packed_8[1]), "=r"(packed_8[2]), "=r"(packed_8[3])
                                            : "r"(address_9 ^ (address_9 & 1023) >> 7 << 4)
                                            : "memory");
                                        #pragma unroll
                                        for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                            float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(packed_8[pair_8]));
                                            up[tile_col_7 * 8 + pair_8 * 2] = _cvt_f32_5.x;
                                            up[tile_col_7 * 8 + pair_8 * 2 + 1] = _cvt_f32_5.y;
                                        }
                                    }
                                    #pragma unroll
                                    for (int elem_11 = 0; elem_11 < 64; elem_11++) {
                                        dhidden[elem_11] = dhidden[elem_11] * up[elem_11];
                                    }
                                    int warp_7_1 = tid / 32;
                                    int local_warp_8_1 = warp_7_1 / 4 + warp_7_1 % 4 * 2;
                                    int lane_9_1 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_8 = 0; tile_col_8 < 8; tile_col_8++) {
                                        unsigned int packed_9[4];
                                        #pragma unroll
                                        for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(dhidden[tile_col_8 * 8 + pair_9 * 2], dhidden[tile_col_8 * 8 + pair_9 * 2 + 1]));
                                            packed_9[pair_9] = __as_u32(_bf16x2_6);
                                        }
                                        unsigned int address_10 = sw_gate_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_8 * 16 + lane_9_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_8_1 * 16 + lane_9_1 % 16) * 64 + (tile_col_8 * 16 + lane_9_1 / 16 * 8) % 64) * 2);
                                        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_10 ^ (address_10 & 1023) >> 7 << 4);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_9[3]))
                                            : "memory");
                                    }
                                    int warp_10_1 = tid / 32;
                                    int local_warp_11_1 = warp_10_1 / 4 + warp_10_1 % 4 * 2;
                                    int lane_12_1 = tid % 32;
                                    #pragma unroll
                                    for (int tile_col_9 = 0; tile_col_9 < 8; tile_col_9++) {
                                        unsigned int packed_10[4];
                                        #pragma unroll
                                        for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(gate[tile_col_9 * 8 + pair_10 * 2], gate[tile_col_9 * 8 + pair_10 * 2 + 1]));
                                            packed_10[pair_10] = __as_u32(_bf16x2_7);
                                        }
                                        unsigned int address_11 = sw_up_addr + (unsigned int)(stage_5 * 32768) + (unsigned int)(((tile_col_9 * 16 + lane_12_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_11_1 * 16 + lane_12_1 % 16) * 64 + (tile_col_9 * 16 + lane_12_1 / 16 * 8) % 64) * 2);
                                        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(address_11 ^ (address_11 & 1023) >> 7 << 4);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[3]))
                                            : "memory");
                                    }
                                    __syncthreads();
                                }
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&dg_sw_s), 0, (row_9 - macro_row_offset) * 128, col_4 * 2, 0, 0, sw_gate_addr + (unsigned int)(stage_5 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&du_sw_s), 0, (row_9 - macro_row_offset) * 128, col_4 * 2, 0, 0, sw_up_addr + (unsigned int)(stage_5 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_6 = 0; stage_6 < 2; stage_6++) {
                                if (tile_end > first_tile_1 + stage_6) {
                                    int row_10 = first_row_1;
                                    if (col_blocks_4 <= first_col_1 + stage_6) {
                                        row_10 = row_10 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (row_10 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                    }
                    if (tid == 0) {
                    }
                    swiglu_phase = phase_bits_4;
                } else {
                    if (kind == 2) {
                        int col_blocks_5 = intermediate / 256;
                        {
                            col_blocks_5 = hidden / 256;
                        }
                        int x_1 = -1;
                        int y_1 = -1;
                        int expert_1 = -1;
                        int k_start_1 = 0;
                        int k_end_1 = 0;
                        int first_1 = 0;
                        int row_blocks_1 = (local_tokens + 255) / 256;
                        if (task_3 < row_blocks_1 * col_blocks_5) {
                            int supergroup_1 = task_3 / (row_blocks_1 * 8);
                            int full_cols_1 = col_blocks_5 / 8 * 8;
                            int row_11 = 0;
                            int col_5 = 0;
                            if (task_3 < row_blocks_1 * full_cols_1) {
                                row_11 = task_3 % (row_blocks_1 * 8) / 8;
                                col_5 = supergroup_1 * 8 + task_3 % 8;
                            } else {
                                row_11 = (task_3 - row_blocks_1 * full_cols_1) / (col_blocks_5 - full_cols_1);
                                col_5 = full_cols_1 + (task_3 - row_blocks_1 * full_cols_1) % (col_blocks_5 - full_cols_1);
                            }
                            if ((supergroup_1 & 1) != 0) {
                                row_11 = row_blocks_1 - row_11 - 1;
                            }
                            x_1 = row_11;
                            y_1 = col_5;
                            expert_1 = 0;
                        }
                        unsigned int phase_bits_5 = gemm_phase;
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
                                            int32_t _relaxed_ld_10;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                            int value_7 = _relaxed_ld_10;
                                            while (value_7 < row_count) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_11;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                                value_7 = _relaxed_ld_11;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_30 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                        int _max_8 = ((0) > (_min_30) ? (0) : (_min_30));
                                        int mini_rows_5 = _max_8;
                                        int required_4 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
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
                                                :: "r"(b_ab_addr + (unsigned int)(ring_2 * 16384)), "l"((&wg_s)), "r"(0), "r"(idx_2 * 64), "r"(y_1 * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
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
                                                :: "r"(b_ab_addr + (unsigned int)(ring_2 * 16384)), "l"((&wu_s)), "r"(0), "r"((idx_2 - intermediate / 64) * 64), "r"(y_1 * 4 + cta_rank_0 * 2), "r"(expert_1), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        }
                                        phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_2);
                                        ring_2 = (ring_2 + 1) % 6;
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
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_5 >> (unsigned int)ring_3 & 1);
                                            int _mma_a_lo_1 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_3) * 1024;
                                            int _mma_b_lo_1 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_3) * 1024;
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)));
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_3) * 8, (uint16_t)(3));
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << ring_3);
                                            ring_3 = (ring_3 + 1) % 6;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_5 >> 6 & 1);
                                phase_bits_5 = phase_bits_5 ^ 64;
                                unsigned int packed_11[128];
                                #pragma unroll
                                for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                                    #pragma unroll
                                    for (int half_5 = 0; half_5 < 2; half_5++) {
                                        unsigned int address_12 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_5 * 16 << 16) + (unsigned int)(chunk_2 * 32);
                                        float _tmem_load_1[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                            : "r"(address_12));
                                        #pragma unroll
                                        for (int pair_11 = 0; pair_11 < 8; pair_11++) {
                                            __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_tmem_load_1[pair_11 * 2], _tmem_load_1[pair_11 * 2 + 1]));
                                            packed_11[chunk_2 * 16 + half_5 * 8 + pair_11] = __as_u32(_bf16x2_8);
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
                                    int output_row_1 = x_1 * 256 + cta_rank_0 * 128;
                                    int _min_31 = ((macro_size) < (tokens - previous_offset_4) ? (macro_size) : (tokens - previous_offset_4));
                                    if (output_row_1 < _min_31) {
                                    }
                                }
                                #pragma unroll
                                for (int chunk_3 = 0; chunk_3 < 8; chunk_3++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_3 = tid / 32;
                                    int lane_5 = tid % 32;
                                    #pragma unroll
                                    for (int half_6 = 0; half_6 < 2; half_6++) {
                                        #pragma unroll
                                        for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                            int row_12 = warp_0_3 * 32 + half_6 * 16 + lane_5 % 16;
                                            int col_6 = col_tile_1 * 16 + lane_5 / 16 * 8;
                                            unsigned int address_13 = d_smem_addr + (unsigned int)(chunk_3 % 3 * 8192) + (unsigned int)((row_12 * 32 + col_6) * 2);
                                            address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                            int offset_3 = chunk_3 * 16 + half_6 * 8 + col_tile_1 * 4;
                                            uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(address_13);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_3 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&dx_s)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(y_1 * 8 + chunk_3), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_3 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                        gemm_phase = phase_bits_5;
                    } else if (kind == 3) {
                        int col_blocks_6 = local_tokens / 256;
                        {
                            col_blocks_6 = intermediate / 256;
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
                        k_end_2 = local_tokens;
                        first_2 = 1;
                        if (k_start_2 < k_end_2) {
                            int supergroup_2 = local_task / (row_blocks_2 * 8);
                            int full_cols_2 = col_blocks_6 / 8 * 8;
                            int row_13 = 0;
                            int col_7 = 0;
                            if (local_task < row_blocks_2 * full_cols_2) {
                                row_13 = local_task % (row_blocks_2 * 8) / 8;
                                col_7 = supergroup_2 * 8 + local_task % 8;
                            } else {
                                row_13 = (local_task - row_blocks_2 * full_cols_2) / (col_blocks_6 - full_cols_2);
                                col_7 = full_cols_2 + (local_task - row_blocks_2 * full_cols_2) % (col_blocks_6 - full_cols_2);
                            }
                            if ((supergroup_2 & 1) != 0) {
                                row_13 = row_blocks_2 - row_13 - 1;
                            }
                            x_2 = row_13;
                            y_2 = col_7;
                            expert_2 = expert_idx;
                        }
                        unsigned int phase_bits_6 = gemm_phase;
                        int global_mini_2 = 0;
                        int macro_rows_2 = 0;
                        int iterations_2 = hidden / 64;
                        iterations_2 = (k_end_2 - k_start_2 + 63) / 64;
                        if (expert_2 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    int ring_4 = 0;
                                    #pragma unroll 1
                                    for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                        int token_row = k_start_2 + idx_4 * 64;
                                        if (idx_4 == 0 || token_row % 256 == 0) {
                                        }
                                        if (idx_4 == 0 || token_row % mini_size == 0) {
                                            int input_mini = token_row / mini_size;
                                            int _min_33 = ((mini_size) < (tokens - input_mini * mini_size) ? (mini_size) : (tokens - input_mini * mini_size));
                                            int input_rows = _min_33;
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
                                            :: "r"(b_ab_addr + (unsigned int)(ring_4 * 16384)), "l"((&h_atb_s)), "r"(0), "r"(local_row), "r"(y_2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
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
                                            int _mma_a_lo_2 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_5) * 1024;
                                            int _mma_b_lo_2 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_5) * 1024;
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
                                unsigned int packed_12[128];
                                #pragma unroll
                                for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                    #pragma unroll
                                    for (int half_7 = 0; half_7 < 2; half_7++) {
                                        unsigned int address_14 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_7 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                        float _tmem_load_2[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                            : "r"(address_14));
                                        #pragma unroll
                                        for (int pair_12 = 0; pair_12 < 8; pair_12++) {
                                            __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_tmem_load_2[pair_12 * 2], _tmem_load_2[pair_12 * 2 + 1]));
                                            packed_12[chunk_4 * 16 + half_7 * 8 + pair_12] = __as_u32(_bf16x2_9);
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    int previous_offset_5 = macro_size;
                                    int output_row_2 = x_2 * 256 + cta_rank_0 * 128;
                                    int _min_34 = ((macro_size) < (tokens - previous_offset_5) ? (macro_size) : (tokens - previous_offset_5));
                                    if (output_row_2 < _min_34) {
                                    }
                                }
                                #pragma unroll
                                for (int chunk_5 = 0; chunk_5 < 8; chunk_5++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_4 = tid / 32;
                                    int lane_7 = tid % 32;
                                    #pragma unroll
                                    for (int half_8 = 0; half_8 < 2; half_8++) {
                                        #pragma unroll
                                        for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                            int row_14 = warp_0_4 * 32 + half_8 * 16 + lane_7 % 16;
                                            int col_8 = col_tile_2 * 16 + lane_7 / 16 * 8;
                                            unsigned int address_15 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_14 * 32 + col_8) * 2);
                                            address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                            int offset_4 = chunk_5 * 16 + half_8 * 8 + col_tile_2 * 4;
                                            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(address_15);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_4 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        if (first_2 != 0) {
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dwd_s)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + chunk_5), "r"(expert_2), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        } else {
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                            #error "TmaReduceAdd5d requires SM90 or newer"
                                            #endif
                                            asm volatile(
                                                "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dwd_s)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + chunk_5), "r"(expert_2), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        }
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
                        gemm_phase = phase_bits_6;
                    } else {
                        if (kind == 4) {
                            int col_blocks_7 = local_tokens / 256;
                            {
                                col_blocks_7 = hidden / 256;
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
                            k_end_3 = local_tokens;
                            first_3 = 1;
                            if (k_start_3 < k_end_3) {
                                int supergroup_3 = local_task_1 / (row_blocks_3 * 8);
                                int full_cols_3 = col_blocks_7 / 8 * 8;
                                int row_15 = 0;
                                int col_9 = 0;
                                if (local_task_1 < row_blocks_3 * full_cols_3) {
                                    row_15 = local_task_1 % (row_blocks_3 * 8) / 8;
                                    col_9 = supergroup_3 * 8 + local_task_1 % 8;
                                } else {
                                    row_15 = (local_task_1 - row_blocks_3 * full_cols_3) / (col_blocks_7 - full_cols_3);
                                    col_9 = full_cols_3 + (local_task_1 - row_blocks_3 * full_cols_3) % (col_blocks_7 - full_cols_3);
                                }
                                if ((supergroup_3 & 1) != 0) {
                                    row_15 = row_blocks_3 - row_15 - 1;
                                }
                                x_3 = row_15;
                                y_3 = col_9;
                                expert_3 = expert_idx_1;
                            }
                            unsigned int phase_bits_7 = gemm_phase;
                            int global_mini_3 = 0;
                            int macro_rows_3 = 0;
                            int iterations_3 = intermediate / 64;
                            iterations_3 = (k_end_3 - k_start_3 + 63) / 64;
                            if (expert_3 < 0) {
                                if (tid == 0) {
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        int ring_6 = 0;
                                        #pragma unroll 1
                                        for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                            int token_row_1 = k_start_3 + idx_6 * 64;
                                            if (idx_6 == 0 || token_row_1 % 256 == 0) {
                                                bool enabled_value_4 = 1;
                                                if (enabled_value_4 != 0) {
                                                    int32_t _relaxed_ld_14;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                    int value_8 = _relaxed_ld_14;
                                                    while (value_8 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_15;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                        value_8 = _relaxed_ld_15;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_6 == 0 || token_row_1 % mini_size == 0) {
                                                int input_mini_1 = token_row_1 / mini_size;
                                                int _min_36 = ((mini_size) < (tokens - input_mini_1 * mini_size) ? (mini_size) : (tokens - input_mini_1 * mini_size));
                                                int input_rows_1 = _min_36;
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
                                                :: "r"(b_ab_addr + (unsigned int)(ring_6 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_1), "r"(y_3 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 16 + ring_6);
                                            ring_6 = (ring_6 + 1) % 6;
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
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_7) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_7) * 8, phase_bits_7 >> (unsigned int)ring_7 & 1);
                                                int _mma_a_lo_3 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_7) * 1024;
                                                int _mma_b_lo_3 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_7) * 1024;
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
            :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_7) * 8, (uint16_t)(3));
                                                phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << ring_7);
                                                ring_7 = (ring_7 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_7 >> 6 & 1);
                                    phase_bits_7 = phase_bits_7 ^ 64;
                                    unsigned int packed_13[128];
                                    #pragma unroll
                                    for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                        #pragma unroll
                                        for (int half_9 = 0; half_9 < 2; half_9++) {
                                            unsigned int address_16 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_9 * 16 << 16) + (unsigned int)(chunk_6 * 32);
                                            float _tmem_load_3[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                                : "r"(address_16));
                                            #pragma unroll
                                            for (int pair_13 = 0; pair_13 < 8; pair_13++) {
                                                __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_3[pair_13 * 2], _tmem_load_3[pair_13 * 2 + 1]));
                                                packed_13[chunk_6 * 16 + half_9 * 8 + pair_13] = __as_u32(_bf16x2_10);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        int previous_offset_6 = macro_size;
                                        int output_row_3 = x_3 * 256 + cta_rank_0 * 128;
                                        int _min_37 = ((macro_size) < (tokens - previous_offset_6) ? (macro_size) : (tokens - previous_offset_6));
                                        if (output_row_3 < _min_37) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_7 = 0; chunk_7 < 8; chunk_7++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_5 = tid / 32;
                                        int lane_8 = tid % 32;
                                        #pragma unroll
                                        for (int half_10 = 0; half_10 < 2; half_10++) {
                                            #pragma unroll
                                            for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                                int row_16 = warp_0_5 * 32 + half_10 * 16 + lane_8 % 16;
                                                int col_10 = col_tile_3 * 16 + lane_8 / 16 * 8;
                                                unsigned int address_17 = d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192) + (unsigned int)((row_16 * 32 + col_10) * 2);
                                                address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                                int offset_5 = chunk_7 * 16 + half_10 * 8 + col_tile_3 * 4;
                                                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(address_17);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_5 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            if (first_3 != 0) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwg_s)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + chunk_7), "r"(expert_3), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            } else {
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwg_s)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + chunk_7), "r"(expert_3), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            }
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
                            gemm_phase = phase_bits_7;
                        } else if (kind == 5) {
                            int col_blocks_8 = local_tokens / 256;
                            {
                                col_blocks_8 = hidden / 256;
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
                            k_end_4 = local_tokens;
                            first_4 = 1;
                            if (k_start_4 < k_end_4) {
                                int supergroup_4 = local_task_2 / (row_blocks_4 * 8);
                                int full_cols_4 = col_blocks_8 / 8 * 8;
                                int row_17 = 0;
                                int col_11 = 0;
                                if (local_task_2 < row_blocks_4 * full_cols_4) {
                                    row_17 = local_task_2 % (row_blocks_4 * 8) / 8;
                                    col_11 = supergroup_4 * 8 + local_task_2 % 8;
                                } else {
                                    row_17 = (local_task_2 - row_blocks_4 * full_cols_4) / (col_blocks_8 - full_cols_4);
                                    col_11 = full_cols_4 + (local_task_2 - row_blocks_4 * full_cols_4) % (col_blocks_8 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_17 = row_blocks_4 - row_17 - 1;
                                }
                                x_4 = row_17;
                                y_4 = col_11;
                                expert_4 = expert_idx_2;
                            }
                            unsigned int phase_bits_8 = gemm_phase;
                            int global_mini_4 = 0;
                            int macro_rows_4 = 0;
                            int iterations_4 = intermediate / 64;
                            iterations_4 = (k_end_4 - k_start_4 + 63) / 64;
                            if (expert_4 < 0) {
                                if (tid == 0) {
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        int ring_8 = 0;
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_4; idx_8++) {
                                            int token_row_2 = k_start_4 + idx_8 * 64;
                                            if (idx_8 == 0 || token_row_2 % 256 == 0) {
                                                bool enabled_value_5 = 1;
                                                if (enabled_value_5 != 0) {
                                                    int32_t _relaxed_ld_18;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_18) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                    int value_9 = _relaxed_ld_18;
                                                    while (value_9 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_19;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_19) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                        value_9 = _relaxed_ld_19;
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
                                                :: "r"(b_ab_addr + (unsigned int)(ring_8 * 16384)), "l"((&x_atb_s)), "r"(0), "r"(local_row_2), "r"(y_4 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << 16 + ring_8);
                                            ring_8 = (ring_8 + 1) % 6;
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
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_9) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_9) * 8, phase_bits_8 >> (unsigned int)ring_9 & 1);
                                                int _mma_a_lo_4 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_9) * 1024;
                                                int _mma_b_lo_4 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_9) * 1024;
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
            :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_accumulator), "r"(((idx_9 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_9) * 8, (uint16_t)(3));
                                                phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << ring_9);
                                                ring_9 = (ring_9 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_8 >> 6 & 1);
                                    phase_bits_8 = phase_bits_8 ^ 64;
                                    unsigned int packed_14[128];
                                    #pragma unroll
                                    for (int chunk_8 = 0; chunk_8 < 8; chunk_8++) {
                                        #pragma unroll
                                        for (int half_11 = 0; half_11 < 2; half_11++) {
                                            unsigned int address_18 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_11 * 16 << 16) + (unsigned int)(chunk_8 * 32);
                                            float _tmem_load_4[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                                                : "r"(address_18));
                                            #pragma unroll
                                            for (int pair_14 = 0; pair_14 < 8; pair_14++) {
                                                __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_tmem_load_4[pair_14 * 2], _tmem_load_4[pair_14 * 2 + 1]));
                                                packed_14[chunk_8 * 16 + half_11 * 8 + pair_14] = __as_u32(_bf16x2_11);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        int previous_offset_7 = macro_size;
                                        int output_row_4 = x_4 * 256 + cta_rank_0 * 128;
                                        int _min_40 = ((macro_size) < (tokens - previous_offset_7) ? (macro_size) : (tokens - previous_offset_7));
                                        if (output_row_4 < _min_40) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_9 = 0; chunk_9 < 8; chunk_9++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_6 = tid / 32;
                                        int lane_10 = tid % 32;
                                        #pragma unroll
                                        for (int half_12 = 0; half_12 < 2; half_12++) {
                                            #pragma unroll
                                            for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                                int row_18 = warp_0_6 * 32 + half_12 * 16 + lane_10 % 16;
                                                int col_12 = col_tile_4 * 16 + lane_10 / 16 * 8;
                                                unsigned int address_19 = d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192) + (unsigned int)((row_18 * 32 + col_12) * 2);
                                                address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                                int offset_6 = chunk_9 * 16 + half_12 * 8 + col_tile_4 * 4;
                                                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(address_19);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[offset_6 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            if (first_4 != 0) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwu_s)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(y_4 * 8 + chunk_9), "r"(expert_4), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            } else {
                                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                #error "TmaReduceAdd5d requires SM90 or newer"
                                                #endif
                                                asm volatile(
                                                    "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dwu_s)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(y_4 * 8 + chunk_9), "r"(expert_4), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            }
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
                            gemm_phase = phase_bits_8;
                        }
                    }
                }
            } else if (kind == 6) {
                int col_blocks_9 = intermediate / 256;
                int x_5 = -1;
                int y_5 = -1;
                int expert_5 = -1;
                int k_start_5 = 0;
                int k_end_5 = 0;
                int first_5 = 0;
                int first_block = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                int offset_7 = 0;
                int remaining = task_3;
                #pragma unroll 1
                for (int index = 0; index < experts; index++) {
                    int blocks = counts[index] / 256;
                    int _max_12 = ((first_block) > (offset_7) ? (first_block) : (offset_7));
                    int first_row_2 = _max_12;
                    int _min_41 = ((first_block + mini_size / 256) < (offset_7 + blocks) ? (first_block + mini_size / 256) : (offset_7 + blocks));
                    int _max_13 = ((0) > (_min_41 - first_row_2) ? (0) : (_min_41 - first_row_2));
                    int rows_3 = _max_13;
                    int tasks = rows_3 * col_blocks_9;
                    if (remaining < tasks) {
                        int supergroup_5 = remaining / (rows_3 * 8);
                        int full_cols_5 = col_blocks_9 / 8 * 8;
                        int row_19 = 0;
                        int col_13 = 0;
                        if (remaining < rows_3 * full_cols_5) {
                            row_19 = remaining % (rows_3 * 8) / 8;
                            col_13 = supergroup_5 * 8 + remaining % 8;
                        } else {
                            row_19 = (remaining - rows_3 * full_cols_5) / (col_blocks_9 - full_cols_5);
                            col_13 = full_cols_5 + (remaining - rows_3 * full_cols_5) % (col_blocks_9 - full_cols_5);
                        }
                        if ((supergroup_5 & 1) != 0) {
                            row_19 = rows_3 - row_19 - 1;
                        }
                        x_5 = first_row_2 + row_19 - macro_1 * (macro_size / 256);
                        y_5 = col_13;
                        expert_5 = index;
                        break;
                    }
                    remaining = remaining - tasks;
                    offset_7 = offset_7 + blocks;
                }
                unsigned int phase_bits_9 = gemm_phase;
                int global_mini_5 = macro_1 * (macro_size / mini_size) + mini_1;
                int macro_rows_5 = macro_1 * (macro_size / 256);
                int iterations_5 = hidden / 128;
                int k_blocks = hidden / 128;
                int n_blocks = intermediate / 128;
                if (expert_5 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_42 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                int _max_14 = ((0) > (_min_42) ? (0) : (_min_42));
                                int mini_rows_6 = _max_14;
                                int required_5 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                bool enabled_value_6 = 1;
                                if (enabled_value_6 != 0) {
                                    int32_t _relaxed_ld_20;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_20) : "l"(replay_x + global_mini_5) : "memory");
                                    int value_10 = _relaxed_ld_20;
                                    while (value_10 < required_5) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_21;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_21) : "l"(replay_x + global_mini_5) : "memory");
                                        value_10 = _relaxed_ld_21;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                            }
                            int ring_10 = 0;
                            #pragma unroll 1
                            for (int idx_10 = 0; idx_10 < iterations_5; idx_10++) {
                                mbarrier_wait(gemm_finished_addr + (ring_10) * 8, phase_bits_9 >> (unsigned int)(16 + ring_10) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_fp8_smem_addr + (unsigned int)(ring_10 * 16384)), "l"((&x_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_fp8_smem_addr + (unsigned int)(ring_10 * 16384)), "l"((&wg_r)), "r"(0), "r"(y_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(expert_5), "r"(0),
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
                                    int _max_15 = ((0) > (_min_43) ? (0) : (_min_43));
                                    int mini_rows_7 = _max_15;
                                    int required_6 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_7 = 1;
                                    if (enabled_value_7 != 0) {
                                        int32_t _relaxed_ld_22;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_22) : "l"(replay_x + global_mini_5) : "memory");
                                        int value_11 = _relaxed_ld_22;
                                        while (value_11 < required_6) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_23;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_23) : "l"(replay_x + global_mini_5) : "memory");
                                            value_11 = _relaxed_ld_23;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                int ring_11 = 0;
                                #pragma unroll 1
                                for (int idx_11 = 0; idx_11 < iterations_5; idx_11++) {
                                    mbarrier_wait(scales_finished_addr + (ring_11) * 8, phase_bits_9 >> (unsigned int)(23 + ring_11) & 1);
                                    int a_tile = (x_5 * 2 + cta_rank_0) * k_blocks + idx_11;
                                    int b_tile = (expert_5 * n_blocks + y_5 * 2 + cta_rank_0) * k_blocks + idx_11;
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(a_sc_smem_addr + (unsigned int)(ring_11 * 512)), "l"((&x_sc_r)), "r"(0), "r"(a_tile * 32), "r"(0),
                                           "r"(((scales_arrived_addr + (ring_11) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(b_sc_smem_addr + (unsigned int)(ring_11 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_sc_r)), "r"(0), "r"(b_tile * 32), "r"(0),
                                           "r"(((scales_arrived_addr + (ring_11) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                    phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << 23 + ring_11);
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
                                    mbarrier_wait(scales_arrived_addr + (ring_12) * 8, phase_bits_9 >> (unsigned int)(7 + ring_12) & 1);
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_12 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_12 * 512)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_12 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_12 * 1024)));
                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_12 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_12 * 1024) + 512)));
                                    tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_12) * 8, (uint16_t)(3));
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_12) * 8, 65536);
                                    mbarrier_wait(gemm_arrived_addr + (ring_12) * 8, phase_bits_9 >> (unsigned int)ring_12 & 1);
                                    int _mma_a_lo_5 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_12) * 1024;
                                    int _mma_b_lo_5 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_12) * 1024;
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                            (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_12 * 4, tmem_sf_b + ring_12 * 8, ((idx_12 == 0) ? 0 : 1));
                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                            (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_12 * 4, tmem_sf_b + ring_12 * 8, 1);
                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                            (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_12 * 4, tmem_sf_b + ring_12 * 8, 1);
                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                            (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_12 * 4, tmem_sf_b + ring_12 * 8, 1);
                                    }
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_12) * 8, (uint16_t)(3));
                                    phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << ring_12);
                                    phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << 7 + ring_12);
                                    ring_12 = (ring_12 + 1) % 6;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else {
                        if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_9 >> 6 & 1);
                            phase_bits_9 = phase_bits_9 ^ 64;
                            int tile_row = tid;
                            float inv_e4m3_max_3 = 0.002232142857f;
                            float scale_floor_3 = 1e-12f;
                            unsigned int scale_word = 0;
                            #pragma unroll 1
                            for (int i_12 = 0; i_12 < 8; i_12++) {
                                float _tmem_load_5[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                                    : "r"(taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + (unsigned int)(i_12 * 32)));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                unsigned int words_3[16];
                                #pragma unroll
                                for (int j_9 = 0; j_9 < 16; j_9++) {
                                    __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_tmem_load_5[2 * j_9], _tmem_load_5[2 * j_9 + 1]));
                                    words_3[j_9] = __as_u32(_bf16x2_12);
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                #pragma unroll
                                for (int j_10 = 0; j_10 < 4; j_10++) {
                                    unsigned int address_20 = d_smem_addr + (unsigned int)(i_12 % 2 * 8192) + (unsigned int)((tile_row * 32 + j_10 * 8) * 2);
                                    address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                        "r"(address_20), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_10)[0])), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_10)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_10)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(words_3 + 4 * j_10)[(0) + 3])));
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + i_12), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(i_12 % 2 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                unsigned int packed_15[8];
                                uint32_t _bf16x2_abs_12;
                                asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_12) : "r"(words_3[0]));
                                unsigned int amax2_6 = _bf16x2_abs_12;
                                #pragma unroll
                                for (int k_12 = 1; k_12 < 16; k_12++) {
                                    uint32_t _bf16x2_abs_13;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_13) : "r"(words_3[k_12]));
                                    uint32_t _bf16x2_max_6;
                                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(amax2_6), "r"(_bf16x2_abs_13));
                                    amax2_6 = _bf16x2_max_6;
                                }
                                uint16_t _bf16_max_6;
                                asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_6) : "h"((uint16_t)(amax2_6 & 65535)), "h"((uint16_t)(amax2_6 >> 16)));
                                float _cvt_f32_bf16_30;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_30) : "h"((uint16_t)(_bf16_max_6)));
                                float amax_6 = _cvt_f32_bf16_30;
                                float _max_16 = max_noftz(amax_6 * inv_e4m3_max_3, scale_floor_3);
                                float scale_6 = _max_16;
                                uint16_t _ue8m0x2_f32_6;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_6) : "f"(scale_6), "f"(scale_6));
                                uint16_t codes_6 = _ue8m0x2_f32_6;
                                unsigned int scale_byte_6 = (unsigned int)codes_6 & 255;
                                unsigned int inv_bits_6 = 254 - scale_byte_6 << 23;
                                float inv_6 = 0.0f;
                                inv_6 = __uint_as_float(inv_bits_6);
                                #pragma unroll
                                for (int i_13 = 0; i_13 < 8; i_13++) {
                                    unsigned int w0_6 = words_3[2 * i_13];
                                    unsigned int w1_6 = words_3[2 * i_13 + 1];
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
                                    packed_15[i_13] = (unsigned int)lo_6 | (unsigned int)hi_6 << 16;
                                }
                                unsigned int scale_byte_0 = scale_byte_6;
                                #pragma unroll
                                for (int m = 0; m < 2; m++) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                        "r"(d_fp8_smem_addr + (unsigned int)(tile_row * 32) + (unsigned int)(m * 16)), "r"(*reinterpret_cast<uint32_t*>(&(packed_15 + 4 * m)[0])), "r"(*reinterpret_cast<uint32_t*>(&(packed_15 + 4 * m)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(packed_15 + 4 * m)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(packed_15 + 4 * m)[(0) + 3])));
                                }
                                scale_word = scale_word | scale_byte_0 << (unsigned int)(i_12 % 4 * 8);
                                if (i_12 % 4 == 3) {
                                    d_sc_smem[i_12 / 4 * 128 + tile_row % 32 * 4 + tile_row / 32] = scale_word;
                                    scale_word = 0;
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&gate_fp8_out_r)), "r"(y_5 * 256 + i_12 * 32), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(d_fp8_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    if (i_12 % 4 == 3) {
                                        tma_store_3d((&gate_sc_r), 0, ((x_5 * 2 + cta_rank_0) * n_blocks + y_5 * 2 + i_12 / 4) * 32, 0, d_sc_smem_addr + (unsigned int)(i_12 / 4 * 512));
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
                    int col_blocks_10 = intermediate / 256;
                    int x_6 = -1;
                    int y_6 = -1;
                    int expert_6 = -1;
                    int k_start_6 = 0;
                    int k_end_6 = 0;
                    int first_6 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                    int offset_8 = 0;
                    int remaining_1 = task_3;
                    #pragma unroll 1
                    for (int index_1 = 0; index_1 < experts; index_1++) {
                        int blocks_1 = counts[index_1] / 256;
                        int _max_17 = ((first_block_1) > (offset_8) ? (first_block_1) : (offset_8));
                        int first_row_3 = _max_17;
                        int _min_44 = ((first_block_1 + mini_size / 256) < (offset_8 + blocks_1) ? (first_block_1 + mini_size / 256) : (offset_8 + blocks_1));
                        int _max_18 = ((0) > (_min_44 - first_row_3) ? (0) : (_min_44 - first_row_3));
                        int rows_4 = _max_18;
                        int tasks_1 = rows_4 * col_blocks_10;
                        if (remaining_1 < tasks_1) {
                            int supergroup_6 = remaining_1 / (rows_4 * 8);
                            int full_cols_6 = col_blocks_10 / 8 * 8;
                            int row_20 = 0;
                            int col_14 = 0;
                            if (remaining_1 < rows_4 * full_cols_6) {
                                row_20 = remaining_1 % (rows_4 * 8) / 8;
                                col_14 = supergroup_6 * 8 + remaining_1 % 8;
                            } else {
                                row_20 = (remaining_1 - rows_4 * full_cols_6) / (col_blocks_10 - full_cols_6);
                                col_14 = full_cols_6 + (remaining_1 - rows_4 * full_cols_6) % (col_blocks_10 - full_cols_6);
                            }
                            if ((supergroup_6 & 1) != 0) {
                                row_20 = rows_4 - row_20 - 1;
                            }
                            x_6 = first_row_3 + row_20 - macro_1 * (macro_size / 256);
                            y_6 = col_14;
                            expert_6 = index_1;
                            break;
                        }
                        remaining_1 = remaining_1 - tasks_1;
                        offset_8 = offset_8 + blocks_1;
                    }
                    unsigned int phase_bits_10 = gemm_phase;
                    int global_mini_6 = macro_1 * (macro_size / mini_size) + mini_1;
                    int macro_rows_6 = macro_1 * (macro_size / 256);
                    int iterations_6 = hidden / 128;
                    int k_blocks_1 = hidden / 128;
                    int n_blocks_1 = intermediate / 128;
                    if (expert_6 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_45 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                    int _max_19 = ((0) > (_min_45) ? (0) : (_min_45));
                                    int mini_rows_8 = _max_19;
                                    int required_7 = (mini_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_9 = 1;
                                    if (enabled_value_9 != 0) {
                                        int32_t _relaxed_ld_24;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_24) : "l"(replay_x + global_mini_6) : "memory");
                                        int value_12 = _relaxed_ld_24;
                                        while (value_12 < required_7) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_25;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_25) : "l"(replay_x + global_mini_6) : "memory");
                                            value_12 = _relaxed_ld_25;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                int ring_13 = 0;
                                #pragma unroll 1
                                for (int idx_13 = 0; idx_13 < iterations_6; idx_13++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_13) * 8, phase_bits_10 >> (unsigned int)(16 + ring_13) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_fp8_smem_addr + (unsigned int)(ring_13 * 16384)), "l"((&x_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(idx_13), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_13) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_fp8_smem_addr + (unsigned int)(ring_13 * 16384)), "l"((&wu_r)), "r"(0), "r"(y_6 * 256 + cta_rank_0 * 128), "r"(idx_13), "r"(expert_6), "r"(0),
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
                                        int _min_46 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                        int _max_20 = ((0) > (_min_46) ? (0) : (_min_46));
                                        int mini_rows_9 = _max_20;
                                        int required_8 = (mini_rows_9 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_10 = 1;
                                        if (enabled_value_10 != 0) {
                                            int32_t _relaxed_ld_26;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_26) : "l"(replay_x + global_mini_6) : "memory");
                                            int value_13 = _relaxed_ld_26;
                                            while (value_13 < required_8) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_27;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_27) : "l"(replay_x + global_mini_6) : "memory");
                                                value_13 = _relaxed_ld_27;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_14 = 0;
                                    #pragma unroll 1
                                    for (int idx_14 = 0; idx_14 < iterations_6; idx_14++) {
                                        mbarrier_wait(scales_finished_addr + (ring_14) * 8, phase_bits_10 >> (unsigned int)(23 + ring_14) & 1);
                                        int a_tile_1 = (x_6 * 2 + cta_rank_0) * k_blocks_1 + idx_14;
                                        int b_tile_1 = (expert_6 * n_blocks_1 + y_6 * 2 + cta_rank_0) * k_blocks_1 + idx_14;
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(a_sc_smem_addr + (unsigned int)(ring_14 * 512)), "l"((&x_sc_r)), "r"(0), "r"(a_tile_1 * 32), "r"(0),
                                               "r"(((scales_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(b_sc_smem_addr + (unsigned int)(ring_14 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_sc_r)), "r"(0), "r"(b_tile_1 * 32), "r"(0),
                                               "r"(((scales_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                        phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 23 + ring_14);
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
                                        mbarrier_wait(scales_arrived_addr + (ring_15) * 8, phase_bits_10 >> (unsigned int)(7 + ring_15) & 1);
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_15 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_15 * 512)));
                                        tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_15 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_15 * 1024)));
                                        tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_15 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_15 * 1024) + 512)));
                                        tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_15) * 8, (uint16_t)(3));
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_15) * 8, 65536);
                                        mbarrier_wait(gemm_arrived_addr + (ring_15) * 8, phase_bits_10 >> (unsigned int)ring_15 & 1);
                                        int _mma_a_lo_6 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_15) * 1024;
                                        int _mma_b_lo_6 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_15) * 1024;
                                        {
                                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_15 * 4, tmem_sf_b + ring_15 * 8, ((idx_15 == 0) ? 0 : 1));
                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_15 * 4, tmem_sf_b + ring_15 * 8, 1);
                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_15 * 4, tmem_sf_b + ring_15 * 8, 1);
                                            tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_15 * 4, tmem_sf_b + ring_15 * 8, 1);
                                        }
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_15) * 8, (uint16_t)(3));
                                        phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << ring_15);
                                        phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 7 + ring_15);
                                        ring_15 = (ring_15 + 1) % 6;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else {
                            if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_10 >> 6 & 1);
                                phase_bits_10 = phase_bits_10 ^ 64;
                                int tile_row_1 = tid;
                                float inv_e4m3_max_4 = 0.002232142857f;
                                float scale_floor_4 = 1e-12f;
                                unsigned int scale_word_1 = 0;
                                #pragma unroll 1
                                for (int i_14 = 0; i_14 < 8; i_14++) {
                                    float _tmem_load_6[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                                        : "r"(taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + (unsigned int)(i_14 * 32)));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    unsigned int words_4[16];
                                    #pragma unroll
                                    for (int j_11 = 0; j_11 < 16; j_11++) {
                                        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_tmem_load_6[2 * j_11], _tmem_load_6[2 * j_11 + 1]));
                                        words_4[j_11] = __as_u32(_bf16x2_13);
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    #pragma unroll
                                    for (int j_12 = 0; j_12 < 4; j_12++) {
                                        unsigned int address_21 = d_smem_addr + (unsigned int)(i_14 % 2 * 8192) + (unsigned int)((tile_row_1 * 32 + j_12 * 8) * 2);
                                        address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                            "r"(address_21), "r"(*reinterpret_cast<uint32_t*>(&(words_4 + 4 * j_12)[0])), "r"(*reinterpret_cast<uint32_t*>(&(words_4 + 4 * j_12)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(words_4 + 4 * j_12)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(words_4 + 4 * j_12)[(0) + 3])));
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 8 + i_14), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(i_14 % 2 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    unsigned int packed_16[8];
                                    uint32_t _bf16x2_abs_14;
                                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_14) : "r"(words_4[0]));
                                    unsigned int amax2_7 = _bf16x2_abs_14;
                                    #pragma unroll
                                    for (int k_13 = 1; k_13 < 16; k_13++) {
                                        uint32_t _bf16x2_abs_15;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_15) : "r"(words_4[k_13]));
                                        uint32_t _bf16x2_max_7;
                                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(amax2_7), "r"(_bf16x2_abs_15));
                                        amax2_7 = _bf16x2_max_7;
                                    }
                                    uint16_t _bf16_max_7;
                                    asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_7) : "h"((uint16_t)(amax2_7 & 65535)), "h"((uint16_t)(amax2_7 >> 16)));
                                    float _cvt_f32_bf16_35;
                                    asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_35) : "h"((uint16_t)(_bf16_max_7)));
                                    float amax_7 = _cvt_f32_bf16_35;
                                    float _max_21 = max_noftz(amax_7 * inv_e4m3_max_4, scale_floor_4);
                                    float scale_7 = _max_21;
                                    uint16_t _ue8m0x2_f32_7;
                                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_7) : "f"(scale_7), "f"(scale_7));
                                    uint16_t codes_7 = _ue8m0x2_f32_7;
                                    unsigned int scale_byte_7 = (unsigned int)codes_7 & 255;
                                    unsigned int inv_bits_7 = 254 - scale_byte_7 << 23;
                                    float inv_7 = 0.0f;
                                    inv_7 = __uint_as_float(inv_bits_7);
                                    #pragma unroll
                                    for (int i_15 = 0; i_15 < 8; i_15++) {
                                        unsigned int w0_7 = words_4[2 * i_15];
                                        unsigned int w1_7 = words_4[2 * i_15 + 1];
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
                                        packed_16[i_15] = (unsigned int)lo_7 | (unsigned int)hi_7 << 16;
                                    }
                                    unsigned int scale_byte_0_1 = scale_byte_7;
                                    #pragma unroll
                                    for (int m_1 = 0; m_1 < 2; m_1++) {
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                            "r"(d_fp8_smem_addr + (unsigned int)(tile_row_1 * 32) + (unsigned int)(m_1 * 16)), "r"(*reinterpret_cast<uint32_t*>(&(packed_16 + 4 * m_1)[0])), "r"(*reinterpret_cast<uint32_t*>(&(packed_16 + 4 * m_1)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(packed_16 + 4 * m_1)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(packed_16 + 4 * m_1)[(0) + 3])));
                                    }
                                    scale_word_1 = scale_word_1 | scale_byte_0_1 << (unsigned int)(i_14 % 4 * 8);
                                    if (i_14 % 4 == 3) {
                                        d_sc_smem[i_14 / 4 * 128 + tile_row_1 % 32 * 4 + tile_row_1 / 32] = scale_word_1;
                                        scale_word_1 = 0;
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2}], [%3], %4;"
                                            :: "l"((&up_fp8_out_r)), "r"(y_6 * 256 + i_14 * 32), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(d_fp8_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                        if (i_14 % 4 == 3) {
                                            tma_store_3d((&up_sc_r), 0, ((x_6 * 2 + cta_rank_0) * n_blocks_1 + y_6 * 2 + i_14 / 4) * 32, 0, d_sc_smem_addr + (unsigned int)(i_14 / 4 * 512));
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
                    int num_tiles_1 = (tokens + 127) / 128 * col_blocks_11;
                    int macro_row_offset_1 = macro_1 * (macro_size / 128);
                    int macro_row_blocks_1 = macro_size / 128;
                    int global_mini_7 = macro_1 * (macro_size / mini_size) + mini_1;
                    int mini_tiles = mini_size / 128 * col_blocks_11;
                    int first_tile_2 = task_3 * 6 + cta_rank_0 * 3 + global_mini_7 * mini_tiles;
                    int _min_47 = ((num_tiles_1) < ((global_mini_7 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_7 + 1) * mini_tiles));
                    int tile_end_1 = _min_47;
                    if (first_tile_2 < tile_end_1) {
                        int first_row_4 = first_tile_2 / col_blocks_11;
                        int first_col_2 = first_tile_2 % col_blocks_11;
                        if (tid == 0) {
                            #pragma unroll
                            for (int stage_7 = 0; stage_7 < 3; stage_7++) {
                                if (tile_end_1 > first_tile_2 + stage_7) {
                                    int row_21 = first_row_4;
                                    int col_15 = first_col_2 + stage_7;
                                    if (col_15 >= col_blocks_11) {
                                        row_21 = row_21 + 1;
                                        col_15 = col_15 - col_blocks_11;
                                    }
                                    mbarrier_arrive_expect_tx(replay_arrived_addr + (stage_7) * 8, 65536);
                                    int parent_1 = row_21 / 2 * (intermediate / 256) + col_15 / 2;
                                    int32_t _relaxed_ld_28;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_28) : "l"(replay_gu + parent_1) : "memory");
                                    int value_14 = _relaxed_ld_28;
                                    while (value_14 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_29;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_29) : "l"(replay_gu + parent_1) : "memory");
                                        value_14 = _relaxed_ld_29;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3}], [%4];"
                                        :: "r"(gate_flat_addr + (unsigned int)(stage_7 * 32768)), "l"((&gate_rows_r)), "r"(col_15 * 128), "r"((row_21 - macro_row_offset_1) * 128), "r"(replay_arrived_addr + (stage_7) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3}], [%4];"
                                        :: "r"(up_flat_addr + (unsigned int)(stage_7 * 32768)), "l"((&up_rows_r)), "r"(col_15 * 128), "r"((row_21 - macro_row_offset_1) * 128), "r"(replay_arrived_addr + (stage_7) * 8) : "memory");
                                }
                            }
                        }
                        float inv_e4m3_max_5 = 0.002232142857f;
                        float scale_floor_5 = 1e-12f;
                        #pragma unroll 1
                        for (int stage_8 = 0; stage_8 < 3; stage_8++) {
                            if (tile_end_1 > first_tile_2 + stage_8) {
                                mbarrier_wait(replay_arrived_addr + (stage_8) * 8, phase_bits_11 >> (unsigned int)stage_8 & 1);
                                phase_bits_11 = phase_bits_11 ^ (unsigned int)(1 << stage_8);
                                int row_22 = first_row_4;
                                int col_16 = first_col_2 + stage_8;
                                if (col_16 >= col_blocks_11) {
                                    row_22 = row_22 + 1;
                                    col_16 = col_16 - col_blocks_11;
                                }
                                if (swiglu_clamped != 0) {
                                    #pragma unroll
                                    for (int step = 0; step < 32; step++) {
                                        int pair_15 = step * 256 + tid;
                                        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(gate_words[stage_8 * 8192 + pair_15]));
                                        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(up_words[stage_8 * 8192 + pair_15]));
                                        float gx = _cvt_f32_6.x;
                                        float gy = _cvt_f32_6.y;
                                        float ux = _cvt_f32_7.x;
                                        float uy = _cvt_f32_7.y;
                                        float _min_48 = fminf(gx, swiglu_limit);
                                        gx = _min_48;
                                        float _min_49 = fminf(gy, swiglu_limit);
                                        gy = _min_49;
                                        float _max_22 = max_noftz(ux, -swiglu_limit);
                                        float _min_50 = fminf(_max_22, swiglu_limit);
                                        ux = _min_50;
                                        float _max_23 = max_noftz(uy, -swiglu_limit);
                                        float _min_51 = fminf(_max_23, swiglu_limit);
                                        uy = _min_51;
                                        float _exp_2 = expf(gx * -1.0f);
                                        float hx = gx / (_exp_2 + 1.0f) * ux;
                                        float _exp_3 = expf(gy * -1.0f);
                                        float hy = gy / (_exp_3 + 1.0f) * uy;
                                        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(hx, hy));
                                        hidden_words[pair_15] = __as_u32(_bf16x2_14);
                                    }
                                } else {
                                    #pragma unroll
                                    for (int step_1 = 0; step_1 < 32; step_1++) {
                                        int pair_16 = step_1 * 256 + tid;
                                        float2 _cvt_f32_8 = __bfloat1622float2(__as_bf16x2(gate_words[stage_8 * 8192 + pair_16]));
                                        float2 _cvt_f32_9 = __bfloat1622float2(__as_bf16x2(up_words[stage_8 * 8192 + pair_16]));
                                        float gx_1 = _cvt_f32_8.x;
                                        float gy_1 = _cvt_f32_8.y;
                                        float ux_1 = _cvt_f32_9.x;
                                        float uy_1 = _cvt_f32_9.y;
                                        float _exp_4 = expf(gx_1 * -1.0f);
                                        float hx_1 = gx_1 / (_exp_4 + 1.0f) * ux_1;
                                        float _exp_5 = expf(gy_1 * -1.0f);
                                        float hy_1 = gy_1 / (_exp_5 + 1.0f) * uy_1;
                                        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(hx_1, hy_1));
                                        hidden_words[pair_16] = __as_u32(_bf16x2_15);
                                    }
                                }
                                __syncthreads();
                                if (tid < 128) {
                                    int t_row_3 = tid % 64 * 2 + tid / 64;
                                    int rotation_6 = tid / 8;
                                    unsigned int t_scale_word_3 = 0;
                                    #pragma unroll 1
                                    for (int j_13 = 0; j_13 < 4; j_13++) {
                                        int k_block_3 = (j_13 + rotation_6) % 4;
                                        unsigned int t_words_3[16];
                                        #pragma unroll
                                        for (int k_14 = 0; k_14 < 16; k_14++) {
                                            int src_row_3 = k_block_3 * 32 + (tid * 4 + k_14 * 2) % 32;
                                            unsigned int lo_8 = (unsigned int)hidden_halves[src_row_3 * 128 + t_row_3];
                                            unsigned int hi_8 = (unsigned int)hidden_halves[(src_row_3 + 1) * 128 + t_row_3];
                                            uint32_t _prmt_b32_0;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_0) : "r"(lo_8), "r"(hi_8));
                                            t_words_3[k_14] = _prmt_b32_0;
                                        }
                                        unsigned int t_packed_3[8];
                                        uint32_t _bf16x2_abs_16;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(t_words_3[0]));
                                        unsigned int amax2_8 = _bf16x2_abs_16;
                                        #pragma unroll
                                        for (int k_15 = 1; k_15 < 16; k_15++) {
                                            uint32_t _bf16x2_abs_17;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(t_words_3[k_15]));
                                            uint32_t _bf16x2_max_8;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_8) : "r"(amax2_8), "r"(_bf16x2_abs_17));
                                            amax2_8 = _bf16x2_max_8;
                                        }
                                        uint16_t _bf16_max_8;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_8) : "h"((uint16_t)(amax2_8 & 65535)), "h"((uint16_t)(amax2_8 >> 16)));
                                        float _cvt_f32_bf16_40;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_40) : "h"((uint16_t)(_bf16_max_8)));
                                        float amax_8 = _cvt_f32_bf16_40;
                                        float _max_24 = max_noftz(amax_8 * inv_e4m3_max_5, scale_floor_5);
                                        float scale_8 = _max_24;
                                        uint16_t _ue8m0x2_f32_8;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_8) : "f"(scale_8), "f"(scale_8));
                                        uint16_t codes_8 = _ue8m0x2_f32_8;
                                        unsigned int scale_byte_8 = (unsigned int)codes_8 & 255;
                                        unsigned int inv_bits_8 = 254 - scale_byte_8 << 23;
                                        float inv_8 = 0.0f;
                                        inv_8 = __uint_as_float(inv_bits_8);
                                        #pragma unroll
                                        for (int i_16 = 0; i_16 < 8; i_16++) {
                                            unsigned int w0_8 = t_words_3[2 * i_16];
                                            unsigned int w1_8 = t_words_3[2 * i_16 + 1];
                                            float _cvt_f32_bf16_41;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_41) : "h"((uint16_t)(w0_8 & 65535)));
                                            float v0_11 = _cvt_f32_bf16_41;
                                            float _cvt_f32_bf16_42;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_42) : "h"((uint16_t)(w0_8 >> 16)));
                                            float v1_11 = _cvt_f32_bf16_42;
                                            float _cvt_f32_bf16_43;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_43) : "h"((uint16_t)(w1_8 & 65535)));
                                            float v2_8 = _cvt_f32_bf16_43;
                                            float _cvt_f32_bf16_44;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_44) : "h"((uint16_t)(w1_8 >> 16)));
                                            float v3_8 = _cvt_f32_bf16_44;
                                            uint16_t _e4m3x2_f32_16;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_16) : "f"(v1_11 * inv_8), "f"(v0_11 * inv_8));
                                            uint16_t lo_9 = _e4m3x2_f32_16;
                                            uint16_t _e4m3x2_f32_17;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_17) : "f"(v3_8 * inv_8), "f"(v2_8 * inv_8));
                                            uint16_t hi_9 = _e4m3x2_f32_17;
                                            t_packed_3[i_16] = (unsigned int)lo_9 | (unsigned int)hi_9 << 16;
                                        }
                                        unsigned int t_scale_byte_3 = scale_byte_8;
                                        #pragma unroll
                                        for (int i_17 = 0; i_17 < 8; i_17++) {
                                            int t_col_3 = k_block_3 * 32 + (tid * 4 + i_17 * 4) % 32;
                                            up_words[stage_8 * 8192 + t_row_3 * 32 + t_col_3 / 4] = t_packed_3[i_17];
                                        }
                                        t_scale_word_3 = t_scale_word_3 | t_scale_byte_3 << (unsigned int)(k_block_3 * 8);
                                    }
                                    up_words[stage_8 * 8192 + 4096 + t_row_3 % 32 * 4 + t_row_3 / 32] = t_scale_word_3;
                                    int n_row_3 = tid;
                                    int rotation_0 = tid / 8;
                                    unsigned int words_5[64];
                                    #pragma unroll
                                    for (int j_14 = 0; j_14 < 4; j_14++) {
                                        int k_block_j_3 = (j_14 + rotation_0) % 4;
                                        #pragma unroll
                                        for (int k_16 = 0; k_16 < 16; k_16++) {
                                            int src_col_3 = k_block_j_3 * 32 + (tid * 4 + k_16 * 2) % 32;
                                            words_5[j_14 * 16 + k_16] = hidden_words[n_row_3 * 64 + src_col_3 / 2];
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    unsigned int n_scale_word_3 = 0;
                                    #pragma unroll
                                    for (int j_15 = 0; j_15 < 4; j_15++) {
                                        int k_block_n_3 = (j_15 + rotation_0) % 4;
                                        unsigned int n_packed_3[8];
                                        uint32_t _bf16x2_abs_18;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(words_5[j_15 * 16]));
                                        unsigned int amax2_9 = _bf16x2_abs_18;
                                        #pragma unroll
                                        for (int k_17 = 1; k_17 < 16; k_17++) {
                                            uint32_t _bf16x2_abs_19;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(words_5[j_15 * 16 + k_17]));
                                            uint32_t _bf16x2_max_9;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_9) : "r"(amax2_9), "r"(_bf16x2_abs_19));
                                            amax2_9 = _bf16x2_max_9;
                                        }
                                        uint16_t _bf16_max_9;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_9) : "h"((uint16_t)(amax2_9 & 65535)), "h"((uint16_t)(amax2_9 >> 16)));
                                        float _cvt_f32_bf16_45;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_45) : "h"((uint16_t)(_bf16_max_9)));
                                        float amax_9 = _cvt_f32_bf16_45;
                                        float _max_25 = max_noftz(amax_9 * inv_e4m3_max_5, scale_floor_5);
                                        float scale_9 = _max_25;
                                        uint16_t _ue8m0x2_f32_9;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_9) : "f"(scale_9), "f"(scale_9));
                                        uint16_t codes_9 = _ue8m0x2_f32_9;
                                        unsigned int scale_byte_9 = (unsigned int)codes_9 & 255;
                                        unsigned int inv_bits_9 = 254 - scale_byte_9 << 23;
                                        float inv_9 = 0.0f;
                                        inv_9 = __uint_as_float(inv_bits_9);
                                        #pragma unroll
                                        for (int i_18 = 0; i_18 < 8; i_18++) {
                                            unsigned int w0_9 = words_5[j_15 * 16 + 2 * i_18];
                                            unsigned int w1_9 = words_5[j_15 * 16 + 2 * i_18 + 1];
                                            float _cvt_f32_bf16_46;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_46) : "h"((uint16_t)(w0_9 & 65535)));
                                            float v0_12 = _cvt_f32_bf16_46;
                                            float _cvt_f32_bf16_47;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_47) : "h"((uint16_t)(w0_9 >> 16)));
                                            float v1_12 = _cvt_f32_bf16_47;
                                            float _cvt_f32_bf16_48;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_48) : "h"((uint16_t)(w1_9 & 65535)));
                                            float v2_9 = _cvt_f32_bf16_48;
                                            float _cvt_f32_bf16_49;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_49) : "h"((uint16_t)(w1_9 >> 16)));
                                            float v3_9 = _cvt_f32_bf16_49;
                                            uint16_t _e4m3x2_f32_18;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_18) : "f"(v1_12 * inv_9), "f"(v0_12 * inv_9));
                                            uint16_t lo_10 = _e4m3x2_f32_18;
                                            uint16_t _e4m3x2_f32_19;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_19) : "f"(v3_9 * inv_9), "f"(v2_9 * inv_9));
                                            uint16_t hi_10 = _e4m3x2_f32_19;
                                            n_packed_3[i_18] = (unsigned int)lo_10 | (unsigned int)hi_10 << 16;
                                        }
                                        unsigned int n_scale_byte_3 = scale_byte_9;
                                        #pragma unroll
                                        for (int i_19 = 0; i_19 < 8; i_19++) {
                                            int n_col_3 = k_block_n_3 * 32 + (tid * 4 + i_19 * 4) % 32;
                                            gate_words[stage_8 * 8192 + n_row_3 * 32 + n_col_3 / 4] = n_packed_3[i_19];
                                        }
                                        n_scale_word_3 = n_scale_word_3 | n_scale_byte_3 << (unsigned int)(k_block_n_3 * 8);
                                    }
                                    gate_words[stage_8 * 8192 + 4096 + n_row_3 % 32 * 4 + n_row_3 / 32] = n_scale_word_3;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    int local_row_3 = row_22 - macro_row_offset_1;
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_2d((&h_fp8_r), col_16 * 128, local_row_3 * 128, gate_flat_addr + (unsigned int)(stage_8 * 32768));
                                    tma_store_3d((&h_sc_r), 0, (local_row_3 * col_blocks_11 + col_16) * 32, 0, gate_flat_addr + (unsigned int)(stage_8 * 32768) + 16384);
                                    tma_store_2d((&h_fp8_t_r), local_row_3 * 128, col_16 * 128, up_flat_addr + (unsigned int)(stage_8 * 32768));
                                    tma_store_3d((&h_sc_t_r), 0, (col_16 * macro_row_blocks_1 + local_row_3) * 32, 0, up_flat_addr + (unsigned int)(stage_8 * 32768) + 16384);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_9 = 0; stage_9 < 3; stage_9++) {
                                if (tile_end_1 > first_tile_2 + stage_9) {
                                    int row_23 = first_row_4;
                                    if (col_blocks_11 <= first_col_2 + stage_9) {
                                        row_23 = row_23 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_23 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                    }
                    replay_phase = phase_bits_11;
                } else {
                    if (kind == 0) {
                        int col_blocks_12 = intermediate / 256;
                        int x_7 = -1;
                        int y_7 = -1;
                        int expert_7 = -1;
                        int k_start_7 = 0;
                        int k_end_7 = 0;
                        int first_7 = 0;
                        int first_block_2 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                        int offset_9 = 0;
                        int remaining_2 = task_3;
                        #pragma unroll 1
                        for (int index_2 = 0; index_2 < experts; index_2++) {
                            int blocks_2 = counts[index_2] / 256;
                            int _max_26 = ((first_block_2) > (offset_9) ? (first_block_2) : (offset_9));
                            int first_row_5 = _max_26;
                            int _min_52 = ((first_block_2 + mini_size / 256) < (offset_9 + blocks_2) ? (first_block_2 + mini_size / 256) : (offset_9 + blocks_2));
                            int _max_27 = ((0) > (_min_52 - first_row_5) ? (0) : (_min_52 - first_row_5));
                            int rows_5 = _max_27;
                            int tasks_2 = rows_5 * col_blocks_12;
                            if (remaining_2 < tasks_2) {
                                int supergroup_7 = remaining_2 / (rows_5 * 8);
                                int full_cols_7 = col_blocks_12 / 8 * 8;
                                int row_24 = 0;
                                int col_17 = 0;
                                if (remaining_2 < rows_5 * full_cols_7) {
                                    row_24 = remaining_2 % (rows_5 * 8) / 8;
                                    col_17 = supergroup_7 * 8 + remaining_2 % 8;
                                } else {
                                    row_24 = (remaining_2 - rows_5 * full_cols_7) / (col_blocks_12 - full_cols_7);
                                    col_17 = full_cols_7 + (remaining_2 - rows_5 * full_cols_7) % (col_blocks_12 - full_cols_7);
                                }
                                if ((supergroup_7 & 1) != 0) {
                                    row_24 = rows_5 - row_24 - 1;
                                }
                                x_7 = first_row_5 + row_24 - macro_1 * (macro_size / 256);
                                y_7 = col_17;
                                expert_7 = index_2;
                                break;
                            }
                            remaining_2 = remaining_2 - tasks_2;
                            offset_9 = offset_9 + blocks_2;
                        }
                        unsigned int phase_bits_12 = gemm_phase;
                        int global_mini_8 = macro_1 * (macro_size / mini_size) + mini_1;
                        int macro_rows_7 = macro_1 * (macro_size / 256);
                        int iterations_7 = hidden / 128;
                        int k_blocks_2 = hidden / 128;
                        int n_blocks_2 = intermediate / 128;
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
                                        int _min_53 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                        int _max_28 = ((0) > (_min_53) ? (0) : (_min_53));
                                        int mini_rows_10 = _max_28;
                                        int required_9 = (mini_rows_10 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_13 = 1;
                                        if (enabled_value_13 != 0) {
                                            int32_t _relaxed_ld_30;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_30) : "l"(dy_ready + global_mini_8) : "memory");
                                            int value_15 = _relaxed_ld_30;
                                            while (value_15 < required_9) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_31;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_31) : "l"(dy_ready + global_mini_8) : "memory");
                                                value_15 = _relaxed_ld_31;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_16 = 0;
                                    #pragma unroll 1
                                    for (int idx_16 = 0; idx_16 < iterations_7; idx_16++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_16) * 8, phase_bits_12 >> (unsigned int)(16 + ring_16) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_fp8_smem_addr + (unsigned int)(ring_16 * 16384)), "l"((&dy_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(idx_16), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_fp8_smem_addr + (unsigned int)(ring_16 * 16384)), "l"((&wd_t_r)), "r"(0), "r"(y_7 * 256 + cta_rank_0 * 128), "r"(idx_16), "r"(expert_7), "r"(0),
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
                                            int _min_54 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                            int _max_29 = ((0) > (_min_54) ? (0) : (_min_54));
                                            int mini_rows_11 = _max_29;
                                            int required_10 = (mini_rows_11 + 127) / 128 * ((hidden + 511) / 512);
                                            bool enabled_value_14 = 1;
                                            if (enabled_value_14 != 0) {
                                                int32_t _relaxed_ld_32;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_32) : "l"(dy_ready + global_mini_8) : "memory");
                                                int value_16 = _relaxed_ld_32;
                                                while (value_16 < required_10) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_33;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_33) : "l"(dy_ready + global_mini_8) : "memory");
                                                    value_16 = _relaxed_ld_33;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                        int ring_17 = 0;
                                        #pragma unroll 1
                                        for (int idx_17 = 0; idx_17 < iterations_7; idx_17++) {
                                            mbarrier_wait(scales_finished_addr + (ring_17) * 8, phase_bits_12 >> (unsigned int)(23 + ring_17) & 1);
                                            int a_tile_2 = (x_7 * 2 + cta_rank_0) * k_blocks_2 + idx_17;
                                            int b_tile_2 = (expert_7 * n_blocks_2 + y_7 * 2 + cta_rank_0) * k_blocks_2 + idx_17;
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(a_sc_smem_addr + (unsigned int)(ring_17 * 512)), "l"((&dy_sc_r)), "r"(0), "r"(a_tile_2 * 32), "r"(0),
                                                   "r"(((scales_arrived_addr + (ring_17) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(b_sc_smem_addr + (unsigned int)(ring_17 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wd_t_sc_r)), "r"(0), "r"(b_tile_2 * 32), "r"(0),
                                                   "r"(((scales_arrived_addr + (ring_17) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                            phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << 23 + ring_17);
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
                                            mbarrier_wait(scales_arrived_addr + (ring_18) * 8, phase_bits_12 >> (unsigned int)(7 + ring_18) & 1);
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_18 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_18 * 512)));
                                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_18 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_18 * 1024)));
                                            tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_18 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_18 * 1024) + 512)));
                                            tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_18) * 8, (uint16_t)(3));
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_18) * 8, 65536);
                                            mbarrier_wait(gemm_arrived_addr + (ring_18) * 8, phase_bits_12 >> (unsigned int)ring_18 & 1);
                                            int _mma_a_lo_7 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_18) * 1024;
                                            int _mma_b_lo_7 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_18) * 1024;
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                    (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_18 * 4, tmem_sf_b + ring_18 * 8, ((idx_18 == 0) ? 0 : 1));
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                    (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_18 * 4, tmem_sf_b + ring_18 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                    (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_18 * 4, tmem_sf_b + ring_18 * 8, 1);
                                                tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                    (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_18 * 4, tmem_sf_b + ring_18 * 8, 1);
                                            }
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_18) * 8, (uint16_t)(3));
                                            phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << ring_18);
                                            phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << 7 + ring_18);
                                            ring_18 = (ring_18 + 1) % 6;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else {
                                if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_12 >> 6 & 1);
                                    phase_bits_12 = phase_bits_12 ^ 64;
                                    unsigned int packed_17[128];
                                    #pragma unroll
                                    for (int chunk_10 = 0; chunk_10 < 8; chunk_10++) {
                                        #pragma unroll
                                        for (int half_13 = 0; half_13 < 2; half_13++) {
                                            unsigned int address_22 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_13 * 16 << 16) + (unsigned int)(chunk_10 * 32);
                                            float _tmem_load_7[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                                                : "r"(address_22));
                                            #pragma unroll
                                            for (int pair_17 = 0; pair_17 < 8; pair_17++) {
                                                __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_tmem_load_7[pair_17 * 2], _tmem_load_7[pair_17 * 2 + 1]));
                                                packed_17[chunk_10 * 16 + half_13 * 8 + pair_17] = __as_u32(_bf16x2_16);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        int previous_offset_8 = (macro_1 + 1) * macro_size;
                                        int output_row_5 = x_7 * 256 + cta_rank_0 * 128;
                                        int _min_55 = ((macro_size) < (tokens - previous_offset_8) ? (macro_size) : (tokens - previous_offset_8));
                                        if (output_row_5 < _min_55) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_11 = 0; chunk_11 < 8; chunk_11++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_7 = tid / 32;
                                        int lane_11 = tid % 32;
                                        #pragma unroll
                                        for (int half_14 = 0; half_14 < 2; half_14++) {
                                            #pragma unroll
                                            for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                                int row_25 = warp_0_7 * 32 + half_14 * 16 + lane_11 % 16;
                                                int col_18 = col_tile_5 * 16 + lane_11 / 16 * 8;
                                                unsigned int address_23 = d_smem_addr + (unsigned int)(chunk_11 % 3 * 8192) + (unsigned int)((row_25 * 32 + col_18) * 2);
                                                address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                                int offset_0 = chunk_11 * 16 + half_14 * 8 + col_tile_5 * 4;
                                                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(address_23);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[offset_0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[offset_0 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[offset_0 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_17[offset_0 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dh_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(y_7 * 8 + chunk_11), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_11 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                bool enabled_value_15 = 1;
                                                if (enabled_value_15 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_0 + (macro_rows_7 + x_7) * (intermediate / 256) + y_7))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                                bool enabled_value_0 = macros > 1;
                                                if (enabled_value_0 != 0) {
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
                        int num_tiles_2 = (tokens + 127) / 128 * col_blocks_13;
                        int macro_row_offset_2 = macro_1 * (macro_size / 128);
                        int macro_row_blocks_2 = macro_size / 128;
                        int global_mini_9 = macro_1 * (macro_size / mini_size) + mini_1;
                        int mini_tiles_1 = mini_size / 128 * col_blocks_13;
                        int first_tile_3 = task_3 * 4 + cta_rank_0 * 2 + global_mini_9 * mini_tiles_1;
                        int _min_56 = ((num_tiles_2) < ((global_mini_9 + 1) * mini_tiles_1) ? (num_tiles_2) : ((global_mini_9 + 1) * mini_tiles_1));
                        int tile_end_2 = _min_56;
                        if (first_tile_3 < tile_end_2) {
                            int first_row_6 = first_tile_3 / col_blocks_13;
                            int first_col_3 = first_tile_3 % col_blocks_13;
                            if (tid == 0) {
                                #pragma unroll
                                for (int stage_10 = 0; stage_10 < 2; stage_10++) {
                                    if (tile_end_2 > first_tile_3 + stage_10) {
                                        int row_26 = first_row_6;
                                        int col_19 = first_col_3 + stage_10;
                                        if (col_19 >= col_blocks_13) {
                                            row_26 = row_26 + 1;
                                            col_19 = col_19 - col_blocks_13;
                                        }
                                        mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_10) * 8, 66560);
                                        int parent_2 = row_26 / 2 * (intermediate / 256) + col_19 / 2;
                                        int32_t _relaxed_ld_34;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_34) : "l"(dh_ready + (shared_down_0 + parent_2)) : "memory");
                                        int value_17 = _relaxed_ld_34;
                                        while (value_17 < 2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_35;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_35) : "l"(dh_ready + (shared_down_0 + parent_2)) : "memory");
                                            value_17 = _relaxed_ld_35;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                        bool enabled_value_16 = macro_1 > 0;
                                        if (enabled_value_16 != 0) {
                                            int32_t _relaxed_ld_36;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_36) : "l"(replay_gu + parent_2) : "memory");
                                            int value_0 = _relaxed_ld_36;
                                            while (value_0 < 4) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_37;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_37) : "l"(replay_gu + parent_2) : "memory");
                                                value_0 = _relaxed_ld_37;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int local_block = row_26 - macro_row_offset_2;
                                        int scale_tile = (local_block * col_blocks_13 + col_19) * 32;
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3}], [%4];"
                                            :: "r"(q_flat_addr + (unsigned int)(stage_10 * 32768)), "l"((&dh_rows_r)), "r"(col_19 * 128), "r"(local_block * 128), "r"(swiglu_arrived_addr + (stage_10) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3}], [%4];"
                                            :: "r"(q_flat_addr + 65536 + (unsigned int)(stage_10 * 16384)), "l"((&gate_fp8_r)), "r"(col_19 * 128), "r"(local_block * 128), "r"(swiglu_arrived_addr + (stage_10) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3}], [%4];"
                                            :: "r"(q_flat_addr + 98304 + (unsigned int)(stage_10 * 16384)), "l"((&up_fp8_r)), "r"(col_19 * 128), "r"(local_block * 128), "r"(swiglu_arrived_addr + (stage_10) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4}], [%5];"
                                            :: "r"(q_flat_addr + 131072 + (unsigned int)(stage_10 * 512)), "l"((&gate_sc_r)), "r"(0), "r"(scale_tile), "r"(0), "r"(swiglu_arrived_addr + (stage_10) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4}], [%5];"
                                            :: "r"(q_flat_addr + 132096 + (unsigned int)(stage_10 * 512)), "l"((&up_sc_r)), "r"(0), "r"(scale_tile), "r"(0), "r"(swiglu_arrived_addr + (stage_10) * 8) : "memory");
                                    }
                                }
                            }
                            float inv_e4m3_max_6 = 0.002232142857f;
                            float scale_floor_6 = 1e-12f;
                            int tile_row_2 = tid % 128;
                            int half_15 = tid / 128;
                            int k_pair = tile_row_2 >> 1 & 1 ^ half_15;
                            int scale_index = tile_row_2 % 32 * 4 + tile_row_2 / 32;
                            #pragma unroll
                            for (int stage_11 = 0; stage_11 < 2; stage_11++) {
                                if (tile_end_2 > first_tile_3 + stage_11) {
                                    int row_27 = first_row_6;
                                    int col_20 = first_col_3 + stage_11;
                                    if (col_20 >= col_blocks_13) {
                                        row_27 = row_27 + 1;
                                        col_20 = col_20 - col_blocks_13;
                                    }
                                    int local_block_1 = row_27 - macro_row_offset_2;
                                    int local_row_4 = local_block_1 * 128 + tile_row_2;
                                    mbarrier_wait(swiglu_arrived_addr + (stage_11) * 8, phase_bits_13 >> (unsigned int)stage_11 & 1);
                                    phase_bits_13 = phase_bits_13 ^ (unsigned int)(1 << stage_11);
                                    int32_t _relaxed_ld_38;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_38) : "l"(schedule_rank + (row_27 * 128 + tile_row_2)) : "memory");
                                    int peer_5 = _relaxed_ld_38;
                                    float weight = weights[local_row_4];
                                    float router_gradient = 0.0f;
                                    unsigned int gate_scales = q_words[(131072 + stage_11 * 512) / 4 + scale_index];
                                    unsigned int up_scales = q_words[(132096 + stage_11 * 512) / 4 + scale_index];
                                    int gate_base = (65536 + stage_11 * 16384) / 4 + tile_row_2 * 32;
                                    int up_base = (98304 + stage_11 * 16384) / 4 + tile_row_2 * 32;
                                    int dh_base = stage_11 * 32768 / 4 + tile_row_2 * 64;
                                    int dup_stage_base = 33280 + tile_row_2 * 64;
                                    unsigned int dgate_scale_pair = 0;
                                    unsigned int dup_scale_pair = 0;
                                    #pragma unroll 1
                                    for (int j_16 = 0; j_16 < 2; j_16++) {
                                        int k_sub = tile_row_2 + j_16 & 1;
                                        int k_block_4 = k_pair * 2 + k_sub;
                                        unsigned int gate_scale_bits = (gate_scales >> (unsigned int)(k_block_4 * 8) & 255) << 23;
                                        unsigned int up_scale_bits = (up_scales >> (unsigned int)(k_block_4 * 8) & 255) << 23;
                                        float gate_scale = 0.0f;
                                        float up_scale = 0.0f;
                                        gate_scale = __uint_as_float(gate_scale_bits);
                                        up_scale = __uint_as_float(up_scale_bits);
                                        unsigned int dgate_words[16];
                                        unsigned int dup_words[16];
                                        if (swiglu_clamped != 0) {
                                            #pragma unroll
                                            for (int k_18 = 0; k_18 < 8; k_18++) {
                                                int col_idx = k_block_4 * 32 + (tile_row_2 / 4 + k_18) % 8 * 4;
                                                unsigned int gate_word = q_words[gate_base + col_idx / 4];
                                                unsigned int up_word = q_words[up_base + col_idx / 4];
                                                uint32_t _q_words_reg_0[2];
                                                uint64_t _smem_raw_11;
                                                asm volatile("ld.weak.shared::cta.b64 %0, [%1];" : "=l"(_smem_raw_11) : "r"(q_words_addr + (dh_base + col_idx / 2) * 4) : "memory");
                                                _q_words_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                                                _q_words_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                                                #pragma unroll
                                                for (int h = 0; h < 2; h++) {
                                                    float2 _fp8x2_decode_0;
                                                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                        "mov.b32 {lo, hi}, pair;\n"
                                                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                        : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"((uint16_t)(gate_word >> (unsigned int)(16 * h) & 65535)));
                                                    float2 _fp8x2_decode_1;
                                                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                        "mov.b32 {lo, hi}, pair;\n"
                                                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                        : "=f"(_fp8x2_decode_1.x), "=f"(_fp8x2_decode_1.y) : "h"((uint16_t)(up_word >> (unsigned int)(16 * h) & 65535)));
                                                    float2 _cvt_f32_10 = __bfloat1622float2(__as_bf16x2(_q_words_reg_0[h]));
                                                    float g_x = _fp8x2_decode_0.x * gate_scale;
                                                    float g_y = _fp8x2_decode_0.y * gate_scale;
                                                    float u_x = _fp8x2_decode_1.x * up_scale;
                                                    float u_y = _fp8x2_decode_1.y * up_scale;
                                                    float dh_x = _cvt_f32_10.x * weight;
                                                    float dh_y = _cvt_f32_10.y * weight;
                                                    float g_cx = g_x;
                                                    float g_cy = g_y;
                                                    float u_cx = u_x;
                                                    float u_cy = u_y;
                                                    bool gate_mask_x = g_x <= swiglu_limit;
                                                    bool gate_mask_y = g_y <= swiglu_limit;
                                                    bool up_mask_x = u_x >= -swiglu_limit && u_x <= swiglu_limit;
                                                    bool up_mask_y = u_y >= -swiglu_limit && u_y <= swiglu_limit;
                                                    float _min_57 = fminf(g_x, swiglu_limit);
                                                    g_cx = _min_57;
                                                    float _min_58 = fminf(g_y, swiglu_limit);
                                                    g_cy = _min_58;
                                                    float _max_30 = max_noftz(u_x, -swiglu_limit);
                                                    float _min_59 = fminf(_max_30, swiglu_limit);
                                                    u_cx = _min_59;
                                                    float _max_31 = max_noftz(u_y, -swiglu_limit);
                                                    float _min_60 = fminf(_max_31, swiglu_limit);
                                                    u_cy = _min_60;
                                                    float _exp_6 = expf(-g_cx);
                                                    float sigmoid_x = 1.0f / (1.0f + _exp_6);
                                                    float _exp_7 = expf(-g_cy);
                                                    float sigmoid_y = 1.0f / (1.0f + _exp_7);
                                                    float silu_x = g_cx * sigmoid_x;
                                                    float silu_y = g_cy * sigmoid_y;
                                                    float dsilu_x = (1.0f - silu_x) * sigmoid_x + silu_x;
                                                    float dsilu_y = (1.0f - silu_y) * sigmoid_y + silu_y;
                                                    float hidden_x = silu_x * u_cx;
                                                    float hidden_y = silu_y * u_cy;
                                                    float dgate_x = dsilu_x * u_cx * dh_x;
                                                    float dgate_y = dsilu_y * u_cy * dh_y;
                                                    float dup_x = silu_x * dh_x;
                                                    float dup_y = silu_y * dh_y;
                                                    dgate_x = ((gate_mask_x) ? dgate_x : 0.0f);
                                                    dgate_y = ((gate_mask_y) ? dgate_y : 0.0f);
                                                    dup_x = ((up_mask_x) ? dup_x : 0.0f);
                                                    dup_y = ((up_mask_y) ? dup_y : 0.0f);
                                                    router_gradient = router_gradient + (_cvt_f32_10.x * hidden_x + _cvt_f32_10.y * hidden_y);
                                                    __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(dgate_x, dgate_y));
                                                    dgate_words[k_18 * 2 + h] = __as_u32(_bf16x2_17);
                                                    __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(dup_x, dup_y));
                                                    dup_words[k_18 * 2 + h] = __as_u32(_bf16x2_18);
                                                }
                                            }
                                        } else {
                                            #pragma unroll
                                            for (int k_19 = 0; k_19 < 8; k_19++) {
                                                int col_idx_1 = k_block_4 * 32 + (tile_row_2 / 4 + k_19) % 8 * 4;
                                                unsigned int gate_word_1 = q_words[gate_base + col_idx_1 / 4];
                                                unsigned int up_word_1 = q_words[up_base + col_idx_1 / 4];
                                                uint32_t _q_words_reg_1[2];
                                                uint64_t _smem_raw_12;
                                                asm volatile("ld.weak.shared::cta.b64 %0, [%1];" : "=l"(_smem_raw_12) : "r"(q_words_addr + (dh_base + col_idx_1 / 2) * 4) : "memory");
                                                _q_words_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
                                                _q_words_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
                                                #pragma unroll
                                                for (int h_1 = 0; h_1 < 2; h_1++) {
                                                    float2 _fp8x2_decode_2;
                                                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                        "mov.b32 {lo, hi}, pair;\n"
                                                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                        : "=f"(_fp8x2_decode_2.x), "=f"(_fp8x2_decode_2.y) : "h"((uint16_t)(gate_word_1 >> (unsigned int)(16 * h_1) & 65535)));
                                                    float2 _fp8x2_decode_3;
                                                    asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                                        "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                                        "mov.b32 {lo, hi}, pair;\n"
                                                        "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                                        : "=f"(_fp8x2_decode_3.x), "=f"(_fp8x2_decode_3.y) : "h"((uint16_t)(up_word_1 >> (unsigned int)(16 * h_1) & 65535)));
                                                    float2 _cvt_f32_11 = __bfloat1622float2(__as_bf16x2(_q_words_reg_1[h_1]));
                                                    float g_x_1 = _fp8x2_decode_2.x * gate_scale;
                                                    float g_y_1 = _fp8x2_decode_2.y * gate_scale;
                                                    float u_x_1 = _fp8x2_decode_3.x * up_scale;
                                                    float u_y_1 = _fp8x2_decode_3.y * up_scale;
                                                    float dh_x_1 = _cvt_f32_11.x * weight;
                                                    float dh_y_1 = _cvt_f32_11.y * weight;
                                                    float g_cx_1 = g_x_1;
                                                    float g_cy_1 = g_y_1;
                                                    float u_cx_1 = u_x_1;
                                                    float u_cy_1 = u_y_1;
                                                    float _exp_8 = expf(-g_cx_1);
                                                    float sigmoid_x_1 = 1.0f / (1.0f + _exp_8);
                                                    float _exp_9 = expf(-g_cy_1);
                                                    float sigmoid_y_1 = 1.0f / (1.0f + _exp_9);
                                                    float silu_x_1 = g_cx_1 * sigmoid_x_1;
                                                    float silu_y_1 = g_cy_1 * sigmoid_y_1;
                                                    float dsilu_x_1 = (1.0f - silu_x_1) * sigmoid_x_1 + silu_x_1;
                                                    float dsilu_y_1 = (1.0f - silu_y_1) * sigmoid_y_1 + silu_y_1;
                                                    float hidden_x_1 = silu_x_1 * u_cx_1;
                                                    float hidden_y_1 = silu_y_1 * u_cy_1;
                                                    float dgate_x_1 = dsilu_x_1 * u_cx_1 * dh_x_1;
                                                    float dgate_y_1 = dsilu_y_1 * u_cy_1 * dh_y_1;
                                                    float dup_x_1 = silu_x_1 * dh_x_1;
                                                    float dup_y_1 = silu_y_1 * dh_y_1;
                                                    router_gradient = router_gradient + (_cvt_f32_11.x * hidden_x_1 + _cvt_f32_11.y * hidden_y_1);
                                                    __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(dgate_x_1, dgate_y_1));
                                                    dgate_words[k_19 * 2 + h_1] = __as_u32(_bf16x2_19);
                                                    __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(dup_x_1, dup_y_1));
                                                    dup_words[k_19 * 2 + h_1] = __as_u32(_bf16x2_20);
                                                }
                                            }
                                        }
                                        unsigned int dgate_packed[8];
                                        unsigned int dup_packed[8];
                                        uint32_t _bf16x2_abs_20;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(dgate_words[0]));
                                        unsigned int amax2_10 = _bf16x2_abs_20;
                                        #pragma unroll
                                        for (int k_20 = 1; k_20 < 16; k_20++) {
                                            uint32_t _bf16x2_abs_21;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(dgate_words[k_20]));
                                            uint32_t _bf16x2_max_10;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_10) : "r"(amax2_10), "r"(_bf16x2_abs_21));
                                            amax2_10 = _bf16x2_max_10;
                                        }
                                        uint16_t _bf16_max_10;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_10) : "h"((uint16_t)(amax2_10 & 65535)), "h"((uint16_t)(amax2_10 >> 16)));
                                        float _cvt_f32_bf16_50;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_50) : "h"((uint16_t)(_bf16_max_10)));
                                        float amax_10 = _cvt_f32_bf16_50;
                                        float _max_32 = max_noftz(amax_10 * inv_e4m3_max_6, scale_floor_6);
                                        float scale_10 = _max_32;
                                        uint16_t _ue8m0x2_f32_10;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_10) : "f"(scale_10), "f"(scale_10));
                                        uint16_t codes_10 = _ue8m0x2_f32_10;
                                        unsigned int scale_byte_10 = (unsigned int)codes_10 & 255;
                                        unsigned int inv_bits_10 = 254 - scale_byte_10 << 23;
                                        float inv_10 = 0.0f;
                                        inv_10 = __uint_as_float(inv_bits_10);
                                        #pragma unroll
                                        for (int i_20 = 0; i_20 < 8; i_20++) {
                                            unsigned int w0_10 = dgate_words[2 * i_20];
                                            unsigned int w1_10 = dgate_words[2 * i_20 + 1];
                                            float _cvt_f32_bf16_51;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_51) : "h"((uint16_t)(w0_10 & 65535)));
                                            float v0_13 = _cvt_f32_bf16_51;
                                            float _cvt_f32_bf16_52;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_52) : "h"((uint16_t)(w0_10 >> 16)));
                                            float v1_13 = _cvt_f32_bf16_52;
                                            float _cvt_f32_bf16_53;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_53) : "h"((uint16_t)(w1_10 & 65535)));
                                            float v2_10 = _cvt_f32_bf16_53;
                                            float _cvt_f32_bf16_54;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_54) : "h"((uint16_t)(w1_10 >> 16)));
                                            float v3_10 = _cvt_f32_bf16_54;
                                            uint16_t _e4m3x2_f32_20;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_20) : "f"(v1_13 * inv_10), "f"(v0_13 * inv_10));
                                            uint16_t lo_11 = _e4m3x2_f32_20;
                                            uint16_t _e4m3x2_f32_21;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_21) : "f"(v3_10 * inv_10), "f"(v2_10 * inv_10));
                                            uint16_t hi_11 = _e4m3x2_f32_21;
                                            dgate_packed[i_20] = (unsigned int)lo_11 | (unsigned int)hi_11 << 16;
                                        }
                                        unsigned int dgate_byte = scale_byte_10;
                                        uint32_t _bf16x2_abs_22;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(dup_words[0]));
                                        unsigned int amax2_0 = _bf16x2_abs_22;
                                        #pragma unroll
                                        for (int k_21 = 1; k_21 < 16; k_21++) {
                                            uint32_t _bf16x2_abs_23;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(dup_words[k_21]));
                                            uint32_t _bf16x2_max_11;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_11) : "r"(amax2_0), "r"(_bf16x2_abs_23));
                                            amax2_0 = _bf16x2_max_11;
                                        }
                                        uint16_t _bf16_max_11;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_11) : "h"((uint16_t)(amax2_0 & 65535)), "h"((uint16_t)(amax2_0 >> 16)));
                                        float _cvt_f32_bf16_55;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_55) : "h"((uint16_t)(_bf16_max_11)));
                                        float amax_1_1 = _cvt_f32_bf16_55;
                                        float _max_33 = max_noftz(amax_1_1 * inv_e4m3_max_6, scale_floor_6);
                                        float scale_2_1 = _max_33;
                                        uint16_t _ue8m0x2_f32_11;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_11) : "f"(scale_2_1), "f"(scale_2_1));
                                        uint16_t codes_3_1 = _ue8m0x2_f32_11;
                                        unsigned int scale_byte_4_1 = (unsigned int)codes_3_1 & 255;
                                        unsigned int inv_bits_5_1 = 254 - scale_byte_4_1 << 23;
                                        float inv_6_1 = 0.0f;
                                        inv_6_1 = __uint_as_float(inv_bits_5_1);
                                        #pragma unroll
                                        for (int i_21 = 0; i_21 < 8; i_21++) {
                                            unsigned int w0_11 = dup_words[2 * i_21];
                                            unsigned int w1_11 = dup_words[2 * i_21 + 1];
                                            float _cvt_f32_bf16_56;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_56) : "h"((uint16_t)(w0_11 & 65535)));
                                            float v0_14 = _cvt_f32_bf16_56;
                                            float _cvt_f32_bf16_57;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_57) : "h"((uint16_t)(w0_11 >> 16)));
                                            float v1_14 = _cvt_f32_bf16_57;
                                            float _cvt_f32_bf16_58;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_58) : "h"((uint16_t)(w1_11 & 65535)));
                                            float v2_11 = _cvt_f32_bf16_58;
                                            float _cvt_f32_bf16_59;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_59) : "h"((uint16_t)(w1_11 >> 16)));
                                            float v3_11 = _cvt_f32_bf16_59;
                                            uint16_t _e4m3x2_f32_22;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_22) : "f"(v1_14 * inv_6_1), "f"(v0_14 * inv_6_1));
                                            uint16_t lo_12 = _e4m3x2_f32_22;
                                            uint16_t _e4m3x2_f32_23;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_23) : "f"(v3_11 * inv_6_1), "f"(v2_11 * inv_6_1));
                                            uint16_t hi_12 = _e4m3x2_f32_23;
                                            dup_packed[i_21] = (unsigned int)lo_12 | (unsigned int)hi_12 << 16;
                                        }
                                        unsigned int dup_byte = scale_byte_4_1;
                                        dgate_scale_pair = dgate_scale_pair | dgate_byte << (unsigned int)(k_sub * 8);
                                        dup_scale_pair = dup_scale_pair | dup_byte << (unsigned int)(k_sub * 8);
                                        #pragma unroll
                                        for (int k_22 = 0; k_22 < 8; k_22++) {
                                            int col_idx_2 = k_block_4 * 32 + (tile_row_2 / 4 + k_22) % 8 * 4;
                                            q_words[gate_base + col_idx_2 / 4] = dgate_packed[k_22];
                                            q_words[up_base + col_idx_2 / 4] = dup_packed[k_22];
                                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(q_words_addr + (unsigned int)((dh_base + col_idx_2 / 2) * 4)), "r"(dgate_words[k_22 * 2]), "r"(dgate_words[k_22 * 2 + 1]) : "memory");
                                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(q_words_addr + (unsigned int)((dup_stage_base + col_idx_2 / 2) * 4)), "r"(dup_words[k_22 * 2]), "r"(dup_words[k_22 * 2 + 1]) : "memory");
                                        }
                                    }
                                    q_halves[(131072 + stage_11 * 512) / 2 + scale_index * 2 + k_pair] = (uint16_t)dgate_scale_pair;
                                    q_halves[(132096 + stage_11 * 512) / 2 + scale_index * 2 + k_pair] = (uint16_t)dup_scale_pair;
                                    if (stage_11 == 1) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                    }
                                    __syncthreads();
                                    if (tid == 0) {
                                        int scale_tile_1 = (local_block_1 * col_blocks_13 + col_20) * 32;
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_2d((&dg_fp8_r), col_20 * 128, local_block_1 * 128, q_flat_addr + 65536 + (unsigned int)(stage_11 * 16384));
                                        tma_store_3d((&dg_sc_r), 0, scale_tile_1, 0, q_flat_addr + 131072 + (unsigned int)(stage_11 * 512));
                                        tma_store_2d((&du_fp8_r), col_20 * 128, local_block_1 * 128, q_flat_addr + 98304 + (unsigned int)(stage_11 * 16384));
                                        tma_store_3d((&du_sc_r), 0, scale_tile_1, 0, q_flat_addr + 132096 + (unsigned int)(stage_11 * 512));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    int src_offset = ((half_15 == 0) ? stage_11 * 32768 / 2 : 66560);
                                    int t_row_4 = tile_row_2 % 64 * 2 + tile_row_2 / 64;
                                    int rotation_7 = tile_row_2 / 8;
                                    unsigned int t_scale_word_4 = 0;
                                    #pragma unroll 1
                                    for (int j_17 = 0; j_17 < 4; j_17++) {
                                        int k_block_5 = (j_17 + rotation_7) % 4;
                                        unsigned int t_words_4[16];
                                        #pragma unroll
                                        for (int k_23 = 0; k_23 < 16; k_23++) {
                                            int src_row_4 = k_block_5 * 32 + (tile_row_2 * 4 + k_23 * 2) % 32;
                                            unsigned int lo_13 = (unsigned int)q_halves[src_row_4 * 128 + src_offset + t_row_4];
                                            unsigned int hi_13 = (unsigned int)q_halves[(src_row_4 + 1) * 128 + src_offset + t_row_4];
                                            uint32_t _prmt_b32_1;
                                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_1) : "r"(lo_13), "r"(hi_13));
                                            t_words_4[k_23] = _prmt_b32_1;
                                        }
                                        unsigned int t_packed_4[8];
                                        uint32_t _bf16x2_abs_24;
                                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(t_words_4[0]));
                                        unsigned int amax2_11 = _bf16x2_abs_24;
                                        #pragma unroll
                                        for (int k_24 = 1; k_24 < 16; k_24++) {
                                            uint32_t _bf16x2_abs_25;
                                            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(t_words_4[k_24]));
                                            uint32_t _bf16x2_max_12;
                                            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_12) : "r"(amax2_11), "r"(_bf16x2_abs_25));
                                            amax2_11 = _bf16x2_max_12;
                                        }
                                        uint16_t _bf16_max_12;
                                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_12) : "h"((uint16_t)(amax2_11 & 65535)), "h"((uint16_t)(amax2_11 >> 16)));
                                        float _cvt_f32_bf16_60;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_60) : "h"((uint16_t)(_bf16_max_12)));
                                        float amax_11 = _cvt_f32_bf16_60;
                                        float _max_34 = max_noftz(amax_11 * inv_e4m3_max_6, scale_floor_6);
                                        float scale_11 = _max_34;
                                        uint16_t _ue8m0x2_f32_12;
                                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_12) : "f"(scale_11), "f"(scale_11));
                                        uint16_t codes_11 = _ue8m0x2_f32_12;
                                        unsigned int scale_byte_11 = (unsigned int)codes_11 & 255;
                                        unsigned int inv_bits_11 = 254 - scale_byte_11 << 23;
                                        float inv_11 = 0.0f;
                                        inv_11 = __uint_as_float(inv_bits_11);
                                        #pragma unroll
                                        for (int i_22 = 0; i_22 < 8; i_22++) {
                                            unsigned int w0_12 = t_words_4[2 * i_22];
                                            unsigned int w1_12 = t_words_4[2 * i_22 + 1];
                                            float _cvt_f32_bf16_61;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_61) : "h"((uint16_t)(w0_12 & 65535)));
                                            float v0_15 = _cvt_f32_bf16_61;
                                            float _cvt_f32_bf16_62;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_62) : "h"((uint16_t)(w0_12 >> 16)));
                                            float v1_15 = _cvt_f32_bf16_62;
                                            float _cvt_f32_bf16_63;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_63) : "h"((uint16_t)(w1_12 & 65535)));
                                            float v2_12 = _cvt_f32_bf16_63;
                                            float _cvt_f32_bf16_64;
                                            asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_64) : "h"((uint16_t)(w1_12 >> 16)));
                                            float v3_12 = _cvt_f32_bf16_64;
                                            uint16_t _e4m3x2_f32_24;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_24) : "f"(v1_15 * inv_11), "f"(v0_15 * inv_11));
                                            uint16_t lo_14 = _e4m3x2_f32_24;
                                            uint16_t _e4m3x2_f32_25;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_25) : "f"(v3_12 * inv_11), "f"(v2_12 * inv_11));
                                            uint16_t hi_14 = _e4m3x2_f32_25;
                                            t_packed_4[i_22] = (unsigned int)lo_14 | (unsigned int)hi_14 << 16;
                                        }
                                        unsigned int t_scale_byte_4 = scale_byte_11;
                                        #pragma unroll
                                        for (int i_23 = 0; i_23 < 8; i_23++) {
                                            int t_col_4 = k_block_5 * 32 + (tile_row_2 * 4 + i_23 * 4) % 32;
                                            q_words[41472 + half_15 * 4096 + t_row_4 * 32 + t_col_4 / 4] = t_packed_4[i_23];
                                        }
                                        t_scale_word_4 = t_scale_word_4 | t_scale_byte_4 << (unsigned int)(k_block_5 * 8);
                                    }
                                    q_words[49664 + half_15 * 128 + t_row_4 % 32 * 4 + t_row_4 / 32] = t_scale_word_4;
                                    if (half_15 != 0) {
                                        q_router[stage_11 * 128 + tile_row_2] = router_gradient;
                                    }
                                    __syncthreads();
                                    if (tid == 0) {
                                        int t_scale_tile = (col_20 * macro_row_blocks_2 + local_block_1) * 32;
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_2d((&dg_fp8_t_r), local_block_1 * 128, col_20 * 128, q_flat_addr + 165888);
                                        tma_store_3d((&dg_sc_t_r), 0, t_scale_tile, 0, q_flat_addr + 198656);
                                        tma_store_2d((&du_fp8_t_r), local_block_1 * 128, col_20 * 128, q_flat_addr + 165888 + 16384);
                                        tma_store_3d((&du_sc_t_r), 0, t_scale_tile, 0, q_flat_addr + 198656 + 512);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                    if (half_15 == 0 && peer_5 >= 0) {
                                        router_gradient = router_gradient + q_router[stage_11 * 128 + tile_row_2];
                                        partials[local_row_4 * col_blocks_13 + col_20] = router_gradient;
                                    }
                                    if (stage_11 == 1) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                    } else if (tile_end_2 <= first_tile_3 + 1) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                    }
                                    __syncthreads();
                                }
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                #pragma unroll
                                for (int stage_12 = 0; stage_12 < 2; stage_12++) {
                                    if (tile_end_2 > first_tile_3 + stage_12) {
                                        int row_28 = first_row_6;
                                        if (col_blocks_13 <= first_col_3 + stage_12) {
                                            row_28 = row_28 + 1;
                                        }
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (shared_rows + row_28 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            }
                        }
                        if (tid == 0) {
                            bool enabled_value_17 = macros > 1;
                            if (enabled_value_17 != 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        swiglu_phase = phase_bits_13;
                    } else {
                        if (kind == 2) {
                            int col_blocks_14 = hidden / 256;
                            int x_8 = -1;
                            int y_8 = -1;
                            int expert_8 = -1;
                            int k_start_8 = 0;
                            int k_end_8 = 0;
                            int first_8 = 0;
                            int first_block_3 = (macro_1 * (macro_size / mini_size) + mini_1) * (mini_size / 256);
                            int offset_10 = 0;
                            int remaining_3 = task_3;
                            #pragma unroll 1
                            for (int index_3 = 0; index_3 < experts; index_3++) {
                                int blocks_3 = counts[index_3] / 256;
                                int _max_35 = ((first_block_3) > (offset_10) ? (first_block_3) : (offset_10));
                                int first_row_7 = _max_35;
                                int _min_61 = ((first_block_3 + mini_size / 256) < (offset_10 + blocks_3) ? (first_block_3 + mini_size / 256) : (offset_10 + blocks_3));
                                int _max_36 = ((0) > (_min_61 - first_row_7) ? (0) : (_min_61 - first_row_7));
                                int rows_6 = _max_36;
                                int tasks_3 = rows_6 * col_blocks_14;
                                if (remaining_3 < tasks_3) {
                                    int supergroup_8 = remaining_3 / (rows_6 * 8);
                                    int full_cols_8 = col_blocks_14 / 8 * 8;
                                    int row_29 = 0;
                                    int col_21 = 0;
                                    if (remaining_3 < rows_6 * full_cols_8) {
                                        row_29 = remaining_3 % (rows_6 * 8) / 8;
                                        col_21 = supergroup_8 * 8 + remaining_3 % 8;
                                    } else {
                                        row_29 = (remaining_3 - rows_6 * full_cols_8) / (col_blocks_14 - full_cols_8);
                                        col_21 = full_cols_8 + (remaining_3 - rows_6 * full_cols_8) % (col_blocks_14 - full_cols_8);
                                    }
                                    if ((supergroup_8 & 1) != 0) {
                                        row_29 = rows_6 - row_29 - 1;
                                    }
                                    x_8 = first_row_7 + row_29 - macro_1 * (macro_size / 256);
                                    y_8 = col_21;
                                    expert_8 = index_3;
                                    break;
                                }
                                remaining_3 = remaining_3 - tasks_3;
                                offset_10 = offset_10 + blocks_3;
                            }
                            unsigned int phase_bits_14 = gemm_phase;
                            int global_mini_10 = macro_1 * (macro_size / mini_size) + mini_1;
                            int macro_rows_8 = macro_1 * (macro_size / 256);
                            int iterations_8 = intermediate / 128 + intermediate / 128;
                            int k_blocks_3 = intermediate / 128;
                            int n_blocks_3 = hidden / 128;
                            if (expert_8 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_18 = macros > 1;
                                    if (enabled_value_18 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_19 = 1;
                                            if (enabled_value_19 != 0) {
                                                int32_t _relaxed_ld_39;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_39) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                int value_18 = _relaxed_ld_39;
                                                while (value_18 < row_count) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_40;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_40) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    value_18 = _relaxed_ld_40;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_62 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                            int _max_37 = ((0) > (_min_62) ? (0) : (_min_62));
                                            int mini_rows_12 = _max_37;
                                            int required_11 = (mini_rows_12 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        int ring_19 = 0;
                                        #pragma unroll 1
                                        for (int idx_19 = 0; idx_19 < iterations_8; idx_19++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_19) * 8, phase_bits_14 >> (unsigned int)(16 + ring_19) & 1);
                                            if (idx_19 < intermediate / 128) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_fp8_smem_addr + (unsigned int)(ring_19 * 16384)), "l"((&dg_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_19), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_fp8_smem_addr + (unsigned int)(ring_19 * 16384)), "l"((&wg_t_r)), "r"(0), "r"(y_8 * 256 + cta_rank_0 * 128), "r"(idx_19), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            } else {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_fp8_smem_addr + (unsigned int)(ring_19 * 16384)), "l"((&du_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_19) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_fp8_smem_addr + (unsigned int)(ring_19 * 16384)), "l"((&wu_t_r)), "r"(0), "r"(y_8 * 256 + cta_rank_0 * 128), "r"(idx_19 - intermediate / 128), "r"(expert_8), "r"(0),
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
                                                bool enabled_value_20 = 1;
                                                if (enabled_value_20 != 0) {
                                                    int32_t _relaxed_ld_41;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_41) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    int value_19 = _relaxed_ld_41;
                                                    while (value_19 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_42;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_42) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                        value_19 = _relaxed_ld_42;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                                int _min_63 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                                int _max_38 = ((0) > (_min_63) ? (0) : (_min_63));
                                                int mini_rows_13 = _max_38;
                                                int required_12 = (mini_rows_13 + 127) / 128 * ((intermediate + 511) / 512);
                                            }
                                            int ring_20 = 0;
                                            #pragma unroll 1
                                            for (int idx_20 = 0; idx_20 < iterations_8; idx_20++) {
                                                mbarrier_wait(scales_finished_addr + (ring_20) * 8, phase_bits_14 >> (unsigned int)(23 + ring_20) & 1);
                                                if (idx_20 < intermediate / 128) {
                                                    int a_tile_3 = (x_8 * 2 + cta_rank_0) * k_blocks_3 + idx_20;
                                                    int b_tile_3 = (expert_8 * n_blocks_3 + y_8 * 2 + cta_rank_0) * k_blocks_3 + idx_20;
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(a_sc_smem_addr + (unsigned int)(ring_20 * 512)), "l"((&dg_sc_r)), "r"(0), "r"(a_tile_3 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(b_sc_smem_addr + (unsigned int)(ring_20 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wg_t_sc_r)), "r"(0), "r"(b_tile_3 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                } else {
                                                    int a_tile_4 = (x_8 * 2 + cta_rank_0) * k_blocks_3 + (idx_20 - intermediate / 128);
                                                    int b_tile_4 = (expert_8 * n_blocks_3 + y_8 * 2 + cta_rank_0) * k_blocks_3 + (idx_20 - intermediate / 128);
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(a_sc_smem_addr + (unsigned int)(ring_20 * 512)), "l"((&du_sc_r)), "r"(0), "r"(a_tile_4 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(b_sc_smem_addr + (unsigned int)(ring_20 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&wu_t_sc_r)), "r"(0), "r"(b_tile_4 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                }
                                                phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << 23 + ring_20);
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
                                                mbarrier_wait(scales_arrived_addr + (ring_21) * 8, phase_bits_14 >> (unsigned int)(7 + ring_21) & 1);
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_21 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_21 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_21 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_21 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_21 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_21 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_21) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_21) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_21) * 8, phase_bits_14 >> (unsigned int)ring_21 & 1);
                                                int _mma_a_lo_8 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_21) * 1024;
                                                int _mma_b_lo_8 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_21) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_21 * 4, tmem_sf_b + ring_21 * 8, ((idx_21 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                        (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_21 * 4, tmem_sf_b + ring_21 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_21 * 4, tmem_sf_b + ring_21 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                        (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_21 * 4, tmem_sf_b + ring_21 * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_21) * 8, (uint16_t)(3));
                                                phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << ring_21);
                                                phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << 7 + ring_21);
                                                ring_21 = (ring_21 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_14 >> 6 & 1);
                                        phase_bits_14 = phase_bits_14 ^ 64;
                                        unsigned int packed_18[128];
                                        #pragma unroll
                                        for (int chunk_12 = 0; chunk_12 < 8; chunk_12++) {
                                            #pragma unroll
                                            for (int half_16 = 0; half_16 < 2; half_16++) {
                                                unsigned int address_24 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_16 * 16 << 16) + (unsigned int)(chunk_12 * 32);
                                                float _tmem_load_8[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                                                    : "r"(address_24));
                                                #pragma unroll
                                                for (int pair_18 = 0; pair_18 < 8; pair_18++) {
                                                    __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_tmem_load_8[pair_18 * 2], _tmem_load_8[pair_18 * 2 + 1]));
                                                    packed_18[chunk_12 * 16 + half_16 * 8 + pair_18] = __as_u32(_bf16x2_21);
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            int previous_offset_9 = (macro_1 + 1) * macro_size;
                                            int output_row_6 = x_8 * 256 + cta_rank_0 * 128;
                                            int _min_64 = ((macro_size) < (tokens - previous_offset_9) ? (macro_size) : (tokens - previous_offset_9));
                                            if (output_row_6 < _min_64) {
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_13 = 0; chunk_13 < 8; chunk_13++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_8 = tid / 32;
                                            int lane_13 = tid % 32;
                                            #pragma unroll
                                            for (int half_17 = 0; half_17 < 2; half_17++) {
                                                #pragma unroll
                                                for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                                    int row_30 = warp_0_8 * 32 + half_17 * 16 + lane_13 % 16;
                                                    int col_22 = col_tile_6 * 16 + lane_13 / 16 * 8;
                                                    unsigned int address_25 = d_smem_addr + (unsigned int)(chunk_13 % 3 * 8192) + (unsigned int)((row_30 * 32 + col_22) * 2);
                                                    address_25 = address_25 ^ (address_25 & 511) >> 7 << 4;
                                                    int offset_0_1 = chunk_13 * 16 + half_17 * 8 + col_tile_6 * 4;
                                                    uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(address_25);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_0_1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_0_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_0_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_18[offset_0_1 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dx_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(y_8 * 8 + chunk_13), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_13 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                    bool enabled_value_21 = 1;
                                                    if (enabled_value_21 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_10))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                    bool enabled_value_0_1 = macros > 1;
                                                    if (enabled_value_0_1 != 0) {
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
                            int col_blocks_15 = macro_size / 256;
                            {
                                col_blocks_15 = intermediate / 256;
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
                            expert_idx_3 = task_3 / (row_blocks_5 * col_blocks_15);
                            local_task_3 = task_3 % (row_blocks_5 * col_blocks_15);
                            int offset_11 = 0;
                            #pragma unroll 1
                            for (int index_4 = 0; index_4 < expert_idx_3; index_4++) {
                                offset_11 = offset_11 + counts[index_4];
                            }
                            int _max_39 = ((offset_11) > (macro_1 * macro_size) ? (offset_11) : (macro_1 * macro_size));
                            k_start_9 = _max_39;
                            int _min_65 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                            int _min_66 = ((offset_11 + counts[expert_idx_3]) < (_min_65) ? (offset_11 + counts[expert_idx_3]) : (_min_65));
                            k_end_9 = _min_66;
                            first_9 = (int)(k_start_9 == offset_11);
                            if (k_start_9 < k_end_9) {
                                int supergroup_9 = local_task_3 / (row_blocks_5 * 8);
                                int full_cols_9 = col_blocks_15 / 8 * 8;
                                int row_31 = 0;
                                int col_23 = 0;
                                if (local_task_3 < row_blocks_5 * full_cols_9) {
                                    row_31 = local_task_3 % (row_blocks_5 * 8) / 8;
                                    col_23 = supergroup_9 * 8 + local_task_3 % 8;
                                } else {
                                    row_31 = (local_task_3 - row_blocks_5 * full_cols_9) / (col_blocks_15 - full_cols_9);
                                    col_23 = full_cols_9 + (local_task_3 - row_blocks_5 * full_cols_9) % (col_blocks_15 - full_cols_9);
                                }
                                if ((supergroup_9 & 1) != 0) {
                                    row_31 = row_blocks_5 - row_31 - 1;
                                }
                                x_9 = row_31;
                                y_9 = col_23;
                                expert_9 = expert_idx_3;
                            }
                            unsigned int phase_bits_15 = gemm_phase;
                            int global_mini_11 = macro_1 * (macro_size / mini_size);
                            int macro_rows_9 = macro_1 * (macro_size / 256);
                            int iterations_9 = hidden / 128;
                            iterations_9 = (k_end_9 - k_start_9 + 127) / 128;
                            int k_blocks_4 = hidden / 128;
                            k_blocks_4 = macro_size / 128;
                            int n_blocks_4 = intermediate / 128;
                            if (expert_9 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_22 = macros > 1;
                                    if (enabled_value_22 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        int ring_22 = 0;
                                        #pragma unroll 1
                                        for (int idx_22 = 0; idx_22 < iterations_9; idx_22++) {
                                            int token_row_3 = k_start_9 + idx_22 * 128;
                                            if (idx_22 == 0 || token_row_3 % 256 == 0) {
                                                bool enabled_value_23 = macro_1 > 0;
                                                if (enabled_value_23 != 0) {
                                                    int32_t _relaxed_ld_47;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_47) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                    int value_20 = _relaxed_ld_47;
                                                    while (value_20 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_48;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_48) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                        value_20 = _relaxed_ld_48;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_22 == 0 || token_row_3 % mini_size == 0) {
                                                int input_mini_3 = token_row_3 / mini_size;
                                                int _min_68 = ((mini_size) < (tokens - input_mini_3 * mini_size) ? (mini_size) : (tokens - input_mini_3 * mini_size));
                                                int input_rows_3 = _min_68;
                                                int input_count_3 = (input_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                                bool enabled_value_24 = 1;
                                                if (enabled_value_24 != 0) {
                                                    int32_t _relaxed_ld_49;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_49) : "l"(dy_ready + input_mini_3) : "memory");
                                                    int value_21 = _relaxed_ld_49;
                                                    while (value_21 < input_count_3) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_50;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_50) : "l"(dy_ready + input_mini_3) : "memory");
                                                        value_21 = _relaxed_ld_50;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_22) * 8, phase_bits_15 >> (unsigned int)(16 + ring_22) & 1);
                                            int k_tile = (k_start_9 + idx_22 * 128 - macro_1 * macro_size) / 128;
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_fp8_smem_addr + (unsigned int)(ring_22 * 16384)), "l"((&dy_t_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(k_tile), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_22) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_fp8_smem_addr + (unsigned int)(ring_22 * 16384)), "l"((&h_t_r)), "r"(0), "r"(y_9 * 256 + cta_rank_0 * 128), "r"(k_tile), "r"(0), "r"(0),
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
                                            int ring_23 = 0;
                                            #pragma unroll 1
                                            for (int idx_23 = 0; idx_23 < iterations_9; idx_23++) {
                                                int token_row_4 = k_start_9 + idx_23 * 128;
                                                if (idx_23 == 0 || token_row_4 % 256 == 0) {
                                                    bool enabled_value_25 = macro_1 > 0;
                                                    if (enabled_value_25 != 0) {
                                                        int32_t _relaxed_ld_55;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_55) : "l"(replay_h + (token_row_4 / 256)) : "memory");
                                                        int value_22 = _relaxed_ld_55;
                                                        while (value_22 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_56;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_56) : "l"(replay_h + (token_row_4 / 256)) : "memory");
                                                            value_22 = _relaxed_ld_56;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_23 == 0 || token_row_4 % mini_size == 0) {
                                                    int input_mini_4 = token_row_4 / mini_size;
                                                    int _min_70 = ((mini_size) < (tokens - input_mini_4 * mini_size) ? (mini_size) : (tokens - input_mini_4 * mini_size));
                                                    int input_rows_4 = _min_70;
                                                    int input_count_4 = (input_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_26 = 1;
                                                    if (enabled_value_26 != 0) {
                                                        int32_t _relaxed_ld_57;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_57) : "l"(dy_ready + input_mini_4) : "memory");
                                                        int value_23 = _relaxed_ld_57;
                                                        while (value_23 < input_count_4) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_58;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_58) : "l"(dy_ready + input_mini_4) : "memory");
                                                            value_23 = _relaxed_ld_58;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(scales_finished_addr + (ring_23) * 8, phase_bits_15 >> (unsigned int)(23 + ring_23) & 1);
                                                int k_tile_1 = (k_start_9 + idx_23 * 128 - macro_1 * macro_size) / 128;
                                                int a_tile_5 = (x_9 * 2 + cta_rank_0) * k_blocks_4 + k_tile_1;
                                                int b_tile_5 = (y_9 * 2 + cta_rank_0) * k_blocks_4 + k_tile_1;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(a_sc_smem_addr + (unsigned int)(ring_23 * 512)), "l"((&dy_sc_t_r)), "r"(0), "r"(a_tile_5 * 32), "r"(0),
                                                       "r"(((scales_arrived_addr + (ring_23) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                    :: "r"(b_sc_smem_addr + (unsigned int)(ring_23 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&h_sc_t_r)), "r"(0), "r"(b_tile_5 * 32), "r"(0),
                                                       "r"(((scales_arrived_addr + (ring_23) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << 23 + ring_23);
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
                                                mbarrier_wait(scales_arrived_addr + (ring_24) * 8, phase_bits_15 >> (unsigned int)(7 + ring_24) & 1);
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_24 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_24 * 512)));
                                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_24 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_24 * 1024)));
                                                tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_24 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_24 * 1024) + 512)));
                                                tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_24) * 8, (uint16_t)(3));
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_24) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_24) * 8, phase_bits_15 >> (unsigned int)ring_24 & 1);
                                                int _mma_a_lo_9 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_24) * 1024;
                                                int _mma_b_lo_9 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_24) * 1024;
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                        (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_24 * 4, tmem_sf_b + ring_24 * 8, ((idx_24 == 0) ? 0 : 1));
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                        (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_24 * 4, tmem_sf_b + ring_24 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                        (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_24 * 4, tmem_sf_b + ring_24 * 8, 1);
                                                    tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                        (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_24 * 4, tmem_sf_b + ring_24 * 8, 1);
                                                }
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_24) * 8, (uint16_t)(3));
                                                phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << ring_24);
                                                phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << 7 + ring_24);
                                                ring_24 = (ring_24 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else {
                                    if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_15 >> 6 & 1);
                                        phase_bits_15 = phase_bits_15 ^ 64;
                                        unsigned int packed_19[128];
                                        #pragma unroll
                                        for (int chunk_14 = 0; chunk_14 < 8; chunk_14++) {
                                            #pragma unroll
                                            for (int half_18 = 0; half_18 < 2; half_18++) {
                                                unsigned int address_26 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_18 * 16 << 16) + (unsigned int)(chunk_14 * 32);
                                                float _tmem_load_9[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                                                    : "r"(address_26));
                                                #pragma unroll
                                                for (int pair_19 = 0; pair_19 < 8; pair_19++) {
                                                    __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_tmem_load_9[pair_19 * 2], _tmem_load_9[pair_19 * 2 + 1]));
                                                    packed_19[chunk_14 * 16 + half_18 * 8 + pair_19] = __as_u32(_bf16x2_22);
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            int previous_offset_10 = (macro_1 + 1) * macro_size;
                                            int output_row_7 = x_9 * 256 + cta_rank_0 * 128;
                                            int _min_71 = ((macro_size) < (tokens - previous_offset_10) ? (macro_size) : (tokens - previous_offset_10));
                                            if (output_row_7 < _min_71) {
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_15 = 0; chunk_15 < 8; chunk_15++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_9 = tid / 32;
                                            int lane_14 = tid % 32;
                                            #pragma unroll
                                            for (int half_19 = 0; half_19 < 2; half_19++) {
                                                #pragma unroll
                                                for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                                    int row_32 = warp_0_9 * 32 + half_19 * 16 + lane_14 % 16;
                                                    int col_24 = col_tile_7 * 16 + lane_14 / 16 * 8;
                                                    unsigned int address_27 = d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192) + (unsigned int)((row_32 * 32 + col_24) * 2);
                                                    address_27 = address_27 ^ (address_27 & 511) >> 7 << 4;
                                                    int offset_0_2 = chunk_15 * 16 + half_19 * 8 + col_tile_7 * 4;
                                                    uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(address_27);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_0_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_0_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_0_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_19[offset_0_2 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                if (first_9 != 0) {
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(y_9 * 8 + chunk_15), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                } else {
                                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                    #error "TmaReduceAdd5d requires SM90 or newer"
                                                    #endif
                                                    asm volatile(
                                                        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                        :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(y_9 * 8 + chunk_15), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                }
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
                                                    bool enabled_value_27 = macros > 1;
                                                    if (enabled_value_27 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_0_2 = 1;
                                                        if (enabled_value_0_2 != 0) {
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
                                int col_blocks_16 = macro_size / 256;
                                {
                                    col_blocks_16 = hidden / 256;
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
                                expert_idx_4 = task_3 / (row_blocks_6 * col_blocks_16);
                                local_task_4 = task_3 % (row_blocks_6 * col_blocks_16);
                                int offset_12 = 0;
                                #pragma unroll 1
                                for (int index_5 = 0; index_5 < expert_idx_4; index_5++) {
                                    offset_12 = offset_12 + counts[index_5];
                                }
                                int _max_42 = ((offset_12) > (macro_1 * macro_size) ? (offset_12) : (macro_1 * macro_size));
                                k_start_10 = _max_42;
                                int _min_72 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                                int _min_73 = ((offset_12 + counts[expert_idx_4]) < (_min_72) ? (offset_12 + counts[expert_idx_4]) : (_min_72));
                                k_end_10 = _min_73;
                                first_10 = (int)(k_start_10 == offset_12);
                                if (k_start_10 < k_end_10) {
                                    int supergroup_10 = local_task_4 / (row_blocks_6 * 8);
                                    int full_cols_10 = col_blocks_16 / 8 * 8;
                                    int row_33 = 0;
                                    int col_25 = 0;
                                    if (local_task_4 < row_blocks_6 * full_cols_10) {
                                        row_33 = local_task_4 % (row_blocks_6 * 8) / 8;
                                        col_25 = supergroup_10 * 8 + local_task_4 % 8;
                                    } else {
                                        row_33 = (local_task_4 - row_blocks_6 * full_cols_10) / (col_blocks_16 - full_cols_10);
                                        col_25 = full_cols_10 + (local_task_4 - row_blocks_6 * full_cols_10) % (col_blocks_16 - full_cols_10);
                                    }
                                    if ((supergroup_10 & 1) != 0) {
                                        row_33 = row_blocks_6 - row_33 - 1;
                                    }
                                    x_10 = row_33;
                                    y_10 = col_25;
                                    expert_10 = expert_idx_4;
                                }
                                unsigned int phase_bits_16 = gemm_phase;
                                int global_mini_12 = macro_1 * (macro_size / mini_size);
                                int macro_rows_10 = macro_1 * (macro_size / 256);
                                int iterations_10 = intermediate / 128;
                                iterations_10 = (k_end_10 - k_start_10 + 127) / 128;
                                int k_blocks_5 = intermediate / 128;
                                k_blocks_5 = macro_size / 128;
                                int n_blocks_5 = hidden / 128;
                                if (expert_10 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_28 = macros > 1;
                                        if (enabled_value_28 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            int ring_25 = 0;
                                            #pragma unroll 1
                                            for (int idx_25 = 0; idx_25 < iterations_10; idx_25++) {
                                                int token_row_5 = k_start_10 + idx_25 * 128;
                                                if (idx_25 == 0 || token_row_5 % 256 == 0) {
                                                    bool enabled_value_29 = 1;
                                                    if (enabled_value_29 != 0) {
                                                        int32_t _relaxed_ld_63;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_63) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                        int value_24 = _relaxed_ld_63;
                                                        while (value_24 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_64;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_64) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                            value_24 = _relaxed_ld_64;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_25 == 0 || token_row_5 % mini_size == 0) {
                                                    int input_mini_5 = token_row_5 / mini_size;
                                                    int _min_75 = ((mini_size) < (tokens - input_mini_5 * mini_size) ? (mini_size) : (tokens - input_mini_5 * mini_size));
                                                    int input_rows_5 = _min_75;
                                                    int input_count_5 = (input_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_30 = macro_1 > 0;
                                                    if (enabled_value_30 != 0) {
                                                        int32_t _relaxed_ld_65;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_65) : "l"(replay_x + input_mini_5) : "memory");
                                                        int value_25 = _relaxed_ld_65;
                                                        while (value_25 < input_count_5) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_66;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_66) : "l"(replay_x + input_mini_5) : "memory");
                                                            value_25 = _relaxed_ld_66;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(gemm_finished_addr + (ring_25) * 8, phase_bits_16 >> (unsigned int)(16 + ring_25) & 1);
                                                int k_tile_2 = (k_start_10 + idx_25 * 128 - macro_1 * macro_size) / 128;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_fp8_smem_addr + (unsigned int)(ring_25 * 16384)), "l"((&dg_t_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(k_tile_2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_25) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_fp8_smem_addr + (unsigned int)(ring_25 * 16384)), "l"((&x_t_r)), "r"(0), "r"(y_10 * 256 + cta_rank_0 * 128), "r"(k_tile_2), "r"(0), "r"(0),
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
                                                int ring_26 = 0;
                                                #pragma unroll 1
                                                for (int idx_26 = 0; idx_26 < iterations_10; idx_26++) {
                                                    int token_row_6 = k_start_10 + idx_26 * 128;
                                                    if (idx_26 == 0 || token_row_6 % 256 == 0) {
                                                        bool enabled_value_31 = 1;
                                                        if (enabled_value_31 != 0) {
                                                            int32_t _relaxed_ld_71;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_71) : "l"(dg_ready + (shared_rows + token_row_6 / 256)) : "memory");
                                                            int value_26 = _relaxed_ld_71;
                                                            while (value_26 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_72;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_72) : "l"(dg_ready + (shared_rows + token_row_6 / 256)) : "memory");
                                                                value_26 = _relaxed_ld_72;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_26 == 0 || token_row_6 % mini_size == 0) {
                                                        int input_mini_6 = token_row_6 / mini_size;
                                                        int _min_77 = ((mini_size) < (tokens - input_mini_6 * mini_size) ? (mini_size) : (tokens - input_mini_6 * mini_size));
                                                        int input_rows_6 = _min_77;
                                                        int input_count_6 = (input_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_32 = macro_1 > 0;
                                                        if (enabled_value_32 != 0) {
                                                            int32_t _relaxed_ld_73;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_73) : "l"(replay_x + input_mini_6) : "memory");
                                                            int value_27 = _relaxed_ld_73;
                                                            while (value_27 < input_count_6) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_74;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_74) : "l"(replay_x + input_mini_6) : "memory");
                                                                value_27 = _relaxed_ld_74;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    mbarrier_wait(scales_finished_addr + (ring_26) * 8, phase_bits_16 >> (unsigned int)(23 + ring_26) & 1);
                                                    int k_tile_3 = (k_start_10 + idx_26 * 128 - macro_1 * macro_size) / 128;
                                                    int a_tile_6 = (x_10 * 2 + cta_rank_0) * k_blocks_5 + k_tile_3;
                                                    int b_tile_6 = (y_10 * 2 + cta_rank_0) * k_blocks_5 + k_tile_3;
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(a_sc_smem_addr + (unsigned int)(ring_26 * 512)), "l"((&dg_sc_t_r)), "r"(0), "r"(a_tile_6 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_26) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(b_sc_smem_addr + (unsigned int)(ring_26 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t_r)), "r"(0), "r"(b_tile_6 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_26) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << 23 + ring_26);
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
                                                    mbarrier_wait(scales_arrived_addr + (ring_27) * 8, phase_bits_16 >> (unsigned int)(7 + ring_27) & 1);
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_27 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_27 * 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_27 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_27 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_27 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_27 * 1024) + 512)));
                                                    tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_27) * 8, (uint16_t)(3));
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_27) * 8, 65536);
                                                    mbarrier_wait(gemm_arrived_addr + (ring_27) * 8, phase_bits_16 >> (unsigned int)ring_27 & 1);
                                                    int _mma_a_lo_10 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_27) * 1024;
                                                    int _mma_b_lo_10 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_27) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                            (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_27 * 4, tmem_sf_b + ring_27 * 8, ((idx_27 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                            (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_27 * 4, tmem_sf_b + ring_27 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                            (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_27 * 4, tmem_sf_b + ring_27 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                            (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_27 * 4, tmem_sf_b + ring_27 * 8, 1);
                                                    }
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_27) * 8, (uint16_t)(3));
                                                    phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << ring_27);
                                                    phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << 7 + ring_27);
                                                    ring_27 = (ring_27 + 1) % 6;
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else {
                                        if (tid < 128) {
                                            mbarrier_wait(output_arrived_addr, phase_bits_16 >> 6 & 1);
                                            phase_bits_16 = phase_bits_16 ^ 64;
                                            unsigned int packed_20[128];
                                            #pragma unroll
                                            for (int chunk_16 = 0; chunk_16 < 8; chunk_16++) {
                                                #pragma unroll
                                                for (int half_20 = 0; half_20 < 2; half_20++) {
                                                    unsigned int address_28 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_20 * 16 << 16) + (unsigned int)(chunk_16 * 32);
                                                    float _tmem_load_10[16];
                                                    asm volatile(
                                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                                                        : "r"(address_28));
                                                    #pragma unroll
                                                    for (int pair_20 = 0; pair_20 < 8; pair_20++) {
                                                        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_tmem_load_10[pair_20 * 2], _tmem_load_10[pair_20 * 2 + 1]));
                                                        packed_20[chunk_16 * 16 + half_20 * 8 + pair_20] = __as_u32(_bf16x2_23);
                                                    }
                                                }
                                            }
                                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                                int previous_offset_11 = (macro_1 + 1) * macro_size;
                                                int output_row_8 = x_10 * 256 + cta_rank_0 * 128;
                                                int _min_78 = ((macro_size) < (tokens - previous_offset_11) ? (macro_size) : (tokens - previous_offset_11));
                                                if (output_row_8 < _min_78) {
                                                }
                                            }
                                            #pragma unroll
                                            for (int chunk_17 = 0; chunk_17 < 8; chunk_17++) {
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                int warp_0_10 = tid / 32;
                                                int lane_15 = tid % 32;
                                                #pragma unroll
                                                for (int half_21 = 0; half_21 < 2; half_21++) {
                                                    #pragma unroll
                                                    for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                                        int row_34 = warp_0_10 * 32 + half_21 * 16 + lane_15 % 16;
                                                        int col_26 = col_tile_8 * 16 + lane_15 / 16 * 8;
                                                        unsigned int address_29 = d_smem_addr + (unsigned int)(chunk_17 % 3 * 8192) + (unsigned int)((row_34 * 32 + col_26) * 2);
                                                        address_29 = address_29 ^ (address_29 & 511) >> 7 << 4;
                                                        int offset_0_3 = chunk_17 * 16 + half_21 * 8 + col_tile_8 * 4;
                                                        uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(address_29);
                                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                            :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&packed_20[offset_0_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_20[offset_0_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_20[offset_0_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_20[offset_0_3 + 3]))
                                                            : "memory");
                                                    }
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    if (first_10 != 0) {
                                                        asm volatile(
                                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                            :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(y_10 * 8 + chunk_17), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_17 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    } else {
                                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                        #error "TmaReduceAdd5d requires SM90 or newer"
                                                        #endif
                                                        asm volatile(
                                                            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                            :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(y_10 * 8 + chunk_17), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_17 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    }
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
                                                        bool enabled_value_33 = macros > 1;
                                                        if (enabled_value_33 != 0) {
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
                                gemm_phase = phase_bits_16;
                            } else if (kind == 5) {
                                int col_blocks_17 = macro_size / 256;
                                {
                                    col_blocks_17 = hidden / 256;
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
                                expert_idx_5 = task_3 / (row_blocks_7 * col_blocks_17);
                                local_task_5 = task_3 % (row_blocks_7 * col_blocks_17);
                                int offset_13 = 0;
                                #pragma unroll 1
                                for (int index_6 = 0; index_6 < expert_idx_5; index_6++) {
                                    offset_13 = offset_13 + counts[index_6];
                                }
                                int _max_45 = ((offset_13) > (macro_1 * macro_size) ? (offset_13) : (macro_1 * macro_size));
                                k_start_11 = _max_45;
                                int _min_79 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                                int _min_80 = ((offset_13 + counts[expert_idx_5]) < (_min_79) ? (offset_13 + counts[expert_idx_5]) : (_min_79));
                                k_end_11 = _min_80;
                                first_11 = (int)(k_start_11 == offset_13);
                                if (k_start_11 < k_end_11) {
                                    int supergroup_11 = local_task_5 / (row_blocks_7 * 8);
                                    int full_cols_11 = col_blocks_17 / 8 * 8;
                                    int row_35 = 0;
                                    int col_27 = 0;
                                    if (local_task_5 < row_blocks_7 * full_cols_11) {
                                        row_35 = local_task_5 % (row_blocks_7 * 8) / 8;
                                        col_27 = supergroup_11 * 8 + local_task_5 % 8;
                                    } else {
                                        row_35 = (local_task_5 - row_blocks_7 * full_cols_11) / (col_blocks_17 - full_cols_11);
                                        col_27 = full_cols_11 + (local_task_5 - row_blocks_7 * full_cols_11) % (col_blocks_17 - full_cols_11);
                                    }
                                    if ((supergroup_11 & 1) != 0) {
                                        row_35 = row_blocks_7 - row_35 - 1;
                                    }
                                    x_11 = row_35;
                                    y_11 = col_27;
                                    expert_11 = expert_idx_5;
                                }
                                unsigned int phase_bits_17 = gemm_phase;
                                int global_mini_13 = macro_1 * (macro_size / mini_size);
                                int macro_rows_11 = macro_1 * (macro_size / 256);
                                int iterations_11 = intermediate / 128;
                                iterations_11 = (k_end_11 - k_start_11 + 127) / 128;
                                int k_blocks_6 = intermediate / 128;
                                k_blocks_6 = macro_size / 128;
                                int n_blocks_6 = hidden / 128;
                                if (expert_11 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_34 = macros > 1;
                                        if (enabled_value_34 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            int ring_28 = 0;
                                            #pragma unroll 1
                                            for (int idx_28 = 0; idx_28 < iterations_11; idx_28++) {
                                                int token_row_7 = k_start_11 + idx_28 * 128;
                                                if (idx_28 == 0 || token_row_7 % 256 == 0) {
                                                    bool enabled_value_35 = 1;
                                                    if (enabled_value_35 != 0) {
                                                        int32_t _relaxed_ld_79;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_79) : "l"(dg_ready + (shared_rows + token_row_7 / 256)) : "memory");
                                                        int value_28 = _relaxed_ld_79;
                                                        while (value_28 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_80;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_80) : "l"(dg_ready + (shared_rows + token_row_7 / 256)) : "memory");
                                                            value_28 = _relaxed_ld_80;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_28 == 0 || token_row_7 % mini_size == 0) {
                                                    int input_mini_7 = token_row_7 / mini_size;
                                                    int _min_82 = ((mini_size) < (tokens - input_mini_7 * mini_size) ? (mini_size) : (tokens - input_mini_7 * mini_size));
                                                    int input_rows_7 = _min_82;
                                                    int input_count_7 = (input_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_36 = macro_1 > 0;
                                                    if (enabled_value_36 != 0) {
                                                        int32_t _relaxed_ld_81;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_81) : "l"(replay_x + input_mini_7) : "memory");
                                                        int value_29 = _relaxed_ld_81;
                                                        while (value_29 < input_count_7) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_82;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_82) : "l"(replay_x + input_mini_7) : "memory");
                                                            value_29 = _relaxed_ld_82;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(gemm_finished_addr + (ring_28) * 8, phase_bits_17 >> (unsigned int)(16 + ring_28) & 1);
                                                int k_tile_4 = (k_start_11 + idx_28 * 128 - macro_1 * macro_size) / 128;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_fp8_smem_addr + (unsigned int)(ring_28 * 16384)), "l"((&du_t_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(k_tile_4), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_28) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_fp8_smem_addr + (unsigned int)(ring_28 * 16384)), "l"((&x_t_r)), "r"(0), "r"(y_11 * 256 + cta_rank_0 * 128), "r"(k_tile_4), "r"(0), "r"(0),
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
                                                int ring_29 = 0;
                                                #pragma unroll 1
                                                for (int idx_29 = 0; idx_29 < iterations_11; idx_29++) {
                                                    int token_row_8 = k_start_11 + idx_29 * 128;
                                                    if (idx_29 == 0 || token_row_8 % 256 == 0) {
                                                        bool enabled_value_37 = 1;
                                                        if (enabled_value_37 != 0) {
                                                            int32_t _relaxed_ld_87;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_87) : "l"(dg_ready + (shared_rows + token_row_8 / 256)) : "memory");
                                                            int value_30 = _relaxed_ld_87;
                                                            while (value_30 < row_count) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_88;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_88) : "l"(dg_ready + (shared_rows + token_row_8 / 256)) : "memory");
                                                                value_30 = _relaxed_ld_88;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    if (idx_29 == 0 || token_row_8 % mini_size == 0) {
                                                        int input_mini_8 = token_row_8 / mini_size;
                                                        int _min_84 = ((mini_size) < (tokens - input_mini_8 * mini_size) ? (mini_size) : (tokens - input_mini_8 * mini_size));
                                                        int input_rows_8 = _min_84;
                                                        int input_count_8 = (input_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                                        bool enabled_value_38 = macro_1 > 0;
                                                        if (enabled_value_38 != 0) {
                                                            int32_t _relaxed_ld_89;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_89) : "l"(replay_x + input_mini_8) : "memory");
                                                            int value_31 = _relaxed_ld_89;
                                                            while (value_31 < input_count_8) {
                                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                                int32_t _relaxed_ld_90;
                                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_90) : "l"(replay_x + input_mini_8) : "memory");
                                                                value_31 = _relaxed_ld_90;
                                                            }
                                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                                        }
                                                    }
                                                    mbarrier_wait(scales_finished_addr + (ring_29) * 8, phase_bits_17 >> (unsigned int)(23 + ring_29) & 1);
                                                    int k_tile_5 = (k_start_11 + idx_29 * 128 - macro_1 * macro_size) / 128;
                                                    int a_tile_7 = (x_11 * 2 + cta_rank_0) * k_blocks_6 + k_tile_5;
                                                    int b_tile_7 = (y_11 * 2 + cta_rank_0) * k_blocks_6 + k_tile_5;
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(a_sc_smem_addr + (unsigned int)(ring_29 * 512)), "l"((&du_sc_t_r)), "r"(0), "r"(a_tile_7 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_29) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                    asm volatile(
                                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                        :: "r"(b_sc_smem_addr + (unsigned int)(ring_29 * 1024) + (unsigned int)(cta_rank_0 * 512)), "l"((&x_sc_t_r)), "r"(0), "r"(b_tile_7 * 32), "r"(0),
                                                           "r"(((scales_arrived_addr + (ring_29) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                                    phase_bits_17 = phase_bits_17 ^ (unsigned int)(1 << 23 + ring_29);
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
                                                    mbarrier_wait(scales_arrived_addr + (ring_30) * 8, phase_bits_17 >> (unsigned int)(7 + ring_30) & 1);
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a + ring_30 * 4, make_sf_cp_desc_sbo128(a_sc_smem_addr + (unsigned int)(ring_30 * 512)));
                                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b + ring_30 * 8, make_sf_cp_desc_sbo128(b_sc_smem_addr + (unsigned int)(ring_30 * 1024)));
                                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + ring_30 * 8 + 4), make_sf_cp_desc_sbo128((b_sc_smem_addr + (unsigned int)(ring_30 * 1024) + 512)));
                                                    tcgen05_commit_cg2_multicast(scales_finished_addr + (ring_30) * 8, (uint16_t)(3));
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_30) * 8, 65536);
                                                    mbarrier_wait(gemm_arrived_addr + (ring_30) * 8, phase_bits_17 >> (unsigned int)ring_30 & 1);
                                                    int _mma_a_lo_11 = (((a_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_30) * 1024;
                                                    int _mma_b_lo_11 = (((b_fp8_smem_addr) >> 4) & 0x3FFF) + (ring_30) * 1024;
                                                    {
                                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x40004040 << 32);
                                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x40004040 << 32);

                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 0, b_desc + 0,
                                                            (0x10c00000U | ((0) << 29) | ((0) << 4)), tmem_sf_a + ring_30 * 4, tmem_sf_b + ring_30 * 8, ((idx_30 == 0) ? 0 : 1));
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 2, b_desc + 2,
                                                            (0x10c00000U | ((1) << 29) | ((1) << 4)), tmem_sf_a + ring_30 * 4, tmem_sf_b + ring_30 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 4, b_desc + 4,
                                                            (0x10c00000U | ((2) << 29) | ((2) << 4)), tmem_sf_a + ring_30 * 4, tmem_sf_b + ring_30 * 8, 1);
                                                        tcgen05_mma_mxf8_bs_cta2(tmem_accumulator, a_desc + 6, b_desc + 6,
                                                            (0x10c00000U | ((3) << 29) | ((3) << 4)), tmem_sf_a + ring_30 * 4, tmem_sf_b + ring_30 * 8, 1);
                                                    }
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_30) * 8, (uint16_t)(3));
                                                    phase_bits_17 = phase_bits_17 ^ (unsigned int)(1 << ring_30);
                                                    phase_bits_17 = phase_bits_17 ^ (unsigned int)(1 << 7 + ring_30);
                                                    ring_30 = (ring_30 + 1) % 6;
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else {
                                        if (tid < 128) {
                                            mbarrier_wait(output_arrived_addr, phase_bits_17 >> 6 & 1);
                                            phase_bits_17 = phase_bits_17 ^ 64;
                                            unsigned int packed_21[128];
                                            #pragma unroll
                                            for (int chunk_18 = 0; chunk_18 < 8; chunk_18++) {
                                                #pragma unroll
                                                for (int half_22 = 0; half_22 < 2; half_22++) {
                                                    unsigned int address_30 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_22 * 16 << 16) + (unsigned int)(chunk_18 * 32);
                                                    float _tmem_load_11[16];
                                                    asm volatile(
                                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                                                        : "r"(address_30));
                                                    #pragma unroll
                                                    for (int pair_21 = 0; pair_21 < 8; pair_21++) {
                                                        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_tmem_load_11[pair_21 * 2], _tmem_load_11[pair_21 * 2 + 1]));
                                                        packed_21[chunk_18 * 16 + half_22 * 8 + pair_21] = __as_u32(_bf16x2_24);
                                                    }
                                                }
                                            }
                                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                                int previous_offset_12 = (macro_1 + 1) * macro_size;
                                                int output_row_9 = x_11 * 256 + cta_rank_0 * 128;
                                                int _min_85 = ((macro_size) < (tokens - previous_offset_12) ? (macro_size) : (tokens - previous_offset_12));
                                                if (output_row_9 < _min_85) {
                                                }
                                            }
                                            #pragma unroll
                                            for (int chunk_19 = 0; chunk_19 < 8; chunk_19++) {
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                int warp_0_11 = tid / 32;
                                                int lane_16 = tid % 32;
                                                #pragma unroll
                                                for (int half_23 = 0; half_23 < 2; half_23++) {
                                                    #pragma unroll
                                                    for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                                        int row_36 = warp_0_11 * 32 + half_23 * 16 + lane_16 % 16;
                                                        int col_28 = col_tile_9 * 16 + lane_16 / 16 * 8;
                                                        unsigned int address_31 = d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192) + (unsigned int)((row_36 * 32 + col_28) * 2);
                                                        address_31 = address_31 ^ (address_31 & 511) >> 7 << 4;
                                                        int offset_0_4 = chunk_19 * 16 + half_23 * 8 + col_tile_9 * 4;
                                                        uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(address_31);
                                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                            :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&packed_21[offset_0_4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_21[offset_0_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_21[offset_0_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_21[offset_0_4 + 3]))
                                                            : "memory");
                                                    }
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                if (tid == 0) {
                                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                    if (first_11 != 0) {
                                                        asm volatile(
                                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                            :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(y_11 * 8 + chunk_19), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    } else {
                                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                                        #error "TmaReduceAdd5d requires SM90 or newer"
                                                        #endif
                                                        asm volatile(
                                                            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
                                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                            :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(y_11 * 8 + chunk_19), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                    }
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
                                                        bool enabled_value_39 = macros > 1;
                                                        if (enabled_value_39 != 0) {
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
            int kind_1 = -1;
            int task_2_1 = 0;
            int macro_3 = 0;
            int mini_4 = 0;
            int shared_5 = 0;
            if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
                if (shared_tasks > cluster - comm_clusters) {
                    shared_5 = 1;
                    if (shared_down > cluster - comm_clusters) {
                        kind_1 = 0;
                        task_2_1 = cluster - comm_clusters;
                    } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                        kind_1 = 1;
                        task_2_1 = cluster - comm_clusters - shared_down;
                    } else {
                        if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                            kind_1 = 2;
                            task_2_1 = cluster - comm_clusters - shared_down - shared_swiglu;
                        } else {
                            int weight_task_2 = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                            kind_1 = 3 + weight_task_2 / shared_wgrad;
                            task_2_1 = weight_task_2 % shared_wgrad;
                        }
                    }
                } else {
                    int routed_1 = cluster - comm_clusters - shared_tasks;
                    int macro_task_1 = routed_1;
                    int macro_minis_1 = saved_minis;
                    int replay_tasks_1 = 0;
                    if (routed_1 >= saved_tasks) {
                        macro_3 = 1 + (routed_1 - saved_tasks) / replay_macro_tasks;
                        macro_task_1 = (routed_1 - saved_tasks) % replay_macro_tasks;
                        int _min_86 = ((tokens - macro_3 * macro_size) < (macro_size) ? (tokens - macro_3 * macro_size) : (macro_size));
                        macro_minis_1 = (_min_86 + mini_size - 1) / mini_size;
                        replay_tasks_1 = macro_minis_1 * mini_replay;
                    }
                    if (macro_task_1 < replay_tasks_1) {
                        mini_4 = macro_task_1 / mini_replay;
                        int mini_task_2 = macro_task_1 % mini_replay;
                        if (mini_task_2 < mini_down) {
                            kind_1 = 6;
                            task_2_1 = mini_task_2;
                        } else if (mini_task_2 < 2 * mini_down) {
                            kind_1 = 7;
                            task_2_1 = mini_task_2 - mini_down;
                        } else {
                            kind_1 = 8;
                            task_2_1 = mini_task_2 - 2 * mini_down;
                        }
                    } else {
                        int bwd_task_1 = macro_task_1 - replay_tasks_1;
                        if (bwd_task_1 < macro_minis_1 * mini_bwd) {
                            mini_4 = bwd_task_1 / mini_bwd;
                            int mini_task_3 = bwd_task_1 % mini_bwd;
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
                            kind_1 = 3 + weight_task_3 / weight_tasks;
                            task_2_1 = weight_task_3 % weight_tasks;
                        }
                    }
                }
            }
            if ((kind == 1 || kind == 8) && cluster >= 0 && kind_1 != 1 && kind_1 != 8) {
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            }
            kind = kind_1;
            task_3 = task_2_1;
            macro_1 = macro_3;
            mini_1 = mini_4;
            shared = shared_5;
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
