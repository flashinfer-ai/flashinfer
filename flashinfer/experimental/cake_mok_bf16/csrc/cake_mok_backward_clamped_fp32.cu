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
#define SMEM_D_SMEM_OFF 206848
#define SMEM_D_SMEM_STAGE_BYTES 8192
#define SMEM_D_SMEM_STRIDE 8192
#define SMEM_D_WORDS_OFF 206848
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
#define SMEM_REPLAY_HIDDEN_OFF 197632
#define SMEM_REPLAY_HIDDEN_STAGE_BYTES 32768
#define SMEM_REPLAY_HIDDEN_STRIDE 32768
#define SMEM_DISPATCH_SMEM_OFF 1024
#define SMEM_DISPATCH_SMEM_STAGE_BYTES 196608
#define SMEM_DISPATCH_SMEM_STRIDE 196608
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 196608
#define SMEM_DISPATCH_WORDS_STRIDE 196608
#define SMEM_RANGE_WEIGHTS_OFF 197632
#define SMEM_RANGE_WEIGHTS_STAGE_BYTES 1536
#define SMEM_RANGE_WEIGHTS_STRIDE 1536
#define SMEM_RANGE_FLAGS_OFF 199168
#define SMEM_RANGE_FLAGS_STAGE_BYTES 512
#define SMEM_RANGE_FLAGS_STRIDE 512
#define SMEM_COMBINE_SCHEDULE_OFF 197632
#define SMEM_COMBINE_SCHEDULE_STAGE_BYTES 3072
#define SMEM_COMBINE_SCHEDULE_STRIDE 3072
#define SMEM_DISPATCH_WEIGHTS_OFF 199680
#define SMEM_DISPATCH_WEIGHTS_STAGE_BYTES 512
#define SMEM_DISPATCH_WEIGHTS_STRIDE 512
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 320

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
kernel_cake_mok_backward_clamped_fp32(const __grid_constant__ CUtensorMap dy_s, const __grid_constant__ CUtensorMap dy_r, const __grid_constant__ CUtensorMap dg_s, const __grid_constant__ CUtensorMap dg_r, const __grid_constant__ CUtensorMap du_s, const __grid_constant__ CUtensorMap du_r, const __grid_constant__ CUtensorMap x_nt_r, const __grid_constant__ CUtensorMap dy_atb_s, const __grid_constant__ CUtensorMap dy_atb_r, const __grid_constant__ CUtensorMap dg_atb_s, const __grid_constant__ CUtensorMap dg_atb_r, const __grid_constant__ CUtensorMap du_atb_s, const __grid_constant__ CUtensorMap du_atb_r, const __grid_constant__ CUtensorMap x_atb_s, const __grid_constant__ CUtensorMap x_atb_r, const __grid_constant__ CUtensorMap h_atb_s, const __grid_constant__ CUtensorMap h_atb_r, const __grid_constant__ CUtensorMap wg_s, const __grid_constant__ CUtensorMap wu_s, const __grid_constant__ CUtensorMap wd_s, const __grid_constant__ CUtensorMap wg_r, const __grid_constant__ CUtensorMap wu_r, const __grid_constant__ CUtensorMap wd_r, const __grid_constant__ CUtensorMap wg_nt_r, const __grid_constant__ CUtensorMap wu_nt_r, const __grid_constant__ CUtensorMap dh_s, const __grid_constant__ CUtensorMap dh_r, const __grid_constant__ CUtensorMap dx_s, const __grid_constant__ CUtensorMap dx_r, const __grid_constant__ CUtensorMap gate_out_r, const __grid_constant__ CUtensorMap up_out_r, const __grid_constant__ CUtensorMap dwg_s, const __grid_constant__ CUtensorMap dwu_s, const __grid_constant__ CUtensorMap dwd_s, const __grid_constant__ CUtensorMap dwg_r, const __grid_constant__ CUtensorMap dwu_r, const __grid_constant__ CUtensorMap dwd_r, const __grid_constant__ CUtensorMap dh_sw_s, const __grid_constant__ CUtensorMap dh_sw_r, const __grid_constant__ CUtensorMap gate_sw_s, const __grid_constant__ CUtensorMap gate_sw_r, const __grid_constant__ CUtensorMap up_sw_s, const __grid_constant__ CUtensorMap up_sw_r, const __grid_constant__ CUtensorMap dg_sw_s, const __grid_constant__ CUtensorMap dg_sw_r, const __grid_constant__ CUtensorMap du_sw_s, const __grid_constant__ CUtensorMap du_sw_r, const __grid_constant__ CUtensorMap h_sw_r, __nv_bfloat16* __restrict__ x_routed_ptr, __nv_bfloat16* __restrict__ dy_routed_ptr, __nv_bfloat16* __restrict__ dx_routed_ptr, float* __restrict__ weights, float* __restrict__ partials, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ dy_peers, unsigned long long* __restrict__ dx_peers, unsigned long long* __restrict__ weight_peers, unsigned long long* __restrict__ dweight_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ dh_ready, int* __restrict__ dg_ready, int* __restrict__ dy_ready, int* __restrict__ dx_ready, int* __restrict__ replay_x, int* __restrict__ replay_gu, int* __restrict__ replay_h, int* __restrict__ buffers_done, int* __restrict__ weight_ready, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit)
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
    #define gemm_finished_addr (mbar_base + 72)
    #define output_arrived_addr (mbar_base + 104)
    #define output_finished_addr (mbar_base + 112)
    #define schedule_arrived_addr (mbar_base + 120)
    #define schedule_finished_addr (mbar_base + 128)
    #define drain_arrived_0_addr (mbar_base + 136)
    #define drain_arrived_1_addr (mbar_base + 144)
    #define drain_arrived_2_addr (mbar_base + 152)
    #define drain_arrived_3_addr (mbar_base + 160)
    #define drain_arrived_4_addr (mbar_base + 168)
    #define drain_arrived_5_addr (mbar_base + 176)
    #define drain_arrived_6_addr (mbar_base + 184)
    #define drain_arrived_7_addr (mbar_base + 192)
    #define drain_finished_0_addr (mbar_base + 200)
    #define drain_finished_1_addr (mbar_base + 208)
    #define drain_finished_2_addr (mbar_base + 216)
    #define drain_finished_3_addr (mbar_base + 224)
    #define drain_finished_4_addr (mbar_base + 232)
    #define drain_finished_5_addr (mbar_base + 240)
    #define drain_finished_6_addr (mbar_base + 248)
    #define drain_finished_7_addr (mbar_base + 256)
    #define dispatch_arrived_addr (mbar_base + 264)
    #define range_arrived_addr (mbar_base + 272)
    #define rows_arrived_addr (mbar_base + 296)

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
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
    unsigned int* d_words = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int d_words_addr = smem + 206848;
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
    __nv_bfloat16* replay_hidden = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int replay_hidden_addr = smem + 197632;
    __nv_bfloat16* dispatch_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int dispatch_smem_addr = smem + 1024;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int dispatch_words_addr = smem + 1024;
    float* range_weights = reinterpret_cast<float*>(smem_raw + 197632);
    const int range_weights_addr = smem + 197632;
    int* range_flags = reinterpret_cast<int*>(smem_raw + 199168);
    const int range_flags_addr = smem + 199168;
    int* combine_schedule = reinterpret_cast<int*>(smem_raw + 197632);
    const int combine_schedule_addr = smem + 197632;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + 199680);
    const int dispatch_weights_addr = smem + 199680;
    int tokens = num_tokens[0];
    int shared_down = local_tokens / 256 * ((intermediate + 511) / 512);
    int shared_swiglu = (local_tokens / 128 * (intermediate / 128) + 16 - 1) / 16;
    int shared_dx = local_tokens / 256 * ((hidden + 511) / 512);
    int _max_0 = ((intermediate / 256 * ((hidden + 512 - 1) / 512)) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (intermediate / 256 * ((hidden + 512 - 1) / 512)) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
    int shared_wgrad = _max_0;
    int shared_tasks = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
    int mini_down = mini_size / 256 * ((intermediate + 511) / 512);
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 16 - 1) / 16;
    int mini_dx = mini_size / 256 * ((hidden + 511) / 512);
    int mini_replay_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int mini_bwd = mini_down + mini_swiglu + mini_dx;
    int mini_replay = 2 * mini_down + mini_replay_swiglu;
    int wgrad_tasks = 3 * experts * shared_wgrad;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis = (tokens + mini_size - 1) / mini_size;
    int _min_0 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
    int saved_minis = (_min_0 + mini_size - 1) / mini_size;
    int true_compute = shared_tasks + minis * mini_bwd + (minis - saved_minis) * mini_replay + macros * wgrad_tasks;
    int comm_clusters = comm_sms / 2;
    int true_clusters = comm_clusters + true_compute;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (27 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[0..320)

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
        // gemm_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 72, 1);
        mbarrier_init(smem + 80, 1);
        mbarrier_init(smem + 88, 1);
        mbarrier_init(smem + 96, 1);
        // output_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 104, 1);
        // output_finished: 1 barriers, init_count=2
        mbarrier_init(smem + 112, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 120, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 128, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 144, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 152, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 160, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 176, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 184, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 192, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 200, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 208, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 216, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 224, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 232, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 240, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 248, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 256, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 264, 1);
        // range_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 272, 1);
        mbarrier_init(smem + 280, 1);
        mbarrier_init(smem + 288, 1);
        // rows_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 296, 1);
        mbarrier_init(smem + 304, 1);
        mbarrier_init(smem + 312, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 320);
    if (warp == 0) {
        int _tmem_hold = smem + 320;
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
        int combine_units = 0;
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
        int col_blocks = (hidden + 511) / 512;
        int _min_2 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
        int unit_rows = _min_2;
        int block_units = (128 + unit_rows - 1) / unit_rows;
        int macro_offset = 0;
        int _min_3 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
        int macro_blocks = _min_3 / 128;
        int blocks = 0;
        if (macro_blocks > cluster * 2 + cta_rank_0) {
            blocks = (macro_blocks - (cluster * 2 + cta_rank_0) + comm_sms - 1) / comm_sms;
        }
        int units = blocks * block_units;
        int _min_4 = ((units) < (2) ? (units) : (2));
        #pragma unroll 1
        for (int unit = 0; unit < _min_4; unit++) {
            int buffer = unit % 3;
            int block = cluster * 2 + cta_rank_0 + unit / block_units * comm_sms;
            int first_row = block * 128 + unit % block_units * unit_rows;
            int _min_5 = ((unit_rows) < (block * 128 + 128 - first_row) ? (unit_rows) : (block * 128 + 128 - first_row));
            int rows_0 = _min_5;
            int peer_1 = -1;
            int peer_token = -1;
            if (rows_0 > tid) {
                peer_1 = schedule_rank[macro_offset + first_row + tid];
                peer_token = schedule_token[macro_offset + first_row + tid];
                range_flags[tid] = peer_1;
                range_weights[buffer * 128 + tid] = ((peer_1 >= 0) ? weights[first_row + tid] : 0.0f);
            }
            uint32_t _cta_count_0 = __syncthreads_count(peer_1 >= 0);
            if (tid == 0) {
                mbarrier_arrive_expect_tx(range_arrived_addr + (buffer) * 8, (unsigned int)(_cta_count_0 * (unsigned int)hidden * 2));
            }
            if (_cta_count_0 < (unsigned int)rows_0) {
                #pragma unroll 1
                for (int pad = 0; pad < rows_0; pad++) {
                    int pad_peer = range_flags[pad];
                    if (pad_peer < 0) {
                        #pragma unroll 1
                        for (int vec = tid; vec < hidden / 8; vec += 256) {
                            asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (buffer * 65536 + pad * hidden * 2 + vec * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                        }
                    }
                }
            }
            __syncthreads();
            if (peer_1 >= 0) {
                cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)((buffer * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden) * (unsigned long long)2)), hidden * 2, range_arrived_addr + (buffer) * 8);
            }
        }
        #pragma unroll 1
        for (int unit_1 = 0; unit_1 < units; unit_1++) {
            int buffer_1 = unit_1 % 3;
            int block_1 = cluster * 2 + cta_rank_0 + unit_1 / block_units * comm_sms;
            int first_row_1 = block_1 * 128 + unit_1 % block_units * unit_rows;
            int _min_6 = ((unit_rows) < (block_1 * 128 + 128 - first_row_1) ? (unit_rows) : (block_1 * 128 + 128 - first_row_1));
            int rows_0_1 = _min_6;
            mbarrier_wait(range_arrived_addr + (buffer_1) * 8, unit_1 / 3 & 1);
            __syncthreads();
            #pragma unroll 4
            for (int word = tid; word < rows_0_1 * (hidden / 2); word += 256) {
                float weight = range_weights[buffer_1 * 128 + word / (hidden / 2)];
                int index = buffer_1 * 16384 + word;
                unsigned int pair = dispatch_words[index];
                uint32_t _bf16x2_scale_0;
                {
                    uint32_t _bf16x2_pair_0 = pair;
                    float _bf16x2_lo_0;
                    float _bf16x2_hi_0;
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_0) : "h"((uint16_t)(_bf16x2_pair_0 & 0xFFFFu)));
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_0) : "h"((uint16_t)(_bf16x2_pair_0 >> 16)));
                    _bf16x2_lo_0 *= weight;
                    _bf16x2_hi_0 *= weight;
                    uint32_t _bf16x2_out_0;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_0) : "f"(_bf16x2_hi_0), "f"(_bf16x2_lo_0));
                    _bf16x2_scale_0 = _bf16x2_out_0;
                }
                dispatch_words[index] = _bf16x2_scale_0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid == 0) {
                {
                    void* _cpbulk_dst_1 = reinterpret_cast<void*>(dy_routed_ptr + ((unsigned long long)first_row_1 * (unsigned long long)hidden));
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                        :: "l"(_cpbulk_dst_1), "r"(dispatch_smem_addr + (unsigned int)(buffer_1 * 65536)), "r"((uint32_t)((unsigned int)(rows_0_1 * hidden * 2)))
                        : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;");
                if (units > unit_1 + 2) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                if (unit_1 >= 2 && (unit_1 - 2) % block_units == block_units - 1) {
                    asm volatile("cp.async.bulk.wait_group 2;");
                    int done_block = cluster * 2 + cta_rank_0 + (unit_1 - 2) / block_units * comm_sms;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset + done_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                }
            }
            if (units > unit_1 + 2) {
                int buffer_0 = (unit_1 + 2) % 3;
                int block_1_1 = cluster * 2 + cta_rank_0 + (unit_1 + 2) / block_units * comm_sms;
                int first_row_2 = block_1_1 * 128 + (unit_1 + 2) % block_units * unit_rows;
                int _min_7 = ((unit_rows) < (block_1_1 * 128 + 128 - first_row_2) ? (unit_rows) : (block_1_1 * 128 + 128 - first_row_2));
                int rows_3 = _min_7;
                int peer_2 = -1;
                int peer_token_1 = -1;
                if (rows_3 > tid) {
                    peer_2 = schedule_rank[macro_offset + first_row_2 + tid];
                    peer_token_1 = schedule_token[macro_offset + first_row_2 + tid];
                    range_flags[tid] = peer_2;
                    range_weights[buffer_0 * 128 + tid] = ((peer_2 >= 0) ? weights[first_row_2 + tid] : 0.0f);
                }
                uint32_t _cta_count_1 = __syncthreads_count(peer_2 >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(range_arrived_addr + (buffer_0) * 8, (unsigned int)(_cta_count_1 * (unsigned int)hidden * 2));
                }
                if (_cta_count_1 < (unsigned int)rows_3) {
                    #pragma unroll 1
                    for (int pad_1 = 0; pad_1 < rows_3; pad_1++) {
                        int pad_peer_1 = range_flags[pad_1];
                        if (pad_peer_1 < 0) {
                            #pragma unroll 1
                            for (int vec_1 = tid; vec_1 < hidden / 8; vec_1 += 256) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_words) + (buffer_0 * 65536 + pad_1 * hidden * 2 + vec_1 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                }
                __syncthreads();
                if (peer_2 >= 0) {
                    cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)((buffer_0 * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_2])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden) * (unsigned long long)2)), hidden * 2, range_arrived_addr + (buffer_0) * 8);
                }
            }
        }
        if (tid == 0) {
            asm volatile("cp.async.bulk.wait_group 0;");
            int _max_1 = ((0) > (units - 2) ? (0) : (units - 2));
            #pragma unroll 1
            for (int tail = _max_1; tail < units; tail++) {
                if (tail % block_units == block_units - 1) {
                    int tail_block = cluster * 2 + cta_rank_0 + tail / block_units * comm_sms;
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset + tail_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                }
            }
        }
        #pragma unroll 1
        for (int macro = 0; macro < macros; macro++) {
            int _min_8 = ((tokens - macro * macro_size) < (macro_size) ? (tokens - macro * macro_size) : (macro_size));
            int rows_0_2 = _min_8;
            int _min_9 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int unit_rows_1 = _min_9;
            int block_units_2 = (128 + unit_rows_1 - 1) / unit_rows_1;
            int macro_offset_3 = macro * macro_size;
            int _min_10 = ((macro_size) < (tokens - macro_offset_3) ? (macro_size) : (tokens - macro_offset_3));
            int macro_blocks_4 = _min_10 / 128;
            int blocks_5 = 0;
            if (macro_blocks_4 > cluster * 2 + cta_rank_0) {
                blocks_5 = (macro_blocks_4 - (cluster * 2 + cta_rank_0) + comm_sms - 1) / comm_sms;
            }
            int units_6 = blocks_5 * block_units_2;
            int _min_11 = ((units_6) < (2) ? (units_6) : (2));
            #pragma unroll 1
            for (int unit_2 = 0; unit_2 < _min_11; unit_2++) {
                int slot = (combine_units + unit_2) % 3;
                int block_2 = cluster * 2 + cta_rank_0 + unit_2 / block_units_2 * comm_sms;
                int first_row_3 = block_2 * 128 + unit_2 % block_units_2 * unit_rows_1;
                int _min_12 = ((unit_rows_1) < (block_2 * 128 + 128 - first_row_3) ? (unit_rows_1) : (block_2 * 128 + 128 - first_row_3));
                if (_min_12 > tid) {
                    combine_schedule[slot * 128 + tid] = schedule_rank[macro_offset_3 + first_row_3 + tid];
                    combine_schedule[384 + slot * 128 + tid] = schedule_token[macro_offset_3 + first_row_3 + tid];
                }
            }
            if (tid == 0) {
                int _min_13 = ((units_6) < (2) ? (units_6) : (2));
                #pragma unroll 1
                for (int unit_3 = 0; unit_3 < _min_13; unit_3++) {
                    int buffer_2 = (combine_units + unit_3) % 3;
                    int block_3 = cluster * 2 + cta_rank_0 + unit_3 / block_units_2 * comm_sms;
                    int first_row_4 = block_3 * 128 + unit_3 % block_units_2 * unit_rows_1;
                    int _min_14 = ((unit_rows_1) < (block_3 * 128 + 128 - first_row_4) ? (unit_rows_1) : (block_3 * 128 + 128 - first_row_4));
                    int rows_1 = _min_14;
                    int mini = (macro_offset_3 + first_row_4) / mini_size;
                    int _min_15 = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                    int mini_rows = _min_15;
                    int32_t _relaxed_ld_2;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_2) : "l"(dx_ready + mini) : "memory");
                    int value_2 = _relaxed_ld_2;
                    while (value_2 < (mini_rows + 255) / 256 * (hidden / 256) * 2) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int32_t _relaxed_ld_3;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_3) : "l"(dx_ready + mini) : "memory");
                        value_2 = _relaxed_ld_3;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_2) * 8, (unsigned int)(rows_1 * hidden * 2));
                    cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(buffer_2 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)first_row_4 * (unsigned long long)hidden) * (unsigned long long)2)), rows_1 * hidden * 2, rows_arrived_addr + (buffer_2) * 8);
                }
            }
            __syncthreads();
            #pragma unroll 1
            for (int unit_4 = 0; unit_4 < units_6; unit_4++) {
                int buffer_3 = (combine_units + unit_4) % 3;
                int block_4 = cluster * 2 + cta_rank_0 + unit_4 / block_units_2 * comm_sms;
                int first_row_5 = block_4 * 128 + unit_4 % block_units_2 * unit_rows_1;
                int _min_16 = ((unit_rows_1) < (block_4 * 128 + 128 - first_row_5) ? (unit_rows_1) : (block_4 * 128 + 128 - first_row_5));
                int rows_1_1 = _min_16;
                mbarrier_wait(rows_arrived_addr + (buffer_3) * 8, (combine_units + unit_4) / 3 & 1);
                if (rows_1_1 > tid) {
                    int peer_3 = combine_schedule[buffer_3 * 128 + tid];
                    int token_1 = combine_schedule[384 + buffer_3 * 128 + tid];
                    if (peer_3 >= 0) {
                        {
                            float gradient = 0.0f;
                            #pragma unroll 1
                            for (int col = 0; col < intermediate / 128; col++) {
                                gradient = gradient + partials[(first_row_5 + tid) * (intermediate / 128) + col];
                            }
                            reinterpret_cast<float*>(dweight_peers[peer_3])[token_1] = gradient;
                        }
                    }
                }
                if (tid == 0) {
                    #pragma unroll 1
                    for (int row_1 = 0; row_1 < rows_1_1; row_1++) {
                        int row_peer = combine_schedule[buffer_3 * 128 + row_1];
                        int row_token = combine_schedule[384 + buffer_3 * 128 + row_1];
                        if (row_peer >= 0) {
                            {
                                void* _cpbulk_dst_2 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(dx_peers[row_peer]) + ((unsigned long long)row_token * (unsigned long long)hidden));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_2), "r"(dispatch_smem_addr + (unsigned int)(buffer_3 * 65536) + (unsigned int)(row_1 * hidden * 2)), "r"((uint32_t)((unsigned int)(hidden * 2)))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    if (units_6 > unit_4 + 2) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        int buffer_0_1 = (combine_units + (unit_4 + 2)) % 3;
                        int block_1_2 = cluster * 2 + cta_rank_0 + (unit_4 + 2) / block_units_2 * comm_sms;
                        int first_row_2_1 = block_1_2 * 128 + (unit_4 + 2) % block_units_2 * unit_rows_1;
                        int _min_17 = ((unit_rows_1) < (block_1_2 * 128 + 128 - first_row_2_1) ? (unit_rows_1) : (block_1_2 * 128 + 128 - first_row_2_1));
                        int rows_3_1 = _min_17;
                        int mini_1 = (macro_offset_3 + first_row_2_1) / mini_size;
                        int _min_18 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                        int mini_rows_1 = _min_18;
                        int32_t _relaxed_ld_4;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(dx_ready + mini_1) : "memory");
                        int value_3 = _relaxed_ld_4;
                        while (value_3 < (mini_rows_1 + 255) / 256 * (hidden / 256) * 2) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int32_t _relaxed_ld_5;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(dx_ready + mini_1) : "memory");
                            value_3 = _relaxed_ld_5;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_0_1) * 8, (unsigned int)(rows_3_1 * hidden * 2));
                        cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(buffer_0_1 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dx_routed_ptr) + ((unsigned long long)((unsigned long long)first_row_2_1 * (unsigned long long)hidden) * (unsigned long long)2)), rows_3_1 * hidden * 2, rows_arrived_addr + (buffer_0_1) * 8);
                    }
                }
                if (units_6 > unit_4 + 2) {
                    int slot_1 = (combine_units + (unit_4 + 2)) % 3;
                    int block_0 = cluster * 2 + cta_rank_0 + (unit_4 + 2) / block_units_2 * comm_sms;
                    int first_row_1_1 = block_0 * 128 + (unit_4 + 2) % block_units_2 * unit_rows_1;
                    int _min_19 = ((unit_rows_1) < (block_0 * 128 + 128 - first_row_1_1) ? (unit_rows_1) : (block_0 * 128 + 128 - first_row_1_1));
                    if (_min_19 > tid) {
                        combine_schedule[slot_1 * 128 + tid] = schedule_rank[macro_offset_3 + first_row_1_1 + tid];
                        combine_schedule[384 + slot_1 * 128 + tid] = schedule_token[macro_offset_3 + first_row_1_1 + tid];
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
                {
                    bool enabled_value = macros > 1;
                    if (enabled_value != 0) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            combine_units = combine_units + units_6;
            if (macros > macro + 1) {
                int minis_0 = (rows_0_2 + mini_size - 1) / mini_size;
                int required = 2 * (minis_0 * mini_bwd + wgrad_tasks) + comm_sms;
                if (tid == 0) {
                    bool enabled_value_1 = 1;
                    if (enabled_value_1 != 0) {
                        int32_t _relaxed_ld_6;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(buffers_done + macro) : "memory");
                        int value_4 = _relaxed_ld_6;
                        while (value_4 < required) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int32_t _relaxed_ld_7;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(buffers_done + macro) : "memory");
                            value_4 = _relaxed_ld_7;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                    }
                }
                __syncthreads();
                int offset_1 = (macro + 1) * macro_size;
                int _min_20 = ((macro_size) < (tokens - offset_1) ? (macro_size) : (tokens - offset_1));
                int rows_2 = _min_20;
                #pragma unroll 1
                for (int row_2 = (cluster * 2 + cta_rank_0) * 256 + tid; row_2 < rows_2; row_2 += comm_sms * 256) {
                    int peer_4 = schedule_rank[offset_1 + row_2];
                    int token_2 = schedule_token[offset_1 + row_2];
                    float value_5 = 0.0f;
                    if (peer_4 >= 0) {
                        value_5 = reinterpret_cast<float*>(weight_peers[peer_4])[(int)token_2];
                    }
                    weights[row_2] = value_5;
                }
                __syncthreads();
                if (tid == 0) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(weight_ready)) + (macro + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    int32_t _relaxed_ld_8;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(weight_ready + (macro + 1)) : "memory");
                    int value_6 = _relaxed_ld_8;
                    while (value_6 < comm_sms) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int32_t _relaxed_ld_9;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(weight_ready + (macro + 1)) : "memory");
                        value_6 = _relaxed_ld_9;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                }
                __syncthreads();
                int _min_21 = ((tokens - (macro + 1) * macro_size) < (macro_size) ? (tokens - (macro + 1) * macro_size) : (macro_size));
                int next_rows = _min_21;
                #pragma unroll 1
                for (int task = cluster * 2 + cta_rank_0; task < next_rows / 128 * ((hidden + 511) / 512); task += comm_sms) {
                    unsigned int phase_bits = dispatch_bits;
                    int col_blocks_0 = (hidden + 511) / 512;
                    int macro_offset_1 = (macro + 1) * macro_size;
                    int _min_22 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                    int macro_tokens = _min_22;
                    if (task < macro_tokens / 128 * col_blocks_0) {
                        int row_3 = task / col_blocks_0 * 128;
                        int col_block = task % col_blocks_0;
                        int _min_23 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                        int chunk_cols = _min_23;
                        unsigned int chunk_bytes = (unsigned int)(chunk_cols * 2);
                        int peer_5 = -1;
                        int peer_token_2 = -1;
                        if (tid < 128) {
                            peer_5 = schedule_rank[macro_offset_1 + row_3 + tid];
                            peer_token_2 = schedule_token[macro_offset_1 + row_3 + tid];
                        }
                        uint32_t _cta_count_2 = __syncthreads_count(peer_5 >= 0);
                        if (tid == 0) {
                            int previous_offset = -1 * macro_size;
                            int _min_24 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                            int previous_tokens = _min_24;
                            if (row_3 < previous_tokens) {
                                int previous_mini = (previous_offset + row_3) / mini_size;
                                int _min_25 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                int mini_rows_2 = _min_25;
                                int required_0 = (mini_rows_2 + 255) / 256 * (hidden / 256) * 2;
                            }
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * chunk_bytes);
                        }
                        __syncthreads();
                        if (tid < 128) {
                            dispatch_weights[tid] = ((peer_5 >= 0) ? weights[row_3 + tid] : 0.0f);
                        }
                        if (peer_5 >= 0) {
                            cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(dy_peers[peer_5])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2)), chunk_cols * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_smem) + (tid * 1024 + vec_2 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        mbarrier_wait(dispatch_arrived_addr, phase_bits & 1);
                        phase_bits = phase_bits ^ 1;
                        __syncthreads();
                        if (tid < chunk_cols / 2) {
                            #pragma unroll 1
                            for (int scale_row = 0; scale_row < 128; scale_row++) {
                                float weight_1 = dispatch_weights[scale_row];
                                unsigned int pair_1 = dispatch_words[scale_row * 256 + tid];
                                uint32_t _bf16x2_scale_1;
                                {
                                    uint32_t _bf16x2_pair_3 = pair_1;
                                    float _bf16x2_lo_3;
                                    float _bf16x2_hi_3;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_3) : "h"((uint16_t)(_bf16x2_pair_3 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_3) : "h"((uint16_t)(_bf16x2_pair_3 >> 16)));
                                    _bf16x2_lo_3 *= weight_1;
                                    _bf16x2_hi_3 *= weight_1;
                                    uint32_t _bf16x2_out_3;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_3) : "f"(_bf16x2_hi_3), "f"(_bf16x2_lo_3));
                                    _bf16x2_scale_1 = _bf16x2_out_3;
                                }
                                dispatch_words[scale_row * 256 + tid] = _bf16x2_scale_1;
                            }
                        }
                        __syncthreads();
                        if (tid < 128) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_4 = reinterpret_cast<void*>(dy_routed_ptr + ((unsigned long long)(row_3 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_4), "r"(dispatch_smem_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group 0;");
                        }
                        __syncthreads();
                        if (tid == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dy_ready)) + ((macro_offset_1 + row_3) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                    dispatch_bits = phase_bits;
                    unsigned int phase_bits_2 = dispatch_bits;
                    int col_blocks_3 = (hidden + 511) / 512;
                    int macro_offset_4 = (macro + 1) * macro_size;
                    int _min_26 = ((macro_size) < (tokens - macro_offset_4) ? (macro_size) : (tokens - macro_offset_4));
                    int macro_tokens_5 = _min_26;
                    if (task < macro_tokens_5 / 128 * col_blocks_3) {
                        int row_4 = task / col_blocks_3 * 128;
                        int col_block_1 = task % col_blocks_3;
                        int _min_27 = ((512) < (hidden - col_block_1 * 512) ? (512) : (hidden - col_block_1 * 512));
                        int chunk_cols_1 = _min_27;
                        unsigned int chunk_bytes_1 = (unsigned int)(chunk_cols_1 * 2);
                        int peer_6 = -1;
                        int peer_token_3 = -1;
                        if (tid < 128) {
                            peer_6 = schedule_rank[macro_offset_4 + row_4 + tid];
                            peer_token_3 = schedule_token[macro_offset_4 + row_4 + tid];
                        }
                        uint32_t _cta_count_3 = __syncthreads_count(peer_6 >= 0);
                        if (tid == 0) {
                            int previous_offset_1 = -1 * macro_size;
                            int _min_28 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            int previous_tokens_1 = _min_28;
                            if (row_4 < previous_tokens_1) {
                                int previous_mini_1 = (previous_offset_1 + row_4) / mini_size;
                                int _min_29 = ((mini_size) < (tokens - previous_mini_1 * mini_size) ? (mini_size) : (tokens - previous_mini_1 * mini_size));
                                int mini_rows_3 = _min_29;
                                int required_0_1 = (mini_rows_3 + 255) / 256 * (hidden / 256) * 2;
                            }
                            mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_3 * chunk_bytes_1);
                        }
                        __syncthreads();
                        if (peer_6 >= 0) {
                            cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_6])) + ((unsigned long long)((unsigned long long)(peer_token_3 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)) * (unsigned long long)2)), chunk_cols_1 * 2, dispatch_arrived_addr);
                        } else if (tid < 128) {
                            #pragma unroll
                            for (int vec_3 = 0; vec_3 < 64; vec_3++) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_smem) + (tid * 1024 + vec_3 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                        mbarrier_wait(dispatch_arrived_addr, phase_bits_2 & 1);
                        phase_bits_2 = phase_bits_2 ^ 1;
                        if (tid < 128) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                void* _cpbulk_dst_5 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row_4 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_5), "r"(dispatch_smem_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes_1))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group 0;");
                        }
                        __syncthreads();
                        if (tid == 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_x)) + ((macro_offset_4 + row_4) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                    dispatch_bits = phase_bits_2;
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
            int mini_replay_2 = 2 * mini_down + mini_replay_swiglu;
            int weight_tasks = experts * shared_wgrad;
            int _min_30 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
            int saved_minis_3 = (_min_30 + mini_size - 1) / mini_size;
            int saved_tasks = saved_minis_3 * mini_bwd_1 + 3 * weight_tasks;
            int replay_macro_tasks = macro_size / mini_size * (mini_replay_2 + mini_bwd_1) + 3 * weight_tasks;
            int kind = -1;
            int task_1 = 0;
            int macro_1 = 0;
            int mini_2 = 0;
            int shared = 0;
            if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
                if (shared_tasks_0 > cluster - comm_clusters) {
                    shared = 1;
                    if (shared_down > cluster - comm_clusters) {
                        kind = 0;
                        task_1 = cluster - comm_clusters;
                    } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                        kind = 1;
                        task_1 = cluster - comm_clusters - shared_down;
                    } else {
                        if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                            kind = 2;
                            task_1 = cluster - comm_clusters - shared_down - shared_swiglu;
                        } else {
                            int weight_task = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                            kind = 3 + weight_task / shared_wgrad;
                            task_1 = weight_task % shared_wgrad;
                        }
                    }
                } else {
                    int routed = cluster - comm_clusters - shared_tasks_0;
                    int macro_task = routed;
                    int replay_tasks = 0;
                    if (routed >= saved_tasks) {
                        macro_1 = 1 + (routed - saved_tasks) / replay_macro_tasks;
                        macro_task = (routed - saved_tasks) % replay_macro_tasks;
                        int _min_31 = ((tokens - macro_1 * macro_size) < (macro_size) ? (tokens - macro_1 * macro_size) : (macro_size));
                        int macro_minis = (_min_31 + mini_size - 1) / mini_size;
                        replay_tasks = macro_minis * mini_replay_2;
                    }
                    int _min_32 = ((tokens - macro_1 * macro_size) < (macro_size) ? (tokens - macro_1 * macro_size) : (macro_size));
                    int macro_minis_1 = (_min_32 + mini_size - 1) / mini_size;
                    if (macro_task < replay_tasks) {
                        mini_2 = macro_task / mini_replay_2;
                        int mini_task = macro_task % mini_replay_2;
                        if (mini_task < mini_down) {
                            kind = 6;
                            task_1 = mini_task;
                        } else if (mini_task < 2 * mini_down) {
                            kind = 7;
                            task_1 = mini_task - mini_down;
                        } else {
                            kind = 8;
                            task_1 = mini_task - 2 * mini_down;
                        }
                    } else {
                        int bwd_task = macro_task - replay_tasks;
                        if (bwd_task < macro_minis_1 * mini_bwd_1) {
                            mini_2 = bwd_task / mini_bwd_1;
                            int mini_task_1 = bwd_task % mini_bwd_1;
                            if (mini_task_1 < mini_down) {
                                kind = 0;
                                task_1 = mini_task_1;
                            } else if (mini_task_1 < mini_down + mini_swiglu) {
                                kind = 1;
                                task_1 = mini_task_1 - mini_down;
                            } else {
                                kind = 2;
                                task_1 = mini_task_1 - mini_down - mini_swiglu;
                            }
                        } else {
                            int weight_task_1 = bwd_task - macro_minis_1 * mini_bwd_1;
                            kind = 3 + weight_task_1 / weight_tasks;
                            task_1 = weight_task_1 % weight_tasks;
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
            if (shared != 0) {
                if (kind == 0) {
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
                    if (task_1 < row_blocks * col_blocks_1) {
                        int supergroup = task_1 / (row_blocks * 8);
                        int full_cols = col_blocks_1 / 8 * 8;
                        int row_5 = 0;
                        int col_1 = 0;
                        if (task_1 < row_blocks * full_cols) {
                            row_5 = task_1 % (row_blocks * 8) / 8;
                            col_1 = supergroup * 8 + task_1 % 8;
                        } else {
                            row_5 = (task_1 - row_blocks * full_cols) / (col_blocks_1 - full_cols);
                            col_1 = full_cols + (task_1 - row_blocks * full_cols) % (col_blocks_1 - full_cols);
                        }
                        if ((supergroup & 1) != 0) {
                            row_5 = row_blocks - row_5 - 1;
                        }
                        x = row_5;
                        y = col_1;
                        expert = 0;
                    }
                    unsigned int phase_bits_1 = gemm_phase;
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
                                    int _min_33 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                    int _max_2 = ((0) > (_min_33) ? (0) : (_min_33));
                                    int mini_rows_4 = _max_2;
                                    int required_1 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                }
                                int ring = 0;
                                #pragma unroll 1
                                for (int idx = 0; idx < iterations; idx++) {
                                    mbarrier_wait(gemm_finished_addr + (ring) * 8, phase_bits_1 >> (unsigned int)(16 + ring) & 1);
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
                                    for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[pair_2 * 2], _tmem_load_0[pair_2 * 2 + 1]));
                                        packed[chunk * 16 + sub * 8 + pair_2] = __as_u32(_bf16x2_0);
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
                                int previous_offset_2 = macro_size;
                                int output_row = x * 256 + cta_rank_0 * 128;
                                int _min_34 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                if (output_row < _min_34) {
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
                                    for (int col_tile = 0; col_tile < 2; col_tile++) {
                                        int row_6 = warp_0 * 32 + half * 16 + lane_1 % 16;
                                        int col_2 = col_tile * 16 + lane_1 / 16 * 8;
                                        unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_6 * 32 + col_2) * 2);
                                        address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                        int offset_2 = chunk_1 * 16 + half * 8 + col_tile * 4;
                                        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(address_1);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_2 + 3]))
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
                                        for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_1[pair_3 * 2], _tmem_load_1[pair_3 * 2 + 1]));
                                            packed_0[chunk_2 * 16 + sub_1 * 8 + pair_3] = __as_u32(_bf16x2_1);
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
                                        for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                            int row_7 = warp_0_1 * 32 + half_1 * 16 + lane_2 % 16;
                                            int col_3 = col_tile_1 * 16 + lane_2 / 16 * 8;
                                            unsigned int address_3 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_7 * 32 + col_3) * 2);
                                            address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                            int offset_3 = chunk_3 * 16 + half_1 * 8 + col_tile_1 * 4;
                                            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(address_3);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_3 + 3]))
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
                    gemm_phase = phase_bits_1;
                } else if (kind == 1) {
                    unsigned int phase_bits_3 = swiglu_phase;
                    int col_blocks_2 = intermediate / 128;
                    int num_tiles = local_tokens / 128 * col_blocks_2;
                    int macro_row_offset = 0;
                    int first_tile = task_1 * 16 + cta_rank_0 * 8;
                    int tile_end = num_tiles;
                    int _min_36 = ((8) < (tile_end - first_tile) ? (8) : (tile_end - first_tile));
                    int _max_3 = ((0) > (_min_36) ? (0) : (_min_36));
                    int tiles = _max_3;
                    if (tiles > 0) {
                        if (tid == 0) {
                            int _min_37 = ((tiles) < (2) ? (tiles) : (2));
                            #pragma unroll 1
                            for (int stage = 0; stage < _min_37; stage++) {
                                int col_blocks_0_1 = intermediate / 128;
                                int row_8 = (first_tile + stage) / col_blocks_0_1;
                                int col_4 = (first_tile + stage) % col_blocks_0_1;
                                mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage) * 8, 98304);
                                int parent = row_8 / 2 * (intermediate / 256) + col_4 / 2;
                                int32_t _relaxed_ld_10;
                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(dh_ready + parent) : "memory");
                                int value_7 = _relaxed_ld_10;
                                while (value_7 < 2) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                    int32_t _relaxed_ld_11;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(dh_ready + parent) : "memory");
                                    value_7 = _relaxed_ld_11;
                                }
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_dh_addr + (unsigned int)(stage * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_8 - macro_row_offset) * 128), "r"(col_4 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_gate_addr + (unsigned int)(stage * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_8 - macro_row_offset) * 128), "r"(col_4 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage) * 8) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                    :: "r"(sw_up_addr + (unsigned int)(stage * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_8 - macro_row_offset) * 128), "r"(col_4 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage) * 8) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int index_1 = 0; index_1 < tiles; index_1++) {
                            int stage_1 = index_1 % 2;
                            mbarrier_wait(swiglu_arrived_addr + (stage_1) * 8, phase_bits_3 >> (unsigned int)stage_1 & 1);
                            phase_bits_3 = phase_bits_3 ^ (unsigned int)(1 << stage_1);
                            int row_9 = (first_tile + index_1) / col_blocks_2;
                            int col_5 = (first_tile + index_1) % col_blocks_2;
                            float gate[64];
                            float up[64];
                            float dhidden[64];
                            int warp_0_2 = tid / 32;
                            int local_warp = warp_0_2 / 4 + warp_0_2 % 4 * 2;
                            int lane_3 = tid % 32;
                            #pragma unroll
                            for (int tile_col = 0; tile_col < 8; tile_col++) {
                                unsigned int packed_1[4];
                                unsigned int address_4 = sw_gate_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col * 16 + lane_3 / 16 * 8) / 64 * 128 * 64 + (local_warp * 16 + lane_3 % 16) * 64 + (tile_col * 16 + lane_3 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_1[0]), "=r"(packed_1[1]), "=r"(packed_1[2]), "=r"(packed_1[3])
                                    : "r"(address_4 ^ (address_4 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                    float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(packed_1[pair_4]));
                                    gate[tile_col * 8 + pair_4 * 2] = _cvt_f32_0.x;
                                    gate[tile_col * 8 + pair_4 * 2 + 1] = _cvt_f32_0.y;
                                }
                            }
                            int warp_1 = tid / 32;
                            int local_warp_2 = warp_1 / 4 + warp_1 % 4 * 2;
                            int lane_3_1 = tid % 32;
                            #pragma unroll
                            for (int tile_col_1 = 0; tile_col_1 < 8; tile_col_1++) {
                                unsigned int packed_2[4];
                                unsigned int address_5 = sw_up_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col_1 * 16 + lane_3_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_2 * 16 + lane_3_1 % 16) * 64 + (tile_col_1 * 16 + lane_3_1 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_2[0]), "=r"(packed_2[1]), "=r"(packed_2[2]), "=r"(packed_2[3])
                                    : "r"(address_5 ^ (address_5 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_5 = 0; pair_5 < 4; pair_5++) {
                                    float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(packed_2[pair_5]));
                                    up[tile_col_1 * 8 + pair_5 * 2] = _cvt_f32_1.x;
                                    up[tile_col_1 * 8 + pair_5 * 2 + 1] = _cvt_f32_1.y;
                                }
                            }
                            int warp_4 = tid / 32;
                            int local_warp_5 = warp_4 / 4 + warp_4 % 4 * 2;
                            int lane_6 = tid % 32;
                            #pragma unroll
                            for (int tile_col_2 = 0; tile_col_2 < 8; tile_col_2++) {
                                unsigned int packed_3[4];
                                unsigned int address_6 = sw_dh_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(packed_3[0]), "=r"(packed_3[1]), "=r"(packed_3[2]), "=r"(packed_3[3])
                                    : "r"(address_6 ^ (address_6 & 1023) >> 7 << 4)
                                    : "memory");
                                #pragma unroll
                                for (int pair_6 = 0; pair_6 < 4; pair_6++) {
                                    float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(packed_3[pair_6]));
                                    dhidden[tile_col_2 * 8 + pair_6 * 2] = _cvt_f32_2.x;
                                    dhidden[tile_col_2 * 8 + pair_6 * 2 + 1] = _cvt_f32_2.y;
                                }
                            }
                            #pragma unroll
                            for (int elem = 0; elem < 64; elem++) {
                                float gate_mask = ((gate[elem] <= swiglu_limit) ? 1.0f : 0.0f);
                                float up_mask = ((up[elem] >= -swiglu_limit && up[elem] <= swiglu_limit) ? 1.0f : 0.0f);
                                float _min_38 = fminf(gate[elem], swiglu_limit);
                                float clamped_gate = _min_38;
                                float _fmax_0 = fmaxf(up[elem], -swiglu_limit);
                                float _min_39 = fminf(_fmax_0, swiglu_limit);
                                float clamped_up = _min_39;
                                float _exp_0 = expf(-clamped_gate);
                                float sigmoid = 1.0f / (1.0f + _exp_0);
                                float silu = clamped_gate * sigmoid;
                                float dsilu = (1.0f - silu) * sigmoid + silu;
                                float dgate = ((gate_mask != 0.0f) ? dsilu * clamped_up * dhidden[elem] : 0.0f);
                                float dup = ((up_mask != 0.0f) ? silu * dhidden[elem] : 0.0f);
                                dhidden[elem] = dgate;
                                gate[elem] = dup;
                            }
                            int warp_7 = tid / 32;
                            int local_warp_8 = warp_7 / 4 + warp_7 % 4 * 2;
                            int lane_9 = tid % 32;
                            #pragma unroll
                            for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                unsigned int packed_4[4];
                                #pragma unroll
                                for (int pair_7 = 0; pair_7 < 4; pair_7++) {
                                    __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(dhidden[tile_col_3 * 8 + pair_7 * 2], dhidden[tile_col_3 * 8 + pair_7 * 2 + 1]));
                                    packed_4[pair_7] = __as_u32(_bf16x2_2);
                                }
                                unsigned int address_7 = sw_gate_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_9 / 16 * 8) / 64 * 128 * 64 + (local_warp_8 * 16 + lane_9 % 16) * 64 + (tile_col_3 * 16 + lane_9 / 16 * 8) % 64) * 2);
                                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(address_7 ^ (address_7 & 1023) >> 7 << 4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
                                    : "memory");
                            }
                            int warp_10 = tid / 32;
                            int local_warp_11 = warp_10 / 4 + warp_10 % 4 * 2;
                            int lane_12 = tid % 32;
                            #pragma unroll
                            for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                unsigned int packed_5[4];
                                #pragma unroll
                                for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                    __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(gate[tile_col_4 * 8 + pair_8 * 2], gate[tile_col_4 * 8 + pair_8 * 2 + 1]));
                                    packed_5[pair_8] = __as_u32(_bf16x2_3);
                                }
                                unsigned int address_8 = sw_up_addr + (unsigned int)(stage_1 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_12 / 16 * 8) / 64 * 128 * 64 + (local_warp_11 * 16 + lane_12 % 16) * 64 + (tile_col_4 * 16 + lane_12 / 16 * 8) % 64) * 2);
                                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(address_8 ^ (address_8 & 1023) >> 7 << 4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[3]))
                                    : "memory");
                            }
                            __syncthreads();
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&dg_sw_s), 0, (row_9 - macro_row_offset) * 128, col_5 * 2, 0, 0, sw_gate_addr + (unsigned int)(stage_1 * 32768));
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_5d((&du_sw_s), 0, (row_9 - macro_row_offset) * 128, col_5 * 2, 0, 0, sw_up_addr + (unsigned int)(stage_1 * 32768));
                                asm volatile("cp.async.bulk.commit_group;");
                                if (tiles > index_1 + 2) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                    int col_blocks_0_2 = intermediate / 128;
                                    int row_1_1 = (first_tile + index_1 + 2) / col_blocks_0_2;
                                    int col_2_1 = (first_tile + index_1 + 2) % col_blocks_0_2;
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_1) * 8, 98304);
                                    int parent_1 = row_1_1 / 2 * (intermediate / 256) + col_2_1 / 2;
                                    int32_t _relaxed_ld_12;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(dh_ready + parent_1) : "memory");
                                    int value_8 = _relaxed_ld_12;
                                    while (value_8 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_13;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(dh_ready + parent_1) : "memory");
                                        value_8 = _relaxed_ld_13;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_dh_addr + (unsigned int)(stage_1 * 32768)), "l"((&dh_sw_s)), "r"(0), "r"((row_1_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_1) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_gate_addr + (unsigned int)(stage_1 * 32768)), "l"((&gate_sw_s)), "r"(0), "r"((row_1_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_1) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_up_addr + (unsigned int)(stage_1 * 32768)), "l"((&up_sw_s)), "r"(0), "r"((row_1_1 - macro_row_offset) * 128), "r"(col_2_1 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_1) * 8) : "memory");
                                }
                            }
                            __syncthreads();
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll 1
                            for (int index_2 = 0; index_2 < tiles; index_2++) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + ((first_tile + index_2) / col_blocks_2 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                    }
                    if (tid == 0) {
                    }
                    swiglu_phase = phase_bits_3;
                } else {
                    if (kind == 2) {
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
                        if (task_1 < row_blocks_1 * col_blocks_4) {
                            int supergroup_1 = task_1 / (row_blocks_1 * 8);
                            int full_cols_1 = col_blocks_4 / 8 * 8;
                            int row_10 = 0;
                            int col_6 = 0;
                            if (task_1 < row_blocks_1 * full_cols_1) {
                                row_10 = task_1 % (row_blocks_1 * 8) / 8;
                                col_6 = supergroup_1 * 8 + task_1 % 8;
                            } else {
                                row_10 = (task_1 - row_blocks_1 * full_cols_1) / (col_blocks_4 - full_cols_1);
                                col_6 = full_cols_1 + (task_1 - row_blocks_1 * full_cols_1) % (col_blocks_4 - full_cols_1);
                            }
                            if ((supergroup_1 & 1) != 0) {
                                row_10 = row_blocks_1 - row_10 - 1;
                            }
                            x_1 = row_10;
                            y_1 = col_6;
                            expert_1 = 0;
                        }
                        unsigned int phase_bits_4 = gemm_phase;
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
                                            int32_t _relaxed_ld_14;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                            int value_9 = _relaxed_ld_14;
                                            while (value_9 < row_count) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_15;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(dg_ready + (macro_rows_1 + x_1)) : "memory");
                                                value_9 = _relaxed_ld_15;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        int _min_40 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                        int _max_4 = ((0) > (_min_40) ? (0) : (_min_40));
                                        int mini_rows_5 = _max_4;
                                        int required_2 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
                                    }
                                    int ring_2 = 0;
                                    #pragma unroll 1
                                    for (int idx_2 = 0; idx_2 < iterations_1; idx_2++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_2) * 8, phase_bits_4 >> (unsigned int)(16 + ring_2) & 1);
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
                                        phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_2);
                                        ring_2 = (ring_2 + 1) % 4;
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
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_3) * 8, 98304);
                                            mbarrier_wait(gemm_arrived_addr + (ring_3) * 8, phase_bits_4 >> (unsigned int)ring_3 & 1);
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
                                            phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << ring_3);
                                            ring_3 = (ring_3 + 1) % 4;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_4 >> 6 & 1);
                                phase_bits_4 = phase_bits_4 ^ 64;
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
                                        for (int pair_9 = 0; pair_9 < 8; pair_9++) {
                                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_2[pair_9 * 2], _tmem_load_2[pair_9 * 2 + 1]));
                                            packed_6[chunk_4 * 16 + sub_2 * 8 + pair_9] = __as_u32(_bf16x2_4);
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
                                    int previous_offset_3 = macro_size;
                                    int output_row_1 = x_1 * 256 + cta_rank_0 * 128;
                                    int _min_41 = ((macro_size) < (tokens - previous_offset_3) ? (macro_size) : (tokens - previous_offset_3));
                                    if (output_row_1 < _min_41) {
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
                                        for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                            int row_11 = warp_0_3 * 32 + half_2 * 16 + lane_4 % 16;
                                            int col_7 = col_tile_2 * 16 + lane_4 / 16 * 8;
                                            unsigned int address_10 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_11 * 32 + col_7) * 2);
                                            address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                            int offset_4 = chunk_5 * 16 + half_2 * 8 + col_tile_2 * 4;
                                            uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(address_10);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_4 + 3]))
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
                                            for (int pair_10 = 0; pair_10 < 8; pair_10++) {
                                                __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_3[pair_10 * 2], _tmem_load_3[pair_10 * 2 + 1]));
                                                packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_10] = __as_u32(_bf16x2_5);
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
                                            for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                                int row_12 = warp_0_4 * 32 + half_3 * 16 + lane_5 % 16;
                                                int col_8 = col_tile_3 * 16 + lane_5 / 16 * 8;
                                                unsigned int address_12 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_12 * 32 + col_8) * 2);
                                                address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                                int offset_5 = chunk_7 * 16 + half_3 * 8 + col_tile_3 * 4;
                                                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(address_12);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_5 + 3]))
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
                        gemm_phase = phase_bits_4;
                    } else if (kind == 3) {
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
                        int expert_idx = 0;
                        int local_task = task_1;
                        int _max_5 = ((row_blocks_2 * col_blocks_5) > (intermediate / 256 * ((hidden + 512 - 1) / 512)) ? (row_blocks_2 * col_blocks_5) : (intermediate / 256 * ((hidden + 512 - 1) / 512)));
                        int stride = _max_5;
                        k_end_2 = local_tokens;
                        first_2 = 1;
                        if (k_start_2 < k_end_2 && local_task < row_blocks_2 * col_blocks_5) {
                            int supergroup_2 = local_task / (row_blocks_2 * 8);
                            int full_cols_2 = col_blocks_5 / 8 * 8;
                            int row_13 = 0;
                            int col_9 = 0;
                            if (local_task < row_blocks_2 * full_cols_2) {
                                row_13 = local_task % (row_blocks_2 * 8) / 8;
                                col_9 = supergroup_2 * 8 + local_task % 8;
                            } else {
                                row_13 = (local_task - row_blocks_2 * full_cols_2) / (col_blocks_5 - full_cols_2);
                                col_9 = full_cols_2 + (local_task - row_blocks_2 * full_cols_2) % (col_blocks_5 - full_cols_2);
                            }
                            if ((supergroup_2 & 1) != 0) {
                                row_13 = row_blocks_2 - row_13 - 1;
                            }
                            x_2 = row_13;
                            y_2 = col_9;
                            expert_2 = expert_idx;
                        }
                        unsigned int phase_bits_5 = gemm_phase;
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
                                    int ring_4 = 0;
                                    #pragma unroll 1
                                    for (int idx_4 = 0; idx_4 < iterations_2; idx_4++) {
                                        int token_row = k_start_2 + idx_4 * 64;
                                        if (idx_4 == 0 || token_row % 256 == 0) {
                                        }
                                        if (idx_4 == 0 || token_row % mini_size == 0) {
                                            int input_mini = token_row / mini_size;
                                            int _min_43 = ((mini_size) < (tokens - input_mini * mini_size) ? (mini_size) : (tokens - input_mini * mini_size));
                                            int input_rows = _min_43;
                                            int input_count = (input_rows + 127) / 128 * ((hidden + 511) / 512);
                                        }
                                        mbarrier_wait(gemm_finished_addr + (ring_4) * 8, phase_bits_5 >> (unsigned int)(16 + ring_4) & 1);
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
                                        phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_4);
                                        ring_4 = (ring_4 + 1) % 4;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_5 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_5 >> 22 & 1);
                                        phase_bits_5 = phase_bits_5 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_5 = 0; idx_5 < iterations_2; idx_5++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_5) * 8, 98304);
                                            mbarrier_wait(gemm_arrived_addr + (ring_5) * 8, phase_bits_5 >> (unsigned int)ring_5 & 1);
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
                                            phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << ring_5);
                                            ring_5 = (ring_5 + 1) % 4;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_5 >> 6 & 1);
                                phase_bits_5 = phase_bits_5 ^ 64;
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
                                    for (int vec_4 = 0; vec_4 < 4; vec_4++) {
                                        unsigned int address_13 = d_smem_addr + (unsigned int)(chunk_8 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_4 * 16);
                                        address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_13 - d_words_addr)), "f"(_tmem_load_4[vec_4 * 4]), "f"(_tmem_load_4[vec_4 * 4 + 1]), "f"(_tmem_load_4[vec_4 * 4 + 2]), "f"(_tmem_load_4[vec_4 * 4 + 3]) : "memory");
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
                                        for (int vec_5 = 0; vec_5 < 4; vec_5++) {
                                            unsigned int address_14 = d_smem_addr + (unsigned int)((16 + chunk_9) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_5 * 16);
                                            address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_14 - d_words_addr)), "f"(_tmem_load_5[vec_5 * 4]), "f"(_tmem_load_5[vec_5 * 4 + 1]), "f"(_tmem_load_5[vec_5 * 4 + 2]), "f"(_tmem_load_5[vec_5 * 4 + 3]) : "memory");
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
                        gemm_phase = phase_bits_5;
                    } else {
                        if (kind == 4) {
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
                            int expert_idx_1 = 0;
                            int local_task_1 = task_1;
                            int _max_7 = ((row_blocks_3 * col_blocks_6) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (row_blocks_3 * col_blocks_6) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
                            int stride_1 = _max_7;
                            k_end_3 = local_tokens;
                            first_3 = 1;
                            if (k_start_3 < k_end_3 && local_task_1 < row_blocks_3 * col_blocks_6) {
                                int supergroup_3 = local_task_1 / (row_blocks_3 * 8);
                                int full_cols_3 = col_blocks_6 / 8 * 8;
                                int row_14 = 0;
                                int col_10 = 0;
                                if (local_task_1 < row_blocks_3 * full_cols_3) {
                                    row_14 = local_task_1 % (row_blocks_3 * 8) / 8;
                                    col_10 = supergroup_3 * 8 + local_task_1 % 8;
                                } else {
                                    row_14 = (local_task_1 - row_blocks_3 * full_cols_3) / (col_blocks_6 - full_cols_3);
                                    col_10 = full_cols_3 + (local_task_1 - row_blocks_3 * full_cols_3) % (col_blocks_6 - full_cols_3);
                                }
                                if ((supergroup_3 & 1) != 0) {
                                    row_14 = row_blocks_3 - row_14 - 1;
                                }
                                x_3 = row_14;
                                y_3 = col_10;
                                expert_3 = expert_idx_1;
                            }
                            unsigned int phase_bits_6 = gemm_phase;
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
                                        int ring_6 = 0;
                                        #pragma unroll 1
                                        for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                            int token_row_1 = k_start_3 + idx_6 * 64;
                                            if (idx_6 == 0 || token_row_1 % 256 == 0) {
                                                bool enabled_value_4 = 1;
                                                if (enabled_value_4 != 0) {
                                                    int32_t _relaxed_ld_18;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_18) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                    int value_10 = _relaxed_ld_18;
                                                    while (value_10 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_19;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_19) : "l"(dg_ready + (token_row_1 / 256)) : "memory");
                                                        value_10 = _relaxed_ld_19;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_6 == 0 || token_row_1 % mini_size == 0) {
                                                int input_mini_1 = token_row_1 / mini_size;
                                                int _min_45 = ((mini_size) < (tokens - input_mini_1 * mini_size) ? (mini_size) : (tokens - input_mini_1 * mini_size));
                                                int input_rows_1 = _min_45;
                                                int input_count_1 = (input_rows_1 + 127) / 128 * ((hidden + 511) / 512);
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_6) * 8, phase_bits_6 >> (unsigned int)(16 + ring_6) & 1);
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
                                            phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << 16 + ring_6);
                                            ring_6 = (ring_6 + 1) % 4;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_7 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_6 >> 22 & 1);
                                            phase_bits_6 = phase_bits_6 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_7) * 8, 98304);
                                                mbarrier_wait(gemm_arrived_addr + (ring_7) * 8, phase_bits_6 >> (unsigned int)ring_7 & 1);
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
                                                phase_bits_6 = phase_bits_6 ^ (unsigned int)(1 << ring_7);
                                                ring_7 = (ring_7 + 1) % 4;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_6 >> 6 & 1);
                                    phase_bits_6 = phase_bits_6 ^ 64;
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
                                        for (int vec_6 = 0; vec_6 < 4; vec_6++) {
                                            unsigned int address_15 = d_smem_addr + (unsigned int)(chunk_10 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_6 * 16);
                                            address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_15 - d_words_addr)), "f"(_tmem_load_6[vec_6 * 4]), "f"(_tmem_load_6[vec_6 * 4 + 1]), "f"(_tmem_load_6[vec_6 * 4 + 2]), "f"(_tmem_load_6[vec_6 * 4 + 3]) : "memory");
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
                                            for (int vec_7 = 0; vec_7 < 4; vec_7++) {
                                                unsigned int address_16 = d_smem_addr + (unsigned int)((16 + chunk_11) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_7 * 16);
                                                address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_16 - d_words_addr)), "f"(_tmem_load_7[vec_7 * 4]), "f"(_tmem_load_7[vec_7 * 4 + 1]), "f"(_tmem_load_7[vec_7 * 4 + 2]), "f"(_tmem_load_7[vec_7 * 4 + 3]) : "memory");
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
                            gemm_phase = phase_bits_6;
                        } else if (kind == 5) {
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
                            int expert_idx_2 = 0;
                            int local_task_2 = task_1;
                            int _max_9 = ((row_blocks_4 * col_blocks_7) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (row_blocks_4 * col_blocks_7) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
                            int stride_2 = _max_9;
                            k_end_4 = local_tokens;
                            first_4 = 1;
                            if (k_start_4 < k_end_4 && local_task_2 < row_blocks_4 * col_blocks_7) {
                                int supergroup_4 = local_task_2 / (row_blocks_4 * 8);
                                int full_cols_4 = col_blocks_7 / 8 * 8;
                                int row_15 = 0;
                                int col_11 = 0;
                                if (local_task_2 < row_blocks_4 * full_cols_4) {
                                    row_15 = local_task_2 % (row_blocks_4 * 8) / 8;
                                    col_11 = supergroup_4 * 8 + local_task_2 % 8;
                                } else {
                                    row_15 = (local_task_2 - row_blocks_4 * full_cols_4) / (col_blocks_7 - full_cols_4);
                                    col_11 = full_cols_4 + (local_task_2 - row_blocks_4 * full_cols_4) % (col_blocks_7 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_15 = row_blocks_4 - row_15 - 1;
                                }
                                x_4 = row_15;
                                y_4 = col_11;
                                expert_4 = expert_idx_2;
                            }
                            unsigned int phase_bits_7 = gemm_phase;
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
                                        int ring_8 = 0;
                                        #pragma unroll 1
                                        for (int idx_8 = 0; idx_8 < iterations_4; idx_8++) {
                                            int token_row_2 = k_start_4 + idx_8 * 64;
                                            if (idx_8 == 0 || token_row_2 % 256 == 0) {
                                                bool enabled_value_5 = 1;
                                                if (enabled_value_5 != 0) {
                                                    int32_t _relaxed_ld_22;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_22) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                    int value_11 = _relaxed_ld_22;
                                                    while (value_11 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_23;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_23) : "l"(dg_ready + (token_row_2 / 256)) : "memory");
                                                        value_11 = _relaxed_ld_23;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_8 == 0 || token_row_2 % mini_size == 0) {
                                                int input_mini_2 = token_row_2 / mini_size;
                                                int _min_47 = ((mini_size) < (tokens - input_mini_2 * mini_size) ? (mini_size) : (tokens - input_mini_2 * mini_size));
                                                int input_rows_2 = _min_47;
                                                int input_count_2 = (input_rows_2 + 127) / 128 * ((hidden + 511) / 512);
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_8) * 8, phase_bits_7 >> (unsigned int)(16 + ring_8) & 1);
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
                                            phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << 16 + ring_8);
                                            ring_8 = (ring_8 + 1) % 4;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_9 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_7 >> 22 & 1);
                                            phase_bits_7 = phase_bits_7 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_9 = 0; idx_9 < iterations_4; idx_9++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_9) * 8, 98304);
                                                mbarrier_wait(gemm_arrived_addr + (ring_9) * 8, phase_bits_7 >> (unsigned int)ring_9 & 1);
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
                                                phase_bits_7 = phase_bits_7 ^ (unsigned int)(1 << ring_9);
                                                ring_9 = (ring_9 + 1) % 4;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_7 >> 6 & 1);
                                    phase_bits_7 = phase_bits_7 ^ 64;
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
                                        for (int vec_8 = 0; vec_8 < 4; vec_8++) {
                                            unsigned int address_17 = d_smem_addr + (unsigned int)(chunk_12 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_8 * 16);
                                            address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_17 - d_words_addr)), "f"(_tmem_load_8[vec_8 * 4]), "f"(_tmem_load_8[vec_8 * 4 + 1]), "f"(_tmem_load_8[vec_8 * 4 + 2]), "f"(_tmem_load_8[vec_8 * 4 + 3]) : "memory");
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
                                            for (int vec_9 = 0; vec_9 < 4; vec_9++) {
                                                unsigned int address_18 = d_smem_addr + (unsigned int)((16 + chunk_13) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_9 * 16);
                                                address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_18 - d_words_addr)), "f"(_tmem_load_9[vec_9 * 4]), "f"(_tmem_load_9[vec_9 * 4 + 1]), "f"(_tmem_load_9[vec_9 * 4 + 2]), "f"(_tmem_load_9[vec_9 * 4 + 3]) : "memory");
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
                            gemm_phase = phase_bits_7;
                        }
                    }
                }
            } else if (kind == 6) {
                int col_blocks_8 = (intermediate + 512 - 1) / 512;
                int x_5 = -1;
                int y_5 = -1;
                int expert_5 = -1;
                int k_start_5 = 0;
                int k_end_5 = 0;
                int first_5 = 0;
                int first_block = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                int _min_48 = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                int end_block = _min_48;
                int block_5 = first_block + task_1 / col_blocks_8;
                if (block_5 < end_block) {
                    int index_3 = counts[3 * experts + block_5];
                    int offset_6 = counts[experts + index_3] / 256;
                    int _max_11 = ((first_block) > (offset_6) ? (first_block) : (offset_6));
                    int first_row_6 = _max_11;
                    int _min_49 = ((end_block) < (offset_6 + counts[index_3] / 256) ? (end_block) : (offset_6 + counts[index_3] / 256));
                    int rows_4 = _min_49 - first_row_6;
                    int supergroup_5 = (task_1 - (first_row_6 - first_block) * col_blocks_8) / (rows_4 * 8);
                    int full_cols_5 = col_blocks_8 / 8 * 8;
                    int row_16 = 0;
                    int col_12 = 0;
                    if (task_1 - (first_row_6 - first_block) * col_blocks_8 < rows_4 * full_cols_5) {
                        row_16 = (task_1 - (first_row_6 - first_block) * col_blocks_8) % (rows_4 * 8) / 8;
                        col_12 = supergroup_5 * 8 + (task_1 - (first_row_6 - first_block) * col_blocks_8) % 8;
                    } else {
                        row_16 = (task_1 - (first_row_6 - first_block) * col_blocks_8 - rows_4 * full_cols_5) / (col_blocks_8 - full_cols_5);
                        col_12 = full_cols_5 + (task_1 - (first_row_6 - first_block) * col_blocks_8 - rows_4 * full_cols_5) % (col_blocks_8 - full_cols_5);
                    }
                    if ((supergroup_5 & 1) != 0) {
                        row_16 = rows_4 - row_16 - 1;
                    }
                    x_5 = first_row_6 + row_16 - macro_1 * (macro_size / 256);
                    y_5 = col_12;
                    expert_5 = index_3;
                }
                unsigned int phase_bits_8 = gemm_phase;
                int has_hi_5 = 0;
                has_hi_5 = (int)((y_5 * 2 + 1) * 256 < intermediate);
                int global_mini_5 = macro_1 * (macro_size / mini_size) + mini_2;
                int macro_rows_5 = macro_1 * (macro_size / 256);
                int iterations_5 = hidden / 64;
                if (expert_5 < 0) {
                    if (tid == 0) {
                    }
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                int _min_50 = ((mini_size) < (tokens - global_mini_5 * mini_size) ? (mini_size) : (tokens - global_mini_5 * mini_size));
                                int _max_12 = ((0) > (_min_50) ? (0) : (_min_50));
                                int mini_rows_6 = _max_12;
                                int required_3 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                bool enabled_value_6 = 1;
                                if (enabled_value_6 != 0) {
                                    int32_t _relaxed_ld_24;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_24) : "l"(replay_x + global_mini_5) : "memory");
                                    int value_12 = _relaxed_ld_24;
                                    while (value_12 < required_3) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_25;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_25) : "l"(replay_x + global_mini_5) : "memory");
                                        value_12 = _relaxed_ld_25;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                            }
                            int ring_10 = 0;
                            #pragma unroll 1
                            for (int idx_10 = 0; idx_10 < iterations_5; idx_10++) {
                                mbarrier_wait(gemm_finished_addr + (ring_10) * 8, phase_bits_8 >> (unsigned int)(16 + ring_10) & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_nt_addr + (unsigned int)(ring_10 * 16384)), "l"((&x_nt_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_nt_addr + (unsigned int)(ring_10 * 16384)), "l"((&wg_nt_r)), "r"(0), "r"(y_5 * 2 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(expert_5), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_nt_hi_addr + (unsigned int)(ring_10 * 16384)), "l"((&wg_nt_r)), "r"(0), "r"((y_5 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(expert_5), "r"(0),
                                       "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << 16 + ring_10);
                                ring_10 = (ring_10 + 1) % 4;
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int ring_11 = 0;
                                mbarrier_wait(output_finished_addr, phase_bits_8 >> 22 & 1);
                                phase_bits_8 = phase_bits_8 ^ 4194304;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_11 = 0; idx_11 < iterations_5; idx_11++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_11) * 8, 98304);
                                    mbarrier_wait(gemm_arrived_addr + (ring_11) * 8, phase_bits_8 >> (unsigned int)ring_11 & 1);
                                    int _mma_a_lo_10 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
                                    int _mma_b_lo_10 = (((b_nt_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
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
            :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_accumulator), "r"(((idx_11 == 0) ? 0 : 1)));
                                    int _mma_a_lo_11 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
                                    int _mma_b_lo_11 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
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
            :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_accumulator + (256))), "r"(((idx_11 == 0) ? 0 : 1)));
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_11) * 8, (uint16_t)(3));
                                    phase_bits_8 = phase_bits_8 ^ (unsigned int)(1 << ring_11);
                                    ring_11 = (ring_11 + 1) % 4;
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        mbarrier_wait(output_arrived_addr, phase_bits_8 >> 6 & 1);
                        phase_bits_8 = phase_bits_8 ^ 64;
                        unsigned int packed_7[128];
                        #pragma unroll
                        for (int chunk_14 = 0; chunk_14 < 8; chunk_14++) {
                            #pragma unroll
                            for (int sub_4 = 0; sub_4 < 2; sub_4++) {
                                unsigned int address_19 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_4 * 16 << 16) + (unsigned int)(chunk_14 * 32);
                                float _tmem_load_10[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                                    : "r"(address_19));
                                #pragma unroll
                                for (int pair_11 = 0; pair_11 < 8; pair_11++) {
                                    __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_tmem_load_10[pair_11 * 2], _tmem_load_10[pair_11 * 2 + 1]));
                                    packed_7[chunk_14 * 16 + sub_4 * 8 + pair_11] = __as_u32(_bf16x2_6);
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int last_3 = 1;
                        last_3 = 1 - has_hi_5;
                        if (last_3 != 0) {
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        if (tid == 0) {
                            int previous_offset_4 = (macro_1 + 1) * macro_size;
                            int output_row_2 = x_5 * 256 + cta_rank_0 * 128;
                            int _min_51 = ((macro_size) < (tokens - previous_offset_4) ? (macro_size) : (tokens - previous_offset_4));
                            if (output_row_2 < _min_51) {
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
                            for (int half_4 = 0; half_4 < 2; half_4++) {
                                #pragma unroll
                                for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                    int row_17 = warp_0_5 * 32 + half_4 * 16 + lane_7 % 16;
                                    int col_13 = col_tile_4 * 16 + lane_7 / 16 * 8;
                                    unsigned int address_20 = d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192) + (unsigned int)((row_17 * 32 + col_13) * 2);
                                    address_20 = address_20 ^ (address_20 & 511) >> 7 << 4;
                                    int offset_7 = chunk_15 * 16 + half_4 * 8 + col_tile_4 * 4;
                                    uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(address_20);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_7 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 2 * 8 + chunk_15), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_15 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        if (has_hi_5 != 0) {
                            unsigned int packed_0_2[128];
                            #pragma unroll
                            for (int chunk_16 = 0; chunk_16 < 8; chunk_16++) {
                                #pragma unroll
                                for (int sub_5 = 0; sub_5 < 2; sub_5++) {
                                    unsigned int address_21 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_5 * 16 << 16) + 256 + (unsigned int)(chunk_16 * 32);
                                    float _tmem_load_11[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                                        : "r"(address_21));
                                    #pragma unroll
                                    for (int pair_12 = 0; pair_12 < 8; pair_12++) {
                                        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_11[pair_12 * 2], _tmem_load_11[pair_12 * 2 + 1]));
                                        packed_0_2[chunk_16 * 16 + sub_5 * 8 + pair_12] = __as_u32(_bf16x2_7);
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
                                for (int half_5 = 0; half_5 < 2; half_5++) {
                                    #pragma unroll
                                    for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                        int row_18 = warp_0_6 * 32 + half_5 * 16 + lane_8 % 16;
                                        int col_14 = col_tile_5 * 16 + lane_8 / 16 * 8;
                                        unsigned int address_22 = d_smem_addr + (unsigned int)((8 + chunk_17) % 3 * 8192) + (unsigned int)((row_18 * 32 + col_14) * 2);
                                        address_22 = address_22 ^ (address_22 & 511) >> 7 << 4;
                                        int offset_8 = chunk_17 * 16 + half_5 * 8 + col_tile_5 * 4;
                                        uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(address_22);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_2[offset_8 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&gate_out_r)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"((y_5 * 2 + 1) * 8 + chunk_17), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_17) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                    bool enabled_value_7 = 1;
                                    if (enabled_value_7 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_5 + x_5) * (intermediate / 256) + y_5 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                    if (has_hi_5 != 0) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_0_1 = 1;
                                        if (enabled_value_0_1 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_5 + x_5) * (intermediate / 256) + y_5 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                gemm_phase = phase_bits_8;
            } else {
                if (kind == 7) {
                    int col_blocks_9 = (intermediate + 512 - 1) / 512;
                    int x_6 = -1;
                    int y_6 = -1;
                    int expert_6 = -1;
                    int k_start_6 = 0;
                    int k_end_6 = 0;
                    int first_6 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                    int _min_52 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int end_block_1 = _min_52;
                    int block_6 = first_block_1 + task_1 / col_blocks_9;
                    if (block_6 < end_block_1) {
                        int index_4 = counts[3 * experts + block_6];
                        int offset_9 = counts[experts + index_4] / 256;
                        int _max_13 = ((first_block_1) > (offset_9) ? (first_block_1) : (offset_9));
                        int first_row_7 = _max_13;
                        int _min_53 = ((end_block_1) < (offset_9 + counts[index_4] / 256) ? (end_block_1) : (offset_9 + counts[index_4] / 256));
                        int rows_5 = _min_53 - first_row_7;
                        int supergroup_6 = (task_1 - (first_row_7 - first_block_1) * col_blocks_9) / (rows_5 * 8);
                        int full_cols_6 = col_blocks_9 / 8 * 8;
                        int row_19 = 0;
                        int col_15 = 0;
                        if (task_1 - (first_row_7 - first_block_1) * col_blocks_9 < rows_5 * full_cols_6) {
                            row_19 = (task_1 - (first_row_7 - first_block_1) * col_blocks_9) % (rows_5 * 8) / 8;
                            col_15 = supergroup_6 * 8 + (task_1 - (first_row_7 - first_block_1) * col_blocks_9) % 8;
                        } else {
                            row_19 = (task_1 - (first_row_7 - first_block_1) * col_blocks_9 - rows_5 * full_cols_6) / (col_blocks_9 - full_cols_6);
                            col_15 = full_cols_6 + (task_1 - (first_row_7 - first_block_1) * col_blocks_9 - rows_5 * full_cols_6) % (col_blocks_9 - full_cols_6);
                        }
                        if ((supergroup_6 & 1) != 0) {
                            row_19 = rows_5 - row_19 - 1;
                        }
                        x_6 = first_row_7 + row_19 - macro_1 * (macro_size / 256);
                        y_6 = col_15;
                        expert_6 = index_4;
                    }
                    unsigned int phase_bits_9 = gemm_phase;
                    int has_hi_6 = 0;
                    has_hi_6 = (int)((y_6 * 2 + 1) * 256 < intermediate);
                    int global_mini_6 = macro_1 * (macro_size / mini_size) + mini_2;
                    int macro_rows_6 = macro_1 * (macro_size / 256);
                    int iterations_6 = hidden / 64;
                    if (expert_6 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_54 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                    int _max_14 = ((0) > (_min_54) ? (0) : (_min_54));
                                    int mini_rows_7 = _max_14;
                                    int required_4 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_8 = 1;
                                    if (enabled_value_8 != 0) {
                                        int32_t _relaxed_ld_26;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_26) : "l"(replay_x + global_mini_6) : "memory");
                                        int value_13 = _relaxed_ld_26;
                                        while (value_13 < required_4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_27;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_27) : "l"(replay_x + global_mini_6) : "memory");
                                            value_13 = _relaxed_ld_27;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                int ring_12 = 0;
                                #pragma unroll 1
                                for (int idx_12 = 0; idx_12 < iterations_6; idx_12++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_12) * 8, phase_bits_9 >> (unsigned int)(16 + ring_12) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_nt_addr + (unsigned int)(ring_12 * 16384)), "l"((&x_nt_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_12) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_nt_addr + (unsigned int)(ring_12 * 16384)), "l"((&wu_nt_r)), "r"(0), "r"(y_6 * 2 * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(expert_6), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_12) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_nt_hi_addr + (unsigned int)(ring_12 * 16384)), "l"((&wu_nt_r)), "r"(0), "r"((y_6 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(expert_6), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_12) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << 16 + ring_12);
                                    ring_12 = (ring_12 + 1) % 4;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_13 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_9 >> 22 & 1);
                                    phase_bits_9 = phase_bits_9 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_13 = 0; idx_13 < iterations_6; idx_13++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_13) * 8, 98304);
                                        mbarrier_wait(gemm_arrived_addr + (ring_13) * 8, phase_bits_9 >> (unsigned int)ring_13 & 1);
                                        int _mma_a_lo_12 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_13) * 1024;
                                        int _mma_b_lo_12 = (((b_nt_addr) >> 4) & 0x3FFF) + (ring_13) * 1024;
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
            :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"(tmem_accumulator), "r"(((idx_13 == 0) ? 0 : 1)));
                                        int _mma_a_lo_13 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_13) * 1024;
                                        int _mma_b_lo_13 = (((b_nt_hi_addr) >> 4) & 0x3FFF) + (ring_13) * 1024;
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
            :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_13), "r"((tmem_accumulator + (256))), "r"(((idx_13 == 0) ? 0 : 1)));
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_13) * 8, (uint16_t)(3));
                                        phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << ring_13);
                                        ring_13 = (ring_13 + 1) % 4;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_9 >> 6 & 1);
                            phase_bits_9 = phase_bits_9 ^ 64;
                            unsigned int packed_8[128];
                            #pragma unroll
                            for (int chunk_18 = 0; chunk_18 < 8; chunk_18++) {
                                #pragma unroll
                                for (int sub_6 = 0; sub_6 < 2; sub_6++) {
                                    unsigned int address_23 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_6 * 16 << 16) + (unsigned int)(chunk_18 * 32);
                                    float _tmem_load_12[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                                        : "r"(address_23));
                                    #pragma unroll
                                    for (int pair_13 = 0; pair_13 < 8; pair_13++) {
                                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_tmem_load_12[pair_13 * 2], _tmem_load_12[pair_13 * 2 + 1]));
                                        packed_8[chunk_18 * 16 + sub_6 * 8 + pair_13] = __as_u32(_bf16x2_8);
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int last_4 = 1;
                            last_4 = 1 - has_hi_6;
                            if (last_4 != 0) {
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            if (tid == 0) {
                                int previous_offset_5 = (macro_1 + 1) * macro_size;
                                int output_row_3 = x_6 * 256 + cta_rank_0 * 128;
                                int _min_55 = ((macro_size) < (tokens - previous_offset_5) ? (macro_size) : (tokens - previous_offset_5));
                                if (output_row_3 < _min_55) {
                                }
                            }
                            #pragma unroll
                            for (int chunk_19 = 0; chunk_19 < 8; chunk_19++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_0_7 = tid / 32;
                                int lane_10 = tid % 32;
                                #pragma unroll
                                for (int half_6 = 0; half_6 < 2; half_6++) {
                                    #pragma unroll
                                    for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                        int row_20 = warp_0_7 * 32 + half_6 * 16 + lane_10 % 16;
                                        int col_16 = col_tile_6 * 16 + lane_10 / 16 * 8;
                                        unsigned int address_24 = d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192) + (unsigned int)((row_20 * 32 + col_16) * 2);
                                        address_24 = address_24 ^ (address_24 & 511) >> 7 << 4;
                                        int offset_10 = chunk_19 * 16 + half_6 * 8 + col_tile_6 * 4;
                                        uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(address_24);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_8[offset_10 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                        :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"(y_6 * 2 * 8 + chunk_19), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_19 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (has_hi_6 != 0) {
                                unsigned int packed_0_3[128];
                                #pragma unroll
                                for (int chunk_20 = 0; chunk_20 < 8; chunk_20++) {
                                    #pragma unroll
                                    for (int sub_7 = 0; sub_7 < 2; sub_7++) {
                                        unsigned int address_25 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_7 * 16 << 16) + 256 + (unsigned int)(chunk_20 * 32);
                                        float _tmem_load_13[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                                            : "r"(address_25));
                                        #pragma unroll
                                        for (int pair_14 = 0; pair_14 < 8; pair_14++) {
                                            __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_tmem_load_13[pair_14 * 2], _tmem_load_13[pair_14 * 2 + 1]));
                                            packed_0_3[chunk_20 * 16 + sub_7 * 8 + pair_14] = __as_u32(_bf16x2_9);
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
                                    int lane_11 = tid % 32;
                                    #pragma unroll
                                    for (int half_7 = 0; half_7 < 2; half_7++) {
                                        #pragma unroll
                                        for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                            int row_21 = warp_0_8 * 32 + half_7 * 16 + lane_11 % 16;
                                            int col_17 = col_tile_7 * 16 + lane_11 / 16 * 8;
                                            unsigned int address_26 = d_smem_addr + (unsigned int)((8 + chunk_21) % 3 * 8192) + (unsigned int)((row_21 * 32 + col_17) * 2);
                                            address_26 = address_26 ^ (address_26 & 511) >> 7 << 4;
                                            int offset_11 = chunk_21 * 16 + half_7 * 8 + col_tile_7 * 4;
                                            uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(address_26);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_3[offset_11 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_out_r)), "r"(0), "r"(x_6 * 256 + cta_rank_0 * 128), "r"((y_6 * 2 + 1) * 8 + chunk_21), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_21) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                        bool enabled_value_9 = 1;
                                        if (enabled_value_9 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_6 + x_6) * (intermediate / 256) + y_6 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                        if (has_hi_6 != 0) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_0_2 = 1;
                                            if (enabled_value_0_2 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_gu)) + ((macro_rows_6 + x_6) * (intermediate / 256) + y_6 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_phase = phase_bits_9;
                } else if (kind == 8) {
                    unsigned int phase_bits_10 = replay_phase;
                    int col_blocks_10 = intermediate / 128;
                    int num_tiles_1 = tokens / 128 * col_blocks_10;
                    int macro_row_offset_1 = macro_1 * (macro_size / 128);
                    int first_tile_1 = task_1 * 6 + cta_rank_0 * 3;
                    int tile_end_1 = num_tiles_1;
                    {
                        int global_mini_7 = macro_1 * (macro_size / mini_size) + mini_2;
                        int mini_tiles = mini_size / 128 * col_blocks_10;
                        first_tile_1 = first_tile_1 + global_mini_7 * mini_tiles;
                        int _min_56 = ((num_tiles_1) < ((global_mini_7 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_7 + 1) * mini_tiles));
                        tile_end_1 = _min_56;
                    }
                    if (first_tile_1 < tile_end_1) {
                        int first_row_8 = first_tile_1 / col_blocks_10;
                        int first_col = first_tile_1 % col_blocks_10;
                        if (tid == 0) {
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 3; stage_2++) {
                                if (tile_end_1 > first_tile_1 + stage_2) {
                                    int row_22 = first_row_8;
                                    int col_18 = first_col + stage_2;
                                    if (col_18 >= col_blocks_10) {
                                        row_22 = row_22 + 1;
                                        col_18 = col_18 - col_blocks_10;
                                    }
                                    mbarrier_arrive_expect_tx(replay_arrived_addr + (stage_2) * 8, 65536);
                                    int parent_2 = row_22 / 2 * (intermediate / 256) + col_18 / 2;
                                    int32_t _relaxed_ld_28;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_28) : "l"(replay_gu + parent_2) : "memory");
                                    int value_14 = _relaxed_ld_28;
                                    while (value_14 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_29;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_29) : "l"(replay_gu + parent_2) : "memory");
                                        value_14 = _relaxed_ld_29;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(replay_gate_addr + (unsigned int)(stage_2 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_22 - macro_row_offset_1) * 128), "r"(col_18 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + (stage_2) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(replay_up_addr + (unsigned int)(stage_2 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_22 - macro_row_offset_1) * 128), "r"(col_18 * 2), "r"(0), "r"(0), "r"(replay_arrived_addr + (stage_2) * 8) : "memory");
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int stage_3 = 0; stage_3 < 3; stage_3++) {
                            if (tile_end_1 > first_tile_1 + stage_3) {
                                mbarrier_wait(replay_arrived_addr + (stage_3) * 8, phase_bits_10 >> (unsigned int)stage_3 & 1);
                                phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << stage_3);
                                int row_23 = first_row_8;
                                int col_19 = first_col + stage_3;
                                if (col_19 >= col_blocks_10) {
                                    row_23 = row_23 + 1;
                                    col_19 = col_19 - col_blocks_10;
                                }
                                float gate_1[64];
                                float up_1[64];
                                float denominator[64];
                                int warp_0_9 = tid / 32;
                                int local_warp_1 = warp_0_9 / 4 + warp_0_9 % 4 * 2;
                                int lane_13 = tid % 32;
                                #pragma unroll
                                for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                    unsigned int packed_9[4];
                                    unsigned int address_27 = replay_gate_addr + (unsigned int)(stage_3 * 32768) + (unsigned int)(((tile_col_5 * 16 + lane_13 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_13 % 16) * 64 + (tile_col_5 * 16 + lane_13 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_9[0]), "=r"(packed_9[1]), "=r"(packed_9[2]), "=r"(packed_9[3])
                                        : "r"(address_27 ^ (address_27 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_15 = 0; pair_15 < 4; pair_15++) {
                                        float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(packed_9[pair_15]));
                                        gate_1[tile_col_5 * 8 + pair_15 * 2] = _cvt_f32_3.x;
                                        gate_1[tile_col_5 * 8 + pair_15 * 2 + 1] = _cvt_f32_3.y;
                                    }
                                }
                                int warp_1_1 = tid / 32;
                                int local_warp_2_1 = warp_1_1 / 4 + warp_1_1 % 4 * 2;
                                int lane_3_2 = tid % 32;
                                #pragma unroll
                                for (int tile_col_6 = 0; tile_col_6 < 8; tile_col_6++) {
                                    unsigned int packed_10[4];
                                    unsigned int address_28 = replay_up_addr + (unsigned int)(stage_3 * 32768) + (unsigned int)(((tile_col_6 * 16 + lane_3_2 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_1 * 16 + lane_3_2 % 16) * 64 + (tile_col_6 * 16 + lane_3_2 / 16 * 8) % 64) * 2);
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(packed_10[0]), "=r"(packed_10[1]), "=r"(packed_10[2]), "=r"(packed_10[3])
                                        : "r"(address_28 ^ (address_28 & 1023) >> 7 << 4)
                                        : "memory");
                                    #pragma unroll
                                    for (int pair_16 = 0; pair_16 < 4; pair_16++) {
                                        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(packed_10[pair_16]));
                                        up_1[tile_col_6 * 8 + pair_16 * 2] = _cvt_f32_4.x;
                                        up_1[tile_col_6 * 8 + pair_16 * 2 + 1] = _cvt_f32_4.y;
                                    }
                                }
                                #pragma unroll
                                for (int elem_1 = 0; elem_1 < 64; elem_1++) {
                                    float _min_57 = fminf(gate_1[elem_1], swiglu_limit);
                                    gate_1[elem_1] = _min_57;
                                }
                                #pragma unroll
                                for (int elem_2 = 0; elem_2 < 64; elem_2++) {
                                    float _fmax_1 = fmaxf(up_1[elem_2], -swiglu_limit);
                                    float _min_58 = fminf(_fmax_1, swiglu_limit);
                                    up_1[elem_2] = _min_58;
                                }
                                #pragma unroll
                                for (int elem_3 = 0; elem_3 < 64; elem_3++) {
                                    denominator[elem_3] = gate_1[elem_3] * -1.0f;
                                }
                                #pragma unroll
                                for (int elem_4 = 0; elem_4 < 64; elem_4++) {
                                    float _exp_1 = expf(denominator[elem_4]);
                                    denominator[elem_4] = _exp_1;
                                }
                                #pragma unroll
                                for (int elem_5 = 0; elem_5 < 64; elem_5++) {
                                    denominator[elem_5] = denominator[elem_5] + 1.0f;
                                }
                                #pragma unroll
                                for (int elem_6 = 0; elem_6 < 64; elem_6++) {
                                    gate_1[elem_6] = gate_1[elem_6] / denominator[elem_6];
                                }
                                #pragma unroll
                                for (int elem_7 = 0; elem_7 < 64; elem_7++) {
                                    gate_1[elem_7] = gate_1[elem_7] * up_1[elem_7];
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                __syncthreads();
                                int warp_4_1 = tid / 32;
                                int local_warp_5_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                                int lane_6_1 = tid % 32;
                                #pragma unroll
                                for (int tile_col_7 = 0; tile_col_7 < 8; tile_col_7++) {
                                    unsigned int packed_11[4];
                                    #pragma unroll
                                    for (int pair_17 = 0; pair_17 < 4; pair_17++) {
                                        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(gate_1[tile_col_7 * 8 + pair_17 * 2], gate_1[tile_col_7 * 8 + pair_17 * 2 + 1]));
                                        packed_11[pair_17] = __as_u32(_bf16x2_10);
                                    }
                                    unsigned int address_29 = replay_hidden_addr + (unsigned int)(((tile_col_7 * 16 + lane_6_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_5_1 * 16 + lane_6_1 % 16) * 64 + (tile_col_7 * 16 + lane_6_1 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(address_29 ^ (address_29 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[3]))
                                        : "memory");
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&h_sw_r), 0, (row_23 - macro_row_offset_1) * 128, col_19 * 2, 0, 0, replay_hidden_addr);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            #pragma unroll
                            for (int stage_4 = 0; stage_4 < 3; stage_4++) {
                                if (tile_end_1 > first_tile_1 + stage_4) {
                                    int row_24 = first_row_8;
                                    if (col_blocks_10 <= first_col + stage_4) {
                                        row_24 = row_24 + 1;
                                    }
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(replay_h)) + (row_24 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                    }
                    replay_phase = phase_bits_10;
                } else {
                    if (kind == 0) {
                        int col_blocks_11 = (hidden + 512 - 1) / 512;
                        {
                            col_blocks_11 = (intermediate + 512 - 1) / 512;
                        }
                        int x_7 = -1;
                        int y_7 = -1;
                        int expert_7 = -1;
                        int k_start_7 = 0;
                        int k_end_7 = 0;
                        int first_7 = 0;
                        int first_block_2 = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                        int _min_59 = ((first_block_2 + mini_size / 256) < (tokens / 256) ? (first_block_2 + mini_size / 256) : (tokens / 256));
                        int end_block_2 = _min_59;
                        int block_7 = first_block_2 + task_1 / col_blocks_11;
                        if (block_7 < end_block_2) {
                            int index_5 = counts[3 * experts + block_7];
                            int offset_12 = counts[experts + index_5] / 256;
                            int _max_15 = ((first_block_2) > (offset_12) ? (first_block_2) : (offset_12));
                            int first_row_9 = _max_15;
                            int _min_60 = ((end_block_2) < (offset_12 + counts[index_5] / 256) ? (end_block_2) : (offset_12 + counts[index_5] / 256));
                            int rows_6 = _min_60 - first_row_9;
                            int supergroup_7 = (task_1 - (first_row_9 - first_block_2) * col_blocks_11) / (rows_6 * 8);
                            int full_cols_7 = col_blocks_11 / 8 * 8;
                            int row_25 = 0;
                            int col_20 = 0;
                            if (task_1 - (first_row_9 - first_block_2) * col_blocks_11 < rows_6 * full_cols_7) {
                                row_25 = (task_1 - (first_row_9 - first_block_2) * col_blocks_11) % (rows_6 * 8) / 8;
                                col_20 = supergroup_7 * 8 + (task_1 - (first_row_9 - first_block_2) * col_blocks_11) % 8;
                            } else {
                                row_25 = (task_1 - (first_row_9 - first_block_2) * col_blocks_11 - rows_6 * full_cols_7) / (col_blocks_11 - full_cols_7);
                                col_20 = full_cols_7 + (task_1 - (first_row_9 - first_block_2) * col_blocks_11 - rows_6 * full_cols_7) % (col_blocks_11 - full_cols_7);
                            }
                            if ((supergroup_7 & 1) != 0) {
                                row_25 = rows_6 - row_25 - 1;
                            }
                            x_7 = first_row_9 + row_25 - macro_1 * (macro_size / 256);
                            y_7 = col_20;
                            expert_7 = index_5;
                        }
                        unsigned int phase_bits_11 = gemm_phase;
                        int has_hi_7 = 0;
                        has_hi_7 = (int)((y_7 * 2 + 1) * 256 < intermediate);
                        int global_mini_8 = macro_1 * (macro_size / mini_size) + mini_2;
                        int macro_rows_7 = macro_1 * (macro_size / 256);
                        int iterations_7 = hidden / 64;
                        if (expert_7 < 0) {
                            if (tid == 0) {
                                bool enabled_value_10 = macros > 1;
                                if (enabled_value_10 != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_61 = ((mini_size) < (tokens - global_mini_8 * mini_size) ? (mini_size) : (tokens - global_mini_8 * mini_size));
                                        int _max_16 = ((0) > (_min_61) ? (0) : (_min_61));
                                        int mini_rows_8 = _max_16;
                                        int required_5 = (mini_rows_8 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_11 = 1;
                                        if (enabled_value_11 != 0) {
                                            int32_t _relaxed_ld_30;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_30) : "l"(dy_ready + global_mini_8) : "memory");
                                            int value_15 = _relaxed_ld_30;
                                            while (value_15 < required_5) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_31;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_31) : "l"(dy_ready + global_mini_8) : "memory");
                                                value_15 = _relaxed_ld_31;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_14 = 0;
                                    #pragma unroll 1
                                    for (int idx_14 = 0; idx_14 < iterations_7; idx_14++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_14) * 8, phase_bits_11 >> (unsigned int)(16 + ring_14) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_nt_addr + (unsigned int)(ring_14 * 16384)), "l"((&dy_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(idx_14), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_ab_addr + (unsigned int)(ring_14 * 16384)), "l"((&wd_r)), "r"(0), "r"(idx_14 * 64), "r"(y_7 * 2 * 4 + cta_rank_0 * 2), "r"(expert_7), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_ab_hi_addr + (unsigned int)(ring_14 * 16384)), "l"((&wd_r)), "r"(0), "r"(idx_14 * 64), "r"((y_7 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_7), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_14) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        phase_bits_11 = phase_bits_11 ^ (unsigned int)(1 << 16 + ring_14);
                                        ring_14 = (ring_14 + 1) % 4;
                                    }
                                }
                            }
                        } else {
                            if (tid / 32 == 4 && cta_rank_0 == 0) {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        int ring_15 = 0;
                                        mbarrier_wait(output_finished_addr, phase_bits_11 >> 22 & 1);
                                        phase_bits_11 = phase_bits_11 ^ 4194304;
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        #pragma unroll 1
                                        for (int idx_15 = 0; idx_15 < iterations_7; idx_15++) {
                                            mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_15) * 8, 98304);
                                            mbarrier_wait(gemm_arrived_addr + (ring_15) * 8, phase_bits_11 >> (unsigned int)ring_15 & 1);
                                            int _mma_a_lo_14 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_15) * 1024;
                                            int _mma_b_lo_14 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_15) * 1024;
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
            :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"(tmem_accumulator), "r"(((idx_15 == 0) ? 0 : 1)));
                                            int _mma_a_lo_15 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_15) * 1024;
                                            int _mma_b_lo_15 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_15) * 1024;
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
            :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_15), "r"((tmem_accumulator + (256))), "r"(((idx_15 == 0) ? 0 : 1)));
                                            tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_15) * 8, (uint16_t)(3));
                                            phase_bits_11 = phase_bits_11 ^ (unsigned int)(1 << ring_15);
                                            ring_15 = (ring_15 + 1) % 4;
                                        }
                                        tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    }
                                }
                            } else if (tid < 128) {
                                mbarrier_wait(output_arrived_addr, phase_bits_11 >> 6 & 1);
                                phase_bits_11 = phase_bits_11 ^ 64;
                                unsigned int packed_12[128];
                                #pragma unroll
                                for (int chunk_22 = 0; chunk_22 < 8; chunk_22++) {
                                    #pragma unroll
                                    for (int sub_8 = 0; sub_8 < 2; sub_8++) {
                                        unsigned int address_30 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_8 * 16 << 16) + (unsigned int)(chunk_22 * 32);
                                        float _tmem_load_14[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                                            : "r"(address_30));
                                        #pragma unroll
                                        for (int pair_18 = 0; pair_18 < 8; pair_18++) {
                                            __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_tmem_load_14[pair_18 * 2], _tmem_load_14[pair_18 * 2 + 1]));
                                            packed_12[chunk_22 * 16 + sub_8 * 8 + pair_18] = __as_u32(_bf16x2_11);
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int last_5 = 1;
                                last_5 = 1 - has_hi_7;
                                if (last_5 != 0) {
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                if (tid == 0) {
                                    int previous_offset_6 = (macro_1 + 1) * macro_size;
                                    int output_row_4 = x_7 * 256 + cta_rank_0 * 128;
                                    int _min_62 = ((macro_size) < (tokens - previous_offset_6) ? (macro_size) : (tokens - previous_offset_6));
                                    if (output_row_4 < _min_62) {
                                    }
                                }
                                #pragma unroll
                                for (int chunk_23 = 0; chunk_23 < 8; chunk_23++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_10 = tid / 32;
                                    int lane_14 = tid % 32;
                                    #pragma unroll
                                    for (int half_8 = 0; half_8 < 2; half_8++) {
                                        #pragma unroll
                                        for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                            int row_26 = warp_0_10 * 32 + half_8 * 16 + lane_14 % 16;
                                            int col_21 = col_tile_8 * 16 + lane_14 / 16 * 8;
                                            unsigned int address_31 = d_smem_addr + (unsigned int)(chunk_23 % 3 * 8192) + (unsigned int)((row_26 * 32 + col_21) * 2);
                                            address_31 = address_31 ^ (address_31 & 511) >> 7 << 4;
                                            int offset_13 = chunk_23 * 16 + half_8 * 8 + col_tile_8 * 4;
                                            uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(address_31);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_12[offset_13 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&dh_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"(y_7 * 2 * 8 + chunk_23), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_23 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                if (has_hi_7 != 0) {
                                    unsigned int packed_0_4[128];
                                    #pragma unroll
                                    for (int chunk_24 = 0; chunk_24 < 8; chunk_24++) {
                                        #pragma unroll
                                        for (int sub_9 = 0; sub_9 < 2; sub_9++) {
                                            unsigned int address_32 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_9 * 16 << 16) + 256 + (unsigned int)(chunk_24 * 32);
                                            float _tmem_load_15[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                                                : "r"(address_32));
                                            #pragma unroll
                                            for (int pair_19 = 0; pair_19 < 8; pair_19++) {
                                                __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_tmem_load_15[pair_19 * 2], _tmem_load_15[pair_19 * 2 + 1]));
                                                packed_0_4[chunk_24 * 16 + sub_9 * 8 + pair_19] = __as_u32(_bf16x2_12);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_1_4 = 1;
                                    if (last_1_4 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_25 = 0; chunk_25 < 8; chunk_25++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_11 = tid / 32;
                                        int lane_15 = tid % 32;
                                        #pragma unroll
                                        for (int half_9 = 0; half_9 < 2; half_9++) {
                                            #pragma unroll
                                            for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                                int row_27 = warp_0_11 * 32 + half_9 * 16 + lane_15 % 16;
                                                int col_22 = col_tile_9 * 16 + lane_15 / 16 * 8;
                                                unsigned int address_33 = d_smem_addr + (unsigned int)((8 + chunk_25) % 3 * 8192) + (unsigned int)((row_27 * 32 + col_22) * 2);
                                                address_33 = address_33 ^ (address_33 & 511) >> 7 << 4;
                                                int offset_14 = chunk_25 * 16 + half_9 * 8 + col_tile_9 * 4;
                                                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(address_33);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_4[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_4[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_4[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_4[offset_14 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dh_r)), "r"(0), "r"(x_7 * 256 + cta_rank_0 * 128), "r"((y_7 * 2 + 1) * 8 + chunk_25), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_25) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                            bool enabled_value_12 = 1;
                                            if (enabled_value_12 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_4 + (macro_rows_7 + x_7) * (intermediate / 256) + y_7 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                            bool enabled_value_0_3 = macros > 1;
                                            if (enabled_value_0_3 != 0) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_1_1 = 1;
                                                if (enabled_value_1_1 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                            if (has_hi_7 != 0) {
                                                asm volatile("cp.async.bulk.wait_group 0;");
                                                bool enabled_value_1_2 = 1;
                                                if (enabled_value_1_2 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dh_ready)) + (shared_down_4 + (macro_rows_7 + x_7) * (intermediate / 256) + y_7 * 2 + 1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        gemm_phase = phase_bits_11;
                    } else if (kind == 1) {
                        unsigned int phase_bits_12 = swiglu_phase;
                        int col_blocks_12 = intermediate / 128;
                        int num_tiles_2 = tokens / 128 * col_blocks_12;
                        int macro_row_offset_2 = macro_1 * (macro_size / 128);
                        int first_tile_2 = task_1 * 16 + cta_rank_0 * 8;
                        int tile_end_2 = num_tiles_2;
                        {
                            int global_mini_9 = macro_1 * (macro_size / mini_size) + mini_2;
                            int mini_tiles_1 = mini_size / 128 * col_blocks_12;
                            first_tile_2 = first_tile_2 + global_mini_9 * mini_tiles_1;
                            int _min_63 = ((num_tiles_2) < ((global_mini_9 + 1) * mini_tiles_1) ? (num_tiles_2) : ((global_mini_9 + 1) * mini_tiles_1));
                            tile_end_2 = _min_63;
                        }
                        int _min_64 = ((8) < (tile_end_2 - first_tile_2) ? (8) : (tile_end_2 - first_tile_2));
                        int _max_17 = ((0) > (_min_64) ? (0) : (_min_64));
                        int tiles_1 = _max_17;
                        if (tiles_1 > 0) {
                            if (tid == 0) {
                                int _min_65 = ((tiles_1) < (2) ? (tiles_1) : (2));
                                #pragma unroll 1
                                for (int stage_5 = 0; stage_5 < _min_65; stage_5++) {
                                    int col_blocks_0_3 = intermediate / 128;
                                    int row_28 = (first_tile_2 + stage_5) / col_blocks_0_3;
                                    int col_23 = (first_tile_2 + stage_5) % col_blocks_0_3;
                                    mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_5) * 8, 98304);
                                    int parent_3 = row_28 / 2 * (intermediate / 256) + col_23 / 2;
                                    int32_t _relaxed_ld_32;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_32) : "l"(dh_ready + (local_tokens / 256 * (intermediate / 256) + parent_3)) : "memory");
                                    int value_16 = _relaxed_ld_32;
                                    while (value_16 < 2) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_33;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_33) : "l"(dh_ready + (local_tokens / 256 * (intermediate / 256) + parent_3)) : "memory");
                                        value_16 = _relaxed_ld_33;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    bool enabled_value_13 = macro_1 > 0;
                                    if (enabled_value_13 != 0) {
                                        int32_t _relaxed_ld_34;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_34) : "l"(replay_gu + parent_3) : "memory");
                                        int value_0 = _relaxed_ld_34;
                                        while (value_0 < 4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_35;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_35) : "l"(replay_gu + parent_3) : "memory");
                                            value_0 = _relaxed_ld_35;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_dh_addr + (unsigned int)(stage_5 * 32768)), "l"((&dh_sw_r)), "r"(0), "r"((row_28 - macro_row_offset_2) * 128), "r"(col_23 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_5) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_gate_addr + (unsigned int)(stage_5 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_28 - macro_row_offset_2) * 128), "r"(col_23 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_5) * 8) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                        :: "r"(sw_up_addr + (unsigned int)(stage_5 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_28 - macro_row_offset_2) * 128), "r"(col_23 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_5) * 8) : "memory");
                                }
                            }
                            #pragma unroll 1
                            for (int index_6 = 0; index_6 < tiles_1; index_6++) {
                                int stage_6 = index_6 % 2;
                                mbarrier_wait(swiglu_arrived_addr + (stage_6) * 8, phase_bits_12 >> (unsigned int)stage_6 & 1);
                                phase_bits_12 = phase_bits_12 ^ (unsigned int)(1 << stage_6);
                                int row_29 = (first_tile_2 + index_6) / col_blocks_12;
                                int col_24 = (first_tile_2 + index_6) % col_blocks_12;
                                int tile_row = tid % 128;
                                int half_10 = tid / 128;
                                int local_row_3 = (row_29 - macro_row_offset_2) * 128 + tile_row;
                                int peer_7 = schedule_rank[row_29 * 128 + tile_row];
                                float weight_2 = weights[local_row_3];
                                float inverse = ((weight_2 > 0.0f) ? 1.0f / weight_2 : 0.0f);
                                float router_gradient = 0.0f;
                                #pragma unroll
                                for (int group = 0; group < 16; group++) {
                                    int tile_col_8 = half_10 * 64 + group * 4;
                                    unsigned int address_34 = sw_gate_addr + (unsigned int)(stage_6 * 32768) + (unsigned int)((tile_col_8 / 64 * 128 * 64 + tile_row * 64 + tile_col_8 % 64) * 2);
                                    unsigned int address_0 = sw_up_addr + (unsigned int)(stage_6 * 32768) + (unsigned int)((tile_col_8 / 64 * 128 * 64 + tile_row * 64 + tile_col_8 % 64) * 2);
                                    unsigned int address_1_1 = sw_dh_addr + (unsigned int)(stage_6 * 32768) + (unsigned int)((tile_col_8 / 64 * 128 * 64 + tile_row * 64 + tile_col_8 % 64) * 2);
                                    uint32_t _sw_gate_words_reg_0[2];
                                    uint64_t _smem_raw_19;
                                    asm volatile("ld.weak.shared::cta.b64 %0, [%1];" : "=l"(_smem_raw_19) : "r"(sw_gate_words_addr + (((address_34 ^ (address_34 & 1023) >> 7 << 4) - sw_gate_words_addr) / 4) * 4) : "memory");
                                    _sw_gate_words_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
                                    _sw_gate_words_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
                                    uint32_t _sw_up_words_reg_0[2];
                                    uint64_t _smem_raw_20;
                                    asm volatile("ld.weak.shared::cta.b64 %0, [%1];" : "=l"(_smem_raw_20) : "r"(sw_up_words_addr + (((address_0 ^ (address_0 & 1023) >> 7 << 4) - sw_up_words_addr) / 4) * 4) : "memory");
                                    _sw_up_words_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[0];
                                    _sw_up_words_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[1];
                                    uint32_t _sw_dh_words_reg_0[2];
                                    uint64_t _smem_raw_21;
                                    asm volatile("ld.weak.shared::cta.b64 %0, [%1];" : "=l"(_smem_raw_21) : "r"(sw_dh_words_addr + (((address_1_1 ^ (address_1_1 & 1023) >> 7 << 4) - sw_dh_words_addr) / 4) * 4) : "memory");
                                    _sw_dh_words_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[0];
                                    _sw_dh_words_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[1];
                                    unsigned int dgate_packed[2];
                                    unsigned int dup_packed[2];
                                    #pragma unroll
                                    for (int pair_20 = 0; pair_20 < 2; pair_20++) {
                                        float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(_sw_gate_words_reg_0[pair_20]));
                                        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(_sw_up_words_reg_0[pair_20]));
                                        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(_sw_dh_words_reg_0[pair_20]));
                                        float gate_mask_1 = ((_cvt_f32_5.x <= swiglu_limit) ? 1.0f : 0.0f);
                                        float up_mask_1 = ((_cvt_f32_6.x >= -swiglu_limit && _cvt_f32_6.x <= swiglu_limit) ? 1.0f : 0.0f);
                                        float _min_66 = fminf(_cvt_f32_5.x, swiglu_limit);
                                        float clamped_gate_1 = _min_66;
                                        float _fmax_2 = fmaxf(_cvt_f32_6.x, -swiglu_limit);
                                        float _min_67 = fminf(_fmax_2, swiglu_limit);
                                        float clamped_up_1 = _min_67;
                                        float _exp_2 = expf(-clamped_gate_1);
                                        float sigmoid_1 = 1.0f / (1.0f + _exp_2);
                                        float silu_1 = clamped_gate_1 * sigmoid_1;
                                        float dsilu_1 = (1.0f - silu_1) * sigmoid_1 + silu_1;
                                        float dgate_1 = ((gate_mask_1 != 0.0f) ? dsilu_1 * clamped_up_1 * _cvt_f32_7.x : 0.0f);
                                        float dup_1 = ((up_mask_1 != 0.0f) ? silu_1 * _cvt_f32_7.x : 0.0f);
                                        float gate_mask_0 = ((_cvt_f32_5.y <= swiglu_limit) ? 1.0f : 0.0f);
                                        float up_mask_1_1 = ((_cvt_f32_6.y >= -swiglu_limit && _cvt_f32_6.y <= swiglu_limit) ? 1.0f : 0.0f);
                                        float _min_68 = fminf(_cvt_f32_5.y, swiglu_limit);
                                        float clamped_gate_2 = _min_68;
                                        float _fmax_3 = fmaxf(_cvt_f32_6.y, -swiglu_limit);
                                        float _min_69 = fminf(_fmax_3, swiglu_limit);
                                        float clamped_up_3 = _min_69;
                                        float _exp_3 = expf(-clamped_gate_2);
                                        float sigmoid_4 = 1.0f / (1.0f + _exp_3);
                                        float silu_5 = clamped_gate_2 * sigmoid_4;
                                        float dsilu_6 = (1.0f - silu_5) * sigmoid_4 + silu_5;
                                        float dgate_7 = ((gate_mask_0 != 0.0f) ? dsilu_6 * clamped_up_3 * _cvt_f32_7.y : 0.0f);
                                        float dup_8 = ((up_mask_1_1 != 0.0f) ? silu_5 * _cvt_f32_7.y : 0.0f);
                                        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(dgate_1, dgate_7));
                                        dgate_packed[pair_20] = __as_u32(_bf16x2_13);
                                        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(dup_1, dup_8));
                                        dup_packed[pair_20] = __as_u32(_bf16x2_14);
                                        router_gradient = router_gradient + (_cvt_f32_7.x * inverse * (silu_1 * clamped_up_1) + _cvt_f32_7.y * inverse * (silu_5 * clamped_up_3));
                                    }
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(sw_gate_words_addr + ((address_34 ^ (address_34 & 1023) >> 7 << 4) - sw_gate_words_addr)), "r"(dgate_packed[0]), "r"(dgate_packed[1]) : "memory");
                                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(sw_up_words_addr + ((address_0 ^ (address_0 & 1023) >> 7 << 4) - sw_up_words_addr)), "r"(dup_packed[0]), "r"(dup_packed[1]) : "memory");
                                }
                                if (half_10 != 0) {
                                    sw_router[stage_6 * 128 + tile_row] = router_gradient;
                                }
                                __syncthreads();
                                if (half_10 == 0 && peer_7 >= 0) {
                                    router_gradient = router_gradient + sw_router[stage_6 * 128 + tile_row];
                                    partials[local_row_3 * col_blocks_12 + col_24] = router_gradient;
                                }
                                __syncthreads();
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&dg_sw_r), 0, (row_29 - macro_row_offset_2) * 128, col_24 * 2, 0, 0, sw_gate_addr + (unsigned int)(stage_6 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&du_sw_r), 0, (row_29 - macro_row_offset_2) * 128, col_24 * 2, 0, 0, sw_up_addr + (unsigned int)(stage_6 * 32768));
                                    asm volatile("cp.async.bulk.commit_group;");
                                    if (tiles_1 > index_6 + 2) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        int col_blocks_0_4 = intermediate / 128;
                                        int row_1_2 = (first_tile_2 + index_6 + 2) / col_blocks_0_4;
                                        int col_2_2 = (first_tile_2 + index_6 + 2) % col_blocks_0_4;
                                        mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_6) * 8, 98304);
                                        int parent_4 = row_1_2 / 2 * (intermediate / 256) + col_2_2 / 2;
                                        int32_t _relaxed_ld_36;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_36) : "l"(dh_ready + (local_tokens / 256 * (intermediate / 256) + parent_4)) : "memory");
                                        int value_17 = _relaxed_ld_36;
                                        while (value_17 < 2) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_37;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_37) : "l"(dh_ready + (local_tokens / 256 * (intermediate / 256) + parent_4)) : "memory");
                                            value_17 = _relaxed_ld_37;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                        bool enabled_value_14 = macro_1 > 0;
                                        if (enabled_value_14 != 0) {
                                            int32_t _relaxed_ld_38;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_38) : "l"(replay_gu + parent_4) : "memory");
                                            int value_0_1 = _relaxed_ld_38;
                                            while (value_0_1 < 4) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_39;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_39) : "l"(replay_gu + parent_4) : "memory");
                                                value_0_1 = _relaxed_ld_39;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                            :: "r"(sw_dh_addr + (unsigned int)(stage_6 * 32768)), "l"((&dh_sw_r)), "r"(0), "r"((row_1_2 - macro_row_offset_2) * 128), "r"(col_2_2 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_6) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                            :: "r"(sw_gate_addr + (unsigned int)(stage_6 * 32768)), "l"((&gate_sw_r)), "r"(0), "r"((row_1_2 - macro_row_offset_2) * 128), "r"(col_2_2 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_6) * 8) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                            :: "r"(sw_up_addr + (unsigned int)(stage_6 * 32768)), "l"((&up_sw_r)), "r"(0), "r"((row_1_2 - macro_row_offset_2) * 128), "r"(col_2_2 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_6) * 8) : "memory");
                                    }
                                }
                                __syncthreads();
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                #pragma unroll 1
                                for (int index_7 = 0; index_7 < tiles_1; index_7++) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dg_ready)) + (local_tokens / 256 + (first_tile_2 + index_7) / col_blocks_12 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                        }
                        if (tid == 0) {
                            bool enabled_value_15 = macros > 1;
                            if (enabled_value_15 != 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        swiglu_phase = phase_bits_12;
                    } else {
                        if (kind == 2) {
                            int col_blocks_13 = (intermediate + 512 - 1) / 512;
                            {
                                col_blocks_13 = (hidden + 512 - 1) / 512;
                            }
                            int x_8 = -1;
                            int y_8 = -1;
                            int expert_8 = -1;
                            int k_start_8 = 0;
                            int k_end_8 = 0;
                            int first_8 = 0;
                            int first_block_3 = (macro_1 * (macro_size / mini_size) + mini_2) * (mini_size / 256);
                            int _min_70 = ((first_block_3 + mini_size / 256) < (tokens / 256) ? (first_block_3 + mini_size / 256) : (tokens / 256));
                            int end_block_3 = _min_70;
                            int block_8 = first_block_3 + task_1 / col_blocks_13;
                            if (block_8 < end_block_3) {
                                int index_8 = counts[3 * experts + block_8];
                                int offset_15 = counts[experts + index_8] / 256;
                                int _max_18 = ((first_block_3) > (offset_15) ? (first_block_3) : (offset_15));
                                int first_row_10 = _max_18;
                                int _min_71 = ((end_block_3) < (offset_15 + counts[index_8] / 256) ? (end_block_3) : (offset_15 + counts[index_8] / 256));
                                int rows_7 = _min_71 - first_row_10;
                                int supergroup_8 = (task_1 - (first_row_10 - first_block_3) * col_blocks_13) / (rows_7 * 8);
                                int full_cols_8 = col_blocks_13 / 8 * 8;
                                int row_30 = 0;
                                int col_25 = 0;
                                if (task_1 - (first_row_10 - first_block_3) * col_blocks_13 < rows_7 * full_cols_8) {
                                    row_30 = (task_1 - (first_row_10 - first_block_3) * col_blocks_13) % (rows_7 * 8) / 8;
                                    col_25 = supergroup_8 * 8 + (task_1 - (first_row_10 - first_block_3) * col_blocks_13) % 8;
                                } else {
                                    row_30 = (task_1 - (first_row_10 - first_block_3) * col_blocks_13 - rows_7 * full_cols_8) / (col_blocks_13 - full_cols_8);
                                    col_25 = full_cols_8 + (task_1 - (first_row_10 - first_block_3) * col_blocks_13 - rows_7 * full_cols_8) % (col_blocks_13 - full_cols_8);
                                }
                                if ((supergroup_8 & 1) != 0) {
                                    row_30 = rows_7 - row_30 - 1;
                                }
                                x_8 = first_row_10 + row_30 - macro_1 * (macro_size / 256);
                                y_8 = col_25;
                                expert_8 = index_8;
                            }
                            unsigned int phase_bits_13 = gemm_phase;
                            int has_hi_8 = 0;
                            has_hi_8 = (int)((y_8 * 2 + 1) * 256 < hidden);
                            int global_mini_10 = macro_1 * (macro_size / mini_size) + mini_2;
                            int macro_rows_8 = macro_1 * (macro_size / 256);
                            int iterations_8 = intermediate / 64 + intermediate / 64;
                            if (expert_8 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_16 = macros > 1;
                                    if (enabled_value_16 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_17 = 1;
                                            if (enabled_value_17 != 0) {
                                                int32_t _relaxed_ld_40;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_40) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                int value_18 = _relaxed_ld_40;
                                                while (value_18 < row_count) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_41;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_41) : "l"(dg_ready + (shared_rows + macro_rows_8 + x_8)) : "memory");
                                                    value_18 = _relaxed_ld_41;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_72 = ((mini_size) < (tokens - global_mini_10 * mini_size) ? (mini_size) : (tokens - global_mini_10 * mini_size));
                                            int _max_19 = ((0) > (_min_72) ? (0) : (_min_72));
                                            int mini_rows_9 = _max_19;
                                            int required_6 = (mini_rows_9 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        int ring_16 = 0;
                                        #pragma unroll 1
                                        for (int idx_16 = 0; idx_16 < iterations_8; idx_16++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_16) * 8, phase_bits_13 >> (unsigned int)(16 + ring_16) & 1);
                                            if (idx_16 < intermediate / 64) {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_nt_addr + (unsigned int)(ring_16 * 16384)), "l"((&dg_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_16), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_addr + (unsigned int)(ring_16 * 16384)), "l"((&wg_r)), "r"(0), "r"(idx_16 * 64), "r"(y_8 * 2 * 4 + cta_rank_0 * 2), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_hi_addr + (unsigned int)(ring_16 * 16384)), "l"((&wg_r)), "r"(0), "r"(idx_16 * 64), "r"((y_8 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            } else {
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_nt_addr + (unsigned int)(ring_16 * 16384)), "l"((&du_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(idx_16 - intermediate / 64), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_addr + (unsigned int)(ring_16 * 16384)), "l"((&wu_r)), "r"(0), "r"((idx_16 - intermediate / 64) * 64), "r"(y_8 * 2 * 4 + cta_rank_0 * 2), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_hi_addr + (unsigned int)(ring_16 * 16384)), "l"((&wu_r)), "r"(0), "r"((idx_16 - intermediate / 64) * 64), "r"((y_8 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(expert_8), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_16) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            }
                                            phase_bits_13 = phase_bits_13 ^ (unsigned int)(1 << 16 + ring_16);
                                            ring_16 = (ring_16 + 1) % 4;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_17 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_13 >> 22 & 1);
                                            phase_bits_13 = phase_bits_13 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_17 = 0; idx_17 < iterations_8; idx_17++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_17) * 8, 98304);
                                                mbarrier_wait(gemm_arrived_addr + (ring_17) * 8, phase_bits_13 >> (unsigned int)ring_17 & 1);
                                                int _mma_a_lo_16 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_17) * 1024;
                                                int _mma_b_lo_16 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_17) * 1024;
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
            :: "r"(_mma_a_lo_16), "r"(_mma_b_lo_16), "r"(tmem_accumulator), "r"(((idx_17 == 0) ? 0 : 1)));
                                                int _mma_a_lo_17 = (((a_nt_addr) >> 4) & 0x3FFF) + (ring_17) * 1024;
                                                int _mma_b_lo_17 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_17) * 1024;
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
            :: "r"(_mma_a_lo_17), "r"(_mma_b_lo_17), "r"((tmem_accumulator + (256))), "r"(((idx_17 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_17) * 8, (uint16_t)(3));
                                                phase_bits_13 = phase_bits_13 ^ (unsigned int)(1 << ring_17);
                                                ring_17 = (ring_17 + 1) % 4;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_13 >> 6 & 1);
                                    phase_bits_13 = phase_bits_13 ^ 64;
                                    unsigned int packed_13[128];
                                    #pragma unroll
                                    for (int chunk_26 = 0; chunk_26 < 8; chunk_26++) {
                                        #pragma unroll
                                        for (int sub_10 = 0; sub_10 < 2; sub_10++) {
                                            unsigned int address_35 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_10 * 16 << 16) + (unsigned int)(chunk_26 * 32);
                                            float _tmem_load_16[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                                                : "r"(address_35));
                                            #pragma unroll
                                            for (int pair_21 = 0; pair_21 < 8; pair_21++) {
                                                __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_tmem_load_16[pair_21 * 2], _tmem_load_16[pair_21 * 2 + 1]));
                                                packed_13[chunk_26 * 16 + sub_10 * 8 + pair_21] = __as_u32(_bf16x2_15);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    int last_6 = 1;
                                    last_6 = 1 - has_hi_8;
                                    if (last_6 != 0) {
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile(
                                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        }
                                    }
                                    if (tid == 0) {
                                        int previous_offset_7 = (macro_1 + 1) * macro_size;
                                        int output_row_5 = x_8 * 256 + cta_rank_0 * 128;
                                        int _min_73 = ((macro_size) < (tokens - previous_offset_7) ? (macro_size) : (tokens - previous_offset_7));
                                        if (output_row_5 < _min_73) {
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_27 = 0; chunk_27 < 8; chunk_27++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_12 = tid / 32;
                                        int lane_16 = tid % 32;
                                        #pragma unroll
                                        for (int half_11 = 0; half_11 < 2; half_11++) {
                                            #pragma unroll
                                            for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                                int row_31 = warp_0_12 * 32 + half_11 * 16 + lane_16 % 16;
                                                int col_26 = col_tile_10 * 16 + lane_16 / 16 * 8;
                                                unsigned int address_36 = d_smem_addr + (unsigned int)(chunk_27 % 3 * 8192) + (unsigned int)((row_31 * 32 + col_26) * 2);
                                                address_36 = address_36 ^ (address_36 & 511) >> 7 << 4;
                                                int offset_16 = chunk_27 * 16 + half_11 * 8 + col_tile_10 * 4;
                                                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(address_36);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_13[offset_16 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&dx_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"(y_8 * 2 * 8 + chunk_27), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_27 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (has_hi_8 != 0) {
                                        unsigned int packed_0_5[128];
                                        #pragma unroll
                                        for (int chunk_28 = 0; chunk_28 < 8; chunk_28++) {
                                            #pragma unroll
                                            for (int sub_11 = 0; sub_11 < 2; sub_11++) {
                                                unsigned int address_37 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_11 * 16 << 16) + 256 + (unsigned int)(chunk_28 * 32);
                                                float _tmem_load_17[16];
                                                asm volatile(
                                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15]))
                                                    : "r"(address_37));
                                                #pragma unroll
                                                for (int pair_22 = 0; pair_22 < 8; pair_22++) {
                                                    __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_tmem_load_17[pair_22 * 2], _tmem_load_17[pair_22 * 2 + 1]));
                                                    packed_0_5[chunk_28 * 16 + sub_11 * 8 + pair_22] = __as_u32(_bf16x2_16);
                                                }
                                            }
                                        }
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        int last_1_5 = 1;
                                        if (last_1_5 != 0) {
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile(
                                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                            }
                                        }
                                        #pragma unroll
                                        for (int chunk_29 = 0; chunk_29 < 8; chunk_29++) {
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            int warp_0_13 = tid / 32;
                                            int lane_17 = tid % 32;
                                            #pragma unroll
                                            for (int half_12 = 0; half_12 < 2; half_12++) {
                                                #pragma unroll
                                                for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                                    int row_32 = warp_0_13 * 32 + half_12 * 16 + lane_17 % 16;
                                                    int col_27 = col_tile_11 * 16 + lane_17 / 16 * 8;
                                                    unsigned int address_38 = d_smem_addr + (unsigned int)((8 + chunk_29) % 3 * 8192) + (unsigned int)((row_32 * 32 + col_27) * 2);
                                                    address_38 = address_38 ^ (address_38 & 511) >> 7 << 4;
                                                    int offset_17 = chunk_29 * 16 + half_12 * 8 + col_tile_11 * 4;
                                                    uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(address_38);
                                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                        :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_5[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_5[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_5[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_5[offset_17 + 3]))
                                                        : "memory");
                                                }
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            if (tid == 0) {
                                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                    :: "l"((&dx_r)), "r"(0), "r"(x_8 * 256 + cta_rank_0 * 128), "r"((y_8 * 2 + 1) * 8 + chunk_29), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)((8 + chunk_29) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                bool enabled_value_18 = 1;
                                                if (enabled_value_18 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_10))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                }
                                                bool enabled_value_0_4 = macros > 1;
                                                if (enabled_value_0_4 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_1_3 = 1;
                                                    if (enabled_value_1_3 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                                if (has_hi_8 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_1_4 = 1;
                                                    if (enabled_value_1_4 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(dx_ready)) + (global_mini_10))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_phase = phase_bits_13;
                        } else if (kind == 3) {
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
                            int expert_idx_3 = 0;
                            int local_task_3 = task_1;
                            int _max_20 = ((row_blocks_5 * col_blocks_14) > (intermediate / 256 * ((hidden + 512 - 1) / 512)) ? (row_blocks_5 * col_blocks_14) : (intermediate / 256 * ((hidden + 512 - 1) / 512)));
                            int stride_3 = _max_20;
                            expert_idx_3 = task_1 / stride_3;
                            local_task_3 = task_1 % stride_3;
                            int offset_18 = counts[experts + expert_idx_3];
                            int real = counts[2 * experts + expert_idx_3];
                            real = (real + 64 - 1) / 64 * 64;
                            int _max_21 = ((offset_18) > (macro_1 * macro_size) ? (offset_18) : (macro_1 * macro_size));
                            k_start_9 = _max_21;
                            int _min_74 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                            int _min_75 = ((offset_18 + real) < (_min_74) ? (offset_18 + real) : (_min_74));
                            k_end_9 = _min_75;
                            first_9 = (int)(k_start_9 == offset_18);
                            if (k_start_9 < k_end_9 && local_task_3 < row_blocks_5 * col_blocks_14) {
                                int supergroup_9 = local_task_3 / (row_blocks_5 * 8);
                                int full_cols_9 = col_blocks_14 / 8 * 8;
                                int row_33 = 0;
                                int col_28 = 0;
                                if (local_task_3 < row_blocks_5 * full_cols_9) {
                                    row_33 = local_task_3 % (row_blocks_5 * 8) / 8;
                                    col_28 = supergroup_9 * 8 + local_task_3 % 8;
                                } else {
                                    row_33 = (local_task_3 - row_blocks_5 * full_cols_9) / (col_blocks_14 - full_cols_9);
                                    col_28 = full_cols_9 + (local_task_3 - row_blocks_5 * full_cols_9) % (col_blocks_14 - full_cols_9);
                                }
                                if ((supergroup_9 & 1) != 0) {
                                    row_33 = row_blocks_5 - row_33 - 1;
                                }
                                x_9 = row_33;
                                y_9 = col_28;
                                expert_9 = expert_idx_3;
                            }
                            unsigned int phase_bits_14 = gemm_phase;
                            int has_hi_9 = 0;
                            has_hi_9 = (int)((y_9 * 2 + 1) * 256 < intermediate);
                            int global_mini_11 = macro_1 * (macro_size / mini_size);
                            int macro_rows_9 = macro_1 * (macro_size / 256);
                            int iterations_9 = hidden / 64;
                            iterations_9 = (k_end_9 - k_start_9) / 64;
                            if (expert_9 < 0) {
                                if (tid == 0) {
                                    bool enabled_value_19 = macros > 1;
                                    if (enabled_value_19 != 0) {
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        int ring_18 = 0;
                                        #pragma unroll 1
                                        for (int idx_18 = 0; idx_18 < iterations_9; idx_18++) {
                                            int token_row_3 = k_start_9 + idx_18 * 64;
                                            if (idx_18 == 0 || token_row_3 % 256 == 0) {
                                                bool enabled_value_20 = macro_1 > 0;
                                                if (enabled_value_20 != 0) {
                                                    int32_t _relaxed_ld_46;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_46) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                    int value_19 = _relaxed_ld_46;
                                                    while (value_19 < row_count) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_47;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_47) : "l"(replay_h + (token_row_3 / 256)) : "memory");
                                                        value_19 = _relaxed_ld_47;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            if (idx_18 == 0 || token_row_3 % mini_size == 0) {
                                                int input_mini_3 = token_row_3 / mini_size;
                                                int _min_77 = ((mini_size) < (tokens - input_mini_3 * mini_size) ? (mini_size) : (tokens - input_mini_3 * mini_size));
                                                int input_rows_3 = _min_77;
                                                int input_count_3 = (input_rows_3 + 127) / 128 * ((hidden + 511) / 512);
                                                bool enabled_value_21 = 1;
                                                if (enabled_value_21 != 0) {
                                                    int32_t _relaxed_ld_48;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_48) : "l"(dy_ready + input_mini_3) : "memory");
                                                    int value_20 = _relaxed_ld_48;
                                                    while (value_20 < input_count_3) {
                                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                        int32_t _relaxed_ld_49;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_49) : "l"(dy_ready + input_mini_3) : "memory");
                                                        value_20 = _relaxed_ld_49;
                                                    }
                                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                                }
                                            }
                                            mbarrier_wait(gemm_finished_addr + (ring_18) * 8, phase_bits_14 >> (unsigned int)(16 + ring_18) & 1);
                                            int local_row_4 = k_start_9 + idx_18 * 64 - macro_1 * macro_size;
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_atb_addr + (unsigned int)(ring_18 * 16384)), "l"((&dy_atb_r)), "r"(0), "r"(local_row_4), "r"(x_9 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_18) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_addr + (unsigned int)(ring_18 * 16384)), "l"((&h_atb_r)), "r"(0), "r"(local_row_4), "r"(y_9 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_18) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_ab_hi_addr + (unsigned int)(ring_18 * 16384)), "l"((&h_atb_r)), "r"(0), "r"(local_row_4), "r"((y_9 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_18) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << 16 + ring_18);
                                            ring_18 = (ring_18 + 1) % 4;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_19 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_14 >> 22 & 1);
                                            phase_bits_14 = phase_bits_14 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_19 = 0; idx_19 < iterations_9; idx_19++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_19) * 8, 98304);
                                                mbarrier_wait(gemm_arrived_addr + (ring_19) * 8, phase_bits_14 >> (unsigned int)ring_19 & 1);
                                                int _mma_a_lo_18 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_19) * 1024;
                                                int _mma_b_lo_18 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_19) * 1024;
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
            :: "r"(_mma_a_lo_18), "r"(_mma_b_lo_18), "r"(tmem_accumulator), "r"(((idx_19 == 0) ? 0 : 1)));
                                                int _mma_a_lo_19 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_19) * 1024;
                                                int _mma_b_lo_19 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_19) * 1024;
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
            :: "r"(_mma_a_lo_19), "r"(_mma_b_lo_19), "r"((tmem_accumulator + (256))), "r"(((idx_19 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_19) * 8, (uint16_t)(3));
                                                phase_bits_14 = phase_bits_14 ^ (unsigned int)(1 << ring_19);
                                                ring_19 = (ring_19 + 1) % 4;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_14 >> 6 & 1);
                                    phase_bits_14 = phase_bits_14 ^ 64;
                                    int warp_row_3 = tid / 32 * 32;
                                    #pragma unroll
                                    for (int chunk_30 = 0; chunk_30 < 16; chunk_30++) {
                                        float _tmem_load_18[16];
                                        tmem_ld_x16(&_tmem_load_18[0], taddr_1 + (unsigned int)(warp_row_3 << 16) + (unsigned int)(chunk_30 * 16));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        #pragma unroll
                                        for (int vec_10 = 0; vec_10 < 4; vec_10++) {
                                            unsigned int address_39 = d_smem_addr + (unsigned int)(chunk_30 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_10 * 16);
                                            address_39 = address_39 ^ (address_39 & 511) >> 7 << 4;
                                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_39 - d_words_addr)), "f"(_tmem_load_18[vec_10 * 4]), "f"(_tmem_load_18[vec_10 * 4 + 1]), "f"(_tmem_load_18[vec_10 * 4 + 2]), "f"(_tmem_load_18[vec_10 * 4 + 3]) : "memory");
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
                                                :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"(y_9 * 2 * 16 + chunk_30), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_30 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                    if (has_hi_9 != 0) {
                                        #pragma unroll
                                        for (int chunk_31 = 0; chunk_31 < 16; chunk_31++) {
                                            float _tmem_load_19[16];
                                            tmem_ld_x16(&_tmem_load_19[0], taddr_1 + (unsigned int)(warp_row_3 << 16) + 256 + (unsigned int)(chunk_31 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_11 = 0; vec_11 < 4; vec_11++) {
                                                unsigned int address_40 = d_smem_addr + (unsigned int)((16 + chunk_31) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_11 * 16);
                                                address_40 = address_40 ^ (address_40 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_40 - d_words_addr)), "f"(_tmem_load_19[vec_11 * 4]), "f"(_tmem_load_19[vec_11 * 4 + 1]), "f"(_tmem_load_19[vec_11 * 4 + 2]), "f"(_tmem_load_19[vec_11 * 4 + 3]) : "memory");
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
                                                    :: "l"((&dwd_r)), "r"(0), "r"(x_9 * 256 + cta_rank_0 * 128), "r"((y_9 * 2 + 1) * 16 + chunk_31), "r"(expert_9), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_31) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                bool enabled_value_22 = macros > 1;
                                                if (enabled_value_22 != 0) {
                                                    asm volatile("cp.async.bulk.wait_group 0;");
                                                    bool enabled_value_0_5 = 1;
                                                    if (enabled_value_0_5 != 0) {
                                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                    }
                                                }
                                                if (has_hi_9 != 0) {
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            gemm_phase = phase_bits_14;
                        } else {
                            if (kind == 4) {
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
                                int expert_idx_4 = 0;
                                int local_task_4 = task_1;
                                int _max_23 = ((row_blocks_6 * col_blocks_15) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (row_blocks_6 * col_blocks_15) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
                                int stride_4 = _max_23;
                                expert_idx_4 = task_1 / stride_4;
                                local_task_4 = task_1 % stride_4;
                                int offset_19 = counts[experts + expert_idx_4];
                                int real_1 = counts[2 * experts + expert_idx_4];
                                real_1 = (real_1 + 64 - 1) / 64 * 64;
                                int _max_24 = ((offset_19) > (macro_1 * macro_size) ? (offset_19) : (macro_1 * macro_size));
                                k_start_10 = _max_24;
                                int _min_78 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                                int _min_79 = ((offset_19 + real_1) < (_min_78) ? (offset_19 + real_1) : (_min_78));
                                k_end_10 = _min_79;
                                first_10 = (int)(k_start_10 == offset_19);
                                if (k_start_10 < k_end_10 && local_task_4 < row_blocks_6 * col_blocks_15) {
                                    int supergroup_10 = local_task_4 / (row_blocks_6 * 8);
                                    int full_cols_10 = col_blocks_15 / 8 * 8;
                                    int row_34 = 0;
                                    int col_29 = 0;
                                    if (local_task_4 < row_blocks_6 * full_cols_10) {
                                        row_34 = local_task_4 % (row_blocks_6 * 8) / 8;
                                        col_29 = supergroup_10 * 8 + local_task_4 % 8;
                                    } else {
                                        row_34 = (local_task_4 - row_blocks_6 * full_cols_10) / (col_blocks_15 - full_cols_10);
                                        col_29 = full_cols_10 + (local_task_4 - row_blocks_6 * full_cols_10) % (col_blocks_15 - full_cols_10);
                                    }
                                    if ((supergroup_10 & 1) != 0) {
                                        row_34 = row_blocks_6 - row_34 - 1;
                                    }
                                    x_10 = row_34;
                                    y_10 = col_29;
                                    expert_10 = expert_idx_4;
                                }
                                unsigned int phase_bits_15 = gemm_phase;
                                int has_hi_10 = 0;
                                has_hi_10 = (int)((y_10 * 2 + 1) * 256 < hidden);
                                int global_mini_12 = macro_1 * (macro_size / mini_size);
                                int macro_rows_10 = macro_1 * (macro_size / 256);
                                int iterations_10 = intermediate / 64;
                                iterations_10 = (k_end_10 - k_start_10) / 64;
                                if (expert_10 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_23 = macros > 1;
                                        if (enabled_value_23 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            int ring_20 = 0;
                                            #pragma unroll 1
                                            for (int idx_20 = 0; idx_20 < iterations_10; idx_20++) {
                                                int token_row_4 = k_start_10 + idx_20 * 64;
                                                if (idx_20 == 0 || token_row_4 % 256 == 0) {
                                                    bool enabled_value_24 = 1;
                                                    if (enabled_value_24 != 0) {
                                                        int32_t _relaxed_ld_54;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_54) : "l"(dg_ready + (shared_rows + token_row_4 / 256)) : "memory");
                                                        int value_21 = _relaxed_ld_54;
                                                        while (value_21 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_55;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_55) : "l"(dg_ready + (shared_rows + token_row_4 / 256)) : "memory");
                                                            value_21 = _relaxed_ld_55;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_20 == 0 || token_row_4 % mini_size == 0) {
                                                    int input_mini_4 = token_row_4 / mini_size;
                                                    int _min_81 = ((mini_size) < (tokens - input_mini_4 * mini_size) ? (mini_size) : (tokens - input_mini_4 * mini_size));
                                                    int input_rows_4 = _min_81;
                                                    int input_count_4 = (input_rows_4 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_25 = macro_1 > 0;
                                                    if (enabled_value_25 != 0) {
                                                        int32_t _relaxed_ld_56;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_56) : "l"(replay_x + input_mini_4) : "memory");
                                                        int value_22 = _relaxed_ld_56;
                                                        while (value_22 < input_count_4) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_57;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_57) : "l"(replay_x + input_mini_4) : "memory");
                                                            value_22 = _relaxed_ld_57;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(gemm_finished_addr + (ring_20) * 8, phase_bits_15 >> (unsigned int)(16 + ring_20) & 1);
                                                int local_row_5 = k_start_10 + idx_20 * 64 - macro_1 * macro_size;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_atb_addr + (unsigned int)(ring_20 * 16384)), "l"((&dg_atb_r)), "r"(0), "r"(local_row_5), "r"(x_10 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_addr + (unsigned int)(ring_20 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_5), "r"(y_10 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_hi_addr + (unsigned int)(ring_20 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_5), "r"((y_10 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_20) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << 16 + ring_20);
                                                ring_20 = (ring_20 + 1) % 4;
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int ring_21 = 0;
                                                mbarrier_wait(output_finished_addr, phase_bits_15 >> 22 & 1);
                                                phase_bits_15 = phase_bits_15 ^ 4194304;
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_21 = 0; idx_21 < iterations_10; idx_21++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_21) * 8, 98304);
                                                    mbarrier_wait(gemm_arrived_addr + (ring_21) * 8, phase_bits_15 >> (unsigned int)ring_21 & 1);
                                                    int _mma_a_lo_20 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_21) * 1024;
                                                    int _mma_b_lo_20 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_21) * 1024;
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
            :: "r"(_mma_a_lo_20), "r"(_mma_b_lo_20), "r"(tmem_accumulator), "r"(((idx_21 == 0) ? 0 : 1)));
                                                    int _mma_a_lo_21 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_21) * 1024;
                                                    int _mma_b_lo_21 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_21) * 1024;
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
            :: "r"(_mma_a_lo_21), "r"(_mma_b_lo_21), "r"((tmem_accumulator + (256))), "r"(((idx_21 == 0) ? 0 : 1)));
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_21) * 8, (uint16_t)(3));
                                                    phase_bits_15 = phase_bits_15 ^ (unsigned int)(1 << ring_21);
                                                    ring_21 = (ring_21 + 1) % 4;
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_15 >> 6 & 1);
                                        phase_bits_15 = phase_bits_15 ^ 64;
                                        int warp_row_4 = tid / 32 * 32;
                                        #pragma unroll
                                        for (int chunk_32 = 0; chunk_32 < 16; chunk_32++) {
                                            float _tmem_load_20[16];
                                            tmem_ld_x16(&_tmem_load_20[0], taddr_1 + (unsigned int)(warp_row_4 << 16) + (unsigned int)(chunk_32 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_12 = 0; vec_12 < 4; vec_12++) {
                                                unsigned int address_41 = d_smem_addr + (unsigned int)(chunk_32 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_12 * 16);
                                                address_41 = address_41 ^ (address_41 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_41 - d_words_addr)), "f"(_tmem_load_20[vec_12 * 4]), "f"(_tmem_load_20[vec_12 * 4 + 1]), "f"(_tmem_load_20[vec_12 * 4 + 2]), "f"(_tmem_load_20[vec_12 * 4 + 3]) : "memory");
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
                                                    :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"(y_10 * 2 * 16 + chunk_32), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_32 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        if (has_hi_10 != 0) {
                                            #pragma unroll
                                            for (int chunk_33 = 0; chunk_33 < 16; chunk_33++) {
                                                float _tmem_load_21[16];
                                                tmem_ld_x16(&_tmem_load_21[0], taddr_1 + (unsigned int)(warp_row_4 << 16) + 256 + (unsigned int)(chunk_33 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_13 = 0; vec_13 < 4; vec_13++) {
                                                    unsigned int address_42 = d_smem_addr + (unsigned int)((16 + chunk_33) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_13 * 16);
                                                    address_42 = address_42 ^ (address_42 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_42 - d_words_addr)), "f"(_tmem_load_21[vec_13 * 4]), "f"(_tmem_load_21[vec_13 * 4 + 1]), "f"(_tmem_load_21[vec_13 * 4 + 2]), "f"(_tmem_load_21[vec_13 * 4 + 3]) : "memory");
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
                                                        :: "l"((&dwg_r)), "r"(0), "r"(x_10 * 256 + cta_rank_0 * 128), "r"((y_10 * 2 + 1) * 16 + chunk_33), "r"(expert_10), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_33) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                    bool enabled_value_26 = macros > 1;
                                                    if (enabled_value_26 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_0_6 = 1;
                                                        if (enabled_value_0_6 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                    if (has_hi_10 != 0) {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                gemm_phase = phase_bits_15;
                            } else if (kind == 5) {
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
                                int expert_idx_5 = 0;
                                int local_task_5 = task_1;
                                int _max_26 = ((row_blocks_7 * col_blocks_16) > (hidden / 256 * ((intermediate + 512 - 1) / 512)) ? (row_blocks_7 * col_blocks_16) : (hidden / 256 * ((intermediate + 512 - 1) / 512)));
                                int stride_5 = _max_26;
                                expert_idx_5 = task_1 / stride_5;
                                local_task_5 = task_1 % stride_5;
                                int offset_20 = counts[experts + expert_idx_5];
                                int real_2 = counts[2 * experts + expert_idx_5];
                                real_2 = (real_2 + 64 - 1) / 64 * 64;
                                int _max_27 = ((offset_20) > (macro_1 * macro_size) ? (offset_20) : (macro_1 * macro_size));
                                k_start_11 = _max_27;
                                int _min_82 = (((macro_1 + 1) * macro_size) < (tokens) ? ((macro_1 + 1) * macro_size) : (tokens));
                                int _min_83 = ((offset_20 + real_2) < (_min_82) ? (offset_20 + real_2) : (_min_82));
                                k_end_11 = _min_83;
                                first_11 = (int)(k_start_11 == offset_20);
                                if (k_start_11 < k_end_11 && local_task_5 < row_blocks_7 * col_blocks_16) {
                                    int supergroup_11 = local_task_5 / (row_blocks_7 * 8);
                                    int full_cols_11 = col_blocks_16 / 8 * 8;
                                    int row_35 = 0;
                                    int col_30 = 0;
                                    if (local_task_5 < row_blocks_7 * full_cols_11) {
                                        row_35 = local_task_5 % (row_blocks_7 * 8) / 8;
                                        col_30 = supergroup_11 * 8 + local_task_5 % 8;
                                    } else {
                                        row_35 = (local_task_5 - row_blocks_7 * full_cols_11) / (col_blocks_16 - full_cols_11);
                                        col_30 = full_cols_11 + (local_task_5 - row_blocks_7 * full_cols_11) % (col_blocks_16 - full_cols_11);
                                    }
                                    if ((supergroup_11 & 1) != 0) {
                                        row_35 = row_blocks_7 - row_35 - 1;
                                    }
                                    x_11 = row_35;
                                    y_11 = col_30;
                                    expert_11 = expert_idx_5;
                                }
                                unsigned int phase_bits_16 = gemm_phase;
                                int has_hi_11 = 0;
                                has_hi_11 = (int)((y_11 * 2 + 1) * 256 < hidden);
                                int global_mini_13 = macro_1 * (macro_size / mini_size);
                                int macro_rows_11 = macro_1 * (macro_size / 256);
                                int iterations_11 = intermediate / 64;
                                iterations_11 = (k_end_11 - k_start_11) / 64;
                                if (expert_11 < 0) {
                                    if (tid == 0) {
                                        bool enabled_value_27 = macros > 1;
                                        if (enabled_value_27 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                    }
                                } else if (tid / 32 == 7) {
                                    if (warp == 7) {
                                        if (elect_sync()) {
                                            int ring_22 = 0;
                                            #pragma unroll 1
                                            for (int idx_22 = 0; idx_22 < iterations_11; idx_22++) {
                                                int token_row_5 = k_start_11 + idx_22 * 64;
                                                if (idx_22 == 0 || token_row_5 % 256 == 0) {
                                                    bool enabled_value_28 = 1;
                                                    if (enabled_value_28 != 0) {
                                                        int32_t _relaxed_ld_62;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_62) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                        int value_23 = _relaxed_ld_62;
                                                        while (value_23 < row_count) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_63;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_63) : "l"(dg_ready + (shared_rows + token_row_5 / 256)) : "memory");
                                                            value_23 = _relaxed_ld_63;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                if (idx_22 == 0 || token_row_5 % mini_size == 0) {
                                                    int input_mini_5 = token_row_5 / mini_size;
                                                    int _min_85 = ((mini_size) < (tokens - input_mini_5 * mini_size) ? (mini_size) : (tokens - input_mini_5 * mini_size));
                                                    int input_rows_5 = _min_85;
                                                    int input_count_5 = (input_rows_5 + 127) / 128 * ((hidden + 511) / 512);
                                                    bool enabled_value_29 = macro_1 > 0;
                                                    if (enabled_value_29 != 0) {
                                                        int32_t _relaxed_ld_64;
                                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_64) : "l"(replay_x + input_mini_5) : "memory");
                                                        int value_24 = _relaxed_ld_64;
                                                        while (value_24 < input_count_5) {
                                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                            int32_t _relaxed_ld_65;
                                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_65) : "l"(replay_x + input_mini_5) : "memory");
                                                            value_24 = _relaxed_ld_65;
                                                        }
                                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                                    }
                                                }
                                                mbarrier_wait(gemm_finished_addr + (ring_22) * 8, phase_bits_16 >> (unsigned int)(16 + ring_22) & 1);
                                                int local_row_6 = k_start_11 + idx_22 * 64 - macro_1 * macro_size;
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(a_atb_addr + (unsigned int)(ring_22 * 16384)), "l"((&du_atb_r)), "r"(0), "r"(local_row_6), "r"(x_11 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_22) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_addr + (unsigned int)(ring_22 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_6), "r"(y_11 * 2 * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_22) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                asm volatile(
                                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                    :: "r"(b_ab_hi_addr + (unsigned int)(ring_22 * 16384)), "l"((&x_atb_r)), "r"(0), "r"(local_row_6), "r"((y_11 * 2 + 1) * 4 + cta_rank_0 * 2), "r"(0), "r"(0),
                                                       "r"(((gemm_arrived_addr + (ring_22) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                                phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << 16 + ring_22);
                                                ring_22 = (ring_22 + 1) % 4;
                                            }
                                        }
                                    }
                                } else {
                                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                                        if (warp == 4) {
                                            if (elect_sync()) {
                                                int ring_23 = 0;
                                                mbarrier_wait(output_finished_addr, phase_bits_16 >> 22 & 1);
                                                phase_bits_16 = phase_bits_16 ^ 4194304;
                                                asm volatile("tcgen05.fence::after_thread_sync;");
                                                #pragma unroll 1
                                                for (int idx_23 = 0; idx_23 < iterations_11; idx_23++) {
                                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_23) * 8, 98304);
                                                    mbarrier_wait(gemm_arrived_addr + (ring_23) * 8, phase_bits_16 >> (unsigned int)ring_23 & 1);
                                                    int _mma_a_lo_22 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_23) * 1024;
                                                    int _mma_b_lo_22 = ((((b_ab_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_23) * 1024;
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
            :: "r"(_mma_a_lo_22), "r"(_mma_b_lo_22), "r"(tmem_accumulator), "r"(((idx_23 == 0) ? 0 : 1)));
                                                    int _mma_a_lo_23 = ((((a_atb_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_23) * 1024;
                                                    int _mma_b_lo_23 = ((((b_ab_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (ring_23) * 1024;
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
            :: "r"(_mma_a_lo_23), "r"(_mma_b_lo_23), "r"((tmem_accumulator + (256))), "r"(((idx_23 == 0) ? 0 : 1)));
                                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_23) * 8, (uint16_t)(3));
                                                    phase_bits_16 = phase_bits_16 ^ (unsigned int)(1 << ring_23);
                                                    ring_23 = (ring_23 + 1) % 4;
                                                }
                                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                            }
                                        }
                                    } else if (tid < 128) {
                                        mbarrier_wait(output_arrived_addr, phase_bits_16 >> 6 & 1);
                                        phase_bits_16 = phase_bits_16 ^ 64;
                                        int warp_row_5 = tid / 32 * 32;
                                        #pragma unroll
                                        for (int chunk_34 = 0; chunk_34 < 16; chunk_34++) {
                                            float _tmem_load_22[16];
                                            tmem_ld_x16(&_tmem_load_22[0], taddr_1 + (unsigned int)(warp_row_5 << 16) + (unsigned int)(chunk_34 * 16));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            if (tid == 0) {
                                                asm volatile("cp.async.bulk.wait_group.read 2;");
                                            }
                                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                                            #pragma unroll
                                            for (int vec_14 = 0; vec_14 < 4; vec_14++) {
                                                unsigned int address_43 = d_smem_addr + (unsigned int)(chunk_34 % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_14 * 16);
                                                address_43 = address_43 ^ (address_43 & 511) >> 7 << 4;
                                                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_43 - d_words_addr)), "f"(_tmem_load_22[vec_14 * 4]), "f"(_tmem_load_22[vec_14 * 4 + 1]), "f"(_tmem_load_22[vec_14 * 4 + 2]), "f"(_tmem_load_22[vec_14 * 4 + 3]) : "memory");
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
                                                    :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"(y_11 * 2 * 16 + chunk_34), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_34 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
                                                asm volatile("cp.async.bulk.commit_group;");
                                            }
                                        }
                                        if (has_hi_11 != 0) {
                                            #pragma unroll
                                            for (int chunk_35 = 0; chunk_35 < 16; chunk_35++) {
                                                float _tmem_load_23[16];
                                                tmem_ld_x16(&_tmem_load_23[0], taddr_1 + (unsigned int)(warp_row_5 << 16) + 256 + (unsigned int)(chunk_35 * 16));
                                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                                if (tid == 0) {
                                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                                }
                                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                                #pragma unroll
                                                for (int vec_15 = 0; vec_15 < 4; vec_15++) {
                                                    unsigned int address_44 = d_smem_addr + (unsigned int)((16 + chunk_35) % 3 * 8192) + (unsigned int)(tid * 64) + (unsigned int)(vec_15 * 16);
                                                    address_44 = address_44 ^ (address_44 & 511) >> 7 << 4;
                                                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(d_words_addr + (address_44 - d_words_addr)), "f"(_tmem_load_23[vec_15 * 4]), "f"(_tmem_load_23[vec_15 * 4 + 1]), "f"(_tmem_load_23[vec_15 * 4 + 2]), "f"(_tmem_load_23[vec_15 * 4 + 3]) : "memory");
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
                                                        :: "l"((&dwu_r)), "r"(0), "r"(x_11 * 256 + cta_rank_0 * 128), "r"((y_11 * 2 + 1) * 16 + chunk_35), "r"(expert_11), "r"(0), "r"(d_smem_addr + (unsigned int)((16 + chunk_35) % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                    bool enabled_value_30 = macros > 1;
                                                    if (enabled_value_30 != 0) {
                                                        asm volatile("cp.async.bulk.wait_group 0;");
                                                        bool enabled_value_0_7 = 1;
                                                        if (enabled_value_0_7 != 0) {
                                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(buffers_done)) + (macro_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                                        }
                                                    }
                                                    if (has_hi_11 != 0) {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                gemm_phase = phase_bits_16;
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
            int shared_tasks_5 = shared_down + shared_swiglu + shared_dx + 3 * shared_wgrad;
            int mini_bwd_6 = mini_down + mini_swiglu + mini_dx;
            int mini_replay_7 = 2 * mini_down + mini_replay_swiglu;
            int weight_tasks_8 = experts * shared_wgrad;
            int _min_86 = ((tokens) < (macro_size) ? (tokens) : (macro_size));
            int saved_minis_9 = (_min_86 + mini_size - 1) / mini_size;
            int saved_tasks_10 = saved_minis_9 * mini_bwd_6 + 3 * weight_tasks_8;
            int replay_macro_tasks_11 = macro_size / mini_size * (mini_replay_7 + mini_bwd_6) + 3 * weight_tasks_8;
            int kind_12 = -1;
            int task_13 = 0;
            int macro_14 = 0;
            int mini_15 = 0;
            int shared_16 = 0;
            if (cluster - comm_clusters >= 0 && true_compute > cluster - comm_clusters) {
                if (shared_tasks_5 > cluster - comm_clusters) {
                    shared_16 = 1;
                    if (shared_down > cluster - comm_clusters) {
                        kind_12 = 0;
                        task_13 = cluster - comm_clusters;
                    } else if (cluster - comm_clusters < shared_down + shared_swiglu) {
                        kind_12 = 1;
                        task_13 = cluster - comm_clusters - shared_down;
                    } else {
                        if (cluster - comm_clusters < shared_down + shared_swiglu + shared_dx) {
                            kind_12 = 2;
                            task_13 = cluster - comm_clusters - shared_down - shared_swiglu;
                        } else {
                            int weight_task_2 = cluster - comm_clusters - shared_down - shared_swiglu - shared_dx;
                            kind_12 = 3 + weight_task_2 / shared_wgrad;
                            task_13 = weight_task_2 % shared_wgrad;
                        }
                    }
                } else {
                    int routed_1 = cluster - comm_clusters - shared_tasks_5;
                    int macro_task_1 = routed_1;
                    int replay_tasks_1 = 0;
                    if (routed_1 >= saved_tasks_10) {
                        macro_14 = 1 + (routed_1 - saved_tasks_10) / replay_macro_tasks_11;
                        macro_task_1 = (routed_1 - saved_tasks_10) % replay_macro_tasks_11;
                        int _min_87 = ((tokens - macro_14 * macro_size) < (macro_size) ? (tokens - macro_14 * macro_size) : (macro_size));
                        int macro_minis_2 = (_min_87 + mini_size - 1) / mini_size;
                        replay_tasks_1 = macro_minis_2 * mini_replay_7;
                    }
                    int _min_88 = ((tokens - macro_14 * macro_size) < (macro_size) ? (tokens - macro_14 * macro_size) : (macro_size));
                    int macro_minis_3 = (_min_88 + mini_size - 1) / mini_size;
                    if (macro_task_1 < replay_tasks_1) {
                        mini_15 = macro_task_1 / mini_replay_7;
                        int mini_task_2 = macro_task_1 % mini_replay_7;
                        if (mini_task_2 < mini_down) {
                            kind_12 = 6;
                            task_13 = mini_task_2;
                        } else if (mini_task_2 < 2 * mini_down) {
                            kind_12 = 7;
                            task_13 = mini_task_2 - mini_down;
                        } else {
                            kind_12 = 8;
                            task_13 = mini_task_2 - 2 * mini_down;
                        }
                    } else {
                        int bwd_task_1 = macro_task_1 - replay_tasks_1;
                        if (bwd_task_1 < macro_minis_3 * mini_bwd_6) {
                            mini_15 = bwd_task_1 / mini_bwd_6;
                            int mini_task_3 = bwd_task_1 % mini_bwd_6;
                            if (mini_task_3 < mini_down) {
                                kind_12 = 0;
                                task_13 = mini_task_3;
                            } else if (mini_task_3 < mini_down + mini_swiglu) {
                                kind_12 = 1;
                                task_13 = mini_task_3 - mini_down;
                            } else {
                                kind_12 = 2;
                                task_13 = mini_task_3 - mini_down - mini_swiglu;
                            }
                        } else {
                            int weight_task_3 = bwd_task_1 - macro_minis_3 * mini_bwd_6;
                            kind_12 = 3 + weight_task_3 / weight_tasks_8;
                            task_13 = weight_task_3 % weight_tasks_8;
                        }
                    }
                }
            }
            if ((kind == 1 || kind == 8) && cluster >= 0 && kind_12 != 1 && kind_12 != 8) {
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
