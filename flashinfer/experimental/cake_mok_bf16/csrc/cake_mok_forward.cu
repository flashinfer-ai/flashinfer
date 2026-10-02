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
#define TMEM_RESERVED_OFFSET 256
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
#define SMEM_D_SMEM_OFF 206848
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
#define SMEM_DISPATCH_SMEM_OFF 1024
#define SMEM_DISPATCH_SMEM_STAGE_BYTES 131072
#define SMEM_DISPATCH_SMEM_STRIDE 131072
#define SMEM_DISPATCH_WORDS_OFF 1024
#define SMEM_DISPATCH_WORDS_STAGE_BYTES 131072
#define SMEM_DISPATCH_WORDS_STRIDE 131072
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



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};




__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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
kernel_cake_mok_forward(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap x_routed, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap wg_routed, const __grid_constant__ CUtensorMap wu_routed, const __grid_constant__ CUtensorMap wd_routed, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap gate_shared_in, const __grid_constant__ CUtensorMap up_shared_in, const __grid_constant__ CUtensorMap gate_routed_in, const __grid_constant__ CUtensorMap up_routed_in, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_routed_out, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap hidden_routed_in, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ x_routed_ptr, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ gate_ready, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size)
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
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
    __nv_bfloat16* gate_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int gate_smem_addr = smem + 1024;
    __nv_bfloat16* up_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int up_smem_addr = smem + 99328;
    __nv_bfloat16* hidden_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int hidden_smem_addr = smem + 197632;
    __nv_bfloat16* dispatch_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int dispatch_smem_addr = smem + 1024;
    unsigned int* dispatch_words = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int dispatch_words_addr = smem + 1024;
    float* dispatch_weights = reinterpret_cast<float*>(smem_raw + 199680);
    const int dispatch_weights_addr = smem + 199680;
    __nv_bfloat16* combine_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int combine_smem_addr = smem + 1024;
    int tokens = num_tokens[0];
    int shared_rows = local_tokens / 256;
    int shared_gate = shared_rows * (intermediate / 256);
    int mini_gate = mini_size / 256 * (intermediate / 256);
    int shared_swiglu = (local_tokens / 128 * (intermediate / 128) + 5) / 6;
    int mini_swiglu = (mini_size / 128 * (intermediate / 128) + 5) / 6;
    int shared_tasks = 2 * shared_gate + shared_swiglu + shared_rows * (hidden / 256);
    int mini_tasks = 2 * mini_gate + mini_swiglu + mini_size / 256 * (hidden / 256);
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
    unsigned int taddr_1 = reinterpret_cast<const volatile unsigned int*>(reinterpret_cast<uint8_t*>(smem_raw) + CAKE_TMEM_HOLD_OFFSET)[0];
    int cluster = bid / 2;
    int cta_rank_0 = cta_rank;
    unsigned int gemm_bits = 4294901760;
    unsigned int swiglu_bits = 4294901760;
    unsigned int dispatch_bits = 4294901760;
    unsigned int combine_bits = 4294901760;
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
                    if (tid < 128) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row + tid) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_0), "r"(dispatch_smem_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                    __syncthreads();
                    if (tid == 0) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + row) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                dispatch_bits = phase_bits;
            }
            int macro = macros - 1;
            while (macro >= 0) {
                int _min_5 = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int macro_rows = _min_5;
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                int dispatch_tasks = 0;
                if (macro > 0) {
                    int _min_6 = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                    int previous_rows = _min_6;
                    dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                }
                int _max_0 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task_1 = comm_cta; task_1 < _max_0; task_1 += comm_sms) {
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
                                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[stage_3]]) + ((unsigned long long)tokens_0[stage_3] * (unsigned long long)hidden + (unsigned long long)(columns[stage_3] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_1), "r"(combine_smem_addr + (unsigned int)((stage_3 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_2))
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
                                    bool enabled_value_1 = 1;
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
                            if (tid < 128) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_2 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row_2 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block_1 * 512)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_2), "r"(dispatch_smem_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes_3))
                                        : "memory");
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncthreads();
                            if (tid == 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_2 + row_2) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
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
                int row_blocks = local_tokens / 256;
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
                                int _max_1 = ((0) > (_min_17) ? (0) : (_min_17));
                                int mini_rows_3 = _max_1;
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
                            for (int half = 0; half < 2; half++) {
                                unsigned int address = taddr_1 + (unsigned int)(tid / 32 * 32 + half * 16 << 16) + (unsigned int)(chunk * 32);
                                float _tmem_load_0[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                    : "r"(address));
                                #pragma unroll
                                for (int pair = 0; pair < 8; pair++) {
                                    __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]));
                                    packed[chunk * 16 + half * 8 + pair] = __as_u32(_bf16x2_0);
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
                            for (int half_1 = 0; half_1 < 2; half_1++) {
                                #pragma unroll
                                for (int col_tile = 0; col_tile < 2; col_tile++) {
                                    int row_4 = warp_0 * 32 + half_1 * 16 + lane_1 % 16;
                                    int col_1 = col_tile * 16 + lane_1 / 16 * 8;
                                    unsigned int address_1 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_4 * 32 + col_1) * 2);
                                    address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                    int offset = chunk_1 * 16 + half_1 * 8 + col_tile * 4;
                                    uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(address_1);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset + 3]))
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
                int row_blocks_1 = local_tokens / 256;
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
                                int _max_2 = ((0) > (_min_19) ? (0) : (_min_19));
                                int mini_rows_4 = _max_2;
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
                            for (int half_2 = 0; half_2 < 2; half_2++) {
                                unsigned int address_2 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_2 * 16 << 16) + (unsigned int)(chunk_2 * 32);
                                float _tmem_load_1[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                    : "r"(address_2));
                                #pragma unroll
                                for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                    __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_1[pair_1 * 2], _tmem_load_1[pair_1 * 2 + 1]));
                                    packed_1[chunk_2 * 16 + half_2 * 8 + pair_1] = __as_u32(_bf16x2_1);
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
                            for (int half_3 = 0; half_3 < 2; half_3++) {
                                #pragma unroll
                                for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                    int row_6 = warp_0_1 * 32 + half_3 * 16 + lane_2 % 16;
                                    int col_3 = col_tile_1 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_3 = d_smem_addr + (unsigned int)(chunk_3 % 3 * 8192) + (unsigned int)((row_6 * 32 + col_3) * 2);
                                    address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                    int offset_1 = chunk_3 * 16 + half_3 * 8 + col_tile_1 * 4;
                                    uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_3);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_1 + 3]))
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
                    int num_tiles = local_tokens / 128 * col_blocks_5;
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
                                    int32_t _relaxed_ld_4;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(gate_ready + parent) : "memory");
                                    int value_2 = _relaxed_ld_4;
                                    while (value_2 < 4) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_5;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(gate_ready + parent) : "memory");
                                        value_2 = _relaxed_ld_5;
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
                                    for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                                        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(gate[tile_col_2 * 8 + pair_4 * 2], gate[tile_col_2 * 8 + pair_4 * 2 + 1]));
                                        packed_4[pair_4] = __as_u32(_bf16x2_2);
                                    }
                                    unsigned int address_6 = hidden_smem_addr + (unsigned int)(((tile_col_2 * 16 + lane_6 / 16 * 8) / 64 * 128 * 64 + (local_warp_5 * 16 + lane_6 % 16) * 64 + (tile_col_2 * 16 + lane_6 / 16 * 8) % 64) * 2);
                                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(address_6 ^ (address_6 & 1023) >> 7 << 4);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3]))
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
                    int row_blocks_2 = local_tokens / 256;
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
                                        int32_t _relaxed_ld_6;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(hidden_ready + (macro_rows_3 + x_2)) : "memory");
                                        int value_3 = _relaxed_ld_6;
                                        while (value_3 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_7;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(hidden_ready + (macro_rows_3 + x_2)) : "memory");
                                            value_3 = _relaxed_ld_7;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_22 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                    int _max_3 = ((0) > (_min_22) ? (0) : (_min_22));
                                    int mini_rows_5 = _max_3;
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
                                for (int half_4 = 0; half_4 < 2; half_4++) {
                                    unsigned int address_7 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_4 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                    float _tmem_load_2[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                        : "r"(address_7));
                                    #pragma unroll
                                    for (int pair_5 = 0; pair_5 < 8; pair_5++) {
                                        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_2[pair_5 * 2], _tmem_load_2[pair_5 * 2 + 1]));
                                        packed_5[chunk_4 * 16 + half_4 * 8 + pair_5] = __as_u32(_bf16x2_3);
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
                                int _min_23 = ((macro_size) < (tokens - previous_offset_4) ? (macro_size) : (tokens - previous_offset_4));
                                if (output_row_2 < _min_23) {
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
                                for (int half_5 = 0; half_5 < 2; half_5++) {
                                    #pragma unroll
                                    for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                        int row_11 = warp_0_3 * 32 + half_5 * 16 + lane_4 % 16;
                                        int col_7 = col_tile_2 * 16 + lane_4 / 16 * 8;
                                        unsigned int address_8 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_11 * 32 + col_7) * 2);
                                        address_8 = address_8 ^ (address_8 & 511) >> 7 << 4;
                                        int offset_2 = chunk_5 * 16 + half_5 * 8 + col_tile_2 * 4;
                                        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(address_8);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_5[offset_2 + 3]))
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
                            int _max_4 = ((first_block) > (offset_3) ? (first_block) : (offset_3));
                            int first_row_2 = _max_4;
                            int _min_24 = ((first_block + mini_size / 256) < (offset_3 + blocks) ? (first_block + mini_size / 256) : (offset_3 + blocks));
                            int _max_5 = ((0) > (_min_24 - first_row_2) ? (0) : (_min_24 - first_row_2));
                            int rows_1 = _max_5;
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
                        int iterations_3 = hidden / 64;
                        if (expert_3 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_25 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                        int _max_6 = ((0) > (_min_25) ? (0) : (_min_25));
                                        int mini_rows_6 = _max_6;
                                        int required_6 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_5 = 1;
                                        if (enabled_value_5 != 0) {
                                            int32_t _relaxed_ld_8;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(x_ready + global_mini_3) : "memory");
                                            int value_4 = _relaxed_ld_8;
                                            while (value_4 < required_6) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_9;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(x_ready + global_mini_3) : "memory");
                                                value_4 = _relaxed_ld_9;
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
                                            :: "r"(a_smem_addr + (unsigned int)(ring_6 * 16384)), "l"((&x_routed)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_smem_addr + (unsigned int)(ring_6 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y_3 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_3), "r"(0),
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
                                            int _mma_a_lo_3 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_7) * 1024;
                                            int _mma_b_lo_3 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_7) * 1024;
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
                                unsigned int packed_6[128];
                                #pragma unroll
                                for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                    #pragma unroll
                                    for (int half_6 = 0; half_6 < 2; half_6++) {
                                        unsigned int address_9 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_6 * 16 << 16) + (unsigned int)(chunk_6 * 32);
                                        float _tmem_load_3[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                            : "r"(address_9));
                                        #pragma unroll
                                        for (int pair_6 = 0; pair_6 < 8; pair_6++) {
                                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_3[pair_6 * 2], _tmem_load_3[pair_6 * 2 + 1]));
                                            packed_6[chunk_6 * 16 + half_6 * 8 + pair_6] = __as_u32(_bf16x2_4);
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
                                    int output_row_3 = x_3 * 256 + cta_rank_0 * 128;
                                    int _min_26 = ((macro_size) < (tokens - previous_offset_5) ? (macro_size) : (tokens - previous_offset_5));
                                    if (output_row_3 < _min_26) {
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
                                    for (int half_7 = 0; half_7 < 2; half_7++) {
                                        #pragma unroll
                                        for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                            int row_13 = warp_0_4 * 32 + half_7 * 16 + lane_5 % 16;
                                            int col_9 = col_tile_3 * 16 + lane_5 / 16 * 8;
                                            unsigned int address_10 = d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192) + (unsigned int)((row_13 * 32 + col_9) * 2);
                                            address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                            int offset_0 = chunk_7 * 16 + half_7 * 8 + col_tile_3 * 4;
                                            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(address_10);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_0 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_0 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_6[offset_0 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&gate_routed_out)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(y_3 * 8 + chunk_7), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_7 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                            bool enabled_value_6 = 1;
                                            if (enabled_value_6 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_4 + x_3) * (intermediate / 256) + y_3))), "r"(static_cast<unsigned int>(1)) : "memory");
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
                            int _max_7 = ((first_block_1) > (offset_4) ? (first_block_1) : (offset_4));
                            int first_row_3 = _max_7;
                            int _min_27 = ((first_block_1 + mini_size / 256) < (offset_4 + blocks_1) ? (first_block_1 + mini_size / 256) : (offset_4 + blocks_1));
                            int _max_8 = ((0) > (_min_27 - first_row_3) ? (0) : (_min_27 - first_row_3));
                            int rows_2 = _max_8;
                            int tasks_1 = rows_2 * col_blocks_8;
                            if (remaining_1 < tasks_1) {
                                int supergroup_4 = remaining_1 / (rows_2 * 8);
                                int full_cols_4 = col_blocks_8 / 8 * 8;
                                int row_14 = 0;
                                int col_10 = 0;
                                if (remaining_1 < rows_2 * full_cols_4) {
                                    row_14 = remaining_1 % (rows_2 * 8) / 8;
                                    col_10 = supergroup_4 * 8 + remaining_1 % 8;
                                } else {
                                    row_14 = (remaining_1 - rows_2 * full_cols_4) / (col_blocks_8 - full_cols_4);
                                    col_10 = full_cols_4 + (remaining_1 - rows_2 * full_cols_4) % (col_blocks_8 - full_cols_4);
                                }
                                if ((supergroup_4 & 1) != 0) {
                                    row_14 = rows_2 - row_14 - 1;
                                }
                                x_4 = first_row_3 + row_14 - macro_1 * (macro_size / 256);
                                y_4 = col_10;
                                expert_4 = index_1;
                                break;
                            }
                            remaining_1 = remaining_1 - tasks_1;
                            offset_4 = offset_4 + blocks_1;
                        }
                        unsigned int phase_bits_8 = gemm_bits;
                        int global_mini_4 = macro_1 * (macro_size / mini_size) + mini_1;
                        int macro_rows_5 = macro_1 * (macro_size / 256);
                        int iterations_4 = hidden / 64;
                        if (expert_4 < 0) {
                            if (tid == 0) {
                            }
                        } else if (tid / 32 == 7) {
                            if (warp == 7) {
                                if (elect_sync()) {
                                    {
                                        int _min_28 = ((mini_size) < (tokens - global_mini_4 * mini_size) ? (mini_size) : (tokens - global_mini_4 * mini_size));
                                        int _max_9 = ((0) > (_min_28) ? (0) : (_min_28));
                                        int mini_rows_7 = _max_9;
                                        int required_7 = (mini_rows_7 + 127) / 128 * ((hidden + 511) / 512);
                                        bool enabled_value_7 = 1;
                                        if (enabled_value_7 != 0) {
                                            int32_t _relaxed_ld_10;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(x_ready + global_mini_4) : "memory");
                                            int value_5 = _relaxed_ld_10;
                                            while (value_5 < required_7) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_11;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(x_ready + global_mini_4) : "memory");
                                                value_5 = _relaxed_ld_11;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                        }
                                    }
                                    int ring_8 = 0;
                                    #pragma unroll 1
                                    for (int idx_8 = 0; idx_8 < iterations_4; idx_8++) {
                                        mbarrier_wait(gemm_finished_addr + (ring_8) * 8, phase_bits_8 >> (unsigned int)(16 + ring_8) & 1);
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(a_smem_addr + (unsigned int)(ring_8 * 16384)), "l"((&x_routed)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(idx_8), "r"(0), "r"(0),
                                               "r"(((gemm_arrived_addr + (ring_8) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_smem_addr + (unsigned int)(ring_8 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y_4 * 256 + cta_rank_0 * 128), "r"(idx_8), "r"(expert_4), "r"(0),
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
                                            int _mma_a_lo_4 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
                                            int _mma_b_lo_4 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_9) * 1024;
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
                                unsigned int packed_7[128];
                                #pragma unroll
                                for (int chunk_8 = 0; chunk_8 < 8; chunk_8++) {
                                    #pragma unroll
                                    for (int half_8 = 0; half_8 < 2; half_8++) {
                                        unsigned int address_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_8 * 16 << 16) + (unsigned int)(chunk_8 * 32);
                                        float _tmem_load_4[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                                            : "r"(address_11));
                                        #pragma unroll
                                        for (int pair_7 = 0; pair_7 < 8; pair_7++) {
                                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_4[pair_7 * 2], _tmem_load_4[pair_7 * 2 + 1]));
                                            packed_7[chunk_8 * 16 + half_8 * 8 + pair_7] = __as_u32(_bf16x2_5);
                                        }
                                    }
                                }
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                    int previous_offset_6 = (macro_1 + 1) * macro_size;
                                    int output_row_4 = x_4 * 256 + cta_rank_0 * 128;
                                    int _min_29 = ((macro_size) < (tokens - previous_offset_6) ? (macro_size) : (tokens - previous_offset_6));
                                    if (output_row_4 < _min_29) {
                                    }
                                }
                                #pragma unroll
                                for (int chunk_9 = 0; chunk_9 < 8; chunk_9++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_0_5 = tid / 32;
                                    int lane_7 = tid % 32;
                                    #pragma unroll
                                    for (int half_9 = 0; half_9 < 2; half_9++) {
                                        #pragma unroll
                                        for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                            int row_15 = warp_0_5 * 32 + half_9 * 16 + lane_7 % 16;
                                            int col_11 = col_tile_4 * 16 + lane_7 / 16 * 8;
                                            unsigned int address_12 = d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192) + (unsigned int)((row_15 * 32 + col_11) * 2);
                                            address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                            int offset_0_1 = chunk_9 * 16 + half_9 * 8 + col_tile_4 * 4;
                                            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(address_12);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_0_1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_0_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_0_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_7[offset_0_1 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                            " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                            :: "l"((&up_routed_out)), "r"(0), "r"(x_4 * 256 + cta_rank_0 * 128), "r"(y_4 * 8 + chunk_9), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_9 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                            bool enabled_value_8 = 1;
                                            if (enabled_value_8 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(gate_ready)) + (shared_gate + (macro_rows_5 + x_4) * (intermediate / 256) + y_4))), "r"(static_cast<unsigned int>(1)) : "memory");
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
                            int num_tiles_1 = tokens / 128 * col_blocks_9;
                            int macro_row_offset_1 = macro_1 * (macro_size / 128);
                            int first_tile_2 = (task_2 - 2 * mini_gate) * 6 + cta_rank_0 * 3;
                            int tile_end_1 = num_tiles_1;
                            {
                                int global_mini_5 = macro_1 * (macro_size / mini_size) + mini_1;
                                int mini_tiles = mini_size / 128 * col_blocks_9;
                                first_tile_2 = first_tile_2 + global_mini_5 * mini_tiles;
                                int _min_30 = ((num_tiles_1) < ((global_mini_5 + 1) * mini_tiles) ? (num_tiles_1) : ((global_mini_5 + 1) * mini_tiles));
                                tile_end_1 = _min_30;
                            }
                            if (first_tile_2 < tile_end_1) {
                                int first_row_4 = first_tile_2 / col_blocks_9;
                                int first_col_2 = first_tile_2 % col_blocks_9;
                                if (tid == 0) {
                                    #pragma unroll
                                    for (int stage_7 = 0; stage_7 < 3; stage_7++) {
                                        if (tile_end_1 > first_tile_2 + stage_7) {
                                            int row_16 = first_row_4;
                                            int col_12 = first_col_2 + stage_7;
                                            if (col_12 >= col_blocks_9) {
                                                row_16 = row_16 + 1;
                                                col_12 = col_12 - col_blocks_9;
                                            }
                                            mbarrier_arrive_expect_tx(swiglu_arrived_addr + (stage_7) * 8, 65536);
                                            int parent_1 = row_16 / 2 * (intermediate / 256) + col_12 / 2;
                                            int32_t _relaxed_ld_12;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                            int value_6 = _relaxed_ld_12;
                                            while (value_6 < 4) {
                                                asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                int32_t _relaxed_ld_13;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(gate_ready + (shared_gate + parent_1)) : "memory");
                                                value_6 = _relaxed_ld_13;
                                            }
                                            asm volatile("fence.acquire.gpu;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                                :: "r"(gate_smem_addr + (unsigned int)(stage_7 * 32768)), "l"((&gate_routed_in)), "r"(0), "r"((row_16 - macro_row_offset_1) * 128), "r"(col_12 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_7) * 8) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                                                :: "r"(up_smem_addr + (unsigned int)(stage_7 * 32768)), "l"((&up_routed_in)), "r"(0), "r"((row_16 - macro_row_offset_1) * 128), "r"(col_12 * 2), "r"(0), "r"(0), "r"(swiglu_arrived_addr + (stage_7) * 8) : "memory");
                                        }
                                    }
                                }
                                #pragma unroll 1
                                for (int stage_8 = 0; stage_8 < 3; stage_8++) {
                                    if (tile_end_1 > first_tile_2 + stage_8) {
                                        mbarrier_wait(swiglu_arrived_addr + (stage_8) * 8, phase_bits_9 >> (unsigned int)stage_8 & 1);
                                        phase_bits_9 = phase_bits_9 ^ (unsigned int)(1 << stage_8);
                                        int row_17 = first_row_4;
                                        int col_13 = first_col_2 + stage_8;
                                        if (col_13 >= col_blocks_9) {
                                            row_17 = row_17 + 1;
                                            col_13 = col_13 - col_blocks_9;
                                        }
                                        float gate_1[64];
                                        float up_1[64];
                                        float denominator_1[64];
                                        int warp_0_6 = tid / 32;
                                        int local_warp_1 = warp_0_6 / 4 + warp_0_6 % 4 * 2;
                                        int lane_8 = tid % 32;
                                        #pragma unroll
                                        for (int tile_col_3 = 0; tile_col_3 < 8; tile_col_3++) {
                                            unsigned int packed_8[4];
                                            unsigned int address_13 = gate_smem_addr + (unsigned int)(stage_8 * 32768) + (unsigned int)(((tile_col_3 * 16 + lane_8 / 16 * 8) / 64 * 128 * 64 + (local_warp_1 * 16 + lane_8 % 16) * 64 + (tile_col_3 * 16 + lane_8 / 16 * 8) % 64) * 2);
                                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                                : "=r"(packed_8[0]), "=r"(packed_8[1]), "=r"(packed_8[2]), "=r"(packed_8[3])
                                                : "r"(address_13 ^ (address_13 & 1023) >> 7 << 4)
                                                : "memory");
                                            #pragma unroll
                                            for (int pair_8 = 0; pair_8 < 4; pair_8++) {
                                                float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(packed_8[pair_8]));
                                                gate_1[tile_col_3 * 8 + pair_8 * 2] = _cvt_f32_2.x;
                                                gate_1[tile_col_3 * 8 + pair_8 * 2 + 1] = _cvt_f32_2.y;
                                            }
                                        }
                                        int warp_1_2 = tid / 32;
                                        int local_warp_2_1 = warp_1_2 / 4 + warp_1_2 % 4 * 2;
                                        int lane_3_2 = tid % 32;
                                        #pragma unroll
                                        for (int tile_col_4 = 0; tile_col_4 < 8; tile_col_4++) {
                                            unsigned int packed_9[4];
                                            unsigned int address_14 = up_smem_addr + (unsigned int)(stage_8 * 32768) + (unsigned int)(((tile_col_4 * 16 + lane_3_2 / 16 * 8) / 64 * 128 * 64 + (local_warp_2_1 * 16 + lane_3_2 % 16) * 64 + (tile_col_4 * 16 + lane_3_2 / 16 * 8) % 64) * 2);
                                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                                : "=r"(packed_9[0]), "=r"(packed_9[1]), "=r"(packed_9[2]), "=r"(packed_9[3])
                                                : "r"(address_14 ^ (address_14 & 1023) >> 7 << 4)
                                                : "memory");
                                            #pragma unroll
                                            for (int pair_9 = 0; pair_9 < 4; pair_9++) {
                                                float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(packed_9[pair_9]));
                                                up_1[tile_col_4 * 8 + pair_9 * 2] = _cvt_f32_3.x;
                                                up_1[tile_col_4 * 8 + pair_9 * 2 + 1] = _cvt_f32_3.y;
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
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 0;");
                                        }
                                        __syncthreads();
                                        int warp_4_1 = tid / 32;
                                        int local_warp_5_1 = warp_4_1 / 4 + warp_4_1 % 4 * 2;
                                        int lane_6_1 = tid % 32;
                                        #pragma unroll
                                        for (int tile_col_5 = 0; tile_col_5 < 8; tile_col_5++) {
                                            unsigned int packed_10[4];
                                            #pragma unroll
                                            for (int pair_10 = 0; pair_10 < 4; pair_10++) {
                                                __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(gate_1[tile_col_5 * 8 + pair_10 * 2], gate_1[tile_col_5 * 8 + pair_10 * 2 + 1]));
                                                packed_10[pair_10] = __as_u32(_bf16x2_6);
                                            }
                                            unsigned int address_15 = hidden_smem_addr + (unsigned int)(((tile_col_5 * 16 + lane_6_1 / 16 * 8) / 64 * 128 * 64 + (local_warp_5_1 * 16 + lane_6_1 % 16) * 64 + (tile_col_5 * 16 + lane_6_1 / 16 * 8) % 64) * 2);
                                            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(address_15 ^ (address_15 & 1023) >> 7 << 4);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_10[3]))
                                                : "memory");
                                        }
                                        __syncthreads();
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            tma_store_5d((&hidden_routed_out), 0, (row_17 - macro_row_offset_1) * 128, col_13 * 2, 0, 0, hidden_smem_addr);
                                            asm volatile("cp.async.bulk.commit_group;");
                                        }
                                    }
                                }
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group 0;");
                                    #pragma unroll
                                    for (int stage_9 = 0; stage_9 < 3; stage_9++) {
                                        if (tile_end_1 > first_tile_2 + stage_9) {
                                            int row_18 = first_row_4;
                                            if (col_blocks_9 <= first_col_2 + stage_9) {
                                                row_18 = row_18 + 1;
                                            }
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + row_18 / 2))), "r"(static_cast<unsigned int>(1)) : "memory");
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
                                int _max_10 = ((first_block_2) > (offset_5) ? (first_block_2) : (offset_5));
                                int first_row_5 = _max_10;
                                int _min_31 = ((first_block_2 + mini_size / 256) < (offset_5 + blocks_2) ? (first_block_2 + mini_size / 256) : (offset_5 + blocks_2));
                                int _max_11 = ((0) > (_min_31 - first_row_5) ? (0) : (_min_31 - first_row_5));
                                int rows_3 = _max_11;
                                int tasks_2 = rows_3 * col_blocks_10;
                                if (remaining_2 < tasks_2) {
                                    int supergroup_5 = remaining_2 / (rows_3 * 8);
                                    int full_cols_5 = col_blocks_10 / 8 * 8;
                                    int row_19 = 0;
                                    int col_14 = 0;
                                    if (remaining_2 < rows_3 * full_cols_5) {
                                        row_19 = remaining_2 % (rows_3 * 8) / 8;
                                        col_14 = supergroup_5 * 8 + remaining_2 % 8;
                                    } else {
                                        row_19 = (remaining_2 - rows_3 * full_cols_5) / (col_blocks_10 - full_cols_5);
                                        col_14 = full_cols_5 + (remaining_2 - rows_3 * full_cols_5) % (col_blocks_10 - full_cols_5);
                                    }
                                    if ((supergroup_5 & 1) != 0) {
                                        row_19 = rows_3 - row_19 - 1;
                                    }
                                    x_5 = first_row_5 + row_19 - macro_1 * (macro_size / 256);
                                    y_5 = col_14;
                                    expert_5 = index_2;
                                    break;
                                }
                                remaining_2 = remaining_2 - tasks_2;
                                offset_5 = offset_5 + blocks_2;
                            }
                            unsigned int phase_bits_10 = gemm_bits;
                            int global_mini_6 = macro_1 * (macro_size / mini_size) + mini_1;
                            int macro_rows_6 = macro_1 * (macro_size / 256);
                            int iterations_5 = intermediate / 64;
                            if (expert_5 < 0) {
                                if (tid == 0) {
                                }
                            } else if (tid / 32 == 7) {
                                if (warp == 7) {
                                    if (elect_sync()) {
                                        {
                                            bool enabled_value_9 = 1;
                                            if (enabled_value_9 != 0) {
                                                int32_t _relaxed_ld_14;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(hidden_ready + (shared_rows + macro_rows_6 + x_5)) : "memory");
                                                int value_7 = _relaxed_ld_14;
                                                while (value_7 < 2 * (intermediate / 128)) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_15;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(hidden_ready + (shared_rows + macro_rows_6 + x_5)) : "memory");
                                                    value_7 = _relaxed_ld_15;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                            int _min_32 = ((mini_size) < (tokens - global_mini_6 * mini_size) ? (mini_size) : (tokens - global_mini_6 * mini_size));
                                            int _max_12 = ((0) > (_min_32) ? (0) : (_min_32));
                                            int mini_rows_8 = _max_12;
                                            int required_8 = (mini_rows_8 + 127) / 128 * ((intermediate + 511) / 512);
                                        }
                                        int ring_10 = 0;
                                        #pragma unroll 1
                                        for (int idx_10 = 0; idx_10 < iterations_5; idx_10++) {
                                            mbarrier_wait(gemm_finished_addr + (ring_10) * 8, phase_bits_10 >> (unsigned int)(16 + ring_10) & 1);
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(a_smem_addr + (unsigned int)(ring_10 * 16384)), "l"((&hidden_routed_in)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(0), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                                :: "r"(b_smem_addr + (unsigned int)(ring_10 * 16384)), "l"((&wd_routed)), "r"(0), "r"(y_5 * 256 + cta_rank_0 * 128), "r"(idx_10), "r"(expert_5), "r"(0),
                                                   "r"(((gemm_arrived_addr + (ring_10) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                            phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << 16 + ring_10);
                                            ring_10 = (ring_10 + 1) % 6;
                                        }
                                    }
                                }
                            } else {
                                if (tid / 32 == 4 && cta_rank_0 == 0) {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            int ring_11 = 0;
                                            mbarrier_wait(output_finished_addr, phase_bits_10 >> 22 & 1);
                                            phase_bits_10 = phase_bits_10 ^ 4194304;
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            #pragma unroll 1
                                            for (int idx_11 = 0; idx_11 < iterations_5; idx_11++) {
                                                mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_11) * 8, 65536);
                                                mbarrier_wait(gemm_arrived_addr + (ring_11) * 8, phase_bits_10 >> (unsigned int)ring_11 & 1);
                                                int _mma_a_lo_5 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
                                                int _mma_b_lo_5 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_11) * 1024;
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
            :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_accumulator), "r"(((idx_11 == 0) ? 0 : 1)));
                                                tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_11) * 8, (uint16_t)(3));
                                                phase_bits_10 = phase_bits_10 ^ (unsigned int)(1 << ring_11);
                                                ring_11 = (ring_11 + 1) % 6;
                                            }
                                            tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                        }
                                    }
                                } else if (tid < 128) {
                                    mbarrier_wait(output_arrived_addr, phase_bits_10 >> 6 & 1);
                                    phase_bits_10 = phase_bits_10 ^ 64;
                                    unsigned int packed_11[128];
                                    #pragma unroll
                                    for (int chunk_10 = 0; chunk_10 < 8; chunk_10++) {
                                        #pragma unroll
                                        for (int half_10 = 0; half_10 < 2; half_10++) {
                                            unsigned int address_16 = taddr_1 + (unsigned int)(tid / 32 * 32 + half_10 * 16 << 16) + (unsigned int)(chunk_10 * 32);
                                            float _tmem_load_5[16];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                                                : "r"(address_16));
                                            #pragma unroll
                                            for (int pair_11 = 0; pair_11 < 8; pair_11++) {
                                                __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_5[pair_11 * 2], _tmem_load_5[pair_11 * 2 + 1]));
                                                packed_11[chunk_10 * 16 + half_10 * 8 + pair_11] = __as_u32(_bf16x2_7);
                                            }
                                        }
                                    }
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile(
                                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                            :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                                        int previous_offset_7 = (macro_1 + 1) * macro_size;
                                        int output_row_5 = x_5 * 256 + cta_rank_0 * 128;
                                        int _min_33 = ((macro_size) < (tokens - previous_offset_7) ? (macro_size) : (tokens - previous_offset_7));
                                        if (output_row_5 < _min_33) {
                                            bool enabled_value_10 = 1;
                                            if (enabled_value_10 != 0) {
                                                int32_t _relaxed_ld_16;
                                                asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_16) : "l"(y_done + ((previous_offset_7 + output_row_5) / 128)) : "memory");
                                                int value_8 = _relaxed_ld_16;
                                                while (value_8 < 8 * ((hidden + 1023) / 1024)) {
                                                    asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                                    int32_t _relaxed_ld_17;
                                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_17) : "l"(y_done + ((previous_offset_7 + output_row_5) / 128)) : "memory");
                                                    value_8 = _relaxed_ld_17;
                                                }
                                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int chunk_11 = 0; chunk_11 < 8; chunk_11++) {
                                        if (tid == 0) {
                                            asm volatile("cp.async.bulk.wait_group.read 2;");
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        int warp_0_7 = tid / 32;
                                        int lane_9 = tid % 32;
                                        #pragma unroll
                                        for (int half_11 = 0; half_11 < 2; half_11++) {
                                            #pragma unroll
                                            for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                                int row_20 = warp_0_7 * 32 + half_11 * 16 + lane_9 % 16;
                                                int col_15 = col_tile_5 * 16 + lane_9 / 16 * 8;
                                                unsigned int address_17 = d_smem_addr + (unsigned int)(chunk_11 % 3 * 8192) + (unsigned int)((row_20 * 32 + col_15) * 2);
                                                address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                                int offset_0_2 = chunk_11 * 16 + half_11 * 8 + col_tile_5 * 4;
                                                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(address_17);
                                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_0_2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_0_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_0_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_11[offset_0_2 + 3]))
                                                    : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                                        if (tid == 0) {
                                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                            asm volatile(
                                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                                :: "l"((&y_routed)), "r"(0), "r"(x_5 * 256 + cta_rank_0 * 128), "r"(y_5 * 8 + chunk_11), "r"(0), "r"(0), "r"(d_smem_addr + (unsigned int)(chunk_11 % 3 * 8192)), "l"(0x12F0000000000000ULL) : "memory");
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
                                                bool enabled_value_11 = 1;
                                                if (enabled_value_11 != 0) {
                                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_6))), "r"(static_cast<unsigned int>(1)) : "memory");
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
