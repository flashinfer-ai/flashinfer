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
#define SMEM_A_SMEM_OFF 1024
#define SMEM_A_SMEM_STAGE_BYTES 16384
#define SMEM_A_SMEM_STRIDE 16384
#define SMEM_B_SMEM_OFF 66560
#define SMEM_B_SMEM_STAGE_BYTES 16384
#define SMEM_B_SMEM_STRIDE 16384
#define SMEM_B_HI_OFF 132096
#define SMEM_B_HI_STAGE_BYTES 16384
#define SMEM_B_HI_STRIDE 16384
#define SMEM_D_SMEM_OFF 206848
#define SMEM_D_SMEM_STAGE_BYTES 8192
#define SMEM_D_SMEM_STRIDE 8192
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
#define SMEM_COMBINE_SMEM_OFF 1024
#define SMEM_COMBINE_SMEM_STAGE_BYTES 229376
#define SMEM_COMBINE_SMEM_STRIDE 229376
#define SMEM_TOTAL 232448
#define THREADS 256
#define CAKE_TMEM_HOLD_OFFSET 336

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
kernel_cake_mok_forward(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap x_routed, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap wg_routed, const __grid_constant__ CUtensorMap wu_routed, const __grid_constant__ CUtensorMap wd_routed, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_routed_out, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap hidden_routed_in, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ x_routed_ptr, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit)
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
    #define gemm_arrived_addr (mbar_base + 0)
    #define gemm_finished_addr (mbar_base + 32)
    #define output_arrived_addr (mbar_base + 64)
    #define output_finished_addr (mbar_base + 72)
    #define schedule_arrived_addr (mbar_base + 80)
    #define schedule_finished_addr (mbar_base + 88)
    #define drain_arrived_0_addr (mbar_base + 96)
    #define drain_arrived_1_addr (mbar_base + 104)
    #define drain_arrived_2_addr (mbar_base + 112)
    #define drain_arrived_3_addr (mbar_base + 120)
    #define drain_arrived_4_addr (mbar_base + 128)
    #define drain_arrived_5_addr (mbar_base + 136)
    #define drain_arrived_6_addr (mbar_base + 144)
    #define drain_arrived_7_addr (mbar_base + 152)
    #define drain_finished_0_addr (mbar_base + 160)
    #define drain_finished_1_addr (mbar_base + 168)
    #define drain_finished_2_addr (mbar_base + 176)
    #define drain_finished_3_addr (mbar_base + 184)
    #define drain_finished_4_addr (mbar_base + 192)
    #define drain_finished_5_addr (mbar_base + 200)
    #define drain_finished_6_addr (mbar_base + 208)
    #define drain_finished_7_addr (mbar_base + 216)
    #define dispatch_arrived_addr (mbar_base + 224)
    #define range_arrived_addr (mbar_base + 232)
    #define rows_arrived_addr (mbar_base + 256)
    #define combine_arrived_addr (mbar_base + 280)

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
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
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
    __nv_bfloat16* combine_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int combine_smem_addr = smem + 1024;
    int tokens = num_tokens[0];
    int shared_rows = local_tokens / 256;
    int shared_fused = shared_rows * (intermediate / 256);
    int mini_fused = mini_size / 256 * (intermediate / 256);
    int shared_tasks = shared_fused + shared_rows * ((hidden + 511) / 512);
    int mini_tasks = mini_fused + mini_size / 256 * ((hidden + 511) / 512);
    int comm_clusters = comm_sms / 2;
    int macros = (tokens + macro_size - 1) / macro_size;
    int minis_per_macro = macro_size / mini_size;
    int true_minis = (tokens + mini_size - 1) / mini_size;
    int last_minis = true_minis - (macros - 1) * minis_per_macro;
    int true_clusters = comm_clusters + shared_tasks + true_minis * mini_tasks;
    if (true_clusters <= bid / 2) return;
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (26 pipeline groups, 0 ordered-sequence groups, 42 barriers)
    // Mbarriers at smem_raw[0..336)

    if (threadIdx.x == 0) {
        // gemm_arrived: 4 barriers, init_count=1
        mbarrier_init(smem + 0, 1);
        mbarrier_init(smem + 8, 1);
        mbarrier_init(smem + 16, 1);
        mbarrier_init(smem + 24, 1);
        // gemm_finished: 4 barriers, init_count=1
        mbarrier_init(smem + 32, 1);
        mbarrier_init(smem + 40, 1);
        mbarrier_init(smem + 48, 1);
        mbarrier_init(smem + 56, 1);
        // output_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 64, 1);
        // output_finished: 1 barriers, init_count=2
        mbarrier_init(smem + 72, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 80, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 88, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 96, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 104, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 112, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 120, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 128, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 144, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 152, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 160, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 168, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 176, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 184, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 192, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 200, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 208, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 216, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 224, 1);
        // range_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 232, 1);
        mbarrier_init(smem + 240, 1);
        mbarrier_init(smem + 248, 1);
        // rows_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 256, 1);
        mbarrier_init(smem + 264, 1);
        mbarrier_init(smem + 272, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 280, 1);
        mbarrier_init(smem + 288, 1);
        mbarrier_init(smem + 296, 1);
        mbarrier_init(smem + 304, 1);
        mbarrier_init(smem + 312, 1);
        mbarrier_init(smem + 320, 1);
        mbarrier_init(smem + 328, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 336);
    if (warp == 0) {
        int _tmem_hold = smem + 336;
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
    unsigned int dispatch_bits = 4294901760;
    unsigned int combine_bits = 4294901760;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        if (macros > 0) {
            int col_blocks = (hidden + 511) / 512;
            int _min_0 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int unit_rows = _min_0;
            int block_units = (128 + unit_rows - 1) / unit_rows;
            int macro_offset = (macros - 1) * macro_size;
            int _min_1 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
            int macro_blocks = _min_1 / 128;
            int blocks = 0;
            if (comm_cta < macro_blocks) {
                blocks = (macro_blocks - comm_cta + comm_sms - 1) / comm_sms;
            }
            int units = blocks * block_units;
            int _min_2 = ((units) < (2) ? (units) : (2));
            #pragma unroll 1
            for (int unit = 0; unit < _min_2; unit++) {
                int buffer = unit % 3;
                int block = comm_cta + unit / block_units * comm_sms;
                int first_row = block * 128 + unit % block_units * unit_rows;
                int _min_3 = ((unit_rows) < (block * 128 + 128 - first_row) ? (unit_rows) : (block * 128 + 128 - first_row));
                int rows = _min_3;
                int peer = -1;
                int peer_token = -1;
                if (rows > tid) {
                    peer = schedule_rank[macro_offset + first_row + tid];
                    peer_token = schedule_token[macro_offset + first_row + tid];
                    range_flags[tid] = peer;
                }
                uint32_t _cta_count_0 = __syncthreads_count(peer >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(range_arrived_addr + (buffer) * 8, (unsigned int)(_cta_count_0 * (unsigned int)hidden * 2));
                }
                if (_cta_count_0 < (unsigned int)rows) {
                    #pragma unroll 1
                    for (int pad = 0; pad < rows; pad++) {
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
                if (peer >= 0) {
                    cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)((buffer * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + ((unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden) * (unsigned long long)2)), hidden * 2, range_arrived_addr + (buffer) * 8);
                }
            }
            #pragma unroll 1
            for (int unit_1 = 0; unit_1 < units; unit_1++) {
                int buffer_1 = unit_1 % 3;
                int block_1 = comm_cta + unit_1 / block_units * comm_sms;
                int first_row_1 = block_1 * 128 + unit_1 % block_units * unit_rows;
                int _min_4 = ((unit_rows) < (block_1 * 128 + 128 - first_row_1) ? (unit_rows) : (block_1 * 128 + 128 - first_row_1));
                int rows_1 = _min_4;
                mbarrier_wait(range_arrived_addr + (buffer_1) * 8, unit_1 / 3 & 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid == 0) {
                    {
                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)first_row_1 * (unsigned long long)hidden));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_0), "r"(dispatch_smem_addr + (unsigned int)(buffer_1 * 65536)), "r"((uint32_t)((unsigned int)(rows_1 * hidden * 2)))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    if (units > unit_1 + 2) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    if (unit_1 >= 2 && (unit_1 - 2) % block_units == block_units - 1) {
                        asm volatile("cp.async.bulk.wait_group 2;");
                        int done_block = comm_cta + (unit_1 - 2) / block_units * comm_sms;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + done_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                    }
                }
                if (units > unit_1 + 2) {
                    int buffer_0 = (unit_1 + 2) % 3;
                    int block_1_1 = comm_cta + (unit_1 + 2) / block_units * comm_sms;
                    int first_row_2 = block_1_1 * 128 + (unit_1 + 2) % block_units * unit_rows;
                    int _min_5 = ((unit_rows) < (block_1_1 * 128 + 128 - first_row_2) ? (unit_rows) : (block_1_1 * 128 + 128 - first_row_2));
                    int rows_3 = _min_5;
                    int peer_1 = -1;
                    int peer_token_1 = -1;
                    if (rows_3 > tid) {
                        peer_1 = schedule_rank[macro_offset + first_row_2 + tid];
                        peer_token_1 = schedule_token[macro_offset + first_row_2 + tid];
                        range_flags[tid] = peer_1;
                    }
                    uint32_t _cta_count_1 = __syncthreads_count(peer_1 >= 0);
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
                    if (peer_1 >= 0) {
                        cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)((buffer_0 * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + ((unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden) * (unsigned long long)2)), hidden * 2, range_arrived_addr + (buffer_0) * 8);
                    }
                }
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
                int _max_0 = ((0) > (units - 2) ? (0) : (units - 2));
                #pragma unroll 1
                for (int tail = _max_0; tail < units; tail++) {
                    if (tail % block_units == block_units - 1) {
                        int tail_block = comm_cta + tail / block_units * comm_sms;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset + tail_block * 128) / mini_size))), "r"(static_cast<unsigned int>(col_blocks)) : "memory");
                    }
                }
            }
            int macro = macros - 1;
            while (macro > 0) {
                int _min_6 = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int macro_rows = _min_6;
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                int _min_7 = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                int previous_rows = _min_7;
                int dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                int _max_1 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task = comm_cta; task < _max_1; task += comm_sms) {
                    if (combine_tasks > task) {
                        unsigned int phase_bits = combine_bits;
                        int col_blocks_0 = (hidden + 1023) / 1024;
                        int first_tile = task * 7;
                        int macro_offset_1 = macro * macro_size;
                        int _min_8 = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                        int macro_tokens = _min_8;
                        int _min_9 = ((7) < (macro_tokens / 16 * col_blocks_0 - first_tile) ? (7) : (macro_tokens / 16 * col_blocks_0 - first_tile));
                        int valid_tiles = _min_9;
                        if (valid_tiles > 0) {
                            int first_row_3 = first_tile / col_blocks_0 * 16 + tid;
                            int first_col = first_tile % col_blocks_0;
                            int rows_2[7];
                            int columns[7];
                            int peers[7];
                            int tokens_0[7];
                            unsigned int counts_1[7];
                            int row = first_row_3;
                            int column = first_col;
                            #pragma unroll
                            for (int stage = 0; stage < 7; stage++) {
                                rows_2[stage] = row;
                                columns[stage] = column;
                                peers[stage] = -1;
                                tokens_0[stage] = -1;
                                if (valid_tiles > stage && tid < 16) {
                                    peers[stage] = schedule_rank[macro_offset_1 + row];
                                    tokens_0[stage] = schedule_token[macro_offset_1 + row];
                                }
                                counts_1[stage] = 0;
                                if (valid_tiles > stage) {
                                    if (stage == 0 || column == 0) {
                                        uint32_t _cta_count_2 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _cta_count_2;
                                    } else {
                                        counts_1[stage] = counts_1[stage - 1];
                                    }
                                }
                                column = column + 1;
                                if (column == col_blocks_0) {
                                    column = 0;
                                    row = row + 16;
                                }
                            }
                            if (tid == 0) {
                                int first_mini = (macro_offset_1 + first_row_3) / mini_size;
                                int last_mini = (macro_offset_1 + (first_tile + valid_tiles - 1) / col_blocks_0 * 16) / mini_size;
                                #pragma unroll 1
                                for (int mini = first_mini; mini < last_mini + 1; mini++) {
                                    int _min_10 = ((mini_size) < (tokens - mini * mini_size) ? (mini_size) : (tokens - mini * mini_size));
                                    int mini_rows = _min_10;
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
                                        int _min_11 = ((1024) < (hidden - columns[stage_1] * 1024) ? (1024) : (hidden - columns[stage_1] * 1024));
                                        unsigned int chunk_bytes = (unsigned int)(_min_11 * 2);
                                        mbarrier_arrive_expect_tx(combine_arrived_addr + (stage_1) * 8, counts_1[stage_1] * chunk_bytes);
                                    }
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int stage_2 = 0; stage_2 < 7; stage_2++) {
                                if (peers[stage_2] >= 0) {
                                    int _min_12 = ((1024) < (hidden - columns[stage_2] * 1024) ? (1024) : (hidden - columns[stage_2] * 1024));
                                    int chunk_cols = _min_12;
                                    cp_async_bulk_gmem2smem(combine_smem_addr + (unsigned int)((stage_2 * 16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)rows_2[stage_2] * (unsigned long long)hidden + (unsigned long long)(columns[stage_2] * 1024)) * (unsigned long long)2)), chunk_cols * 2, combine_arrived_addr + (stage_2) * 8);
                                }
                            }
                            #pragma unroll
                            for (int stage_3 = 0; stage_3 < 7; stage_3++) {
                                if (valid_tiles > stage_3) {
                                    mbarrier_wait(combine_arrived_addr + (stage_3) * 8, phase_bits >> (unsigned int)stage_3 & 1);
                                    phase_bits = phase_bits ^ (unsigned int)(1 << stage_3);
                                    if (peers[stage_3] >= 0) {
                                        int _min_13 = ((1024) < (hidden - columns[stage_3] * 1024) ? (1024) : (hidden - columns[stage_3] * 1024));
                                        unsigned int chunk_bytes_1 = (unsigned int)(_min_13 * 2);
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        {
                                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[stage_3]]) + ((unsigned long long)tokens_0[stage_3] * (unsigned long long)hidden + (unsigned long long)(columns[stage_3] * 1024)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_1), "r"(combine_smem_addr + (unsigned int)((stage_3 * 16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_1))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            int warp_1 = tid / 32;
                            if (tid % 32 == 0 && warp_1 < valid_tiles) {
                                int row_done = (first_tile + warp_1) / col_blocks_0 * 16;
                                bool enabled_value = 1;
                                if (enabled_value != 0) {
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_1 + row_done) / 128))), "r"(static_cast<unsigned int>(1)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            __syncthreads();
                        }
                        combine_bits = phase_bits;
                    }
                    if (dispatch_tasks > task) {
                        unsigned int phase_bits_1 = dispatch_bits;
                        int col_blocks_0_1 = (hidden + 511) / 512;
                        int macro_offset_1_1 = (macro - 1) * macro_size;
                        int _min_14 = ((macro_size) < (tokens - macro_offset_1_1) ? (macro_size) : (tokens - macro_offset_1_1));
                        int macro_tokens_1 = _min_14;
                        if (task < macro_tokens_1 / 128 * col_blocks_0_1) {
                            int row_1 = task / col_blocks_0_1 * 128;
                            int col_block = task % col_blocks_0_1;
                            int _min_15 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                            int chunk_cols_1 = _min_15;
                            unsigned int chunk_bytes_2 = (unsigned int)(chunk_cols_1 * 2);
                            int peer_2 = -1;
                            int peer_token_2 = -1;
                            if (tid < 128) {
                                peer_2 = schedule_rank[macro_offset_1_1 + row_1 + tid];
                                peer_token_2 = schedule_token[macro_offset_1_1 + row_1 + tid];
                            }
                            uint32_t _cta_count_3 = __syncthreads_count(peer_2 >= 0);
                            if (tid == 0) {
                                int previous_offset = macro * macro_size;
                                int _min_16 = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                int previous_tokens = _min_16;
                                if (row_1 < previous_tokens) {
                                    int previous_mini = (previous_offset + row_1) / mini_size;
                                    int _min_17 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                    int mini_rows_1 = _min_17;
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    bool enabled_value_1 = 1;
                                    if (enabled_value_1 != 0) {
                                        int32_t _relaxed_ld_2;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_2) : "l"(y_ready + previous_mini) : "memory");
                                        int value_1 = _relaxed_ld_2;
                                        while (value_1 < required_1) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_3;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_3) : "l"(y_ready + previous_mini) : "memory");
                                            value_1 = _relaxed_ld_3;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_3 * chunk_bytes_2);
                            }
                            __syncthreads();
                            if (peer_2 >= 0) {
                                cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + ((unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2)), chunk_cols_1 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(dispatch_smem) + (tid * 1024 + vec_2 * 16))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                            mbarrier_wait(dispatch_arrived_addr, phase_bits_1 & 1);
                            phase_bits_1 = phase_bits_1 ^ 1;
                            if (tid < 128) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_2 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row_1 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_2), "r"(dispatch_smem_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes_2))
                                        : "memory");
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncthreads();
                            if (tid == 0) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready)) + ((macro_offset_1_1 + row_1) / mini_size))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        dispatch_bits = phase_bits_1;
                    }
                }
                macro = macro - 1;
            }
            int _min_18 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int unit_rows_0 = _min_18;
            int block_units_1 = (128 + unit_rows_0 - 1) / unit_rows_0;
            int macro_offset_2 = 0;
            int _min_19 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
            int macro_blocks_3 = _min_19 / 128;
            int blocks_4 = 0;
            if (comm_cta < macro_blocks_3) {
                blocks_4 = (macro_blocks_3 - comm_cta + comm_sms - 1) / comm_sms;
            }
            int units_5 = blocks_4 * block_units_1;
            int _min_20 = ((units_5) < (2) ? (units_5) : (2));
            #pragma unroll 1
            for (int unit_2 = 0; unit_2 < _min_20; unit_2++) {
                int slot = unit_2 % 3;
                int block_2 = comm_cta + unit_2 / block_units_1 * comm_sms;
                int first_row_4 = block_2 * 128 + unit_2 % block_units_1 * unit_rows_0;
                int _min_21 = ((unit_rows_0) < (block_2 * 128 + 128 - first_row_4) ? (unit_rows_0) : (block_2 * 128 + 128 - first_row_4));
                if (_min_21 > tid) {
                    combine_schedule[slot * 128 + tid] = schedule_rank[macro_offset_2 + first_row_4 + tid];
                    combine_schedule[384 + slot * 128 + tid] = schedule_token[macro_offset_2 + first_row_4 + tid];
                }
            }
            if (tid == 0) {
                int _min_22 = ((units_5) < (2) ? (units_5) : (2));
                #pragma unroll 1
                for (int unit_3 = 0; unit_3 < _min_22; unit_3++) {
                    int buffer_2 = unit_3 % 3;
                    int block_3 = comm_cta + unit_3 / block_units_1 * comm_sms;
                    int first_row_5 = block_3 * 128 + unit_3 % block_units_1 * unit_rows_0;
                    int _min_23 = ((unit_rows_0) < (block_3 * 128 + 128 - first_row_5) ? (unit_rows_0) : (block_3 * 128 + 128 - first_row_5));
                    int rows_4 = _min_23;
                    int mini_1 = (macro_offset_2 + first_row_5) / mini_size;
                    int _min_24 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                    int mini_rows_2 = _min_24;
                    int32_t _relaxed_ld_4;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_4) : "l"(y_ready + mini_1) : "memory");
                    int value_2 = _relaxed_ld_4;
                    while (value_2 < (mini_rows_2 + 255) / 256 * (hidden / 256) * 2) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int32_t _relaxed_ld_5;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_5) : "l"(y_ready + mini_1) : "memory");
                        value_2 = _relaxed_ld_5;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_2) * 8, (unsigned int)(rows_4 * hidden * 2));
                    cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(buffer_2 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)first_row_5 * (unsigned long long)hidden) * (unsigned long long)2)), rows_4 * hidden * 2, rows_arrived_addr + (buffer_2) * 8);
                }
            }
            __syncthreads();
            #pragma unroll 1
            for (int unit_4 = 0; unit_4 < units_5; unit_4++) {
                int buffer_3 = unit_4 % 3;
                int block_4 = comm_cta + unit_4 / block_units_1 * comm_sms;
                int first_row_6 = block_4 * 128 + unit_4 % block_units_1 * unit_rows_0;
                int _min_25 = ((unit_rows_0) < (block_4 * 128 + 128 - first_row_6) ? (unit_rows_0) : (block_4 * 128 + 128 - first_row_6));
                int rows_5 = _min_25;
                mbarrier_wait(rows_arrived_addr + (buffer_3) * 8, unit_4 / 3 & 1);
                if (rows_5 > tid) {
                    int peer_3 = combine_schedule[buffer_3 * 128 + tid];
                    int token = combine_schedule[384 + buffer_3 * 128 + tid];
                    if (peer_3 >= 0) {
                    }
                }
                if (tid == 0) {
                    #pragma unroll 1
                    for (int row_2 = 0; row_2 < rows_5; row_2++) {
                        int row_peer = combine_schedule[buffer_3 * 128 + row_2];
                        int row_token = combine_schedule[384 + buffer_3 * 128 + row_2];
                        if (row_peer >= 0) {
                            {
                                void* _cpbulk_dst_3 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[row_peer]) + ((unsigned long long)row_token * (unsigned long long)hidden));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_3), "r"(dispatch_smem_addr + (unsigned int)(buffer_3 * 65536) + (unsigned int)(row_2 * hidden * 2)), "r"((uint32_t)((unsigned int)(hidden * 2)))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    if (unit_4 % block_units_1 == block_units_1 - 1) {
                        bool enabled_value_2 = 0;
                        if (enabled_value_2 != 0) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_done)) + ((macro_offset_2 + block_4 * 128) / 128))), "r"(static_cast<unsigned int>(8 * ((hidden + 1023) / 1024))) : "memory");
                        }
                    }
                    if (units_5 > unit_4 + 2) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        int buffer_0_1 = (unit_4 + 2) % 3;
                        int block_1_2 = comm_cta + (unit_4 + 2) / block_units_1 * comm_sms;
                        int first_row_2_1 = block_1_2 * 128 + (unit_4 + 2) % block_units_1 * unit_rows_0;
                        int _min_26 = ((unit_rows_0) < (block_1_2 * 128 + 128 - first_row_2_1) ? (unit_rows_0) : (block_1_2 * 128 + 128 - first_row_2_1));
                        int rows_3_1 = _min_26;
                        int mini_2 = (macro_offset_2 + first_row_2_1) / mini_size;
                        int _min_27 = ((mini_size) < (tokens - mini_2 * mini_size) ? (mini_size) : (tokens - mini_2 * mini_size));
                        int mini_rows_3 = _min_27;
                        int32_t _relaxed_ld_6;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_6) : "l"(y_ready + mini_2) : "memory");
                        int value_3 = _relaxed_ld_6;
                        while (value_3 < (mini_rows_3 + 255) / 256 * (hidden / 256) * 2) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int32_t _relaxed_ld_7;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_7) : "l"(y_ready + mini_2) : "memory");
                            value_3 = _relaxed_ld_7;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_0_1) * 8, (unsigned int)(rows_3_1 * hidden * 2));
                        cp_async_bulk_gmem2smem(dispatch_smem_addr + (unsigned int)(buffer_0_1 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + ((unsigned long long)((unsigned long long)first_row_2_1 * (unsigned long long)hidden) * (unsigned long long)2)), rows_3_1 * hidden * 2, rows_arrived_addr + (buffer_0_1) * 8);
                    }
                }
                if (units_5 > unit_4 + 2) {
                    int slot_1 = (unit_4 + 2) % 3;
                    int block_0 = comm_cta + (unit_4 + 2) / block_units_1 * comm_sms;
                    int first_row_1_1 = block_0 * 128 + (unit_4 + 2) % block_units_1 * unit_rows_0;
                    int _min_28 = ((unit_rows_0) < (block_0 * 128 + 128 - first_row_1_1) ? (unit_rows_0) : (block_0 * 128 + 128 - first_row_1_1));
                    if (_min_28 > tid) {
                        combine_schedule[slot_1 * 128 + tid] = schedule_rank[macro_offset_2 + first_row_1_1 + tid];
                        combine_schedule[384 + slot_1 * 128 + tid] = schedule_token[macro_offset_2 + first_row_1_1 + tid];
                    }
                }
                __syncthreads();
            }
            if (tid == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
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
            if (compute < shared_fused) {
                int col_blocks_1 = (intermediate + 256 - 1) / 256;
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
                    int row_3 = 0;
                    int col = 0;
                    if (compute < row_blocks * full_cols) {
                        row_3 = compute % (row_blocks * 8) / 8;
                        col = supergroup * 8 + compute % 8;
                    } else {
                        row_3 = (compute - row_blocks * full_cols) / (col_blocks_1 - full_cols);
                        col = full_cols + (compute - row_blocks * full_cols) % (col_blocks_1 - full_cols);
                    }
                    if ((supergroup & 1) != 0) {
                        row_3 = row_blocks - row_3 - 1;
                    }
                    x = row_3;
                    y = col;
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
                                int _min_29 = ((mini_size) < (tokens - global_mini * mini_size) ? (mini_size) : (tokens - global_mini * mini_size));
                                int _max_2 = ((0) > (_min_29) ? (0) : (_min_29));
                                int mini_rows_4 = _max_2;
                                int required_2 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
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
                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        gate_packed[0] = __as_u32(_bf16x2_0);
                        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                        up_packed[0] = __as_u32(_bf16x2_1);
                        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        gate_packed[1] = __as_u32(_bf16x2_2);
                        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                        up_packed[1] = __as_u32(_bf16x2_3);
                        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                        gate_packed[2] = __as_u32(_bf16x2_4);
                        __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_1[4], _tmem_load_1[5]));
                        up_packed[2] = __as_u32(_bf16x2_5);
                        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                        gate_packed[3] = __as_u32(_bf16x2_6);
                        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_1[6], _tmem_load_1[7]));
                        up_packed[3] = __as_u32(_bf16x2_7);
                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                        gate_packed[4] = __as_u32(_bf16x2_8);
                        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_tmem_load_1[8], _tmem_load_1[9]));
                        up_packed[4] = __as_u32(_bf16x2_9);
                        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                        gate_packed[5] = __as_u32(_bf16x2_10);
                        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_tmem_load_1[10], _tmem_load_1[11]));
                        up_packed[5] = __as_u32(_bf16x2_11);
                        __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                        gate_packed[6] = __as_u32(_bf16x2_12);
                        __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_tmem_load_1[12], _tmem_load_1[13]));
                        up_packed[6] = __as_u32(_bf16x2_13);
                        __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                        gate_packed[7] = __as_u32(_bf16x2_14);
                        __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_tmem_load_1[14], _tmem_load_1[15]));
                        up_packed[7] = __as_u32(_bf16x2_15);
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
                        __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_tmem_load_2[0], _tmem_load_2[1]));
                        gate_packed[8] = __as_u32(_bf16x2_16);
                        __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_tmem_load_3[0], _tmem_load_3[1]));
                        up_packed[8] = __as_u32(_bf16x2_17);
                        __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_tmem_load_2[2], _tmem_load_2[3]));
                        gate_packed[9] = __as_u32(_bf16x2_18);
                        __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_tmem_load_3[2], _tmem_load_3[3]));
                        up_packed[9] = __as_u32(_bf16x2_19);
                        __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_tmem_load_2[4], _tmem_load_2[5]));
                        gate_packed[10] = __as_u32(_bf16x2_20);
                        __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_tmem_load_3[4], _tmem_load_3[5]));
                        up_packed[10] = __as_u32(_bf16x2_21);
                        __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(_tmem_load_2[6], _tmem_load_2[7]));
                        gate_packed[11] = __as_u32(_bf16x2_22);
                        __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(_tmem_load_3[6], _tmem_load_3[7]));
                        up_packed[11] = __as_u32(_bf16x2_23);
                        __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(_tmem_load_2[8], _tmem_load_2[9]));
                        gate_packed[12] = __as_u32(_bf16x2_24);
                        __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(_tmem_load_3[8], _tmem_load_3[9]));
                        up_packed[12] = __as_u32(_bf16x2_25);
                        __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(_tmem_load_2[10], _tmem_load_2[11]));
                        gate_packed[13] = __as_u32(_bf16x2_26);
                        __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(_tmem_load_3[10], _tmem_load_3[11]));
                        up_packed[13] = __as_u32(_bf16x2_27);
                        __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(_tmem_load_2[12], _tmem_load_2[13]));
                        gate_packed[14] = __as_u32(_bf16x2_28);
                        __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(_tmem_load_3[12], _tmem_load_3[13]));
                        up_packed[14] = __as_u32(_bf16x2_29);
                        __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(_tmem_load_2[14], _tmem_load_2[15]));
                        gate_packed[15] = __as_u32(_bf16x2_30);
                        __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(_tmem_load_3[14], _tmem_load_3[15]));
                        up_packed[15] = __as_u32(_bf16x2_31);
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
                        __nv_bfloat162 _bf16x2_32 = __float22bfloat162_rn(make_float2(_tmem_load_4[0], _tmem_load_4[1]));
                        gate_packed[16] = __as_u32(_bf16x2_32);
                        __nv_bfloat162 _bf16x2_33 = __float22bfloat162_rn(make_float2(_tmem_load_5[0], _tmem_load_5[1]));
                        up_packed[16] = __as_u32(_bf16x2_33);
                        __nv_bfloat162 _bf16x2_34 = __float22bfloat162_rn(make_float2(_tmem_load_4[2], _tmem_load_4[3]));
                        gate_packed[17] = __as_u32(_bf16x2_34);
                        __nv_bfloat162 _bf16x2_35 = __float22bfloat162_rn(make_float2(_tmem_load_5[2], _tmem_load_5[3]));
                        up_packed[17] = __as_u32(_bf16x2_35);
                        __nv_bfloat162 _bf16x2_36 = __float22bfloat162_rn(make_float2(_tmem_load_4[4], _tmem_load_4[5]));
                        gate_packed[18] = __as_u32(_bf16x2_36);
                        __nv_bfloat162 _bf16x2_37 = __float22bfloat162_rn(make_float2(_tmem_load_5[4], _tmem_load_5[5]));
                        up_packed[18] = __as_u32(_bf16x2_37);
                        __nv_bfloat162 _bf16x2_38 = __float22bfloat162_rn(make_float2(_tmem_load_4[6], _tmem_load_4[7]));
                        gate_packed[19] = __as_u32(_bf16x2_38);
                        __nv_bfloat162 _bf16x2_39 = __float22bfloat162_rn(make_float2(_tmem_load_5[6], _tmem_load_5[7]));
                        up_packed[19] = __as_u32(_bf16x2_39);
                        __nv_bfloat162 _bf16x2_40 = __float22bfloat162_rn(make_float2(_tmem_load_4[8], _tmem_load_4[9]));
                        gate_packed[20] = __as_u32(_bf16x2_40);
                        __nv_bfloat162 _bf16x2_41 = __float22bfloat162_rn(make_float2(_tmem_load_5[8], _tmem_load_5[9]));
                        up_packed[20] = __as_u32(_bf16x2_41);
                        __nv_bfloat162 _bf16x2_42 = __float22bfloat162_rn(make_float2(_tmem_load_4[10], _tmem_load_4[11]));
                        gate_packed[21] = __as_u32(_bf16x2_42);
                        __nv_bfloat162 _bf16x2_43 = __float22bfloat162_rn(make_float2(_tmem_load_5[10], _tmem_load_5[11]));
                        up_packed[21] = __as_u32(_bf16x2_43);
                        __nv_bfloat162 _bf16x2_44 = __float22bfloat162_rn(make_float2(_tmem_load_4[12], _tmem_load_4[13]));
                        gate_packed[22] = __as_u32(_bf16x2_44);
                        __nv_bfloat162 _bf16x2_45 = __float22bfloat162_rn(make_float2(_tmem_load_5[12], _tmem_load_5[13]));
                        up_packed[22] = __as_u32(_bf16x2_45);
                        __nv_bfloat162 _bf16x2_46 = __float22bfloat162_rn(make_float2(_tmem_load_4[14], _tmem_load_4[15]));
                        gate_packed[23] = __as_u32(_bf16x2_46);
                        __nv_bfloat162 _bf16x2_47 = __float22bfloat162_rn(make_float2(_tmem_load_5[14], _tmem_load_5[15]));
                        up_packed[23] = __as_u32(_bf16x2_47);
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
                        __nv_bfloat162 _bf16x2_48 = __float22bfloat162_rn(make_float2(_tmem_load_6[0], _tmem_load_6[1]));
                        gate_packed[24] = __as_u32(_bf16x2_48);
                        __nv_bfloat162 _bf16x2_49 = __float22bfloat162_rn(make_float2(_tmem_load_7[0], _tmem_load_7[1]));
                        up_packed[24] = __as_u32(_bf16x2_49);
                        __nv_bfloat162 _bf16x2_50 = __float22bfloat162_rn(make_float2(_tmem_load_6[2], _tmem_load_6[3]));
                        gate_packed[25] = __as_u32(_bf16x2_50);
                        __nv_bfloat162 _bf16x2_51 = __float22bfloat162_rn(make_float2(_tmem_load_7[2], _tmem_load_7[3]));
                        up_packed[25] = __as_u32(_bf16x2_51);
                        __nv_bfloat162 _bf16x2_52 = __float22bfloat162_rn(make_float2(_tmem_load_6[4], _tmem_load_6[5]));
                        gate_packed[26] = __as_u32(_bf16x2_52);
                        __nv_bfloat162 _bf16x2_53 = __float22bfloat162_rn(make_float2(_tmem_load_7[4], _tmem_load_7[5]));
                        up_packed[26] = __as_u32(_bf16x2_53);
                        __nv_bfloat162 _bf16x2_54 = __float22bfloat162_rn(make_float2(_tmem_load_6[6], _tmem_load_6[7]));
                        gate_packed[27] = __as_u32(_bf16x2_54);
                        __nv_bfloat162 _bf16x2_55 = __float22bfloat162_rn(make_float2(_tmem_load_7[6], _tmem_load_7[7]));
                        up_packed[27] = __as_u32(_bf16x2_55);
                        __nv_bfloat162 _bf16x2_56 = __float22bfloat162_rn(make_float2(_tmem_load_6[8], _tmem_load_6[9]));
                        gate_packed[28] = __as_u32(_bf16x2_56);
                        __nv_bfloat162 _bf16x2_57 = __float22bfloat162_rn(make_float2(_tmem_load_7[8], _tmem_load_7[9]));
                        up_packed[28] = __as_u32(_bf16x2_57);
                        __nv_bfloat162 _bf16x2_58 = __float22bfloat162_rn(make_float2(_tmem_load_6[10], _tmem_load_6[11]));
                        gate_packed[29] = __as_u32(_bf16x2_58);
                        __nv_bfloat162 _bf16x2_59 = __float22bfloat162_rn(make_float2(_tmem_load_7[10], _tmem_load_7[11]));
                        up_packed[29] = __as_u32(_bf16x2_59);
                        __nv_bfloat162 _bf16x2_60 = __float22bfloat162_rn(make_float2(_tmem_load_6[12], _tmem_load_6[13]));
                        gate_packed[30] = __as_u32(_bf16x2_60);
                        __nv_bfloat162 _bf16x2_61 = __float22bfloat162_rn(make_float2(_tmem_load_7[12], _tmem_load_7[13]));
                        up_packed[30] = __as_u32(_bf16x2_61);
                        __nv_bfloat162 _bf16x2_62 = __float22bfloat162_rn(make_float2(_tmem_load_6[14], _tmem_load_6[15]));
                        gate_packed[31] = __as_u32(_bf16x2_62);
                        __nv_bfloat162 _bf16x2_63 = __float22bfloat162_rn(make_float2(_tmem_load_7[14], _tmem_load_7[15]));
                        up_packed[31] = __as_u32(_bf16x2_63);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(gate_packed[0]));
                        float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(up_packed[0]));
                        float gate_x = _cvt_f32_0.x;
                        float gate_y = _cvt_f32_0.y;
                        float up_x = _cvt_f32_1.x;
                        float up_y = _cvt_f32_1.y;
                        float _exp_0 = expf(gate_x * -1.0f);
                        float denominator_x = _exp_0 + 1.0f;
                        float _exp_1 = expf(gate_y * -1.0f);
                        float denominator_y = _exp_1 + 1.0f;
                        float hidden_x = gate_x / denominator_x * up_x;
                        float hidden_y = gate_y / denominator_y * up_y;
                        __nv_bfloat162 _bf16x2_64 = __float22bfloat162_rn(make_float2(hidden_x, hidden_y));
                        hidden_packed[0] = __as_u32(_bf16x2_64);
                        float2 _cvt_f32_2 = __bfloat1622float2(__as_bf16x2(gate_packed[1]));
                        float2 _cvt_f32_3 = __bfloat1622float2(__as_bf16x2(up_packed[1]));
                        float gate_x_3 = _cvt_f32_2.x;
                        float gate_y_4 = _cvt_f32_2.y;
                        float up_x_5 = _cvt_f32_3.x;
                        float up_y_6 = _cvt_f32_3.y;
                        float _exp_2 = expf(gate_x_3 * -1.0f);
                        float denominator_x_7 = _exp_2 + 1.0f;
                        float _exp_3 = expf(gate_y_4 * -1.0f);
                        float denominator_y_8 = _exp_3 + 1.0f;
                        float hidden_x_9 = gate_x_3 / denominator_x_7 * up_x_5;
                        float hidden_y_10 = gate_y_4 / denominator_y_8 * up_y_6;
                        __nv_bfloat162 _bf16x2_65 = __float22bfloat162_rn(make_float2(hidden_x_9, hidden_y_10));
                        hidden_packed[1] = __as_u32(_bf16x2_65);
                        float2 _cvt_f32_4 = __bfloat1622float2(__as_bf16x2(gate_packed[2]));
                        float2 _cvt_f32_5 = __bfloat1622float2(__as_bf16x2(up_packed[2]));
                        float gate_x_11 = _cvt_f32_4.x;
                        float gate_y_12 = _cvt_f32_4.y;
                        float up_x_13 = _cvt_f32_5.x;
                        float up_y_14 = _cvt_f32_5.y;
                        float _exp_4 = expf(gate_x_11 * -1.0f);
                        float denominator_x_15 = _exp_4 + 1.0f;
                        float _exp_5 = expf(gate_y_12 * -1.0f);
                        float denominator_y_16 = _exp_5 + 1.0f;
                        float hidden_x_17 = gate_x_11 / denominator_x_15 * up_x_13;
                        float hidden_y_18 = gate_y_12 / denominator_y_16 * up_y_14;
                        __nv_bfloat162 _bf16x2_66 = __float22bfloat162_rn(make_float2(hidden_x_17, hidden_y_18));
                        hidden_packed[2] = __as_u32(_bf16x2_66);
                        float2 _cvt_f32_6 = __bfloat1622float2(__as_bf16x2(gate_packed[3]));
                        float2 _cvt_f32_7 = __bfloat1622float2(__as_bf16x2(up_packed[3]));
                        float gate_x_19 = _cvt_f32_6.x;
                        float gate_y_20 = _cvt_f32_6.y;
                        float up_x_21 = _cvt_f32_7.x;
                        float up_y_22 = _cvt_f32_7.y;
                        float _exp_6 = expf(gate_x_19 * -1.0f);
                        float denominator_x_23 = _exp_6 + 1.0f;
                        float _exp_7 = expf(gate_y_20 * -1.0f);
                        float denominator_y_24 = _exp_7 + 1.0f;
                        float hidden_x_25 = gate_x_19 / denominator_x_23 * up_x_21;
                        float hidden_y_26 = gate_y_20 / denominator_y_24 * up_y_22;
                        __nv_bfloat162 _bf16x2_67 = __float22bfloat162_rn(make_float2(hidden_x_25, hidden_y_26));
                        hidden_packed[3] = __as_u32(_bf16x2_67);
                        float2 _cvt_f32_8 = __bfloat1622float2(__as_bf16x2(gate_packed[4]));
                        float2 _cvt_f32_9 = __bfloat1622float2(__as_bf16x2(up_packed[4]));
                        float gate_x_27 = _cvt_f32_8.x;
                        float gate_y_28 = _cvt_f32_8.y;
                        float up_x_29 = _cvt_f32_9.x;
                        float up_y_30 = _cvt_f32_9.y;
                        float _exp_8 = expf(gate_x_27 * -1.0f);
                        float denominator_x_31 = _exp_8 + 1.0f;
                        float _exp_9 = expf(gate_y_28 * -1.0f);
                        float denominator_y_32 = _exp_9 + 1.0f;
                        float hidden_x_33 = gate_x_27 / denominator_x_31 * up_x_29;
                        float hidden_y_34 = gate_y_28 / denominator_y_32 * up_y_30;
                        __nv_bfloat162 _bf16x2_68 = __float22bfloat162_rn(make_float2(hidden_x_33, hidden_y_34));
                        hidden_packed[4] = __as_u32(_bf16x2_68);
                        float2 _cvt_f32_10 = __bfloat1622float2(__as_bf16x2(gate_packed[5]));
                        float2 _cvt_f32_11 = __bfloat1622float2(__as_bf16x2(up_packed[5]));
                        float gate_x_35 = _cvt_f32_10.x;
                        float gate_y_36 = _cvt_f32_10.y;
                        float up_x_37 = _cvt_f32_11.x;
                        float up_y_38 = _cvt_f32_11.y;
                        float _exp_10 = expf(gate_x_35 * -1.0f);
                        float denominator_x_39 = _exp_10 + 1.0f;
                        float _exp_11 = expf(gate_y_36 * -1.0f);
                        float denominator_y_40 = _exp_11 + 1.0f;
                        float hidden_x_41 = gate_x_35 / denominator_x_39 * up_x_37;
                        float hidden_y_42 = gate_y_36 / denominator_y_40 * up_y_38;
                        __nv_bfloat162 _bf16x2_69 = __float22bfloat162_rn(make_float2(hidden_x_41, hidden_y_42));
                        hidden_packed[5] = __as_u32(_bf16x2_69);
                        float2 _cvt_f32_12 = __bfloat1622float2(__as_bf16x2(gate_packed[6]));
                        float2 _cvt_f32_13 = __bfloat1622float2(__as_bf16x2(up_packed[6]));
                        float gate_x_43 = _cvt_f32_12.x;
                        float gate_y_44 = _cvt_f32_12.y;
                        float up_x_45 = _cvt_f32_13.x;
                        float up_y_46 = _cvt_f32_13.y;
                        float _exp_12 = expf(gate_x_43 * -1.0f);
                        float denominator_x_47 = _exp_12 + 1.0f;
                        float _exp_13 = expf(gate_y_44 * -1.0f);
                        float denominator_y_48 = _exp_13 + 1.0f;
                        float hidden_x_49 = gate_x_43 / denominator_x_47 * up_x_45;
                        float hidden_y_50 = gate_y_44 / denominator_y_48 * up_y_46;
                        __nv_bfloat162 _bf16x2_70 = __float22bfloat162_rn(make_float2(hidden_x_49, hidden_y_50));
                        hidden_packed[6] = __as_u32(_bf16x2_70);
                        float2 _cvt_f32_14 = __bfloat1622float2(__as_bf16x2(gate_packed[7]));
                        float2 _cvt_f32_15 = __bfloat1622float2(__as_bf16x2(up_packed[7]));
                        float gate_x_51 = _cvt_f32_14.x;
                        float gate_y_52 = _cvt_f32_14.y;
                        float up_x_53 = _cvt_f32_15.x;
                        float up_y_54 = _cvt_f32_15.y;
                        float _exp_14 = expf(gate_x_51 * -1.0f);
                        float denominator_x_55 = _exp_14 + 1.0f;
                        float _exp_15 = expf(gate_y_52 * -1.0f);
                        float denominator_y_56 = _exp_15 + 1.0f;
                        float hidden_x_57 = gate_x_51 / denominator_x_55 * up_x_53;
                        float hidden_y_58 = gate_y_52 / denominator_y_56 * up_y_54;
                        __nv_bfloat162 _bf16x2_71 = __float22bfloat162_rn(make_float2(hidden_x_57, hidden_y_58));
                        hidden_packed[7] = __as_u32(_bf16x2_71);
                        float2 _cvt_f32_16 = __bfloat1622float2(__as_bf16x2(gate_packed[8]));
                        float2 _cvt_f32_17 = __bfloat1622float2(__as_bf16x2(up_packed[8]));
                        float gate_x_59 = _cvt_f32_16.x;
                        float gate_y_60 = _cvt_f32_16.y;
                        float up_x_61 = _cvt_f32_17.x;
                        float up_y_62 = _cvt_f32_17.y;
                        float _exp_16 = expf(gate_x_59 * -1.0f);
                        float denominator_x_63 = _exp_16 + 1.0f;
                        float _exp_17 = expf(gate_y_60 * -1.0f);
                        float denominator_y_64 = _exp_17 + 1.0f;
                        float hidden_x_65 = gate_x_59 / denominator_x_63 * up_x_61;
                        float hidden_y_66 = gate_y_60 / denominator_y_64 * up_y_62;
                        __nv_bfloat162 _bf16x2_72 = __float22bfloat162_rn(make_float2(hidden_x_65, hidden_y_66));
                        hidden_packed[8] = __as_u32(_bf16x2_72);
                        float2 _cvt_f32_18 = __bfloat1622float2(__as_bf16x2(gate_packed[9]));
                        float2 _cvt_f32_19 = __bfloat1622float2(__as_bf16x2(up_packed[9]));
                        float gate_x_67 = _cvt_f32_18.x;
                        float gate_y_68 = _cvt_f32_18.y;
                        float up_x_69 = _cvt_f32_19.x;
                        float up_y_70 = _cvt_f32_19.y;
                        float _exp_18 = expf(gate_x_67 * -1.0f);
                        float denominator_x_71 = _exp_18 + 1.0f;
                        float _exp_19 = expf(gate_y_68 * -1.0f);
                        float denominator_y_72 = _exp_19 + 1.0f;
                        float hidden_x_73 = gate_x_67 / denominator_x_71 * up_x_69;
                        float hidden_y_74 = gate_y_68 / denominator_y_72 * up_y_70;
                        __nv_bfloat162 _bf16x2_73 = __float22bfloat162_rn(make_float2(hidden_x_73, hidden_y_74));
                        hidden_packed[9] = __as_u32(_bf16x2_73);
                        float2 _cvt_f32_20 = __bfloat1622float2(__as_bf16x2(gate_packed[10]));
                        float2 _cvt_f32_21 = __bfloat1622float2(__as_bf16x2(up_packed[10]));
                        float gate_x_75 = _cvt_f32_20.x;
                        float gate_y_76 = _cvt_f32_20.y;
                        float up_x_77 = _cvt_f32_21.x;
                        float up_y_78 = _cvt_f32_21.y;
                        float _exp_20 = expf(gate_x_75 * -1.0f);
                        float denominator_x_79 = _exp_20 + 1.0f;
                        float _exp_21 = expf(gate_y_76 * -1.0f);
                        float denominator_y_80 = _exp_21 + 1.0f;
                        float hidden_x_81 = gate_x_75 / denominator_x_79 * up_x_77;
                        float hidden_y_82 = gate_y_76 / denominator_y_80 * up_y_78;
                        __nv_bfloat162 _bf16x2_74 = __float22bfloat162_rn(make_float2(hidden_x_81, hidden_y_82));
                        hidden_packed[10] = __as_u32(_bf16x2_74);
                        float2 _cvt_f32_22 = __bfloat1622float2(__as_bf16x2(gate_packed[11]));
                        float2 _cvt_f32_23 = __bfloat1622float2(__as_bf16x2(up_packed[11]));
                        float gate_x_83 = _cvt_f32_22.x;
                        float gate_y_84 = _cvt_f32_22.y;
                        float up_x_85 = _cvt_f32_23.x;
                        float up_y_86 = _cvt_f32_23.y;
                        float _exp_22 = expf(gate_x_83 * -1.0f);
                        float denominator_x_87 = _exp_22 + 1.0f;
                        float _exp_23 = expf(gate_y_84 * -1.0f);
                        float denominator_y_88 = _exp_23 + 1.0f;
                        float hidden_x_89 = gate_x_83 / denominator_x_87 * up_x_85;
                        float hidden_y_90 = gate_y_84 / denominator_y_88 * up_y_86;
                        __nv_bfloat162 _bf16x2_75 = __float22bfloat162_rn(make_float2(hidden_x_89, hidden_y_90));
                        hidden_packed[11] = __as_u32(_bf16x2_75);
                        float2 _cvt_f32_24 = __bfloat1622float2(__as_bf16x2(gate_packed[12]));
                        float2 _cvt_f32_25 = __bfloat1622float2(__as_bf16x2(up_packed[12]));
                        float gate_x_91 = _cvt_f32_24.x;
                        float gate_y_92 = _cvt_f32_24.y;
                        float up_x_93 = _cvt_f32_25.x;
                        float up_y_94 = _cvt_f32_25.y;
                        float _exp_24 = expf(gate_x_91 * -1.0f);
                        float denominator_x_95 = _exp_24 + 1.0f;
                        float _exp_25 = expf(gate_y_92 * -1.0f);
                        float denominator_y_96 = _exp_25 + 1.0f;
                        float hidden_x_97 = gate_x_91 / denominator_x_95 * up_x_93;
                        float hidden_y_98 = gate_y_92 / denominator_y_96 * up_y_94;
                        __nv_bfloat162 _bf16x2_76 = __float22bfloat162_rn(make_float2(hidden_x_97, hidden_y_98));
                        hidden_packed[12] = __as_u32(_bf16x2_76);
                        float2 _cvt_f32_26 = __bfloat1622float2(__as_bf16x2(gate_packed[13]));
                        float2 _cvt_f32_27 = __bfloat1622float2(__as_bf16x2(up_packed[13]));
                        float gate_x_99 = _cvt_f32_26.x;
                        float gate_y_100 = _cvt_f32_26.y;
                        float up_x_101 = _cvt_f32_27.x;
                        float up_y_102 = _cvt_f32_27.y;
                        float _exp_26 = expf(gate_x_99 * -1.0f);
                        float denominator_x_103 = _exp_26 + 1.0f;
                        float _exp_27 = expf(gate_y_100 * -1.0f);
                        float denominator_y_104 = _exp_27 + 1.0f;
                        float hidden_x_105 = gate_x_99 / denominator_x_103 * up_x_101;
                        float hidden_y_106 = gate_y_100 / denominator_y_104 * up_y_102;
                        __nv_bfloat162 _bf16x2_77 = __float22bfloat162_rn(make_float2(hidden_x_105, hidden_y_106));
                        hidden_packed[13] = __as_u32(_bf16x2_77);
                        float2 _cvt_f32_28 = __bfloat1622float2(__as_bf16x2(gate_packed[14]));
                        float2 _cvt_f32_29 = __bfloat1622float2(__as_bf16x2(up_packed[14]));
                        float gate_x_107 = _cvt_f32_28.x;
                        float gate_y_108 = _cvt_f32_28.y;
                        float up_x_109 = _cvt_f32_29.x;
                        float up_y_110 = _cvt_f32_29.y;
                        float _exp_28 = expf(gate_x_107 * -1.0f);
                        float denominator_x_111 = _exp_28 + 1.0f;
                        float _exp_29 = expf(gate_y_108 * -1.0f);
                        float denominator_y_112 = _exp_29 + 1.0f;
                        float hidden_x_113 = gate_x_107 / denominator_x_111 * up_x_109;
                        float hidden_y_114 = gate_y_108 / denominator_y_112 * up_y_110;
                        __nv_bfloat162 _bf16x2_78 = __float22bfloat162_rn(make_float2(hidden_x_113, hidden_y_114));
                        hidden_packed[14] = __as_u32(_bf16x2_78);
                        float2 _cvt_f32_30 = __bfloat1622float2(__as_bf16x2(gate_packed[15]));
                        float2 _cvt_f32_31 = __bfloat1622float2(__as_bf16x2(up_packed[15]));
                        float gate_x_115 = _cvt_f32_30.x;
                        float gate_y_116 = _cvt_f32_30.y;
                        float up_x_117 = _cvt_f32_31.x;
                        float up_y_118 = _cvt_f32_31.y;
                        float _exp_30 = expf(gate_x_115 * -1.0f);
                        float denominator_x_119 = _exp_30 + 1.0f;
                        float _exp_31 = expf(gate_y_116 * -1.0f);
                        float denominator_y_120 = _exp_31 + 1.0f;
                        float hidden_x_121 = gate_x_115 / denominator_x_119 * up_x_117;
                        float hidden_y_122 = gate_y_116 / denominator_y_120 * up_y_118;
                        __nv_bfloat162 _bf16x2_79 = __float22bfloat162_rn(make_float2(hidden_x_121, hidden_y_122));
                        hidden_packed[15] = __as_u32(_bf16x2_79);
                        float2 _cvt_f32_32 = __bfloat1622float2(__as_bf16x2(gate_packed[16]));
                        float2 _cvt_f32_33 = __bfloat1622float2(__as_bf16x2(up_packed[16]));
                        float gate_x_123 = _cvt_f32_32.x;
                        float gate_y_124 = _cvt_f32_32.y;
                        float up_x_125 = _cvt_f32_33.x;
                        float up_y_126 = _cvt_f32_33.y;
                        float _exp_32 = expf(gate_x_123 * -1.0f);
                        float denominator_x_127 = _exp_32 + 1.0f;
                        float _exp_33 = expf(gate_y_124 * -1.0f);
                        float denominator_y_128 = _exp_33 + 1.0f;
                        float hidden_x_129 = gate_x_123 / denominator_x_127 * up_x_125;
                        float hidden_y_130 = gate_y_124 / denominator_y_128 * up_y_126;
                        __nv_bfloat162 _bf16x2_80 = __float22bfloat162_rn(make_float2(hidden_x_129, hidden_y_130));
                        hidden_packed[16] = __as_u32(_bf16x2_80);
                        float2 _cvt_f32_34 = __bfloat1622float2(__as_bf16x2(gate_packed[17]));
                        float2 _cvt_f32_35 = __bfloat1622float2(__as_bf16x2(up_packed[17]));
                        float gate_x_131 = _cvt_f32_34.x;
                        float gate_y_132 = _cvt_f32_34.y;
                        float up_x_133 = _cvt_f32_35.x;
                        float up_y_134 = _cvt_f32_35.y;
                        float _exp_34 = expf(gate_x_131 * -1.0f);
                        float denominator_x_135 = _exp_34 + 1.0f;
                        float _exp_35 = expf(gate_y_132 * -1.0f);
                        float denominator_y_136 = _exp_35 + 1.0f;
                        float hidden_x_137 = gate_x_131 / denominator_x_135 * up_x_133;
                        float hidden_y_138 = gate_y_132 / denominator_y_136 * up_y_134;
                        __nv_bfloat162 _bf16x2_81 = __float22bfloat162_rn(make_float2(hidden_x_137, hidden_y_138));
                        hidden_packed[17] = __as_u32(_bf16x2_81);
                        float2 _cvt_f32_36 = __bfloat1622float2(__as_bf16x2(gate_packed[18]));
                        float2 _cvt_f32_37 = __bfloat1622float2(__as_bf16x2(up_packed[18]));
                        float gate_x_139 = _cvt_f32_36.x;
                        float gate_y_140 = _cvt_f32_36.y;
                        float up_x_141 = _cvt_f32_37.x;
                        float up_y_142 = _cvt_f32_37.y;
                        float _exp_36 = expf(gate_x_139 * -1.0f);
                        float denominator_x_143 = _exp_36 + 1.0f;
                        float _exp_37 = expf(gate_y_140 * -1.0f);
                        float denominator_y_144 = _exp_37 + 1.0f;
                        float hidden_x_145 = gate_x_139 / denominator_x_143 * up_x_141;
                        float hidden_y_146 = gate_y_140 / denominator_y_144 * up_y_142;
                        __nv_bfloat162 _bf16x2_82 = __float22bfloat162_rn(make_float2(hidden_x_145, hidden_y_146));
                        hidden_packed[18] = __as_u32(_bf16x2_82);
                        float2 _cvt_f32_38 = __bfloat1622float2(__as_bf16x2(gate_packed[19]));
                        float2 _cvt_f32_39 = __bfloat1622float2(__as_bf16x2(up_packed[19]));
                        float gate_x_147 = _cvt_f32_38.x;
                        float gate_y_148 = _cvt_f32_38.y;
                        float up_x_149 = _cvt_f32_39.x;
                        float up_y_150 = _cvt_f32_39.y;
                        float _exp_38 = expf(gate_x_147 * -1.0f);
                        float denominator_x_151 = _exp_38 + 1.0f;
                        float _exp_39 = expf(gate_y_148 * -1.0f);
                        float denominator_y_152 = _exp_39 + 1.0f;
                        float hidden_x_153 = gate_x_147 / denominator_x_151 * up_x_149;
                        float hidden_y_154 = gate_y_148 / denominator_y_152 * up_y_150;
                        __nv_bfloat162 _bf16x2_83 = __float22bfloat162_rn(make_float2(hidden_x_153, hidden_y_154));
                        hidden_packed[19] = __as_u32(_bf16x2_83);
                        float2 _cvt_f32_40 = __bfloat1622float2(__as_bf16x2(gate_packed[20]));
                        float2 _cvt_f32_41 = __bfloat1622float2(__as_bf16x2(up_packed[20]));
                        float gate_x_155 = _cvt_f32_40.x;
                        float gate_y_156 = _cvt_f32_40.y;
                        float up_x_157 = _cvt_f32_41.x;
                        float up_y_158 = _cvt_f32_41.y;
                        float _exp_40 = expf(gate_x_155 * -1.0f);
                        float denominator_x_159 = _exp_40 + 1.0f;
                        float _exp_41 = expf(gate_y_156 * -1.0f);
                        float denominator_y_160 = _exp_41 + 1.0f;
                        float hidden_x_161 = gate_x_155 / denominator_x_159 * up_x_157;
                        float hidden_y_162 = gate_y_156 / denominator_y_160 * up_y_158;
                        __nv_bfloat162 _bf16x2_84 = __float22bfloat162_rn(make_float2(hidden_x_161, hidden_y_162));
                        hidden_packed[20] = __as_u32(_bf16x2_84);
                        float2 _cvt_f32_42 = __bfloat1622float2(__as_bf16x2(gate_packed[21]));
                        float2 _cvt_f32_43 = __bfloat1622float2(__as_bf16x2(up_packed[21]));
                        float gate_x_163 = _cvt_f32_42.x;
                        float gate_y_164 = _cvt_f32_42.y;
                        float up_x_165 = _cvt_f32_43.x;
                        float up_y_166 = _cvt_f32_43.y;
                        float _exp_42 = expf(gate_x_163 * -1.0f);
                        float denominator_x_167 = _exp_42 + 1.0f;
                        float _exp_43 = expf(gate_y_164 * -1.0f);
                        float denominator_y_168 = _exp_43 + 1.0f;
                        float hidden_x_169 = gate_x_163 / denominator_x_167 * up_x_165;
                        float hidden_y_170 = gate_y_164 / denominator_y_168 * up_y_166;
                        __nv_bfloat162 _bf16x2_85 = __float22bfloat162_rn(make_float2(hidden_x_169, hidden_y_170));
                        hidden_packed[21] = __as_u32(_bf16x2_85);
                        float2 _cvt_f32_44 = __bfloat1622float2(__as_bf16x2(gate_packed[22]));
                        float2 _cvt_f32_45 = __bfloat1622float2(__as_bf16x2(up_packed[22]));
                        float gate_x_171 = _cvt_f32_44.x;
                        float gate_y_172 = _cvt_f32_44.y;
                        float up_x_173 = _cvt_f32_45.x;
                        float up_y_174 = _cvt_f32_45.y;
                        float _exp_44 = expf(gate_x_171 * -1.0f);
                        float denominator_x_175 = _exp_44 + 1.0f;
                        float _exp_45 = expf(gate_y_172 * -1.0f);
                        float denominator_y_176 = _exp_45 + 1.0f;
                        float hidden_x_177 = gate_x_171 / denominator_x_175 * up_x_173;
                        float hidden_y_178 = gate_y_172 / denominator_y_176 * up_y_174;
                        __nv_bfloat162 _bf16x2_86 = __float22bfloat162_rn(make_float2(hidden_x_177, hidden_y_178));
                        hidden_packed[22] = __as_u32(_bf16x2_86);
                        float2 _cvt_f32_46 = __bfloat1622float2(__as_bf16x2(gate_packed[23]));
                        float2 _cvt_f32_47 = __bfloat1622float2(__as_bf16x2(up_packed[23]));
                        float gate_x_179 = _cvt_f32_46.x;
                        float gate_y_180 = _cvt_f32_46.y;
                        float up_x_181 = _cvt_f32_47.x;
                        float up_y_182 = _cvt_f32_47.y;
                        float _exp_46 = expf(gate_x_179 * -1.0f);
                        float denominator_x_183 = _exp_46 + 1.0f;
                        float _exp_47 = expf(gate_y_180 * -1.0f);
                        float denominator_y_184 = _exp_47 + 1.0f;
                        float hidden_x_185 = gate_x_179 / denominator_x_183 * up_x_181;
                        float hidden_y_186 = gate_y_180 / denominator_y_184 * up_y_182;
                        __nv_bfloat162 _bf16x2_87 = __float22bfloat162_rn(make_float2(hidden_x_185, hidden_y_186));
                        hidden_packed[23] = __as_u32(_bf16x2_87);
                        float2 _cvt_f32_48 = __bfloat1622float2(__as_bf16x2(gate_packed[24]));
                        float2 _cvt_f32_49 = __bfloat1622float2(__as_bf16x2(up_packed[24]));
                        float gate_x_187 = _cvt_f32_48.x;
                        float gate_y_188 = _cvt_f32_48.y;
                        float up_x_189 = _cvt_f32_49.x;
                        float up_y_190 = _cvt_f32_49.y;
                        float _exp_48 = expf(gate_x_187 * -1.0f);
                        float denominator_x_191 = _exp_48 + 1.0f;
                        float _exp_49 = expf(gate_y_188 * -1.0f);
                        float denominator_y_192 = _exp_49 + 1.0f;
                        float hidden_x_193 = gate_x_187 / denominator_x_191 * up_x_189;
                        float hidden_y_194 = gate_y_188 / denominator_y_192 * up_y_190;
                        __nv_bfloat162 _bf16x2_88 = __float22bfloat162_rn(make_float2(hidden_x_193, hidden_y_194));
                        hidden_packed[24] = __as_u32(_bf16x2_88);
                        float2 _cvt_f32_50 = __bfloat1622float2(__as_bf16x2(gate_packed[25]));
                        float2 _cvt_f32_51 = __bfloat1622float2(__as_bf16x2(up_packed[25]));
                        float gate_x_195 = _cvt_f32_50.x;
                        float gate_y_196 = _cvt_f32_50.y;
                        float up_x_197 = _cvt_f32_51.x;
                        float up_y_198 = _cvt_f32_51.y;
                        float _exp_50 = expf(gate_x_195 * -1.0f);
                        float denominator_x_199 = _exp_50 + 1.0f;
                        float _exp_51 = expf(gate_y_196 * -1.0f);
                        float denominator_y_200 = _exp_51 + 1.0f;
                        float hidden_x_201 = gate_x_195 / denominator_x_199 * up_x_197;
                        float hidden_y_202 = gate_y_196 / denominator_y_200 * up_y_198;
                        __nv_bfloat162 _bf16x2_89 = __float22bfloat162_rn(make_float2(hidden_x_201, hidden_y_202));
                        hidden_packed[25] = __as_u32(_bf16x2_89);
                        float2 _cvt_f32_52 = __bfloat1622float2(__as_bf16x2(gate_packed[26]));
                        float2 _cvt_f32_53 = __bfloat1622float2(__as_bf16x2(up_packed[26]));
                        float gate_x_203 = _cvt_f32_52.x;
                        float gate_y_204 = _cvt_f32_52.y;
                        float up_x_205 = _cvt_f32_53.x;
                        float up_y_206 = _cvt_f32_53.y;
                        float _exp_52 = expf(gate_x_203 * -1.0f);
                        float denominator_x_207 = _exp_52 + 1.0f;
                        float _exp_53 = expf(gate_y_204 * -1.0f);
                        float denominator_y_208 = _exp_53 + 1.0f;
                        float hidden_x_209 = gate_x_203 / denominator_x_207 * up_x_205;
                        float hidden_y_210 = gate_y_204 / denominator_y_208 * up_y_206;
                        __nv_bfloat162 _bf16x2_90 = __float22bfloat162_rn(make_float2(hidden_x_209, hidden_y_210));
                        hidden_packed[26] = __as_u32(_bf16x2_90);
                        float2 _cvt_f32_54 = __bfloat1622float2(__as_bf16x2(gate_packed[27]));
                        float2 _cvt_f32_55 = __bfloat1622float2(__as_bf16x2(up_packed[27]));
                        float gate_x_211 = _cvt_f32_54.x;
                        float gate_y_212 = _cvt_f32_54.y;
                        float up_x_213 = _cvt_f32_55.x;
                        float up_y_214 = _cvt_f32_55.y;
                        float _exp_54 = expf(gate_x_211 * -1.0f);
                        float denominator_x_215 = _exp_54 + 1.0f;
                        float _exp_55 = expf(gate_y_212 * -1.0f);
                        float denominator_y_216 = _exp_55 + 1.0f;
                        float hidden_x_217 = gate_x_211 / denominator_x_215 * up_x_213;
                        float hidden_y_218 = gate_y_212 / denominator_y_216 * up_y_214;
                        __nv_bfloat162 _bf16x2_91 = __float22bfloat162_rn(make_float2(hidden_x_217, hidden_y_218));
                        hidden_packed[27] = __as_u32(_bf16x2_91);
                        float2 _cvt_f32_56 = __bfloat1622float2(__as_bf16x2(gate_packed[28]));
                        float2 _cvt_f32_57 = __bfloat1622float2(__as_bf16x2(up_packed[28]));
                        float gate_x_219 = _cvt_f32_56.x;
                        float gate_y_220 = _cvt_f32_56.y;
                        float up_x_221 = _cvt_f32_57.x;
                        float up_y_222 = _cvt_f32_57.y;
                        float _exp_56 = expf(gate_x_219 * -1.0f);
                        float denominator_x_223 = _exp_56 + 1.0f;
                        float _exp_57 = expf(gate_y_220 * -1.0f);
                        float denominator_y_224 = _exp_57 + 1.0f;
                        float hidden_x_225 = gate_x_219 / denominator_x_223 * up_x_221;
                        float hidden_y_226 = gate_y_220 / denominator_y_224 * up_y_222;
                        __nv_bfloat162 _bf16x2_92 = __float22bfloat162_rn(make_float2(hidden_x_225, hidden_y_226));
                        hidden_packed[28] = __as_u32(_bf16x2_92);
                        float2 _cvt_f32_58 = __bfloat1622float2(__as_bf16x2(gate_packed[29]));
                        float2 _cvt_f32_59 = __bfloat1622float2(__as_bf16x2(up_packed[29]));
                        float gate_x_227 = _cvt_f32_58.x;
                        float gate_y_228 = _cvt_f32_58.y;
                        float up_x_229 = _cvt_f32_59.x;
                        float up_y_230 = _cvt_f32_59.y;
                        float _exp_58 = expf(gate_x_227 * -1.0f);
                        float denominator_x_231 = _exp_58 + 1.0f;
                        float _exp_59 = expf(gate_y_228 * -1.0f);
                        float denominator_y_232 = _exp_59 + 1.0f;
                        float hidden_x_233 = gate_x_227 / denominator_x_231 * up_x_229;
                        float hidden_y_234 = gate_y_228 / denominator_y_232 * up_y_230;
                        __nv_bfloat162 _bf16x2_93 = __float22bfloat162_rn(make_float2(hidden_x_233, hidden_y_234));
                        hidden_packed[29] = __as_u32(_bf16x2_93);
                        float2 _cvt_f32_60 = __bfloat1622float2(__as_bf16x2(gate_packed[30]));
                        float2 _cvt_f32_61 = __bfloat1622float2(__as_bf16x2(up_packed[30]));
                        float gate_x_235 = _cvt_f32_60.x;
                        float gate_y_236 = _cvt_f32_60.y;
                        float up_x_237 = _cvt_f32_61.x;
                        float up_y_238 = _cvt_f32_61.y;
                        float _exp_60 = expf(gate_x_235 * -1.0f);
                        float denominator_x_239 = _exp_60 + 1.0f;
                        float _exp_61 = expf(gate_y_236 * -1.0f);
                        float denominator_y_240 = _exp_61 + 1.0f;
                        float hidden_x_241 = gate_x_235 / denominator_x_239 * up_x_237;
                        float hidden_y_242 = gate_y_236 / denominator_y_240 * up_y_238;
                        __nv_bfloat162 _bf16x2_94 = __float22bfloat162_rn(make_float2(hidden_x_241, hidden_y_242));
                        hidden_packed[30] = __as_u32(_bf16x2_94);
                        float2 _cvt_f32_62 = __bfloat1622float2(__as_bf16x2(gate_packed[31]));
                        float2 _cvt_f32_63 = __bfloat1622float2(__as_bf16x2(up_packed[31]));
                        float gate_x_243 = _cvt_f32_62.x;
                        float gate_y_244 = _cvt_f32_62.y;
                        float up_x_245 = _cvt_f32_63.x;
                        float up_y_246 = _cvt_f32_63.y;
                        float _exp_62 = expf(gate_x_243 * -1.0f);
                        float denominator_x_247 = _exp_62 + 1.0f;
                        float _exp_63 = expf(gate_y_244 * -1.0f);
                        float denominator_y_248 = _exp_63 + 1.0f;
                        float hidden_x_249 = gate_x_243 / denominator_x_247 * up_x_245;
                        float hidden_y_250 = gate_y_244 / denominator_y_248 * up_y_246;
                        __nv_bfloat162 _bf16x2_95 = __float22bfloat162_rn(make_float2(hidden_x_249, hidden_y_250));
                        hidden_packed[31] = __as_u32(_bf16x2_95);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_251 = tid / 32;
                        int lane_1 = tid % 32;
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            #pragma unroll
                            for (int col_tile = 0; col_tile < 2; col_tile++) {
                                int row_4 = warp_251 * 32 + half * 16 + lane_1 % 16;
                                int col_1 = col_tile * 16 + lane_1 / 16 * 8;
                                unsigned int address_3 = d_smem_addr + (unsigned int)((row_4 * 32 + col_1) * 2);
                                address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                int offset = half * 8 + col_tile * 4;
                                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(address_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset + 3]))
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
                            for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                int row_5 = warp_252 * 32 + half_1 * 16 + lane_253 % 16;
                                int col_2 = col_tile_1 * 16 + lane_253 / 16 * 8;
                                unsigned int address_3_1 = d_smem_addr + 8192 + (unsigned int)((row_5 * 32 + col_2) * 2);
                                address_3_1 = address_3_1 ^ (address_3_1 & 511) >> 7 << 4;
                                int offset_1 = half_1 * 8 + col_tile_1 * 4;
                                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(address_3_1);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_1 + 3]))
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
                            for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                int row_6 = warp_254 * 32 + half_2 * 16 + lane_255 % 16;
                                int col_3 = col_tile_2 * 16 + lane_255 / 16 * 8;
                                unsigned int address_3_2 = d_smem_addr + 16384 + (unsigned int)((row_6 * 32 + col_3) * 2);
                                address_3_2 = address_3_2 ^ (address_3_2 & 511) >> 7 << 4;
                                int offset_2 = half_2 * 8 + col_tile_2 * 4;
                                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(address_3_2);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_2 + 3]))
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
                            for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                int row_7 = warp_256 * 32 + half_3 * 16 + lane_257 % 16;
                                int col_4 = col_tile_3 * 16 + lane_257 / 16 * 8;
                                unsigned int address_3_3 = d_smem_addr + (unsigned int)((row_7 * 32 + col_4) * 2);
                                address_3_3 = address_3_3 ^ (address_3_3 & 511) >> 7 << 4;
                                int offset_3 = 16 + half_3 * 8 + col_tile_3 * 4;
                                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(address_3_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed[offset_3 + 3]))
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
                            for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                int row_8 = warp_258 * 32 + half_4 * 16 + lane_259 % 16;
                                int col_5 = col_tile_4 * 16 + lane_259 / 16 * 8;
                                unsigned int address_3_4 = d_smem_addr + 8192 + (unsigned int)((row_8 * 32 + col_5) * 2);
                                address_3_4 = address_3_4 ^ (address_3_4 & 511) >> 7 << 4;
                                int offset_4 = 16 + half_4 * 8 + col_tile_4 * 4;
                                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(address_3_4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed[offset_4 + 3]))
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
                            for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                int row_9 = warp_260 * 32 + half_5 * 16 + lane_261 % 16;
                                int col_6 = col_tile_5 * 16 + lane_261 / 16 * 8;
                                unsigned int address_3_5 = d_smem_addr + 16384 + (unsigned int)((row_9 * 32 + col_6) * 2);
                                address_3_5 = address_3_5 ^ (address_3_5 & 511) >> 7 << 4;
                                int offset_5 = 16 + half_5 * 8 + col_tile_5 * 4;
                                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(address_3_5);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_5 + 3]))
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
                        __nv_bfloat162 _bf16x2_96 = __float22bfloat162_rn(make_float2(_tmem_load_8[0], _tmem_load_8[1]));
                        gate_packed_262[0] = __as_u32(_bf16x2_96);
                        __nv_bfloat162 _bf16x2_97 = __float22bfloat162_rn(make_float2(_tmem_load_9[0], _tmem_load_9[1]));
                        up_packed_263[0] = __as_u32(_bf16x2_97);
                        __nv_bfloat162 _bf16x2_98 = __float22bfloat162_rn(make_float2(_tmem_load_8[2], _tmem_load_8[3]));
                        gate_packed_262[1] = __as_u32(_bf16x2_98);
                        __nv_bfloat162 _bf16x2_99 = __float22bfloat162_rn(make_float2(_tmem_load_9[2], _tmem_load_9[3]));
                        up_packed_263[1] = __as_u32(_bf16x2_99);
                        __nv_bfloat162 _bf16x2_100 = __float22bfloat162_rn(make_float2(_tmem_load_8[4], _tmem_load_8[5]));
                        gate_packed_262[2] = __as_u32(_bf16x2_100);
                        __nv_bfloat162 _bf16x2_101 = __float22bfloat162_rn(make_float2(_tmem_load_9[4], _tmem_load_9[5]));
                        up_packed_263[2] = __as_u32(_bf16x2_101);
                        __nv_bfloat162 _bf16x2_102 = __float22bfloat162_rn(make_float2(_tmem_load_8[6], _tmem_load_8[7]));
                        gate_packed_262[3] = __as_u32(_bf16x2_102);
                        __nv_bfloat162 _bf16x2_103 = __float22bfloat162_rn(make_float2(_tmem_load_9[6], _tmem_load_9[7]));
                        up_packed_263[3] = __as_u32(_bf16x2_103);
                        __nv_bfloat162 _bf16x2_104 = __float22bfloat162_rn(make_float2(_tmem_load_8[8], _tmem_load_8[9]));
                        gate_packed_262[4] = __as_u32(_bf16x2_104);
                        __nv_bfloat162 _bf16x2_105 = __float22bfloat162_rn(make_float2(_tmem_load_9[8], _tmem_load_9[9]));
                        up_packed_263[4] = __as_u32(_bf16x2_105);
                        __nv_bfloat162 _bf16x2_106 = __float22bfloat162_rn(make_float2(_tmem_load_8[10], _tmem_load_8[11]));
                        gate_packed_262[5] = __as_u32(_bf16x2_106);
                        __nv_bfloat162 _bf16x2_107 = __float22bfloat162_rn(make_float2(_tmem_load_9[10], _tmem_load_9[11]));
                        up_packed_263[5] = __as_u32(_bf16x2_107);
                        __nv_bfloat162 _bf16x2_108 = __float22bfloat162_rn(make_float2(_tmem_load_8[12], _tmem_load_8[13]));
                        gate_packed_262[6] = __as_u32(_bf16x2_108);
                        __nv_bfloat162 _bf16x2_109 = __float22bfloat162_rn(make_float2(_tmem_load_9[12], _tmem_load_9[13]));
                        up_packed_263[6] = __as_u32(_bf16x2_109);
                        __nv_bfloat162 _bf16x2_110 = __float22bfloat162_rn(make_float2(_tmem_load_8[14], _tmem_load_8[15]));
                        gate_packed_262[7] = __as_u32(_bf16x2_110);
                        __nv_bfloat162 _bf16x2_111 = __float22bfloat162_rn(make_float2(_tmem_load_9[14], _tmem_load_9[15]));
                        up_packed_263[7] = __as_u32(_bf16x2_111);
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
                        __nv_bfloat162 _bf16x2_112 = __float22bfloat162_rn(make_float2(_tmem_load_10[0], _tmem_load_10[1]));
                        gate_packed_262[8] = __as_u32(_bf16x2_112);
                        __nv_bfloat162 _bf16x2_113 = __float22bfloat162_rn(make_float2(_tmem_load_11[0], _tmem_load_11[1]));
                        up_packed_263[8] = __as_u32(_bf16x2_113);
                        __nv_bfloat162 _bf16x2_114 = __float22bfloat162_rn(make_float2(_tmem_load_10[2], _tmem_load_10[3]));
                        gate_packed_262[9] = __as_u32(_bf16x2_114);
                        __nv_bfloat162 _bf16x2_115 = __float22bfloat162_rn(make_float2(_tmem_load_11[2], _tmem_load_11[3]));
                        up_packed_263[9] = __as_u32(_bf16x2_115);
                        __nv_bfloat162 _bf16x2_116 = __float22bfloat162_rn(make_float2(_tmem_load_10[4], _tmem_load_10[5]));
                        gate_packed_262[10] = __as_u32(_bf16x2_116);
                        __nv_bfloat162 _bf16x2_117 = __float22bfloat162_rn(make_float2(_tmem_load_11[4], _tmem_load_11[5]));
                        up_packed_263[10] = __as_u32(_bf16x2_117);
                        __nv_bfloat162 _bf16x2_118 = __float22bfloat162_rn(make_float2(_tmem_load_10[6], _tmem_load_10[7]));
                        gate_packed_262[11] = __as_u32(_bf16x2_118);
                        __nv_bfloat162 _bf16x2_119 = __float22bfloat162_rn(make_float2(_tmem_load_11[6], _tmem_load_11[7]));
                        up_packed_263[11] = __as_u32(_bf16x2_119);
                        __nv_bfloat162 _bf16x2_120 = __float22bfloat162_rn(make_float2(_tmem_load_10[8], _tmem_load_10[9]));
                        gate_packed_262[12] = __as_u32(_bf16x2_120);
                        __nv_bfloat162 _bf16x2_121 = __float22bfloat162_rn(make_float2(_tmem_load_11[8], _tmem_load_11[9]));
                        up_packed_263[12] = __as_u32(_bf16x2_121);
                        __nv_bfloat162 _bf16x2_122 = __float22bfloat162_rn(make_float2(_tmem_load_10[10], _tmem_load_10[11]));
                        gate_packed_262[13] = __as_u32(_bf16x2_122);
                        __nv_bfloat162 _bf16x2_123 = __float22bfloat162_rn(make_float2(_tmem_load_11[10], _tmem_load_11[11]));
                        up_packed_263[13] = __as_u32(_bf16x2_123);
                        __nv_bfloat162 _bf16x2_124 = __float22bfloat162_rn(make_float2(_tmem_load_10[12], _tmem_load_10[13]));
                        gate_packed_262[14] = __as_u32(_bf16x2_124);
                        __nv_bfloat162 _bf16x2_125 = __float22bfloat162_rn(make_float2(_tmem_load_11[12], _tmem_load_11[13]));
                        up_packed_263[14] = __as_u32(_bf16x2_125);
                        __nv_bfloat162 _bf16x2_126 = __float22bfloat162_rn(make_float2(_tmem_load_10[14], _tmem_load_10[15]));
                        gate_packed_262[15] = __as_u32(_bf16x2_126);
                        __nv_bfloat162 _bf16x2_127 = __float22bfloat162_rn(make_float2(_tmem_load_11[14], _tmem_load_11[15]));
                        up_packed_263[15] = __as_u32(_bf16x2_127);
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
                        __nv_bfloat162 _bf16x2_128 = __float22bfloat162_rn(make_float2(_tmem_load_12[0], _tmem_load_12[1]));
                        gate_packed_262[16] = __as_u32(_bf16x2_128);
                        __nv_bfloat162 _bf16x2_129 = __float22bfloat162_rn(make_float2(_tmem_load_13[0], _tmem_load_13[1]));
                        up_packed_263[16] = __as_u32(_bf16x2_129);
                        __nv_bfloat162 _bf16x2_130 = __float22bfloat162_rn(make_float2(_tmem_load_12[2], _tmem_load_12[3]));
                        gate_packed_262[17] = __as_u32(_bf16x2_130);
                        __nv_bfloat162 _bf16x2_131 = __float22bfloat162_rn(make_float2(_tmem_load_13[2], _tmem_load_13[3]));
                        up_packed_263[17] = __as_u32(_bf16x2_131);
                        __nv_bfloat162 _bf16x2_132 = __float22bfloat162_rn(make_float2(_tmem_load_12[4], _tmem_load_12[5]));
                        gate_packed_262[18] = __as_u32(_bf16x2_132);
                        __nv_bfloat162 _bf16x2_133 = __float22bfloat162_rn(make_float2(_tmem_load_13[4], _tmem_load_13[5]));
                        up_packed_263[18] = __as_u32(_bf16x2_133);
                        __nv_bfloat162 _bf16x2_134 = __float22bfloat162_rn(make_float2(_tmem_load_12[6], _tmem_load_12[7]));
                        gate_packed_262[19] = __as_u32(_bf16x2_134);
                        __nv_bfloat162 _bf16x2_135 = __float22bfloat162_rn(make_float2(_tmem_load_13[6], _tmem_load_13[7]));
                        up_packed_263[19] = __as_u32(_bf16x2_135);
                        __nv_bfloat162 _bf16x2_136 = __float22bfloat162_rn(make_float2(_tmem_load_12[8], _tmem_load_12[9]));
                        gate_packed_262[20] = __as_u32(_bf16x2_136);
                        __nv_bfloat162 _bf16x2_137 = __float22bfloat162_rn(make_float2(_tmem_load_13[8], _tmem_load_13[9]));
                        up_packed_263[20] = __as_u32(_bf16x2_137);
                        __nv_bfloat162 _bf16x2_138 = __float22bfloat162_rn(make_float2(_tmem_load_12[10], _tmem_load_12[11]));
                        gate_packed_262[21] = __as_u32(_bf16x2_138);
                        __nv_bfloat162 _bf16x2_139 = __float22bfloat162_rn(make_float2(_tmem_load_13[10], _tmem_load_13[11]));
                        up_packed_263[21] = __as_u32(_bf16x2_139);
                        __nv_bfloat162 _bf16x2_140 = __float22bfloat162_rn(make_float2(_tmem_load_12[12], _tmem_load_12[13]));
                        gate_packed_262[22] = __as_u32(_bf16x2_140);
                        __nv_bfloat162 _bf16x2_141 = __float22bfloat162_rn(make_float2(_tmem_load_13[12], _tmem_load_13[13]));
                        up_packed_263[22] = __as_u32(_bf16x2_141);
                        __nv_bfloat162 _bf16x2_142 = __float22bfloat162_rn(make_float2(_tmem_load_12[14], _tmem_load_12[15]));
                        gate_packed_262[23] = __as_u32(_bf16x2_142);
                        __nv_bfloat162 _bf16x2_143 = __float22bfloat162_rn(make_float2(_tmem_load_13[14], _tmem_load_13[15]));
                        up_packed_263[23] = __as_u32(_bf16x2_143);
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
                        __nv_bfloat162 _bf16x2_144 = __float22bfloat162_rn(make_float2(_tmem_load_14[0], _tmem_load_14[1]));
                        gate_packed_262[24] = __as_u32(_bf16x2_144);
                        __nv_bfloat162 _bf16x2_145 = __float22bfloat162_rn(make_float2(_tmem_load_15[0], _tmem_load_15[1]));
                        up_packed_263[24] = __as_u32(_bf16x2_145);
                        __nv_bfloat162 _bf16x2_146 = __float22bfloat162_rn(make_float2(_tmem_load_14[2], _tmem_load_14[3]));
                        gate_packed_262[25] = __as_u32(_bf16x2_146);
                        __nv_bfloat162 _bf16x2_147 = __float22bfloat162_rn(make_float2(_tmem_load_15[2], _tmem_load_15[3]));
                        up_packed_263[25] = __as_u32(_bf16x2_147);
                        __nv_bfloat162 _bf16x2_148 = __float22bfloat162_rn(make_float2(_tmem_load_14[4], _tmem_load_14[5]));
                        gate_packed_262[26] = __as_u32(_bf16x2_148);
                        __nv_bfloat162 _bf16x2_149 = __float22bfloat162_rn(make_float2(_tmem_load_15[4], _tmem_load_15[5]));
                        up_packed_263[26] = __as_u32(_bf16x2_149);
                        __nv_bfloat162 _bf16x2_150 = __float22bfloat162_rn(make_float2(_tmem_load_14[6], _tmem_load_14[7]));
                        gate_packed_262[27] = __as_u32(_bf16x2_150);
                        __nv_bfloat162 _bf16x2_151 = __float22bfloat162_rn(make_float2(_tmem_load_15[6], _tmem_load_15[7]));
                        up_packed_263[27] = __as_u32(_bf16x2_151);
                        __nv_bfloat162 _bf16x2_152 = __float22bfloat162_rn(make_float2(_tmem_load_14[8], _tmem_load_14[9]));
                        gate_packed_262[28] = __as_u32(_bf16x2_152);
                        __nv_bfloat162 _bf16x2_153 = __float22bfloat162_rn(make_float2(_tmem_load_15[8], _tmem_load_15[9]));
                        up_packed_263[28] = __as_u32(_bf16x2_153);
                        __nv_bfloat162 _bf16x2_154 = __float22bfloat162_rn(make_float2(_tmem_load_14[10], _tmem_load_14[11]));
                        gate_packed_262[29] = __as_u32(_bf16x2_154);
                        __nv_bfloat162 _bf16x2_155 = __float22bfloat162_rn(make_float2(_tmem_load_15[10], _tmem_load_15[11]));
                        up_packed_263[29] = __as_u32(_bf16x2_155);
                        __nv_bfloat162 _bf16x2_156 = __float22bfloat162_rn(make_float2(_tmem_load_14[12], _tmem_load_14[13]));
                        gate_packed_262[30] = __as_u32(_bf16x2_156);
                        __nv_bfloat162 _bf16x2_157 = __float22bfloat162_rn(make_float2(_tmem_load_15[12], _tmem_load_15[13]));
                        up_packed_263[30] = __as_u32(_bf16x2_157);
                        __nv_bfloat162 _bf16x2_158 = __float22bfloat162_rn(make_float2(_tmem_load_14[14], _tmem_load_14[15]));
                        gate_packed_262[31] = __as_u32(_bf16x2_158);
                        __nv_bfloat162 _bf16x2_159 = __float22bfloat162_rn(make_float2(_tmem_load_15[14], _tmem_load_15[15]));
                        up_packed_263[31] = __as_u32(_bf16x2_159);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_64 = __bfloat1622float2(__as_bf16x2(gate_packed_262[0]));
                        float2 _cvt_f32_65 = __bfloat1622float2(__as_bf16x2(up_packed_263[0]));
                        float gate_x_269 = _cvt_f32_64.x;
                        float gate_y_270 = _cvt_f32_64.y;
                        float up_x_271 = _cvt_f32_65.x;
                        float up_y_272 = _cvt_f32_65.y;
                        float _exp_64 = expf(gate_x_269 * -1.0f);
                        float denominator_x_273 = _exp_64 + 1.0f;
                        float _exp_65 = expf(gate_y_270 * -1.0f);
                        float denominator_y_274 = _exp_65 + 1.0f;
                        float hidden_x_275 = gate_x_269 / denominator_x_273 * up_x_271;
                        float hidden_y_276 = gate_y_270 / denominator_y_274 * up_y_272;
                        __nv_bfloat162 _bf16x2_160 = __float22bfloat162_rn(make_float2(hidden_x_275, hidden_y_276));
                        hidden_packed_264[0] = __as_u32(_bf16x2_160);
                        float2 _cvt_f32_66 = __bfloat1622float2(__as_bf16x2(gate_packed_262[1]));
                        float2 _cvt_f32_67 = __bfloat1622float2(__as_bf16x2(up_packed_263[1]));
                        float gate_x_277 = _cvt_f32_66.x;
                        float gate_y_278 = _cvt_f32_66.y;
                        float up_x_279 = _cvt_f32_67.x;
                        float up_y_280 = _cvt_f32_67.y;
                        float _exp_66 = expf(gate_x_277 * -1.0f);
                        float denominator_x_281 = _exp_66 + 1.0f;
                        float _exp_67 = expf(gate_y_278 * -1.0f);
                        float denominator_y_282 = _exp_67 + 1.0f;
                        float hidden_x_283 = gate_x_277 / denominator_x_281 * up_x_279;
                        float hidden_y_284 = gate_y_278 / denominator_y_282 * up_y_280;
                        __nv_bfloat162 _bf16x2_161 = __float22bfloat162_rn(make_float2(hidden_x_283, hidden_y_284));
                        hidden_packed_264[1] = __as_u32(_bf16x2_161);
                        float2 _cvt_f32_68 = __bfloat1622float2(__as_bf16x2(gate_packed_262[2]));
                        float2 _cvt_f32_69 = __bfloat1622float2(__as_bf16x2(up_packed_263[2]));
                        float gate_x_285 = _cvt_f32_68.x;
                        float gate_y_286 = _cvt_f32_68.y;
                        float up_x_287 = _cvt_f32_69.x;
                        float up_y_288 = _cvt_f32_69.y;
                        float _exp_68 = expf(gate_x_285 * -1.0f);
                        float denominator_x_289 = _exp_68 + 1.0f;
                        float _exp_69 = expf(gate_y_286 * -1.0f);
                        float denominator_y_290 = _exp_69 + 1.0f;
                        float hidden_x_291 = gate_x_285 / denominator_x_289 * up_x_287;
                        float hidden_y_292 = gate_y_286 / denominator_y_290 * up_y_288;
                        __nv_bfloat162 _bf16x2_162 = __float22bfloat162_rn(make_float2(hidden_x_291, hidden_y_292));
                        hidden_packed_264[2] = __as_u32(_bf16x2_162);
                        float2 _cvt_f32_70 = __bfloat1622float2(__as_bf16x2(gate_packed_262[3]));
                        float2 _cvt_f32_71 = __bfloat1622float2(__as_bf16x2(up_packed_263[3]));
                        float gate_x_293 = _cvt_f32_70.x;
                        float gate_y_294 = _cvt_f32_70.y;
                        float up_x_295 = _cvt_f32_71.x;
                        float up_y_296 = _cvt_f32_71.y;
                        float _exp_70 = expf(gate_x_293 * -1.0f);
                        float denominator_x_297 = _exp_70 + 1.0f;
                        float _exp_71 = expf(gate_y_294 * -1.0f);
                        float denominator_y_298 = _exp_71 + 1.0f;
                        float hidden_x_299 = gate_x_293 / denominator_x_297 * up_x_295;
                        float hidden_y_300 = gate_y_294 / denominator_y_298 * up_y_296;
                        __nv_bfloat162 _bf16x2_163 = __float22bfloat162_rn(make_float2(hidden_x_299, hidden_y_300));
                        hidden_packed_264[3] = __as_u32(_bf16x2_163);
                        float2 _cvt_f32_72 = __bfloat1622float2(__as_bf16x2(gate_packed_262[4]));
                        float2 _cvt_f32_73 = __bfloat1622float2(__as_bf16x2(up_packed_263[4]));
                        float gate_x_301 = _cvt_f32_72.x;
                        float gate_y_302 = _cvt_f32_72.y;
                        float up_x_303 = _cvt_f32_73.x;
                        float up_y_304 = _cvt_f32_73.y;
                        float _exp_72 = expf(gate_x_301 * -1.0f);
                        float denominator_x_305 = _exp_72 + 1.0f;
                        float _exp_73 = expf(gate_y_302 * -1.0f);
                        float denominator_y_306 = _exp_73 + 1.0f;
                        float hidden_x_307 = gate_x_301 / denominator_x_305 * up_x_303;
                        float hidden_y_308 = gate_y_302 / denominator_y_306 * up_y_304;
                        __nv_bfloat162 _bf16x2_164 = __float22bfloat162_rn(make_float2(hidden_x_307, hidden_y_308));
                        hidden_packed_264[4] = __as_u32(_bf16x2_164);
                        float2 _cvt_f32_74 = __bfloat1622float2(__as_bf16x2(gate_packed_262[5]));
                        float2 _cvt_f32_75 = __bfloat1622float2(__as_bf16x2(up_packed_263[5]));
                        float gate_x_309 = _cvt_f32_74.x;
                        float gate_y_310 = _cvt_f32_74.y;
                        float up_x_311 = _cvt_f32_75.x;
                        float up_y_312 = _cvt_f32_75.y;
                        float _exp_74 = expf(gate_x_309 * -1.0f);
                        float denominator_x_313 = _exp_74 + 1.0f;
                        float _exp_75 = expf(gate_y_310 * -1.0f);
                        float denominator_y_314 = _exp_75 + 1.0f;
                        float hidden_x_315 = gate_x_309 / denominator_x_313 * up_x_311;
                        float hidden_y_316 = gate_y_310 / denominator_y_314 * up_y_312;
                        __nv_bfloat162 _bf16x2_165 = __float22bfloat162_rn(make_float2(hidden_x_315, hidden_y_316));
                        hidden_packed_264[5] = __as_u32(_bf16x2_165);
                        float2 _cvt_f32_76 = __bfloat1622float2(__as_bf16x2(gate_packed_262[6]));
                        float2 _cvt_f32_77 = __bfloat1622float2(__as_bf16x2(up_packed_263[6]));
                        float gate_x_317 = _cvt_f32_76.x;
                        float gate_y_318 = _cvt_f32_76.y;
                        float up_x_319 = _cvt_f32_77.x;
                        float up_y_320 = _cvt_f32_77.y;
                        float _exp_76 = expf(gate_x_317 * -1.0f);
                        float denominator_x_321 = _exp_76 + 1.0f;
                        float _exp_77 = expf(gate_y_318 * -1.0f);
                        float denominator_y_322 = _exp_77 + 1.0f;
                        float hidden_x_323 = gate_x_317 / denominator_x_321 * up_x_319;
                        float hidden_y_324 = gate_y_318 / denominator_y_322 * up_y_320;
                        __nv_bfloat162 _bf16x2_166 = __float22bfloat162_rn(make_float2(hidden_x_323, hidden_y_324));
                        hidden_packed_264[6] = __as_u32(_bf16x2_166);
                        float2 _cvt_f32_78 = __bfloat1622float2(__as_bf16x2(gate_packed_262[7]));
                        float2 _cvt_f32_79 = __bfloat1622float2(__as_bf16x2(up_packed_263[7]));
                        float gate_x_325 = _cvt_f32_78.x;
                        float gate_y_326 = _cvt_f32_78.y;
                        float up_x_327 = _cvt_f32_79.x;
                        float up_y_328 = _cvt_f32_79.y;
                        float _exp_78 = expf(gate_x_325 * -1.0f);
                        float denominator_x_329 = _exp_78 + 1.0f;
                        float _exp_79 = expf(gate_y_326 * -1.0f);
                        float denominator_y_330 = _exp_79 + 1.0f;
                        float hidden_x_331 = gate_x_325 / denominator_x_329 * up_x_327;
                        float hidden_y_332 = gate_y_326 / denominator_y_330 * up_y_328;
                        __nv_bfloat162 _bf16x2_167 = __float22bfloat162_rn(make_float2(hidden_x_331, hidden_y_332));
                        hidden_packed_264[7] = __as_u32(_bf16x2_167);
                        float2 _cvt_f32_80 = __bfloat1622float2(__as_bf16x2(gate_packed_262[8]));
                        float2 _cvt_f32_81 = __bfloat1622float2(__as_bf16x2(up_packed_263[8]));
                        float gate_x_333 = _cvt_f32_80.x;
                        float gate_y_334 = _cvt_f32_80.y;
                        float up_x_335 = _cvt_f32_81.x;
                        float up_y_336 = _cvt_f32_81.y;
                        float _exp_80 = expf(gate_x_333 * -1.0f);
                        float denominator_x_337 = _exp_80 + 1.0f;
                        float _exp_81 = expf(gate_y_334 * -1.0f);
                        float denominator_y_338 = _exp_81 + 1.0f;
                        float hidden_x_339 = gate_x_333 / denominator_x_337 * up_x_335;
                        float hidden_y_340 = gate_y_334 / denominator_y_338 * up_y_336;
                        __nv_bfloat162 _bf16x2_168 = __float22bfloat162_rn(make_float2(hidden_x_339, hidden_y_340));
                        hidden_packed_264[8] = __as_u32(_bf16x2_168);
                        float2 _cvt_f32_82 = __bfloat1622float2(__as_bf16x2(gate_packed_262[9]));
                        float2 _cvt_f32_83 = __bfloat1622float2(__as_bf16x2(up_packed_263[9]));
                        float gate_x_341 = _cvt_f32_82.x;
                        float gate_y_342 = _cvt_f32_82.y;
                        float up_x_343 = _cvt_f32_83.x;
                        float up_y_344 = _cvt_f32_83.y;
                        float _exp_82 = expf(gate_x_341 * -1.0f);
                        float denominator_x_345 = _exp_82 + 1.0f;
                        float _exp_83 = expf(gate_y_342 * -1.0f);
                        float denominator_y_346 = _exp_83 + 1.0f;
                        float hidden_x_347 = gate_x_341 / denominator_x_345 * up_x_343;
                        float hidden_y_348 = gate_y_342 / denominator_y_346 * up_y_344;
                        __nv_bfloat162 _bf16x2_169 = __float22bfloat162_rn(make_float2(hidden_x_347, hidden_y_348));
                        hidden_packed_264[9] = __as_u32(_bf16x2_169);
                        float2 _cvt_f32_84 = __bfloat1622float2(__as_bf16x2(gate_packed_262[10]));
                        float2 _cvt_f32_85 = __bfloat1622float2(__as_bf16x2(up_packed_263[10]));
                        float gate_x_349 = _cvt_f32_84.x;
                        float gate_y_350 = _cvt_f32_84.y;
                        float up_x_351 = _cvt_f32_85.x;
                        float up_y_352 = _cvt_f32_85.y;
                        float _exp_84 = expf(gate_x_349 * -1.0f);
                        float denominator_x_353 = _exp_84 + 1.0f;
                        float _exp_85 = expf(gate_y_350 * -1.0f);
                        float denominator_y_354 = _exp_85 + 1.0f;
                        float hidden_x_355 = gate_x_349 / denominator_x_353 * up_x_351;
                        float hidden_y_356 = gate_y_350 / denominator_y_354 * up_y_352;
                        __nv_bfloat162 _bf16x2_170 = __float22bfloat162_rn(make_float2(hidden_x_355, hidden_y_356));
                        hidden_packed_264[10] = __as_u32(_bf16x2_170);
                        float2 _cvt_f32_86 = __bfloat1622float2(__as_bf16x2(gate_packed_262[11]));
                        float2 _cvt_f32_87 = __bfloat1622float2(__as_bf16x2(up_packed_263[11]));
                        float gate_x_357 = _cvt_f32_86.x;
                        float gate_y_358 = _cvt_f32_86.y;
                        float up_x_359 = _cvt_f32_87.x;
                        float up_y_360 = _cvt_f32_87.y;
                        float _exp_86 = expf(gate_x_357 * -1.0f);
                        float denominator_x_361 = _exp_86 + 1.0f;
                        float _exp_87 = expf(gate_y_358 * -1.0f);
                        float denominator_y_362 = _exp_87 + 1.0f;
                        float hidden_x_363 = gate_x_357 / denominator_x_361 * up_x_359;
                        float hidden_y_364 = gate_y_358 / denominator_y_362 * up_y_360;
                        __nv_bfloat162 _bf16x2_171 = __float22bfloat162_rn(make_float2(hidden_x_363, hidden_y_364));
                        hidden_packed_264[11] = __as_u32(_bf16x2_171);
                        float2 _cvt_f32_88 = __bfloat1622float2(__as_bf16x2(gate_packed_262[12]));
                        float2 _cvt_f32_89 = __bfloat1622float2(__as_bf16x2(up_packed_263[12]));
                        float gate_x_365 = _cvt_f32_88.x;
                        float gate_y_366 = _cvt_f32_88.y;
                        float up_x_367 = _cvt_f32_89.x;
                        float up_y_368 = _cvt_f32_89.y;
                        float _exp_88 = expf(gate_x_365 * -1.0f);
                        float denominator_x_369 = _exp_88 + 1.0f;
                        float _exp_89 = expf(gate_y_366 * -1.0f);
                        float denominator_y_370 = _exp_89 + 1.0f;
                        float hidden_x_371 = gate_x_365 / denominator_x_369 * up_x_367;
                        float hidden_y_372 = gate_y_366 / denominator_y_370 * up_y_368;
                        __nv_bfloat162 _bf16x2_172 = __float22bfloat162_rn(make_float2(hidden_x_371, hidden_y_372));
                        hidden_packed_264[12] = __as_u32(_bf16x2_172);
                        float2 _cvt_f32_90 = __bfloat1622float2(__as_bf16x2(gate_packed_262[13]));
                        float2 _cvt_f32_91 = __bfloat1622float2(__as_bf16x2(up_packed_263[13]));
                        float gate_x_373 = _cvt_f32_90.x;
                        float gate_y_374 = _cvt_f32_90.y;
                        float up_x_375 = _cvt_f32_91.x;
                        float up_y_376 = _cvt_f32_91.y;
                        float _exp_90 = expf(gate_x_373 * -1.0f);
                        float denominator_x_377 = _exp_90 + 1.0f;
                        float _exp_91 = expf(gate_y_374 * -1.0f);
                        float denominator_y_378 = _exp_91 + 1.0f;
                        float hidden_x_379 = gate_x_373 / denominator_x_377 * up_x_375;
                        float hidden_y_380 = gate_y_374 / denominator_y_378 * up_y_376;
                        __nv_bfloat162 _bf16x2_173 = __float22bfloat162_rn(make_float2(hidden_x_379, hidden_y_380));
                        hidden_packed_264[13] = __as_u32(_bf16x2_173);
                        float2 _cvt_f32_92 = __bfloat1622float2(__as_bf16x2(gate_packed_262[14]));
                        float2 _cvt_f32_93 = __bfloat1622float2(__as_bf16x2(up_packed_263[14]));
                        float gate_x_381 = _cvt_f32_92.x;
                        float gate_y_382 = _cvt_f32_92.y;
                        float up_x_383 = _cvt_f32_93.x;
                        float up_y_384 = _cvt_f32_93.y;
                        float _exp_92 = expf(gate_x_381 * -1.0f);
                        float denominator_x_385 = _exp_92 + 1.0f;
                        float _exp_93 = expf(gate_y_382 * -1.0f);
                        float denominator_y_386 = _exp_93 + 1.0f;
                        float hidden_x_387 = gate_x_381 / denominator_x_385 * up_x_383;
                        float hidden_y_388 = gate_y_382 / denominator_y_386 * up_y_384;
                        __nv_bfloat162 _bf16x2_174 = __float22bfloat162_rn(make_float2(hidden_x_387, hidden_y_388));
                        hidden_packed_264[14] = __as_u32(_bf16x2_174);
                        float2 _cvt_f32_94 = __bfloat1622float2(__as_bf16x2(gate_packed_262[15]));
                        float2 _cvt_f32_95 = __bfloat1622float2(__as_bf16x2(up_packed_263[15]));
                        float gate_x_389 = _cvt_f32_94.x;
                        float gate_y_390 = _cvt_f32_94.y;
                        float up_x_391 = _cvt_f32_95.x;
                        float up_y_392 = _cvt_f32_95.y;
                        float _exp_94 = expf(gate_x_389 * -1.0f);
                        float denominator_x_393 = _exp_94 + 1.0f;
                        float _exp_95 = expf(gate_y_390 * -1.0f);
                        float denominator_y_394 = _exp_95 + 1.0f;
                        float hidden_x_395 = gate_x_389 / denominator_x_393 * up_x_391;
                        float hidden_y_396 = gate_y_390 / denominator_y_394 * up_y_392;
                        __nv_bfloat162 _bf16x2_175 = __float22bfloat162_rn(make_float2(hidden_x_395, hidden_y_396));
                        hidden_packed_264[15] = __as_u32(_bf16x2_175);
                        float2 _cvt_f32_96 = __bfloat1622float2(__as_bf16x2(gate_packed_262[16]));
                        float2 _cvt_f32_97 = __bfloat1622float2(__as_bf16x2(up_packed_263[16]));
                        float gate_x_397 = _cvt_f32_96.x;
                        float gate_y_398 = _cvt_f32_96.y;
                        float up_x_399 = _cvt_f32_97.x;
                        float up_y_400 = _cvt_f32_97.y;
                        float _exp_96 = expf(gate_x_397 * -1.0f);
                        float denominator_x_401 = _exp_96 + 1.0f;
                        float _exp_97 = expf(gate_y_398 * -1.0f);
                        float denominator_y_402 = _exp_97 + 1.0f;
                        float hidden_x_403 = gate_x_397 / denominator_x_401 * up_x_399;
                        float hidden_y_404 = gate_y_398 / denominator_y_402 * up_y_400;
                        __nv_bfloat162 _bf16x2_176 = __float22bfloat162_rn(make_float2(hidden_x_403, hidden_y_404));
                        hidden_packed_264[16] = __as_u32(_bf16x2_176);
                        float2 _cvt_f32_98 = __bfloat1622float2(__as_bf16x2(gate_packed_262[17]));
                        float2 _cvt_f32_99 = __bfloat1622float2(__as_bf16x2(up_packed_263[17]));
                        float gate_x_405 = _cvt_f32_98.x;
                        float gate_y_406 = _cvt_f32_98.y;
                        float up_x_407 = _cvt_f32_99.x;
                        float up_y_408 = _cvt_f32_99.y;
                        float _exp_98 = expf(gate_x_405 * -1.0f);
                        float denominator_x_409 = _exp_98 + 1.0f;
                        float _exp_99 = expf(gate_y_406 * -1.0f);
                        float denominator_y_410 = _exp_99 + 1.0f;
                        float hidden_x_411 = gate_x_405 / denominator_x_409 * up_x_407;
                        float hidden_y_412 = gate_y_406 / denominator_y_410 * up_y_408;
                        __nv_bfloat162 _bf16x2_177 = __float22bfloat162_rn(make_float2(hidden_x_411, hidden_y_412));
                        hidden_packed_264[17] = __as_u32(_bf16x2_177);
                        float2 _cvt_f32_100 = __bfloat1622float2(__as_bf16x2(gate_packed_262[18]));
                        float2 _cvt_f32_101 = __bfloat1622float2(__as_bf16x2(up_packed_263[18]));
                        float gate_x_413 = _cvt_f32_100.x;
                        float gate_y_414 = _cvt_f32_100.y;
                        float up_x_415 = _cvt_f32_101.x;
                        float up_y_416 = _cvt_f32_101.y;
                        float _exp_100 = expf(gate_x_413 * -1.0f);
                        float denominator_x_417 = _exp_100 + 1.0f;
                        float _exp_101 = expf(gate_y_414 * -1.0f);
                        float denominator_y_418 = _exp_101 + 1.0f;
                        float hidden_x_419 = gate_x_413 / denominator_x_417 * up_x_415;
                        float hidden_y_420 = gate_y_414 / denominator_y_418 * up_y_416;
                        __nv_bfloat162 _bf16x2_178 = __float22bfloat162_rn(make_float2(hidden_x_419, hidden_y_420));
                        hidden_packed_264[18] = __as_u32(_bf16x2_178);
                        float2 _cvt_f32_102 = __bfloat1622float2(__as_bf16x2(gate_packed_262[19]));
                        float2 _cvt_f32_103 = __bfloat1622float2(__as_bf16x2(up_packed_263[19]));
                        float gate_x_421 = _cvt_f32_102.x;
                        float gate_y_422 = _cvt_f32_102.y;
                        float up_x_423 = _cvt_f32_103.x;
                        float up_y_424 = _cvt_f32_103.y;
                        float _exp_102 = expf(gate_x_421 * -1.0f);
                        float denominator_x_425 = _exp_102 + 1.0f;
                        float _exp_103 = expf(gate_y_422 * -1.0f);
                        float denominator_y_426 = _exp_103 + 1.0f;
                        float hidden_x_427 = gate_x_421 / denominator_x_425 * up_x_423;
                        float hidden_y_428 = gate_y_422 / denominator_y_426 * up_y_424;
                        __nv_bfloat162 _bf16x2_179 = __float22bfloat162_rn(make_float2(hidden_x_427, hidden_y_428));
                        hidden_packed_264[19] = __as_u32(_bf16x2_179);
                        float2 _cvt_f32_104 = __bfloat1622float2(__as_bf16x2(gate_packed_262[20]));
                        float2 _cvt_f32_105 = __bfloat1622float2(__as_bf16x2(up_packed_263[20]));
                        float gate_x_429 = _cvt_f32_104.x;
                        float gate_y_430 = _cvt_f32_104.y;
                        float up_x_431 = _cvt_f32_105.x;
                        float up_y_432 = _cvt_f32_105.y;
                        float _exp_104 = expf(gate_x_429 * -1.0f);
                        float denominator_x_433 = _exp_104 + 1.0f;
                        float _exp_105 = expf(gate_y_430 * -1.0f);
                        float denominator_y_434 = _exp_105 + 1.0f;
                        float hidden_x_435 = gate_x_429 / denominator_x_433 * up_x_431;
                        float hidden_y_436 = gate_y_430 / denominator_y_434 * up_y_432;
                        __nv_bfloat162 _bf16x2_180 = __float22bfloat162_rn(make_float2(hidden_x_435, hidden_y_436));
                        hidden_packed_264[20] = __as_u32(_bf16x2_180);
                        float2 _cvt_f32_106 = __bfloat1622float2(__as_bf16x2(gate_packed_262[21]));
                        float2 _cvt_f32_107 = __bfloat1622float2(__as_bf16x2(up_packed_263[21]));
                        float gate_x_437 = _cvt_f32_106.x;
                        float gate_y_438 = _cvt_f32_106.y;
                        float up_x_439 = _cvt_f32_107.x;
                        float up_y_440 = _cvt_f32_107.y;
                        float _exp_106 = expf(gate_x_437 * -1.0f);
                        float denominator_x_441 = _exp_106 + 1.0f;
                        float _exp_107 = expf(gate_y_438 * -1.0f);
                        float denominator_y_442 = _exp_107 + 1.0f;
                        float hidden_x_443 = gate_x_437 / denominator_x_441 * up_x_439;
                        float hidden_y_444 = gate_y_438 / denominator_y_442 * up_y_440;
                        __nv_bfloat162 _bf16x2_181 = __float22bfloat162_rn(make_float2(hidden_x_443, hidden_y_444));
                        hidden_packed_264[21] = __as_u32(_bf16x2_181);
                        float2 _cvt_f32_108 = __bfloat1622float2(__as_bf16x2(gate_packed_262[22]));
                        float2 _cvt_f32_109 = __bfloat1622float2(__as_bf16x2(up_packed_263[22]));
                        float gate_x_445 = _cvt_f32_108.x;
                        float gate_y_446 = _cvt_f32_108.y;
                        float up_x_447 = _cvt_f32_109.x;
                        float up_y_448 = _cvt_f32_109.y;
                        float _exp_108 = expf(gate_x_445 * -1.0f);
                        float denominator_x_449 = _exp_108 + 1.0f;
                        float _exp_109 = expf(gate_y_446 * -1.0f);
                        float denominator_y_450 = _exp_109 + 1.0f;
                        float hidden_x_451 = gate_x_445 / denominator_x_449 * up_x_447;
                        float hidden_y_452 = gate_y_446 / denominator_y_450 * up_y_448;
                        __nv_bfloat162 _bf16x2_182 = __float22bfloat162_rn(make_float2(hidden_x_451, hidden_y_452));
                        hidden_packed_264[22] = __as_u32(_bf16x2_182);
                        float2 _cvt_f32_110 = __bfloat1622float2(__as_bf16x2(gate_packed_262[23]));
                        float2 _cvt_f32_111 = __bfloat1622float2(__as_bf16x2(up_packed_263[23]));
                        float gate_x_453 = _cvt_f32_110.x;
                        float gate_y_454 = _cvt_f32_110.y;
                        float up_x_455 = _cvt_f32_111.x;
                        float up_y_456 = _cvt_f32_111.y;
                        float _exp_110 = expf(gate_x_453 * -1.0f);
                        float denominator_x_457 = _exp_110 + 1.0f;
                        float _exp_111 = expf(gate_y_454 * -1.0f);
                        float denominator_y_458 = _exp_111 + 1.0f;
                        float hidden_x_459 = gate_x_453 / denominator_x_457 * up_x_455;
                        float hidden_y_460 = gate_y_454 / denominator_y_458 * up_y_456;
                        __nv_bfloat162 _bf16x2_183 = __float22bfloat162_rn(make_float2(hidden_x_459, hidden_y_460));
                        hidden_packed_264[23] = __as_u32(_bf16x2_183);
                        float2 _cvt_f32_112 = __bfloat1622float2(__as_bf16x2(gate_packed_262[24]));
                        float2 _cvt_f32_113 = __bfloat1622float2(__as_bf16x2(up_packed_263[24]));
                        float gate_x_461 = _cvt_f32_112.x;
                        float gate_y_462 = _cvt_f32_112.y;
                        float up_x_463 = _cvt_f32_113.x;
                        float up_y_464 = _cvt_f32_113.y;
                        float _exp_112 = expf(gate_x_461 * -1.0f);
                        float denominator_x_465 = _exp_112 + 1.0f;
                        float _exp_113 = expf(gate_y_462 * -1.0f);
                        float denominator_y_466 = _exp_113 + 1.0f;
                        float hidden_x_467 = gate_x_461 / denominator_x_465 * up_x_463;
                        float hidden_y_468 = gate_y_462 / denominator_y_466 * up_y_464;
                        __nv_bfloat162 _bf16x2_184 = __float22bfloat162_rn(make_float2(hidden_x_467, hidden_y_468));
                        hidden_packed_264[24] = __as_u32(_bf16x2_184);
                        float2 _cvt_f32_114 = __bfloat1622float2(__as_bf16x2(gate_packed_262[25]));
                        float2 _cvt_f32_115 = __bfloat1622float2(__as_bf16x2(up_packed_263[25]));
                        float gate_x_469 = _cvt_f32_114.x;
                        float gate_y_470 = _cvt_f32_114.y;
                        float up_x_471 = _cvt_f32_115.x;
                        float up_y_472 = _cvt_f32_115.y;
                        float _exp_114 = expf(gate_x_469 * -1.0f);
                        float denominator_x_473 = _exp_114 + 1.0f;
                        float _exp_115 = expf(gate_y_470 * -1.0f);
                        float denominator_y_474 = _exp_115 + 1.0f;
                        float hidden_x_475 = gate_x_469 / denominator_x_473 * up_x_471;
                        float hidden_y_476 = gate_y_470 / denominator_y_474 * up_y_472;
                        __nv_bfloat162 _bf16x2_185 = __float22bfloat162_rn(make_float2(hidden_x_475, hidden_y_476));
                        hidden_packed_264[25] = __as_u32(_bf16x2_185);
                        float2 _cvt_f32_116 = __bfloat1622float2(__as_bf16x2(gate_packed_262[26]));
                        float2 _cvt_f32_117 = __bfloat1622float2(__as_bf16x2(up_packed_263[26]));
                        float gate_x_477 = _cvt_f32_116.x;
                        float gate_y_478 = _cvt_f32_116.y;
                        float up_x_479 = _cvt_f32_117.x;
                        float up_y_480 = _cvt_f32_117.y;
                        float _exp_116 = expf(gate_x_477 * -1.0f);
                        float denominator_x_481 = _exp_116 + 1.0f;
                        float _exp_117 = expf(gate_y_478 * -1.0f);
                        float denominator_y_482 = _exp_117 + 1.0f;
                        float hidden_x_483 = gate_x_477 / denominator_x_481 * up_x_479;
                        float hidden_y_484 = gate_y_478 / denominator_y_482 * up_y_480;
                        __nv_bfloat162 _bf16x2_186 = __float22bfloat162_rn(make_float2(hidden_x_483, hidden_y_484));
                        hidden_packed_264[26] = __as_u32(_bf16x2_186);
                        float2 _cvt_f32_118 = __bfloat1622float2(__as_bf16x2(gate_packed_262[27]));
                        float2 _cvt_f32_119 = __bfloat1622float2(__as_bf16x2(up_packed_263[27]));
                        float gate_x_485 = _cvt_f32_118.x;
                        float gate_y_486 = _cvt_f32_118.y;
                        float up_x_487 = _cvt_f32_119.x;
                        float up_y_488 = _cvt_f32_119.y;
                        float _exp_118 = expf(gate_x_485 * -1.0f);
                        float denominator_x_489 = _exp_118 + 1.0f;
                        float _exp_119 = expf(gate_y_486 * -1.0f);
                        float denominator_y_490 = _exp_119 + 1.0f;
                        float hidden_x_491 = gate_x_485 / denominator_x_489 * up_x_487;
                        float hidden_y_492 = gate_y_486 / denominator_y_490 * up_y_488;
                        __nv_bfloat162 _bf16x2_187 = __float22bfloat162_rn(make_float2(hidden_x_491, hidden_y_492));
                        hidden_packed_264[27] = __as_u32(_bf16x2_187);
                        float2 _cvt_f32_120 = __bfloat1622float2(__as_bf16x2(gate_packed_262[28]));
                        float2 _cvt_f32_121 = __bfloat1622float2(__as_bf16x2(up_packed_263[28]));
                        float gate_x_493 = _cvt_f32_120.x;
                        float gate_y_494 = _cvt_f32_120.y;
                        float up_x_495 = _cvt_f32_121.x;
                        float up_y_496 = _cvt_f32_121.y;
                        float _exp_120 = expf(gate_x_493 * -1.0f);
                        float denominator_x_497 = _exp_120 + 1.0f;
                        float _exp_121 = expf(gate_y_494 * -1.0f);
                        float denominator_y_498 = _exp_121 + 1.0f;
                        float hidden_x_499 = gate_x_493 / denominator_x_497 * up_x_495;
                        float hidden_y_500 = gate_y_494 / denominator_y_498 * up_y_496;
                        __nv_bfloat162 _bf16x2_188 = __float22bfloat162_rn(make_float2(hidden_x_499, hidden_y_500));
                        hidden_packed_264[28] = __as_u32(_bf16x2_188);
                        float2 _cvt_f32_122 = __bfloat1622float2(__as_bf16x2(gate_packed_262[29]));
                        float2 _cvt_f32_123 = __bfloat1622float2(__as_bf16x2(up_packed_263[29]));
                        float gate_x_501 = _cvt_f32_122.x;
                        float gate_y_502 = _cvt_f32_122.y;
                        float up_x_503 = _cvt_f32_123.x;
                        float up_y_504 = _cvt_f32_123.y;
                        float _exp_122 = expf(gate_x_501 * -1.0f);
                        float denominator_x_505 = _exp_122 + 1.0f;
                        float _exp_123 = expf(gate_y_502 * -1.0f);
                        float denominator_y_506 = _exp_123 + 1.0f;
                        float hidden_x_507 = gate_x_501 / denominator_x_505 * up_x_503;
                        float hidden_y_508 = gate_y_502 / denominator_y_506 * up_y_504;
                        __nv_bfloat162 _bf16x2_189 = __float22bfloat162_rn(make_float2(hidden_x_507, hidden_y_508));
                        hidden_packed_264[29] = __as_u32(_bf16x2_189);
                        float2 _cvt_f32_124 = __bfloat1622float2(__as_bf16x2(gate_packed_262[30]));
                        float2 _cvt_f32_125 = __bfloat1622float2(__as_bf16x2(up_packed_263[30]));
                        float gate_x_509 = _cvt_f32_124.x;
                        float gate_y_510 = _cvt_f32_124.y;
                        float up_x_511 = _cvt_f32_125.x;
                        float up_y_512 = _cvt_f32_125.y;
                        float _exp_124 = expf(gate_x_509 * -1.0f);
                        float denominator_x_513 = _exp_124 + 1.0f;
                        float _exp_125 = expf(gate_y_510 * -1.0f);
                        float denominator_y_514 = _exp_125 + 1.0f;
                        float hidden_x_515 = gate_x_509 / denominator_x_513 * up_x_511;
                        float hidden_y_516 = gate_y_510 / denominator_y_514 * up_y_512;
                        __nv_bfloat162 _bf16x2_190 = __float22bfloat162_rn(make_float2(hidden_x_515, hidden_y_516));
                        hidden_packed_264[30] = __as_u32(_bf16x2_190);
                        float2 _cvt_f32_126 = __bfloat1622float2(__as_bf16x2(gate_packed_262[31]));
                        float2 _cvt_f32_127 = __bfloat1622float2(__as_bf16x2(up_packed_263[31]));
                        float gate_x_517 = _cvt_f32_126.x;
                        float gate_y_518 = _cvt_f32_126.y;
                        float up_x_519 = _cvt_f32_127.x;
                        float up_y_520 = _cvt_f32_127.y;
                        float _exp_126 = expf(gate_x_517 * -1.0f);
                        float denominator_x_521 = _exp_126 + 1.0f;
                        float _exp_127 = expf(gate_y_518 * -1.0f);
                        float denominator_y_522 = _exp_127 + 1.0f;
                        float hidden_x_523 = gate_x_517 / denominator_x_521 * up_x_519;
                        float hidden_y_524 = gate_y_518 / denominator_y_522 * up_y_520;
                        __nv_bfloat162 _bf16x2_191 = __float22bfloat162_rn(make_float2(hidden_x_523, hidden_y_524));
                        hidden_packed_264[31] = __as_u32(_bf16x2_191);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_525 = tid / 32;
                        int lane_526 = tid % 32;
                        #pragma unroll
                        for (int half_6 = 0; half_6 < 2; half_6++) {
                            #pragma unroll
                            for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                int row_10 = warp_525 * 32 + half_6 * 16 + lane_526 % 16;
                                int col_7 = col_tile_6 * 16 + lane_526 / 16 * 8;
                                unsigned int address_3_6 = d_smem_addr + (unsigned int)((row_10 * 32 + col_7) * 2);
                                address_3_6 = address_3_6 ^ (address_3_6 & 511) >> 7 << 4;
                                int offset_6 = half_6 * 8 + col_tile_6 * 4;
                                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(address_3_6);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_6 + 3]))
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
                            for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                int row_11 = warp_527 * 32 + half_7 * 16 + lane_528 % 16;
                                int col_8 = col_tile_7 * 16 + lane_528 / 16 * 8;
                                unsigned int address_3_7 = d_smem_addr + 8192 + (unsigned int)((row_11 * 32 + col_8) * 2);
                                address_3_7 = address_3_7 ^ (address_3_7 & 511) >> 7 << 4;
                                int offset_7 = half_7 * 8 + col_tile_7 * 4;
                                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(address_3_7);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_7 + 3]))
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
                            for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                int row_12 = warp_529 * 32 + half_8 * 16 + lane_530 % 16;
                                int col_9 = col_tile_8 * 16 + lane_530 / 16 * 8;
                                unsigned int address_3_8 = d_smem_addr + 16384 + (unsigned int)((row_12 * 32 + col_9) * 2);
                                address_3_8 = address_3_8 ^ (address_3_8 & 511) >> 7 << 4;
                                int offset_8 = half_8 * 8 + col_tile_8 * 4;
                                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(address_3_8);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_8 + 3]))
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
                            for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                int row_13 = warp_531 * 32 + half_9 * 16 + lane_532 % 16;
                                int col_10 = col_tile_9 * 16 + lane_532 / 16 * 8;
                                unsigned int address_3_9 = d_smem_addr + (unsigned int)((row_13 * 32 + col_10) * 2);
                                address_3_9 = address_3_9 ^ (address_3_9 & 511) >> 7 << 4;
                                int offset_9 = 16 + half_9 * 8 + col_tile_9 * 4;
                                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(address_3_9);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262[offset_9 + 3]))
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
                            for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                int row_14 = warp_533 * 32 + half_10 * 16 + lane_534 % 16;
                                int col_11 = col_tile_10 * 16 + lane_534 / 16 * 8;
                                unsigned int address_3_10 = d_smem_addr + 8192 + (unsigned int)((row_14 * 32 + col_11) * 2);
                                address_3_10 = address_3_10 ^ (address_3_10 & 511) >> 7 << 4;
                                int offset_10 = 16 + half_10 * 8 + col_tile_10 * 4;
                                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(address_3_10);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263[offset_10 + 3]))
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
                            for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                int row_15 = warp_535 * 32 + half_11 * 16 + lane_536 % 16;
                                int col_12 = col_tile_11 * 16 + lane_536 / 16 * 8;
                                unsigned int address_3_11 = d_smem_addr + 16384 + (unsigned int)((row_15 * 32 + col_12) * 2);
                                address_3_11 = address_3_11 ^ (address_3_11 & 511) >> 7 << 4;
                                int offset_11 = 16 + half_11 * 8 + col_tile_11 * 4;
                                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(address_3_11);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264[offset_11 + 3]))
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
                        __nv_bfloat162 _bf16x2_192 = __float22bfloat162_rn(make_float2(_tmem_load_16[0], _tmem_load_16[1]));
                        gate_packed_537[0] = __as_u32(_bf16x2_192);
                        __nv_bfloat162 _bf16x2_193 = __float22bfloat162_rn(make_float2(_tmem_load_17[0], _tmem_load_17[1]));
                        up_packed_538[0] = __as_u32(_bf16x2_193);
                        __nv_bfloat162 _bf16x2_194 = __float22bfloat162_rn(make_float2(_tmem_load_16[2], _tmem_load_16[3]));
                        gate_packed_537[1] = __as_u32(_bf16x2_194);
                        __nv_bfloat162 _bf16x2_195 = __float22bfloat162_rn(make_float2(_tmem_load_17[2], _tmem_load_17[3]));
                        up_packed_538[1] = __as_u32(_bf16x2_195);
                        __nv_bfloat162 _bf16x2_196 = __float22bfloat162_rn(make_float2(_tmem_load_16[4], _tmem_load_16[5]));
                        gate_packed_537[2] = __as_u32(_bf16x2_196);
                        __nv_bfloat162 _bf16x2_197 = __float22bfloat162_rn(make_float2(_tmem_load_17[4], _tmem_load_17[5]));
                        up_packed_538[2] = __as_u32(_bf16x2_197);
                        __nv_bfloat162 _bf16x2_198 = __float22bfloat162_rn(make_float2(_tmem_load_16[6], _tmem_load_16[7]));
                        gate_packed_537[3] = __as_u32(_bf16x2_198);
                        __nv_bfloat162 _bf16x2_199 = __float22bfloat162_rn(make_float2(_tmem_load_17[6], _tmem_load_17[7]));
                        up_packed_538[3] = __as_u32(_bf16x2_199);
                        __nv_bfloat162 _bf16x2_200 = __float22bfloat162_rn(make_float2(_tmem_load_16[8], _tmem_load_16[9]));
                        gate_packed_537[4] = __as_u32(_bf16x2_200);
                        __nv_bfloat162 _bf16x2_201 = __float22bfloat162_rn(make_float2(_tmem_load_17[8], _tmem_load_17[9]));
                        up_packed_538[4] = __as_u32(_bf16x2_201);
                        __nv_bfloat162 _bf16x2_202 = __float22bfloat162_rn(make_float2(_tmem_load_16[10], _tmem_load_16[11]));
                        gate_packed_537[5] = __as_u32(_bf16x2_202);
                        __nv_bfloat162 _bf16x2_203 = __float22bfloat162_rn(make_float2(_tmem_load_17[10], _tmem_load_17[11]));
                        up_packed_538[5] = __as_u32(_bf16x2_203);
                        __nv_bfloat162 _bf16x2_204 = __float22bfloat162_rn(make_float2(_tmem_load_16[12], _tmem_load_16[13]));
                        gate_packed_537[6] = __as_u32(_bf16x2_204);
                        __nv_bfloat162 _bf16x2_205 = __float22bfloat162_rn(make_float2(_tmem_load_17[12], _tmem_load_17[13]));
                        up_packed_538[6] = __as_u32(_bf16x2_205);
                        __nv_bfloat162 _bf16x2_206 = __float22bfloat162_rn(make_float2(_tmem_load_16[14], _tmem_load_16[15]));
                        gate_packed_537[7] = __as_u32(_bf16x2_206);
                        __nv_bfloat162 _bf16x2_207 = __float22bfloat162_rn(make_float2(_tmem_load_17[14], _tmem_load_17[15]));
                        up_packed_538[7] = __as_u32(_bf16x2_207);
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
                        __nv_bfloat162 _bf16x2_208 = __float22bfloat162_rn(make_float2(_tmem_load_18[0], _tmem_load_18[1]));
                        gate_packed_537[8] = __as_u32(_bf16x2_208);
                        __nv_bfloat162 _bf16x2_209 = __float22bfloat162_rn(make_float2(_tmem_load_19[0], _tmem_load_19[1]));
                        up_packed_538[8] = __as_u32(_bf16x2_209);
                        __nv_bfloat162 _bf16x2_210 = __float22bfloat162_rn(make_float2(_tmem_load_18[2], _tmem_load_18[3]));
                        gate_packed_537[9] = __as_u32(_bf16x2_210);
                        __nv_bfloat162 _bf16x2_211 = __float22bfloat162_rn(make_float2(_tmem_load_19[2], _tmem_load_19[3]));
                        up_packed_538[9] = __as_u32(_bf16x2_211);
                        __nv_bfloat162 _bf16x2_212 = __float22bfloat162_rn(make_float2(_tmem_load_18[4], _tmem_load_18[5]));
                        gate_packed_537[10] = __as_u32(_bf16x2_212);
                        __nv_bfloat162 _bf16x2_213 = __float22bfloat162_rn(make_float2(_tmem_load_19[4], _tmem_load_19[5]));
                        up_packed_538[10] = __as_u32(_bf16x2_213);
                        __nv_bfloat162 _bf16x2_214 = __float22bfloat162_rn(make_float2(_tmem_load_18[6], _tmem_load_18[7]));
                        gate_packed_537[11] = __as_u32(_bf16x2_214);
                        __nv_bfloat162 _bf16x2_215 = __float22bfloat162_rn(make_float2(_tmem_load_19[6], _tmem_load_19[7]));
                        up_packed_538[11] = __as_u32(_bf16x2_215);
                        __nv_bfloat162 _bf16x2_216 = __float22bfloat162_rn(make_float2(_tmem_load_18[8], _tmem_load_18[9]));
                        gate_packed_537[12] = __as_u32(_bf16x2_216);
                        __nv_bfloat162 _bf16x2_217 = __float22bfloat162_rn(make_float2(_tmem_load_19[8], _tmem_load_19[9]));
                        up_packed_538[12] = __as_u32(_bf16x2_217);
                        __nv_bfloat162 _bf16x2_218 = __float22bfloat162_rn(make_float2(_tmem_load_18[10], _tmem_load_18[11]));
                        gate_packed_537[13] = __as_u32(_bf16x2_218);
                        __nv_bfloat162 _bf16x2_219 = __float22bfloat162_rn(make_float2(_tmem_load_19[10], _tmem_load_19[11]));
                        up_packed_538[13] = __as_u32(_bf16x2_219);
                        __nv_bfloat162 _bf16x2_220 = __float22bfloat162_rn(make_float2(_tmem_load_18[12], _tmem_load_18[13]));
                        gate_packed_537[14] = __as_u32(_bf16x2_220);
                        __nv_bfloat162 _bf16x2_221 = __float22bfloat162_rn(make_float2(_tmem_load_19[12], _tmem_load_19[13]));
                        up_packed_538[14] = __as_u32(_bf16x2_221);
                        __nv_bfloat162 _bf16x2_222 = __float22bfloat162_rn(make_float2(_tmem_load_18[14], _tmem_load_18[15]));
                        gate_packed_537[15] = __as_u32(_bf16x2_222);
                        __nv_bfloat162 _bf16x2_223 = __float22bfloat162_rn(make_float2(_tmem_load_19[14], _tmem_load_19[15]));
                        up_packed_538[15] = __as_u32(_bf16x2_223);
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
                        __nv_bfloat162 _bf16x2_224 = __float22bfloat162_rn(make_float2(_tmem_load_20[0], _tmem_load_20[1]));
                        gate_packed_537[16] = __as_u32(_bf16x2_224);
                        __nv_bfloat162 _bf16x2_225 = __float22bfloat162_rn(make_float2(_tmem_load_21[0], _tmem_load_21[1]));
                        up_packed_538[16] = __as_u32(_bf16x2_225);
                        __nv_bfloat162 _bf16x2_226 = __float22bfloat162_rn(make_float2(_tmem_load_20[2], _tmem_load_20[3]));
                        gate_packed_537[17] = __as_u32(_bf16x2_226);
                        __nv_bfloat162 _bf16x2_227 = __float22bfloat162_rn(make_float2(_tmem_load_21[2], _tmem_load_21[3]));
                        up_packed_538[17] = __as_u32(_bf16x2_227);
                        __nv_bfloat162 _bf16x2_228 = __float22bfloat162_rn(make_float2(_tmem_load_20[4], _tmem_load_20[5]));
                        gate_packed_537[18] = __as_u32(_bf16x2_228);
                        __nv_bfloat162 _bf16x2_229 = __float22bfloat162_rn(make_float2(_tmem_load_21[4], _tmem_load_21[5]));
                        up_packed_538[18] = __as_u32(_bf16x2_229);
                        __nv_bfloat162 _bf16x2_230 = __float22bfloat162_rn(make_float2(_tmem_load_20[6], _tmem_load_20[7]));
                        gate_packed_537[19] = __as_u32(_bf16x2_230);
                        __nv_bfloat162 _bf16x2_231 = __float22bfloat162_rn(make_float2(_tmem_load_21[6], _tmem_load_21[7]));
                        up_packed_538[19] = __as_u32(_bf16x2_231);
                        __nv_bfloat162 _bf16x2_232 = __float22bfloat162_rn(make_float2(_tmem_load_20[8], _tmem_load_20[9]));
                        gate_packed_537[20] = __as_u32(_bf16x2_232);
                        __nv_bfloat162 _bf16x2_233 = __float22bfloat162_rn(make_float2(_tmem_load_21[8], _tmem_load_21[9]));
                        up_packed_538[20] = __as_u32(_bf16x2_233);
                        __nv_bfloat162 _bf16x2_234 = __float22bfloat162_rn(make_float2(_tmem_load_20[10], _tmem_load_20[11]));
                        gate_packed_537[21] = __as_u32(_bf16x2_234);
                        __nv_bfloat162 _bf16x2_235 = __float22bfloat162_rn(make_float2(_tmem_load_21[10], _tmem_load_21[11]));
                        up_packed_538[21] = __as_u32(_bf16x2_235);
                        __nv_bfloat162 _bf16x2_236 = __float22bfloat162_rn(make_float2(_tmem_load_20[12], _tmem_load_20[13]));
                        gate_packed_537[22] = __as_u32(_bf16x2_236);
                        __nv_bfloat162 _bf16x2_237 = __float22bfloat162_rn(make_float2(_tmem_load_21[12], _tmem_load_21[13]));
                        up_packed_538[22] = __as_u32(_bf16x2_237);
                        __nv_bfloat162 _bf16x2_238 = __float22bfloat162_rn(make_float2(_tmem_load_20[14], _tmem_load_20[15]));
                        gate_packed_537[23] = __as_u32(_bf16x2_238);
                        __nv_bfloat162 _bf16x2_239 = __float22bfloat162_rn(make_float2(_tmem_load_21[14], _tmem_load_21[15]));
                        up_packed_538[23] = __as_u32(_bf16x2_239);
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
                        __nv_bfloat162 _bf16x2_240 = __float22bfloat162_rn(make_float2(_tmem_load_22[0], _tmem_load_22[1]));
                        gate_packed_537[24] = __as_u32(_bf16x2_240);
                        __nv_bfloat162 _bf16x2_241 = __float22bfloat162_rn(make_float2(_tmem_load_23[0], _tmem_load_23[1]));
                        up_packed_538[24] = __as_u32(_bf16x2_241);
                        __nv_bfloat162 _bf16x2_242 = __float22bfloat162_rn(make_float2(_tmem_load_22[2], _tmem_load_22[3]));
                        gate_packed_537[25] = __as_u32(_bf16x2_242);
                        __nv_bfloat162 _bf16x2_243 = __float22bfloat162_rn(make_float2(_tmem_load_23[2], _tmem_load_23[3]));
                        up_packed_538[25] = __as_u32(_bf16x2_243);
                        __nv_bfloat162 _bf16x2_244 = __float22bfloat162_rn(make_float2(_tmem_load_22[4], _tmem_load_22[5]));
                        gate_packed_537[26] = __as_u32(_bf16x2_244);
                        __nv_bfloat162 _bf16x2_245 = __float22bfloat162_rn(make_float2(_tmem_load_23[4], _tmem_load_23[5]));
                        up_packed_538[26] = __as_u32(_bf16x2_245);
                        __nv_bfloat162 _bf16x2_246 = __float22bfloat162_rn(make_float2(_tmem_load_22[6], _tmem_load_22[7]));
                        gate_packed_537[27] = __as_u32(_bf16x2_246);
                        __nv_bfloat162 _bf16x2_247 = __float22bfloat162_rn(make_float2(_tmem_load_23[6], _tmem_load_23[7]));
                        up_packed_538[27] = __as_u32(_bf16x2_247);
                        __nv_bfloat162 _bf16x2_248 = __float22bfloat162_rn(make_float2(_tmem_load_22[8], _tmem_load_22[9]));
                        gate_packed_537[28] = __as_u32(_bf16x2_248);
                        __nv_bfloat162 _bf16x2_249 = __float22bfloat162_rn(make_float2(_tmem_load_23[8], _tmem_load_23[9]));
                        up_packed_538[28] = __as_u32(_bf16x2_249);
                        __nv_bfloat162 _bf16x2_250 = __float22bfloat162_rn(make_float2(_tmem_load_22[10], _tmem_load_22[11]));
                        gate_packed_537[29] = __as_u32(_bf16x2_250);
                        __nv_bfloat162 _bf16x2_251 = __float22bfloat162_rn(make_float2(_tmem_load_23[10], _tmem_load_23[11]));
                        up_packed_538[29] = __as_u32(_bf16x2_251);
                        __nv_bfloat162 _bf16x2_252 = __float22bfloat162_rn(make_float2(_tmem_load_22[12], _tmem_load_22[13]));
                        gate_packed_537[30] = __as_u32(_bf16x2_252);
                        __nv_bfloat162 _bf16x2_253 = __float22bfloat162_rn(make_float2(_tmem_load_23[12], _tmem_load_23[13]));
                        up_packed_538[30] = __as_u32(_bf16x2_253);
                        __nv_bfloat162 _bf16x2_254 = __float22bfloat162_rn(make_float2(_tmem_load_22[14], _tmem_load_22[15]));
                        gate_packed_537[31] = __as_u32(_bf16x2_254);
                        __nv_bfloat162 _bf16x2_255 = __float22bfloat162_rn(make_float2(_tmem_load_23[14], _tmem_load_23[15]));
                        up_packed_538[31] = __as_u32(_bf16x2_255);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _cvt_f32_128 = __bfloat1622float2(__as_bf16x2(gate_packed_537[0]));
                        float2 _cvt_f32_129 = __bfloat1622float2(__as_bf16x2(up_packed_538[0]));
                        float gate_x_544 = _cvt_f32_128.x;
                        float gate_y_545 = _cvt_f32_128.y;
                        float up_x_546 = _cvt_f32_129.x;
                        float up_y_547 = _cvt_f32_129.y;
                        float _exp_128 = expf(gate_x_544 * -1.0f);
                        float denominator_x_548 = _exp_128 + 1.0f;
                        float _exp_129 = expf(gate_y_545 * -1.0f);
                        float denominator_y_549 = _exp_129 + 1.0f;
                        float hidden_x_550 = gate_x_544 / denominator_x_548 * up_x_546;
                        float hidden_y_551 = gate_y_545 / denominator_y_549 * up_y_547;
                        __nv_bfloat162 _bf16x2_256 = __float22bfloat162_rn(make_float2(hidden_x_550, hidden_y_551));
                        hidden_packed_539[0] = __as_u32(_bf16x2_256);
                        float2 _cvt_f32_130 = __bfloat1622float2(__as_bf16x2(gate_packed_537[1]));
                        float2 _cvt_f32_131 = __bfloat1622float2(__as_bf16x2(up_packed_538[1]));
                        float gate_x_552 = _cvt_f32_130.x;
                        float gate_y_553 = _cvt_f32_130.y;
                        float up_x_554 = _cvt_f32_131.x;
                        float up_y_555 = _cvt_f32_131.y;
                        float _exp_130 = expf(gate_x_552 * -1.0f);
                        float denominator_x_556 = _exp_130 + 1.0f;
                        float _exp_131 = expf(gate_y_553 * -1.0f);
                        float denominator_y_557 = _exp_131 + 1.0f;
                        float hidden_x_558 = gate_x_552 / denominator_x_556 * up_x_554;
                        float hidden_y_559 = gate_y_553 / denominator_y_557 * up_y_555;
                        __nv_bfloat162 _bf16x2_257 = __float22bfloat162_rn(make_float2(hidden_x_558, hidden_y_559));
                        hidden_packed_539[1] = __as_u32(_bf16x2_257);
                        float2 _cvt_f32_132 = __bfloat1622float2(__as_bf16x2(gate_packed_537[2]));
                        float2 _cvt_f32_133 = __bfloat1622float2(__as_bf16x2(up_packed_538[2]));
                        float gate_x_560 = _cvt_f32_132.x;
                        float gate_y_561 = _cvt_f32_132.y;
                        float up_x_562 = _cvt_f32_133.x;
                        float up_y_563 = _cvt_f32_133.y;
                        float _exp_132 = expf(gate_x_560 * -1.0f);
                        float denominator_x_564 = _exp_132 + 1.0f;
                        float _exp_133 = expf(gate_y_561 * -1.0f);
                        float denominator_y_565 = _exp_133 + 1.0f;
                        float hidden_x_566 = gate_x_560 / denominator_x_564 * up_x_562;
                        float hidden_y_567 = gate_y_561 / denominator_y_565 * up_y_563;
                        __nv_bfloat162 _bf16x2_258 = __float22bfloat162_rn(make_float2(hidden_x_566, hidden_y_567));
                        hidden_packed_539[2] = __as_u32(_bf16x2_258);
                        float2 _cvt_f32_134 = __bfloat1622float2(__as_bf16x2(gate_packed_537[3]));
                        float2 _cvt_f32_135 = __bfloat1622float2(__as_bf16x2(up_packed_538[3]));
                        float gate_x_568 = _cvt_f32_134.x;
                        float gate_y_569 = _cvt_f32_134.y;
                        float up_x_570 = _cvt_f32_135.x;
                        float up_y_571 = _cvt_f32_135.y;
                        float _exp_134 = expf(gate_x_568 * -1.0f);
                        float denominator_x_572 = _exp_134 + 1.0f;
                        float _exp_135 = expf(gate_y_569 * -1.0f);
                        float denominator_y_573 = _exp_135 + 1.0f;
                        float hidden_x_574 = gate_x_568 / denominator_x_572 * up_x_570;
                        float hidden_y_575 = gate_y_569 / denominator_y_573 * up_y_571;
                        __nv_bfloat162 _bf16x2_259 = __float22bfloat162_rn(make_float2(hidden_x_574, hidden_y_575));
                        hidden_packed_539[3] = __as_u32(_bf16x2_259);
                        float2 _cvt_f32_136 = __bfloat1622float2(__as_bf16x2(gate_packed_537[4]));
                        float2 _cvt_f32_137 = __bfloat1622float2(__as_bf16x2(up_packed_538[4]));
                        float gate_x_576 = _cvt_f32_136.x;
                        float gate_y_577 = _cvt_f32_136.y;
                        float up_x_578 = _cvt_f32_137.x;
                        float up_y_579 = _cvt_f32_137.y;
                        float _exp_136 = expf(gate_x_576 * -1.0f);
                        float denominator_x_580 = _exp_136 + 1.0f;
                        float _exp_137 = expf(gate_y_577 * -1.0f);
                        float denominator_y_581 = _exp_137 + 1.0f;
                        float hidden_x_582 = gate_x_576 / denominator_x_580 * up_x_578;
                        float hidden_y_583 = gate_y_577 / denominator_y_581 * up_y_579;
                        __nv_bfloat162 _bf16x2_260 = __float22bfloat162_rn(make_float2(hidden_x_582, hidden_y_583));
                        hidden_packed_539[4] = __as_u32(_bf16x2_260);
                        float2 _cvt_f32_138 = __bfloat1622float2(__as_bf16x2(gate_packed_537[5]));
                        float2 _cvt_f32_139 = __bfloat1622float2(__as_bf16x2(up_packed_538[5]));
                        float gate_x_584 = _cvt_f32_138.x;
                        float gate_y_585 = _cvt_f32_138.y;
                        float up_x_586 = _cvt_f32_139.x;
                        float up_y_587 = _cvt_f32_139.y;
                        float _exp_138 = expf(gate_x_584 * -1.0f);
                        float denominator_x_588 = _exp_138 + 1.0f;
                        float _exp_139 = expf(gate_y_585 * -1.0f);
                        float denominator_y_589 = _exp_139 + 1.0f;
                        float hidden_x_590 = gate_x_584 / denominator_x_588 * up_x_586;
                        float hidden_y_591 = gate_y_585 / denominator_y_589 * up_y_587;
                        __nv_bfloat162 _bf16x2_261 = __float22bfloat162_rn(make_float2(hidden_x_590, hidden_y_591));
                        hidden_packed_539[5] = __as_u32(_bf16x2_261);
                        float2 _cvt_f32_140 = __bfloat1622float2(__as_bf16x2(gate_packed_537[6]));
                        float2 _cvt_f32_141 = __bfloat1622float2(__as_bf16x2(up_packed_538[6]));
                        float gate_x_592 = _cvt_f32_140.x;
                        float gate_y_593 = _cvt_f32_140.y;
                        float up_x_594 = _cvt_f32_141.x;
                        float up_y_595 = _cvt_f32_141.y;
                        float _exp_140 = expf(gate_x_592 * -1.0f);
                        float denominator_x_596 = _exp_140 + 1.0f;
                        float _exp_141 = expf(gate_y_593 * -1.0f);
                        float denominator_y_597 = _exp_141 + 1.0f;
                        float hidden_x_598 = gate_x_592 / denominator_x_596 * up_x_594;
                        float hidden_y_599 = gate_y_593 / denominator_y_597 * up_y_595;
                        __nv_bfloat162 _bf16x2_262 = __float22bfloat162_rn(make_float2(hidden_x_598, hidden_y_599));
                        hidden_packed_539[6] = __as_u32(_bf16x2_262);
                        float2 _cvt_f32_142 = __bfloat1622float2(__as_bf16x2(gate_packed_537[7]));
                        float2 _cvt_f32_143 = __bfloat1622float2(__as_bf16x2(up_packed_538[7]));
                        float gate_x_600 = _cvt_f32_142.x;
                        float gate_y_601 = _cvt_f32_142.y;
                        float up_x_602 = _cvt_f32_143.x;
                        float up_y_603 = _cvt_f32_143.y;
                        float _exp_142 = expf(gate_x_600 * -1.0f);
                        float denominator_x_604 = _exp_142 + 1.0f;
                        float _exp_143 = expf(gate_y_601 * -1.0f);
                        float denominator_y_605 = _exp_143 + 1.0f;
                        float hidden_x_606 = gate_x_600 / denominator_x_604 * up_x_602;
                        float hidden_y_607 = gate_y_601 / denominator_y_605 * up_y_603;
                        __nv_bfloat162 _bf16x2_263 = __float22bfloat162_rn(make_float2(hidden_x_606, hidden_y_607));
                        hidden_packed_539[7] = __as_u32(_bf16x2_263);
                        float2 _cvt_f32_144 = __bfloat1622float2(__as_bf16x2(gate_packed_537[8]));
                        float2 _cvt_f32_145 = __bfloat1622float2(__as_bf16x2(up_packed_538[8]));
                        float gate_x_608 = _cvt_f32_144.x;
                        float gate_y_609 = _cvt_f32_144.y;
                        float up_x_610 = _cvt_f32_145.x;
                        float up_y_611 = _cvt_f32_145.y;
                        float _exp_144 = expf(gate_x_608 * -1.0f);
                        float denominator_x_612 = _exp_144 + 1.0f;
                        float _exp_145 = expf(gate_y_609 * -1.0f);
                        float denominator_y_613 = _exp_145 + 1.0f;
                        float hidden_x_614 = gate_x_608 / denominator_x_612 * up_x_610;
                        float hidden_y_615 = gate_y_609 / denominator_y_613 * up_y_611;
                        __nv_bfloat162 _bf16x2_264 = __float22bfloat162_rn(make_float2(hidden_x_614, hidden_y_615));
                        hidden_packed_539[8] = __as_u32(_bf16x2_264);
                        float2 _cvt_f32_146 = __bfloat1622float2(__as_bf16x2(gate_packed_537[9]));
                        float2 _cvt_f32_147 = __bfloat1622float2(__as_bf16x2(up_packed_538[9]));
                        float gate_x_616 = _cvt_f32_146.x;
                        float gate_y_617 = _cvt_f32_146.y;
                        float up_x_618 = _cvt_f32_147.x;
                        float up_y_619 = _cvt_f32_147.y;
                        float _exp_146 = expf(gate_x_616 * -1.0f);
                        float denominator_x_620 = _exp_146 + 1.0f;
                        float _exp_147 = expf(gate_y_617 * -1.0f);
                        float denominator_y_621 = _exp_147 + 1.0f;
                        float hidden_x_622 = gate_x_616 / denominator_x_620 * up_x_618;
                        float hidden_y_623 = gate_y_617 / denominator_y_621 * up_y_619;
                        __nv_bfloat162 _bf16x2_265 = __float22bfloat162_rn(make_float2(hidden_x_622, hidden_y_623));
                        hidden_packed_539[9] = __as_u32(_bf16x2_265);
                        float2 _cvt_f32_148 = __bfloat1622float2(__as_bf16x2(gate_packed_537[10]));
                        float2 _cvt_f32_149 = __bfloat1622float2(__as_bf16x2(up_packed_538[10]));
                        float gate_x_624 = _cvt_f32_148.x;
                        float gate_y_625 = _cvt_f32_148.y;
                        float up_x_626 = _cvt_f32_149.x;
                        float up_y_627 = _cvt_f32_149.y;
                        float _exp_148 = expf(gate_x_624 * -1.0f);
                        float denominator_x_628 = _exp_148 + 1.0f;
                        float _exp_149 = expf(gate_y_625 * -1.0f);
                        float denominator_y_629 = _exp_149 + 1.0f;
                        float hidden_x_630 = gate_x_624 / denominator_x_628 * up_x_626;
                        float hidden_y_631 = gate_y_625 / denominator_y_629 * up_y_627;
                        __nv_bfloat162 _bf16x2_266 = __float22bfloat162_rn(make_float2(hidden_x_630, hidden_y_631));
                        hidden_packed_539[10] = __as_u32(_bf16x2_266);
                        float2 _cvt_f32_150 = __bfloat1622float2(__as_bf16x2(gate_packed_537[11]));
                        float2 _cvt_f32_151 = __bfloat1622float2(__as_bf16x2(up_packed_538[11]));
                        float gate_x_632 = _cvt_f32_150.x;
                        float gate_y_633 = _cvt_f32_150.y;
                        float up_x_634 = _cvt_f32_151.x;
                        float up_y_635 = _cvt_f32_151.y;
                        float _exp_150 = expf(gate_x_632 * -1.0f);
                        float denominator_x_636 = _exp_150 + 1.0f;
                        float _exp_151 = expf(gate_y_633 * -1.0f);
                        float denominator_y_637 = _exp_151 + 1.0f;
                        float hidden_x_638 = gate_x_632 / denominator_x_636 * up_x_634;
                        float hidden_y_639 = gate_y_633 / denominator_y_637 * up_y_635;
                        __nv_bfloat162 _bf16x2_267 = __float22bfloat162_rn(make_float2(hidden_x_638, hidden_y_639));
                        hidden_packed_539[11] = __as_u32(_bf16x2_267);
                        float2 _cvt_f32_152 = __bfloat1622float2(__as_bf16x2(gate_packed_537[12]));
                        float2 _cvt_f32_153 = __bfloat1622float2(__as_bf16x2(up_packed_538[12]));
                        float gate_x_640 = _cvt_f32_152.x;
                        float gate_y_641 = _cvt_f32_152.y;
                        float up_x_642 = _cvt_f32_153.x;
                        float up_y_643 = _cvt_f32_153.y;
                        float _exp_152 = expf(gate_x_640 * -1.0f);
                        float denominator_x_644 = _exp_152 + 1.0f;
                        float _exp_153 = expf(gate_y_641 * -1.0f);
                        float denominator_y_645 = _exp_153 + 1.0f;
                        float hidden_x_646 = gate_x_640 / denominator_x_644 * up_x_642;
                        float hidden_y_647 = gate_y_641 / denominator_y_645 * up_y_643;
                        __nv_bfloat162 _bf16x2_268 = __float22bfloat162_rn(make_float2(hidden_x_646, hidden_y_647));
                        hidden_packed_539[12] = __as_u32(_bf16x2_268);
                        float2 _cvt_f32_154 = __bfloat1622float2(__as_bf16x2(gate_packed_537[13]));
                        float2 _cvt_f32_155 = __bfloat1622float2(__as_bf16x2(up_packed_538[13]));
                        float gate_x_648 = _cvt_f32_154.x;
                        float gate_y_649 = _cvt_f32_154.y;
                        float up_x_650 = _cvt_f32_155.x;
                        float up_y_651 = _cvt_f32_155.y;
                        float _exp_154 = expf(gate_x_648 * -1.0f);
                        float denominator_x_652 = _exp_154 + 1.0f;
                        float _exp_155 = expf(gate_y_649 * -1.0f);
                        float denominator_y_653 = _exp_155 + 1.0f;
                        float hidden_x_654 = gate_x_648 / denominator_x_652 * up_x_650;
                        float hidden_y_655 = gate_y_649 / denominator_y_653 * up_y_651;
                        __nv_bfloat162 _bf16x2_269 = __float22bfloat162_rn(make_float2(hidden_x_654, hidden_y_655));
                        hidden_packed_539[13] = __as_u32(_bf16x2_269);
                        float2 _cvt_f32_156 = __bfloat1622float2(__as_bf16x2(gate_packed_537[14]));
                        float2 _cvt_f32_157 = __bfloat1622float2(__as_bf16x2(up_packed_538[14]));
                        float gate_x_656 = _cvt_f32_156.x;
                        float gate_y_657 = _cvt_f32_156.y;
                        float up_x_658 = _cvt_f32_157.x;
                        float up_y_659 = _cvt_f32_157.y;
                        float _exp_156 = expf(gate_x_656 * -1.0f);
                        float denominator_x_660 = _exp_156 + 1.0f;
                        float _exp_157 = expf(gate_y_657 * -1.0f);
                        float denominator_y_661 = _exp_157 + 1.0f;
                        float hidden_x_662 = gate_x_656 / denominator_x_660 * up_x_658;
                        float hidden_y_663 = gate_y_657 / denominator_y_661 * up_y_659;
                        __nv_bfloat162 _bf16x2_270 = __float22bfloat162_rn(make_float2(hidden_x_662, hidden_y_663));
                        hidden_packed_539[14] = __as_u32(_bf16x2_270);
                        float2 _cvt_f32_158 = __bfloat1622float2(__as_bf16x2(gate_packed_537[15]));
                        float2 _cvt_f32_159 = __bfloat1622float2(__as_bf16x2(up_packed_538[15]));
                        float gate_x_664 = _cvt_f32_158.x;
                        float gate_y_665 = _cvt_f32_158.y;
                        float up_x_666 = _cvt_f32_159.x;
                        float up_y_667 = _cvt_f32_159.y;
                        float _exp_158 = expf(gate_x_664 * -1.0f);
                        float denominator_x_668 = _exp_158 + 1.0f;
                        float _exp_159 = expf(gate_y_665 * -1.0f);
                        float denominator_y_669 = _exp_159 + 1.0f;
                        float hidden_x_670 = gate_x_664 / denominator_x_668 * up_x_666;
                        float hidden_y_671 = gate_y_665 / denominator_y_669 * up_y_667;
                        __nv_bfloat162 _bf16x2_271 = __float22bfloat162_rn(make_float2(hidden_x_670, hidden_y_671));
                        hidden_packed_539[15] = __as_u32(_bf16x2_271);
                        float2 _cvt_f32_160 = __bfloat1622float2(__as_bf16x2(gate_packed_537[16]));
                        float2 _cvt_f32_161 = __bfloat1622float2(__as_bf16x2(up_packed_538[16]));
                        float gate_x_672 = _cvt_f32_160.x;
                        float gate_y_673 = _cvt_f32_160.y;
                        float up_x_674 = _cvt_f32_161.x;
                        float up_y_675 = _cvt_f32_161.y;
                        float _exp_160 = expf(gate_x_672 * -1.0f);
                        float denominator_x_676 = _exp_160 + 1.0f;
                        float _exp_161 = expf(gate_y_673 * -1.0f);
                        float denominator_y_677 = _exp_161 + 1.0f;
                        float hidden_x_678 = gate_x_672 / denominator_x_676 * up_x_674;
                        float hidden_y_679 = gate_y_673 / denominator_y_677 * up_y_675;
                        __nv_bfloat162 _bf16x2_272 = __float22bfloat162_rn(make_float2(hidden_x_678, hidden_y_679));
                        hidden_packed_539[16] = __as_u32(_bf16x2_272);
                        float2 _cvt_f32_162 = __bfloat1622float2(__as_bf16x2(gate_packed_537[17]));
                        float2 _cvt_f32_163 = __bfloat1622float2(__as_bf16x2(up_packed_538[17]));
                        float gate_x_680 = _cvt_f32_162.x;
                        float gate_y_681 = _cvt_f32_162.y;
                        float up_x_682 = _cvt_f32_163.x;
                        float up_y_683 = _cvt_f32_163.y;
                        float _exp_162 = expf(gate_x_680 * -1.0f);
                        float denominator_x_684 = _exp_162 + 1.0f;
                        float _exp_163 = expf(gate_y_681 * -1.0f);
                        float denominator_y_685 = _exp_163 + 1.0f;
                        float hidden_x_686 = gate_x_680 / denominator_x_684 * up_x_682;
                        float hidden_y_687 = gate_y_681 / denominator_y_685 * up_y_683;
                        __nv_bfloat162 _bf16x2_273 = __float22bfloat162_rn(make_float2(hidden_x_686, hidden_y_687));
                        hidden_packed_539[17] = __as_u32(_bf16x2_273);
                        float2 _cvt_f32_164 = __bfloat1622float2(__as_bf16x2(gate_packed_537[18]));
                        float2 _cvt_f32_165 = __bfloat1622float2(__as_bf16x2(up_packed_538[18]));
                        float gate_x_688 = _cvt_f32_164.x;
                        float gate_y_689 = _cvt_f32_164.y;
                        float up_x_690 = _cvt_f32_165.x;
                        float up_y_691 = _cvt_f32_165.y;
                        float _exp_164 = expf(gate_x_688 * -1.0f);
                        float denominator_x_692 = _exp_164 + 1.0f;
                        float _exp_165 = expf(gate_y_689 * -1.0f);
                        float denominator_y_693 = _exp_165 + 1.0f;
                        float hidden_x_694 = gate_x_688 / denominator_x_692 * up_x_690;
                        float hidden_y_695 = gate_y_689 / denominator_y_693 * up_y_691;
                        __nv_bfloat162 _bf16x2_274 = __float22bfloat162_rn(make_float2(hidden_x_694, hidden_y_695));
                        hidden_packed_539[18] = __as_u32(_bf16x2_274);
                        float2 _cvt_f32_166 = __bfloat1622float2(__as_bf16x2(gate_packed_537[19]));
                        float2 _cvt_f32_167 = __bfloat1622float2(__as_bf16x2(up_packed_538[19]));
                        float gate_x_696 = _cvt_f32_166.x;
                        float gate_y_697 = _cvt_f32_166.y;
                        float up_x_698 = _cvt_f32_167.x;
                        float up_y_699 = _cvt_f32_167.y;
                        float _exp_166 = expf(gate_x_696 * -1.0f);
                        float denominator_x_700 = _exp_166 + 1.0f;
                        float _exp_167 = expf(gate_y_697 * -1.0f);
                        float denominator_y_701 = _exp_167 + 1.0f;
                        float hidden_x_702 = gate_x_696 / denominator_x_700 * up_x_698;
                        float hidden_y_703 = gate_y_697 / denominator_y_701 * up_y_699;
                        __nv_bfloat162 _bf16x2_275 = __float22bfloat162_rn(make_float2(hidden_x_702, hidden_y_703));
                        hidden_packed_539[19] = __as_u32(_bf16x2_275);
                        float2 _cvt_f32_168 = __bfloat1622float2(__as_bf16x2(gate_packed_537[20]));
                        float2 _cvt_f32_169 = __bfloat1622float2(__as_bf16x2(up_packed_538[20]));
                        float gate_x_704 = _cvt_f32_168.x;
                        float gate_y_705 = _cvt_f32_168.y;
                        float up_x_706 = _cvt_f32_169.x;
                        float up_y_707 = _cvt_f32_169.y;
                        float _exp_168 = expf(gate_x_704 * -1.0f);
                        float denominator_x_708 = _exp_168 + 1.0f;
                        float _exp_169 = expf(gate_y_705 * -1.0f);
                        float denominator_y_709 = _exp_169 + 1.0f;
                        float hidden_x_710 = gate_x_704 / denominator_x_708 * up_x_706;
                        float hidden_y_711 = gate_y_705 / denominator_y_709 * up_y_707;
                        __nv_bfloat162 _bf16x2_276 = __float22bfloat162_rn(make_float2(hidden_x_710, hidden_y_711));
                        hidden_packed_539[20] = __as_u32(_bf16x2_276);
                        float2 _cvt_f32_170 = __bfloat1622float2(__as_bf16x2(gate_packed_537[21]));
                        float2 _cvt_f32_171 = __bfloat1622float2(__as_bf16x2(up_packed_538[21]));
                        float gate_x_712 = _cvt_f32_170.x;
                        float gate_y_713 = _cvt_f32_170.y;
                        float up_x_714 = _cvt_f32_171.x;
                        float up_y_715 = _cvt_f32_171.y;
                        float _exp_170 = expf(gate_x_712 * -1.0f);
                        float denominator_x_716 = _exp_170 + 1.0f;
                        float _exp_171 = expf(gate_y_713 * -1.0f);
                        float denominator_y_717 = _exp_171 + 1.0f;
                        float hidden_x_718 = gate_x_712 / denominator_x_716 * up_x_714;
                        float hidden_y_719 = gate_y_713 / denominator_y_717 * up_y_715;
                        __nv_bfloat162 _bf16x2_277 = __float22bfloat162_rn(make_float2(hidden_x_718, hidden_y_719));
                        hidden_packed_539[21] = __as_u32(_bf16x2_277);
                        float2 _cvt_f32_172 = __bfloat1622float2(__as_bf16x2(gate_packed_537[22]));
                        float2 _cvt_f32_173 = __bfloat1622float2(__as_bf16x2(up_packed_538[22]));
                        float gate_x_720 = _cvt_f32_172.x;
                        float gate_y_721 = _cvt_f32_172.y;
                        float up_x_722 = _cvt_f32_173.x;
                        float up_y_723 = _cvt_f32_173.y;
                        float _exp_172 = expf(gate_x_720 * -1.0f);
                        float denominator_x_724 = _exp_172 + 1.0f;
                        float _exp_173 = expf(gate_y_721 * -1.0f);
                        float denominator_y_725 = _exp_173 + 1.0f;
                        float hidden_x_726 = gate_x_720 / denominator_x_724 * up_x_722;
                        float hidden_y_727 = gate_y_721 / denominator_y_725 * up_y_723;
                        __nv_bfloat162 _bf16x2_278 = __float22bfloat162_rn(make_float2(hidden_x_726, hidden_y_727));
                        hidden_packed_539[22] = __as_u32(_bf16x2_278);
                        float2 _cvt_f32_174 = __bfloat1622float2(__as_bf16x2(gate_packed_537[23]));
                        float2 _cvt_f32_175 = __bfloat1622float2(__as_bf16x2(up_packed_538[23]));
                        float gate_x_728 = _cvt_f32_174.x;
                        float gate_y_729 = _cvt_f32_174.y;
                        float up_x_730 = _cvt_f32_175.x;
                        float up_y_731 = _cvt_f32_175.y;
                        float _exp_174 = expf(gate_x_728 * -1.0f);
                        float denominator_x_732 = _exp_174 + 1.0f;
                        float _exp_175 = expf(gate_y_729 * -1.0f);
                        float denominator_y_733 = _exp_175 + 1.0f;
                        float hidden_x_734 = gate_x_728 / denominator_x_732 * up_x_730;
                        float hidden_y_735 = gate_y_729 / denominator_y_733 * up_y_731;
                        __nv_bfloat162 _bf16x2_279 = __float22bfloat162_rn(make_float2(hidden_x_734, hidden_y_735));
                        hidden_packed_539[23] = __as_u32(_bf16x2_279);
                        float2 _cvt_f32_176 = __bfloat1622float2(__as_bf16x2(gate_packed_537[24]));
                        float2 _cvt_f32_177 = __bfloat1622float2(__as_bf16x2(up_packed_538[24]));
                        float gate_x_736 = _cvt_f32_176.x;
                        float gate_y_737 = _cvt_f32_176.y;
                        float up_x_738 = _cvt_f32_177.x;
                        float up_y_739 = _cvt_f32_177.y;
                        float _exp_176 = expf(gate_x_736 * -1.0f);
                        float denominator_x_740 = _exp_176 + 1.0f;
                        float _exp_177 = expf(gate_y_737 * -1.0f);
                        float denominator_y_741 = _exp_177 + 1.0f;
                        float hidden_x_742 = gate_x_736 / denominator_x_740 * up_x_738;
                        float hidden_y_743 = gate_y_737 / denominator_y_741 * up_y_739;
                        __nv_bfloat162 _bf16x2_280 = __float22bfloat162_rn(make_float2(hidden_x_742, hidden_y_743));
                        hidden_packed_539[24] = __as_u32(_bf16x2_280);
                        float2 _cvt_f32_178 = __bfloat1622float2(__as_bf16x2(gate_packed_537[25]));
                        float2 _cvt_f32_179 = __bfloat1622float2(__as_bf16x2(up_packed_538[25]));
                        float gate_x_744 = _cvt_f32_178.x;
                        float gate_y_745 = _cvt_f32_178.y;
                        float up_x_746 = _cvt_f32_179.x;
                        float up_y_747 = _cvt_f32_179.y;
                        float _exp_178 = expf(gate_x_744 * -1.0f);
                        float denominator_x_748 = _exp_178 + 1.0f;
                        float _exp_179 = expf(gate_y_745 * -1.0f);
                        float denominator_y_749 = _exp_179 + 1.0f;
                        float hidden_x_750 = gate_x_744 / denominator_x_748 * up_x_746;
                        float hidden_y_751 = gate_y_745 / denominator_y_749 * up_y_747;
                        __nv_bfloat162 _bf16x2_281 = __float22bfloat162_rn(make_float2(hidden_x_750, hidden_y_751));
                        hidden_packed_539[25] = __as_u32(_bf16x2_281);
                        float2 _cvt_f32_180 = __bfloat1622float2(__as_bf16x2(gate_packed_537[26]));
                        float2 _cvt_f32_181 = __bfloat1622float2(__as_bf16x2(up_packed_538[26]));
                        float gate_x_752 = _cvt_f32_180.x;
                        float gate_y_753 = _cvt_f32_180.y;
                        float up_x_754 = _cvt_f32_181.x;
                        float up_y_755 = _cvt_f32_181.y;
                        float _exp_180 = expf(gate_x_752 * -1.0f);
                        float denominator_x_756 = _exp_180 + 1.0f;
                        float _exp_181 = expf(gate_y_753 * -1.0f);
                        float denominator_y_757 = _exp_181 + 1.0f;
                        float hidden_x_758 = gate_x_752 / denominator_x_756 * up_x_754;
                        float hidden_y_759 = gate_y_753 / denominator_y_757 * up_y_755;
                        __nv_bfloat162 _bf16x2_282 = __float22bfloat162_rn(make_float2(hidden_x_758, hidden_y_759));
                        hidden_packed_539[26] = __as_u32(_bf16x2_282);
                        float2 _cvt_f32_182 = __bfloat1622float2(__as_bf16x2(gate_packed_537[27]));
                        float2 _cvt_f32_183 = __bfloat1622float2(__as_bf16x2(up_packed_538[27]));
                        float gate_x_760 = _cvt_f32_182.x;
                        float gate_y_761 = _cvt_f32_182.y;
                        float up_x_762 = _cvt_f32_183.x;
                        float up_y_763 = _cvt_f32_183.y;
                        float _exp_182 = expf(gate_x_760 * -1.0f);
                        float denominator_x_764 = _exp_182 + 1.0f;
                        float _exp_183 = expf(gate_y_761 * -1.0f);
                        float denominator_y_765 = _exp_183 + 1.0f;
                        float hidden_x_766 = gate_x_760 / denominator_x_764 * up_x_762;
                        float hidden_y_767 = gate_y_761 / denominator_y_765 * up_y_763;
                        __nv_bfloat162 _bf16x2_283 = __float22bfloat162_rn(make_float2(hidden_x_766, hidden_y_767));
                        hidden_packed_539[27] = __as_u32(_bf16x2_283);
                        float2 _cvt_f32_184 = __bfloat1622float2(__as_bf16x2(gate_packed_537[28]));
                        float2 _cvt_f32_185 = __bfloat1622float2(__as_bf16x2(up_packed_538[28]));
                        float gate_x_768 = _cvt_f32_184.x;
                        float gate_y_769 = _cvt_f32_184.y;
                        float up_x_770 = _cvt_f32_185.x;
                        float up_y_771 = _cvt_f32_185.y;
                        float _exp_184 = expf(gate_x_768 * -1.0f);
                        float denominator_x_772 = _exp_184 + 1.0f;
                        float _exp_185 = expf(gate_y_769 * -1.0f);
                        float denominator_y_773 = _exp_185 + 1.0f;
                        float hidden_x_774 = gate_x_768 / denominator_x_772 * up_x_770;
                        float hidden_y_775 = gate_y_769 / denominator_y_773 * up_y_771;
                        __nv_bfloat162 _bf16x2_284 = __float22bfloat162_rn(make_float2(hidden_x_774, hidden_y_775));
                        hidden_packed_539[28] = __as_u32(_bf16x2_284);
                        float2 _cvt_f32_186 = __bfloat1622float2(__as_bf16x2(gate_packed_537[29]));
                        float2 _cvt_f32_187 = __bfloat1622float2(__as_bf16x2(up_packed_538[29]));
                        float gate_x_776 = _cvt_f32_186.x;
                        float gate_y_777 = _cvt_f32_186.y;
                        float up_x_778 = _cvt_f32_187.x;
                        float up_y_779 = _cvt_f32_187.y;
                        float _exp_186 = expf(gate_x_776 * -1.0f);
                        float denominator_x_780 = _exp_186 + 1.0f;
                        float _exp_187 = expf(gate_y_777 * -1.0f);
                        float denominator_y_781 = _exp_187 + 1.0f;
                        float hidden_x_782 = gate_x_776 / denominator_x_780 * up_x_778;
                        float hidden_y_783 = gate_y_777 / denominator_y_781 * up_y_779;
                        __nv_bfloat162 _bf16x2_285 = __float22bfloat162_rn(make_float2(hidden_x_782, hidden_y_783));
                        hidden_packed_539[29] = __as_u32(_bf16x2_285);
                        float2 _cvt_f32_188 = __bfloat1622float2(__as_bf16x2(gate_packed_537[30]));
                        float2 _cvt_f32_189 = __bfloat1622float2(__as_bf16x2(up_packed_538[30]));
                        float gate_x_784 = _cvt_f32_188.x;
                        float gate_y_785 = _cvt_f32_188.y;
                        float up_x_786 = _cvt_f32_189.x;
                        float up_y_787 = _cvt_f32_189.y;
                        float _exp_188 = expf(gate_x_784 * -1.0f);
                        float denominator_x_788 = _exp_188 + 1.0f;
                        float _exp_189 = expf(gate_y_785 * -1.0f);
                        float denominator_y_789 = _exp_189 + 1.0f;
                        float hidden_x_790 = gate_x_784 / denominator_x_788 * up_x_786;
                        float hidden_y_791 = gate_y_785 / denominator_y_789 * up_y_787;
                        __nv_bfloat162 _bf16x2_286 = __float22bfloat162_rn(make_float2(hidden_x_790, hidden_y_791));
                        hidden_packed_539[30] = __as_u32(_bf16x2_286);
                        float2 _cvt_f32_190 = __bfloat1622float2(__as_bf16x2(gate_packed_537[31]));
                        float2 _cvt_f32_191 = __bfloat1622float2(__as_bf16x2(up_packed_538[31]));
                        float gate_x_792 = _cvt_f32_190.x;
                        float gate_y_793 = _cvt_f32_190.y;
                        float up_x_794 = _cvt_f32_191.x;
                        float up_y_795 = _cvt_f32_191.y;
                        float _exp_190 = expf(gate_x_792 * -1.0f);
                        float denominator_x_796 = _exp_190 + 1.0f;
                        float _exp_191 = expf(gate_y_793 * -1.0f);
                        float denominator_y_797 = _exp_191 + 1.0f;
                        float hidden_x_798 = gate_x_792 / denominator_x_796 * up_x_794;
                        float hidden_y_799 = gate_y_793 / denominator_y_797 * up_y_795;
                        __nv_bfloat162 _bf16x2_287 = __float22bfloat162_rn(make_float2(hidden_x_798, hidden_y_799));
                        hidden_packed_539[31] = __as_u32(_bf16x2_287);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_800 = tid / 32;
                        int lane_801 = tid % 32;
                        #pragma unroll
                        for (int half_12 = 0; half_12 < 2; half_12++) {
                            #pragma unroll
                            for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                int row_16 = warp_800 * 32 + half_12 * 16 + lane_801 % 16;
                                int col_13 = col_tile_12 * 16 + lane_801 / 16 * 8;
                                unsigned int address_3_12 = d_smem_addr + (unsigned int)((row_16 * 32 + col_13) * 2);
                                address_3_12 = address_3_12 ^ (address_3_12 & 511) >> 7 << 4;
                                int offset_12 = half_12 * 8 + col_tile_12 * 4;
                                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(address_3_12);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_12 + 3]))
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
                            for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                int row_17 = warp_802 * 32 + half_13 * 16 + lane_803 % 16;
                                int col_14 = col_tile_13 * 16 + lane_803 / 16 * 8;
                                unsigned int address_3_13 = d_smem_addr + 8192 + (unsigned int)((row_17 * 32 + col_14) * 2);
                                address_3_13 = address_3_13 ^ (address_3_13 & 511) >> 7 << 4;
                                int offset_13 = half_13 * 8 + col_tile_13 * 4;
                                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(address_3_13);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_13 + 3]))
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
                            for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                int row_18 = warp_804 * 32 + half_14 * 16 + lane_805 % 16;
                                int col_15 = col_tile_14 * 16 + lane_805 / 16 * 8;
                                unsigned int address_3_14 = d_smem_addr + 16384 + (unsigned int)((row_18 * 32 + col_15) * 2);
                                address_3_14 = address_3_14 ^ (address_3_14 & 511) >> 7 << 4;
                                int offset_14 = half_14 * 8 + col_tile_14 * 4;
                                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(address_3_14);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_14 + 3]))
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
                            for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                int row_19 = warp_806 * 32 + half_15 * 16 + lane_807 % 16;
                                int col_16 = col_tile_15 * 16 + lane_807 / 16 * 8;
                                unsigned int address_3_15 = d_smem_addr + (unsigned int)((row_19 * 32 + col_16) * 2);
                                address_3_15 = address_3_15 ^ (address_3_15 & 511) >> 7 << 4;
                                int offset_15 = 16 + half_15 * 8 + col_tile_15 * 4;
                                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(address_3_15);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537[offset_15 + 3]))
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
                            for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                int row_20 = warp_808 * 32 + half_16 * 16 + lane_809 % 16;
                                int col_17 = col_tile_16 * 16 + lane_809 / 16 * 8;
                                unsigned int address_3_16 = d_smem_addr + 8192 + (unsigned int)((row_20 * 32 + col_17) * 2);
                                address_3_16 = address_3_16 ^ (address_3_16 & 511) >> 7 << 4;
                                int offset_16 = 16 + half_16 * 8 + col_tile_16 * 4;
                                uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(address_3_16);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538[offset_16 + 3]))
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
                            for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                int row_21 = warp_810 * 32 + half_17 * 16 + lane_811 % 16;
                                int col_18 = col_tile_17 * 16 + lane_811 / 16 * 8;
                                unsigned int address_3_17 = d_smem_addr + 16384 + (unsigned int)((row_21 * 32 + col_18) * 2);
                                address_3_17 = address_3_17 ^ (address_3_17 & 511) >> 7 << 4;
                                int offset_17 = 16 + half_17 * 8 + col_tile_17 * 4;
                                uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(address_3_17);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539[offset_17 + 3]))
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
                        __nv_bfloat162 _bf16x2_288 = __float22bfloat162_rn(make_float2(_tmem_load_24[0], _tmem_load_24[1]));
                        gate_packed_812[0] = __as_u32(_bf16x2_288);
                        __nv_bfloat162 _bf16x2_289 = __float22bfloat162_rn(make_float2(_tmem_load_25[0], _tmem_load_25[1]));
                        up_packed_813[0] = __as_u32(_bf16x2_289);
                        __nv_bfloat162 _bf16x2_290 = __float22bfloat162_rn(make_float2(_tmem_load_24[2], _tmem_load_24[3]));
                        gate_packed_812[1] = __as_u32(_bf16x2_290);
                        __nv_bfloat162 _bf16x2_291 = __float22bfloat162_rn(make_float2(_tmem_load_25[2], _tmem_load_25[3]));
                        up_packed_813[1] = __as_u32(_bf16x2_291);
                        __nv_bfloat162 _bf16x2_292 = __float22bfloat162_rn(make_float2(_tmem_load_24[4], _tmem_load_24[5]));
                        gate_packed_812[2] = __as_u32(_bf16x2_292);
                        __nv_bfloat162 _bf16x2_293 = __float22bfloat162_rn(make_float2(_tmem_load_25[4], _tmem_load_25[5]));
                        up_packed_813[2] = __as_u32(_bf16x2_293);
                        __nv_bfloat162 _bf16x2_294 = __float22bfloat162_rn(make_float2(_tmem_load_24[6], _tmem_load_24[7]));
                        gate_packed_812[3] = __as_u32(_bf16x2_294);
                        __nv_bfloat162 _bf16x2_295 = __float22bfloat162_rn(make_float2(_tmem_load_25[6], _tmem_load_25[7]));
                        up_packed_813[3] = __as_u32(_bf16x2_295);
                        __nv_bfloat162 _bf16x2_296 = __float22bfloat162_rn(make_float2(_tmem_load_24[8], _tmem_load_24[9]));
                        gate_packed_812[4] = __as_u32(_bf16x2_296);
                        __nv_bfloat162 _bf16x2_297 = __float22bfloat162_rn(make_float2(_tmem_load_25[8], _tmem_load_25[9]));
                        up_packed_813[4] = __as_u32(_bf16x2_297);
                        __nv_bfloat162 _bf16x2_298 = __float22bfloat162_rn(make_float2(_tmem_load_24[10], _tmem_load_24[11]));
                        gate_packed_812[5] = __as_u32(_bf16x2_298);
                        __nv_bfloat162 _bf16x2_299 = __float22bfloat162_rn(make_float2(_tmem_load_25[10], _tmem_load_25[11]));
                        up_packed_813[5] = __as_u32(_bf16x2_299);
                        __nv_bfloat162 _bf16x2_300 = __float22bfloat162_rn(make_float2(_tmem_load_24[12], _tmem_load_24[13]));
                        gate_packed_812[6] = __as_u32(_bf16x2_300);
                        __nv_bfloat162 _bf16x2_301 = __float22bfloat162_rn(make_float2(_tmem_load_25[12], _tmem_load_25[13]));
                        up_packed_813[6] = __as_u32(_bf16x2_301);
                        __nv_bfloat162 _bf16x2_302 = __float22bfloat162_rn(make_float2(_tmem_load_24[14], _tmem_load_24[15]));
                        gate_packed_812[7] = __as_u32(_bf16x2_302);
                        __nv_bfloat162 _bf16x2_303 = __float22bfloat162_rn(make_float2(_tmem_load_25[14], _tmem_load_25[15]));
                        up_packed_813[7] = __as_u32(_bf16x2_303);
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
                        __nv_bfloat162 _bf16x2_304 = __float22bfloat162_rn(make_float2(_tmem_load_26[0], _tmem_load_26[1]));
                        gate_packed_812[8] = __as_u32(_bf16x2_304);
                        __nv_bfloat162 _bf16x2_305 = __float22bfloat162_rn(make_float2(_tmem_load_27[0], _tmem_load_27[1]));
                        up_packed_813[8] = __as_u32(_bf16x2_305);
                        __nv_bfloat162 _bf16x2_306 = __float22bfloat162_rn(make_float2(_tmem_load_26[2], _tmem_load_26[3]));
                        gate_packed_812[9] = __as_u32(_bf16x2_306);
                        __nv_bfloat162 _bf16x2_307 = __float22bfloat162_rn(make_float2(_tmem_load_27[2], _tmem_load_27[3]));
                        up_packed_813[9] = __as_u32(_bf16x2_307);
                        __nv_bfloat162 _bf16x2_308 = __float22bfloat162_rn(make_float2(_tmem_load_26[4], _tmem_load_26[5]));
                        gate_packed_812[10] = __as_u32(_bf16x2_308);
                        __nv_bfloat162 _bf16x2_309 = __float22bfloat162_rn(make_float2(_tmem_load_27[4], _tmem_load_27[5]));
                        up_packed_813[10] = __as_u32(_bf16x2_309);
                        __nv_bfloat162 _bf16x2_310 = __float22bfloat162_rn(make_float2(_tmem_load_26[6], _tmem_load_26[7]));
                        gate_packed_812[11] = __as_u32(_bf16x2_310);
                        __nv_bfloat162 _bf16x2_311 = __float22bfloat162_rn(make_float2(_tmem_load_27[6], _tmem_load_27[7]));
                        up_packed_813[11] = __as_u32(_bf16x2_311);
                        __nv_bfloat162 _bf16x2_312 = __float22bfloat162_rn(make_float2(_tmem_load_26[8], _tmem_load_26[9]));
                        gate_packed_812[12] = __as_u32(_bf16x2_312);
                        __nv_bfloat162 _bf16x2_313 = __float22bfloat162_rn(make_float2(_tmem_load_27[8], _tmem_load_27[9]));
                        up_packed_813[12] = __as_u32(_bf16x2_313);
                        __nv_bfloat162 _bf16x2_314 = __float22bfloat162_rn(make_float2(_tmem_load_26[10], _tmem_load_26[11]));
                        gate_packed_812[13] = __as_u32(_bf16x2_314);
                        __nv_bfloat162 _bf16x2_315 = __float22bfloat162_rn(make_float2(_tmem_load_27[10], _tmem_load_27[11]));
                        up_packed_813[13] = __as_u32(_bf16x2_315);
                        __nv_bfloat162 _bf16x2_316 = __float22bfloat162_rn(make_float2(_tmem_load_26[12], _tmem_load_26[13]));
                        gate_packed_812[14] = __as_u32(_bf16x2_316);
                        __nv_bfloat162 _bf16x2_317 = __float22bfloat162_rn(make_float2(_tmem_load_27[12], _tmem_load_27[13]));
                        up_packed_813[14] = __as_u32(_bf16x2_317);
                        __nv_bfloat162 _bf16x2_318 = __float22bfloat162_rn(make_float2(_tmem_load_26[14], _tmem_load_26[15]));
                        gate_packed_812[15] = __as_u32(_bf16x2_318);
                        __nv_bfloat162 _bf16x2_319 = __float22bfloat162_rn(make_float2(_tmem_load_27[14], _tmem_load_27[15]));
                        up_packed_813[15] = __as_u32(_bf16x2_319);
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
                        __nv_bfloat162 _bf16x2_320 = __float22bfloat162_rn(make_float2(_tmem_load_28[0], _tmem_load_28[1]));
                        gate_packed_812[16] = __as_u32(_bf16x2_320);
                        __nv_bfloat162 _bf16x2_321 = __float22bfloat162_rn(make_float2(_tmem_load_29[0], _tmem_load_29[1]));
                        up_packed_813[16] = __as_u32(_bf16x2_321);
                        __nv_bfloat162 _bf16x2_322 = __float22bfloat162_rn(make_float2(_tmem_load_28[2], _tmem_load_28[3]));
                        gate_packed_812[17] = __as_u32(_bf16x2_322);
                        __nv_bfloat162 _bf16x2_323 = __float22bfloat162_rn(make_float2(_tmem_load_29[2], _tmem_load_29[3]));
                        up_packed_813[17] = __as_u32(_bf16x2_323);
                        __nv_bfloat162 _bf16x2_324 = __float22bfloat162_rn(make_float2(_tmem_load_28[4], _tmem_load_28[5]));
                        gate_packed_812[18] = __as_u32(_bf16x2_324);
                        __nv_bfloat162 _bf16x2_325 = __float22bfloat162_rn(make_float2(_tmem_load_29[4], _tmem_load_29[5]));
                        up_packed_813[18] = __as_u32(_bf16x2_325);
                        __nv_bfloat162 _bf16x2_326 = __float22bfloat162_rn(make_float2(_tmem_load_28[6], _tmem_load_28[7]));
                        gate_packed_812[19] = __as_u32(_bf16x2_326);
                        __nv_bfloat162 _bf16x2_327 = __float22bfloat162_rn(make_float2(_tmem_load_29[6], _tmem_load_29[7]));
                        up_packed_813[19] = __as_u32(_bf16x2_327);
                        __nv_bfloat162 _bf16x2_328 = __float22bfloat162_rn(make_float2(_tmem_load_28[8], _tmem_load_28[9]));
                        gate_packed_812[20] = __as_u32(_bf16x2_328);
                        __nv_bfloat162 _bf16x2_329 = __float22bfloat162_rn(make_float2(_tmem_load_29[8], _tmem_load_29[9]));
                        up_packed_813[20] = __as_u32(_bf16x2_329);
                        __nv_bfloat162 _bf16x2_330 = __float22bfloat162_rn(make_float2(_tmem_load_28[10], _tmem_load_28[11]));
                        gate_packed_812[21] = __as_u32(_bf16x2_330);
                        __nv_bfloat162 _bf16x2_331 = __float22bfloat162_rn(make_float2(_tmem_load_29[10], _tmem_load_29[11]));
                        up_packed_813[21] = __as_u32(_bf16x2_331);
                        __nv_bfloat162 _bf16x2_332 = __float22bfloat162_rn(make_float2(_tmem_load_28[12], _tmem_load_28[13]));
                        gate_packed_812[22] = __as_u32(_bf16x2_332);
                        __nv_bfloat162 _bf16x2_333 = __float22bfloat162_rn(make_float2(_tmem_load_29[12], _tmem_load_29[13]));
                        up_packed_813[22] = __as_u32(_bf16x2_333);
                        __nv_bfloat162 _bf16x2_334 = __float22bfloat162_rn(make_float2(_tmem_load_28[14], _tmem_load_28[15]));
                        gate_packed_812[23] = __as_u32(_bf16x2_334);
                        __nv_bfloat162 _bf16x2_335 = __float22bfloat162_rn(make_float2(_tmem_load_29[14], _tmem_load_29[15]));
                        up_packed_813[23] = __as_u32(_bf16x2_335);
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
                        __nv_bfloat162 _bf16x2_336 = __float22bfloat162_rn(make_float2(_tmem_load_30[0], _tmem_load_30[1]));
                        gate_packed_812[24] = __as_u32(_bf16x2_336);
                        __nv_bfloat162 _bf16x2_337 = __float22bfloat162_rn(make_float2(_tmem_load_31[0], _tmem_load_31[1]));
                        up_packed_813[24] = __as_u32(_bf16x2_337);
                        __nv_bfloat162 _bf16x2_338 = __float22bfloat162_rn(make_float2(_tmem_load_30[2], _tmem_load_30[3]));
                        gate_packed_812[25] = __as_u32(_bf16x2_338);
                        __nv_bfloat162 _bf16x2_339 = __float22bfloat162_rn(make_float2(_tmem_load_31[2], _tmem_load_31[3]));
                        up_packed_813[25] = __as_u32(_bf16x2_339);
                        __nv_bfloat162 _bf16x2_340 = __float22bfloat162_rn(make_float2(_tmem_load_30[4], _tmem_load_30[5]));
                        gate_packed_812[26] = __as_u32(_bf16x2_340);
                        __nv_bfloat162 _bf16x2_341 = __float22bfloat162_rn(make_float2(_tmem_load_31[4], _tmem_load_31[5]));
                        up_packed_813[26] = __as_u32(_bf16x2_341);
                        __nv_bfloat162 _bf16x2_342 = __float22bfloat162_rn(make_float2(_tmem_load_30[6], _tmem_load_30[7]));
                        gate_packed_812[27] = __as_u32(_bf16x2_342);
                        __nv_bfloat162 _bf16x2_343 = __float22bfloat162_rn(make_float2(_tmem_load_31[6], _tmem_load_31[7]));
                        up_packed_813[27] = __as_u32(_bf16x2_343);
                        __nv_bfloat162 _bf16x2_344 = __float22bfloat162_rn(make_float2(_tmem_load_30[8], _tmem_load_30[9]));
                        gate_packed_812[28] = __as_u32(_bf16x2_344);
                        __nv_bfloat162 _bf16x2_345 = __float22bfloat162_rn(make_float2(_tmem_load_31[8], _tmem_load_31[9]));
                        up_packed_813[28] = __as_u32(_bf16x2_345);
                        __nv_bfloat162 _bf16x2_346 = __float22bfloat162_rn(make_float2(_tmem_load_30[10], _tmem_load_30[11]));
                        gate_packed_812[29] = __as_u32(_bf16x2_346);
                        __nv_bfloat162 _bf16x2_347 = __float22bfloat162_rn(make_float2(_tmem_load_31[10], _tmem_load_31[11]));
                        up_packed_813[29] = __as_u32(_bf16x2_347);
                        __nv_bfloat162 _bf16x2_348 = __float22bfloat162_rn(make_float2(_tmem_load_30[12], _tmem_load_30[13]));
                        gate_packed_812[30] = __as_u32(_bf16x2_348);
                        __nv_bfloat162 _bf16x2_349 = __float22bfloat162_rn(make_float2(_tmem_load_31[12], _tmem_load_31[13]));
                        up_packed_813[30] = __as_u32(_bf16x2_349);
                        __nv_bfloat162 _bf16x2_350 = __float22bfloat162_rn(make_float2(_tmem_load_30[14], _tmem_load_30[15]));
                        gate_packed_812[31] = __as_u32(_bf16x2_350);
                        __nv_bfloat162 _bf16x2_351 = __float22bfloat162_rn(make_float2(_tmem_load_31[14], _tmem_load_31[15]));
                        up_packed_813[31] = __as_u32(_bf16x2_351);
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
                        float _exp_192 = expf(gate_x_819 * -1.0f);
                        float denominator_x_823 = _exp_192 + 1.0f;
                        float _exp_193 = expf(gate_y_820 * -1.0f);
                        float denominator_y_824 = _exp_193 + 1.0f;
                        float hidden_x_825 = gate_x_819 / denominator_x_823 * up_x_821;
                        float hidden_y_826 = gate_y_820 / denominator_y_824 * up_y_822;
                        __nv_bfloat162 _bf16x2_352 = __float22bfloat162_rn(make_float2(hidden_x_825, hidden_y_826));
                        hidden_packed_814[0] = __as_u32(_bf16x2_352);
                        float2 _cvt_f32_194 = __bfloat1622float2(__as_bf16x2(gate_packed_812[1]));
                        float2 _cvt_f32_195 = __bfloat1622float2(__as_bf16x2(up_packed_813[1]));
                        float gate_x_827 = _cvt_f32_194.x;
                        float gate_y_828 = _cvt_f32_194.y;
                        float up_x_829 = _cvt_f32_195.x;
                        float up_y_830 = _cvt_f32_195.y;
                        float _exp_194 = expf(gate_x_827 * -1.0f);
                        float denominator_x_831 = _exp_194 + 1.0f;
                        float _exp_195 = expf(gate_y_828 * -1.0f);
                        float denominator_y_832 = _exp_195 + 1.0f;
                        float hidden_x_833 = gate_x_827 / denominator_x_831 * up_x_829;
                        float hidden_y_834 = gate_y_828 / denominator_y_832 * up_y_830;
                        __nv_bfloat162 _bf16x2_353 = __float22bfloat162_rn(make_float2(hidden_x_833, hidden_y_834));
                        hidden_packed_814[1] = __as_u32(_bf16x2_353);
                        float2 _cvt_f32_196 = __bfloat1622float2(__as_bf16x2(gate_packed_812[2]));
                        float2 _cvt_f32_197 = __bfloat1622float2(__as_bf16x2(up_packed_813[2]));
                        float gate_x_835 = _cvt_f32_196.x;
                        float gate_y_836 = _cvt_f32_196.y;
                        float up_x_837 = _cvt_f32_197.x;
                        float up_y_838 = _cvt_f32_197.y;
                        float _exp_196 = expf(gate_x_835 * -1.0f);
                        float denominator_x_839 = _exp_196 + 1.0f;
                        float _exp_197 = expf(gate_y_836 * -1.0f);
                        float denominator_y_840 = _exp_197 + 1.0f;
                        float hidden_x_841 = gate_x_835 / denominator_x_839 * up_x_837;
                        float hidden_y_842 = gate_y_836 / denominator_y_840 * up_y_838;
                        __nv_bfloat162 _bf16x2_354 = __float22bfloat162_rn(make_float2(hidden_x_841, hidden_y_842));
                        hidden_packed_814[2] = __as_u32(_bf16x2_354);
                        float2 _cvt_f32_198 = __bfloat1622float2(__as_bf16x2(gate_packed_812[3]));
                        float2 _cvt_f32_199 = __bfloat1622float2(__as_bf16x2(up_packed_813[3]));
                        float gate_x_843 = _cvt_f32_198.x;
                        float gate_y_844 = _cvt_f32_198.y;
                        float up_x_845 = _cvt_f32_199.x;
                        float up_y_846 = _cvt_f32_199.y;
                        float _exp_198 = expf(gate_x_843 * -1.0f);
                        float denominator_x_847 = _exp_198 + 1.0f;
                        float _exp_199 = expf(gate_y_844 * -1.0f);
                        float denominator_y_848 = _exp_199 + 1.0f;
                        float hidden_x_849 = gate_x_843 / denominator_x_847 * up_x_845;
                        float hidden_y_850 = gate_y_844 / denominator_y_848 * up_y_846;
                        __nv_bfloat162 _bf16x2_355 = __float22bfloat162_rn(make_float2(hidden_x_849, hidden_y_850));
                        hidden_packed_814[3] = __as_u32(_bf16x2_355);
                        float2 _cvt_f32_200 = __bfloat1622float2(__as_bf16x2(gate_packed_812[4]));
                        float2 _cvt_f32_201 = __bfloat1622float2(__as_bf16x2(up_packed_813[4]));
                        float gate_x_851 = _cvt_f32_200.x;
                        float gate_y_852 = _cvt_f32_200.y;
                        float up_x_853 = _cvt_f32_201.x;
                        float up_y_854 = _cvt_f32_201.y;
                        float _exp_200 = expf(gate_x_851 * -1.0f);
                        float denominator_x_855 = _exp_200 + 1.0f;
                        float _exp_201 = expf(gate_y_852 * -1.0f);
                        float denominator_y_856 = _exp_201 + 1.0f;
                        float hidden_x_857 = gate_x_851 / denominator_x_855 * up_x_853;
                        float hidden_y_858 = gate_y_852 / denominator_y_856 * up_y_854;
                        __nv_bfloat162 _bf16x2_356 = __float22bfloat162_rn(make_float2(hidden_x_857, hidden_y_858));
                        hidden_packed_814[4] = __as_u32(_bf16x2_356);
                        float2 _cvt_f32_202 = __bfloat1622float2(__as_bf16x2(gate_packed_812[5]));
                        float2 _cvt_f32_203 = __bfloat1622float2(__as_bf16x2(up_packed_813[5]));
                        float gate_x_859 = _cvt_f32_202.x;
                        float gate_y_860 = _cvt_f32_202.y;
                        float up_x_861 = _cvt_f32_203.x;
                        float up_y_862 = _cvt_f32_203.y;
                        float _exp_202 = expf(gate_x_859 * -1.0f);
                        float denominator_x_863 = _exp_202 + 1.0f;
                        float _exp_203 = expf(gate_y_860 * -1.0f);
                        float denominator_y_864 = _exp_203 + 1.0f;
                        float hidden_x_865 = gate_x_859 / denominator_x_863 * up_x_861;
                        float hidden_y_866 = gate_y_860 / denominator_y_864 * up_y_862;
                        __nv_bfloat162 _bf16x2_357 = __float22bfloat162_rn(make_float2(hidden_x_865, hidden_y_866));
                        hidden_packed_814[5] = __as_u32(_bf16x2_357);
                        float2 _cvt_f32_204 = __bfloat1622float2(__as_bf16x2(gate_packed_812[6]));
                        float2 _cvt_f32_205 = __bfloat1622float2(__as_bf16x2(up_packed_813[6]));
                        float gate_x_867 = _cvt_f32_204.x;
                        float gate_y_868 = _cvt_f32_204.y;
                        float up_x_869 = _cvt_f32_205.x;
                        float up_y_870 = _cvt_f32_205.y;
                        float _exp_204 = expf(gate_x_867 * -1.0f);
                        float denominator_x_871 = _exp_204 + 1.0f;
                        float _exp_205 = expf(gate_y_868 * -1.0f);
                        float denominator_y_872 = _exp_205 + 1.0f;
                        float hidden_x_873 = gate_x_867 / denominator_x_871 * up_x_869;
                        float hidden_y_874 = gate_y_868 / denominator_y_872 * up_y_870;
                        __nv_bfloat162 _bf16x2_358 = __float22bfloat162_rn(make_float2(hidden_x_873, hidden_y_874));
                        hidden_packed_814[6] = __as_u32(_bf16x2_358);
                        float2 _cvt_f32_206 = __bfloat1622float2(__as_bf16x2(gate_packed_812[7]));
                        float2 _cvt_f32_207 = __bfloat1622float2(__as_bf16x2(up_packed_813[7]));
                        float gate_x_875 = _cvt_f32_206.x;
                        float gate_y_876 = _cvt_f32_206.y;
                        float up_x_877 = _cvt_f32_207.x;
                        float up_y_878 = _cvt_f32_207.y;
                        float _exp_206 = expf(gate_x_875 * -1.0f);
                        float denominator_x_879 = _exp_206 + 1.0f;
                        float _exp_207 = expf(gate_y_876 * -1.0f);
                        float denominator_y_880 = _exp_207 + 1.0f;
                        float hidden_x_881 = gate_x_875 / denominator_x_879 * up_x_877;
                        float hidden_y_882 = gate_y_876 / denominator_y_880 * up_y_878;
                        __nv_bfloat162 _bf16x2_359 = __float22bfloat162_rn(make_float2(hidden_x_881, hidden_y_882));
                        hidden_packed_814[7] = __as_u32(_bf16x2_359);
                        float2 _cvt_f32_208 = __bfloat1622float2(__as_bf16x2(gate_packed_812[8]));
                        float2 _cvt_f32_209 = __bfloat1622float2(__as_bf16x2(up_packed_813[8]));
                        float gate_x_883 = _cvt_f32_208.x;
                        float gate_y_884 = _cvt_f32_208.y;
                        float up_x_885 = _cvt_f32_209.x;
                        float up_y_886 = _cvt_f32_209.y;
                        float _exp_208 = expf(gate_x_883 * -1.0f);
                        float denominator_x_887 = _exp_208 + 1.0f;
                        float _exp_209 = expf(gate_y_884 * -1.0f);
                        float denominator_y_888 = _exp_209 + 1.0f;
                        float hidden_x_889 = gate_x_883 / denominator_x_887 * up_x_885;
                        float hidden_y_890 = gate_y_884 / denominator_y_888 * up_y_886;
                        __nv_bfloat162 _bf16x2_360 = __float22bfloat162_rn(make_float2(hidden_x_889, hidden_y_890));
                        hidden_packed_814[8] = __as_u32(_bf16x2_360);
                        float2 _cvt_f32_210 = __bfloat1622float2(__as_bf16x2(gate_packed_812[9]));
                        float2 _cvt_f32_211 = __bfloat1622float2(__as_bf16x2(up_packed_813[9]));
                        float gate_x_891 = _cvt_f32_210.x;
                        float gate_y_892 = _cvt_f32_210.y;
                        float up_x_893 = _cvt_f32_211.x;
                        float up_y_894 = _cvt_f32_211.y;
                        float _exp_210 = expf(gate_x_891 * -1.0f);
                        float denominator_x_895 = _exp_210 + 1.0f;
                        float _exp_211 = expf(gate_y_892 * -1.0f);
                        float denominator_y_896 = _exp_211 + 1.0f;
                        float hidden_x_897 = gate_x_891 / denominator_x_895 * up_x_893;
                        float hidden_y_898 = gate_y_892 / denominator_y_896 * up_y_894;
                        __nv_bfloat162 _bf16x2_361 = __float22bfloat162_rn(make_float2(hidden_x_897, hidden_y_898));
                        hidden_packed_814[9] = __as_u32(_bf16x2_361);
                        float2 _cvt_f32_212 = __bfloat1622float2(__as_bf16x2(gate_packed_812[10]));
                        float2 _cvt_f32_213 = __bfloat1622float2(__as_bf16x2(up_packed_813[10]));
                        float gate_x_899 = _cvt_f32_212.x;
                        float gate_y_900 = _cvt_f32_212.y;
                        float up_x_901 = _cvt_f32_213.x;
                        float up_y_902 = _cvt_f32_213.y;
                        float _exp_212 = expf(gate_x_899 * -1.0f);
                        float denominator_x_903 = _exp_212 + 1.0f;
                        float _exp_213 = expf(gate_y_900 * -1.0f);
                        float denominator_y_904 = _exp_213 + 1.0f;
                        float hidden_x_905 = gate_x_899 / denominator_x_903 * up_x_901;
                        float hidden_y_906 = gate_y_900 / denominator_y_904 * up_y_902;
                        __nv_bfloat162 _bf16x2_362 = __float22bfloat162_rn(make_float2(hidden_x_905, hidden_y_906));
                        hidden_packed_814[10] = __as_u32(_bf16x2_362);
                        float2 _cvt_f32_214 = __bfloat1622float2(__as_bf16x2(gate_packed_812[11]));
                        float2 _cvt_f32_215 = __bfloat1622float2(__as_bf16x2(up_packed_813[11]));
                        float gate_x_907 = _cvt_f32_214.x;
                        float gate_y_908 = _cvt_f32_214.y;
                        float up_x_909 = _cvt_f32_215.x;
                        float up_y_910 = _cvt_f32_215.y;
                        float _exp_214 = expf(gate_x_907 * -1.0f);
                        float denominator_x_911 = _exp_214 + 1.0f;
                        float _exp_215 = expf(gate_y_908 * -1.0f);
                        float denominator_y_912 = _exp_215 + 1.0f;
                        float hidden_x_913 = gate_x_907 / denominator_x_911 * up_x_909;
                        float hidden_y_914 = gate_y_908 / denominator_y_912 * up_y_910;
                        __nv_bfloat162 _bf16x2_363 = __float22bfloat162_rn(make_float2(hidden_x_913, hidden_y_914));
                        hidden_packed_814[11] = __as_u32(_bf16x2_363);
                        float2 _cvt_f32_216 = __bfloat1622float2(__as_bf16x2(gate_packed_812[12]));
                        float2 _cvt_f32_217 = __bfloat1622float2(__as_bf16x2(up_packed_813[12]));
                        float gate_x_915 = _cvt_f32_216.x;
                        float gate_y_916 = _cvt_f32_216.y;
                        float up_x_917 = _cvt_f32_217.x;
                        float up_y_918 = _cvt_f32_217.y;
                        float _exp_216 = expf(gate_x_915 * -1.0f);
                        float denominator_x_919 = _exp_216 + 1.0f;
                        float _exp_217 = expf(gate_y_916 * -1.0f);
                        float denominator_y_920 = _exp_217 + 1.0f;
                        float hidden_x_921 = gate_x_915 / denominator_x_919 * up_x_917;
                        float hidden_y_922 = gate_y_916 / denominator_y_920 * up_y_918;
                        __nv_bfloat162 _bf16x2_364 = __float22bfloat162_rn(make_float2(hidden_x_921, hidden_y_922));
                        hidden_packed_814[12] = __as_u32(_bf16x2_364);
                        float2 _cvt_f32_218 = __bfloat1622float2(__as_bf16x2(gate_packed_812[13]));
                        float2 _cvt_f32_219 = __bfloat1622float2(__as_bf16x2(up_packed_813[13]));
                        float gate_x_923 = _cvt_f32_218.x;
                        float gate_y_924 = _cvt_f32_218.y;
                        float up_x_925 = _cvt_f32_219.x;
                        float up_y_926 = _cvt_f32_219.y;
                        float _exp_218 = expf(gate_x_923 * -1.0f);
                        float denominator_x_927 = _exp_218 + 1.0f;
                        float _exp_219 = expf(gate_y_924 * -1.0f);
                        float denominator_y_928 = _exp_219 + 1.0f;
                        float hidden_x_929 = gate_x_923 / denominator_x_927 * up_x_925;
                        float hidden_y_930 = gate_y_924 / denominator_y_928 * up_y_926;
                        __nv_bfloat162 _bf16x2_365 = __float22bfloat162_rn(make_float2(hidden_x_929, hidden_y_930));
                        hidden_packed_814[13] = __as_u32(_bf16x2_365);
                        float2 _cvt_f32_220 = __bfloat1622float2(__as_bf16x2(gate_packed_812[14]));
                        float2 _cvt_f32_221 = __bfloat1622float2(__as_bf16x2(up_packed_813[14]));
                        float gate_x_931 = _cvt_f32_220.x;
                        float gate_y_932 = _cvt_f32_220.y;
                        float up_x_933 = _cvt_f32_221.x;
                        float up_y_934 = _cvt_f32_221.y;
                        float _exp_220 = expf(gate_x_931 * -1.0f);
                        float denominator_x_935 = _exp_220 + 1.0f;
                        float _exp_221 = expf(gate_y_932 * -1.0f);
                        float denominator_y_936 = _exp_221 + 1.0f;
                        float hidden_x_937 = gate_x_931 / denominator_x_935 * up_x_933;
                        float hidden_y_938 = gate_y_932 / denominator_y_936 * up_y_934;
                        __nv_bfloat162 _bf16x2_366 = __float22bfloat162_rn(make_float2(hidden_x_937, hidden_y_938));
                        hidden_packed_814[14] = __as_u32(_bf16x2_366);
                        float2 _cvt_f32_222 = __bfloat1622float2(__as_bf16x2(gate_packed_812[15]));
                        float2 _cvt_f32_223 = __bfloat1622float2(__as_bf16x2(up_packed_813[15]));
                        float gate_x_939 = _cvt_f32_222.x;
                        float gate_y_940 = _cvt_f32_222.y;
                        float up_x_941 = _cvt_f32_223.x;
                        float up_y_942 = _cvt_f32_223.y;
                        float _exp_222 = expf(gate_x_939 * -1.0f);
                        float denominator_x_943 = _exp_222 + 1.0f;
                        float _exp_223 = expf(gate_y_940 * -1.0f);
                        float denominator_y_944 = _exp_223 + 1.0f;
                        float hidden_x_945 = gate_x_939 / denominator_x_943 * up_x_941;
                        float hidden_y_946 = gate_y_940 / denominator_y_944 * up_y_942;
                        __nv_bfloat162 _bf16x2_367 = __float22bfloat162_rn(make_float2(hidden_x_945, hidden_y_946));
                        hidden_packed_814[15] = __as_u32(_bf16x2_367);
                        float2 _cvt_f32_224 = __bfloat1622float2(__as_bf16x2(gate_packed_812[16]));
                        float2 _cvt_f32_225 = __bfloat1622float2(__as_bf16x2(up_packed_813[16]));
                        float gate_x_947 = _cvt_f32_224.x;
                        float gate_y_948 = _cvt_f32_224.y;
                        float up_x_949 = _cvt_f32_225.x;
                        float up_y_950 = _cvt_f32_225.y;
                        float _exp_224 = expf(gate_x_947 * -1.0f);
                        float denominator_x_951 = _exp_224 + 1.0f;
                        float _exp_225 = expf(gate_y_948 * -1.0f);
                        float denominator_y_952 = _exp_225 + 1.0f;
                        float hidden_x_953 = gate_x_947 / denominator_x_951 * up_x_949;
                        float hidden_y_954 = gate_y_948 / denominator_y_952 * up_y_950;
                        __nv_bfloat162 _bf16x2_368 = __float22bfloat162_rn(make_float2(hidden_x_953, hidden_y_954));
                        hidden_packed_814[16] = __as_u32(_bf16x2_368);
                        float2 _cvt_f32_226 = __bfloat1622float2(__as_bf16x2(gate_packed_812[17]));
                        float2 _cvt_f32_227 = __bfloat1622float2(__as_bf16x2(up_packed_813[17]));
                        float gate_x_955 = _cvt_f32_226.x;
                        float gate_y_956 = _cvt_f32_226.y;
                        float up_x_957 = _cvt_f32_227.x;
                        float up_y_958 = _cvt_f32_227.y;
                        float _exp_226 = expf(gate_x_955 * -1.0f);
                        float denominator_x_959 = _exp_226 + 1.0f;
                        float _exp_227 = expf(gate_y_956 * -1.0f);
                        float denominator_y_960 = _exp_227 + 1.0f;
                        float hidden_x_961 = gate_x_955 / denominator_x_959 * up_x_957;
                        float hidden_y_962 = gate_y_956 / denominator_y_960 * up_y_958;
                        __nv_bfloat162 _bf16x2_369 = __float22bfloat162_rn(make_float2(hidden_x_961, hidden_y_962));
                        hidden_packed_814[17] = __as_u32(_bf16x2_369);
                        float2 _cvt_f32_228 = __bfloat1622float2(__as_bf16x2(gate_packed_812[18]));
                        float2 _cvt_f32_229 = __bfloat1622float2(__as_bf16x2(up_packed_813[18]));
                        float gate_x_963 = _cvt_f32_228.x;
                        float gate_y_964 = _cvt_f32_228.y;
                        float up_x_965 = _cvt_f32_229.x;
                        float up_y_966 = _cvt_f32_229.y;
                        float _exp_228 = expf(gate_x_963 * -1.0f);
                        float denominator_x_967 = _exp_228 + 1.0f;
                        float _exp_229 = expf(gate_y_964 * -1.0f);
                        float denominator_y_968 = _exp_229 + 1.0f;
                        float hidden_x_969 = gate_x_963 / denominator_x_967 * up_x_965;
                        float hidden_y_970 = gate_y_964 / denominator_y_968 * up_y_966;
                        __nv_bfloat162 _bf16x2_370 = __float22bfloat162_rn(make_float2(hidden_x_969, hidden_y_970));
                        hidden_packed_814[18] = __as_u32(_bf16x2_370);
                        float2 _cvt_f32_230 = __bfloat1622float2(__as_bf16x2(gate_packed_812[19]));
                        float2 _cvt_f32_231 = __bfloat1622float2(__as_bf16x2(up_packed_813[19]));
                        float gate_x_971 = _cvt_f32_230.x;
                        float gate_y_972 = _cvt_f32_230.y;
                        float up_x_973 = _cvt_f32_231.x;
                        float up_y_974 = _cvt_f32_231.y;
                        float _exp_230 = expf(gate_x_971 * -1.0f);
                        float denominator_x_975 = _exp_230 + 1.0f;
                        float _exp_231 = expf(gate_y_972 * -1.0f);
                        float denominator_y_976 = _exp_231 + 1.0f;
                        float hidden_x_977 = gate_x_971 / denominator_x_975 * up_x_973;
                        float hidden_y_978 = gate_y_972 / denominator_y_976 * up_y_974;
                        __nv_bfloat162 _bf16x2_371 = __float22bfloat162_rn(make_float2(hidden_x_977, hidden_y_978));
                        hidden_packed_814[19] = __as_u32(_bf16x2_371);
                        float2 _cvt_f32_232 = __bfloat1622float2(__as_bf16x2(gate_packed_812[20]));
                        float2 _cvt_f32_233 = __bfloat1622float2(__as_bf16x2(up_packed_813[20]));
                        float gate_x_979 = _cvt_f32_232.x;
                        float gate_y_980 = _cvt_f32_232.y;
                        float up_x_981 = _cvt_f32_233.x;
                        float up_y_982 = _cvt_f32_233.y;
                        float _exp_232 = expf(gate_x_979 * -1.0f);
                        float denominator_x_983 = _exp_232 + 1.0f;
                        float _exp_233 = expf(gate_y_980 * -1.0f);
                        float denominator_y_984 = _exp_233 + 1.0f;
                        float hidden_x_985 = gate_x_979 / denominator_x_983 * up_x_981;
                        float hidden_y_986 = gate_y_980 / denominator_y_984 * up_y_982;
                        __nv_bfloat162 _bf16x2_372 = __float22bfloat162_rn(make_float2(hidden_x_985, hidden_y_986));
                        hidden_packed_814[20] = __as_u32(_bf16x2_372);
                        float2 _cvt_f32_234 = __bfloat1622float2(__as_bf16x2(gate_packed_812[21]));
                        float2 _cvt_f32_235 = __bfloat1622float2(__as_bf16x2(up_packed_813[21]));
                        float gate_x_987 = _cvt_f32_234.x;
                        float gate_y_988 = _cvt_f32_234.y;
                        float up_x_989 = _cvt_f32_235.x;
                        float up_y_990 = _cvt_f32_235.y;
                        float _exp_234 = expf(gate_x_987 * -1.0f);
                        float denominator_x_991 = _exp_234 + 1.0f;
                        float _exp_235 = expf(gate_y_988 * -1.0f);
                        float denominator_y_992 = _exp_235 + 1.0f;
                        float hidden_x_993 = gate_x_987 / denominator_x_991 * up_x_989;
                        float hidden_y_994 = gate_y_988 / denominator_y_992 * up_y_990;
                        __nv_bfloat162 _bf16x2_373 = __float22bfloat162_rn(make_float2(hidden_x_993, hidden_y_994));
                        hidden_packed_814[21] = __as_u32(_bf16x2_373);
                        float2 _cvt_f32_236 = __bfloat1622float2(__as_bf16x2(gate_packed_812[22]));
                        float2 _cvt_f32_237 = __bfloat1622float2(__as_bf16x2(up_packed_813[22]));
                        float gate_x_995 = _cvt_f32_236.x;
                        float gate_y_996 = _cvt_f32_236.y;
                        float up_x_997 = _cvt_f32_237.x;
                        float up_y_998 = _cvt_f32_237.y;
                        float _exp_236 = expf(gate_x_995 * -1.0f);
                        float denominator_x_999 = _exp_236 + 1.0f;
                        float _exp_237 = expf(gate_y_996 * -1.0f);
                        float denominator_y_1000 = _exp_237 + 1.0f;
                        float hidden_x_1001 = gate_x_995 / denominator_x_999 * up_x_997;
                        float hidden_y_1002 = gate_y_996 / denominator_y_1000 * up_y_998;
                        __nv_bfloat162 _bf16x2_374 = __float22bfloat162_rn(make_float2(hidden_x_1001, hidden_y_1002));
                        hidden_packed_814[22] = __as_u32(_bf16x2_374);
                        float2 _cvt_f32_238 = __bfloat1622float2(__as_bf16x2(gate_packed_812[23]));
                        float2 _cvt_f32_239 = __bfloat1622float2(__as_bf16x2(up_packed_813[23]));
                        float gate_x_1003 = _cvt_f32_238.x;
                        float gate_y_1004 = _cvt_f32_238.y;
                        float up_x_1005 = _cvt_f32_239.x;
                        float up_y_1006 = _cvt_f32_239.y;
                        float _exp_238 = expf(gate_x_1003 * -1.0f);
                        float denominator_x_1007 = _exp_238 + 1.0f;
                        float _exp_239 = expf(gate_y_1004 * -1.0f);
                        float denominator_y_1008 = _exp_239 + 1.0f;
                        float hidden_x_1009 = gate_x_1003 / denominator_x_1007 * up_x_1005;
                        float hidden_y_1010 = gate_y_1004 / denominator_y_1008 * up_y_1006;
                        __nv_bfloat162 _bf16x2_375 = __float22bfloat162_rn(make_float2(hidden_x_1009, hidden_y_1010));
                        hidden_packed_814[23] = __as_u32(_bf16x2_375);
                        float2 _cvt_f32_240 = __bfloat1622float2(__as_bf16x2(gate_packed_812[24]));
                        float2 _cvt_f32_241 = __bfloat1622float2(__as_bf16x2(up_packed_813[24]));
                        float gate_x_1011 = _cvt_f32_240.x;
                        float gate_y_1012 = _cvt_f32_240.y;
                        float up_x_1013 = _cvt_f32_241.x;
                        float up_y_1014 = _cvt_f32_241.y;
                        float _exp_240 = expf(gate_x_1011 * -1.0f);
                        float denominator_x_1015 = _exp_240 + 1.0f;
                        float _exp_241 = expf(gate_y_1012 * -1.0f);
                        float denominator_y_1016 = _exp_241 + 1.0f;
                        float hidden_x_1017 = gate_x_1011 / denominator_x_1015 * up_x_1013;
                        float hidden_y_1018 = gate_y_1012 / denominator_y_1016 * up_y_1014;
                        __nv_bfloat162 _bf16x2_376 = __float22bfloat162_rn(make_float2(hidden_x_1017, hidden_y_1018));
                        hidden_packed_814[24] = __as_u32(_bf16x2_376);
                        float2 _cvt_f32_242 = __bfloat1622float2(__as_bf16x2(gate_packed_812[25]));
                        float2 _cvt_f32_243 = __bfloat1622float2(__as_bf16x2(up_packed_813[25]));
                        float gate_x_1019 = _cvt_f32_242.x;
                        float gate_y_1020 = _cvt_f32_242.y;
                        float up_x_1021 = _cvt_f32_243.x;
                        float up_y_1022 = _cvt_f32_243.y;
                        float _exp_242 = expf(gate_x_1019 * -1.0f);
                        float denominator_x_1023 = _exp_242 + 1.0f;
                        float _exp_243 = expf(gate_y_1020 * -1.0f);
                        float denominator_y_1024 = _exp_243 + 1.0f;
                        float hidden_x_1025 = gate_x_1019 / denominator_x_1023 * up_x_1021;
                        float hidden_y_1026 = gate_y_1020 / denominator_y_1024 * up_y_1022;
                        __nv_bfloat162 _bf16x2_377 = __float22bfloat162_rn(make_float2(hidden_x_1025, hidden_y_1026));
                        hidden_packed_814[25] = __as_u32(_bf16x2_377);
                        float2 _cvt_f32_244 = __bfloat1622float2(__as_bf16x2(gate_packed_812[26]));
                        float2 _cvt_f32_245 = __bfloat1622float2(__as_bf16x2(up_packed_813[26]));
                        float gate_x_1027 = _cvt_f32_244.x;
                        float gate_y_1028 = _cvt_f32_244.y;
                        float up_x_1029 = _cvt_f32_245.x;
                        float up_y_1030 = _cvt_f32_245.y;
                        float _exp_244 = expf(gate_x_1027 * -1.0f);
                        float denominator_x_1031 = _exp_244 + 1.0f;
                        float _exp_245 = expf(gate_y_1028 * -1.0f);
                        float denominator_y_1032 = _exp_245 + 1.0f;
                        float hidden_x_1033 = gate_x_1027 / denominator_x_1031 * up_x_1029;
                        float hidden_y_1034 = gate_y_1028 / denominator_y_1032 * up_y_1030;
                        __nv_bfloat162 _bf16x2_378 = __float22bfloat162_rn(make_float2(hidden_x_1033, hidden_y_1034));
                        hidden_packed_814[26] = __as_u32(_bf16x2_378);
                        float2 _cvt_f32_246 = __bfloat1622float2(__as_bf16x2(gate_packed_812[27]));
                        float2 _cvt_f32_247 = __bfloat1622float2(__as_bf16x2(up_packed_813[27]));
                        float gate_x_1035 = _cvt_f32_246.x;
                        float gate_y_1036 = _cvt_f32_246.y;
                        float up_x_1037 = _cvt_f32_247.x;
                        float up_y_1038 = _cvt_f32_247.y;
                        float _exp_246 = expf(gate_x_1035 * -1.0f);
                        float denominator_x_1039 = _exp_246 + 1.0f;
                        float _exp_247 = expf(gate_y_1036 * -1.0f);
                        float denominator_y_1040 = _exp_247 + 1.0f;
                        float hidden_x_1041 = gate_x_1035 / denominator_x_1039 * up_x_1037;
                        float hidden_y_1042 = gate_y_1036 / denominator_y_1040 * up_y_1038;
                        __nv_bfloat162 _bf16x2_379 = __float22bfloat162_rn(make_float2(hidden_x_1041, hidden_y_1042));
                        hidden_packed_814[27] = __as_u32(_bf16x2_379);
                        float2 _cvt_f32_248 = __bfloat1622float2(__as_bf16x2(gate_packed_812[28]));
                        float2 _cvt_f32_249 = __bfloat1622float2(__as_bf16x2(up_packed_813[28]));
                        float gate_x_1043 = _cvt_f32_248.x;
                        float gate_y_1044 = _cvt_f32_248.y;
                        float up_x_1045 = _cvt_f32_249.x;
                        float up_y_1046 = _cvt_f32_249.y;
                        float _exp_248 = expf(gate_x_1043 * -1.0f);
                        float denominator_x_1047 = _exp_248 + 1.0f;
                        float _exp_249 = expf(gate_y_1044 * -1.0f);
                        float denominator_y_1048 = _exp_249 + 1.0f;
                        float hidden_x_1049 = gate_x_1043 / denominator_x_1047 * up_x_1045;
                        float hidden_y_1050 = gate_y_1044 / denominator_y_1048 * up_y_1046;
                        __nv_bfloat162 _bf16x2_380 = __float22bfloat162_rn(make_float2(hidden_x_1049, hidden_y_1050));
                        hidden_packed_814[28] = __as_u32(_bf16x2_380);
                        float2 _cvt_f32_250 = __bfloat1622float2(__as_bf16x2(gate_packed_812[29]));
                        float2 _cvt_f32_251 = __bfloat1622float2(__as_bf16x2(up_packed_813[29]));
                        float gate_x_1051 = _cvt_f32_250.x;
                        float gate_y_1052 = _cvt_f32_250.y;
                        float up_x_1053 = _cvt_f32_251.x;
                        float up_y_1054 = _cvt_f32_251.y;
                        float _exp_250 = expf(gate_x_1051 * -1.0f);
                        float denominator_x_1055 = _exp_250 + 1.0f;
                        float _exp_251 = expf(gate_y_1052 * -1.0f);
                        float denominator_y_1056 = _exp_251 + 1.0f;
                        float hidden_x_1057 = gate_x_1051 / denominator_x_1055 * up_x_1053;
                        float hidden_y_1058 = gate_y_1052 / denominator_y_1056 * up_y_1054;
                        __nv_bfloat162 _bf16x2_381 = __float22bfloat162_rn(make_float2(hidden_x_1057, hidden_y_1058));
                        hidden_packed_814[29] = __as_u32(_bf16x2_381);
                        float2 _cvt_f32_252 = __bfloat1622float2(__as_bf16x2(gate_packed_812[30]));
                        float2 _cvt_f32_253 = __bfloat1622float2(__as_bf16x2(up_packed_813[30]));
                        float gate_x_1059 = _cvt_f32_252.x;
                        float gate_y_1060 = _cvt_f32_252.y;
                        float up_x_1061 = _cvt_f32_253.x;
                        float up_y_1062 = _cvt_f32_253.y;
                        float _exp_252 = expf(gate_x_1059 * -1.0f);
                        float denominator_x_1063 = _exp_252 + 1.0f;
                        float _exp_253 = expf(gate_y_1060 * -1.0f);
                        float denominator_y_1064 = _exp_253 + 1.0f;
                        float hidden_x_1065 = gate_x_1059 / denominator_x_1063 * up_x_1061;
                        float hidden_y_1066 = gate_y_1060 / denominator_y_1064 * up_y_1062;
                        __nv_bfloat162 _bf16x2_382 = __float22bfloat162_rn(make_float2(hidden_x_1065, hidden_y_1066));
                        hidden_packed_814[30] = __as_u32(_bf16x2_382);
                        float2 _cvt_f32_254 = __bfloat1622float2(__as_bf16x2(gate_packed_812[31]));
                        float2 _cvt_f32_255 = __bfloat1622float2(__as_bf16x2(up_packed_813[31]));
                        float gate_x_1067 = _cvt_f32_254.x;
                        float gate_y_1068 = _cvt_f32_254.y;
                        float up_x_1069 = _cvt_f32_255.x;
                        float up_y_1070 = _cvt_f32_255.y;
                        float _exp_254 = expf(gate_x_1067 * -1.0f);
                        float denominator_x_1071 = _exp_254 + 1.0f;
                        float _exp_255 = expf(gate_y_1068 * -1.0f);
                        float denominator_y_1072 = _exp_255 + 1.0f;
                        float hidden_x_1073 = gate_x_1067 / denominator_x_1071 * up_x_1069;
                        float hidden_y_1074 = gate_y_1068 / denominator_y_1072 * up_y_1070;
                        __nv_bfloat162 _bf16x2_383 = __float22bfloat162_rn(make_float2(hidden_x_1073, hidden_y_1074));
                        hidden_packed_814[31] = __as_u32(_bf16x2_383);
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1075 = tid / 32;
                        int lane_1076 = tid % 32;
                        #pragma unroll
                        for (int half_18 = 0; half_18 < 2; half_18++) {
                            #pragma unroll
                            for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                int row_22 = warp_1075 * 32 + half_18 * 16 + lane_1076 % 16;
                                int col_19 = col_tile_18 * 16 + lane_1076 / 16 * 8;
                                unsigned int address_3_18 = d_smem_addr + (unsigned int)((row_22 * 32 + col_19) * 2);
                                address_3_18 = address_3_18 ^ (address_3_18 & 511) >> 7 << 4;
                                int offset_18 = half_18 * 8 + col_tile_18 * 4;
                                uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(address_3_18);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_18 + 3]))
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
                            for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                int row_23 = warp_1077 * 32 + half_19 * 16 + lane_1078 % 16;
                                int col_20 = col_tile_19 * 16 + lane_1078 / 16 * 8;
                                unsigned int address_3_19 = d_smem_addr + 8192 + (unsigned int)((row_23 * 32 + col_20) * 2);
                                address_3_19 = address_3_19 ^ (address_3_19 & 511) >> 7 << 4;
                                int offset_19 = half_19 * 8 + col_tile_19 * 4;
                                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(address_3_19);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_19 + 3]))
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
                            for (int col_tile_20 = 0; col_tile_20 < 2; col_tile_20++) {
                                int row_24 = warp_1079 * 32 + half_20 * 16 + lane_1080 % 16;
                                int col_21 = col_tile_20 * 16 + lane_1080 / 16 * 8;
                                unsigned int address_3_20 = d_smem_addr + 16384 + (unsigned int)((row_24 * 32 + col_21) * 2);
                                address_3_20 = address_3_20 ^ (address_3_20 & 511) >> 7 << 4;
                                int offset_20 = half_20 * 8 + col_tile_20 * 4;
                                uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(address_3_20);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_20 + 3]))
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
                            for (int col_tile_21 = 0; col_tile_21 < 2; col_tile_21++) {
                                int row_25 = warp_1081 * 32 + half_21 * 16 + lane_1082 % 16;
                                int col_22 = col_tile_21 * 16 + lane_1082 / 16 * 8;
                                unsigned int address_3_21 = d_smem_addr + (unsigned int)((row_25 * 32 + col_22) * 2);
                                address_3_21 = address_3_21 ^ (address_3_21 & 511) >> 7 << 4;
                                int offset_21 = 16 + half_21 * 8 + col_tile_21 * 4;
                                uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(address_3_21);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812[offset_21 + 3]))
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
                            for (int col_tile_22 = 0; col_tile_22 < 2; col_tile_22++) {
                                int row_26 = warp_1083 * 32 + half_22 * 16 + lane_1084 % 16;
                                int col_23 = col_tile_22 * 16 + lane_1084 / 16 * 8;
                                unsigned int address_3_22 = d_smem_addr + 8192 + (unsigned int)((row_26 * 32 + col_23) * 2);
                                address_3_22 = address_3_22 ^ (address_3_22 & 511) >> 7 << 4;
                                int offset_22 = 16 + half_22 * 8 + col_tile_22 * 4;
                                uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(address_3_22);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813[offset_22 + 3]))
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
                            for (int col_tile_23 = 0; col_tile_23 < 2; col_tile_23++) {
                                int row_27 = warp_1085 * 32 + half_23 * 16 + lane_1086 % 16;
                                int col_24 = col_tile_23 * 16 + lane_1086 / 16 * 8;
                                unsigned int address_3_23 = d_smem_addr + 16384 + (unsigned int)((row_27 * 32 + col_24) * 2);
                                address_3_23 = address_3_23 ^ (address_3_23 & 511) >> 7 << 4;
                                int offset_23 = 16 + half_23 * 8 + col_tile_23 * 4;
                                uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(address_3_23);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814[offset_23 + 3]))
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
                int col_blocks_2 = (hidden + 512 - 1) / 512;
                int x_1 = -1;
                int y_1 = -1;
                int expert_1 = -1;
                int k_start_1 = 0;
                int k_end_1 = 0;
                int first_1 = 0;
                int row_blocks_1 = local_tokens / 256;
                if (compute - shared_fused < row_blocks_1 * col_blocks_2) {
                    int supergroup_1 = (compute - shared_fused) / (row_blocks_1 * 8);
                    int full_cols_1 = col_blocks_2 / 8 * 8;
                    int row_28 = 0;
                    int col_25 = 0;
                    if (compute - shared_fused < row_blocks_1 * full_cols_1) {
                        row_28 = (compute - shared_fused) % (row_blocks_1 * 8) / 8;
                        col_25 = supergroup_1 * 8 + (compute - shared_fused) % 8;
                    } else {
                        row_28 = (compute - shared_fused - row_blocks_1 * full_cols_1) / (col_blocks_2 - full_cols_1);
                        col_25 = full_cols_1 + (compute - shared_fused - row_blocks_1 * full_cols_1) % (col_blocks_2 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_28 = row_blocks_1 - row_28 - 1;
                    }
                    x_1 = row_28;
                    y_1 = col_25;
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
                                bool enabled_value_3 = 1;
                                if (enabled_value_3 != 0) {
                                    int32_t _relaxed_ld_8;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_8) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                    int value_4 = _relaxed_ld_8;
                                    while (value_4 < 2 * (intermediate / 128)) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int32_t _relaxed_ld_9;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_9) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                        value_4 = _relaxed_ld_9;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                int _min_30 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                int _max_3 = ((0) > (_min_30) ? (0) : (_min_30));
                                int mini_rows_5 = _max_3;
                                int required_3 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
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
                                    __nv_bfloat162 _bf16x2_384 = __float22bfloat162_rn(make_float2(_tmem_load_32[pair * 2], _tmem_load_32[pair * 2 + 1]));
                                    packed[chunk * 16 + sub * 8 + pair] = __as_u32(_bf16x2_384);
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
                            int _min_31 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            if (output_row < _min_31) {
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
                                for (int col_tile_24 = 0; col_tile_24 < 2; col_tile_24++) {
                                    int row_29 = warp_0 * 32 + half_24 * 16 + lane_2 % 16;
                                    int col_26 = col_tile_24 * 16 + lane_2 / 16 * 8;
                                    unsigned int address_5 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_29 * 32 + col_26) * 2);
                                    address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                    int offset_24 = chunk_1 * 16 + half_24 * 8 + col_tile_24 * 4;
                                    uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(address_5);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_24 + 3]))
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
                                        __nv_bfloat162 _bf16x2_385 = __float22bfloat162_rn(make_float2(_tmem_load_33[pair_1 * 2], _tmem_load_33[pair_1 * 2 + 1]));
                                        packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(_bf16x2_385);
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
                                    for (int col_tile_25 = 0; col_tile_25 < 2; col_tile_25++) {
                                        int row_30 = warp_0_1 * 32 + half_25 * 16 + lane_3 % 16;
                                        int col_27 = col_tile_25 * 16 + lane_3 / 16 * 8;
                                        unsigned int address_7 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_30 * 32 + col_27) * 2);
                                        address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                        int offset_25 = chunk_3 * 16 + half_25 * 8 + col_tile_25 * 4;
                                        uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(address_7);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_25 + 3]))
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
                int task_1 = (compute - shared_tasks) % mini_tasks;
                int macro_1 = macros - 1;
                int mini_3 = ordered_mini;
                if (ordered_mini >= last_minis) {
                    macro_1 = macros - 2 - (ordered_mini - last_minis) / minis_per_macro;
                    mini_3 = (ordered_mini - last_minis) % minis_per_macro;
                }
                if (task_1 < mini_fused) {
                    int col_blocks_3 = (intermediate + 256 - 1) / 256;
                    int x_2 = -1;
                    int y_2 = -1;
                    int expert_2 = -1;
                    int k_start_2 = 0;
                    int k_end_2 = 0;
                    int first_2 = 0;
                    int first_block = (macro_1 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                    int _min_32 = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    int end_block = _min_32;
                    int block_5 = first_block + task_1 / col_blocks_3;
                    if (block_5 < end_block) {
                        int index = counts[3 * experts + block_5];
                        int offset_26 = counts[experts + index] / 256;
                        int _max_4 = ((first_block) > (offset_26) ? (first_block) : (offset_26));
                        int first_row_7 = _max_4;
                        int _min_33 = ((end_block) < (offset_26 + counts[index] / 256) ? (end_block) : (offset_26 + counts[index] / 256));
                        int rows_6 = _min_33 - first_row_7;
                        int supergroup_2 = (task_1 - (first_row_7 - first_block) * col_blocks_3) / (rows_6 * 8);
                        int full_cols_2 = col_blocks_3 / 8 * 8;
                        int row_31 = 0;
                        int col_28 = 0;
                        if (task_1 - (first_row_7 - first_block) * col_blocks_3 < rows_6 * full_cols_2) {
                            row_31 = (task_1 - (first_row_7 - first_block) * col_blocks_3) % (rows_6 * 8) / 8;
                            col_28 = supergroup_2 * 8 + (task_1 - (first_row_7 - first_block) * col_blocks_3) % 8;
                        } else {
                            row_31 = (task_1 - (first_row_7 - first_block) * col_blocks_3 - rows_6 * full_cols_2) / (col_blocks_3 - full_cols_2);
                            col_28 = full_cols_2 + (task_1 - (first_row_7 - first_block) * col_blocks_3 - rows_6 * full_cols_2) % (col_blocks_3 - full_cols_2);
                        }
                        if ((supergroup_2 & 1) != 0) {
                            row_31 = rows_6 - row_31 - 1;
                        }
                        x_2 = first_row_7 + row_31 - macro_1 * (macro_size / 256);
                        y_2 = col_28;
                        expert_2 = index;
                    }
                    unsigned int phase_bits_4 = gemm_bits;
                    int has_hi_2 = 0;
                    has_hi_2 = (int)((y_2 * 2 + 1) * 256 < intermediate);
                    int global_mini_2 = macro_1 * (macro_size / mini_size) + mini_3;
                    int macro_rows_3 = macro_1 * (macro_size / 256);
                    int iterations_2 = hidden / 64;
                    if (expert_2 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    int _min_34 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                    int _max_5 = ((0) > (_min_34) ? (0) : (_min_34));
                                    int mini_rows_6 = _max_5;
                                    int required_4 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_4 = 1;
                                    if (enabled_value_4 != 0) {
                                        int32_t _relaxed_ld_10;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_10) : "l"(x_ready + global_mini_2) : "memory");
                                        int value_5 = _relaxed_ld_10;
                                        while (value_5 < required_4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_11;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_11) : "l"(x_ready + global_mini_2) : "memory");
                                            value_5 = _relaxed_ld_11;
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
                                        :: "r"(a_smem_addr + (unsigned int)(ring_4 * 16384)), "l"((&x_routed)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)(ring_4 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)(ring_4 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y_2 * 256 + cta_rank_0 * 128), "r"(idx_4), "r"(expert_2), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << 16 + ring_4);
                                    ring_4 = (ring_4 + 1) % 4;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_5 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_4 >> 22 & 1);
                                    phase_bits_4 = phase_bits_4 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_5 = 0; idx_5 < iterations_2; idx_5++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_5) * 8, 98304);
                                        mbarrier_wait(gemm_arrived_addr + (ring_5) * 8, phase_bits_4 >> (unsigned int)ring_5 & 1);
                                        int _mma_a_lo_4 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_5) * 1024;
                                        int _mma_b_lo_4 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_5) * 1024;
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
            :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_accumulator), "r"(((idx_5 == 0) ? 0 : 1)));
                                        int _mma_a_lo_5 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_5) * 1024;
                                        int _mma_b_lo_5 = (((b_hi_addr) >> 4) & 0x3FFF) + (ring_5) * 1024;
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
            :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)));
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_5) * 8, (uint16_t)(3));
                                        phase_bits_4 = phase_bits_4 ^ (unsigned int)(1 << ring_5);
                                        ring_5 = (ring_5 + 1) % 4;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_4 >> 6 & 1);
                            phase_bits_4 = phase_bits_4 ^ 64;
                            unsigned int gate_packed_1[32];
                            unsigned int up_packed_1[32];
                            unsigned int hidden_packed_1[32];
                            unsigned int address_8 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16);
                            float _tmem_load_34[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_34[15]))
                                : "r"(address_8));
                            float _tmem_load_35[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_35[15]))
                                : "r"(address_8 + 256));
                            __nv_bfloat162 _bf16x2_386 = __float22bfloat162_rn(make_float2(_tmem_load_34[0], _tmem_load_34[1]));
                            gate_packed_1[0] = __as_u32(_bf16x2_386);
                            __nv_bfloat162 _bf16x2_387 = __float22bfloat162_rn(make_float2(_tmem_load_35[0], _tmem_load_35[1]));
                            up_packed_1[0] = __as_u32(_bf16x2_387);
                            __nv_bfloat162 _bf16x2_388 = __float22bfloat162_rn(make_float2(_tmem_load_34[2], _tmem_load_34[3]));
                            gate_packed_1[1] = __as_u32(_bf16x2_388);
                            __nv_bfloat162 _bf16x2_389 = __float22bfloat162_rn(make_float2(_tmem_load_35[2], _tmem_load_35[3]));
                            up_packed_1[1] = __as_u32(_bf16x2_389);
                            __nv_bfloat162 _bf16x2_390 = __float22bfloat162_rn(make_float2(_tmem_load_34[4], _tmem_load_34[5]));
                            gate_packed_1[2] = __as_u32(_bf16x2_390);
                            __nv_bfloat162 _bf16x2_391 = __float22bfloat162_rn(make_float2(_tmem_load_35[4], _tmem_load_35[5]));
                            up_packed_1[2] = __as_u32(_bf16x2_391);
                            __nv_bfloat162 _bf16x2_392 = __float22bfloat162_rn(make_float2(_tmem_load_34[6], _tmem_load_34[7]));
                            gate_packed_1[3] = __as_u32(_bf16x2_392);
                            __nv_bfloat162 _bf16x2_393 = __float22bfloat162_rn(make_float2(_tmem_load_35[6], _tmem_load_35[7]));
                            up_packed_1[3] = __as_u32(_bf16x2_393);
                            __nv_bfloat162 _bf16x2_394 = __float22bfloat162_rn(make_float2(_tmem_load_34[8], _tmem_load_34[9]));
                            gate_packed_1[4] = __as_u32(_bf16x2_394);
                            __nv_bfloat162 _bf16x2_395 = __float22bfloat162_rn(make_float2(_tmem_load_35[8], _tmem_load_35[9]));
                            up_packed_1[4] = __as_u32(_bf16x2_395);
                            __nv_bfloat162 _bf16x2_396 = __float22bfloat162_rn(make_float2(_tmem_load_34[10], _tmem_load_34[11]));
                            gate_packed_1[5] = __as_u32(_bf16x2_396);
                            __nv_bfloat162 _bf16x2_397 = __float22bfloat162_rn(make_float2(_tmem_load_35[10], _tmem_load_35[11]));
                            up_packed_1[5] = __as_u32(_bf16x2_397);
                            __nv_bfloat162 _bf16x2_398 = __float22bfloat162_rn(make_float2(_tmem_load_34[12], _tmem_load_34[13]));
                            gate_packed_1[6] = __as_u32(_bf16x2_398);
                            __nv_bfloat162 _bf16x2_399 = __float22bfloat162_rn(make_float2(_tmem_load_35[12], _tmem_load_35[13]));
                            up_packed_1[6] = __as_u32(_bf16x2_399);
                            __nv_bfloat162 _bf16x2_400 = __float22bfloat162_rn(make_float2(_tmem_load_34[14], _tmem_load_34[15]));
                            gate_packed_1[7] = __as_u32(_bf16x2_400);
                            __nv_bfloat162 _bf16x2_401 = __float22bfloat162_rn(make_float2(_tmem_load_35[14], _tmem_load_35[15]));
                            up_packed_1[7] = __as_u32(_bf16x2_401);
                            unsigned int address_0_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16);
                            float _tmem_load_36[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[15]))
                                : "r"(address_0_1));
                            float _tmem_load_37[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_37[15]))
                                : "r"(address_0_1 + 256));
                            __nv_bfloat162 _bf16x2_402 = __float22bfloat162_rn(make_float2(_tmem_load_36[0], _tmem_load_36[1]));
                            gate_packed_1[8] = __as_u32(_bf16x2_402);
                            __nv_bfloat162 _bf16x2_403 = __float22bfloat162_rn(make_float2(_tmem_load_37[0], _tmem_load_37[1]));
                            up_packed_1[8] = __as_u32(_bf16x2_403);
                            __nv_bfloat162 _bf16x2_404 = __float22bfloat162_rn(make_float2(_tmem_load_36[2], _tmem_load_36[3]));
                            gate_packed_1[9] = __as_u32(_bf16x2_404);
                            __nv_bfloat162 _bf16x2_405 = __float22bfloat162_rn(make_float2(_tmem_load_37[2], _tmem_load_37[3]));
                            up_packed_1[9] = __as_u32(_bf16x2_405);
                            __nv_bfloat162 _bf16x2_406 = __float22bfloat162_rn(make_float2(_tmem_load_36[4], _tmem_load_36[5]));
                            gate_packed_1[10] = __as_u32(_bf16x2_406);
                            __nv_bfloat162 _bf16x2_407 = __float22bfloat162_rn(make_float2(_tmem_load_37[4], _tmem_load_37[5]));
                            up_packed_1[10] = __as_u32(_bf16x2_407);
                            __nv_bfloat162 _bf16x2_408 = __float22bfloat162_rn(make_float2(_tmem_load_36[6], _tmem_load_36[7]));
                            gate_packed_1[11] = __as_u32(_bf16x2_408);
                            __nv_bfloat162 _bf16x2_409 = __float22bfloat162_rn(make_float2(_tmem_load_37[6], _tmem_load_37[7]));
                            up_packed_1[11] = __as_u32(_bf16x2_409);
                            __nv_bfloat162 _bf16x2_410 = __float22bfloat162_rn(make_float2(_tmem_load_36[8], _tmem_load_36[9]));
                            gate_packed_1[12] = __as_u32(_bf16x2_410);
                            __nv_bfloat162 _bf16x2_411 = __float22bfloat162_rn(make_float2(_tmem_load_37[8], _tmem_load_37[9]));
                            up_packed_1[12] = __as_u32(_bf16x2_411);
                            __nv_bfloat162 _bf16x2_412 = __float22bfloat162_rn(make_float2(_tmem_load_36[10], _tmem_load_36[11]));
                            gate_packed_1[13] = __as_u32(_bf16x2_412);
                            __nv_bfloat162 _bf16x2_413 = __float22bfloat162_rn(make_float2(_tmem_load_37[10], _tmem_load_37[11]));
                            up_packed_1[13] = __as_u32(_bf16x2_413);
                            __nv_bfloat162 _bf16x2_414 = __float22bfloat162_rn(make_float2(_tmem_load_36[12], _tmem_load_36[13]));
                            gate_packed_1[14] = __as_u32(_bf16x2_414);
                            __nv_bfloat162 _bf16x2_415 = __float22bfloat162_rn(make_float2(_tmem_load_37[12], _tmem_load_37[13]));
                            up_packed_1[14] = __as_u32(_bf16x2_415);
                            __nv_bfloat162 _bf16x2_416 = __float22bfloat162_rn(make_float2(_tmem_load_36[14], _tmem_load_36[15]));
                            gate_packed_1[15] = __as_u32(_bf16x2_416);
                            __nv_bfloat162 _bf16x2_417 = __float22bfloat162_rn(make_float2(_tmem_load_37[14], _tmem_load_37[15]));
                            up_packed_1[15] = __as_u32(_bf16x2_417);
                            unsigned int address_1_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 32;
                            float _tmem_load_38[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_38[15]))
                                : "r"(address_1_1));
                            float _tmem_load_39[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_39[15]))
                                : "r"(address_1_1 + 256));
                            __nv_bfloat162 _bf16x2_418 = __float22bfloat162_rn(make_float2(_tmem_load_38[0], _tmem_load_38[1]));
                            gate_packed_1[16] = __as_u32(_bf16x2_418);
                            __nv_bfloat162 _bf16x2_419 = __float22bfloat162_rn(make_float2(_tmem_load_39[0], _tmem_load_39[1]));
                            up_packed_1[16] = __as_u32(_bf16x2_419);
                            __nv_bfloat162 _bf16x2_420 = __float22bfloat162_rn(make_float2(_tmem_load_38[2], _tmem_load_38[3]));
                            gate_packed_1[17] = __as_u32(_bf16x2_420);
                            __nv_bfloat162 _bf16x2_421 = __float22bfloat162_rn(make_float2(_tmem_load_39[2], _tmem_load_39[3]));
                            up_packed_1[17] = __as_u32(_bf16x2_421);
                            __nv_bfloat162 _bf16x2_422 = __float22bfloat162_rn(make_float2(_tmem_load_38[4], _tmem_load_38[5]));
                            gate_packed_1[18] = __as_u32(_bf16x2_422);
                            __nv_bfloat162 _bf16x2_423 = __float22bfloat162_rn(make_float2(_tmem_load_39[4], _tmem_load_39[5]));
                            up_packed_1[18] = __as_u32(_bf16x2_423);
                            __nv_bfloat162 _bf16x2_424 = __float22bfloat162_rn(make_float2(_tmem_load_38[6], _tmem_load_38[7]));
                            gate_packed_1[19] = __as_u32(_bf16x2_424);
                            __nv_bfloat162 _bf16x2_425 = __float22bfloat162_rn(make_float2(_tmem_load_39[6], _tmem_load_39[7]));
                            up_packed_1[19] = __as_u32(_bf16x2_425);
                            __nv_bfloat162 _bf16x2_426 = __float22bfloat162_rn(make_float2(_tmem_load_38[8], _tmem_load_38[9]));
                            gate_packed_1[20] = __as_u32(_bf16x2_426);
                            __nv_bfloat162 _bf16x2_427 = __float22bfloat162_rn(make_float2(_tmem_load_39[8], _tmem_load_39[9]));
                            up_packed_1[20] = __as_u32(_bf16x2_427);
                            __nv_bfloat162 _bf16x2_428 = __float22bfloat162_rn(make_float2(_tmem_load_38[10], _tmem_load_38[11]));
                            gate_packed_1[21] = __as_u32(_bf16x2_428);
                            __nv_bfloat162 _bf16x2_429 = __float22bfloat162_rn(make_float2(_tmem_load_39[10], _tmem_load_39[11]));
                            up_packed_1[21] = __as_u32(_bf16x2_429);
                            __nv_bfloat162 _bf16x2_430 = __float22bfloat162_rn(make_float2(_tmem_load_38[12], _tmem_load_38[13]));
                            gate_packed_1[22] = __as_u32(_bf16x2_430);
                            __nv_bfloat162 _bf16x2_431 = __float22bfloat162_rn(make_float2(_tmem_load_39[12], _tmem_load_39[13]));
                            up_packed_1[22] = __as_u32(_bf16x2_431);
                            __nv_bfloat162 _bf16x2_432 = __float22bfloat162_rn(make_float2(_tmem_load_38[14], _tmem_load_38[15]));
                            gate_packed_1[23] = __as_u32(_bf16x2_432);
                            __nv_bfloat162 _bf16x2_433 = __float22bfloat162_rn(make_float2(_tmem_load_39[14], _tmem_load_39[15]));
                            up_packed_1[23] = __as_u32(_bf16x2_433);
                            unsigned int address_2_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 32;
                            float _tmem_load_40[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_40[15]))
                                : "r"(address_2_1));
                            float _tmem_load_41[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[15]))
                                : "r"(address_2_1 + 256));
                            __nv_bfloat162 _bf16x2_434 = __float22bfloat162_rn(make_float2(_tmem_load_40[0], _tmem_load_40[1]));
                            gate_packed_1[24] = __as_u32(_bf16x2_434);
                            __nv_bfloat162 _bf16x2_435 = __float22bfloat162_rn(make_float2(_tmem_load_41[0], _tmem_load_41[1]));
                            up_packed_1[24] = __as_u32(_bf16x2_435);
                            __nv_bfloat162 _bf16x2_436 = __float22bfloat162_rn(make_float2(_tmem_load_40[2], _tmem_load_40[3]));
                            gate_packed_1[25] = __as_u32(_bf16x2_436);
                            __nv_bfloat162 _bf16x2_437 = __float22bfloat162_rn(make_float2(_tmem_load_41[2], _tmem_load_41[3]));
                            up_packed_1[25] = __as_u32(_bf16x2_437);
                            __nv_bfloat162 _bf16x2_438 = __float22bfloat162_rn(make_float2(_tmem_load_40[4], _tmem_load_40[5]));
                            gate_packed_1[26] = __as_u32(_bf16x2_438);
                            __nv_bfloat162 _bf16x2_439 = __float22bfloat162_rn(make_float2(_tmem_load_41[4], _tmem_load_41[5]));
                            up_packed_1[26] = __as_u32(_bf16x2_439);
                            __nv_bfloat162 _bf16x2_440 = __float22bfloat162_rn(make_float2(_tmem_load_40[6], _tmem_load_40[7]));
                            gate_packed_1[27] = __as_u32(_bf16x2_440);
                            __nv_bfloat162 _bf16x2_441 = __float22bfloat162_rn(make_float2(_tmem_load_41[6], _tmem_load_41[7]));
                            up_packed_1[27] = __as_u32(_bf16x2_441);
                            __nv_bfloat162 _bf16x2_442 = __float22bfloat162_rn(make_float2(_tmem_load_40[8], _tmem_load_40[9]));
                            gate_packed_1[28] = __as_u32(_bf16x2_442);
                            __nv_bfloat162 _bf16x2_443 = __float22bfloat162_rn(make_float2(_tmem_load_41[8], _tmem_load_41[9]));
                            up_packed_1[28] = __as_u32(_bf16x2_443);
                            __nv_bfloat162 _bf16x2_444 = __float22bfloat162_rn(make_float2(_tmem_load_40[10], _tmem_load_40[11]));
                            gate_packed_1[29] = __as_u32(_bf16x2_444);
                            __nv_bfloat162 _bf16x2_445 = __float22bfloat162_rn(make_float2(_tmem_load_41[10], _tmem_load_41[11]));
                            up_packed_1[29] = __as_u32(_bf16x2_445);
                            __nv_bfloat162 _bf16x2_446 = __float22bfloat162_rn(make_float2(_tmem_load_40[12], _tmem_load_40[13]));
                            gate_packed_1[30] = __as_u32(_bf16x2_446);
                            __nv_bfloat162 _bf16x2_447 = __float22bfloat162_rn(make_float2(_tmem_load_41[12], _tmem_load_41[13]));
                            up_packed_1[30] = __as_u32(_bf16x2_447);
                            __nv_bfloat162 _bf16x2_448 = __float22bfloat162_rn(make_float2(_tmem_load_40[14], _tmem_load_40[15]));
                            gate_packed_1[31] = __as_u32(_bf16x2_448);
                            __nv_bfloat162 _bf16x2_449 = __float22bfloat162_rn(make_float2(_tmem_load_41[14], _tmem_load_41[15]));
                            up_packed_1[31] = __as_u32(_bf16x2_449);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _cvt_f32_256 = __bfloat1622float2(__as_bf16x2(gate_packed_1[0]));
                            float2 _cvt_f32_257 = __bfloat1622float2(__as_bf16x2(up_packed_1[0]));
                            float gate_x_1 = _cvt_f32_256.x;
                            float gate_y_1 = _cvt_f32_256.y;
                            float up_x_1 = _cvt_f32_257.x;
                            float up_y_1 = _cvt_f32_257.y;
                            float _exp_256 = expf(gate_x_1 * -1.0f);
                            float denominator_x_1 = _exp_256 + 1.0f;
                            float _exp_257 = expf(gate_y_1 * -1.0f);
                            float denominator_y_1 = _exp_257 + 1.0f;
                            float hidden_x_1 = gate_x_1 / denominator_x_1 * up_x_1;
                            float hidden_y_1 = gate_y_1 / denominator_y_1 * up_y_1;
                            __nv_bfloat162 _bf16x2_450 = __float22bfloat162_rn(make_float2(hidden_x_1, hidden_y_1));
                            hidden_packed_1[0] = __as_u32(_bf16x2_450);
                            float2 _cvt_f32_258 = __bfloat1622float2(__as_bf16x2(gate_packed_1[1]));
                            float2 _cvt_f32_259 = __bfloat1622float2(__as_bf16x2(up_packed_1[1]));
                            float gate_x_3_1 = _cvt_f32_258.x;
                            float gate_y_4_1 = _cvt_f32_258.y;
                            float up_x_5_1 = _cvt_f32_259.x;
                            float up_y_6_1 = _cvt_f32_259.y;
                            float _exp_258 = expf(gate_x_3_1 * -1.0f);
                            float denominator_x_7_1 = _exp_258 + 1.0f;
                            float _exp_259 = expf(gate_y_4_1 * -1.0f);
                            float denominator_y_8_1 = _exp_259 + 1.0f;
                            float hidden_x_9_1 = gate_x_3_1 / denominator_x_7_1 * up_x_5_1;
                            float hidden_y_10_1 = gate_y_4_1 / denominator_y_8_1 * up_y_6_1;
                            __nv_bfloat162 _bf16x2_451 = __float22bfloat162_rn(make_float2(hidden_x_9_1, hidden_y_10_1));
                            hidden_packed_1[1] = __as_u32(_bf16x2_451);
                            float2 _cvt_f32_260 = __bfloat1622float2(__as_bf16x2(gate_packed_1[2]));
                            float2 _cvt_f32_261 = __bfloat1622float2(__as_bf16x2(up_packed_1[2]));
                            float gate_x_11_1 = _cvt_f32_260.x;
                            float gate_y_12_1 = _cvt_f32_260.y;
                            float up_x_13_1 = _cvt_f32_261.x;
                            float up_y_14_1 = _cvt_f32_261.y;
                            float _exp_260 = expf(gate_x_11_1 * -1.0f);
                            float denominator_x_15_1 = _exp_260 + 1.0f;
                            float _exp_261 = expf(gate_y_12_1 * -1.0f);
                            float denominator_y_16_1 = _exp_261 + 1.0f;
                            float hidden_x_17_1 = gate_x_11_1 / denominator_x_15_1 * up_x_13_1;
                            float hidden_y_18_1 = gate_y_12_1 / denominator_y_16_1 * up_y_14_1;
                            __nv_bfloat162 _bf16x2_452 = __float22bfloat162_rn(make_float2(hidden_x_17_1, hidden_y_18_1));
                            hidden_packed_1[2] = __as_u32(_bf16x2_452);
                            float2 _cvt_f32_262 = __bfloat1622float2(__as_bf16x2(gate_packed_1[3]));
                            float2 _cvt_f32_263 = __bfloat1622float2(__as_bf16x2(up_packed_1[3]));
                            float gate_x_19_1 = _cvt_f32_262.x;
                            float gate_y_20_1 = _cvt_f32_262.y;
                            float up_x_21_1 = _cvt_f32_263.x;
                            float up_y_22_1 = _cvt_f32_263.y;
                            float _exp_262 = expf(gate_x_19_1 * -1.0f);
                            float denominator_x_23_1 = _exp_262 + 1.0f;
                            float _exp_263 = expf(gate_y_20_1 * -1.0f);
                            float denominator_y_24_1 = _exp_263 + 1.0f;
                            float hidden_x_25_1 = gate_x_19_1 / denominator_x_23_1 * up_x_21_1;
                            float hidden_y_26_1 = gate_y_20_1 / denominator_y_24_1 * up_y_22_1;
                            __nv_bfloat162 _bf16x2_453 = __float22bfloat162_rn(make_float2(hidden_x_25_1, hidden_y_26_1));
                            hidden_packed_1[3] = __as_u32(_bf16x2_453);
                            float2 _cvt_f32_264 = __bfloat1622float2(__as_bf16x2(gate_packed_1[4]));
                            float2 _cvt_f32_265 = __bfloat1622float2(__as_bf16x2(up_packed_1[4]));
                            float gate_x_27_1 = _cvt_f32_264.x;
                            float gate_y_28_1 = _cvt_f32_264.y;
                            float up_x_29_1 = _cvt_f32_265.x;
                            float up_y_30_1 = _cvt_f32_265.y;
                            float _exp_264 = expf(gate_x_27_1 * -1.0f);
                            float denominator_x_31_1 = _exp_264 + 1.0f;
                            float _exp_265 = expf(gate_y_28_1 * -1.0f);
                            float denominator_y_32_1 = _exp_265 + 1.0f;
                            float hidden_x_33_1 = gate_x_27_1 / denominator_x_31_1 * up_x_29_1;
                            float hidden_y_34_1 = gate_y_28_1 / denominator_y_32_1 * up_y_30_1;
                            __nv_bfloat162 _bf16x2_454 = __float22bfloat162_rn(make_float2(hidden_x_33_1, hidden_y_34_1));
                            hidden_packed_1[4] = __as_u32(_bf16x2_454);
                            float2 _cvt_f32_266 = __bfloat1622float2(__as_bf16x2(gate_packed_1[5]));
                            float2 _cvt_f32_267 = __bfloat1622float2(__as_bf16x2(up_packed_1[5]));
                            float gate_x_35_1 = _cvt_f32_266.x;
                            float gate_y_36_1 = _cvt_f32_266.y;
                            float up_x_37_1 = _cvt_f32_267.x;
                            float up_y_38_1 = _cvt_f32_267.y;
                            float _exp_266 = expf(gate_x_35_1 * -1.0f);
                            float denominator_x_39_1 = _exp_266 + 1.0f;
                            float _exp_267 = expf(gate_y_36_1 * -1.0f);
                            float denominator_y_40_1 = _exp_267 + 1.0f;
                            float hidden_x_41_1 = gate_x_35_1 / denominator_x_39_1 * up_x_37_1;
                            float hidden_y_42_1 = gate_y_36_1 / denominator_y_40_1 * up_y_38_1;
                            __nv_bfloat162 _bf16x2_455 = __float22bfloat162_rn(make_float2(hidden_x_41_1, hidden_y_42_1));
                            hidden_packed_1[5] = __as_u32(_bf16x2_455);
                            float2 _cvt_f32_268 = __bfloat1622float2(__as_bf16x2(gate_packed_1[6]));
                            float2 _cvt_f32_269 = __bfloat1622float2(__as_bf16x2(up_packed_1[6]));
                            float gate_x_43_1 = _cvt_f32_268.x;
                            float gate_y_44_1 = _cvt_f32_268.y;
                            float up_x_45_1 = _cvt_f32_269.x;
                            float up_y_46_1 = _cvt_f32_269.y;
                            float _exp_268 = expf(gate_x_43_1 * -1.0f);
                            float denominator_x_47_1 = _exp_268 + 1.0f;
                            float _exp_269 = expf(gate_y_44_1 * -1.0f);
                            float denominator_y_48_1 = _exp_269 + 1.0f;
                            float hidden_x_49_1 = gate_x_43_1 / denominator_x_47_1 * up_x_45_1;
                            float hidden_y_50_1 = gate_y_44_1 / denominator_y_48_1 * up_y_46_1;
                            __nv_bfloat162 _bf16x2_456 = __float22bfloat162_rn(make_float2(hidden_x_49_1, hidden_y_50_1));
                            hidden_packed_1[6] = __as_u32(_bf16x2_456);
                            float2 _cvt_f32_270 = __bfloat1622float2(__as_bf16x2(gate_packed_1[7]));
                            float2 _cvt_f32_271 = __bfloat1622float2(__as_bf16x2(up_packed_1[7]));
                            float gate_x_51_1 = _cvt_f32_270.x;
                            float gate_y_52_1 = _cvt_f32_270.y;
                            float up_x_53_1 = _cvt_f32_271.x;
                            float up_y_54_1 = _cvt_f32_271.y;
                            float _exp_270 = expf(gate_x_51_1 * -1.0f);
                            float denominator_x_55_1 = _exp_270 + 1.0f;
                            float _exp_271 = expf(gate_y_52_1 * -1.0f);
                            float denominator_y_56_1 = _exp_271 + 1.0f;
                            float hidden_x_57_1 = gate_x_51_1 / denominator_x_55_1 * up_x_53_1;
                            float hidden_y_58_1 = gate_y_52_1 / denominator_y_56_1 * up_y_54_1;
                            __nv_bfloat162 _bf16x2_457 = __float22bfloat162_rn(make_float2(hidden_x_57_1, hidden_y_58_1));
                            hidden_packed_1[7] = __as_u32(_bf16x2_457);
                            float2 _cvt_f32_272 = __bfloat1622float2(__as_bf16x2(gate_packed_1[8]));
                            float2 _cvt_f32_273 = __bfloat1622float2(__as_bf16x2(up_packed_1[8]));
                            float gate_x_59_1 = _cvt_f32_272.x;
                            float gate_y_60_1 = _cvt_f32_272.y;
                            float up_x_61_1 = _cvt_f32_273.x;
                            float up_y_62_1 = _cvt_f32_273.y;
                            float _exp_272 = expf(gate_x_59_1 * -1.0f);
                            float denominator_x_63_1 = _exp_272 + 1.0f;
                            float _exp_273 = expf(gate_y_60_1 * -1.0f);
                            float denominator_y_64_1 = _exp_273 + 1.0f;
                            float hidden_x_65_1 = gate_x_59_1 / denominator_x_63_1 * up_x_61_1;
                            float hidden_y_66_1 = gate_y_60_1 / denominator_y_64_1 * up_y_62_1;
                            __nv_bfloat162 _bf16x2_458 = __float22bfloat162_rn(make_float2(hidden_x_65_1, hidden_y_66_1));
                            hidden_packed_1[8] = __as_u32(_bf16x2_458);
                            float2 _cvt_f32_274 = __bfloat1622float2(__as_bf16x2(gate_packed_1[9]));
                            float2 _cvt_f32_275 = __bfloat1622float2(__as_bf16x2(up_packed_1[9]));
                            float gate_x_67_1 = _cvt_f32_274.x;
                            float gate_y_68_1 = _cvt_f32_274.y;
                            float up_x_69_1 = _cvt_f32_275.x;
                            float up_y_70_1 = _cvt_f32_275.y;
                            float _exp_274 = expf(gate_x_67_1 * -1.0f);
                            float denominator_x_71_1 = _exp_274 + 1.0f;
                            float _exp_275 = expf(gate_y_68_1 * -1.0f);
                            float denominator_y_72_1 = _exp_275 + 1.0f;
                            float hidden_x_73_1 = gate_x_67_1 / denominator_x_71_1 * up_x_69_1;
                            float hidden_y_74_1 = gate_y_68_1 / denominator_y_72_1 * up_y_70_1;
                            __nv_bfloat162 _bf16x2_459 = __float22bfloat162_rn(make_float2(hidden_x_73_1, hidden_y_74_1));
                            hidden_packed_1[9] = __as_u32(_bf16x2_459);
                            float2 _cvt_f32_276 = __bfloat1622float2(__as_bf16x2(gate_packed_1[10]));
                            float2 _cvt_f32_277 = __bfloat1622float2(__as_bf16x2(up_packed_1[10]));
                            float gate_x_75_1 = _cvt_f32_276.x;
                            float gate_y_76_1 = _cvt_f32_276.y;
                            float up_x_77_1 = _cvt_f32_277.x;
                            float up_y_78_1 = _cvt_f32_277.y;
                            float _exp_276 = expf(gate_x_75_1 * -1.0f);
                            float denominator_x_79_1 = _exp_276 + 1.0f;
                            float _exp_277 = expf(gate_y_76_1 * -1.0f);
                            float denominator_y_80_1 = _exp_277 + 1.0f;
                            float hidden_x_81_1 = gate_x_75_1 / denominator_x_79_1 * up_x_77_1;
                            float hidden_y_82_1 = gate_y_76_1 / denominator_y_80_1 * up_y_78_1;
                            __nv_bfloat162 _bf16x2_460 = __float22bfloat162_rn(make_float2(hidden_x_81_1, hidden_y_82_1));
                            hidden_packed_1[10] = __as_u32(_bf16x2_460);
                            float2 _cvt_f32_278 = __bfloat1622float2(__as_bf16x2(gate_packed_1[11]));
                            float2 _cvt_f32_279 = __bfloat1622float2(__as_bf16x2(up_packed_1[11]));
                            float gate_x_83_1 = _cvt_f32_278.x;
                            float gate_y_84_1 = _cvt_f32_278.y;
                            float up_x_85_1 = _cvt_f32_279.x;
                            float up_y_86_1 = _cvt_f32_279.y;
                            float _exp_278 = expf(gate_x_83_1 * -1.0f);
                            float denominator_x_87_1 = _exp_278 + 1.0f;
                            float _exp_279 = expf(gate_y_84_1 * -1.0f);
                            float denominator_y_88_1 = _exp_279 + 1.0f;
                            float hidden_x_89_1 = gate_x_83_1 / denominator_x_87_1 * up_x_85_1;
                            float hidden_y_90_1 = gate_y_84_1 / denominator_y_88_1 * up_y_86_1;
                            __nv_bfloat162 _bf16x2_461 = __float22bfloat162_rn(make_float2(hidden_x_89_1, hidden_y_90_1));
                            hidden_packed_1[11] = __as_u32(_bf16x2_461);
                            float2 _cvt_f32_280 = __bfloat1622float2(__as_bf16x2(gate_packed_1[12]));
                            float2 _cvt_f32_281 = __bfloat1622float2(__as_bf16x2(up_packed_1[12]));
                            float gate_x_91_1 = _cvt_f32_280.x;
                            float gate_y_92_1 = _cvt_f32_280.y;
                            float up_x_93_1 = _cvt_f32_281.x;
                            float up_y_94_1 = _cvt_f32_281.y;
                            float _exp_280 = expf(gate_x_91_1 * -1.0f);
                            float denominator_x_95_1 = _exp_280 + 1.0f;
                            float _exp_281 = expf(gate_y_92_1 * -1.0f);
                            float denominator_y_96_1 = _exp_281 + 1.0f;
                            float hidden_x_97_1 = gate_x_91_1 / denominator_x_95_1 * up_x_93_1;
                            float hidden_y_98_1 = gate_y_92_1 / denominator_y_96_1 * up_y_94_1;
                            __nv_bfloat162 _bf16x2_462 = __float22bfloat162_rn(make_float2(hidden_x_97_1, hidden_y_98_1));
                            hidden_packed_1[12] = __as_u32(_bf16x2_462);
                            float2 _cvt_f32_282 = __bfloat1622float2(__as_bf16x2(gate_packed_1[13]));
                            float2 _cvt_f32_283 = __bfloat1622float2(__as_bf16x2(up_packed_1[13]));
                            float gate_x_99_1 = _cvt_f32_282.x;
                            float gate_y_100_1 = _cvt_f32_282.y;
                            float up_x_101_1 = _cvt_f32_283.x;
                            float up_y_102_1 = _cvt_f32_283.y;
                            float _exp_282 = expf(gate_x_99_1 * -1.0f);
                            float denominator_x_103_1 = _exp_282 + 1.0f;
                            float _exp_283 = expf(gate_y_100_1 * -1.0f);
                            float denominator_y_104_1 = _exp_283 + 1.0f;
                            float hidden_x_105_1 = gate_x_99_1 / denominator_x_103_1 * up_x_101_1;
                            float hidden_y_106_1 = gate_y_100_1 / denominator_y_104_1 * up_y_102_1;
                            __nv_bfloat162 _bf16x2_463 = __float22bfloat162_rn(make_float2(hidden_x_105_1, hidden_y_106_1));
                            hidden_packed_1[13] = __as_u32(_bf16x2_463);
                            float2 _cvt_f32_284 = __bfloat1622float2(__as_bf16x2(gate_packed_1[14]));
                            float2 _cvt_f32_285 = __bfloat1622float2(__as_bf16x2(up_packed_1[14]));
                            float gate_x_107_1 = _cvt_f32_284.x;
                            float gate_y_108_1 = _cvt_f32_284.y;
                            float up_x_109_1 = _cvt_f32_285.x;
                            float up_y_110_1 = _cvt_f32_285.y;
                            float _exp_284 = expf(gate_x_107_1 * -1.0f);
                            float denominator_x_111_1 = _exp_284 + 1.0f;
                            float _exp_285 = expf(gate_y_108_1 * -1.0f);
                            float denominator_y_112_1 = _exp_285 + 1.0f;
                            float hidden_x_113_1 = gate_x_107_1 / denominator_x_111_1 * up_x_109_1;
                            float hidden_y_114_1 = gate_y_108_1 / denominator_y_112_1 * up_y_110_1;
                            __nv_bfloat162 _bf16x2_464 = __float22bfloat162_rn(make_float2(hidden_x_113_1, hidden_y_114_1));
                            hidden_packed_1[14] = __as_u32(_bf16x2_464);
                            float2 _cvt_f32_286 = __bfloat1622float2(__as_bf16x2(gate_packed_1[15]));
                            float2 _cvt_f32_287 = __bfloat1622float2(__as_bf16x2(up_packed_1[15]));
                            float gate_x_115_1 = _cvt_f32_286.x;
                            float gate_y_116_1 = _cvt_f32_286.y;
                            float up_x_117_1 = _cvt_f32_287.x;
                            float up_y_118_1 = _cvt_f32_287.y;
                            float _exp_286 = expf(gate_x_115_1 * -1.0f);
                            float denominator_x_119_1 = _exp_286 + 1.0f;
                            float _exp_287 = expf(gate_y_116_1 * -1.0f);
                            float denominator_y_120_1 = _exp_287 + 1.0f;
                            float hidden_x_121_1 = gate_x_115_1 / denominator_x_119_1 * up_x_117_1;
                            float hidden_y_122_1 = gate_y_116_1 / denominator_y_120_1 * up_y_118_1;
                            __nv_bfloat162 _bf16x2_465 = __float22bfloat162_rn(make_float2(hidden_x_121_1, hidden_y_122_1));
                            hidden_packed_1[15] = __as_u32(_bf16x2_465);
                            float2 _cvt_f32_288 = __bfloat1622float2(__as_bf16x2(gate_packed_1[16]));
                            float2 _cvt_f32_289 = __bfloat1622float2(__as_bf16x2(up_packed_1[16]));
                            float gate_x_123_1 = _cvt_f32_288.x;
                            float gate_y_124_1 = _cvt_f32_288.y;
                            float up_x_125_1 = _cvt_f32_289.x;
                            float up_y_126_1 = _cvt_f32_289.y;
                            float _exp_288 = expf(gate_x_123_1 * -1.0f);
                            float denominator_x_127_1 = _exp_288 + 1.0f;
                            float _exp_289 = expf(gate_y_124_1 * -1.0f);
                            float denominator_y_128_1 = _exp_289 + 1.0f;
                            float hidden_x_129_1 = gate_x_123_1 / denominator_x_127_1 * up_x_125_1;
                            float hidden_y_130_1 = gate_y_124_1 / denominator_y_128_1 * up_y_126_1;
                            __nv_bfloat162 _bf16x2_466 = __float22bfloat162_rn(make_float2(hidden_x_129_1, hidden_y_130_1));
                            hidden_packed_1[16] = __as_u32(_bf16x2_466);
                            float2 _cvt_f32_290 = __bfloat1622float2(__as_bf16x2(gate_packed_1[17]));
                            float2 _cvt_f32_291 = __bfloat1622float2(__as_bf16x2(up_packed_1[17]));
                            float gate_x_131_1 = _cvt_f32_290.x;
                            float gate_y_132_1 = _cvt_f32_290.y;
                            float up_x_133_1 = _cvt_f32_291.x;
                            float up_y_134_1 = _cvt_f32_291.y;
                            float _exp_290 = expf(gate_x_131_1 * -1.0f);
                            float denominator_x_135_1 = _exp_290 + 1.0f;
                            float _exp_291 = expf(gate_y_132_1 * -1.0f);
                            float denominator_y_136_1 = _exp_291 + 1.0f;
                            float hidden_x_137_1 = gate_x_131_1 / denominator_x_135_1 * up_x_133_1;
                            float hidden_y_138_1 = gate_y_132_1 / denominator_y_136_1 * up_y_134_1;
                            __nv_bfloat162 _bf16x2_467 = __float22bfloat162_rn(make_float2(hidden_x_137_1, hidden_y_138_1));
                            hidden_packed_1[17] = __as_u32(_bf16x2_467);
                            float2 _cvt_f32_292 = __bfloat1622float2(__as_bf16x2(gate_packed_1[18]));
                            float2 _cvt_f32_293 = __bfloat1622float2(__as_bf16x2(up_packed_1[18]));
                            float gate_x_139_1 = _cvt_f32_292.x;
                            float gate_y_140_1 = _cvt_f32_292.y;
                            float up_x_141_1 = _cvt_f32_293.x;
                            float up_y_142_1 = _cvt_f32_293.y;
                            float _exp_292 = expf(gate_x_139_1 * -1.0f);
                            float denominator_x_143_1 = _exp_292 + 1.0f;
                            float _exp_293 = expf(gate_y_140_1 * -1.0f);
                            float denominator_y_144_1 = _exp_293 + 1.0f;
                            float hidden_x_145_1 = gate_x_139_1 / denominator_x_143_1 * up_x_141_1;
                            float hidden_y_146_1 = gate_y_140_1 / denominator_y_144_1 * up_y_142_1;
                            __nv_bfloat162 _bf16x2_468 = __float22bfloat162_rn(make_float2(hidden_x_145_1, hidden_y_146_1));
                            hidden_packed_1[18] = __as_u32(_bf16x2_468);
                            float2 _cvt_f32_294 = __bfloat1622float2(__as_bf16x2(gate_packed_1[19]));
                            float2 _cvt_f32_295 = __bfloat1622float2(__as_bf16x2(up_packed_1[19]));
                            float gate_x_147_1 = _cvt_f32_294.x;
                            float gate_y_148_1 = _cvt_f32_294.y;
                            float up_x_149_1 = _cvt_f32_295.x;
                            float up_y_150_1 = _cvt_f32_295.y;
                            float _exp_294 = expf(gate_x_147_1 * -1.0f);
                            float denominator_x_151_1 = _exp_294 + 1.0f;
                            float _exp_295 = expf(gate_y_148_1 * -1.0f);
                            float denominator_y_152_1 = _exp_295 + 1.0f;
                            float hidden_x_153_1 = gate_x_147_1 / denominator_x_151_1 * up_x_149_1;
                            float hidden_y_154_1 = gate_y_148_1 / denominator_y_152_1 * up_y_150_1;
                            __nv_bfloat162 _bf16x2_469 = __float22bfloat162_rn(make_float2(hidden_x_153_1, hidden_y_154_1));
                            hidden_packed_1[19] = __as_u32(_bf16x2_469);
                            float2 _cvt_f32_296 = __bfloat1622float2(__as_bf16x2(gate_packed_1[20]));
                            float2 _cvt_f32_297 = __bfloat1622float2(__as_bf16x2(up_packed_1[20]));
                            float gate_x_155_1 = _cvt_f32_296.x;
                            float gate_y_156_1 = _cvt_f32_296.y;
                            float up_x_157_1 = _cvt_f32_297.x;
                            float up_y_158_1 = _cvt_f32_297.y;
                            float _exp_296 = expf(gate_x_155_1 * -1.0f);
                            float denominator_x_159_1 = _exp_296 + 1.0f;
                            float _exp_297 = expf(gate_y_156_1 * -1.0f);
                            float denominator_y_160_1 = _exp_297 + 1.0f;
                            float hidden_x_161_1 = gate_x_155_1 / denominator_x_159_1 * up_x_157_1;
                            float hidden_y_162_1 = gate_y_156_1 / denominator_y_160_1 * up_y_158_1;
                            __nv_bfloat162 _bf16x2_470 = __float22bfloat162_rn(make_float2(hidden_x_161_1, hidden_y_162_1));
                            hidden_packed_1[20] = __as_u32(_bf16x2_470);
                            float2 _cvt_f32_298 = __bfloat1622float2(__as_bf16x2(gate_packed_1[21]));
                            float2 _cvt_f32_299 = __bfloat1622float2(__as_bf16x2(up_packed_1[21]));
                            float gate_x_163_1 = _cvt_f32_298.x;
                            float gate_y_164_1 = _cvt_f32_298.y;
                            float up_x_165_1 = _cvt_f32_299.x;
                            float up_y_166_1 = _cvt_f32_299.y;
                            float _exp_298 = expf(gate_x_163_1 * -1.0f);
                            float denominator_x_167_1 = _exp_298 + 1.0f;
                            float _exp_299 = expf(gate_y_164_1 * -1.0f);
                            float denominator_y_168_1 = _exp_299 + 1.0f;
                            float hidden_x_169_1 = gate_x_163_1 / denominator_x_167_1 * up_x_165_1;
                            float hidden_y_170_1 = gate_y_164_1 / denominator_y_168_1 * up_y_166_1;
                            __nv_bfloat162 _bf16x2_471 = __float22bfloat162_rn(make_float2(hidden_x_169_1, hidden_y_170_1));
                            hidden_packed_1[21] = __as_u32(_bf16x2_471);
                            float2 _cvt_f32_300 = __bfloat1622float2(__as_bf16x2(gate_packed_1[22]));
                            float2 _cvt_f32_301 = __bfloat1622float2(__as_bf16x2(up_packed_1[22]));
                            float gate_x_171_1 = _cvt_f32_300.x;
                            float gate_y_172_1 = _cvt_f32_300.y;
                            float up_x_173_1 = _cvt_f32_301.x;
                            float up_y_174_1 = _cvt_f32_301.y;
                            float _exp_300 = expf(gate_x_171_1 * -1.0f);
                            float denominator_x_175_1 = _exp_300 + 1.0f;
                            float _exp_301 = expf(gate_y_172_1 * -1.0f);
                            float denominator_y_176_1 = _exp_301 + 1.0f;
                            float hidden_x_177_1 = gate_x_171_1 / denominator_x_175_1 * up_x_173_1;
                            float hidden_y_178_1 = gate_y_172_1 / denominator_y_176_1 * up_y_174_1;
                            __nv_bfloat162 _bf16x2_472 = __float22bfloat162_rn(make_float2(hidden_x_177_1, hidden_y_178_1));
                            hidden_packed_1[22] = __as_u32(_bf16x2_472);
                            float2 _cvt_f32_302 = __bfloat1622float2(__as_bf16x2(gate_packed_1[23]));
                            float2 _cvt_f32_303 = __bfloat1622float2(__as_bf16x2(up_packed_1[23]));
                            float gate_x_179_1 = _cvt_f32_302.x;
                            float gate_y_180_1 = _cvt_f32_302.y;
                            float up_x_181_1 = _cvt_f32_303.x;
                            float up_y_182_1 = _cvt_f32_303.y;
                            float _exp_302 = expf(gate_x_179_1 * -1.0f);
                            float denominator_x_183_1 = _exp_302 + 1.0f;
                            float _exp_303 = expf(gate_y_180_1 * -1.0f);
                            float denominator_y_184_1 = _exp_303 + 1.0f;
                            float hidden_x_185_1 = gate_x_179_1 / denominator_x_183_1 * up_x_181_1;
                            float hidden_y_186_1 = gate_y_180_1 / denominator_y_184_1 * up_y_182_1;
                            __nv_bfloat162 _bf16x2_473 = __float22bfloat162_rn(make_float2(hidden_x_185_1, hidden_y_186_1));
                            hidden_packed_1[23] = __as_u32(_bf16x2_473);
                            float2 _cvt_f32_304 = __bfloat1622float2(__as_bf16x2(gate_packed_1[24]));
                            float2 _cvt_f32_305 = __bfloat1622float2(__as_bf16x2(up_packed_1[24]));
                            float gate_x_187_1 = _cvt_f32_304.x;
                            float gate_y_188_1 = _cvt_f32_304.y;
                            float up_x_189_1 = _cvt_f32_305.x;
                            float up_y_190_1 = _cvt_f32_305.y;
                            float _exp_304 = expf(gate_x_187_1 * -1.0f);
                            float denominator_x_191_1 = _exp_304 + 1.0f;
                            float _exp_305 = expf(gate_y_188_1 * -1.0f);
                            float denominator_y_192_1 = _exp_305 + 1.0f;
                            float hidden_x_193_1 = gate_x_187_1 / denominator_x_191_1 * up_x_189_1;
                            float hidden_y_194_1 = gate_y_188_1 / denominator_y_192_1 * up_y_190_1;
                            __nv_bfloat162 _bf16x2_474 = __float22bfloat162_rn(make_float2(hidden_x_193_1, hidden_y_194_1));
                            hidden_packed_1[24] = __as_u32(_bf16x2_474);
                            float2 _cvt_f32_306 = __bfloat1622float2(__as_bf16x2(gate_packed_1[25]));
                            float2 _cvt_f32_307 = __bfloat1622float2(__as_bf16x2(up_packed_1[25]));
                            float gate_x_195_1 = _cvt_f32_306.x;
                            float gate_y_196_1 = _cvt_f32_306.y;
                            float up_x_197_1 = _cvt_f32_307.x;
                            float up_y_198_1 = _cvt_f32_307.y;
                            float _exp_306 = expf(gate_x_195_1 * -1.0f);
                            float denominator_x_199_1 = _exp_306 + 1.0f;
                            float _exp_307 = expf(gate_y_196_1 * -1.0f);
                            float denominator_y_200_1 = _exp_307 + 1.0f;
                            float hidden_x_201_1 = gate_x_195_1 / denominator_x_199_1 * up_x_197_1;
                            float hidden_y_202_1 = gate_y_196_1 / denominator_y_200_1 * up_y_198_1;
                            __nv_bfloat162 _bf16x2_475 = __float22bfloat162_rn(make_float2(hidden_x_201_1, hidden_y_202_1));
                            hidden_packed_1[25] = __as_u32(_bf16x2_475);
                            float2 _cvt_f32_308 = __bfloat1622float2(__as_bf16x2(gate_packed_1[26]));
                            float2 _cvt_f32_309 = __bfloat1622float2(__as_bf16x2(up_packed_1[26]));
                            float gate_x_203_1 = _cvt_f32_308.x;
                            float gate_y_204_1 = _cvt_f32_308.y;
                            float up_x_205_1 = _cvt_f32_309.x;
                            float up_y_206_1 = _cvt_f32_309.y;
                            float _exp_308 = expf(gate_x_203_1 * -1.0f);
                            float denominator_x_207_1 = _exp_308 + 1.0f;
                            float _exp_309 = expf(gate_y_204_1 * -1.0f);
                            float denominator_y_208_1 = _exp_309 + 1.0f;
                            float hidden_x_209_1 = gate_x_203_1 / denominator_x_207_1 * up_x_205_1;
                            float hidden_y_210_1 = gate_y_204_1 / denominator_y_208_1 * up_y_206_1;
                            __nv_bfloat162 _bf16x2_476 = __float22bfloat162_rn(make_float2(hidden_x_209_1, hidden_y_210_1));
                            hidden_packed_1[26] = __as_u32(_bf16x2_476);
                            float2 _cvt_f32_310 = __bfloat1622float2(__as_bf16x2(gate_packed_1[27]));
                            float2 _cvt_f32_311 = __bfloat1622float2(__as_bf16x2(up_packed_1[27]));
                            float gate_x_211_1 = _cvt_f32_310.x;
                            float gate_y_212_1 = _cvt_f32_310.y;
                            float up_x_213_1 = _cvt_f32_311.x;
                            float up_y_214_1 = _cvt_f32_311.y;
                            float _exp_310 = expf(gate_x_211_1 * -1.0f);
                            float denominator_x_215_1 = _exp_310 + 1.0f;
                            float _exp_311 = expf(gate_y_212_1 * -1.0f);
                            float denominator_y_216_1 = _exp_311 + 1.0f;
                            float hidden_x_217_1 = gate_x_211_1 / denominator_x_215_1 * up_x_213_1;
                            float hidden_y_218_1 = gate_y_212_1 / denominator_y_216_1 * up_y_214_1;
                            __nv_bfloat162 _bf16x2_477 = __float22bfloat162_rn(make_float2(hidden_x_217_1, hidden_y_218_1));
                            hidden_packed_1[27] = __as_u32(_bf16x2_477);
                            float2 _cvt_f32_312 = __bfloat1622float2(__as_bf16x2(gate_packed_1[28]));
                            float2 _cvt_f32_313 = __bfloat1622float2(__as_bf16x2(up_packed_1[28]));
                            float gate_x_219_1 = _cvt_f32_312.x;
                            float gate_y_220_1 = _cvt_f32_312.y;
                            float up_x_221_1 = _cvt_f32_313.x;
                            float up_y_222_1 = _cvt_f32_313.y;
                            float _exp_312 = expf(gate_x_219_1 * -1.0f);
                            float denominator_x_223_1 = _exp_312 + 1.0f;
                            float _exp_313 = expf(gate_y_220_1 * -1.0f);
                            float denominator_y_224_1 = _exp_313 + 1.0f;
                            float hidden_x_225_1 = gate_x_219_1 / denominator_x_223_1 * up_x_221_1;
                            float hidden_y_226_1 = gate_y_220_1 / denominator_y_224_1 * up_y_222_1;
                            __nv_bfloat162 _bf16x2_478 = __float22bfloat162_rn(make_float2(hidden_x_225_1, hidden_y_226_1));
                            hidden_packed_1[28] = __as_u32(_bf16x2_478);
                            float2 _cvt_f32_314 = __bfloat1622float2(__as_bf16x2(gate_packed_1[29]));
                            float2 _cvt_f32_315 = __bfloat1622float2(__as_bf16x2(up_packed_1[29]));
                            float gate_x_227_1 = _cvt_f32_314.x;
                            float gate_y_228_1 = _cvt_f32_314.y;
                            float up_x_229_1 = _cvt_f32_315.x;
                            float up_y_230_1 = _cvt_f32_315.y;
                            float _exp_314 = expf(gate_x_227_1 * -1.0f);
                            float denominator_x_231_1 = _exp_314 + 1.0f;
                            float _exp_315 = expf(gate_y_228_1 * -1.0f);
                            float denominator_y_232_1 = _exp_315 + 1.0f;
                            float hidden_x_233_1 = gate_x_227_1 / denominator_x_231_1 * up_x_229_1;
                            float hidden_y_234_1 = gate_y_228_1 / denominator_y_232_1 * up_y_230_1;
                            __nv_bfloat162 _bf16x2_479 = __float22bfloat162_rn(make_float2(hidden_x_233_1, hidden_y_234_1));
                            hidden_packed_1[29] = __as_u32(_bf16x2_479);
                            float2 _cvt_f32_316 = __bfloat1622float2(__as_bf16x2(gate_packed_1[30]));
                            float2 _cvt_f32_317 = __bfloat1622float2(__as_bf16x2(up_packed_1[30]));
                            float gate_x_235_1 = _cvt_f32_316.x;
                            float gate_y_236_1 = _cvt_f32_316.y;
                            float up_x_237_1 = _cvt_f32_317.x;
                            float up_y_238_1 = _cvt_f32_317.y;
                            float _exp_316 = expf(gate_x_235_1 * -1.0f);
                            float denominator_x_239_1 = _exp_316 + 1.0f;
                            float _exp_317 = expf(gate_y_236_1 * -1.0f);
                            float denominator_y_240_1 = _exp_317 + 1.0f;
                            float hidden_x_241_1 = gate_x_235_1 / denominator_x_239_1 * up_x_237_1;
                            float hidden_y_242_1 = gate_y_236_1 / denominator_y_240_1 * up_y_238_1;
                            __nv_bfloat162 _bf16x2_480 = __float22bfloat162_rn(make_float2(hidden_x_241_1, hidden_y_242_1));
                            hidden_packed_1[30] = __as_u32(_bf16x2_480);
                            float2 _cvt_f32_318 = __bfloat1622float2(__as_bf16x2(gate_packed_1[31]));
                            float2 _cvt_f32_319 = __bfloat1622float2(__as_bf16x2(up_packed_1[31]));
                            float gate_x_243_1 = _cvt_f32_318.x;
                            float gate_y_244_1 = _cvt_f32_318.y;
                            float up_x_245_1 = _cvt_f32_319.x;
                            float up_y_246_1 = _cvt_f32_319.y;
                            float _exp_318 = expf(gate_x_243_1 * -1.0f);
                            float denominator_x_247_1 = _exp_318 + 1.0f;
                            float _exp_319 = expf(gate_y_244_1 * -1.0f);
                            float denominator_y_248_1 = _exp_319 + 1.0f;
                            float hidden_x_249_1 = gate_x_243_1 / denominator_x_247_1 * up_x_245_1;
                            float hidden_y_250_1 = gate_y_244_1 / denominator_y_248_1 * up_y_246_1;
                            __nv_bfloat162 _bf16x2_481 = __float22bfloat162_rn(make_float2(hidden_x_249_1, hidden_y_250_1));
                            hidden_packed_1[31] = __as_u32(_bf16x2_481);
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_251_1 = tid / 32;
                            int lane_4 = tid % 32;
                            #pragma unroll
                            for (int half_26 = 0; half_26 < 2; half_26++) {
                                #pragma unroll
                                for (int col_tile_26 = 0; col_tile_26 < 2; col_tile_26++) {
                                    int row_32 = warp_251_1 * 32 + half_26 * 16 + lane_4 % 16;
                                    int col_29 = col_tile_26 * 16 + lane_4 / 16 * 8;
                                    unsigned int address_3_24 = d_smem_addr + (unsigned int)((row_32 * 32 + col_29) * 2);
                                    address_3_24 = address_3_24 ^ (address_3_24 & 511) >> 7 << 4;
                                    int offset_27 = half_26 * 8 + col_tile_26 * 4;
                                    uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(address_3_24);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_27])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_27 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_27 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_27 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_252_1 = tid / 32;
                            int lane_253_1 = tid % 32;
                            #pragma unroll
                            for (int half_27 = 0; half_27 < 2; half_27++) {
                                #pragma unroll
                                for (int col_tile_27 = 0; col_tile_27 < 2; col_tile_27++) {
                                    int row_33 = warp_252_1 * 32 + half_27 * 16 + lane_253_1 % 16;
                                    int col_30 = col_tile_27 * 16 + lane_253_1 / 16 * 8;
                                    unsigned int address_3_25 = d_smem_addr + 8192 + (unsigned int)((row_33 * 32 + col_30) * 2);
                                    address_3_25 = address_3_25 ^ (address_3_25 & 511) >> 7 << 4;
                                    int offset_28 = half_27 * 8 + col_tile_27 * 4;
                                    uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(address_3_25);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_28])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_28 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_28 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_28 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_254_1 = tid / 32;
                            int lane_255_1 = tid % 32;
                            #pragma unroll
                            for (int half_28 = 0; half_28 < 2; half_28++) {
                                #pragma unroll
                                for (int col_tile_28 = 0; col_tile_28 < 2; col_tile_28++) {
                                    int row_34 = warp_254_1 * 32 + half_28 * 16 + lane_255_1 % 16;
                                    int col_31 = col_tile_28 * 16 + lane_255_1 / 16 * 8;
                                    unsigned int address_3_26 = d_smem_addr + 16384 + (unsigned int)((row_34 * 32 + col_31) * 2);
                                    address_3_26 = address_3_26 ^ (address_3_26 & 511) >> 7 << 4;
                                    int offset_29 = half_28 * 8 + col_tile_28 * 4;
                                    uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(address_3_26);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_29])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_29 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_29 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_29 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_256_1 = tid / 32;
                            int lane_257_1 = tid % 32;
                            #pragma unroll
                            for (int half_29 = 0; half_29 < 2; half_29++) {
                                #pragma unroll
                                for (int col_tile_29 = 0; col_tile_29 < 2; col_tile_29++) {
                                    int row_35 = warp_256_1 * 32 + half_29 * 16 + lane_257_1 % 16;
                                    int col_32 = col_tile_29 * 16 + lane_257_1 / 16 * 8;
                                    unsigned int address_3_27 = d_smem_addr + (unsigned int)((row_35 * 32 + col_32) * 2);
                                    address_3_27 = address_3_27 ^ (address_3_27 & 511) >> 7 << 4;
                                    int offset_30 = 16 + half_29 * 8 + col_tile_29 * 4;
                                    uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(address_3_27);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_30])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_30 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_30 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_1[offset_30 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_258_1 = tid / 32;
                            int lane_259_1 = tid % 32;
                            #pragma unroll
                            for (int half_30 = 0; half_30 < 2; half_30++) {
                                #pragma unroll
                                for (int col_tile_30 = 0; col_tile_30 < 2; col_tile_30++) {
                                    int row_36 = warp_258_1 * 32 + half_30 * 16 + lane_259_1 % 16;
                                    int col_33 = col_tile_30 * 16 + lane_259_1 / 16 * 8;
                                    unsigned int address_3_28 = d_smem_addr + 8192 + (unsigned int)((row_36 * 32 + col_33) * 2);
                                    address_3_28 = address_3_28 ^ (address_3_28 & 511) >> 7 << 4;
                                    int offset_31 = 16 + half_30 * 8 + col_tile_30 * 4;
                                    uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(address_3_28);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_31])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_31 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_31 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_1[offset_31 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_260_1 = tid / 32;
                            int lane_261_1 = tid % 32;
                            #pragma unroll
                            for (int half_31 = 0; half_31 < 2; half_31++) {
                                #pragma unroll
                                for (int col_tile_31 = 0; col_tile_31 < 2; col_tile_31++) {
                                    int row_37 = warp_260_1 * 32 + half_31 * 16 + lane_261_1 % 16;
                                    int col_34 = col_tile_31 * 16 + lane_261_1 / 16 * 8;
                                    unsigned int address_3_29 = d_smem_addr + 16384 + (unsigned int)((row_37 * 32 + col_34) * 2);
                                    address_3_29 = address_3_29 ^ (address_3_29 & 511) >> 7 << 4;
                                    int offset_32 = 16 + half_31 * 8 + col_tile_31 * 4;
                                    uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(address_3_29);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_32])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_32 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_32 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_32 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int gate_packed_262_1[32];
                            unsigned int up_packed_263_1[32];
                            unsigned int hidden_packed_264_1[32];
                            unsigned int address_265_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 64;
                            float _tmem_load_42[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_42[15]))
                                : "r"(address_265_1));
                            float _tmem_load_43[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[15]))
                                : "r"(address_265_1 + 256));
                            __nv_bfloat162 _bf16x2_482 = __float22bfloat162_rn(make_float2(_tmem_load_42[0], _tmem_load_42[1]));
                            gate_packed_262_1[0] = __as_u32(_bf16x2_482);
                            __nv_bfloat162 _bf16x2_483 = __float22bfloat162_rn(make_float2(_tmem_load_43[0], _tmem_load_43[1]));
                            up_packed_263_1[0] = __as_u32(_bf16x2_483);
                            __nv_bfloat162 _bf16x2_484 = __float22bfloat162_rn(make_float2(_tmem_load_42[2], _tmem_load_42[3]));
                            gate_packed_262_1[1] = __as_u32(_bf16x2_484);
                            __nv_bfloat162 _bf16x2_485 = __float22bfloat162_rn(make_float2(_tmem_load_43[2], _tmem_load_43[3]));
                            up_packed_263_1[1] = __as_u32(_bf16x2_485);
                            __nv_bfloat162 _bf16x2_486 = __float22bfloat162_rn(make_float2(_tmem_load_42[4], _tmem_load_42[5]));
                            gate_packed_262_1[2] = __as_u32(_bf16x2_486);
                            __nv_bfloat162 _bf16x2_487 = __float22bfloat162_rn(make_float2(_tmem_load_43[4], _tmem_load_43[5]));
                            up_packed_263_1[2] = __as_u32(_bf16x2_487);
                            __nv_bfloat162 _bf16x2_488 = __float22bfloat162_rn(make_float2(_tmem_load_42[6], _tmem_load_42[7]));
                            gate_packed_262_1[3] = __as_u32(_bf16x2_488);
                            __nv_bfloat162 _bf16x2_489 = __float22bfloat162_rn(make_float2(_tmem_load_43[6], _tmem_load_43[7]));
                            up_packed_263_1[3] = __as_u32(_bf16x2_489);
                            __nv_bfloat162 _bf16x2_490 = __float22bfloat162_rn(make_float2(_tmem_load_42[8], _tmem_load_42[9]));
                            gate_packed_262_1[4] = __as_u32(_bf16x2_490);
                            __nv_bfloat162 _bf16x2_491 = __float22bfloat162_rn(make_float2(_tmem_load_43[8], _tmem_load_43[9]));
                            up_packed_263_1[4] = __as_u32(_bf16x2_491);
                            __nv_bfloat162 _bf16x2_492 = __float22bfloat162_rn(make_float2(_tmem_load_42[10], _tmem_load_42[11]));
                            gate_packed_262_1[5] = __as_u32(_bf16x2_492);
                            __nv_bfloat162 _bf16x2_493 = __float22bfloat162_rn(make_float2(_tmem_load_43[10], _tmem_load_43[11]));
                            up_packed_263_1[5] = __as_u32(_bf16x2_493);
                            __nv_bfloat162 _bf16x2_494 = __float22bfloat162_rn(make_float2(_tmem_load_42[12], _tmem_load_42[13]));
                            gate_packed_262_1[6] = __as_u32(_bf16x2_494);
                            __nv_bfloat162 _bf16x2_495 = __float22bfloat162_rn(make_float2(_tmem_load_43[12], _tmem_load_43[13]));
                            up_packed_263_1[6] = __as_u32(_bf16x2_495);
                            __nv_bfloat162 _bf16x2_496 = __float22bfloat162_rn(make_float2(_tmem_load_42[14], _tmem_load_42[15]));
                            gate_packed_262_1[7] = __as_u32(_bf16x2_496);
                            __nv_bfloat162 _bf16x2_497 = __float22bfloat162_rn(make_float2(_tmem_load_43[14], _tmem_load_43[15]));
                            up_packed_263_1[7] = __as_u32(_bf16x2_497);
                            unsigned int address_266_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 64;
                            float _tmem_load_44[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_44[15]))
                                : "r"(address_266_1));
                            float _tmem_load_45[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[15]))
                                : "r"(address_266_1 + 256));
                            __nv_bfloat162 _bf16x2_498 = __float22bfloat162_rn(make_float2(_tmem_load_44[0], _tmem_load_44[1]));
                            gate_packed_262_1[8] = __as_u32(_bf16x2_498);
                            __nv_bfloat162 _bf16x2_499 = __float22bfloat162_rn(make_float2(_tmem_load_45[0], _tmem_load_45[1]));
                            up_packed_263_1[8] = __as_u32(_bf16x2_499);
                            __nv_bfloat162 _bf16x2_500 = __float22bfloat162_rn(make_float2(_tmem_load_44[2], _tmem_load_44[3]));
                            gate_packed_262_1[9] = __as_u32(_bf16x2_500);
                            __nv_bfloat162 _bf16x2_501 = __float22bfloat162_rn(make_float2(_tmem_load_45[2], _tmem_load_45[3]));
                            up_packed_263_1[9] = __as_u32(_bf16x2_501);
                            __nv_bfloat162 _bf16x2_502 = __float22bfloat162_rn(make_float2(_tmem_load_44[4], _tmem_load_44[5]));
                            gate_packed_262_1[10] = __as_u32(_bf16x2_502);
                            __nv_bfloat162 _bf16x2_503 = __float22bfloat162_rn(make_float2(_tmem_load_45[4], _tmem_load_45[5]));
                            up_packed_263_1[10] = __as_u32(_bf16x2_503);
                            __nv_bfloat162 _bf16x2_504 = __float22bfloat162_rn(make_float2(_tmem_load_44[6], _tmem_load_44[7]));
                            gate_packed_262_1[11] = __as_u32(_bf16x2_504);
                            __nv_bfloat162 _bf16x2_505 = __float22bfloat162_rn(make_float2(_tmem_load_45[6], _tmem_load_45[7]));
                            up_packed_263_1[11] = __as_u32(_bf16x2_505);
                            __nv_bfloat162 _bf16x2_506 = __float22bfloat162_rn(make_float2(_tmem_load_44[8], _tmem_load_44[9]));
                            gate_packed_262_1[12] = __as_u32(_bf16x2_506);
                            __nv_bfloat162 _bf16x2_507 = __float22bfloat162_rn(make_float2(_tmem_load_45[8], _tmem_load_45[9]));
                            up_packed_263_1[12] = __as_u32(_bf16x2_507);
                            __nv_bfloat162 _bf16x2_508 = __float22bfloat162_rn(make_float2(_tmem_load_44[10], _tmem_load_44[11]));
                            gate_packed_262_1[13] = __as_u32(_bf16x2_508);
                            __nv_bfloat162 _bf16x2_509 = __float22bfloat162_rn(make_float2(_tmem_load_45[10], _tmem_load_45[11]));
                            up_packed_263_1[13] = __as_u32(_bf16x2_509);
                            __nv_bfloat162 _bf16x2_510 = __float22bfloat162_rn(make_float2(_tmem_load_44[12], _tmem_load_44[13]));
                            gate_packed_262_1[14] = __as_u32(_bf16x2_510);
                            __nv_bfloat162 _bf16x2_511 = __float22bfloat162_rn(make_float2(_tmem_load_45[12], _tmem_load_45[13]));
                            up_packed_263_1[14] = __as_u32(_bf16x2_511);
                            __nv_bfloat162 _bf16x2_512 = __float22bfloat162_rn(make_float2(_tmem_load_44[14], _tmem_load_44[15]));
                            gate_packed_262_1[15] = __as_u32(_bf16x2_512);
                            __nv_bfloat162 _bf16x2_513 = __float22bfloat162_rn(make_float2(_tmem_load_45[14], _tmem_load_45[15]));
                            up_packed_263_1[15] = __as_u32(_bf16x2_513);
                            unsigned int address_267_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 96;
                            float _tmem_load_46[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_46[15]))
                                : "r"(address_267_1));
                            float _tmem_load_47[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[15]))
                                : "r"(address_267_1 + 256));
                            __nv_bfloat162 _bf16x2_514 = __float22bfloat162_rn(make_float2(_tmem_load_46[0], _tmem_load_46[1]));
                            gate_packed_262_1[16] = __as_u32(_bf16x2_514);
                            __nv_bfloat162 _bf16x2_515 = __float22bfloat162_rn(make_float2(_tmem_load_47[0], _tmem_load_47[1]));
                            up_packed_263_1[16] = __as_u32(_bf16x2_515);
                            __nv_bfloat162 _bf16x2_516 = __float22bfloat162_rn(make_float2(_tmem_load_46[2], _tmem_load_46[3]));
                            gate_packed_262_1[17] = __as_u32(_bf16x2_516);
                            __nv_bfloat162 _bf16x2_517 = __float22bfloat162_rn(make_float2(_tmem_load_47[2], _tmem_load_47[3]));
                            up_packed_263_1[17] = __as_u32(_bf16x2_517);
                            __nv_bfloat162 _bf16x2_518 = __float22bfloat162_rn(make_float2(_tmem_load_46[4], _tmem_load_46[5]));
                            gate_packed_262_1[18] = __as_u32(_bf16x2_518);
                            __nv_bfloat162 _bf16x2_519 = __float22bfloat162_rn(make_float2(_tmem_load_47[4], _tmem_load_47[5]));
                            up_packed_263_1[18] = __as_u32(_bf16x2_519);
                            __nv_bfloat162 _bf16x2_520 = __float22bfloat162_rn(make_float2(_tmem_load_46[6], _tmem_load_46[7]));
                            gate_packed_262_1[19] = __as_u32(_bf16x2_520);
                            __nv_bfloat162 _bf16x2_521 = __float22bfloat162_rn(make_float2(_tmem_load_47[6], _tmem_load_47[7]));
                            up_packed_263_1[19] = __as_u32(_bf16x2_521);
                            __nv_bfloat162 _bf16x2_522 = __float22bfloat162_rn(make_float2(_tmem_load_46[8], _tmem_load_46[9]));
                            gate_packed_262_1[20] = __as_u32(_bf16x2_522);
                            __nv_bfloat162 _bf16x2_523 = __float22bfloat162_rn(make_float2(_tmem_load_47[8], _tmem_load_47[9]));
                            up_packed_263_1[20] = __as_u32(_bf16x2_523);
                            __nv_bfloat162 _bf16x2_524 = __float22bfloat162_rn(make_float2(_tmem_load_46[10], _tmem_load_46[11]));
                            gate_packed_262_1[21] = __as_u32(_bf16x2_524);
                            __nv_bfloat162 _bf16x2_525 = __float22bfloat162_rn(make_float2(_tmem_load_47[10], _tmem_load_47[11]));
                            up_packed_263_1[21] = __as_u32(_bf16x2_525);
                            __nv_bfloat162 _bf16x2_526 = __float22bfloat162_rn(make_float2(_tmem_load_46[12], _tmem_load_46[13]));
                            gate_packed_262_1[22] = __as_u32(_bf16x2_526);
                            __nv_bfloat162 _bf16x2_527 = __float22bfloat162_rn(make_float2(_tmem_load_47[12], _tmem_load_47[13]));
                            up_packed_263_1[22] = __as_u32(_bf16x2_527);
                            __nv_bfloat162 _bf16x2_528 = __float22bfloat162_rn(make_float2(_tmem_load_46[14], _tmem_load_46[15]));
                            gate_packed_262_1[23] = __as_u32(_bf16x2_528);
                            __nv_bfloat162 _bf16x2_529 = __float22bfloat162_rn(make_float2(_tmem_load_47[14], _tmem_load_47[15]));
                            up_packed_263_1[23] = __as_u32(_bf16x2_529);
                            unsigned int address_268_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 96;
                            float _tmem_load_48[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_48[15]))
                                : "r"(address_268_1));
                            float _tmem_load_49[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_49[15]))
                                : "r"(address_268_1 + 256));
                            __nv_bfloat162 _bf16x2_530 = __float22bfloat162_rn(make_float2(_tmem_load_48[0], _tmem_load_48[1]));
                            gate_packed_262_1[24] = __as_u32(_bf16x2_530);
                            __nv_bfloat162 _bf16x2_531 = __float22bfloat162_rn(make_float2(_tmem_load_49[0], _tmem_load_49[1]));
                            up_packed_263_1[24] = __as_u32(_bf16x2_531);
                            __nv_bfloat162 _bf16x2_532 = __float22bfloat162_rn(make_float2(_tmem_load_48[2], _tmem_load_48[3]));
                            gate_packed_262_1[25] = __as_u32(_bf16x2_532);
                            __nv_bfloat162 _bf16x2_533 = __float22bfloat162_rn(make_float2(_tmem_load_49[2], _tmem_load_49[3]));
                            up_packed_263_1[25] = __as_u32(_bf16x2_533);
                            __nv_bfloat162 _bf16x2_534 = __float22bfloat162_rn(make_float2(_tmem_load_48[4], _tmem_load_48[5]));
                            gate_packed_262_1[26] = __as_u32(_bf16x2_534);
                            __nv_bfloat162 _bf16x2_535 = __float22bfloat162_rn(make_float2(_tmem_load_49[4], _tmem_load_49[5]));
                            up_packed_263_1[26] = __as_u32(_bf16x2_535);
                            __nv_bfloat162 _bf16x2_536 = __float22bfloat162_rn(make_float2(_tmem_load_48[6], _tmem_load_48[7]));
                            gate_packed_262_1[27] = __as_u32(_bf16x2_536);
                            __nv_bfloat162 _bf16x2_537 = __float22bfloat162_rn(make_float2(_tmem_load_49[6], _tmem_load_49[7]));
                            up_packed_263_1[27] = __as_u32(_bf16x2_537);
                            __nv_bfloat162 _bf16x2_538 = __float22bfloat162_rn(make_float2(_tmem_load_48[8], _tmem_load_48[9]));
                            gate_packed_262_1[28] = __as_u32(_bf16x2_538);
                            __nv_bfloat162 _bf16x2_539 = __float22bfloat162_rn(make_float2(_tmem_load_49[8], _tmem_load_49[9]));
                            up_packed_263_1[28] = __as_u32(_bf16x2_539);
                            __nv_bfloat162 _bf16x2_540 = __float22bfloat162_rn(make_float2(_tmem_load_48[10], _tmem_load_48[11]));
                            gate_packed_262_1[29] = __as_u32(_bf16x2_540);
                            __nv_bfloat162 _bf16x2_541 = __float22bfloat162_rn(make_float2(_tmem_load_49[10], _tmem_load_49[11]));
                            up_packed_263_1[29] = __as_u32(_bf16x2_541);
                            __nv_bfloat162 _bf16x2_542 = __float22bfloat162_rn(make_float2(_tmem_load_48[12], _tmem_load_48[13]));
                            gate_packed_262_1[30] = __as_u32(_bf16x2_542);
                            __nv_bfloat162 _bf16x2_543 = __float22bfloat162_rn(make_float2(_tmem_load_49[12], _tmem_load_49[13]));
                            up_packed_263_1[30] = __as_u32(_bf16x2_543);
                            __nv_bfloat162 _bf16x2_544 = __float22bfloat162_rn(make_float2(_tmem_load_48[14], _tmem_load_48[15]));
                            gate_packed_262_1[31] = __as_u32(_bf16x2_544);
                            __nv_bfloat162 _bf16x2_545 = __float22bfloat162_rn(make_float2(_tmem_load_49[14], _tmem_load_49[15]));
                            up_packed_263_1[31] = __as_u32(_bf16x2_545);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _cvt_f32_320 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[0]));
                            float2 _cvt_f32_321 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[0]));
                            float gate_x_269_1 = _cvt_f32_320.x;
                            float gate_y_270_1 = _cvt_f32_320.y;
                            float up_x_271_1 = _cvt_f32_321.x;
                            float up_y_272_1 = _cvt_f32_321.y;
                            float _exp_320 = expf(gate_x_269_1 * -1.0f);
                            float denominator_x_273_1 = _exp_320 + 1.0f;
                            float _exp_321 = expf(gate_y_270_1 * -1.0f);
                            float denominator_y_274_1 = _exp_321 + 1.0f;
                            float hidden_x_275_1 = gate_x_269_1 / denominator_x_273_1 * up_x_271_1;
                            float hidden_y_276_1 = gate_y_270_1 / denominator_y_274_1 * up_y_272_1;
                            __nv_bfloat162 _bf16x2_546 = __float22bfloat162_rn(make_float2(hidden_x_275_1, hidden_y_276_1));
                            hidden_packed_264_1[0] = __as_u32(_bf16x2_546);
                            float2 _cvt_f32_322 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[1]));
                            float2 _cvt_f32_323 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[1]));
                            float gate_x_277_1 = _cvt_f32_322.x;
                            float gate_y_278_1 = _cvt_f32_322.y;
                            float up_x_279_1 = _cvt_f32_323.x;
                            float up_y_280_1 = _cvt_f32_323.y;
                            float _exp_322 = expf(gate_x_277_1 * -1.0f);
                            float denominator_x_281_1 = _exp_322 + 1.0f;
                            float _exp_323 = expf(gate_y_278_1 * -1.0f);
                            float denominator_y_282_1 = _exp_323 + 1.0f;
                            float hidden_x_283_1 = gate_x_277_1 / denominator_x_281_1 * up_x_279_1;
                            float hidden_y_284_1 = gate_y_278_1 / denominator_y_282_1 * up_y_280_1;
                            __nv_bfloat162 _bf16x2_547 = __float22bfloat162_rn(make_float2(hidden_x_283_1, hidden_y_284_1));
                            hidden_packed_264_1[1] = __as_u32(_bf16x2_547);
                            float2 _cvt_f32_324 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[2]));
                            float2 _cvt_f32_325 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[2]));
                            float gate_x_285_1 = _cvt_f32_324.x;
                            float gate_y_286_1 = _cvt_f32_324.y;
                            float up_x_287_1 = _cvt_f32_325.x;
                            float up_y_288_1 = _cvt_f32_325.y;
                            float _exp_324 = expf(gate_x_285_1 * -1.0f);
                            float denominator_x_289_1 = _exp_324 + 1.0f;
                            float _exp_325 = expf(gate_y_286_1 * -1.0f);
                            float denominator_y_290_1 = _exp_325 + 1.0f;
                            float hidden_x_291_1 = gate_x_285_1 / denominator_x_289_1 * up_x_287_1;
                            float hidden_y_292_1 = gate_y_286_1 / denominator_y_290_1 * up_y_288_1;
                            __nv_bfloat162 _bf16x2_548 = __float22bfloat162_rn(make_float2(hidden_x_291_1, hidden_y_292_1));
                            hidden_packed_264_1[2] = __as_u32(_bf16x2_548);
                            float2 _cvt_f32_326 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[3]));
                            float2 _cvt_f32_327 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[3]));
                            float gate_x_293_1 = _cvt_f32_326.x;
                            float gate_y_294_1 = _cvt_f32_326.y;
                            float up_x_295_1 = _cvt_f32_327.x;
                            float up_y_296_1 = _cvt_f32_327.y;
                            float _exp_326 = expf(gate_x_293_1 * -1.0f);
                            float denominator_x_297_1 = _exp_326 + 1.0f;
                            float _exp_327 = expf(gate_y_294_1 * -1.0f);
                            float denominator_y_298_1 = _exp_327 + 1.0f;
                            float hidden_x_299_1 = gate_x_293_1 / denominator_x_297_1 * up_x_295_1;
                            float hidden_y_300_1 = gate_y_294_1 / denominator_y_298_1 * up_y_296_1;
                            __nv_bfloat162 _bf16x2_549 = __float22bfloat162_rn(make_float2(hidden_x_299_1, hidden_y_300_1));
                            hidden_packed_264_1[3] = __as_u32(_bf16x2_549);
                            float2 _cvt_f32_328 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[4]));
                            float2 _cvt_f32_329 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[4]));
                            float gate_x_301_1 = _cvt_f32_328.x;
                            float gate_y_302_1 = _cvt_f32_328.y;
                            float up_x_303_1 = _cvt_f32_329.x;
                            float up_y_304_1 = _cvt_f32_329.y;
                            float _exp_328 = expf(gate_x_301_1 * -1.0f);
                            float denominator_x_305_1 = _exp_328 + 1.0f;
                            float _exp_329 = expf(gate_y_302_1 * -1.0f);
                            float denominator_y_306_1 = _exp_329 + 1.0f;
                            float hidden_x_307_1 = gate_x_301_1 / denominator_x_305_1 * up_x_303_1;
                            float hidden_y_308_1 = gate_y_302_1 / denominator_y_306_1 * up_y_304_1;
                            __nv_bfloat162 _bf16x2_550 = __float22bfloat162_rn(make_float2(hidden_x_307_1, hidden_y_308_1));
                            hidden_packed_264_1[4] = __as_u32(_bf16x2_550);
                            float2 _cvt_f32_330 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[5]));
                            float2 _cvt_f32_331 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[5]));
                            float gate_x_309_1 = _cvt_f32_330.x;
                            float gate_y_310_1 = _cvt_f32_330.y;
                            float up_x_311_1 = _cvt_f32_331.x;
                            float up_y_312_1 = _cvt_f32_331.y;
                            float _exp_330 = expf(gate_x_309_1 * -1.0f);
                            float denominator_x_313_1 = _exp_330 + 1.0f;
                            float _exp_331 = expf(gate_y_310_1 * -1.0f);
                            float denominator_y_314_1 = _exp_331 + 1.0f;
                            float hidden_x_315_1 = gate_x_309_1 / denominator_x_313_1 * up_x_311_1;
                            float hidden_y_316_1 = gate_y_310_1 / denominator_y_314_1 * up_y_312_1;
                            __nv_bfloat162 _bf16x2_551 = __float22bfloat162_rn(make_float2(hidden_x_315_1, hidden_y_316_1));
                            hidden_packed_264_1[5] = __as_u32(_bf16x2_551);
                            float2 _cvt_f32_332 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[6]));
                            float2 _cvt_f32_333 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[6]));
                            float gate_x_317_1 = _cvt_f32_332.x;
                            float gate_y_318_1 = _cvt_f32_332.y;
                            float up_x_319_1 = _cvt_f32_333.x;
                            float up_y_320_1 = _cvt_f32_333.y;
                            float _exp_332 = expf(gate_x_317_1 * -1.0f);
                            float denominator_x_321_1 = _exp_332 + 1.0f;
                            float _exp_333 = expf(gate_y_318_1 * -1.0f);
                            float denominator_y_322_1 = _exp_333 + 1.0f;
                            float hidden_x_323_1 = gate_x_317_1 / denominator_x_321_1 * up_x_319_1;
                            float hidden_y_324_1 = gate_y_318_1 / denominator_y_322_1 * up_y_320_1;
                            __nv_bfloat162 _bf16x2_552 = __float22bfloat162_rn(make_float2(hidden_x_323_1, hidden_y_324_1));
                            hidden_packed_264_1[6] = __as_u32(_bf16x2_552);
                            float2 _cvt_f32_334 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[7]));
                            float2 _cvt_f32_335 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[7]));
                            float gate_x_325_1 = _cvt_f32_334.x;
                            float gate_y_326_1 = _cvt_f32_334.y;
                            float up_x_327_1 = _cvt_f32_335.x;
                            float up_y_328_1 = _cvt_f32_335.y;
                            float _exp_334 = expf(gate_x_325_1 * -1.0f);
                            float denominator_x_329_1 = _exp_334 + 1.0f;
                            float _exp_335 = expf(gate_y_326_1 * -1.0f);
                            float denominator_y_330_1 = _exp_335 + 1.0f;
                            float hidden_x_331_1 = gate_x_325_1 / denominator_x_329_1 * up_x_327_1;
                            float hidden_y_332_1 = gate_y_326_1 / denominator_y_330_1 * up_y_328_1;
                            __nv_bfloat162 _bf16x2_553 = __float22bfloat162_rn(make_float2(hidden_x_331_1, hidden_y_332_1));
                            hidden_packed_264_1[7] = __as_u32(_bf16x2_553);
                            float2 _cvt_f32_336 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[8]));
                            float2 _cvt_f32_337 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[8]));
                            float gate_x_333_1 = _cvt_f32_336.x;
                            float gate_y_334_1 = _cvt_f32_336.y;
                            float up_x_335_1 = _cvt_f32_337.x;
                            float up_y_336_1 = _cvt_f32_337.y;
                            float _exp_336 = expf(gate_x_333_1 * -1.0f);
                            float denominator_x_337_1 = _exp_336 + 1.0f;
                            float _exp_337 = expf(gate_y_334_1 * -1.0f);
                            float denominator_y_338_1 = _exp_337 + 1.0f;
                            float hidden_x_339_1 = gate_x_333_1 / denominator_x_337_1 * up_x_335_1;
                            float hidden_y_340_1 = gate_y_334_1 / denominator_y_338_1 * up_y_336_1;
                            __nv_bfloat162 _bf16x2_554 = __float22bfloat162_rn(make_float2(hidden_x_339_1, hidden_y_340_1));
                            hidden_packed_264_1[8] = __as_u32(_bf16x2_554);
                            float2 _cvt_f32_338 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[9]));
                            float2 _cvt_f32_339 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[9]));
                            float gate_x_341_1 = _cvt_f32_338.x;
                            float gate_y_342_1 = _cvt_f32_338.y;
                            float up_x_343_1 = _cvt_f32_339.x;
                            float up_y_344_1 = _cvt_f32_339.y;
                            float _exp_338 = expf(gate_x_341_1 * -1.0f);
                            float denominator_x_345_1 = _exp_338 + 1.0f;
                            float _exp_339 = expf(gate_y_342_1 * -1.0f);
                            float denominator_y_346_1 = _exp_339 + 1.0f;
                            float hidden_x_347_1 = gate_x_341_1 / denominator_x_345_1 * up_x_343_1;
                            float hidden_y_348_1 = gate_y_342_1 / denominator_y_346_1 * up_y_344_1;
                            __nv_bfloat162 _bf16x2_555 = __float22bfloat162_rn(make_float2(hidden_x_347_1, hidden_y_348_1));
                            hidden_packed_264_1[9] = __as_u32(_bf16x2_555);
                            float2 _cvt_f32_340 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[10]));
                            float2 _cvt_f32_341 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[10]));
                            float gate_x_349_1 = _cvt_f32_340.x;
                            float gate_y_350_1 = _cvt_f32_340.y;
                            float up_x_351_1 = _cvt_f32_341.x;
                            float up_y_352_1 = _cvt_f32_341.y;
                            float _exp_340 = expf(gate_x_349_1 * -1.0f);
                            float denominator_x_353_1 = _exp_340 + 1.0f;
                            float _exp_341 = expf(gate_y_350_1 * -1.0f);
                            float denominator_y_354_1 = _exp_341 + 1.0f;
                            float hidden_x_355_1 = gate_x_349_1 / denominator_x_353_1 * up_x_351_1;
                            float hidden_y_356_1 = gate_y_350_1 / denominator_y_354_1 * up_y_352_1;
                            __nv_bfloat162 _bf16x2_556 = __float22bfloat162_rn(make_float2(hidden_x_355_1, hidden_y_356_1));
                            hidden_packed_264_1[10] = __as_u32(_bf16x2_556);
                            float2 _cvt_f32_342 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[11]));
                            float2 _cvt_f32_343 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[11]));
                            float gate_x_357_1 = _cvt_f32_342.x;
                            float gate_y_358_1 = _cvt_f32_342.y;
                            float up_x_359_1 = _cvt_f32_343.x;
                            float up_y_360_1 = _cvt_f32_343.y;
                            float _exp_342 = expf(gate_x_357_1 * -1.0f);
                            float denominator_x_361_1 = _exp_342 + 1.0f;
                            float _exp_343 = expf(gate_y_358_1 * -1.0f);
                            float denominator_y_362_1 = _exp_343 + 1.0f;
                            float hidden_x_363_1 = gate_x_357_1 / denominator_x_361_1 * up_x_359_1;
                            float hidden_y_364_1 = gate_y_358_1 / denominator_y_362_1 * up_y_360_1;
                            __nv_bfloat162 _bf16x2_557 = __float22bfloat162_rn(make_float2(hidden_x_363_1, hidden_y_364_1));
                            hidden_packed_264_1[11] = __as_u32(_bf16x2_557);
                            float2 _cvt_f32_344 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[12]));
                            float2 _cvt_f32_345 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[12]));
                            float gate_x_365_1 = _cvt_f32_344.x;
                            float gate_y_366_1 = _cvt_f32_344.y;
                            float up_x_367_1 = _cvt_f32_345.x;
                            float up_y_368_1 = _cvt_f32_345.y;
                            float _exp_344 = expf(gate_x_365_1 * -1.0f);
                            float denominator_x_369_1 = _exp_344 + 1.0f;
                            float _exp_345 = expf(gate_y_366_1 * -1.0f);
                            float denominator_y_370_1 = _exp_345 + 1.0f;
                            float hidden_x_371_1 = gate_x_365_1 / denominator_x_369_1 * up_x_367_1;
                            float hidden_y_372_1 = gate_y_366_1 / denominator_y_370_1 * up_y_368_1;
                            __nv_bfloat162 _bf16x2_558 = __float22bfloat162_rn(make_float2(hidden_x_371_1, hidden_y_372_1));
                            hidden_packed_264_1[12] = __as_u32(_bf16x2_558);
                            float2 _cvt_f32_346 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[13]));
                            float2 _cvt_f32_347 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[13]));
                            float gate_x_373_1 = _cvt_f32_346.x;
                            float gate_y_374_1 = _cvt_f32_346.y;
                            float up_x_375_1 = _cvt_f32_347.x;
                            float up_y_376_1 = _cvt_f32_347.y;
                            float _exp_346 = expf(gate_x_373_1 * -1.0f);
                            float denominator_x_377_1 = _exp_346 + 1.0f;
                            float _exp_347 = expf(gate_y_374_1 * -1.0f);
                            float denominator_y_378_1 = _exp_347 + 1.0f;
                            float hidden_x_379_1 = gate_x_373_1 / denominator_x_377_1 * up_x_375_1;
                            float hidden_y_380_1 = gate_y_374_1 / denominator_y_378_1 * up_y_376_1;
                            __nv_bfloat162 _bf16x2_559 = __float22bfloat162_rn(make_float2(hidden_x_379_1, hidden_y_380_1));
                            hidden_packed_264_1[13] = __as_u32(_bf16x2_559);
                            float2 _cvt_f32_348 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[14]));
                            float2 _cvt_f32_349 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[14]));
                            float gate_x_381_1 = _cvt_f32_348.x;
                            float gate_y_382_1 = _cvt_f32_348.y;
                            float up_x_383_1 = _cvt_f32_349.x;
                            float up_y_384_1 = _cvt_f32_349.y;
                            float _exp_348 = expf(gate_x_381_1 * -1.0f);
                            float denominator_x_385_1 = _exp_348 + 1.0f;
                            float _exp_349 = expf(gate_y_382_1 * -1.0f);
                            float denominator_y_386_1 = _exp_349 + 1.0f;
                            float hidden_x_387_1 = gate_x_381_1 / denominator_x_385_1 * up_x_383_1;
                            float hidden_y_388_1 = gate_y_382_1 / denominator_y_386_1 * up_y_384_1;
                            __nv_bfloat162 _bf16x2_560 = __float22bfloat162_rn(make_float2(hidden_x_387_1, hidden_y_388_1));
                            hidden_packed_264_1[14] = __as_u32(_bf16x2_560);
                            float2 _cvt_f32_350 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[15]));
                            float2 _cvt_f32_351 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[15]));
                            float gate_x_389_1 = _cvt_f32_350.x;
                            float gate_y_390_1 = _cvt_f32_350.y;
                            float up_x_391_1 = _cvt_f32_351.x;
                            float up_y_392_1 = _cvt_f32_351.y;
                            float _exp_350 = expf(gate_x_389_1 * -1.0f);
                            float denominator_x_393_1 = _exp_350 + 1.0f;
                            float _exp_351 = expf(gate_y_390_1 * -1.0f);
                            float denominator_y_394_1 = _exp_351 + 1.0f;
                            float hidden_x_395_1 = gate_x_389_1 / denominator_x_393_1 * up_x_391_1;
                            float hidden_y_396_1 = gate_y_390_1 / denominator_y_394_1 * up_y_392_1;
                            __nv_bfloat162 _bf16x2_561 = __float22bfloat162_rn(make_float2(hidden_x_395_1, hidden_y_396_1));
                            hidden_packed_264_1[15] = __as_u32(_bf16x2_561);
                            float2 _cvt_f32_352 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[16]));
                            float2 _cvt_f32_353 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[16]));
                            float gate_x_397_1 = _cvt_f32_352.x;
                            float gate_y_398_1 = _cvt_f32_352.y;
                            float up_x_399_1 = _cvt_f32_353.x;
                            float up_y_400_1 = _cvt_f32_353.y;
                            float _exp_352 = expf(gate_x_397_1 * -1.0f);
                            float denominator_x_401_1 = _exp_352 + 1.0f;
                            float _exp_353 = expf(gate_y_398_1 * -1.0f);
                            float denominator_y_402_1 = _exp_353 + 1.0f;
                            float hidden_x_403_1 = gate_x_397_1 / denominator_x_401_1 * up_x_399_1;
                            float hidden_y_404_1 = gate_y_398_1 / denominator_y_402_1 * up_y_400_1;
                            __nv_bfloat162 _bf16x2_562 = __float22bfloat162_rn(make_float2(hidden_x_403_1, hidden_y_404_1));
                            hidden_packed_264_1[16] = __as_u32(_bf16x2_562);
                            float2 _cvt_f32_354 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[17]));
                            float2 _cvt_f32_355 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[17]));
                            float gate_x_405_1 = _cvt_f32_354.x;
                            float gate_y_406_1 = _cvt_f32_354.y;
                            float up_x_407_1 = _cvt_f32_355.x;
                            float up_y_408_1 = _cvt_f32_355.y;
                            float _exp_354 = expf(gate_x_405_1 * -1.0f);
                            float denominator_x_409_1 = _exp_354 + 1.0f;
                            float _exp_355 = expf(gate_y_406_1 * -1.0f);
                            float denominator_y_410_1 = _exp_355 + 1.0f;
                            float hidden_x_411_1 = gate_x_405_1 / denominator_x_409_1 * up_x_407_1;
                            float hidden_y_412_1 = gate_y_406_1 / denominator_y_410_1 * up_y_408_1;
                            __nv_bfloat162 _bf16x2_563 = __float22bfloat162_rn(make_float2(hidden_x_411_1, hidden_y_412_1));
                            hidden_packed_264_1[17] = __as_u32(_bf16x2_563);
                            float2 _cvt_f32_356 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[18]));
                            float2 _cvt_f32_357 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[18]));
                            float gate_x_413_1 = _cvt_f32_356.x;
                            float gate_y_414_1 = _cvt_f32_356.y;
                            float up_x_415_1 = _cvt_f32_357.x;
                            float up_y_416_1 = _cvt_f32_357.y;
                            float _exp_356 = expf(gate_x_413_1 * -1.0f);
                            float denominator_x_417_1 = _exp_356 + 1.0f;
                            float _exp_357 = expf(gate_y_414_1 * -1.0f);
                            float denominator_y_418_1 = _exp_357 + 1.0f;
                            float hidden_x_419_1 = gate_x_413_1 / denominator_x_417_1 * up_x_415_1;
                            float hidden_y_420_1 = gate_y_414_1 / denominator_y_418_1 * up_y_416_1;
                            __nv_bfloat162 _bf16x2_564 = __float22bfloat162_rn(make_float2(hidden_x_419_1, hidden_y_420_1));
                            hidden_packed_264_1[18] = __as_u32(_bf16x2_564);
                            float2 _cvt_f32_358 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[19]));
                            float2 _cvt_f32_359 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[19]));
                            float gate_x_421_1 = _cvt_f32_358.x;
                            float gate_y_422_1 = _cvt_f32_358.y;
                            float up_x_423_1 = _cvt_f32_359.x;
                            float up_y_424_1 = _cvt_f32_359.y;
                            float _exp_358 = expf(gate_x_421_1 * -1.0f);
                            float denominator_x_425_1 = _exp_358 + 1.0f;
                            float _exp_359 = expf(gate_y_422_1 * -1.0f);
                            float denominator_y_426_1 = _exp_359 + 1.0f;
                            float hidden_x_427_1 = gate_x_421_1 / denominator_x_425_1 * up_x_423_1;
                            float hidden_y_428_1 = gate_y_422_1 / denominator_y_426_1 * up_y_424_1;
                            __nv_bfloat162 _bf16x2_565 = __float22bfloat162_rn(make_float2(hidden_x_427_1, hidden_y_428_1));
                            hidden_packed_264_1[19] = __as_u32(_bf16x2_565);
                            float2 _cvt_f32_360 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[20]));
                            float2 _cvt_f32_361 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[20]));
                            float gate_x_429_1 = _cvt_f32_360.x;
                            float gate_y_430_1 = _cvt_f32_360.y;
                            float up_x_431_1 = _cvt_f32_361.x;
                            float up_y_432_1 = _cvt_f32_361.y;
                            float _exp_360 = expf(gate_x_429_1 * -1.0f);
                            float denominator_x_433_1 = _exp_360 + 1.0f;
                            float _exp_361 = expf(gate_y_430_1 * -1.0f);
                            float denominator_y_434_1 = _exp_361 + 1.0f;
                            float hidden_x_435_1 = gate_x_429_1 / denominator_x_433_1 * up_x_431_1;
                            float hidden_y_436_1 = gate_y_430_1 / denominator_y_434_1 * up_y_432_1;
                            __nv_bfloat162 _bf16x2_566 = __float22bfloat162_rn(make_float2(hidden_x_435_1, hidden_y_436_1));
                            hidden_packed_264_1[20] = __as_u32(_bf16x2_566);
                            float2 _cvt_f32_362 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[21]));
                            float2 _cvt_f32_363 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[21]));
                            float gate_x_437_1 = _cvt_f32_362.x;
                            float gate_y_438_1 = _cvt_f32_362.y;
                            float up_x_439_1 = _cvt_f32_363.x;
                            float up_y_440_1 = _cvt_f32_363.y;
                            float _exp_362 = expf(gate_x_437_1 * -1.0f);
                            float denominator_x_441_1 = _exp_362 + 1.0f;
                            float _exp_363 = expf(gate_y_438_1 * -1.0f);
                            float denominator_y_442_1 = _exp_363 + 1.0f;
                            float hidden_x_443_1 = gate_x_437_1 / denominator_x_441_1 * up_x_439_1;
                            float hidden_y_444_1 = gate_y_438_1 / denominator_y_442_1 * up_y_440_1;
                            __nv_bfloat162 _bf16x2_567 = __float22bfloat162_rn(make_float2(hidden_x_443_1, hidden_y_444_1));
                            hidden_packed_264_1[21] = __as_u32(_bf16x2_567);
                            float2 _cvt_f32_364 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[22]));
                            float2 _cvt_f32_365 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[22]));
                            float gate_x_445_1 = _cvt_f32_364.x;
                            float gate_y_446_1 = _cvt_f32_364.y;
                            float up_x_447_1 = _cvt_f32_365.x;
                            float up_y_448_1 = _cvt_f32_365.y;
                            float _exp_364 = expf(gate_x_445_1 * -1.0f);
                            float denominator_x_449_1 = _exp_364 + 1.0f;
                            float _exp_365 = expf(gate_y_446_1 * -1.0f);
                            float denominator_y_450_1 = _exp_365 + 1.0f;
                            float hidden_x_451_1 = gate_x_445_1 / denominator_x_449_1 * up_x_447_1;
                            float hidden_y_452_1 = gate_y_446_1 / denominator_y_450_1 * up_y_448_1;
                            __nv_bfloat162 _bf16x2_568 = __float22bfloat162_rn(make_float2(hidden_x_451_1, hidden_y_452_1));
                            hidden_packed_264_1[22] = __as_u32(_bf16x2_568);
                            float2 _cvt_f32_366 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[23]));
                            float2 _cvt_f32_367 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[23]));
                            float gate_x_453_1 = _cvt_f32_366.x;
                            float gate_y_454_1 = _cvt_f32_366.y;
                            float up_x_455_1 = _cvt_f32_367.x;
                            float up_y_456_1 = _cvt_f32_367.y;
                            float _exp_366 = expf(gate_x_453_1 * -1.0f);
                            float denominator_x_457_1 = _exp_366 + 1.0f;
                            float _exp_367 = expf(gate_y_454_1 * -1.0f);
                            float denominator_y_458_1 = _exp_367 + 1.0f;
                            float hidden_x_459_1 = gate_x_453_1 / denominator_x_457_1 * up_x_455_1;
                            float hidden_y_460_1 = gate_y_454_1 / denominator_y_458_1 * up_y_456_1;
                            __nv_bfloat162 _bf16x2_569 = __float22bfloat162_rn(make_float2(hidden_x_459_1, hidden_y_460_1));
                            hidden_packed_264_1[23] = __as_u32(_bf16x2_569);
                            float2 _cvt_f32_368 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[24]));
                            float2 _cvt_f32_369 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[24]));
                            float gate_x_461_1 = _cvt_f32_368.x;
                            float gate_y_462_1 = _cvt_f32_368.y;
                            float up_x_463_1 = _cvt_f32_369.x;
                            float up_y_464_1 = _cvt_f32_369.y;
                            float _exp_368 = expf(gate_x_461_1 * -1.0f);
                            float denominator_x_465_1 = _exp_368 + 1.0f;
                            float _exp_369 = expf(gate_y_462_1 * -1.0f);
                            float denominator_y_466_1 = _exp_369 + 1.0f;
                            float hidden_x_467_1 = gate_x_461_1 / denominator_x_465_1 * up_x_463_1;
                            float hidden_y_468_1 = gate_y_462_1 / denominator_y_466_1 * up_y_464_1;
                            __nv_bfloat162 _bf16x2_570 = __float22bfloat162_rn(make_float2(hidden_x_467_1, hidden_y_468_1));
                            hidden_packed_264_1[24] = __as_u32(_bf16x2_570);
                            float2 _cvt_f32_370 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[25]));
                            float2 _cvt_f32_371 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[25]));
                            float gate_x_469_1 = _cvt_f32_370.x;
                            float gate_y_470_1 = _cvt_f32_370.y;
                            float up_x_471_1 = _cvt_f32_371.x;
                            float up_y_472_1 = _cvt_f32_371.y;
                            float _exp_370 = expf(gate_x_469_1 * -1.0f);
                            float denominator_x_473_1 = _exp_370 + 1.0f;
                            float _exp_371 = expf(gate_y_470_1 * -1.0f);
                            float denominator_y_474_1 = _exp_371 + 1.0f;
                            float hidden_x_475_1 = gate_x_469_1 / denominator_x_473_1 * up_x_471_1;
                            float hidden_y_476_1 = gate_y_470_1 / denominator_y_474_1 * up_y_472_1;
                            __nv_bfloat162 _bf16x2_571 = __float22bfloat162_rn(make_float2(hidden_x_475_1, hidden_y_476_1));
                            hidden_packed_264_1[25] = __as_u32(_bf16x2_571);
                            float2 _cvt_f32_372 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[26]));
                            float2 _cvt_f32_373 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[26]));
                            float gate_x_477_1 = _cvt_f32_372.x;
                            float gate_y_478_1 = _cvt_f32_372.y;
                            float up_x_479_1 = _cvt_f32_373.x;
                            float up_y_480_1 = _cvt_f32_373.y;
                            float _exp_372 = expf(gate_x_477_1 * -1.0f);
                            float denominator_x_481_1 = _exp_372 + 1.0f;
                            float _exp_373 = expf(gate_y_478_1 * -1.0f);
                            float denominator_y_482_1 = _exp_373 + 1.0f;
                            float hidden_x_483_1 = gate_x_477_1 / denominator_x_481_1 * up_x_479_1;
                            float hidden_y_484_1 = gate_y_478_1 / denominator_y_482_1 * up_y_480_1;
                            __nv_bfloat162 _bf16x2_572 = __float22bfloat162_rn(make_float2(hidden_x_483_1, hidden_y_484_1));
                            hidden_packed_264_1[26] = __as_u32(_bf16x2_572);
                            float2 _cvt_f32_374 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[27]));
                            float2 _cvt_f32_375 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[27]));
                            float gate_x_485_1 = _cvt_f32_374.x;
                            float gate_y_486_1 = _cvt_f32_374.y;
                            float up_x_487_1 = _cvt_f32_375.x;
                            float up_y_488_1 = _cvt_f32_375.y;
                            float _exp_374 = expf(gate_x_485_1 * -1.0f);
                            float denominator_x_489_1 = _exp_374 + 1.0f;
                            float _exp_375 = expf(gate_y_486_1 * -1.0f);
                            float denominator_y_490_1 = _exp_375 + 1.0f;
                            float hidden_x_491_1 = gate_x_485_1 / denominator_x_489_1 * up_x_487_1;
                            float hidden_y_492_1 = gate_y_486_1 / denominator_y_490_1 * up_y_488_1;
                            __nv_bfloat162 _bf16x2_573 = __float22bfloat162_rn(make_float2(hidden_x_491_1, hidden_y_492_1));
                            hidden_packed_264_1[27] = __as_u32(_bf16x2_573);
                            float2 _cvt_f32_376 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[28]));
                            float2 _cvt_f32_377 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[28]));
                            float gate_x_493_1 = _cvt_f32_376.x;
                            float gate_y_494_1 = _cvt_f32_376.y;
                            float up_x_495_1 = _cvt_f32_377.x;
                            float up_y_496_1 = _cvt_f32_377.y;
                            float _exp_376 = expf(gate_x_493_1 * -1.0f);
                            float denominator_x_497_1 = _exp_376 + 1.0f;
                            float _exp_377 = expf(gate_y_494_1 * -1.0f);
                            float denominator_y_498_1 = _exp_377 + 1.0f;
                            float hidden_x_499_1 = gate_x_493_1 / denominator_x_497_1 * up_x_495_1;
                            float hidden_y_500_1 = gate_y_494_1 / denominator_y_498_1 * up_y_496_1;
                            __nv_bfloat162 _bf16x2_574 = __float22bfloat162_rn(make_float2(hidden_x_499_1, hidden_y_500_1));
                            hidden_packed_264_1[28] = __as_u32(_bf16x2_574);
                            float2 _cvt_f32_378 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[29]));
                            float2 _cvt_f32_379 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[29]));
                            float gate_x_501_1 = _cvt_f32_378.x;
                            float gate_y_502_1 = _cvt_f32_378.y;
                            float up_x_503_1 = _cvt_f32_379.x;
                            float up_y_504_1 = _cvt_f32_379.y;
                            float _exp_378 = expf(gate_x_501_1 * -1.0f);
                            float denominator_x_505_1 = _exp_378 + 1.0f;
                            float _exp_379 = expf(gate_y_502_1 * -1.0f);
                            float denominator_y_506_1 = _exp_379 + 1.0f;
                            float hidden_x_507_1 = gate_x_501_1 / denominator_x_505_1 * up_x_503_1;
                            float hidden_y_508_1 = gate_y_502_1 / denominator_y_506_1 * up_y_504_1;
                            __nv_bfloat162 _bf16x2_575 = __float22bfloat162_rn(make_float2(hidden_x_507_1, hidden_y_508_1));
                            hidden_packed_264_1[29] = __as_u32(_bf16x2_575);
                            float2 _cvt_f32_380 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[30]));
                            float2 _cvt_f32_381 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[30]));
                            float gate_x_509_1 = _cvt_f32_380.x;
                            float gate_y_510_1 = _cvt_f32_380.y;
                            float up_x_511_1 = _cvt_f32_381.x;
                            float up_y_512_1 = _cvt_f32_381.y;
                            float _exp_380 = expf(gate_x_509_1 * -1.0f);
                            float denominator_x_513_1 = _exp_380 + 1.0f;
                            float _exp_381 = expf(gate_y_510_1 * -1.0f);
                            float denominator_y_514_1 = _exp_381 + 1.0f;
                            float hidden_x_515_1 = gate_x_509_1 / denominator_x_513_1 * up_x_511_1;
                            float hidden_y_516_1 = gate_y_510_1 / denominator_y_514_1 * up_y_512_1;
                            __nv_bfloat162 _bf16x2_576 = __float22bfloat162_rn(make_float2(hidden_x_515_1, hidden_y_516_1));
                            hidden_packed_264_1[30] = __as_u32(_bf16x2_576);
                            float2 _cvt_f32_382 = __bfloat1622float2(__as_bf16x2(gate_packed_262_1[31]));
                            float2 _cvt_f32_383 = __bfloat1622float2(__as_bf16x2(up_packed_263_1[31]));
                            float gate_x_517_1 = _cvt_f32_382.x;
                            float gate_y_518_1 = _cvt_f32_382.y;
                            float up_x_519_1 = _cvt_f32_383.x;
                            float up_y_520_1 = _cvt_f32_383.y;
                            float _exp_382 = expf(gate_x_517_1 * -1.0f);
                            float denominator_x_521_1 = _exp_382 + 1.0f;
                            float _exp_383 = expf(gate_y_518_1 * -1.0f);
                            float denominator_y_522_1 = _exp_383 + 1.0f;
                            float hidden_x_523_1 = gate_x_517_1 / denominator_x_521_1 * up_x_519_1;
                            float hidden_y_524_1 = gate_y_518_1 / denominator_y_522_1 * up_y_520_1;
                            __nv_bfloat162 _bf16x2_577 = __float22bfloat162_rn(make_float2(hidden_x_523_1, hidden_y_524_1));
                            hidden_packed_264_1[31] = __as_u32(_bf16x2_577);
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_525_1 = tid / 32;
                            int lane_526_1 = tid % 32;
                            #pragma unroll
                            for (int half_32 = 0; half_32 < 2; half_32++) {
                                #pragma unroll
                                for (int col_tile_32 = 0; col_tile_32 < 2; col_tile_32++) {
                                    int row_38 = warp_525_1 * 32 + half_32 * 16 + lane_526_1 % 16;
                                    int col_35 = col_tile_32 * 16 + lane_526_1 / 16 * 8;
                                    unsigned int address_3_30 = d_smem_addr + (unsigned int)((row_38 * 32 + col_35) * 2);
                                    address_3_30 = address_3_30 ^ (address_3_30 & 511) >> 7 << 4;
                                    int offset_33 = half_32 * 8 + col_tile_32 * 4;
                                    uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(address_3_30);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_33])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_33 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_33 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_33 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_527_1 = tid / 32;
                            int lane_528_1 = tid % 32;
                            #pragma unroll
                            for (int half_33 = 0; half_33 < 2; half_33++) {
                                #pragma unroll
                                for (int col_tile_33 = 0; col_tile_33 < 2; col_tile_33++) {
                                    int row_39 = warp_527_1 * 32 + half_33 * 16 + lane_528_1 % 16;
                                    int col_36 = col_tile_33 * 16 + lane_528_1 / 16 * 8;
                                    unsigned int address_3_31 = d_smem_addr + 8192 + (unsigned int)((row_39 * 32 + col_36) * 2);
                                    address_3_31 = address_3_31 ^ (address_3_31 & 511) >> 7 << 4;
                                    int offset_34 = half_33 * 8 + col_tile_33 * 4;
                                    uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(address_3_31);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_34])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_34 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_34 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_34 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_529_1 = tid / 32;
                            int lane_530_1 = tid % 32;
                            #pragma unroll
                            for (int half_34 = 0; half_34 < 2; half_34++) {
                                #pragma unroll
                                for (int col_tile_34 = 0; col_tile_34 < 2; col_tile_34++) {
                                    int row_40 = warp_529_1 * 32 + half_34 * 16 + lane_530_1 % 16;
                                    int col_37 = col_tile_34 * 16 + lane_530_1 / 16 * 8;
                                    unsigned int address_3_32 = d_smem_addr + 16384 + (unsigned int)((row_40 * 32 + col_37) * 2);
                                    address_3_32 = address_3_32 ^ (address_3_32 & 511) >> 7 << 4;
                                    int offset_35 = half_34 * 8 + col_tile_34 * 4;
                                    uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(address_3_32);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_35])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_35 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_35 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_35 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_531_1 = tid / 32;
                            int lane_532_1 = tid % 32;
                            #pragma unroll
                            for (int half_35 = 0; half_35 < 2; half_35++) {
                                #pragma unroll
                                for (int col_tile_35 = 0; col_tile_35 < 2; col_tile_35++) {
                                    int row_41 = warp_531_1 * 32 + half_35 * 16 + lane_532_1 % 16;
                                    int col_38 = col_tile_35 * 16 + lane_532_1 / 16 * 8;
                                    unsigned int address_3_33 = d_smem_addr + (unsigned int)((row_41 * 32 + col_38) * 2);
                                    address_3_33 = address_3_33 ^ (address_3_33 & 511) >> 7 << 4;
                                    int offset_36 = 16 + half_35 * 8 + col_tile_35 * 4;
                                    uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(address_3_33);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_36])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_36 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_36 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_262_1[offset_36 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_533_1 = tid / 32;
                            int lane_534_1 = tid % 32;
                            #pragma unroll
                            for (int half_36 = 0; half_36 < 2; half_36++) {
                                #pragma unroll
                                for (int col_tile_36 = 0; col_tile_36 < 2; col_tile_36++) {
                                    int row_42 = warp_533_1 * 32 + half_36 * 16 + lane_534_1 % 16;
                                    int col_39 = col_tile_36 * 16 + lane_534_1 / 16 * 8;
                                    unsigned int address_3_34 = d_smem_addr + 8192 + (unsigned int)((row_42 * 32 + col_39) * 2);
                                    address_3_34 = address_3_34 ^ (address_3_34 & 511) >> 7 << 4;
                                    int offset_37 = 16 + half_36 * 8 + col_tile_36 * 4;
                                    uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(address_3_34);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_37])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_37 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_37 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_263_1[offset_37 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_535_1 = tid / 32;
                            int lane_536_1 = tid % 32;
                            #pragma unroll
                            for (int half_37 = 0; half_37 < 2; half_37++) {
                                #pragma unroll
                                for (int col_tile_37 = 0; col_tile_37 < 2; col_tile_37++) {
                                    int row_43 = warp_535_1 * 32 + half_37 * 16 + lane_536_1 % 16;
                                    int col_40 = col_tile_37 * 16 + lane_536_1 / 16 * 8;
                                    unsigned int address_3_35 = d_smem_addr + 16384 + (unsigned int)((row_43 * 32 + col_40) * 2);
                                    address_3_35 = address_3_35 ^ (address_3_35 & 511) >> 7 << 4;
                                    int offset_38 = 16 + half_37 * 8 + col_tile_37 * 4;
                                    uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(address_3_35);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_38])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_38 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_38 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_264_1[offset_38 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int gate_packed_537_1[32];
                            unsigned int up_packed_538_1[32];
                            unsigned int hidden_packed_539_1[32];
                            unsigned int address_540_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 128;
                            float _tmem_load_50[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_50[15]))
                                : "r"(address_540_1));
                            float _tmem_load_51[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_51[15]))
                                : "r"(address_540_1 + 256));
                            __nv_bfloat162 _bf16x2_578 = __float22bfloat162_rn(make_float2(_tmem_load_50[0], _tmem_load_50[1]));
                            gate_packed_537_1[0] = __as_u32(_bf16x2_578);
                            __nv_bfloat162 _bf16x2_579 = __float22bfloat162_rn(make_float2(_tmem_load_51[0], _tmem_load_51[1]));
                            up_packed_538_1[0] = __as_u32(_bf16x2_579);
                            __nv_bfloat162 _bf16x2_580 = __float22bfloat162_rn(make_float2(_tmem_load_50[2], _tmem_load_50[3]));
                            gate_packed_537_1[1] = __as_u32(_bf16x2_580);
                            __nv_bfloat162 _bf16x2_581 = __float22bfloat162_rn(make_float2(_tmem_load_51[2], _tmem_load_51[3]));
                            up_packed_538_1[1] = __as_u32(_bf16x2_581);
                            __nv_bfloat162 _bf16x2_582 = __float22bfloat162_rn(make_float2(_tmem_load_50[4], _tmem_load_50[5]));
                            gate_packed_537_1[2] = __as_u32(_bf16x2_582);
                            __nv_bfloat162 _bf16x2_583 = __float22bfloat162_rn(make_float2(_tmem_load_51[4], _tmem_load_51[5]));
                            up_packed_538_1[2] = __as_u32(_bf16x2_583);
                            __nv_bfloat162 _bf16x2_584 = __float22bfloat162_rn(make_float2(_tmem_load_50[6], _tmem_load_50[7]));
                            gate_packed_537_1[3] = __as_u32(_bf16x2_584);
                            __nv_bfloat162 _bf16x2_585 = __float22bfloat162_rn(make_float2(_tmem_load_51[6], _tmem_load_51[7]));
                            up_packed_538_1[3] = __as_u32(_bf16x2_585);
                            __nv_bfloat162 _bf16x2_586 = __float22bfloat162_rn(make_float2(_tmem_load_50[8], _tmem_load_50[9]));
                            gate_packed_537_1[4] = __as_u32(_bf16x2_586);
                            __nv_bfloat162 _bf16x2_587 = __float22bfloat162_rn(make_float2(_tmem_load_51[8], _tmem_load_51[9]));
                            up_packed_538_1[4] = __as_u32(_bf16x2_587);
                            __nv_bfloat162 _bf16x2_588 = __float22bfloat162_rn(make_float2(_tmem_load_50[10], _tmem_load_50[11]));
                            gate_packed_537_1[5] = __as_u32(_bf16x2_588);
                            __nv_bfloat162 _bf16x2_589 = __float22bfloat162_rn(make_float2(_tmem_load_51[10], _tmem_load_51[11]));
                            up_packed_538_1[5] = __as_u32(_bf16x2_589);
                            __nv_bfloat162 _bf16x2_590 = __float22bfloat162_rn(make_float2(_tmem_load_50[12], _tmem_load_50[13]));
                            gate_packed_537_1[6] = __as_u32(_bf16x2_590);
                            __nv_bfloat162 _bf16x2_591 = __float22bfloat162_rn(make_float2(_tmem_load_51[12], _tmem_load_51[13]));
                            up_packed_538_1[6] = __as_u32(_bf16x2_591);
                            __nv_bfloat162 _bf16x2_592 = __float22bfloat162_rn(make_float2(_tmem_load_50[14], _tmem_load_50[15]));
                            gate_packed_537_1[7] = __as_u32(_bf16x2_592);
                            __nv_bfloat162 _bf16x2_593 = __float22bfloat162_rn(make_float2(_tmem_load_51[14], _tmem_load_51[15]));
                            up_packed_538_1[7] = __as_u32(_bf16x2_593);
                            unsigned int address_541_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 128;
                            float _tmem_load_52[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_52[15]))
                                : "r"(address_541_1));
                            float _tmem_load_53[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_53[15]))
                                : "r"(address_541_1 + 256));
                            __nv_bfloat162 _bf16x2_594 = __float22bfloat162_rn(make_float2(_tmem_load_52[0], _tmem_load_52[1]));
                            gate_packed_537_1[8] = __as_u32(_bf16x2_594);
                            __nv_bfloat162 _bf16x2_595 = __float22bfloat162_rn(make_float2(_tmem_load_53[0], _tmem_load_53[1]));
                            up_packed_538_1[8] = __as_u32(_bf16x2_595);
                            __nv_bfloat162 _bf16x2_596 = __float22bfloat162_rn(make_float2(_tmem_load_52[2], _tmem_load_52[3]));
                            gate_packed_537_1[9] = __as_u32(_bf16x2_596);
                            __nv_bfloat162 _bf16x2_597 = __float22bfloat162_rn(make_float2(_tmem_load_53[2], _tmem_load_53[3]));
                            up_packed_538_1[9] = __as_u32(_bf16x2_597);
                            __nv_bfloat162 _bf16x2_598 = __float22bfloat162_rn(make_float2(_tmem_load_52[4], _tmem_load_52[5]));
                            gate_packed_537_1[10] = __as_u32(_bf16x2_598);
                            __nv_bfloat162 _bf16x2_599 = __float22bfloat162_rn(make_float2(_tmem_load_53[4], _tmem_load_53[5]));
                            up_packed_538_1[10] = __as_u32(_bf16x2_599);
                            __nv_bfloat162 _bf16x2_600 = __float22bfloat162_rn(make_float2(_tmem_load_52[6], _tmem_load_52[7]));
                            gate_packed_537_1[11] = __as_u32(_bf16x2_600);
                            __nv_bfloat162 _bf16x2_601 = __float22bfloat162_rn(make_float2(_tmem_load_53[6], _tmem_load_53[7]));
                            up_packed_538_1[11] = __as_u32(_bf16x2_601);
                            __nv_bfloat162 _bf16x2_602 = __float22bfloat162_rn(make_float2(_tmem_load_52[8], _tmem_load_52[9]));
                            gate_packed_537_1[12] = __as_u32(_bf16x2_602);
                            __nv_bfloat162 _bf16x2_603 = __float22bfloat162_rn(make_float2(_tmem_load_53[8], _tmem_load_53[9]));
                            up_packed_538_1[12] = __as_u32(_bf16x2_603);
                            __nv_bfloat162 _bf16x2_604 = __float22bfloat162_rn(make_float2(_tmem_load_52[10], _tmem_load_52[11]));
                            gate_packed_537_1[13] = __as_u32(_bf16x2_604);
                            __nv_bfloat162 _bf16x2_605 = __float22bfloat162_rn(make_float2(_tmem_load_53[10], _tmem_load_53[11]));
                            up_packed_538_1[13] = __as_u32(_bf16x2_605);
                            __nv_bfloat162 _bf16x2_606 = __float22bfloat162_rn(make_float2(_tmem_load_52[12], _tmem_load_52[13]));
                            gate_packed_537_1[14] = __as_u32(_bf16x2_606);
                            __nv_bfloat162 _bf16x2_607 = __float22bfloat162_rn(make_float2(_tmem_load_53[12], _tmem_load_53[13]));
                            up_packed_538_1[14] = __as_u32(_bf16x2_607);
                            __nv_bfloat162 _bf16x2_608 = __float22bfloat162_rn(make_float2(_tmem_load_52[14], _tmem_load_52[15]));
                            gate_packed_537_1[15] = __as_u32(_bf16x2_608);
                            __nv_bfloat162 _bf16x2_609 = __float22bfloat162_rn(make_float2(_tmem_load_53[14], _tmem_load_53[15]));
                            up_packed_538_1[15] = __as_u32(_bf16x2_609);
                            unsigned int address_542_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 160;
                            float _tmem_load_54[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_54[15]))
                                : "r"(address_542_1));
                            float _tmem_load_55[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_55[15]))
                                : "r"(address_542_1 + 256));
                            __nv_bfloat162 _bf16x2_610 = __float22bfloat162_rn(make_float2(_tmem_load_54[0], _tmem_load_54[1]));
                            gate_packed_537_1[16] = __as_u32(_bf16x2_610);
                            __nv_bfloat162 _bf16x2_611 = __float22bfloat162_rn(make_float2(_tmem_load_55[0], _tmem_load_55[1]));
                            up_packed_538_1[16] = __as_u32(_bf16x2_611);
                            __nv_bfloat162 _bf16x2_612 = __float22bfloat162_rn(make_float2(_tmem_load_54[2], _tmem_load_54[3]));
                            gate_packed_537_1[17] = __as_u32(_bf16x2_612);
                            __nv_bfloat162 _bf16x2_613 = __float22bfloat162_rn(make_float2(_tmem_load_55[2], _tmem_load_55[3]));
                            up_packed_538_1[17] = __as_u32(_bf16x2_613);
                            __nv_bfloat162 _bf16x2_614 = __float22bfloat162_rn(make_float2(_tmem_load_54[4], _tmem_load_54[5]));
                            gate_packed_537_1[18] = __as_u32(_bf16x2_614);
                            __nv_bfloat162 _bf16x2_615 = __float22bfloat162_rn(make_float2(_tmem_load_55[4], _tmem_load_55[5]));
                            up_packed_538_1[18] = __as_u32(_bf16x2_615);
                            __nv_bfloat162 _bf16x2_616 = __float22bfloat162_rn(make_float2(_tmem_load_54[6], _tmem_load_54[7]));
                            gate_packed_537_1[19] = __as_u32(_bf16x2_616);
                            __nv_bfloat162 _bf16x2_617 = __float22bfloat162_rn(make_float2(_tmem_load_55[6], _tmem_load_55[7]));
                            up_packed_538_1[19] = __as_u32(_bf16x2_617);
                            __nv_bfloat162 _bf16x2_618 = __float22bfloat162_rn(make_float2(_tmem_load_54[8], _tmem_load_54[9]));
                            gate_packed_537_1[20] = __as_u32(_bf16x2_618);
                            __nv_bfloat162 _bf16x2_619 = __float22bfloat162_rn(make_float2(_tmem_load_55[8], _tmem_load_55[9]));
                            up_packed_538_1[20] = __as_u32(_bf16x2_619);
                            __nv_bfloat162 _bf16x2_620 = __float22bfloat162_rn(make_float2(_tmem_load_54[10], _tmem_load_54[11]));
                            gate_packed_537_1[21] = __as_u32(_bf16x2_620);
                            __nv_bfloat162 _bf16x2_621 = __float22bfloat162_rn(make_float2(_tmem_load_55[10], _tmem_load_55[11]));
                            up_packed_538_1[21] = __as_u32(_bf16x2_621);
                            __nv_bfloat162 _bf16x2_622 = __float22bfloat162_rn(make_float2(_tmem_load_54[12], _tmem_load_54[13]));
                            gate_packed_537_1[22] = __as_u32(_bf16x2_622);
                            __nv_bfloat162 _bf16x2_623 = __float22bfloat162_rn(make_float2(_tmem_load_55[12], _tmem_load_55[13]));
                            up_packed_538_1[22] = __as_u32(_bf16x2_623);
                            __nv_bfloat162 _bf16x2_624 = __float22bfloat162_rn(make_float2(_tmem_load_54[14], _tmem_load_54[15]));
                            gate_packed_537_1[23] = __as_u32(_bf16x2_624);
                            __nv_bfloat162 _bf16x2_625 = __float22bfloat162_rn(make_float2(_tmem_load_55[14], _tmem_load_55[15]));
                            up_packed_538_1[23] = __as_u32(_bf16x2_625);
                            unsigned int address_543_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 160;
                            float _tmem_load_56[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_56[15]))
                                : "r"(address_543_1));
                            float _tmem_load_57[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_57[15]))
                                : "r"(address_543_1 + 256));
                            __nv_bfloat162 _bf16x2_626 = __float22bfloat162_rn(make_float2(_tmem_load_56[0], _tmem_load_56[1]));
                            gate_packed_537_1[24] = __as_u32(_bf16x2_626);
                            __nv_bfloat162 _bf16x2_627 = __float22bfloat162_rn(make_float2(_tmem_load_57[0], _tmem_load_57[1]));
                            up_packed_538_1[24] = __as_u32(_bf16x2_627);
                            __nv_bfloat162 _bf16x2_628 = __float22bfloat162_rn(make_float2(_tmem_load_56[2], _tmem_load_56[3]));
                            gate_packed_537_1[25] = __as_u32(_bf16x2_628);
                            __nv_bfloat162 _bf16x2_629 = __float22bfloat162_rn(make_float2(_tmem_load_57[2], _tmem_load_57[3]));
                            up_packed_538_1[25] = __as_u32(_bf16x2_629);
                            __nv_bfloat162 _bf16x2_630 = __float22bfloat162_rn(make_float2(_tmem_load_56[4], _tmem_load_56[5]));
                            gate_packed_537_1[26] = __as_u32(_bf16x2_630);
                            __nv_bfloat162 _bf16x2_631 = __float22bfloat162_rn(make_float2(_tmem_load_57[4], _tmem_load_57[5]));
                            up_packed_538_1[26] = __as_u32(_bf16x2_631);
                            __nv_bfloat162 _bf16x2_632 = __float22bfloat162_rn(make_float2(_tmem_load_56[6], _tmem_load_56[7]));
                            gate_packed_537_1[27] = __as_u32(_bf16x2_632);
                            __nv_bfloat162 _bf16x2_633 = __float22bfloat162_rn(make_float2(_tmem_load_57[6], _tmem_load_57[7]));
                            up_packed_538_1[27] = __as_u32(_bf16x2_633);
                            __nv_bfloat162 _bf16x2_634 = __float22bfloat162_rn(make_float2(_tmem_load_56[8], _tmem_load_56[9]));
                            gate_packed_537_1[28] = __as_u32(_bf16x2_634);
                            __nv_bfloat162 _bf16x2_635 = __float22bfloat162_rn(make_float2(_tmem_load_57[8], _tmem_load_57[9]));
                            up_packed_538_1[28] = __as_u32(_bf16x2_635);
                            __nv_bfloat162 _bf16x2_636 = __float22bfloat162_rn(make_float2(_tmem_load_56[10], _tmem_load_56[11]));
                            gate_packed_537_1[29] = __as_u32(_bf16x2_636);
                            __nv_bfloat162 _bf16x2_637 = __float22bfloat162_rn(make_float2(_tmem_load_57[10], _tmem_load_57[11]));
                            up_packed_538_1[29] = __as_u32(_bf16x2_637);
                            __nv_bfloat162 _bf16x2_638 = __float22bfloat162_rn(make_float2(_tmem_load_56[12], _tmem_load_56[13]));
                            gate_packed_537_1[30] = __as_u32(_bf16x2_638);
                            __nv_bfloat162 _bf16x2_639 = __float22bfloat162_rn(make_float2(_tmem_load_57[12], _tmem_load_57[13]));
                            up_packed_538_1[30] = __as_u32(_bf16x2_639);
                            __nv_bfloat162 _bf16x2_640 = __float22bfloat162_rn(make_float2(_tmem_load_56[14], _tmem_load_56[15]));
                            gate_packed_537_1[31] = __as_u32(_bf16x2_640);
                            __nv_bfloat162 _bf16x2_641 = __float22bfloat162_rn(make_float2(_tmem_load_57[14], _tmem_load_57[15]));
                            up_packed_538_1[31] = __as_u32(_bf16x2_641);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _cvt_f32_384 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[0]));
                            float2 _cvt_f32_385 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[0]));
                            float gate_x_544_1 = _cvt_f32_384.x;
                            float gate_y_545_1 = _cvt_f32_384.y;
                            float up_x_546_1 = _cvt_f32_385.x;
                            float up_y_547_1 = _cvt_f32_385.y;
                            float _exp_384 = expf(gate_x_544_1 * -1.0f);
                            float denominator_x_548_1 = _exp_384 + 1.0f;
                            float _exp_385 = expf(gate_y_545_1 * -1.0f);
                            float denominator_y_549_1 = _exp_385 + 1.0f;
                            float hidden_x_550_1 = gate_x_544_1 / denominator_x_548_1 * up_x_546_1;
                            float hidden_y_551_1 = gate_y_545_1 / denominator_y_549_1 * up_y_547_1;
                            __nv_bfloat162 _bf16x2_642 = __float22bfloat162_rn(make_float2(hidden_x_550_1, hidden_y_551_1));
                            hidden_packed_539_1[0] = __as_u32(_bf16x2_642);
                            float2 _cvt_f32_386 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[1]));
                            float2 _cvt_f32_387 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[1]));
                            float gate_x_552_1 = _cvt_f32_386.x;
                            float gate_y_553_1 = _cvt_f32_386.y;
                            float up_x_554_1 = _cvt_f32_387.x;
                            float up_y_555_1 = _cvt_f32_387.y;
                            float _exp_386 = expf(gate_x_552_1 * -1.0f);
                            float denominator_x_556_1 = _exp_386 + 1.0f;
                            float _exp_387 = expf(gate_y_553_1 * -1.0f);
                            float denominator_y_557_1 = _exp_387 + 1.0f;
                            float hidden_x_558_1 = gate_x_552_1 / denominator_x_556_1 * up_x_554_1;
                            float hidden_y_559_1 = gate_y_553_1 / denominator_y_557_1 * up_y_555_1;
                            __nv_bfloat162 _bf16x2_643 = __float22bfloat162_rn(make_float2(hidden_x_558_1, hidden_y_559_1));
                            hidden_packed_539_1[1] = __as_u32(_bf16x2_643);
                            float2 _cvt_f32_388 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[2]));
                            float2 _cvt_f32_389 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[2]));
                            float gate_x_560_1 = _cvt_f32_388.x;
                            float gate_y_561_1 = _cvt_f32_388.y;
                            float up_x_562_1 = _cvt_f32_389.x;
                            float up_y_563_1 = _cvt_f32_389.y;
                            float _exp_388 = expf(gate_x_560_1 * -1.0f);
                            float denominator_x_564_1 = _exp_388 + 1.0f;
                            float _exp_389 = expf(gate_y_561_1 * -1.0f);
                            float denominator_y_565_1 = _exp_389 + 1.0f;
                            float hidden_x_566_1 = gate_x_560_1 / denominator_x_564_1 * up_x_562_1;
                            float hidden_y_567_1 = gate_y_561_1 / denominator_y_565_1 * up_y_563_1;
                            __nv_bfloat162 _bf16x2_644 = __float22bfloat162_rn(make_float2(hidden_x_566_1, hidden_y_567_1));
                            hidden_packed_539_1[2] = __as_u32(_bf16x2_644);
                            float2 _cvt_f32_390 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[3]));
                            float2 _cvt_f32_391 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[3]));
                            float gate_x_568_1 = _cvt_f32_390.x;
                            float gate_y_569_1 = _cvt_f32_390.y;
                            float up_x_570_1 = _cvt_f32_391.x;
                            float up_y_571_1 = _cvt_f32_391.y;
                            float _exp_390 = expf(gate_x_568_1 * -1.0f);
                            float denominator_x_572_1 = _exp_390 + 1.0f;
                            float _exp_391 = expf(gate_y_569_1 * -1.0f);
                            float denominator_y_573_1 = _exp_391 + 1.0f;
                            float hidden_x_574_1 = gate_x_568_1 / denominator_x_572_1 * up_x_570_1;
                            float hidden_y_575_1 = gate_y_569_1 / denominator_y_573_1 * up_y_571_1;
                            __nv_bfloat162 _bf16x2_645 = __float22bfloat162_rn(make_float2(hidden_x_574_1, hidden_y_575_1));
                            hidden_packed_539_1[3] = __as_u32(_bf16x2_645);
                            float2 _cvt_f32_392 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[4]));
                            float2 _cvt_f32_393 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[4]));
                            float gate_x_576_1 = _cvt_f32_392.x;
                            float gate_y_577_1 = _cvt_f32_392.y;
                            float up_x_578_1 = _cvt_f32_393.x;
                            float up_y_579_1 = _cvt_f32_393.y;
                            float _exp_392 = expf(gate_x_576_1 * -1.0f);
                            float denominator_x_580_1 = _exp_392 + 1.0f;
                            float _exp_393 = expf(gate_y_577_1 * -1.0f);
                            float denominator_y_581_1 = _exp_393 + 1.0f;
                            float hidden_x_582_1 = gate_x_576_1 / denominator_x_580_1 * up_x_578_1;
                            float hidden_y_583_1 = gate_y_577_1 / denominator_y_581_1 * up_y_579_1;
                            __nv_bfloat162 _bf16x2_646 = __float22bfloat162_rn(make_float2(hidden_x_582_1, hidden_y_583_1));
                            hidden_packed_539_1[4] = __as_u32(_bf16x2_646);
                            float2 _cvt_f32_394 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[5]));
                            float2 _cvt_f32_395 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[5]));
                            float gate_x_584_1 = _cvt_f32_394.x;
                            float gate_y_585_1 = _cvt_f32_394.y;
                            float up_x_586_1 = _cvt_f32_395.x;
                            float up_y_587_1 = _cvt_f32_395.y;
                            float _exp_394 = expf(gate_x_584_1 * -1.0f);
                            float denominator_x_588_1 = _exp_394 + 1.0f;
                            float _exp_395 = expf(gate_y_585_1 * -1.0f);
                            float denominator_y_589_1 = _exp_395 + 1.0f;
                            float hidden_x_590_1 = gate_x_584_1 / denominator_x_588_1 * up_x_586_1;
                            float hidden_y_591_1 = gate_y_585_1 / denominator_y_589_1 * up_y_587_1;
                            __nv_bfloat162 _bf16x2_647 = __float22bfloat162_rn(make_float2(hidden_x_590_1, hidden_y_591_1));
                            hidden_packed_539_1[5] = __as_u32(_bf16x2_647);
                            float2 _cvt_f32_396 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[6]));
                            float2 _cvt_f32_397 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[6]));
                            float gate_x_592_1 = _cvt_f32_396.x;
                            float gate_y_593_1 = _cvt_f32_396.y;
                            float up_x_594_1 = _cvt_f32_397.x;
                            float up_y_595_1 = _cvt_f32_397.y;
                            float _exp_396 = expf(gate_x_592_1 * -1.0f);
                            float denominator_x_596_1 = _exp_396 + 1.0f;
                            float _exp_397 = expf(gate_y_593_1 * -1.0f);
                            float denominator_y_597_1 = _exp_397 + 1.0f;
                            float hidden_x_598_1 = gate_x_592_1 / denominator_x_596_1 * up_x_594_1;
                            float hidden_y_599_1 = gate_y_593_1 / denominator_y_597_1 * up_y_595_1;
                            __nv_bfloat162 _bf16x2_648 = __float22bfloat162_rn(make_float2(hidden_x_598_1, hidden_y_599_1));
                            hidden_packed_539_1[6] = __as_u32(_bf16x2_648);
                            float2 _cvt_f32_398 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[7]));
                            float2 _cvt_f32_399 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[7]));
                            float gate_x_600_1 = _cvt_f32_398.x;
                            float gate_y_601_1 = _cvt_f32_398.y;
                            float up_x_602_1 = _cvt_f32_399.x;
                            float up_y_603_1 = _cvt_f32_399.y;
                            float _exp_398 = expf(gate_x_600_1 * -1.0f);
                            float denominator_x_604_1 = _exp_398 + 1.0f;
                            float _exp_399 = expf(gate_y_601_1 * -1.0f);
                            float denominator_y_605_1 = _exp_399 + 1.0f;
                            float hidden_x_606_1 = gate_x_600_1 / denominator_x_604_1 * up_x_602_1;
                            float hidden_y_607_1 = gate_y_601_1 / denominator_y_605_1 * up_y_603_1;
                            __nv_bfloat162 _bf16x2_649 = __float22bfloat162_rn(make_float2(hidden_x_606_1, hidden_y_607_1));
                            hidden_packed_539_1[7] = __as_u32(_bf16x2_649);
                            float2 _cvt_f32_400 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[8]));
                            float2 _cvt_f32_401 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[8]));
                            float gate_x_608_1 = _cvt_f32_400.x;
                            float gate_y_609_1 = _cvt_f32_400.y;
                            float up_x_610_1 = _cvt_f32_401.x;
                            float up_y_611_1 = _cvt_f32_401.y;
                            float _exp_400 = expf(gate_x_608_1 * -1.0f);
                            float denominator_x_612_1 = _exp_400 + 1.0f;
                            float _exp_401 = expf(gate_y_609_1 * -1.0f);
                            float denominator_y_613_1 = _exp_401 + 1.0f;
                            float hidden_x_614_1 = gate_x_608_1 / denominator_x_612_1 * up_x_610_1;
                            float hidden_y_615_1 = gate_y_609_1 / denominator_y_613_1 * up_y_611_1;
                            __nv_bfloat162 _bf16x2_650 = __float22bfloat162_rn(make_float2(hidden_x_614_1, hidden_y_615_1));
                            hidden_packed_539_1[8] = __as_u32(_bf16x2_650);
                            float2 _cvt_f32_402 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[9]));
                            float2 _cvt_f32_403 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[9]));
                            float gate_x_616_1 = _cvt_f32_402.x;
                            float gate_y_617_1 = _cvt_f32_402.y;
                            float up_x_618_1 = _cvt_f32_403.x;
                            float up_y_619_1 = _cvt_f32_403.y;
                            float _exp_402 = expf(gate_x_616_1 * -1.0f);
                            float denominator_x_620_1 = _exp_402 + 1.0f;
                            float _exp_403 = expf(gate_y_617_1 * -1.0f);
                            float denominator_y_621_1 = _exp_403 + 1.0f;
                            float hidden_x_622_1 = gate_x_616_1 / denominator_x_620_1 * up_x_618_1;
                            float hidden_y_623_1 = gate_y_617_1 / denominator_y_621_1 * up_y_619_1;
                            __nv_bfloat162 _bf16x2_651 = __float22bfloat162_rn(make_float2(hidden_x_622_1, hidden_y_623_1));
                            hidden_packed_539_1[9] = __as_u32(_bf16x2_651);
                            float2 _cvt_f32_404 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[10]));
                            float2 _cvt_f32_405 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[10]));
                            float gate_x_624_1 = _cvt_f32_404.x;
                            float gate_y_625_1 = _cvt_f32_404.y;
                            float up_x_626_1 = _cvt_f32_405.x;
                            float up_y_627_1 = _cvt_f32_405.y;
                            float _exp_404 = expf(gate_x_624_1 * -1.0f);
                            float denominator_x_628_1 = _exp_404 + 1.0f;
                            float _exp_405 = expf(gate_y_625_1 * -1.0f);
                            float denominator_y_629_1 = _exp_405 + 1.0f;
                            float hidden_x_630_1 = gate_x_624_1 / denominator_x_628_1 * up_x_626_1;
                            float hidden_y_631_1 = gate_y_625_1 / denominator_y_629_1 * up_y_627_1;
                            __nv_bfloat162 _bf16x2_652 = __float22bfloat162_rn(make_float2(hidden_x_630_1, hidden_y_631_1));
                            hidden_packed_539_1[10] = __as_u32(_bf16x2_652);
                            float2 _cvt_f32_406 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[11]));
                            float2 _cvt_f32_407 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[11]));
                            float gate_x_632_1 = _cvt_f32_406.x;
                            float gate_y_633_1 = _cvt_f32_406.y;
                            float up_x_634_1 = _cvt_f32_407.x;
                            float up_y_635_1 = _cvt_f32_407.y;
                            float _exp_406 = expf(gate_x_632_1 * -1.0f);
                            float denominator_x_636_1 = _exp_406 + 1.0f;
                            float _exp_407 = expf(gate_y_633_1 * -1.0f);
                            float denominator_y_637_1 = _exp_407 + 1.0f;
                            float hidden_x_638_1 = gate_x_632_1 / denominator_x_636_1 * up_x_634_1;
                            float hidden_y_639_1 = gate_y_633_1 / denominator_y_637_1 * up_y_635_1;
                            __nv_bfloat162 _bf16x2_653 = __float22bfloat162_rn(make_float2(hidden_x_638_1, hidden_y_639_1));
                            hidden_packed_539_1[11] = __as_u32(_bf16x2_653);
                            float2 _cvt_f32_408 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[12]));
                            float2 _cvt_f32_409 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[12]));
                            float gate_x_640_1 = _cvt_f32_408.x;
                            float gate_y_641_1 = _cvt_f32_408.y;
                            float up_x_642_1 = _cvt_f32_409.x;
                            float up_y_643_1 = _cvt_f32_409.y;
                            float _exp_408 = expf(gate_x_640_1 * -1.0f);
                            float denominator_x_644_1 = _exp_408 + 1.0f;
                            float _exp_409 = expf(gate_y_641_1 * -1.0f);
                            float denominator_y_645_1 = _exp_409 + 1.0f;
                            float hidden_x_646_1 = gate_x_640_1 / denominator_x_644_1 * up_x_642_1;
                            float hidden_y_647_1 = gate_y_641_1 / denominator_y_645_1 * up_y_643_1;
                            __nv_bfloat162 _bf16x2_654 = __float22bfloat162_rn(make_float2(hidden_x_646_1, hidden_y_647_1));
                            hidden_packed_539_1[12] = __as_u32(_bf16x2_654);
                            float2 _cvt_f32_410 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[13]));
                            float2 _cvt_f32_411 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[13]));
                            float gate_x_648_1 = _cvt_f32_410.x;
                            float gate_y_649_1 = _cvt_f32_410.y;
                            float up_x_650_1 = _cvt_f32_411.x;
                            float up_y_651_1 = _cvt_f32_411.y;
                            float _exp_410 = expf(gate_x_648_1 * -1.0f);
                            float denominator_x_652_1 = _exp_410 + 1.0f;
                            float _exp_411 = expf(gate_y_649_1 * -1.0f);
                            float denominator_y_653_1 = _exp_411 + 1.0f;
                            float hidden_x_654_1 = gate_x_648_1 / denominator_x_652_1 * up_x_650_1;
                            float hidden_y_655_1 = gate_y_649_1 / denominator_y_653_1 * up_y_651_1;
                            __nv_bfloat162 _bf16x2_655 = __float22bfloat162_rn(make_float2(hidden_x_654_1, hidden_y_655_1));
                            hidden_packed_539_1[13] = __as_u32(_bf16x2_655);
                            float2 _cvt_f32_412 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[14]));
                            float2 _cvt_f32_413 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[14]));
                            float gate_x_656_1 = _cvt_f32_412.x;
                            float gate_y_657_1 = _cvt_f32_412.y;
                            float up_x_658_1 = _cvt_f32_413.x;
                            float up_y_659_1 = _cvt_f32_413.y;
                            float _exp_412 = expf(gate_x_656_1 * -1.0f);
                            float denominator_x_660_1 = _exp_412 + 1.0f;
                            float _exp_413 = expf(gate_y_657_1 * -1.0f);
                            float denominator_y_661_1 = _exp_413 + 1.0f;
                            float hidden_x_662_1 = gate_x_656_1 / denominator_x_660_1 * up_x_658_1;
                            float hidden_y_663_1 = gate_y_657_1 / denominator_y_661_1 * up_y_659_1;
                            __nv_bfloat162 _bf16x2_656 = __float22bfloat162_rn(make_float2(hidden_x_662_1, hidden_y_663_1));
                            hidden_packed_539_1[14] = __as_u32(_bf16x2_656);
                            float2 _cvt_f32_414 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[15]));
                            float2 _cvt_f32_415 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[15]));
                            float gate_x_664_1 = _cvt_f32_414.x;
                            float gate_y_665_1 = _cvt_f32_414.y;
                            float up_x_666_1 = _cvt_f32_415.x;
                            float up_y_667_1 = _cvt_f32_415.y;
                            float _exp_414 = expf(gate_x_664_1 * -1.0f);
                            float denominator_x_668_1 = _exp_414 + 1.0f;
                            float _exp_415 = expf(gate_y_665_1 * -1.0f);
                            float denominator_y_669_1 = _exp_415 + 1.0f;
                            float hidden_x_670_1 = gate_x_664_1 / denominator_x_668_1 * up_x_666_1;
                            float hidden_y_671_1 = gate_y_665_1 / denominator_y_669_1 * up_y_667_1;
                            __nv_bfloat162 _bf16x2_657 = __float22bfloat162_rn(make_float2(hidden_x_670_1, hidden_y_671_1));
                            hidden_packed_539_1[15] = __as_u32(_bf16x2_657);
                            float2 _cvt_f32_416 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[16]));
                            float2 _cvt_f32_417 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[16]));
                            float gate_x_672_1 = _cvt_f32_416.x;
                            float gate_y_673_1 = _cvt_f32_416.y;
                            float up_x_674_1 = _cvt_f32_417.x;
                            float up_y_675_1 = _cvt_f32_417.y;
                            float _exp_416 = expf(gate_x_672_1 * -1.0f);
                            float denominator_x_676_1 = _exp_416 + 1.0f;
                            float _exp_417 = expf(gate_y_673_1 * -1.0f);
                            float denominator_y_677_1 = _exp_417 + 1.0f;
                            float hidden_x_678_1 = gate_x_672_1 / denominator_x_676_1 * up_x_674_1;
                            float hidden_y_679_1 = gate_y_673_1 / denominator_y_677_1 * up_y_675_1;
                            __nv_bfloat162 _bf16x2_658 = __float22bfloat162_rn(make_float2(hidden_x_678_1, hidden_y_679_1));
                            hidden_packed_539_1[16] = __as_u32(_bf16x2_658);
                            float2 _cvt_f32_418 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[17]));
                            float2 _cvt_f32_419 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[17]));
                            float gate_x_680_1 = _cvt_f32_418.x;
                            float gate_y_681_1 = _cvt_f32_418.y;
                            float up_x_682_1 = _cvt_f32_419.x;
                            float up_y_683_1 = _cvt_f32_419.y;
                            float _exp_418 = expf(gate_x_680_1 * -1.0f);
                            float denominator_x_684_1 = _exp_418 + 1.0f;
                            float _exp_419 = expf(gate_y_681_1 * -1.0f);
                            float denominator_y_685_1 = _exp_419 + 1.0f;
                            float hidden_x_686_1 = gate_x_680_1 / denominator_x_684_1 * up_x_682_1;
                            float hidden_y_687_1 = gate_y_681_1 / denominator_y_685_1 * up_y_683_1;
                            __nv_bfloat162 _bf16x2_659 = __float22bfloat162_rn(make_float2(hidden_x_686_1, hidden_y_687_1));
                            hidden_packed_539_1[17] = __as_u32(_bf16x2_659);
                            float2 _cvt_f32_420 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[18]));
                            float2 _cvt_f32_421 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[18]));
                            float gate_x_688_1 = _cvt_f32_420.x;
                            float gate_y_689_1 = _cvt_f32_420.y;
                            float up_x_690_1 = _cvt_f32_421.x;
                            float up_y_691_1 = _cvt_f32_421.y;
                            float _exp_420 = expf(gate_x_688_1 * -1.0f);
                            float denominator_x_692_1 = _exp_420 + 1.0f;
                            float _exp_421 = expf(gate_y_689_1 * -1.0f);
                            float denominator_y_693_1 = _exp_421 + 1.0f;
                            float hidden_x_694_1 = gate_x_688_1 / denominator_x_692_1 * up_x_690_1;
                            float hidden_y_695_1 = gate_y_689_1 / denominator_y_693_1 * up_y_691_1;
                            __nv_bfloat162 _bf16x2_660 = __float22bfloat162_rn(make_float2(hidden_x_694_1, hidden_y_695_1));
                            hidden_packed_539_1[18] = __as_u32(_bf16x2_660);
                            float2 _cvt_f32_422 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[19]));
                            float2 _cvt_f32_423 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[19]));
                            float gate_x_696_1 = _cvt_f32_422.x;
                            float gate_y_697_1 = _cvt_f32_422.y;
                            float up_x_698_1 = _cvt_f32_423.x;
                            float up_y_699_1 = _cvt_f32_423.y;
                            float _exp_422 = expf(gate_x_696_1 * -1.0f);
                            float denominator_x_700_1 = _exp_422 + 1.0f;
                            float _exp_423 = expf(gate_y_697_1 * -1.0f);
                            float denominator_y_701_1 = _exp_423 + 1.0f;
                            float hidden_x_702_1 = gate_x_696_1 / denominator_x_700_1 * up_x_698_1;
                            float hidden_y_703_1 = gate_y_697_1 / denominator_y_701_1 * up_y_699_1;
                            __nv_bfloat162 _bf16x2_661 = __float22bfloat162_rn(make_float2(hidden_x_702_1, hidden_y_703_1));
                            hidden_packed_539_1[19] = __as_u32(_bf16x2_661);
                            float2 _cvt_f32_424 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[20]));
                            float2 _cvt_f32_425 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[20]));
                            float gate_x_704_1 = _cvt_f32_424.x;
                            float gate_y_705_1 = _cvt_f32_424.y;
                            float up_x_706_1 = _cvt_f32_425.x;
                            float up_y_707_1 = _cvt_f32_425.y;
                            float _exp_424 = expf(gate_x_704_1 * -1.0f);
                            float denominator_x_708_1 = _exp_424 + 1.0f;
                            float _exp_425 = expf(gate_y_705_1 * -1.0f);
                            float denominator_y_709_1 = _exp_425 + 1.0f;
                            float hidden_x_710_1 = gate_x_704_1 / denominator_x_708_1 * up_x_706_1;
                            float hidden_y_711_1 = gate_y_705_1 / denominator_y_709_1 * up_y_707_1;
                            __nv_bfloat162 _bf16x2_662 = __float22bfloat162_rn(make_float2(hidden_x_710_1, hidden_y_711_1));
                            hidden_packed_539_1[20] = __as_u32(_bf16x2_662);
                            float2 _cvt_f32_426 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[21]));
                            float2 _cvt_f32_427 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[21]));
                            float gate_x_712_1 = _cvt_f32_426.x;
                            float gate_y_713_1 = _cvt_f32_426.y;
                            float up_x_714_1 = _cvt_f32_427.x;
                            float up_y_715_1 = _cvt_f32_427.y;
                            float _exp_426 = expf(gate_x_712_1 * -1.0f);
                            float denominator_x_716_1 = _exp_426 + 1.0f;
                            float _exp_427 = expf(gate_y_713_1 * -1.0f);
                            float denominator_y_717_1 = _exp_427 + 1.0f;
                            float hidden_x_718_1 = gate_x_712_1 / denominator_x_716_1 * up_x_714_1;
                            float hidden_y_719_1 = gate_y_713_1 / denominator_y_717_1 * up_y_715_1;
                            __nv_bfloat162 _bf16x2_663 = __float22bfloat162_rn(make_float2(hidden_x_718_1, hidden_y_719_1));
                            hidden_packed_539_1[21] = __as_u32(_bf16x2_663);
                            float2 _cvt_f32_428 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[22]));
                            float2 _cvt_f32_429 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[22]));
                            float gate_x_720_1 = _cvt_f32_428.x;
                            float gate_y_721_1 = _cvt_f32_428.y;
                            float up_x_722_1 = _cvt_f32_429.x;
                            float up_y_723_1 = _cvt_f32_429.y;
                            float _exp_428 = expf(gate_x_720_1 * -1.0f);
                            float denominator_x_724_1 = _exp_428 + 1.0f;
                            float _exp_429 = expf(gate_y_721_1 * -1.0f);
                            float denominator_y_725_1 = _exp_429 + 1.0f;
                            float hidden_x_726_1 = gate_x_720_1 / denominator_x_724_1 * up_x_722_1;
                            float hidden_y_727_1 = gate_y_721_1 / denominator_y_725_1 * up_y_723_1;
                            __nv_bfloat162 _bf16x2_664 = __float22bfloat162_rn(make_float2(hidden_x_726_1, hidden_y_727_1));
                            hidden_packed_539_1[22] = __as_u32(_bf16x2_664);
                            float2 _cvt_f32_430 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[23]));
                            float2 _cvt_f32_431 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[23]));
                            float gate_x_728_1 = _cvt_f32_430.x;
                            float gate_y_729_1 = _cvt_f32_430.y;
                            float up_x_730_1 = _cvt_f32_431.x;
                            float up_y_731_1 = _cvt_f32_431.y;
                            float _exp_430 = expf(gate_x_728_1 * -1.0f);
                            float denominator_x_732_1 = _exp_430 + 1.0f;
                            float _exp_431 = expf(gate_y_729_1 * -1.0f);
                            float denominator_y_733_1 = _exp_431 + 1.0f;
                            float hidden_x_734_1 = gate_x_728_1 / denominator_x_732_1 * up_x_730_1;
                            float hidden_y_735_1 = gate_y_729_1 / denominator_y_733_1 * up_y_731_1;
                            __nv_bfloat162 _bf16x2_665 = __float22bfloat162_rn(make_float2(hidden_x_734_1, hidden_y_735_1));
                            hidden_packed_539_1[23] = __as_u32(_bf16x2_665);
                            float2 _cvt_f32_432 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[24]));
                            float2 _cvt_f32_433 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[24]));
                            float gate_x_736_1 = _cvt_f32_432.x;
                            float gate_y_737_1 = _cvt_f32_432.y;
                            float up_x_738_1 = _cvt_f32_433.x;
                            float up_y_739_1 = _cvt_f32_433.y;
                            float _exp_432 = expf(gate_x_736_1 * -1.0f);
                            float denominator_x_740_1 = _exp_432 + 1.0f;
                            float _exp_433 = expf(gate_y_737_1 * -1.0f);
                            float denominator_y_741_1 = _exp_433 + 1.0f;
                            float hidden_x_742_1 = gate_x_736_1 / denominator_x_740_1 * up_x_738_1;
                            float hidden_y_743_1 = gate_y_737_1 / denominator_y_741_1 * up_y_739_1;
                            __nv_bfloat162 _bf16x2_666 = __float22bfloat162_rn(make_float2(hidden_x_742_1, hidden_y_743_1));
                            hidden_packed_539_1[24] = __as_u32(_bf16x2_666);
                            float2 _cvt_f32_434 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[25]));
                            float2 _cvt_f32_435 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[25]));
                            float gate_x_744_1 = _cvt_f32_434.x;
                            float gate_y_745_1 = _cvt_f32_434.y;
                            float up_x_746_1 = _cvt_f32_435.x;
                            float up_y_747_1 = _cvt_f32_435.y;
                            float _exp_434 = expf(gate_x_744_1 * -1.0f);
                            float denominator_x_748_1 = _exp_434 + 1.0f;
                            float _exp_435 = expf(gate_y_745_1 * -1.0f);
                            float denominator_y_749_1 = _exp_435 + 1.0f;
                            float hidden_x_750_1 = gate_x_744_1 / denominator_x_748_1 * up_x_746_1;
                            float hidden_y_751_1 = gate_y_745_1 / denominator_y_749_1 * up_y_747_1;
                            __nv_bfloat162 _bf16x2_667 = __float22bfloat162_rn(make_float2(hidden_x_750_1, hidden_y_751_1));
                            hidden_packed_539_1[25] = __as_u32(_bf16x2_667);
                            float2 _cvt_f32_436 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[26]));
                            float2 _cvt_f32_437 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[26]));
                            float gate_x_752_1 = _cvt_f32_436.x;
                            float gate_y_753_1 = _cvt_f32_436.y;
                            float up_x_754_1 = _cvt_f32_437.x;
                            float up_y_755_1 = _cvt_f32_437.y;
                            float _exp_436 = expf(gate_x_752_1 * -1.0f);
                            float denominator_x_756_1 = _exp_436 + 1.0f;
                            float _exp_437 = expf(gate_y_753_1 * -1.0f);
                            float denominator_y_757_1 = _exp_437 + 1.0f;
                            float hidden_x_758_1 = gate_x_752_1 / denominator_x_756_1 * up_x_754_1;
                            float hidden_y_759_1 = gate_y_753_1 / denominator_y_757_1 * up_y_755_1;
                            __nv_bfloat162 _bf16x2_668 = __float22bfloat162_rn(make_float2(hidden_x_758_1, hidden_y_759_1));
                            hidden_packed_539_1[26] = __as_u32(_bf16x2_668);
                            float2 _cvt_f32_438 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[27]));
                            float2 _cvt_f32_439 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[27]));
                            float gate_x_760_1 = _cvt_f32_438.x;
                            float gate_y_761_1 = _cvt_f32_438.y;
                            float up_x_762_1 = _cvt_f32_439.x;
                            float up_y_763_1 = _cvt_f32_439.y;
                            float _exp_438 = expf(gate_x_760_1 * -1.0f);
                            float denominator_x_764_1 = _exp_438 + 1.0f;
                            float _exp_439 = expf(gate_y_761_1 * -1.0f);
                            float denominator_y_765_1 = _exp_439 + 1.0f;
                            float hidden_x_766_1 = gate_x_760_1 / denominator_x_764_1 * up_x_762_1;
                            float hidden_y_767_1 = gate_y_761_1 / denominator_y_765_1 * up_y_763_1;
                            __nv_bfloat162 _bf16x2_669 = __float22bfloat162_rn(make_float2(hidden_x_766_1, hidden_y_767_1));
                            hidden_packed_539_1[27] = __as_u32(_bf16x2_669);
                            float2 _cvt_f32_440 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[28]));
                            float2 _cvt_f32_441 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[28]));
                            float gate_x_768_1 = _cvt_f32_440.x;
                            float gate_y_769_1 = _cvt_f32_440.y;
                            float up_x_770_1 = _cvt_f32_441.x;
                            float up_y_771_1 = _cvt_f32_441.y;
                            float _exp_440 = expf(gate_x_768_1 * -1.0f);
                            float denominator_x_772_1 = _exp_440 + 1.0f;
                            float _exp_441 = expf(gate_y_769_1 * -1.0f);
                            float denominator_y_773_1 = _exp_441 + 1.0f;
                            float hidden_x_774_1 = gate_x_768_1 / denominator_x_772_1 * up_x_770_1;
                            float hidden_y_775_1 = gate_y_769_1 / denominator_y_773_1 * up_y_771_1;
                            __nv_bfloat162 _bf16x2_670 = __float22bfloat162_rn(make_float2(hidden_x_774_1, hidden_y_775_1));
                            hidden_packed_539_1[28] = __as_u32(_bf16x2_670);
                            float2 _cvt_f32_442 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[29]));
                            float2 _cvt_f32_443 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[29]));
                            float gate_x_776_1 = _cvt_f32_442.x;
                            float gate_y_777_1 = _cvt_f32_442.y;
                            float up_x_778_1 = _cvt_f32_443.x;
                            float up_y_779_1 = _cvt_f32_443.y;
                            float _exp_442 = expf(gate_x_776_1 * -1.0f);
                            float denominator_x_780_1 = _exp_442 + 1.0f;
                            float _exp_443 = expf(gate_y_777_1 * -1.0f);
                            float denominator_y_781_1 = _exp_443 + 1.0f;
                            float hidden_x_782_1 = gate_x_776_1 / denominator_x_780_1 * up_x_778_1;
                            float hidden_y_783_1 = gate_y_777_1 / denominator_y_781_1 * up_y_779_1;
                            __nv_bfloat162 _bf16x2_671 = __float22bfloat162_rn(make_float2(hidden_x_782_1, hidden_y_783_1));
                            hidden_packed_539_1[29] = __as_u32(_bf16x2_671);
                            float2 _cvt_f32_444 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[30]));
                            float2 _cvt_f32_445 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[30]));
                            float gate_x_784_1 = _cvt_f32_444.x;
                            float gate_y_785_1 = _cvt_f32_444.y;
                            float up_x_786_1 = _cvt_f32_445.x;
                            float up_y_787_1 = _cvt_f32_445.y;
                            float _exp_444 = expf(gate_x_784_1 * -1.0f);
                            float denominator_x_788_1 = _exp_444 + 1.0f;
                            float _exp_445 = expf(gate_y_785_1 * -1.0f);
                            float denominator_y_789_1 = _exp_445 + 1.0f;
                            float hidden_x_790_1 = gate_x_784_1 / denominator_x_788_1 * up_x_786_1;
                            float hidden_y_791_1 = gate_y_785_1 / denominator_y_789_1 * up_y_787_1;
                            __nv_bfloat162 _bf16x2_672 = __float22bfloat162_rn(make_float2(hidden_x_790_1, hidden_y_791_1));
                            hidden_packed_539_1[30] = __as_u32(_bf16x2_672);
                            float2 _cvt_f32_446 = __bfloat1622float2(__as_bf16x2(gate_packed_537_1[31]));
                            float2 _cvt_f32_447 = __bfloat1622float2(__as_bf16x2(up_packed_538_1[31]));
                            float gate_x_792_1 = _cvt_f32_446.x;
                            float gate_y_793_1 = _cvt_f32_446.y;
                            float up_x_794_1 = _cvt_f32_447.x;
                            float up_y_795_1 = _cvt_f32_447.y;
                            float _exp_446 = expf(gate_x_792_1 * -1.0f);
                            float denominator_x_796_1 = _exp_446 + 1.0f;
                            float _exp_447 = expf(gate_y_793_1 * -1.0f);
                            float denominator_y_797_1 = _exp_447 + 1.0f;
                            float hidden_x_798_1 = gate_x_792_1 / denominator_x_796_1 * up_x_794_1;
                            float hidden_y_799_1 = gate_y_793_1 / denominator_y_797_1 * up_y_795_1;
                            __nv_bfloat162 _bf16x2_673 = __float22bfloat162_rn(make_float2(hidden_x_798_1, hidden_y_799_1));
                            hidden_packed_539_1[31] = __as_u32(_bf16x2_673);
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_800_1 = tid / 32;
                            int lane_801_1 = tid % 32;
                            #pragma unroll
                            for (int half_38 = 0; half_38 < 2; half_38++) {
                                #pragma unroll
                                for (int col_tile_38 = 0; col_tile_38 < 2; col_tile_38++) {
                                    int row_44 = warp_800_1 * 32 + half_38 * 16 + lane_801_1 % 16;
                                    int col_41 = col_tile_38 * 16 + lane_801_1 / 16 * 8;
                                    unsigned int address_3_36 = d_smem_addr + (unsigned int)((row_44 * 32 + col_41) * 2);
                                    address_3_36 = address_3_36 ^ (address_3_36 & 511) >> 7 << 4;
                                    int offset_39 = half_38 * 8 + col_tile_38 * 4;
                                    uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(address_3_36);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_39])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_39 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_39 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_39 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_802_1 = tid / 32;
                            int lane_803_1 = tid % 32;
                            #pragma unroll
                            for (int half_39 = 0; half_39 < 2; half_39++) {
                                #pragma unroll
                                for (int col_tile_39 = 0; col_tile_39 < 2; col_tile_39++) {
                                    int row_45 = warp_802_1 * 32 + half_39 * 16 + lane_803_1 % 16;
                                    int col_42 = col_tile_39 * 16 + lane_803_1 / 16 * 8;
                                    unsigned int address_3_37 = d_smem_addr + 8192 + (unsigned int)((row_45 * 32 + col_42) * 2);
                                    address_3_37 = address_3_37 ^ (address_3_37 & 511) >> 7 << 4;
                                    int offset_40 = half_39 * 8 + col_tile_39 * 4;
                                    uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(address_3_37);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_40])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_40 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_40 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_40 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_804_1 = tid / 32;
                            int lane_805_1 = tid % 32;
                            #pragma unroll
                            for (int half_40 = 0; half_40 < 2; half_40++) {
                                #pragma unroll
                                for (int col_tile_40 = 0; col_tile_40 < 2; col_tile_40++) {
                                    int row_46 = warp_804_1 * 32 + half_40 * 16 + lane_805_1 % 16;
                                    int col_43 = col_tile_40 * 16 + lane_805_1 / 16 * 8;
                                    unsigned int address_3_38 = d_smem_addr + 16384 + (unsigned int)((row_46 * 32 + col_43) * 2);
                                    address_3_38 = address_3_38 ^ (address_3_38 & 511) >> 7 << 4;
                                    int offset_41 = half_40 * 8 + col_tile_40 * 4;
                                    uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(address_3_38);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_41])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_41 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_41 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_41 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_806_1 = tid / 32;
                            int lane_807_1 = tid % 32;
                            #pragma unroll
                            for (int half_41 = 0; half_41 < 2; half_41++) {
                                #pragma unroll
                                for (int col_tile_41 = 0; col_tile_41 < 2; col_tile_41++) {
                                    int row_47 = warp_806_1 * 32 + half_41 * 16 + lane_807_1 % 16;
                                    int col_44 = col_tile_41 * 16 + lane_807_1 / 16 * 8;
                                    unsigned int address_3_39 = d_smem_addr + (unsigned int)((row_47 * 32 + col_44) * 2);
                                    address_3_39 = address_3_39 ^ (address_3_39 & 511) >> 7 << 4;
                                    int offset_42 = 16 + half_41 * 8 + col_tile_41 * 4;
                                    uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(address_3_39);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_42])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_42 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_42 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_537_1[offset_42 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_808_1 = tid / 32;
                            int lane_809_1 = tid % 32;
                            #pragma unroll
                            for (int half_42 = 0; half_42 < 2; half_42++) {
                                #pragma unroll
                                for (int col_tile_42 = 0; col_tile_42 < 2; col_tile_42++) {
                                    int row_48 = warp_808_1 * 32 + half_42 * 16 + lane_809_1 % 16;
                                    int col_45 = col_tile_42 * 16 + lane_809_1 / 16 * 8;
                                    unsigned int address_3_40 = d_smem_addr + 8192 + (unsigned int)((row_48 * 32 + col_45) * 2);
                                    address_3_40 = address_3_40 ^ (address_3_40 & 511) >> 7 << 4;
                                    int offset_43 = 16 + half_42 * 8 + col_tile_42 * 4;
                                    uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(address_3_40);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_43])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_43 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_43 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_538_1[offset_43 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_810_1 = tid / 32;
                            int lane_811_1 = tid % 32;
                            #pragma unroll
                            for (int half_43 = 0; half_43 < 2; half_43++) {
                                #pragma unroll
                                for (int col_tile_43 = 0; col_tile_43 < 2; col_tile_43++) {
                                    int row_49 = warp_810_1 * 32 + half_43 * 16 + lane_811_1 % 16;
                                    int col_46 = col_tile_43 * 16 + lane_811_1 / 16 * 8;
                                    unsigned int address_3_41 = d_smem_addr + 16384 + (unsigned int)((row_49 * 32 + col_46) * 2);
                                    address_3_41 = address_3_41 ^ (address_3_41 & 511) >> 7 << 4;
                                    int offset_44 = 16 + half_43 * 8 + col_tile_43 * 4;
                                    uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(address_3_41);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_44])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_44 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_44 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_539_1[offset_44 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int gate_packed_812_1[32];
                            unsigned int up_packed_813_1[32];
                            unsigned int hidden_packed_814_1[32];
                            unsigned int address_815_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 192;
                            float _tmem_load_58[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_58[15]))
                                : "r"(address_815_1));
                            float _tmem_load_59[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_59[15]))
                                : "r"(address_815_1 + 256));
                            __nv_bfloat162 _bf16x2_674 = __float22bfloat162_rn(make_float2(_tmem_load_58[0], _tmem_load_58[1]));
                            gate_packed_812_1[0] = __as_u32(_bf16x2_674);
                            __nv_bfloat162 _bf16x2_675 = __float22bfloat162_rn(make_float2(_tmem_load_59[0], _tmem_load_59[1]));
                            up_packed_813_1[0] = __as_u32(_bf16x2_675);
                            __nv_bfloat162 _bf16x2_676 = __float22bfloat162_rn(make_float2(_tmem_load_58[2], _tmem_load_58[3]));
                            gate_packed_812_1[1] = __as_u32(_bf16x2_676);
                            __nv_bfloat162 _bf16x2_677 = __float22bfloat162_rn(make_float2(_tmem_load_59[2], _tmem_load_59[3]));
                            up_packed_813_1[1] = __as_u32(_bf16x2_677);
                            __nv_bfloat162 _bf16x2_678 = __float22bfloat162_rn(make_float2(_tmem_load_58[4], _tmem_load_58[5]));
                            gate_packed_812_1[2] = __as_u32(_bf16x2_678);
                            __nv_bfloat162 _bf16x2_679 = __float22bfloat162_rn(make_float2(_tmem_load_59[4], _tmem_load_59[5]));
                            up_packed_813_1[2] = __as_u32(_bf16x2_679);
                            __nv_bfloat162 _bf16x2_680 = __float22bfloat162_rn(make_float2(_tmem_load_58[6], _tmem_load_58[7]));
                            gate_packed_812_1[3] = __as_u32(_bf16x2_680);
                            __nv_bfloat162 _bf16x2_681 = __float22bfloat162_rn(make_float2(_tmem_load_59[6], _tmem_load_59[7]));
                            up_packed_813_1[3] = __as_u32(_bf16x2_681);
                            __nv_bfloat162 _bf16x2_682 = __float22bfloat162_rn(make_float2(_tmem_load_58[8], _tmem_load_58[9]));
                            gate_packed_812_1[4] = __as_u32(_bf16x2_682);
                            __nv_bfloat162 _bf16x2_683 = __float22bfloat162_rn(make_float2(_tmem_load_59[8], _tmem_load_59[9]));
                            up_packed_813_1[4] = __as_u32(_bf16x2_683);
                            __nv_bfloat162 _bf16x2_684 = __float22bfloat162_rn(make_float2(_tmem_load_58[10], _tmem_load_58[11]));
                            gate_packed_812_1[5] = __as_u32(_bf16x2_684);
                            __nv_bfloat162 _bf16x2_685 = __float22bfloat162_rn(make_float2(_tmem_load_59[10], _tmem_load_59[11]));
                            up_packed_813_1[5] = __as_u32(_bf16x2_685);
                            __nv_bfloat162 _bf16x2_686 = __float22bfloat162_rn(make_float2(_tmem_load_58[12], _tmem_load_58[13]));
                            gate_packed_812_1[6] = __as_u32(_bf16x2_686);
                            __nv_bfloat162 _bf16x2_687 = __float22bfloat162_rn(make_float2(_tmem_load_59[12], _tmem_load_59[13]));
                            up_packed_813_1[6] = __as_u32(_bf16x2_687);
                            __nv_bfloat162 _bf16x2_688 = __float22bfloat162_rn(make_float2(_tmem_load_58[14], _tmem_load_58[15]));
                            gate_packed_812_1[7] = __as_u32(_bf16x2_688);
                            __nv_bfloat162 _bf16x2_689 = __float22bfloat162_rn(make_float2(_tmem_load_59[14], _tmem_load_59[15]));
                            up_packed_813_1[7] = __as_u32(_bf16x2_689);
                            unsigned int address_816_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 192;
                            float _tmem_load_60[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_60[15]))
                                : "r"(address_816_1));
                            float _tmem_load_61[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_61[15]))
                                : "r"(address_816_1 + 256));
                            __nv_bfloat162 _bf16x2_690 = __float22bfloat162_rn(make_float2(_tmem_load_60[0], _tmem_load_60[1]));
                            gate_packed_812_1[8] = __as_u32(_bf16x2_690);
                            __nv_bfloat162 _bf16x2_691 = __float22bfloat162_rn(make_float2(_tmem_load_61[0], _tmem_load_61[1]));
                            up_packed_813_1[8] = __as_u32(_bf16x2_691);
                            __nv_bfloat162 _bf16x2_692 = __float22bfloat162_rn(make_float2(_tmem_load_60[2], _tmem_load_60[3]));
                            gate_packed_812_1[9] = __as_u32(_bf16x2_692);
                            __nv_bfloat162 _bf16x2_693 = __float22bfloat162_rn(make_float2(_tmem_load_61[2], _tmem_load_61[3]));
                            up_packed_813_1[9] = __as_u32(_bf16x2_693);
                            __nv_bfloat162 _bf16x2_694 = __float22bfloat162_rn(make_float2(_tmem_load_60[4], _tmem_load_60[5]));
                            gate_packed_812_1[10] = __as_u32(_bf16x2_694);
                            __nv_bfloat162 _bf16x2_695 = __float22bfloat162_rn(make_float2(_tmem_load_61[4], _tmem_load_61[5]));
                            up_packed_813_1[10] = __as_u32(_bf16x2_695);
                            __nv_bfloat162 _bf16x2_696 = __float22bfloat162_rn(make_float2(_tmem_load_60[6], _tmem_load_60[7]));
                            gate_packed_812_1[11] = __as_u32(_bf16x2_696);
                            __nv_bfloat162 _bf16x2_697 = __float22bfloat162_rn(make_float2(_tmem_load_61[6], _tmem_load_61[7]));
                            up_packed_813_1[11] = __as_u32(_bf16x2_697);
                            __nv_bfloat162 _bf16x2_698 = __float22bfloat162_rn(make_float2(_tmem_load_60[8], _tmem_load_60[9]));
                            gate_packed_812_1[12] = __as_u32(_bf16x2_698);
                            __nv_bfloat162 _bf16x2_699 = __float22bfloat162_rn(make_float2(_tmem_load_61[8], _tmem_load_61[9]));
                            up_packed_813_1[12] = __as_u32(_bf16x2_699);
                            __nv_bfloat162 _bf16x2_700 = __float22bfloat162_rn(make_float2(_tmem_load_60[10], _tmem_load_60[11]));
                            gate_packed_812_1[13] = __as_u32(_bf16x2_700);
                            __nv_bfloat162 _bf16x2_701 = __float22bfloat162_rn(make_float2(_tmem_load_61[10], _tmem_load_61[11]));
                            up_packed_813_1[13] = __as_u32(_bf16x2_701);
                            __nv_bfloat162 _bf16x2_702 = __float22bfloat162_rn(make_float2(_tmem_load_60[12], _tmem_load_60[13]));
                            gate_packed_812_1[14] = __as_u32(_bf16x2_702);
                            __nv_bfloat162 _bf16x2_703 = __float22bfloat162_rn(make_float2(_tmem_load_61[12], _tmem_load_61[13]));
                            up_packed_813_1[14] = __as_u32(_bf16x2_703);
                            __nv_bfloat162 _bf16x2_704 = __float22bfloat162_rn(make_float2(_tmem_load_60[14], _tmem_load_60[15]));
                            gate_packed_812_1[15] = __as_u32(_bf16x2_704);
                            __nv_bfloat162 _bf16x2_705 = __float22bfloat162_rn(make_float2(_tmem_load_61[14], _tmem_load_61[15]));
                            up_packed_813_1[15] = __as_u32(_bf16x2_705);
                            unsigned int address_817_1 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 224;
                            float _tmem_load_62[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_62[15]))
                                : "r"(address_817_1));
                            float _tmem_load_63[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_63[15]))
                                : "r"(address_817_1 + 256));
                            __nv_bfloat162 _bf16x2_706 = __float22bfloat162_rn(make_float2(_tmem_load_62[0], _tmem_load_62[1]));
                            gate_packed_812_1[16] = __as_u32(_bf16x2_706);
                            __nv_bfloat162 _bf16x2_707 = __float22bfloat162_rn(make_float2(_tmem_load_63[0], _tmem_load_63[1]));
                            up_packed_813_1[16] = __as_u32(_bf16x2_707);
                            __nv_bfloat162 _bf16x2_708 = __float22bfloat162_rn(make_float2(_tmem_load_62[2], _tmem_load_62[3]));
                            gate_packed_812_1[17] = __as_u32(_bf16x2_708);
                            __nv_bfloat162 _bf16x2_709 = __float22bfloat162_rn(make_float2(_tmem_load_63[2], _tmem_load_63[3]));
                            up_packed_813_1[17] = __as_u32(_bf16x2_709);
                            __nv_bfloat162 _bf16x2_710 = __float22bfloat162_rn(make_float2(_tmem_load_62[4], _tmem_load_62[5]));
                            gate_packed_812_1[18] = __as_u32(_bf16x2_710);
                            __nv_bfloat162 _bf16x2_711 = __float22bfloat162_rn(make_float2(_tmem_load_63[4], _tmem_load_63[5]));
                            up_packed_813_1[18] = __as_u32(_bf16x2_711);
                            __nv_bfloat162 _bf16x2_712 = __float22bfloat162_rn(make_float2(_tmem_load_62[6], _tmem_load_62[7]));
                            gate_packed_812_1[19] = __as_u32(_bf16x2_712);
                            __nv_bfloat162 _bf16x2_713 = __float22bfloat162_rn(make_float2(_tmem_load_63[6], _tmem_load_63[7]));
                            up_packed_813_1[19] = __as_u32(_bf16x2_713);
                            __nv_bfloat162 _bf16x2_714 = __float22bfloat162_rn(make_float2(_tmem_load_62[8], _tmem_load_62[9]));
                            gate_packed_812_1[20] = __as_u32(_bf16x2_714);
                            __nv_bfloat162 _bf16x2_715 = __float22bfloat162_rn(make_float2(_tmem_load_63[8], _tmem_load_63[9]));
                            up_packed_813_1[20] = __as_u32(_bf16x2_715);
                            __nv_bfloat162 _bf16x2_716 = __float22bfloat162_rn(make_float2(_tmem_load_62[10], _tmem_load_62[11]));
                            gate_packed_812_1[21] = __as_u32(_bf16x2_716);
                            __nv_bfloat162 _bf16x2_717 = __float22bfloat162_rn(make_float2(_tmem_load_63[10], _tmem_load_63[11]));
                            up_packed_813_1[21] = __as_u32(_bf16x2_717);
                            __nv_bfloat162 _bf16x2_718 = __float22bfloat162_rn(make_float2(_tmem_load_62[12], _tmem_load_62[13]));
                            gate_packed_812_1[22] = __as_u32(_bf16x2_718);
                            __nv_bfloat162 _bf16x2_719 = __float22bfloat162_rn(make_float2(_tmem_load_63[12], _tmem_load_63[13]));
                            up_packed_813_1[22] = __as_u32(_bf16x2_719);
                            __nv_bfloat162 _bf16x2_720 = __float22bfloat162_rn(make_float2(_tmem_load_62[14], _tmem_load_62[15]));
                            gate_packed_812_1[23] = __as_u32(_bf16x2_720);
                            __nv_bfloat162 _bf16x2_721 = __float22bfloat162_rn(make_float2(_tmem_load_63[14], _tmem_load_63[15]));
                            up_packed_813_1[23] = __as_u32(_bf16x2_721);
                            unsigned int address_818_1 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 224;
                            float _tmem_load_64[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_64[15]))
                                : "r"(address_818_1));
                            float _tmem_load_65[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_65[15]))
                                : "r"(address_818_1 + 256));
                            __nv_bfloat162 _bf16x2_722 = __float22bfloat162_rn(make_float2(_tmem_load_64[0], _tmem_load_64[1]));
                            gate_packed_812_1[24] = __as_u32(_bf16x2_722);
                            __nv_bfloat162 _bf16x2_723 = __float22bfloat162_rn(make_float2(_tmem_load_65[0], _tmem_load_65[1]));
                            up_packed_813_1[24] = __as_u32(_bf16x2_723);
                            __nv_bfloat162 _bf16x2_724 = __float22bfloat162_rn(make_float2(_tmem_load_64[2], _tmem_load_64[3]));
                            gate_packed_812_1[25] = __as_u32(_bf16x2_724);
                            __nv_bfloat162 _bf16x2_725 = __float22bfloat162_rn(make_float2(_tmem_load_65[2], _tmem_load_65[3]));
                            up_packed_813_1[25] = __as_u32(_bf16x2_725);
                            __nv_bfloat162 _bf16x2_726 = __float22bfloat162_rn(make_float2(_tmem_load_64[4], _tmem_load_64[5]));
                            gate_packed_812_1[26] = __as_u32(_bf16x2_726);
                            __nv_bfloat162 _bf16x2_727 = __float22bfloat162_rn(make_float2(_tmem_load_65[4], _tmem_load_65[5]));
                            up_packed_813_1[26] = __as_u32(_bf16x2_727);
                            __nv_bfloat162 _bf16x2_728 = __float22bfloat162_rn(make_float2(_tmem_load_64[6], _tmem_load_64[7]));
                            gate_packed_812_1[27] = __as_u32(_bf16x2_728);
                            __nv_bfloat162 _bf16x2_729 = __float22bfloat162_rn(make_float2(_tmem_load_65[6], _tmem_load_65[7]));
                            up_packed_813_1[27] = __as_u32(_bf16x2_729);
                            __nv_bfloat162 _bf16x2_730 = __float22bfloat162_rn(make_float2(_tmem_load_64[8], _tmem_load_64[9]));
                            gate_packed_812_1[28] = __as_u32(_bf16x2_730);
                            __nv_bfloat162 _bf16x2_731 = __float22bfloat162_rn(make_float2(_tmem_load_65[8], _tmem_load_65[9]));
                            up_packed_813_1[28] = __as_u32(_bf16x2_731);
                            __nv_bfloat162 _bf16x2_732 = __float22bfloat162_rn(make_float2(_tmem_load_64[10], _tmem_load_64[11]));
                            gate_packed_812_1[29] = __as_u32(_bf16x2_732);
                            __nv_bfloat162 _bf16x2_733 = __float22bfloat162_rn(make_float2(_tmem_load_65[10], _tmem_load_65[11]));
                            up_packed_813_1[29] = __as_u32(_bf16x2_733);
                            __nv_bfloat162 _bf16x2_734 = __float22bfloat162_rn(make_float2(_tmem_load_64[12], _tmem_load_64[13]));
                            gate_packed_812_1[30] = __as_u32(_bf16x2_734);
                            __nv_bfloat162 _bf16x2_735 = __float22bfloat162_rn(make_float2(_tmem_load_65[12], _tmem_load_65[13]));
                            up_packed_813_1[30] = __as_u32(_bf16x2_735);
                            __nv_bfloat162 _bf16x2_736 = __float22bfloat162_rn(make_float2(_tmem_load_64[14], _tmem_load_64[15]));
                            gate_packed_812_1[31] = __as_u32(_bf16x2_736);
                            __nv_bfloat162 _bf16x2_737 = __float22bfloat162_rn(make_float2(_tmem_load_65[14], _tmem_load_65[15]));
                            up_packed_813_1[31] = __as_u32(_bf16x2_737);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr) & 0xFEFFFFFF) : "memory");
                            }
                            float2 _cvt_f32_448 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[0]));
                            float2 _cvt_f32_449 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[0]));
                            float gate_x_819_1 = _cvt_f32_448.x;
                            float gate_y_820_1 = _cvt_f32_448.y;
                            float up_x_821_1 = _cvt_f32_449.x;
                            float up_y_822_1 = _cvt_f32_449.y;
                            float _exp_448 = expf(gate_x_819_1 * -1.0f);
                            float denominator_x_823_1 = _exp_448 + 1.0f;
                            float _exp_449 = expf(gate_y_820_1 * -1.0f);
                            float denominator_y_824_1 = _exp_449 + 1.0f;
                            float hidden_x_825_1 = gate_x_819_1 / denominator_x_823_1 * up_x_821_1;
                            float hidden_y_826_1 = gate_y_820_1 / denominator_y_824_1 * up_y_822_1;
                            __nv_bfloat162 _bf16x2_738 = __float22bfloat162_rn(make_float2(hidden_x_825_1, hidden_y_826_1));
                            hidden_packed_814_1[0] = __as_u32(_bf16x2_738);
                            float2 _cvt_f32_450 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[1]));
                            float2 _cvt_f32_451 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[1]));
                            float gate_x_827_1 = _cvt_f32_450.x;
                            float gate_y_828_1 = _cvt_f32_450.y;
                            float up_x_829_1 = _cvt_f32_451.x;
                            float up_y_830_1 = _cvt_f32_451.y;
                            float _exp_450 = expf(gate_x_827_1 * -1.0f);
                            float denominator_x_831_1 = _exp_450 + 1.0f;
                            float _exp_451 = expf(gate_y_828_1 * -1.0f);
                            float denominator_y_832_1 = _exp_451 + 1.0f;
                            float hidden_x_833_1 = gate_x_827_1 / denominator_x_831_1 * up_x_829_1;
                            float hidden_y_834_1 = gate_y_828_1 / denominator_y_832_1 * up_y_830_1;
                            __nv_bfloat162 _bf16x2_739 = __float22bfloat162_rn(make_float2(hidden_x_833_1, hidden_y_834_1));
                            hidden_packed_814_1[1] = __as_u32(_bf16x2_739);
                            float2 _cvt_f32_452 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[2]));
                            float2 _cvt_f32_453 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[2]));
                            float gate_x_835_1 = _cvt_f32_452.x;
                            float gate_y_836_1 = _cvt_f32_452.y;
                            float up_x_837_1 = _cvt_f32_453.x;
                            float up_y_838_1 = _cvt_f32_453.y;
                            float _exp_452 = expf(gate_x_835_1 * -1.0f);
                            float denominator_x_839_1 = _exp_452 + 1.0f;
                            float _exp_453 = expf(gate_y_836_1 * -1.0f);
                            float denominator_y_840_1 = _exp_453 + 1.0f;
                            float hidden_x_841_1 = gate_x_835_1 / denominator_x_839_1 * up_x_837_1;
                            float hidden_y_842_1 = gate_y_836_1 / denominator_y_840_1 * up_y_838_1;
                            __nv_bfloat162 _bf16x2_740 = __float22bfloat162_rn(make_float2(hidden_x_841_1, hidden_y_842_1));
                            hidden_packed_814_1[2] = __as_u32(_bf16x2_740);
                            float2 _cvt_f32_454 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[3]));
                            float2 _cvt_f32_455 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[3]));
                            float gate_x_843_1 = _cvt_f32_454.x;
                            float gate_y_844_1 = _cvt_f32_454.y;
                            float up_x_845_1 = _cvt_f32_455.x;
                            float up_y_846_1 = _cvt_f32_455.y;
                            float _exp_454 = expf(gate_x_843_1 * -1.0f);
                            float denominator_x_847_1 = _exp_454 + 1.0f;
                            float _exp_455 = expf(gate_y_844_1 * -1.0f);
                            float denominator_y_848_1 = _exp_455 + 1.0f;
                            float hidden_x_849_1 = gate_x_843_1 / denominator_x_847_1 * up_x_845_1;
                            float hidden_y_850_1 = gate_y_844_1 / denominator_y_848_1 * up_y_846_1;
                            __nv_bfloat162 _bf16x2_741 = __float22bfloat162_rn(make_float2(hidden_x_849_1, hidden_y_850_1));
                            hidden_packed_814_1[3] = __as_u32(_bf16x2_741);
                            float2 _cvt_f32_456 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[4]));
                            float2 _cvt_f32_457 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[4]));
                            float gate_x_851_1 = _cvt_f32_456.x;
                            float gate_y_852_1 = _cvt_f32_456.y;
                            float up_x_853_1 = _cvt_f32_457.x;
                            float up_y_854_1 = _cvt_f32_457.y;
                            float _exp_456 = expf(gate_x_851_1 * -1.0f);
                            float denominator_x_855_1 = _exp_456 + 1.0f;
                            float _exp_457 = expf(gate_y_852_1 * -1.0f);
                            float denominator_y_856_1 = _exp_457 + 1.0f;
                            float hidden_x_857_1 = gate_x_851_1 / denominator_x_855_1 * up_x_853_1;
                            float hidden_y_858_1 = gate_y_852_1 / denominator_y_856_1 * up_y_854_1;
                            __nv_bfloat162 _bf16x2_742 = __float22bfloat162_rn(make_float2(hidden_x_857_1, hidden_y_858_1));
                            hidden_packed_814_1[4] = __as_u32(_bf16x2_742);
                            float2 _cvt_f32_458 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[5]));
                            float2 _cvt_f32_459 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[5]));
                            float gate_x_859_1 = _cvt_f32_458.x;
                            float gate_y_860_1 = _cvt_f32_458.y;
                            float up_x_861_1 = _cvt_f32_459.x;
                            float up_y_862_1 = _cvt_f32_459.y;
                            float _exp_458 = expf(gate_x_859_1 * -1.0f);
                            float denominator_x_863_1 = _exp_458 + 1.0f;
                            float _exp_459 = expf(gate_y_860_1 * -1.0f);
                            float denominator_y_864_1 = _exp_459 + 1.0f;
                            float hidden_x_865_1 = gate_x_859_1 / denominator_x_863_1 * up_x_861_1;
                            float hidden_y_866_1 = gate_y_860_1 / denominator_y_864_1 * up_y_862_1;
                            __nv_bfloat162 _bf16x2_743 = __float22bfloat162_rn(make_float2(hidden_x_865_1, hidden_y_866_1));
                            hidden_packed_814_1[5] = __as_u32(_bf16x2_743);
                            float2 _cvt_f32_460 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[6]));
                            float2 _cvt_f32_461 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[6]));
                            float gate_x_867_1 = _cvt_f32_460.x;
                            float gate_y_868_1 = _cvt_f32_460.y;
                            float up_x_869_1 = _cvt_f32_461.x;
                            float up_y_870_1 = _cvt_f32_461.y;
                            float _exp_460 = expf(gate_x_867_1 * -1.0f);
                            float denominator_x_871_1 = _exp_460 + 1.0f;
                            float _exp_461 = expf(gate_y_868_1 * -1.0f);
                            float denominator_y_872_1 = _exp_461 + 1.0f;
                            float hidden_x_873_1 = gate_x_867_1 / denominator_x_871_1 * up_x_869_1;
                            float hidden_y_874_1 = gate_y_868_1 / denominator_y_872_1 * up_y_870_1;
                            __nv_bfloat162 _bf16x2_744 = __float22bfloat162_rn(make_float2(hidden_x_873_1, hidden_y_874_1));
                            hidden_packed_814_1[6] = __as_u32(_bf16x2_744);
                            float2 _cvt_f32_462 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[7]));
                            float2 _cvt_f32_463 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[7]));
                            float gate_x_875_1 = _cvt_f32_462.x;
                            float gate_y_876_1 = _cvt_f32_462.y;
                            float up_x_877_1 = _cvt_f32_463.x;
                            float up_y_878_1 = _cvt_f32_463.y;
                            float _exp_462 = expf(gate_x_875_1 * -1.0f);
                            float denominator_x_879_1 = _exp_462 + 1.0f;
                            float _exp_463 = expf(gate_y_876_1 * -1.0f);
                            float denominator_y_880_1 = _exp_463 + 1.0f;
                            float hidden_x_881_1 = gate_x_875_1 / denominator_x_879_1 * up_x_877_1;
                            float hidden_y_882_1 = gate_y_876_1 / denominator_y_880_1 * up_y_878_1;
                            __nv_bfloat162 _bf16x2_745 = __float22bfloat162_rn(make_float2(hidden_x_881_1, hidden_y_882_1));
                            hidden_packed_814_1[7] = __as_u32(_bf16x2_745);
                            float2 _cvt_f32_464 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[8]));
                            float2 _cvt_f32_465 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[8]));
                            float gate_x_883_1 = _cvt_f32_464.x;
                            float gate_y_884_1 = _cvt_f32_464.y;
                            float up_x_885_1 = _cvt_f32_465.x;
                            float up_y_886_1 = _cvt_f32_465.y;
                            float _exp_464 = expf(gate_x_883_1 * -1.0f);
                            float denominator_x_887_1 = _exp_464 + 1.0f;
                            float _exp_465 = expf(gate_y_884_1 * -1.0f);
                            float denominator_y_888_1 = _exp_465 + 1.0f;
                            float hidden_x_889_1 = gate_x_883_1 / denominator_x_887_1 * up_x_885_1;
                            float hidden_y_890_1 = gate_y_884_1 / denominator_y_888_1 * up_y_886_1;
                            __nv_bfloat162 _bf16x2_746 = __float22bfloat162_rn(make_float2(hidden_x_889_1, hidden_y_890_1));
                            hidden_packed_814_1[8] = __as_u32(_bf16x2_746);
                            float2 _cvt_f32_466 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[9]));
                            float2 _cvt_f32_467 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[9]));
                            float gate_x_891_1 = _cvt_f32_466.x;
                            float gate_y_892_1 = _cvt_f32_466.y;
                            float up_x_893_1 = _cvt_f32_467.x;
                            float up_y_894_1 = _cvt_f32_467.y;
                            float _exp_466 = expf(gate_x_891_1 * -1.0f);
                            float denominator_x_895_1 = _exp_466 + 1.0f;
                            float _exp_467 = expf(gate_y_892_1 * -1.0f);
                            float denominator_y_896_1 = _exp_467 + 1.0f;
                            float hidden_x_897_1 = gate_x_891_1 / denominator_x_895_1 * up_x_893_1;
                            float hidden_y_898_1 = gate_y_892_1 / denominator_y_896_1 * up_y_894_1;
                            __nv_bfloat162 _bf16x2_747 = __float22bfloat162_rn(make_float2(hidden_x_897_1, hidden_y_898_1));
                            hidden_packed_814_1[9] = __as_u32(_bf16x2_747);
                            float2 _cvt_f32_468 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[10]));
                            float2 _cvt_f32_469 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[10]));
                            float gate_x_899_1 = _cvt_f32_468.x;
                            float gate_y_900_1 = _cvt_f32_468.y;
                            float up_x_901_1 = _cvt_f32_469.x;
                            float up_y_902_1 = _cvt_f32_469.y;
                            float _exp_468 = expf(gate_x_899_1 * -1.0f);
                            float denominator_x_903_1 = _exp_468 + 1.0f;
                            float _exp_469 = expf(gate_y_900_1 * -1.0f);
                            float denominator_y_904_1 = _exp_469 + 1.0f;
                            float hidden_x_905_1 = gate_x_899_1 / denominator_x_903_1 * up_x_901_1;
                            float hidden_y_906_1 = gate_y_900_1 / denominator_y_904_1 * up_y_902_1;
                            __nv_bfloat162 _bf16x2_748 = __float22bfloat162_rn(make_float2(hidden_x_905_1, hidden_y_906_1));
                            hidden_packed_814_1[10] = __as_u32(_bf16x2_748);
                            float2 _cvt_f32_470 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[11]));
                            float2 _cvt_f32_471 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[11]));
                            float gate_x_907_1 = _cvt_f32_470.x;
                            float gate_y_908_1 = _cvt_f32_470.y;
                            float up_x_909_1 = _cvt_f32_471.x;
                            float up_y_910_1 = _cvt_f32_471.y;
                            float _exp_470 = expf(gate_x_907_1 * -1.0f);
                            float denominator_x_911_1 = _exp_470 + 1.0f;
                            float _exp_471 = expf(gate_y_908_1 * -1.0f);
                            float denominator_y_912_1 = _exp_471 + 1.0f;
                            float hidden_x_913_1 = gate_x_907_1 / denominator_x_911_1 * up_x_909_1;
                            float hidden_y_914_1 = gate_y_908_1 / denominator_y_912_1 * up_y_910_1;
                            __nv_bfloat162 _bf16x2_749 = __float22bfloat162_rn(make_float2(hidden_x_913_1, hidden_y_914_1));
                            hidden_packed_814_1[11] = __as_u32(_bf16x2_749);
                            float2 _cvt_f32_472 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[12]));
                            float2 _cvt_f32_473 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[12]));
                            float gate_x_915_1 = _cvt_f32_472.x;
                            float gate_y_916_1 = _cvt_f32_472.y;
                            float up_x_917_1 = _cvt_f32_473.x;
                            float up_y_918_1 = _cvt_f32_473.y;
                            float _exp_472 = expf(gate_x_915_1 * -1.0f);
                            float denominator_x_919_1 = _exp_472 + 1.0f;
                            float _exp_473 = expf(gate_y_916_1 * -1.0f);
                            float denominator_y_920_1 = _exp_473 + 1.0f;
                            float hidden_x_921_1 = gate_x_915_1 / denominator_x_919_1 * up_x_917_1;
                            float hidden_y_922_1 = gate_y_916_1 / denominator_y_920_1 * up_y_918_1;
                            __nv_bfloat162 _bf16x2_750 = __float22bfloat162_rn(make_float2(hidden_x_921_1, hidden_y_922_1));
                            hidden_packed_814_1[12] = __as_u32(_bf16x2_750);
                            float2 _cvt_f32_474 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[13]));
                            float2 _cvt_f32_475 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[13]));
                            float gate_x_923_1 = _cvt_f32_474.x;
                            float gate_y_924_1 = _cvt_f32_474.y;
                            float up_x_925_1 = _cvt_f32_475.x;
                            float up_y_926_1 = _cvt_f32_475.y;
                            float _exp_474 = expf(gate_x_923_1 * -1.0f);
                            float denominator_x_927_1 = _exp_474 + 1.0f;
                            float _exp_475 = expf(gate_y_924_1 * -1.0f);
                            float denominator_y_928_1 = _exp_475 + 1.0f;
                            float hidden_x_929_1 = gate_x_923_1 / denominator_x_927_1 * up_x_925_1;
                            float hidden_y_930_1 = gate_y_924_1 / denominator_y_928_1 * up_y_926_1;
                            __nv_bfloat162 _bf16x2_751 = __float22bfloat162_rn(make_float2(hidden_x_929_1, hidden_y_930_1));
                            hidden_packed_814_1[13] = __as_u32(_bf16x2_751);
                            float2 _cvt_f32_476 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[14]));
                            float2 _cvt_f32_477 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[14]));
                            float gate_x_931_1 = _cvt_f32_476.x;
                            float gate_y_932_1 = _cvt_f32_476.y;
                            float up_x_933_1 = _cvt_f32_477.x;
                            float up_y_934_1 = _cvt_f32_477.y;
                            float _exp_476 = expf(gate_x_931_1 * -1.0f);
                            float denominator_x_935_1 = _exp_476 + 1.0f;
                            float _exp_477 = expf(gate_y_932_1 * -1.0f);
                            float denominator_y_936_1 = _exp_477 + 1.0f;
                            float hidden_x_937_1 = gate_x_931_1 / denominator_x_935_1 * up_x_933_1;
                            float hidden_y_938_1 = gate_y_932_1 / denominator_y_936_1 * up_y_934_1;
                            __nv_bfloat162 _bf16x2_752 = __float22bfloat162_rn(make_float2(hidden_x_937_1, hidden_y_938_1));
                            hidden_packed_814_1[14] = __as_u32(_bf16x2_752);
                            float2 _cvt_f32_478 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[15]));
                            float2 _cvt_f32_479 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[15]));
                            float gate_x_939_1 = _cvt_f32_478.x;
                            float gate_y_940_1 = _cvt_f32_478.y;
                            float up_x_941_1 = _cvt_f32_479.x;
                            float up_y_942_1 = _cvt_f32_479.y;
                            float _exp_478 = expf(gate_x_939_1 * -1.0f);
                            float denominator_x_943_1 = _exp_478 + 1.0f;
                            float _exp_479 = expf(gate_y_940_1 * -1.0f);
                            float denominator_y_944_1 = _exp_479 + 1.0f;
                            float hidden_x_945_1 = gate_x_939_1 / denominator_x_943_1 * up_x_941_1;
                            float hidden_y_946_1 = gate_y_940_1 / denominator_y_944_1 * up_y_942_1;
                            __nv_bfloat162 _bf16x2_753 = __float22bfloat162_rn(make_float2(hidden_x_945_1, hidden_y_946_1));
                            hidden_packed_814_1[15] = __as_u32(_bf16x2_753);
                            float2 _cvt_f32_480 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[16]));
                            float2 _cvt_f32_481 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[16]));
                            float gate_x_947_1 = _cvt_f32_480.x;
                            float gate_y_948_1 = _cvt_f32_480.y;
                            float up_x_949_1 = _cvt_f32_481.x;
                            float up_y_950_1 = _cvt_f32_481.y;
                            float _exp_480 = expf(gate_x_947_1 * -1.0f);
                            float denominator_x_951_1 = _exp_480 + 1.0f;
                            float _exp_481 = expf(gate_y_948_1 * -1.0f);
                            float denominator_y_952_1 = _exp_481 + 1.0f;
                            float hidden_x_953_1 = gate_x_947_1 / denominator_x_951_1 * up_x_949_1;
                            float hidden_y_954_1 = gate_y_948_1 / denominator_y_952_1 * up_y_950_1;
                            __nv_bfloat162 _bf16x2_754 = __float22bfloat162_rn(make_float2(hidden_x_953_1, hidden_y_954_1));
                            hidden_packed_814_1[16] = __as_u32(_bf16x2_754);
                            float2 _cvt_f32_482 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[17]));
                            float2 _cvt_f32_483 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[17]));
                            float gate_x_955_1 = _cvt_f32_482.x;
                            float gate_y_956_1 = _cvt_f32_482.y;
                            float up_x_957_1 = _cvt_f32_483.x;
                            float up_y_958_1 = _cvt_f32_483.y;
                            float _exp_482 = expf(gate_x_955_1 * -1.0f);
                            float denominator_x_959_1 = _exp_482 + 1.0f;
                            float _exp_483 = expf(gate_y_956_1 * -1.0f);
                            float denominator_y_960_1 = _exp_483 + 1.0f;
                            float hidden_x_961_1 = gate_x_955_1 / denominator_x_959_1 * up_x_957_1;
                            float hidden_y_962_1 = gate_y_956_1 / denominator_y_960_1 * up_y_958_1;
                            __nv_bfloat162 _bf16x2_755 = __float22bfloat162_rn(make_float2(hidden_x_961_1, hidden_y_962_1));
                            hidden_packed_814_1[17] = __as_u32(_bf16x2_755);
                            float2 _cvt_f32_484 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[18]));
                            float2 _cvt_f32_485 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[18]));
                            float gate_x_963_1 = _cvt_f32_484.x;
                            float gate_y_964_1 = _cvt_f32_484.y;
                            float up_x_965_1 = _cvt_f32_485.x;
                            float up_y_966_1 = _cvt_f32_485.y;
                            float _exp_484 = expf(gate_x_963_1 * -1.0f);
                            float denominator_x_967_1 = _exp_484 + 1.0f;
                            float _exp_485 = expf(gate_y_964_1 * -1.0f);
                            float denominator_y_968_1 = _exp_485 + 1.0f;
                            float hidden_x_969_1 = gate_x_963_1 / denominator_x_967_1 * up_x_965_1;
                            float hidden_y_970_1 = gate_y_964_1 / denominator_y_968_1 * up_y_966_1;
                            __nv_bfloat162 _bf16x2_756 = __float22bfloat162_rn(make_float2(hidden_x_969_1, hidden_y_970_1));
                            hidden_packed_814_1[18] = __as_u32(_bf16x2_756);
                            float2 _cvt_f32_486 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[19]));
                            float2 _cvt_f32_487 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[19]));
                            float gate_x_971_1 = _cvt_f32_486.x;
                            float gate_y_972_1 = _cvt_f32_486.y;
                            float up_x_973_1 = _cvt_f32_487.x;
                            float up_y_974_1 = _cvt_f32_487.y;
                            float _exp_486 = expf(gate_x_971_1 * -1.0f);
                            float denominator_x_975_1 = _exp_486 + 1.0f;
                            float _exp_487 = expf(gate_y_972_1 * -1.0f);
                            float denominator_y_976_1 = _exp_487 + 1.0f;
                            float hidden_x_977_1 = gate_x_971_1 / denominator_x_975_1 * up_x_973_1;
                            float hidden_y_978_1 = gate_y_972_1 / denominator_y_976_1 * up_y_974_1;
                            __nv_bfloat162 _bf16x2_757 = __float22bfloat162_rn(make_float2(hidden_x_977_1, hidden_y_978_1));
                            hidden_packed_814_1[19] = __as_u32(_bf16x2_757);
                            float2 _cvt_f32_488 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[20]));
                            float2 _cvt_f32_489 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[20]));
                            float gate_x_979_1 = _cvt_f32_488.x;
                            float gate_y_980_1 = _cvt_f32_488.y;
                            float up_x_981_1 = _cvt_f32_489.x;
                            float up_y_982_1 = _cvt_f32_489.y;
                            float _exp_488 = expf(gate_x_979_1 * -1.0f);
                            float denominator_x_983_1 = _exp_488 + 1.0f;
                            float _exp_489 = expf(gate_y_980_1 * -1.0f);
                            float denominator_y_984_1 = _exp_489 + 1.0f;
                            float hidden_x_985_1 = gate_x_979_1 / denominator_x_983_1 * up_x_981_1;
                            float hidden_y_986_1 = gate_y_980_1 / denominator_y_984_1 * up_y_982_1;
                            __nv_bfloat162 _bf16x2_758 = __float22bfloat162_rn(make_float2(hidden_x_985_1, hidden_y_986_1));
                            hidden_packed_814_1[20] = __as_u32(_bf16x2_758);
                            float2 _cvt_f32_490 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[21]));
                            float2 _cvt_f32_491 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[21]));
                            float gate_x_987_1 = _cvt_f32_490.x;
                            float gate_y_988_1 = _cvt_f32_490.y;
                            float up_x_989_1 = _cvt_f32_491.x;
                            float up_y_990_1 = _cvt_f32_491.y;
                            float _exp_490 = expf(gate_x_987_1 * -1.0f);
                            float denominator_x_991_1 = _exp_490 + 1.0f;
                            float _exp_491 = expf(gate_y_988_1 * -1.0f);
                            float denominator_y_992_1 = _exp_491 + 1.0f;
                            float hidden_x_993_1 = gate_x_987_1 / denominator_x_991_1 * up_x_989_1;
                            float hidden_y_994_1 = gate_y_988_1 / denominator_y_992_1 * up_y_990_1;
                            __nv_bfloat162 _bf16x2_759 = __float22bfloat162_rn(make_float2(hidden_x_993_1, hidden_y_994_1));
                            hidden_packed_814_1[21] = __as_u32(_bf16x2_759);
                            float2 _cvt_f32_492 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[22]));
                            float2 _cvt_f32_493 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[22]));
                            float gate_x_995_1 = _cvt_f32_492.x;
                            float gate_y_996_1 = _cvt_f32_492.y;
                            float up_x_997_1 = _cvt_f32_493.x;
                            float up_y_998_1 = _cvt_f32_493.y;
                            float _exp_492 = expf(gate_x_995_1 * -1.0f);
                            float denominator_x_999_1 = _exp_492 + 1.0f;
                            float _exp_493 = expf(gate_y_996_1 * -1.0f);
                            float denominator_y_1000_1 = _exp_493 + 1.0f;
                            float hidden_x_1001_1 = gate_x_995_1 / denominator_x_999_1 * up_x_997_1;
                            float hidden_y_1002_1 = gate_y_996_1 / denominator_y_1000_1 * up_y_998_1;
                            __nv_bfloat162 _bf16x2_760 = __float22bfloat162_rn(make_float2(hidden_x_1001_1, hidden_y_1002_1));
                            hidden_packed_814_1[22] = __as_u32(_bf16x2_760);
                            float2 _cvt_f32_494 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[23]));
                            float2 _cvt_f32_495 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[23]));
                            float gate_x_1003_1 = _cvt_f32_494.x;
                            float gate_y_1004_1 = _cvt_f32_494.y;
                            float up_x_1005_1 = _cvt_f32_495.x;
                            float up_y_1006_1 = _cvt_f32_495.y;
                            float _exp_494 = expf(gate_x_1003_1 * -1.0f);
                            float denominator_x_1007_1 = _exp_494 + 1.0f;
                            float _exp_495 = expf(gate_y_1004_1 * -1.0f);
                            float denominator_y_1008_1 = _exp_495 + 1.0f;
                            float hidden_x_1009_1 = gate_x_1003_1 / denominator_x_1007_1 * up_x_1005_1;
                            float hidden_y_1010_1 = gate_y_1004_1 / denominator_y_1008_1 * up_y_1006_1;
                            __nv_bfloat162 _bf16x2_761 = __float22bfloat162_rn(make_float2(hidden_x_1009_1, hidden_y_1010_1));
                            hidden_packed_814_1[23] = __as_u32(_bf16x2_761);
                            float2 _cvt_f32_496 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[24]));
                            float2 _cvt_f32_497 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[24]));
                            float gate_x_1011_1 = _cvt_f32_496.x;
                            float gate_y_1012_1 = _cvt_f32_496.y;
                            float up_x_1013_1 = _cvt_f32_497.x;
                            float up_y_1014_1 = _cvt_f32_497.y;
                            float _exp_496 = expf(gate_x_1011_1 * -1.0f);
                            float denominator_x_1015_1 = _exp_496 + 1.0f;
                            float _exp_497 = expf(gate_y_1012_1 * -1.0f);
                            float denominator_y_1016_1 = _exp_497 + 1.0f;
                            float hidden_x_1017_1 = gate_x_1011_1 / denominator_x_1015_1 * up_x_1013_1;
                            float hidden_y_1018_1 = gate_y_1012_1 / denominator_y_1016_1 * up_y_1014_1;
                            __nv_bfloat162 _bf16x2_762 = __float22bfloat162_rn(make_float2(hidden_x_1017_1, hidden_y_1018_1));
                            hidden_packed_814_1[24] = __as_u32(_bf16x2_762);
                            float2 _cvt_f32_498 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[25]));
                            float2 _cvt_f32_499 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[25]));
                            float gate_x_1019_1 = _cvt_f32_498.x;
                            float gate_y_1020_1 = _cvt_f32_498.y;
                            float up_x_1021_1 = _cvt_f32_499.x;
                            float up_y_1022_1 = _cvt_f32_499.y;
                            float _exp_498 = expf(gate_x_1019_1 * -1.0f);
                            float denominator_x_1023_1 = _exp_498 + 1.0f;
                            float _exp_499 = expf(gate_y_1020_1 * -1.0f);
                            float denominator_y_1024_1 = _exp_499 + 1.0f;
                            float hidden_x_1025_1 = gate_x_1019_1 / denominator_x_1023_1 * up_x_1021_1;
                            float hidden_y_1026_1 = gate_y_1020_1 / denominator_y_1024_1 * up_y_1022_1;
                            __nv_bfloat162 _bf16x2_763 = __float22bfloat162_rn(make_float2(hidden_x_1025_1, hidden_y_1026_1));
                            hidden_packed_814_1[25] = __as_u32(_bf16x2_763);
                            float2 _cvt_f32_500 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[26]));
                            float2 _cvt_f32_501 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[26]));
                            float gate_x_1027_1 = _cvt_f32_500.x;
                            float gate_y_1028_1 = _cvt_f32_500.y;
                            float up_x_1029_1 = _cvt_f32_501.x;
                            float up_y_1030_1 = _cvt_f32_501.y;
                            float _exp_500 = expf(gate_x_1027_1 * -1.0f);
                            float denominator_x_1031_1 = _exp_500 + 1.0f;
                            float _exp_501 = expf(gate_y_1028_1 * -1.0f);
                            float denominator_y_1032_1 = _exp_501 + 1.0f;
                            float hidden_x_1033_1 = gate_x_1027_1 / denominator_x_1031_1 * up_x_1029_1;
                            float hidden_y_1034_1 = gate_y_1028_1 / denominator_y_1032_1 * up_y_1030_1;
                            __nv_bfloat162 _bf16x2_764 = __float22bfloat162_rn(make_float2(hidden_x_1033_1, hidden_y_1034_1));
                            hidden_packed_814_1[26] = __as_u32(_bf16x2_764);
                            float2 _cvt_f32_502 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[27]));
                            float2 _cvt_f32_503 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[27]));
                            float gate_x_1035_1 = _cvt_f32_502.x;
                            float gate_y_1036_1 = _cvt_f32_502.y;
                            float up_x_1037_1 = _cvt_f32_503.x;
                            float up_y_1038_1 = _cvt_f32_503.y;
                            float _exp_502 = expf(gate_x_1035_1 * -1.0f);
                            float denominator_x_1039_1 = _exp_502 + 1.0f;
                            float _exp_503 = expf(gate_y_1036_1 * -1.0f);
                            float denominator_y_1040_1 = _exp_503 + 1.0f;
                            float hidden_x_1041_1 = gate_x_1035_1 / denominator_x_1039_1 * up_x_1037_1;
                            float hidden_y_1042_1 = gate_y_1036_1 / denominator_y_1040_1 * up_y_1038_1;
                            __nv_bfloat162 _bf16x2_765 = __float22bfloat162_rn(make_float2(hidden_x_1041_1, hidden_y_1042_1));
                            hidden_packed_814_1[27] = __as_u32(_bf16x2_765);
                            float2 _cvt_f32_504 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[28]));
                            float2 _cvt_f32_505 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[28]));
                            float gate_x_1043_1 = _cvt_f32_504.x;
                            float gate_y_1044_1 = _cvt_f32_504.y;
                            float up_x_1045_1 = _cvt_f32_505.x;
                            float up_y_1046_1 = _cvt_f32_505.y;
                            float _exp_504 = expf(gate_x_1043_1 * -1.0f);
                            float denominator_x_1047_1 = _exp_504 + 1.0f;
                            float _exp_505 = expf(gate_y_1044_1 * -1.0f);
                            float denominator_y_1048_1 = _exp_505 + 1.0f;
                            float hidden_x_1049_1 = gate_x_1043_1 / denominator_x_1047_1 * up_x_1045_1;
                            float hidden_y_1050_1 = gate_y_1044_1 / denominator_y_1048_1 * up_y_1046_1;
                            __nv_bfloat162 _bf16x2_766 = __float22bfloat162_rn(make_float2(hidden_x_1049_1, hidden_y_1050_1));
                            hidden_packed_814_1[28] = __as_u32(_bf16x2_766);
                            float2 _cvt_f32_506 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[29]));
                            float2 _cvt_f32_507 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[29]));
                            float gate_x_1051_1 = _cvt_f32_506.x;
                            float gate_y_1052_1 = _cvt_f32_506.y;
                            float up_x_1053_1 = _cvt_f32_507.x;
                            float up_y_1054_1 = _cvt_f32_507.y;
                            float _exp_506 = expf(gate_x_1051_1 * -1.0f);
                            float denominator_x_1055_1 = _exp_506 + 1.0f;
                            float _exp_507 = expf(gate_y_1052_1 * -1.0f);
                            float denominator_y_1056_1 = _exp_507 + 1.0f;
                            float hidden_x_1057_1 = gate_x_1051_1 / denominator_x_1055_1 * up_x_1053_1;
                            float hidden_y_1058_1 = gate_y_1052_1 / denominator_y_1056_1 * up_y_1054_1;
                            __nv_bfloat162 _bf16x2_767 = __float22bfloat162_rn(make_float2(hidden_x_1057_1, hidden_y_1058_1));
                            hidden_packed_814_1[29] = __as_u32(_bf16x2_767);
                            float2 _cvt_f32_508 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[30]));
                            float2 _cvt_f32_509 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[30]));
                            float gate_x_1059_1 = _cvt_f32_508.x;
                            float gate_y_1060_1 = _cvt_f32_508.y;
                            float up_x_1061_1 = _cvt_f32_509.x;
                            float up_y_1062_1 = _cvt_f32_509.y;
                            float _exp_508 = expf(gate_x_1059_1 * -1.0f);
                            float denominator_x_1063_1 = _exp_508 + 1.0f;
                            float _exp_509 = expf(gate_y_1060_1 * -1.0f);
                            float denominator_y_1064_1 = _exp_509 + 1.0f;
                            float hidden_x_1065_1 = gate_x_1059_1 / denominator_x_1063_1 * up_x_1061_1;
                            float hidden_y_1066_1 = gate_y_1060_1 / denominator_y_1064_1 * up_y_1062_1;
                            __nv_bfloat162 _bf16x2_768 = __float22bfloat162_rn(make_float2(hidden_x_1065_1, hidden_y_1066_1));
                            hidden_packed_814_1[30] = __as_u32(_bf16x2_768);
                            float2 _cvt_f32_510 = __bfloat1622float2(__as_bf16x2(gate_packed_812_1[31]));
                            float2 _cvt_f32_511 = __bfloat1622float2(__as_bf16x2(up_packed_813_1[31]));
                            float gate_x_1067_1 = _cvt_f32_510.x;
                            float gate_y_1068_1 = _cvt_f32_510.y;
                            float up_x_1069_1 = _cvt_f32_511.x;
                            float up_y_1070_1 = _cvt_f32_511.y;
                            float _exp_510 = expf(gate_x_1067_1 * -1.0f);
                            float denominator_x_1071_1 = _exp_510 + 1.0f;
                            float _exp_511 = expf(gate_y_1068_1 * -1.0f);
                            float denominator_y_1072_1 = _exp_511 + 1.0f;
                            float hidden_x_1073_1 = gate_x_1067_1 / denominator_x_1071_1 * up_x_1069_1;
                            float hidden_y_1074_1 = gate_y_1068_1 / denominator_y_1072_1 * up_y_1070_1;
                            __nv_bfloat162 _bf16x2_769 = __float22bfloat162_rn(make_float2(hidden_x_1073_1, hidden_y_1074_1));
                            hidden_packed_814_1[31] = __as_u32(_bf16x2_769);
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1075_1 = tid / 32;
                            int lane_1076_1 = tid % 32;
                            #pragma unroll
                            for (int half_44 = 0; half_44 < 2; half_44++) {
                                #pragma unroll
                                for (int col_tile_44 = 0; col_tile_44 < 2; col_tile_44++) {
                                    int row_50 = warp_1075_1 * 32 + half_44 * 16 + lane_1076_1 % 16;
                                    int col_47 = col_tile_44 * 16 + lane_1076_1 / 16 * 8;
                                    unsigned int address_3_42 = d_smem_addr + (unsigned int)((row_50 * 32 + col_47) * 2);
                                    address_3_42 = address_3_42 ^ (address_3_42 & 511) >> 7 << 4;
                                    int offset_45 = half_44 * 8 + col_tile_44 * 4;
                                    uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(address_3_42);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_45])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_45 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_45 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_45 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1077_1 = tid / 32;
                            int lane_1078_1 = tid % 32;
                            #pragma unroll
                            for (int half_45 = 0; half_45 < 2; half_45++) {
                                #pragma unroll
                                for (int col_tile_45 = 0; col_tile_45 < 2; col_tile_45++) {
                                    int row_51 = warp_1077_1 * 32 + half_45 * 16 + lane_1078_1 % 16;
                                    int col_48 = col_tile_45 * 16 + lane_1078_1 / 16 * 8;
                                    unsigned int address_3_43 = d_smem_addr + 8192 + (unsigned int)((row_51 * 32 + col_48) * 2);
                                    address_3_43 = address_3_43 ^ (address_3_43 & 511) >> 7 << 4;
                                    int offset_46 = half_45 * 8 + col_tile_45 * 4;
                                    uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(address_3_43);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_46])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_46 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_46 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_46 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1079_1 = tid / 32;
                            int lane_1080_1 = tid % 32;
                            #pragma unroll
                            for (int half_46 = 0; half_46 < 2; half_46++) {
                                #pragma unroll
                                for (int col_tile_46 = 0; col_tile_46 < 2; col_tile_46++) {
                                    int row_52 = warp_1079_1 * 32 + half_46 * 16 + lane_1080_1 % 16;
                                    int col_49 = col_tile_46 * 16 + lane_1080_1 / 16 * 8;
                                    unsigned int address_3_44 = d_smem_addr + 16384 + (unsigned int)((row_52 * 32 + col_49) * 2);
                                    address_3_44 = address_3_44 ^ (address_3_44 & 511) >> 7 << 4;
                                    int offset_47 = half_46 * 8 + col_tile_46 * 4;
                                    uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(address_3_44);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_47])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_47 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_47 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_47 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1081_1 = tid / 32;
                            int lane_1082_1 = tid % 32;
                            #pragma unroll
                            for (int half_47 = 0; half_47 < 2; half_47++) {
                                #pragma unroll
                                for (int col_tile_47 = 0; col_tile_47 < 2; col_tile_47++) {
                                    int row_53 = warp_1081_1 * 32 + half_47 * 16 + lane_1082_1 % 16;
                                    int col_50 = col_tile_47 * 16 + lane_1082_1 / 16 * 8;
                                    unsigned int address_3_45 = d_smem_addr + (unsigned int)((row_53 * 32 + col_50) * 2);
                                    address_3_45 = address_3_45 ^ (address_3_45 & 511) >> 7 << 4;
                                    int offset_48 = 16 + half_47 * 8 + col_tile_47 * 4;
                                    uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(address_3_45);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_48])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_48 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_48 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&gate_packed_812_1[offset_48 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&gate_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1083_1 = tid / 32;
                            int lane_1084_1 = tid % 32;
                            #pragma unroll
                            for (int half_48 = 0; half_48 < 2; half_48++) {
                                #pragma unroll
                                for (int col_tile_48 = 0; col_tile_48 < 2; col_tile_48++) {
                                    int row_54 = warp_1083_1 * 32 + half_48 * 16 + lane_1084_1 % 16;
                                    int col_51 = col_tile_48 * 16 + lane_1084_1 / 16 * 8;
                                    unsigned int address_3_46 = d_smem_addr + 8192 + (unsigned int)((row_54 * 32 + col_51) * 2);
                                    address_3_46 = address_3_46 ^ (address_3_46 & 511) >> 7 << 4;
                                    int offset_49 = 16 + half_48 * 8 + col_tile_48 * 4;
                                    uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(address_3_46);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_49])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_49 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_49 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&up_packed_813_1[offset_49 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&up_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_1085_1 = tid / 32;
                            int lane_1086_1 = tid % 32;
                            #pragma unroll
                            for (int half_49 = 0; half_49 < 2; half_49++) {
                                #pragma unroll
                                for (int col_tile_49 = 0; col_tile_49 < 2; col_tile_49++) {
                                    int row_55 = warp_1085_1 * 32 + half_49 * 16 + lane_1086_1 % 16;
                                    int col_52 = col_tile_49 * 16 + lane_1086_1 / 16 * 8;
                                    unsigned int address_3_47 = d_smem_addr + 16384 + (unsigned int)((row_55 * 32 + col_52) * 2);
                                    address_3_47 = address_3_47 ^ (address_3_47 & 511) >> 7 << 4;
                                    int offset_50 = 16 + half_49 * 8 + col_tile_49 * 4;
                                    uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(address_3_47);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_50])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_50 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_50 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_814_1[offset_50 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 6 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 16384), "l"(0x12F0000000000000ULL) : "memory");
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
                                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(hidden_ready)) + (shared_rows + macro_1 * (macro_size / 256) + x_2))), "r"(static_cast<unsigned int>(2)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    gemm_bits = phase_bits_4;
                } else {
                    int col_blocks_4 = (hidden + 512 - 1) / 512;
                    int x_3 = -1;
                    int y_3 = -1;
                    int expert_3 = -1;
                    int k_start_3 = 0;
                    int k_end_3 = 0;
                    int first_3 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                    int _min_35 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int end_block_1 = _min_35;
                    int block_6 = first_block_1 + (task_1 - mini_fused) / col_blocks_4;
                    if (block_6 < end_block_1) {
                        int index_1 = counts[3 * experts + block_6];
                        int offset_51 = counts[experts + index_1] / 256;
                        int _max_6 = ((first_block_1) > (offset_51) ? (first_block_1) : (offset_51));
                        int first_row_8 = _max_6;
                        int _min_36 = ((end_block_1) < (offset_51 + counts[index_1] / 256) ? (end_block_1) : (offset_51 + counts[index_1] / 256));
                        int rows_7 = _min_36 - first_row_8;
                        int supergroup_3 = (task_1 - mini_fused - (first_row_8 - first_block_1) * col_blocks_4) / (rows_7 * 8);
                        int full_cols_3 = col_blocks_4 / 8 * 8;
                        int row_56 = 0;
                        int col_53 = 0;
                        if (task_1 - mini_fused - (first_row_8 - first_block_1) * col_blocks_4 < rows_7 * full_cols_3) {
                            row_56 = (task_1 - mini_fused - (first_row_8 - first_block_1) * col_blocks_4) % (rows_7 * 8) / 8;
                            col_53 = supergroup_3 * 8 + (task_1 - mini_fused - (first_row_8 - first_block_1) * col_blocks_4) % 8;
                        } else {
                            row_56 = (task_1 - mini_fused - (first_row_8 - first_block_1) * col_blocks_4 - rows_7 * full_cols_3) / (col_blocks_4 - full_cols_3);
                            col_53 = full_cols_3 + (task_1 - mini_fused - (first_row_8 - first_block_1) * col_blocks_4 - rows_7 * full_cols_3) % (col_blocks_4 - full_cols_3);
                        }
                        if ((supergroup_3 & 1) != 0) {
                            row_56 = rows_7 - row_56 - 1;
                        }
                        x_3 = first_row_8 + row_56 - macro_1 * (macro_size / 256);
                        y_3 = col_53;
                        expert_3 = index_1;
                    }
                    unsigned int phase_bits_5 = gemm_bits;
                    int has_hi_3 = 0;
                    has_hi_3 = (int)((y_3 * 2 + 1) * 256 < hidden);
                    int global_mini_3 = macro_1 * (macro_size / mini_size) + mini_3;
                    int macro_rows_4 = macro_1 * (macro_size / 256);
                    int iterations_3 = intermediate / 64;
                    if (expert_3 < 0) {
                        if (tid == 0) {
                        }
                    } else if (tid / 32 == 7) {
                        if (warp == 7) {
                            if (elect_sync()) {
                                {
                                    bool enabled_value_5 = 1;
                                    if (enabled_value_5 != 0) {
                                        int32_t _relaxed_ld_12;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_12) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                        int value_6 = _relaxed_ld_12;
                                        while (value_6 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_13;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_13) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                            value_6 = _relaxed_ld_13;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_37 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                    int _max_7 = ((0) > (_min_37) ? (0) : (_min_37));
                                    int mini_rows_7 = _max_7;
                                    int required_5 = (mini_rows_7 + 127) / 128 * ((intermediate + 511) / 512);
                                }
                                int ring_6 = 0;
                                #pragma unroll 1
                                for (int idx_6 = 0; idx_6 < iterations_3; idx_6++) {
                                    mbarrier_wait(gemm_finished_addr + (ring_6) * 8, phase_bits_5 >> (unsigned int)(16 + ring_6) & 1);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)(ring_6 * 16384)), "l"((&hidden_routed_in)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)(ring_6 * 16384)), "l"((&wd_routed)), "r"(0), "r"(y_3 * 2 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_3), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)(ring_6 * 16384)), "l"((&wd_routed)), "r"(0), "r"((y_3 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_3), "r"(0),
                                           "r"(((gemm_arrived_addr + (ring_6) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << 16 + ring_6);
                                    ring_6 = (ring_6 + 1) % 4;
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int ring_7 = 0;
                                    mbarrier_wait(output_finished_addr, phase_bits_5 >> 22 & 1);
                                    phase_bits_5 = phase_bits_5 ^ 4194304;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_7 = 0; idx_7 < iterations_3; idx_7++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + (ring_7) * 8, 98304);
                                        mbarrier_wait(gemm_arrived_addr + (ring_7) * 8, phase_bits_5 >> (unsigned int)ring_7 & 1);
                                        int _mma_a_lo_6 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_7) * 1024;
                                        int _mma_b_lo_6 = (((b_smem_addr) >> 4) & 0x3FFF) + (ring_7) * 1024;
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
            :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)));
                                        int _mma_a_lo_7 = (((a_smem_addr) >> 4) & 0x3FFF) + (ring_7) * 1024;
                                        int _mma_b_lo_7 = (((b_hi_addr) >> 4) & 0x3FFF) + (ring_7) * 1024;
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
            :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_accumulator + (256))), "r"(((idx_7 == 0) ? 0 : 1)));
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + (ring_7) * 8, (uint16_t)(3));
                                        phase_bits_5 = phase_bits_5 ^ (unsigned int)(1 << ring_7);
                                        ring_7 = (ring_7 + 1) % 4;
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            mbarrier_wait(output_arrived_addr, phase_bits_5 >> 6 & 1);
                            phase_bits_5 = phase_bits_5 ^ 64;
                            unsigned int packed_1[128];
                            #pragma unroll
                            for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                #pragma unroll
                                for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                    unsigned int address_9 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                    float _tmem_load_66[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_66[15]))
                                        : "r"(address_9));
                                    #pragma unroll
                                    for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                                        __nv_bfloat162 _bf16x2_770 = __float22bfloat162_rn(make_float2(_tmem_load_66[pair_2 * 2], _tmem_load_66[pair_2 * 2 + 1]));
                                        packed_1[chunk_4 * 16 + sub_2 * 8 + pair_2] = __as_u32(_bf16x2_770);
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
                                int _min_38 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                if (output_row_1 < _min_38) {
                                    bool enabled_value_6 = 1;
                                    if (enabled_value_6 != 0) {
                                        int32_t _relaxed_ld_14;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_14) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                        int value_7 = _relaxed_ld_14;
                                        while (value_7 < 8 * ((hidden + 1023) / 1024)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int32_t _relaxed_ld_15;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_15) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                            value_7 = _relaxed_ld_15;
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
                                int warp_0_2 = tid / 32;
                                int lane_5 = tid % 32;
                                #pragma unroll
                                for (int half_50 = 0; half_50 < 2; half_50++) {
                                    #pragma unroll
                                    for (int col_tile_50 = 0; col_tile_50 < 2; col_tile_50++) {
                                        int row_57 = warp_0_2 * 32 + half_50 * 16 + lane_5 % 16;
                                        int col_54 = col_tile_50 * 16 + lane_5 / 16 * 8;
                                        unsigned int address_10 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_57 * 32 + col_54) * 2);
                                        address_10 = address_10 ^ (address_10 & 511) >> 7 << 4;
                                        int offset_52 = chunk_5 * 16 + half_50 * 8 + col_tile_50 * 4;
                                        uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(address_10);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_52])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_52 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_52 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_52 + 3]))
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
                                        unsigned int address_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                        float _tmem_load_67[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_67[15]))
                                            : "r"(address_11));
                                        #pragma unroll
                                        for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                            __nv_bfloat162 _bf16x2_771 = __float22bfloat162_rn(make_float2(_tmem_load_67[pair_3 * 2], _tmem_load_67[pair_3 * 2 + 1]));
                                            packed_0_1[chunk_6 * 16 + sub_3 * 8 + pair_3] = __as_u32(_bf16x2_771);
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
                                    int lane_6 = tid % 32;
                                    #pragma unroll
                                    for (int half_51 = 0; half_51 < 2; half_51++) {
                                        #pragma unroll
                                        for (int col_tile_51 = 0; col_tile_51 < 2; col_tile_51++) {
                                            int row_58 = warp_0_3 * 32 + half_51 * 16 + lane_6 % 16;
                                            int col_55 = col_tile_51 * 16 + lane_6 / 16 * 8;
                                            unsigned int address_12 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_58 * 32 + col_55) * 2);
                                            address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                            int offset_53 = chunk_7 * 16 + half_51 * 8 + col_tile_51 * 4;
                                            uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(address_12);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_53])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_53 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_53 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[offset_53 + 3]))
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
                            if (tid / 32 == 0) {
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        bool enabled_value_7 = 1;
                                        if (enabled_value_7 != 0) {
                                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_3))), "r"(static_cast<unsigned int>(1)) : "memory");
                                        }
                                        if (has_hi_3 != 0) {
                                            asm volatile("cp.async.bulk.wait_group 0;");
                                            bool enabled_value_0 = 1;
                                            if (enabled_value_0 != 0) {
                                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(y_ready)) + (global_mini_3))), "r"(static_cast<unsigned int>(1)) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    gemm_bits = phase_bits_5;
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
