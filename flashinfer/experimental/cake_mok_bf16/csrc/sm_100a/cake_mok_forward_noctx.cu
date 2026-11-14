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

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mok_forward_noctx(const __grid_constant__ CUtensorMap x_shared, const __grid_constant__ CUtensorMap x_routed, const __grid_constant__ CUtensorMap wg_shared, const __grid_constant__ CUtensorMap wu_shared, const __grid_constant__ CUtensorMap wd_shared, const __grid_constant__ CUtensorMap wg_routed, const __grid_constant__ CUtensorMap wu_routed, const __grid_constant__ CUtensorMap wd_routed, const __grid_constant__ CUtensorMap gate_shared_out, const __grid_constant__ CUtensorMap up_shared_out, const __grid_constant__ CUtensorMap gate_routed_out, const __grid_constant__ CUtensorMap up_routed_out, const __grid_constant__ CUtensorMap hidden_shared_out, const __grid_constant__ CUtensorMap hidden_routed_out, const __grid_constant__ CUtensorMap hidden_shared_in, const __grid_constant__ CUtensorMap hidden_routed_in, const __grid_constant__ CUtensorMap y_shared, const __grid_constant__ CUtensorMap y_routed, __nv_bfloat16* __restrict__ x_routed_ptr, __nv_bfloat16* __restrict__ y_routed_ptr, unsigned long long* __restrict__ x_peers, unsigned long long* __restrict__ y_peers, int* __restrict__ schedule_rank, int* __restrict__ schedule_token, int* __restrict__ num_tokens, int* __restrict__ counts, int* __restrict__ hidden_ready, int* __restrict__ x_ready, int* __restrict__ y_ready, int* __restrict__ y_done, int local_tokens, int hidden, int intermediate, int experts, int topk, int comm_sms, int macro_size, int mini_size, float swiglu_limit, CakeFastDivmod mini_tasks_div, CakeFastDivmod minis_per_macro_div, CakeFastDivmod fused_col_blocks_div, CakeFastDivmod down_col_blocks_div)
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
    #define output_finished_addr (mbar_base + 80)
    #define schedule_arrived_addr (mbar_base + 96)
    #define schedule_finished_addr (mbar_base + 104)
    #define drain_arrived_0_addr (mbar_base + 112)
    #define drain_arrived_1_addr (mbar_base + 120)
    #define drain_arrived_2_addr (mbar_base + 128)
    #define drain_arrived_3_addr (mbar_base + 136)
    #define drain_arrived_4_addr (mbar_base + 144)
    #define drain_arrived_5_addr (mbar_base + 152)
    #define drain_arrived_6_addr (mbar_base + 160)
    #define drain_arrived_7_addr (mbar_base + 168)
    #define drain_finished_0_addr (mbar_base + 176)
    #define drain_finished_1_addr (mbar_base + 184)
    #define drain_finished_2_addr (mbar_base + 192)
    #define drain_finished_3_addr (mbar_base + 200)
    #define drain_finished_4_addr (mbar_base + 208)
    #define drain_finished_5_addr (mbar_base + 216)
    #define drain_finished_6_addr (mbar_base + 224)
    #define drain_finished_7_addr (mbar_base + 232)
    #define dispatch_arrived_addr (mbar_base + 240)
    #define range_arrived_addr (mbar_base + 248)
    #define rows_arrived_addr (mbar_base + 272)
    #define combine_arrived_addr (mbar_base + 296)

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
    __nv_bfloat16* d_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int d_smem_addr = smem + 206848;
    __nv_bfloat16* flat_view = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr = smem + 1024;
    unsigned int* flat_view_0 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int flat_view_addr_0 = smem + 1024;
    float* flat_view_1 = reinterpret_cast<float*>(smem_raw + 197632);
    const int flat_view_addr_1 = smem + 197632;
    int* flat_view_2 = reinterpret_cast<int*>(smem_raw + 199168);
    const int flat_view_addr_2 = smem + 199168;
    int* flat_view_3 = reinterpret_cast<int*>(smem_raw + 197632);
    const int flat_view_addr_3 = smem + 197632;
    float* flat_view_4 = reinterpret_cast<float*>(smem_raw + 199680);
    const int flat_view_addr_4 = smem + 199680;
    __nv_bfloat16* flat_view_5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int flat_view_addr_5 = smem + 1024;

    // Kernel setup ops
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
    if (true_clusters <= bid / 2) {
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");

    // Mbarrier init (44 barriers at smem_raw[0..352))

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
        // output_arrived: 2 barriers, init_count=1
        mbarrier_init(smem + 64, 1);
        mbarrier_init(smem + 72, 1);
        // output_finished: 2 barriers, init_count=2
        mbarrier_init(smem + 80, 2);
        mbarrier_init(smem + 88, 2);
        // --- pipeline 'schedule_pipe' ---
        // schedule_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 96, 1);
        // schedule_finished: 1 barriers, init_count=16
        mbarrier_init(smem + 104, 16);
        // --- pipeline 'drain_pipe_0' ---
        // drain_arrived_0: 1 barriers, init_count=1
        mbarrier_init(smem + 112, 1);
        // --- pipeline 'drain_pipe_1' ---
        // drain_arrived_1: 1 barriers, init_count=1
        mbarrier_init(smem + 120, 1);
        // --- pipeline 'drain_pipe_2' ---
        // drain_arrived_2: 1 barriers, init_count=1
        mbarrier_init(smem + 128, 1);
        // --- pipeline 'drain_pipe_3' ---
        // drain_arrived_3: 1 barriers, init_count=1
        mbarrier_init(smem + 136, 1);
        // --- pipeline 'drain_pipe_4' ---
        // drain_arrived_4: 1 barriers, init_count=1
        mbarrier_init(smem + 144, 1);
        // --- pipeline 'drain_pipe_5' ---
        // drain_arrived_5: 1 barriers, init_count=1
        mbarrier_init(smem + 152, 1);
        // --- pipeline 'drain_pipe_6' ---
        // drain_arrived_6: 1 barriers, init_count=1
        mbarrier_init(smem + 160, 1);
        // --- pipeline 'drain_pipe_7' ---
        // drain_arrived_7: 1 barriers, init_count=1
        mbarrier_init(smem + 168, 1);
        // --- pipeline 'drain_pipe_0' ---
        // drain_finished_0: 1 barriers, init_count=2
        mbarrier_init(smem + 176, 2);
        // --- pipeline 'drain_pipe_1' ---
        // drain_finished_1: 1 barriers, init_count=2
        mbarrier_init(smem + 184, 2);
        // --- pipeline 'drain_pipe_2' ---
        // drain_finished_2: 1 barriers, init_count=2
        mbarrier_init(smem + 192, 2);
        // --- pipeline 'drain_pipe_3' ---
        // drain_finished_3: 1 barriers, init_count=2
        mbarrier_init(smem + 200, 2);
        // --- pipeline 'drain_pipe_4' ---
        // drain_finished_4: 1 barriers, init_count=2
        mbarrier_init(smem + 208, 2);
        // --- pipeline 'drain_pipe_5' ---
        // drain_finished_5: 1 barriers, init_count=2
        mbarrier_init(smem + 216, 2);
        // --- pipeline 'drain_pipe_6' ---
        // drain_finished_6: 1 barriers, init_count=2
        mbarrier_init(smem + 224, 2);
        // --- pipeline 'drain_pipe_7' ---
        // drain_finished_7: 1 barriers, init_count=2
        mbarrier_init(smem + 232, 2);
        // dispatch_arrived: 1 barriers, init_count=1
        mbarrier_init(smem + 240, 1);
        // range_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 248, 1);
        mbarrier_init(smem + 256, 1);
        mbarrier_init(smem + 264, 1);
        // rows_arrived: 3 barriers, init_count=1
        mbarrier_init(smem + 272, 1);
        mbarrier_init(smem + 280, 1);
        mbarrier_init(smem + 288, 1);
        // combine_arrived: 7 barriers, init_count=1
        mbarrier_init(smem + 296, 1);
        mbarrier_init(smem + 304, 1);
        mbarrier_init(smem + 312, 1);
        mbarrier_init(smem + 320, 1);
        mbarrier_init(smem + 328, 1);
        mbarrier_init(smem + 336, 1);
        mbarrier_init(smem + 344, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    if (warp == 0) {
        int _tmem_hold = smem + 352;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
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
    int dispatch_seq = 0;
    int combine_seq_0 = 0;
    int combine_seq_1 = 0;
    int combine_seq_2 = 0;
    int combine_seq_3 = 0;
    int combine_seq_4 = 0;
    int combine_seq_5 = 0;
    int combine_seq_6 = 0;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (cluster < comm_clusters) {
        int comm_cta = cluster * 2 + cta_rank_0;
        if (macros > 0) {
            int col_blocks = (hidden + 511) / 512;
            int unit_rows = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int block_units = (128 + unit_rows - 1) / unit_rows;
            int macro_offset = (macros - 1) * macro_size;
            int _min_0 = ((macro_size) < (tokens - macro_offset) ? (macro_size) : (tokens - macro_offset));
            int macro_blocks = _min_0 / 128;
            int blocks = 0;
            if (comm_cta < macro_blocks) {
                blocks = (macro_blocks - comm_cta + comm_sms - 1) / comm_sms;
            }
            int units = blocks * block_units;
            int _min_0_1 = ((units) < (2) ? (units) : (2));
            #pragma unroll 1
            for (int unit = 0; unit < _min_0_1; unit++) {
                int buffer = unit % 3;
                int block = comm_cta + unit / block_units * comm_sms;
                int first_row = block * 128 + unit % block_units * unit_rows;
                int rows = ((unit_rows) < (block * 128 + 128 - first_row) ? (unit_rows) : (block * 128 + 128 - first_row));
                int peer = -1;
                int peer_token = -1;
                if (rows > tid) {
                    peer = schedule_rank[macro_offset + first_row + tid];
                    peer_token = schedule_token[macro_offset + first_row + tid];
                    flat_view_2[tid] = peer;
                }
                uint32_t cta_count = __syncthreads_count(peer >= 0);
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(range_arrived_addr + (buffer) * 8, (unsigned int)(cta_count * (unsigned int)hidden * 2));
                }
                if (cta_count < (unsigned int)rows) {
                    #pragma unroll 1
                    for (int pad = 0; pad < rows; pad++) {
                        int pad_peer = flat_view_2[pad];
                        if (pad_peer < 0) {
                            #pragma unroll 1
                            for (int vec = tid; vec < hidden / 8; vec += 256) {
                                asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_0) + (buffer * 65536 + pad * hidden * 2 + vec * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                            }
                        }
                    }
                }
                __syncthreads();
                if (peer >= 0) {
                    cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)((buffer * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer])) + (unsigned long long)((unsigned long long)(peer_token / topk) * (unsigned long long)hidden) * (unsigned long long)2), hidden * 2, range_arrived_addr + (buffer) * 8);
                }
            }
            #pragma unroll 1
            for (int unit_1 = 0; unit_1 < units; unit_1++) {
                int buffer_1 = unit_1 % 3;
                int block_1 = comm_cta + unit_1 / block_units * comm_sms;
                int first_row_1 = block_1 * 128 + unit_1 % block_units * unit_rows;
                int rows_1 = ((unit_rows) < (block_1 * 128 + 128 - first_row_1) ? (unit_rows) : (block_1 * 128 + 128 - first_row_1));
                int _parity_range_arrived = (int)((unsigned int)unit_1 / 3 % 2);
                mbarrier_wait(range_arrived_addr + (buffer_1) * 8, _parity_range_arrived);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid == 0) {
                    {
                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(x_routed_ptr + (unsigned long long)first_row_1 * (unsigned long long)hidden);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_1), "r"(flat_view_addr + (unsigned int)(buffer_1 * 65536)), "r"((uint32_t)((unsigned int)(rows_1 * hidden * 2)))
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
                    int block_0 = comm_cta + (unit_1 + 2) / block_units * comm_sms;
                    int first_row_0 = block_0 * 128 + (unit_1 + 2) % block_units * unit_rows;
                    int rows_0 = ((unit_rows) < (block_0 * 128 + 128 - first_row_0) ? (unit_rows) : (block_0 * 128 + 128 - first_row_0));
                    int peer_1 = -1;
                    int peer_token_1 = -1;
                    if (rows_0 > tid) {
                        peer_1 = schedule_rank[macro_offset + first_row_0 + tid];
                        peer_token_1 = schedule_token[macro_offset + first_row_0 + tid];
                        flat_view_2[tid] = peer_1;
                    }
                    uint32_t cta_count_1 = __syncthreads_count(peer_1 >= 0);
                    if (tid == 0) {
                        mbarrier_arrive_expect_tx(range_arrived_addr + (buffer_0) * 8, (unsigned int)(cta_count_1 * (unsigned int)hidden * 2));
                    }
                    if (cta_count_1 < (unsigned int)rows_0) {
                        #pragma unroll 1
                        for (int pad_1 = 0; pad_1 < rows_0; pad_1++) {
                            int pad_peer_1 = flat_view_2[pad_1];
                            if (pad_peer_1 < 0) {
                                #pragma unroll 1
                                for (int vec_1 = tid; vec_1 < hidden / 8; vec_1 += 256) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view_0) + (buffer_0 * 65536 + pad_1 * hidden * 2 + vec_1 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                        }
                    }
                    __syncthreads();
                    if (peer_1 >= 0) {
                        cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)((buffer_0 * 32768 + tid * hidden) * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_1])) + (unsigned long long)((unsigned long long)(peer_token_1 / topk) * (unsigned long long)hidden) * (unsigned long long)2), hidden * 2, range_arrived_addr + (buffer_0) * 8);
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
                int macro_rows = ((macro_size) < (tokens - macro * macro_size) ? (macro_size) : (tokens - macro * macro_size));
                int combine_tasks = (macro_rows / 16 * ((hidden + 1023) / 1024) + 6) / 7;
                int previous_rows = ((macro_size) < (tokens - (macro - 1) * macro_size) ? (macro_size) : (tokens - (macro - 1) * macro_size));
                int dispatch_tasks = previous_rows / 128 * ((hidden + 511) / 512);
                int _max_1 = ((combine_tasks) > (dispatch_tasks) ? (combine_tasks) : (dispatch_tasks));
                #pragma unroll 1
                for (int task = comm_cta; task < _max_1; task += comm_sms) {
                    if (combine_tasks > task) {
                        int col_blocks_0 = (hidden + 1023) / 1024;
                        int first_tile = task * 7;
                        int macro_offset_1 = macro * macro_size;
                        int macro_tokens = ((macro_size) < (tokens - macro_offset_1) ? (macro_size) : (tokens - macro_offset_1));
                        int valid_tiles = ((7) < (macro_tokens / 16 * col_blocks_0 - first_tile) ? (7) : (macro_tokens / 16 * col_blocks_0 - first_tile));
                        if (valid_tiles > 0) {
                            int first_row_2 = first_tile / col_blocks_0 * 16 + tid;
                            int first_col = first_tile % col_blocks_0;
                            int rows_2[7];
                            int columns[7];
                            int peers[7];
                            int tokens_0[7];
                            unsigned int counts_1[7];
                            int row = first_row_2;
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
                                        uint32_t _barrier_vote_2 = __syncthreads_count(peers[stage] >= 0);
                                        counts_1[stage] = _barrier_vote_2;
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
                                int first_mini = (macro_offset_1 + first_row_2) / mini_size;
                                int last_mini = (macro_offset_1 + (first_tile + valid_tiles - 1) / col_blocks_0 * 16) / mini_size;
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
                                    int _min_3 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                    unsigned int chunk_bytes = (unsigned int)(_min_3 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr, counts_1[0] * chunk_bytes);
                                }
                                if (valid_tiles > 1) {
                                    int _min_4 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                    unsigned int chunk_bytes_1 = (unsigned int)(_min_4 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 8, counts_1[1] * chunk_bytes_1);
                                }
                                if (valid_tiles > 2) {
                                    int _min_5 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                    unsigned int chunk_bytes_2 = (unsigned int)(_min_5 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 16, counts_1[2] * chunk_bytes_2);
                                }
                                if (valid_tiles > 3) {
                                    int _min_6 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                    unsigned int chunk_bytes_3 = (unsigned int)(_min_6 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 24, counts_1[3] * chunk_bytes_3);
                                }
                                if (valid_tiles > 4) {
                                    int _min_7 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                    unsigned int chunk_bytes_4 = (unsigned int)(_min_7 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 32, counts_1[4] * chunk_bytes_4);
                                }
                                if (valid_tiles > 5) {
                                    int _min_8 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                    unsigned int chunk_bytes_5 = (unsigned int)(_min_8 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 40, counts_1[5] * chunk_bytes_5);
                                }
                                if (valid_tiles > 6) {
                                    int _min_9 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                    unsigned int chunk_bytes_6 = (unsigned int)(_min_9 * 2);
                                    mbarrier_arrive_expect_tx(combine_arrived_addr + 48, counts_1[6] * chunk_bytes_6);
                                }
                            }
                            __syncthreads();
                            if (peers[0] >= 0) {
                                int chunk_cols = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)(tid * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)) * (unsigned long long)2), chunk_cols * 2, combine_arrived_addr);
                            }
                            if (peers[1] >= 0) {
                                int chunk_cols_1 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((16 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)) * (unsigned long long)2), chunk_cols_1 * 2, combine_arrived_addr + 8);
                            }
                            if (peers[2] >= 0) {
                                int chunk_cols_2 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((32 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)) * (unsigned long long)2), chunk_cols_2 * 2, combine_arrived_addr + 16);
                            }
                            if (peers[3] >= 0) {
                                int chunk_cols_3 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((48 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)) * (unsigned long long)2), chunk_cols_3 * 2, combine_arrived_addr + 24);
                            }
                            if (peers[4] >= 0) {
                                int chunk_cols_4 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((64 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)) * (unsigned long long)2), chunk_cols_4 * 2, combine_arrived_addr + 32);
                            }
                            if (peers[5] >= 0) {
                                int chunk_cols_5 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((80 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)) * (unsigned long long)2), chunk_cols_5 * 2, combine_arrived_addr + 40);
                            }
                            if (peers[6] >= 0) {
                                int chunk_cols_6 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                cp_async_bulk_gmem2smem(flat_view_addr_5 + (unsigned int)((96 + tid) * 1024 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)rows_2[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)) * (unsigned long long)2), chunk_cols_6 * 2, combine_arrived_addr + 48);
                            }
                            if (valid_tiles > 0) {
                                int _parity_combine_arrived = (int)((unsigned int)combine_seq_0 % 2);
                                mbarrier_wait(combine_arrived_addr, _parity_combine_arrived);
                                combine_seq_0 += 1;
                                if (peers[0] >= 0) {
                                    int _min_10 = ((1024) < (hidden - columns[0] * 1024) ? (1024) : (hidden - columns[0] * 1024));
                                    unsigned int chunk_bytes_7 = (unsigned int)(_min_10 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_11 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[0]]) + ((unsigned long long)tokens_0[0] * (unsigned long long)hidden + (unsigned long long)(columns[0] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_11), "r"(flat_view_addr_5 + (unsigned int)(tid * 2048)), "r"((uint32_t)(chunk_bytes_7))
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
                                    int _min_12 = ((1024) < (hidden - columns[1] * 1024) ? (1024) : (hidden - columns[1] * 1024));
                                    unsigned int chunk_bytes_8 = (unsigned int)(_min_12 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_13 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[1]]) + ((unsigned long long)tokens_0[1] * (unsigned long long)hidden + (unsigned long long)(columns[1] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_13), "r"(flat_view_addr_5 + (unsigned int)((16 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_8))
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
                                    int _min_14 = ((1024) < (hidden - columns[2] * 1024) ? (1024) : (hidden - columns[2] * 1024));
                                    unsigned int chunk_bytes_9 = (unsigned int)(_min_14 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_15 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[2]]) + ((unsigned long long)tokens_0[2] * (unsigned long long)hidden + (unsigned long long)(columns[2] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_15), "r"(flat_view_addr_5 + (unsigned int)((32 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_9))
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
                                    int _min_16 = ((1024) < (hidden - columns[3] * 1024) ? (1024) : (hidden - columns[3] * 1024));
                                    unsigned int chunk_bytes_10 = (unsigned int)(_min_16 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_17 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[3]]) + ((unsigned long long)tokens_0[3] * (unsigned long long)hidden + (unsigned long long)(columns[3] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_17), "r"(flat_view_addr_5 + (unsigned int)((48 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_10))
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
                                    int _min_18 = ((1024) < (hidden - columns[4] * 1024) ? (1024) : (hidden - columns[4] * 1024));
                                    unsigned int chunk_bytes_11 = (unsigned int)(_min_18 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_19 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[4]]) + ((unsigned long long)tokens_0[4] * (unsigned long long)hidden + (unsigned long long)(columns[4] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_19), "r"(flat_view_addr_5 + (unsigned int)((64 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_11))
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
                                    int _min_20 = ((1024) < (hidden - columns[5] * 1024) ? (1024) : (hidden - columns[5] * 1024));
                                    unsigned int chunk_bytes_12 = (unsigned int)(_min_20 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_21 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[5]]) + ((unsigned long long)tokens_0[5] * (unsigned long long)hidden + (unsigned long long)(columns[5] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_21), "r"(flat_view_addr_5 + (unsigned int)((80 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_12))
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
                                    int _min_22 = ((1024) < (hidden - columns[6] * 1024) ? (1024) : (hidden - columns[6] * 1024));
                                    unsigned int chunk_bytes_13 = (unsigned int)(_min_22 * 2);
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    {
                                        void* _cpbulk_dst_23 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[peers[6]]) + ((unsigned long long)tokens_0[6] * (unsigned long long)hidden + (unsigned long long)(columns[6] * 1024)));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_23), "r"(flat_view_addr_5 + (unsigned int)((96 + tid) * 2048)), "r"((uint32_t)(chunk_bytes_13))
                                            : "memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
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
                    }
                    if (dispatch_tasks > task) {
                        int col_blocks_0_1 = (hidden + 511) / 512;
                        int macro_offset_1_1 = (macro - 1) * macro_size;
                        int macro_tokens_1 = ((macro_size) < (tokens - macro_offset_1_1) ? (macro_size) : (tokens - macro_offset_1_1));
                        if (task < macro_tokens_1 / 128 * col_blocks_0_1) {
                            int row_1 = task / col_blocks_0_1 * 128;
                            int col_block = task % col_blocks_0_1;
                            int chunk_cols_7 = ((512) < (hidden - col_block * 512) ? (512) : (hidden - col_block * 512));
                            unsigned int chunk_bytes_14 = (unsigned int)(chunk_cols_7 * 2);
                            int peer_2 = -1;
                            int peer_token_2 = -1;
                            if (tid < 128) {
                                peer_2 = schedule_rank[macro_offset_1_1 + row_1 + tid];
                                peer_token_2 = schedule_token[macro_offset_1_1 + row_1 + tid];
                            }
                            uint32_t _cta_count_2 = __syncthreads_count(peer_2 >= 0);
                            if (tid == 0) {
                                int previous_offset = macro * macro_size;
                                int previous_tokens = ((macro_size) < (tokens - previous_offset) ? (macro_size) : (tokens - previous_offset));
                                if (row_1 < previous_tokens) {
                                    int previous_mini = (previous_offset + row_1) / mini_size;
                                    int mini_rows_1 = ((mini_size) < (tokens - previous_mini * mini_size) ? (mini_size) : (tokens - previous_mini * mini_size));
                                    int required_1 = (mini_rows_1 + 255) / 256 * (hidden / 256) * 2;
                                    bool enabled_value_1 = 1;
                                    if (enabled_value_1 != 0) {
                                        int poll_1_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_first) : "l"(y_ready + previous_mini) : "memory");
                                        int poll_1 = poll_1_first;
                                        while (poll_1 < required_1) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_1_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_1_next) : "l"(y_ready + previous_mini) : "memory");
                                            poll_1 = poll_1_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                mbarrier_arrive_expect_tx(dispatch_arrived_addr, _cta_count_2 * chunk_bytes_14);
                            }
                            __syncthreads();
                            if (peer_2 >= 0) {
                                cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)(tid * 512 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(reinterpret_cast<__nv_bfloat16*>(x_peers[peer_2])) + (unsigned long long)((unsigned long long)(peer_token_2 / topk) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)) * (unsigned long long)2), chunk_cols_7 * 2, dispatch_arrived_addr);
                            } else if (tid < 128) {
                                #pragma unroll
                                for (int vec_2 = 0; vec_2 < 64; vec_2++) {
                                    asm volatile("st.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"((uint64_t)((unsigned long long)(reinterpret_cast<uint8_t*>(flat_view) + (tid * 1024 + vec_2 * 16)))), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)), "r"(static_cast<uint32_t>((unsigned int)0)) : "memory");
                                }
                            }
                            int _parity_dispatch_arrived = (int)((unsigned int)dispatch_seq % 2);
                            mbarrier_wait(dispatch_arrived_addr, _parity_dispatch_arrived);
                            dispatch_seq += 1;
                            if (tid < 128) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                {
                                    void* _cpbulk_dst_24 = reinterpret_cast<void*>(x_routed_ptr + ((unsigned long long)(row_1 + tid) * (unsigned long long)hidden + (unsigned long long)(col_block * 512)));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                        :: "l"(_cpbulk_dst_24), "r"(flat_view_addr + (unsigned int)(tid * 1024)), "r"((uint32_t)(chunk_bytes_14))
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
                    }
                }
                macro = macro - 1;
            }
            int unit_rows_0 = ((128) < (65536 / (hidden * 2)) ? (128) : (65536 / (hidden * 2)));
            int block_units_1 = (128 + unit_rows_0 - 1) / unit_rows_0;
            int macro_offset_2 = 0;
            int _min_25 = ((macro_size) < (tokens - macro_offset_2) ? (macro_size) : (tokens - macro_offset_2));
            int macro_blocks_3 = _min_25 / 128;
            int blocks_4 = 0;
            if (comm_cta < macro_blocks_3) {
                blocks_4 = (macro_blocks_3 - comm_cta + comm_sms - 1) / comm_sms;
            }
            int units_5 = blocks_4 * block_units_1;
            int _min_1 = ((units_5) < (2) ? (units_5) : (2));
            #pragma unroll 1
            for (int unit_2 = 0; unit_2 < _min_1; unit_2++) {
                int slot = unit_2 % 3;
                int block_2 = comm_cta + unit_2 / block_units_1 * comm_sms;
                int first_row_3 = block_2 * 128 + unit_2 % block_units_1 * unit_rows_0;
                int _min_26 = ((unit_rows_0) < (block_2 * 128 + 128 - first_row_3) ? (unit_rows_0) : (block_2 * 128 + 128 - first_row_3));
                if (_min_26 > tid) {
                    flat_view_3[slot * 128 + tid] = schedule_rank[macro_offset_2 + first_row_3 + tid];
                    flat_view_3[384 + slot * 128 + tid] = schedule_token[macro_offset_2 + first_row_3 + tid];
                }
            }
            if (tid == 0) {
                int _min_2 = ((units_5) < (2) ? (units_5) : (2));
                #pragma unroll 1
                for (int unit_3 = 0; unit_3 < _min_2; unit_3++) {
                    int buffer_2 = unit_3 % 3;
                    int block_3 = comm_cta + unit_3 / block_units_1 * comm_sms;
                    int first_row_4 = block_3 * 128 + unit_3 % block_units_1 * unit_rows_0;
                    int rows_3 = ((unit_rows_0) < (block_3 * 128 + 128 - first_row_4) ? (unit_rows_0) : (block_3 * 128 + 128 - first_row_4));
                    int mini_1 = (macro_offset_2 + first_row_4) / mini_size;
                    int mini_rows_2 = ((mini_size) < (tokens - mini_1 * mini_size) ? (mini_size) : (tokens - mini_1 * mini_size));
                    int poll_2_first;
                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_first) : "l"(y_ready + mini_1) : "memory");
                    int poll_2 = poll_2_first;
                    while (poll_2 < (mini_rows_2 + 255) / 256 * (hidden / 256) * 2) {
                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                        int poll_2_next;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_2_next) : "l"(y_ready + mini_1) : "memory");
                        poll_2 = poll_2_next;
                    }
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_2) * 8, (unsigned int)(rows_3 * hidden * 2));
                    cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)(buffer_2 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)first_row_4 * (unsigned long long)hidden) * (unsigned long long)2), rows_3 * hidden * 2, rows_arrived_addr + (buffer_2) * 8);
                }
            }
            __syncthreads();
            #pragma unroll 1
            for (int unit_4 = 0; unit_4 < units_5; unit_4++) {
                int buffer_3 = unit_4 % 3;
                int block_4 = comm_cta + unit_4 / block_units_1 * comm_sms;
                int first_row_5 = block_4 * 128 + unit_4 % block_units_1 * unit_rows_0;
                int rows_4 = ((unit_rows_0) < (block_4 * 128 + 128 - first_row_5) ? (unit_rows_0) : (block_4 * 128 + 128 - first_row_5));
                int _parity_rows_arrived = (int)((unsigned int)unit_4 / 3 % 2);
                mbarrier_wait(rows_arrived_addr + (buffer_3) * 8, _parity_rows_arrived);
                if (rows_4 > tid) {
                    int peer_3 = flat_view_3[buffer_3 * 128 + tid];
                    int token = flat_view_3[384 + buffer_3 * 128 + tid];
                    if (peer_3 >= 0) {
                    }
                }
                if (tid == 0) {
                    #pragma unroll 1
                    for (int row_2 = 0; row_2 < rows_4; row_2++) {
                        int row_peer = flat_view_3[buffer_3 * 128 + row_2];
                        int row_token = flat_view_3[384 + buffer_3 * 128 + row_2];
                        if (row_peer >= 0) {
                            {
                                void* _cpbulk_dst_27 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(y_peers[row_peer]) + (unsigned long long)row_token * (unsigned long long)hidden);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_27), "r"(flat_view_addr + (unsigned int)(buffer_3 * 65536) + (unsigned int)(row_2 * hidden * 2)), "r"((uint32_t)((unsigned int)(hidden * 2)))
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
                        int buffer_1_1 = (unit_4 + 2) % 3;
                        int block_1_1 = comm_cta + (unit_4 + 2) / block_units_1 * comm_sms;
                        int first_row_1_1 = block_1_1 * 128 + (unit_4 + 2) % block_units_1 * unit_rows_0;
                        int rows_1_1 = ((unit_rows_0) < (block_1_1 * 128 + 128 - first_row_1_1) ? (unit_rows_0) : (block_1_1 * 128 + 128 - first_row_1_1));
                        int mini_2 = (macro_offset_2 + first_row_1_1) / mini_size;
                        int mini_rows_3 = ((mini_size) < (tokens - mini_2 * mini_size) ? (mini_size) : (tokens - mini_2 * mini_size));
                        int poll_3_first;
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_first) : "l"(y_ready + mini_2) : "memory");
                        int poll_3 = poll_3_first;
                        while (poll_3 < (mini_rows_3 + 255) / 256 * (hidden / 256) * 2) {
                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                            int poll_3_next;
                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_3_next) : "l"(y_ready + mini_2) : "memory");
                            poll_3 = poll_3_next;
                        }
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        mbarrier_arrive_expect_tx(rows_arrived_addr + (buffer_1_1) * 8, (unsigned int)(rows_1_1 * hidden * 2));
                        cp_async_bulk_gmem2smem(flat_view_addr + (unsigned int)(buffer_1_1 * 32768 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(y_routed_ptr) + (unsigned long long)((unsigned long long)first_row_1_1 * (unsigned long long)hidden) * (unsigned long long)2), rows_1_1 * hidden * 2, rows_arrived_addr + (buffer_1_1) * 8);
                    }
                }
                if (units_5 > unit_4 + 2) {
                    int slot_1 = (unit_4 + 2) % 3;
                    int block_2_1 = comm_cta + (unit_4 + 2) / block_units_1 * comm_sms;
                    int first_row_2_1 = block_2_1 * 128 + (unit_4 + 2) % block_units_1 * unit_rows_0;
                    int _min_28 = ((unit_rows_0) < (block_2_1 * 128 + 128 - first_row_2_1) ? (unit_rows_0) : (block_2_1 * 128 + 128 - first_row_2_1));
                    if (_min_28 > tid) {
                        flat_view_3[slot_1 * 128 + tid] = schedule_rank[macro_offset_2 + first_row_2_1 + tid];
                        flat_view_3[384 + slot_1 * 128 + tid] = schedule_token[macro_offset_2 + first_row_2_1 + tid];
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
                                int mini_rows_4 = ((0) > (_min_29) ? (0) : (_min_29));
                                int required_2 = (mini_rows_4 + 127) / 128 * ((hidden + 511) / 512);
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
                                if (cta_rank_0 == 0) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_shared)), "r"(0), "r"(y * 256), "r"(idx), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                } else {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wu_shared)), "r"(0), "r"(y * 256), "r"(idx), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                }
                                if (cta_rank_0 == 0) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wg_shared)), "r"(0), "r"(y * 256 + 128), "r"(idx), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                } else {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx) % 4 * 16384)), "l"((&wu_shared)), "r"(0), "r"(y * 256 + 128), "r"(idx), "r"(expert), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                }
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int head = ((3) < (iterations) ? (3) : (iterations));
                                int tail_1 = 0;
                                int _min_30 = ((3) < (iterations - head) ? (3) : (iterations - head));
                                tail_1 = _min_30;
                                int _stage_output_finished = (int)((unsigned int)gemm_acc_tiles % 2);
                                int _parity_output_finished = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished) * 8, _parity_output_finished);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_1 = 0; idx_1 < head; idx_1++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_1) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived = (int)((unsigned int)(gemm_wide + idx_1) % 4);
                                    int _parity_gemm_arrived = (int)((unsigned int)(gemm_wide + idx_1) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived) * 8, _parity_gemm_arrived);
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_31 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_32 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_1) % 4) * 1024;
#else
                                    int _smem_desc_lo_31 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
                                    int _smem_desc_lo_32 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_1) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_31), "r"(_smem_desc_lo_32), "r"(tmem_accumulator), "r"(((idx_1 == 0) ? 0 : 1)) : "memory");
                                }
                                int _stage_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                int _parity_output_finished_1 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished_1) * 8, _parity_output_finished_1);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_2 = 0; idx_2 < head; idx_2++) {
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_33 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_2) % 4) * 1024;
                                    int _smem_desc_lo_34 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_2) % 4) * 1024;
#else
                                    int _smem_desc_lo_33 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
                                    int _smem_desc_lo_34 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_2) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_33), "r"(_smem_desc_lo_34), "r"((tmem_accumulator + (256))), "r"(((idx_2 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_2) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_3 = head; idx_3 < iterations - tail_1; idx_3++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) % 4);
                                    int _parity_gemm_arrived_1 = (int)((unsigned int)(gemm_wide + idx_3) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_1) * 8, _parity_gemm_arrived_1);
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_35 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_36 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
#else
                                    int _smem_desc_lo_35 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_36 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_35), "r"(_smem_desc_lo_36), "r"(tmem_accumulator), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_37 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_38 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_3) % 4) * 1024;
#else
                                    int _smem_desc_lo_37 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
                                    int _smem_desc_lo_38 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_3) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_37), "r"(_smem_desc_lo_38), "r"((tmem_accumulator + (256))), "r"(((idx_3 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_3) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_4 = iterations - tail_1; idx_4 < iterations; idx_4++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_4) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) % 4);
                                    int _parity_gemm_arrived_2 = (int)((unsigned int)(gemm_wide + idx_4) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_2) * 8, _parity_gemm_arrived_2);
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_39 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_4) % 4) * 1024;
                                    int _smem_desc_lo_40 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_4) % 4) * 1024;
#else
                                    int _smem_desc_lo_39 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
                                    int _smem_desc_lo_40 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_4) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_39), "r"(_smem_desc_lo_40), "r"(tmem_accumulator), "r"(((idx_4 == 0) ? 0 : 1)) : "memory");
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                #pragma unroll 1
                                for (int idx_5 = iterations - tail_1; idx_5 < iterations; idx_5++) {
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_41 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_5) % 4) * 1024;
                                    int _smem_desc_lo_42 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_5) % 4) * 1024;
#else
                                    int _smem_desc_lo_41 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
                                    int _smem_desc_lo_42 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_5) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_41), "r"(_smem_desc_lo_42), "r"((tmem_accumulator + (256))), "r"(((idx_5 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_5) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        int _parity_output_arrived = (int)((unsigned int)gemm_out0 % 2);
                        mbarrier_wait(output_arrived_addr, _parity_output_arrived);
                        unsigned int gate_packed_0[32];
                        unsigned int up_packed_0[32];
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
                            : "r"(address_3_0 + 128));
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
                            : "r"(address_3_1 + 128));
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
                            : "r"(address_3_2 + 128));
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
                            : "r"(address_3_3 + 128));
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
                        unsigned int gate_packed_1[32];
                        unsigned int up_packed_1[32];
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
                            : "r"(address_3_4 + 128));
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
                            : "r"(address_3_5 + 128));
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
                            : "r"(address_3_6 + 128));
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
                            : "r"(address_3_7 + 128));
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
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        unsigned int hidden_packed[32];
                        float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[0]));
                        float2 _cvt_f32_256_0 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[0]));
                        float gate_x_1_0 = _cvt_f32_0.x;
                        float gate_y_1_0 = _cvt_f32_0.y;
                        float up_x_1_0 = _cvt_f32_256_0.x;
                        float up_y_1_0 = _cvt_f32_256_0.y;
                        float _exp_43 = expf(gate_x_1_0 * -1.0f);
                        float denominator_x_1_0 = _exp_43 + 1.0f;
                        float _exp_44 = expf(gate_y_1_0 * -1.0f);
                        float denominator_y_1_0 = _exp_44 + 1.0f;
                        float hidden_x_1_0 = gate_x_1_0 / denominator_x_1_0 * up_x_1_0;
                        float hidden_y_1_0 = gate_y_1_0 / denominator_y_1_0 * up_y_1_0;
                        hidden_packed[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_0, hidden_y_1_0)));
                        float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[1]));
                        float2 _cvt_f32_256_1 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[1]));
                        float gate_x_1_1 = _cvt_f32_1.x;
                        float gate_y_1_1 = _cvt_f32_1.y;
                        float up_x_1_1 = _cvt_f32_256_1.x;
                        float up_y_1_1 = _cvt_f32_256_1.y;
                        float _exp_45 = expf(gate_x_1_1 * -1.0f);
                        float denominator_x_1_1 = _exp_45 + 1.0f;
                        float _exp_46 = expf(gate_y_1_1 * -1.0f);
                        float denominator_y_1_1 = _exp_46 + 1.0f;
                        float hidden_x_1_1 = gate_x_1_1 / denominator_x_1_1 * up_x_1_1;
                        float hidden_y_1_1 = gate_y_1_1 / denominator_y_1_1 * up_y_1_1;
                        hidden_packed[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_1, hidden_y_1_1)));
                        float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[2]));
                        float2 _cvt_f32_256_2 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[2]));
                        float gate_x_1_2 = _cvt_f32_2.x;
                        float gate_y_1_2 = _cvt_f32_2.y;
                        float up_x_1_2 = _cvt_f32_256_2.x;
                        float up_y_1_2 = _cvt_f32_256_2.y;
                        float _exp_47 = expf(gate_x_1_2 * -1.0f);
                        float denominator_x_1_2 = _exp_47 + 1.0f;
                        float _exp_48 = expf(gate_y_1_2 * -1.0f);
                        float denominator_y_1_2 = _exp_48 + 1.0f;
                        float hidden_x_1_2 = gate_x_1_2 / denominator_x_1_2 * up_x_1_2;
                        float hidden_y_1_2 = gate_y_1_2 / denominator_y_1_2 * up_y_1_2;
                        hidden_packed[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_2, hidden_y_1_2)));
                        float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[3]));
                        float2 _cvt_f32_256_3 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[3]));
                        float gate_x_1_3 = _cvt_f32_3.x;
                        float gate_y_1_3 = _cvt_f32_3.y;
                        float up_x_1_3 = _cvt_f32_256_3.x;
                        float up_y_1_3 = _cvt_f32_256_3.y;
                        float _exp_49 = expf(gate_x_1_3 * -1.0f);
                        float denominator_x_1_3 = _exp_49 + 1.0f;
                        float _exp_50 = expf(gate_y_1_3 * -1.0f);
                        float denominator_y_1_3 = _exp_50 + 1.0f;
                        float hidden_x_1_3 = gate_x_1_3 / denominator_x_1_3 * up_x_1_3;
                        float hidden_y_1_3 = gate_y_1_3 / denominator_y_1_3 * up_y_1_3;
                        hidden_packed[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_3, hidden_y_1_3)));
                        float2 _cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[4]));
                        float2 _cvt_f32_256_4 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[4]));
                        float gate_x_1_4 = _cvt_f32_4.x;
                        float gate_y_1_4 = _cvt_f32_4.y;
                        float up_x_1_4 = _cvt_f32_256_4.x;
                        float up_y_1_4 = _cvt_f32_256_4.y;
                        float _exp_51 = expf(gate_x_1_4 * -1.0f);
                        float denominator_x_1_4 = _exp_51 + 1.0f;
                        float _exp_52 = expf(gate_y_1_4 * -1.0f);
                        float denominator_y_1_4 = _exp_52 + 1.0f;
                        float hidden_x_1_4 = gate_x_1_4 / denominator_x_1_4 * up_x_1_4;
                        float hidden_y_1_4 = gate_y_1_4 / denominator_y_1_4 * up_y_1_4;
                        hidden_packed[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_4, hidden_y_1_4)));
                        float2 _cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[5]));
                        float2 _cvt_f32_256_5 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[5]));
                        float gate_x_1_5 = _cvt_f32_5.x;
                        float gate_y_1_5 = _cvt_f32_5.y;
                        float up_x_1_5 = _cvt_f32_256_5.x;
                        float up_y_1_5 = _cvt_f32_256_5.y;
                        float _exp_53 = expf(gate_x_1_5 * -1.0f);
                        float denominator_x_1_5 = _exp_53 + 1.0f;
                        float _exp_54 = expf(gate_y_1_5 * -1.0f);
                        float denominator_y_1_5 = _exp_54 + 1.0f;
                        float hidden_x_1_5 = gate_x_1_5 / denominator_x_1_5 * up_x_1_5;
                        float hidden_y_1_5 = gate_y_1_5 / denominator_y_1_5 * up_y_1_5;
                        hidden_packed[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_5, hidden_y_1_5)));
                        float2 _cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[6]));
                        float2 _cvt_f32_256_6 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[6]));
                        float gate_x_1_6 = _cvt_f32_6.x;
                        float gate_y_1_6 = _cvt_f32_6.y;
                        float up_x_1_6 = _cvt_f32_256_6.x;
                        float up_y_1_6 = _cvt_f32_256_6.y;
                        float _exp_55 = expf(gate_x_1_6 * -1.0f);
                        float denominator_x_1_6 = _exp_55 + 1.0f;
                        float _exp_56 = expf(gate_y_1_6 * -1.0f);
                        float denominator_y_1_6 = _exp_56 + 1.0f;
                        float hidden_x_1_6 = gate_x_1_6 / denominator_x_1_6 * up_x_1_6;
                        float hidden_y_1_6 = gate_y_1_6 / denominator_y_1_6 * up_y_1_6;
                        hidden_packed[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_6, hidden_y_1_6)));
                        float2 _cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[7]));
                        float2 _cvt_f32_256_7 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[7]));
                        float gate_x_1_7 = _cvt_f32_7.x;
                        float gate_y_1_7 = _cvt_f32_7.y;
                        float up_x_1_7 = _cvt_f32_256_7.x;
                        float up_y_1_7 = _cvt_f32_256_7.y;
                        float _exp_57 = expf(gate_x_1_7 * -1.0f);
                        float denominator_x_1_7 = _exp_57 + 1.0f;
                        float _exp_58 = expf(gate_y_1_7 * -1.0f);
                        float denominator_y_1_7 = _exp_58 + 1.0f;
                        float hidden_x_1_7 = gate_x_1_7 / denominator_x_1_7 * up_x_1_7;
                        float hidden_y_1_7 = gate_y_1_7 / denominator_y_1_7 * up_y_1_7;
                        hidden_packed[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_7, hidden_y_1_7)));
                        float2 _cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[8]));
                        float2 _cvt_f32_256_8 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[8]));
                        float gate_x_1_8 = _cvt_f32_8.x;
                        float gate_y_1_8 = _cvt_f32_8.y;
                        float up_x_1_8 = _cvt_f32_256_8.x;
                        float up_y_1_8 = _cvt_f32_256_8.y;
                        float _exp_59 = expf(gate_x_1_8 * -1.0f);
                        float denominator_x_1_8 = _exp_59 + 1.0f;
                        float _exp_60 = expf(gate_y_1_8 * -1.0f);
                        float denominator_y_1_8 = _exp_60 + 1.0f;
                        float hidden_x_1_8 = gate_x_1_8 / denominator_x_1_8 * up_x_1_8;
                        float hidden_y_1_8 = gate_y_1_8 / denominator_y_1_8 * up_y_1_8;
                        hidden_packed[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_8, hidden_y_1_8)));
                        float2 _cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[9]));
                        float2 _cvt_f32_256_9 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[9]));
                        float gate_x_1_9 = _cvt_f32_9.x;
                        float gate_y_1_9 = _cvt_f32_9.y;
                        float up_x_1_9 = _cvt_f32_256_9.x;
                        float up_y_1_9 = _cvt_f32_256_9.y;
                        float _exp_61 = expf(gate_x_1_9 * -1.0f);
                        float denominator_x_1_9 = _exp_61 + 1.0f;
                        float _exp_62 = expf(gate_y_1_9 * -1.0f);
                        float denominator_y_1_9 = _exp_62 + 1.0f;
                        float hidden_x_1_9 = gate_x_1_9 / denominator_x_1_9 * up_x_1_9;
                        float hidden_y_1_9 = gate_y_1_9 / denominator_y_1_9 * up_y_1_9;
                        hidden_packed[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_9, hidden_y_1_9)));
                        float2 _cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[10]));
                        float2 _cvt_f32_256_10 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[10]));
                        float gate_x_1_10 = _cvt_f32_10.x;
                        float gate_y_1_10 = _cvt_f32_10.y;
                        float up_x_1_10 = _cvt_f32_256_10.x;
                        float up_y_1_10 = _cvt_f32_256_10.y;
                        float _exp_63 = expf(gate_x_1_10 * -1.0f);
                        float denominator_x_1_10 = _exp_63 + 1.0f;
                        float _exp_64 = expf(gate_y_1_10 * -1.0f);
                        float denominator_y_1_10 = _exp_64 + 1.0f;
                        float hidden_x_1_10 = gate_x_1_10 / denominator_x_1_10 * up_x_1_10;
                        float hidden_y_1_10 = gate_y_1_10 / denominator_y_1_10 * up_y_1_10;
                        hidden_packed[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_10, hidden_y_1_10)));
                        float2 _cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[11]));
                        float2 _cvt_f32_256_11 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[11]));
                        float gate_x_1_11 = _cvt_f32_11.x;
                        float gate_y_1_11 = _cvt_f32_11.y;
                        float up_x_1_11 = _cvt_f32_256_11.x;
                        float up_y_1_11 = _cvt_f32_256_11.y;
                        float _exp_65 = expf(gate_x_1_11 * -1.0f);
                        float denominator_x_1_11 = _exp_65 + 1.0f;
                        float _exp_66 = expf(gate_y_1_11 * -1.0f);
                        float denominator_y_1_11 = _exp_66 + 1.0f;
                        float hidden_x_1_11 = gate_x_1_11 / denominator_x_1_11 * up_x_1_11;
                        float hidden_y_1_11 = gate_y_1_11 / denominator_y_1_11 * up_y_1_11;
                        hidden_packed[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_11, hidden_y_1_11)));
                        float2 _cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[12]));
                        float2 _cvt_f32_256_12 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[12]));
                        float gate_x_1_12 = _cvt_f32_12.x;
                        float gate_y_1_12 = _cvt_f32_12.y;
                        float up_x_1_12 = _cvt_f32_256_12.x;
                        float up_y_1_12 = _cvt_f32_256_12.y;
                        float _exp_67 = expf(gate_x_1_12 * -1.0f);
                        float denominator_x_1_12 = _exp_67 + 1.0f;
                        float _exp_68 = expf(gate_y_1_12 * -1.0f);
                        float denominator_y_1_12 = _exp_68 + 1.0f;
                        float hidden_x_1_12 = gate_x_1_12 / denominator_x_1_12 * up_x_1_12;
                        float hidden_y_1_12 = gate_y_1_12 / denominator_y_1_12 * up_y_1_12;
                        hidden_packed[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_12, hidden_y_1_12)));
                        float2 _cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[13]));
                        float2 _cvt_f32_256_13 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[13]));
                        float gate_x_1_13 = _cvt_f32_13.x;
                        float gate_y_1_13 = _cvt_f32_13.y;
                        float up_x_1_13 = _cvt_f32_256_13.x;
                        float up_y_1_13 = _cvt_f32_256_13.y;
                        float _exp_69 = expf(gate_x_1_13 * -1.0f);
                        float denominator_x_1_13 = _exp_69 + 1.0f;
                        float _exp_70 = expf(gate_y_1_13 * -1.0f);
                        float denominator_y_1_13 = _exp_70 + 1.0f;
                        float hidden_x_1_13 = gate_x_1_13 / denominator_x_1_13 * up_x_1_13;
                        float hidden_y_1_13 = gate_y_1_13 / denominator_y_1_13 * up_y_1_13;
                        hidden_packed[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_13, hidden_y_1_13)));
                        float2 _cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[14]));
                        float2 _cvt_f32_256_14 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[14]));
                        float gate_x_1_14 = _cvt_f32_14.x;
                        float gate_y_1_14 = _cvt_f32_14.y;
                        float up_x_1_14 = _cvt_f32_256_14.x;
                        float up_y_1_14 = _cvt_f32_256_14.y;
                        float _exp_71 = expf(gate_x_1_14 * -1.0f);
                        float denominator_x_1_14 = _exp_71 + 1.0f;
                        float _exp_72 = expf(gate_y_1_14 * -1.0f);
                        float denominator_y_1_14 = _exp_72 + 1.0f;
                        float hidden_x_1_14 = gate_x_1_14 / denominator_x_1_14 * up_x_1_14;
                        float hidden_y_1_14 = gate_y_1_14 / denominator_y_1_14 * up_y_1_14;
                        hidden_packed[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_14, hidden_y_1_14)));
                        float2 _cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[15]));
                        float2 _cvt_f32_256_15 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[15]));
                        float gate_x_1_15 = _cvt_f32_15.x;
                        float gate_y_1_15 = _cvt_f32_15.y;
                        float up_x_1_15 = _cvt_f32_256_15.x;
                        float up_y_1_15 = _cvt_f32_256_15.y;
                        float _exp_73 = expf(gate_x_1_15 * -1.0f);
                        float denominator_x_1_15 = _exp_73 + 1.0f;
                        float _exp_74 = expf(gate_y_1_15 * -1.0f);
                        float denominator_y_1_15 = _exp_74 + 1.0f;
                        float hidden_x_1_15 = gate_x_1_15 / denominator_x_1_15 * up_x_1_15;
                        float hidden_y_1_15 = gate_y_1_15 / denominator_y_1_15 * up_y_1_15;
                        hidden_packed[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_15, hidden_y_1_15)));
                        float2 _cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[16]));
                        float2 _cvt_f32_256_16 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[16]));
                        float gate_x_1_16 = _cvt_f32_16.x;
                        float gate_y_1_16 = _cvt_f32_16.y;
                        float up_x_1_16 = _cvt_f32_256_16.x;
                        float up_y_1_16 = _cvt_f32_256_16.y;
                        float _exp_75 = expf(gate_x_1_16 * -1.0f);
                        float denominator_x_1_16 = _exp_75 + 1.0f;
                        float _exp_76 = expf(gate_y_1_16 * -1.0f);
                        float denominator_y_1_16 = _exp_76 + 1.0f;
                        float hidden_x_1_16 = gate_x_1_16 / denominator_x_1_16 * up_x_1_16;
                        float hidden_y_1_16 = gate_y_1_16 / denominator_y_1_16 * up_y_1_16;
                        hidden_packed[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_16, hidden_y_1_16)));
                        float2 _cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[17]));
                        float2 _cvt_f32_256_17 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[17]));
                        float gate_x_1_17 = _cvt_f32_17.x;
                        float gate_y_1_17 = _cvt_f32_17.y;
                        float up_x_1_17 = _cvt_f32_256_17.x;
                        float up_y_1_17 = _cvt_f32_256_17.y;
                        float _exp_77 = expf(gate_x_1_17 * -1.0f);
                        float denominator_x_1_17 = _exp_77 + 1.0f;
                        float _exp_78 = expf(gate_y_1_17 * -1.0f);
                        float denominator_y_1_17 = _exp_78 + 1.0f;
                        float hidden_x_1_17 = gate_x_1_17 / denominator_x_1_17 * up_x_1_17;
                        float hidden_y_1_17 = gate_y_1_17 / denominator_y_1_17 * up_y_1_17;
                        hidden_packed[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_17, hidden_y_1_17)));
                        float2 _cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[18]));
                        float2 _cvt_f32_256_18 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[18]));
                        float gate_x_1_18 = _cvt_f32_18.x;
                        float gate_y_1_18 = _cvt_f32_18.y;
                        float up_x_1_18 = _cvt_f32_256_18.x;
                        float up_y_1_18 = _cvt_f32_256_18.y;
                        float _exp_79 = expf(gate_x_1_18 * -1.0f);
                        float denominator_x_1_18 = _exp_79 + 1.0f;
                        float _exp_80 = expf(gate_y_1_18 * -1.0f);
                        float denominator_y_1_18 = _exp_80 + 1.0f;
                        float hidden_x_1_18 = gate_x_1_18 / denominator_x_1_18 * up_x_1_18;
                        float hidden_y_1_18 = gate_y_1_18 / denominator_y_1_18 * up_y_1_18;
                        hidden_packed[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_18, hidden_y_1_18)));
                        float2 _cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[19]));
                        float2 _cvt_f32_256_19 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[19]));
                        float gate_x_1_19 = _cvt_f32_19.x;
                        float gate_y_1_19 = _cvt_f32_19.y;
                        float up_x_1_19 = _cvt_f32_256_19.x;
                        float up_y_1_19 = _cvt_f32_256_19.y;
                        float _exp_81 = expf(gate_x_1_19 * -1.0f);
                        float denominator_x_1_19 = _exp_81 + 1.0f;
                        float _exp_82 = expf(gate_y_1_19 * -1.0f);
                        float denominator_y_1_19 = _exp_82 + 1.0f;
                        float hidden_x_1_19 = gate_x_1_19 / denominator_x_1_19 * up_x_1_19;
                        float hidden_y_1_19 = gate_y_1_19 / denominator_y_1_19 * up_y_1_19;
                        hidden_packed[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_19, hidden_y_1_19)));
                        float2 _cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[20]));
                        float2 _cvt_f32_256_20 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[20]));
                        float gate_x_1_20 = _cvt_f32_20.x;
                        float gate_y_1_20 = _cvt_f32_20.y;
                        float up_x_1_20 = _cvt_f32_256_20.x;
                        float up_y_1_20 = _cvt_f32_256_20.y;
                        float _exp_83 = expf(gate_x_1_20 * -1.0f);
                        float denominator_x_1_20 = _exp_83 + 1.0f;
                        float _exp_84 = expf(gate_y_1_20 * -1.0f);
                        float denominator_y_1_20 = _exp_84 + 1.0f;
                        float hidden_x_1_20 = gate_x_1_20 / denominator_x_1_20 * up_x_1_20;
                        float hidden_y_1_20 = gate_y_1_20 / denominator_y_1_20 * up_y_1_20;
                        hidden_packed[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_20, hidden_y_1_20)));
                        float2 _cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[21]));
                        float2 _cvt_f32_256_21 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[21]));
                        float gate_x_1_21 = _cvt_f32_21.x;
                        float gate_y_1_21 = _cvt_f32_21.y;
                        float up_x_1_21 = _cvt_f32_256_21.x;
                        float up_y_1_21 = _cvt_f32_256_21.y;
                        float _exp_85 = expf(gate_x_1_21 * -1.0f);
                        float denominator_x_1_21 = _exp_85 + 1.0f;
                        float _exp_86 = expf(gate_y_1_21 * -1.0f);
                        float denominator_y_1_21 = _exp_86 + 1.0f;
                        float hidden_x_1_21 = gate_x_1_21 / denominator_x_1_21 * up_x_1_21;
                        float hidden_y_1_21 = gate_y_1_21 / denominator_y_1_21 * up_y_1_21;
                        hidden_packed[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_21, hidden_y_1_21)));
                        float2 _cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[22]));
                        float2 _cvt_f32_256_22 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[22]));
                        float gate_x_1_22 = _cvt_f32_22.x;
                        float gate_y_1_22 = _cvt_f32_22.y;
                        float up_x_1_22 = _cvt_f32_256_22.x;
                        float up_y_1_22 = _cvt_f32_256_22.y;
                        float _exp_87 = expf(gate_x_1_22 * -1.0f);
                        float denominator_x_1_22 = _exp_87 + 1.0f;
                        float _exp_88 = expf(gate_y_1_22 * -1.0f);
                        float denominator_y_1_22 = _exp_88 + 1.0f;
                        float hidden_x_1_22 = gate_x_1_22 / denominator_x_1_22 * up_x_1_22;
                        float hidden_y_1_22 = gate_y_1_22 / denominator_y_1_22 * up_y_1_22;
                        hidden_packed[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_22, hidden_y_1_22)));
                        float2 _cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[23]));
                        float2 _cvt_f32_256_23 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[23]));
                        float gate_x_1_23 = _cvt_f32_23.x;
                        float gate_y_1_23 = _cvt_f32_23.y;
                        float up_x_1_23 = _cvt_f32_256_23.x;
                        float up_y_1_23 = _cvt_f32_256_23.y;
                        float _exp_89 = expf(gate_x_1_23 * -1.0f);
                        float denominator_x_1_23 = _exp_89 + 1.0f;
                        float _exp_90 = expf(gate_y_1_23 * -1.0f);
                        float denominator_y_1_23 = _exp_90 + 1.0f;
                        float hidden_x_1_23 = gate_x_1_23 / denominator_x_1_23 * up_x_1_23;
                        float hidden_y_1_23 = gate_y_1_23 / denominator_y_1_23 * up_y_1_23;
                        hidden_packed[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_23, hidden_y_1_23)));
                        float2 _cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[24]));
                        float2 _cvt_f32_256_24 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[24]));
                        float gate_x_1_24 = _cvt_f32_24.x;
                        float gate_y_1_24 = _cvt_f32_24.y;
                        float up_x_1_24 = _cvt_f32_256_24.x;
                        float up_y_1_24 = _cvt_f32_256_24.y;
                        float _exp_91 = expf(gate_x_1_24 * -1.0f);
                        float denominator_x_1_24 = _exp_91 + 1.0f;
                        float _exp_92 = expf(gate_y_1_24 * -1.0f);
                        float denominator_y_1_24 = _exp_92 + 1.0f;
                        float hidden_x_1_24 = gate_x_1_24 / denominator_x_1_24 * up_x_1_24;
                        float hidden_y_1_24 = gate_y_1_24 / denominator_y_1_24 * up_y_1_24;
                        hidden_packed[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_24, hidden_y_1_24)));
                        float2 _cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[25]));
                        float2 _cvt_f32_256_25 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[25]));
                        float gate_x_1_25 = _cvt_f32_25.x;
                        float gate_y_1_25 = _cvt_f32_25.y;
                        float up_x_1_25 = _cvt_f32_256_25.x;
                        float up_y_1_25 = _cvt_f32_256_25.y;
                        float _exp_93 = expf(gate_x_1_25 * -1.0f);
                        float denominator_x_1_25 = _exp_93 + 1.0f;
                        float _exp_94 = expf(gate_y_1_25 * -1.0f);
                        float denominator_y_1_25 = _exp_94 + 1.0f;
                        float hidden_x_1_25 = gate_x_1_25 / denominator_x_1_25 * up_x_1_25;
                        float hidden_y_1_25 = gate_y_1_25 / denominator_y_1_25 * up_y_1_25;
                        hidden_packed[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_25, hidden_y_1_25)));
                        float2 _cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[26]));
                        float2 _cvt_f32_256_26 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[26]));
                        float gate_x_1_26 = _cvt_f32_26.x;
                        float gate_y_1_26 = _cvt_f32_26.y;
                        float up_x_1_26 = _cvt_f32_256_26.x;
                        float up_y_1_26 = _cvt_f32_256_26.y;
                        float _exp_95 = expf(gate_x_1_26 * -1.0f);
                        float denominator_x_1_26 = _exp_95 + 1.0f;
                        float _exp_96 = expf(gate_y_1_26 * -1.0f);
                        float denominator_y_1_26 = _exp_96 + 1.0f;
                        float hidden_x_1_26 = gate_x_1_26 / denominator_x_1_26 * up_x_1_26;
                        float hidden_y_1_26 = gate_y_1_26 / denominator_y_1_26 * up_y_1_26;
                        hidden_packed[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_26, hidden_y_1_26)));
                        float2 _cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[27]));
                        float2 _cvt_f32_256_27 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[27]));
                        float gate_x_1_27 = _cvt_f32_27.x;
                        float gate_y_1_27 = _cvt_f32_27.y;
                        float up_x_1_27 = _cvt_f32_256_27.x;
                        float up_y_1_27 = _cvt_f32_256_27.y;
                        float _exp_97 = expf(gate_x_1_27 * -1.0f);
                        float denominator_x_1_27 = _exp_97 + 1.0f;
                        float _exp_98 = expf(gate_y_1_27 * -1.0f);
                        float denominator_y_1_27 = _exp_98 + 1.0f;
                        float hidden_x_1_27 = gate_x_1_27 / denominator_x_1_27 * up_x_1_27;
                        float hidden_y_1_27 = gate_y_1_27 / denominator_y_1_27 * up_y_1_27;
                        hidden_packed[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_27, hidden_y_1_27)));
                        float2 _cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[28]));
                        float2 _cvt_f32_256_28 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[28]));
                        float gate_x_1_28 = _cvt_f32_28.x;
                        float gate_y_1_28 = _cvt_f32_28.y;
                        float up_x_1_28 = _cvt_f32_256_28.x;
                        float up_y_1_28 = _cvt_f32_256_28.y;
                        float _exp_99 = expf(gate_x_1_28 * -1.0f);
                        float denominator_x_1_28 = _exp_99 + 1.0f;
                        float _exp_100 = expf(gate_y_1_28 * -1.0f);
                        float denominator_y_1_28 = _exp_100 + 1.0f;
                        float hidden_x_1_28 = gate_x_1_28 / denominator_x_1_28 * up_x_1_28;
                        float hidden_y_1_28 = gate_y_1_28 / denominator_y_1_28 * up_y_1_28;
                        hidden_packed[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_28, hidden_y_1_28)));
                        float2 _cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[29]));
                        float2 _cvt_f32_256_29 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[29]));
                        float gate_x_1_29 = _cvt_f32_29.x;
                        float gate_y_1_29 = _cvt_f32_29.y;
                        float up_x_1_29 = _cvt_f32_256_29.x;
                        float up_y_1_29 = _cvt_f32_256_29.y;
                        float _exp_101 = expf(gate_x_1_29 * -1.0f);
                        float denominator_x_1_29 = _exp_101 + 1.0f;
                        float _exp_102 = expf(gate_y_1_29 * -1.0f);
                        float denominator_y_1_29 = _exp_102 + 1.0f;
                        float hidden_x_1_29 = gate_x_1_29 / denominator_x_1_29 * up_x_1_29;
                        float hidden_y_1_29 = gate_y_1_29 / denominator_y_1_29 * up_y_1_29;
                        hidden_packed[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_29, hidden_y_1_29)));
                        float2 _cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[30]));
                        float2 _cvt_f32_256_30 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[30]));
                        float gate_x_1_30 = _cvt_f32_30.x;
                        float gate_y_1_30 = _cvt_f32_30.y;
                        float up_x_1_30 = _cvt_f32_256_30.x;
                        float up_y_1_30 = _cvt_f32_256_30.y;
                        float _exp_103 = expf(gate_x_1_30 * -1.0f);
                        float denominator_x_1_30 = _exp_103 + 1.0f;
                        float _exp_104 = expf(gate_y_1_30 * -1.0f);
                        float denominator_y_1_30 = _exp_104 + 1.0f;
                        float hidden_x_1_30 = gate_x_1_30 / denominator_x_1_30 * up_x_1_30;
                        float hidden_y_1_30 = gate_y_1_30 / denominator_y_1_30 * up_y_1_30;
                        hidden_packed[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_30, hidden_y_1_30)));
                        float2 _cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_0[31]));
                        float2 _cvt_f32_256_31 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_0[31]));
                        float gate_x_1_31 = _cvt_f32_31.x;
                        float gate_y_1_31 = _cvt_f32_31.y;
                        float up_x_1_31 = _cvt_f32_256_31.x;
                        float up_y_1_31 = _cvt_f32_256_31.y;
                        float _exp_105 = expf(gate_x_1_31 * -1.0f);
                        float denominator_x_1_31 = _exp_105 + 1.0f;
                        float _exp_106 = expf(gate_y_1_31 * -1.0f);
                        float denominator_y_1_31 = _exp_106 + 1.0f;
                        float hidden_x_1_31 = gate_x_1_31 / denominator_x_1_31 * up_x_1_31;
                        float hidden_y_1_31 = gate_y_1_31 / denominator_y_1_31 * up_y_1_31;
                        hidden_packed[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_31, hidden_y_1_31)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_2 = tid / 32;
                        int lane_1 = tid % 32;
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            #pragma unroll
                            for (int col_tile = 0; col_tile < 2; col_tile++) {
                                int row_4 = warp_2 * 32 + half * 16 + lane_1 % 16;
                                int col_1 = col_tile * 16 + lane_1 / 16 * 8;
                                unsigned int address = d_smem_addr + (unsigned int)((row_4 * 32 + col_1) * 2);
                                address = address ^ (address & 511) >> 7 << 4;
                                int offset = half * 8 + col_tile * 4;
                                uint32_t _stmatrix_addr_107 = static_cast<uint32_t>(address);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_107), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_0 = tid / 32;
                        int lane_0 = tid % 32;
                        #pragma unroll
                        for (int half_1 = 0; half_1 < 2; half_1++) {
                            #pragma unroll
                            for (int col_tile_1 = 0; col_tile_1 < 2; col_tile_1++) {
                                int row_5 = warp_0 * 32 + half_1 * 16 + lane_0 % 16;
                                int col_2 = col_tile_1 * 16 + lane_0 / 16 * 8;
                                unsigned int address_1 = d_smem_addr + 8192 + (unsigned int)((row_5 * 32 + col_2) * 2);
                                address_1 = address_1 ^ (address_1 & 511) >> 7 << 4;
                                int offset_1 = 16 + half_1 * 8 + col_tile_1 * 4;
                                uint32_t _stmatrix_addr_108 = static_cast<uint32_t>(address_1);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_108), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed[offset_1 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        unsigned int hidden_packed_0[32];
                        float2 _cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[0]));
                        float2 _cvt_f32_256_32 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[0]));
                        float gate_x_1_32 = _cvt_f32_32.x;
                        float gate_y_1_32 = _cvt_f32_32.y;
                        float up_x_1_32 = _cvt_f32_256_32.x;
                        float up_y_1_32 = _cvt_f32_256_32.y;
                        float _exp_109 = expf(gate_x_1_32 * -1.0f);
                        float denominator_x_1_32 = _exp_109 + 1.0f;
                        float _exp_110 = expf(gate_y_1_32 * -1.0f);
                        float denominator_y_1_32 = _exp_110 + 1.0f;
                        float hidden_x_1_32 = gate_x_1_32 / denominator_x_1_32 * up_x_1_32;
                        float hidden_y_1_32 = gate_y_1_32 / denominator_y_1_32 * up_y_1_32;
                        hidden_packed_0[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_32, hidden_y_1_32)));
                        float2 _cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[1]));
                        float2 _cvt_f32_256_33 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[1]));
                        float gate_x_1_33 = _cvt_f32_33.x;
                        float gate_y_1_33 = _cvt_f32_33.y;
                        float up_x_1_33 = _cvt_f32_256_33.x;
                        float up_y_1_33 = _cvt_f32_256_33.y;
                        float _exp_111 = expf(gate_x_1_33 * -1.0f);
                        float denominator_x_1_33 = _exp_111 + 1.0f;
                        float _exp_112 = expf(gate_y_1_33 * -1.0f);
                        float denominator_y_1_33 = _exp_112 + 1.0f;
                        float hidden_x_1_33 = gate_x_1_33 / denominator_x_1_33 * up_x_1_33;
                        float hidden_y_1_33 = gate_y_1_33 / denominator_y_1_33 * up_y_1_33;
                        hidden_packed_0[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_33, hidden_y_1_33)));
                        float2 _cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[2]));
                        float2 _cvt_f32_256_34 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[2]));
                        float gate_x_1_34 = _cvt_f32_34.x;
                        float gate_y_1_34 = _cvt_f32_34.y;
                        float up_x_1_34 = _cvt_f32_256_34.x;
                        float up_y_1_34 = _cvt_f32_256_34.y;
                        float _exp_113 = expf(gate_x_1_34 * -1.0f);
                        float denominator_x_1_34 = _exp_113 + 1.0f;
                        float _exp_114 = expf(gate_y_1_34 * -1.0f);
                        float denominator_y_1_34 = _exp_114 + 1.0f;
                        float hidden_x_1_34 = gate_x_1_34 / denominator_x_1_34 * up_x_1_34;
                        float hidden_y_1_34 = gate_y_1_34 / denominator_y_1_34 * up_y_1_34;
                        hidden_packed_0[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_34, hidden_y_1_34)));
                        float2 _cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[3]));
                        float2 _cvt_f32_256_35 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[3]));
                        float gate_x_1_35 = _cvt_f32_35.x;
                        float gate_y_1_35 = _cvt_f32_35.y;
                        float up_x_1_35 = _cvt_f32_256_35.x;
                        float up_y_1_35 = _cvt_f32_256_35.y;
                        float _exp_115 = expf(gate_x_1_35 * -1.0f);
                        float denominator_x_1_35 = _exp_115 + 1.0f;
                        float _exp_116 = expf(gate_y_1_35 * -1.0f);
                        float denominator_y_1_35 = _exp_116 + 1.0f;
                        float hidden_x_1_35 = gate_x_1_35 / denominator_x_1_35 * up_x_1_35;
                        float hidden_y_1_35 = gate_y_1_35 / denominator_y_1_35 * up_y_1_35;
                        hidden_packed_0[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_35, hidden_y_1_35)));
                        float2 _cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[4]));
                        float2 _cvt_f32_256_36 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[4]));
                        float gate_x_1_36 = _cvt_f32_36.x;
                        float gate_y_1_36 = _cvt_f32_36.y;
                        float up_x_1_36 = _cvt_f32_256_36.x;
                        float up_y_1_36 = _cvt_f32_256_36.y;
                        float _exp_117 = expf(gate_x_1_36 * -1.0f);
                        float denominator_x_1_36 = _exp_117 + 1.0f;
                        float _exp_118 = expf(gate_y_1_36 * -1.0f);
                        float denominator_y_1_36 = _exp_118 + 1.0f;
                        float hidden_x_1_36 = gate_x_1_36 / denominator_x_1_36 * up_x_1_36;
                        float hidden_y_1_36 = gate_y_1_36 / denominator_y_1_36 * up_y_1_36;
                        hidden_packed_0[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_36, hidden_y_1_36)));
                        float2 _cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[5]));
                        float2 _cvt_f32_256_37 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[5]));
                        float gate_x_1_37 = _cvt_f32_37.x;
                        float gate_y_1_37 = _cvt_f32_37.y;
                        float up_x_1_37 = _cvt_f32_256_37.x;
                        float up_y_1_37 = _cvt_f32_256_37.y;
                        float _exp_119 = expf(gate_x_1_37 * -1.0f);
                        float denominator_x_1_37 = _exp_119 + 1.0f;
                        float _exp_120 = expf(gate_y_1_37 * -1.0f);
                        float denominator_y_1_37 = _exp_120 + 1.0f;
                        float hidden_x_1_37 = gate_x_1_37 / denominator_x_1_37 * up_x_1_37;
                        float hidden_y_1_37 = gate_y_1_37 / denominator_y_1_37 * up_y_1_37;
                        hidden_packed_0[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_37, hidden_y_1_37)));
                        float2 _cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[6]));
                        float2 _cvt_f32_256_38 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[6]));
                        float gate_x_1_38 = _cvt_f32_38.x;
                        float gate_y_1_38 = _cvt_f32_38.y;
                        float up_x_1_38 = _cvt_f32_256_38.x;
                        float up_y_1_38 = _cvt_f32_256_38.y;
                        float _exp_121 = expf(gate_x_1_38 * -1.0f);
                        float denominator_x_1_38 = _exp_121 + 1.0f;
                        float _exp_122 = expf(gate_y_1_38 * -1.0f);
                        float denominator_y_1_38 = _exp_122 + 1.0f;
                        float hidden_x_1_38 = gate_x_1_38 / denominator_x_1_38 * up_x_1_38;
                        float hidden_y_1_38 = gate_y_1_38 / denominator_y_1_38 * up_y_1_38;
                        hidden_packed_0[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_38, hidden_y_1_38)));
                        float2 _cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[7]));
                        float2 _cvt_f32_256_39 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[7]));
                        float gate_x_1_39 = _cvt_f32_39.x;
                        float gate_y_1_39 = _cvt_f32_39.y;
                        float up_x_1_39 = _cvt_f32_256_39.x;
                        float up_y_1_39 = _cvt_f32_256_39.y;
                        float _exp_123 = expf(gate_x_1_39 * -1.0f);
                        float denominator_x_1_39 = _exp_123 + 1.0f;
                        float _exp_124 = expf(gate_y_1_39 * -1.0f);
                        float denominator_y_1_39 = _exp_124 + 1.0f;
                        float hidden_x_1_39 = gate_x_1_39 / denominator_x_1_39 * up_x_1_39;
                        float hidden_y_1_39 = gate_y_1_39 / denominator_y_1_39 * up_y_1_39;
                        hidden_packed_0[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_39, hidden_y_1_39)));
                        float2 _cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[8]));
                        float2 _cvt_f32_256_40 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[8]));
                        float gate_x_1_40 = _cvt_f32_40.x;
                        float gate_y_1_40 = _cvt_f32_40.y;
                        float up_x_1_40 = _cvt_f32_256_40.x;
                        float up_y_1_40 = _cvt_f32_256_40.y;
                        float _exp_125 = expf(gate_x_1_40 * -1.0f);
                        float denominator_x_1_40 = _exp_125 + 1.0f;
                        float _exp_126 = expf(gate_y_1_40 * -1.0f);
                        float denominator_y_1_40 = _exp_126 + 1.0f;
                        float hidden_x_1_40 = gate_x_1_40 / denominator_x_1_40 * up_x_1_40;
                        float hidden_y_1_40 = gate_y_1_40 / denominator_y_1_40 * up_y_1_40;
                        hidden_packed_0[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_40, hidden_y_1_40)));
                        float2 _cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[9]));
                        float2 _cvt_f32_256_41 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[9]));
                        float gate_x_1_41 = _cvt_f32_41.x;
                        float gate_y_1_41 = _cvt_f32_41.y;
                        float up_x_1_41 = _cvt_f32_256_41.x;
                        float up_y_1_41 = _cvt_f32_256_41.y;
                        float _exp_127 = expf(gate_x_1_41 * -1.0f);
                        float denominator_x_1_41 = _exp_127 + 1.0f;
                        float _exp_128 = expf(gate_y_1_41 * -1.0f);
                        float denominator_y_1_41 = _exp_128 + 1.0f;
                        float hidden_x_1_41 = gate_x_1_41 / denominator_x_1_41 * up_x_1_41;
                        float hidden_y_1_41 = gate_y_1_41 / denominator_y_1_41 * up_y_1_41;
                        hidden_packed_0[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_41, hidden_y_1_41)));
                        float2 _cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[10]));
                        float2 _cvt_f32_256_42 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[10]));
                        float gate_x_1_42 = _cvt_f32_42.x;
                        float gate_y_1_42 = _cvt_f32_42.y;
                        float up_x_1_42 = _cvt_f32_256_42.x;
                        float up_y_1_42 = _cvt_f32_256_42.y;
                        float _exp_129 = expf(gate_x_1_42 * -1.0f);
                        float denominator_x_1_42 = _exp_129 + 1.0f;
                        float _exp_130 = expf(gate_y_1_42 * -1.0f);
                        float denominator_y_1_42 = _exp_130 + 1.0f;
                        float hidden_x_1_42 = gate_x_1_42 / denominator_x_1_42 * up_x_1_42;
                        float hidden_y_1_42 = gate_y_1_42 / denominator_y_1_42 * up_y_1_42;
                        hidden_packed_0[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_42, hidden_y_1_42)));
                        float2 _cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[11]));
                        float2 _cvt_f32_256_43 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[11]));
                        float gate_x_1_43 = _cvt_f32_43.x;
                        float gate_y_1_43 = _cvt_f32_43.y;
                        float up_x_1_43 = _cvt_f32_256_43.x;
                        float up_y_1_43 = _cvt_f32_256_43.y;
                        float _exp_131 = expf(gate_x_1_43 * -1.0f);
                        float denominator_x_1_43 = _exp_131 + 1.0f;
                        float _exp_132 = expf(gate_y_1_43 * -1.0f);
                        float denominator_y_1_43 = _exp_132 + 1.0f;
                        float hidden_x_1_43 = gate_x_1_43 / denominator_x_1_43 * up_x_1_43;
                        float hidden_y_1_43 = gate_y_1_43 / denominator_y_1_43 * up_y_1_43;
                        hidden_packed_0[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_43, hidden_y_1_43)));
                        float2 _cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[12]));
                        float2 _cvt_f32_256_44 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[12]));
                        float gate_x_1_44 = _cvt_f32_44.x;
                        float gate_y_1_44 = _cvt_f32_44.y;
                        float up_x_1_44 = _cvt_f32_256_44.x;
                        float up_y_1_44 = _cvt_f32_256_44.y;
                        float _exp_133 = expf(gate_x_1_44 * -1.0f);
                        float denominator_x_1_44 = _exp_133 + 1.0f;
                        float _exp_134 = expf(gate_y_1_44 * -1.0f);
                        float denominator_y_1_44 = _exp_134 + 1.0f;
                        float hidden_x_1_44 = gate_x_1_44 / denominator_x_1_44 * up_x_1_44;
                        float hidden_y_1_44 = gate_y_1_44 / denominator_y_1_44 * up_y_1_44;
                        hidden_packed_0[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_44, hidden_y_1_44)));
                        float2 _cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[13]));
                        float2 _cvt_f32_256_45 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[13]));
                        float gate_x_1_45 = _cvt_f32_45.x;
                        float gate_y_1_45 = _cvt_f32_45.y;
                        float up_x_1_45 = _cvt_f32_256_45.x;
                        float up_y_1_45 = _cvt_f32_256_45.y;
                        float _exp_135 = expf(gate_x_1_45 * -1.0f);
                        float denominator_x_1_45 = _exp_135 + 1.0f;
                        float _exp_136 = expf(gate_y_1_45 * -1.0f);
                        float denominator_y_1_45 = _exp_136 + 1.0f;
                        float hidden_x_1_45 = gate_x_1_45 / denominator_x_1_45 * up_x_1_45;
                        float hidden_y_1_45 = gate_y_1_45 / denominator_y_1_45 * up_y_1_45;
                        hidden_packed_0[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_45, hidden_y_1_45)));
                        float2 _cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[14]));
                        float2 _cvt_f32_256_46 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[14]));
                        float gate_x_1_46 = _cvt_f32_46.x;
                        float gate_y_1_46 = _cvt_f32_46.y;
                        float up_x_1_46 = _cvt_f32_256_46.x;
                        float up_y_1_46 = _cvt_f32_256_46.y;
                        float _exp_137 = expf(gate_x_1_46 * -1.0f);
                        float denominator_x_1_46 = _exp_137 + 1.0f;
                        float _exp_138 = expf(gate_y_1_46 * -1.0f);
                        float denominator_y_1_46 = _exp_138 + 1.0f;
                        float hidden_x_1_46 = gate_x_1_46 / denominator_x_1_46 * up_x_1_46;
                        float hidden_y_1_46 = gate_y_1_46 / denominator_y_1_46 * up_y_1_46;
                        hidden_packed_0[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_46, hidden_y_1_46)));
                        float2 _cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[15]));
                        float2 _cvt_f32_256_47 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[15]));
                        float gate_x_1_47 = _cvt_f32_47.x;
                        float gate_y_1_47 = _cvt_f32_47.y;
                        float up_x_1_47 = _cvt_f32_256_47.x;
                        float up_y_1_47 = _cvt_f32_256_47.y;
                        float _exp_139 = expf(gate_x_1_47 * -1.0f);
                        float denominator_x_1_47 = _exp_139 + 1.0f;
                        float _exp_140 = expf(gate_y_1_47 * -1.0f);
                        float denominator_y_1_47 = _exp_140 + 1.0f;
                        float hidden_x_1_47 = gate_x_1_47 / denominator_x_1_47 * up_x_1_47;
                        float hidden_y_1_47 = gate_y_1_47 / denominator_y_1_47 * up_y_1_47;
                        hidden_packed_0[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_47, hidden_y_1_47)));
                        float2 _cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[16]));
                        float2 _cvt_f32_256_48 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[16]));
                        float gate_x_1_48 = _cvt_f32_48.x;
                        float gate_y_1_48 = _cvt_f32_48.y;
                        float up_x_1_48 = _cvt_f32_256_48.x;
                        float up_y_1_48 = _cvt_f32_256_48.y;
                        float _exp_141 = expf(gate_x_1_48 * -1.0f);
                        float denominator_x_1_48 = _exp_141 + 1.0f;
                        float _exp_142 = expf(gate_y_1_48 * -1.0f);
                        float denominator_y_1_48 = _exp_142 + 1.0f;
                        float hidden_x_1_48 = gate_x_1_48 / denominator_x_1_48 * up_x_1_48;
                        float hidden_y_1_48 = gate_y_1_48 / denominator_y_1_48 * up_y_1_48;
                        hidden_packed_0[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_48, hidden_y_1_48)));
                        float2 _cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[17]));
                        float2 _cvt_f32_256_49 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[17]));
                        float gate_x_1_49 = _cvt_f32_49.x;
                        float gate_y_1_49 = _cvt_f32_49.y;
                        float up_x_1_49 = _cvt_f32_256_49.x;
                        float up_y_1_49 = _cvt_f32_256_49.y;
                        float _exp_143 = expf(gate_x_1_49 * -1.0f);
                        float denominator_x_1_49 = _exp_143 + 1.0f;
                        float _exp_144 = expf(gate_y_1_49 * -1.0f);
                        float denominator_y_1_49 = _exp_144 + 1.0f;
                        float hidden_x_1_49 = gate_x_1_49 / denominator_x_1_49 * up_x_1_49;
                        float hidden_y_1_49 = gate_y_1_49 / denominator_y_1_49 * up_y_1_49;
                        hidden_packed_0[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_49, hidden_y_1_49)));
                        float2 _cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[18]));
                        float2 _cvt_f32_256_50 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[18]));
                        float gate_x_1_50 = _cvt_f32_50.x;
                        float gate_y_1_50 = _cvt_f32_50.y;
                        float up_x_1_50 = _cvt_f32_256_50.x;
                        float up_y_1_50 = _cvt_f32_256_50.y;
                        float _exp_145 = expf(gate_x_1_50 * -1.0f);
                        float denominator_x_1_50 = _exp_145 + 1.0f;
                        float _exp_146 = expf(gate_y_1_50 * -1.0f);
                        float denominator_y_1_50 = _exp_146 + 1.0f;
                        float hidden_x_1_50 = gate_x_1_50 / denominator_x_1_50 * up_x_1_50;
                        float hidden_y_1_50 = gate_y_1_50 / denominator_y_1_50 * up_y_1_50;
                        hidden_packed_0[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_50, hidden_y_1_50)));
                        float2 _cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[19]));
                        float2 _cvt_f32_256_51 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[19]));
                        float gate_x_1_51 = _cvt_f32_51.x;
                        float gate_y_1_51 = _cvt_f32_51.y;
                        float up_x_1_51 = _cvt_f32_256_51.x;
                        float up_y_1_51 = _cvt_f32_256_51.y;
                        float _exp_147 = expf(gate_x_1_51 * -1.0f);
                        float denominator_x_1_51 = _exp_147 + 1.0f;
                        float _exp_148 = expf(gate_y_1_51 * -1.0f);
                        float denominator_y_1_51 = _exp_148 + 1.0f;
                        float hidden_x_1_51 = gate_x_1_51 / denominator_x_1_51 * up_x_1_51;
                        float hidden_y_1_51 = gate_y_1_51 / denominator_y_1_51 * up_y_1_51;
                        hidden_packed_0[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_51, hidden_y_1_51)));
                        float2 _cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[20]));
                        float2 _cvt_f32_256_52 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[20]));
                        float gate_x_1_52 = _cvt_f32_52.x;
                        float gate_y_1_52 = _cvt_f32_52.y;
                        float up_x_1_52 = _cvt_f32_256_52.x;
                        float up_y_1_52 = _cvt_f32_256_52.y;
                        float _exp_149 = expf(gate_x_1_52 * -1.0f);
                        float denominator_x_1_52 = _exp_149 + 1.0f;
                        float _exp_150 = expf(gate_y_1_52 * -1.0f);
                        float denominator_y_1_52 = _exp_150 + 1.0f;
                        float hidden_x_1_52 = gate_x_1_52 / denominator_x_1_52 * up_x_1_52;
                        float hidden_y_1_52 = gate_y_1_52 / denominator_y_1_52 * up_y_1_52;
                        hidden_packed_0[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_52, hidden_y_1_52)));
                        float2 _cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[21]));
                        float2 _cvt_f32_256_53 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[21]));
                        float gate_x_1_53 = _cvt_f32_53.x;
                        float gate_y_1_53 = _cvt_f32_53.y;
                        float up_x_1_53 = _cvt_f32_256_53.x;
                        float up_y_1_53 = _cvt_f32_256_53.y;
                        float _exp_151 = expf(gate_x_1_53 * -1.0f);
                        float denominator_x_1_53 = _exp_151 + 1.0f;
                        float _exp_152 = expf(gate_y_1_53 * -1.0f);
                        float denominator_y_1_53 = _exp_152 + 1.0f;
                        float hidden_x_1_53 = gate_x_1_53 / denominator_x_1_53 * up_x_1_53;
                        float hidden_y_1_53 = gate_y_1_53 / denominator_y_1_53 * up_y_1_53;
                        hidden_packed_0[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_53, hidden_y_1_53)));
                        float2 _cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[22]));
                        float2 _cvt_f32_256_54 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[22]));
                        float gate_x_1_54 = _cvt_f32_54.x;
                        float gate_y_1_54 = _cvt_f32_54.y;
                        float up_x_1_54 = _cvt_f32_256_54.x;
                        float up_y_1_54 = _cvt_f32_256_54.y;
                        float _exp_153 = expf(gate_x_1_54 * -1.0f);
                        float denominator_x_1_54 = _exp_153 + 1.0f;
                        float _exp_154 = expf(gate_y_1_54 * -1.0f);
                        float denominator_y_1_54 = _exp_154 + 1.0f;
                        float hidden_x_1_54 = gate_x_1_54 / denominator_x_1_54 * up_x_1_54;
                        float hidden_y_1_54 = gate_y_1_54 / denominator_y_1_54 * up_y_1_54;
                        hidden_packed_0[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_54, hidden_y_1_54)));
                        float2 _cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[23]));
                        float2 _cvt_f32_256_55 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[23]));
                        float gate_x_1_55 = _cvt_f32_55.x;
                        float gate_y_1_55 = _cvt_f32_55.y;
                        float up_x_1_55 = _cvt_f32_256_55.x;
                        float up_y_1_55 = _cvt_f32_256_55.y;
                        float _exp_155 = expf(gate_x_1_55 * -1.0f);
                        float denominator_x_1_55 = _exp_155 + 1.0f;
                        float _exp_156 = expf(gate_y_1_55 * -1.0f);
                        float denominator_y_1_55 = _exp_156 + 1.0f;
                        float hidden_x_1_55 = gate_x_1_55 / denominator_x_1_55 * up_x_1_55;
                        float hidden_y_1_55 = gate_y_1_55 / denominator_y_1_55 * up_y_1_55;
                        hidden_packed_0[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_55, hidden_y_1_55)));
                        float2 _cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[24]));
                        float2 _cvt_f32_256_56 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[24]));
                        float gate_x_1_56 = _cvt_f32_56.x;
                        float gate_y_1_56 = _cvt_f32_56.y;
                        float up_x_1_56 = _cvt_f32_256_56.x;
                        float up_y_1_56 = _cvt_f32_256_56.y;
                        float _exp_157 = expf(gate_x_1_56 * -1.0f);
                        float denominator_x_1_56 = _exp_157 + 1.0f;
                        float _exp_158 = expf(gate_y_1_56 * -1.0f);
                        float denominator_y_1_56 = _exp_158 + 1.0f;
                        float hidden_x_1_56 = gate_x_1_56 / denominator_x_1_56 * up_x_1_56;
                        float hidden_y_1_56 = gate_y_1_56 / denominator_y_1_56 * up_y_1_56;
                        hidden_packed_0[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_56, hidden_y_1_56)));
                        float2 _cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[25]));
                        float2 _cvt_f32_256_57 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[25]));
                        float gate_x_1_57 = _cvt_f32_57.x;
                        float gate_y_1_57 = _cvt_f32_57.y;
                        float up_x_1_57 = _cvt_f32_256_57.x;
                        float up_y_1_57 = _cvt_f32_256_57.y;
                        float _exp_159 = expf(gate_x_1_57 * -1.0f);
                        float denominator_x_1_57 = _exp_159 + 1.0f;
                        float _exp_160 = expf(gate_y_1_57 * -1.0f);
                        float denominator_y_1_57 = _exp_160 + 1.0f;
                        float hidden_x_1_57 = gate_x_1_57 / denominator_x_1_57 * up_x_1_57;
                        float hidden_y_1_57 = gate_y_1_57 / denominator_y_1_57 * up_y_1_57;
                        hidden_packed_0[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_57, hidden_y_1_57)));
                        float2 _cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[26]));
                        float2 _cvt_f32_256_58 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[26]));
                        float gate_x_1_58 = _cvt_f32_58.x;
                        float gate_y_1_58 = _cvt_f32_58.y;
                        float up_x_1_58 = _cvt_f32_256_58.x;
                        float up_y_1_58 = _cvt_f32_256_58.y;
                        float _exp_161 = expf(gate_x_1_58 * -1.0f);
                        float denominator_x_1_58 = _exp_161 + 1.0f;
                        float _exp_162 = expf(gate_y_1_58 * -1.0f);
                        float denominator_y_1_58 = _exp_162 + 1.0f;
                        float hidden_x_1_58 = gate_x_1_58 / denominator_x_1_58 * up_x_1_58;
                        float hidden_y_1_58 = gate_y_1_58 / denominator_y_1_58 * up_y_1_58;
                        hidden_packed_0[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_58, hidden_y_1_58)));
                        float2 _cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[27]));
                        float2 _cvt_f32_256_59 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[27]));
                        float gate_x_1_59 = _cvt_f32_59.x;
                        float gate_y_1_59 = _cvt_f32_59.y;
                        float up_x_1_59 = _cvt_f32_256_59.x;
                        float up_y_1_59 = _cvt_f32_256_59.y;
                        float _exp_163 = expf(gate_x_1_59 * -1.0f);
                        float denominator_x_1_59 = _exp_163 + 1.0f;
                        float _exp_164 = expf(gate_y_1_59 * -1.0f);
                        float denominator_y_1_59 = _exp_164 + 1.0f;
                        float hidden_x_1_59 = gate_x_1_59 / denominator_x_1_59 * up_x_1_59;
                        float hidden_y_1_59 = gate_y_1_59 / denominator_y_1_59 * up_y_1_59;
                        hidden_packed_0[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_59, hidden_y_1_59)));
                        float2 _cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[28]));
                        float2 _cvt_f32_256_60 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[28]));
                        float gate_x_1_60 = _cvt_f32_60.x;
                        float gate_y_1_60 = _cvt_f32_60.y;
                        float up_x_1_60 = _cvt_f32_256_60.x;
                        float up_y_1_60 = _cvt_f32_256_60.y;
                        float _exp_165 = expf(gate_x_1_60 * -1.0f);
                        float denominator_x_1_60 = _exp_165 + 1.0f;
                        float _exp_166 = expf(gate_y_1_60 * -1.0f);
                        float denominator_y_1_60 = _exp_166 + 1.0f;
                        float hidden_x_1_60 = gate_x_1_60 / denominator_x_1_60 * up_x_1_60;
                        float hidden_y_1_60 = gate_y_1_60 / denominator_y_1_60 * up_y_1_60;
                        hidden_packed_0[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_60, hidden_y_1_60)));
                        float2 _cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[29]));
                        float2 _cvt_f32_256_61 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[29]));
                        float gate_x_1_61 = _cvt_f32_61.x;
                        float gate_y_1_61 = _cvt_f32_61.y;
                        float up_x_1_61 = _cvt_f32_256_61.x;
                        float up_y_1_61 = _cvt_f32_256_61.y;
                        float _exp_167 = expf(gate_x_1_61 * -1.0f);
                        float denominator_x_1_61 = _exp_167 + 1.0f;
                        float _exp_168 = expf(gate_y_1_61 * -1.0f);
                        float denominator_y_1_61 = _exp_168 + 1.0f;
                        float hidden_x_1_61 = gate_x_1_61 / denominator_x_1_61 * up_x_1_61;
                        float hidden_y_1_61 = gate_y_1_61 / denominator_y_1_61 * up_y_1_61;
                        hidden_packed_0[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_61, hidden_y_1_61)));
                        float2 _cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[30]));
                        float2 _cvt_f32_256_62 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[30]));
                        float gate_x_1_62 = _cvt_f32_62.x;
                        float gate_y_1_62 = _cvt_f32_62.y;
                        float up_x_1_62 = _cvt_f32_256_62.x;
                        float up_y_1_62 = _cvt_f32_256_62.y;
                        float _exp_169 = expf(gate_x_1_62 * -1.0f);
                        float denominator_x_1_62 = _exp_169 + 1.0f;
                        float _exp_170 = expf(gate_y_1_62 * -1.0f);
                        float denominator_y_1_62 = _exp_170 + 1.0f;
                        float hidden_x_1_62 = gate_x_1_62 / denominator_x_1_62 * up_x_1_62;
                        float hidden_y_1_62 = gate_y_1_62 / denominator_y_1_62 * up_y_1_62;
                        hidden_packed_0[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_62, hidden_y_1_62)));
                        float2 _cvt_f32_63 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_1[31]));
                        float2 _cvt_f32_256_63 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_1[31]));
                        float gate_x_1_63 = _cvt_f32_63.x;
                        float gate_y_1_63 = _cvt_f32_63.y;
                        float up_x_1_63 = _cvt_f32_256_63.x;
                        float up_y_1_63 = _cvt_f32_256_63.y;
                        float _exp_171 = expf(gate_x_1_63 * -1.0f);
                        float denominator_x_1_63 = _exp_171 + 1.0f;
                        float _exp_172 = expf(gate_y_1_63 * -1.0f);
                        float denominator_y_1_63 = _exp_172 + 1.0f;
                        float hidden_x_1_63 = gate_x_1_63 / denominator_x_1_63 * up_x_1_63;
                        float hidden_y_1_63 = gate_y_1_63 / denominator_y_1_63 * up_y_1_63;
                        hidden_packed_0[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_63, hidden_y_1_63)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_1_1 = tid / 32;
                        int lane_1_1 = tid % 32;
                        #pragma unroll
                        for (int half_2 = 0; half_2 < 2; half_2++) {
                            #pragma unroll
                            for (int col_tile_2 = 0; col_tile_2 < 2; col_tile_2++) {
                                int row_6 = warp_1_1 * 32 + half_2 * 16 + lane_1_1 % 16;
                                int col_3 = col_tile_2 * 16 + lane_1_1 / 16 * 8;
                                unsigned int address_2 = d_smem_addr + 16384 + (unsigned int)((row_6 * 32 + col_3) * 2);
                                address_2 = address_2 ^ (address_2 & 511) >> 7 << 4;
                                int offset_2 = half_2 * 8 + col_tile_2 * 4;
                                uint32_t _stmatrix_addr_173 = static_cast<uint32_t>(address_2);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_173), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_2 + 3]))
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
                        int warp_2_1 = tid / 32;
                        int lane_2 = tid % 32;
                        #pragma unroll
                        for (int half_3 = 0; half_3 < 2; half_3++) {
                            #pragma unroll
                            for (int col_tile_3 = 0; col_tile_3 < 2; col_tile_3++) {
                                int row_7 = warp_2_1 * 32 + half_3 * 16 + lane_2 % 16;
                                int col_4 = col_tile_3 * 16 + lane_2 / 16 * 8;
                                unsigned int address_3 = d_smem_addr + (unsigned int)((row_7 * 32 + col_4) * 2);
                                address_3 = address_3 ^ (address_3 & 511) >> 7 << 4;
                                int offset_3 = 16 + half_3 * 8 + col_tile_3 * 4;
                                uint32_t _stmatrix_addr_174 = static_cast<uint32_t>(address_3);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_174), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_3])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_3 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_3 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_0[offset_3 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        int _parity_output_arrived_1 = (int)((unsigned int)gemm_out1 % 2);
                        mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_1);
                        unsigned int gate_packed_2[32];
                        unsigned int up_packed_2[32];
                        unsigned int address_3_8 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 256;
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
                            : "r"(address_3_8 + 128));
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
                        unsigned int address_3_9 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 256;
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
                            : "r"(address_3_9 + 128));
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
                        unsigned int address_3_10 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 288;
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
                            : "r"(address_3_10 + 128));
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
                        unsigned int address_3_11 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 288;
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
                            : "r"(address_3_11 + 128));
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
                        unsigned int gate_packed_3[32];
                        unsigned int up_packed_3[32];
                        unsigned int address_3_12 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 320;
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
                            : "r"(address_3_12 + 128));
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
                        unsigned int address_3_13 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 320;
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
                            : "r"(address_3_13 + 128));
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
                        unsigned int address_3_14 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 352;
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
                            : "r"(address_3_14 + 128));
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
                        unsigned int address_3_15 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 352;
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
                            : "r"(address_3_15 + 128));
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
                                :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        unsigned int hidden_packed_1[32];
                        float2 _cvt_f32_64 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[0]));
                        float2 _cvt_f32_256_64 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[0]));
                        float gate_x_1_64 = _cvt_f32_64.x;
                        float gate_y_1_64 = _cvt_f32_64.y;
                        float up_x_1_64 = _cvt_f32_256_64.x;
                        float up_y_1_64 = _cvt_f32_256_64.y;
                        float _exp_175 = expf(gate_x_1_64 * -1.0f);
                        float denominator_x_1_64 = _exp_175 + 1.0f;
                        float _exp_176 = expf(gate_y_1_64 * -1.0f);
                        float denominator_y_1_64 = _exp_176 + 1.0f;
                        float hidden_x_1_64 = gate_x_1_64 / denominator_x_1_64 * up_x_1_64;
                        float hidden_y_1_64 = gate_y_1_64 / denominator_y_1_64 * up_y_1_64;
                        hidden_packed_1[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_64, hidden_y_1_64)));
                        float2 _cvt_f32_65 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[1]));
                        float2 _cvt_f32_256_65 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[1]));
                        float gate_x_1_65 = _cvt_f32_65.x;
                        float gate_y_1_65 = _cvt_f32_65.y;
                        float up_x_1_65 = _cvt_f32_256_65.x;
                        float up_y_1_65 = _cvt_f32_256_65.y;
                        float _exp_177 = expf(gate_x_1_65 * -1.0f);
                        float denominator_x_1_65 = _exp_177 + 1.0f;
                        float _exp_178 = expf(gate_y_1_65 * -1.0f);
                        float denominator_y_1_65 = _exp_178 + 1.0f;
                        float hidden_x_1_65 = gate_x_1_65 / denominator_x_1_65 * up_x_1_65;
                        float hidden_y_1_65 = gate_y_1_65 / denominator_y_1_65 * up_y_1_65;
                        hidden_packed_1[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_65, hidden_y_1_65)));
                        float2 _cvt_f32_66 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[2]));
                        float2 _cvt_f32_256_66 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[2]));
                        float gate_x_1_66 = _cvt_f32_66.x;
                        float gate_y_1_66 = _cvt_f32_66.y;
                        float up_x_1_66 = _cvt_f32_256_66.x;
                        float up_y_1_66 = _cvt_f32_256_66.y;
                        float _exp_179 = expf(gate_x_1_66 * -1.0f);
                        float denominator_x_1_66 = _exp_179 + 1.0f;
                        float _exp_180 = expf(gate_y_1_66 * -1.0f);
                        float denominator_y_1_66 = _exp_180 + 1.0f;
                        float hidden_x_1_66 = gate_x_1_66 / denominator_x_1_66 * up_x_1_66;
                        float hidden_y_1_66 = gate_y_1_66 / denominator_y_1_66 * up_y_1_66;
                        hidden_packed_1[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_66, hidden_y_1_66)));
                        float2 _cvt_f32_67 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[3]));
                        float2 _cvt_f32_256_67 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[3]));
                        float gate_x_1_67 = _cvt_f32_67.x;
                        float gate_y_1_67 = _cvt_f32_67.y;
                        float up_x_1_67 = _cvt_f32_256_67.x;
                        float up_y_1_67 = _cvt_f32_256_67.y;
                        float _exp_181 = expf(gate_x_1_67 * -1.0f);
                        float denominator_x_1_67 = _exp_181 + 1.0f;
                        float _exp_182 = expf(gate_y_1_67 * -1.0f);
                        float denominator_y_1_67 = _exp_182 + 1.0f;
                        float hidden_x_1_67 = gate_x_1_67 / denominator_x_1_67 * up_x_1_67;
                        float hidden_y_1_67 = gate_y_1_67 / denominator_y_1_67 * up_y_1_67;
                        hidden_packed_1[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_67, hidden_y_1_67)));
                        float2 _cvt_f32_68 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[4]));
                        float2 _cvt_f32_256_68 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[4]));
                        float gate_x_1_68 = _cvt_f32_68.x;
                        float gate_y_1_68 = _cvt_f32_68.y;
                        float up_x_1_68 = _cvt_f32_256_68.x;
                        float up_y_1_68 = _cvt_f32_256_68.y;
                        float _exp_183 = expf(gate_x_1_68 * -1.0f);
                        float denominator_x_1_68 = _exp_183 + 1.0f;
                        float _exp_184 = expf(gate_y_1_68 * -1.0f);
                        float denominator_y_1_68 = _exp_184 + 1.0f;
                        float hidden_x_1_68 = gate_x_1_68 / denominator_x_1_68 * up_x_1_68;
                        float hidden_y_1_68 = gate_y_1_68 / denominator_y_1_68 * up_y_1_68;
                        hidden_packed_1[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_68, hidden_y_1_68)));
                        float2 _cvt_f32_69 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[5]));
                        float2 _cvt_f32_256_69 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[5]));
                        float gate_x_1_69 = _cvt_f32_69.x;
                        float gate_y_1_69 = _cvt_f32_69.y;
                        float up_x_1_69 = _cvt_f32_256_69.x;
                        float up_y_1_69 = _cvt_f32_256_69.y;
                        float _exp_185 = expf(gate_x_1_69 * -1.0f);
                        float denominator_x_1_69 = _exp_185 + 1.0f;
                        float _exp_186 = expf(gate_y_1_69 * -1.0f);
                        float denominator_y_1_69 = _exp_186 + 1.0f;
                        float hidden_x_1_69 = gate_x_1_69 / denominator_x_1_69 * up_x_1_69;
                        float hidden_y_1_69 = gate_y_1_69 / denominator_y_1_69 * up_y_1_69;
                        hidden_packed_1[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_69, hidden_y_1_69)));
                        float2 _cvt_f32_70 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[6]));
                        float2 _cvt_f32_256_70 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[6]));
                        float gate_x_1_70 = _cvt_f32_70.x;
                        float gate_y_1_70 = _cvt_f32_70.y;
                        float up_x_1_70 = _cvt_f32_256_70.x;
                        float up_y_1_70 = _cvt_f32_256_70.y;
                        float _exp_187 = expf(gate_x_1_70 * -1.0f);
                        float denominator_x_1_70 = _exp_187 + 1.0f;
                        float _exp_188 = expf(gate_y_1_70 * -1.0f);
                        float denominator_y_1_70 = _exp_188 + 1.0f;
                        float hidden_x_1_70 = gate_x_1_70 / denominator_x_1_70 * up_x_1_70;
                        float hidden_y_1_70 = gate_y_1_70 / denominator_y_1_70 * up_y_1_70;
                        hidden_packed_1[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_70, hidden_y_1_70)));
                        float2 _cvt_f32_71 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[7]));
                        float2 _cvt_f32_256_71 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[7]));
                        float gate_x_1_71 = _cvt_f32_71.x;
                        float gate_y_1_71 = _cvt_f32_71.y;
                        float up_x_1_71 = _cvt_f32_256_71.x;
                        float up_y_1_71 = _cvt_f32_256_71.y;
                        float _exp_189 = expf(gate_x_1_71 * -1.0f);
                        float denominator_x_1_71 = _exp_189 + 1.0f;
                        float _exp_190 = expf(gate_y_1_71 * -1.0f);
                        float denominator_y_1_71 = _exp_190 + 1.0f;
                        float hidden_x_1_71 = gate_x_1_71 / denominator_x_1_71 * up_x_1_71;
                        float hidden_y_1_71 = gate_y_1_71 / denominator_y_1_71 * up_y_1_71;
                        hidden_packed_1[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_71, hidden_y_1_71)));
                        float2 _cvt_f32_72 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[8]));
                        float2 _cvt_f32_256_72 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[8]));
                        float gate_x_1_72 = _cvt_f32_72.x;
                        float gate_y_1_72 = _cvt_f32_72.y;
                        float up_x_1_72 = _cvt_f32_256_72.x;
                        float up_y_1_72 = _cvt_f32_256_72.y;
                        float _exp_191 = expf(gate_x_1_72 * -1.0f);
                        float denominator_x_1_72 = _exp_191 + 1.0f;
                        float _exp_192 = expf(gate_y_1_72 * -1.0f);
                        float denominator_y_1_72 = _exp_192 + 1.0f;
                        float hidden_x_1_72 = gate_x_1_72 / denominator_x_1_72 * up_x_1_72;
                        float hidden_y_1_72 = gate_y_1_72 / denominator_y_1_72 * up_y_1_72;
                        hidden_packed_1[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_72, hidden_y_1_72)));
                        float2 _cvt_f32_73 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[9]));
                        float2 _cvt_f32_256_73 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[9]));
                        float gate_x_1_73 = _cvt_f32_73.x;
                        float gate_y_1_73 = _cvt_f32_73.y;
                        float up_x_1_73 = _cvt_f32_256_73.x;
                        float up_y_1_73 = _cvt_f32_256_73.y;
                        float _exp_193 = expf(gate_x_1_73 * -1.0f);
                        float denominator_x_1_73 = _exp_193 + 1.0f;
                        float _exp_194 = expf(gate_y_1_73 * -1.0f);
                        float denominator_y_1_73 = _exp_194 + 1.0f;
                        float hidden_x_1_73 = gate_x_1_73 / denominator_x_1_73 * up_x_1_73;
                        float hidden_y_1_73 = gate_y_1_73 / denominator_y_1_73 * up_y_1_73;
                        hidden_packed_1[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_73, hidden_y_1_73)));
                        float2 _cvt_f32_74 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[10]));
                        float2 _cvt_f32_256_74 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[10]));
                        float gate_x_1_74 = _cvt_f32_74.x;
                        float gate_y_1_74 = _cvt_f32_74.y;
                        float up_x_1_74 = _cvt_f32_256_74.x;
                        float up_y_1_74 = _cvt_f32_256_74.y;
                        float _exp_195 = expf(gate_x_1_74 * -1.0f);
                        float denominator_x_1_74 = _exp_195 + 1.0f;
                        float _exp_196 = expf(gate_y_1_74 * -1.0f);
                        float denominator_y_1_74 = _exp_196 + 1.0f;
                        float hidden_x_1_74 = gate_x_1_74 / denominator_x_1_74 * up_x_1_74;
                        float hidden_y_1_74 = gate_y_1_74 / denominator_y_1_74 * up_y_1_74;
                        hidden_packed_1[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_74, hidden_y_1_74)));
                        float2 _cvt_f32_75 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[11]));
                        float2 _cvt_f32_256_75 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[11]));
                        float gate_x_1_75 = _cvt_f32_75.x;
                        float gate_y_1_75 = _cvt_f32_75.y;
                        float up_x_1_75 = _cvt_f32_256_75.x;
                        float up_y_1_75 = _cvt_f32_256_75.y;
                        float _exp_197 = expf(gate_x_1_75 * -1.0f);
                        float denominator_x_1_75 = _exp_197 + 1.0f;
                        float _exp_198 = expf(gate_y_1_75 * -1.0f);
                        float denominator_y_1_75 = _exp_198 + 1.0f;
                        float hidden_x_1_75 = gate_x_1_75 / denominator_x_1_75 * up_x_1_75;
                        float hidden_y_1_75 = gate_y_1_75 / denominator_y_1_75 * up_y_1_75;
                        hidden_packed_1[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_75, hidden_y_1_75)));
                        float2 _cvt_f32_76 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[12]));
                        float2 _cvt_f32_256_76 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[12]));
                        float gate_x_1_76 = _cvt_f32_76.x;
                        float gate_y_1_76 = _cvt_f32_76.y;
                        float up_x_1_76 = _cvt_f32_256_76.x;
                        float up_y_1_76 = _cvt_f32_256_76.y;
                        float _exp_199 = expf(gate_x_1_76 * -1.0f);
                        float denominator_x_1_76 = _exp_199 + 1.0f;
                        float _exp_200 = expf(gate_y_1_76 * -1.0f);
                        float denominator_y_1_76 = _exp_200 + 1.0f;
                        float hidden_x_1_76 = gate_x_1_76 / denominator_x_1_76 * up_x_1_76;
                        float hidden_y_1_76 = gate_y_1_76 / denominator_y_1_76 * up_y_1_76;
                        hidden_packed_1[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_76, hidden_y_1_76)));
                        float2 _cvt_f32_77 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[13]));
                        float2 _cvt_f32_256_77 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[13]));
                        float gate_x_1_77 = _cvt_f32_77.x;
                        float gate_y_1_77 = _cvt_f32_77.y;
                        float up_x_1_77 = _cvt_f32_256_77.x;
                        float up_y_1_77 = _cvt_f32_256_77.y;
                        float _exp_201 = expf(gate_x_1_77 * -1.0f);
                        float denominator_x_1_77 = _exp_201 + 1.0f;
                        float _exp_202 = expf(gate_y_1_77 * -1.0f);
                        float denominator_y_1_77 = _exp_202 + 1.0f;
                        float hidden_x_1_77 = gate_x_1_77 / denominator_x_1_77 * up_x_1_77;
                        float hidden_y_1_77 = gate_y_1_77 / denominator_y_1_77 * up_y_1_77;
                        hidden_packed_1[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_77, hidden_y_1_77)));
                        float2 _cvt_f32_78 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[14]));
                        float2 _cvt_f32_256_78 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[14]));
                        float gate_x_1_78 = _cvt_f32_78.x;
                        float gate_y_1_78 = _cvt_f32_78.y;
                        float up_x_1_78 = _cvt_f32_256_78.x;
                        float up_y_1_78 = _cvt_f32_256_78.y;
                        float _exp_203 = expf(gate_x_1_78 * -1.0f);
                        float denominator_x_1_78 = _exp_203 + 1.0f;
                        float _exp_204 = expf(gate_y_1_78 * -1.0f);
                        float denominator_y_1_78 = _exp_204 + 1.0f;
                        float hidden_x_1_78 = gate_x_1_78 / denominator_x_1_78 * up_x_1_78;
                        float hidden_y_1_78 = gate_y_1_78 / denominator_y_1_78 * up_y_1_78;
                        hidden_packed_1[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_78, hidden_y_1_78)));
                        float2 _cvt_f32_79 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[15]));
                        float2 _cvt_f32_256_79 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[15]));
                        float gate_x_1_79 = _cvt_f32_79.x;
                        float gate_y_1_79 = _cvt_f32_79.y;
                        float up_x_1_79 = _cvt_f32_256_79.x;
                        float up_y_1_79 = _cvt_f32_256_79.y;
                        float _exp_205 = expf(gate_x_1_79 * -1.0f);
                        float denominator_x_1_79 = _exp_205 + 1.0f;
                        float _exp_206 = expf(gate_y_1_79 * -1.0f);
                        float denominator_y_1_79 = _exp_206 + 1.0f;
                        float hidden_x_1_79 = gate_x_1_79 / denominator_x_1_79 * up_x_1_79;
                        float hidden_y_1_79 = gate_y_1_79 / denominator_y_1_79 * up_y_1_79;
                        hidden_packed_1[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_79, hidden_y_1_79)));
                        float2 _cvt_f32_80 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[16]));
                        float2 _cvt_f32_256_80 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[16]));
                        float gate_x_1_80 = _cvt_f32_80.x;
                        float gate_y_1_80 = _cvt_f32_80.y;
                        float up_x_1_80 = _cvt_f32_256_80.x;
                        float up_y_1_80 = _cvt_f32_256_80.y;
                        float _exp_207 = expf(gate_x_1_80 * -1.0f);
                        float denominator_x_1_80 = _exp_207 + 1.0f;
                        float _exp_208 = expf(gate_y_1_80 * -1.0f);
                        float denominator_y_1_80 = _exp_208 + 1.0f;
                        float hidden_x_1_80 = gate_x_1_80 / denominator_x_1_80 * up_x_1_80;
                        float hidden_y_1_80 = gate_y_1_80 / denominator_y_1_80 * up_y_1_80;
                        hidden_packed_1[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_80, hidden_y_1_80)));
                        float2 _cvt_f32_81 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[17]));
                        float2 _cvt_f32_256_81 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[17]));
                        float gate_x_1_81 = _cvt_f32_81.x;
                        float gate_y_1_81 = _cvt_f32_81.y;
                        float up_x_1_81 = _cvt_f32_256_81.x;
                        float up_y_1_81 = _cvt_f32_256_81.y;
                        float _exp_209 = expf(gate_x_1_81 * -1.0f);
                        float denominator_x_1_81 = _exp_209 + 1.0f;
                        float _exp_210 = expf(gate_y_1_81 * -1.0f);
                        float denominator_y_1_81 = _exp_210 + 1.0f;
                        float hidden_x_1_81 = gate_x_1_81 / denominator_x_1_81 * up_x_1_81;
                        float hidden_y_1_81 = gate_y_1_81 / denominator_y_1_81 * up_y_1_81;
                        hidden_packed_1[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_81, hidden_y_1_81)));
                        float2 _cvt_f32_82 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[18]));
                        float2 _cvt_f32_256_82 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[18]));
                        float gate_x_1_82 = _cvt_f32_82.x;
                        float gate_y_1_82 = _cvt_f32_82.y;
                        float up_x_1_82 = _cvt_f32_256_82.x;
                        float up_y_1_82 = _cvt_f32_256_82.y;
                        float _exp_211 = expf(gate_x_1_82 * -1.0f);
                        float denominator_x_1_82 = _exp_211 + 1.0f;
                        float _exp_212 = expf(gate_y_1_82 * -1.0f);
                        float denominator_y_1_82 = _exp_212 + 1.0f;
                        float hidden_x_1_82 = gate_x_1_82 / denominator_x_1_82 * up_x_1_82;
                        float hidden_y_1_82 = gate_y_1_82 / denominator_y_1_82 * up_y_1_82;
                        hidden_packed_1[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_82, hidden_y_1_82)));
                        float2 _cvt_f32_83 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[19]));
                        float2 _cvt_f32_256_83 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[19]));
                        float gate_x_1_83 = _cvt_f32_83.x;
                        float gate_y_1_83 = _cvt_f32_83.y;
                        float up_x_1_83 = _cvt_f32_256_83.x;
                        float up_y_1_83 = _cvt_f32_256_83.y;
                        float _exp_213 = expf(gate_x_1_83 * -1.0f);
                        float denominator_x_1_83 = _exp_213 + 1.0f;
                        float _exp_214 = expf(gate_y_1_83 * -1.0f);
                        float denominator_y_1_83 = _exp_214 + 1.0f;
                        float hidden_x_1_83 = gate_x_1_83 / denominator_x_1_83 * up_x_1_83;
                        float hidden_y_1_83 = gate_y_1_83 / denominator_y_1_83 * up_y_1_83;
                        hidden_packed_1[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_83, hidden_y_1_83)));
                        float2 _cvt_f32_84 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[20]));
                        float2 _cvt_f32_256_84 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[20]));
                        float gate_x_1_84 = _cvt_f32_84.x;
                        float gate_y_1_84 = _cvt_f32_84.y;
                        float up_x_1_84 = _cvt_f32_256_84.x;
                        float up_y_1_84 = _cvt_f32_256_84.y;
                        float _exp_215 = expf(gate_x_1_84 * -1.0f);
                        float denominator_x_1_84 = _exp_215 + 1.0f;
                        float _exp_216 = expf(gate_y_1_84 * -1.0f);
                        float denominator_y_1_84 = _exp_216 + 1.0f;
                        float hidden_x_1_84 = gate_x_1_84 / denominator_x_1_84 * up_x_1_84;
                        float hidden_y_1_84 = gate_y_1_84 / denominator_y_1_84 * up_y_1_84;
                        hidden_packed_1[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_84, hidden_y_1_84)));
                        float2 _cvt_f32_85 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[21]));
                        float2 _cvt_f32_256_85 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[21]));
                        float gate_x_1_85 = _cvt_f32_85.x;
                        float gate_y_1_85 = _cvt_f32_85.y;
                        float up_x_1_85 = _cvt_f32_256_85.x;
                        float up_y_1_85 = _cvt_f32_256_85.y;
                        float _exp_217 = expf(gate_x_1_85 * -1.0f);
                        float denominator_x_1_85 = _exp_217 + 1.0f;
                        float _exp_218 = expf(gate_y_1_85 * -1.0f);
                        float denominator_y_1_85 = _exp_218 + 1.0f;
                        float hidden_x_1_85 = gate_x_1_85 / denominator_x_1_85 * up_x_1_85;
                        float hidden_y_1_85 = gate_y_1_85 / denominator_y_1_85 * up_y_1_85;
                        hidden_packed_1[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_85, hidden_y_1_85)));
                        float2 _cvt_f32_86 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[22]));
                        float2 _cvt_f32_256_86 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[22]));
                        float gate_x_1_86 = _cvt_f32_86.x;
                        float gate_y_1_86 = _cvt_f32_86.y;
                        float up_x_1_86 = _cvt_f32_256_86.x;
                        float up_y_1_86 = _cvt_f32_256_86.y;
                        float _exp_219 = expf(gate_x_1_86 * -1.0f);
                        float denominator_x_1_86 = _exp_219 + 1.0f;
                        float _exp_220 = expf(gate_y_1_86 * -1.0f);
                        float denominator_y_1_86 = _exp_220 + 1.0f;
                        float hidden_x_1_86 = gate_x_1_86 / denominator_x_1_86 * up_x_1_86;
                        float hidden_y_1_86 = gate_y_1_86 / denominator_y_1_86 * up_y_1_86;
                        hidden_packed_1[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_86, hidden_y_1_86)));
                        float2 _cvt_f32_87 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[23]));
                        float2 _cvt_f32_256_87 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[23]));
                        float gate_x_1_87 = _cvt_f32_87.x;
                        float gate_y_1_87 = _cvt_f32_87.y;
                        float up_x_1_87 = _cvt_f32_256_87.x;
                        float up_y_1_87 = _cvt_f32_256_87.y;
                        float _exp_221 = expf(gate_x_1_87 * -1.0f);
                        float denominator_x_1_87 = _exp_221 + 1.0f;
                        float _exp_222 = expf(gate_y_1_87 * -1.0f);
                        float denominator_y_1_87 = _exp_222 + 1.0f;
                        float hidden_x_1_87 = gate_x_1_87 / denominator_x_1_87 * up_x_1_87;
                        float hidden_y_1_87 = gate_y_1_87 / denominator_y_1_87 * up_y_1_87;
                        hidden_packed_1[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_87, hidden_y_1_87)));
                        float2 _cvt_f32_88 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[24]));
                        float2 _cvt_f32_256_88 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[24]));
                        float gate_x_1_88 = _cvt_f32_88.x;
                        float gate_y_1_88 = _cvt_f32_88.y;
                        float up_x_1_88 = _cvt_f32_256_88.x;
                        float up_y_1_88 = _cvt_f32_256_88.y;
                        float _exp_223 = expf(gate_x_1_88 * -1.0f);
                        float denominator_x_1_88 = _exp_223 + 1.0f;
                        float _exp_224 = expf(gate_y_1_88 * -1.0f);
                        float denominator_y_1_88 = _exp_224 + 1.0f;
                        float hidden_x_1_88 = gate_x_1_88 / denominator_x_1_88 * up_x_1_88;
                        float hidden_y_1_88 = gate_y_1_88 / denominator_y_1_88 * up_y_1_88;
                        hidden_packed_1[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_88, hidden_y_1_88)));
                        float2 _cvt_f32_89 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[25]));
                        float2 _cvt_f32_256_89 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[25]));
                        float gate_x_1_89 = _cvt_f32_89.x;
                        float gate_y_1_89 = _cvt_f32_89.y;
                        float up_x_1_89 = _cvt_f32_256_89.x;
                        float up_y_1_89 = _cvt_f32_256_89.y;
                        float _exp_225 = expf(gate_x_1_89 * -1.0f);
                        float denominator_x_1_89 = _exp_225 + 1.0f;
                        float _exp_226 = expf(gate_y_1_89 * -1.0f);
                        float denominator_y_1_89 = _exp_226 + 1.0f;
                        float hidden_x_1_89 = gate_x_1_89 / denominator_x_1_89 * up_x_1_89;
                        float hidden_y_1_89 = gate_y_1_89 / denominator_y_1_89 * up_y_1_89;
                        hidden_packed_1[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_89, hidden_y_1_89)));
                        float2 _cvt_f32_90 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[26]));
                        float2 _cvt_f32_256_90 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[26]));
                        float gate_x_1_90 = _cvt_f32_90.x;
                        float gate_y_1_90 = _cvt_f32_90.y;
                        float up_x_1_90 = _cvt_f32_256_90.x;
                        float up_y_1_90 = _cvt_f32_256_90.y;
                        float _exp_227 = expf(gate_x_1_90 * -1.0f);
                        float denominator_x_1_90 = _exp_227 + 1.0f;
                        float _exp_228 = expf(gate_y_1_90 * -1.0f);
                        float denominator_y_1_90 = _exp_228 + 1.0f;
                        float hidden_x_1_90 = gate_x_1_90 / denominator_x_1_90 * up_x_1_90;
                        float hidden_y_1_90 = gate_y_1_90 / denominator_y_1_90 * up_y_1_90;
                        hidden_packed_1[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_90, hidden_y_1_90)));
                        float2 _cvt_f32_91 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[27]));
                        float2 _cvt_f32_256_91 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[27]));
                        float gate_x_1_91 = _cvt_f32_91.x;
                        float gate_y_1_91 = _cvt_f32_91.y;
                        float up_x_1_91 = _cvt_f32_256_91.x;
                        float up_y_1_91 = _cvt_f32_256_91.y;
                        float _exp_229 = expf(gate_x_1_91 * -1.0f);
                        float denominator_x_1_91 = _exp_229 + 1.0f;
                        float _exp_230 = expf(gate_y_1_91 * -1.0f);
                        float denominator_y_1_91 = _exp_230 + 1.0f;
                        float hidden_x_1_91 = gate_x_1_91 / denominator_x_1_91 * up_x_1_91;
                        float hidden_y_1_91 = gate_y_1_91 / denominator_y_1_91 * up_y_1_91;
                        hidden_packed_1[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_91, hidden_y_1_91)));
                        float2 _cvt_f32_92 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[28]));
                        float2 _cvt_f32_256_92 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[28]));
                        float gate_x_1_92 = _cvt_f32_92.x;
                        float gate_y_1_92 = _cvt_f32_92.y;
                        float up_x_1_92 = _cvt_f32_256_92.x;
                        float up_y_1_92 = _cvt_f32_256_92.y;
                        float _exp_231 = expf(gate_x_1_92 * -1.0f);
                        float denominator_x_1_92 = _exp_231 + 1.0f;
                        float _exp_232 = expf(gate_y_1_92 * -1.0f);
                        float denominator_y_1_92 = _exp_232 + 1.0f;
                        float hidden_x_1_92 = gate_x_1_92 / denominator_x_1_92 * up_x_1_92;
                        float hidden_y_1_92 = gate_y_1_92 / denominator_y_1_92 * up_y_1_92;
                        hidden_packed_1[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_92, hidden_y_1_92)));
                        float2 _cvt_f32_93 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[29]));
                        float2 _cvt_f32_256_93 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[29]));
                        float gate_x_1_93 = _cvt_f32_93.x;
                        float gate_y_1_93 = _cvt_f32_93.y;
                        float up_x_1_93 = _cvt_f32_256_93.x;
                        float up_y_1_93 = _cvt_f32_256_93.y;
                        float _exp_233 = expf(gate_x_1_93 * -1.0f);
                        float denominator_x_1_93 = _exp_233 + 1.0f;
                        float _exp_234 = expf(gate_y_1_93 * -1.0f);
                        float denominator_y_1_93 = _exp_234 + 1.0f;
                        float hidden_x_1_93 = gate_x_1_93 / denominator_x_1_93 * up_x_1_93;
                        float hidden_y_1_93 = gate_y_1_93 / denominator_y_1_93 * up_y_1_93;
                        hidden_packed_1[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_93, hidden_y_1_93)));
                        float2 _cvt_f32_94 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[30]));
                        float2 _cvt_f32_256_94 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[30]));
                        float gate_x_1_94 = _cvt_f32_94.x;
                        float gate_y_1_94 = _cvt_f32_94.y;
                        float up_x_1_94 = _cvt_f32_256_94.x;
                        float up_y_1_94 = _cvt_f32_256_94.y;
                        float _exp_235 = expf(gate_x_1_94 * -1.0f);
                        float denominator_x_1_94 = _exp_235 + 1.0f;
                        float _exp_236 = expf(gate_y_1_94 * -1.0f);
                        float denominator_y_1_94 = _exp_236 + 1.0f;
                        float hidden_x_1_94 = gate_x_1_94 / denominator_x_1_94 * up_x_1_94;
                        float hidden_y_1_94 = gate_y_1_94 / denominator_y_1_94 * up_y_1_94;
                        hidden_packed_1[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_94, hidden_y_1_94)));
                        float2 _cvt_f32_95 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_2[31]));
                        float2 _cvt_f32_256_95 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_2[31]));
                        float gate_x_1_95 = _cvt_f32_95.x;
                        float gate_y_1_95 = _cvt_f32_95.y;
                        float up_x_1_95 = _cvt_f32_256_95.x;
                        float up_y_1_95 = _cvt_f32_256_95.y;
                        float _exp_237 = expf(gate_x_1_95 * -1.0f);
                        float denominator_x_1_95 = _exp_237 + 1.0f;
                        float _exp_238 = expf(gate_y_1_95 * -1.0f);
                        float denominator_y_1_95 = _exp_238 + 1.0f;
                        float hidden_x_1_95 = gate_x_1_95 / denominator_x_1_95 * up_x_1_95;
                        float hidden_y_1_95 = gate_y_1_95 / denominator_y_1_95 * up_y_1_95;
                        hidden_packed_1[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_95, hidden_y_1_95)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_3 = tid / 32;
                        int lane_3 = tid % 32;
                        #pragma unroll
                        for (int half_4 = 0; half_4 < 2; half_4++) {
                            #pragma unroll
                            for (int col_tile_4 = 0; col_tile_4 < 2; col_tile_4++) {
                                int row_8 = warp_3 * 32 + half_4 * 16 + lane_3 % 16;
                                int col_5 = col_tile_4 * 16 + lane_3 / 16 * 8;
                                unsigned int address_4 = d_smem_addr + 8192 + (unsigned int)((row_8 * 32 + col_5) * 2);
                                address_4 = address_4 ^ (address_4 & 511) >> 7 << 4;
                                int offset_4 = half_4 * 8 + col_tile_4 * 4;
                                uint32_t _stmatrix_addr_239 = static_cast<uint32_t>(address_4);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_239), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_4])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_4 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_4 = tid / 32;
                        int lane_4 = tid % 32;
                        #pragma unroll
                        for (int half_5 = 0; half_5 < 2; half_5++) {
                            #pragma unroll
                            for (int col_tile_5 = 0; col_tile_5 < 2; col_tile_5++) {
                                int row_9 = warp_4 * 32 + half_5 * 16 + lane_4 % 16;
                                int col_6 = col_tile_5 * 16 + lane_4 / 16 * 8;
                                unsigned int address_5 = d_smem_addr + 16384 + (unsigned int)((row_9 * 32 + col_6) * 2);
                                address_5 = address_5 ^ (address_5 & 511) >> 7 << 4;
                                int offset_5 = 16 + half_5 * 8 + col_tile_5 * 4;
                                uint32_t _stmatrix_addr_240 = static_cast<uint32_t>(address_5);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_240), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_5])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_5 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_5 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_1[offset_5 + 3]))
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
                        unsigned int hidden_packed_2[32];
                        float2 _cvt_f32_96 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[0]));
                        float2 _cvt_f32_256_96 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[0]));
                        float gate_x_1_96 = _cvt_f32_96.x;
                        float gate_y_1_96 = _cvt_f32_96.y;
                        float up_x_1_96 = _cvt_f32_256_96.x;
                        float up_y_1_96 = _cvt_f32_256_96.y;
                        float _exp_241 = expf(gate_x_1_96 * -1.0f);
                        float denominator_x_1_96 = _exp_241 + 1.0f;
                        float _exp_242 = expf(gate_y_1_96 * -1.0f);
                        float denominator_y_1_96 = _exp_242 + 1.0f;
                        float hidden_x_1_96 = gate_x_1_96 / denominator_x_1_96 * up_x_1_96;
                        float hidden_y_1_96 = gate_y_1_96 / denominator_y_1_96 * up_y_1_96;
                        hidden_packed_2[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_96, hidden_y_1_96)));
                        float2 _cvt_f32_97 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[1]));
                        float2 _cvt_f32_256_97 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[1]));
                        float gate_x_1_97 = _cvt_f32_97.x;
                        float gate_y_1_97 = _cvt_f32_97.y;
                        float up_x_1_97 = _cvt_f32_256_97.x;
                        float up_y_1_97 = _cvt_f32_256_97.y;
                        float _exp_243 = expf(gate_x_1_97 * -1.0f);
                        float denominator_x_1_97 = _exp_243 + 1.0f;
                        float _exp_244 = expf(gate_y_1_97 * -1.0f);
                        float denominator_y_1_97 = _exp_244 + 1.0f;
                        float hidden_x_1_97 = gate_x_1_97 / denominator_x_1_97 * up_x_1_97;
                        float hidden_y_1_97 = gate_y_1_97 / denominator_y_1_97 * up_y_1_97;
                        hidden_packed_2[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_97, hidden_y_1_97)));
                        float2 _cvt_f32_98 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[2]));
                        float2 _cvt_f32_256_98 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[2]));
                        float gate_x_1_98 = _cvt_f32_98.x;
                        float gate_y_1_98 = _cvt_f32_98.y;
                        float up_x_1_98 = _cvt_f32_256_98.x;
                        float up_y_1_98 = _cvt_f32_256_98.y;
                        float _exp_245 = expf(gate_x_1_98 * -1.0f);
                        float denominator_x_1_98 = _exp_245 + 1.0f;
                        float _exp_246 = expf(gate_y_1_98 * -1.0f);
                        float denominator_y_1_98 = _exp_246 + 1.0f;
                        float hidden_x_1_98 = gate_x_1_98 / denominator_x_1_98 * up_x_1_98;
                        float hidden_y_1_98 = gate_y_1_98 / denominator_y_1_98 * up_y_1_98;
                        hidden_packed_2[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_98, hidden_y_1_98)));
                        float2 _cvt_f32_99 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[3]));
                        float2 _cvt_f32_256_99 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[3]));
                        float gate_x_1_99 = _cvt_f32_99.x;
                        float gate_y_1_99 = _cvt_f32_99.y;
                        float up_x_1_99 = _cvt_f32_256_99.x;
                        float up_y_1_99 = _cvt_f32_256_99.y;
                        float _exp_247 = expf(gate_x_1_99 * -1.0f);
                        float denominator_x_1_99 = _exp_247 + 1.0f;
                        float _exp_248 = expf(gate_y_1_99 * -1.0f);
                        float denominator_y_1_99 = _exp_248 + 1.0f;
                        float hidden_x_1_99 = gate_x_1_99 / denominator_x_1_99 * up_x_1_99;
                        float hidden_y_1_99 = gate_y_1_99 / denominator_y_1_99 * up_y_1_99;
                        hidden_packed_2[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_99, hidden_y_1_99)));
                        float2 _cvt_f32_100 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[4]));
                        float2 _cvt_f32_256_100 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[4]));
                        float gate_x_1_100 = _cvt_f32_100.x;
                        float gate_y_1_100 = _cvt_f32_100.y;
                        float up_x_1_100 = _cvt_f32_256_100.x;
                        float up_y_1_100 = _cvt_f32_256_100.y;
                        float _exp_249 = expf(gate_x_1_100 * -1.0f);
                        float denominator_x_1_100 = _exp_249 + 1.0f;
                        float _exp_250 = expf(gate_y_1_100 * -1.0f);
                        float denominator_y_1_100 = _exp_250 + 1.0f;
                        float hidden_x_1_100 = gate_x_1_100 / denominator_x_1_100 * up_x_1_100;
                        float hidden_y_1_100 = gate_y_1_100 / denominator_y_1_100 * up_y_1_100;
                        hidden_packed_2[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_100, hidden_y_1_100)));
                        float2 _cvt_f32_101 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[5]));
                        float2 _cvt_f32_256_101 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[5]));
                        float gate_x_1_101 = _cvt_f32_101.x;
                        float gate_y_1_101 = _cvt_f32_101.y;
                        float up_x_1_101 = _cvt_f32_256_101.x;
                        float up_y_1_101 = _cvt_f32_256_101.y;
                        float _exp_251 = expf(gate_x_1_101 * -1.0f);
                        float denominator_x_1_101 = _exp_251 + 1.0f;
                        float _exp_252 = expf(gate_y_1_101 * -1.0f);
                        float denominator_y_1_101 = _exp_252 + 1.0f;
                        float hidden_x_1_101 = gate_x_1_101 / denominator_x_1_101 * up_x_1_101;
                        float hidden_y_1_101 = gate_y_1_101 / denominator_y_1_101 * up_y_1_101;
                        hidden_packed_2[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_101, hidden_y_1_101)));
                        float2 _cvt_f32_102 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[6]));
                        float2 _cvt_f32_256_102 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[6]));
                        float gate_x_1_102 = _cvt_f32_102.x;
                        float gate_y_1_102 = _cvt_f32_102.y;
                        float up_x_1_102 = _cvt_f32_256_102.x;
                        float up_y_1_102 = _cvt_f32_256_102.y;
                        float _exp_253 = expf(gate_x_1_102 * -1.0f);
                        float denominator_x_1_102 = _exp_253 + 1.0f;
                        float _exp_254 = expf(gate_y_1_102 * -1.0f);
                        float denominator_y_1_102 = _exp_254 + 1.0f;
                        float hidden_x_1_102 = gate_x_1_102 / denominator_x_1_102 * up_x_1_102;
                        float hidden_y_1_102 = gate_y_1_102 / denominator_y_1_102 * up_y_1_102;
                        hidden_packed_2[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_102, hidden_y_1_102)));
                        float2 _cvt_f32_103 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[7]));
                        float2 _cvt_f32_256_103 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[7]));
                        float gate_x_1_103 = _cvt_f32_103.x;
                        float gate_y_1_103 = _cvt_f32_103.y;
                        float up_x_1_103 = _cvt_f32_256_103.x;
                        float up_y_1_103 = _cvt_f32_256_103.y;
                        float _exp_255 = expf(gate_x_1_103 * -1.0f);
                        float denominator_x_1_103 = _exp_255 + 1.0f;
                        float _exp_256 = expf(gate_y_1_103 * -1.0f);
                        float denominator_y_1_103 = _exp_256 + 1.0f;
                        float hidden_x_1_103 = gate_x_1_103 / denominator_x_1_103 * up_x_1_103;
                        float hidden_y_1_103 = gate_y_1_103 / denominator_y_1_103 * up_y_1_103;
                        hidden_packed_2[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_103, hidden_y_1_103)));
                        float2 _cvt_f32_104 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[8]));
                        float2 _cvt_f32_256_104 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[8]));
                        float gate_x_1_104 = _cvt_f32_104.x;
                        float gate_y_1_104 = _cvt_f32_104.y;
                        float up_x_1_104 = _cvt_f32_256_104.x;
                        float up_y_1_104 = _cvt_f32_256_104.y;
                        float _exp_257 = expf(gate_x_1_104 * -1.0f);
                        float denominator_x_1_104 = _exp_257 + 1.0f;
                        float _exp_258 = expf(gate_y_1_104 * -1.0f);
                        float denominator_y_1_104 = _exp_258 + 1.0f;
                        float hidden_x_1_104 = gate_x_1_104 / denominator_x_1_104 * up_x_1_104;
                        float hidden_y_1_104 = gate_y_1_104 / denominator_y_1_104 * up_y_1_104;
                        hidden_packed_2[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_104, hidden_y_1_104)));
                        float2 _cvt_f32_105 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[9]));
                        float2 _cvt_f32_256_105 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[9]));
                        float gate_x_1_105 = _cvt_f32_105.x;
                        float gate_y_1_105 = _cvt_f32_105.y;
                        float up_x_1_105 = _cvt_f32_256_105.x;
                        float up_y_1_105 = _cvt_f32_256_105.y;
                        float _exp_259 = expf(gate_x_1_105 * -1.0f);
                        float denominator_x_1_105 = _exp_259 + 1.0f;
                        float _exp_260 = expf(gate_y_1_105 * -1.0f);
                        float denominator_y_1_105 = _exp_260 + 1.0f;
                        float hidden_x_1_105 = gate_x_1_105 / denominator_x_1_105 * up_x_1_105;
                        float hidden_y_1_105 = gate_y_1_105 / denominator_y_1_105 * up_y_1_105;
                        hidden_packed_2[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_105, hidden_y_1_105)));
                        float2 _cvt_f32_106 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[10]));
                        float2 _cvt_f32_256_106 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[10]));
                        float gate_x_1_106 = _cvt_f32_106.x;
                        float gate_y_1_106 = _cvt_f32_106.y;
                        float up_x_1_106 = _cvt_f32_256_106.x;
                        float up_y_1_106 = _cvt_f32_256_106.y;
                        float _exp_261 = expf(gate_x_1_106 * -1.0f);
                        float denominator_x_1_106 = _exp_261 + 1.0f;
                        float _exp_262 = expf(gate_y_1_106 * -1.0f);
                        float denominator_y_1_106 = _exp_262 + 1.0f;
                        float hidden_x_1_106 = gate_x_1_106 / denominator_x_1_106 * up_x_1_106;
                        float hidden_y_1_106 = gate_y_1_106 / denominator_y_1_106 * up_y_1_106;
                        hidden_packed_2[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_106, hidden_y_1_106)));
                        float2 _cvt_f32_107 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[11]));
                        float2 _cvt_f32_256_107 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[11]));
                        float gate_x_1_107 = _cvt_f32_107.x;
                        float gate_y_1_107 = _cvt_f32_107.y;
                        float up_x_1_107 = _cvt_f32_256_107.x;
                        float up_y_1_107 = _cvt_f32_256_107.y;
                        float _exp_263 = expf(gate_x_1_107 * -1.0f);
                        float denominator_x_1_107 = _exp_263 + 1.0f;
                        float _exp_264 = expf(gate_y_1_107 * -1.0f);
                        float denominator_y_1_107 = _exp_264 + 1.0f;
                        float hidden_x_1_107 = gate_x_1_107 / denominator_x_1_107 * up_x_1_107;
                        float hidden_y_1_107 = gate_y_1_107 / denominator_y_1_107 * up_y_1_107;
                        hidden_packed_2[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_107, hidden_y_1_107)));
                        float2 _cvt_f32_108 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[12]));
                        float2 _cvt_f32_256_108 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[12]));
                        float gate_x_1_108 = _cvt_f32_108.x;
                        float gate_y_1_108 = _cvt_f32_108.y;
                        float up_x_1_108 = _cvt_f32_256_108.x;
                        float up_y_1_108 = _cvt_f32_256_108.y;
                        float _exp_265 = expf(gate_x_1_108 * -1.0f);
                        float denominator_x_1_108 = _exp_265 + 1.0f;
                        float _exp_266 = expf(gate_y_1_108 * -1.0f);
                        float denominator_y_1_108 = _exp_266 + 1.0f;
                        float hidden_x_1_108 = gate_x_1_108 / denominator_x_1_108 * up_x_1_108;
                        float hidden_y_1_108 = gate_y_1_108 / denominator_y_1_108 * up_y_1_108;
                        hidden_packed_2[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_108, hidden_y_1_108)));
                        float2 _cvt_f32_109 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[13]));
                        float2 _cvt_f32_256_109 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[13]));
                        float gate_x_1_109 = _cvt_f32_109.x;
                        float gate_y_1_109 = _cvt_f32_109.y;
                        float up_x_1_109 = _cvt_f32_256_109.x;
                        float up_y_1_109 = _cvt_f32_256_109.y;
                        float _exp_267 = expf(gate_x_1_109 * -1.0f);
                        float denominator_x_1_109 = _exp_267 + 1.0f;
                        float _exp_268 = expf(gate_y_1_109 * -1.0f);
                        float denominator_y_1_109 = _exp_268 + 1.0f;
                        float hidden_x_1_109 = gate_x_1_109 / denominator_x_1_109 * up_x_1_109;
                        float hidden_y_1_109 = gate_y_1_109 / denominator_y_1_109 * up_y_1_109;
                        hidden_packed_2[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_109, hidden_y_1_109)));
                        float2 _cvt_f32_110 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[14]));
                        float2 _cvt_f32_256_110 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[14]));
                        float gate_x_1_110 = _cvt_f32_110.x;
                        float gate_y_1_110 = _cvt_f32_110.y;
                        float up_x_1_110 = _cvt_f32_256_110.x;
                        float up_y_1_110 = _cvt_f32_256_110.y;
                        float _exp_269 = expf(gate_x_1_110 * -1.0f);
                        float denominator_x_1_110 = _exp_269 + 1.0f;
                        float _exp_270 = expf(gate_y_1_110 * -1.0f);
                        float denominator_y_1_110 = _exp_270 + 1.0f;
                        float hidden_x_1_110 = gate_x_1_110 / denominator_x_1_110 * up_x_1_110;
                        float hidden_y_1_110 = gate_y_1_110 / denominator_y_1_110 * up_y_1_110;
                        hidden_packed_2[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_110, hidden_y_1_110)));
                        float2 _cvt_f32_111 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[15]));
                        float2 _cvt_f32_256_111 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[15]));
                        float gate_x_1_111 = _cvt_f32_111.x;
                        float gate_y_1_111 = _cvt_f32_111.y;
                        float up_x_1_111 = _cvt_f32_256_111.x;
                        float up_y_1_111 = _cvt_f32_256_111.y;
                        float _exp_271 = expf(gate_x_1_111 * -1.0f);
                        float denominator_x_1_111 = _exp_271 + 1.0f;
                        float _exp_272 = expf(gate_y_1_111 * -1.0f);
                        float denominator_y_1_111 = _exp_272 + 1.0f;
                        float hidden_x_1_111 = gate_x_1_111 / denominator_x_1_111 * up_x_1_111;
                        float hidden_y_1_111 = gate_y_1_111 / denominator_y_1_111 * up_y_1_111;
                        hidden_packed_2[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_111, hidden_y_1_111)));
                        float2 _cvt_f32_112 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[16]));
                        float2 _cvt_f32_256_112 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[16]));
                        float gate_x_1_112 = _cvt_f32_112.x;
                        float gate_y_1_112 = _cvt_f32_112.y;
                        float up_x_1_112 = _cvt_f32_256_112.x;
                        float up_y_1_112 = _cvt_f32_256_112.y;
                        float _exp_273 = expf(gate_x_1_112 * -1.0f);
                        float denominator_x_1_112 = _exp_273 + 1.0f;
                        float _exp_274 = expf(gate_y_1_112 * -1.0f);
                        float denominator_y_1_112 = _exp_274 + 1.0f;
                        float hidden_x_1_112 = gate_x_1_112 / denominator_x_1_112 * up_x_1_112;
                        float hidden_y_1_112 = gate_y_1_112 / denominator_y_1_112 * up_y_1_112;
                        hidden_packed_2[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_112, hidden_y_1_112)));
                        float2 _cvt_f32_113 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[17]));
                        float2 _cvt_f32_256_113 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[17]));
                        float gate_x_1_113 = _cvt_f32_113.x;
                        float gate_y_1_113 = _cvt_f32_113.y;
                        float up_x_1_113 = _cvt_f32_256_113.x;
                        float up_y_1_113 = _cvt_f32_256_113.y;
                        float _exp_275 = expf(gate_x_1_113 * -1.0f);
                        float denominator_x_1_113 = _exp_275 + 1.0f;
                        float _exp_276 = expf(gate_y_1_113 * -1.0f);
                        float denominator_y_1_113 = _exp_276 + 1.0f;
                        float hidden_x_1_113 = gate_x_1_113 / denominator_x_1_113 * up_x_1_113;
                        float hidden_y_1_113 = gate_y_1_113 / denominator_y_1_113 * up_y_1_113;
                        hidden_packed_2[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_113, hidden_y_1_113)));
                        float2 _cvt_f32_114 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[18]));
                        float2 _cvt_f32_256_114 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[18]));
                        float gate_x_1_114 = _cvt_f32_114.x;
                        float gate_y_1_114 = _cvt_f32_114.y;
                        float up_x_1_114 = _cvt_f32_256_114.x;
                        float up_y_1_114 = _cvt_f32_256_114.y;
                        float _exp_277 = expf(gate_x_1_114 * -1.0f);
                        float denominator_x_1_114 = _exp_277 + 1.0f;
                        float _exp_278 = expf(gate_y_1_114 * -1.0f);
                        float denominator_y_1_114 = _exp_278 + 1.0f;
                        float hidden_x_1_114 = gate_x_1_114 / denominator_x_1_114 * up_x_1_114;
                        float hidden_y_1_114 = gate_y_1_114 / denominator_y_1_114 * up_y_1_114;
                        hidden_packed_2[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_114, hidden_y_1_114)));
                        float2 _cvt_f32_115 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[19]));
                        float2 _cvt_f32_256_115 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[19]));
                        float gate_x_1_115 = _cvt_f32_115.x;
                        float gate_y_1_115 = _cvt_f32_115.y;
                        float up_x_1_115 = _cvt_f32_256_115.x;
                        float up_y_1_115 = _cvt_f32_256_115.y;
                        float _exp_279 = expf(gate_x_1_115 * -1.0f);
                        float denominator_x_1_115 = _exp_279 + 1.0f;
                        float _exp_280 = expf(gate_y_1_115 * -1.0f);
                        float denominator_y_1_115 = _exp_280 + 1.0f;
                        float hidden_x_1_115 = gate_x_1_115 / denominator_x_1_115 * up_x_1_115;
                        float hidden_y_1_115 = gate_y_1_115 / denominator_y_1_115 * up_y_1_115;
                        hidden_packed_2[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_115, hidden_y_1_115)));
                        float2 _cvt_f32_116 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[20]));
                        float2 _cvt_f32_256_116 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[20]));
                        float gate_x_1_116 = _cvt_f32_116.x;
                        float gate_y_1_116 = _cvt_f32_116.y;
                        float up_x_1_116 = _cvt_f32_256_116.x;
                        float up_y_1_116 = _cvt_f32_256_116.y;
                        float _exp_281 = expf(gate_x_1_116 * -1.0f);
                        float denominator_x_1_116 = _exp_281 + 1.0f;
                        float _exp_282 = expf(gate_y_1_116 * -1.0f);
                        float denominator_y_1_116 = _exp_282 + 1.0f;
                        float hidden_x_1_116 = gate_x_1_116 / denominator_x_1_116 * up_x_1_116;
                        float hidden_y_1_116 = gate_y_1_116 / denominator_y_1_116 * up_y_1_116;
                        hidden_packed_2[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_116, hidden_y_1_116)));
                        float2 _cvt_f32_117 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[21]));
                        float2 _cvt_f32_256_117 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[21]));
                        float gate_x_1_117 = _cvt_f32_117.x;
                        float gate_y_1_117 = _cvt_f32_117.y;
                        float up_x_1_117 = _cvt_f32_256_117.x;
                        float up_y_1_117 = _cvt_f32_256_117.y;
                        float _exp_283 = expf(gate_x_1_117 * -1.0f);
                        float denominator_x_1_117 = _exp_283 + 1.0f;
                        float _exp_284 = expf(gate_y_1_117 * -1.0f);
                        float denominator_y_1_117 = _exp_284 + 1.0f;
                        float hidden_x_1_117 = gate_x_1_117 / denominator_x_1_117 * up_x_1_117;
                        float hidden_y_1_117 = gate_y_1_117 / denominator_y_1_117 * up_y_1_117;
                        hidden_packed_2[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_117, hidden_y_1_117)));
                        float2 _cvt_f32_118 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[22]));
                        float2 _cvt_f32_256_118 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[22]));
                        float gate_x_1_118 = _cvt_f32_118.x;
                        float gate_y_1_118 = _cvt_f32_118.y;
                        float up_x_1_118 = _cvt_f32_256_118.x;
                        float up_y_1_118 = _cvt_f32_256_118.y;
                        float _exp_285 = expf(gate_x_1_118 * -1.0f);
                        float denominator_x_1_118 = _exp_285 + 1.0f;
                        float _exp_286 = expf(gate_y_1_118 * -1.0f);
                        float denominator_y_1_118 = _exp_286 + 1.0f;
                        float hidden_x_1_118 = gate_x_1_118 / denominator_x_1_118 * up_x_1_118;
                        float hidden_y_1_118 = gate_y_1_118 / denominator_y_1_118 * up_y_1_118;
                        hidden_packed_2[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_118, hidden_y_1_118)));
                        float2 _cvt_f32_119 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[23]));
                        float2 _cvt_f32_256_119 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[23]));
                        float gate_x_1_119 = _cvt_f32_119.x;
                        float gate_y_1_119 = _cvt_f32_119.y;
                        float up_x_1_119 = _cvt_f32_256_119.x;
                        float up_y_1_119 = _cvt_f32_256_119.y;
                        float _exp_287 = expf(gate_x_1_119 * -1.0f);
                        float denominator_x_1_119 = _exp_287 + 1.0f;
                        float _exp_288 = expf(gate_y_1_119 * -1.0f);
                        float denominator_y_1_119 = _exp_288 + 1.0f;
                        float hidden_x_1_119 = gate_x_1_119 / denominator_x_1_119 * up_x_1_119;
                        float hidden_y_1_119 = gate_y_1_119 / denominator_y_1_119 * up_y_1_119;
                        hidden_packed_2[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_119, hidden_y_1_119)));
                        float2 _cvt_f32_120 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[24]));
                        float2 _cvt_f32_256_120 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[24]));
                        float gate_x_1_120 = _cvt_f32_120.x;
                        float gate_y_1_120 = _cvt_f32_120.y;
                        float up_x_1_120 = _cvt_f32_256_120.x;
                        float up_y_1_120 = _cvt_f32_256_120.y;
                        float _exp_289 = expf(gate_x_1_120 * -1.0f);
                        float denominator_x_1_120 = _exp_289 + 1.0f;
                        float _exp_290 = expf(gate_y_1_120 * -1.0f);
                        float denominator_y_1_120 = _exp_290 + 1.0f;
                        float hidden_x_1_120 = gate_x_1_120 / denominator_x_1_120 * up_x_1_120;
                        float hidden_y_1_120 = gate_y_1_120 / denominator_y_1_120 * up_y_1_120;
                        hidden_packed_2[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_120, hidden_y_1_120)));
                        float2 _cvt_f32_121 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[25]));
                        float2 _cvt_f32_256_121 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[25]));
                        float gate_x_1_121 = _cvt_f32_121.x;
                        float gate_y_1_121 = _cvt_f32_121.y;
                        float up_x_1_121 = _cvt_f32_256_121.x;
                        float up_y_1_121 = _cvt_f32_256_121.y;
                        float _exp_291 = expf(gate_x_1_121 * -1.0f);
                        float denominator_x_1_121 = _exp_291 + 1.0f;
                        float _exp_292 = expf(gate_y_1_121 * -1.0f);
                        float denominator_y_1_121 = _exp_292 + 1.0f;
                        float hidden_x_1_121 = gate_x_1_121 / denominator_x_1_121 * up_x_1_121;
                        float hidden_y_1_121 = gate_y_1_121 / denominator_y_1_121 * up_y_1_121;
                        hidden_packed_2[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_121, hidden_y_1_121)));
                        float2 _cvt_f32_122 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[26]));
                        float2 _cvt_f32_256_122 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[26]));
                        float gate_x_1_122 = _cvt_f32_122.x;
                        float gate_y_1_122 = _cvt_f32_122.y;
                        float up_x_1_122 = _cvt_f32_256_122.x;
                        float up_y_1_122 = _cvt_f32_256_122.y;
                        float _exp_293 = expf(gate_x_1_122 * -1.0f);
                        float denominator_x_1_122 = _exp_293 + 1.0f;
                        float _exp_294 = expf(gate_y_1_122 * -1.0f);
                        float denominator_y_1_122 = _exp_294 + 1.0f;
                        float hidden_x_1_122 = gate_x_1_122 / denominator_x_1_122 * up_x_1_122;
                        float hidden_y_1_122 = gate_y_1_122 / denominator_y_1_122 * up_y_1_122;
                        hidden_packed_2[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_122, hidden_y_1_122)));
                        float2 _cvt_f32_123 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[27]));
                        float2 _cvt_f32_256_123 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[27]));
                        float gate_x_1_123 = _cvt_f32_123.x;
                        float gate_y_1_123 = _cvt_f32_123.y;
                        float up_x_1_123 = _cvt_f32_256_123.x;
                        float up_y_1_123 = _cvt_f32_256_123.y;
                        float _exp_295 = expf(gate_x_1_123 * -1.0f);
                        float denominator_x_1_123 = _exp_295 + 1.0f;
                        float _exp_296 = expf(gate_y_1_123 * -1.0f);
                        float denominator_y_1_123 = _exp_296 + 1.0f;
                        float hidden_x_1_123 = gate_x_1_123 / denominator_x_1_123 * up_x_1_123;
                        float hidden_y_1_123 = gate_y_1_123 / denominator_y_1_123 * up_y_1_123;
                        hidden_packed_2[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_123, hidden_y_1_123)));
                        float2 _cvt_f32_124 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[28]));
                        float2 _cvt_f32_256_124 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[28]));
                        float gate_x_1_124 = _cvt_f32_124.x;
                        float gate_y_1_124 = _cvt_f32_124.y;
                        float up_x_1_124 = _cvt_f32_256_124.x;
                        float up_y_1_124 = _cvt_f32_256_124.y;
                        float _exp_297 = expf(gate_x_1_124 * -1.0f);
                        float denominator_x_1_124 = _exp_297 + 1.0f;
                        float _exp_298 = expf(gate_y_1_124 * -1.0f);
                        float denominator_y_1_124 = _exp_298 + 1.0f;
                        float hidden_x_1_124 = gate_x_1_124 / denominator_x_1_124 * up_x_1_124;
                        float hidden_y_1_124 = gate_y_1_124 / denominator_y_1_124 * up_y_1_124;
                        hidden_packed_2[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_124, hidden_y_1_124)));
                        float2 _cvt_f32_125 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[29]));
                        float2 _cvt_f32_256_125 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[29]));
                        float gate_x_1_125 = _cvt_f32_125.x;
                        float gate_y_1_125 = _cvt_f32_125.y;
                        float up_x_1_125 = _cvt_f32_256_125.x;
                        float up_y_1_125 = _cvt_f32_256_125.y;
                        float _exp_299 = expf(gate_x_1_125 * -1.0f);
                        float denominator_x_1_125 = _exp_299 + 1.0f;
                        float _exp_300 = expf(gate_y_1_125 * -1.0f);
                        float denominator_y_1_125 = _exp_300 + 1.0f;
                        float hidden_x_1_125 = gate_x_1_125 / denominator_x_1_125 * up_x_1_125;
                        float hidden_y_1_125 = gate_y_1_125 / denominator_y_1_125 * up_y_1_125;
                        hidden_packed_2[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_125, hidden_y_1_125)));
                        float2 _cvt_f32_126 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[30]));
                        float2 _cvt_f32_256_126 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[30]));
                        float gate_x_1_126 = _cvt_f32_126.x;
                        float gate_y_1_126 = _cvt_f32_126.y;
                        float up_x_1_126 = _cvt_f32_256_126.x;
                        float up_y_1_126 = _cvt_f32_256_126.y;
                        float _exp_301 = expf(gate_x_1_126 * -1.0f);
                        float denominator_x_1_126 = _exp_301 + 1.0f;
                        float _exp_302 = expf(gate_y_1_126 * -1.0f);
                        float denominator_y_1_126 = _exp_302 + 1.0f;
                        float hidden_x_1_126 = gate_x_1_126 / denominator_x_1_126 * up_x_1_126;
                        float hidden_y_1_126 = gate_y_1_126 / denominator_y_1_126 * up_y_1_126;
                        hidden_packed_2[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_126, hidden_y_1_126)));
                        float2 _cvt_f32_127 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_3[31]));
                        float2 _cvt_f32_256_127 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_3[31]));
                        float gate_x_1_127 = _cvt_f32_127.x;
                        float gate_y_1_127 = _cvt_f32_127.y;
                        float up_x_1_127 = _cvt_f32_256_127.x;
                        float up_y_1_127 = _cvt_f32_256_127.y;
                        float _exp_303 = expf(gate_x_1_127 * -1.0f);
                        float denominator_x_1_127 = _exp_303 + 1.0f;
                        float _exp_304 = expf(gate_y_1_127 * -1.0f);
                        float denominator_y_1_127 = _exp_304 + 1.0f;
                        float hidden_x_1_127 = gate_x_1_127 / denominator_x_1_127 * up_x_1_127;
                        float hidden_y_1_127 = gate_y_1_127 / denominator_y_1_127 * up_y_1_127;
                        hidden_packed_2[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_127, hidden_y_1_127)));
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_5 = tid / 32;
                        int lane_5 = tid % 32;
                        #pragma unroll
                        for (int half_6 = 0; half_6 < 2; half_6++) {
                            #pragma unroll
                            for (int col_tile_6 = 0; col_tile_6 < 2; col_tile_6++) {
                                int row_10 = warp_5 * 32 + half_6 * 16 + lane_5 % 16;
                                int col_7 = col_tile_6 * 16 + lane_5 / 16 * 8;
                                unsigned int address_6 = d_smem_addr + (unsigned int)((row_10 * 32 + col_7) * 2);
                                address_6 = address_6 ^ (address_6 & 511) >> 7 << 4;
                                int offset_6 = half_6 * 8 + col_tile_6 * 4;
                                uint32_t _stmatrix_addr_305 = static_cast<uint32_t>(address_6);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_305), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_6])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_6 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_6 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_6 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (tid == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int warp_6 = tid / 32;
                        int lane_6 = tid % 32;
                        #pragma unroll
                        for (int half_7 = 0; half_7 < 2; half_7++) {
                            #pragma unroll
                            for (int col_tile_7 = 0; col_tile_7 < 2; col_tile_7++) {
                                int row_11 = warp_6 * 32 + half_7 * 16 + lane_6 % 16;
                                int col_8 = col_tile_7 * 16 + lane_6 / 16 * 8;
                                unsigned int address_7 = d_smem_addr + 8192 + (unsigned int)((row_11 * 32 + col_8) * 2);
                                address_7 = address_7 ^ (address_7 & 511) >> 7 << 4;
                                int offset_7 = 16 + half_7 * 8 + col_tile_7 * 4;
                                uint32_t _stmatrix_addr_306 = static_cast<uint32_t>(address_7);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_306), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_7])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_7 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_7 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_2[offset_7 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid == 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                :: "l"((&hidden_shared_out)), "r"(0), "r"(x * 256 + cta_rank_0 * 128), "r"(y * 8 + 4 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
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
                gemm_wide += iterations * (int)(expert >= 0);
                gemm_out0 += (int)(expert >= 0);
                gemm_out1 += (int)(expert >= 0);
                gemm_acc_tiles += 2 * (int)(expert >= 0);
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
                    int row_12 = 0;
                    int col_9 = 0;
                    if (compute - shared_fused < row_blocks_1 * full_cols_1) {
                        row_12 = (compute - shared_fused) % (row_blocks_1 * 8) / 8;
                        col_9 = supergroup_1 * 8 + (compute - shared_fused) % 8;
                    } else {
                        row_12 = (compute - shared_fused - row_blocks_1 * full_cols_1) / (col_blocks_2 - full_cols_1);
                        col_9 = full_cols_1 + (compute - shared_fused - row_blocks_1 * full_cols_1) % (col_blocks_2 - full_cols_1);
                    }
                    if ((supergroup_1 & 1) != 0) {
                        row_12 = row_blocks_1 - row_12 - 1;
                    }
                    x_1 = row_12;
                    y_1 = col_9;
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
                } else if (tid / 32 == 7) {
                    if (warp == 7) {
                        if (elect_sync()) {
                            {
                                bool enabled_value_3 = 1;
                                if (enabled_value_3 != 0) {
                                    int poll_4_first;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_first) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                    int poll_4 = poll_4_first;
                                    while (poll_4 < 2 * (intermediate / 128)) {
                                        asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                        int poll_4_next;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_4_next) : "l"(hidden_ready + (macro_rows_2 + x_1)) : "memory");
                                        poll_4 = poll_4_next;
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                int _min_307 = ((mini_size) < (tokens - global_mini_1 * mini_size) ? (mini_size) : (tokens - global_mini_1 * mini_size));
                                int mini_rows_5 = ((0) > (_min_307) ? (0) : (_min_307));
                                int required_3 = (mini_rows_5 + 127) / 128 * ((intermediate + 511) / 512);
                            }
                            #pragma unroll 1
                            for (int idx_6 = 0; idx_6 < iterations_1; idx_6++) {
                                int _stage_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_6) % 4);
                                int _parity_gemm_finished_1 = (int)((unsigned int)(gemm_wide + idx_6) / 4 % 2) ^ 1;
                                mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_1) * 8, _parity_gemm_finished_1);
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&hidden_shared_in)), "r"(0), "r"(x_1 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(0), "r"(0),
                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&wd_shared)), "r"(0), "r"(y_1 * 2 * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_6) % 4 * 16384)), "l"((&wd_shared)), "r"(0), "r"((y_1 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_6), "r"(expert_1), "r"(0),
                                       "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_6) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                            }
                        }
                    }
                } else {
                    if (tid / 32 == 4 && cta_rank_0 == 0) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                int head_1 = ((3) < (iterations_1) ? (3) : (iterations_1));
                                int tail_2 = 0;
                                int _min_308 = ((3) < (iterations_1 - head_1) ? (3) : (iterations_1 - head_1));
                                tail_2 = _min_308;
                                int _stage_output_finished_2 = (int)((unsigned int)gemm_acc_tiles % 2);
                                int _parity_output_finished_2 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished_2) * 8, _parity_output_finished_2);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_7 = 0; idx_7 < head_1; idx_7++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_7) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_7) % 4);
                                    int _parity_gemm_arrived_3 = (int)((unsigned int)(gemm_wide + idx_7) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_3) * 8, _parity_gemm_arrived_3);
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_309 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_7) % 4) * 1024;
                                    int _smem_desc_lo_310 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_7) % 4) * 1024;
#else
                                    int _smem_desc_lo_309 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_7) % 4) * 1024;
                                    int _smem_desc_lo_310 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_7) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_309), "r"(_smem_desc_lo_310), "r"(tmem_accumulator), "r"(((idx_7 == 0) ? 0 : 1)) : "memory");
                                }
                                int _stage_output_finished_3 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                int _parity_output_finished_3 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                mbarrier_wait(output_finished_addr + (_stage_output_finished_3) * 8, _parity_output_finished_3);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                #pragma unroll 1
                                for (int idx_8 = 0; idx_8 < head_1; idx_8++) {
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_311 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_8) % 4) * 1024;
                                    int _smem_desc_lo_312 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_8) % 4) * 1024;
#else
                                    int _smem_desc_lo_311 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_8) % 4) * 1024;
                                    int _smem_desc_lo_312 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_8) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_311), "r"(_smem_desc_lo_312), "r"((tmem_accumulator + (256))), "r"(((idx_8 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_8) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_9 = head_1; idx_9 < iterations_1 - tail_2; idx_9++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) % 4);
                                    int _parity_gemm_arrived_4 = (int)((unsigned int)(gemm_wide + idx_9) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_4) * 8, _parity_gemm_arrived_4);
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_313 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                    int _smem_desc_lo_314 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
#else
                                    int _smem_desc_lo_313 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                    int _smem_desc_lo_314 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_313), "r"(_smem_desc_lo_314), "r"(tmem_accumulator), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_315 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                    int _smem_desc_lo_316 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_9) % 4) * 1024;
#else
                                    int _smem_desc_lo_315 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
                                    int _smem_desc_lo_316 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_9) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_315), "r"(_smem_desc_lo_316), "r"((tmem_accumulator + (256))), "r"(((idx_9 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_9) % 4)) * 8, (uint16_t)(3));
                                }
                                #pragma unroll 1
                                for (int idx_10 = iterations_1 - tail_2; idx_10 < iterations_1; idx_10++) {
                                    mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_10) % 4)) * 8, 98304);
                                    int _stage_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_10) % 4);
                                    int _parity_gemm_arrived_5 = (int)((unsigned int)(gemm_wide + idx_10) / 4 % 2);
                                    mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_5) * 8, _parity_gemm_arrived_5);
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_317 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_10) % 4) * 1024;
                                    int _smem_desc_lo_318 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_10) % 4) * 1024;
#else
                                    int _smem_desc_lo_317 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_10) % 4) * 1024;
                                    int _smem_desc_lo_318 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_10) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_317), "r"(_smem_desc_lo_318), "r"(tmem_accumulator), "r"(((idx_10 == 0) ? 0 : 1)) : "memory");
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                #pragma unroll 1
                                for (int idx_11 = iterations_1 - tail_2; idx_11 < iterations_1; idx_11++) {
#if __CUDA_ARCH__ == 1070
                                    int _smem_desc_lo_319 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_11) % 4) * 1024;
                                    int _smem_desc_lo_320 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_11) % 4) * 1024;
#else
                                    int _smem_desc_lo_319 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_11) % 4) * 1024;
                                    int _smem_desc_lo_320 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_11) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_319), "r"(_smem_desc_lo_320), "r"((tmem_accumulator + (256))), "r"(((idx_11 == 0) ? 0 : 1)) : "memory");
                                    tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_11) % 4)) * 8, (uint16_t)(3));
                                }
                                tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                            }
                        }
                    } else if (tid < 128) {
                        int _parity_output_arrived_2 = (int)((unsigned int)gemm_out0 % 2);
                        mbarrier_wait(output_arrived_addr, _parity_output_arrived_2);
                        unsigned int packed[128];
                        #pragma unroll
                        for (int chunk = 0; chunk < 8; chunk++) {
                            #pragma unroll
                            for (int sub = 0; sub < 2; sub++) {
                                unsigned int address_8 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub * 16 << 16) + (unsigned int)(chunk * 32);
                                float _tmem_load_32[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[15]))
                                    : "r"(address_8));
                                #pragma unroll
                                for (int pair = 0; pair < 8; pair++) {
                                    packed[chunk * 16 + sub * 8 + pair] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32[pair * 2], _tmem_load_32[pair * 2 + 1])));
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int last = 1;
                        if (last != 0) {
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        if (tid == 0) {
                            int previous_offset_1 = macro_size;
                            int output_row = x_1 * 256 + cta_rank_0 * 128;
                            int _min_321 = ((macro_size) < (tokens - previous_offset_1) ? (macro_size) : (tokens - previous_offset_1));
                            if (output_row < _min_321) {
                            }
                        }
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < 8; chunk_1++) {
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_7 = tid / 32;
                            int lane_7 = tid % 32;
                            #pragma unroll
                            for (int half_8 = 0; half_8 < 2; half_8++) {
                                #pragma unroll
                                for (int col_tile_8 = 0; col_tile_8 < 2; col_tile_8++) {
                                    int row_13 = warp_7 * 32 + half_8 * 16 + lane_7 % 16;
                                    int col_10 = col_tile_8 * 16 + lane_7 / 16 * 8;
                                    unsigned int address_9 = d_smem_addr + (unsigned int)(chunk_1 % 3 * 8192) + (unsigned int)((row_13 * 32 + col_10) * 2);
                                    address_9 = address_9 ^ (address_9 & 511) >> 7 << 4;
                                    int offset_8 = chunk_1 * 16 + half_8 * 8 + col_tile_8 * 4;
                                    uint32_t _stmatrix_addr_322 = static_cast<uint32_t>(address_9);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_322), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_8])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_8 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_8 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[offset_8 + 3]))
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
                        int _parity_output_arrived_3 = (int)((unsigned int)gemm_out1 % 2);
                        mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_3);
                        if (has_hi_1 != 0) {
                            unsigned int packed_0[128];
                            #pragma unroll
                            for (int chunk_2 = 0; chunk_2 < 8; chunk_2++) {
                                #pragma unroll
                                for (int sub_1 = 0; sub_1 < 2; sub_1++) {
                                    unsigned int address_10 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_1 * 16 << 16) + 256 + (unsigned int)(chunk_2 * 32);
                                    float _tmem_load_32_16[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16[15]))
                                        : "r"(address_10));
                                    #pragma unroll
                                    for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                                        packed_0[chunk_2 * 16 + sub_1 * 8 + pair_1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16[pair_1 * 2], _tmem_load_32_16[pair_1 * 2 + 1])));
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
                                        :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            #pragma unroll
                            for (int chunk_3 = 0; chunk_3 < 8; chunk_3++) {
                                if (tid == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 2;");
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                int warp_8 = tid / 32;
                                int lane_8 = tid % 32;
                                #pragma unroll
                                for (int half_9 = 0; half_9 < 2; half_9++) {
                                    #pragma unroll
                                    for (int col_tile_9 = 0; col_tile_9 < 2; col_tile_9++) {
                                        int row_14 = warp_8 * 32 + half_9 * 16 + lane_8 % 16;
                                        int col_11 = col_tile_9 * 16 + lane_8 / 16 * 8;
                                        unsigned int address_11 = d_smem_addr + (unsigned int)((8 + chunk_3) % 3 * 8192) + (unsigned int)((row_14 * 32 + col_11) * 2);
                                        address_11 = address_11 ^ (address_11 & 511) >> 7 << 4;
                                        int offset_9 = chunk_3 * 16 + half_9 * 8 + col_tile_9 * 4;
                                        uint32_t _stmatrix_addr_323 = static_cast<uint32_t>(address_11);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_323), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_9 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_9 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0[offset_9 + 3]))
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
                        if (has_hi_1 == 0) {
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
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
                gemm_wide += iterations_1 * (int)(expert_1 >= 0);
                gemm_out0 += (int)(expert_1 >= 0);
                gemm_out1 += (int)(expert_1 >= 0);
                gemm_acc_tiles += 2 * (int)(expert_1 >= 0);
            } else {
                int task_1 = 0;
                int macro_1 = 0;
                int mini_3 = 0;
                int swiglu = 0;
                if (compute >= shared_tasks) {
                    uint32_t _fast_div_q_0 = (mini_tasks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(compute - shared_tasks)), mini_tasks_div.multiplier) >> mini_tasks_div.shift_right) : (uint32_t)((unsigned int)(compute - shared_tasks));
                    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)(compute - shared_tasks)) - _fast_div_q_0 * (uint32_t)(mini_tasks_div.divisor);
                    int ordered_mini = (int)_fast_div_q_0;
                    task_1 = (int)_fast_div_r_0;
                    macro_1 = macros - 1;
                    mini_3 = ordered_mini;
                    if (ordered_mini >= last_minis) {
                        uint32_t _fast_div_q_1 = (minis_per_macro_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(ordered_mini - last_minis)), minis_per_macro_div.multiplier) >> minis_per_macro_div.shift_right) : (uint32_t)((unsigned int)(ordered_mini - last_minis));
                        uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)(ordered_mini - last_minis)) - _fast_div_q_1 * (uint32_t)(minis_per_macro_div.divisor);
                        macro_1 = macros - 2 - (int)_fast_div_q_1;
                        mini_3 = (int)_fast_div_r_1;
                    }
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
                    int end_block = ((first_block + mini_size / 256) < (tokens / 256) ? (first_block + mini_size / 256) : (tokens / 256));
                    int block_5 = first_block;
                    uint32_t fast_div_q = (fused_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)task_1), fused_col_blocks_div.multiplier) >> fused_col_blocks_div.shift_right) : (uint32_t)((unsigned int)task_1);
                    uint32_t fast_div_r = (uint32_t)((unsigned int)task_1) - fast_div_q * (uint32_t)(fused_col_blocks_div.divisor);
                    block_5 = first_block + (int)fast_div_q;
                    if (block_5 < end_block) {
                        int index = counts[3 * experts + block_5];
                        int offset_10 = counts[experts + index] / 256;
                        int first_row_6 = ((first_block) > (offset_10) ? (first_block) : (offset_10));
                        int _min_324 = ((end_block) < (offset_10 + counts[index] / 256) ? (end_block) : (offset_10 + counts[index] / 256));
                        int rows_5 = _min_324 - first_row_6;
                        int supergroup_2 = (task_1 - (first_row_6 - first_block) * col_blocks_3) / (rows_5 * 8);
                        int full_cols_2 = col_blocks_3 / 8 * 8;
                        int row_15 = 0;
                        int col_12 = 0;
                        if (task_1 - (first_row_6 - first_block) * col_blocks_3 < rows_5 * full_cols_2) {
                            row_15 = (task_1 - (first_row_6 - first_block) * col_blocks_3) % (rows_5 * 8) / 8;
                            col_12 = supergroup_2 * 8 + (task_1 - (first_row_6 - first_block) * col_blocks_3) % 8;
                        } else {
                            row_15 = (task_1 - (first_row_6 - first_block) * col_blocks_3 - rows_5 * full_cols_2) / (col_blocks_3 - full_cols_2);
                            col_12 = full_cols_2 + (task_1 - (first_row_6 - first_block) * col_blocks_3 - rows_5 * full_cols_2) % (col_blocks_3 - full_cols_2);
                        }
                        if ((supergroup_2 & 1) != 0) {
                            row_15 = rows_5 - row_15 - 1;
                        }
                        x_2 = first_row_6 + row_15 - macro_1 * (macro_size / 256);
                        y_2 = col_12;
                        expert_2 = index;
                    }
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
                                    int _min_325 = ((mini_size) < (tokens - global_mini_2 * mini_size) ? (mini_size) : (tokens - global_mini_2 * mini_size));
                                    int mini_rows_6 = ((0) > (_min_325) ? (0) : (_min_325));
                                    int required_4 = (mini_rows_6 + 127) / 128 * ((hidden + 511) / 512);
                                    bool enabled_value_4 = 1;
                                    if (enabled_value_4 != 0) {
                                        int poll_5_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_first) : "l"(x_ready + global_mini_2) : "memory");
                                        int poll_5 = poll_5_first;
                                        while (poll_5 < required_4) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_5_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_5_next) : "l"(x_ready + global_mini_2) : "memory");
                                            poll_5 = poll_5_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                }
                                #pragma unroll 1
                                for (int idx_12 = 0; idx_12 < iterations_2; idx_12++) {
                                    int _stage_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_12) % 4);
                                    int _parity_gemm_finished_2 = (int)((unsigned int)(gemm_wide + idx_12) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_2) * 8, _parity_gemm_finished_2);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&x_routed)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(idx_12), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    if (cta_rank_0 == 0) {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y_2 * 256), "r"(idx_12), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    } else {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y_2 * 256), "r"(idx_12), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                    if (cta_rank_0 == 0) {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&wg_routed)), "r"(0), "r"(y_2 * 256 + 128), "r"(idx_12), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    } else {
                                        asm volatile(
                                            "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                            :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_12) % 4 * 16384)), "l"((&wu_routed)), "r"(0), "r"(y_2 * 256 + 128), "r"(idx_12), "r"(expert_2), "r"(0),
                                               "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_12) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    }
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int head_2 = ((3) < (iterations_2) ? (3) : (iterations_2));
                                    int tail_3 = 0;
                                    int _min_326 = ((3) < (iterations_2 - head_2) ? (3) : (iterations_2 - head_2));
                                    tail_3 = _min_326;
                                    int _stage_output_finished_4 = (int)((unsigned int)gemm_acc_tiles % 2);
                                    int _parity_output_finished_4 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_4) * 8, _parity_output_finished_4);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_13 = 0; idx_13 < head_2; idx_13++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_13) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_6 = (int)((unsigned int)(gemm_wide + idx_13) % 4);
                                        int _parity_gemm_arrived_6 = (int)((unsigned int)(gemm_wide + idx_13) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_6) * 8, _parity_gemm_arrived_6);
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_327 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_13) % 4) * 1024;
                                        int _smem_desc_lo_328 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_13) % 4) * 1024;
#else
                                        int _smem_desc_lo_327 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_13) % 4) * 1024;
                                        int _smem_desc_lo_328 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_13) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_327), "r"(_smem_desc_lo_328), "r"(tmem_accumulator), "r"(((idx_13 == 0) ? 0 : 1)) : "memory");
                                    }
                                    int _stage_output_finished_5 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                    int _parity_output_finished_5 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_5) * 8, _parity_output_finished_5);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_14 = 0; idx_14 < head_2; idx_14++) {
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_329 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_14) % 4) * 1024;
                                        int _smem_desc_lo_330 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_14) % 4) * 1024;
#else
                                        int _smem_desc_lo_329 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_14) % 4) * 1024;
                                        int _smem_desc_lo_330 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_14) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_329), "r"(_smem_desc_lo_330), "r"((tmem_accumulator + (256))), "r"(((idx_14 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_14) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_15 = head_2; idx_15 < iterations_2 - tail_3; idx_15++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_15) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_7 = (int)((unsigned int)(gemm_wide + idx_15) % 4);
                                        int _parity_gemm_arrived_7 = (int)((unsigned int)(gemm_wide + idx_15) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_7) * 8, _parity_gemm_arrived_7);
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_331 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                        int _smem_desc_lo_332 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_15) % 4) * 1024;
#else
                                        int _smem_desc_lo_331 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                        int _smem_desc_lo_332 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_15) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_331), "r"(_smem_desc_lo_332), "r"(tmem_accumulator), "r"(((idx_15 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_333 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                        int _smem_desc_lo_334 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_15) % 4) * 1024;
#else
                                        int _smem_desc_lo_333 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_15) % 4) * 1024;
                                        int _smem_desc_lo_334 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_15) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_333), "r"(_smem_desc_lo_334), "r"((tmem_accumulator + (256))), "r"(((idx_15 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_15) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_16 = iterations_2 - tail_3; idx_16 < iterations_2; idx_16++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_16) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_8 = (int)((unsigned int)(gemm_wide + idx_16) % 4);
                                        int _parity_gemm_arrived_8 = (int)((unsigned int)(gemm_wide + idx_16) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_8) * 8, _parity_gemm_arrived_8);
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_335 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_16) % 4) * 1024;
                                        int _smem_desc_lo_336 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_16) % 4) * 1024;
#else
                                        int _smem_desc_lo_335 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_16) % 4) * 1024;
                                        int _smem_desc_lo_336 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_16) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_335), "r"(_smem_desc_lo_336), "r"(tmem_accumulator), "r"(((idx_16 == 0) ? 0 : 1)) : "memory");
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    #pragma unroll 1
                                    for (int idx_17 = iterations_2 - tail_3; idx_17 < iterations_2; idx_17++) {
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_337 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_17) % 4) * 1024;
                                        int _smem_desc_lo_338 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_17) % 4) * 1024;
#else
                                        int _smem_desc_lo_337 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_17) % 4) * 1024;
                                        int _smem_desc_lo_338 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_17) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_337), "r"(_smem_desc_lo_338), "r"((tmem_accumulator + (256))), "r"(((idx_17 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_17) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            int _parity_output_arrived_4 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_4);
                            unsigned int gate_packed_4[32];
                            unsigned int up_packed_4[32];
                            unsigned int address_3_16 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16);
                            float _tmem_load_16[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                                : "r"(address_3_16));
                            float _tmem_load_32_16_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_16_1[15]))
                                : "r"(address_3_16 + 128));
                            gate_packed_4[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[0], _tmem_load_16[1])));
                            up_packed_4[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[0], _tmem_load_32_16_1[1])));
                            gate_packed_4[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[2], _tmem_load_16[3])));
                            up_packed_4[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[2], _tmem_load_32_16_1[3])));
                            gate_packed_4[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[4], _tmem_load_16[5])));
                            up_packed_4[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[4], _tmem_load_32_16_1[5])));
                            gate_packed_4[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[6], _tmem_load_16[7])));
                            up_packed_4[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[6], _tmem_load_32_16_1[7])));
                            gate_packed_4[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[8], _tmem_load_16[9])));
                            up_packed_4[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[8], _tmem_load_32_16_1[9])));
                            gate_packed_4[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[10], _tmem_load_16[11])));
                            up_packed_4[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[10], _tmem_load_32_16_1[11])));
                            gate_packed_4[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[12], _tmem_load_16[13])));
                            up_packed_4[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[12], _tmem_load_32_16_1[13])));
                            gate_packed_4[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_16[14], _tmem_load_16[15])));
                            up_packed_4[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_16_1[14], _tmem_load_32_16_1[15])));
                            unsigned int address_3_17 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16);
                            float _tmem_load_17[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15]))
                                : "r"(address_3_17));
                            float _tmem_load_32_17[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_17[15]))
                                : "r"(address_3_17 + 128));
                            gate_packed_4[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[0], _tmem_load_17[1])));
                            up_packed_4[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[0], _tmem_load_32_17[1])));
                            gate_packed_4[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[2], _tmem_load_17[3])));
                            up_packed_4[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[2], _tmem_load_32_17[3])));
                            gate_packed_4[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[4], _tmem_load_17[5])));
                            up_packed_4[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[4], _tmem_load_32_17[5])));
                            gate_packed_4[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[6], _tmem_load_17[7])));
                            up_packed_4[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[6], _tmem_load_32_17[7])));
                            gate_packed_4[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[8], _tmem_load_17[9])));
                            up_packed_4[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[8], _tmem_load_32_17[9])));
                            gate_packed_4[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[10], _tmem_load_17[11])));
                            up_packed_4[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[10], _tmem_load_32_17[11])));
                            gate_packed_4[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[12], _tmem_load_17[13])));
                            up_packed_4[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[12], _tmem_load_32_17[13])));
                            gate_packed_4[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_17[14], _tmem_load_17[15])));
                            up_packed_4[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_17[14], _tmem_load_32_17[15])));
                            unsigned int address_3_18 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 32;
                            float _tmem_load_18[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[15]))
                                : "r"(address_3_18));
                            float _tmem_load_32_18[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_18[15]))
                                : "r"(address_3_18 + 128));
                            gate_packed_4[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[0], _tmem_load_18[1])));
                            up_packed_4[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[0], _tmem_load_32_18[1])));
                            gate_packed_4[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[2], _tmem_load_18[3])));
                            up_packed_4[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[2], _tmem_load_32_18[3])));
                            gate_packed_4[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[4], _tmem_load_18[5])));
                            up_packed_4[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[4], _tmem_load_32_18[5])));
                            gate_packed_4[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[6], _tmem_load_18[7])));
                            up_packed_4[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[6], _tmem_load_32_18[7])));
                            gate_packed_4[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[8], _tmem_load_18[9])));
                            up_packed_4[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[8], _tmem_load_32_18[9])));
                            gate_packed_4[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[10], _tmem_load_18[11])));
                            up_packed_4[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[10], _tmem_load_32_18[11])));
                            gate_packed_4[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[12], _tmem_load_18[13])));
                            up_packed_4[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[12], _tmem_load_32_18[13])));
                            gate_packed_4[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_18[14], _tmem_load_18[15])));
                            up_packed_4[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_18[14], _tmem_load_32_18[15])));
                            unsigned int address_3_19 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 32;
                            float _tmem_load_19[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[15]))
                                : "r"(address_3_19));
                            float _tmem_load_32_19[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_19[15]))
                                : "r"(address_3_19 + 128));
                            gate_packed_4[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[0], _tmem_load_19[1])));
                            up_packed_4[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[0], _tmem_load_32_19[1])));
                            gate_packed_4[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[2], _tmem_load_19[3])));
                            up_packed_4[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[2], _tmem_load_32_19[3])));
                            gate_packed_4[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[4], _tmem_load_19[5])));
                            up_packed_4[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[4], _tmem_load_32_19[5])));
                            gate_packed_4[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[6], _tmem_load_19[7])));
                            up_packed_4[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[6], _tmem_load_32_19[7])));
                            gate_packed_4[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[8], _tmem_load_19[9])));
                            up_packed_4[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[8], _tmem_load_32_19[9])));
                            gate_packed_4[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[10], _tmem_load_19[11])));
                            up_packed_4[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[10], _tmem_load_32_19[11])));
                            gate_packed_4[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[12], _tmem_load_19[13])));
                            up_packed_4[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[12], _tmem_load_32_19[13])));
                            gate_packed_4[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_19[14], _tmem_load_19[15])));
                            up_packed_4[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_19[14], _tmem_load_32_19[15])));
                            unsigned int gate_packed_5[32];
                            unsigned int up_packed_5[32];
                            unsigned int address_3_20 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 64;
                            float _tmem_load_20[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[15]))
                                : "r"(address_3_20));
                            float _tmem_load_32_20[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_20[15]))
                                : "r"(address_3_20 + 128));
                            gate_packed_5[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[0], _tmem_load_20[1])));
                            up_packed_5[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[0], _tmem_load_32_20[1])));
                            gate_packed_5[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[2], _tmem_load_20[3])));
                            up_packed_5[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[2], _tmem_load_32_20[3])));
                            gate_packed_5[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[4], _tmem_load_20[5])));
                            up_packed_5[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[4], _tmem_load_32_20[5])));
                            gate_packed_5[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[6], _tmem_load_20[7])));
                            up_packed_5[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[6], _tmem_load_32_20[7])));
                            gate_packed_5[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[8], _tmem_load_20[9])));
                            up_packed_5[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[8], _tmem_load_32_20[9])));
                            gate_packed_5[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[10], _tmem_load_20[11])));
                            up_packed_5[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[10], _tmem_load_32_20[11])));
                            gate_packed_5[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[12], _tmem_load_20[13])));
                            up_packed_5[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[12], _tmem_load_32_20[13])));
                            gate_packed_5[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_20[14], _tmem_load_20[15])));
                            up_packed_5[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_20[14], _tmem_load_32_20[15])));
                            unsigned int address_3_21 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 64;
                            float _tmem_load_21[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[15]))
                                : "r"(address_3_21));
                            float _tmem_load_32_21[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_21[15]))
                                : "r"(address_3_21 + 128));
                            gate_packed_5[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[0], _tmem_load_21[1])));
                            up_packed_5[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[0], _tmem_load_32_21[1])));
                            gate_packed_5[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[2], _tmem_load_21[3])));
                            up_packed_5[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[2], _tmem_load_32_21[3])));
                            gate_packed_5[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[4], _tmem_load_21[5])));
                            up_packed_5[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[4], _tmem_load_32_21[5])));
                            gate_packed_5[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[6], _tmem_load_21[7])));
                            up_packed_5[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[6], _tmem_load_32_21[7])));
                            gate_packed_5[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[8], _tmem_load_21[9])));
                            up_packed_5[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[8], _tmem_load_32_21[9])));
                            gate_packed_5[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[10], _tmem_load_21[11])));
                            up_packed_5[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[10], _tmem_load_32_21[11])));
                            gate_packed_5[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[12], _tmem_load_21[13])));
                            up_packed_5[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[12], _tmem_load_32_21[13])));
                            gate_packed_5[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_21[14], _tmem_load_21[15])));
                            up_packed_5[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_21[14], _tmem_load_32_21[15])));
                            unsigned int address_3_22 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 96;
                            float _tmem_load_22[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[15]))
                                : "r"(address_3_22));
                            float _tmem_load_32_22[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_22[15]))
                                : "r"(address_3_22 + 128));
                            gate_packed_5[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[0], _tmem_load_22[1])));
                            up_packed_5[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[0], _tmem_load_32_22[1])));
                            gate_packed_5[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[2], _tmem_load_22[3])));
                            up_packed_5[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[2], _tmem_load_32_22[3])));
                            gate_packed_5[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[4], _tmem_load_22[5])));
                            up_packed_5[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[4], _tmem_load_32_22[5])));
                            gate_packed_5[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[6], _tmem_load_22[7])));
                            up_packed_5[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[6], _tmem_load_32_22[7])));
                            gate_packed_5[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[8], _tmem_load_22[9])));
                            up_packed_5[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[8], _tmem_load_32_22[9])));
                            gate_packed_5[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[10], _tmem_load_22[11])));
                            up_packed_5[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[10], _tmem_load_32_22[11])));
                            gate_packed_5[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[12], _tmem_load_22[13])));
                            up_packed_5[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[12], _tmem_load_32_22[13])));
                            gate_packed_5[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_22[14], _tmem_load_22[15])));
                            up_packed_5[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_22[14], _tmem_load_32_22[15])));
                            unsigned int address_3_23 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 96;
                            float _tmem_load_23[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[15]))
                                : "r"(address_3_23));
                            float _tmem_load_32_23[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_23[15]))
                                : "r"(address_3_23 + 128));
                            gate_packed_5[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[0], _tmem_load_23[1])));
                            up_packed_5[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[0], _tmem_load_32_23[1])));
                            gate_packed_5[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[2], _tmem_load_23[3])));
                            up_packed_5[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[2], _tmem_load_32_23[3])));
                            gate_packed_5[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[4], _tmem_load_23[5])));
                            up_packed_5[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[4], _tmem_load_32_23[5])));
                            gate_packed_5[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[6], _tmem_load_23[7])));
                            up_packed_5[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[6], _tmem_load_32_23[7])));
                            gate_packed_5[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[8], _tmem_load_23[9])));
                            up_packed_5[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[8], _tmem_load_32_23[9])));
                            gate_packed_5[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[10], _tmem_load_23[11])));
                            up_packed_5[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[10], _tmem_load_32_23[11])));
                            gate_packed_5[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[12], _tmem_load_23[13])));
                            up_packed_5[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[12], _tmem_load_32_23[13])));
                            gate_packed_5[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_23[14], _tmem_load_23[15])));
                            up_packed_5[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_23[14], _tmem_load_32_23[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            unsigned int hidden_packed_3[32];
                            float2 _cvt_f32_128 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[0]));
                            float2 _cvt_f32_256_128 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[0]));
                            float gate_x_1_128 = _cvt_f32_128.x;
                            float gate_y_1_128 = _cvt_f32_128.y;
                            float up_x_1_128 = _cvt_f32_256_128.x;
                            float up_y_1_128 = _cvt_f32_256_128.y;
                            float _exp_339 = expf(gate_x_1_128 * -1.0f);
                            float denominator_x_1_128 = _exp_339 + 1.0f;
                            float _exp_340 = expf(gate_y_1_128 * -1.0f);
                            float denominator_y_1_128 = _exp_340 + 1.0f;
                            float hidden_x_1_128 = gate_x_1_128 / denominator_x_1_128 * up_x_1_128;
                            float hidden_y_1_128 = gate_y_1_128 / denominator_y_1_128 * up_y_1_128;
                            hidden_packed_3[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_128, hidden_y_1_128)));
                            float2 _cvt_f32_129 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[1]));
                            float2 _cvt_f32_256_129 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[1]));
                            float gate_x_1_129 = _cvt_f32_129.x;
                            float gate_y_1_129 = _cvt_f32_129.y;
                            float up_x_1_129 = _cvt_f32_256_129.x;
                            float up_y_1_129 = _cvt_f32_256_129.y;
                            float _exp_341 = expf(gate_x_1_129 * -1.0f);
                            float denominator_x_1_129 = _exp_341 + 1.0f;
                            float _exp_342 = expf(gate_y_1_129 * -1.0f);
                            float denominator_y_1_129 = _exp_342 + 1.0f;
                            float hidden_x_1_129 = gate_x_1_129 / denominator_x_1_129 * up_x_1_129;
                            float hidden_y_1_129 = gate_y_1_129 / denominator_y_1_129 * up_y_1_129;
                            hidden_packed_3[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_129, hidden_y_1_129)));
                            float2 _cvt_f32_130 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[2]));
                            float2 _cvt_f32_256_130 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[2]));
                            float gate_x_1_130 = _cvt_f32_130.x;
                            float gate_y_1_130 = _cvt_f32_130.y;
                            float up_x_1_130 = _cvt_f32_256_130.x;
                            float up_y_1_130 = _cvt_f32_256_130.y;
                            float _exp_343 = expf(gate_x_1_130 * -1.0f);
                            float denominator_x_1_130 = _exp_343 + 1.0f;
                            float _exp_344 = expf(gate_y_1_130 * -1.0f);
                            float denominator_y_1_130 = _exp_344 + 1.0f;
                            float hidden_x_1_130 = gate_x_1_130 / denominator_x_1_130 * up_x_1_130;
                            float hidden_y_1_130 = gate_y_1_130 / denominator_y_1_130 * up_y_1_130;
                            hidden_packed_3[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_130, hidden_y_1_130)));
                            float2 _cvt_f32_131 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[3]));
                            float2 _cvt_f32_256_131 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[3]));
                            float gate_x_1_131 = _cvt_f32_131.x;
                            float gate_y_1_131 = _cvt_f32_131.y;
                            float up_x_1_131 = _cvt_f32_256_131.x;
                            float up_y_1_131 = _cvt_f32_256_131.y;
                            float _exp_345 = expf(gate_x_1_131 * -1.0f);
                            float denominator_x_1_131 = _exp_345 + 1.0f;
                            float _exp_346 = expf(gate_y_1_131 * -1.0f);
                            float denominator_y_1_131 = _exp_346 + 1.0f;
                            float hidden_x_1_131 = gate_x_1_131 / denominator_x_1_131 * up_x_1_131;
                            float hidden_y_1_131 = gate_y_1_131 / denominator_y_1_131 * up_y_1_131;
                            hidden_packed_3[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_131, hidden_y_1_131)));
                            float2 _cvt_f32_132 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[4]));
                            float2 _cvt_f32_256_132 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[4]));
                            float gate_x_1_132 = _cvt_f32_132.x;
                            float gate_y_1_132 = _cvt_f32_132.y;
                            float up_x_1_132 = _cvt_f32_256_132.x;
                            float up_y_1_132 = _cvt_f32_256_132.y;
                            float _exp_347 = expf(gate_x_1_132 * -1.0f);
                            float denominator_x_1_132 = _exp_347 + 1.0f;
                            float _exp_348 = expf(gate_y_1_132 * -1.0f);
                            float denominator_y_1_132 = _exp_348 + 1.0f;
                            float hidden_x_1_132 = gate_x_1_132 / denominator_x_1_132 * up_x_1_132;
                            float hidden_y_1_132 = gate_y_1_132 / denominator_y_1_132 * up_y_1_132;
                            hidden_packed_3[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_132, hidden_y_1_132)));
                            float2 _cvt_f32_133 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[5]));
                            float2 _cvt_f32_256_133 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[5]));
                            float gate_x_1_133 = _cvt_f32_133.x;
                            float gate_y_1_133 = _cvt_f32_133.y;
                            float up_x_1_133 = _cvt_f32_256_133.x;
                            float up_y_1_133 = _cvt_f32_256_133.y;
                            float _exp_349 = expf(gate_x_1_133 * -1.0f);
                            float denominator_x_1_133 = _exp_349 + 1.0f;
                            float _exp_350 = expf(gate_y_1_133 * -1.0f);
                            float denominator_y_1_133 = _exp_350 + 1.0f;
                            float hidden_x_1_133 = gate_x_1_133 / denominator_x_1_133 * up_x_1_133;
                            float hidden_y_1_133 = gate_y_1_133 / denominator_y_1_133 * up_y_1_133;
                            hidden_packed_3[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_133, hidden_y_1_133)));
                            float2 _cvt_f32_134 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[6]));
                            float2 _cvt_f32_256_134 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[6]));
                            float gate_x_1_134 = _cvt_f32_134.x;
                            float gate_y_1_134 = _cvt_f32_134.y;
                            float up_x_1_134 = _cvt_f32_256_134.x;
                            float up_y_1_134 = _cvt_f32_256_134.y;
                            float _exp_351 = expf(gate_x_1_134 * -1.0f);
                            float denominator_x_1_134 = _exp_351 + 1.0f;
                            float _exp_352 = expf(gate_y_1_134 * -1.0f);
                            float denominator_y_1_134 = _exp_352 + 1.0f;
                            float hidden_x_1_134 = gate_x_1_134 / denominator_x_1_134 * up_x_1_134;
                            float hidden_y_1_134 = gate_y_1_134 / denominator_y_1_134 * up_y_1_134;
                            hidden_packed_3[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_134, hidden_y_1_134)));
                            float2 _cvt_f32_135 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[7]));
                            float2 _cvt_f32_256_135 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[7]));
                            float gate_x_1_135 = _cvt_f32_135.x;
                            float gate_y_1_135 = _cvt_f32_135.y;
                            float up_x_1_135 = _cvt_f32_256_135.x;
                            float up_y_1_135 = _cvt_f32_256_135.y;
                            float _exp_353 = expf(gate_x_1_135 * -1.0f);
                            float denominator_x_1_135 = _exp_353 + 1.0f;
                            float _exp_354 = expf(gate_y_1_135 * -1.0f);
                            float denominator_y_1_135 = _exp_354 + 1.0f;
                            float hidden_x_1_135 = gate_x_1_135 / denominator_x_1_135 * up_x_1_135;
                            float hidden_y_1_135 = gate_y_1_135 / denominator_y_1_135 * up_y_1_135;
                            hidden_packed_3[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_135, hidden_y_1_135)));
                            float2 _cvt_f32_136 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[8]));
                            float2 _cvt_f32_256_136 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[8]));
                            float gate_x_1_136 = _cvt_f32_136.x;
                            float gate_y_1_136 = _cvt_f32_136.y;
                            float up_x_1_136 = _cvt_f32_256_136.x;
                            float up_y_1_136 = _cvt_f32_256_136.y;
                            float _exp_355 = expf(gate_x_1_136 * -1.0f);
                            float denominator_x_1_136 = _exp_355 + 1.0f;
                            float _exp_356 = expf(gate_y_1_136 * -1.0f);
                            float denominator_y_1_136 = _exp_356 + 1.0f;
                            float hidden_x_1_136 = gate_x_1_136 / denominator_x_1_136 * up_x_1_136;
                            float hidden_y_1_136 = gate_y_1_136 / denominator_y_1_136 * up_y_1_136;
                            hidden_packed_3[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_136, hidden_y_1_136)));
                            float2 _cvt_f32_137 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[9]));
                            float2 _cvt_f32_256_137 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[9]));
                            float gate_x_1_137 = _cvt_f32_137.x;
                            float gate_y_1_137 = _cvt_f32_137.y;
                            float up_x_1_137 = _cvt_f32_256_137.x;
                            float up_y_1_137 = _cvt_f32_256_137.y;
                            float _exp_357 = expf(gate_x_1_137 * -1.0f);
                            float denominator_x_1_137 = _exp_357 + 1.0f;
                            float _exp_358 = expf(gate_y_1_137 * -1.0f);
                            float denominator_y_1_137 = _exp_358 + 1.0f;
                            float hidden_x_1_137 = gate_x_1_137 / denominator_x_1_137 * up_x_1_137;
                            float hidden_y_1_137 = gate_y_1_137 / denominator_y_1_137 * up_y_1_137;
                            hidden_packed_3[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_137, hidden_y_1_137)));
                            float2 _cvt_f32_138 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[10]));
                            float2 _cvt_f32_256_138 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[10]));
                            float gate_x_1_138 = _cvt_f32_138.x;
                            float gate_y_1_138 = _cvt_f32_138.y;
                            float up_x_1_138 = _cvt_f32_256_138.x;
                            float up_y_1_138 = _cvt_f32_256_138.y;
                            float _exp_359 = expf(gate_x_1_138 * -1.0f);
                            float denominator_x_1_138 = _exp_359 + 1.0f;
                            float _exp_360 = expf(gate_y_1_138 * -1.0f);
                            float denominator_y_1_138 = _exp_360 + 1.0f;
                            float hidden_x_1_138 = gate_x_1_138 / denominator_x_1_138 * up_x_1_138;
                            float hidden_y_1_138 = gate_y_1_138 / denominator_y_1_138 * up_y_1_138;
                            hidden_packed_3[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_138, hidden_y_1_138)));
                            float2 _cvt_f32_139 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[11]));
                            float2 _cvt_f32_256_139 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[11]));
                            float gate_x_1_139 = _cvt_f32_139.x;
                            float gate_y_1_139 = _cvt_f32_139.y;
                            float up_x_1_139 = _cvt_f32_256_139.x;
                            float up_y_1_139 = _cvt_f32_256_139.y;
                            float _exp_361 = expf(gate_x_1_139 * -1.0f);
                            float denominator_x_1_139 = _exp_361 + 1.0f;
                            float _exp_362 = expf(gate_y_1_139 * -1.0f);
                            float denominator_y_1_139 = _exp_362 + 1.0f;
                            float hidden_x_1_139 = gate_x_1_139 / denominator_x_1_139 * up_x_1_139;
                            float hidden_y_1_139 = gate_y_1_139 / denominator_y_1_139 * up_y_1_139;
                            hidden_packed_3[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_139, hidden_y_1_139)));
                            float2 _cvt_f32_140 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[12]));
                            float2 _cvt_f32_256_140 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[12]));
                            float gate_x_1_140 = _cvt_f32_140.x;
                            float gate_y_1_140 = _cvt_f32_140.y;
                            float up_x_1_140 = _cvt_f32_256_140.x;
                            float up_y_1_140 = _cvt_f32_256_140.y;
                            float _exp_363 = expf(gate_x_1_140 * -1.0f);
                            float denominator_x_1_140 = _exp_363 + 1.0f;
                            float _exp_364 = expf(gate_y_1_140 * -1.0f);
                            float denominator_y_1_140 = _exp_364 + 1.0f;
                            float hidden_x_1_140 = gate_x_1_140 / denominator_x_1_140 * up_x_1_140;
                            float hidden_y_1_140 = gate_y_1_140 / denominator_y_1_140 * up_y_1_140;
                            hidden_packed_3[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_140, hidden_y_1_140)));
                            float2 _cvt_f32_141 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[13]));
                            float2 _cvt_f32_256_141 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[13]));
                            float gate_x_1_141 = _cvt_f32_141.x;
                            float gate_y_1_141 = _cvt_f32_141.y;
                            float up_x_1_141 = _cvt_f32_256_141.x;
                            float up_y_1_141 = _cvt_f32_256_141.y;
                            float _exp_365 = expf(gate_x_1_141 * -1.0f);
                            float denominator_x_1_141 = _exp_365 + 1.0f;
                            float _exp_366 = expf(gate_y_1_141 * -1.0f);
                            float denominator_y_1_141 = _exp_366 + 1.0f;
                            float hidden_x_1_141 = gate_x_1_141 / denominator_x_1_141 * up_x_1_141;
                            float hidden_y_1_141 = gate_y_1_141 / denominator_y_1_141 * up_y_1_141;
                            hidden_packed_3[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_141, hidden_y_1_141)));
                            float2 _cvt_f32_142 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[14]));
                            float2 _cvt_f32_256_142 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[14]));
                            float gate_x_1_142 = _cvt_f32_142.x;
                            float gate_y_1_142 = _cvt_f32_142.y;
                            float up_x_1_142 = _cvt_f32_256_142.x;
                            float up_y_1_142 = _cvt_f32_256_142.y;
                            float _exp_367 = expf(gate_x_1_142 * -1.0f);
                            float denominator_x_1_142 = _exp_367 + 1.0f;
                            float _exp_368 = expf(gate_y_1_142 * -1.0f);
                            float denominator_y_1_142 = _exp_368 + 1.0f;
                            float hidden_x_1_142 = gate_x_1_142 / denominator_x_1_142 * up_x_1_142;
                            float hidden_y_1_142 = gate_y_1_142 / denominator_y_1_142 * up_y_1_142;
                            hidden_packed_3[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_142, hidden_y_1_142)));
                            float2 _cvt_f32_143 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[15]));
                            float2 _cvt_f32_256_143 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[15]));
                            float gate_x_1_143 = _cvt_f32_143.x;
                            float gate_y_1_143 = _cvt_f32_143.y;
                            float up_x_1_143 = _cvt_f32_256_143.x;
                            float up_y_1_143 = _cvt_f32_256_143.y;
                            float _exp_369 = expf(gate_x_1_143 * -1.0f);
                            float denominator_x_1_143 = _exp_369 + 1.0f;
                            float _exp_370 = expf(gate_y_1_143 * -1.0f);
                            float denominator_y_1_143 = _exp_370 + 1.0f;
                            float hidden_x_1_143 = gate_x_1_143 / denominator_x_1_143 * up_x_1_143;
                            float hidden_y_1_143 = gate_y_1_143 / denominator_y_1_143 * up_y_1_143;
                            hidden_packed_3[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_143, hidden_y_1_143)));
                            float2 _cvt_f32_144 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[16]));
                            float2 _cvt_f32_256_144 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[16]));
                            float gate_x_1_144 = _cvt_f32_144.x;
                            float gate_y_1_144 = _cvt_f32_144.y;
                            float up_x_1_144 = _cvt_f32_256_144.x;
                            float up_y_1_144 = _cvt_f32_256_144.y;
                            float _exp_371 = expf(gate_x_1_144 * -1.0f);
                            float denominator_x_1_144 = _exp_371 + 1.0f;
                            float _exp_372 = expf(gate_y_1_144 * -1.0f);
                            float denominator_y_1_144 = _exp_372 + 1.0f;
                            float hidden_x_1_144 = gate_x_1_144 / denominator_x_1_144 * up_x_1_144;
                            float hidden_y_1_144 = gate_y_1_144 / denominator_y_1_144 * up_y_1_144;
                            hidden_packed_3[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_144, hidden_y_1_144)));
                            float2 _cvt_f32_145 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[17]));
                            float2 _cvt_f32_256_145 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[17]));
                            float gate_x_1_145 = _cvt_f32_145.x;
                            float gate_y_1_145 = _cvt_f32_145.y;
                            float up_x_1_145 = _cvt_f32_256_145.x;
                            float up_y_1_145 = _cvt_f32_256_145.y;
                            float _exp_373 = expf(gate_x_1_145 * -1.0f);
                            float denominator_x_1_145 = _exp_373 + 1.0f;
                            float _exp_374 = expf(gate_y_1_145 * -1.0f);
                            float denominator_y_1_145 = _exp_374 + 1.0f;
                            float hidden_x_1_145 = gate_x_1_145 / denominator_x_1_145 * up_x_1_145;
                            float hidden_y_1_145 = gate_y_1_145 / denominator_y_1_145 * up_y_1_145;
                            hidden_packed_3[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_145, hidden_y_1_145)));
                            float2 _cvt_f32_146 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[18]));
                            float2 _cvt_f32_256_146 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[18]));
                            float gate_x_1_146 = _cvt_f32_146.x;
                            float gate_y_1_146 = _cvt_f32_146.y;
                            float up_x_1_146 = _cvt_f32_256_146.x;
                            float up_y_1_146 = _cvt_f32_256_146.y;
                            float _exp_375 = expf(gate_x_1_146 * -1.0f);
                            float denominator_x_1_146 = _exp_375 + 1.0f;
                            float _exp_376 = expf(gate_y_1_146 * -1.0f);
                            float denominator_y_1_146 = _exp_376 + 1.0f;
                            float hidden_x_1_146 = gate_x_1_146 / denominator_x_1_146 * up_x_1_146;
                            float hidden_y_1_146 = gate_y_1_146 / denominator_y_1_146 * up_y_1_146;
                            hidden_packed_3[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_146, hidden_y_1_146)));
                            float2 _cvt_f32_147 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[19]));
                            float2 _cvt_f32_256_147 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[19]));
                            float gate_x_1_147 = _cvt_f32_147.x;
                            float gate_y_1_147 = _cvt_f32_147.y;
                            float up_x_1_147 = _cvt_f32_256_147.x;
                            float up_y_1_147 = _cvt_f32_256_147.y;
                            float _exp_377 = expf(gate_x_1_147 * -1.0f);
                            float denominator_x_1_147 = _exp_377 + 1.0f;
                            float _exp_378 = expf(gate_y_1_147 * -1.0f);
                            float denominator_y_1_147 = _exp_378 + 1.0f;
                            float hidden_x_1_147 = gate_x_1_147 / denominator_x_1_147 * up_x_1_147;
                            float hidden_y_1_147 = gate_y_1_147 / denominator_y_1_147 * up_y_1_147;
                            hidden_packed_3[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_147, hidden_y_1_147)));
                            float2 _cvt_f32_148 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[20]));
                            float2 _cvt_f32_256_148 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[20]));
                            float gate_x_1_148 = _cvt_f32_148.x;
                            float gate_y_1_148 = _cvt_f32_148.y;
                            float up_x_1_148 = _cvt_f32_256_148.x;
                            float up_y_1_148 = _cvt_f32_256_148.y;
                            float _exp_379 = expf(gate_x_1_148 * -1.0f);
                            float denominator_x_1_148 = _exp_379 + 1.0f;
                            float _exp_380 = expf(gate_y_1_148 * -1.0f);
                            float denominator_y_1_148 = _exp_380 + 1.0f;
                            float hidden_x_1_148 = gate_x_1_148 / denominator_x_1_148 * up_x_1_148;
                            float hidden_y_1_148 = gate_y_1_148 / denominator_y_1_148 * up_y_1_148;
                            hidden_packed_3[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_148, hidden_y_1_148)));
                            float2 _cvt_f32_149 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[21]));
                            float2 _cvt_f32_256_149 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[21]));
                            float gate_x_1_149 = _cvt_f32_149.x;
                            float gate_y_1_149 = _cvt_f32_149.y;
                            float up_x_1_149 = _cvt_f32_256_149.x;
                            float up_y_1_149 = _cvt_f32_256_149.y;
                            float _exp_381 = expf(gate_x_1_149 * -1.0f);
                            float denominator_x_1_149 = _exp_381 + 1.0f;
                            float _exp_382 = expf(gate_y_1_149 * -1.0f);
                            float denominator_y_1_149 = _exp_382 + 1.0f;
                            float hidden_x_1_149 = gate_x_1_149 / denominator_x_1_149 * up_x_1_149;
                            float hidden_y_1_149 = gate_y_1_149 / denominator_y_1_149 * up_y_1_149;
                            hidden_packed_3[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_149, hidden_y_1_149)));
                            float2 _cvt_f32_150 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[22]));
                            float2 _cvt_f32_256_150 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[22]));
                            float gate_x_1_150 = _cvt_f32_150.x;
                            float gate_y_1_150 = _cvt_f32_150.y;
                            float up_x_1_150 = _cvt_f32_256_150.x;
                            float up_y_1_150 = _cvt_f32_256_150.y;
                            float _exp_383 = expf(gate_x_1_150 * -1.0f);
                            float denominator_x_1_150 = _exp_383 + 1.0f;
                            float _exp_384 = expf(gate_y_1_150 * -1.0f);
                            float denominator_y_1_150 = _exp_384 + 1.0f;
                            float hidden_x_1_150 = gate_x_1_150 / denominator_x_1_150 * up_x_1_150;
                            float hidden_y_1_150 = gate_y_1_150 / denominator_y_1_150 * up_y_1_150;
                            hidden_packed_3[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_150, hidden_y_1_150)));
                            float2 _cvt_f32_151 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[23]));
                            float2 _cvt_f32_256_151 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[23]));
                            float gate_x_1_151 = _cvt_f32_151.x;
                            float gate_y_1_151 = _cvt_f32_151.y;
                            float up_x_1_151 = _cvt_f32_256_151.x;
                            float up_y_1_151 = _cvt_f32_256_151.y;
                            float _exp_385 = expf(gate_x_1_151 * -1.0f);
                            float denominator_x_1_151 = _exp_385 + 1.0f;
                            float _exp_386 = expf(gate_y_1_151 * -1.0f);
                            float denominator_y_1_151 = _exp_386 + 1.0f;
                            float hidden_x_1_151 = gate_x_1_151 / denominator_x_1_151 * up_x_1_151;
                            float hidden_y_1_151 = gate_y_1_151 / denominator_y_1_151 * up_y_1_151;
                            hidden_packed_3[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_151, hidden_y_1_151)));
                            float2 _cvt_f32_152 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[24]));
                            float2 _cvt_f32_256_152 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[24]));
                            float gate_x_1_152 = _cvt_f32_152.x;
                            float gate_y_1_152 = _cvt_f32_152.y;
                            float up_x_1_152 = _cvt_f32_256_152.x;
                            float up_y_1_152 = _cvt_f32_256_152.y;
                            float _exp_387 = expf(gate_x_1_152 * -1.0f);
                            float denominator_x_1_152 = _exp_387 + 1.0f;
                            float _exp_388 = expf(gate_y_1_152 * -1.0f);
                            float denominator_y_1_152 = _exp_388 + 1.0f;
                            float hidden_x_1_152 = gate_x_1_152 / denominator_x_1_152 * up_x_1_152;
                            float hidden_y_1_152 = gate_y_1_152 / denominator_y_1_152 * up_y_1_152;
                            hidden_packed_3[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_152, hidden_y_1_152)));
                            float2 _cvt_f32_153 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[25]));
                            float2 _cvt_f32_256_153 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[25]));
                            float gate_x_1_153 = _cvt_f32_153.x;
                            float gate_y_1_153 = _cvt_f32_153.y;
                            float up_x_1_153 = _cvt_f32_256_153.x;
                            float up_y_1_153 = _cvt_f32_256_153.y;
                            float _exp_389 = expf(gate_x_1_153 * -1.0f);
                            float denominator_x_1_153 = _exp_389 + 1.0f;
                            float _exp_390 = expf(gate_y_1_153 * -1.0f);
                            float denominator_y_1_153 = _exp_390 + 1.0f;
                            float hidden_x_1_153 = gate_x_1_153 / denominator_x_1_153 * up_x_1_153;
                            float hidden_y_1_153 = gate_y_1_153 / denominator_y_1_153 * up_y_1_153;
                            hidden_packed_3[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_153, hidden_y_1_153)));
                            float2 _cvt_f32_154 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[26]));
                            float2 _cvt_f32_256_154 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[26]));
                            float gate_x_1_154 = _cvt_f32_154.x;
                            float gate_y_1_154 = _cvt_f32_154.y;
                            float up_x_1_154 = _cvt_f32_256_154.x;
                            float up_y_1_154 = _cvt_f32_256_154.y;
                            float _exp_391 = expf(gate_x_1_154 * -1.0f);
                            float denominator_x_1_154 = _exp_391 + 1.0f;
                            float _exp_392 = expf(gate_y_1_154 * -1.0f);
                            float denominator_y_1_154 = _exp_392 + 1.0f;
                            float hidden_x_1_154 = gate_x_1_154 / denominator_x_1_154 * up_x_1_154;
                            float hidden_y_1_154 = gate_y_1_154 / denominator_y_1_154 * up_y_1_154;
                            hidden_packed_3[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_154, hidden_y_1_154)));
                            float2 _cvt_f32_155 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[27]));
                            float2 _cvt_f32_256_155 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[27]));
                            float gate_x_1_155 = _cvt_f32_155.x;
                            float gate_y_1_155 = _cvt_f32_155.y;
                            float up_x_1_155 = _cvt_f32_256_155.x;
                            float up_y_1_155 = _cvt_f32_256_155.y;
                            float _exp_393 = expf(gate_x_1_155 * -1.0f);
                            float denominator_x_1_155 = _exp_393 + 1.0f;
                            float _exp_394 = expf(gate_y_1_155 * -1.0f);
                            float denominator_y_1_155 = _exp_394 + 1.0f;
                            float hidden_x_1_155 = gate_x_1_155 / denominator_x_1_155 * up_x_1_155;
                            float hidden_y_1_155 = gate_y_1_155 / denominator_y_1_155 * up_y_1_155;
                            hidden_packed_3[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_155, hidden_y_1_155)));
                            float2 _cvt_f32_156 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[28]));
                            float2 _cvt_f32_256_156 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[28]));
                            float gate_x_1_156 = _cvt_f32_156.x;
                            float gate_y_1_156 = _cvt_f32_156.y;
                            float up_x_1_156 = _cvt_f32_256_156.x;
                            float up_y_1_156 = _cvt_f32_256_156.y;
                            float _exp_395 = expf(gate_x_1_156 * -1.0f);
                            float denominator_x_1_156 = _exp_395 + 1.0f;
                            float _exp_396 = expf(gate_y_1_156 * -1.0f);
                            float denominator_y_1_156 = _exp_396 + 1.0f;
                            float hidden_x_1_156 = gate_x_1_156 / denominator_x_1_156 * up_x_1_156;
                            float hidden_y_1_156 = gate_y_1_156 / denominator_y_1_156 * up_y_1_156;
                            hidden_packed_3[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_156, hidden_y_1_156)));
                            float2 _cvt_f32_157 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[29]));
                            float2 _cvt_f32_256_157 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[29]));
                            float gate_x_1_157 = _cvt_f32_157.x;
                            float gate_y_1_157 = _cvt_f32_157.y;
                            float up_x_1_157 = _cvt_f32_256_157.x;
                            float up_y_1_157 = _cvt_f32_256_157.y;
                            float _exp_397 = expf(gate_x_1_157 * -1.0f);
                            float denominator_x_1_157 = _exp_397 + 1.0f;
                            float _exp_398 = expf(gate_y_1_157 * -1.0f);
                            float denominator_y_1_157 = _exp_398 + 1.0f;
                            float hidden_x_1_157 = gate_x_1_157 / denominator_x_1_157 * up_x_1_157;
                            float hidden_y_1_157 = gate_y_1_157 / denominator_y_1_157 * up_y_1_157;
                            hidden_packed_3[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_157, hidden_y_1_157)));
                            float2 _cvt_f32_158 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[30]));
                            float2 _cvt_f32_256_158 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[30]));
                            float gate_x_1_158 = _cvt_f32_158.x;
                            float gate_y_1_158 = _cvt_f32_158.y;
                            float up_x_1_158 = _cvt_f32_256_158.x;
                            float up_y_1_158 = _cvt_f32_256_158.y;
                            float _exp_399 = expf(gate_x_1_158 * -1.0f);
                            float denominator_x_1_158 = _exp_399 + 1.0f;
                            float _exp_400 = expf(gate_y_1_158 * -1.0f);
                            float denominator_y_1_158 = _exp_400 + 1.0f;
                            float hidden_x_1_158 = gate_x_1_158 / denominator_x_1_158 * up_x_1_158;
                            float hidden_y_1_158 = gate_y_1_158 / denominator_y_1_158 * up_y_1_158;
                            hidden_packed_3[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_158, hidden_y_1_158)));
                            float2 _cvt_f32_159 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_4[31]));
                            float2 _cvt_f32_256_159 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_4[31]));
                            float gate_x_1_159 = _cvt_f32_159.x;
                            float gate_y_1_159 = _cvt_f32_159.y;
                            float up_x_1_159 = _cvt_f32_256_159.x;
                            float up_y_1_159 = _cvt_f32_256_159.y;
                            float _exp_401 = expf(gate_x_1_159 * -1.0f);
                            float denominator_x_1_159 = _exp_401 + 1.0f;
                            float _exp_402 = expf(gate_y_1_159 * -1.0f);
                            float denominator_y_1_159 = _exp_402 + 1.0f;
                            float hidden_x_1_159 = gate_x_1_159 / denominator_x_1_159 * up_x_1_159;
                            float hidden_y_1_159 = gate_y_1_159 / denominator_y_1_159 * up_y_1_159;
                            hidden_packed_3[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_159, hidden_y_1_159)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_9 = tid / 32;
                            int lane_9 = tid % 32;
                            #pragma unroll
                            for (int half_10 = 0; half_10 < 2; half_10++) {
                                #pragma unroll
                                for (int col_tile_10 = 0; col_tile_10 < 2; col_tile_10++) {
                                    int row_16 = warp_9 * 32 + half_10 * 16 + lane_9 % 16;
                                    int col_13 = col_tile_10 * 16 + lane_9 / 16 * 8;
                                    unsigned int address_12 = d_smem_addr + (unsigned int)((row_16 * 32 + col_13) * 2);
                                    address_12 = address_12 ^ (address_12 & 511) >> 7 << 4;
                                    int offset_11 = half_10 * 8 + col_tile_10 * 4;
                                    uint32_t _stmatrix_addr_403 = static_cast<uint32_t>(address_12);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_403), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_11])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_11 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_11 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_11 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_7_1 = tid / 32;
                            int lane_7_1 = tid % 32;
                            #pragma unroll
                            for (int half_11 = 0; half_11 < 2; half_11++) {
                                #pragma unroll
                                for (int col_tile_11 = 0; col_tile_11 < 2; col_tile_11++) {
                                    int row_17 = warp_7_1 * 32 + half_11 * 16 + lane_7_1 % 16;
                                    int col_14 = col_tile_11 * 16 + lane_7_1 / 16 * 8;
                                    unsigned int address_13 = d_smem_addr + 8192 + (unsigned int)((row_17 * 32 + col_14) * 2);
                                    address_13 = address_13 ^ (address_13 & 511) >> 7 << 4;
                                    int offset_12 = 16 + half_11 * 8 + col_tile_11 * 4;
                                    uint32_t _stmatrix_addr_404 = static_cast<uint32_t>(address_13);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_404), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_12])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_12 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_12 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3[offset_12 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            unsigned int hidden_packed_3_1[32];
                            float2 _cvt_f32_160 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[0]));
                            float2 _cvt_f32_256_160 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[0]));
                            float gate_x_1_160 = _cvt_f32_160.x;
                            float gate_y_1_160 = _cvt_f32_160.y;
                            float up_x_1_160 = _cvt_f32_256_160.x;
                            float up_y_1_160 = _cvt_f32_256_160.y;
                            float _exp_405 = expf(gate_x_1_160 * -1.0f);
                            float denominator_x_1_160 = _exp_405 + 1.0f;
                            float _exp_406 = expf(gate_y_1_160 * -1.0f);
                            float denominator_y_1_160 = _exp_406 + 1.0f;
                            float hidden_x_1_160 = gate_x_1_160 / denominator_x_1_160 * up_x_1_160;
                            float hidden_y_1_160 = gate_y_1_160 / denominator_y_1_160 * up_y_1_160;
                            hidden_packed_3_1[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_160, hidden_y_1_160)));
                            float2 _cvt_f32_161 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[1]));
                            float2 _cvt_f32_256_161 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[1]));
                            float gate_x_1_161 = _cvt_f32_161.x;
                            float gate_y_1_161 = _cvt_f32_161.y;
                            float up_x_1_161 = _cvt_f32_256_161.x;
                            float up_y_1_161 = _cvt_f32_256_161.y;
                            float _exp_407 = expf(gate_x_1_161 * -1.0f);
                            float denominator_x_1_161 = _exp_407 + 1.0f;
                            float _exp_408 = expf(gate_y_1_161 * -1.0f);
                            float denominator_y_1_161 = _exp_408 + 1.0f;
                            float hidden_x_1_161 = gate_x_1_161 / denominator_x_1_161 * up_x_1_161;
                            float hidden_y_1_161 = gate_y_1_161 / denominator_y_1_161 * up_y_1_161;
                            hidden_packed_3_1[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_161, hidden_y_1_161)));
                            float2 _cvt_f32_162 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[2]));
                            float2 _cvt_f32_256_162 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[2]));
                            float gate_x_1_162 = _cvt_f32_162.x;
                            float gate_y_1_162 = _cvt_f32_162.y;
                            float up_x_1_162 = _cvt_f32_256_162.x;
                            float up_y_1_162 = _cvt_f32_256_162.y;
                            float _exp_409 = expf(gate_x_1_162 * -1.0f);
                            float denominator_x_1_162 = _exp_409 + 1.0f;
                            float _exp_410 = expf(gate_y_1_162 * -1.0f);
                            float denominator_y_1_162 = _exp_410 + 1.0f;
                            float hidden_x_1_162 = gate_x_1_162 / denominator_x_1_162 * up_x_1_162;
                            float hidden_y_1_162 = gate_y_1_162 / denominator_y_1_162 * up_y_1_162;
                            hidden_packed_3_1[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_162, hidden_y_1_162)));
                            float2 _cvt_f32_163 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[3]));
                            float2 _cvt_f32_256_163 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[3]));
                            float gate_x_1_163 = _cvt_f32_163.x;
                            float gate_y_1_163 = _cvt_f32_163.y;
                            float up_x_1_163 = _cvt_f32_256_163.x;
                            float up_y_1_163 = _cvt_f32_256_163.y;
                            float _exp_411 = expf(gate_x_1_163 * -1.0f);
                            float denominator_x_1_163 = _exp_411 + 1.0f;
                            float _exp_412 = expf(gate_y_1_163 * -1.0f);
                            float denominator_y_1_163 = _exp_412 + 1.0f;
                            float hidden_x_1_163 = gate_x_1_163 / denominator_x_1_163 * up_x_1_163;
                            float hidden_y_1_163 = gate_y_1_163 / denominator_y_1_163 * up_y_1_163;
                            hidden_packed_3_1[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_163, hidden_y_1_163)));
                            float2 _cvt_f32_164 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[4]));
                            float2 _cvt_f32_256_164 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[4]));
                            float gate_x_1_164 = _cvt_f32_164.x;
                            float gate_y_1_164 = _cvt_f32_164.y;
                            float up_x_1_164 = _cvt_f32_256_164.x;
                            float up_y_1_164 = _cvt_f32_256_164.y;
                            float _exp_413 = expf(gate_x_1_164 * -1.0f);
                            float denominator_x_1_164 = _exp_413 + 1.0f;
                            float _exp_414 = expf(gate_y_1_164 * -1.0f);
                            float denominator_y_1_164 = _exp_414 + 1.0f;
                            float hidden_x_1_164 = gate_x_1_164 / denominator_x_1_164 * up_x_1_164;
                            float hidden_y_1_164 = gate_y_1_164 / denominator_y_1_164 * up_y_1_164;
                            hidden_packed_3_1[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_164, hidden_y_1_164)));
                            float2 _cvt_f32_165 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[5]));
                            float2 _cvt_f32_256_165 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[5]));
                            float gate_x_1_165 = _cvt_f32_165.x;
                            float gate_y_1_165 = _cvt_f32_165.y;
                            float up_x_1_165 = _cvt_f32_256_165.x;
                            float up_y_1_165 = _cvt_f32_256_165.y;
                            float _exp_415 = expf(gate_x_1_165 * -1.0f);
                            float denominator_x_1_165 = _exp_415 + 1.0f;
                            float _exp_416 = expf(gate_y_1_165 * -1.0f);
                            float denominator_y_1_165 = _exp_416 + 1.0f;
                            float hidden_x_1_165 = gate_x_1_165 / denominator_x_1_165 * up_x_1_165;
                            float hidden_y_1_165 = gate_y_1_165 / denominator_y_1_165 * up_y_1_165;
                            hidden_packed_3_1[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_165, hidden_y_1_165)));
                            float2 _cvt_f32_166 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[6]));
                            float2 _cvt_f32_256_166 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[6]));
                            float gate_x_1_166 = _cvt_f32_166.x;
                            float gate_y_1_166 = _cvt_f32_166.y;
                            float up_x_1_166 = _cvt_f32_256_166.x;
                            float up_y_1_166 = _cvt_f32_256_166.y;
                            float _exp_417 = expf(gate_x_1_166 * -1.0f);
                            float denominator_x_1_166 = _exp_417 + 1.0f;
                            float _exp_418 = expf(gate_y_1_166 * -1.0f);
                            float denominator_y_1_166 = _exp_418 + 1.0f;
                            float hidden_x_1_166 = gate_x_1_166 / denominator_x_1_166 * up_x_1_166;
                            float hidden_y_1_166 = gate_y_1_166 / denominator_y_1_166 * up_y_1_166;
                            hidden_packed_3_1[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_166, hidden_y_1_166)));
                            float2 _cvt_f32_167 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[7]));
                            float2 _cvt_f32_256_167 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[7]));
                            float gate_x_1_167 = _cvt_f32_167.x;
                            float gate_y_1_167 = _cvt_f32_167.y;
                            float up_x_1_167 = _cvt_f32_256_167.x;
                            float up_y_1_167 = _cvt_f32_256_167.y;
                            float _exp_419 = expf(gate_x_1_167 * -1.0f);
                            float denominator_x_1_167 = _exp_419 + 1.0f;
                            float _exp_420 = expf(gate_y_1_167 * -1.0f);
                            float denominator_y_1_167 = _exp_420 + 1.0f;
                            float hidden_x_1_167 = gate_x_1_167 / denominator_x_1_167 * up_x_1_167;
                            float hidden_y_1_167 = gate_y_1_167 / denominator_y_1_167 * up_y_1_167;
                            hidden_packed_3_1[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_167, hidden_y_1_167)));
                            float2 _cvt_f32_168 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[8]));
                            float2 _cvt_f32_256_168 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[8]));
                            float gate_x_1_168 = _cvt_f32_168.x;
                            float gate_y_1_168 = _cvt_f32_168.y;
                            float up_x_1_168 = _cvt_f32_256_168.x;
                            float up_y_1_168 = _cvt_f32_256_168.y;
                            float _exp_421 = expf(gate_x_1_168 * -1.0f);
                            float denominator_x_1_168 = _exp_421 + 1.0f;
                            float _exp_422 = expf(gate_y_1_168 * -1.0f);
                            float denominator_y_1_168 = _exp_422 + 1.0f;
                            float hidden_x_1_168 = gate_x_1_168 / denominator_x_1_168 * up_x_1_168;
                            float hidden_y_1_168 = gate_y_1_168 / denominator_y_1_168 * up_y_1_168;
                            hidden_packed_3_1[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_168, hidden_y_1_168)));
                            float2 _cvt_f32_169 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[9]));
                            float2 _cvt_f32_256_169 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[9]));
                            float gate_x_1_169 = _cvt_f32_169.x;
                            float gate_y_1_169 = _cvt_f32_169.y;
                            float up_x_1_169 = _cvt_f32_256_169.x;
                            float up_y_1_169 = _cvt_f32_256_169.y;
                            float _exp_423 = expf(gate_x_1_169 * -1.0f);
                            float denominator_x_1_169 = _exp_423 + 1.0f;
                            float _exp_424 = expf(gate_y_1_169 * -1.0f);
                            float denominator_y_1_169 = _exp_424 + 1.0f;
                            float hidden_x_1_169 = gate_x_1_169 / denominator_x_1_169 * up_x_1_169;
                            float hidden_y_1_169 = gate_y_1_169 / denominator_y_1_169 * up_y_1_169;
                            hidden_packed_3_1[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_169, hidden_y_1_169)));
                            float2 _cvt_f32_170 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[10]));
                            float2 _cvt_f32_256_170 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[10]));
                            float gate_x_1_170 = _cvt_f32_170.x;
                            float gate_y_1_170 = _cvt_f32_170.y;
                            float up_x_1_170 = _cvt_f32_256_170.x;
                            float up_y_1_170 = _cvt_f32_256_170.y;
                            float _exp_425 = expf(gate_x_1_170 * -1.0f);
                            float denominator_x_1_170 = _exp_425 + 1.0f;
                            float _exp_426 = expf(gate_y_1_170 * -1.0f);
                            float denominator_y_1_170 = _exp_426 + 1.0f;
                            float hidden_x_1_170 = gate_x_1_170 / denominator_x_1_170 * up_x_1_170;
                            float hidden_y_1_170 = gate_y_1_170 / denominator_y_1_170 * up_y_1_170;
                            hidden_packed_3_1[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_170, hidden_y_1_170)));
                            float2 _cvt_f32_171 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[11]));
                            float2 _cvt_f32_256_171 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[11]));
                            float gate_x_1_171 = _cvt_f32_171.x;
                            float gate_y_1_171 = _cvt_f32_171.y;
                            float up_x_1_171 = _cvt_f32_256_171.x;
                            float up_y_1_171 = _cvt_f32_256_171.y;
                            float _exp_427 = expf(gate_x_1_171 * -1.0f);
                            float denominator_x_1_171 = _exp_427 + 1.0f;
                            float _exp_428 = expf(gate_y_1_171 * -1.0f);
                            float denominator_y_1_171 = _exp_428 + 1.0f;
                            float hidden_x_1_171 = gate_x_1_171 / denominator_x_1_171 * up_x_1_171;
                            float hidden_y_1_171 = gate_y_1_171 / denominator_y_1_171 * up_y_1_171;
                            hidden_packed_3_1[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_171, hidden_y_1_171)));
                            float2 _cvt_f32_172 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[12]));
                            float2 _cvt_f32_256_172 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[12]));
                            float gate_x_1_172 = _cvt_f32_172.x;
                            float gate_y_1_172 = _cvt_f32_172.y;
                            float up_x_1_172 = _cvt_f32_256_172.x;
                            float up_y_1_172 = _cvt_f32_256_172.y;
                            float _exp_429 = expf(gate_x_1_172 * -1.0f);
                            float denominator_x_1_172 = _exp_429 + 1.0f;
                            float _exp_430 = expf(gate_y_1_172 * -1.0f);
                            float denominator_y_1_172 = _exp_430 + 1.0f;
                            float hidden_x_1_172 = gate_x_1_172 / denominator_x_1_172 * up_x_1_172;
                            float hidden_y_1_172 = gate_y_1_172 / denominator_y_1_172 * up_y_1_172;
                            hidden_packed_3_1[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_172, hidden_y_1_172)));
                            float2 _cvt_f32_173 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[13]));
                            float2 _cvt_f32_256_173 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[13]));
                            float gate_x_1_173 = _cvt_f32_173.x;
                            float gate_y_1_173 = _cvt_f32_173.y;
                            float up_x_1_173 = _cvt_f32_256_173.x;
                            float up_y_1_173 = _cvt_f32_256_173.y;
                            float _exp_431 = expf(gate_x_1_173 * -1.0f);
                            float denominator_x_1_173 = _exp_431 + 1.0f;
                            float _exp_432 = expf(gate_y_1_173 * -1.0f);
                            float denominator_y_1_173 = _exp_432 + 1.0f;
                            float hidden_x_1_173 = gate_x_1_173 / denominator_x_1_173 * up_x_1_173;
                            float hidden_y_1_173 = gate_y_1_173 / denominator_y_1_173 * up_y_1_173;
                            hidden_packed_3_1[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_173, hidden_y_1_173)));
                            float2 _cvt_f32_174 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[14]));
                            float2 _cvt_f32_256_174 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[14]));
                            float gate_x_1_174 = _cvt_f32_174.x;
                            float gate_y_1_174 = _cvt_f32_174.y;
                            float up_x_1_174 = _cvt_f32_256_174.x;
                            float up_y_1_174 = _cvt_f32_256_174.y;
                            float _exp_433 = expf(gate_x_1_174 * -1.0f);
                            float denominator_x_1_174 = _exp_433 + 1.0f;
                            float _exp_434 = expf(gate_y_1_174 * -1.0f);
                            float denominator_y_1_174 = _exp_434 + 1.0f;
                            float hidden_x_1_174 = gate_x_1_174 / denominator_x_1_174 * up_x_1_174;
                            float hidden_y_1_174 = gate_y_1_174 / denominator_y_1_174 * up_y_1_174;
                            hidden_packed_3_1[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_174, hidden_y_1_174)));
                            float2 _cvt_f32_175 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[15]));
                            float2 _cvt_f32_256_175 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[15]));
                            float gate_x_1_175 = _cvt_f32_175.x;
                            float gate_y_1_175 = _cvt_f32_175.y;
                            float up_x_1_175 = _cvt_f32_256_175.x;
                            float up_y_1_175 = _cvt_f32_256_175.y;
                            float _exp_435 = expf(gate_x_1_175 * -1.0f);
                            float denominator_x_1_175 = _exp_435 + 1.0f;
                            float _exp_436 = expf(gate_y_1_175 * -1.0f);
                            float denominator_y_1_175 = _exp_436 + 1.0f;
                            float hidden_x_1_175 = gate_x_1_175 / denominator_x_1_175 * up_x_1_175;
                            float hidden_y_1_175 = gate_y_1_175 / denominator_y_1_175 * up_y_1_175;
                            hidden_packed_3_1[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_175, hidden_y_1_175)));
                            float2 _cvt_f32_176 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[16]));
                            float2 _cvt_f32_256_176 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[16]));
                            float gate_x_1_176 = _cvt_f32_176.x;
                            float gate_y_1_176 = _cvt_f32_176.y;
                            float up_x_1_176 = _cvt_f32_256_176.x;
                            float up_y_1_176 = _cvt_f32_256_176.y;
                            float _exp_437 = expf(gate_x_1_176 * -1.0f);
                            float denominator_x_1_176 = _exp_437 + 1.0f;
                            float _exp_438 = expf(gate_y_1_176 * -1.0f);
                            float denominator_y_1_176 = _exp_438 + 1.0f;
                            float hidden_x_1_176 = gate_x_1_176 / denominator_x_1_176 * up_x_1_176;
                            float hidden_y_1_176 = gate_y_1_176 / denominator_y_1_176 * up_y_1_176;
                            hidden_packed_3_1[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_176, hidden_y_1_176)));
                            float2 _cvt_f32_177 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[17]));
                            float2 _cvt_f32_256_177 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[17]));
                            float gate_x_1_177 = _cvt_f32_177.x;
                            float gate_y_1_177 = _cvt_f32_177.y;
                            float up_x_1_177 = _cvt_f32_256_177.x;
                            float up_y_1_177 = _cvt_f32_256_177.y;
                            float _exp_439 = expf(gate_x_1_177 * -1.0f);
                            float denominator_x_1_177 = _exp_439 + 1.0f;
                            float _exp_440 = expf(gate_y_1_177 * -1.0f);
                            float denominator_y_1_177 = _exp_440 + 1.0f;
                            float hidden_x_1_177 = gate_x_1_177 / denominator_x_1_177 * up_x_1_177;
                            float hidden_y_1_177 = gate_y_1_177 / denominator_y_1_177 * up_y_1_177;
                            hidden_packed_3_1[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_177, hidden_y_1_177)));
                            float2 _cvt_f32_178 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[18]));
                            float2 _cvt_f32_256_178 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[18]));
                            float gate_x_1_178 = _cvt_f32_178.x;
                            float gate_y_1_178 = _cvt_f32_178.y;
                            float up_x_1_178 = _cvt_f32_256_178.x;
                            float up_y_1_178 = _cvt_f32_256_178.y;
                            float _exp_441 = expf(gate_x_1_178 * -1.0f);
                            float denominator_x_1_178 = _exp_441 + 1.0f;
                            float _exp_442 = expf(gate_y_1_178 * -1.0f);
                            float denominator_y_1_178 = _exp_442 + 1.0f;
                            float hidden_x_1_178 = gate_x_1_178 / denominator_x_1_178 * up_x_1_178;
                            float hidden_y_1_178 = gate_y_1_178 / denominator_y_1_178 * up_y_1_178;
                            hidden_packed_3_1[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_178, hidden_y_1_178)));
                            float2 _cvt_f32_179 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[19]));
                            float2 _cvt_f32_256_179 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[19]));
                            float gate_x_1_179 = _cvt_f32_179.x;
                            float gate_y_1_179 = _cvt_f32_179.y;
                            float up_x_1_179 = _cvt_f32_256_179.x;
                            float up_y_1_179 = _cvt_f32_256_179.y;
                            float _exp_443 = expf(gate_x_1_179 * -1.0f);
                            float denominator_x_1_179 = _exp_443 + 1.0f;
                            float _exp_444 = expf(gate_y_1_179 * -1.0f);
                            float denominator_y_1_179 = _exp_444 + 1.0f;
                            float hidden_x_1_179 = gate_x_1_179 / denominator_x_1_179 * up_x_1_179;
                            float hidden_y_1_179 = gate_y_1_179 / denominator_y_1_179 * up_y_1_179;
                            hidden_packed_3_1[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_179, hidden_y_1_179)));
                            float2 _cvt_f32_180 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[20]));
                            float2 _cvt_f32_256_180 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[20]));
                            float gate_x_1_180 = _cvt_f32_180.x;
                            float gate_y_1_180 = _cvt_f32_180.y;
                            float up_x_1_180 = _cvt_f32_256_180.x;
                            float up_y_1_180 = _cvt_f32_256_180.y;
                            float _exp_445 = expf(gate_x_1_180 * -1.0f);
                            float denominator_x_1_180 = _exp_445 + 1.0f;
                            float _exp_446 = expf(gate_y_1_180 * -1.0f);
                            float denominator_y_1_180 = _exp_446 + 1.0f;
                            float hidden_x_1_180 = gate_x_1_180 / denominator_x_1_180 * up_x_1_180;
                            float hidden_y_1_180 = gate_y_1_180 / denominator_y_1_180 * up_y_1_180;
                            hidden_packed_3_1[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_180, hidden_y_1_180)));
                            float2 _cvt_f32_181 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[21]));
                            float2 _cvt_f32_256_181 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[21]));
                            float gate_x_1_181 = _cvt_f32_181.x;
                            float gate_y_1_181 = _cvt_f32_181.y;
                            float up_x_1_181 = _cvt_f32_256_181.x;
                            float up_y_1_181 = _cvt_f32_256_181.y;
                            float _exp_447 = expf(gate_x_1_181 * -1.0f);
                            float denominator_x_1_181 = _exp_447 + 1.0f;
                            float _exp_448 = expf(gate_y_1_181 * -1.0f);
                            float denominator_y_1_181 = _exp_448 + 1.0f;
                            float hidden_x_1_181 = gate_x_1_181 / denominator_x_1_181 * up_x_1_181;
                            float hidden_y_1_181 = gate_y_1_181 / denominator_y_1_181 * up_y_1_181;
                            hidden_packed_3_1[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_181, hidden_y_1_181)));
                            float2 _cvt_f32_182 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[22]));
                            float2 _cvt_f32_256_182 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[22]));
                            float gate_x_1_182 = _cvt_f32_182.x;
                            float gate_y_1_182 = _cvt_f32_182.y;
                            float up_x_1_182 = _cvt_f32_256_182.x;
                            float up_y_1_182 = _cvt_f32_256_182.y;
                            float _exp_449 = expf(gate_x_1_182 * -1.0f);
                            float denominator_x_1_182 = _exp_449 + 1.0f;
                            float _exp_450 = expf(gate_y_1_182 * -1.0f);
                            float denominator_y_1_182 = _exp_450 + 1.0f;
                            float hidden_x_1_182 = gate_x_1_182 / denominator_x_1_182 * up_x_1_182;
                            float hidden_y_1_182 = gate_y_1_182 / denominator_y_1_182 * up_y_1_182;
                            hidden_packed_3_1[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_182, hidden_y_1_182)));
                            float2 _cvt_f32_183 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[23]));
                            float2 _cvt_f32_256_183 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[23]));
                            float gate_x_1_183 = _cvt_f32_183.x;
                            float gate_y_1_183 = _cvt_f32_183.y;
                            float up_x_1_183 = _cvt_f32_256_183.x;
                            float up_y_1_183 = _cvt_f32_256_183.y;
                            float _exp_451 = expf(gate_x_1_183 * -1.0f);
                            float denominator_x_1_183 = _exp_451 + 1.0f;
                            float _exp_452 = expf(gate_y_1_183 * -1.0f);
                            float denominator_y_1_183 = _exp_452 + 1.0f;
                            float hidden_x_1_183 = gate_x_1_183 / denominator_x_1_183 * up_x_1_183;
                            float hidden_y_1_183 = gate_y_1_183 / denominator_y_1_183 * up_y_1_183;
                            hidden_packed_3_1[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_183, hidden_y_1_183)));
                            float2 _cvt_f32_184 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[24]));
                            float2 _cvt_f32_256_184 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[24]));
                            float gate_x_1_184 = _cvt_f32_184.x;
                            float gate_y_1_184 = _cvt_f32_184.y;
                            float up_x_1_184 = _cvt_f32_256_184.x;
                            float up_y_1_184 = _cvt_f32_256_184.y;
                            float _exp_453 = expf(gate_x_1_184 * -1.0f);
                            float denominator_x_1_184 = _exp_453 + 1.0f;
                            float _exp_454 = expf(gate_y_1_184 * -1.0f);
                            float denominator_y_1_184 = _exp_454 + 1.0f;
                            float hidden_x_1_184 = gate_x_1_184 / denominator_x_1_184 * up_x_1_184;
                            float hidden_y_1_184 = gate_y_1_184 / denominator_y_1_184 * up_y_1_184;
                            hidden_packed_3_1[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_184, hidden_y_1_184)));
                            float2 _cvt_f32_185 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[25]));
                            float2 _cvt_f32_256_185 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[25]));
                            float gate_x_1_185 = _cvt_f32_185.x;
                            float gate_y_1_185 = _cvt_f32_185.y;
                            float up_x_1_185 = _cvt_f32_256_185.x;
                            float up_y_1_185 = _cvt_f32_256_185.y;
                            float _exp_455 = expf(gate_x_1_185 * -1.0f);
                            float denominator_x_1_185 = _exp_455 + 1.0f;
                            float _exp_456 = expf(gate_y_1_185 * -1.0f);
                            float denominator_y_1_185 = _exp_456 + 1.0f;
                            float hidden_x_1_185 = gate_x_1_185 / denominator_x_1_185 * up_x_1_185;
                            float hidden_y_1_185 = gate_y_1_185 / denominator_y_1_185 * up_y_1_185;
                            hidden_packed_3_1[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_185, hidden_y_1_185)));
                            float2 _cvt_f32_186 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[26]));
                            float2 _cvt_f32_256_186 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[26]));
                            float gate_x_1_186 = _cvt_f32_186.x;
                            float gate_y_1_186 = _cvt_f32_186.y;
                            float up_x_1_186 = _cvt_f32_256_186.x;
                            float up_y_1_186 = _cvt_f32_256_186.y;
                            float _exp_457 = expf(gate_x_1_186 * -1.0f);
                            float denominator_x_1_186 = _exp_457 + 1.0f;
                            float _exp_458 = expf(gate_y_1_186 * -1.0f);
                            float denominator_y_1_186 = _exp_458 + 1.0f;
                            float hidden_x_1_186 = gate_x_1_186 / denominator_x_1_186 * up_x_1_186;
                            float hidden_y_1_186 = gate_y_1_186 / denominator_y_1_186 * up_y_1_186;
                            hidden_packed_3_1[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_186, hidden_y_1_186)));
                            float2 _cvt_f32_187 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[27]));
                            float2 _cvt_f32_256_187 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[27]));
                            float gate_x_1_187 = _cvt_f32_187.x;
                            float gate_y_1_187 = _cvt_f32_187.y;
                            float up_x_1_187 = _cvt_f32_256_187.x;
                            float up_y_1_187 = _cvt_f32_256_187.y;
                            float _exp_459 = expf(gate_x_1_187 * -1.0f);
                            float denominator_x_1_187 = _exp_459 + 1.0f;
                            float _exp_460 = expf(gate_y_1_187 * -1.0f);
                            float denominator_y_1_187 = _exp_460 + 1.0f;
                            float hidden_x_1_187 = gate_x_1_187 / denominator_x_1_187 * up_x_1_187;
                            float hidden_y_1_187 = gate_y_1_187 / denominator_y_1_187 * up_y_1_187;
                            hidden_packed_3_1[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_187, hidden_y_1_187)));
                            float2 _cvt_f32_188 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[28]));
                            float2 _cvt_f32_256_188 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[28]));
                            float gate_x_1_188 = _cvt_f32_188.x;
                            float gate_y_1_188 = _cvt_f32_188.y;
                            float up_x_1_188 = _cvt_f32_256_188.x;
                            float up_y_1_188 = _cvt_f32_256_188.y;
                            float _exp_461 = expf(gate_x_1_188 * -1.0f);
                            float denominator_x_1_188 = _exp_461 + 1.0f;
                            float _exp_462 = expf(gate_y_1_188 * -1.0f);
                            float denominator_y_1_188 = _exp_462 + 1.0f;
                            float hidden_x_1_188 = gate_x_1_188 / denominator_x_1_188 * up_x_1_188;
                            float hidden_y_1_188 = gate_y_1_188 / denominator_y_1_188 * up_y_1_188;
                            hidden_packed_3_1[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_188, hidden_y_1_188)));
                            float2 _cvt_f32_189 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[29]));
                            float2 _cvt_f32_256_189 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[29]));
                            float gate_x_1_189 = _cvt_f32_189.x;
                            float gate_y_1_189 = _cvt_f32_189.y;
                            float up_x_1_189 = _cvt_f32_256_189.x;
                            float up_y_1_189 = _cvt_f32_256_189.y;
                            float _exp_463 = expf(gate_x_1_189 * -1.0f);
                            float denominator_x_1_189 = _exp_463 + 1.0f;
                            float _exp_464 = expf(gate_y_1_189 * -1.0f);
                            float denominator_y_1_189 = _exp_464 + 1.0f;
                            float hidden_x_1_189 = gate_x_1_189 / denominator_x_1_189 * up_x_1_189;
                            float hidden_y_1_189 = gate_y_1_189 / denominator_y_1_189 * up_y_1_189;
                            hidden_packed_3_1[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_189, hidden_y_1_189)));
                            float2 _cvt_f32_190 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[30]));
                            float2 _cvt_f32_256_190 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[30]));
                            float gate_x_1_190 = _cvt_f32_190.x;
                            float gate_y_1_190 = _cvt_f32_190.y;
                            float up_x_1_190 = _cvt_f32_256_190.x;
                            float up_y_1_190 = _cvt_f32_256_190.y;
                            float _exp_465 = expf(gate_x_1_190 * -1.0f);
                            float denominator_x_1_190 = _exp_465 + 1.0f;
                            float _exp_466 = expf(gate_y_1_190 * -1.0f);
                            float denominator_y_1_190 = _exp_466 + 1.0f;
                            float hidden_x_1_190 = gate_x_1_190 / denominator_x_1_190 * up_x_1_190;
                            float hidden_y_1_190 = gate_y_1_190 / denominator_y_1_190 * up_y_1_190;
                            hidden_packed_3_1[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_190, hidden_y_1_190)));
                            float2 _cvt_f32_191 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_5[31]));
                            float2 _cvt_f32_256_191 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_5[31]));
                            float gate_x_1_191 = _cvt_f32_191.x;
                            float gate_y_1_191 = _cvt_f32_191.y;
                            float up_x_1_191 = _cvt_f32_256_191.x;
                            float up_y_1_191 = _cvt_f32_256_191.y;
                            float _exp_467 = expf(gate_x_1_191 * -1.0f);
                            float denominator_x_1_191 = _exp_467 + 1.0f;
                            float _exp_468 = expf(gate_y_1_191 * -1.0f);
                            float denominator_y_1_191 = _exp_468 + 1.0f;
                            float hidden_x_1_191 = gate_x_1_191 / denominator_x_1_191 * up_x_1_191;
                            float hidden_y_1_191 = gate_y_1_191 / denominator_y_1_191 * up_y_1_191;
                            hidden_packed_3_1[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_191, hidden_y_1_191)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_8_1 = tid / 32;
                            int lane_8_1 = tid % 32;
                            #pragma unroll
                            for (int half_12 = 0; half_12 < 2; half_12++) {
                                #pragma unroll
                                for (int col_tile_12 = 0; col_tile_12 < 2; col_tile_12++) {
                                    int row_18 = warp_8_1 * 32 + half_12 * 16 + lane_8_1 % 16;
                                    int col_15 = col_tile_12 * 16 + lane_8_1 / 16 * 8;
                                    unsigned int address_14 = d_smem_addr + 16384 + (unsigned int)((row_18 * 32 + col_15) * 2);
                                    address_14 = address_14 ^ (address_14 & 511) >> 7 << 4;
                                    int offset_13 = half_12 * 8 + col_tile_12 * 4;
                                    uint32_t _stmatrix_addr_469 = static_cast<uint32_t>(address_14);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_469), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_13])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_13 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_13 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_13 + 3]))
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
                            int warp_9_1 = tid / 32;
                            int lane_9_1 = tid % 32;
                            #pragma unroll
                            for (int half_13 = 0; half_13 < 2; half_13++) {
                                #pragma unroll
                                for (int col_tile_13 = 0; col_tile_13 < 2; col_tile_13++) {
                                    int row_19 = warp_9_1 * 32 + half_13 * 16 + lane_9_1 % 16;
                                    int col_16 = col_tile_13 * 16 + lane_9_1 / 16 * 8;
                                    unsigned int address_15 = d_smem_addr + (unsigned int)((row_19 * 32 + col_16) * 2);
                                    address_15 = address_15 ^ (address_15 & 511) >> 7 << 4;
                                    int offset_14 = 16 + half_13 * 8 + col_tile_13 * 4;
                                    uint32_t _stmatrix_addr_470 = static_cast<uint32_t>(address_15);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_470), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_14])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_14 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_14 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_3_1[offset_14 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            int _parity_output_arrived_5 = (int)((unsigned int)gemm_out1 % 2);
                            mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_5);
                            unsigned int gate_packed_6[32];
                            unsigned int up_packed_6[32];
                            unsigned int address_3_24 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 256;
                            float _tmem_load_24[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[15]))
                                : "r"(address_3_24));
                            float _tmem_load_32_24[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_24[15]))
                                : "r"(address_3_24 + 128));
                            gate_packed_6[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[0], _tmem_load_24[1])));
                            up_packed_6[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[0], _tmem_load_32_24[1])));
                            gate_packed_6[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[2], _tmem_load_24[3])));
                            up_packed_6[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[2], _tmem_load_32_24[3])));
                            gate_packed_6[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[4], _tmem_load_24[5])));
                            up_packed_6[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[4], _tmem_load_32_24[5])));
                            gate_packed_6[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[6], _tmem_load_24[7])));
                            up_packed_6[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[6], _tmem_load_32_24[7])));
                            gate_packed_6[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[8], _tmem_load_24[9])));
                            up_packed_6[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[8], _tmem_load_32_24[9])));
                            gate_packed_6[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[10], _tmem_load_24[11])));
                            up_packed_6[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[10], _tmem_load_32_24[11])));
                            gate_packed_6[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[12], _tmem_load_24[13])));
                            up_packed_6[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[12], _tmem_load_32_24[13])));
                            gate_packed_6[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_24[14], _tmem_load_24[15])));
                            up_packed_6[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_24[14], _tmem_load_32_24[15])));
                            unsigned int address_3_25 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 256;
                            float _tmem_load_25[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[15]))
                                : "r"(address_3_25));
                            float _tmem_load_32_25[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_25[15]))
                                : "r"(address_3_25 + 128));
                            gate_packed_6[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[0], _tmem_load_25[1])));
                            up_packed_6[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[0], _tmem_load_32_25[1])));
                            gate_packed_6[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[2], _tmem_load_25[3])));
                            up_packed_6[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[2], _tmem_load_32_25[3])));
                            gate_packed_6[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[4], _tmem_load_25[5])));
                            up_packed_6[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[4], _tmem_load_32_25[5])));
                            gate_packed_6[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[6], _tmem_load_25[7])));
                            up_packed_6[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[6], _tmem_load_32_25[7])));
                            gate_packed_6[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[8], _tmem_load_25[9])));
                            up_packed_6[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[8], _tmem_load_32_25[9])));
                            gate_packed_6[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[10], _tmem_load_25[11])));
                            up_packed_6[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[10], _tmem_load_32_25[11])));
                            gate_packed_6[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[12], _tmem_load_25[13])));
                            up_packed_6[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[12], _tmem_load_32_25[13])));
                            gate_packed_6[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_25[14], _tmem_load_25[15])));
                            up_packed_6[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_25[14], _tmem_load_32_25[15])));
                            unsigned int address_3_26 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 288;
                            float _tmem_load_26[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[15]))
                                : "r"(address_3_26));
                            float _tmem_load_32_26[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_26[15]))
                                : "r"(address_3_26 + 128));
                            gate_packed_6[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[0], _tmem_load_26[1])));
                            up_packed_6[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[0], _tmem_load_32_26[1])));
                            gate_packed_6[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[2], _tmem_load_26[3])));
                            up_packed_6[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[2], _tmem_load_32_26[3])));
                            gate_packed_6[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[4], _tmem_load_26[5])));
                            up_packed_6[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[4], _tmem_load_32_26[5])));
                            gate_packed_6[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[6], _tmem_load_26[7])));
                            up_packed_6[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[6], _tmem_load_32_26[7])));
                            gate_packed_6[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[8], _tmem_load_26[9])));
                            up_packed_6[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[8], _tmem_load_32_26[9])));
                            gate_packed_6[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[10], _tmem_load_26[11])));
                            up_packed_6[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[10], _tmem_load_32_26[11])));
                            gate_packed_6[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[12], _tmem_load_26[13])));
                            up_packed_6[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[12], _tmem_load_32_26[13])));
                            gate_packed_6[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_26[14], _tmem_load_26[15])));
                            up_packed_6[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_26[14], _tmem_load_32_26[15])));
                            unsigned int address_3_27 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 288;
                            float _tmem_load_27[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[15]))
                                : "r"(address_3_27));
                            float _tmem_load_32_27[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_27[15]))
                                : "r"(address_3_27 + 128));
                            gate_packed_6[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[0], _tmem_load_27[1])));
                            up_packed_6[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[0], _tmem_load_32_27[1])));
                            gate_packed_6[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[2], _tmem_load_27[3])));
                            up_packed_6[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[2], _tmem_load_32_27[3])));
                            gate_packed_6[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[4], _tmem_load_27[5])));
                            up_packed_6[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[4], _tmem_load_32_27[5])));
                            gate_packed_6[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[6], _tmem_load_27[7])));
                            up_packed_6[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[6], _tmem_load_32_27[7])));
                            gate_packed_6[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[8], _tmem_load_27[9])));
                            up_packed_6[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[8], _tmem_load_32_27[9])));
                            gate_packed_6[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[10], _tmem_load_27[11])));
                            up_packed_6[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[10], _tmem_load_32_27[11])));
                            gate_packed_6[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[12], _tmem_load_27[13])));
                            up_packed_6[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[12], _tmem_load_32_27[13])));
                            gate_packed_6[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_27[14], _tmem_load_27[15])));
                            up_packed_6[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_27[14], _tmem_load_32_27[15])));
                            unsigned int gate_packed_7[32];
                            unsigned int up_packed_7[32];
                            unsigned int address_3_28 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 320;
                            float _tmem_load_28[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[15]))
                                : "r"(address_3_28));
                            float _tmem_load_32_28[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_28[15]))
                                : "r"(address_3_28 + 128));
                            gate_packed_7[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[0], _tmem_load_28[1])));
                            up_packed_7[0] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[0], _tmem_load_32_28[1])));
                            gate_packed_7[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[2], _tmem_load_28[3])));
                            up_packed_7[1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[2], _tmem_load_32_28[3])));
                            gate_packed_7[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[4], _tmem_load_28[5])));
                            up_packed_7[2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[4], _tmem_load_32_28[5])));
                            gate_packed_7[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[6], _tmem_load_28[7])));
                            up_packed_7[3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[6], _tmem_load_32_28[7])));
                            gate_packed_7[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[8], _tmem_load_28[9])));
                            up_packed_7[4] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[8], _tmem_load_32_28[9])));
                            gate_packed_7[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[10], _tmem_load_28[11])));
                            up_packed_7[5] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[10], _tmem_load_32_28[11])));
                            gate_packed_7[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[12], _tmem_load_28[13])));
                            up_packed_7[6] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[12], _tmem_load_32_28[13])));
                            gate_packed_7[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_28[14], _tmem_load_28[15])));
                            up_packed_7[7] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_28[14], _tmem_load_32_28[15])));
                            unsigned int address_3_29 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 320;
                            float _tmem_load_29[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[15]))
                                : "r"(address_3_29));
                            float _tmem_load_32_29[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_29[15]))
                                : "r"(address_3_29 + 128));
                            gate_packed_7[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[0], _tmem_load_29[1])));
                            up_packed_7[8] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[0], _tmem_load_32_29[1])));
                            gate_packed_7[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[2], _tmem_load_29[3])));
                            up_packed_7[9] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[2], _tmem_load_32_29[3])));
                            gate_packed_7[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[4], _tmem_load_29[5])));
                            up_packed_7[10] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[4], _tmem_load_32_29[5])));
                            gate_packed_7[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[6], _tmem_load_29[7])));
                            up_packed_7[11] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[6], _tmem_load_32_29[7])));
                            gate_packed_7[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[8], _tmem_load_29[9])));
                            up_packed_7[12] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[8], _tmem_load_32_29[9])));
                            gate_packed_7[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[10], _tmem_load_29[11])));
                            up_packed_7[13] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[10], _tmem_load_32_29[11])));
                            gate_packed_7[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[12], _tmem_load_29[13])));
                            up_packed_7[14] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[12], _tmem_load_32_29[13])));
                            gate_packed_7[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_29[14], _tmem_load_29[15])));
                            up_packed_7[15] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_29[14], _tmem_load_32_29[15])));
                            unsigned int address_3_30 = taddr_1 + (unsigned int)(tid / 32 * 32 << 16) + 352;
                            float _tmem_load_30[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[15]))
                                : "r"(address_3_30));
                            float _tmem_load_32_30[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_30[15]))
                                : "r"(address_3_30 + 128));
                            gate_packed_7[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[0], _tmem_load_30[1])));
                            up_packed_7[16] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[0], _tmem_load_32_30[1])));
                            gate_packed_7[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[2], _tmem_load_30[3])));
                            up_packed_7[17] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[2], _tmem_load_32_30[3])));
                            gate_packed_7[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[4], _tmem_load_30[5])));
                            up_packed_7[18] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[4], _tmem_load_32_30[5])));
                            gate_packed_7[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[6], _tmem_load_30[7])));
                            up_packed_7[19] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[6], _tmem_load_32_30[7])));
                            gate_packed_7[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[8], _tmem_load_30[9])));
                            up_packed_7[20] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[8], _tmem_load_32_30[9])));
                            gate_packed_7[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[10], _tmem_load_30[11])));
                            up_packed_7[21] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[10], _tmem_load_32_30[11])));
                            gate_packed_7[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[12], _tmem_load_30[13])));
                            up_packed_7[22] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[12], _tmem_load_32_30[13])));
                            gate_packed_7[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_30[14], _tmem_load_30[15])));
                            up_packed_7[23] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_30[14], _tmem_load_32_30[15])));
                            unsigned int address_3_31 = taddr_1 + (unsigned int)(tid / 32 * 32 + 16 << 16) + 352;
                            float _tmem_load_31[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[15]))
                                : "r"(address_3_31));
                            float _tmem_load_32_31[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_31[15]))
                                : "r"(address_3_31 + 128));
                            gate_packed_7[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[0], _tmem_load_31[1])));
                            up_packed_7[24] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[0], _tmem_load_32_31[1])));
                            gate_packed_7[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[2], _tmem_load_31[3])));
                            up_packed_7[25] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[2], _tmem_load_32_31[3])));
                            gate_packed_7[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[4], _tmem_load_31[5])));
                            up_packed_7[26] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[4], _tmem_load_32_31[5])));
                            gate_packed_7[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[6], _tmem_load_31[7])));
                            up_packed_7[27] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[6], _tmem_load_32_31[7])));
                            gate_packed_7[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[8], _tmem_load_31[9])));
                            up_packed_7[28] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[8], _tmem_load_32_31[9])));
                            gate_packed_7[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[10], _tmem_load_31[11])));
                            up_packed_7[29] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[10], _tmem_load_32_31[11])));
                            gate_packed_7[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[12], _tmem_load_31[13])));
                            up_packed_7[30] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[12], _tmem_load_32_31[13])));
                            gate_packed_7[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_31[14], _tmem_load_31[15])));
                            up_packed_7[31] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_31[14], _tmem_load_32_31[15])));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            unsigned int hidden_packed_4[32];
                            float2 _cvt_f32_192 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[0]));
                            float2 _cvt_f32_256_192 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[0]));
                            float gate_x_1_192 = _cvt_f32_192.x;
                            float gate_y_1_192 = _cvt_f32_192.y;
                            float up_x_1_192 = _cvt_f32_256_192.x;
                            float up_y_1_192 = _cvt_f32_256_192.y;
                            float _exp_471 = expf(gate_x_1_192 * -1.0f);
                            float denominator_x_1_192 = _exp_471 + 1.0f;
                            float _exp_472 = expf(gate_y_1_192 * -1.0f);
                            float denominator_y_1_192 = _exp_472 + 1.0f;
                            float hidden_x_1_192 = gate_x_1_192 / denominator_x_1_192 * up_x_1_192;
                            float hidden_y_1_192 = gate_y_1_192 / denominator_y_1_192 * up_y_1_192;
                            hidden_packed_4[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_192, hidden_y_1_192)));
                            float2 _cvt_f32_193 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[1]));
                            float2 _cvt_f32_256_193 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[1]));
                            float gate_x_1_193 = _cvt_f32_193.x;
                            float gate_y_1_193 = _cvt_f32_193.y;
                            float up_x_1_193 = _cvt_f32_256_193.x;
                            float up_y_1_193 = _cvt_f32_256_193.y;
                            float _exp_473 = expf(gate_x_1_193 * -1.0f);
                            float denominator_x_1_193 = _exp_473 + 1.0f;
                            float _exp_474 = expf(gate_y_1_193 * -1.0f);
                            float denominator_y_1_193 = _exp_474 + 1.0f;
                            float hidden_x_1_193 = gate_x_1_193 / denominator_x_1_193 * up_x_1_193;
                            float hidden_y_1_193 = gate_y_1_193 / denominator_y_1_193 * up_y_1_193;
                            hidden_packed_4[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_193, hidden_y_1_193)));
                            float2 _cvt_f32_194 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[2]));
                            float2 _cvt_f32_256_194 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[2]));
                            float gate_x_1_194 = _cvt_f32_194.x;
                            float gate_y_1_194 = _cvt_f32_194.y;
                            float up_x_1_194 = _cvt_f32_256_194.x;
                            float up_y_1_194 = _cvt_f32_256_194.y;
                            float _exp_475 = expf(gate_x_1_194 * -1.0f);
                            float denominator_x_1_194 = _exp_475 + 1.0f;
                            float _exp_476 = expf(gate_y_1_194 * -1.0f);
                            float denominator_y_1_194 = _exp_476 + 1.0f;
                            float hidden_x_1_194 = gate_x_1_194 / denominator_x_1_194 * up_x_1_194;
                            float hidden_y_1_194 = gate_y_1_194 / denominator_y_1_194 * up_y_1_194;
                            hidden_packed_4[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_194, hidden_y_1_194)));
                            float2 _cvt_f32_195 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[3]));
                            float2 _cvt_f32_256_195 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[3]));
                            float gate_x_1_195 = _cvt_f32_195.x;
                            float gate_y_1_195 = _cvt_f32_195.y;
                            float up_x_1_195 = _cvt_f32_256_195.x;
                            float up_y_1_195 = _cvt_f32_256_195.y;
                            float _exp_477 = expf(gate_x_1_195 * -1.0f);
                            float denominator_x_1_195 = _exp_477 + 1.0f;
                            float _exp_478 = expf(gate_y_1_195 * -1.0f);
                            float denominator_y_1_195 = _exp_478 + 1.0f;
                            float hidden_x_1_195 = gate_x_1_195 / denominator_x_1_195 * up_x_1_195;
                            float hidden_y_1_195 = gate_y_1_195 / denominator_y_1_195 * up_y_1_195;
                            hidden_packed_4[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_195, hidden_y_1_195)));
                            float2 _cvt_f32_196 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[4]));
                            float2 _cvt_f32_256_196 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[4]));
                            float gate_x_1_196 = _cvt_f32_196.x;
                            float gate_y_1_196 = _cvt_f32_196.y;
                            float up_x_1_196 = _cvt_f32_256_196.x;
                            float up_y_1_196 = _cvt_f32_256_196.y;
                            float _exp_479 = expf(gate_x_1_196 * -1.0f);
                            float denominator_x_1_196 = _exp_479 + 1.0f;
                            float _exp_480 = expf(gate_y_1_196 * -1.0f);
                            float denominator_y_1_196 = _exp_480 + 1.0f;
                            float hidden_x_1_196 = gate_x_1_196 / denominator_x_1_196 * up_x_1_196;
                            float hidden_y_1_196 = gate_y_1_196 / denominator_y_1_196 * up_y_1_196;
                            hidden_packed_4[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_196, hidden_y_1_196)));
                            float2 _cvt_f32_197 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[5]));
                            float2 _cvt_f32_256_197 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[5]));
                            float gate_x_1_197 = _cvt_f32_197.x;
                            float gate_y_1_197 = _cvt_f32_197.y;
                            float up_x_1_197 = _cvt_f32_256_197.x;
                            float up_y_1_197 = _cvt_f32_256_197.y;
                            float _exp_481 = expf(gate_x_1_197 * -1.0f);
                            float denominator_x_1_197 = _exp_481 + 1.0f;
                            float _exp_482 = expf(gate_y_1_197 * -1.0f);
                            float denominator_y_1_197 = _exp_482 + 1.0f;
                            float hidden_x_1_197 = gate_x_1_197 / denominator_x_1_197 * up_x_1_197;
                            float hidden_y_1_197 = gate_y_1_197 / denominator_y_1_197 * up_y_1_197;
                            hidden_packed_4[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_197, hidden_y_1_197)));
                            float2 _cvt_f32_198 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[6]));
                            float2 _cvt_f32_256_198 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[6]));
                            float gate_x_1_198 = _cvt_f32_198.x;
                            float gate_y_1_198 = _cvt_f32_198.y;
                            float up_x_1_198 = _cvt_f32_256_198.x;
                            float up_y_1_198 = _cvt_f32_256_198.y;
                            float _exp_483 = expf(gate_x_1_198 * -1.0f);
                            float denominator_x_1_198 = _exp_483 + 1.0f;
                            float _exp_484 = expf(gate_y_1_198 * -1.0f);
                            float denominator_y_1_198 = _exp_484 + 1.0f;
                            float hidden_x_1_198 = gate_x_1_198 / denominator_x_1_198 * up_x_1_198;
                            float hidden_y_1_198 = gate_y_1_198 / denominator_y_1_198 * up_y_1_198;
                            hidden_packed_4[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_198, hidden_y_1_198)));
                            float2 _cvt_f32_199 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[7]));
                            float2 _cvt_f32_256_199 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[7]));
                            float gate_x_1_199 = _cvt_f32_199.x;
                            float gate_y_1_199 = _cvt_f32_199.y;
                            float up_x_1_199 = _cvt_f32_256_199.x;
                            float up_y_1_199 = _cvt_f32_256_199.y;
                            float _exp_485 = expf(gate_x_1_199 * -1.0f);
                            float denominator_x_1_199 = _exp_485 + 1.0f;
                            float _exp_486 = expf(gate_y_1_199 * -1.0f);
                            float denominator_y_1_199 = _exp_486 + 1.0f;
                            float hidden_x_1_199 = gate_x_1_199 / denominator_x_1_199 * up_x_1_199;
                            float hidden_y_1_199 = gate_y_1_199 / denominator_y_1_199 * up_y_1_199;
                            hidden_packed_4[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_199, hidden_y_1_199)));
                            float2 _cvt_f32_200 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[8]));
                            float2 _cvt_f32_256_200 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[8]));
                            float gate_x_1_200 = _cvt_f32_200.x;
                            float gate_y_1_200 = _cvt_f32_200.y;
                            float up_x_1_200 = _cvt_f32_256_200.x;
                            float up_y_1_200 = _cvt_f32_256_200.y;
                            float _exp_487 = expf(gate_x_1_200 * -1.0f);
                            float denominator_x_1_200 = _exp_487 + 1.0f;
                            float _exp_488 = expf(gate_y_1_200 * -1.0f);
                            float denominator_y_1_200 = _exp_488 + 1.0f;
                            float hidden_x_1_200 = gate_x_1_200 / denominator_x_1_200 * up_x_1_200;
                            float hidden_y_1_200 = gate_y_1_200 / denominator_y_1_200 * up_y_1_200;
                            hidden_packed_4[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_200, hidden_y_1_200)));
                            float2 _cvt_f32_201 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[9]));
                            float2 _cvt_f32_256_201 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[9]));
                            float gate_x_1_201 = _cvt_f32_201.x;
                            float gate_y_1_201 = _cvt_f32_201.y;
                            float up_x_1_201 = _cvt_f32_256_201.x;
                            float up_y_1_201 = _cvt_f32_256_201.y;
                            float _exp_489 = expf(gate_x_1_201 * -1.0f);
                            float denominator_x_1_201 = _exp_489 + 1.0f;
                            float _exp_490 = expf(gate_y_1_201 * -1.0f);
                            float denominator_y_1_201 = _exp_490 + 1.0f;
                            float hidden_x_1_201 = gate_x_1_201 / denominator_x_1_201 * up_x_1_201;
                            float hidden_y_1_201 = gate_y_1_201 / denominator_y_1_201 * up_y_1_201;
                            hidden_packed_4[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_201, hidden_y_1_201)));
                            float2 _cvt_f32_202 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[10]));
                            float2 _cvt_f32_256_202 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[10]));
                            float gate_x_1_202 = _cvt_f32_202.x;
                            float gate_y_1_202 = _cvt_f32_202.y;
                            float up_x_1_202 = _cvt_f32_256_202.x;
                            float up_y_1_202 = _cvt_f32_256_202.y;
                            float _exp_491 = expf(gate_x_1_202 * -1.0f);
                            float denominator_x_1_202 = _exp_491 + 1.0f;
                            float _exp_492 = expf(gate_y_1_202 * -1.0f);
                            float denominator_y_1_202 = _exp_492 + 1.0f;
                            float hidden_x_1_202 = gate_x_1_202 / denominator_x_1_202 * up_x_1_202;
                            float hidden_y_1_202 = gate_y_1_202 / denominator_y_1_202 * up_y_1_202;
                            hidden_packed_4[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_202, hidden_y_1_202)));
                            float2 _cvt_f32_203 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[11]));
                            float2 _cvt_f32_256_203 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[11]));
                            float gate_x_1_203 = _cvt_f32_203.x;
                            float gate_y_1_203 = _cvt_f32_203.y;
                            float up_x_1_203 = _cvt_f32_256_203.x;
                            float up_y_1_203 = _cvt_f32_256_203.y;
                            float _exp_493 = expf(gate_x_1_203 * -1.0f);
                            float denominator_x_1_203 = _exp_493 + 1.0f;
                            float _exp_494 = expf(gate_y_1_203 * -1.0f);
                            float denominator_y_1_203 = _exp_494 + 1.0f;
                            float hidden_x_1_203 = gate_x_1_203 / denominator_x_1_203 * up_x_1_203;
                            float hidden_y_1_203 = gate_y_1_203 / denominator_y_1_203 * up_y_1_203;
                            hidden_packed_4[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_203, hidden_y_1_203)));
                            float2 _cvt_f32_204 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[12]));
                            float2 _cvt_f32_256_204 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[12]));
                            float gate_x_1_204 = _cvt_f32_204.x;
                            float gate_y_1_204 = _cvt_f32_204.y;
                            float up_x_1_204 = _cvt_f32_256_204.x;
                            float up_y_1_204 = _cvt_f32_256_204.y;
                            float _exp_495 = expf(gate_x_1_204 * -1.0f);
                            float denominator_x_1_204 = _exp_495 + 1.0f;
                            float _exp_496 = expf(gate_y_1_204 * -1.0f);
                            float denominator_y_1_204 = _exp_496 + 1.0f;
                            float hidden_x_1_204 = gate_x_1_204 / denominator_x_1_204 * up_x_1_204;
                            float hidden_y_1_204 = gate_y_1_204 / denominator_y_1_204 * up_y_1_204;
                            hidden_packed_4[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_204, hidden_y_1_204)));
                            float2 _cvt_f32_205 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[13]));
                            float2 _cvt_f32_256_205 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[13]));
                            float gate_x_1_205 = _cvt_f32_205.x;
                            float gate_y_1_205 = _cvt_f32_205.y;
                            float up_x_1_205 = _cvt_f32_256_205.x;
                            float up_y_1_205 = _cvt_f32_256_205.y;
                            float _exp_497 = expf(gate_x_1_205 * -1.0f);
                            float denominator_x_1_205 = _exp_497 + 1.0f;
                            float _exp_498 = expf(gate_y_1_205 * -1.0f);
                            float denominator_y_1_205 = _exp_498 + 1.0f;
                            float hidden_x_1_205 = gate_x_1_205 / denominator_x_1_205 * up_x_1_205;
                            float hidden_y_1_205 = gate_y_1_205 / denominator_y_1_205 * up_y_1_205;
                            hidden_packed_4[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_205, hidden_y_1_205)));
                            float2 _cvt_f32_206 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[14]));
                            float2 _cvt_f32_256_206 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[14]));
                            float gate_x_1_206 = _cvt_f32_206.x;
                            float gate_y_1_206 = _cvt_f32_206.y;
                            float up_x_1_206 = _cvt_f32_256_206.x;
                            float up_y_1_206 = _cvt_f32_256_206.y;
                            float _exp_499 = expf(gate_x_1_206 * -1.0f);
                            float denominator_x_1_206 = _exp_499 + 1.0f;
                            float _exp_500 = expf(gate_y_1_206 * -1.0f);
                            float denominator_y_1_206 = _exp_500 + 1.0f;
                            float hidden_x_1_206 = gate_x_1_206 / denominator_x_1_206 * up_x_1_206;
                            float hidden_y_1_206 = gate_y_1_206 / denominator_y_1_206 * up_y_1_206;
                            hidden_packed_4[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_206, hidden_y_1_206)));
                            float2 _cvt_f32_207 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[15]));
                            float2 _cvt_f32_256_207 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[15]));
                            float gate_x_1_207 = _cvt_f32_207.x;
                            float gate_y_1_207 = _cvt_f32_207.y;
                            float up_x_1_207 = _cvt_f32_256_207.x;
                            float up_y_1_207 = _cvt_f32_256_207.y;
                            float _exp_501 = expf(gate_x_1_207 * -1.0f);
                            float denominator_x_1_207 = _exp_501 + 1.0f;
                            float _exp_502 = expf(gate_y_1_207 * -1.0f);
                            float denominator_y_1_207 = _exp_502 + 1.0f;
                            float hidden_x_1_207 = gate_x_1_207 / denominator_x_1_207 * up_x_1_207;
                            float hidden_y_1_207 = gate_y_1_207 / denominator_y_1_207 * up_y_1_207;
                            hidden_packed_4[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_207, hidden_y_1_207)));
                            float2 _cvt_f32_208 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[16]));
                            float2 _cvt_f32_256_208 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[16]));
                            float gate_x_1_208 = _cvt_f32_208.x;
                            float gate_y_1_208 = _cvt_f32_208.y;
                            float up_x_1_208 = _cvt_f32_256_208.x;
                            float up_y_1_208 = _cvt_f32_256_208.y;
                            float _exp_503 = expf(gate_x_1_208 * -1.0f);
                            float denominator_x_1_208 = _exp_503 + 1.0f;
                            float _exp_504 = expf(gate_y_1_208 * -1.0f);
                            float denominator_y_1_208 = _exp_504 + 1.0f;
                            float hidden_x_1_208 = gate_x_1_208 / denominator_x_1_208 * up_x_1_208;
                            float hidden_y_1_208 = gate_y_1_208 / denominator_y_1_208 * up_y_1_208;
                            hidden_packed_4[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_208, hidden_y_1_208)));
                            float2 _cvt_f32_209 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[17]));
                            float2 _cvt_f32_256_209 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[17]));
                            float gate_x_1_209 = _cvt_f32_209.x;
                            float gate_y_1_209 = _cvt_f32_209.y;
                            float up_x_1_209 = _cvt_f32_256_209.x;
                            float up_y_1_209 = _cvt_f32_256_209.y;
                            float _exp_505 = expf(gate_x_1_209 * -1.0f);
                            float denominator_x_1_209 = _exp_505 + 1.0f;
                            float _exp_506 = expf(gate_y_1_209 * -1.0f);
                            float denominator_y_1_209 = _exp_506 + 1.0f;
                            float hidden_x_1_209 = gate_x_1_209 / denominator_x_1_209 * up_x_1_209;
                            float hidden_y_1_209 = gate_y_1_209 / denominator_y_1_209 * up_y_1_209;
                            hidden_packed_4[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_209, hidden_y_1_209)));
                            float2 _cvt_f32_210 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[18]));
                            float2 _cvt_f32_256_210 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[18]));
                            float gate_x_1_210 = _cvt_f32_210.x;
                            float gate_y_1_210 = _cvt_f32_210.y;
                            float up_x_1_210 = _cvt_f32_256_210.x;
                            float up_y_1_210 = _cvt_f32_256_210.y;
                            float _exp_507 = expf(gate_x_1_210 * -1.0f);
                            float denominator_x_1_210 = _exp_507 + 1.0f;
                            float _exp_508 = expf(gate_y_1_210 * -1.0f);
                            float denominator_y_1_210 = _exp_508 + 1.0f;
                            float hidden_x_1_210 = gate_x_1_210 / denominator_x_1_210 * up_x_1_210;
                            float hidden_y_1_210 = gate_y_1_210 / denominator_y_1_210 * up_y_1_210;
                            hidden_packed_4[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_210, hidden_y_1_210)));
                            float2 _cvt_f32_211 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[19]));
                            float2 _cvt_f32_256_211 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[19]));
                            float gate_x_1_211 = _cvt_f32_211.x;
                            float gate_y_1_211 = _cvt_f32_211.y;
                            float up_x_1_211 = _cvt_f32_256_211.x;
                            float up_y_1_211 = _cvt_f32_256_211.y;
                            float _exp_509 = expf(gate_x_1_211 * -1.0f);
                            float denominator_x_1_211 = _exp_509 + 1.0f;
                            float _exp_510 = expf(gate_y_1_211 * -1.0f);
                            float denominator_y_1_211 = _exp_510 + 1.0f;
                            float hidden_x_1_211 = gate_x_1_211 / denominator_x_1_211 * up_x_1_211;
                            float hidden_y_1_211 = gate_y_1_211 / denominator_y_1_211 * up_y_1_211;
                            hidden_packed_4[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_211, hidden_y_1_211)));
                            float2 _cvt_f32_212 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[20]));
                            float2 _cvt_f32_256_212 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[20]));
                            float gate_x_1_212 = _cvt_f32_212.x;
                            float gate_y_1_212 = _cvt_f32_212.y;
                            float up_x_1_212 = _cvt_f32_256_212.x;
                            float up_y_1_212 = _cvt_f32_256_212.y;
                            float _exp_511 = expf(gate_x_1_212 * -1.0f);
                            float denominator_x_1_212 = _exp_511 + 1.0f;
                            float _exp_512 = expf(gate_y_1_212 * -1.0f);
                            float denominator_y_1_212 = _exp_512 + 1.0f;
                            float hidden_x_1_212 = gate_x_1_212 / denominator_x_1_212 * up_x_1_212;
                            float hidden_y_1_212 = gate_y_1_212 / denominator_y_1_212 * up_y_1_212;
                            hidden_packed_4[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_212, hidden_y_1_212)));
                            float2 _cvt_f32_213 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[21]));
                            float2 _cvt_f32_256_213 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[21]));
                            float gate_x_1_213 = _cvt_f32_213.x;
                            float gate_y_1_213 = _cvt_f32_213.y;
                            float up_x_1_213 = _cvt_f32_256_213.x;
                            float up_y_1_213 = _cvt_f32_256_213.y;
                            float _exp_513 = expf(gate_x_1_213 * -1.0f);
                            float denominator_x_1_213 = _exp_513 + 1.0f;
                            float _exp_514 = expf(gate_y_1_213 * -1.0f);
                            float denominator_y_1_213 = _exp_514 + 1.0f;
                            float hidden_x_1_213 = gate_x_1_213 / denominator_x_1_213 * up_x_1_213;
                            float hidden_y_1_213 = gate_y_1_213 / denominator_y_1_213 * up_y_1_213;
                            hidden_packed_4[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_213, hidden_y_1_213)));
                            float2 _cvt_f32_214 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[22]));
                            float2 _cvt_f32_256_214 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[22]));
                            float gate_x_1_214 = _cvt_f32_214.x;
                            float gate_y_1_214 = _cvt_f32_214.y;
                            float up_x_1_214 = _cvt_f32_256_214.x;
                            float up_y_1_214 = _cvt_f32_256_214.y;
                            float _exp_515 = expf(gate_x_1_214 * -1.0f);
                            float denominator_x_1_214 = _exp_515 + 1.0f;
                            float _exp_516 = expf(gate_y_1_214 * -1.0f);
                            float denominator_y_1_214 = _exp_516 + 1.0f;
                            float hidden_x_1_214 = gate_x_1_214 / denominator_x_1_214 * up_x_1_214;
                            float hidden_y_1_214 = gate_y_1_214 / denominator_y_1_214 * up_y_1_214;
                            hidden_packed_4[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_214, hidden_y_1_214)));
                            float2 _cvt_f32_215 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[23]));
                            float2 _cvt_f32_256_215 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[23]));
                            float gate_x_1_215 = _cvt_f32_215.x;
                            float gate_y_1_215 = _cvt_f32_215.y;
                            float up_x_1_215 = _cvt_f32_256_215.x;
                            float up_y_1_215 = _cvt_f32_256_215.y;
                            float _exp_517 = expf(gate_x_1_215 * -1.0f);
                            float denominator_x_1_215 = _exp_517 + 1.0f;
                            float _exp_518 = expf(gate_y_1_215 * -1.0f);
                            float denominator_y_1_215 = _exp_518 + 1.0f;
                            float hidden_x_1_215 = gate_x_1_215 / denominator_x_1_215 * up_x_1_215;
                            float hidden_y_1_215 = gate_y_1_215 / denominator_y_1_215 * up_y_1_215;
                            hidden_packed_4[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_215, hidden_y_1_215)));
                            float2 _cvt_f32_216 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[24]));
                            float2 _cvt_f32_256_216 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[24]));
                            float gate_x_1_216 = _cvt_f32_216.x;
                            float gate_y_1_216 = _cvt_f32_216.y;
                            float up_x_1_216 = _cvt_f32_256_216.x;
                            float up_y_1_216 = _cvt_f32_256_216.y;
                            float _exp_519 = expf(gate_x_1_216 * -1.0f);
                            float denominator_x_1_216 = _exp_519 + 1.0f;
                            float _exp_520 = expf(gate_y_1_216 * -1.0f);
                            float denominator_y_1_216 = _exp_520 + 1.0f;
                            float hidden_x_1_216 = gate_x_1_216 / denominator_x_1_216 * up_x_1_216;
                            float hidden_y_1_216 = gate_y_1_216 / denominator_y_1_216 * up_y_1_216;
                            hidden_packed_4[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_216, hidden_y_1_216)));
                            float2 _cvt_f32_217 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[25]));
                            float2 _cvt_f32_256_217 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[25]));
                            float gate_x_1_217 = _cvt_f32_217.x;
                            float gate_y_1_217 = _cvt_f32_217.y;
                            float up_x_1_217 = _cvt_f32_256_217.x;
                            float up_y_1_217 = _cvt_f32_256_217.y;
                            float _exp_521 = expf(gate_x_1_217 * -1.0f);
                            float denominator_x_1_217 = _exp_521 + 1.0f;
                            float _exp_522 = expf(gate_y_1_217 * -1.0f);
                            float denominator_y_1_217 = _exp_522 + 1.0f;
                            float hidden_x_1_217 = gate_x_1_217 / denominator_x_1_217 * up_x_1_217;
                            float hidden_y_1_217 = gate_y_1_217 / denominator_y_1_217 * up_y_1_217;
                            hidden_packed_4[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_217, hidden_y_1_217)));
                            float2 _cvt_f32_218 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[26]));
                            float2 _cvt_f32_256_218 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[26]));
                            float gate_x_1_218 = _cvt_f32_218.x;
                            float gate_y_1_218 = _cvt_f32_218.y;
                            float up_x_1_218 = _cvt_f32_256_218.x;
                            float up_y_1_218 = _cvt_f32_256_218.y;
                            float _exp_523 = expf(gate_x_1_218 * -1.0f);
                            float denominator_x_1_218 = _exp_523 + 1.0f;
                            float _exp_524 = expf(gate_y_1_218 * -1.0f);
                            float denominator_y_1_218 = _exp_524 + 1.0f;
                            float hidden_x_1_218 = gate_x_1_218 / denominator_x_1_218 * up_x_1_218;
                            float hidden_y_1_218 = gate_y_1_218 / denominator_y_1_218 * up_y_1_218;
                            hidden_packed_4[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_218, hidden_y_1_218)));
                            float2 _cvt_f32_219 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[27]));
                            float2 _cvt_f32_256_219 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[27]));
                            float gate_x_1_219 = _cvt_f32_219.x;
                            float gate_y_1_219 = _cvt_f32_219.y;
                            float up_x_1_219 = _cvt_f32_256_219.x;
                            float up_y_1_219 = _cvt_f32_256_219.y;
                            float _exp_525 = expf(gate_x_1_219 * -1.0f);
                            float denominator_x_1_219 = _exp_525 + 1.0f;
                            float _exp_526 = expf(gate_y_1_219 * -1.0f);
                            float denominator_y_1_219 = _exp_526 + 1.0f;
                            float hidden_x_1_219 = gate_x_1_219 / denominator_x_1_219 * up_x_1_219;
                            float hidden_y_1_219 = gate_y_1_219 / denominator_y_1_219 * up_y_1_219;
                            hidden_packed_4[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_219, hidden_y_1_219)));
                            float2 _cvt_f32_220 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[28]));
                            float2 _cvt_f32_256_220 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[28]));
                            float gate_x_1_220 = _cvt_f32_220.x;
                            float gate_y_1_220 = _cvt_f32_220.y;
                            float up_x_1_220 = _cvt_f32_256_220.x;
                            float up_y_1_220 = _cvt_f32_256_220.y;
                            float _exp_527 = expf(gate_x_1_220 * -1.0f);
                            float denominator_x_1_220 = _exp_527 + 1.0f;
                            float _exp_528 = expf(gate_y_1_220 * -1.0f);
                            float denominator_y_1_220 = _exp_528 + 1.0f;
                            float hidden_x_1_220 = gate_x_1_220 / denominator_x_1_220 * up_x_1_220;
                            float hidden_y_1_220 = gate_y_1_220 / denominator_y_1_220 * up_y_1_220;
                            hidden_packed_4[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_220, hidden_y_1_220)));
                            float2 _cvt_f32_221 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[29]));
                            float2 _cvt_f32_256_221 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[29]));
                            float gate_x_1_221 = _cvt_f32_221.x;
                            float gate_y_1_221 = _cvt_f32_221.y;
                            float up_x_1_221 = _cvt_f32_256_221.x;
                            float up_y_1_221 = _cvt_f32_256_221.y;
                            float _exp_529 = expf(gate_x_1_221 * -1.0f);
                            float denominator_x_1_221 = _exp_529 + 1.0f;
                            float _exp_530 = expf(gate_y_1_221 * -1.0f);
                            float denominator_y_1_221 = _exp_530 + 1.0f;
                            float hidden_x_1_221 = gate_x_1_221 / denominator_x_1_221 * up_x_1_221;
                            float hidden_y_1_221 = gate_y_1_221 / denominator_y_1_221 * up_y_1_221;
                            hidden_packed_4[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_221, hidden_y_1_221)));
                            float2 _cvt_f32_222 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[30]));
                            float2 _cvt_f32_256_222 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[30]));
                            float gate_x_1_222 = _cvt_f32_222.x;
                            float gate_y_1_222 = _cvt_f32_222.y;
                            float up_x_1_222 = _cvt_f32_256_222.x;
                            float up_y_1_222 = _cvt_f32_256_222.y;
                            float _exp_531 = expf(gate_x_1_222 * -1.0f);
                            float denominator_x_1_222 = _exp_531 + 1.0f;
                            float _exp_532 = expf(gate_y_1_222 * -1.0f);
                            float denominator_y_1_222 = _exp_532 + 1.0f;
                            float hidden_x_1_222 = gate_x_1_222 / denominator_x_1_222 * up_x_1_222;
                            float hidden_y_1_222 = gate_y_1_222 / denominator_y_1_222 * up_y_1_222;
                            hidden_packed_4[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_222, hidden_y_1_222)));
                            float2 _cvt_f32_223 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_6[31]));
                            float2 _cvt_f32_256_223 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_6[31]));
                            float gate_x_1_223 = _cvt_f32_223.x;
                            float gate_y_1_223 = _cvt_f32_223.y;
                            float up_x_1_223 = _cvt_f32_256_223.x;
                            float up_y_1_223 = _cvt_f32_256_223.y;
                            float _exp_533 = expf(gate_x_1_223 * -1.0f);
                            float denominator_x_1_223 = _exp_533 + 1.0f;
                            float _exp_534 = expf(gate_y_1_223 * -1.0f);
                            float denominator_y_1_223 = _exp_534 + 1.0f;
                            float hidden_x_1_223 = gate_x_1_223 / denominator_x_1_223 * up_x_1_223;
                            float hidden_y_1_223 = gate_y_1_223 / denominator_y_1_223 * up_y_1_223;
                            hidden_packed_4[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_223, hidden_y_1_223)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_10 = tid / 32;
                            int lane_10 = tid % 32;
                            #pragma unroll
                            for (int half_14 = 0; half_14 < 2; half_14++) {
                                #pragma unroll
                                for (int col_tile_14 = 0; col_tile_14 < 2; col_tile_14++) {
                                    int row_20 = warp_10 * 32 + half_14 * 16 + lane_10 % 16;
                                    int col_17 = col_tile_14 * 16 + lane_10 / 16 * 8;
                                    unsigned int address_16 = d_smem_addr + 8192 + (unsigned int)((row_20 * 32 + col_17) * 2);
                                    address_16 = address_16 ^ (address_16 & 511) >> 7 << 4;
                                    int offset_15 = half_14 * 8 + col_tile_14 * 4;
                                    uint32_t _stmatrix_addr_535 = static_cast<uint32_t>(address_16);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_535), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_15])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_15 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_15 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_15 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_11 = tid / 32;
                            int lane_11 = tid % 32;
                            #pragma unroll
                            for (int half_15 = 0; half_15 < 2; half_15++) {
                                #pragma unroll
                                for (int col_tile_15 = 0; col_tile_15 < 2; col_tile_15++) {
                                    int row_21 = warp_11 * 32 + half_15 * 16 + lane_11 % 16;
                                    int col_18 = col_tile_15 * 16 + lane_11 / 16 * 8;
                                    unsigned int address_17 = d_smem_addr + 16384 + (unsigned int)((row_21 * 32 + col_18) * 2);
                                    address_17 = address_17 ^ (address_17 & 511) >> 7 << 4;
                                    int offset_16 = 16 + half_15 * 8 + col_tile_15 * 4;
                                    uint32_t _stmatrix_addr_536 = static_cast<uint32_t>(address_17);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_536), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_16])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_16 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_16 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_4[offset_16 + 3]))
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
                            unsigned int hidden_packed_5[32];
                            float2 _cvt_f32_224 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[0]));
                            float2 _cvt_f32_256_224 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[0]));
                            float gate_x_1_224 = _cvt_f32_224.x;
                            float gate_y_1_224 = _cvt_f32_224.y;
                            float up_x_1_224 = _cvt_f32_256_224.x;
                            float up_y_1_224 = _cvt_f32_256_224.y;
                            float _exp_537 = expf(gate_x_1_224 * -1.0f);
                            float denominator_x_1_224 = _exp_537 + 1.0f;
                            float _exp_538 = expf(gate_y_1_224 * -1.0f);
                            float denominator_y_1_224 = _exp_538 + 1.0f;
                            float hidden_x_1_224 = gate_x_1_224 / denominator_x_1_224 * up_x_1_224;
                            float hidden_y_1_224 = gate_y_1_224 / denominator_y_1_224 * up_y_1_224;
                            hidden_packed_5[0] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_224, hidden_y_1_224)));
                            float2 _cvt_f32_225 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[1]));
                            float2 _cvt_f32_256_225 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[1]));
                            float gate_x_1_225 = _cvt_f32_225.x;
                            float gate_y_1_225 = _cvt_f32_225.y;
                            float up_x_1_225 = _cvt_f32_256_225.x;
                            float up_y_1_225 = _cvt_f32_256_225.y;
                            float _exp_539 = expf(gate_x_1_225 * -1.0f);
                            float denominator_x_1_225 = _exp_539 + 1.0f;
                            float _exp_540 = expf(gate_y_1_225 * -1.0f);
                            float denominator_y_1_225 = _exp_540 + 1.0f;
                            float hidden_x_1_225 = gate_x_1_225 / denominator_x_1_225 * up_x_1_225;
                            float hidden_y_1_225 = gate_y_1_225 / denominator_y_1_225 * up_y_1_225;
                            hidden_packed_5[1] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_225, hidden_y_1_225)));
                            float2 _cvt_f32_226 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[2]));
                            float2 _cvt_f32_256_226 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[2]));
                            float gate_x_1_226 = _cvt_f32_226.x;
                            float gate_y_1_226 = _cvt_f32_226.y;
                            float up_x_1_226 = _cvt_f32_256_226.x;
                            float up_y_1_226 = _cvt_f32_256_226.y;
                            float _exp_541 = expf(gate_x_1_226 * -1.0f);
                            float denominator_x_1_226 = _exp_541 + 1.0f;
                            float _exp_542 = expf(gate_y_1_226 * -1.0f);
                            float denominator_y_1_226 = _exp_542 + 1.0f;
                            float hidden_x_1_226 = gate_x_1_226 / denominator_x_1_226 * up_x_1_226;
                            float hidden_y_1_226 = gate_y_1_226 / denominator_y_1_226 * up_y_1_226;
                            hidden_packed_5[2] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_226, hidden_y_1_226)));
                            float2 _cvt_f32_227 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[3]));
                            float2 _cvt_f32_256_227 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[3]));
                            float gate_x_1_227 = _cvt_f32_227.x;
                            float gate_y_1_227 = _cvt_f32_227.y;
                            float up_x_1_227 = _cvt_f32_256_227.x;
                            float up_y_1_227 = _cvt_f32_256_227.y;
                            float _exp_543 = expf(gate_x_1_227 * -1.0f);
                            float denominator_x_1_227 = _exp_543 + 1.0f;
                            float _exp_544 = expf(gate_y_1_227 * -1.0f);
                            float denominator_y_1_227 = _exp_544 + 1.0f;
                            float hidden_x_1_227 = gate_x_1_227 / denominator_x_1_227 * up_x_1_227;
                            float hidden_y_1_227 = gate_y_1_227 / denominator_y_1_227 * up_y_1_227;
                            hidden_packed_5[3] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_227, hidden_y_1_227)));
                            float2 _cvt_f32_228 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[4]));
                            float2 _cvt_f32_256_228 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[4]));
                            float gate_x_1_228 = _cvt_f32_228.x;
                            float gate_y_1_228 = _cvt_f32_228.y;
                            float up_x_1_228 = _cvt_f32_256_228.x;
                            float up_y_1_228 = _cvt_f32_256_228.y;
                            float _exp_545 = expf(gate_x_1_228 * -1.0f);
                            float denominator_x_1_228 = _exp_545 + 1.0f;
                            float _exp_546 = expf(gate_y_1_228 * -1.0f);
                            float denominator_y_1_228 = _exp_546 + 1.0f;
                            float hidden_x_1_228 = gate_x_1_228 / denominator_x_1_228 * up_x_1_228;
                            float hidden_y_1_228 = gate_y_1_228 / denominator_y_1_228 * up_y_1_228;
                            hidden_packed_5[4] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_228, hidden_y_1_228)));
                            float2 _cvt_f32_229 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[5]));
                            float2 _cvt_f32_256_229 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[5]));
                            float gate_x_1_229 = _cvt_f32_229.x;
                            float gate_y_1_229 = _cvt_f32_229.y;
                            float up_x_1_229 = _cvt_f32_256_229.x;
                            float up_y_1_229 = _cvt_f32_256_229.y;
                            float _exp_547 = expf(gate_x_1_229 * -1.0f);
                            float denominator_x_1_229 = _exp_547 + 1.0f;
                            float _exp_548 = expf(gate_y_1_229 * -1.0f);
                            float denominator_y_1_229 = _exp_548 + 1.0f;
                            float hidden_x_1_229 = gate_x_1_229 / denominator_x_1_229 * up_x_1_229;
                            float hidden_y_1_229 = gate_y_1_229 / denominator_y_1_229 * up_y_1_229;
                            hidden_packed_5[5] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_229, hidden_y_1_229)));
                            float2 _cvt_f32_230 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[6]));
                            float2 _cvt_f32_256_230 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[6]));
                            float gate_x_1_230 = _cvt_f32_230.x;
                            float gate_y_1_230 = _cvt_f32_230.y;
                            float up_x_1_230 = _cvt_f32_256_230.x;
                            float up_y_1_230 = _cvt_f32_256_230.y;
                            float _exp_549 = expf(gate_x_1_230 * -1.0f);
                            float denominator_x_1_230 = _exp_549 + 1.0f;
                            float _exp_550 = expf(gate_y_1_230 * -1.0f);
                            float denominator_y_1_230 = _exp_550 + 1.0f;
                            float hidden_x_1_230 = gate_x_1_230 / denominator_x_1_230 * up_x_1_230;
                            float hidden_y_1_230 = gate_y_1_230 / denominator_y_1_230 * up_y_1_230;
                            hidden_packed_5[6] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_230, hidden_y_1_230)));
                            float2 _cvt_f32_231 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[7]));
                            float2 _cvt_f32_256_231 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[7]));
                            float gate_x_1_231 = _cvt_f32_231.x;
                            float gate_y_1_231 = _cvt_f32_231.y;
                            float up_x_1_231 = _cvt_f32_256_231.x;
                            float up_y_1_231 = _cvt_f32_256_231.y;
                            float _exp_551 = expf(gate_x_1_231 * -1.0f);
                            float denominator_x_1_231 = _exp_551 + 1.0f;
                            float _exp_552 = expf(gate_y_1_231 * -1.0f);
                            float denominator_y_1_231 = _exp_552 + 1.0f;
                            float hidden_x_1_231 = gate_x_1_231 / denominator_x_1_231 * up_x_1_231;
                            float hidden_y_1_231 = gate_y_1_231 / denominator_y_1_231 * up_y_1_231;
                            hidden_packed_5[7] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_231, hidden_y_1_231)));
                            float2 _cvt_f32_232 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[8]));
                            float2 _cvt_f32_256_232 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[8]));
                            float gate_x_1_232 = _cvt_f32_232.x;
                            float gate_y_1_232 = _cvt_f32_232.y;
                            float up_x_1_232 = _cvt_f32_256_232.x;
                            float up_y_1_232 = _cvt_f32_256_232.y;
                            float _exp_553 = expf(gate_x_1_232 * -1.0f);
                            float denominator_x_1_232 = _exp_553 + 1.0f;
                            float _exp_554 = expf(gate_y_1_232 * -1.0f);
                            float denominator_y_1_232 = _exp_554 + 1.0f;
                            float hidden_x_1_232 = gate_x_1_232 / denominator_x_1_232 * up_x_1_232;
                            float hidden_y_1_232 = gate_y_1_232 / denominator_y_1_232 * up_y_1_232;
                            hidden_packed_5[8] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_232, hidden_y_1_232)));
                            float2 _cvt_f32_233 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[9]));
                            float2 _cvt_f32_256_233 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[9]));
                            float gate_x_1_233 = _cvt_f32_233.x;
                            float gate_y_1_233 = _cvt_f32_233.y;
                            float up_x_1_233 = _cvt_f32_256_233.x;
                            float up_y_1_233 = _cvt_f32_256_233.y;
                            float _exp_555 = expf(gate_x_1_233 * -1.0f);
                            float denominator_x_1_233 = _exp_555 + 1.0f;
                            float _exp_556 = expf(gate_y_1_233 * -1.0f);
                            float denominator_y_1_233 = _exp_556 + 1.0f;
                            float hidden_x_1_233 = gate_x_1_233 / denominator_x_1_233 * up_x_1_233;
                            float hidden_y_1_233 = gate_y_1_233 / denominator_y_1_233 * up_y_1_233;
                            hidden_packed_5[9] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_233, hidden_y_1_233)));
                            float2 _cvt_f32_234 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[10]));
                            float2 _cvt_f32_256_234 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[10]));
                            float gate_x_1_234 = _cvt_f32_234.x;
                            float gate_y_1_234 = _cvt_f32_234.y;
                            float up_x_1_234 = _cvt_f32_256_234.x;
                            float up_y_1_234 = _cvt_f32_256_234.y;
                            float _exp_557 = expf(gate_x_1_234 * -1.0f);
                            float denominator_x_1_234 = _exp_557 + 1.0f;
                            float _exp_558 = expf(gate_y_1_234 * -1.0f);
                            float denominator_y_1_234 = _exp_558 + 1.0f;
                            float hidden_x_1_234 = gate_x_1_234 / denominator_x_1_234 * up_x_1_234;
                            float hidden_y_1_234 = gate_y_1_234 / denominator_y_1_234 * up_y_1_234;
                            hidden_packed_5[10] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_234, hidden_y_1_234)));
                            float2 _cvt_f32_235 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[11]));
                            float2 _cvt_f32_256_235 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[11]));
                            float gate_x_1_235 = _cvt_f32_235.x;
                            float gate_y_1_235 = _cvt_f32_235.y;
                            float up_x_1_235 = _cvt_f32_256_235.x;
                            float up_y_1_235 = _cvt_f32_256_235.y;
                            float _exp_559 = expf(gate_x_1_235 * -1.0f);
                            float denominator_x_1_235 = _exp_559 + 1.0f;
                            float _exp_560 = expf(gate_y_1_235 * -1.0f);
                            float denominator_y_1_235 = _exp_560 + 1.0f;
                            float hidden_x_1_235 = gate_x_1_235 / denominator_x_1_235 * up_x_1_235;
                            float hidden_y_1_235 = gate_y_1_235 / denominator_y_1_235 * up_y_1_235;
                            hidden_packed_5[11] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_235, hidden_y_1_235)));
                            float2 _cvt_f32_236 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[12]));
                            float2 _cvt_f32_256_236 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[12]));
                            float gate_x_1_236 = _cvt_f32_236.x;
                            float gate_y_1_236 = _cvt_f32_236.y;
                            float up_x_1_236 = _cvt_f32_256_236.x;
                            float up_y_1_236 = _cvt_f32_256_236.y;
                            float _exp_561 = expf(gate_x_1_236 * -1.0f);
                            float denominator_x_1_236 = _exp_561 + 1.0f;
                            float _exp_562 = expf(gate_y_1_236 * -1.0f);
                            float denominator_y_1_236 = _exp_562 + 1.0f;
                            float hidden_x_1_236 = gate_x_1_236 / denominator_x_1_236 * up_x_1_236;
                            float hidden_y_1_236 = gate_y_1_236 / denominator_y_1_236 * up_y_1_236;
                            hidden_packed_5[12] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_236, hidden_y_1_236)));
                            float2 _cvt_f32_237 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[13]));
                            float2 _cvt_f32_256_237 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[13]));
                            float gate_x_1_237 = _cvt_f32_237.x;
                            float gate_y_1_237 = _cvt_f32_237.y;
                            float up_x_1_237 = _cvt_f32_256_237.x;
                            float up_y_1_237 = _cvt_f32_256_237.y;
                            float _exp_563 = expf(gate_x_1_237 * -1.0f);
                            float denominator_x_1_237 = _exp_563 + 1.0f;
                            float _exp_564 = expf(gate_y_1_237 * -1.0f);
                            float denominator_y_1_237 = _exp_564 + 1.0f;
                            float hidden_x_1_237 = gate_x_1_237 / denominator_x_1_237 * up_x_1_237;
                            float hidden_y_1_237 = gate_y_1_237 / denominator_y_1_237 * up_y_1_237;
                            hidden_packed_5[13] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_237, hidden_y_1_237)));
                            float2 _cvt_f32_238 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[14]));
                            float2 _cvt_f32_256_238 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[14]));
                            float gate_x_1_238 = _cvt_f32_238.x;
                            float gate_y_1_238 = _cvt_f32_238.y;
                            float up_x_1_238 = _cvt_f32_256_238.x;
                            float up_y_1_238 = _cvt_f32_256_238.y;
                            float _exp_565 = expf(gate_x_1_238 * -1.0f);
                            float denominator_x_1_238 = _exp_565 + 1.0f;
                            float _exp_566 = expf(gate_y_1_238 * -1.0f);
                            float denominator_y_1_238 = _exp_566 + 1.0f;
                            float hidden_x_1_238 = gate_x_1_238 / denominator_x_1_238 * up_x_1_238;
                            float hidden_y_1_238 = gate_y_1_238 / denominator_y_1_238 * up_y_1_238;
                            hidden_packed_5[14] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_238, hidden_y_1_238)));
                            float2 _cvt_f32_239 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[15]));
                            float2 _cvt_f32_256_239 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[15]));
                            float gate_x_1_239 = _cvt_f32_239.x;
                            float gate_y_1_239 = _cvt_f32_239.y;
                            float up_x_1_239 = _cvt_f32_256_239.x;
                            float up_y_1_239 = _cvt_f32_256_239.y;
                            float _exp_567 = expf(gate_x_1_239 * -1.0f);
                            float denominator_x_1_239 = _exp_567 + 1.0f;
                            float _exp_568 = expf(gate_y_1_239 * -1.0f);
                            float denominator_y_1_239 = _exp_568 + 1.0f;
                            float hidden_x_1_239 = gate_x_1_239 / denominator_x_1_239 * up_x_1_239;
                            float hidden_y_1_239 = gate_y_1_239 / denominator_y_1_239 * up_y_1_239;
                            hidden_packed_5[15] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_239, hidden_y_1_239)));
                            float2 _cvt_f32_240 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[16]));
                            float2 _cvt_f32_256_240 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[16]));
                            float gate_x_1_240 = _cvt_f32_240.x;
                            float gate_y_1_240 = _cvt_f32_240.y;
                            float up_x_1_240 = _cvt_f32_256_240.x;
                            float up_y_1_240 = _cvt_f32_256_240.y;
                            float _exp_569 = expf(gate_x_1_240 * -1.0f);
                            float denominator_x_1_240 = _exp_569 + 1.0f;
                            float _exp_570 = expf(gate_y_1_240 * -1.0f);
                            float denominator_y_1_240 = _exp_570 + 1.0f;
                            float hidden_x_1_240 = gate_x_1_240 / denominator_x_1_240 * up_x_1_240;
                            float hidden_y_1_240 = gate_y_1_240 / denominator_y_1_240 * up_y_1_240;
                            hidden_packed_5[16] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_240, hidden_y_1_240)));
                            float2 _cvt_f32_241 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[17]));
                            float2 _cvt_f32_256_241 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[17]));
                            float gate_x_1_241 = _cvt_f32_241.x;
                            float gate_y_1_241 = _cvt_f32_241.y;
                            float up_x_1_241 = _cvt_f32_256_241.x;
                            float up_y_1_241 = _cvt_f32_256_241.y;
                            float _exp_571 = expf(gate_x_1_241 * -1.0f);
                            float denominator_x_1_241 = _exp_571 + 1.0f;
                            float _exp_572 = expf(gate_y_1_241 * -1.0f);
                            float denominator_y_1_241 = _exp_572 + 1.0f;
                            float hidden_x_1_241 = gate_x_1_241 / denominator_x_1_241 * up_x_1_241;
                            float hidden_y_1_241 = gate_y_1_241 / denominator_y_1_241 * up_y_1_241;
                            hidden_packed_5[17] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_241, hidden_y_1_241)));
                            float2 _cvt_f32_242 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[18]));
                            float2 _cvt_f32_256_242 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[18]));
                            float gate_x_1_242 = _cvt_f32_242.x;
                            float gate_y_1_242 = _cvt_f32_242.y;
                            float up_x_1_242 = _cvt_f32_256_242.x;
                            float up_y_1_242 = _cvt_f32_256_242.y;
                            float _exp_573 = expf(gate_x_1_242 * -1.0f);
                            float denominator_x_1_242 = _exp_573 + 1.0f;
                            float _exp_574 = expf(gate_y_1_242 * -1.0f);
                            float denominator_y_1_242 = _exp_574 + 1.0f;
                            float hidden_x_1_242 = gate_x_1_242 / denominator_x_1_242 * up_x_1_242;
                            float hidden_y_1_242 = gate_y_1_242 / denominator_y_1_242 * up_y_1_242;
                            hidden_packed_5[18] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_242, hidden_y_1_242)));
                            float2 _cvt_f32_243 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[19]));
                            float2 _cvt_f32_256_243 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[19]));
                            float gate_x_1_243 = _cvt_f32_243.x;
                            float gate_y_1_243 = _cvt_f32_243.y;
                            float up_x_1_243 = _cvt_f32_256_243.x;
                            float up_y_1_243 = _cvt_f32_256_243.y;
                            float _exp_575 = expf(gate_x_1_243 * -1.0f);
                            float denominator_x_1_243 = _exp_575 + 1.0f;
                            float _exp_576 = expf(gate_y_1_243 * -1.0f);
                            float denominator_y_1_243 = _exp_576 + 1.0f;
                            float hidden_x_1_243 = gate_x_1_243 / denominator_x_1_243 * up_x_1_243;
                            float hidden_y_1_243 = gate_y_1_243 / denominator_y_1_243 * up_y_1_243;
                            hidden_packed_5[19] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_243, hidden_y_1_243)));
                            float2 _cvt_f32_244 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[20]));
                            float2 _cvt_f32_256_244 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[20]));
                            float gate_x_1_244 = _cvt_f32_244.x;
                            float gate_y_1_244 = _cvt_f32_244.y;
                            float up_x_1_244 = _cvt_f32_256_244.x;
                            float up_y_1_244 = _cvt_f32_256_244.y;
                            float _exp_577 = expf(gate_x_1_244 * -1.0f);
                            float denominator_x_1_244 = _exp_577 + 1.0f;
                            float _exp_578 = expf(gate_y_1_244 * -1.0f);
                            float denominator_y_1_244 = _exp_578 + 1.0f;
                            float hidden_x_1_244 = gate_x_1_244 / denominator_x_1_244 * up_x_1_244;
                            float hidden_y_1_244 = gate_y_1_244 / denominator_y_1_244 * up_y_1_244;
                            hidden_packed_5[20] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_244, hidden_y_1_244)));
                            float2 _cvt_f32_245 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[21]));
                            float2 _cvt_f32_256_245 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[21]));
                            float gate_x_1_245 = _cvt_f32_245.x;
                            float gate_y_1_245 = _cvt_f32_245.y;
                            float up_x_1_245 = _cvt_f32_256_245.x;
                            float up_y_1_245 = _cvt_f32_256_245.y;
                            float _exp_579 = expf(gate_x_1_245 * -1.0f);
                            float denominator_x_1_245 = _exp_579 + 1.0f;
                            float _exp_580 = expf(gate_y_1_245 * -1.0f);
                            float denominator_y_1_245 = _exp_580 + 1.0f;
                            float hidden_x_1_245 = gate_x_1_245 / denominator_x_1_245 * up_x_1_245;
                            float hidden_y_1_245 = gate_y_1_245 / denominator_y_1_245 * up_y_1_245;
                            hidden_packed_5[21] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_245, hidden_y_1_245)));
                            float2 _cvt_f32_246 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[22]));
                            float2 _cvt_f32_256_246 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[22]));
                            float gate_x_1_246 = _cvt_f32_246.x;
                            float gate_y_1_246 = _cvt_f32_246.y;
                            float up_x_1_246 = _cvt_f32_256_246.x;
                            float up_y_1_246 = _cvt_f32_256_246.y;
                            float _exp_581 = expf(gate_x_1_246 * -1.0f);
                            float denominator_x_1_246 = _exp_581 + 1.0f;
                            float _exp_582 = expf(gate_y_1_246 * -1.0f);
                            float denominator_y_1_246 = _exp_582 + 1.0f;
                            float hidden_x_1_246 = gate_x_1_246 / denominator_x_1_246 * up_x_1_246;
                            float hidden_y_1_246 = gate_y_1_246 / denominator_y_1_246 * up_y_1_246;
                            hidden_packed_5[22] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_246, hidden_y_1_246)));
                            float2 _cvt_f32_247 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[23]));
                            float2 _cvt_f32_256_247 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[23]));
                            float gate_x_1_247 = _cvt_f32_247.x;
                            float gate_y_1_247 = _cvt_f32_247.y;
                            float up_x_1_247 = _cvt_f32_256_247.x;
                            float up_y_1_247 = _cvt_f32_256_247.y;
                            float _exp_583 = expf(gate_x_1_247 * -1.0f);
                            float denominator_x_1_247 = _exp_583 + 1.0f;
                            float _exp_584 = expf(gate_y_1_247 * -1.0f);
                            float denominator_y_1_247 = _exp_584 + 1.0f;
                            float hidden_x_1_247 = gate_x_1_247 / denominator_x_1_247 * up_x_1_247;
                            float hidden_y_1_247 = gate_y_1_247 / denominator_y_1_247 * up_y_1_247;
                            hidden_packed_5[23] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_247, hidden_y_1_247)));
                            float2 _cvt_f32_248 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[24]));
                            float2 _cvt_f32_256_248 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[24]));
                            float gate_x_1_248 = _cvt_f32_248.x;
                            float gate_y_1_248 = _cvt_f32_248.y;
                            float up_x_1_248 = _cvt_f32_256_248.x;
                            float up_y_1_248 = _cvt_f32_256_248.y;
                            float _exp_585 = expf(gate_x_1_248 * -1.0f);
                            float denominator_x_1_248 = _exp_585 + 1.0f;
                            float _exp_586 = expf(gate_y_1_248 * -1.0f);
                            float denominator_y_1_248 = _exp_586 + 1.0f;
                            float hidden_x_1_248 = gate_x_1_248 / denominator_x_1_248 * up_x_1_248;
                            float hidden_y_1_248 = gate_y_1_248 / denominator_y_1_248 * up_y_1_248;
                            hidden_packed_5[24] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_248, hidden_y_1_248)));
                            float2 _cvt_f32_249 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[25]));
                            float2 _cvt_f32_256_249 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[25]));
                            float gate_x_1_249 = _cvt_f32_249.x;
                            float gate_y_1_249 = _cvt_f32_249.y;
                            float up_x_1_249 = _cvt_f32_256_249.x;
                            float up_y_1_249 = _cvt_f32_256_249.y;
                            float _exp_587 = expf(gate_x_1_249 * -1.0f);
                            float denominator_x_1_249 = _exp_587 + 1.0f;
                            float _exp_588 = expf(gate_y_1_249 * -1.0f);
                            float denominator_y_1_249 = _exp_588 + 1.0f;
                            float hidden_x_1_249 = gate_x_1_249 / denominator_x_1_249 * up_x_1_249;
                            float hidden_y_1_249 = gate_y_1_249 / denominator_y_1_249 * up_y_1_249;
                            hidden_packed_5[25] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_249, hidden_y_1_249)));
                            float2 _cvt_f32_250 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[26]));
                            float2 _cvt_f32_256_250 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[26]));
                            float gate_x_1_250 = _cvt_f32_250.x;
                            float gate_y_1_250 = _cvt_f32_250.y;
                            float up_x_1_250 = _cvt_f32_256_250.x;
                            float up_y_1_250 = _cvt_f32_256_250.y;
                            float _exp_589 = expf(gate_x_1_250 * -1.0f);
                            float denominator_x_1_250 = _exp_589 + 1.0f;
                            float _exp_590 = expf(gate_y_1_250 * -1.0f);
                            float denominator_y_1_250 = _exp_590 + 1.0f;
                            float hidden_x_1_250 = gate_x_1_250 / denominator_x_1_250 * up_x_1_250;
                            float hidden_y_1_250 = gate_y_1_250 / denominator_y_1_250 * up_y_1_250;
                            hidden_packed_5[26] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_250, hidden_y_1_250)));
                            float2 _cvt_f32_251 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[27]));
                            float2 _cvt_f32_256_251 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[27]));
                            float gate_x_1_251 = _cvt_f32_251.x;
                            float gate_y_1_251 = _cvt_f32_251.y;
                            float up_x_1_251 = _cvt_f32_256_251.x;
                            float up_y_1_251 = _cvt_f32_256_251.y;
                            float _exp_591 = expf(gate_x_1_251 * -1.0f);
                            float denominator_x_1_251 = _exp_591 + 1.0f;
                            float _exp_592 = expf(gate_y_1_251 * -1.0f);
                            float denominator_y_1_251 = _exp_592 + 1.0f;
                            float hidden_x_1_251 = gate_x_1_251 / denominator_x_1_251 * up_x_1_251;
                            float hidden_y_1_251 = gate_y_1_251 / denominator_y_1_251 * up_y_1_251;
                            hidden_packed_5[27] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_251, hidden_y_1_251)));
                            float2 _cvt_f32_252 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[28]));
                            float2 _cvt_f32_256_252 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[28]));
                            float gate_x_1_252 = _cvt_f32_252.x;
                            float gate_y_1_252 = _cvt_f32_252.y;
                            float up_x_1_252 = _cvt_f32_256_252.x;
                            float up_y_1_252 = _cvt_f32_256_252.y;
                            float _exp_593 = expf(gate_x_1_252 * -1.0f);
                            float denominator_x_1_252 = _exp_593 + 1.0f;
                            float _exp_594 = expf(gate_y_1_252 * -1.0f);
                            float denominator_y_1_252 = _exp_594 + 1.0f;
                            float hidden_x_1_252 = gate_x_1_252 / denominator_x_1_252 * up_x_1_252;
                            float hidden_y_1_252 = gate_y_1_252 / denominator_y_1_252 * up_y_1_252;
                            hidden_packed_5[28] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_252, hidden_y_1_252)));
                            float2 _cvt_f32_253 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[29]));
                            float2 _cvt_f32_256_253 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[29]));
                            float gate_x_1_253 = _cvt_f32_253.x;
                            float gate_y_1_253 = _cvt_f32_253.y;
                            float up_x_1_253 = _cvt_f32_256_253.x;
                            float up_y_1_253 = _cvt_f32_256_253.y;
                            float _exp_595 = expf(gate_x_1_253 * -1.0f);
                            float denominator_x_1_253 = _exp_595 + 1.0f;
                            float _exp_596 = expf(gate_y_1_253 * -1.0f);
                            float denominator_y_1_253 = _exp_596 + 1.0f;
                            float hidden_x_1_253 = gate_x_1_253 / denominator_x_1_253 * up_x_1_253;
                            float hidden_y_1_253 = gate_y_1_253 / denominator_y_1_253 * up_y_1_253;
                            hidden_packed_5[29] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_253, hidden_y_1_253)));
                            float2 _cvt_f32_254 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[30]));
                            float2 _cvt_f32_256_254 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[30]));
                            float gate_x_1_254 = _cvt_f32_254.x;
                            float gate_y_1_254 = _cvt_f32_254.y;
                            float up_x_1_254 = _cvt_f32_256_254.x;
                            float up_y_1_254 = _cvt_f32_256_254.y;
                            float _exp_597 = expf(gate_x_1_254 * -1.0f);
                            float denominator_x_1_254 = _exp_597 + 1.0f;
                            float _exp_598 = expf(gate_y_1_254 * -1.0f);
                            float denominator_y_1_254 = _exp_598 + 1.0f;
                            float hidden_x_1_254 = gate_x_1_254 / denominator_x_1_254 * up_x_1_254;
                            float hidden_y_1_254 = gate_y_1_254 / denominator_y_1_254 * up_y_1_254;
                            hidden_packed_5[30] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_254, hidden_y_1_254)));
                            float2 _cvt_f32_255 = cake_bf16x2_to_f32x2(__as_bf16x2(gate_packed_7[31]));
                            float2 _cvt_f32_256_255 = cake_bf16x2_to_f32x2(__as_bf16x2(up_packed_7[31]));
                            float gate_x_1_255 = _cvt_f32_255.x;
                            float gate_y_1_255 = _cvt_f32_255.y;
                            float up_x_1_255 = _cvt_f32_256_255.x;
                            float up_y_1_255 = _cvt_f32_256_255.y;
                            float _exp_599 = expf(gate_x_1_255 * -1.0f);
                            float denominator_x_1_255 = _exp_599 + 1.0f;
                            float _exp_600 = expf(gate_y_1_255 * -1.0f);
                            float denominator_y_1_255 = _exp_600 + 1.0f;
                            float hidden_x_1_255 = gate_x_1_255 / denominator_x_1_255 * up_x_1_255;
                            float hidden_y_1_255 = gate_y_1_255 / denominator_y_1_255 * up_y_1_255;
                            hidden_packed_5[31] = __as_u32(__float22bfloat162_rn(make_float2(hidden_x_1_255, hidden_y_1_255)));
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_12 = tid / 32;
                            int lane_12 = tid % 32;
                            #pragma unroll
                            for (int half_16 = 0; half_16 < 2; half_16++) {
                                #pragma unroll
                                for (int col_tile_16 = 0; col_tile_16 < 2; col_tile_16++) {
                                    int row_22 = warp_12 * 32 + half_16 * 16 + lane_12 % 16;
                                    int col_19 = col_tile_16 * 16 + lane_12 / 16 * 8;
                                    unsigned int address_18 = d_smem_addr + (unsigned int)((row_22 * 32 + col_19) * 2);
                                    address_18 = address_18 ^ (address_18 & 511) >> 7 << 4;
                                    int offset_17 = half_16 * 8 + col_tile_16 * 4;
                                    uint32_t _stmatrix_addr_601 = static_cast<uint32_t>(address_18);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_601), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_17])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_17 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_17 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_17 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4 + 2), "r"(0), "r"(0), "r"(d_smem_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (tid == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            int warp_13 = tid / 32;
                            int lane_13 = tid % 32;
                            #pragma unroll
                            for (int half_17 = 0; half_17 < 2; half_17++) {
                                #pragma unroll
                                for (int col_tile_17 = 0; col_tile_17 < 2; col_tile_17++) {
                                    int row_23 = warp_13 * 32 + half_17 * 16 + lane_13 % 16;
                                    int col_20 = col_tile_17 * 16 + lane_13 / 16 * 8;
                                    unsigned int address_19 = d_smem_addr + 8192 + (unsigned int)((row_23 * 32 + col_20) * 2);
                                    address_19 = address_19 ^ (address_19 & 511) >> 7 << 4;
                                    int offset_18 = 16 + half_17 * 8 + col_tile_17 * 4;
                                    uint32_t _stmatrix_addr_602 = static_cast<uint32_t>(address_19);
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_602), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_18])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_18 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_18 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&hidden_packed_5[offset_18 + 3]))
                                        : "memory");
                                }
                            }
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (tid == 0) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                    " [%0, {%1, %2, %3, %4, %5}], [%6], %7;"
                                    :: "l"((&hidden_routed_out)), "r"(0), "r"(x_2 * 256 + cta_rank_0 * 128), "r"(y_2 * 8 + 4 + 2 + 1), "r"(0), "r"(0), "r"(d_smem_addr + 8192), "l"(0x12F0000000000000ULL) : "memory");
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
                    gemm_wide += iterations_2 * (int)(expert_2 >= 0);
                    gemm_out0 += (int)(expert_2 >= 0);
                    gemm_out1 += (int)(expert_2 >= 0);
                    gemm_acc_tiles += 2 * (int)(expert_2 >= 0);
                } else {
                    int col_blocks_4 = (hidden + 512 - 1) / 512;
                    int x_3 = -1;
                    int y_3 = -1;
                    int expert_3 = -1;
                    int k_start_3 = 0;
                    int k_end_3 = 0;
                    int first_3 = 0;
                    int first_block_1 = (macro_1 * (macro_size / mini_size) + mini_3) * (mini_size / 256);
                    int end_block_1 = ((first_block_1 + mini_size / 256) < (tokens / 256) ? (first_block_1 + mini_size / 256) : (tokens / 256));
                    int block_6 = first_block_1;
                    uint32_t fast_div_q_1 = (down_col_blocks_div.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(task_1 - mini_fused)), down_col_blocks_div.multiplier) >> down_col_blocks_div.shift_right) : (uint32_t)((unsigned int)(task_1 - mini_fused));
                    uint32_t fast_div_r_1 = (uint32_t)((unsigned int)(task_1 - mini_fused)) - fast_div_q_1 * (uint32_t)(down_col_blocks_div.divisor);
                    block_6 = first_block_1 + (int)fast_div_q_1;
                    if (block_6 < end_block_1) {
                        int index_1 = counts[3 * experts + block_6];
                        int offset_19 = counts[experts + index_1] / 256;
                        int first_row_7 = ((first_block_1) > (offset_19) ? (first_block_1) : (offset_19));
                        int _min_603 = ((end_block_1) < (offset_19 + counts[index_1] / 256) ? (end_block_1) : (offset_19 + counts[index_1] / 256));
                        int rows_6 = _min_603 - first_row_7;
                        int supergroup_3 = (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_4) / (rows_6 * 8);
                        int full_cols_3 = col_blocks_4 / 8 * 8;
                        int row_24 = 0;
                        int col_21 = 0;
                        if (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_4 < rows_6 * full_cols_3) {
                            row_24 = (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_4) % (rows_6 * 8) / 8;
                            col_21 = supergroup_3 * 8 + (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_4) % 8;
                        } else {
                            row_24 = (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_4 - rows_6 * full_cols_3) / (col_blocks_4 - full_cols_3);
                            col_21 = full_cols_3 + (task_1 - mini_fused - (first_row_7 - first_block_1) * col_blocks_4 - rows_6 * full_cols_3) % (col_blocks_4 - full_cols_3);
                        }
                        if ((supergroup_3 & 1) != 0) {
                            row_24 = rows_6 - row_24 - 1;
                        }
                        x_3 = first_row_7 + row_24 - macro_1 * (macro_size / 256);
                        y_3 = col_21;
                        expert_3 = index_1;
                    }
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
                                        int poll_6_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_first) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                        int poll_6 = poll_6_first;
                                        while (poll_6 < 2 * (intermediate / 128)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_6_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_6_next) : "l"(hidden_ready + (shared_rows + macro_rows_4 + x_3)) : "memory");
                                            poll_6 = poll_6_next;
                                        }
                                        asm volatile("fence.acquire.gpu;" ::: "memory");
                                    }
                                    int _min_604 = ((mini_size) < (tokens - global_mini_3 * mini_size) ? (mini_size) : (tokens - global_mini_3 * mini_size));
                                    int mini_rows_7 = ((0) > (_min_604) ? (0) : (_min_604));
                                    int required_5 = (mini_rows_7 + 127) / 128 * ((intermediate + 511) / 512);
                                }
                                #pragma unroll 1
                                for (int idx_18 = 0; idx_18 < iterations_3; idx_18++) {
                                    int _stage_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_18) % 4);
                                    int _parity_gemm_finished_3 = (int)((unsigned int)(gemm_wide + idx_18) / 4 % 2) ^ 1;
                                    mbarrier_wait(gemm_finished_addr + (_stage_gemm_finished_3) * 8, _parity_gemm_finished_3);
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(a_smem_addr + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&hidden_routed_in)), "r"(0), "r"(x_3 * 256 + cta_rank_0 * 128), "r"(idx_18), "r"(0), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_smem_addr + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&wd_routed)), "r"(0), "r"(y_3 * 2 * 256 + cta_rank_0 * 128), "r"(idx_18), "r"(expert_3), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(b_hi_addr + (unsigned int)((gemm_wide + idx_18) % 4 * 16384)), "l"((&wd_routed)), "r"(0), "r"((y_3 * 2 + 1) * 256 + cta_rank_0 * 128), "r"(idx_18), "r"(expert_3), "r"(0),
                                           "r"(((gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_18) % 4)) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(1 << cta_rank_0)) : "memory");
                                }
                            }
                        }
                    } else {
                        if (tid / 32 == 4 && cta_rank_0 == 0) {
                            if (warp == 4) {
                                if (elect_sync()) {
                                    int head_3 = ((3) < (iterations_3) ? (3) : (iterations_3));
                                    int tail_4 = 0;
                                    int _min_605 = ((3) < (iterations_3 - head_3) ? (3) : (iterations_3 - head_3));
                                    tail_4 = _min_605;
                                    int _stage_output_finished_6 = (int)((unsigned int)gemm_acc_tiles % 2);
                                    int _parity_output_finished_6 = (int)((unsigned int)gemm_acc_tiles / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_6) * 8, _parity_output_finished_6);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_19 = 0; idx_19 < head_3; idx_19++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_19) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_9 = (int)((unsigned int)(gemm_wide + idx_19) % 4);
                                        int _parity_gemm_arrived_9 = (int)((unsigned int)(gemm_wide + idx_19) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_9) * 8, _parity_gemm_arrived_9);
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_606 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_19) % 4) * 1024;
                                        int _smem_desc_lo_607 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_19) % 4) * 1024;
#else
                                        int _smem_desc_lo_606 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_19) % 4) * 1024;
                                        int _smem_desc_lo_607 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_19) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_606), "r"(_smem_desc_lo_607), "r"(tmem_accumulator), "r"(((idx_19 == 0) ? 0 : 1)) : "memory");
                                    }
                                    int _stage_output_finished_7 = (int)((unsigned int)(gemm_acc_tiles + 1) % 2);
                                    int _parity_output_finished_7 = (int)((unsigned int)(gemm_acc_tiles + 1) / 2 % 2) ^ 1;
                                    mbarrier_wait(output_finished_addr + (_stage_output_finished_7) * 8, _parity_output_finished_7);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    #pragma unroll 1
                                    for (int idx_20 = 0; idx_20 < head_3; idx_20++) {
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_608 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_20) % 4) * 1024;
                                        int _smem_desc_lo_609 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_20) % 4) * 1024;
#else
                                        int _smem_desc_lo_608 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_20) % 4) * 1024;
                                        int _smem_desc_lo_609 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_20) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_608), "r"(_smem_desc_lo_609), "r"((tmem_accumulator + (256))), "r"(((idx_20 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_20) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_21 = head_3; idx_21 < iterations_3 - tail_4; idx_21++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_21) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_10 = (int)((unsigned int)(gemm_wide + idx_21) % 4);
                                        int _parity_gemm_arrived_10 = (int)((unsigned int)(gemm_wide + idx_21) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_10) * 8, _parity_gemm_arrived_10);
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_610 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                        int _smem_desc_lo_611 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
#else
                                        int _smem_desc_lo_610 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                        int _smem_desc_lo_611 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_21) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_610), "r"(_smem_desc_lo_611), "r"(tmem_accumulator), "r"(((idx_21 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_612 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                        int _smem_desc_lo_613 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_21) % 4) * 1024;
#else
                                        int _smem_desc_lo_612 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_21) % 4) * 1024;
                                        int _smem_desc_lo_613 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_21) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_612), "r"(_smem_desc_lo_613), "r"((tmem_accumulator + (256))), "r"(((idx_21 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_21) % 4)) * 8, (uint16_t)(3));
                                    }
                                    #pragma unroll 1
                                    for (int idx_22 = iterations_3 - tail_4; idx_22 < iterations_3; idx_22++) {
                                        mbarrier_arrive_expect_tx(gemm_arrived_addr + ((int)((unsigned int)(gemm_wide + idx_22) % 4)) * 8, 98304);
                                        int _stage_gemm_arrived_11 = (int)((unsigned int)(gemm_wide + idx_22) % 4);
                                        int _parity_gemm_arrived_11 = (int)((unsigned int)(gemm_wide + idx_22) / 4 % 2);
                                        mbarrier_wait(gemm_arrived_addr + (_stage_gemm_arrived_11) * 8, _parity_gemm_arrived_11);
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_614 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_22) % 4) * 1024;
                                        int _smem_desc_lo_615 = (((b_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_22) % 4) * 1024;
#else
                                        int _smem_desc_lo_614 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_22) % 4) * 1024;
                                        int _smem_desc_lo_615 = (((b_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_22) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_614), "r"(_smem_desc_lo_615), "r"(tmem_accumulator), "r"(((idx_22 == 0) ? 0 : 1)) : "memory");
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr, (uint16_t)(3));
                                    #pragma unroll 1
                                    for (int idx_23 = iterations_3 - tail_4; idx_23 < iterations_3; idx_23++) {
#if __CUDA_ARCH__ == 1070
                                        int _smem_desc_lo_616 = (((a_smem_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_23) % 4) * 1024;
                                        int _smem_desc_lo_617 = (((b_hi_addr) >> 4) & 0x7FFF) + ((gemm_wide + idx_23) % 4) * 1024;
#else
                                        int _smem_desc_lo_616 = (((a_smem_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_23) % 4) * 1024;
                                        int _smem_desc_lo_617 = (((b_hi_addr) >> 4) & 0x3FFF) + ((gemm_wide + idx_23) % 4) * 1024;
#endif
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
            :: "r"(_smem_desc_lo_616), "r"(_smem_desc_lo_617), "r"((tmem_accumulator + (256))), "r"(((idx_23 == 0) ? 0 : 1)) : "memory");
                                        tcgen05_commit_cg2_multicast(gemm_finished_addr + ((int)((unsigned int)(gemm_wide + idx_23) % 4)) * 8, (uint16_t)(3));
                                    }
                                    tcgen05_commit_cg2_multicast(output_arrived_addr + 8, (uint16_t)(3));
                                }
                            }
                        } else if (tid < 128) {
                            int _parity_output_arrived_6 = (int)((unsigned int)gemm_out0 % 2);
                            mbarrier_wait(output_arrived_addr, _parity_output_arrived_6);
                            unsigned int packed_1[128];
                            #pragma unroll
                            for (int chunk_4 = 0; chunk_4 < 8; chunk_4++) {
                                #pragma unroll
                                for (int sub_2 = 0; sub_2 < 2; sub_2++) {
                                    unsigned int address_20 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_2 * 16 << 16) + (unsigned int)(chunk_4 * 32);
                                    float _tmem_load_32_32[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_32[15]))
                                        : "r"(address_20));
                                    #pragma unroll
                                    for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                                        packed_1[chunk_4 * 16 + sub_2 * 8 + pair_2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_32[pair_2 * 2], _tmem_load_32_32[pair_2 * 2 + 1])));
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
                                        :: "r"((output_finished_addr + ((int)((unsigned int)gemm_acc_tiles % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            if (tid == 0) {
                                int previous_offset_2 = (macro_1 + 1) * macro_size;
                                int output_row_1 = x_3 * 256 + cta_rank_0 * 128;
                                int _min_618 = ((macro_size) < (tokens - previous_offset_2) ? (macro_size) : (tokens - previous_offset_2));
                                if (output_row_1 < _min_618) {
                                    bool enabled_value_6 = 1;
                                    if (enabled_value_6 != 0) {
                                        int poll_7_first;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_first) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                        int poll_7 = poll_7_first;
                                        while (poll_7 < 8 * ((hidden + 1023) / 1024)) {
                                            asm volatile("nanosleep.u32 %0;" :: "r"(16));
                                            int poll_7_next;
                                            asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(poll_7_next) : "l"(y_done + ((previous_offset_2 + output_row_1) / 128)) : "memory");
                                            poll_7 = poll_7_next;
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
                                int warp_14 = tid / 32;
                                int lane_14 = tid % 32;
                                #pragma unroll
                                for (int half_18 = 0; half_18 < 2; half_18++) {
                                    #pragma unroll
                                    for (int col_tile_18 = 0; col_tile_18 < 2; col_tile_18++) {
                                        int row_25 = warp_14 * 32 + half_18 * 16 + lane_14 % 16;
                                        int col_22 = col_tile_18 * 16 + lane_14 / 16 * 8;
                                        unsigned int address_21 = d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192) + (unsigned int)((row_25 * 32 + col_22) * 2);
                                        address_21 = address_21 ^ (address_21 & 511) >> 7 << 4;
                                        int offset_20 = chunk_5 * 16 + half_18 * 8 + col_tile_18 * 4;
                                        uint32_t _stmatrix_addr_619 = static_cast<uint32_t>(address_21);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_619), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_20])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_20 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_20 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[offset_20 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 1, 128;" ::: "memory");
                                if (tid == 0) {
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    tma_store_5d((&y_routed), 0, x_3 * 256 + cta_rank_0 * 128, y_3 * 2 * 8 + chunk_5, 0, 0, d_smem_addr + (unsigned int)(chunk_5 % 3 * 8192));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            int _parity_output_arrived_7 = (int)((unsigned int)gemm_out1 % 2);
                            mbarrier_wait(output_arrived_addr + 8, _parity_output_arrived_7);
                            if (has_hi_3 != 0) {
                                unsigned int packed_1_1[128];
                                #pragma unroll
                                for (int chunk_6 = 0; chunk_6 < 8; chunk_6++) {
                                    #pragma unroll
                                    for (int sub_3 = 0; sub_3 < 2; sub_3++) {
                                        unsigned int address_22 = taddr_1 + (unsigned int)(tid / 32 * 32 + sub_3 * 16 << 16) + 256 + (unsigned int)(chunk_6 * 32);
                                        float _tmem_load_32_33[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32_33[15]))
                                            : "r"(address_22));
                                        #pragma unroll
                                        for (int pair_3 = 0; pair_3 < 8; pair_3++) {
                                            packed_1_1[chunk_6 * 16 + sub_3 * 8 + pair_3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_32_33[pair_3 * 2], _tmem_load_32_33[pair_3 * 2 + 1])));
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
                                            :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
                                    }
                                }
                                #pragma unroll
                                for (int chunk_7 = 0; chunk_7 < 8; chunk_7++) {
                                    if (tid == 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 2;");
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    int warp_15 = tid / 32;
                                    int lane_15 = tid % 32;
                                    #pragma unroll
                                    for (int half_19 = 0; half_19 < 2; half_19++) {
                                        #pragma unroll
                                        for (int col_tile_19 = 0; col_tile_19 < 2; col_tile_19++) {
                                            int row_26 = warp_15 * 32 + half_19 * 16 + lane_15 % 16;
                                            int col_23 = col_tile_19 * 16 + lane_15 / 16 * 8;
                                            unsigned int address_23 = d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192) + (unsigned int)((row_26 * 32 + col_23) * 2);
                                            address_23 = address_23 ^ (address_23 & 511) >> 7 << 4;
                                            int offset_21 = chunk_7 * 16 + half_19 * 8 + col_tile_19 * 4;
                                            uint32_t _stmatrix_addr_620 = static_cast<uint32_t>(address_23);
                                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                                :: "r"(_stmatrix_addr_620), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_21])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_21 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_21 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1_1[offset_21 + 3]))
                                                : "memory");
                                        }
                                    }
                                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                                    if (tid == 0) {
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                        tma_store_5d((&y_routed), 0, x_3 * 256 + cta_rank_0 * 128, (y_3 * 2 + 1) * 8 + chunk_7, 0, 0, d_smem_addr + (unsigned int)((8 + chunk_7) % 3 * 8192));
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            if (has_hi_3 == 0) {
                                if (tid == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((output_finished_addr + ((int)((unsigned int)(gemm_acc_tiles + 1) % 2)) * 8) & 0xFEFFFFFF) : "memory");
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
                    gemm_wide += iterations_3 * (int)(expert_3 >= 0);
                    gemm_out0 += (int)(expert_3 >= 0);
                    gemm_out1 += (int)(expert_3 >= 0);
                    gemm_acc_tiles += 2 * (int)(expert_3 >= 0);
                }
            }
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
