/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 80
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 72
#define TMEM_TMEM_O0_OFFSET 8
#define TMEM_TMEM_O1_OFFSET 16
#define TMEM_TMEM_O2_OFFSET 24
#define TMEM_TMEM_O3_OFFSET 32
#define TMEM_TMEM_O0B_OFFSET 40
#define TMEM_TMEM_O1B_OFFSET 48
#define TMEM_TMEM_O2B_OFFSET 56
#define TMEM_TMEM_O3B_OFFSET 64
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 2048
#define SMEM_SMEM_Q_STRIDE 2048
#define SMEM_SMEM_Q_HALF_OFF 1024
#define SMEM_SMEM_Q_HALF_STAGE_BYTES 4096
#define SMEM_SMEM_Q_HALF_STRIDE 4096
#define SMEM_SMEM_KV_OFF 9216
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 9216
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 140288
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_SMEM_P_HALF_OFF 140288
#define SMEM_SMEM_P_HALF_STAGE_BYTES 1024
#define SMEM_SMEM_P_HALF_STRIDE 1024
#define SMEM_SMEM_EXCH_OFF 142336
#define SMEM_SMEM_EXCH_STAGE_BYTES 128
#define SMEM_SMEM_EXCH_STRIDE 128
#define SMEM_SMEM_EXCH1_OFF 143040
#define SMEM_SMEM_EXCH1_STAGE_BYTES 128
#define SMEM_SMEM_EXCH1_STRIDE 128
#define SMEM_SMEM_SUM_OFF 142464
#define SMEM_SMEM_SUM_STAGE_BYTES 32
#define SMEM_SMEM_SUM_STRIDE 32
#define SMEM_SMEM_ALPHA_OFF 143168
#define SMEM_SMEM_ALPHA_STAGE_BYTES 32
#define SMEM_SMEM_ALPHA_STRIDE 32
#define SMEM_SMEM_INDICES_OFF 142528
#define SMEM_SMEM_INDICES_STAGE_BYTES 512
#define SMEM_SMEM_INDICES_STRIDE 512
#define SMEM_TOTAL 143232
#define THREADS 512
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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


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



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
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

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_f81486848bf3b4e05349(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ partition_arrivals, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int sparse_topk, int num_splits, int num_head_tiles, int has_sinks, int ragged_query, int max_q_len, int batch_size)
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
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define s_full_addr (mbar_base + 40)
    #define p_full_addr (mbar_base + 56)
    #define stats_full_addr (mbar_base + 72)
    #define o_full_lo_addr (mbar_base + 80)
    #define o_full_hi_addr (mbar_base + 96)
    #define tmem_dealloc_addr (mbar_base + 112)
    #define partial_done_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_q_half = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_half_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kv_addr = smem + 9216;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_v_addr = smem + 9216;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int smem_p_addr = smem + 140288;
    __nv_bfloat16* smem_p_half = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int smem_p_half_addr = smem + 140288;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 142336);
    const int smem_exch_addr = smem + 142336;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 143040);
    const int smem_exch1_addr = smem + 143040;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 142464);
    const int smem_sum_addr = smem + 142464;
    float* smem_alpha = reinterpret_cast<float*>(smem_raw + 143168);
    const int smem_alpha_addr = smem + 143168;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 142528);
    const int smem_indices_addr = smem + 142528;
    if (warp == 13) {
        int early_split_work = blockIdx.x / num_head_tiles;
        int early_split_idx = early_split_work % num_splits;
        int early_query_idx = early_split_work / num_splits;
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(sparse_topk_lens + early_query_idx)));
        int w120_total_groups = (sparse_topk + 3) / 4;
        int w120_full_splits = w120_total_groups / 32;
        int w120_split32 = early_split_idx * 32;
        int w120_next32 = w120_split32 + 32;
        int w120_swa_lo = ((w120_split32 / w120_full_splits) < (32) ? (w120_split32 / w120_full_splits) : (32));
        int w120_swa_hi = ((w120_next32 / w120_full_splits) < (32) ? (w120_next32 / w120_full_splits) : (32));
        int w120_nswa = w120_swa_hi - w120_swa_lo;
        int w120_comp_lo = w120_split32 - w120_swa_lo;
        int w120_comp_left = w120_total_groups - 32 - w120_comp_lo;
        int w120_nswa_left = w120_swa_lo - w120_swa_hi + 32;
        int w120_ncomp = ((w120_nswa_left) < (w120_comp_left) ? (w120_nswa_left) : (w120_comp_left));
        int lane_in_swa = ((w120_nswa > lane) ? 1 : 0);
        int lane_col_base = ((lane_in_swa == 1) ? (w120_swa_lo + lane) * 4 : 128 + (w120_comp_lo + lane - w120_nswa) * 4);
        int* row_ptr = ((lane_in_swa == 1) ? (swa_indices + (early_query_idx * swa_index_stride + lane_col_base)) : (compressed_indices + (early_query_idx * compressed_index_stride + (lane_col_base - 128))));
        int vector_offset = lane * 4;
        int row_stride = ((lane_in_swa == 1) ? swa_index_stride : compressed_index_stride);
        int row_vec_misaligned = early_query_idx * row_stride & 3;
        if (lane_col_base + 3 < sparse_topk && row_vec_misaligned == 0) {
            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                :: "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)), "l"(row_ptr));
        } else {
            int staged_rows[4];
            #pragma unroll
            for (int row_i = 0; row_i < 4; row_i++) {
                staged_rows[row_i] = -1;
                if (lane_col_base + row_i < sparse_topk) {
                    staged_rows[row_i] = row_ptr[row_i];
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)), "r"(staged_rows[0]), "r"(staged_rows[1]), "r"(staged_rows[2]), "r"(staged_rows[3]) : "memory");
        }
        asm volatile("cp.async.commit_group;");
    }

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
#if __CUDA_ARCH__ == 1030
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            mbarrier_init(smem + 64, 128);
            // stats_full: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // o_full_lo: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // o_full_hi: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // tmem_dealloc: 1 barriers, init_count=256
            mbarrier_init(smem + 112, 256);
            // partial_done: 1 barriers, init_count=256
            mbarrier_init(smem + 120, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
        // q_full: 1 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 4 barriers, init_count=1
        // s_full: 2 barriers, init_count=1
        // p_full: 2 barriers, init_count=128
        // stats_full: 1 barriers, init_count=128
        // o_full_lo: 2 barriers, init_count=1
        // o_full_hi: 2 barriers, init_count=1
        // tmem_dealloc: 1 barriers, init_count=256
        // partial_done: 1 barriers, init_count=256
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 256;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(14), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(7), "r"((uint32_t)(1)));
        if (lane < 16) {
            mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
#endif
        }
#if __CUDA_ARCH__ == 1030
#else
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
#if __CUDA_ARCH__ == 1030
    if (warp == 0) {
#else
    if (warp == 12) {
#endif
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_s1 = taddr + 72;
    const int tmem_tmem_o0 = taddr + 8;
    const int tmem_tmem_o1 = taddr + 16;
    const int tmem_tmem_o2 = taddr + 24;
    const int tmem_tmem_o3 = taddr + 32;
    const int tmem_tmem_o0b = taddr + 40;
    const int tmem_tmem_o1b = taddr + 48;
    const int tmem_tmem_o2b = taddr + 56;
    const int tmem_tmem_o3b = taddr + 64;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int work_idx = blockIdx.x;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int _max_0 = ((sparse_topk_lens[query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx] + sparse_topk_lens_offset) : (0));
            int active_topk = ((_max_0) < (sparse_topk) ? (_max_0) : (sparse_topk));
            int head_base = head_tile * 8;
            const int key_lane = lane % 16;
            const int head_lo = lane / 16 * 4;
            const int half_pos = warp * 16 + key_lane;
            const int pos0 = half_pos;
            const int pos1 = 64 + half_pos;
            int w120_total_groups_1 = (sparse_topk + 3) / 4;
            int w120_full_splits_1 = w120_total_groups_1 / 32;
            int w120_split32_1 = split_idx * 32;
            int w120_next32_1 = w120_split32_1 + 32;
            int w120_swa_lo_1 = ((w120_split32_1 / w120_full_splits_1) < (32) ? (w120_split32_1 / w120_full_splits_1) : (32));
            int w120_swa_hi_1 = ((w120_next32_1 / w120_full_splits_1) < (32) ? (w120_next32_1 / w120_full_splits_1) : (32));
            int w120_nswa_1 = w120_swa_hi_1 - w120_swa_lo_1;
            int w120_comp_lo_1 = w120_split32_1 - w120_swa_lo_1;
            int w120_comp_left_1 = w120_total_groups_1 - 32 - w120_comp_lo_1;
            int w120_nswa_left_1 = w120_swa_lo_1 - w120_swa_hi_1 + 32;
            int w120_ncomp_1 = ((w120_nswa_left_1) < (w120_comp_left_1) ? (w120_nswa_left_1) : (w120_comp_left_1));
            int w120_h0_group = pos0 / 4;
            int w120_h0_row = pos0 % 4;
            int global_token0 = ((w120_h0_group < w120_nswa_1) ? (w120_swa_lo_1 + w120_h0_group) * 4 + w120_h0_row : 128 + (w120_comp_lo_1 + w120_h0_group - w120_nswa_1) * 4 + w120_h0_row);
            int w120_h0_group_0 = pos1 / 4;
            int w120_h0_row_0 = pos1 % 4;
            int global_token1 = ((w120_h0_group_0 < w120_nswa_1) ? (w120_swa_lo_1 + w120_h0_group_0) * 4 + w120_h0_row_0 : 128 + (w120_comp_lo_1 + w120_h0_group_0 - w120_nswa_1) * 4 + w120_h0_row_0);
            float w29_scale_log2 = bmm1_scale[0] * 1.442695f;
            float w29_sinks[4];
            w29_sinks[0] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo < num_heads) {
                w29_sinks[0] = sinks[head_base + head_lo];
            }
            w29_sinks[1] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo + 1 < num_heads) {
                w29_sinks[1] = sinks[head_base + head_lo + 1];
            }
            w29_sinks[2] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo + 2 < num_heads) {
                w29_sinks[2] = sinks[head_base + head_lo + 2];
            }
            w29_sinks[3] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo + 3 < num_heads) {
                w29_sinks[3] = sinks[head_base + head_lo + 3];
            }
            int token_valid0 = 1;
            if (global_token0 >= active_topk) {
                token_valid0 = 0;
            }
            int token_valid1 = 1;
            if (global_token1 >= active_topk) {
                token_valid1 = 0;
            }
            int query_batch = query_idx / max_q_len;
            int query_offset = query_idx - query_batch * max_q_len;
            int query_length = max_q_len;
            if (ragged_query != 0) {
                query_batch = 0;
                #pragma unroll 2
                for (int chunk = 0; chunk < (batch_size + 31) / 32; chunk++) {
                    int lane_entry = chunk * 32 + lane + 1;
                    int lane_load = ((lane_entry) < (batch_size) ? (lane_entry) : (batch_size));
                    unsigned int started = __ballot_sync(0xFFFFFFFF, lane_entry <= batch_size && query_idx >= cum_seq_lens_q[lane_load]);
                    query_batch = query_batch + (int)(unsigned int)__popc(started);
                }
                int query_begin = cum_seq_lens_q[query_batch];
                query_length = cum_seq_lens_q[query_batch + 1] - query_begin;
                query_offset = query_idx - query_begin;
            }
            int visible = seq_lens[query_batch] - query_length + query_offset + 1;
            if (visible < 0) {
                visible = 0;
            }
            if (visible > 128) {
                visible = 128;
            }
            int swa_visible = visible;
            if (global_token0 < 128 && global_token0 >= swa_visible) {
                token_valid0 = 0;
            }
            if (global_token1 < 128 && global_token1 >= swa_visible) {
                token_valid1 = 0;
            }
            const int row_addr = warp * 32 << 16;
            unsigned int _phase_s_full_0 = 0;
            mbarrier_wait(s_full_addr, _phase_s_full_0);
            _phase_s_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x4.b32"
                " {%0, %1, %2, %3}, [%4], 4;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr + (unsigned int)row_addr));
            int staged_index0 = smem_indices[pos0];
            if (staged_index0 < 0) {
                token_valid0 = 0;
            }
            if (token_valid0 == 0) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    _tmem_load_0[j] = -CUDART_INF_F;
                }
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                if (head_base + head_lo + j_1 >= num_heads) {
                    _tmem_load_0[j_1] = -CUDART_INF_F;
                }
            }
            float partial_max0[4];
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, _tmem_load_0[j_2], 8);
                float max_1;
                asm("max.f32 %0, %1, %2;" : "=f"(max_1) : "f"(_tmem_load_0[j_2]), "f"(_shfl_xor_sync_1));
                float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, max_1, 4);
                float max_2;
                asm("max.f32 %0, %1, %2;" : "=f"(max_2) : "f"(max_1), "f"(_shfl_xor_sync_2));
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, max_2, 2);
                float max_3;
                asm("max.f32 %0, %1, %2;" : "=f"(max_3) : "f"(max_2), "f"(_shfl_xor_sync_3));
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, max_3, 1);
                partial_max0[j_2] = max_noftz(max_3, _shfl_xor_sync_4);
            }
            if (key_lane == 0) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 4; j_3++) {
                    smem_exch[warp * 8 + head_lo + j_3] = partial_max0[j_3];
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            float exch_max0 = 0.0f;
            if (lane < 8) {
                int h_lane = lane;
                exch_max0 = max_noftz(max_noftz(smem_exch[h_lane], smem_exch[8 + h_lane]), max_noftz(smem_exch[16 + h_lane], smem_exch[24 + h_lane]));
            }
            float tile_max0[4];
            #pragma unroll
            for (int j_4 = 0; j_4 < 4; j_4++) {
                float _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, exch_max0, head_lo + j_4);
                tile_max0[j_4] = _shfl_sync_5;
            }
            float max0_values[4];
            float max0_scaled[4];
            float exp0[4];
            float warp_sum0[4];
            #pragma unroll
            for (int j_5 = 0; j_5 < 4; j_5++) {
                int global_head = head_base + head_lo + j_5;
                float row_max0 = tile_max0[j_5];
                if (has_sinks != 0 && split_idx == 0 && global_head < num_heads) {
                    float sink_unscaled = w29_sinks[j_5] * 1.442695f / w29_scale_log2;
                    row_max0 = max_noftz(row_max0, sink_unscaled);
                }
                max0_values[j_5] = row_max0;
                float safe_max0 = ((row_max0 == -CUDART_INF_F) ? 0.0f : row_max0);
                max0_scaled[j_5] = safe_max0 * w29_scale_log2;
                float _exp2_6 = approx_exp2(__fmaf_rn(_tmem_load_0[j_5], w29_scale_log2, -max0_scaled[j_5]));
                exp0[j_5] = ((_tmem_load_0[j_5] != -CUDART_INF_F) ? _exp2_6 : 0.0f);
                float shfl_xor = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5], 8);
                float shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5] + shfl_xor, 4);
                float shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5] + shfl_xor + shfl_xor_1, 2);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5] + shfl_xor + shfl_xor_1 + shfl_xor_2, 1);
                warp_sum0[j_5] = exp0[j_5] + shfl_xor + shfl_xor_1 + shfl_xor_2 + _shfl_xor_sync_7;
            }
            #pragma unroll
            for (int j_6 = 0; j_6 < 4; j_6++) {
                {
                    __nv_bfloat16 _bval_8 = __float2bfloat16_rn(exp0[j_6]);
                    uint16_t _bits_8 = *(uint16_t*)&_bval_8;
                    uint32_t _addr_8 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(half_pos / 64 * 1024 + (head_lo + j_6) * 128 + half_pos % 64 * 2 ^ (half_pos / 64 * 1024 + (head_lo + j_6) * 128 + half_pos % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_8), "h"(_bits_8) : "memory");
                }
            }
            asm volatile("fence.proxy.async;" ::: "memory");
            mbarrier_arrive(p_full_addr);
            unsigned int _phase_s_full_1 = 0;
            mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
            _phase_s_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_1[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x4.b32"
                " {%0, %1, %2, %3}, [%4], 4;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                : "r"(taddr + 72 + (unsigned int)row_addr));
            int staged_index1 = smem_indices[pos1];
            if (staged_index1 < 0) {
                token_valid1 = 0;
            }
            if (token_valid1 == 0) {
                #pragma unroll
                for (int j_7 = 0; j_7 < 4; j_7++) {
                    _tmem_load_1[j_7] = -CUDART_INF_F;
                }
            }
            #pragma unroll
            for (int j_8 = 0; j_8 < 4; j_8++) {
                if (head_base + head_lo + j_8 >= num_heads) {
                    _tmem_load_1[j_8] = -CUDART_INF_F;
                }
            }
            float partial_max1[4];
            #pragma unroll
            for (int j_9 = 0; j_9 < 4; j_9++) {
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, _tmem_load_1[j_9], 8);
                float max_1_1;
                asm("max.f32 %0, %1, %2;" : "=f"(max_1_1) : "f"(_tmem_load_1[j_9]), "f"(_shfl_xor_sync_9));
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, max_1_1, 4);
                float max_2_1;
                asm("max.f32 %0, %1, %2;" : "=f"(max_2_1) : "f"(max_1_1), "f"(_shfl_xor_sync_10));
                float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, max_2_1, 2);
                float max_3_1;
                asm("max.f32 %0, %1, %2;" : "=f"(max_3_1) : "f"(max_2_1), "f"(_shfl_xor_sync_11));
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, max_3_1, 1);
                partial_max1[j_9] = max_noftz(max_3_1, _shfl_xor_sync_12);
            }
            if (key_lane == 0) {
                #pragma unroll
                for (int j_10 = 0; j_10 < 4; j_10++) {
                    smem_exch1[warp * 8 + head_lo + j_10] = partial_max1[j_10];
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            float exch_max1 = 0.0f;
            if (lane < 8) {
                int h_lane_1 = lane;
                exch_max1 = max_noftz(max_noftz(smem_exch1[h_lane_1], smem_exch1[8 + h_lane_1]), max_noftz(smem_exch1[16 + h_lane_1], smem_exch1[24 + h_lane_1]));
            }
            float tile_max1[4];
            #pragma unroll
            for (int j_11 = 0; j_11 < 4; j_11++) {
                float _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, exch_max1, head_lo + j_11);
                tile_max1[j_11] = _shfl_sync_13;
            }
            float row_max_values[4];
            float max_scaled[4];
            float alpha0[4];
            float exp1[4];
            float warp_sum[4];
            #pragma unroll
            for (int j_12 = 0; j_12 < 4; j_12++) {
                float row_max;
                asm("max.f32 %0, %1, %2;" : "=f"(row_max) : "f"(max0_values[j_12]), "f"(tile_max1[j_12]));
                row_max_values[j_12] = row_max;
                float safe_max = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                max_scaled[j_12] = safe_max * w29_scale_log2;
                float _exp2_14 = approx_exp2(max0_scaled[j_12] - max_scaled[j_12]);
                alpha0[j_12] = ((max0_values[j_12] == -CUDART_INF_F) ? 0.0f : _exp2_14);
                float _exp2_15 = approx_exp2(__fmaf_rn(_tmem_load_1[j_12], w29_scale_log2, -max_scaled[j_12]));
                exp1[j_12] = ((_tmem_load_1[j_12] != -CUDART_INF_F) ? _exp2_15 : 0.0f);
                float shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12], 8);
                float shfl_xor_1_1 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12] + shfl_xor_3, 4);
                float shfl_xor_2_1 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12] + shfl_xor_3 + shfl_xor_1_1, 2);
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12] + shfl_xor_3 + shfl_xor_1_1 + shfl_xor_2_1, 1);
                float warp_sum1 = exp1[j_12] + shfl_xor_3 + shfl_xor_1_1 + shfl_xor_2_1 + _shfl_xor_sync_16;
                warp_sum[j_12] = __fmaf_rn(warp_sum0[j_12], alpha0[j_12], warp_sum1);
            }
            #pragma unroll
            for (int j_13 = 0; j_13 < 4; j_13++) {
                {
                    __nv_bfloat16 _bval_17 = __float2bfloat16_rn(exp1[j_13]);
                    uint16_t _bits_17 = *(uint16_t*)&_bval_17;
                    uint32_t _addr_17 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(half_pos / 64 * 1024 + (8 + head_lo + j_13) * 128 + half_pos % 64 * 2 ^ (half_pos / 64 * 1024 + (8 + head_lo + j_13) * 128 + half_pos % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_17), "h"(_bits_17) : "memory");
                }
            }
            asm volatile("fence.proxy.async;" ::: "memory");
            mbarrier_arrive(p_full_addr + 8);
            if (key_lane == 0) {
                #pragma unroll
                for (int j_14 = 0; j_14 < 4; j_14++) {
                    smem_exch[warp * 8 + head_lo + j_14] = warp_sum[j_14];
                }
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            if (warp == 0) {
                if (key_lane < 4) {
                    int h_lane_2 = head_lo + key_lane;
                    int global_head_1 = head_base + h_lane_2;
                    float row_sum = smem_exch[h_lane_2] + smem_exch[8 + h_lane_2] + smem_exch[16 + h_lane_2] + smem_exch[24 + h_lane_2];
                    float row_max_1 = row_max_values[key_lane];
                    if (has_sinks != 0 && split_idx == 0 && global_head_1 < num_heads) {
                        float sink_scaled = w29_sinks[key_lane] * 1.442695f;
                        float _exp2_18 = approx_exp2(sink_scaled - max_scaled[key_lane]);
                        row_sum = row_sum + _exp2_18;
                    }
                    smem_sum[h_lane_2] = row_sum;
                    smem_alpha[h_lane_2] = alpha0[key_lane];
                    if (global_head_1 < num_heads) {
                        int lse_offset = (query_idx * num_heads + global_head_1) * num_splits + split_idx;
                        float _log2_19;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_19) : "f"(row_sum));
                        partial_lse[lse_offset] = ((row_sum > 0.0f) ? __fmaf_rn(row_max_1, w29_scale_log2, _log2_19) : -CUDART_INF_F);
                    }
                }
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            mbarrier_arrive(stats_full_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_lo_main
            int work_idx_1 = blockIdx.x;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 8;
            const int local_warp = warp - 4;
            const int d_lane = local_warp * 32 + lane;
            const int row_addr_1 = local_warp * 32 << 16;
            float output_scale = bmm2_scale[0];
            int merge_group = query_idx_1 * num_head_tiles + head_tile_1;
            int _max_0_1 = ((sparse_topk_lens[query_idx_1] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_1] + sparse_topk_lens_offset) : (0));
            int w29_active = ((_max_0_1) < (sparse_topk) ? (_max_0_1) : (sparse_topk));
            int w120_total_groups_2 = (sparse_topk + 3) / 4;
            int w120_full_splits_2 = w120_total_groups_2 / 32;
            int w120_split32_2 = split_idx_1 * 32;
            int w120_next32_2 = w120_split32_2 + 32;
            int w120_swa_lo_2 = ((w120_split32_2 / w120_full_splits_2) < (32) ? (w120_split32_2 / w120_full_splits_2) : (32));
            int w120_swa_hi_2 = ((w120_next32_2 / w120_full_splits_2) < (32) ? (w120_next32_2 / w120_full_splits_2) : (32));
            int w120_nswa_2 = w120_swa_hi_2 - w120_swa_lo_2;
            int w120_comp_lo_2 = w120_split32_2 - w120_swa_lo_2;
            int w120_comp_left_2 = w120_total_groups_2 - 32 - w120_comp_lo_2;
            int w120_nswa_left_2 = w120_swa_lo_2 - w120_swa_hi_2 + 32;
            int w120_ncomp_2 = ((w120_nswa_left_2) < (w120_comp_left_2) ? (w120_nswa_left_2) : (w120_comp_left_2));
            int w120_active_group = w29_active / 4;
            int w120_active_rem = w29_active % 4;
            int w120_swa_off = w120_active_group - w120_swa_lo_2;
            int _max_1 = ((w120_swa_off) > (0) ? (w120_swa_off) : (0));
            int w120_full_swa = ((_max_1) < (w120_nswa_2) ? (_max_1) : (w120_nswa_2));
            int w120_comp_off = w120_active_group - (w120_comp_lo_2 + 32);
            int _max_2 = ((w120_comp_off) > (0) ? (w120_comp_off) : (0));
            int w120_full_comp = ((_max_2) < (w120_ncomp_2) ? (_max_2) : (w120_ncomp_2));
            int w120_full_rows = (w120_full_swa + w120_full_comp) * 4;
            int w29_valid_rows = w120_full_rows + ((w120_swa_off >= 0 && w120_swa_off < w120_nswa_2) ? w120_active_rem : ((w120_comp_off >= 0 && w120_comp_off < w120_ncomp_2) ? w120_active_rem : 0));
            int w29_valid_groups = (w29_valid_rows + 3) / 4;
            asm volatile("barrier.sync 1, 352;" ::: "memory");
            if (local_warp == 0) {
                int w29_u2_quad = lane % 16;
                int w29_u2_quad_pos = w29_u2_quad;
                int w29_u2_atom = 2 + lane / 16;
                if (w29_u2_quad_pos < w29_valid_groups) {
                    int w29_u2_raw0 = smem_indices[w29_u2_quad_pos * 4];
                    int w29_u2_row0 = ((w29_u2_raw0 >= 0) ? w29_u2_raw0 : 0);
                    int w29_u2_raw1 = smem_indices[w29_u2_quad_pos * 4 + 1];
                    int w29_u2_row1 = ((w29_u2_raw1 >= 0) ? w29_u2_raw1 : 0);
                    int w29_u2_raw2 = smem_indices[w29_u2_quad_pos * 4 + 2];
                    int w29_u2_row2 = ((w29_u2_raw2 >= 0) ? w29_u2_raw2 : 0);
                    int w29_u2_raw3 = smem_indices[w29_u2_quad_pos * 4 + 3];
                    int w29_u2_row3 = ((w29_u2_raw3 >= 0) ? w29_u2_raw3 : 0);
                    int w120_group_in_swa_w29_u2 = ((w29_u2_quad_pos < w120_nswa_2) ? 1 : 0);
                    if (w120_group_in_swa_w29_u2 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u2_atom * 8192) + (unsigned int)(w29_u2_quad * 512), (&tmap_swa_kv), 256 + w29_u2_atom * 64, w29_u2_row0, w29_u2_row1, w29_u2_row2, w29_u2_row3, kv_full_addr + 8);
                    }
                    if (w120_group_in_swa_w29_u2 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u2_atom * 8192) + (unsigned int)(w29_u2_quad * 512), (&tmap_compressed_kv), 256 + w29_u2_atom * 64, w29_u2_row0, w29_u2_row1, w29_u2_row2, w29_u2_row3, kv_full_addr + 8);
                    }
                }
            }
            if (local_warp == 1) {
                int w29_u4_quad = lane % 16;
                int w29_u4_quad_pos = 16 + w29_u4_quad;
                int w29_u4_atom = lane / 16;
                if (w29_u4_quad_pos < w29_valid_groups) {
                    int w29_u4_raw0 = smem_indices[w29_u4_quad_pos * 4];
                    int w29_u4_row0 = ((w29_u4_raw0 >= 0) ? w29_u4_raw0 : 0);
                    int w29_u4_raw1 = smem_indices[w29_u4_quad_pos * 4 + 1];
                    int w29_u4_row1 = ((w29_u4_raw1 >= 0) ? w29_u4_raw1 : 0);
                    int w29_u4_raw2 = smem_indices[w29_u4_quad_pos * 4 + 2];
                    int w29_u4_row2 = ((w29_u4_raw2 >= 0) ? w29_u4_raw2 : 0);
                    int w29_u4_raw3 = smem_indices[w29_u4_quad_pos * 4 + 3];
                    int w29_u4_row3 = ((w29_u4_raw3 >= 0) ? w29_u4_raw3 : 0);
                    int w120_group_in_swa_w29_u4 = ((w29_u4_quad_pos < w120_nswa_2) ? 1 : 0);
                    if (w120_group_in_swa_w29_u4 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u4_atom * 8192) + (unsigned int)(w29_u4_quad * 512), (&tmap_swa_kv), w29_u4_atom * 64, w29_u4_row0, w29_u4_row1, w29_u4_row2, w29_u4_row3, kv_full_addr + 16);
                    }
                    if (w120_group_in_swa_w29_u4 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u4_atom * 8192) + (unsigned int)(w29_u4_quad * 512), (&tmap_compressed_kv), w29_u4_atom * 64, w29_u4_row0, w29_u4_row1, w29_u4_row2, w29_u4_row3, kv_full_addr + 16);
                    }
                }
            }
            if (local_warp == 2) {
                int w29_u4_quad_1 = lane % 16;
                int w29_u4_quad_pos_1 = 16 + w29_u4_quad_1;
                int w29_u4_atom_1 = 2 + lane / 16;
                if (w29_u4_quad_pos_1 < w29_valid_groups) {
                    int w29_u4_raw0_1 = smem_indices[w29_u4_quad_pos_1 * 4];
                    int w29_u4_row0_1 = ((w29_u4_raw0_1 >= 0) ? w29_u4_raw0_1 : 0);
                    int w29_u4_raw1_1 = smem_indices[w29_u4_quad_pos_1 * 4 + 1];
                    int w29_u4_row1_1 = ((w29_u4_raw1_1 >= 0) ? w29_u4_raw1_1 : 0);
                    int w29_u4_raw2_1 = smem_indices[w29_u4_quad_pos_1 * 4 + 2];
                    int w29_u4_row2_1 = ((w29_u4_raw2_1 >= 0) ? w29_u4_raw2_1 : 0);
                    int w29_u4_raw3_1 = smem_indices[w29_u4_quad_pos_1 * 4 + 3];
                    int w29_u4_row3_1 = ((w29_u4_raw3_1 >= 0) ? w29_u4_raw3_1 : 0);
                    int w120_group_in_swa_w29_u4_1 = ((w29_u4_quad_pos_1 < w120_nswa_2) ? 1 : 0);
                    if (w120_group_in_swa_w29_u4_1 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u4_atom_1 * 8192) + (unsigned int)(w29_u4_quad_1 * 512), (&tmap_swa_kv), w29_u4_atom_1 * 64, w29_u4_row0_1, w29_u4_row1_1, w29_u4_row2_1, w29_u4_row3_1, kv_full_addr + 16);
                    }
                    if (w120_group_in_swa_w29_u4_1 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u4_atom_1 * 8192) + (unsigned int)(w29_u4_quad_1 * 512), (&tmap_compressed_kv), w29_u4_atom_1 * 64, w29_u4_row0_1, w29_u4_row1_1, w29_u4_row2_1, w29_u4_row3_1, kv_full_addr + 16);
                    }
                }
            }
            if (local_warp == 3) {
                int w29_u6_quad = lane % 16;
                int w29_u6_quad_pos = 16 + w29_u6_quad;
                int w29_u6_atom = lane / 16;
                if (w29_u6_quad_pos < w29_valid_groups) {
                    int w29_u6_raw0 = smem_indices[w29_u6_quad_pos * 4];
                    int w29_u6_row0 = ((w29_u6_raw0 >= 0) ? w29_u6_raw0 : 0);
                    int w29_u6_raw1 = smem_indices[w29_u6_quad_pos * 4 + 1];
                    int w29_u6_row1 = ((w29_u6_raw1 >= 0) ? w29_u6_raw1 : 0);
                    int w29_u6_raw2 = smem_indices[w29_u6_quad_pos * 4 + 2];
                    int w29_u6_row2 = ((w29_u6_raw2 >= 0) ? w29_u6_raw2 : 0);
                    int w29_u6_raw3 = smem_indices[w29_u6_quad_pos * 4 + 3];
                    int w29_u6_row3 = ((w29_u6_raw3 >= 0) ? w29_u6_raw3 : 0);
                    int w120_group_in_swa_w29_u6 = ((w29_u6_quad_pos < w120_nswa_2) ? 1 : 0);
                    if (w120_group_in_swa_w29_u6 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u6_atom * 8192) + (unsigned int)(w29_u6_quad * 512), (&tmap_swa_kv), 256 + w29_u6_atom * 64, w29_u6_row0, w29_u6_row1, w29_u6_row2, w29_u6_row3, kv_full_addr + 24);
                    }
                    if (w120_group_in_swa_w29_u6 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u6_atom * 8192) + (unsigned int)(w29_u6_quad * 512), (&tmap_compressed_kv), 256 + w29_u6_atom * 64, w29_u6_row0, w29_u6_row1, w29_u6_row2, w29_u6_row3, kv_full_addr + 24);
                    }
                }
            }
            unsigned int _phase_stats_full_0 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0);
            _phase_stats_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sums[8];
            float alpha_values[8];
            #pragma unroll
            for (int h = 0; h < 8; h++) {
                float row_sum_1 = smem_sum[h];
                float _rcp_3 = approx_rcp(row_sum_1);
                inv_sums[h] = ((row_sum_1 > 0.0f) ? _rcp_3 : 0.0f);
                alpha_values[h] = smem_alpha[h];
            }
            unsigned int _phase_o_full_lo_0 = 0;
            mbarrier_wait(o_full_lo_addr, _phase_o_full_lo_0);
            _phase_o_full_lo_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + 8 + (unsigned int)row_addr_1);
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], taddr + 40 + (unsigned int)row_addr_1);
            const int d_idx0 = d_lane;
            #pragma unroll
            for (int h_1 = 0; h_1 < 8; h_1++) {
                int global_head_2 = head_base_1 + h_1;
                if (global_head_2 < num_heads) {
                    float merged0 = __fmaf_rn(_tmem_load_2[h_1], alpha_values[h_1], _tmem_load_3[h_1]);
                    int output_offset = ((query_idx_1 * num_heads + global_head_2) * num_splits + split_idx_1) * 512 + d_idx0;
                    partial_O[output_offset] = merged0 * inv_sums[h_1] * output_scale;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int _phase_o_full_lo_1 = 0;
            mbarrier_wait(o_full_lo_addr + 8, _phase_o_full_lo_1);
            _phase_o_full_lo_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_4[8];
            tmem_ld_x8(&_tmem_load_4[0], taddr + 16 + (unsigned int)row_addr_1);
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + 48 + (unsigned int)row_addr_1);
            const int d_idx1 = 128 + d_lane;
            #pragma unroll
            for (int h_2 = 0; h_2 < 8; h_2++) {
                int global_head_3 = head_base_1 + h_2;
                if (global_head_3 < num_heads) {
                    float merged1 = __fmaf_rn(_tmem_load_4[h_2], alpha_values[h_2], _tmem_load_5[h_2]);
                    int output_offset_1 = ((query_idx_1 * num_heads + global_head_3) * num_splits + split_idx_1) * 512 + d_idx1;
                    partial_O[output_offset_1] = merged1 * inv_sums[h_2] * output_scale;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_partial_done_0 = 0;
            if (num_splits > 1) {
                mbarrier_arrive(partial_done_addr);
                mbarrier_wait(partial_done_addr, _phase_partial_done_0);
                _phase_partial_done_0 ^= 1;
                if (warp == 4) {
                    if (elect_sync()) {
                        unsigned int prior_arrivals;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(prior_arrivals) : "l"(&partition_arrivals[merge_group]), "r"(static_cast<uint32_t>(1)) : "memory");
                        smem_exch[0] = __uint_as_float(prior_arrivals);
                    }
                }
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                float flag_bits = smem_exch[0];
                int prior_count = __as_u32(flag_bits);
                int last_split = num_splits - 1;
                if (prior_count == last_split) {
                    if (num_splits <= 4) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(partition_arrivals + merge_group)), "r"(static_cast<unsigned int>(0)) : "memory");
                            }
                        }
                        #pragma unroll
                        for (int local_h = 0; local_h < 1; local_h++) {
                            int global_head_4 = head_base_1 + local_warp + local_h;
                            if (global_head_4 < num_heads) {
                                int lse_base = (query_idx_1 * num_heads + global_head_4) * num_splits;
                                int partial_base = lse_base * 512;
                                int lane_d = lane * 8;
                                int split_1 = ((1) < (last_split) ? (1) : (last_split));
                                int split_2 = ((2) < (last_split) ? (2) : (last_split));
                                float _vec_load_0[8];
                                {
                                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_O + (partial_base + lane_d) + 0);
                                    uint4 _vld_4[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_4[_blk] = _vptr_4[_blk];
                                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                            (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_1[8];
                                {
                                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_1 * 512 + lane_d) + 0);
                                    uint4 _vld_5[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_5[_blk] = _vptr_5[_blk];
                                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                            (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_2[8];
                                {
                                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_2 * 512 + lane_d) + 0);
                                    uint4 _vld_6[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_6[_blk] = _vptr_6[_blk];
                                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                            (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_3[8];
                                {
                                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(partial_O + (partial_base + 256 + lane_d) + 0);
                                    uint4 _vld_7[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_7[_blk] = _vptr_7[_blk];
                                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                            (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_4[8];
                                {
                                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_1 * 512 + 256 + lane_d) + 0);
                                    uint4 _vld_8[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_8[_blk] = _vptr_8[_blk];
                                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                            (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_5[8];
                                {
                                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_2 * 512 + 256 + lane_d) + 0);
                                    uint4 _vld_9[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_9[_blk] = _vptr_9[_blk];
                                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                            (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float local_lse = -CUDART_INF_F;
                                if (lane < num_splits) {
                                    local_lse = partial_lse[lse_base + lane];
                                }
                                float _shfl_xor_sync_10_1 = __shfl_xor_sync(0xFFFFFFFF, local_lse, 2);
                                float max_1_2;
                                asm("max.f32 %0, %1, %2;" : "=f"(max_1_2) : "f"(local_lse), "f"(_shfl_xor_sync_10_1));
                                float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, max_1_2, 1);
                                float global_max;
                                asm("max.f32 %0, %1, %2;" : "=f"(global_max) : "f"(max_1_2), "f"(_shfl_xor_sync_11_1));
                                float local_weight = 0.0f;
                                if (lane < num_splits) {
                                    float _exp2_12 = approx_exp2(local_lse - global_max);
                                    local_weight = ((local_lse == -CUDART_INF_F) ? 0.0f : _exp2_12);
                                }
                                float shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, local_weight, 2);
                                float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, local_weight + shfl_xor_4, 1);
                                float global_sum = local_weight + shfl_xor_4 + _shfl_xor_sync_13;
                                float _rcp_14 = approx_rcp(global_sum);
                                float normalized_weight = ((global_sum > 0.0f) ? local_weight * _rcp_14 : 0.0f);
                                float weight_0 = __shfl_sync(0xFFFFFFFF, normalized_weight, 0);
                                float weight_1 = __shfl_sync(0xFFFFFFFF, normalized_weight, 1);
                                float weight_2 = __shfl_sync(0xFFFFFFFF, normalized_weight, 2);
                                float merged_0[8];
                                float merged_1[8];
                                #pragma unroll
                                for (int elem = 0; elem < 8; elem++) {
                                    merged_0[elem] = __fmaf_rn(weight_2, _vec_load_2[elem], __fmaf_rn(weight_0, _vec_load_0[elem], weight_1 * _vec_load_1[elem]));
                                    merged_1[elem] = __fmaf_rn(weight_2, _vec_load_5[elem], __fmaf_rn(weight_0, _vec_load_3[elem], weight_1 * _vec_load_4[elem]));
                                }
                                int output_base = (query_idx_1 * num_heads + global_head_4) * 512 + lane_d;
                                unsigned int _packed_0[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0[_lp*2 + 0], merged_0[_lp*2+1 + 0]));
                                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0[_lp*2 + 2], merged_0[_lp*2+1 + 2]));
                                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0[_lp*2 + 4], merged_0[_lp*2+1 + 4]));
                                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0[_lp*2 + 6], merged_0[_lp*2+1 + 6]));
                                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + output_base)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                                unsigned int _packed_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1[_lp*2 + 0], merged_1[_lp*2+1 + 0]));
                                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1[_lp*2 + 2], merged_1[_lp*2+1 + 2]));
                                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1[_lp*2 + 4], merged_1[_lp*2+1 + 4]));
                                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1[_lp*2 + 6], merged_1[_lp*2+1 + 6]));
                                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + (output_base + 256))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                            }
                        }
                    } else {
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(partition_arrivals + merge_group)), "r"(static_cast<unsigned int>(0)) : "memory");
                            }
                        }
                        #pragma unroll
                        for (int local_h_1 = 0; local_h_1 < 1; local_h_1++) {
                            int global_head_5 = head_base_1 + local_warp + local_h_1;
                            if (global_head_5 < num_heads) {
                                int lse_base_1 = (query_idx_1 * num_heads + global_head_5) * num_splits;
                                int partial_base_1 = lse_base_1 * 512;
                                int lane_d_1 = lane * 8;
                                int split_1_1 = ((1) < (last_split) ? (1) : (last_split));
                                int split_2_1 = ((2) < (last_split) ? (2) : (last_split));
                                float _vec_load_0_1[8];
                                {
                                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(partial_O + (partial_base_1 + lane_d_1) + 0);
                                    uint4 _vld_15[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_15[_blk] = _vptr_15[_blk];
                                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_0_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                                            (&_vec_load_0_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_1_1[8];
                                {
                                    const uint4* _vptr_16 = reinterpret_cast<const uint4*>(partial_O + (partial_base_1 + split_1_1 * 512 + lane_d_1) + 0);
                                    uint4 _vld_16[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_16[_blk] = _vptr_16[_blk];
                                        uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                            (&_vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_2_1[8];
                                {
                                    const uint4* _vptr_17 = reinterpret_cast<const uint4*>(partial_O + (partial_base_1 + split_2_1 * 512 + lane_d_1) + 0);
                                    uint4 _vld_17[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_17[_blk] = _vptr_17[_blk];
                                        uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                            (&_vec_load_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_3_1[8];
                                {
                                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(partial_O + (partial_base_1 + 256 + lane_d_1) + 0);
                                    uint4 _vld_18[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_18[_blk] = _vptr_18[_blk];
                                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                            (&_vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_4_1[8];
                                {
                                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(partial_O + (partial_base_1 + split_1_1 * 512 + 256 + lane_d_1) + 0);
                                    uint4 _vld_19[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_19[_blk] = _vptr_19[_blk];
                                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_4_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                            (&_vec_load_4_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_5_1[8];
                                {
                                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(partial_O + (partial_base_1 + split_2_1 * 512 + 256 + lane_d_1) + 0);
                                    uint4 _vld_20[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_20[_blk] = _vptr_20[_blk];
                                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_5_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                                            (&_vec_load_5_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float local_lse_1 = -CUDART_INF_F;
                                if (lane < num_splits) {
                                    local_lse_1 = partial_lse[lse_base_1 + lane];
                                }
                                float global_max_1 = local_lse_1;
                                #pragma unroll
                                for (int offset = 16; offset > 0; offset >>= 1)
                                    global_max_1 = max_noftz(global_max_1, __shfl_xor_sync(0xFFFFFFFF, global_max_1, offset));
                                float local_weight_1 = 0.0f;
                                if (lane < num_splits) {
                                    float _exp2_21 = approx_exp2(local_lse_1 - global_max_1);
                                    local_weight_1 = ((local_lse_1 == -CUDART_INF_F) ? 0.0f : _exp2_21);
                                }
                                float global_sum_1 = local_weight_1;
                                #pragma unroll
                                for (int offset = 16; offset > 0; offset >>= 1)
                                    global_sum_1 += __shfl_xor_sync(0xFFFFFFFF, global_sum_1, offset);
                                float _rcp_22 = approx_rcp(global_sum_1);
                                float normalized_weight_1 = ((global_sum_1 > 0.0f) ? local_weight_1 * _rcp_22 : 0.0f);
                                float weight_0_1 = __shfl_sync(0xFFFFFFFF, normalized_weight_1, 0);
                                float weight_1_1 = __shfl_sync(0xFFFFFFFF, normalized_weight_1, 1);
                                float weight_2_1 = __shfl_sync(0xFFFFFFFF, normalized_weight_1, 2);
                                float merged_0_1[8];
                                float merged_1_1[8];
                                #pragma unroll
                                for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                    merged_0_1[elem_1] = __fmaf_rn(weight_2_1, _vec_load_2_1[elem_1], __fmaf_rn(weight_0_1, _vec_load_0_1[elem_1], weight_1_1 * _vec_load_1_1[elem_1]));
                                    merged_1_1[elem_1] = __fmaf_rn(weight_2_1, _vec_load_5_1[elem_1], __fmaf_rn(weight_0_1, _vec_load_3_1[elem_1], weight_1_1 * _vec_load_4_1[elem_1]));
                                }
                                int output_base_1 = (query_idx_1 * num_heads + global_head_5) * 512 + lane_d_1;
                                unsigned int _packed_0_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_1[_lp*2 + 0], merged_0_1[_lp*2+1 + 0]));
                                    _packed_0_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_1[_lp*2 + 2], merged_0_1[_lp*2+1 + 2]));
                                    _packed_0_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_1[_lp*2 + 4], merged_0_1[_lp*2+1 + 4]));
                                    _packed_0_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_1[_lp*2 + 6], merged_0_1[_lp*2+1 + 6]));
                                    _packed_0_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + output_base_1)[0] = reinterpret_cast<int4*>(_packed_0_1)[0];
                                unsigned int _packed_1_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_1[_lp*2 + 0], merged_1_1[_lp*2+1 + 0]));
                                    _packed_1_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_1[_lp*2 + 2], merged_1_1[_lp*2+1 + 2]));
                                    _packed_1_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_1[_lp*2 + 4], merged_1_1[_lp*2+1 + 4]));
                                    _packed_1_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_1[_lp*2 + 6], merged_1_1[_lp*2+1 + 6]));
                                    _packed_1_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + (output_base_1 + 256))[0] = reinterpret_cast<int4*>(_packed_1_1)[0];
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_hi_main
            int work_idx_2 = blockIdx.x;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 8;
            const int local_warp_1 = warp - 8;
            const int d_lane_1 = local_warp_1 * 32 + lane;
            const int row_addr_2 = local_warp_1 * 32 << 16;
            float output_scale_1 = bmm2_scale[0];
            int _max_0_2 = ((sparse_topk_lens[query_idx_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_2] + sparse_topk_lens_offset) : (0));
            int w29_active_1 = ((_max_0_2) < (sparse_topk) ? (_max_0_2) : (sparse_topk));
            int w120_total_groups_3 = (sparse_topk + 3) / 4;
            int w120_full_splits_3 = w120_total_groups_3 / 32;
            int w120_split32_3 = split_idx_2 * 32;
            int w120_next32_3 = w120_split32_3 + 32;
            int w120_swa_lo_3 = ((w120_split32_3 / w120_full_splits_3) < (32) ? (w120_split32_3 / w120_full_splits_3) : (32));
            int w120_swa_hi_3 = ((w120_next32_3 / w120_full_splits_3) < (32) ? (w120_next32_3 / w120_full_splits_3) : (32));
            int w120_nswa_3 = w120_swa_hi_3 - w120_swa_lo_3;
            int w120_comp_lo_3 = w120_split32_3 - w120_swa_lo_3;
            int w120_comp_left_3 = w120_total_groups_3 - 32 - w120_comp_lo_3;
            int w120_nswa_left_3 = w120_swa_lo_3 - w120_swa_hi_3 + 32;
            int w120_ncomp_3 = ((w120_nswa_left_3) < (w120_comp_left_3) ? (w120_nswa_left_3) : (w120_comp_left_3));
            int w120_active_group_1 = w29_active_1 / 4;
            int w120_active_rem_1 = w29_active_1 % 4;
            int w120_swa_off_1 = w120_active_group_1 - w120_swa_lo_3;
            int _max_1_1 = ((w120_swa_off_1) > (0) ? (w120_swa_off_1) : (0));
            int w120_full_swa_1 = ((_max_1_1) < (w120_nswa_3) ? (_max_1_1) : (w120_nswa_3));
            int w120_comp_off_1 = w120_active_group_1 - (w120_comp_lo_3 + 32);
            int _max_2_1 = ((w120_comp_off_1) > (0) ? (w120_comp_off_1) : (0));
            int w120_full_comp_1 = ((_max_2_1) < (w120_ncomp_3) ? (_max_2_1) : (w120_ncomp_3));
            int w120_full_rows_1 = (w120_full_swa_1 + w120_full_comp_1) * 4;
            int w29_valid_rows_1 = w120_full_rows_1 + ((w120_swa_off_1 >= 0 && w120_swa_off_1 < w120_nswa_3) ? w120_active_rem_1 : ((w120_comp_off_1 >= 0 && w120_comp_off_1 < w120_ncomp_3) ? w120_active_rem_1 : 0));
            int w29_valid_groups_1 = (w29_valid_rows_1 + 3) / 4;
            asm volatile("barrier.sync 1, 352;" ::: "memory");
            if (local_warp_1 == 0) {
                int w29_u6_quad_1 = lane % 16;
                int w29_u6_quad_pos_1 = 16 + w29_u6_quad_1;
                int w29_u6_atom_1 = 2 + lane / 16;
                if (w29_u6_quad_pos_1 < w29_valid_groups_1) {
                    int w29_u6_raw0_1 = smem_indices[w29_u6_quad_pos_1 * 4];
                    int w29_u6_row0_1 = ((w29_u6_raw0_1 >= 0) ? w29_u6_raw0_1 : 0);
                    int w29_u6_raw1_1 = smem_indices[w29_u6_quad_pos_1 * 4 + 1];
                    int w29_u6_row1_1 = ((w29_u6_raw1_1 >= 0) ? w29_u6_raw1_1 : 0);
                    int w29_u6_raw2_1 = smem_indices[w29_u6_quad_pos_1 * 4 + 2];
                    int w29_u6_row2_1 = ((w29_u6_raw2_1 >= 0) ? w29_u6_raw2_1 : 0);
                    int w29_u6_raw3_1 = smem_indices[w29_u6_quad_pos_1 * 4 + 3];
                    int w29_u6_row3_1 = ((w29_u6_raw3_1 >= 0) ? w29_u6_raw3_1 : 0);
                    int w120_group_in_swa_w29_u6_1 = ((w29_u6_quad_pos_1 < w120_nswa_3) ? 1 : 0);
                    if (w120_group_in_swa_w29_u6_1 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u6_atom_1 * 8192) + (unsigned int)(w29_u6_quad_1 * 512), (&tmap_swa_kv), 256 + w29_u6_atom_1 * 64, w29_u6_row0_1, w29_u6_row1_1, w29_u6_row2_1, w29_u6_row3_1, kv_full_addr + 24);
                    }
                    if (w120_group_in_swa_w29_u6_1 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u6_atom_1 * 8192) + (unsigned int)(w29_u6_quad_1 * 512), (&tmap_compressed_kv), 256 + w29_u6_atom_1 * 64, w29_u6_row0_1, w29_u6_row1_1, w29_u6_row2_1, w29_u6_row3_1, kv_full_addr + 24);
                    }
                }
            }
            unsigned int _phase_stats_full_0_1 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0_1);
            _phase_stats_full_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sums_1[8];
            float alpha_values_1[8];
            #pragma unroll
            for (int h_3 = 0; h_3 < 8; h_3++) {
                float row_sum_2 = smem_sum[h_3];
                float _rcp_3_1 = approx_rcp(row_sum_2);
                inv_sums_1[h_3] = ((row_sum_2 > 0.0f) ? _rcp_3_1 : 0.0f);
                alpha_values_1[h_3] = smem_alpha[h_3];
            }
            unsigned int _phase_o_full_hi_0 = 0;
            mbarrier_wait(o_full_hi_addr, _phase_o_full_hi_0);
            _phase_o_full_hi_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + 24 + (unsigned int)row_addr_2);
            float _tmem_load_7[8];
            tmem_ld_x8(&_tmem_load_7[0], taddr + 56 + (unsigned int)row_addr_2);
            const int d_idx2 = 256 + d_lane_1;
            #pragma unroll
            for (int h_4 = 0; h_4 < 8; h_4++) {
                int global_head_6 = head_base_2 + h_4;
                if (global_head_6 < num_heads) {
                    float merged2 = __fmaf_rn(_tmem_load_6[h_4], alpha_values_1[h_4], _tmem_load_7[h_4]);
                    int output_offset_2 = ((query_idx_2 * num_heads + global_head_6) * num_splits + split_idx_2) * 512 + d_idx2;
                    partial_O[output_offset_2] = merged2 * inv_sums_1[h_4] * output_scale_1;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int _phase_o_full_hi_1 = 0;
            mbarrier_wait(o_full_hi_addr + 8, _phase_o_full_hi_1);
            _phase_o_full_hi_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_8[8];
            tmem_ld_x8(&_tmem_load_8[0], taddr + 32 + (unsigned int)row_addr_2);
            float _tmem_load_9[8];
            tmem_ld_x8(&_tmem_load_9[0], taddr + 64 + (unsigned int)row_addr_2);
            const int d_idx3 = 384 + d_lane_1;
            #pragma unroll
            for (int h_5 = 0; h_5 < 8; h_5++) {
                int global_head_7 = head_base_2 + h_5;
                if (global_head_7 < num_heads) {
                    float merged3 = __fmaf_rn(_tmem_load_8[h_5], alpha_values_1[h_5], _tmem_load_9[h_5]);
                    int output_offset_3 = ((query_idx_2 * num_heads + global_head_7) * num_splits + split_idx_2) * 512 + d_idx3;
                    partial_O[output_offset_3] = merged3 * inv_sums_1[h_5] * output_scale_1;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            if (num_splits > 1) {
                mbarrier_arrive(partial_done_addr);
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                float flag_bits_1 = smem_exch[0];
                int prior_count_1 = __as_u32(flag_bits_1);
                int last_split_1 = num_splits - 1;
                if (prior_count_1 == last_split_1) {
                    if (num_splits <= 4) {
                        #pragma unroll
                        for (int local_h_2 = 0; local_h_2 < 1; local_h_2++) {
                            int global_head_8 = head_base_2 + 4 + local_warp_1 + local_h_2;
                            if (global_head_8 < num_heads) {
                                int lse_base_2 = (query_idx_2 * num_heads + global_head_8) * num_splits;
                                int partial_base_2 = lse_base_2 * 512;
                                int lane_d_2 = lane * 8;
                                int split_1_2 = ((1) < (last_split_1) ? (1) : (last_split_1));
                                int split_2_2 = ((2) < (last_split_1) ? (2) : (last_split_1));
                                float _vec_load_6[8];
                                {
                                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_O + (partial_base_2 + lane_d_2) + 0);
                                    uint4 _vld_4[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_4[_blk] = _vptr_4[_blk];
                                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                            (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_7[8];
                                {
                                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(partial_O + (partial_base_2 + split_1_2 * 512 + lane_d_2) + 0);
                                    uint4 _vld_5[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_5[_blk] = _vptr_5[_blk];
                                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                            (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_8[8];
                                {
                                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_O + (partial_base_2 + split_2_2 * 512 + lane_d_2) + 0);
                                    uint4 _vld_6[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_6[_blk] = _vptr_6[_blk];
                                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                            (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_9[8];
                                {
                                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(partial_O + (partial_base_2 + 256 + lane_d_2) + 0);
                                    uint4 _vld_7[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_7[_blk] = _vptr_7[_blk];
                                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                            (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_10[8];
                                {
                                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(partial_O + (partial_base_2 + split_1_2 * 512 + 256 + lane_d_2) + 0);
                                    uint4 _vld_8[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_8[_blk] = _vptr_8[_blk];
                                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                            (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_11[8];
                                {
                                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(partial_O + (partial_base_2 + split_2_2 * 512 + 256 + lane_d_2) + 0);
                                    uint4 _vld_9[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_9[_blk] = _vptr_9[_blk];
                                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                            (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float local_lse_2 = -CUDART_INF_F;
                                if (lane < num_splits) {
                                    local_lse_2 = partial_lse[lse_base_2 + lane];
                                }
                                float _shfl_xor_sync_10_2 = __shfl_xor_sync(0xFFFFFFFF, local_lse_2, 2);
                                float max_1_3;
                                asm("max.f32 %0, %1, %2;" : "=f"(max_1_3) : "f"(local_lse_2), "f"(_shfl_xor_sync_10_2));
                                float _shfl_xor_sync_11_2 = __shfl_xor_sync(0xFFFFFFFF, max_1_3, 1);
                                float global_max_2;
                                asm("max.f32 %0, %1, %2;" : "=f"(global_max_2) : "f"(max_1_3), "f"(_shfl_xor_sync_11_2));
                                float local_weight_2 = 0.0f;
                                if (lane < num_splits) {
                                    float _exp2_12_1 = approx_exp2(local_lse_2 - global_max_2);
                                    local_weight_2 = ((local_lse_2 == -CUDART_INF_F) ? 0.0f : _exp2_12_1);
                                }
                                float shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, local_weight_2, 2);
                                float _shfl_xor_sync_13_1 = __shfl_xor_sync(0xFFFFFFFF, local_weight_2 + shfl_xor_5, 1);
                                float global_sum_2 = local_weight_2 + shfl_xor_5 + _shfl_xor_sync_13_1;
                                float _rcp_14_1 = approx_rcp(global_sum_2);
                                float normalized_weight_2 = ((global_sum_2 > 0.0f) ? local_weight_2 * _rcp_14_1 : 0.0f);
                                float weight_0_2 = __shfl_sync(0xFFFFFFFF, normalized_weight_2, 0);
                                float weight_1_2 = __shfl_sync(0xFFFFFFFF, normalized_weight_2, 1);
                                float weight_2_2 = __shfl_sync(0xFFFFFFFF, normalized_weight_2, 2);
                                float merged_0_2[8];
                                float merged_1_2[8];
                                #pragma unroll
                                for (int elem_2 = 0; elem_2 < 8; elem_2++) {
                                    merged_0_2[elem_2] = __fmaf_rn(weight_2_2, _vec_load_8[elem_2], __fmaf_rn(weight_0_2, _vec_load_6[elem_2], weight_1_2 * _vec_load_7[elem_2]));
                                    merged_1_2[elem_2] = __fmaf_rn(weight_2_2, _vec_load_11[elem_2], __fmaf_rn(weight_0_2, _vec_load_9[elem_2], weight_1_2 * _vec_load_10[elem_2]));
                                }
                                int output_base_2 = (query_idx_2 * num_heads + global_head_8) * 512 + lane_d_2;
                                unsigned int _packed_2[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_2[_lp*2 + 0], merged_0_2[_lp*2+1 + 0]));
                                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_2[_lp*2 + 2], merged_0_2[_lp*2+1 + 2]));
                                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_2[_lp*2 + 4], merged_0_2[_lp*2+1 + 4]));
                                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_2[_lp*2 + 6], merged_0_2[_lp*2+1 + 6]));
                                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + output_base_2)[0] = reinterpret_cast<int4*>(_packed_2)[0];
                                unsigned int _packed_3[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_2[_lp*2 + 0], merged_1_2[_lp*2+1 + 0]));
                                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_2[_lp*2 + 2], merged_1_2[_lp*2+1 + 2]));
                                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_2[_lp*2 + 4], merged_1_2[_lp*2+1 + 4]));
                                    _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_2[_lp*2 + 6], merged_1_2[_lp*2+1 + 6]));
                                    _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + (output_base_2 + 256))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int local_h_3 = 0; local_h_3 < 1; local_h_3++) {
                            int global_head_9 = head_base_2 + 4 + local_warp_1 + local_h_3;
                            if (global_head_9 < num_heads) {
                                int lse_base_3 = (query_idx_2 * num_heads + global_head_9) * num_splits;
                                int partial_base_3 = lse_base_3 * 512;
                                int lane_d_3 = lane * 8;
                                int split_1_3 = ((1) < (last_split_1) ? (1) : (last_split_1));
                                int split_2_3 = ((2) < (last_split_1) ? (2) : (last_split_1));
                                float _vec_load_6_1[8];
                                {
                                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(partial_O + (partial_base_3 + lane_d_3) + 0);
                                    uint4 _vld_15[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_15[_blk] = _vptr_15[_blk];
                                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_6_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                                            (&_vec_load_6_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_7_1[8];
                                {
                                    const uint4* _vptr_16 = reinterpret_cast<const uint4*>(partial_O + (partial_base_3 + split_1_3 * 512 + lane_d_3) + 0);
                                    uint4 _vld_16[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_16[_blk] = _vptr_16[_blk];
                                        uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_7_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                            (&_vec_load_7_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_8_1[8];
                                {
                                    const uint4* _vptr_17 = reinterpret_cast<const uint4*>(partial_O + (partial_base_3 + split_2_3 * 512 + lane_d_3) + 0);
                                    uint4 _vld_17[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_17[_blk] = _vptr_17[_blk];
                                        uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_8_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                            (&_vec_load_8_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_9_1[8];
                                {
                                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(partial_O + (partial_base_3 + 256 + lane_d_3) + 0);
                                    uint4 _vld_18[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_18[_blk] = _vptr_18[_blk];
                                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_9_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                            (&_vec_load_9_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_10_1[8];
                                {
                                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(partial_O + (partial_base_3 + split_1_3 * 512 + 256 + lane_d_3) + 0);
                                    uint4 _vld_19[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_19[_blk] = _vptr_19[_blk];
                                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_10_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                            (&_vec_load_10_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float _vec_load_11_1[8];
                                {
                                    const uint4* _vptr_20 = reinterpret_cast<const uint4*>(partial_O + (partial_base_3 + split_2_3 * 512 + 256 + lane_d_3) + 0);
                                    uint4 _vld_20[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_20[_blk] = _vptr_20[_blk];
                                        uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&_vec_load_11_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                                            (&_vec_load_11_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                float local_lse_3 = -CUDART_INF_F;
                                if (lane < num_splits) {
                                    local_lse_3 = partial_lse[lse_base_3 + lane];
                                }
                                float global_max_3 = local_lse_3;
                                #pragma unroll
                                for (int offset = 16; offset > 0; offset >>= 1)
                                    global_max_3 = max_noftz(global_max_3, __shfl_xor_sync(0xFFFFFFFF, global_max_3, offset));
                                float local_weight_3 = 0.0f;
                                if (lane < num_splits) {
                                    float _exp2_21_1 = approx_exp2(local_lse_3 - global_max_3);
                                    local_weight_3 = ((local_lse_3 == -CUDART_INF_F) ? 0.0f : _exp2_21_1);
                                }
                                float global_sum_3 = local_weight_3;
                                #pragma unroll
                                for (int offset = 16; offset > 0; offset >>= 1)
                                    global_sum_3 += __shfl_xor_sync(0xFFFFFFFF, global_sum_3, offset);
                                float _rcp_22_1 = approx_rcp(global_sum_3);
                                float normalized_weight_3 = ((global_sum_3 > 0.0f) ? local_weight_3 * _rcp_22_1 : 0.0f);
                                float weight_0_3 = __shfl_sync(0xFFFFFFFF, normalized_weight_3, 0);
                                float weight_1_3 = __shfl_sync(0xFFFFFFFF, normalized_weight_3, 1);
                                float weight_2_3 = __shfl_sync(0xFFFFFFFF, normalized_weight_3, 2);
                                float merged_0_3[8];
                                float merged_1_3[8];
                                #pragma unroll
                                for (int elem_3 = 0; elem_3 < 8; elem_3++) {
                                    merged_0_3[elem_3] = __fmaf_rn(weight_2_3, _vec_load_8_1[elem_3], __fmaf_rn(weight_0_3, _vec_load_6_1[elem_3], weight_1_3 * _vec_load_7_1[elem_3]));
                                    merged_1_3[elem_3] = __fmaf_rn(weight_2_3, _vec_load_11_1[elem_3], __fmaf_rn(weight_0_3, _vec_load_9_1[elem_3], weight_1_3 * _vec_load_10_1[elem_3]));
                                }
                                int output_base_3 = (query_idx_2 * num_heads + global_head_9) * 512 + lane_d_3;
                                unsigned int _packed_2_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_3[_lp*2 + 0], merged_0_3[_lp*2+1 + 0]));
                                    _packed_2_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_3[_lp*2 + 2], merged_0_3[_lp*2+1 + 2]));
                                    _packed_2_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_3[_lp*2 + 4], merged_0_3[_lp*2+1 + 4]));
                                    _packed_2_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_0_3[_lp*2 + 6], merged_0_3[_lp*2+1 + 6]));
                                    _packed_2_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + output_base_3)[0] = reinterpret_cast<int4*>(_packed_2_1)[0];
                                unsigned int _packed_3_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_3[_lp*2 + 0], merged_1_3[_lp*2+1 + 0]));
                                    _packed_3_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_3[_lp*2 + 2], merged_1_3[_lp*2+1 + 2]));
                                    _packed_3_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_3[_lp*2 + 4], merged_1_3[_lp*2+1 + 4]));
                                    _packed_3_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_1_3[_lp*2 + 6], merged_1_3[_lp*2+1 + 6]));
                                    _packed_3_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O + (output_base_3 + 256))[0] = reinterpret_cast<int4*>(_packed_3_1)[0];
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            mbarrier_wait(kv_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
            int _mma_b_lo_0 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (0) * 256);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
            }
            mbarrier_wait(kv_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 2048);
            int _mma_b_lo_1 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (1) * 256);
            {
                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
            }
            elect_commit(s_full_addr);
            mbarrier_wait(kv_full_addr + 16, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (2) * 2048);
            int _mma_b_lo_2 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (0) * 256);
            {
                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
            }
            mbarrier_wait(kv_full_addr + 24, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (3) * 2048);
            int _mma_b_lo_3 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (1) * 256);
            {
                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
            }
            elect_commit(s_full_addr + 8);
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(kv_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 2048);
            int _mma_b_lo_4 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
            }
            int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (0) * 2048);
            int _mma_b_lo_5 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                }
            }
            mbarrier_wait(kv_full_addr + 8, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 2048);
            int _mma_b_lo_6 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                }
            }
            int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (1) * 2048);
            int _mma_b_lo_7 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                }
            }
            unsigned int _phase_p_full_1 = 0;
            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
            _phase_p_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(kv_full_addr + 16, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 2048);
            int _mma_b_lo_8 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                }
            }
            elect_commit(o_full_lo_addr);
            int _mma_a_lo_9 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (2) * 2048);
            int _mma_b_lo_9 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_9);
                uint64_t _mma_ss_b_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_9);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                }
            }
            elect_commit(o_full_lo_addr + 8);
            mbarrier_wait(kv_full_addr + 24, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 2048);
            int _mma_b_lo_10 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_10);
                uint64_t _mma_ss_b_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_10);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 1);
                }
            }
            elect_commit(o_full_hi_addr);
            int _mma_a_lo_11 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (3) * 2048);
            int _mma_b_lo_11 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_11);
                uint64_t _mma_ss_b_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_11);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 1);
                }
            }
            elect_commit(o_full_hi_addr + 8);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_rank = warp - 13;
            int work_idx_3 = blockIdx.x;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int split_idx_3 = split_work_3 % num_splits;
            int query_idx_3 = split_work_3 / num_splits;
            int w120_total_groups_4 = (sparse_topk + 3) / 4;
            int w120_full_splits_4 = w120_total_groups_4 / 32;
            int w120_split32_4 = split_idx_3 * 32;
            int w120_next32_4 = w120_split32_4 + 32;
            int w120_swa_lo_4 = ((w120_split32_4 / w120_full_splits_4) < (32) ? (w120_split32_4 / w120_full_splits_4) : (32));
            int w120_swa_hi_4 = ((w120_next32_4 / w120_full_splits_4) < (32) ? (w120_next32_4 / w120_full_splits_4) : (32));
            int w120_nswa_4 = w120_swa_hi_4 - w120_swa_lo_4;
            int w120_comp_lo_4 = w120_split32_4 - w120_swa_lo_4;
            int w120_comp_left_4 = w120_total_groups_4 - 32 - w120_comp_lo_4;
            int w120_nswa_left_4 = w120_swa_lo_4 - w120_swa_hi_4 + 32;
            int w120_ncomp_4 = ((w120_nswa_left_4) < (w120_comp_left_4) ? (w120_nswa_left_4) : (w120_comp_left_4));
            int vector_offset_1 = lane * 4;
            if (load_rank == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 8192);
                    #pragma unroll
                    for (int q_stage = 0; q_stage < 4; q_stage++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 2048), (&tmap_q), 0, head_tile_3 * 8, q_stage * 2, query_idx_3, q_full_addr);
                    }
                }
            }
            int _max_0_3 = ((sparse_topk_lens[query_idx_3] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_3] + sparse_topk_lens_offset) : (0));
            int active_topk_1 = ((_max_0_3) < (sparse_topk) ? (_max_0_3) : (sparse_topk));
            int w120_active_group_2 = active_topk_1 / 4;
            int w120_active_rem_2 = active_topk_1 % 4;
            int w120_swa_off_2 = w120_active_group_2 - w120_swa_lo_4;
            int _max_1_2 = ((w120_swa_off_2) > (0) ? (w120_swa_off_2) : (0));
            int w120_full_swa_2 = ((_max_1_2) < (w120_nswa_4) ? (_max_1_2) : (w120_nswa_4));
            int w120_comp_off_2 = w120_active_group_2 - (w120_comp_lo_4 + 32);
            int _max_2_2 = ((w120_comp_off_2) > (0) ? (w120_comp_off_2) : (0));
            int w120_full_comp_2 = ((_max_2_2) < (w120_ncomp_4) ? (_max_2_2) : (w120_ncomp_4));
            int w120_full_rows_2 = (w120_full_swa_2 + w120_full_comp_2) * 4;
            int valid_rows = w120_full_rows_2 + ((w120_swa_off_2 >= 0 && w120_swa_off_2 < w120_nswa_4) ? w120_active_rem_2 : ((w120_comp_off_2 >= 0 && w120_comp_off_2 < w120_ncomp_4) ? w120_active_rem_2 : 0));
            int valid_groups = (valid_rows + 3) / 4;
            int live_groups0 = ((valid_groups) < (16) ? (valid_groups) : (16));
            int _max_3 = ((valid_groups - 16) > (0) ? (valid_groups - 16) : (0));
            int live_groups1 = ((_max_3) < (16) ? (_max_3) : (16));
            int loader_thread = load_rank * 32 + lane;
            int dead_vecs0 = (16 - live_groups0) * 128;
            int dead_vecs1 = (16 - live_groups1) * 128;
            for (int tail_vec = loader_thread; tail_vec < dead_vecs0; tail_vec += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(tail_vec % 128 / 32 * 8192 + (live_groups0 + tail_vec / 128) * 512 + tail_vec % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            for (int tail_vec_1 = loader_thread; tail_vec_1 < dead_vecs0; tail_vec_1 += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(32768 + tail_vec_1 % 128 / 32 * 8192 + (live_groups0 + tail_vec_1 / 128) * 512 + tail_vec_1 % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            for (int tail_vec_2 = loader_thread; tail_vec_2 < dead_vecs1; tail_vec_2 += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(65536 + tail_vec_2 % 128 / 32 * 8192 + (live_groups1 + tail_vec_2 / 128) * 512 + tail_vec_2 % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            for (int tail_vec_3 = loader_thread; tail_vec_3 < dead_vecs1; tail_vec_3 += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(98304 + tail_vec_3 % 128 / 32 * 8192 + (live_groups1 + tail_vec_3 / 128) * 512 + tail_vec_3 % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            if (load_rank == 0) {
                asm volatile("cp.async.wait_group 0;");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 352;" ::: "memory");
            int gather_tx0 = live_groups0 * 4 * 512;
            int gather_tx1 = live_groups1 * 4 * 512;
            if (0 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr, gather_tx0);
                }
            }
            if (1 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + 8, gather_tx0);
                }
            }
            if (2 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + 16, gather_tx1);
                }
            }
            if (0 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + 24, gather_tx1);
                }
            }
            if (load_rank == 0) {
                int w29_u0_quad = lane % 16;
                int w29_u0_quad_pos = w29_u0_quad;
                int w29_u0_atom = lane / 16;
                if (w29_u0_quad_pos < valid_groups) {
                    int w29_u0_raw0 = smem_indices[w29_u0_quad_pos * 4];
                    int w29_u0_row0 = ((w29_u0_raw0 >= 0) ? w29_u0_raw0 : 0);
                    int w29_u0_raw1 = smem_indices[w29_u0_quad_pos * 4 + 1];
                    int w29_u0_row1 = ((w29_u0_raw1 >= 0) ? w29_u0_raw1 : 0);
                    int w29_u0_raw2 = smem_indices[w29_u0_quad_pos * 4 + 2];
                    int w29_u0_row2 = ((w29_u0_raw2 >= 0) ? w29_u0_raw2 : 0);
                    int w29_u0_raw3 = smem_indices[w29_u0_quad_pos * 4 + 3];
                    int w29_u0_row3 = ((w29_u0_raw3 >= 0) ? w29_u0_raw3 : 0);
                    int w120_group_in_swa_w29_u0 = ((w29_u0_quad_pos < w120_nswa_4) ? 1 : 0);
                    if (w120_group_in_swa_w29_u0 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u0_atom * 8192) + (unsigned int)(w29_u0_quad * 512), (&tmap_swa_kv), w29_u0_atom * 64, w29_u0_row0, w29_u0_row1, w29_u0_row2, w29_u0_row3, kv_full_addr);
                    }
                    if (w120_group_in_swa_w29_u0 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u0_atom * 8192) + (unsigned int)(w29_u0_quad * 512), (&tmap_compressed_kv), w29_u0_atom * 64, w29_u0_row0, w29_u0_row1, w29_u0_row2, w29_u0_row3, kv_full_addr);
                    }
                }
            }
            if (load_rank == 1) {
                int w29_u0_quad_1 = lane % 16;
                int w29_u0_quad_pos_1 = w29_u0_quad_1;
                int w29_u0_atom_1 = 2 + lane / 16;
                if (w29_u0_quad_pos_1 < valid_groups) {
                    int w29_u0_raw0_1 = smem_indices[w29_u0_quad_pos_1 * 4];
                    int w29_u0_row0_1 = ((w29_u0_raw0_1 >= 0) ? w29_u0_raw0_1 : 0);
                    int w29_u0_raw1_1 = smem_indices[w29_u0_quad_pos_1 * 4 + 1];
                    int w29_u0_row1_1 = ((w29_u0_raw1_1 >= 0) ? w29_u0_raw1_1 : 0);
                    int w29_u0_raw2_1 = smem_indices[w29_u0_quad_pos_1 * 4 + 2];
                    int w29_u0_row2_1 = ((w29_u0_raw2_1 >= 0) ? w29_u0_raw2_1 : 0);
                    int w29_u0_raw3_1 = smem_indices[w29_u0_quad_pos_1 * 4 + 3];
                    int w29_u0_row3_1 = ((w29_u0_raw3_1 >= 0) ? w29_u0_raw3_1 : 0);
                    int w120_group_in_swa_w29_u0_1 = ((w29_u0_quad_pos_1 < w120_nswa_4) ? 1 : 0);
                    if (w120_group_in_swa_w29_u0_1 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u0_atom_1 * 8192) + (unsigned int)(w29_u0_quad_1 * 512), (&tmap_swa_kv), w29_u0_atom_1 * 64, w29_u0_row0_1, w29_u0_row1_1, w29_u0_row2_1, w29_u0_row3_1, kv_full_addr);
                    }
                    if (w120_group_in_swa_w29_u0_1 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u0_atom_1 * 8192) + (unsigned int)(w29_u0_quad_1 * 512), (&tmap_compressed_kv), w29_u0_atom_1 * 64, w29_u0_row0_1, w29_u0_row1_1, w29_u0_row2_1, w29_u0_row3_1, kv_full_addr);
                    }
                }
            }
            if (load_rank == 2) {
                int w29_u2_quad_1 = lane % 16;
                int w29_u2_quad_pos_1 = w29_u2_quad_1;
                int w29_u2_atom_1 = lane / 16;
                if (w29_u2_quad_pos_1 < valid_groups) {
                    int w29_u2_raw0_1 = smem_indices[w29_u2_quad_pos_1 * 4];
                    int w29_u2_row0_1 = ((w29_u2_raw0_1 >= 0) ? w29_u2_raw0_1 : 0);
                    int w29_u2_raw1_1 = smem_indices[w29_u2_quad_pos_1 * 4 + 1];
                    int w29_u2_row1_1 = ((w29_u2_raw1_1 >= 0) ? w29_u2_raw1_1 : 0);
                    int w29_u2_raw2_1 = smem_indices[w29_u2_quad_pos_1 * 4 + 2];
                    int w29_u2_row2_1 = ((w29_u2_raw2_1 >= 0) ? w29_u2_raw2_1 : 0);
                    int w29_u2_raw3_1 = smem_indices[w29_u2_quad_pos_1 * 4 + 3];
                    int w29_u2_row3_1 = ((w29_u2_raw3_1 >= 0) ? w29_u2_raw3_1 : 0);
                    int w120_group_in_swa_w29_u2_1 = ((w29_u2_quad_pos_1 < w120_nswa_4) ? 1 : 0);
                    if (w120_group_in_swa_w29_u2_1 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u2_atom_1 * 8192) + (unsigned int)(w29_u2_quad_1 * 512), (&tmap_swa_kv), 256 + w29_u2_atom_1 * 64, w29_u2_row0_1, w29_u2_row1_1, w29_u2_row2_1, w29_u2_row3_1, kv_full_addr + 8);
                    }
                    if (w120_group_in_swa_w29_u2_1 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u2_atom_1 * 8192) + (unsigned int)(w29_u2_quad_1 * 512), (&tmap_compressed_kv), 256 + w29_u2_atom_1 * 64, w29_u2_row0_1, w29_u2_row1_1, w29_u2_row2_1, w29_u2_row3_1, kv_full_addr + 8);
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
