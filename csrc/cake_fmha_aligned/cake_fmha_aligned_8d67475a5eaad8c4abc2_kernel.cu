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
#define TMEM_NCOLS 40
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_OFFSET 32
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 3
#define SMEM_SMEM_K_OFF 1024
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_SMEM_Q_OFF 99328
#define SMEM_SMEM_Q_STAGE_BYTES 2048
#define SMEM_SMEM_Q_STRIDE 2048
#define SMEM_SMEM_V_OFF 101376
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 199680
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_SMEM_WARP_MAX_OFF 201728
#define SMEM_SMEM_WARP_MAX_STAGE_BYTES 128
#define SMEM_SMEM_WARP_MAX_STRIDE 128
#define SMEM_SMEM_ROW_SUM_OFF 201856
#define SMEM_SMEM_ROW_SUM_STAGE_BYTES 128
#define SMEM_SMEM_ROW_SUM_STRIDE 128
#define SMEM_DSM_MAX_OFF 201984
#define SMEM_DSM_MAX_STAGE_BYTES 32
#define SMEM_DSM_MAX_STRIDE 32
#define SMEM_DSM_SUM_OFF 202016
#define SMEM_DSM_SUM_STAGE_BYTES 32
#define SMEM_DSM_SUM_STRIDE 32
#define SMEM_DSM_O_OFF 202048
#define SMEM_DSM_O_STAGE_BYTES 4096
#define SMEM_DSM_O_STRIDE 4096
#define SMEM_PAGE_IDS_OFF 206144
#define SMEM_PAGE_IDS_STAGE_BYTES 512
#define SMEM_PAGE_IDS_STRIDE 512
#define SMEM_TOTAL 206720
#define THREADS 256
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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ void tmem_st_x8_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
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


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,8) void
kernel_cake_fmha_aligned_8d67475a5eaad8c4abc2(const __grid_constant__ CUtensorMap Qt, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __half* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int uniform_seq_len, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, int batch_size)
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
    #define k_full_addr (mbar_base + 0)
    #define k_empty_addr (mbar_base + 24)
    #define vp_full_addr (mbar_base + 48)
    #define v_empty_addr (mbar_base + 72)
    #define s_full_addr (mbar_base + 96)
    #define o_ready_addr (mbar_base + 104)
    #define tmem_done_addr (mbar_base + 112)
    #define reduce_stats_full_addr (mbar_base + 120)
    #define reduce_o_full_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 8) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 8);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __half* smem_k = reinterpret_cast<__half*>(smem_raw + 1024);
    const int smem_k_addr = smem + 1024;
    __half* smem_q = reinterpret_cast<__half*>(smem_raw + 99328);
    const int smem_q_addr = smem + 99328;
    __half* smem_v = reinterpret_cast<__half*>(smem_raw + 101376);
    const int smem_v_addr = smem + 101376;
    __half* smem_p = reinterpret_cast<__half*>(smem_raw + 199680);
    const int smem_p_addr = smem + 199680;
    float* smem_warp_max = reinterpret_cast<float*>(smem_raw + 201728);
    const int smem_warp_max_addr = smem + 201728;
    float* smem_row_sum = reinterpret_cast<float*>(smem_raw + 201856);
    const int smem_row_sum_addr = smem + 201856;
    float* dsm_max = reinterpret_cast<float*>(smem_raw + 201984);
    const int dsm_max_addr = smem + 201984;
    float* dsm_sum = reinterpret_cast<float*>(smem_raw + 202016);
    const int dsm_sum_addr = smem + 202016;
    float* dsm_o = reinterpret_cast<float*>(smem_raw + 202048);
    const int dsm_o_addr = smem + 202048;
    int* page_ids = reinterpret_cast<int*>(smem_raw + 206144);
    const int page_ids_addr = smem + 206144;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=6
            mbarrier_init(smem + 0, 6);
            mbarrier_init(smem + 8, 6);
            mbarrier_init(smem + 16, 6);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // vp_full: 3 barriers, init_count=5
            mbarrier_init(smem + 48, 5);
            mbarrier_init(smem + 56, 5);
            mbarrier_init(smem + 64, 5);
            // v_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // o_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // tmem_done: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // reduce_stats_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // reduce_o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 120, 64);
            mbarrier_expect_tx(smem + 128, 4096);
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 2) {
        int _tmem_hold = smem + 136;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output = taddr + 32;
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: q_dma ----
    if (warp == 0) {
        { // q_dma_main
            int batch_idx = blockIdx.x / num_kv_heads;
            int kv_head_idx = blockIdx.x % num_kv_heads;
            int seqlen = ((uniform_seq_len > 0) ? uniform_seq_len : seq_lens_kv[batch_idx]);
            int nblocks = (seqlen + 128 - 1) / 128;
            int base = nblocks / 8;
            int rem = nblocks % 8;
            int rank = cta_rank;
            int rank_0 = rank;
            int my_count = base + ((rank_0 < rem) ? 1 : 0);
            if (my_count > 0) {
                unsigned int q_stage = 0;
                unsigned int q_empty_phase = 1;
                if (elect_sync()) {
                    int q_row = (batch_idx * num_kv_heads + kv_head_idx) * 8;
                    mbarrier_wait(k_empty_addr + (q_stage) * 8, q_empty_phase);
                    mbarrier_arrive_expect_tx(k_full_addr + (q_stage) * 8, 2048);
                    tma_3d_gmem2smem(smem_q_addr, (&Qt), 0, q_row, 0, k_full_addr + (q_stage) * 8);
                    q_stage += 1;
                    if (q_stage == 3) { q_stage = 0; q_empty_phase ^= 1; }
                    #pragma unroll 1
                    for (int tile = 1; tile < my_count; tile++) {
                        mbarrier_wait(k_empty_addr + (q_stage) * 8, q_empty_phase);
                        mbarrier_arrive(k_full_addr + (q_stage) * 8);
                        q_stage += 1;
                        if (q_stage == 3) { q_stage = 0; q_empty_phase ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: kv_dma ----
    if (warp == 1) {
        { // kv_dma_main
            int batch_idx_1 = blockIdx.x / num_kv_heads;
            int kv_head_idx_1 = blockIdx.x % num_kv_heads;
            int rank_1 = cta_rank;
            int rank_0_1 = rank_1;
            int seqlen_1 = ((uniform_seq_len > 0) ? uniform_seq_len : seq_lens_kv[batch_idx_1]);
            int nblocks_1 = (seqlen_1 + 128 - 1) / 128;
            int base_1 = nblocks_1 / 8;
            int rem_1 = nblocks_1 % 8;
            int extra_before = ((rank_0_1 < rem_1) ? rank_0_1 : rem_1);
            int my_start = rank_0_1 * base_1 + extra_before;
            int my_count_1 = base_1 + ((rank_0_1 < rem_1) ? 1 : 0);
            int pt_base = batch_idx_1 * max_pages_per_seq;
            unsigned int k_stage = 0;
            unsigned int k_empty_phase = 1;
            unsigned int v_stage = 0;
            unsigned int v_empty_phase = 1;
            if (my_count_1 > 0) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    int page_idx = half * 32 + lane;
                    int _min_0 = ((my_start * 8 + page_idx) < (max_pages_per_seq - 1) ? (my_start * 8 + page_idx) : (max_pages_per_seq - 1));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(page_ids_addr + (unsigned int)(page_idx * 4)), "l"(page_table + (pt_base + _min_0)));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_wait(k_empty_addr + (k_stage) * 8, k_empty_phase);
                    mbarrier_arrive_expect_tx(k_full_addr + (k_stage) * 8, 32768);
                    int k_dst = smem_k_addr + k_stage * 32768;
                    #pragma unroll
                    for (int pg = 0; pg < 8; pg++) {
                        int phys = page_ids[pg] * num_kv_heads + kv_head_idx_1;
                        #pragma unroll
                        for (int hg = 0; hg < 2; hg++) {
                            int off = hg * 16384 + pg * 2048;
                            tma_4d_gmem2smem(k_dst + off, (&K), 0, 0, hg, phys, k_full_addr + (k_stage) * 8);
                        }
                    }
                    k_stage += 1;
                    if (k_stage == 3) { k_stage = 0; k_empty_phase ^= 1; }
                }
                #pragma unroll 1
                for (int tile_1 = 1; tile_1 < my_count_1; tile_1++) {
                    int tile_in_page_stage = tile_1 % 8;
                    int page_stage = tile_1 / 8;
                    int page_slot = page_stage % 2;
                    if (tile_in_page_stage == 0) {
                        asm volatile("cp.async.wait_group 0;");
                        __syncwarp();
                    }
                    if (tile_in_page_stage == 1) {
                        int next_page_stage = page_stage + 1;
                        int next_tile = next_page_stage * 8;
                        if (next_tile < my_count_1) {
                            __syncwarp();
                            int next_page_slot = next_page_stage % 2;
                            #pragma unroll
                            for (int half_1 = 0; half_1 < 2; half_1++) {
                                int page_idx_1 = half_1 * 32 + lane;
                                int logical_page = (my_start + next_tile) * 8 + page_idx_1;
                                int _min_1 = ((logical_page) < (max_pages_per_seq - 1) ? (logical_page) : (max_pages_per_seq - 1));
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                    :: "r"(page_ids_addr + (unsigned int)((next_page_slot * 64 + page_idx_1) * 4)), "l"(page_table + (pt_base + _min_1)));
                            }
                            asm volatile("cp.async.commit_group;");
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_wait(k_empty_addr + (k_stage) * 8, k_empty_phase);
                        mbarrier_arrive_expect_tx(k_full_addr + (k_stage) * 8, 32768);
                        int k_dst_1 = smem_k_addr + k_stage * 32768;
                        #pragma unroll
                        for (int pg_1 = 0; pg_1 < 8; pg_1++) {
                            int phys_1 = page_ids[page_slot * 64 + tile_in_page_stage * 8 + pg_1] * num_kv_heads + kv_head_idx_1;
                            #pragma unroll
                            for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                int off_1 = hg_1 * 16384 + pg_1 * 2048;
                                tma_4d_gmem2smem(k_dst_1 + off_1, (&K), 0, 0, hg_1, phys_1, k_full_addr + (k_stage) * 8);
                            }
                        }
                        k_stage += 1;
                        if (k_stage == 3) { k_stage = 0; k_empty_phase ^= 1; }
                        int prev_tile = tile_1 - 1;
                        int prev_page_stage = prev_tile / 8;
                        int prev_slot = prev_page_stage % 2;
                        int prev_tile_in_page_stage = prev_tile % 8;
                        mbarrier_wait(v_empty_addr + (v_stage) * 8, v_empty_phase);
                        mbarrier_arrive_expect_tx(vp_full_addr + (v_stage) * 8, 32768);
                        int v_dst = smem_v_addr + v_stage * 32768;
                        #pragma unroll
                        for (int pg_2 = 0; pg_2 < 8; pg_2++) {
                            int phys_2 = page_ids[prev_slot * 64 + prev_tile_in_page_stage * 8 + pg_2] * num_kv_heads + kv_head_idx_1;
                            #pragma unroll
                            for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                int off_2 = hg_2 * 16384 + pg_2 * 2048;
                                tma_4d_gmem2smem(v_dst + off_2, (&V), 0, 0, hg_2, phys_2, vp_full_addr + (v_stage) * 8);
                            }
                        }
                        v_stage += 1;
                        if (v_stage == 3) { v_stage = 0; v_empty_phase ^= 1; }
                    }
                }
                int last_tile = my_count_1 - 1;
                int last_page_stage = last_tile / 8;
                int last_slot = last_page_stage % 2;
                int last_tile_in_page_stage = last_tile % 8;
                if (elect_sync()) {
                    mbarrier_wait(v_empty_addr + (v_stage) * 8, v_empty_phase);
                    mbarrier_arrive_expect_tx(vp_full_addr + (v_stage) * 8, 32768);
                    int v_dst_1 = smem_v_addr + v_stage * 32768;
                    #pragma unroll
                    for (int pg_3 = 0; pg_3 < 8; pg_3++) {
                        int phys_3 = page_ids[last_slot * 64 + last_tile_in_page_stage * 8 + pg_3] * num_kv_heads + kv_head_idx_1;
                        #pragma unroll
                        for (int hg_3 = 0; hg_3 < 2; hg_3++) {
                            int off_3 = hg_3 * 16384 + pg_3 * 2048;
                            tma_4d_gmem2smem(v_dst_1 + off_3, (&V), 0, 0, hg_3, phys_3, vp_full_addr + (v_stage) * 8);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 2) {
        { // mma_main
            int batch_idx_2 = blockIdx.x / num_kv_heads;
            int rank_2 = cta_rank;
            int rank_0_2 = rank_2;
            int seqlen_2 = ((uniform_seq_len > 0) ? uniform_seq_len : seq_lens_kv[batch_idx_2]);
            int nblocks_2 = (seqlen_2 + 128 - 1) / 128;
            int base_2 = nblocks_2 / 8;
            int rem_2 = nblocks_2 % 8;
            int my_count_2 = base_2 + ((rank_0_2 < rem_2) ? 1 : 0);
            unsigned int _phase_tmem_done_0 = 0;
            if (my_count_2 > 0) {
                unsigned int k_stage_1 = 0;
                unsigned int k_full_phase = 0;
                unsigned int v_stage_1 = 0;
                unsigned int v_full_phase = 0;
                mbarrier_wait(k_full_addr + (k_stage_1) * 8, k_full_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 2048);
                int _mma_b_lo_0 = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(k_empty_addr + (k_stage_1) * 8);
                k_stage_1 += 1;
                if (k_stage_1 == 3) { k_stage_1 = 0; k_full_phase ^= 1; }
                int first_pv = 1;
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < my_count_2; tile_2++) {
                    if (my_count_2 > tile_2 + 1) {
                        mbarrier_wait(k_full_addr + (k_stage_1) * 8, k_full_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 2048);
                        int _mma_b_lo_1 = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                            }
                        }
                        elect_commit(s_full_addr);
                        elect_commit(k_empty_addr + (k_stage_1) * 8);
                        k_stage_1 += 1;
                        if (k_stage_1 == 3) { k_stage_1 = 0; k_full_phase ^= 1; }
                    }
                    mbarrier_wait(vp_full_addr + (v_stage_1) * 8, v_full_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_1) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, ((first_pv) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_output, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134381584, 1);
                        }
                    }
                    elect_commit2(v_empty_addr + (v_stage_1) * 8, o_ready_addr);
                    first_pv = 0;
                    v_stage_1 += 1;
                    if (v_stage_1 == 3) { v_stage_1 = 0; v_full_phase ^= 1; }
                }
                mbarrier_wait(tmem_done_addr, _phase_tmem_done_0);
                _phase_tmem_done_0 ^= 1;
            }
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        }
    }
    // ---- Role: idle ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // epilogue_main
            int rank_3 = cta_rank;
            int rank_0_3 = rank_3;
            int batch_idx_3 = blockIdx.x / num_kv_heads;
            int kv_head_idx_2 = blockIdx.x % num_kv_heads;
            int seqlen_3 = ((uniform_seq_len > 0) ? uniform_seq_len : seq_lens_kv[batch_idx_3]);
            int nblocks_3 = (seqlen_3 + 128 - 1) / 128;
            int base_3 = nblocks_3 / 8;
            int rem_3 = nblocks_3 % 8;
            int extra_before_1 = ((rank_0_3 < rem_3) ? rank_0_3 : rem_3);
            int my_start_1 = rank_0_3 * base_3 + extra_before_1;
            int my_count_3 = base_3 + ((rank_0_3 < rem_3) ? 1 : 0);
            const int local_warp = warp - 4;
            int local_tid = local_warp * 32 + lane;
            int row_addr = local_warp * 32 << 16;
            if (warp == 4) {
                if (elect_sync()) {
                    mbarrier_arrive(reduce_stats_full_addr);
                    mbarrier_arrive(reduce_o_full_addr);
                }
            }
            float row_max[8];
            float row_sum[8];
            row_max[0] = -CUDART_INF_F;
            row_max[1] = -CUDART_INF_F;
            row_max[2] = -CUDART_INF_F;
            row_max[3] = -CUDART_INF_F;
            row_max[4] = -CUDART_INF_F;
            row_max[5] = -CUDART_INF_F;
            row_max[6] = -CUDART_INF_F;
            row_max[7] = -CUDART_INF_F;
            row_sum[0] = 0.0f;
            row_sum[1] = 0.0f;
            row_sum[2] = 0.0f;
            row_sum[3] = 0.0f;
            row_sum[4] = 0.0f;
            row_sum[5] = 0.0f;
            row_sum[6] = 0.0f;
            row_sum[7] = 0.0f;
            if (elect_sync()) {
                mbarrier_arrive(k_full_addr);
            }
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_o_ready_0 = 0;
            if (my_count_3 > 0) {
                #pragma unroll 1
                for (int tile_3 = 0; tile_3 < my_count_3; tile_3++) {
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)row_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (my_count_3 > tile_3 + 1) {
                        int next_k_stage = (tile_3 + 1) % 3;
                        if (elect_sync()) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(k_full_addr + (next_k_stage) * 8);
                        }
                    }
                    int token = (my_start_1 + tile_3) * 128 + local_tid;
                    if (token >= seqlen_3) {
                        _tmem_load_0[0] = -CUDART_INF_F;
                        _tmem_load_0[1] = -CUDART_INF_F;
                        _tmem_load_0[2] = -CUDART_INF_F;
                        _tmem_load_0[3] = -CUDART_INF_F;
                        _tmem_load_0[4] = -CUDART_INF_F;
                        _tmem_load_0[5] = -CUDART_INF_F;
                        _tmem_load_0[6] = -CUDART_INF_F;
                        _tmem_load_0[7] = -CUDART_INF_F;
                    } else {
                        #pragma unroll
                        for (int h = 0; h < 8; h++) {
                            _tmem_load_0[h] = _tmem_load_0[h] * softmax_scale_log2;
                        }
                    }
                    float tile_max[8];
                    #pragma unroll
                    for (int h_1 = 0; h_1 < 8; h_1++) {
                        tile_max[h_1] = _tmem_load_0[h_1];
                        float _warp_redux_0;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_0) : "f"(tile_max[h_1]));
                        tile_max[h_1] = _warp_redux_0;
                    }
                    if (lane == 0) {
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 8; h_2++) {
                            smem_warp_max[local_warp * 8 + h_2] = tile_max[h_2];
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    float alpha[8];
                    #pragma unroll
                    for (int h_3 = 0; h_3 < 8; h_3++) {
                        float m0 = smem_warp_max[h_3];
                        float m1 = smem_warp_max[8 + h_3];
                        float m2 = smem_warp_max[16 + h_3];
                        float m3 = smem_warp_max[24 + h_3];
                        float cta_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(cta_max) : "f"(max_noftz(m0, m1)), "f"(max_noftz(m2, m3)));
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(row_max[h_3]), "f"(cta_max));
                        float _exp2_1 = approx_exp2(row_max[h_3] - new_max);
                        alpha[h_3] = _exp2_1;
                        float _exp2_2 = approx_exp2(_tmem_load_0[h_3] - new_max);
                        _tmem_load_0[h_3] = _exp2_2;
                        row_max[h_3] = new_max;
                        row_sum[h_3] = __fmaf_rn(row_sum[h_3], alpha[h_3], _tmem_load_0[h_3]);
                    }
                    if (tile_3 > 0) {
                        mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                        _phase_o_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_1[8];
                        tmem_ld_x8(&_tmem_load_1[0], taddr + 32 + (unsigned int)row_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int h_4 = 0; h_4 < 8; h_4++) {
                            _tmem_load_1[h_4] = _tmem_load_1[h_4] * alpha[h_4];
                        }
                        tmem_st_x8_f32(taddr + 32 + (unsigned int)row_addr, _tmem_load_1);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    int token_in_tile = local_tid;
                    int k_group = token_in_tile / 64;
                    int k_in_group = token_in_tile % 64;
                    #pragma unroll
                    for (int h_5 = 0; h_5 < 8; h_5++) {
                        int raw_byte = k_group * 1024 + h_5 * 128 + k_in_group * 2;
                        int swizzled_byte = raw_byte ^ (raw_byte >> 7 & 7) << 4;
                        smem_p[swizzled_byte / 2] = _tmem_load_0[h_5];
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    int vp_stage = tile_3 % 3;
                    if (elect_sync()) {
                        mbarrier_arrive(vp_full_addr + (vp_stage) * 8);
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
                #pragma unroll
                for (int h_6 = 0; h_6 < 8; h_6++) {
                    float partial = row_sum[h_6];
                    float _warp_reduce_3 = partial;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                    partial = _warp_reduce_3;
                    if (lane == 0) {
                        smem_row_sum[local_warp * 8 + h_6] = partial;
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (lane == 0) {
                    #pragma unroll
                    for (int head = 0; head < 8; head++) {
                        if (local_warp == head / 2) {
                            float total_sum = smem_row_sum[head] + smem_row_sum[8 + head] + smem_row_sum[16 + head] + smem_row_sum[24 + head];
                            int owner_rank = head % 8;
                            int owner_slot = head / 8;
                            int owner_offset = owner_slot * 8 + rank_0_3;
                            uint32_t _mapa_0;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_0) : "r"(reduce_stats_full_addr), "r"(owner_rank));
                            uint32_t _mapa_1;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_1) : "r"(dsm_max_addr + (unsigned int)(owner_offset * 4)), "r"(owner_rank));
                            uint32_t _mapa_2;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_2) : "r"(dsm_sum_addr + (unsigned int)(owner_offset * 4)), "r"(owner_rank));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(_mapa_1), "f"(row_max[head]), "r"(_mapa_0) : "memory");
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(_mapa_2), "f"(total_sum), "r"(_mapa_0) : "memory");
                        }
                    }
                }
                mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                _phase_o_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], taddr + 32 + (unsigned int)row_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int h_7 = 0; h_7 < 8; h_7++) {
                    int owner_rank_1 = h_7 % 8;
                    int owner_slot_1 = h_7 / 8;
                    int owner_offset_1 = owner_slot_1 * 8 + rank_0_3;
                    uint32_t _cluster_mapa_4;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_4) : "r"(dsm_o_addr + (unsigned int)((owner_offset_1 * 128 + local_tid) * 4)), "r"(owner_rank_1));
                    uint32_t _cluster_mapa_5;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_5) : "r"(reduce_o_full_addr), "r"(owner_rank_1));
                    asm volatile(
                        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                        :: "r"(_cluster_mapa_4), "f"(_tmem_load_2[h_7]), "r"(_cluster_mapa_5) : "memory");
                }
            } else {
                if (lane == 0) {
                    #pragma unroll
                    for (int head_1 = 0; head_1 < 8; head_1++) {
                        if (local_warp == head_1 / 2) {
                            int owner_rank_2 = head_1 % 8;
                            int owner_slot_2 = head_1 / 8;
                            int owner_offset_2 = owner_slot_2 * 8 + rank_0_3;
                            uint32_t _mapa_3;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_3) : "r"(reduce_stats_full_addr), "r"(owner_rank_2));
                            uint32_t _mapa_4;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_4) : "r"(dsm_max_addr + (unsigned int)(owner_offset_2 * 4)), "r"(owner_rank_2));
                            uint32_t _mapa_5;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_5) : "r"(dsm_sum_addr + (unsigned int)(owner_offset_2 * 4)), "r"(owner_rank_2));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(_mapa_4), "f"(-CUDART_INF_F), "r"(_mapa_3) : "memory");
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(_mapa_5), "f"(0.0f), "r"(_mapa_3) : "memory");
                        }
                    }
                }
                #pragma unroll
                for (int h_8 = 0; h_8 < 8; h_8++) {
                    int owner_rank_3 = h_8 % 8;
                    int owner_slot_3 = h_8 / 8;
                    int owner_offset_3 = owner_slot_3 * 8 + rank_0_3;
                    uint32_t _cluster_mapa_6;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_6) : "r"(dsm_o_addr + (unsigned int)((owner_offset_3 * 128 + local_tid) * 4)), "r"(owner_rank_3));
                    uint32_t _cluster_mapa_7;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_7) : "r"(reduce_o_full_addr), "r"(owner_rank_3));
                    asm volatile(
                        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                        :: "r"(_cluster_mapa_6), "f"(0.0f), "r"(_cluster_mapa_7) : "memory");
                }
            }
            mbarrier_wait(reduce_stats_full_addr, 0);
            #pragma unroll
            for (int owner_slot_4 = 0; owner_slot_4 < 1; owner_slot_4++) {
                const int owner_base = owner_slot_4 * 8;
                float global_max = dsm_max[owner_base];
                #pragma unroll
                for (int source = 1; source < 8; source++) {
                    global_max = max_noftz(global_max, dsm_max[owner_base + source]);
                }
                float beta[8];
                float denom = 0.0f;
                #pragma unroll
                for (int source_1 = 0; source_1 < 8; source_1++) {
                    float _exp2_8 = approx_exp2(dsm_max[owner_base + source_1] - global_max);
                    beta[source_1] = _exp2_8;
                    denom = __fmaf_rn(beta[source_1], dsm_sum[owner_base + source_1], denom);
                }
                if (owner_slot_4 == 0) {
                    mbarrier_wait(reduce_o_full_addr, 0);
                }
                float acc = 0.0f;
                #pragma unroll
                for (int source_2 = 0; source_2 < 8; source_2++) {
                    const int partial_idx = (owner_base + source_2) * 128;
                    acc = __fmaf_rn(beta[source_2], dsm_o[partial_idx + local_tid], acc);
                }
                float _rcp_9 = approx_rcp(denom);
                float final_o = acc * _rcp_9;
                int out_idx = (blockIdx.x * 8 + rank_0_3 + owner_slot_4 * 8) * 128 + local_tid;
                *(reinterpret_cast<__half*>(O + out_idx) + (0)) = __float2half_rn(final_o);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_done_addr);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
