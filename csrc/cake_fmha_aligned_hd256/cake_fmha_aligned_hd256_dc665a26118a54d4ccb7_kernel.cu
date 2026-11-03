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

#define BLOCK_N 128
#define STEP_N 256
#define HEAD_DIM 256
#define TILE_Q 8
#define PAGE_SIZE 16
#define SEQ_LEN_Q 1
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

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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




__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(8,1,1) void
kernel_cake_fmha_aligned_hd256_dc665a26118a54d4ccb7(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, CakeFastDivmod heads_per_kv, int batch_size, float skip_corr_threshold)
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

    const int mbar_base = smem + 177920;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 64)
    #define s_full_0_addr (mbar_base + 96)
    #define s_full_1_addr (mbar_base + 104)
    #define p_empty_addr (mbar_base + 112)
    #define o_full_addr (mbar_base + 128)
    #define page_full_addr (mbar_base + 144)
    #define page_empty_addr (mbar_base + 192)
    #define s_empty_0_addr (mbar_base + 240)
    #define stats_full_0_addr (mbar_base + 248)
    #define stats_empty_0_addr (mbar_base + 256)
    #define s_empty_1_addr (mbar_base + 264)
    #define stats_full_1_addr (mbar_base + 272)
    #define stats_empty_1_addr (mbar_base + 280)
    #define order_p0_addr (mbar_base + 288)
    #define order_p1_addr (mbar_base + 296)
    #define p_full_addr (mbar_base + 304)
    #define o_empty_addr (mbar_base + 320)
    #define reserved_arv128_addr (mbar_base + 336)
    #define cga_full_addr (mbar_base + 512)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_q_addr = smem + 0;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_kv_addr = smem + 8192;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int smem_v_addr = smem + 8192;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 139264);
    const int smem_p0_addr = smem + 139264;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    int* smem_page_ids = reinterpret_cast<int*>(smem_raw + 143360);
    const int smem_page_ids_addr = smem + 143360;
    unsigned int* smem_cga = reinterpret_cast<unsigned int*>(smem_raw + 144128);
    const int smem_cga_addr = smem + 144128;
    unsigned int* smem_red0 = reinterpret_cast<unsigned int*>(smem_raw + 177408);
    const int smem_red0_addr = smem + 177408;
    unsigned int* smem_red1 = reinterpret_cast<unsigned int*>(smem_raw + 177536);
    const int smem_red1_addr = smem + 177536;
    float* smem_corr_red = reinterpret_cast<float*>(smem_raw + 177664);
    const int smem_corr_red_addr = smem + 177664;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (22 pipeline groups, 0 ordered-sequence groups, 65 barriers)
    // Mbarriers at smem_raw[177920..178440)

    if (warp == 0) {
        // --- pipeline 'q_pipe' ---
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 4 barriers, init_count=1
        // kv_empty: 4 barriers, init_count=1
        // s_full_0: 1 barriers, init_count=1
        // s_full_1: 1 barriers, init_count=1
        // p_empty: 2 barriers, init_count=1
        // --- pipeline 'o_pipe' ---
        // o_full: 2 barriers, init_count=1
        // --- pipeline 'page_pipe' ---
        // page_full: 6 barriers, init_count=32
        // page_empty: 6 barriers, init_count=32
        // s_empty_0: 1 barriers, init_count=128
        // stats_full_0: 1 barriers, init_count=128
        // stats_empty_0: 1 barriers, init_count=128
        // s_empty_1: 1 barriers, init_count=128
        // stats_full_1: 1 barriers, init_count=128
        // stats_empty_1: 1 barriers, init_count=128
        // order_p0: 1 barriers, init_count=128
        // order_p1: 1 barriers, init_count=128
        // p_full: 2 barriers, init_count=128
        // --- pipeline 'o_pipe' ---
        // o_empty: 2 barriers, init_count=128
        // reserved_arv128: 22 barriers, init_count=128
        // cga_full: 1 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        mbarrier_init(smem + 177920 + lane * 8, _mbarrier_init_count_0_0);
        mbarrier_init(smem + 178176 + lane * 8, 128);
        if (lane < 1) {
            mbarrier_init(smem + 178432 + lane * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (128 columns, 112 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 178440);
    if (warp == 0) {
        int _tmem_hold = smem + 178440;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0_c0 = taddr + 80;
    const int tmem_tmem_o0_c1 = taddr + 88;
    const int tmem_tmem_o1_c0 = taddr + 96;
    const int tmem_tmem_o1_c1 = taddr + 104;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            const int warp_in_wg = warp % 4;
            const int lane_s = lane;
            const int wg_tid = warp_in_wg * 32 + lane_s;
            const int col_idx = wg_tid % 4 * 2;
            const int row_idx = warp_in_wg * 32 + lane_s / 4;
            int tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 8 : 0)) + (unsigned int)(warp_in_wg * 32 << 16);
            int tmem_stats_base = taddr + (unsigned int)(((is_wg1 != 0) ? 48 : 16)) + (unsigned int)(warp_in_wg * 32 << 16);
            unsigned int* my_red_ptr = ((is_wg1 != 0) ? (smem_red1) : (smem_red0));
            unsigned int my_red_base = ((is_wg1 != 0) ? smem_red1_addr : smem_red0_addr);
            __nv_bfloat16* my_p_base = ((is_wg1 != 0) ? (smem_p1) : (smem_p0));
            const int named_bar_id = 1 + is_wg1;
            uint32_t _fast_div_q_3 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_3 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_s = (int)_fast_div_q_3;
            const int rank_s = cta_rank;
            int cta_q_s = blockIdx.x / 8;
            int batch_s = blockIdx.z;
            const int slice_idx = warp_in_wg / 2;
            const int warp_idx_in_slice = warp_in_wg % 2;
            const int mtx_idx_s = lane_s / 8;
            const int thr_row_idx_s = lane_s % 8;
            const int seg_col_idx_s = warp_idx_in_slice * 4 + mtx_idx_s ^ thr_row_idx_s;
            const int stsm_offset_s = slice_idx * 8 * 128 + thr_row_idx_s * 128 + seg_col_idx_s * 16;
            float s_vals[8];
            float old_max[2];
            float new_max[2];
            float row_sum[2];
            int seq_len_kv_s = seq_lens_kv[batch_s];
            int valid_kv_s = ((cta_q_s * tokens_per_cta_q_s + (seq_len_kv_s - SEQ_LEN_Q) + tokens_per_cta_q_s) < (seq_len_kv_s) ? (cta_q_s * tokens_per_cta_q_s + (seq_len_kv_s - SEQ_LEN_Q) + tokens_per_cta_q_s) : (seq_len_kv_s));
            int num_ctas_kv_s = (((valid_kv_s + (STEP_N - 1)) / STEP_N) < (8) ? ((valid_kv_s + (STEP_N - 1)) / STEP_N) : (8));
            int num_steps_s = 0;
            if (cta_q_s * tokens_per_cta_q_s < SEQ_LEN_Q && rank_s < num_ctas_kv_s) {
                num_steps_s = (valid_kv_s + (num_ctas_kv_s * STEP_N - 1)) / (num_ctas_kv_s * STEP_N);
            }
            unsigned int _phase_stats_empty_1_0 = 1;
            unsigned int _phase_stats_empty_0_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_order_p1_0 = 0;
            unsigned int _phase_order_p0_0 = 1;
            if (num_steps_s > 0) {
                row_sum[0] = 0.0f;
                row_sum[1] = 0.0f;
                new_max[0] = -3.4028235e+38f;
                new_max[1] = -3.4028235e+38f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    my_red_ptr[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(named_bar_id) : "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
                float adjusted_skip_thr = skip_corr_threshold / softmax_scale_log2;
                int tile_offset_q = cta_q_s * tokens_per_cta_q_s + (seq_len_kv_s - SEQ_LEN_Q);
                if (is_wg1 != 0) {
                    mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                } else {
                    mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                #pragma unroll 1
                for (int lo_s = 0; lo_s < num_steps_s; lo_s++) {
                    int is_last_iter_s = ((num_steps_s <= lo_s + 1) ? 1 : 0);
                    if (is_wg1 != 0) {
                        mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                        _phase_s_full_1_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                        _phase_s_full_0_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    old_max[0] = new_max[0];
                    old_max[1] = new_max[1];
                    int tile_offset_k = ((num_steps_s * rank_s + lo_s) * 2 + is_wg1) * BLOCK_N;
                    if (tile_offset_q >= tile_offset_k + BLOCK_N) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[3]))
                            : "r"(tmem_s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[7]))
                            : "r"(tmem_s_base + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int c = 0; c < 2; c++) {
                            new_max[c] = fmaxf(new_max[c], s_vals[c]);
                            new_max[c] = fmaxf(new_max[c], s_vals[c + 2]);
                            new_max[c] = fmaxf(new_max[c], s_vals[c + 4]);
                            new_max[c] = fmaxf(new_max[c], s_vals[c + 6]);
                        }
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 2; c_1++) {
                            float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, new_max[c_1], 16);
                            new_max[c_1] = fmaxf(_shfl_xor_sync_1, new_max[c_1]);
                            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, new_max[c_1], 8);
                            new_max[c_1] = fmaxf(_shfl_xor_sync_2, new_max[c_1]);
                        }
                        if (lane_s < 8) {
                            uint32_t _amf_u_4 = __float_as_uint(new_max[0]);
                            uint32_t _amf_mask_4 = -int32_t(_amf_u_4 >> 31) | 0x80000000u;
                            unsigned int _atomic_max_float_encode_3 = _amf_u_4 ^ _amf_mask_4;
                            atomicMax(&my_red_ptr[col_idx], _atomic_max_float_encode_3);
                            uint32_t _amf_u_6 = __float_as_uint(new_max[1]);
                            uint32_t _amf_mask_6 = -int32_t(_amf_u_6 >> 31) | 0x80000000u;
                            unsigned int _atomic_max_float_encode_5 = _amf_u_6 ^ _amf_mask_6;
                            atomicMax(&my_red_ptr[col_idx + 1], _atomic_max_float_encode_5);
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(named_bar_id) : "memory");
                        unsigned int red_pair[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&red_pair[0])), "=r"(*reinterpret_cast<uint32_t*>(&red_pair[(0) + 1]))
                            : "r"(my_red_base + (unsigned int)(col_idx * 4)));
                        uint32_t _amf_u_8 = red_pair[0];
                        uint32_t _amf_mask_8 = ((_amf_u_8 >> 31) - 1u) | 0x80000000u;
                        float _atomic_max_float_decode_7 = __uint_as_float(_amf_u_8 ^ _amf_mask_8);
                        new_max[0] = _atomic_max_float_decode_7;
                        uint32_t _amf_u_10 = red_pair[1];
                        uint32_t _amf_mask_10 = ((_amf_u_10 >> 31) - 1u) | 0x80000000u;
                        float _atomic_max_float_decode_9 = __uint_as_float(_amf_u_10 ^ _amf_mask_10);
                        new_max[1] = _atomic_max_float_decode_9;
                    } else {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[3]))
                            : "r"(tmem_s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[7]))
                            : "r"(tmem_s_base + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        uint32_t _fast_div_q_4 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)col_idx), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)col_idx);
                        uint32_t _fast_div_r_4 = (uint32_t)((unsigned int)col_idx) - _fast_div_q_4 * (uint32_t)(heads_per_kv.divisor);
                        uint32_t _fast_div_q_5 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 1)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 1));
                        uint32_t _fast_div_r_5 = (uint32_t)((unsigned int)(col_idx + 1)) - _fast_div_q_5 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q0 = ((tile_offset_q + (int)_fast_div_q_4) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fast_div_q_4) : (seq_len_kv_s - 1));
                        int idx_q1 = ((tile_offset_q + (int)_fast_div_q_5) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fast_div_q_5) : (seq_len_kv_s - 1));
                        int idx_k0 = tile_offset_k + row_idx;
                        int idx_k1 = idx_k0 + 8;
                        int idx_k2 = idx_k0 + 16;
                        int idx_k3 = idx_k0 + 24;
                        if (idx_k0 > idx_q0) {
                            s_vals[0] = -3.4028235e+38f;
                        }
                        if (idx_k1 > idx_q0) {
                            s_vals[2] = -3.4028235e+38f;
                        }
                        if (idx_k2 > idx_q0) {
                            s_vals[4] = -3.4028235e+38f;
                        }
                        if (idx_k3 > idx_q0) {
                            s_vals[6] = -3.4028235e+38f;
                        }
                        if (idx_k0 > idx_q1) {
                            s_vals[1] = -3.4028235e+38f;
                        }
                        if (idx_k1 > idx_q1) {
                            s_vals[3] = -3.4028235e+38f;
                        }
                        if (idx_k2 > idx_q1) {
                            s_vals[5] = -3.4028235e+38f;
                        }
                        if (idx_k3 > idx_q1) {
                            s_vals[7] = -3.4028235e+38f;
                        }
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 2; c_2++) {
                            new_max[c_2] = fmaxf(new_max[c_2], s_vals[c_2]);
                            new_max[c_2] = fmaxf(new_max[c_2], s_vals[c_2 + 2]);
                            new_max[c_2] = fmaxf(new_max[c_2], s_vals[c_2 + 4]);
                            new_max[c_2] = fmaxf(new_max[c_2], s_vals[c_2 + 6]);
                        }
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 2; c_3++) {
                            float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, new_max[c_3], 16);
                            new_max[c_3] = fmaxf(_shfl_xor_sync_11, new_max[c_3]);
                            float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, new_max[c_3], 8);
                            new_max[c_3] = fmaxf(_shfl_xor_sync_12, new_max[c_3]);
                        }
                        if (lane_s < 8) {
                            uint32_t _amf_u_14 = __float_as_uint(new_max[0]);
                            uint32_t _amf_mask_14 = -int32_t(_amf_u_14 >> 31) | 0x80000000u;
                            unsigned int _atomic_max_float_encode_13 = _amf_u_14 ^ _amf_mask_14;
                            atomicMax(&my_red_ptr[col_idx], _atomic_max_float_encode_13);
                            uint32_t _amf_u_16 = __float_as_uint(new_max[1]);
                            uint32_t _amf_mask_16 = -int32_t(_amf_u_16 >> 31) | 0x80000000u;
                            unsigned int _atomic_max_float_encode_15 = _amf_u_16 ^ _amf_mask_16;
                            atomicMax(&my_red_ptr[col_idx + 1], _atomic_max_float_encode_15);
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(named_bar_id) : "memory");
                        unsigned int red_pair_1[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&red_pair_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&red_pair_1[(0) + 1]))
                            : "r"(my_red_base + (unsigned int)(col_idx * 4)));
                        uint32_t _amf_u_18 = red_pair_1[0];
                        uint32_t _amf_mask_18 = ((_amf_u_18 >> 31) - 1u) | 0x80000000u;
                        float _atomic_max_float_decode_17 = __uint_as_float(_amf_u_18 ^ _amf_mask_18);
                        new_max[0] = _atomic_max_float_decode_17;
                        uint32_t _amf_u_20 = red_pair_1[1];
                        uint32_t _amf_mask_20 = ((_amf_u_20 >> 31) - 1u) | 0x80000000u;
                        float _atomic_max_float_decode_19 = __uint_as_float(_amf_u_20 ^ _amf_mask_20);
                        new_max[1] = _atomic_max_float_decode_19;
                    }
                    if (adjusted_skip_thr > 0.0f) {
                        #pragma unroll
                        for (int c_4 = 0; c_4 < 2; c_4++) {
                            if (adjusted_skip_thr >= new_max[c_4] - old_max[c_4]) {
                                new_max[c_4] = old_max[c_4];
                            }
                        }
                    }
                    float stats_s[4];
                    stats_s[0] = old_max[0];
                    stats_s[1] = old_max[1];
                    stats_s[2] = new_max[0];
                    stats_s[3] = new_max[1];
                    tmem_st_x4_f32(tmem_stats_base, stats_s);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (is_wg1 != 0) {
                        _phase_stats_empty_1_0 ^= 1;
                        mbarrier_arrive(stats_full_1_addr);
                    } else {
                        _phase_stats_empty_0_0 ^= 1;
                        mbarrier_arrive(stats_full_0_addr);
                    }
                    float neg_scaled_max[2];
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 2; c_5++) {
                        float safe_max = ((new_max[c_5] == -3.4028235e+38f) ? 0.0f : new_max[c_5]);
                        neg_scaled_max[c_5] = safe_max * (-softmax_scale_log2);
                    }
                    s_vals[0] = __fmaf_rn(softmax_scale_log2, s_vals[0], neg_scaled_max[0]);
                    s_vals[1] = __fmaf_rn(softmax_scale_log2, s_vals[1], neg_scaled_max[1]);
                    s_vals[2] = __fmaf_rn(softmax_scale_log2, s_vals[2], neg_scaled_max[0]);
                    s_vals[3] = __fmaf_rn(softmax_scale_log2, s_vals[3], neg_scaled_max[1]);
                    if (is_wg1 != 0) {
                        mbarrier_wait(order_p1_addr, _phase_order_p1_0);
                        _phase_order_p1_0 ^= 1;
                    } else {
                        mbarrier_wait(order_p0_addr, _phase_order_p0_0);
                        _phase_order_p0_0 ^= 1;
                    }
                    float _exp2_21 = approx_exp2(s_vals[0]);
                    s_vals[0] = _exp2_21;
                    s_vals[4] = __fmaf_rn(softmax_scale_log2, s_vals[4], neg_scaled_max[0]);
                    s_vals[5] = __fmaf_rn(softmax_scale_log2, s_vals[5], neg_scaled_max[1]);
                    float _exp2_22 = approx_exp2(s_vals[1]);
                    s_vals[1] = _exp2_22;
                    if (is_wg1 != 0) {
                        mbarrier_arrive(order_p0_addr);
                    } else {
                        mbarrier_arrive(order_p1_addr);
                    }
                    float _exp2_23 = approx_exp2(s_vals[2]);
                    s_vals[2] = _exp2_23;
                    s_vals[6] = __fmaf_rn(softmax_scale_log2, s_vals[6], neg_scaled_max[0]);
                    s_vals[7] = __fmaf_rn(softmax_scale_log2, s_vals[7], neg_scaled_max[1]);
                    float _exp2_24 = approx_exp2(s_vals[3]);
                    s_vals[3] = _exp2_24;
                    float _exp2_25 = approx_exp2(s_vals[4]);
                    s_vals[4] = _exp2_25;
                    float _exp2_26 = approx_exp2(s_vals[5]);
                    s_vals[5] = _exp2_26;
                    float _exp2_27 = approx_exp2(s_vals[6]);
                    s_vals[6] = _exp2_27;
                    float _exp2_28 = approx_exp2(s_vals[7]);
                    s_vals[7] = _exp2_28;
                    unsigned int regs_p[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_vals[_lp*2 + 0], s_vals[_lp*2+1 + 0]));
                        regs_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset_s)), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(s_empty_1_addr);
                    } else {
                        mbarrier_arrive(s_empty_0_addr);
                    }
                    float max_diff0 = old_max[0] - new_max[0];
                    float max_diff1 = old_max[1] - new_max[1];
                    float scale2[2];
                    scale2[0] = 1.0f;
                    scale2[1] = 1.0f;
                    if (max_diff0 != 0.0f || max_diff1 != 0.0f) {
                        float _exp2_29 = approx_exp2(softmax_scale_log2 * max_diff0);
                        scale2[0] = _exp2_29;
                        float _exp2_30 = approx_exp2(softmax_scale_log2 * max_diff1);
                        scale2[1] = _exp2_30;
                    }
                    #pragma unroll
                    for (int c_6 = 0; c_6 < 2; c_6++) {
                        row_sum[c_6] = __fmaf_rn(scale2[c_6], row_sum[c_6], s_vals[c_6]);
                        row_sum[c_6] = row_sum[c_6] + s_vals[c_6 + 2];
                        row_sum[c_6] = row_sum[c_6] + s_vals[c_6 + 4];
                        row_sum[c_6] = row_sum[c_6] + s_vals[c_6 + 6];
                    }
                    if (is_last_iter_s != 0) {
                        float final_stats_s[4];
                        final_stats_s[0] = row_sum[0];
                        final_stats_s[1] = row_sum[1];
                        final_stats_s[2] = new_max[0];
                        final_stats_s[3] = new_max[1];
                        tmem_st_x4_f32(tmem_stats_base, final_stats_s);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (is_wg1 != 0) {
                            _phase_stats_empty_1_0 ^= 1;
                            mbarrier_arrive(stats_full_1_addr);
                        } else {
                            _phase_stats_empty_0_0 ^= 1;
                            mbarrier_arrive(stats_full_0_addr);
                        }
                        if (is_wg1 != 0) {
                            mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                        } else {
                            mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int warp_in_wg_c = warp % 4;
            const int lane_c = lane;
            const int wg_tid_c = warp_in_wg_c * 32 + lane_c;
            const int col_idx_c = wg_tid_c % 4 * 2;
            const int corr_row = warp_in_wg_c * 32 << 16;
            unsigned int* o_words = reinterpret_cast<unsigned int*>(O);
            const int heads_per_kv_c = num_q_heads / num_kv_heads;
            uint32_t _fast_div_q_6 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_6 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_6 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_c = (int)_fast_div_q_6;
            int oc_stage = 0;
            int or_stage = 0;
            const int rank_c = cta_rank;
            int cta_q_c = blockIdx.x / 8;
            int head_c = blockIdx.y;
            int batch_c = blockIdx.z;
            const int slice_idx_c = warp_in_wg_c / 2;
            const int warp_idx_in_slice_c = warp_in_wg_c % 2;
            const int mtx_idx_c = lane_c / 8;
            const int thr_row_idx_c = lane_c % 8;
            const int seg_col_idx_c = warp_idx_in_slice_c * 4 + mtx_idx_c ^ thr_row_idx_c;
            const int stsm_offset_c = slice_idx_c * 8 * 128 + thr_row_idx_c * 128 + seg_col_idx_c * 16;
            const int smem_row_c = wg_tid_c / 8;
            const int load_smem_off_c = wg_tid_c * 16 ^ smem_row_c % 8 * 16;
            const int tok_row_c = smem_row_c % 8;
            const int dst_col_bytes_c = smem_row_c / 8 * 128 + wg_tid_c % 8 * 16;
            const int red_row_in_slice = wg_tid_c / 32;
            const int red_hd = wg_tid_c % 32 * 8;
            int seq_len_kv_c = seq_lens_kv[batch_c];
            int valid_kv_c = ((cta_q_c * tokens_per_cta_q_c + (seq_len_kv_c - SEQ_LEN_Q) + tokens_per_cta_q_c) < (seq_len_kv_c) ? (cta_q_c * tokens_per_cta_q_c + (seq_len_kv_c - SEQ_LEN_Q) + tokens_per_cta_q_c) : (seq_len_kv_c));
            int num_ctas_kv_c = (((valid_kv_c + (STEP_N - 1)) / STEP_N) < (8) ? ((valid_kv_c + (STEP_N - 1)) / STEP_N) : (8));
            int num_steps_c = 0;
            if (cta_q_c * tokens_per_cta_q_c < SEQ_LEN_Q && rank_c < num_ctas_kv_c) {
                num_steps_c = (valid_kv_c + (num_ctas_kv_c * STEP_N - 1)) / (num_ctas_kv_c * STEP_N);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_stats_full_0_0 = 0;
            unsigned int _phase_stats_full_1_0 = 0;
            unsigned int _phase_o_full = 0;
            if (num_steps_c > 0) {
                mbarrier_wait(stats_full_0_addr, _phase_stats_full_0_0);
                _phase_stats_full_0_0 ^= 1;
                mbarrier_arrive(stats_empty_0_addr);
                mbarrier_wait(stats_full_1_addr, _phase_stats_full_1_0);
                _phase_stats_full_1_0 ^= 1;
                mbarrier_arrive(stats_empty_1_addr);
                #pragma unroll 1
                for (int _lo_c = 0; _lo_c < num_steps_c - 1; _lo_c++) {
                    mbarrier_wait(stats_full_0_addr, _phase_stats_full_0_0);
                    _phase_stats_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_empty_0_addr);
                    mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                    oc_stage += 1;
                    if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scales[2];
                    scales[0] = 1.0f;
                    scales[1] = 1.0f;
                    float max_diff0_1 = _tmem_load_0[0] - _tmem_load_0[2];
                    float max_diff1_1 = _tmem_load_0[1] - _tmem_load_0[3];
                    if (max_diff0_1 != 0.0f || max_diff1_1 != 0.0f) {
                        float _exp2_0 = approx_exp2(softmax_scale_log2 * max_diff0_1);
                        scales[0] = _exp2_0;
                        float _exp2_1 = approx_exp2(softmax_scale_log2 * max_diff1_1);
                        scales[1] = _exp2_1;
                    }
                    int _warp_vote_2 = __all_sync(0xFFFFFFFF, scales[0] == 1.0f && scales[1] == 1.0f);
                    if (_warp_vote_2 == 0) {
                        float o_vals[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[3]))
                            : "r"(taddr + 80 + (unsigned int)corr_row));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[7]))
                            : "r"(taddr + 80 + (unsigned int)corr_row + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        o_vals[0] = o_vals[0] * scales[0];
                        o_vals[1] = o_vals[1] * scales[1];
                        o_vals[2] = o_vals[2] * scales[0];
                        o_vals[3] = o_vals[3] * scales[1];
                        o_vals[4] = o_vals[4] * scales[0];
                        o_vals[5] = o_vals[5] * scales[1];
                        o_vals[6] = o_vals[6] * scales[0];
                        o_vals[7] = o_vals[7] * scales[1];
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + (unsigned int)corr_row + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 4)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 4)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 4)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 4)[3])));
                        float o_vals_0[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[3]))
                            : "r"(taddr + 88 + (unsigned int)corr_row));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_0[7]))
                            : "r"(taddr + 88 + (unsigned int)corr_row + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        o_vals_0[0] = o_vals_0[0] * scales[0];
                        o_vals_0[1] = o_vals_0[1] * scales[1];
                        o_vals_0[2] = o_vals_0[2] * scales[0];
                        o_vals_0[3] = o_vals_0[3] * scales[1];
                        o_vals_0[4] = o_vals_0[4] * scales[0];
                        o_vals_0[5] = o_vals_0[5] * scales[1];
                        o_vals_0[6] = o_vals_0[6] * scales[0];
                        o_vals_0[7] = o_vals_0[7] * scales[1];
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_0[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + (unsigned int)corr_row + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_0 + 4)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_0 + 4)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_0 + 4)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_0 + 4)[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr + (or_stage) * 8);
                    or_stage += 1;
                    if (or_stage == 2) { or_stage = 0; }
                    mbarrier_wait(stats_full_1_addr, _phase_stats_full_1_0);
                    _phase_stats_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_2[4];
                    tmem_ld_x4(&_tmem_load_2[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_empty_1_addr);
                    mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                    oc_stage += 1;
                    if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scales_0[2];
                    scales_0[0] = 1.0f;
                    scales_0[1] = 1.0f;
                    float max_diff0_1_1 = _tmem_load_2[0] - _tmem_load_2[2];
                    float max_diff1_2 = _tmem_load_2[1] - _tmem_load_2[3];
                    if (max_diff0_1_1 != 0.0f || max_diff1_2 != 0.0f) {
                        float _exp2_3 = approx_exp2(softmax_scale_log2 * max_diff0_1_1);
                        scales_0[0] = _exp2_3;
                        float _exp2_4 = approx_exp2(softmax_scale_log2 * max_diff1_2);
                        scales_0[1] = _exp2_4;
                    }
                    int _warp_vote_5 = __all_sync(0xFFFFFFFF, scales_0[0] == 1.0f && scales_0[1] == 1.0f);
                    if (_warp_vote_5 == 0) {
                        float o_vals_1[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[3]))
                            : "r"(taddr + 96 + (unsigned int)corr_row));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[7]))
                            : "r"(taddr + 96 + (unsigned int)corr_row + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        o_vals_1[0] = o_vals_1[0] * scales_0[0];
                        o_vals_1[1] = o_vals_1[1] * scales_0[1];
                        o_vals_1[2] = o_vals_1[2] * scales_0[0];
                        o_vals_1[3] = o_vals_1[3] * scales_0[1];
                        o_vals_1[4] = o_vals_1[4] * scales_0[0];
                        o_vals_1[5] = o_vals_1[5] * scales_0[1];
                        o_vals_1[6] = o_vals_1[6] * scales_0[0];
                        o_vals_1[7] = o_vals_1[7] * scales_0[1];
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 96 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 96 + (unsigned int)corr_row + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 4)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 4)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 4)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 4)[3])));
                        float o_vals_1_1[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[3]))
                            : "r"(taddr + 104 + (unsigned int)corr_row));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1_1[7]))
                            : "r"(taddr + 104 + (unsigned int)corr_row + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        o_vals_1_1[0] = o_vals_1_1[0] * scales_0[0];
                        o_vals_1_1[1] = o_vals_1_1[1] * scales_0[1];
                        o_vals_1_1[2] = o_vals_1_1[2] * scales_0[0];
                        o_vals_1_1[3] = o_vals_1_1[3] * scales_0[1];
                        o_vals_1_1[4] = o_vals_1_1[4] * scales_0[0];
                        o_vals_1_1[5] = o_vals_1_1[5] * scales_0[1];
                        o_vals_1_1[6] = o_vals_1_1[6] * scales_0[0];
                        o_vals_1_1[7] = o_vals_1_1[7] * scales_0[1];
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 104 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1_1[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 104 + (unsigned int)corr_row + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1_1 + 4)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1_1 + 4)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1_1 + 4)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1_1 + 4)[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr + (or_stage) * 8);
                    or_stage += 1;
                    if (or_stage == 2) { or_stage = 0; }
                }
                mbarrier_wait(stats_full_0_addr, _phase_stats_full_0_0);
                _phase_stats_full_0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float tmem_load[4];
                tmem_ld_x4(&tmem_load[0], taddr + 16 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(stats_empty_0_addr);
                mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                oc_stage += 1;
                if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                mbarrier_wait(stats_full_1_addr, _phase_stats_full_1_0);
                _phase_stats_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float tmem_load_0[4];
                tmem_ld_x4(&tmem_load_0[0], taddr + 48 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(stats_empty_1_addr);
                mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                oc_stage += 1;
                if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int num_valid_rows_o = SEQ_LEN_Q - cta_q_c * tokens_per_cta_q_c;
                if (num_valid_rows_o > tokens_per_cta_q_c) {
                    num_valid_rows_o = tokens_per_cta_q_c;
                }
                num_valid_rows_o = num_valid_rows_o * heads_per_kv_c;
                int seq_offset_o = batch_c * SEQ_LEN_Q + cta_q_c * tokens_per_cta_q_c;
                int heads_o = heads_per_kv_c;
                if (heads_o > TILE_Q) {
                    heads_o = TILE_Q;
                }
                int cta_offset_o = seq_offset_o * (num_q_heads * HEAD_DIM) + head_c * heads_o * HEAD_DIM;
                int num_valid_slices = (num_valid_rows_o + 3) / 4;
                int num_slices_per_cta = (num_valid_slices + (num_ctas_kv_c - 1)) / num_ctas_kv_c;
                int rows_per_cta = num_slices_per_cta * 4;
                int num_valid_rows_in_cta = ((rows_per_cta) < (num_valid_rows_o - rank_c * rows_per_cta) ? (rows_per_cta) : (num_valid_rows_o - rank_c * rows_per_cta));
                int cga_stats_off = num_ctas_kv_c * rows_per_cta * 512;
                int start_slice = rank_c * num_slices_per_cta;
                int end_slice = ((start_slice + num_slices_per_cta) < (num_valid_slices) ? (start_slice + num_slices_per_cta) : (num_valid_slices));
                int o_dst = tok_row_c / rows_per_cta;
                int o_row_off = (rank_c * rows_per_cta + tok_row_c % rows_per_cta) * 512 + dst_col_bytes_c;
                int stat_sel = wg_tid_c / 4;
                int stat_row = wg_tid_c % 4 * 2 + stat_sel;
                int stat_dst = stat_row / rows_per_cta;
                int stat_off = cga_stats_off + (rank_c * rows_per_cta + stat_row % rows_per_cta) * 8;
                float merged_max[2];
                float scales_i0[2];
                float scales_i1[2];
                scales_i0[0] = 1.0f;
                scales_i0[1] = 1.0f;
                scales_i1[0] = 1.0f;
                scales_i1[1] = 1.0f;
                merged_max[0] = fmaxf(tmem_load[2], tmem_load_0[2]);
                merged_max[1] = fmaxf(tmem_load[3], tmem_load_0[3]);
                float max_diff_i0_0 = tmem_load[2] - merged_max[0];
                float max_diff_i0_1 = tmem_load[3] - merged_max[1];
                float max_diff_i1_0 = tmem_load_0[2] - merged_max[0];
                float max_diff_i1_1 = tmem_load_0[3] - merged_max[1];
                if (max_diff_i0_0 != 0.0f || max_diff_i0_1 != 0.0f || max_diff_i1_0 != 0.0f || max_diff_i1_1 != 0.0f) {
                    float _exp2_6 = approx_exp2(softmax_scale_log2 * max_diff_i0_0);
                    scales_i0[0] = _exp2_6;
                    float _exp2_7 = approx_exp2(softmax_scale_log2 * max_diff_i0_1);
                    scales_i0[1] = _exp2_7;
                    float _exp2_8 = approx_exp2(softmax_scale_log2 * max_diff_i1_0);
                    scales_i1[0] = _exp2_8;
                    float _exp2_9 = approx_exp2(softmax_scale_log2 * max_diff_i1_1);
                    scales_i1[1] = _exp2_9;
                }
                float sums[2];
                sums[0] = __fmaf_rn(scales_i0[0], tmem_load[0], scales_i1[0] * tmem_load_0[0]);
                sums[1] = __fmaf_rn(scales_i0[1], tmem_load[1], scales_i1[1] * tmem_load_0[1]);
                #pragma unroll
                for (int c_7 = 0; c_7 < 2; c_7++) {
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, sums[c_7], 16);
                    sums[c_7] = sums[c_7] + _shfl_xor_sync_10;
                    float _shfl_xor_sync_11_1 = __shfl_xor_sync(0xFFFFFFFF, sums[c_7], 8);
                    sums[c_7] = sums[c_7] + _shfl_xor_sync_11_1;
                    float _shfl_xor_sync_12_1 = __shfl_xor_sync(0xFFFFFFFF, sums[c_7], 4);
                    sums[c_7] = sums[c_7] + _shfl_xor_sync_12_1;
                }
                if (lane_c < 4) {
                    asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(smem_corr_red_addr + (unsigned int)((warp_in_wg_c * 8 + col_idx_c) * 4)), "f"(sums[0]), "f"(sums[1]) : "memory");
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float pair_c[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pair_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&pair_c[(0) + 1]))
                    : "r"(smem_corr_red_addr + (unsigned int)(col_idx_c * 4)));
                float total_sum0 = pair_c[0];
                float total_sum1 = pair_c[1];
                #pragma unroll
                for (int w = 1; w < 4; w++) {
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&pair_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&pair_c[(0) + 1]))
                        : "r"(smem_corr_red_addr + (unsigned int)((w * 8 + col_idx_c) * 4)));
                    total_sum0 = total_sum0 + pair_c[0];
                    total_sum1 = total_sum1 + pair_c[1];
                }
                if (wg_tid_c < 8) {
                    float stat_max = merged_max[0];
                    float stat_sum = total_sum0;
                    if (stat_sel != 0) {
                        stat_max = merged_max[1];
                        stat_sum = total_sum1;
                    }
                    if (stat_row < num_valid_rows_o) {
                        uint32_t stat_addr;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(stat_addr) : "r"(smem_cga_addr + (unsigned int)stat_off), "r"(stat_dst));
                        uint32_t _mbar_peer_13;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_13) : "r"(cga_full_addr), "r"(stat_dst));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(stat_addr), "f"(stat_max), "r"(_mbar_peer_13) : "memory");
                        uint32_t _mbar_peer_14;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_14) : "r"(cga_full_addr), "r"(stat_dst));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(stat_addr + 4), "f"(stat_sum), "r"(_mbar_peer_14) : "memory");
                    }
                }
                float o0_vals[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[3]))
                    : "r"(taddr + 80 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[7]))
                    : "r"(taddr + 80 + (unsigned int)corr_row + 1048576));
                float o1_vals[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[3]))
                    : "r"(taddr + 96 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[7]))
                    : "r"(taddr + 96 + (unsigned int)corr_row + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float merged_o[8];
                merged_o[0] = __fmaf_rn(o0_vals[0], scales_i0[0], o1_vals[0] * scales_i1[0]);
                merged_o[1] = __fmaf_rn(o0_vals[1], scales_i0[1], o1_vals[1] * scales_i1[1]);
                merged_o[2] = __fmaf_rn(o0_vals[2], scales_i0[0], o1_vals[2] * scales_i1[0]);
                merged_o[3] = __fmaf_rn(o0_vals[3], scales_i0[1], o1_vals[3] * scales_i1[1]);
                merged_o[4] = __fmaf_rn(o0_vals[4], scales_i0[0], o1_vals[4] * scales_i1[0]);
                merged_o[5] = __fmaf_rn(o0_vals[5], scales_i0[1], o1_vals[5] * scales_i1[1]);
                merged_o[6] = __fmaf_rn(o0_vals[6], scales_i0[0], o1_vals[6] * scales_i1[0]);
                merged_o[7] = __fmaf_rn(o0_vals[7], scales_i0[1], o1_vals[7] * scales_i1[1]);
                unsigned int regs_o[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_o[_lp*2 + 0], merged_o[_lp*2+1 + 0]));
                    regs_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_p0) + stsm_offset_c)), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[3]))
                    : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (tok_row_c < num_valid_rows_o) {
                    unsigned int o_vec[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                        : "r"(smem_p0_addr + (unsigned int)load_smem_off_c));
                    int o_off = o_row_off;
                    uint32_t _cluster_mapa_15;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_15) : "r"(smem_cga_addr + (unsigned int)o_off), "r"(o_dst));
                    uint32_t _mbar_peer_16;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_16) : "r"(cga_full_addr), "r"(o_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_15), "r"(o_vec[0]), "r"(o_vec[1]), "r"(o_vec[2]), "r"(o_vec[3]), "r"(_mbar_peer_16) : "memory");
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                float o0_vals_0[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[3]))
                    : "r"(taddr + 88 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals_0[7]))
                    : "r"(taddr + 88 + (unsigned int)corr_row + 1048576));
                float o1_vals_0[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[3]))
                    : "r"(taddr + 104 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals_0[7]))
                    : "r"(taddr + 104 + (unsigned int)corr_row + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float merged_o_0[8];
                merged_o_0[0] = __fmaf_rn(o0_vals_0[0], scales_i0[0], o1_vals_0[0] * scales_i1[0]);
                merged_o_0[1] = __fmaf_rn(o0_vals_0[1], scales_i0[1], o1_vals_0[1] * scales_i1[1]);
                merged_o_0[2] = __fmaf_rn(o0_vals_0[2], scales_i0[0], o1_vals_0[2] * scales_i1[0]);
                merged_o_0[3] = __fmaf_rn(o0_vals_0[3], scales_i0[1], o1_vals_0[3] * scales_i1[1]);
                merged_o_0[4] = __fmaf_rn(o0_vals_0[4], scales_i0[0], o1_vals_0[4] * scales_i1[0]);
                merged_o_0[5] = __fmaf_rn(o0_vals_0[5], scales_i0[1], o1_vals_0[5] * scales_i1[1]);
                merged_o_0[6] = __fmaf_rn(o0_vals_0[6], scales_i0[0], o1_vals_0[6] * scales_i1[0]);
                merged_o_0[7] = __fmaf_rn(o0_vals_0[7], scales_i0[1], o1_vals_0[7] * scales_i1[1]);
                unsigned int regs_o_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_o_0[_lp*2 + 0], merged_o_0[_lp*2+1 + 0]));
                    regs_o_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_p0) + stsm_offset_c)), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_0[3]))
                    : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (tok_row_c < num_valid_rows_o) {
                    unsigned int o_vec_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[(0) + 3]))
                        : "r"(smem_p0_addr + (unsigned int)load_smem_off_c));
                    int o_off_1 = o_row_off + 256;
                    uint32_t _cluster_mapa_17;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_17) : "r"(smem_cga_addr + (unsigned int)o_off_1), "r"(o_dst));
                    uint32_t _mbar_peer_18;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_18) : "r"(cga_full_addr), "r"(o_dst));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_17), "r"(o_vec_1[0]), "r"(o_vec_1[1]), "r"(o_vec_1[2]), "r"(o_vec_1[3]), "r"(_mbar_peer_18) : "memory");
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (start_slice < num_valid_slices) {
                    if (wg_tid_c == 0) {
                        mbarrier_arrive_expect_tx(cga_full_addr, num_valid_rows_in_cta * num_ctas_kv_c * 520);
                    }
                    mbarrier_wait_cluster_hint(cga_full_addr, 0, 10000000);
                    #pragma unroll 1
                    for (int slice_r = start_slice; slice_r < end_slice; slice_r++) {
                        int row_r = slice_r * 4 + red_row_in_slice;
                        int valid_row_r = ((row_r) < (num_valid_rows_o - 1) ? (row_r) : (num_valid_rows_o - 1));
                        int load_row_r = valid_row_r - start_slice * 4;
                        float sum_v = 0.0f;
                        float old_max_v = -3.4028235e+38f;
                        float max_v = -3.4028235e+38f;
                        float acc[8];
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        acc[4] = 0.0f;
                        acc[5] = 0.0f;
                        acc[6] = 0.0f;
                        acc[7] = 0.0f;
                        int k_r = 0;
                        float local_max = 0.0f;
                        float local_sum = 0.0f;
                        float corr0 = 1.0f;
                        float corr1 = 0.0f;
                        float part_f[8];
                        #pragma unroll 1
                        for (int ii_r = 0; ii_r < num_ctas_kv_c; ii_r += 4) {
                            float part_stats[8];
                            unsigned int part_o[16];
                            int _min_19 = ((ii_r) < (num_ctas_kv_c - 1) ? (ii_r) : (num_ctas_kv_c - 1));
                            k_r = _min_19;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(0) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(0) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * HEAD_DIM + red_hd) * 2)));
                            int _min_20 = ((ii_r + 1) < (num_ctas_kv_c - 1) ? (ii_r + 1) : (num_ctas_kv_c - 1));
                            k_r = _min_20;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[2])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(2) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(4) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * HEAD_DIM + red_hd) * 2)));
                            int _min_21 = ((ii_r + 2) < (num_ctas_kv_c - 1) ? (ii_r + 2) : (num_ctas_kv_c - 1));
                            k_r = _min_21;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(4) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(8) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * HEAD_DIM + red_hd) * 2)));
                            int _min_22 = ((ii_r + 3) < (num_ctas_kv_c - 1) ? (ii_r + 3) : (num_ctas_kv_c - 1));
                            k_r = _min_22;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[6])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(6) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(12) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * HEAD_DIM + red_hd) * 2)));
                            local_max = part_stats[0];
                            local_sum = part_stats[1];
                            max_v = fmaxf(max_v, local_max);
                            corr0 = 1.0f;
                            corr1 = 0.0f;
                            if (ii_r < num_ctas_kv_c) {
                                float _exp2_23_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_23_1;
                                float _exp2_24_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_24_1;
                            }
                            old_max_v = max_v;
                            sum_v = __fmaf_rn(sum_v, corr0, local_sum * corr1);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&part_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(part_o[_pair]) << 16);
                                (&part_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(part_o[_pair]) & 0xffff0000u);
                            }
                            acc[0] = acc[0] * corr0;
                            acc[0] = __fmaf_rn(part_f[0], corr1, acc[0]);
                            acc[1] = acc[1] * corr0;
                            acc[1] = __fmaf_rn(part_f[1], corr1, acc[1]);
                            acc[2] = acc[2] * corr0;
                            acc[2] = __fmaf_rn(part_f[2], corr1, acc[2]);
                            acc[3] = acc[3] * corr0;
                            acc[3] = __fmaf_rn(part_f[3], corr1, acc[3]);
                            acc[4] = acc[4] * corr0;
                            acc[4] = __fmaf_rn(part_f[4], corr1, acc[4]);
                            acc[5] = acc[5] * corr0;
                            acc[5] = __fmaf_rn(part_f[5], corr1, acc[5]);
                            acc[6] = acc[6] * corr0;
                            acc[6] = __fmaf_rn(part_f[6], corr1, acc[6]);
                            acc[7] = acc[7] * corr0;
                            acc[7] = __fmaf_rn(part_f[7], corr1, acc[7]);
                            local_max = part_stats[2];
                            local_sum = part_stats[3];
                            max_v = fmaxf(max_v, local_max);
                            corr0 = 1.0f;
                            corr1 = 0.0f;
                            if (ii_r + 1 < num_ctas_kv_c) {
                                float _exp2_25_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_25_1;
                                float _exp2_26_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_26_1;
                            }
                            old_max_v = max_v;
                            sum_v = __fmaf_rn(sum_v, corr0, local_sum * corr1);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&part_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(part_o[4 + _pair]) << 16);
                                (&part_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(part_o[4 + _pair]) & 0xffff0000u);
                            }
                            acc[0] = acc[0] * corr0;
                            acc[0] = __fmaf_rn(part_f[0], corr1, acc[0]);
                            acc[1] = acc[1] * corr0;
                            acc[1] = __fmaf_rn(part_f[1], corr1, acc[1]);
                            acc[2] = acc[2] * corr0;
                            acc[2] = __fmaf_rn(part_f[2], corr1, acc[2]);
                            acc[3] = acc[3] * corr0;
                            acc[3] = __fmaf_rn(part_f[3], corr1, acc[3]);
                            acc[4] = acc[4] * corr0;
                            acc[4] = __fmaf_rn(part_f[4], corr1, acc[4]);
                            acc[5] = acc[5] * corr0;
                            acc[5] = __fmaf_rn(part_f[5], corr1, acc[5]);
                            acc[6] = acc[6] * corr0;
                            acc[6] = __fmaf_rn(part_f[6], corr1, acc[6]);
                            acc[7] = acc[7] * corr0;
                            acc[7] = __fmaf_rn(part_f[7], corr1, acc[7]);
                            local_max = part_stats[4];
                            local_sum = part_stats[5];
                            max_v = fmaxf(max_v, local_max);
                            corr0 = 1.0f;
                            corr1 = 0.0f;
                            if (ii_r + 2 < num_ctas_kv_c) {
                                float _exp2_27_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_27_1;
                                float _exp2_28_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_28_1;
                            }
                            old_max_v = max_v;
                            sum_v = __fmaf_rn(sum_v, corr0, local_sum * corr1);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&part_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(part_o[8 + _pair]) << 16);
                                (&part_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(part_o[8 + _pair]) & 0xffff0000u);
                            }
                            acc[0] = acc[0] * corr0;
                            acc[0] = __fmaf_rn(part_f[0], corr1, acc[0]);
                            acc[1] = acc[1] * corr0;
                            acc[1] = __fmaf_rn(part_f[1], corr1, acc[1]);
                            acc[2] = acc[2] * corr0;
                            acc[2] = __fmaf_rn(part_f[2], corr1, acc[2]);
                            acc[3] = acc[3] * corr0;
                            acc[3] = __fmaf_rn(part_f[3], corr1, acc[3]);
                            acc[4] = acc[4] * corr0;
                            acc[4] = __fmaf_rn(part_f[4], corr1, acc[4]);
                            acc[5] = acc[5] * corr0;
                            acc[5] = __fmaf_rn(part_f[5], corr1, acc[5]);
                            acc[6] = acc[6] * corr0;
                            acc[6] = __fmaf_rn(part_f[6], corr1, acc[6]);
                            acc[7] = acc[7] * corr0;
                            acc[7] = __fmaf_rn(part_f[7], corr1, acc[7]);
                            local_max = part_stats[6];
                            local_sum = part_stats[7];
                            max_v = fmaxf(max_v, local_max);
                            corr0 = 1.0f;
                            corr1 = 0.0f;
                            if (ii_r + 3 < num_ctas_kv_c) {
                                float _exp2_29_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_29_1;
                                float _exp2_30_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_30_1;
                            }
                            old_max_v = max_v;
                            sum_v = __fmaf_rn(sum_v, corr0, local_sum * corr1);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&part_f[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(part_o[12 + _pair]) << 16);
                                (&part_f[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(part_o[12 + _pair]) & 0xffff0000u);
                            }
                            acc[0] = acc[0] * corr0;
                            acc[0] = __fmaf_rn(part_f[0], corr1, acc[0]);
                            acc[1] = acc[1] * corr0;
                            acc[1] = __fmaf_rn(part_f[1], corr1, acc[1]);
                            acc[2] = acc[2] * corr0;
                            acc[2] = __fmaf_rn(part_f[2], corr1, acc[2]);
                            acc[3] = acc[3] * corr0;
                            acc[3] = __fmaf_rn(part_f[3], corr1, acc[3]);
                            acc[4] = acc[4] * corr0;
                            acc[4] = __fmaf_rn(part_f[4], corr1, acc[4]);
                            acc[5] = acc[5] * corr0;
                            acc[5] = __fmaf_rn(part_f[5], corr1, acc[5]);
                            acc[6] = acc[6] * corr0;
                            acc[6] = __fmaf_rn(part_f[6], corr1, acc[6]);
                            acc[7] = acc[7] * corr0;
                            acc[7] = __fmaf_rn(part_f[7], corr1, acc[7]);
                        }
                        float norm_r = approx_rcp(sum_v);
                        acc[0] = acc[0] * norm_r;
                        acc[1] = acc[1] * norm_r;
                        acc[2] = acc[2] * norm_r;
                        acc[3] = acc[3] * norm_r;
                        acc[4] = acc[4] * norm_r;
                        acc[5] = acc[5] * norm_r;
                        acc[6] = acc[6] * norm_r;
                        acc[7] = acc[7] * norm_r;
                        unsigned int out_w[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                            out_w[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (row_r < num_valid_rows_o) {
                            uint32_t _fast_div_q_7 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)valid_row_r), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)valid_row_r);
                            uint32_t _fast_div_r_7 = (uint32_t)((unsigned int)valid_row_r) - _fast_div_q_7 * (uint32_t)(heads_per_kv.divisor);
                            int dst_row_r = (int)_fast_div_r_7 + (int)_fast_div_q_7 * num_q_heads;
                            int dst_words_r = cta_offset_o / 2 + dst_row_r * (HEAD_DIM / 2) + red_hd / 2;
                            {
                                auto _gst_31 = o_words + dst_words_r;
                                __builtin_assume(__isGlobal(_gst_31));
                                reinterpret_cast<int4*>(_gst_31)[0] = reinterpret_cast<int4*>(out_w)[0];
                            }
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_empty_addr + (or_stage) * 8);
                or_stage += 1;
                if (or_stage == 2) { or_stage = 0; }
                mbarrier_arrive(o_empty_addr + (or_stage) * 8);
                or_stage += 1;
                if (or_stage == 2) { or_stage = 0; }
            }
            asm volatile("barrier.sync 13, 128;" ::: "memory");
            if (warp == 8) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            uint32_t _fast_div_q_2 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_2 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_m = (int)_fast_div_q_2;
            int qm_stage = 0;
            int mk_stage = 0;
            int om_stage = 0;
            const int rank_m = cta_rank;
            int cta_q_m = blockIdx.x / 8;
            int batch_m = blockIdx.z;
            int seq_len_kv_m = seq_lens_kv[batch_m];
            int valid_kv_m = ((cta_q_m * tokens_per_cta_q_m + (seq_len_kv_m - SEQ_LEN_Q) + tokens_per_cta_q_m) < (seq_len_kv_m) ? (cta_q_m * tokens_per_cta_q_m + (seq_len_kv_m - SEQ_LEN_Q) + tokens_per_cta_q_m) : (seq_len_kv_m));
            int num_ctas_kv_m = (((valid_kv_m + (STEP_N - 1)) / STEP_N) < (8) ? ((valid_kv_m + (STEP_N - 1)) / STEP_N) : (8));
            int num_steps_m = 0;
            if (cta_q_m * tokens_per_cta_q_m < SEQ_LEN_Q && rank_m < num_ctas_kv_m) {
                num_steps_m = (valid_kv_m + (num_ctas_kv_m * STEP_N - 1)) / (num_ctas_kv_m * STEP_N);
            }
            unsigned int _phase_q_full = 0;
            unsigned int _phase_s_empty_0_0 = 1;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_s_empty_1_0 = 1;
            unsigned int _phase_o_empty = 1;
            if (num_steps_m > 0) {
                mbarrier_wait(q_full_addr + (qm_stage) * 8, _phase_q_full);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                int mma_b_lo = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 256);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                int mma_b_lo_0 = make_warp_uniform((((smem_q_addr + 2048) >> 4) & 0x3FFF) + (qm_stage) * 256);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(s_full_0_addr);
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                int mma_b_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 256);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                int mma_b_lo_2 = make_warp_uniform((((smem_q_addr + 2048) >> 4) & 0x3FFF) + (qm_stage) * 256);
                {
                    uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_2);
                    uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_2);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(s_full_1_addr);
                #pragma unroll 1
                for (int lo_m = 0; lo_m < num_steps_m - 1; lo_m++) {
                    int first_pv_m = ((lo_m == 0) ? 1 : 0);
                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                    _phase_s_empty_0_0 ^= 1;
                    mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                    int mma_b_lo_3 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, ((first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                    int mma_b_lo_4 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, ((first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(o_full_addr + (om_stage) * 8);
                    om_stage += 1;
                    if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                    int mma_b_lo_5 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 256);
                    {
                        uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_5);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134349968, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_6 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                    int mma_b_lo_6 = make_warp_uniform((((smem_q_addr + 2048) >> 4) & 0x3FFF) + (qm_stage) * 256);
                    {
                        uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_6);
                        uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_6);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134349968, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(s_full_0_addr);
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
                    mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                    int mma_b_lo_7 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_7);
                        uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, ((first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                    int mma_b_lo_8 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_8);
                        uint64_t _mma_ss_b_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_8);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, ((first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(o_full_addr + (om_stage) * 8);
                    om_stage += 1;
                    if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_9 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                    int mma_b_lo_9 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 256);
                    {
                        uint64_t _mma_ss_a_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_9);
                        uint64_t _mma_ss_b_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_9);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_10, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134349968, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_10 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                    int mma_b_lo_10 = make_warp_uniform((((smem_q_addr + 2048) >> 4) & 0x3FFF) + (qm_stage) * 256);
                    {
                        uint64_t _mma_ss_a_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_10);
                        uint64_t _mma_ss_b_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_10);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134349968, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(s_full_1_addr);
                }
                int last_first_pv_m = ((num_steps_m == 1) ? 1 : 0);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                int mma_b_lo_11 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_12 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_11);
                    uint64_t _mma_ss_b_desc_12 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_11);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, ((last_first_pv_m) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c0, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134382736, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_12 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                int mma_b_lo_12 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_13 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_12);
                    uint64_t _mma_ss_b_desc_13 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_12);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, ((last_first_pv_m) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0_c1, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134382736, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(o_full_addr + (om_stage) * 8);
                om_stage += 1;
                if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                elect_commit(s_full_0_addr);
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_13 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                int mma_b_lo_13 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_13);
                    uint64_t _mma_ss_b_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_13);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, ((last_first_pv_m) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134382736, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_14 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                int mma_b_lo_14 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_14);
                    uint64_t _mma_ss_b_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_14);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, ((last_first_pv_m) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1_c1, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134382736, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(o_full_addr + (om_stage) * 8);
                om_stage += 1;
                if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                elect_commit(s_full_1_addr);
                elect_commit(q_empty_addr + (qm_stage) * 8);
                qm_stage += 1;
                if (qm_stage == 2) { qm_stage = 0; _phase_q_full ^= 1; }
            }
        }
    }
    // ---- Role: load_pgoff ----
    if (warp == 13) {
        { // load_pgoff_main
            const int lane_p = lane;
            uint32_t _fast_div_q_0 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_0 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_p = (int)_fast_div_q_0;
            int pp_stage = 0;
            const int rank_p = cta_rank;
            int cta_q_p = blockIdx.x / 8;
            int batch_p = blockIdx.z;
            int seq_len_kv_p = seq_lens_kv[batch_p];
            int valid_kv_p = ((cta_q_p * tokens_per_cta_q_p + (seq_len_kv_p - SEQ_LEN_Q) + tokens_per_cta_q_p) < (seq_len_kv_p) ? (cta_q_p * tokens_per_cta_q_p + (seq_len_kv_p - SEQ_LEN_Q) + tokens_per_cta_q_p) : (seq_len_kv_p));
            int num_ctas_kv_p = (((valid_kv_p + (STEP_N - 1)) / STEP_N) < (8) ? ((valid_kv_p + (STEP_N - 1)) / STEP_N) : (8));
            int num_steps_p = 0;
            if (cta_q_p * tokens_per_cta_q_p < SEQ_LEN_Q && rank_p < num_ctas_kv_p) {
                num_steps_p = (valid_kv_p + (num_ctas_kv_p * STEP_N - 1)) / (num_ctas_kv_p * STEP_N);
            }
            int num_loop_steps_p = num_steps_p * 2;
            unsigned int _phase_page_empty = 1;
            if (num_loop_steps_p > 0) {
                int pt_base_p = batch_p * max_pages_per_seq;
                int page_idx_ub_p = (seq_len_kv_p + (PAGE_SIZE - 1)) / PAGE_SIZE - 1;
                #pragma unroll 1
                for (int lo_p = 0; lo_p < num_loop_steps_p; lo_p += 4) {
                    int page_idx_p = (((rank_p * num_loop_steps_p + lo_p) * 8 + lane_p) < (page_idx_ub_p) ? ((rank_p * num_loop_steps_p + lo_p) * 8 + lane_p) : (page_idx_ub_p));
                    mbarrier_wait(page_empty_addr + (pp_stage) * 8, _phase_page_empty);
                    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4;"
                        :: "r"(smem_page_ids_addr + (unsigned int)(pp_stage * 128 + lane_p * 4)), "l"(page_table + (pt_base_p + page_idx_p)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(page_full_addr + (pp_stage) * 8) : "memory");
                    mbarrier_arrive(page_full_addr + (pp_stage) * 8);
                    pp_stage += 1;
                    if (pp_stage == 6) { pp_stage = 0; _phase_page_empty ^= 1; }
                    mbarrier_wait(page_empty_addr + (pp_stage) * 8, _phase_page_empty);
                    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4;"
                        :: "r"(smem_page_ids_addr + (unsigned int)(pp_stage * 128 + lane_p * 4)), "l"(page_table + (pt_base_p + page_idx_p)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(page_full_addr + (pp_stage) * 8) : "memory");
                    mbarrier_arrive(page_full_addr + (pp_stage) * 8);
                    pp_stage += 1;
                    if (pp_stage == 6) { pp_stage = 0; _phase_page_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            const int heads_per_kv_l = num_q_heads / num_kv_heads;
            uint32_t _fast_div_q_1 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_1 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_l = (int)_fast_div_q_1;
            int q_stage = 0;
            int kv_stage = 0;
            int pc_stage = 0;
            int pr_stage = 0;
            int k_sid = 0;
            int v_sid = 0;
            const int rank_l = cta_rank;
            int cta_q_l = blockIdx.x / 8;
            int head_l = blockIdx.y;
            int batch_l = blockIdx.z;
            int seq_len_kv_l = seq_lens_kv[batch_l];
            int valid_kv_l = ((cta_q_l * tokens_per_cta_q_l + (seq_len_kv_l - SEQ_LEN_Q) + tokens_per_cta_q_l) < (seq_len_kv_l) ? (cta_q_l * tokens_per_cta_q_l + (seq_len_kv_l - SEQ_LEN_Q) + tokens_per_cta_q_l) : (seq_len_kv_l));
            int num_ctas_kv_l = (((valid_kv_l + (STEP_N - 1)) / STEP_N) < (8) ? ((valid_kv_l + (STEP_N - 1)) / STEP_N) : (8));
            int num_steps_l = 0;
            if (cta_q_l * tokens_per_cta_q_l < SEQ_LEN_Q && rank_l < num_ctas_kv_l) {
                num_steps_l = (valid_kv_l + (num_ctas_kv_l * STEP_N - 1)) / (num_ctas_kv_l * STEP_N);
            }
            int num_loop_steps_l = num_steps_l * 2;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
            if (num_loop_steps_l > 0) {
                mbarrier_wait(q_empty_addr + (q_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 4096);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                if (elect_sync()) {
                    int q_row_l = batch_l * num_q_heads + head_l * heads_per_kv_l;
                    #pragma unroll
                    for (int q_box = 0; q_box < 4; q_box++) {
                        tma_3d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 4096 + q_box * 1024), (&Q), 0, q_row_l, q_box, q_full_addr + (q_stage) * 8);
                    }
                }
                q_stage += 1;
                if (q_stage == 2) { q_stage = 0; _phase_q_empty ^= 1; }
                mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                k_sid = pc_stage;
                pc_stage += 1;
                if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                int ids_lo[4];
                int ids_hi[4];
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_a = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_a = 0; pg_a < 4; pg_a++) {
                        int pool_page_a_lo = ids_lo[pg_a] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_a = 0; slab_a < 2; slab_a++) {
                            tma_4d_gmem2smem(kv_dst_a + pg_a * 2048 + slab_a * 16384, (&K), 0, 0, slab_a, pool_page_a_lo, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + 16)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_b = 0; pg_b < 4; pg_b++) {
                        int pool_page_a_hi = ids_hi[pg_b] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_b = 0; slab_b < 2; slab_b++) {
                            tma_4d_gmem2smem(kv_dst_a + (4 + pg_b) * 2048 + slab_b * 16384, (&K), 0, 0, slab_b, pool_page_a_hi, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_b = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_c = 0; pg_c < 4; pg_c++) {
                        int pool_page_b_lo = ids_lo[pg_c] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_c = 0; slab_c < 2; slab_c++) {
                            tma_4d_gmem2smem(kv_dst_b + pg_c * 2048 + slab_c * 16384, (&K), 0, 0, 2 + slab_c, pool_page_b_lo, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    #pragma unroll
                    for (int pg_d = 0; pg_d < 4; pg_d++) {
                        int pool_page_b_hi = ids_hi[pg_d] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_d = 0; slab_d < 2; slab_d++) {
                            tma_4d_gmem2smem(kv_dst_b + (4 + pg_d) * 2048 + slab_d * 16384, (&K), 0, 0, 2 + slab_d, pool_page_b_hi, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                #pragma unroll 1
                for (int lo_l = 0; lo_l < num_loop_steps_l - 1; lo_l++) {
                    int is_last_iter_l = ((lo_l + 1 >= num_loop_steps_l - 1) ? 1 : 0);
                    if ((lo_l + 1) % 4 == 0) {
                        mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                        k_sid = pc_stage;
                        pc_stage += 1;
                        if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                    }
                    int ids_lo_0[4];
                    int ids_hi_0[4];
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int kv_dst_a_0 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + (lo_l + 1) % 4 * 32)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_a_1 = 0; pg_a_1 < 4; pg_a_1++) {
                            int pool_page_a_lo_1 = ids_lo_0[pg_a_1] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_a_1 = 0; slab_a_1 < 2; slab_a_1++) {
                                tma_4d_gmem2smem(kv_dst_a_0 + pg_a_1 * 2048 + slab_a_1 * 16384, (&K), 0, 0, slab_a_1, pool_page_a_lo_1, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + (lo_l + 1) % 4 * 32 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_b_1 = 0; pg_b_1 < 4; pg_b_1++) {
                            int pool_page_a_hi_1 = ids_hi_0[pg_b_1] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_b_1 = 0; slab_b_1 < 2; slab_b_1++) {
                                tma_4d_gmem2smem(kv_dst_a_0 + (4 + pg_b_1) * 2048 + slab_b_1 * 16384, (&K), 0, 0, slab_b_1, pool_page_a_hi_1, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int kv_dst_b_0 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_c_1 = 0; pg_c_1 < 4; pg_c_1++) {
                            int pool_page_b_lo_1 = ids_lo_0[pg_c_1] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_c_1 = 0; slab_c_1 < 2; slab_c_1++) {
                                tma_4d_gmem2smem(kv_dst_b_0 + pg_c_1 * 2048 + slab_c_1 * 16384, (&K), 0, 0, 2 + slab_c_1, pool_page_b_lo_1, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        #pragma unroll
                        for (int pg_d_1 = 0; pg_d_1 < 4; pg_d_1++) {
                            int pool_page_b_hi_1 = ids_hi_0[pg_d_1] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_d_1 = 0; slab_d_1 < 2; slab_d_1++) {
                                tma_4d_gmem2smem(kv_dst_b_0 + (4 + pg_d_1) * 2048 + slab_d_1 * 16384, (&K), 0, 0, 2 + slab_d_1, pool_page_b_hi_1, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    if (is_last_iter_l == 0 && (lo_l + 1) % 4 == 3) {
                        mbarrier_arrive(page_empty_addr + (pr_stage) * 8);
                        pr_stage += 1;
                        if (pr_stage == 6) { pr_stage = 0; }
                    }
                    if (lo_l % 4 == 0) {
                        mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                        v_sid = pc_stage;
                        pc_stage += 1;
                        if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                    }
                    int ids_lo_1[4];
                    int ids_hi_1[4];
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int kv_dst_a_1 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + lo_l % 4 * 32)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_a_2 = 0; pg_a_2 < 4; pg_a_2++) {
                            int pool_page_a_lo_2 = ids_lo_1[pg_a_2] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_a_2 = 0; slab_a_2 < 2; slab_a_2++) {
                                tma_4d_gmem2smem(kv_dst_a_1 + pg_a_2 * 2048 + slab_a_2 * 16384, (&V), 0, 0, slab_a_2, pool_page_a_lo_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + lo_l % 4 * 32 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_b_2 = 0; pg_b_2 < 4; pg_b_2++) {
                            int pool_page_a_hi_2 = ids_hi_1[pg_b_2] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_b_2 = 0; slab_b_2 < 2; slab_b_2++) {
                                tma_4d_gmem2smem(kv_dst_a_1 + (4 + pg_b_2) * 2048 + slab_b_2 * 16384, (&V), 0, 0, slab_b_2, pool_page_a_hi_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int kv_dst_b_1 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_c_2 = 0; pg_c_2 < 4; pg_c_2++) {
                            int pool_page_b_lo_2 = ids_lo_1[pg_c_2] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_c_2 = 0; slab_c_2 < 2; slab_c_2++) {
                                tma_4d_gmem2smem(kv_dst_b_1 + pg_c_2 * 2048 + slab_c_2 * 16384, (&V), 0, 0, 2 + slab_c_2, pool_page_b_lo_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        #pragma unroll
                        for (int pg_d_2 = 0; pg_d_2 < 4; pg_d_2++) {
                            int pool_page_b_hi_2 = ids_hi_1[pg_d_2] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_d_2 = 0; slab_d_2 < 2; slab_d_2++) {
                                tma_4d_gmem2smem(kv_dst_b_1 + (4 + pg_d_2) * 2048 + slab_d_2 * 16384, (&V), 0, 0, 2 + slab_d_2, pool_page_b_hi_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    if (lo_l % 4 == 3) {
                        mbarrier_arrive(page_empty_addr + (pr_stage) * 8);
                        pr_stage += 1;
                        if (pr_stage == 6) { pr_stage = 0; }
                    }
                }
                int last_l = num_loop_steps_l - 1;
                mbarrier_arrive(page_empty_addr + (pr_stage) * 8);
                pr_stage += 1;
                if (pr_stage == 6) { pr_stage = 0; }
                if (last_l % 4 == 0) {
                    mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                    v_sid = pc_stage;
                    pc_stage += 1;
                    if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                }
                int ids_lo_2[4];
                int ids_hi_2[4];
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_a_2 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + last_l % 4 * 32)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_a_3 = 0; pg_a_3 < 4; pg_a_3++) {
                        int pool_page_a_lo_3 = ids_lo_2[pg_a_3] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_a_3 = 0; slab_a_3 < 2; slab_a_3++) {
                            tma_4d_gmem2smem(kv_dst_a_2 + pg_a_3 * 2048 + slab_a_3 * 16384, (&V), 0, 0, slab_a_3, pool_page_a_lo_3, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + last_l % 4 * 32 + 16)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_b_3 = 0; pg_b_3 < 4; pg_b_3++) {
                        int pool_page_a_hi_3 = ids_hi_2[pg_b_3] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_b_3 = 0; slab_b_3 < 2; slab_b_3++) {
                            tma_4d_gmem2smem(kv_dst_a_2 + (4 + pg_b_3) * 2048 + slab_b_3 * 16384, (&V), 0, 0, slab_b_3, pool_page_a_hi_3, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_b_2 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_c_3 = 0; pg_c_3 < 4; pg_c_3++) {
                        int pool_page_b_lo_3 = ids_lo_2[pg_c_3] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_c_3 = 0; slab_c_3 < 2; slab_c_3++) {
                            tma_4d_gmem2smem(kv_dst_b_2 + pg_c_3 * 2048 + slab_c_3 * 16384, (&V), 0, 0, 2 + slab_c_3, pool_page_b_lo_3, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    #pragma unroll
                    for (int pg_d_3 = 0; pg_d_3 < 4; pg_d_3++) {
                        int pool_page_b_hi_3 = ids_hi_2[pg_d_3] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_d_3 = 0; slab_d_3 < 2; slab_d_3++) {
                            tma_4d_gmem2smem(kv_dst_b_2 + (4 + pg_d_3) * 2048 + slab_d_3 * 16384, (&V), 0, 0, 2 + slab_d_3, pool_page_b_hi_3, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_arrive(page_empty_addr + (pr_stage) * 8);
                pr_stage += 1;
                if (pr_stage == 6) { pr_stage = 0; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
