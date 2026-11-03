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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_fmha_aligned_mtp_q32_a1e5ddfa63d266952503(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, CakeFastDivmod heads_per_kv, int batch_size, int seq_len_q, float skip_corr_threshold)
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

    const int mbar_base = smem + 200448;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 64)
    #define s_full_0_addr (mbar_base + 96)
    #define s_full_1_addr (mbar_base + 104)
    #define p_empty_0_addr (mbar_base + 112)
    #define p_empty_1_addr (mbar_base + 120)
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
    #define p_full_0_addr (mbar_base + 304)
    #define p_full_1_addr (mbar_base + 312)
    #define o_empty_addr (mbar_base + 320)
    #define reserved_arv128_addr (mbar_base + 336)
    #define cga_full_addr (mbar_base + 512)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_q_addr = smem + 0;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int smem_kv_addr = smem + 16384;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int smem_v_addr = smem + 16384;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 147456);
    const int smem_p0_addr = smem + 147456;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 155648);
    const int smem_p1_addr = smem + 155648;
    int* smem_page_ids = reinterpret_cast<int*>(smem_raw + 163840);
    const int smem_page_ids_addr = smem + 163840;
    unsigned int* smem_cga = reinterpret_cast<unsigned int*>(smem_raw + 164608);
    const int smem_cga_addr = smem + 164608;
    unsigned int* smem_red0 = reinterpret_cast<unsigned int*>(smem_raw + 198400);
    const int smem_red0_addr = smem + 198400;
    unsigned int* smem_red1 = reinterpret_cast<unsigned int*>(smem_raw + 198912);
    const int smem_red1_addr = smem + 198912;
    float* smem_corr_red = reinterpret_cast<float*>(smem_raw + 199424);
    const int smem_corr_red_addr = smem + 199424;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (24 pipeline groups, 0 ordered-sequence groups, 65 barriers)
    // Mbarriers at smem_raw[200448..200968)

    if (warp == 0) {
        // --- pipeline 'q_pipe' ---
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 4 barriers, init_count=1
        // kv_empty: 4 barriers, init_count=1
        // s_full_0: 1 barriers, init_count=1
        // s_full_1: 1 barriers, init_count=1
        // p_empty_0: 1 barriers, init_count=1
        // p_empty_1: 1 barriers, init_count=1
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
        // p_full_0: 1 barriers, init_count=128
        // p_full_1: 1 barriers, init_count=128
        // --- pipeline 'o_pipe' ---
        // o_empty: 2 barriers, init_count=128
        // reserved_arv128: 22 barriers, init_count=128
        // cga_full: 1 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        mbarrier_init(smem + 200448 + lane * 8, _mbarrier_init_count_0_0);
        mbarrier_init(smem + 200704 + lane * 8, 128);
        if (lane < 1) {
            mbarrier_init(smem + 200960 + lane * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 200968);
    if (warp == 0) {
        int _tmem_hold = smem + 200968;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
    const int tmem_tmem_s1 = taddr + 32;
    const int tmem_tmem_stats0 = taddr + 64;
    const int tmem_tmem_stats1 = taddr + 96;
    const int tmem_tmem_o0 = taddr + 128;
    const int tmem_tmem_o1 = taddr + 160;
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
            const int red_idx = wg_tid % 4 * 8;
            const int row_idx = warp_in_wg * 32 + lane_s / 4;
            int tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 32 : 0)) + (unsigned int)(warp_in_wg * 32 << 16);
            int tmem_stats_base = taddr + (unsigned int)(((is_wg1 != 0) ? 96 : 64)) + (unsigned int)(warp_in_wg * 32 << 16);
            unsigned int* my_red_ptr = ((is_wg1 != 0) ? (smem_red1) : (smem_red0));
            unsigned int my_red_base = ((is_wg1 != 0) ? smem_red1_addr : smem_red0_addr);
            __nv_bfloat16* my_p_base = ((is_wg1 != 0) ? (smem_p1) : (smem_p0));
            const int named_bar_id = 1 + is_wg1;
            uint32_t _fast_div_q_3 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)(32), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)(32);
            uint32_t _fast_div_r_3 = (uint32_t)(32) - _fast_div_q_3 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_s = (int)_fast_div_q_3;
            const int rank_s = cta_rank;
            int cta_q_s = blockIdx.x / 2;
            int batch_s = blockIdx.z;
            const int slice_idx = warp_in_wg / 2;
            const int warp_idx_in_slice = warp_in_wg % 2;
            const int mtx_idx_s = lane_s / 8;
            const int thr_row_idx_s = lane_s % 8;
            const int mtx_col_s = warp_idx_in_slice * 4 + mtx_idx_s % 2;
            const int stsm_base_s = slice_idx * 32 * 128 + (mtx_idx_s / 2 * 8 + thr_row_idx_s) * 128;
            float s_vals[32];
            float old_max[8];
            float new_max[8];
            float row_sum[8];
            int seq_len_kv_s = seq_lens_kv[batch_s];
            int valid_kv_s = ((cta_q_s * tokens_per_cta_q_s + (seq_len_kv_s - seq_len_q) + tokens_per_cta_q_s) < (seq_len_kv_s) ? (cta_q_s * tokens_per_cta_q_s + (seq_len_kv_s - seq_len_q) + tokens_per_cta_q_s) : (seq_len_kv_s));
            int num_ctas_kv_s = (((valid_kv_s + 255) / 256) < (2) ? ((valid_kv_s + 255) / 256) : (2));
            int num_steps_s = 0;
            if (cta_q_s * tokens_per_cta_q_s < seq_len_q && rank_s < num_ctas_kv_s) {
                num_steps_s = (valid_kv_s + (num_ctas_kv_s * 256 - 1)) / (num_ctas_kv_s * 256);
            }
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_stats_empty_1_0 = 1;
            unsigned int _phase_stats_empty_0_0 = 1;
            unsigned int _phase_p_empty_1_0 = 1;
            unsigned int _phase_p_empty_0_0 = 1;
            unsigned int _phase_order_p1_0 = 0;
            unsigned int _phase_order_p0_0 = 1;
            if (num_steps_s > 0) {
                row_sum[0] = 0.0f;
                new_max[0] = -3.4028235e+38f;
                row_sum[1] = 0.0f;
                new_max[1] = -3.4028235e+38f;
                row_sum[2] = 0.0f;
                new_max[2] = -3.4028235e+38f;
                row_sum[3] = 0.0f;
                new_max[3] = -3.4028235e+38f;
                row_sum[4] = 0.0f;
                new_max[4] = -3.4028235e+38f;
                row_sum[5] = 0.0f;
                new_max[5] = -3.4028235e+38f;
                row_sum[6] = 0.0f;
                new_max[6] = -3.4028235e+38f;
                row_sum[7] = 0.0f;
                new_max[7] = -3.4028235e+38f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 32) {
                    my_red_ptr[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(named_bar_id) : "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
                float adjusted_skip_thr = skip_corr_threshold / softmax_scale_log2;
                int tile_offset_q = cta_q_s * tokens_per_cta_q_s + (seq_len_kv_s - seq_len_q);
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
                    old_max[2] = new_max[2];
                    old_max[3] = new_max[3];
                    old_max[4] = new_max[4];
                    old_max[5] = new_max[5];
                    old_max[6] = new_max[6];
                    old_max[7] = new_max[7];
                    int tile_offset_k = ((num_steps_s * rank_s + lo_s) * 2 + is_wg1) * 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[3])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[7])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[8])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[9])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[10])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[11])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[12])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[13])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[14])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[15]))
                        : "r"(tmem_s_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s_vals[16])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[17])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[18])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[19])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[20])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[21])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[22])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[23])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[24])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[25])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[26])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[27])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[28])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[29])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[30])), "=r"(*reinterpret_cast<uint32_t*>(&s_vals[31]))
                        : "r"(tmem_s_base + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (tile_offset_k + 128 > tile_offset_q) {
                        int idx_k0 = tile_offset_k + row_idx;
                        uint32_t _fq = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)col_idx), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)col_idx);
                        uint32_t _fr = (uint32_t)((unsigned int)col_idx) - _fq * (uint32_t)(heads_per_kv.divisor);
                        int idx_q = ((tile_offset_q + (int)_fq) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q) {
                            s_vals[0] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q) {
                            s_vals[2] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q) {
                            s_vals[16] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q) {
                            s_vals[18] = -3.4028235e+38f;
                        }
                        uint32_t _fq_0 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 1)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 1));
                        uint32_t _fr_0 = (uint32_t)((unsigned int)(col_idx + 1)) - _fq_0 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_0 = ((tile_offset_q + (int)_fq_0) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_0) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_0) {
                            s_vals[1] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_0) {
                            s_vals[3] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_0) {
                            s_vals[17] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_0) {
                            s_vals[19] = -3.4028235e+38f;
                        }
                        uint32_t _fq_1 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 8)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 8));
                        uint32_t _fr_1 = (uint32_t)((unsigned int)(col_idx + 8)) - _fq_1 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_1 = ((tile_offset_q + (int)_fq_1) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_1) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_1) {
                            s_vals[4] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_1) {
                            s_vals[6] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_1) {
                            s_vals[20] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_1) {
                            s_vals[22] = -3.4028235e+38f;
                        }
                        uint32_t _fq_2 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 9)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 9));
                        uint32_t _fr_2 = (uint32_t)((unsigned int)(col_idx + 9)) - _fq_2 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_2 = ((tile_offset_q + (int)_fq_2) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_2) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_2) {
                            s_vals[5] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_2) {
                            s_vals[7] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_2) {
                            s_vals[21] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_2) {
                            s_vals[23] = -3.4028235e+38f;
                        }
                        uint32_t _fq_3 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 16)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 16));
                        uint32_t _fr_3 = (uint32_t)((unsigned int)(col_idx + 16)) - _fq_3 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_3 = ((tile_offset_q + (int)_fq_3) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_3) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_3) {
                            s_vals[8] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_3) {
                            s_vals[10] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_3) {
                            s_vals[24] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_3) {
                            s_vals[26] = -3.4028235e+38f;
                        }
                        uint32_t _fq_4 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 17)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 17));
                        uint32_t _fr_4 = (uint32_t)((unsigned int)(col_idx + 17)) - _fq_4 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_4 = ((tile_offset_q + (int)_fq_4) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_4) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_4) {
                            s_vals[9] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_4) {
                            s_vals[11] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_4) {
                            s_vals[25] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_4) {
                            s_vals[27] = -3.4028235e+38f;
                        }
                        uint32_t _fq_5 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 24)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 24));
                        uint32_t _fr_5 = (uint32_t)((unsigned int)(col_idx + 24)) - _fq_5 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_5 = ((tile_offset_q + (int)_fq_5) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_5) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_5) {
                            s_vals[12] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_5) {
                            s_vals[14] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_5) {
                            s_vals[28] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_5) {
                            s_vals[30] = -3.4028235e+38f;
                        }
                        uint32_t _fq_6 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(col_idx + 25)), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)(col_idx + 25));
                        uint32_t _fr_6 = (uint32_t)((unsigned int)(col_idx + 25)) - _fq_6 * (uint32_t)(heads_per_kv.divisor);
                        int idx_q_6 = ((tile_offset_q + (int)_fq_6) < (seq_len_kv_s - 1) ? (tile_offset_q + (int)_fq_6) : (seq_len_kv_s - 1));
                        if (idx_k0 > idx_q_6) {
                            s_vals[13] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 8 > idx_q_6) {
                            s_vals[15] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 16 > idx_q_6) {
                            s_vals[29] = -3.4028235e+38f;
                        }
                        if (idx_k0 + 24 > idx_q_6) {
                            s_vals[31] = -3.4028235e+38f;
                        }
                    }
                    new_max[0] = fmaxf(new_max[0], s_vals[0]);
                    new_max[0] = fmaxf(new_max[0], s_vals[2]);
                    new_max[0] = fmaxf(new_max[0], s_vals[16]);
                    new_max[0] = fmaxf(new_max[0], s_vals[18]);
                    new_max[1] = fmaxf(new_max[1], s_vals[1]);
                    new_max[1] = fmaxf(new_max[1], s_vals[3]);
                    new_max[1] = fmaxf(new_max[1], s_vals[17]);
                    new_max[1] = fmaxf(new_max[1], s_vals[19]);
                    new_max[2] = fmaxf(new_max[2], s_vals[4]);
                    new_max[2] = fmaxf(new_max[2], s_vals[6]);
                    new_max[2] = fmaxf(new_max[2], s_vals[20]);
                    new_max[2] = fmaxf(new_max[2], s_vals[22]);
                    new_max[3] = fmaxf(new_max[3], s_vals[5]);
                    new_max[3] = fmaxf(new_max[3], s_vals[7]);
                    new_max[3] = fmaxf(new_max[3], s_vals[21]);
                    new_max[3] = fmaxf(new_max[3], s_vals[23]);
                    new_max[4] = fmaxf(new_max[4], s_vals[8]);
                    new_max[4] = fmaxf(new_max[4], s_vals[10]);
                    new_max[4] = fmaxf(new_max[4], s_vals[24]);
                    new_max[4] = fmaxf(new_max[4], s_vals[26]);
                    new_max[5] = fmaxf(new_max[5], s_vals[9]);
                    new_max[5] = fmaxf(new_max[5], s_vals[11]);
                    new_max[5] = fmaxf(new_max[5], s_vals[25]);
                    new_max[5] = fmaxf(new_max[5], s_vals[27]);
                    new_max[6] = fmaxf(new_max[6], s_vals[12]);
                    new_max[6] = fmaxf(new_max[6], s_vals[14]);
                    new_max[6] = fmaxf(new_max[6], s_vals[28]);
                    new_max[6] = fmaxf(new_max[6], s_vals[30]);
                    new_max[7] = fmaxf(new_max[7], s_vals[13]);
                    new_max[7] = fmaxf(new_max[7], s_vals[15]);
                    new_max[7] = fmaxf(new_max[7], s_vals[29]);
                    new_max[7] = fmaxf(new_max[7], s_vals[31]);
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, new_max[0], 16);
                    new_max[0] = fmaxf(_shfl_xor_sync_1, new_max[0]);
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, new_max[0], 8);
                    new_max[0] = fmaxf(_shfl_xor_sync_2, new_max[0]);
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, new_max[0], 4);
                    new_max[0] = fmaxf(_shfl_xor_sync_3, new_max[0]);
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, new_max[1], 16);
                    new_max[1] = fmaxf(_shfl_xor_sync_4, new_max[1]);
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, new_max[1], 8);
                    new_max[1] = fmaxf(_shfl_xor_sync_5, new_max[1]);
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, new_max[1], 4);
                    new_max[1] = fmaxf(_shfl_xor_sync_6, new_max[1]);
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, new_max[2], 16);
                    new_max[2] = fmaxf(_shfl_xor_sync_7, new_max[2]);
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, new_max[2], 8);
                    new_max[2] = fmaxf(_shfl_xor_sync_8, new_max[2]);
                    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, new_max[2], 4);
                    new_max[2] = fmaxf(_shfl_xor_sync_9, new_max[2]);
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, new_max[3], 16);
                    new_max[3] = fmaxf(_shfl_xor_sync_10, new_max[3]);
                    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, new_max[3], 8);
                    new_max[3] = fmaxf(_shfl_xor_sync_11, new_max[3]);
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, new_max[3], 4);
                    new_max[3] = fmaxf(_shfl_xor_sync_12, new_max[3]);
                    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, new_max[4], 16);
                    new_max[4] = fmaxf(_shfl_xor_sync_13, new_max[4]);
                    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, new_max[4], 8);
                    new_max[4] = fmaxf(_shfl_xor_sync_14, new_max[4]);
                    float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, new_max[4], 4);
                    new_max[4] = fmaxf(_shfl_xor_sync_15, new_max[4]);
                    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, new_max[5], 16);
                    new_max[5] = fmaxf(_shfl_xor_sync_16, new_max[5]);
                    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, new_max[5], 8);
                    new_max[5] = fmaxf(_shfl_xor_sync_17, new_max[5]);
                    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, new_max[5], 4);
                    new_max[5] = fmaxf(_shfl_xor_sync_18, new_max[5]);
                    float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, new_max[6], 16);
                    new_max[6] = fmaxf(_shfl_xor_sync_19, new_max[6]);
                    float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, new_max[6], 8);
                    new_max[6] = fmaxf(_shfl_xor_sync_20, new_max[6]);
                    float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, new_max[6], 4);
                    new_max[6] = fmaxf(_shfl_xor_sync_21, new_max[6]);
                    float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, new_max[7], 16);
                    new_max[7] = fmaxf(_shfl_xor_sync_22, new_max[7]);
                    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, new_max[7], 8);
                    new_max[7] = fmaxf(_shfl_xor_sync_23, new_max[7]);
                    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, new_max[7], 4);
                    new_max[7] = fmaxf(_shfl_xor_sync_24, new_max[7]);
                    if (lane_s < 4) {
                        uint32_t _amf_u_26 = __float_as_uint(new_max[0]);
                        uint32_t _amf_mask_26 = -int32_t(_amf_u_26 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_25 = _amf_u_26 ^ _amf_mask_26;
                        atomicMax(&my_red_ptr[red_idx], _atomic_max_float_encode_25);
                        uint32_t _amf_u_28 = __float_as_uint(new_max[1]);
                        uint32_t _amf_mask_28 = -int32_t(_amf_u_28 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_27 = _amf_u_28 ^ _amf_mask_28;
                        atomicMax(&my_red_ptr[red_idx + 1], _atomic_max_float_encode_27);
                        uint32_t _amf_u_30 = __float_as_uint(new_max[2]);
                        uint32_t _amf_mask_30 = -int32_t(_amf_u_30 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_29 = _amf_u_30 ^ _amf_mask_30;
                        atomicMax(&my_red_ptr[red_idx + 2], _atomic_max_float_encode_29);
                        uint32_t _amf_u_32 = __float_as_uint(new_max[3]);
                        uint32_t _amf_mask_32 = -int32_t(_amf_u_32 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_31 = _amf_u_32 ^ _amf_mask_32;
                        atomicMax(&my_red_ptr[red_idx + 3], _atomic_max_float_encode_31);
                        uint32_t _amf_u_34 = __float_as_uint(new_max[4]);
                        uint32_t _amf_mask_34 = -int32_t(_amf_u_34 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_33 = _amf_u_34 ^ _amf_mask_34;
                        atomicMax(&my_red_ptr[red_idx + 4], _atomic_max_float_encode_33);
                        uint32_t _amf_u_36 = __float_as_uint(new_max[5]);
                        uint32_t _amf_mask_36 = -int32_t(_amf_u_36 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_35 = _amf_u_36 ^ _amf_mask_36;
                        atomicMax(&my_red_ptr[red_idx + 5], _atomic_max_float_encode_35);
                        uint32_t _amf_u_38 = __float_as_uint(new_max[6]);
                        uint32_t _amf_mask_38 = -int32_t(_amf_u_38 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_37 = _amf_u_38 ^ _amf_mask_38;
                        atomicMax(&my_red_ptr[red_idx + 6], _atomic_max_float_encode_37);
                        uint32_t _amf_u_40 = __float_as_uint(new_max[7]);
                        uint32_t _amf_mask_40 = -int32_t(_amf_u_40 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_39 = _amf_u_40 ^ _amf_mask_40;
                        atomicMax(&my_red_ptr[red_idx + 7], _atomic_max_float_encode_39);
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(named_bar_id) : "memory");
                    unsigned int red_vals[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&red_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&red_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&red_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&red_vals[(0) + 3]))
                        : "r"(my_red_base + (unsigned int)(red_idx * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&red_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&red_vals[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&red_vals[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&red_vals[(4) + 3]))
                        : "r"(my_red_base + (unsigned int)(red_idx * 4 + 16)));
                    uint32_t _amf_u_42 = red_vals[0];
                    uint32_t _amf_mask_42 = ((_amf_u_42 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_41 = __uint_as_float(_amf_u_42 ^ _amf_mask_42);
                    new_max[0] = _atomic_max_float_decode_41;
                    uint32_t _amf_u_44 = red_vals[1];
                    uint32_t _amf_mask_44 = ((_amf_u_44 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_43 = __uint_as_float(_amf_u_44 ^ _amf_mask_44);
                    new_max[1] = _atomic_max_float_decode_43;
                    uint32_t _amf_u_46 = red_vals[2];
                    uint32_t _amf_mask_46 = ((_amf_u_46 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_45 = __uint_as_float(_amf_u_46 ^ _amf_mask_46);
                    new_max[2] = _atomic_max_float_decode_45;
                    uint32_t _amf_u_48 = red_vals[3];
                    uint32_t _amf_mask_48 = ((_amf_u_48 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_47 = __uint_as_float(_amf_u_48 ^ _amf_mask_48);
                    new_max[3] = _atomic_max_float_decode_47;
                    uint32_t _amf_u_50 = red_vals[4];
                    uint32_t _amf_mask_50 = ((_amf_u_50 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_49 = __uint_as_float(_amf_u_50 ^ _amf_mask_50);
                    new_max[4] = _atomic_max_float_decode_49;
                    uint32_t _amf_u_52 = red_vals[5];
                    uint32_t _amf_mask_52 = ((_amf_u_52 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_51 = __uint_as_float(_amf_u_52 ^ _amf_mask_52);
                    new_max[5] = _atomic_max_float_decode_51;
                    uint32_t _amf_u_54 = red_vals[6];
                    uint32_t _amf_mask_54 = ((_amf_u_54 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_53 = __uint_as_float(_amf_u_54 ^ _amf_mask_54);
                    new_max[6] = _atomic_max_float_decode_53;
                    uint32_t _amf_u_56 = red_vals[7];
                    uint32_t _amf_mask_56 = ((_amf_u_56 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_55 = __uint_as_float(_amf_u_56 ^ _amf_mask_56);
                    new_max[7] = _atomic_max_float_decode_55;
                    if (adjusted_skip_thr > 0.0f) {
                        if (adjusted_skip_thr >= new_max[0] - old_max[0]) {
                            new_max[0] = old_max[0];
                        }
                        if (adjusted_skip_thr >= new_max[1] - old_max[1]) {
                            new_max[1] = old_max[1];
                        }
                        if (adjusted_skip_thr >= new_max[2] - old_max[2]) {
                            new_max[2] = old_max[2];
                        }
                        if (adjusted_skip_thr >= new_max[3] - old_max[3]) {
                            new_max[3] = old_max[3];
                        }
                        if (adjusted_skip_thr >= new_max[4] - old_max[4]) {
                            new_max[4] = old_max[4];
                        }
                        if (adjusted_skip_thr >= new_max[5] - old_max[5]) {
                            new_max[5] = old_max[5];
                        }
                        if (adjusted_skip_thr >= new_max[6] - old_max[6]) {
                            new_max[6] = old_max[6];
                        }
                        if (adjusted_skip_thr >= new_max[7] - old_max[7]) {
                            new_max[7] = old_max[7];
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(s_empty_1_addr);
                    } else {
                        mbarrier_arrive(s_empty_0_addr);
                    }
                    float stats_s[16];
                    stats_s[0] = old_max[0];
                    stats_s[8] = new_max[0];
                    stats_s[1] = old_max[1];
                    stats_s[9] = new_max[1];
                    stats_s[2] = old_max[2];
                    stats_s[10] = new_max[2];
                    stats_s[3] = old_max[3];
                    stats_s[11] = new_max[3];
                    stats_s[4] = old_max[4];
                    stats_s[12] = new_max[4];
                    stats_s[5] = old_max[5];
                    stats_s[13] = new_max[5];
                    stats_s[6] = old_max[6];
                    stats_s[14] = new_max[6];
                    stats_s[7] = old_max[7];
                    stats_s[15] = new_max[7];
                    if (is_wg1 != 0) {
                        mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    tmem_st_x16_f32(tmem_stats_base, stats_s);
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
                        mbarrier_wait(p_empty_1_addr, _phase_p_empty_1_0);
                        _phase_p_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(p_empty_0_addr, _phase_p_empty_0_0);
                        _phase_p_empty_0_0 ^= 1;
                    }
                    float neg_scaled_max[8];
                    float safe_max_0 = ((new_max[0] == -3.4028235e+38f) ? 0.0f : new_max[0]);
                    neg_scaled_max[0] = safe_max_0 * (-softmax_scale_log2);
                    float safe_max_1 = ((new_max[1] == -3.4028235e+38f) ? 0.0f : new_max[1]);
                    neg_scaled_max[1] = safe_max_1 * (-softmax_scale_log2);
                    float safe_max_2 = ((new_max[2] == -3.4028235e+38f) ? 0.0f : new_max[2]);
                    neg_scaled_max[2] = safe_max_2 * (-softmax_scale_log2);
                    float safe_max_3 = ((new_max[3] == -3.4028235e+38f) ? 0.0f : new_max[3]);
                    neg_scaled_max[3] = safe_max_3 * (-softmax_scale_log2);
                    float safe_max_4 = ((new_max[4] == -3.4028235e+38f) ? 0.0f : new_max[4]);
                    neg_scaled_max[4] = safe_max_4 * (-softmax_scale_log2);
                    float safe_max_5 = ((new_max[5] == -3.4028235e+38f) ? 0.0f : new_max[5]);
                    neg_scaled_max[5] = safe_max_5 * (-softmax_scale_log2);
                    float safe_max_6 = ((new_max[6] == -3.4028235e+38f) ? 0.0f : new_max[6]);
                    neg_scaled_max[6] = safe_max_6 * (-softmax_scale_log2);
                    float safe_max_7 = ((new_max[7] == -3.4028235e+38f) ? 0.0f : new_max[7]);
                    neg_scaled_max[7] = safe_max_7 * (-softmax_scale_log2);
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
                    s_vals[4] = __fmaf_rn(softmax_scale_log2, s_vals[4], neg_scaled_max[2]);
                    float _exp2_57 = approx_exp2(s_vals[0]);
                    s_vals[0] = _exp2_57;
                    s_vals[5] = __fmaf_rn(softmax_scale_log2, s_vals[5], neg_scaled_max[3]);
                    float _exp2_58 = approx_exp2(s_vals[1]);
                    s_vals[1] = _exp2_58;
                    s_vals[6] = __fmaf_rn(softmax_scale_log2, s_vals[6], neg_scaled_max[2]);
                    float _exp2_59 = approx_exp2(s_vals[2]);
                    s_vals[2] = _exp2_59;
                    s_vals[7] = __fmaf_rn(softmax_scale_log2, s_vals[7], neg_scaled_max[3]);
                    float _exp2_60 = approx_exp2(s_vals[3]);
                    s_vals[3] = _exp2_60;
                    s_vals[8] = __fmaf_rn(softmax_scale_log2, s_vals[8], neg_scaled_max[4]);
                    float _exp2_61 = approx_exp2(s_vals[4]);
                    s_vals[4] = _exp2_61;
                    s_vals[9] = __fmaf_rn(softmax_scale_log2, s_vals[9], neg_scaled_max[5]);
                    float _exp2_62 = approx_exp2(s_vals[5]);
                    s_vals[5] = _exp2_62;
                    s_vals[10] = __fmaf_rn(softmax_scale_log2, s_vals[10], neg_scaled_max[4]);
                    float _exp2_63 = approx_exp2(s_vals[6]);
                    s_vals[6] = _exp2_63;
                    s_vals[11] = __fmaf_rn(softmax_scale_log2, s_vals[11], neg_scaled_max[5]);
                    float _exp2_64 = approx_exp2(s_vals[7]);
                    s_vals[7] = _exp2_64;
                    s_vals[12] = __fmaf_rn(softmax_scale_log2, s_vals[12], neg_scaled_max[6]);
                    float _exp2_65 = approx_exp2(s_vals[8]);
                    s_vals[8] = _exp2_65;
                    s_vals[13] = __fmaf_rn(softmax_scale_log2, s_vals[13], neg_scaled_max[7]);
                    float _exp2_66 = approx_exp2(s_vals[9]);
                    s_vals[9] = _exp2_66;
                    s_vals[14] = __fmaf_rn(softmax_scale_log2, s_vals[14], neg_scaled_max[6]);
                    float _exp2_67 = approx_exp2(s_vals[10]);
                    s_vals[10] = _exp2_67;
                    s_vals[15] = __fmaf_rn(softmax_scale_log2, s_vals[15], neg_scaled_max[7]);
                    float _exp2_68 = approx_exp2(s_vals[11]);
                    s_vals[11] = _exp2_68;
                    s_vals[16] = __fmaf_rn(softmax_scale_log2, s_vals[16], neg_scaled_max[0]);
                    float _exp2_69 = approx_exp2(s_vals[12]);
                    s_vals[12] = _exp2_69;
                    s_vals[17] = __fmaf_rn(softmax_scale_log2, s_vals[17], neg_scaled_max[1]);
                    float _exp2_70 = approx_exp2(s_vals[13]);
                    s_vals[13] = _exp2_70;
                    s_vals[18] = __fmaf_rn(softmax_scale_log2, s_vals[18], neg_scaled_max[0]);
                    float _exp2_71 = approx_exp2(s_vals[14]);
                    s_vals[14] = _exp2_71;
                    s_vals[19] = __fmaf_rn(softmax_scale_log2, s_vals[19], neg_scaled_max[1]);
                    float _exp2_72 = approx_exp2(s_vals[15]);
                    s_vals[15] = _exp2_72;
                    s_vals[20] = __fmaf_rn(softmax_scale_log2, s_vals[20], neg_scaled_max[2]);
                    float _exp2_73 = approx_exp2(s_vals[16]);
                    s_vals[16] = _exp2_73;
                    s_vals[21] = __fmaf_rn(softmax_scale_log2, s_vals[21], neg_scaled_max[3]);
                    float _exp2_74 = approx_exp2(s_vals[17]);
                    s_vals[17] = _exp2_74;
                    s_vals[22] = __fmaf_rn(softmax_scale_log2, s_vals[22], neg_scaled_max[2]);
                    float _exp2_75 = approx_exp2(s_vals[18]);
                    s_vals[18] = _exp2_75;
                    s_vals[23] = __fmaf_rn(softmax_scale_log2, s_vals[23], neg_scaled_max[3]);
                    float _exp2_76 = approx_exp2(s_vals[19]);
                    s_vals[19] = _exp2_76;
                    s_vals[24] = __fmaf_rn(softmax_scale_log2, s_vals[24], neg_scaled_max[4]);
                    float _exp2_77 = approx_exp2(s_vals[20]);
                    s_vals[20] = _exp2_77;
                    s_vals[25] = __fmaf_rn(softmax_scale_log2, s_vals[25], neg_scaled_max[5]);
                    float _exp2_78 = approx_exp2(s_vals[21]);
                    s_vals[21] = _exp2_78;
                    s_vals[26] = __fmaf_rn(softmax_scale_log2, s_vals[26], neg_scaled_max[4]);
                    float _exp2_79 = approx_exp2(s_vals[22]);
                    s_vals[22] = _exp2_79;
                    s_vals[27] = __fmaf_rn(softmax_scale_log2, s_vals[27], neg_scaled_max[5]);
                    float _exp2_80 = approx_exp2(s_vals[23]);
                    s_vals[23] = _exp2_80;
                    s_vals[28] = __fmaf_rn(softmax_scale_log2, s_vals[28], neg_scaled_max[6]);
                    float _exp2_81 = approx_exp2(s_vals[24]);
                    s_vals[24] = _exp2_81;
                    s_vals[29] = __fmaf_rn(softmax_scale_log2, s_vals[29], neg_scaled_max[7]);
                    float _exp2_82 = approx_exp2(s_vals[25]);
                    s_vals[25] = _exp2_82;
                    if (is_wg1 != 0) {
                        mbarrier_arrive(order_p0_addr);
                    } else {
                        mbarrier_arrive(order_p1_addr);
                    }
                    s_vals[30] = __fmaf_rn(softmax_scale_log2, s_vals[30], neg_scaled_max[6]);
                    float _exp2_83 = approx_exp2(s_vals[26]);
                    s_vals[26] = _exp2_83;
                    s_vals[31] = __fmaf_rn(softmax_scale_log2, s_vals[31], neg_scaled_max[7]);
                    float _exp2_84 = approx_exp2(s_vals[27]);
                    s_vals[27] = _exp2_84;
                    float _exp2_85 = approx_exp2(s_vals[28]);
                    s_vals[28] = _exp2_85;
                    float _exp2_86 = approx_exp2(s_vals[29]);
                    s_vals[29] = _exp2_86;
                    float _exp2_87 = approx_exp2(s_vals[30]);
                    s_vals[30] = _exp2_87;
                    float _exp2_88 = approx_exp2(s_vals[31]);
                    s_vals[31] = _exp2_88;
                    unsigned int regs_p[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_vals[_lp*2 + 0], s_vals[_lp*2+1 + 0]));
                        regs_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(my_p_base) + stsm_base_s + 0 + ((mtx_col_s ^ thr_row_idx_s) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3]))
                        : "memory");
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(my_p_base) + stsm_base_s + 2048 + ((mtx_col_s ^ thr_row_idx_s) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[7]))
                        : "memory");
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(my_p_base) + stsm_base_s + 0 + ((mtx_col_s + 2 ^ thr_row_idx_s) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[11]))
                        : "memory");
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(my_p_base) + stsm_base_s + 2048 + ((mtx_col_s + 2 ^ thr_row_idx_s) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[15]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
                    }
                    float sc_0[2];
                    sc_0[0] = 1.0f;
                    sc_0[1] = 1.0f;
                    float md0_0 = old_max[0] - new_max[0];
                    float md1_0 = old_max[1] - new_max[1];
                    if (md0_0 != 0.0f || md1_0 != 0.0f) {
                        float _exp2_89 = approx_exp2(softmax_scale_log2 * md0_0);
                        sc_0[0] = _exp2_89;
                        float _exp2_90 = approx_exp2(softmax_scale_log2 * md1_0);
                        sc_0[1] = _exp2_90;
                    }
                    row_sum[0] = __fmaf_rn(sc_0[0], row_sum[0], s_vals[0]);
                    row_sum[0] = row_sum[0] + s_vals[2];
                    row_sum[0] = row_sum[0] + s_vals[16];
                    row_sum[0] = row_sum[0] + s_vals[18];
                    row_sum[1] = __fmaf_rn(sc_0[1], row_sum[1], s_vals[1]);
                    row_sum[1] = row_sum[1] + s_vals[3];
                    row_sum[1] = row_sum[1] + s_vals[17];
                    row_sum[1] = row_sum[1] + s_vals[19];
                    float sc_1[2];
                    sc_1[0] = 1.0f;
                    sc_1[1] = 1.0f;
                    float md0_1 = old_max[2] - new_max[2];
                    float md1_1 = old_max[3] - new_max[3];
                    if (md0_1 != 0.0f || md1_1 != 0.0f) {
                        float _exp2_91 = approx_exp2(softmax_scale_log2 * md0_1);
                        sc_1[0] = _exp2_91;
                        float _exp2_92 = approx_exp2(softmax_scale_log2 * md1_1);
                        sc_1[1] = _exp2_92;
                    }
                    row_sum[2] = __fmaf_rn(sc_1[0], row_sum[2], s_vals[4]);
                    row_sum[2] = row_sum[2] + s_vals[6];
                    row_sum[2] = row_sum[2] + s_vals[20];
                    row_sum[2] = row_sum[2] + s_vals[22];
                    row_sum[3] = __fmaf_rn(sc_1[1], row_sum[3], s_vals[5]);
                    row_sum[3] = row_sum[3] + s_vals[7];
                    row_sum[3] = row_sum[3] + s_vals[21];
                    row_sum[3] = row_sum[3] + s_vals[23];
                    float sc_2[2];
                    sc_2[0] = 1.0f;
                    sc_2[1] = 1.0f;
                    float md0_2 = old_max[4] - new_max[4];
                    float md1_2 = old_max[5] - new_max[5];
                    if (md0_2 != 0.0f || md1_2 != 0.0f) {
                        float _exp2_93 = approx_exp2(softmax_scale_log2 * md0_2);
                        sc_2[0] = _exp2_93;
                        float _exp2_94 = approx_exp2(softmax_scale_log2 * md1_2);
                        sc_2[1] = _exp2_94;
                    }
                    row_sum[4] = __fmaf_rn(sc_2[0], row_sum[4], s_vals[8]);
                    row_sum[4] = row_sum[4] + s_vals[10];
                    row_sum[4] = row_sum[4] + s_vals[24];
                    row_sum[4] = row_sum[4] + s_vals[26];
                    row_sum[5] = __fmaf_rn(sc_2[1], row_sum[5], s_vals[9]);
                    row_sum[5] = row_sum[5] + s_vals[11];
                    row_sum[5] = row_sum[5] + s_vals[25];
                    row_sum[5] = row_sum[5] + s_vals[27];
                    float sc_3[2];
                    sc_3[0] = 1.0f;
                    sc_3[1] = 1.0f;
                    float md0_3 = old_max[6] - new_max[6];
                    float md1_3 = old_max[7] - new_max[7];
                    if (md0_3 != 0.0f || md1_3 != 0.0f) {
                        float _exp2_95 = approx_exp2(softmax_scale_log2 * md0_3);
                        sc_3[0] = _exp2_95;
                        float _exp2_96 = approx_exp2(softmax_scale_log2 * md1_3);
                        sc_3[1] = _exp2_96;
                    }
                    row_sum[6] = __fmaf_rn(sc_3[0], row_sum[6], s_vals[12]);
                    row_sum[6] = row_sum[6] + s_vals[14];
                    row_sum[6] = row_sum[6] + s_vals[28];
                    row_sum[6] = row_sum[6] + s_vals[30];
                    row_sum[7] = __fmaf_rn(sc_3[1], row_sum[7], s_vals[13]);
                    row_sum[7] = row_sum[7] + s_vals[15];
                    row_sum[7] = row_sum[7] + s_vals[29];
                    row_sum[7] = row_sum[7] + s_vals[31];
                    if (is_last_iter_s != 0) {
                        float stats_s_0[16];
                        stats_s_0[0] = row_sum[0];
                        stats_s_0[8] = new_max[0];
                        stats_s_0[1] = row_sum[1];
                        stats_s_0[9] = new_max[1];
                        stats_s_0[2] = row_sum[2];
                        stats_s_0[10] = new_max[2];
                        stats_s_0[3] = row_sum[3];
                        stats_s_0[11] = new_max[3];
                        stats_s_0[4] = row_sum[4];
                        stats_s_0[12] = new_max[4];
                        stats_s_0[5] = row_sum[5];
                        stats_s_0[13] = new_max[5];
                        stats_s_0[6] = row_sum[6];
                        stats_s_0[14] = new_max[6];
                        stats_s_0[7] = row_sum[7];
                        stats_s_0[15] = new_max[7];
                        if (is_wg1 != 0) {
                            mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        } else {
                            mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        tmem_st_x16_f32(tmem_stats_base, stats_s_0);
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
            const int quad_c = wg_tid_c % 4;
            const int corr_row = warp_in_wg_c * 32 << 16;
            unsigned int* o_words = reinterpret_cast<unsigned int*>(O);
            const int heads_per_kv_c = num_q_heads / num_kv_heads;
            uint32_t _fast_div_q_6 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)(32), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)(32);
            uint32_t _fast_div_r_6 = (uint32_t)(32) - _fast_div_q_6 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_c = (int)_fast_div_q_6;
            int oc_stage = 0;
            int or_stage = 0;
            const int rank_c = cta_rank;
            int cta_q_c = blockIdx.x / 2;
            int head_c = blockIdx.y;
            int batch_c = blockIdx.z;
            const int slice_idx_c = warp_in_wg_c / 2;
            const int warp_idx_in_slice_c = warp_in_wg_c % 2;
            const int mtx_idx_c = lane_c / 8;
            const int thr_row_idx_c = lane_c % 8;
            const int mtx_col_c = warp_idx_in_slice_c * 4 + mtx_idx_c % 2;
            const int stsm_base_c = slice_idx_c * 32 * 128 + (mtx_idx_c / 2 * 8 + thr_row_idx_c) * 128;
            const int red_row_in_slice = wg_tid_c / 16;
            const int red_hd = wg_tid_c % 16 * 8;
            int seq_len_kv_c = seq_lens_kv[batch_c];
            int valid_kv_c = ((cta_q_c * tokens_per_cta_q_c + (seq_len_kv_c - seq_len_q) + tokens_per_cta_q_c) < (seq_len_kv_c) ? (cta_q_c * tokens_per_cta_q_c + (seq_len_kv_c - seq_len_q) + tokens_per_cta_q_c) : (seq_len_kv_c));
            int num_ctas_kv_c = (((valid_kv_c + 255) / 256) < (2) ? ((valid_kv_c + 255) / 256) : (2));
            int num_steps_c = 0;
            if (cta_q_c * tokens_per_cta_q_c < seq_len_q && rank_c < num_ctas_kv_c) {
                num_steps_c = (valid_kv_c + (num_ctas_kv_c * 256 - 1)) / (num_ctas_kv_c * 256);
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
                    float stats_c[16];
                    tmem_ld_x16(&stats_c[0], taddr + 64 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_empty_0_addr);
                    mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                    oc_stage += 1;
                    if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scales[8];
                    scales[0] = 1.0f;
                    scales[1] = 1.0f;
                    float md0_4 = stats_c[0] - stats_c[8];
                    float md1_4 = stats_c[1] - stats_c[9];
                    if (md0_4 != 0.0f || md1_4 != 0.0f) {
                        float _exp2_0 = approx_exp2(softmax_scale_log2 * md0_4);
                        scales[0] = _exp2_0;
                        float _exp2_1 = approx_exp2(softmax_scale_log2 * md1_4);
                        scales[1] = _exp2_1;
                    }
                    scales[2] = 1.0f;
                    scales[3] = 1.0f;
                    float md0_5 = stats_c[2] - stats_c[10];
                    float md1_5 = stats_c[3] - stats_c[11];
                    if (md0_5 != 0.0f || md1_5 != 0.0f) {
                        float _exp2_2 = approx_exp2(softmax_scale_log2 * md0_5);
                        scales[2] = _exp2_2;
                        float _exp2_3 = approx_exp2(softmax_scale_log2 * md1_5);
                        scales[3] = _exp2_3;
                    }
                    scales[4] = 1.0f;
                    scales[5] = 1.0f;
                    float md0_6 = stats_c[4] - stats_c[12];
                    float md1_6 = stats_c[5] - stats_c[13];
                    if (md0_6 != 0.0f || md1_6 != 0.0f) {
                        float _exp2_4 = approx_exp2(softmax_scale_log2 * md0_6);
                        scales[4] = _exp2_4;
                        float _exp2_5 = approx_exp2(softmax_scale_log2 * md1_6);
                        scales[5] = _exp2_5;
                    }
                    scales[6] = 1.0f;
                    scales[7] = 1.0f;
                    float md0_7 = stats_c[6] - stats_c[14];
                    float md1_7 = stats_c[7] - stats_c[15];
                    if (md0_7 != 0.0f || md1_7 != 0.0f) {
                        float _exp2_6 = approx_exp2(softmax_scale_log2 * md0_7);
                        scales[6] = _exp2_6;
                        float _exp2_7 = approx_exp2(softmax_scale_log2 * md1_7);
                        scales[7] = _exp2_7;
                    }
                    int _warp_vote_8 = __all_sync(0xFFFFFFFF, scales[0] == 1.0f && scales[1] == 1.0f && scales[2] == 1.0f && scales[3] == 1.0f && scales[4] == 1.0f && scales[5] == 1.0f && scales[6] == 1.0f && scales[7] == 1.0f);
                    if (_warp_vote_8 == 0) {
                        float o_vals[32];
                        int o_addr = taddr + 128 + (unsigned int)corr_row;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[15]))
                            : "r"(o_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals[31]))
                            : "r"(o_addr + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        o_vals[0] = o_vals[0] * scales[0];
                        o_vals[1] = o_vals[1] * scales[1];
                        o_vals[2] = o_vals[2] * scales[0];
                        o_vals[3] = o_vals[3] * scales[1];
                        o_vals[4] = o_vals[4] * scales[2];
                        o_vals[5] = o_vals[5] * scales[3];
                        o_vals[6] = o_vals[6] * scales[2];
                        o_vals[7] = o_vals[7] * scales[3];
                        o_vals[8] = o_vals[8] * scales[4];
                        o_vals[9] = o_vals[9] * scales[5];
                        o_vals[10] = o_vals[10] * scales[4];
                        o_vals[11] = o_vals[11] * scales[5];
                        o_vals[12] = o_vals[12] * scales[6];
                        o_vals[13] = o_vals[13] * scales[7];
                        o_vals[14] = o_vals[14] * scales[6];
                        o_vals[15] = o_vals[15] * scales[7];
                        o_vals[16] = o_vals[16] * scales[0];
                        o_vals[17] = o_vals[17] * scales[1];
                        o_vals[18] = o_vals[18] * scales[0];
                        o_vals[19] = o_vals[19] * scales[1];
                        o_vals[20] = o_vals[20] * scales[2];
                        o_vals[21] = o_vals[21] * scales[3];
                        o_vals[22] = o_vals[22] * scales[2];
                        o_vals[23] = o_vals[23] * scales[3];
                        o_vals[24] = o_vals[24] * scales[4];
                        o_vals[25] = o_vals[25] * scales[5];
                        o_vals[26] = o_vals[26] * scales[4];
                        o_vals[27] = o_vals[27] * scales[5];
                        o_vals[28] = o_vals[28] * scales[6];
                        o_vals[29] = o_vals[29] * scales[7];
                        o_vals[30] = o_vals[30] * scales[6];
                        o_vals[31] = o_vals[31] * scales[7];
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(o_addr), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[3])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[4])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[5])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[6])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[7])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[8])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[9])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[10])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[11])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[12])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[13])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[14])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(o_addr + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals + 16)[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr + (or_stage) * 8);
                    or_stage += 1;
                    if (or_stage == 2) { or_stage = 0; }
                    mbarrier_wait(stats_full_1_addr, _phase_stats_full_1_0);
                    _phase_stats_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float stats_c_0[16];
                    tmem_ld_x16(&stats_c_0[0], taddr + 96 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_empty_1_addr);
                    mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                    oc_stage += 1;
                    if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scales_0[8];
                    scales_0[0] = 1.0f;
                    scales_0[1] = 1.0f;
                    float md0_8 = stats_c_0[0] - stats_c_0[8];
                    float md1_8 = stats_c_0[1] - stats_c_0[9];
                    if (md0_8 != 0.0f || md1_8 != 0.0f) {
                        float _exp2_9 = approx_exp2(softmax_scale_log2 * md0_8);
                        scales_0[0] = _exp2_9;
                        float _exp2_10 = approx_exp2(softmax_scale_log2 * md1_8);
                        scales_0[1] = _exp2_10;
                    }
                    scales_0[2] = 1.0f;
                    scales_0[3] = 1.0f;
                    float md0_9 = stats_c_0[2] - stats_c_0[10];
                    float md1_9 = stats_c_0[3] - stats_c_0[11];
                    if (md0_9 != 0.0f || md1_9 != 0.0f) {
                        float _exp2_11 = approx_exp2(softmax_scale_log2 * md0_9);
                        scales_0[2] = _exp2_11;
                        float _exp2_12 = approx_exp2(softmax_scale_log2 * md1_9);
                        scales_0[3] = _exp2_12;
                    }
                    scales_0[4] = 1.0f;
                    scales_0[5] = 1.0f;
                    float md0_10 = stats_c_0[4] - stats_c_0[12];
                    float md1_10 = stats_c_0[5] - stats_c_0[13];
                    if (md0_10 != 0.0f || md1_10 != 0.0f) {
                        float _exp2_13 = approx_exp2(softmax_scale_log2 * md0_10);
                        scales_0[4] = _exp2_13;
                        float _exp2_14 = approx_exp2(softmax_scale_log2 * md1_10);
                        scales_0[5] = _exp2_14;
                    }
                    scales_0[6] = 1.0f;
                    scales_0[7] = 1.0f;
                    float md0_11 = stats_c_0[6] - stats_c_0[14];
                    float md1_11 = stats_c_0[7] - stats_c_0[15];
                    if (md0_11 != 0.0f || md1_11 != 0.0f) {
                        float _exp2_15 = approx_exp2(softmax_scale_log2 * md0_11);
                        scales_0[6] = _exp2_15;
                        float _exp2_16 = approx_exp2(softmax_scale_log2 * md1_11);
                        scales_0[7] = _exp2_16;
                    }
                    int _warp_vote_17 = __all_sync(0xFFFFFFFF, scales_0[0] == 1.0f && scales_0[1] == 1.0f && scales_0[2] == 1.0f && scales_0[3] == 1.0f && scales_0[4] == 1.0f && scales_0[5] == 1.0f && scales_0[6] == 1.0f && scales_0[7] == 1.0f);
                    if (_warp_vote_17 == 0) {
                        float o_vals_1[32];
                        int o_addr_1 = taddr + 160 + (unsigned int)corr_row;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[15]))
                            : "r"(o_addr_1));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_vals_1[31]))
                            : "r"(o_addr_1 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        o_vals_1[0] = o_vals_1[0] * scales_0[0];
                        o_vals_1[1] = o_vals_1[1] * scales_0[1];
                        o_vals_1[2] = o_vals_1[2] * scales_0[0];
                        o_vals_1[3] = o_vals_1[3] * scales_0[1];
                        o_vals_1[4] = o_vals_1[4] * scales_0[2];
                        o_vals_1[5] = o_vals_1[5] * scales_0[3];
                        o_vals_1[6] = o_vals_1[6] * scales_0[2];
                        o_vals_1[7] = o_vals_1[7] * scales_0[3];
                        o_vals_1[8] = o_vals_1[8] * scales_0[4];
                        o_vals_1[9] = o_vals_1[9] * scales_0[5];
                        o_vals_1[10] = o_vals_1[10] * scales_0[4];
                        o_vals_1[11] = o_vals_1[11] * scales_0[5];
                        o_vals_1[12] = o_vals_1[12] * scales_0[6];
                        o_vals_1[13] = o_vals_1[13] * scales_0[7];
                        o_vals_1[14] = o_vals_1[14] * scales_0[6];
                        o_vals_1[15] = o_vals_1[15] * scales_0[7];
                        o_vals_1[16] = o_vals_1[16] * scales_0[0];
                        o_vals_1[17] = o_vals_1[17] * scales_0[1];
                        o_vals_1[18] = o_vals_1[18] * scales_0[0];
                        o_vals_1[19] = o_vals_1[19] * scales_0[1];
                        o_vals_1[20] = o_vals_1[20] * scales_0[2];
                        o_vals_1[21] = o_vals_1[21] * scales_0[3];
                        o_vals_1[22] = o_vals_1[22] * scales_0[2];
                        o_vals_1[23] = o_vals_1[23] * scales_0[3];
                        o_vals_1[24] = o_vals_1[24] * scales_0[4];
                        o_vals_1[25] = o_vals_1[25] * scales_0[5];
                        o_vals_1[26] = o_vals_1[26] * scales_0[4];
                        o_vals_1[27] = o_vals_1[27] * scales_0[5];
                        o_vals_1[28] = o_vals_1[28] * scales_0[6];
                        o_vals_1[29] = o_vals_1[29] * scales_0[7];
                        o_vals_1[30] = o_vals_1[30] * scales_0[6];
                        o_vals_1[31] = o_vals_1[31] * scales_0[7];
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(o_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&o_vals_1[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(o_addr_1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(o_vals_1 + 16)[15])));
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
                float final_stats[16];
                tmem_ld_x16(&final_stats[0], taddr + 64 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(stats_empty_0_addr);
                mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                oc_stage += 1;
                if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                mbarrier_wait(stats_full_1_addr, _phase_stats_full_1_0);
                _phase_stats_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_stats_0[16];
                tmem_ld_x16(&final_stats_0[0], taddr + 96 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(stats_empty_1_addr);
                mbarrier_wait(o_full_addr + (oc_stage) * 8, _phase_o_full);
                oc_stage += 1;
                if (oc_stage == 2) { oc_stage = 0; _phase_o_full ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int num_valid_rows_o = seq_len_q - cta_q_c * tokens_per_cta_q_c;
                if (num_valid_rows_o > tokens_per_cta_q_c) {
                    num_valid_rows_o = tokens_per_cta_q_c;
                }
                num_valid_rows_o = num_valid_rows_o * heads_per_kv_c;
                int seq_offset_o = batch_c * seq_len_q + cta_q_c * tokens_per_cta_q_c;
                int heads_o = heads_per_kv_c;
                if (heads_o > 32) {
                    heads_o = 32;
                }
                int cta_offset_o = seq_offset_o * (num_q_heads * 128) + head_c * heads_o * 128;
                int num_valid_slices = (num_valid_rows_o + 7) / 8;
                int num_slices_per_cta = (num_valid_slices + (num_ctas_kv_c - 1)) / num_ctas_kv_c;
                int num_ctas_kv_for_red = (num_valid_slices + (num_slices_per_cta - 1)) / num_slices_per_cta;
                int rows_per_cta = num_slices_per_cta * 8;
                int num_valid_rows_in_cta = ((rows_per_cta) < (num_valid_rows_o - rank_c * rows_per_cta) ? (rows_per_cta) : (num_valid_rows_o - rank_c * rows_per_cta));
                int cga_stats_off = num_ctas_kv_c * rows_per_cta * 256;
                float scales_i0[8];
                float scales_i1[8];
                float sums[8];
                float merged_max[8];
                merged_max[0] = fmaxf(final_stats[8], final_stats_0[8]);
                merged_max[1] = fmaxf(final_stats[9], final_stats_0[9]);
                scales_i0[0] = 1.0f;
                scales_i0[1] = 1.0f;
                scales_i1[0] = 1.0f;
                scales_i1[1] = 1.0f;
                float d00_0 = final_stats[8] - merged_max[0];
                float d01_0 = final_stats[9] - merged_max[1];
                float d10_0 = final_stats_0[8] - merged_max[0];
                float d11_0 = final_stats_0[9] - merged_max[1];
                if (d00_0 != 0.0f || d01_0 != 0.0f || d10_0 != 0.0f || d11_0 != 0.0f) {
                    float _exp2_18 = approx_exp2(softmax_scale_log2 * d00_0);
                    scales_i0[0] = _exp2_18;
                    float _exp2_19 = approx_exp2(softmax_scale_log2 * d01_0);
                    scales_i0[1] = _exp2_19;
                    float _exp2_20 = approx_exp2(softmax_scale_log2 * d10_0);
                    scales_i1[0] = _exp2_20;
                    float _exp2_21 = approx_exp2(softmax_scale_log2 * d11_0);
                    scales_i1[1] = _exp2_21;
                }
                sums[0] = __fmaf_rn(scales_i0[0], final_stats[0], scales_i1[0] * final_stats_0[0]);
                sums[1] = __fmaf_rn(scales_i0[1], final_stats[1], scales_i1[1] * final_stats_0[1]);
                merged_max[2] = fmaxf(final_stats[10], final_stats_0[10]);
                merged_max[3] = fmaxf(final_stats[11], final_stats_0[11]);
                scales_i0[2] = 1.0f;
                scales_i0[3] = 1.0f;
                scales_i1[2] = 1.0f;
                scales_i1[3] = 1.0f;
                float d00_1 = final_stats[10] - merged_max[2];
                float d01_1 = final_stats[11] - merged_max[3];
                float d10_1 = final_stats_0[10] - merged_max[2];
                float d11_1 = final_stats_0[11] - merged_max[3];
                if (d00_1 != 0.0f || d01_1 != 0.0f || d10_1 != 0.0f || d11_1 != 0.0f) {
                    float _exp2_22 = approx_exp2(softmax_scale_log2 * d00_1);
                    scales_i0[2] = _exp2_22;
                    float _exp2_23 = approx_exp2(softmax_scale_log2 * d01_1);
                    scales_i0[3] = _exp2_23;
                    float _exp2_24 = approx_exp2(softmax_scale_log2 * d10_1);
                    scales_i1[2] = _exp2_24;
                    float _exp2_25 = approx_exp2(softmax_scale_log2 * d11_1);
                    scales_i1[3] = _exp2_25;
                }
                sums[2] = __fmaf_rn(scales_i0[2], final_stats[2], scales_i1[2] * final_stats_0[2]);
                sums[3] = __fmaf_rn(scales_i0[3], final_stats[3], scales_i1[3] * final_stats_0[3]);
                merged_max[4] = fmaxf(final_stats[12], final_stats_0[12]);
                merged_max[5] = fmaxf(final_stats[13], final_stats_0[13]);
                scales_i0[4] = 1.0f;
                scales_i0[5] = 1.0f;
                scales_i1[4] = 1.0f;
                scales_i1[5] = 1.0f;
                float d00_2 = final_stats[12] - merged_max[4];
                float d01_2 = final_stats[13] - merged_max[5];
                float d10_2 = final_stats_0[12] - merged_max[4];
                float d11_2 = final_stats_0[13] - merged_max[5];
                if (d00_2 != 0.0f || d01_2 != 0.0f || d10_2 != 0.0f || d11_2 != 0.0f) {
                    float _exp2_26 = approx_exp2(softmax_scale_log2 * d00_2);
                    scales_i0[4] = _exp2_26;
                    float _exp2_27 = approx_exp2(softmax_scale_log2 * d01_2);
                    scales_i0[5] = _exp2_27;
                    float _exp2_28 = approx_exp2(softmax_scale_log2 * d10_2);
                    scales_i1[4] = _exp2_28;
                    float _exp2_29 = approx_exp2(softmax_scale_log2 * d11_2);
                    scales_i1[5] = _exp2_29;
                }
                sums[4] = __fmaf_rn(scales_i0[4], final_stats[4], scales_i1[4] * final_stats_0[4]);
                sums[5] = __fmaf_rn(scales_i0[5], final_stats[5], scales_i1[5] * final_stats_0[5]);
                merged_max[6] = fmaxf(final_stats[14], final_stats_0[14]);
                merged_max[7] = fmaxf(final_stats[15], final_stats_0[15]);
                scales_i0[6] = 1.0f;
                scales_i0[7] = 1.0f;
                scales_i1[6] = 1.0f;
                scales_i1[7] = 1.0f;
                float d00_3 = final_stats[14] - merged_max[6];
                float d01_3 = final_stats[15] - merged_max[7];
                float d10_3 = final_stats_0[14] - merged_max[6];
                float d11_3 = final_stats_0[15] - merged_max[7];
                if (d00_3 != 0.0f || d01_3 != 0.0f || d10_3 != 0.0f || d11_3 != 0.0f) {
                    float _exp2_30 = approx_exp2(softmax_scale_log2 * d00_3);
                    scales_i0[6] = _exp2_30;
                    float _exp2_31 = approx_exp2(softmax_scale_log2 * d01_3);
                    scales_i0[7] = _exp2_31;
                    float _exp2_32 = approx_exp2(softmax_scale_log2 * d10_3);
                    scales_i1[6] = _exp2_32;
                    float _exp2_33 = approx_exp2(softmax_scale_log2 * d11_3);
                    scales_i1[7] = _exp2_33;
                }
                sums[6] = __fmaf_rn(scales_i0[6], final_stats[6], scales_i1[6] * final_stats_0[6]);
                sums[7] = __fmaf_rn(scales_i0[7], final_stats[7], scales_i1[7] * final_stats_0[7]);
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 16);
                sums[0] = sums[0] + _shfl_xor_sync_34;
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 8);
                sums[0] = sums[0] + _shfl_xor_sync_35;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, sums[0], 4);
                sums[0] = sums[0] + _shfl_xor_sync_36;
                float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 16);
                sums[1] = sums[1] + _shfl_xor_sync_37;
                float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 8);
                sums[1] = sums[1] + _shfl_xor_sync_38;
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, sums[1], 4);
                sums[1] = sums[1] + _shfl_xor_sync_39;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 16);
                sums[2] = sums[2] + _shfl_xor_sync_40;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 8);
                sums[2] = sums[2] + _shfl_xor_sync_41;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, sums[2], 4);
                sums[2] = sums[2] + _shfl_xor_sync_42;
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 16);
                sums[3] = sums[3] + _shfl_xor_sync_43;
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 8);
                sums[3] = sums[3] + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, sums[3], 4);
                sums[3] = sums[3] + _shfl_xor_sync_45;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 16);
                sums[4] = sums[4] + _shfl_xor_sync_46;
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 8);
                sums[4] = sums[4] + _shfl_xor_sync_47;
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, sums[4], 4);
                sums[4] = sums[4] + _shfl_xor_sync_48;
                float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 16);
                sums[5] = sums[5] + _shfl_xor_sync_49;
                float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 8);
                sums[5] = sums[5] + _shfl_xor_sync_50;
                float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, sums[5], 4);
                sums[5] = sums[5] + _shfl_xor_sync_51;
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 16);
                sums[6] = sums[6] + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 8);
                sums[6] = sums[6] + _shfl_xor_sync_53;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, sums[6], 4);
                sums[6] = sums[6] + _shfl_xor_sync_54;
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 16);
                sums[7] = sums[7] + _shfl_xor_sync_55;
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 8);
                sums[7] = sums[7] + _shfl_xor_sync_56;
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, sums[7], 4);
                sums[7] = sums[7] + _shfl_xor_sync_57;
                if (lane_c < 4) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(smem_corr_red_addr + (unsigned int)((warp_in_wg_c * 32 + quad_c * 8) * 4)), "f"(sums[0]), "f"(sums[1]), "f"(sums[2]), "f"(sums[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(smem_corr_red_addr + (unsigned int)((warp_in_wg_c * 32 + quad_c * 8 + 4) * 4)), "f"(sums[4]), "f"(sums[5]), "f"(sums[6]), "f"(sums[7]) : "memory");
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float total_sum[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_sum[0])), "=r"(*reinterpret_cast<uint32_t*>(&total_sum[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_sum[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_sum[(0) + 3]))
                    : "r"(smem_corr_red_addr + (unsigned int)(quad_c * 8 * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&total_sum[4])), "=r"(*reinterpret_cast<uint32_t*>(&total_sum[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total_sum[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total_sum[(4) + 3]))
                    : "r"(smem_corr_red_addr + (unsigned int)((quad_c * 8 + 4) * 4)));
                float part_sum[8];
                #pragma unroll
                for (int w = 1; w < 4; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part_sum[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_sum[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_sum[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_sum[(0) + 3]))
                        : "r"(smem_corr_red_addr + (unsigned int)((w * 32 + quad_c * 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part_sum[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_sum[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_sum[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_sum[(4) + 3]))
                        : "r"(smem_corr_red_addr + (unsigned int)((w * 32 + quad_c * 8 + 4) * 4)));
                    total_sum[0] = total_sum[0] + part_sum[0];
                    total_sum[1] = total_sum[1] + part_sum[1];
                    total_sum[2] = total_sum[2] + part_sum[2];
                    total_sum[3] = total_sum[3] + part_sum[3];
                    total_sum[4] = total_sum[4] + part_sum[4];
                    total_sum[5] = total_sum[5] + part_sum[5];
                    total_sum[6] = total_sum[6] + part_sum[6];
                    total_sum[7] = total_sum[7] + part_sum[7];
                }
                if (wg_tid_c < 32) {
                    int stat_sel = wg_tid_c / 4;
                    int stat_row = quad_c * 2 + stat_sel / 2 * 8 + stat_sel % 2;
                    float stat_max = merged_max[0];
                    float stat_sum = total_sum[0];
                    if (stat_sel == 1) {
                        stat_max = merged_max[1];
                        stat_sum = total_sum[1];
                    }
                    if (stat_sel == 2) {
                        stat_max = merged_max[2];
                        stat_sum = total_sum[2];
                    }
                    if (stat_sel == 3) {
                        stat_max = merged_max[3];
                        stat_sum = total_sum[3];
                    }
                    if (stat_sel == 4) {
                        stat_max = merged_max[4];
                        stat_sum = total_sum[4];
                    }
                    if (stat_sel == 5) {
                        stat_max = merged_max[5];
                        stat_sum = total_sum[5];
                    }
                    if (stat_sel == 6) {
                        stat_max = merged_max[6];
                        stat_sum = total_sum[6];
                    }
                    if (stat_sel == 7) {
                        stat_max = merged_max[7];
                        stat_sum = total_sum[7];
                    }
                    if (stat_row < num_valid_rows_o) {
                        int stat_dst = stat_row / rows_per_cta;
                        uint32_t stat_addr;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(stat_addr) : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (rank_c * rows_per_cta + stat_row % rows_per_cta) * 8)), "r"(stat_dst));
                        uint32_t _mbar_peer_58;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_58) : "r"(cga_full_addr), "r"(stat_dst));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(stat_addr), "f"(stat_max), "r"(_mbar_peer_58) : "memory");
                        uint32_t _mbar_peer_59;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_59) : "r"(cga_full_addr), "r"(stat_dst));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(stat_addr + 4), "f"(stat_sum), "r"(_mbar_peer_59) : "memory");
                    }
                }
                float o0_vals[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[3])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[7])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[8])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[9])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[10])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[11])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[12])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[13])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[14])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[15]))
                    : "r"(taddr + 128 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[16])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[17])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[18])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[19])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[20])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[21])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[22])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[23])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[24])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[25])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[26])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[27])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[28])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[29])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[30])), "=r"(*reinterpret_cast<uint32_t*>(&o0_vals[31]))
                    : "r"(taddr + 128 + (unsigned int)corr_row + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float o1_vals[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[3])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[5])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[6])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[7])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[8])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[9])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[10])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[11])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[12])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[13])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[14])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[15]))
                    : "r"(taddr + 160 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[16])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[17])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[18])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[19])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[20])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[21])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[22])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[23])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[24])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[25])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[26])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[27])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[28])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[29])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[30])), "=r"(*reinterpret_cast<uint32_t*>(&o1_vals[31]))
                    : "r"(taddr + 160 + (unsigned int)corr_row + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                o0_vals[0] = __fmaf_rn(o0_vals[0], scales_i0[0], o1_vals[0] * scales_i1[0]);
                o0_vals[1] = __fmaf_rn(o0_vals[1], scales_i0[1], o1_vals[1] * scales_i1[1]);
                o0_vals[2] = __fmaf_rn(o0_vals[2], scales_i0[0], o1_vals[2] * scales_i1[0]);
                o0_vals[3] = __fmaf_rn(o0_vals[3], scales_i0[1], o1_vals[3] * scales_i1[1]);
                o0_vals[4] = __fmaf_rn(o0_vals[4], scales_i0[2], o1_vals[4] * scales_i1[2]);
                o0_vals[5] = __fmaf_rn(o0_vals[5], scales_i0[3], o1_vals[5] * scales_i1[3]);
                o0_vals[6] = __fmaf_rn(o0_vals[6], scales_i0[2], o1_vals[6] * scales_i1[2]);
                o0_vals[7] = __fmaf_rn(o0_vals[7], scales_i0[3], o1_vals[7] * scales_i1[3]);
                o0_vals[8] = __fmaf_rn(o0_vals[8], scales_i0[4], o1_vals[8] * scales_i1[4]);
                o0_vals[9] = __fmaf_rn(o0_vals[9], scales_i0[5], o1_vals[9] * scales_i1[5]);
                o0_vals[10] = __fmaf_rn(o0_vals[10], scales_i0[4], o1_vals[10] * scales_i1[4]);
                o0_vals[11] = __fmaf_rn(o0_vals[11], scales_i0[5], o1_vals[11] * scales_i1[5]);
                o0_vals[12] = __fmaf_rn(o0_vals[12], scales_i0[6], o1_vals[12] * scales_i1[6]);
                o0_vals[13] = __fmaf_rn(o0_vals[13], scales_i0[7], o1_vals[13] * scales_i1[7]);
                o0_vals[14] = __fmaf_rn(o0_vals[14], scales_i0[6], o1_vals[14] * scales_i1[6]);
                o0_vals[15] = __fmaf_rn(o0_vals[15], scales_i0[7], o1_vals[15] * scales_i1[7]);
                o0_vals[16] = __fmaf_rn(o0_vals[16], scales_i0[0], o1_vals[16] * scales_i1[0]);
                o0_vals[17] = __fmaf_rn(o0_vals[17], scales_i0[1], o1_vals[17] * scales_i1[1]);
                o0_vals[18] = __fmaf_rn(o0_vals[18], scales_i0[0], o1_vals[18] * scales_i1[0]);
                o0_vals[19] = __fmaf_rn(o0_vals[19], scales_i0[1], o1_vals[19] * scales_i1[1]);
                o0_vals[20] = __fmaf_rn(o0_vals[20], scales_i0[2], o1_vals[20] * scales_i1[2]);
                o0_vals[21] = __fmaf_rn(o0_vals[21], scales_i0[3], o1_vals[21] * scales_i1[3]);
                o0_vals[22] = __fmaf_rn(o0_vals[22], scales_i0[2], o1_vals[22] * scales_i1[2]);
                o0_vals[23] = __fmaf_rn(o0_vals[23], scales_i0[3], o1_vals[23] * scales_i1[3]);
                o0_vals[24] = __fmaf_rn(o0_vals[24], scales_i0[4], o1_vals[24] * scales_i1[4]);
                o0_vals[25] = __fmaf_rn(o0_vals[25], scales_i0[5], o1_vals[25] * scales_i1[5]);
                o0_vals[26] = __fmaf_rn(o0_vals[26], scales_i0[4], o1_vals[26] * scales_i1[4]);
                o0_vals[27] = __fmaf_rn(o0_vals[27], scales_i0[5], o1_vals[27] * scales_i1[5]);
                o0_vals[28] = __fmaf_rn(o0_vals[28], scales_i0[6], o1_vals[28] * scales_i1[6]);
                o0_vals[29] = __fmaf_rn(o0_vals[29], scales_i0[7], o1_vals[29] * scales_i1[7]);
                o0_vals[30] = __fmaf_rn(o0_vals[30], scales_i0[6], o1_vals[30] * scales_i1[6]);
                o0_vals[31] = __fmaf_rn(o0_vals[31], scales_i0[7], o1_vals[31] * scales_i1[7]);
                unsigned int regs_o[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o0_vals[_lp*2 + 0], o0_vals[_lp*2+1 + 0]));
                    regs_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_p0) + stsm_base_c + 0 + ((mtx_col_c ^ thr_row_idx_c) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[3]))
                    : "memory");
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_p0) + stsm_base_c + 2048 + ((mtx_col_c ^ thr_row_idx_c) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[7]))
                    : "memory");
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_p0) + stsm_base_c + 0 + ((mtx_col_c + 2 ^ thr_row_idx_c) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[8])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[9])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[10])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[11]))
                    : "memory");
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_p0) + stsm_base_c + 2048 + ((mtx_col_c + 2 ^ thr_row_idx_c) * 16))), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[12])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[13])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[14])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[15]))
                    : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                int copy_off_0 = wg_tid_c * 16;
                int copy_row_0 = copy_off_0 / 128;
                int tok_row_0 = copy_row_0 % 32;
                if (tok_row_0 < num_valid_rows_o) {
                    unsigned int o_vec_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_vec_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_0[(0) + 3]))
                        : "r"(smem_p0_addr + (unsigned int)(copy_off_0 ^ copy_row_0 % 8 * 16)));
                    int o_dst_0 = tok_row_0 / rows_per_cta;
                    int o_off_0 = (rank_c * rows_per_cta + tok_row_0 % rows_per_cta) * 256 + copy_row_0 / 32 * 128 + copy_off_0 % 128;
                    uint32_t _cluster_mapa_60;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_60) : "r"(smem_cga_addr + (unsigned int)o_off_0), "r"(o_dst_0));
                    uint32_t _mbar_peer_61;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_61) : "r"(cga_full_addr), "r"(o_dst_0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_60), "r"(o_vec_0[0]), "r"(o_vec_0[1]), "r"(o_vec_0[2]), "r"(o_vec_0[3]), "r"(_mbar_peer_61) : "memory");
                }
                int copy_off_1 = wg_tid_c * 16 + 2048;
                int copy_row_1 = copy_off_1 / 128;
                int tok_row_1 = copy_row_1 % 32;
                if (tok_row_1 < num_valid_rows_o) {
                    unsigned int o_vec_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_1[(0) + 3]))
                        : "r"(smem_p0_addr + (unsigned int)(copy_off_1 ^ copy_row_1 % 8 * 16)));
                    int o_dst_1 = tok_row_1 / rows_per_cta;
                    int o_off_1 = (rank_c * rows_per_cta + tok_row_1 % rows_per_cta) * 256 + copy_row_1 / 32 * 128 + copy_off_1 % 128;
                    uint32_t _cluster_mapa_62;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_62) : "r"(smem_cga_addr + (unsigned int)o_off_1), "r"(o_dst_1));
                    uint32_t _mbar_peer_63;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_63) : "r"(cga_full_addr), "r"(o_dst_1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_62), "r"(o_vec_1[0]), "r"(o_vec_1[1]), "r"(o_vec_1[2]), "r"(o_vec_1[3]), "r"(_mbar_peer_63) : "memory");
                }
                int copy_off_2 = wg_tid_c * 16 + 4096;
                int copy_row_2 = copy_off_2 / 128;
                int tok_row_2 = copy_row_2 % 32;
                if (tok_row_2 < num_valid_rows_o) {
                    unsigned int o_vec_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_vec_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_2[(0) + 3]))
                        : "r"(smem_p0_addr + (unsigned int)(copy_off_2 ^ copy_row_2 % 8 * 16)));
                    int o_dst_2 = tok_row_2 / rows_per_cta;
                    int o_off_2 = (rank_c * rows_per_cta + tok_row_2 % rows_per_cta) * 256 + copy_row_2 / 32 * 128 + copy_off_2 % 128;
                    uint32_t _cluster_mapa_64;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_64) : "r"(smem_cga_addr + (unsigned int)o_off_2), "r"(o_dst_2));
                    uint32_t _mbar_peer_65;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_65) : "r"(cga_full_addr), "r"(o_dst_2));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_64), "r"(o_vec_2[0]), "r"(o_vec_2[1]), "r"(o_vec_2[2]), "r"(o_vec_2[3]), "r"(_mbar_peer_65) : "memory");
                }
                int copy_off_3 = wg_tid_c * 16 + 6144;
                int copy_row_3 = copy_off_3 / 128;
                int tok_row_3 = copy_row_3 % 32;
                if (tok_row_3 < num_valid_rows_o) {
                    unsigned int o_vec_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&o_vec_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec_3[(0) + 3]))
                        : "r"(smem_p0_addr + (unsigned int)(copy_off_3 ^ copy_row_3 % 8 * 16)));
                    int o_dst_3 = tok_row_3 / rows_per_cta;
                    int o_off_3 = (rank_c * rows_per_cta + tok_row_3 % rows_per_cta) * 256 + copy_row_3 / 32 * 128 + copy_off_3 % 128;
                    uint32_t _cluster_mapa_66;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_66) : "r"(smem_cga_addr + (unsigned int)o_off_3), "r"(o_dst_3));
                    uint32_t _mbar_peer_67;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_67) : "r"(cga_full_addr), "r"(o_dst_3));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_66), "r"(o_vec_3[0]), "r"(o_vec_3[1]), "r"(o_vec_3[2]), "r"(o_vec_3[3]), "r"(_mbar_peer_67) : "memory");
                }
                if (rank_c < num_ctas_kv_for_red) {
                    if (wg_tid_c == 0) {
                        mbarrier_arrive_expect_tx(cga_full_addr, num_valid_rows_in_cta * num_ctas_kv_c * 264);
                    }
                    mbarrier_wait_cluster_hint(cga_full_addr, 0, 10000000);
                    int start_slice = rank_c * num_slices_per_cta;
                    int end_slice = ((start_slice + num_slices_per_cta) < (num_valid_slices) ? (start_slice + num_slices_per_cta) : (num_valid_slices));
                    #pragma unroll 1
                    for (int slice_r = start_slice; slice_r < end_slice; slice_r++) {
                        int row_r = slice_r * 8 + red_row_in_slice;
                        int valid_row_r = ((row_r) < (num_valid_rows_o - 1) ? (row_r) : (num_valid_rows_o - 1));
                        int load_row_r = valid_row_r - start_slice * 8;
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
                            int _min_68 = ((ii_r) < (num_ctas_kv_c - 1) ? (ii_r) : (num_ctas_kv_c - 1));
                            k_r = _min_68;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(0) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(0) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * 128 + red_hd) * 2)));
                            int _min_69 = ((ii_r + 1) < (num_ctas_kv_c - 1) ? (ii_r + 1) : (num_ctas_kv_c - 1));
                            k_r = _min_69;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[2])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(2) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(4) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * 128 + red_hd) * 2)));
                            int _min_70 = ((ii_r + 2) < (num_ctas_kv_c - 1) ? (ii_r + 2) : (num_ctas_kv_c - 1));
                            k_r = _min_70;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(4) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(8) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * 128 + red_hd) * 2)));
                            int _min_71 = ((ii_r + 3) < (num_ctas_kv_c - 1) ? (ii_r + 3) : (num_ctas_kv_c - 1));
                            k_r = _min_71;
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_stats[6])), "=r"(*reinterpret_cast<uint32_t*>(&part_stats[(6) + 1]))
                                : "r"(smem_cga_addr + (unsigned int)(cga_stats_off + (k_r * rows_per_cta + load_row_r) * 8)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&part_o[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_o[(12) + 3]))
                                : "r"(smem_cga_addr + (unsigned int)(((k_r * rows_per_cta + load_row_r) * 128 + red_hd) * 2)));
                            local_max = part_stats[0];
                            local_sum = part_stats[1];
                            max_v = fmaxf(max_v, local_max);
                            corr0 = 1.0f;
                            corr1 = 0.0f;
                            if (ii_r < num_ctas_kv_c) {
                                float _exp2_72_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_72_1;
                                float _exp2_73_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_73_1;
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
                                float _exp2_74_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_74_1;
                                float _exp2_75_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_75_1;
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
                                float _exp2_76_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_76_1;
                                float _exp2_77_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_77_1;
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
                                float _exp2_78_1 = approx_exp2(softmax_scale_log2 * (old_max_v - max_v));
                                corr0 = _exp2_78_1;
                                float _exp2_79_1 = approx_exp2(softmax_scale_log2 * (local_max - max_v));
                                corr1 = _exp2_79_1;
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
                            int dst_words_r = cta_offset_o / 2 + dst_row_r * 64 + red_hd / 2;
                            {
                                auto _gst_80 = o_words + dst_words_r;
                                __builtin_assume(__isGlobal(_gst_80));
                                reinterpret_cast<int4*>(_gst_80)[0] = reinterpret_cast<int4*>(out_w)[0];
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
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            uint32_t _fast_div_q_2 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)(32), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)(32);
            uint32_t _fast_div_r_2 = (uint32_t)(32) - _fast_div_q_2 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_m = (int)_fast_div_q_2;
            int qm_stage = 0;
            int mk_stage = 0;
            int om_stage = 0;
            const int rank_m = cta_rank;
            int cta_q_m = blockIdx.x / 2;
            int batch_m = blockIdx.z;
            int seq_len_kv_m = seq_lens_kv[batch_m];
            int valid_kv_m = ((cta_q_m * tokens_per_cta_q_m + (seq_len_kv_m - seq_len_q) + tokens_per_cta_q_m) < (seq_len_kv_m) ? (cta_q_m * tokens_per_cta_q_m + (seq_len_kv_m - seq_len_q) + tokens_per_cta_q_m) : (seq_len_kv_m));
            int num_ctas_kv_m = (((valid_kv_m + 255) / 256) < (2) ? ((valid_kv_m + 255) / 256) : (2));
            int num_steps_m = 0;
            if (cta_q_m * tokens_per_cta_q_m < seq_len_q && rank_m < num_ctas_kv_m) {
                num_steps_m = (valid_kv_m + (num_ctas_kv_m * 256 - 1)) / (num_ctas_kv_m * 256);
            }
            unsigned int _phase_q_full = 0;
            unsigned int _phase_s_empty_0_0 = 1;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_s_empty_1_0 = 1;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_o_empty = 1;
            unsigned int _phase_p_full_1_0 = 0;
            if (num_steps_m > 0) {
                mbarrier_wait(q_full_addr + (qm_stage) * 8, _phase_q_full);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                int mma_b_lo = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 512);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
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
                int mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                int mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 512);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
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
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                    int mma_b_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 512);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(s_full_0_addr);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                    int mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x1000000);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, ((first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134775952, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(o_full_addr + (om_stage) * 8);
                    om_stage += 1;
                    if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                    elect_commit(p_empty_0_addr);
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 2048);
                    int mma_b_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 512);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134743184, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(s_full_1_addr);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                    int mma_b_lo_4 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x1000000);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, ((first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(o_full_addr + (om_stage) * 8);
                    om_stage += 1;
                    if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                    elect_commit(p_empty_1_addr);
                }
                int last_first_pv_m = ((num_steps_m == 1) ? 1 : 0);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                _phase_p_full_0_0 ^= 1;
                mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                int mma_b_lo_5 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x1000000);
                {
                    uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_5);
                    uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_5);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, ((last_first_pv_m) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134775952, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(o_full_addr + (om_stage) * 8);
                om_stage += 1;
                if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                elect_commit(s_full_0_addr);
                elect_commit(p_empty_0_addr);
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 2048);
                int mma_b_lo_6 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x1000000);
                {
                    uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_6);
                    uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_6);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, ((last_first_pv_m) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134775952, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 4) { mk_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(o_full_addr + (om_stage) * 8);
                om_stage += 1;
                if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                elect_commit(s_full_1_addr);
                elect_commit(p_empty_1_addr);
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
            uint32_t _fast_div_q_0 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)(32), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)(32);
            uint32_t _fast_div_r_0 = (uint32_t)(32) - _fast_div_q_0 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_p = (int)_fast_div_q_0;
            int pp_stage = 0;
            const int rank_p = cta_rank;
            int cta_q_p = blockIdx.x / 2;
            int batch_p = blockIdx.z;
            int seq_len_kv_p = seq_lens_kv[batch_p];
            int valid_kv_p = ((cta_q_p * tokens_per_cta_q_p + (seq_len_kv_p - seq_len_q) + tokens_per_cta_q_p) < (seq_len_kv_p) ? (cta_q_p * tokens_per_cta_q_p + (seq_len_kv_p - seq_len_q) + tokens_per_cta_q_p) : (seq_len_kv_p));
            int num_ctas_kv_p = (((valid_kv_p + 255) / 256) < (2) ? ((valid_kv_p + 255) / 256) : (2));
            int num_steps_p = 0;
            if (cta_q_p * tokens_per_cta_q_p < seq_len_q && rank_p < num_ctas_kv_p) {
                num_steps_p = (valid_kv_p + (num_ctas_kv_p * 256 - 1)) / (num_ctas_kv_p * 256);
            }
            int num_loop_steps_p = num_steps_p * 2;
            unsigned int _phase_page_empty = 1;
            if (num_loop_steps_p > 0) {
                int pt_base_p = batch_p * max_pages_per_seq;
                int page_idx_ub_p = (seq_len_kv_p + 15) / 16 - 1;
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
            uint32_t _fast_div_q_1 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)(32), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)(32);
            uint32_t _fast_div_r_1 = (uint32_t)(32) - _fast_div_q_1 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_l = (int)_fast_div_q_1;
            int q_stage = 0;
            int kv_stage = 0;
            int pc_stage = 0;
            int pr_stage = 0;
            int k_sid = 0;
            int v_sid = 0;
            const int rank_l = cta_rank;
            int cta_q_l = blockIdx.x / 2;
            int head_l = blockIdx.y;
            int batch_l = blockIdx.z;
            int seq_len_kv_l = seq_lens_kv[batch_l];
            int valid_kv_l = ((cta_q_l * tokens_per_cta_q_l + (seq_len_kv_l - seq_len_q) + tokens_per_cta_q_l) < (seq_len_kv_l) ? (cta_q_l * tokens_per_cta_q_l + (seq_len_kv_l - seq_len_q) + tokens_per_cta_q_l) : (seq_len_kv_l));
            int num_ctas_kv_l = (((valid_kv_l + 255) / 256) < (2) ? ((valid_kv_l + 255) / 256) : (2));
            int num_steps_l = 0;
            if (cta_q_l * tokens_per_cta_q_l < seq_len_q && rank_l < num_ctas_kv_l) {
                num_steps_l = (valid_kv_l + (num_ctas_kv_l * 256 - 1)) / (num_ctas_kv_l * 256);
            }
            int num_loop_steps_l = num_steps_l * 2;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
            if (num_loop_steps_l > 0) {
                mbarrier_wait(q_empty_addr + (q_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 8192);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                if (elect_sync()) {
                    int q_token_l = batch_l * seq_len_q + cta_q_l * tokens_per_cta_q_l;
                    #pragma unroll
                    for (int q_box = 0; q_box < 2; q_box++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 8192 + q_box * 4096), (&Q), 0, head_l * 8, q_box, q_token_l, q_full_addr + (q_stage) * 8);
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
                int kv_dst = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_a = 0; pg_a < 4; pg_a++) {
                        int pool_page_lo = ids_lo[pg_a] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_a = 0; slab_a < 2; slab_a++) {
                            tma_4d_gmem2smem(kv_dst + pg_a * 2048 + slab_a * 16384, (&K), 0, 0, slab_a, pool_page_lo, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + 16)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_b = 0; pg_b < 4; pg_b++) {
                        int pool_page_hi = ids_hi[pg_b] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_b = 0; slab_b < 2; slab_b++) {
                            tma_4d_gmem2smem(kv_dst + (4 + pg_b) * 2048 + slab_b * 16384, (&K), 0, 0, slab_b, pool_page_hi, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                int ids_lo_0[4];
                int ids_hi_0[4];
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_0 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + 32)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_a_1 = 0; pg_a_1 < 4; pg_a_1++) {
                        int pool_page_lo_1 = ids_lo_0[pg_a_1] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_a_1 = 0; slab_a_1 < 2; slab_a_1++) {
                            tma_4d_gmem2smem(kv_dst_0 + pg_a_1 * 2048 + slab_a_1 * 16384, (&K), 0, 0, slab_a_1, pool_page_lo_1, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + 32 + 16)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_b_1 = 0; pg_b_1 < 4; pg_b_1++) {
                        int pool_page_hi_1 = ids_hi_0[pg_b_1] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_b_1 = 0; slab_b_1 < 2; slab_b_1++) {
                            tma_4d_gmem2smem(kv_dst_0 + (4 + pg_b_1) * 2048 + slab_b_1 * 16384, (&K), 0, 0, slab_b_1, pool_page_hi_1, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                #pragma unroll 1
                for (int lo_l = 0; lo_l < num_loop_steps_l - 2; lo_l++) {
                    int is_last_iter_l = ((lo_l + 1 >= num_loop_steps_l - 2) ? 1 : 0);
                    if ((lo_l + 2) % 4 == 0) {
                        mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                        k_sid = pc_stage;
                        pc_stage += 1;
                        if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                    }
                    int ids_lo_1[4];
                    int ids_hi_1[4];
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int kv_dst_1 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + (lo_l + 2) % 4 * 32)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_a_2 = 0; pg_a_2 < 4; pg_a_2++) {
                            int pool_page_lo_2 = ids_lo_1[pg_a_2] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_a_2 = 0; slab_a_2 < 2; slab_a_2++) {
                                tma_4d_gmem2smem(kv_dst_1 + pg_a_2 * 2048 + slab_a_2 * 16384, (&K), 0, 0, slab_a_2, pool_page_lo_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(k_sid * 128 + (lo_l + 2) % 4 * 32 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_b_2 = 0; pg_b_2 < 4; pg_b_2++) {
                            int pool_page_hi_2 = ids_hi_1[pg_b_2] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_b_2 = 0; slab_b_2 < 2; slab_b_2++) {
                                tma_4d_gmem2smem(kv_dst_1 + (4 + pg_b_2) * 2048 + slab_b_2 * 16384, (&K), 0, 0, slab_b_2, pool_page_hi_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    if (is_last_iter_l == 0 && (lo_l + 2) % 4 == 3) {
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
                    int ids_lo_2[4];
                    int ids_hi_2[4];
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int kv_dst_2 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + lo_l % 4 * 32)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_a_3 = 0; pg_a_3 < 4; pg_a_3++) {
                            int pool_page_lo_3 = ids_lo_2[pg_a_3] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_a_3 = 0; slab_a_3 < 2; slab_a_3++) {
                                tma_4d_gmem2smem(kv_dst_2 + pg_a_3 * 2048 + slab_a_3 * 16384, (&V), 0, 0, slab_a_3, pool_page_lo_3, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + lo_l % 4 * 32 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_b_3 = 0; pg_b_3 < 4; pg_b_3++) {
                            int pool_page_hi_3 = ids_hi_2[pg_b_3] * num_kv_heads + head_l;
                            #pragma unroll
                            for (int slab_b_3 = 0; slab_b_3 < 2; slab_b_3++) {
                                tma_4d_gmem2smem(kv_dst_2 + (4 + pg_b_3) * 2048 + slab_b_3 * 16384, (&V), 0, 0, slab_b_3, pool_page_hi_3, kv_full_addr + (kv_stage) * 8);
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
                int last_l = num_loop_steps_l - 2;
                mbarrier_arrive(page_empty_addr + (pr_stage) * 8);
                pr_stage += 1;
                if (pr_stage == 6) { pr_stage = 0; }
                if (last_l % 4 == 0) {
                    mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                    v_sid = pc_stage;
                    pc_stage += 1;
                    if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                }
                int ids_lo_3[4];
                int ids_hi_3[4];
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_3 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + last_l % 4 * 32)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_a_4 = 0; pg_a_4 < 4; pg_a_4++) {
                        int pool_page_lo_4 = ids_lo_3[pg_a_4] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_a_4 = 0; slab_a_4 < 2; slab_a_4++) {
                            tma_4d_gmem2smem(kv_dst_3 + pg_a_4 * 2048 + slab_a_4 * 16384, (&V), 0, 0, slab_a_4, pool_page_lo_4, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + last_l % 4 * 32 + 16)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_b_4 = 0; pg_b_4 < 4; pg_b_4++) {
                        int pool_page_hi_4 = ids_hi_3[pg_b_4] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_b_4 = 0; slab_b_4 < 2; slab_b_4++) {
                            tma_4d_gmem2smem(kv_dst_3 + (4 + pg_b_4) * 2048 + slab_b_4 * 16384, (&V), 0, 0, slab_b_4, pool_page_hi_4, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; _phase_kv_empty ^= 1; }
                int ids_lo_4[4];
                int ids_hi_4[4];
                mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                }
                int kv_dst_4 = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + (last_l + 1) % 4 * 32)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_a_5 = 0; pg_a_5 < 4; pg_a_5++) {
                        int pool_page_lo_5 = ids_lo_4[pg_a_5] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_a_5 = 0; slab_a_5 < 2; slab_a_5++) {
                            tma_4d_gmem2smem(kv_dst_4 + pg_a_5 * 2048 + slab_a_5 * 16384, (&V), 0, 0, slab_a_5, pool_page_lo_5, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[(0) + 3]))
                    : "r"(smem_page_ids_addr + (unsigned int)(v_sid * 128 + (last_l + 1) % 4 * 32 + 16)));
                if (elect_sync()) {
                    #pragma unroll
                    for (int pg_b_5 = 0; pg_b_5 < 4; pg_b_5++) {
                        int pool_page_hi_5 = ids_hi_4[pg_b_5] * num_kv_heads + head_l;
                        #pragma unroll
                        for (int slab_b_5 = 0; slab_b_5 < 2; slab_b_5++) {
                            tma_4d_gmem2smem(kv_dst_4 + (4 + pg_b_5) * 2048 + slab_b_5 * 16384, (&V), 0, 0, slab_b_5, pool_page_hi_5, kv_full_addr + (kv_stage) * 8);
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
