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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 96
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 9
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_O_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 0
#define SMEM_SMEM_Q_STAGE_BYTES 1024
#define SMEM_SMEM_Q_STRIDE 1024
#define SMEM_SMEM_KV_OFF 2048
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 2048
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P0_OFF 149504
#define SMEM_SMEM_P0_STAGE_BYTES 2048
#define SMEM_SMEM_P0_STRIDE 2048
#define SMEM_SMEM_P1_OFF 151552
#define SMEM_SMEM_P1_STAGE_BYTES 2048
#define SMEM_SMEM_P1_STRIDE 2048
#define SMEM_SMEM_PAGE_IDS_OFF 153600
#define SMEM_SMEM_PAGE_IDS_STAGE_BYTES 768
#define SMEM_SMEM_PAGE_IDS_STRIDE 768
#define SMEM_SMEM_O_OFF 154368
#define SMEM_SMEM_O_STAGE_BYTES 1024
#define SMEM_SMEM_O_STRIDE 1024
#define SMEM_SMEM_RED0_OFF 155392
#define SMEM_SMEM_RED0_STAGE_BYTES 128
#define SMEM_SMEM_RED0_STRIDE 128
#define SMEM_SMEM_RED1_OFF 155520
#define SMEM_SMEM_RED1_STAGE_BYTES 128
#define SMEM_SMEM_RED1_STRIDE 128
#define SMEM_SMEM_CORR_RED_OFF 155648
#define SMEM_SMEM_CORR_RED_STAGE_BYTES 256
#define SMEM_SMEM_CORR_RED_STRIDE 256
#define SMEM_WORK_RESPONSE_OFF 155904
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 156544
#define THREADS 512
#define BLOCK_N 128
#define STEP_N 256
#define HEAD_DIM 64
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_aligned_hd64_f922a4aaa2d1161d3a26(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, CakeFastDivmod heads_per_kv, int batch_size, float skip_corr_threshold)
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

    const int mbar_base = smem + 156016;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 104)
    #define clc_full_addr (mbar_base + 176)
    #define s_full_0_addr (mbar_base + 192)
    #define s_full_1_addr (mbar_base + 200)
    #define p_empty_0_addr (mbar_base + 208)
    #define p_empty_1_addr (mbar_base + 216)
    #define o_full_addr (mbar_base + 224)
    #define page_full_addr (mbar_base + 240)
    #define page_empty_addr (mbar_base + 288)
    #define throttle_full_addr (mbar_base + 336)
    #define throttle_empty_addr (mbar_base + 352)
    #define s_empty_0_addr (mbar_base + 368)
    #define stats_full_0_addr (mbar_base + 376)
    #define stats_empty_0_addr (mbar_base + 384)
    #define s_empty_1_addr (mbar_base + 392)
    #define stats_full_1_addr (mbar_base + 400)
    #define stats_empty_1_addr (mbar_base + 408)
    #define order_p0_addr (mbar_base + 416)
    #define order_p1_addr (mbar_base + 424)
    #define p_full_0_addr (mbar_base + 432)
    #define p_full_1_addr (mbar_base + 440)
    #define o_empty_addr (mbar_base + 448)
    #define clc_empty_addr (mbar_base + 464)
    #define reserved_arv512_addr (mbar_base + 480)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_q_addr = smem + 0;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_kv_addr = smem + 2048;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_v_addr = smem + 2048;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int smem_p0_addr = smem + 149504;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 151552);
    const int smem_p1_addr = smem + 151552;
    int* smem_page_ids = reinterpret_cast<int*>(smem_raw + 153600);
    const int smem_page_ids_addr = smem + 153600;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 154368);
    const int smem_o_addr = smem + 154368;
    unsigned int* smem_red0 = reinterpret_cast<unsigned int*>(smem_raw + 155392);
    const int smem_red0_addr = smem + 155392;
    unsigned int* smem_red1 = reinterpret_cast<unsigned int*>(smem_raw + 155520);
    const int smem_red1_addr = smem + 155520;
    float* smem_corr_red = reinterpret_cast<float*>(smem_raw + 155648);
    const int smem_corr_red_addr = smem + 155648;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 155904);
    const int work_response_addr = smem + 155904;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (27 pipeline groups, 0 ordered-sequence groups, 64 barriers)
    // Mbarriers at smem_raw[156016..156528)

    if (warp == 0) {
        // --- pipeline 'q_pipe' ---
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 9 barriers, init_count=1
        // kv_empty: 9 barriers, init_count=1
        // --- pipeline 'work_pipe' ---
        // clc_full: 2 barriers, init_count=1
        // s_full_0: 1 barriers, init_count=1
        // s_full_1: 1 barriers, init_count=1
        // p_empty_0: 1 barriers, init_count=1
        // p_empty_1: 1 barriers, init_count=1
        // --- pipeline 'o_pipe' ---
        // o_full: 2 barriers, init_count=1
        // --- pipeline 'page_pipe' ---
        // page_full: 6 barriers, init_count=32
        // page_empty: 6 barriers, init_count=32
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 2 barriers, init_count=32
        // throttle_empty: 2 barriers, init_count=32
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
        // --- pipeline 'work_pipe' ---
        // clc_empty: 2 barriers, init_count=512
        // reserved_arv512: 4 barriers, init_count=512
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(1)));
        mbarrier_init(smem + 156016 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 512;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(26), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(14), "r"((uint32_t)(32)));
        mbarrier_init(smem + 156272 + lane * 8, _mbarrier_init_count_0_32);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 156528);
    if (warp == 0) {
        int _tmem_hold = smem + 156528;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0 = taddr + 80;
    const int tmem_tmem_o1 = taddr + 88;
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
            unsigned int num_tiles_s2 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)gridDim.z;
            uint32_t _fast_div_q_3 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_3 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_s = (int)_fast_div_q_3;
            int ws_sm = 0;
            int cta_x_s = blockIdx.x;
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
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_stats_empty_1_0 = 1;
            unsigned int _phase_stats_empty_0_0 = 1;
            unsigned int _phase_p_empty_1_0 = 1;
            unsigned int _phase_p_empty_0_0 = 1;
            unsigned int _phase_order_p1_0 = 0;
            unsigned int _phase_order_p0_0 = 1;
            unsigned int _phase_clc_full = 0;
            #pragma unroll 1
            while (true) {
                int seq_len_kv_s = seq_lens_kv[batch_s];
                int valid_kv_s = ((cta_x_s * tokens_per_cta_q_s + (seq_len_kv_s - SEQ_LEN_Q) + tokens_per_cta_q_s) < (seq_len_kv_s) ? (cta_x_s * tokens_per_cta_q_s + (seq_len_kv_s - SEQ_LEN_Q) + tokens_per_cta_q_s) : (seq_len_kv_s));
                int num_steps_s = 0;
                if (cta_x_s * tokens_per_cta_q_s < SEQ_LEN_Q && valid_kv_s > 0) {
                    num_steps_s = (valid_kv_s + (STEP_N - 1)) / STEP_N;
                }
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
                int tile_offset_q = cta_x_s * tokens_per_cta_q_s + (seq_len_kv_s - SEQ_LEN_Q);
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
                    int tile_offset_k = (lo_s * 2 + is_wg1) * BLOCK_N;
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
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(s_empty_1_addr);
                    } else {
                        mbarrier_arrive(s_empty_0_addr);
                    }
                    float stats_s[4];
                    stats_s[0] = old_max[0];
                    stats_s[1] = old_max[1];
                    stats_s[2] = new_max[0];
                    stats_s[3] = new_max[1];
                    if (is_wg1 != 0) {
                        mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                        _phase_stats_empty_1_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                        _phase_stats_empty_0_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    tmem_st_x4_f32(tmem_stats_base, stats_s);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(stats_full_1_addr);
                    } else {
                        mbarrier_arrive(stats_full_0_addr);
                    }
                    if (is_wg1 != 0) {
                        mbarrier_wait(p_empty_1_addr, _phase_p_empty_1_0);
                        _phase_p_empty_1_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait(p_empty_0_addr, _phase_p_empty_0_0);
                        _phase_p_empty_0_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
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
                        mbarrier_arrive(p_full_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
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
                        if (is_wg1 != 0) {
                            mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                            _phase_stats_empty_1_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        } else {
                            mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                            _phase_stats_empty_0_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        tmem_st_x4_f32(tmem_stats_base, final_stats_s);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (is_wg1 != 0) {
                            mbarrier_arrive(stats_full_1_addr);
                        } else {
                            mbarrier_arrive(stats_full_0_addr);
                        }
                        if (is_wg1 != 0) {
                            mbarrier_wait(stats_empty_1_addr, _phase_stats_empty_1_0);
                        } else {
                            mbarrier_wait(stats_empty_0_addr, _phase_stats_empty_0_0);
                        }
                    }
                }
                mbarrier_wait(clc_full_addr + (ws_sm) * 8, _phase_clc_full);
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
                    : "r"(work_response_addr + ws_sm * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(clc_empty_addr + (ws_sm) * 8);
                ws_sm += 1;
                if (ws_sm == 2) { ws_sm = 0; _phase_clc_full ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                cta_x_s = (int)_clc_ctaid_x_4;
                batch_s = (int)_clc_ctaid_z_4;
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
            unsigned int num_tiles_c = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)gridDim.z;
            const int heads_per_kv_c = num_q_heads / num_kv_heads;
            uint32_t _fast_div_q_6 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_6 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_6 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_c = (int)_fast_div_q_6;
            int oc_stage = 0;
            int or_stage = 0;
            int ws_c = 0;
            int cta_x_c = blockIdx.x;
            int head_c = blockIdx.y;
            int batch_c = blockIdx.z;
            const int mtx_idx_c = lane_c / 8;
            const int mtx_row_idx_c = mtx_idx_c / 2;
            const int mtx_col_idx_c = mtx_idx_c % 2;
            const int thr_row_idx_c = lane_c % 8;
            const int seg_col_idx_c = warp_in_wg_c * 2 + mtx_col_idx_c ^ thr_row_idx_c;
            const int stsm_offset_c = (mtx_row_idx_c * 8 + thr_row_idx_c) * 128 + seg_col_idx_c * 16;
            const int smem_row_c = wg_tid_c / 8;
            const int load_smem_off_c = wg_tid_c * 16 ^ smem_row_c % 8 * 16;
            const int dst_col_words_c = wg_tid_c % 8 * 4;
            unsigned int _phase_stats_full_0_0 = 0;
            unsigned int _phase_stats_full_1_0 = 0;
            unsigned int _phase_o_full = 0;
            unsigned int _phase_clc_full_1 = 0;
            #pragma unroll 1
            while (true) {
                int seq_len_kv_c = seq_lens_kv[batch_c];
                int valid_kv_c = ((cta_x_c * tokens_per_cta_q_c + (seq_len_kv_c - SEQ_LEN_Q) + tokens_per_cta_q_c) < (seq_len_kv_c) ? (cta_x_c * tokens_per_cta_q_c + (seq_len_kv_c - SEQ_LEN_Q) + tokens_per_cta_q_c) : (seq_len_kv_c));
                int num_steps_c = 0;
                if (cta_x_c * tokens_per_cta_q_c < SEQ_LEN_Q && valid_kv_c > 0) {
                    num_steps_c = (valid_kv_c + (STEP_N - 1)) / STEP_N;
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
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
                            float _tmem_load_1[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                                : "r"(taddr + 80 + (unsigned int)corr_row));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            _tmem_load_1[0] = _tmem_load_1[0] * scales[0];
                            _tmem_load_1[1] = _tmem_load_1[1] * scales[1];
                            _tmem_load_1[2] = _tmem_load_1[2] * scales[0];
                            _tmem_load_1[3] = _tmem_load_1[3] * scales[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(taddr + 80 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])));
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
                            float _tmem_load_3[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
                                : "r"(taddr + 88 + (unsigned int)corr_row));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            _tmem_load_3[0] = _tmem_load_3[0] * scales_0[0];
                            _tmem_load_3[1] = _tmem_load_3[1] * scales_0[1];
                            _tmem_load_3[2] = _tmem_load_3[2] * scales_0[0];
                            _tmem_load_3[3] = _tmem_load_3[3] * scales_0[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(taddr + 88 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[3])));
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
                    int num_valid_rows_o = SEQ_LEN_Q - cta_x_c * tokens_per_cta_q_c;
                    if (num_valid_rows_o > tokens_per_cta_q_c) {
                        num_valid_rows_o = tokens_per_cta_q_c;
                    }
                    num_valid_rows_o = num_valid_rows_o * heads_per_kv_c;
                    int seq_offset_o = batch_c * SEQ_LEN_Q + cta_x_c * tokens_per_cta_q_c;
                    int heads_o = heads_per_kv_c;
                    if (heads_o > TILE_Q) {
                        heads_o = TILE_Q;
                    }
                    int cta_offset_o = seq_offset_o * (num_q_heads * HEAD_DIM) + head_c * heads_o * HEAD_DIM;
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
                    float _rcp_0 = approx_rcp(total_sum0);
                    float _rcp_1 = approx_rcp(total_sum1);
                    float final_scales_0[2];
                    float final_scales_1[2];
                    final_scales_0[0] = _rcp_0 * scales_i0[0];
                    final_scales_0[1] = _rcp_1 * scales_i0[1];
                    final_scales_1[0] = _rcp_0 * scales_i1[0];
                    final_scales_1[1] = _rcp_1 * scales_i1[1];
                    float _tmem_load_6[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3]))
                        : "r"(taddr + 80 + (unsigned int)corr_row));
                    float _tmem_load_7[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3]))
                        : "r"(taddr + 88 + (unsigned int)corr_row));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float merged_o[4];
                    merged_o[0] = __fmaf_rn(_tmem_load_6[0], final_scales_0[0], _tmem_load_7[0] * final_scales_1[0]);
                    merged_o[1] = __fmaf_rn(_tmem_load_6[1], final_scales_0[1], _tmem_load_7[1] * final_scales_1[1]);
                    merged_o[2] = __fmaf_rn(_tmem_load_6[2], final_scales_0[0], _tmem_load_7[2] * final_scales_1[0]);
                    merged_o[3] = __fmaf_rn(_tmem_load_6[3], final_scales_0[1], _tmem_load_7[3] * final_scales_1[1]);
                    unsigned int regs_o[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_o[_lp*2 + 0], merged_o[_lp*2+1 + 0]));
                        regs_o[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(smem_o) + stsm_offset_c)), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o[1]))
                        : "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    if (smem_row_c < num_valid_rows_o) {
                        unsigned int o_vec[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&o_vec[(0) + 3]))
                            : "r"(smem_o_addr + (unsigned int)load_smem_off_c));
                        uint32_t _fast_div_q_7 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)smem_row_c), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)smem_row_c);
                        uint32_t _fast_div_r_7 = (uint32_t)((unsigned int)smem_row_c) - _fast_div_q_7 * (uint32_t)(heads_per_kv.divisor);
                        int dst_row = (int)_fast_div_r_7 + (int)_fast_div_q_7 * num_q_heads;
                        int dst_words = cta_offset_o / 2 + dst_row * (HEAD_DIM / 2) + dst_col_words_c;
                        {
                            auto _gst_13 = o_words + dst_words;
                            __builtin_assume(__isGlobal(_gst_13));
                            reinterpret_cast<int4*>(_gst_13)[0] = reinterpret_cast<int4*>(o_vec)[0];
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
                mbarrier_wait(clc_full_addr + (ws_c) * 8, _phase_clc_full_1);
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
                    : "r"(work_response_addr + ws_c * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(clc_empty_addr + (ws_c) * 8);
                ws_c += 1;
                if (ws_c == 2) { ws_c = 0; _phase_clc_full_1 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
                cta_x_c = (int)_clc_ctaid_x_5;
                head_c = (int)_clc_ctaid_y_5;
                batch_c = (int)_clc_ctaid_z_5;
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
            unsigned int num_tiles_m = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)gridDim.z;
            uint32_t _fast_div_q_2 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_2 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_m = (int)_fast_div_q_2;
            int qm_stage = 0;
            int mk_stage = 0;
            int om_stage = 0;
            int ws_m = 0;
            int cta_x_m = blockIdx.x;
            int batch_m = blockIdx.z;
            unsigned int _phase_s_empty_0_0 = 1;
            mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
            _phase_s_empty_0_0 ^= 1;
            unsigned int _phase_s_empty_1_0 = 1;
            mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
            _phase_s_empty_1_0 ^= 1;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_o_empty = 1;
            unsigned int _phase_p_full_1_0 = 0;
            unsigned int _phase_clc_full_2 = 0;
            #pragma unroll 1
            while (true) {
                int seq_len_kv_m = seq_lens_kv[batch_m];
                int valid_kv_m = ((cta_x_m * tokens_per_cta_q_m + (seq_len_kv_m - SEQ_LEN_Q) + tokens_per_cta_q_m) < (seq_len_kv_m) ? (cta_x_m * tokens_per_cta_q_m + (seq_len_kv_m - SEQ_LEN_Q) + tokens_per_cta_q_m) : (seq_len_kv_m));
                int num_steps_m = 0;
                if (cta_x_m * tokens_per_cta_q_m < SEQ_LEN_Q && valid_kv_m > 0) {
                    num_steps_m = (valid_kv_m + (STEP_N - 1)) / STEP_N;
                }
                if (num_steps_m > 0) {
                    mbarrier_wait(q_full_addr + (qm_stage) * 8, _phase_q_full);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 64);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
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
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(s_full_0_addr);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 1024);
                    int mma_b_lo = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 64);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(s_full_1_addr);
                    #pragma unroll 1
                    for (int lo_m = 0; lo_m < num_steps_m - 1; lo_m++) {
                        int first_pv_m = ((lo_m == 0) ? 1 : 0);
                        mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                        _phase_s_empty_0_0 ^= 1;
                        mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 1024);
                        int mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 64);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (mk_stage) * 8);
                        mk_stage += 1;
                        if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                        elect_commit(s_full_0_addr);
                        mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                        _phase_p_full_0_0 ^= 1;
                        mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                        mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 1024);
                        int mma_b_lo_1 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                        {
                            uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_1);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, ((first_pv_m) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67273872, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (mk_stage) * 8);
                        mk_stage += 1;
                        if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                        elect_commit(o_full_addr + (om_stage) * 8);
                        om_stage += 1;
                        if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                        elect_commit(p_empty_0_addr);
                        mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                        _phase_s_empty_1_0 ^= 1;
                        mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mk_stage) * 1024);
                        int mma_b_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (qm_stage) * 64);
                        {
                            uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (mk_stage) * 8);
                        mk_stage += 1;
                        if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                        elect_commit(s_full_1_addr);
                        mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                        _phase_p_full_1_0 ^= 1;
                        mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                        mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 1024);
                        int mma_b_lo_3 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                        {
                            uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_3);
                            uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_3);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, ((first_pv_m) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 67273872, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (mk_stage) * 8);
                        mk_stage += 1;
                        if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
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
                    int mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 1024);
                    int mma_b_lo_4 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, ((last_first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 67273872, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(o_full_addr + (om_stage) * 8);
                    om_stage += 1;
                    if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                    elect_commit(p_empty_0_addr);
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    mbarrier_wait(o_empty_addr + (om_stage) * 8, _phase_o_empty);
                    mbarrier_wait(kv_full_addr + (mk_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mk_stage) * 1024);
                    int mma_b_lo_5 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_5);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, ((last_first_pv_m) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 67273872, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (mk_stage) * 8);
                    mk_stage += 1;
                    if (mk_stage == 9) { mk_stage = 0; _phase_kv_full ^= 1; }
                    elect_commit(o_full_addr + (om_stage) * 8);
                    om_stage += 1;
                    if (om_stage == 2) { om_stage = 0; _phase_o_empty ^= 1; }
                    elect_commit(p_empty_1_addr);
                    elect_commit(q_empty_addr + (qm_stage) * 8);
                    qm_stage += 1;
                    if (qm_stage == 2) { qm_stage = 0; _phase_q_full ^= 1; }
                }
                mbarrier_wait(clc_full_addr + (ws_m) * 8, _phase_clc_full_2);
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
                    : "r"(work_response_addr + ws_m * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(clc_empty_addr + (ws_m) * 8);
                ws_m += 1;
                if (ws_m == 2) { ws_m = 0; _phase_clc_full_2 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                cta_x_m = (int)_clc_ctaid_x_3;
                batch_m = (int)_clc_ctaid_z_3;
            }
            elect_commit(s_full_0_addr);
            elect_commit(s_full_1_addr);
        }
    }
    // ---- Role: load_pgoff ----
    if (warp == 13) {
        { // load_pgoff_main
            const int lane_p = lane;
            unsigned int num_tiles_p = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)gridDim.z;
            uint32_t _fast_div_q_0 = (heads_per_kv.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)TILE_Q), heads_per_kv.multiplier) >> heads_per_kv.shift_right) : (uint32_t)((unsigned int)TILE_Q);
            uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)TILE_Q) - _fast_div_q_0 * (uint32_t)(heads_per_kv.divisor);
            const int tokens_per_cta_q_p = (int)_fast_div_q_0;
            int pp_stage = 0;
            int ws_p = 0;
            int cta_x_p = blockIdx.x;
            int head_p = blockIdx.y;
            int batch_p = blockIdx.z;
            unsigned int _phase_page_empty = 1;
            unsigned int _phase_clc_full_3 = 0;
            #pragma unroll 1
            while (true) {
                int seq_len_kv_p = seq_lens_kv[batch_p];
                int valid_kv_p = ((cta_x_p * tokens_per_cta_q_p + (seq_len_kv_p - SEQ_LEN_Q) + tokens_per_cta_q_p) < (seq_len_kv_p) ? (cta_x_p * tokens_per_cta_q_p + (seq_len_kv_p - SEQ_LEN_Q) + tokens_per_cta_q_p) : (seq_len_kv_p));
                int num_steps_p = 0;
                if (cta_x_p * tokens_per_cta_q_p < SEQ_LEN_Q && valid_kv_p > 0) {
                    num_steps_p = (valid_kv_p + (STEP_N - 1)) / STEP_N;
                }
                int num_loop_steps_p = num_steps_p * 2;
                int pt_base_p = batch_p * max_pages_per_seq;
                int page_idx_ub_p = (seq_len_kv_p + (PAGE_SIZE - 1)) / PAGE_SIZE - 1;
                #pragma unroll 1
                for (int lo_p = 0; lo_p < num_loop_steps_p; lo_p += 4) {
                    int page_idx_p = ((lo_p * 8 + lane_p) < (page_idx_ub_p) ? (lo_p * 8 + lane_p) : (page_idx_ub_p));
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
                mbarrier_wait(clc_full_addr + (ws_p) * 8, _phase_clc_full_3);
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
                    : "r"(work_response_addr + ws_p * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(clc_empty_addr + (ws_p) * 8);
                ws_p += 1;
                if (ws_p == 2) { ws_p = 0; _phase_clc_full_3 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                cta_x_p = (int)_clc_ctaid_x_0;
                head_p = (int)_clc_ctaid_y_0;
                batch_p = (int)_clc_ctaid_z_0;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        { // scheduler_main
            unsigned int num_tiles_s = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)gridDim.z;
            int th_stage_s = 0;
            int ws_s = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_clc_empty = 1;
            unsigned int _phase_clc_full_4 = 0;
            #pragma unroll 1
            while (true) {
                mbarrier_wait(throttle_full_addr + (th_stage_s) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (th_stage_s) * 8);
                th_stage_s += 1;
                if (th_stage_s == 2) { th_stage_s = 0; _phase_throttle_full ^= 1; }
                mbarrier_wait(clc_empty_addr + (ws_s) * 8, _phase_clc_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(clc_full_addr + (ws_s) * 8, 16);
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + ws_s * 16 + 0 * 16), "r"(clc_full_addr + ws_s * 8)
                        : "memory");
                }
                mbarrier_wait(clc_full_addr + (ws_s) * 8, _phase_clc_full_4);
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
                    : "r"(work_response_addr + ws_s * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(clc_empty_addr + (ws_s) * 8);
                ws_s += 1;
                if (ws_s == 2) { ws_s = 0; _phase_clc_empty ^= 1; _phase_clc_full_4 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            unsigned int num_tiles_l = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)gridDim.z;
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
            int th_stage_l = 0;
            int ws_l = 0;
            int cta_x_l = blockIdx.x;
            int head_l = blockIdx.y;
            int batch_l = blockIdx.z;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_clc_full_5 = 0;
            #pragma unroll 1
            while (true) {
                int seq_len_kv_l = seq_lens_kv[batch_l];
                int valid_kv_l = ((cta_x_l * tokens_per_cta_q_l + (seq_len_kv_l - SEQ_LEN_Q) + tokens_per_cta_q_l) < (seq_len_kv_l) ? (cta_x_l * tokens_per_cta_q_l + (seq_len_kv_l - SEQ_LEN_Q) + tokens_per_cta_q_l) : (seq_len_kv_l));
                int num_steps_l = 0;
                if (cta_x_l * tokens_per_cta_q_l < SEQ_LEN_Q && valid_kv_l > 0) {
                    num_steps_l = (valid_kv_l + (STEP_N - 1)) / STEP_N;
                }
                int num_loop_steps_l = num_steps_l * 2;
                mbarrier_wait(throttle_empty_addr + (th_stage_l) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (th_stage_l) * 8);
                th_stage_l += 1;
                if (th_stage_l == 2) { th_stage_l = 0; _phase_throttle_empty ^= 1; }
                if (num_loop_steps_l > 0) {
                    mbarrier_wait(q_empty_addr + (q_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 1024);
                    }
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                    if (elect_sync()) {
                        int q_row_l = batch_l * num_q_heads + head_l * heads_per_kv_l;
                        tma_3d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 1024), (&Q), 0, q_row_l, 0, q_full_addr + (q_stage) * 8);
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; _phase_q_empty ^= 1; }
                    mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                    k_sid = pc_stage;
                    pc_stage += 1;
                    if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                    }
                    int ids_lo[4];
                    int ids_hi[4];
                    int id_byte_base = k_sid * 128;
                    int kv_dst = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)id_byte_base));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 4; pg_i++) {
                            int pool_page_lo = ids_lo[pg_i] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst + pg_i * 2048, (&K), 0, 0, 0, pool_page_lo, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(id_byte_base + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 4; pg_i_1++) {
                            int pool_page_hi = ids_hi[pg_i_1] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst + 8192 + pg_i_1 * 2048, (&K), 0, 0, 0, pool_page_hi, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 9) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                    }
                    int ids_lo_0[4];
                    int ids_hi_0[4];
                    int id_byte_base_0 = k_sid * 128 + 32;
                    int kv_dst_0 = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_0[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)id_byte_base_0));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_2 = 0; pg_i_2 < 4; pg_i_2++) {
                            int pool_page_lo_1 = ids_lo_0[pg_i_2] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst_0 + pg_i_2 * 2048, (&K), 0, 0, 0, pool_page_lo_1, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_0[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(id_byte_base_0 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_3 = 0; pg_i_3 < 4; pg_i_3++) {
                            int pool_page_hi_1 = ids_hi_0[pg_i_3] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst_0 + 8192 + pg_i_3 * 2048, (&K), 0, 0, 0, pool_page_hi_1, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 9) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    #pragma unroll 1
                    for (int lo_l = 0; lo_l < num_loop_steps_l - 2; lo_l++) {
                        int is_last_iter_l = ((lo_l + 1 >= num_loop_steps_l - 2) ? 1 : 0);
                        if ((lo_l + 2) % 4 == 0) {
                            mbarrier_wait(page_full_addr + (pc_stage) * 8, _phase_page_full);
                            k_sid = pc_stage;
                            pc_stage += 1;
                            if (pc_stage == 6) { pc_stage = 0; _phase_page_full ^= 1; }
                        }
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                        }
                        int ids_lo_1[4];
                        int ids_hi_1[4];
                        int id_byte_base_1 = k_sid * 128 + (lo_l + 2) % 4 * 32;
                        int kv_dst_1 = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_1[(0) + 3]))
                            : "r"(smem_page_ids_addr + (unsigned int)id_byte_base_1));
                        if (elect_sync()) {
                            #pragma unroll
                            for (int pg_i_4 = 0; pg_i_4 < 4; pg_i_4++) {
                                int pool_page_lo_2 = ids_lo_1[pg_i_4] * num_kv_heads + head_l;
                                tma_4d_gmem2smem(kv_dst_1 + pg_i_4 * 2048, (&K), 0, 0, 0, pool_page_lo_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_1[(0) + 3]))
                            : "r"(smem_page_ids_addr + (unsigned int)(id_byte_base_1 + 16)));
                        if (elect_sync()) {
                            #pragma unroll
                            for (int pg_i_5 = 0; pg_i_5 < 4; pg_i_5++) {
                                int pool_page_hi_2 = ids_hi_1[pg_i_5] * num_kv_heads + head_l;
                                tma_4d_gmem2smem(kv_dst_1 + 8192 + pg_i_5 * 2048, (&K), 0, 0, 0, pool_page_hi_2, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 9) { kv_stage = 0; _phase_kv_empty ^= 1; }
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
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                        }
                        int ids_lo_2[4];
                        int ids_hi_2[4];
                        int id_byte_base_2 = v_sid * 128 + lo_l % 4 * 32;
                        int kv_dst_2 = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_2[(0) + 3]))
                            : "r"(smem_page_ids_addr + (unsigned int)id_byte_base_2));
                        if (elect_sync()) {
                            #pragma unroll
                            for (int pg_i_6 = 0; pg_i_6 < 4; pg_i_6++) {
                                int pool_page_lo_3 = ids_lo_2[pg_i_6] * num_kv_heads + head_l;
                                tma_4d_gmem2smem(kv_dst_2 + pg_i_6 * 2048, (&V), 0, 0, 0, pool_page_lo_3, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_2[(0) + 3]))
                            : "r"(smem_page_ids_addr + (unsigned int)(id_byte_base_2 + 16)));
                        if (elect_sync()) {
                            #pragma unroll
                            for (int pg_i_7 = 0; pg_i_7 < 4; pg_i_7++) {
                                int pool_page_hi_3 = ids_hi_2[pg_i_7] * num_kv_heads + head_l;
                                tma_4d_gmem2smem(kv_dst_2 + 8192 + pg_i_7 * 2048, (&V), 0, 0, 0, pool_page_hi_3, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 9) { kv_stage = 0; _phase_kv_empty ^= 1; }
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
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                    }
                    int ids_lo_3[4];
                    int ids_hi_3[4];
                    int id_byte_base_3 = v_sid * 128 + last_l % 4 * 32;
                    int kv_dst_3 = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_3[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)id_byte_base_3));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_8 = 0; pg_i_8 < 4; pg_i_8++) {
                            int pool_page_lo_4 = ids_lo_3[pg_i_8] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst_3 + pg_i_8 * 2048, (&V), 0, 0, 0, pool_page_lo_4, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_3[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(id_byte_base_3 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_9 = 0; pg_i_9 < 4; pg_i_9++) {
                            int pool_page_hi_4 = ids_hi_3[pg_i_9] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst_3 + 8192 + pg_i_9 * 2048, (&V), 0, 0, 0, pool_page_hi_4, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 9) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                    }
                    int ids_lo_4[4];
                    int ids_hi_4[4];
                    int id_byte_base_4 = v_sid * 128 + (last_l + 1) % 4 * 32;
                    int kv_dst_4 = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_lo_4[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)id_byte_base_4));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_10 = 0; pg_i_10 < 4; pg_i_10++) {
                            int pool_page_lo_5 = ids_lo_4[pg_i_10] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst_4 + pg_i_10 * 2048, (&V), 0, 0, 0, pool_page_lo_5, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ids_hi_4[(0) + 3]))
                        : "r"(smem_page_ids_addr + (unsigned int)(id_byte_base_4 + 16)));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int pg_i_11 = 0; pg_i_11 < 4; pg_i_11++) {
                            int pool_page_hi_5 = ids_hi_4[pg_i_11] * num_kv_heads + head_l;
                            tma_4d_gmem2smem(kv_dst_4 + 8192 + pg_i_11 * 2048, (&V), 0, 0, 0, pool_page_hi_5, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 9) { kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_arrive(page_empty_addr + (pr_stage) * 8);
                    pr_stage += 1;
                    if (pr_stage == 6) { pr_stage = 0; }
                }
                mbarrier_wait(clc_full_addr + (ws_l) * 8, _phase_clc_full_5);
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
                    : "r"(work_response_addr + ws_l * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(clc_empty_addr + (ws_l) * 8);
                ws_l += 1;
                if (ws_l == 2) { ws_l = 0; _phase_clc_full_5 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                cta_x_l = (int)_clc_ctaid_x_2;
                head_l = (int)_clc_ctaid_y_2;
                batch_l = (int)_clc_ctaid_z_2;
            }
        }
    }

    // Cleanup
}

} // extern "C"
