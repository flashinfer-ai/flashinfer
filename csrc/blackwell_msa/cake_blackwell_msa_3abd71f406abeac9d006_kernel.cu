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
#define TMEM_NCOLS 256
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_OFFSET 128
#define NUM_KV_RING_STAGES 11
#define NUM_S_RING_STAGES 4
#define NUM_P_RING_STAGES 4
#define NUM_ITEM_RING_STAGES 2
#define SMEM_SMEM_EXCH_MAX_OFF 1024
#define SMEM_SMEM_EXCH_MAX_STAGE_BYTES 512
#define SMEM_SMEM_EXCH_MAX_STRIDE 512
#define SMEM_SMEM_EXCH_MAX_HI_OFF 2304
#define SMEM_SMEM_EXCH_MAX_HI_STAGE_BYTES 512
#define SMEM_SMEM_EXCH_MAX_HI_STRIDE 512
#define SMEM_SMEM_PAGE_OFF 1664
#define SMEM_SMEM_PAGE_STAGE_BYTES 128
#define SMEM_SMEM_PAGE_STRIDE 128
#define SMEM_SMEM_VALID_OFF 1792
#define SMEM_SMEM_VALID_STAGE_BYTES 128
#define SMEM_SMEM_VALID_STRIDE 128
#define SMEM_SMEM_NPAIRS_OFF 1920
#define SMEM_SMEM_NPAIRS_STAGE_BYTES 8
#define SMEM_SMEM_NPAIRS_STRIDE 8
#define SMEM_SMEM_SPLIT_FLAG_OFF 1936
#define SMEM_SMEM_SPLIT_FLAG_STAGE_BYTES 4
#define SMEM_SMEM_SPLIT_FLAG_STRIDE 4
#define SMEM_SMEM_EXCH_SUM_OFF 2048
#define SMEM_SMEM_EXCH_SUM_STAGE_BYTES 256
#define SMEM_SMEM_EXCH_SUM_STRIDE 256
#if __CUDA_ARCH__ == 1030
#define SMEM_SMEM_ISSUE_OFF 2816
#define SMEM_SMEM_ISSUE_STAGE_BYTES 16
#define SMEM_SMEM_ISSUE_STRIDE 16
#else
#endif
#define SMEM_SMEM_QT_OFF 3072
#define SMEM_SMEM_QT_STAGE_BYTES 2048
#define SMEM_SMEM_QT_STRIDE 2048
#define SMEM_SMEM_P_OFF 5120
#define SMEM_SMEM_P_STAGE_BYTES 4096
#define SMEM_SMEM_P_STRIDE 4096
#define SMEM_SMEM_KV_OFF 37888
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 37888
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_TOTAL 218112
#define THREADS 384
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
#if __CUDA_ARCH__ == 1030
#else
}


__device__ __forceinline__ uint32_t mbarrier_test_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
#endif
}


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_blackwell_msa_3abd71f406abeac9d006(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages)
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
    #define q_empty_addr (mbar_base + 8)
    #define meta_full_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 120)
    #define s_full_addr (mbar_base + 208)
    #define s_empty_addr (mbar_base + 240)
    #define p_full_addr (mbar_base + 272)
    #define pv_done_addr (mbar_base + 304)
    #define decode_done_addr (mbar_base + 368)
    #define meta_empty_addr (mbar_base + 376)
    #define tmem_dead_addr (mbar_base + 392)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_exch_max = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_exch_max_addr = smem + 1024;
    float* smem_exch_max_hi = reinterpret_cast<float*>(smem_raw + 2304);
    const int smem_exch_max_hi_addr = smem + 2304;
    int* smem_page = reinterpret_cast<int*>(smem_raw + 1664);
    const int smem_page_addr = smem + 1664;
    int* smem_valid = reinterpret_cast<int*>(smem_raw + 1792);
    const int smem_valid_addr = smem + 1792;
    int* smem_npairs = reinterpret_cast<int*>(smem_raw + 1920);
    const int smem_npairs_addr = smem + 1920;
    int* smem_split_flag = reinterpret_cast<int*>(smem_raw + 1936);
    const int smem_split_flag_addr = smem + 1936;
    float* smem_exch_sum = reinterpret_cast<float*>(smem_raw + 2048);
    const int smem_exch_sum_addr = smem + 2048;
#if __CUDA_ARCH__ == 1030
    unsigned int* smem_issue = reinterpret_cast<unsigned int*>(smem_raw + 2816);
    const int smem_issue_addr = smem + 2816;
#else
#endif
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_qt_addr = smem + 3072;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_p_addr = smem + 5120;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_kv_addr = smem + 37888;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v_addr = smem + 37888;
#if __CUDA_ARCH__ == 1030
    #pragma unroll
    for (int word = 0; word < 3; word++) {
        asm volatile("st.relaxed.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + (unsigned int)(4 * word))), "r"(static_cast<uint32_t>(0)) : "memory");
    }
#else
#endif

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 50 barriers)
    // Mbarriers at smem_raw[0..400)

    if (warp == 0) {
        // q_full: 1 barriers, init_count=1
        // q_empty: 1 barriers, init_count=1
        // --- pipeline 'item_ring' ---
        // meta_full: 2 barriers, init_count=1
        // --- pipeline 'kv_ring' ---
        // kv_full: 11 barriers, init_count=1
        // kv_empty: 11 barriers, init_count=1
        // --- pipeline 's_ring' ---
        // s_full: 4 barriers, init_count=1
        // s_empty: 4 barriers, init_count=8
        // --- pipeline 'p_ring' ---
        // p_full: 4 barriers, init_count=8
        // pv_done: 8 barriers, init_count=1
        // decode_done: 1 barriers, init_count=8
        // --- pipeline 'item_ring' ---
        // meta_empty: 2 barriers, init_count=8
        // tmem_dead: 1 barriers, init_count=8
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 8;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 8;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(14), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(8)));
        if (lane < 18) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 400);
    if (warp == 0) {
        int _tmem_hold = smem + 400;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o = taddr + 128;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax_lo ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // softmax_lo_main
            float tau = 8.0f;
            unsigned int total_items = total_q * num_kv_heads;
            const int warp_in_role = warp % 4;
            const int token = warp_in_role * 32 + lane;
            const int tmem_lane_base = warp % 4 * 32 << 16;
            const int own_half = warp_in_role / 2;
            const int lane_stream = lane / 8;
            const int lane_head = lane % 8;
            const unsigned int byte_mask = 255;
            int item_stage = 0;
            int s_stage = 0;
            int p_stage = 0;
            int item_ord = 0;
            unsigned int _phase_meta_full = 0;
            unsigned int _phase_s_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx = blockIdx.x; work_idx < total_items; work_idx += gridDim.x) {
                int item = work_idx;
                int split = 0;
                int query = item / num_kv_heads;
                int kv_head = item % num_kv_heads;
                int group_size = num_q_heads / num_kv_heads;
                mbarrier_wait(meta_full_addr + (item_stage) * 8, _phase_meta_full);
                int npairs = smem_npairs[item_stage];
                int meta_base = item_stage * 16 + 16 * split;
                item_stage += 1;
                if (item_stage == 2) { item_stage = 0; _phase_meta_full ^= 1; }
                float o_lane = -CUDART_INF_F;
                float ms_lane = 0.0f;
                float lane_sum[16];
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    lane_sum[c] = 0.0f;
                }
                #pragma unroll 1
                for (int pair = 0; pair < 8; pair++) {
                    if (npairs <= pair) {
                        break;
                    }
                    mbarrier_wait(s_full_addr + (s_stage) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)(s_stage * 32) + (unsigned int)tmem_lane_base;
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], s_addr);
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], s_addr + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage) * 8);
                    }
                    int valid0 = smem_valid[meta_base + pair * 2];
                    int valid1 = smem_valid[meta_base + pair * 2 + 1];
                    if (token >= valid0) {
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 8; c_1++) {
                            _tmem_load_0[c_1] = -CUDART_INF_F;
                        }
                    }
                    if (token >= valid1) {
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 8; c_2++) {
                            _tmem_load_1[c_2] = -CUDART_INF_F;
                        }
                    }
                    float lane_hold = -CUDART_INF_F;
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 8; c_3++) {
                        float t0_max_c;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(t0_max_c) : "f"(_tmem_load_0[c_3]));
                        float t1_max_c;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(t1_max_c) : "f"(_tmem_load_1[c_3]));
                        if (lane == c_3) {
                            lane_hold = t0_max_c;
                        }
                        if (lane == 8 + c_3) {
                            lane_hold = t1_max_c;
                        }
                    }
                    int exch_slot = pair % 2 * 64;
                    if (lane < 16) {
                        smem_exch_max[exch_slot + lane / 8 * 32 + warp_in_role * 8 + lane % 8] = lane_hold;
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int m_base = exch_slot + lane_stream / 2 * 32 + lane_stream % 2 * 16 + lane_head;
                    float m_lo = smem_exch_max[m_base];
                    float m_hi = smem_exch_max[m_base + 8];
                    float m_lane;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_lane) : "f"(m_lo), "f"(m_hi));
                    float cand_lane;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_lane) : "f"(o_lane), "f"(m_lane));
                    int grow_lane = 0;
                    if (tau < (cand_lane - o_lane) * softmax_scale_log2) {
                        grow_lane = 1;
                    }
                    unsigned int grow_mask = __ballot_sync(0xFFFFFFFF, grow_lane != 0);
                    unsigned int stream_bits = grow_mask >> (unsigned int)(lane_stream * 8) & byte_mask;
                    int grow = 0;
                    if (grow_mask != 0) {
                        grow = 1;
                    }
                    if (grow != 0) {
                        float alpha_lane = 1.0f;
                        if (stream_bits != 0) {
                            if (o_lane > -CUDART_INF_F) {
                                float _exp2_0 = approx_exp2(softmax_scale_log2 * (o_lane - cand_lane));
                                alpha_lane = _exp2_0;
                            }
                            o_lane = cand_lane;
                            float safe_lane = ((o_lane == -CUDART_INF_F) ? 0.0f : o_lane);
                            ms_lane = safe_lane * softmax_scale_log2;
                        }
                        #pragma unroll
                        for (int c_4 = 0; c_4 < 8; c_4++) {
                            float _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, alpha_lane, own_half * 8 + c_4);
                            lane_sum[c_4] = lane_sum[c_4] * _shfl_sync_1;
                        }
                        #pragma unroll
                        for (int c_5 = 0; c_5 < 8; c_5++) {
                            float _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, alpha_lane, (2 + own_half) * 8 + c_5);
                            lane_sum[8 + c_5] = lane_sum[8 + c_5] * _shfl_sync_2;
                        }
                        if (pair > 0) {
                            int _parity_pv_done = (int)((unsigned int)item_ord % 2);
                            mbarrier_wait(pv_done_addr + (pair - 1) * 8, _parity_pv_done);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if ((grow_mask & byte_mask) != 0) {
                                int o_addr = taddr + 128 + (unsigned int)tmem_lane_base;
                                float _tmem_load_2[8];
                                tmem_ld_x8(&_tmem_load_2[0], o_addr);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_6 = 0; c_6 < 8; c_6++) {
                                    float _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, alpha_lane, c_6);
                                    _tmem_load_2[c_6] = _tmem_load_2[c_6] * _shfl_sync_3;
                                }
                                tmem_st_x8_f32(o_addr, _tmem_load_2);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if ((grow_mask >> 8 & byte_mask) != 0) {
                                int o_addr_0 = taddr + 128 + 32 + (unsigned int)tmem_lane_base;
                                float tmem_load_0[8];
                                tmem_ld_x8(&tmem_load_0[0], o_addr_0);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_0 = 0; c_0 < 8; c_0++) {
                                    float _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, alpha_lane, 8 + c_0);
                                    tmem_load_0[c_0] = tmem_load_0[c_0] * _shfl_sync_4;
                                }
                                tmem_st_x8_f32(o_addr_0, tmem_load_0);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if ((grow_mask >> 16 & byte_mask) != 0) {
                                int o_addr_1 = taddr + 128 + 64 + (unsigned int)tmem_lane_base;
                                float tmem_load_1[8];
                                tmem_ld_x8(&tmem_load_1[0], o_addr_1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_1_1 = 0; c_1_1 < 8; c_1_1++) {
                                    float _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, alpha_lane, 16 + c_1_1);
                                    tmem_load_1[c_1_1] = tmem_load_1[c_1_1] * _shfl_sync_5;
                                }
                                tmem_st_x8_f32(o_addr_1, tmem_load_1);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if ((grow_mask >> 24 & byte_mask) != 0) {
                                int o_addr_2 = taddr + 128 + 96 + (unsigned int)tmem_lane_base;
                                float tmem_load_2[8];
                                tmem_ld_x8(&tmem_load_2[0], o_addr_2);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_2_1 = 0; c_2_1 < 8; c_2_1++) {
                                    float _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, alpha_lane, 24 + c_2_1);
                                    tmem_load_2[c_2_1] = tmem_load_2[c_2_1] * _shfl_sync_6;
                                }
                                tmem_st_x8_f32(o_addr_2, tmem_load_2);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                    }
                    if (pair >= 4) {
                        int _parity_pv_done_1 = (int)((unsigned int)item_ord % 2);
                        mbarrier_wait(pv_done_addr + (pair - 4) * 8, _parity_pv_done_1);
                    }
                    #pragma unroll
                    for (int c_7 = 0; c_7 < 8; c_7++) {
                        float max_scaled_1_0 = __shfl_sync(0xFFFFFFFF, ms_lane, own_half * 8 + c_7);
                        float _exp2_7 = approx_exp2(__fmaf_rn(_tmem_load_0[c_7], softmax_scale_log2, -max_scaled_1_0));
                        _tmem_load_0[c_7] = _exp2_7;
                        lane_sum[c_7] = lane_sum[c_7] + _tmem_load_0[c_7];
                        float max_scaled_1_1 = __shfl_sync(0xFFFFFFFF, ms_lane, (2 + own_half) * 8 + c_7);
                        float _exp2_8 = approx_exp2(__fmaf_rn(_tmem_load_1[c_7], softmax_scale_log2, -max_scaled_1_1));
                        _tmem_load_1[c_7] = _exp2_8;
                        lane_sum[8 + c_7] = lane_sum[8 + c_7] + _tmem_load_1[c_7];
                    }
                    int p0_base = smem_p_addr + (unsigned int)(p_stage * 2 * 4096);
                    int p1_base = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 4096);
                    unsigned int w0[2];
                    unsigned int w1[2];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]),
                                               "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]));
                        w0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]),
                                               "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]));
                        w0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]),
                                               "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]));
                        w1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]),
                                               "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]));
                        w1[1] = _packed;
                    }
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((p0_base + (token * 32 ^ (token * 32 >> 7 & 1) << 4))), "r"(w0[0]), "r"(w0[1]) : "memory");
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((p1_base + (token * 32 ^ (token * 32 >> 7 & 1) << 4))), "r"(w1[0]), "r"(w1[1]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage) * 8);
                    }
                    s_stage += 1;
                    if (s_stage == 4) { s_stage = 0; _phase_s_full ^= 1; }
                    p_stage += 1;
                    if (p_stage == 4) { p_stage = 0; }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)item_ord % 2)) * 8);
                }
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, o_lane, 8);
                float m_fin;
                asm("max.f32 %0, %1, %2;" : "=f"(m_fin) : "f"(o_lane), "f"(_shfl_xor_sync_9));
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, m_fin, 16);
                m_fin = max_noftz(m_fin, _shfl_xor_sync_10);
                float _exp2_11 = approx_exp2((o_lane - m_fin) * softmax_scale_log2);
                float w_lane = ((o_lane == -CUDART_INF_F) ? 0.0f : _exp2_11);
                float lane_sum_w[8];
                #pragma unroll
                for (int c_8 = 0; c_8 < 8; c_8++) {
                    float wt0 = __shfl_sync(0xFFFFFFFF, w_lane, own_half * 8 + c_8);
                    float wt1 = __shfl_sync(0xFFFFFFFF, w_lane, (2 + own_half) * 8 + c_8);
                    lane_sum_w[c_8] = __fmaf_rn(lane_sum[c_8], wt0, lane_sum[8 + c_8] * wt1);
                }
                int b4 = lane / 16 % 2;
                int b3 = lane / 8 % 2;
                int b2 = lane / 4 % 2;
                float r4[4];
                #pragma unroll
                for (int c_9 = 0; c_9 < 4; c_9++) {
                    float snd4 = ((b4 == 0) ? lane_sum_w[c_9 + 4] : lane_sum_w[c_9]);
                    float kp4 = ((b4 == 0) ? lane_sum_w[c_9] : lane_sum_w[c_9 + 4]);
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, snd4, 16);
                    r4[c_9] = kp4 + _shfl_xor_sync_12;
                }
                float r2[2];
                #pragma unroll
                for (int c_10 = 0; c_10 < 2; c_10++) {
                    float snd2 = ((b3 == 0) ? r4[c_10 + 2] : r4[c_10]);
                    float kp2 = ((b3 == 0) ? r4[c_10] : r4[c_10 + 2]);
                    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, snd2, 8);
                    r2[c_10] = kp2 + _shfl_xor_sync_13;
                }
                float snd1 = ((b2 == 0) ? r2[1] : r2[0]);
                float kp1 = ((b2 == 0) ? r2[0] : r2[1]);
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, snd1, 4);
                float r1 = kp1 + _shfl_xor_sync_14;
                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, r1, 2);
                r1 = r1 + _shfl_xor_sync_15;
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, r1, 1);
                r1 = r1 + _shfl_xor_sync_16;
                float xsum = r1;
                if (lane % 4 == 0) {
                    smem_exch_sum[warp_in_role * 8 + lane / 4] = xsum;
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float head_sum = 0.0f;
                if (lane < 8) {
                    float s0e = smem_exch_sum[lane];
                    float s1e = smem_exch_sum[8 + lane];
                    float s2e = smem_exch_sum[16 + lane];
                    float s3e = smem_exch_sum[24 + lane];
                    head_sum = s0e + s1e + s2e + s3e;
                }
                if (npairs > 0) {
                    int _parity_pv_done_2 = (int)((unsigned int)item_ord % 2);
                    mbarrier_wait(pv_done_addr + (npairs - 1) * 8, _parity_pv_done_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                } else {
                    int _parity_pv_done_3 = (int)((unsigned int)item_ord % 2);
                    mbarrier_wait(pv_done_addr, _parity_pv_done_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_epi[8];
                #pragma unroll
                for (int h = 0; h < 8; h++) {
                    o_epi[h] = 0.0f;
                }
                if (npairs > 0) {
                    float _tmem_load_6[8];
                    tmem_ld_x8(&_tmem_load_6[0], taddr + 128 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_1 = 0; h_1 < 8; h_1++) {
                        float _shfl_sync_17 = __shfl_sync(0xFFFFFFFF, w_lane, h_1);
                        o_epi[h_1] = __fmaf_rn(_tmem_load_6[h_1], _shfl_sync_17, o_epi[h_1]);
                    }
                    float _tmem_load_7[8];
                    tmem_ld_x8(&_tmem_load_7[0], taddr + 128 + 32 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_2 = 0; h_2 < 8; h_2++) {
                        float _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, w_lane, 8 + h_2);
                        o_epi[h_2] = __fmaf_rn(_tmem_load_7[h_2], _shfl_sync_18, o_epi[h_2]);
                    }
                    float _tmem_load_8[8];
                    tmem_ld_x8(&_tmem_load_8[0], taddr + 128 + 64 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_3 = 0; h_3 < 8; h_3++) {
                        float _shfl_sync_19 = __shfl_sync(0xFFFFFFFF, w_lane, 16 + h_3);
                        o_epi[h_3] = __fmaf_rn(_tmem_load_8[h_3], _shfl_sync_19, o_epi[h_3]);
                    }
                    float _tmem_load_9[8];
                    tmem_ld_x8(&_tmem_load_9[0], taddr + 128 + 96 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_4 = 0; h_4 < 8; h_4++) {
                        float _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, w_lane, 24 + h_4);
                        o_epi[h_4] = __fmaf_rn(_tmem_load_9[h_4], _shfl_sync_20, o_epi[h_4]);
                    }
                }
                if (total_items <= work_idx + (unsigned int)gridDim.x) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                int q_row0 = query * num_q_heads + kv_head * group_size;
                float _rcp_21 = approx_rcp(head_sum);
                float inv_lane = ((head_sum > 0.0f) ? output_scale * _rcp_21 : 0.0f);
                if (warp_in_role == 0) {
                    if (lane < 8) {
                        if (group_size > lane) {
                            float m_lane_1 = m_fin;
                            float lse_lane = -CUDART_INF_F;
                            if (head_sum > 0.0f) {
                                float _log2_22;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22) : "f"(head_sum));
                                lse_lane = __fmaf_rn(m_lane_1 * softmax_scale_log2, 0.6931472f, _log2_22 * 0.6931472f);
                            }
                            *(reinterpret_cast<float*>(msa_lse + (q_row0 + lane)) + (0)) = lse_lane;
                        }
                    }
                }
                float inv_all[8];
                #pragma unroll
                for (int h_5 = 0; h_5 < 8; h_5++) {
                    float _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, inv_lane, h_5);
                    inv_all[h_5] = _shfl_sync_23;
                }
                #pragma unroll
                for (int h_6 = 0; h_6 < 8; h_6++) {
                    if (group_size > h_6) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + h_6) * 128 + token)) + (0)) = (__nv_bfloat16)(o_epi[h_6] * inv_all[h_6]);
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(decode_done_addr);
                }
                item_ord += 1;
            }
        }
    }
    // ---- Role: softmax_hi ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // softmax_hi_main
            float tau_1 = 8.0f;
            unsigned int total_items_1 = total_q * num_kv_heads;
            const int warp_in_role_1 = warp % 4;
            const int token_1 = warp_in_role_1 * 32 + lane;
            const int tmem_lane_base_1 = warp % 4 * 32 << 16;
            const int own_half_1 = warp_in_role_1 / 2;
            const int lane_stream_1 = lane / 8;
            const int lane_head_1 = lane % 8;
            const unsigned int byte_mask_1 = 255;
            int item_stage_1 = 0;
            int s_stage_1 = 0;
            int p_stage_1 = 0;
            int item_ord_1 = 0;
            unsigned int _phase_meta_full_1 = 0;
            unsigned int _phase_s_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = blockIdx.x; work_idx_1 < total_items_1; work_idx_1 += gridDim.x) {
                int item_1 = work_idx_1;
                int split_1 = 0;
                int query_1 = item_1 / num_kv_heads;
                int kv_head_1 = item_1 % num_kv_heads;
                int group_size_1 = num_q_heads / num_kv_heads;
                mbarrier_wait(meta_full_addr + (item_stage_1) * 8, _phase_meta_full_1);
                int npairs_1 = smem_npairs[item_stage_1];
                int meta_base_1 = item_stage_1 * 16 + 16 * split_1;
                item_stage_1 += 1;
                if (item_stage_1 == 2) { item_stage_1 = 0; _phase_meta_full_1 ^= 1; }
                float o_lane_1 = -CUDART_INF_F;
                float ms_lane_1 = 0.0f;
                float lane_sum_1[16];
                #pragma unroll
                for (int c_11 = 0; c_11 < 16; c_11++) {
                    lane_sum_1[c_11] = 0.0f;
                }
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                    if (npairs_1 <= pair_1) {
                        break;
                    }
                    mbarrier_wait(s_full_addr + (s_stage_1) * 8, _phase_s_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr_1 = taddr + (unsigned int)(s_stage_1 * 32) + 8 + (unsigned int)tmem_lane_base_1;
                    float _tmem_load_10[8];
                    tmem_ld_x8(&_tmem_load_10[0], s_addr_1);
                    float _tmem_load_11[8];
                    tmem_ld_x8(&_tmem_load_11[0], s_addr_1 + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage_1) * 8);
                    }
                    int valid0_1 = smem_valid[meta_base_1 + pair_1 * 2];
                    int valid1_1 = smem_valid[meta_base_1 + pair_1 * 2 + 1];
                    if (token_1 >= valid0_1) {
                        #pragma unroll
                        for (int c_12 = 0; c_12 < 8; c_12++) {
                            _tmem_load_10[c_12] = -CUDART_INF_F;
                        }
                    }
                    if (token_1 >= valid1_1) {
                        #pragma unroll
                        for (int c_13 = 0; c_13 < 8; c_13++) {
                            _tmem_load_11[c_13] = -CUDART_INF_F;
                        }
                    }
                    float lane_hold_1 = -CUDART_INF_F;
                    #pragma unroll
                    for (int c_14 = 0; c_14 < 8; c_14++) {
                        float t0_max_c_1;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(t0_max_c_1) : "f"(_tmem_load_10[c_14]));
                        float t1_max_c_1;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(t1_max_c_1) : "f"(_tmem_load_11[c_14]));
                        if (lane == c_14) {
                            lane_hold_1 = t0_max_c_1;
                        }
                        if (lane == 8 + c_14) {
                            lane_hold_1 = t1_max_c_1;
                        }
                    }
                    int exch_slot_1 = pair_1 % 2 * 64;
                    if (lane < 16) {
                        smem_exch_max_hi[exch_slot_1 + lane / 8 * 32 + warp_in_role_1 * 8 + lane % 8] = lane_hold_1;
                    }
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    int m_base_1 = exch_slot_1 + lane_stream_1 / 2 * 32 + lane_stream_1 % 2 * 16 + lane_head_1;
                    float m_lo_1 = smem_exch_max_hi[m_base_1];
                    float m_hi_1 = smem_exch_max_hi[m_base_1 + 8];
                    float m_lane_2;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_lane_2) : "f"(m_lo_1), "f"(m_hi_1));
                    float cand_lane_1;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_lane_1) : "f"(o_lane_1), "f"(m_lane_2));
                    int grow_lane_1 = 0;
                    if (tau_1 < (cand_lane_1 - o_lane_1) * softmax_scale_log2) {
                        grow_lane_1 = 1;
                    }
                    unsigned int grow_mask_1 = __ballot_sync(0xFFFFFFFF, grow_lane_1 != 0);
                    unsigned int stream_bits_1 = grow_mask_1 >> (unsigned int)(lane_stream_1 * 8) & byte_mask_1;
                    int grow_1 = 0;
                    if (grow_mask_1 != 0) {
                        grow_1 = 1;
                    }
                    if (grow_1 != 0) {
                        float alpha_lane_1 = 1.0f;
                        if (stream_bits_1 != 0) {
                            if (o_lane_1 > -CUDART_INF_F) {
                                float _exp2_0_1 = approx_exp2(softmax_scale_log2 * (o_lane_1 - cand_lane_1));
                                alpha_lane_1 = _exp2_0_1;
                            }
                            o_lane_1 = cand_lane_1;
                            float safe_lane_1 = ((o_lane_1 == -CUDART_INF_F) ? 0.0f : o_lane_1);
                            ms_lane_1 = safe_lane_1 * softmax_scale_log2;
                        }
                        #pragma unroll
                        for (int c_15 = 0; c_15 < 8; c_15++) {
                            float _shfl_sync_1_1 = __shfl_sync(0xFFFFFFFF, alpha_lane_1, own_half_1 * 8 + c_15);
                            lane_sum_1[c_15] = lane_sum_1[c_15] * _shfl_sync_1_1;
                        }
                        #pragma unroll
                        for (int c_16 = 0; c_16 < 8; c_16++) {
                            float _shfl_sync_2_1 = __shfl_sync(0xFFFFFFFF, alpha_lane_1, (2 + own_half_1) * 8 + c_16);
                            lane_sum_1[8 + c_16] = lane_sum_1[8 + c_16] * _shfl_sync_2_1;
                        }
                        if (pair_1 > 0) {
                            int _parity_pv_done_4 = (int)((unsigned int)item_ord_1 % 2);
                            mbarrier_wait(pv_done_addr + (pair_1 - 1) * 8, _parity_pv_done_4);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if ((grow_mask_1 & byte_mask_1) != 0) {
                                int o_addr_3 = taddr + 128 + 8 + (unsigned int)tmem_lane_base_1;
                                float _tmem_load_12[8];
                                tmem_ld_x8(&_tmem_load_12[0], o_addr_3);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_17 = 0; c_17 < 8; c_17++) {
                                    float _shfl_sync_3_1 = __shfl_sync(0xFFFFFFFF, alpha_lane_1, c_17);
                                    _tmem_load_12[c_17] = _tmem_load_12[c_17] * _shfl_sync_3_1;
                                }
                                tmem_st_x8_f32(o_addr_3, _tmem_load_12);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if ((grow_mask_1 >> 8 & byte_mask_1) != 0) {
                                int o_addr_3_1 = taddr + 128 + 32 + 8 + (unsigned int)tmem_lane_base_1;
                                float tmem_load_r_0[8];
                                tmem_ld_x8(&tmem_load_r_0[0], o_addr_3_1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_3_1 = 0; c_3_1 < 8; c_3_1++) {
                                    float _shfl_sync_4_1 = __shfl_sync(0xFFFFFFFF, alpha_lane_1, 8 + c_3_1);
                                    tmem_load_r_0[c_3_1] = tmem_load_r_0[c_3_1] * _shfl_sync_4_1;
                                }
                                tmem_st_x8_f32(o_addr_3_1, tmem_load_r_0);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if ((grow_mask_1 >> 16 & byte_mask_1) != 0) {
                                int o_addr_4 = taddr + 128 + 64 + 8 + (unsigned int)tmem_lane_base_1;
                                float tmem_load_r_1[8];
                                tmem_ld_x8(&tmem_load_r_1[0], o_addr_4);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_4_1 = 0; c_4_1 < 8; c_4_1++) {
                                    float _shfl_sync_5_1 = __shfl_sync(0xFFFFFFFF, alpha_lane_1, 16 + c_4_1);
                                    tmem_load_r_1[c_4_1] = tmem_load_r_1[c_4_1] * _shfl_sync_5_1;
                                }
                                tmem_st_x8_f32(o_addr_4, tmem_load_r_1);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if ((grow_mask_1 >> 24 & byte_mask_1) != 0) {
                                int o_addr_5 = taddr + 128 + 96 + 8 + (unsigned int)tmem_lane_base_1;
                                float tmem_load_r_2[8];
                                tmem_ld_x8(&tmem_load_r_2[0], o_addr_5);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_5_1 = 0; c_5_1 < 8; c_5_1++) {
                                    float _shfl_sync_6_1 = __shfl_sync(0xFFFFFFFF, alpha_lane_1, 24 + c_5_1);
                                    tmem_load_r_2[c_5_1] = tmem_load_r_2[c_5_1] * _shfl_sync_6_1;
                                }
                                tmem_st_x8_f32(o_addr_5, tmem_load_r_2);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                    }
                    if (pair_1 >= 4) {
                        int _parity_pv_done_5 = (int)((unsigned int)item_ord_1 % 2);
                        mbarrier_wait(pv_done_addr + (pair_1 - 4) * 8, _parity_pv_done_5);
                    }
                    #pragma unroll
                    for (int c_18 = 0; c_18 < 8; c_18++) {
                        float max_scaled_1_2 = __shfl_sync(0xFFFFFFFF, ms_lane_1, own_half_1 * 8 + c_18);
                        float _exp2_7_1 = approx_exp2(__fmaf_rn(_tmem_load_10[c_18], softmax_scale_log2, -max_scaled_1_2));
                        _tmem_load_10[c_18] = _exp2_7_1;
                        lane_sum_1[c_18] = lane_sum_1[c_18] + _tmem_load_10[c_18];
                        float max_scaled_1_3 = __shfl_sync(0xFFFFFFFF, ms_lane_1, (2 + own_half_1) * 8 + c_18);
                        float _exp2_8_1 = approx_exp2(__fmaf_rn(_tmem_load_11[c_18], softmax_scale_log2, -max_scaled_1_3));
                        _tmem_load_11[c_18] = _exp2_8_1;
                        lane_sum_1[8 + c_18] = lane_sum_1[8 + c_18] + _tmem_load_11[c_18];
                    }
                    int p0_base_1 = smem_p_addr + (unsigned int)(p_stage_1 * 2 * 4096);
                    int p1_base_1 = smem_p_addr + (unsigned int)((p_stage_1 * 2 + 1) * 4096);
                    unsigned int w0_1[2];
                    unsigned int w1_1[2];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_10[0]), "f"(_tmem_load_10[1]),
                                               "f"(_tmem_load_10[2]), "f"(_tmem_load_10[3]));
                        w0_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_10[4]), "f"(_tmem_load_10[5]),
                                               "f"(_tmem_load_10[6]), "f"(_tmem_load_10[7]));
                        w0_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_11[0]), "f"(_tmem_load_11[1]),
                                               "f"(_tmem_load_11[2]), "f"(_tmem_load_11[3]));
                        w1_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_11[4]), "f"(_tmem_load_11[5]),
                                               "f"(_tmem_load_11[6]), "f"(_tmem_load_11[7]));
                        w1_1[1] = _packed;
                    }
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((p0_base_1 + (token_1 * 32 + 8 ^ (token_1 * 32 + 8 >> 7 & 1) << 4))), "r"(w0_1[0]), "r"(w0_1[1]) : "memory");
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((p1_base_1 + (token_1 * 32 + 8 ^ (token_1 * 32 + 8 >> 7 & 1) << 4))), "r"(w1_1[0]), "r"(w1_1[1]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage_1) * 8);
                    }
                    s_stage_1 += 1;
                    if (s_stage_1 == 4) { s_stage_1 = 0; _phase_s_full_1 ^= 1; }
                    p_stage_1 += 1;
                    if (p_stage_1 == 4) { p_stage_1 = 0; }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)item_ord_1 % 2)) * 8);
                }
                float _shfl_xor_sync_9_1 = __shfl_xor_sync(0xFFFFFFFF, o_lane_1, 8);
                float m_fin_1;
                asm("max.f32 %0, %1, %2;" : "=f"(m_fin_1) : "f"(o_lane_1), "f"(_shfl_xor_sync_9_1));
                float _shfl_xor_sync_10_1 = __shfl_xor_sync(0xFFFFFFFF, m_fin_1, 16);
                m_fin_1 = max_noftz(m_fin_1, _shfl_xor_sync_10_1);
                float _exp2_11_1 = approx_exp2((o_lane_1 - m_fin_1) * softmax_scale_log2);
                float w_lane_1 = ((o_lane_1 == -CUDART_INF_F) ? 0.0f : _exp2_11_1);
                float lane_sum_w_1[8];
                #pragma unroll
                for (int c_19 = 0; c_19 < 8; c_19++) {
                    float wt0_1 = __shfl_sync(0xFFFFFFFF, w_lane_1, own_half_1 * 8 + c_19);
                    float wt1_1 = __shfl_sync(0xFFFFFFFF, w_lane_1, (2 + own_half_1) * 8 + c_19);
                    lane_sum_w_1[c_19] = __fmaf_rn(lane_sum_1[c_19], wt0_1, lane_sum_1[8 + c_19] * wt1_1);
                }
                int b4_1 = lane / 16 % 2;
                int b3_1 = lane / 8 % 2;
                int b2_1 = lane / 4 % 2;
                float r4_1[4];
                #pragma unroll
                for (int c_20 = 0; c_20 < 4; c_20++) {
                    float snd4_1 = ((b4_1 == 0) ? lane_sum_w_1[c_20 + 4] : lane_sum_w_1[c_20]);
                    float kp4_1 = ((b4_1 == 0) ? lane_sum_w_1[c_20] : lane_sum_w_1[c_20 + 4]);
                    float _shfl_xor_sync_12_1 = __shfl_xor_sync(0xFFFFFFFF, snd4_1, 16);
                    r4_1[c_20] = kp4_1 + _shfl_xor_sync_12_1;
                }
                float r2_1[2];
                #pragma unroll
                for (int c_21 = 0; c_21 < 2; c_21++) {
                    float snd2_1 = ((b3_1 == 0) ? r4_1[c_21 + 2] : r4_1[c_21]);
                    float kp2_1 = ((b3_1 == 0) ? r4_1[c_21] : r4_1[c_21 + 2]);
                    float _shfl_xor_sync_13_1 = __shfl_xor_sync(0xFFFFFFFF, snd2_1, 8);
                    r2_1[c_21] = kp2_1 + _shfl_xor_sync_13_1;
                }
                float snd1_1 = ((b2_1 == 0) ? r2_1[1] : r2_1[0]);
                float kp1_1 = ((b2_1 == 0) ? r2_1[0] : r2_1[1]);
                float _shfl_xor_sync_14_1 = __shfl_xor_sync(0xFFFFFFFF, snd1_1, 4);
                float r1_1 = kp1_1 + _shfl_xor_sync_14_1;
                float _shfl_xor_sync_15_1 = __shfl_xor_sync(0xFFFFFFFF, r1_1, 2);
                r1_1 = r1_1 + _shfl_xor_sync_15_1;
                float _shfl_xor_sync_16_1 = __shfl_xor_sync(0xFFFFFFFF, r1_1, 1);
                r1_1 = r1_1 + _shfl_xor_sync_16_1;
                float xsum_1 = r1_1;
                if (lane % 4 == 0) {
                    smem_exch_sum[32 + warp_in_role_1 * 8 + lane / 4] = xsum_1;
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float head_sum_1 = 0.0f;
                if (lane < 8) {
                    float s0e_1 = smem_exch_sum[32 + lane];
                    float s1e_1 = smem_exch_sum[40 + lane];
                    float s2e_1 = smem_exch_sum[48 + lane];
                    float s3e_1 = smem_exch_sum[56 + lane];
                    head_sum_1 = s0e_1 + s1e_1 + s2e_1 + s3e_1;
                }
                if (npairs_1 > 0) {
                    int _parity_pv_done_6 = (int)((unsigned int)item_ord_1 % 2);
                    mbarrier_wait(pv_done_addr + (npairs_1 - 1) * 8, _parity_pv_done_6);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                } else {
                    int _parity_pv_done_7 = (int)((unsigned int)item_ord_1 % 2);
                    mbarrier_wait(pv_done_addr, _parity_pv_done_7);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_epi_1[8];
                #pragma unroll
                for (int h_7 = 0; h_7 < 8; h_7++) {
                    o_epi_1[h_7] = 0.0f;
                }
                if (npairs_1 > 0) {
                    float _tmem_load_16[8];
                    tmem_ld_x8(&_tmem_load_16[0], taddr + 128 + 8 + (unsigned int)tmem_lane_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_8 = 0; h_8 < 8; h_8++) {
                        float _shfl_sync_17_1 = __shfl_sync(0xFFFFFFFF, w_lane_1, h_8);
                        o_epi_1[h_8] = __fmaf_rn(_tmem_load_16[h_8], _shfl_sync_17_1, o_epi_1[h_8]);
                    }
                    float _tmem_load_17[8];
                    tmem_ld_x8(&_tmem_load_17[0], taddr + 128 + 32 + 8 + (unsigned int)tmem_lane_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_9 = 0; h_9 < 8; h_9++) {
                        float _shfl_sync_18_1 = __shfl_sync(0xFFFFFFFF, w_lane_1, 8 + h_9);
                        o_epi_1[h_9] = __fmaf_rn(_tmem_load_17[h_9], _shfl_sync_18_1, o_epi_1[h_9]);
                    }
                    float _tmem_load_18[8];
                    tmem_ld_x8(&_tmem_load_18[0], taddr + 128 + 64 + 8 + (unsigned int)tmem_lane_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_10 = 0; h_10 < 8; h_10++) {
                        float _shfl_sync_19_1 = __shfl_sync(0xFFFFFFFF, w_lane_1, 16 + h_10);
                        o_epi_1[h_10] = __fmaf_rn(_tmem_load_18[h_10], _shfl_sync_19_1, o_epi_1[h_10]);
                    }
                    float _tmem_load_19[8];
                    tmem_ld_x8(&_tmem_load_19[0], taddr + 128 + 96 + 8 + (unsigned int)tmem_lane_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int h_11 = 0; h_11 < 8; h_11++) {
                        float _shfl_sync_20_1 = __shfl_sync(0xFFFFFFFF, w_lane_1, 24 + h_11);
                        o_epi_1[h_11] = __fmaf_rn(_tmem_load_19[h_11], _shfl_sync_20_1, o_epi_1[h_11]);
                    }
                }
                if (total_items_1 <= work_idx_1 + (unsigned int)gridDim.x) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                int q_row0_1 = query_1 * num_q_heads + kv_head_1 * group_size_1 + 8;
                float _rcp_21_1 = approx_rcp(head_sum_1);
                float inv_lane_1 = ((head_sum_1 > 0.0f) ? output_scale * _rcp_21_1 : 0.0f);
                if (warp_in_role_1 == 0) {
                    if (lane < 8) {
                        if (group_size_1 > 8 + lane) {
                            float m_lane_3 = m_fin_1;
                            float lse_lane_1 = -CUDART_INF_F;
                            if (head_sum_1 > 0.0f) {
                                float _log2_22_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22_1) : "f"(head_sum_1));
                                lse_lane_1 = __fmaf_rn(m_lane_3 * softmax_scale_log2, 0.6931472f, _log2_22_1 * 0.6931472f);
                            }
                            *(reinterpret_cast<float*>(msa_lse + (q_row0_1 + lane)) + (0)) = lse_lane_1;
                        }
                    }
                }
                float inv_all_1[8];
                #pragma unroll
                for (int h_12 = 0; h_12 < 8; h_12++) {
                    float _shfl_sync_23_1 = __shfl_sync(0xFFFFFFFF, inv_lane_1, h_12);
                    inv_all_1[h_12] = _shfl_sync_23_1;
                }
                #pragma unroll
                for (int h_13 = 0; h_13 < 8; h_13++) {
                    if (group_size_1 > 8 + h_13) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_1 + h_13) * 128 + token_1)) + (0)) = (__nv_bfloat16)(o_epi_1[h_13] * inv_all_1[h_13]);
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(decode_done_addr);
                }
                item_ord_1 += 1;
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            unsigned int total_items_l = total_q * num_kv_heads;
            int load_stage = 0;
            int load_phase = 1;
            int item_par = 0;
            int cur_page = 0;
            int cur_valid = 0;
            unsigned int first_idx = blockIdx.x;
            if (first_idx < total_items_l) {
                int item_f = first_idx;
                int query_f = item_f / num_kv_heads;
                int kv_head_f = item_f % num_kv_heads;
                if (lane < 16) {
                    int tok_f = 0;
                    int batch = query_f / seqlen_q;
                    int query_in_batch = query_f - batch * seqlen_q;
                    int selected_block = task_kind[(kv_head_f * total_q + query_f) * 16 + lane];
                    int kv_len = task_kv_head[batch];
                    int valid_cols = 0;
                    if (selected_block >= 0) {
                        int block_start = selected_block * 128;
                        valid_cols = kv_len - block_start;
                        if (valid_cols > 128) {
                            valid_cols = 128;
                        }
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                        int query_position = kv_len - seqlen_q + query_in_batch;
                        int causal_cols = query_position - block_start + 1;
                        if (valid_cols > causal_cols) {
                            valid_cols = causal_cols;
                        }
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                    }
                    int token_base = 0;
                    int page_head = 0;
                    int physical_page = 0;
                    if (selected_block >= 0) {
                        physical_page = kv_indices[batch * msa_max_pages + selected_block];
                        if (physical_page < 0) {
                            valid_cols = 0;
                            physical_page = 0;
                        }
                    }
                    page_head = physical_page * num_kv_heads + kv_head_f;
                    tok_f = token_base;
                    cur_page = page_head;
                    cur_valid = valid_cols;
                }
            }
            unsigned int _phase_meta_empty = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int work_idx_l = blockIdx.x; work_idx_l < total_items_l; work_idx_l += gridDim.x) {
                int item_l = work_idx_l;
                int split_l = 0;
                int query_l = item_l / num_kv_heads;
                int kv_head_l = item_l % num_kv_heads;
                int group_size_l = num_q_heads / num_kv_heads;
                int slot_page_head = cur_page;
                int slot_valid = cur_valid;
                __syncwarp();
                mbarrier_wait(meta_empty_addr + (item_par) * 8, _phase_meta_empty);
                int meta_base_l = item_par * 16;
                unsigned int valid_mask = __ballot_sync(0xFFFFFFFF, slot_valid > 0);
                int count_l = (unsigned int)__popc(valid_mask);
                int pos_l = (int)__popc(valid_mask & ((1u << (unsigned int)lane) - 1u));
                int pos_w = count_l - 1 - pos_l;
                if (slot_valid > 0) {
                    smem_page[meta_base_l + pos_w] = slot_page_head;
                    smem_valid[meta_base_l + pos_w] = slot_valid;
                    if (pos_w + 1 == count_l) {
                        if (count_l % 2 == 1) {
                            smem_page[meta_base_l + count_l] = slot_page_head;
                            smem_valid[meta_base_l + count_l] = 0;
                        }
                    }
                }
                int npairs_all = (count_l + count_l % 2) / 2;
                int npairs_l = npairs_all - split_l * 8;
                if (npairs_l > 8) {
                    npairs_l = 8;
                }
                if (npairs_l < 0) {
                    npairs_l = 0;
                }
                int slot0_l = meta_base_l + 16 * split_l;
                __syncwarp();
                if (elect_sync()) {
                    if (npairs_l > 0) {
                        #pragma unroll
                        for (int k_tile = 0; k_tile < 2; k_tile++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[slot0_l + k_tile], kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    int q_row_k = query_l * num_q_heads + kv_head_l * group_size_l;
                    mbarrier_arrive_expect_tx(q_full_addr, 2048);
                    tma_3d_gmem2smem(smem_qt_addr, (&Q), 0, q_row_k, 0, q_full_addr);
                }
                if (elect_sync()) {
                    if (npairs_l > 1) {
                        #pragma unroll
                        for (int k_tile_1 = 0; k_tile_1 < 2; k_tile_1++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[slot0_l + 2 + k_tile_1], kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                if (elect_sync()) {
                    smem_npairs[item_par] = npairs_l;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(meta_full_addr + (item_par) * 8);
                }
                if (elect_sync()) {
                    if (npairs_l > 0) {
                        if (npairs_l > 2) {
                            #pragma unroll
                            for (int k_tile_2 = 0; k_tile_2 < 2; k_tile_2++) {
                                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                                tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[slot0_l + 4 + k_tile_2], kv_full_addr + (load_stage) * 8);
                                load_stage += 1;
                                if (load_stage == 11) { load_stage = 0; _phase_kv_empty ^= 1; }
                            }
                        }
                        #pragma unroll
                        for (int v_tile = 0; v_tile < 2; v_tile++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&V), 0, 0, smem_page[slot0_l + v_tile], kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                unsigned int nxt_idx = work_idx_l + (unsigned int)gridDim.x;
                cur_page = 0;
                cur_valid = 0;
                if (nxt_idx < total_items_l) {
                    int item_n = nxt_idx;
                    int query_n = item_n / num_kv_heads;
                    int kv_head_n = item_n % num_kv_heads;
                    if (lane < 16) {
                        int tok_n = 0;
                        int batch_1 = query_n / seqlen_q;
                        int query_in_batch_1 = query_n - batch_1 * seqlen_q;
                        int selected_block_1 = task_kind[(kv_head_n * total_q + query_n) * 16 + lane];
                        int kv_len_1 = task_kv_head[batch_1];
                        int valid_cols_1 = 0;
                        if (selected_block_1 >= 0) {
                            int block_start_1 = selected_block_1 * 128;
                            valid_cols_1 = kv_len_1 - block_start_1;
                            if (valid_cols_1 > 128) {
                                valid_cols_1 = 128;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                            int query_position_1 = kv_len_1 - seqlen_q + query_in_batch_1;
                            int causal_cols_1 = query_position_1 - block_start_1 + 1;
                            if (valid_cols_1 > causal_cols_1) {
                                valid_cols_1 = causal_cols_1;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                        }
                        int token_base_1 = 0;
                        int page_head_1 = 0;
                        int physical_page_1 = 0;
                        if (selected_block_1 >= 0) {
                            physical_page_1 = kv_indices[batch_1 * msa_max_pages + selected_block_1];
                            if (physical_page_1 < 0) {
                                valid_cols_1 = 0;
                                physical_page_1 = 0;
                            }
                        }
                        page_head_1 = physical_page_1 * num_kv_heads + kv_head_n;
                        tok_n = token_base_1;
                        cur_page = page_head_1;
                        cur_valid = valid_cols_1;
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int pair_2 = 1; pair_2 < 8; pair_2++) {
                        if (npairs_l <= pair_2) {
                            break;
                        }
                        if (npairs_l > pair_2 + 2) {
                            #pragma unroll
                            for (int k_tile_3 = 0; k_tile_3 < 2; k_tile_3++) {
                                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                                tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[slot0_l + pair_2 * 2 + 4 + k_tile_3], kv_full_addr + (load_stage) * 8);
                                load_stage += 1;
                                if (load_stage == 11) { load_stage = 0; _phase_kv_empty ^= 1; }
                            }
                        }
                        #pragma unroll
                        for (int v_tile_1 = 0; v_tile_1 < 2; v_tile_1++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&V), 0, 0, smem_page[slot0_l + pair_2 * 2 + v_tile_1], kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                item_par += 1;
                if (item_par == 2) { item_par = 0; _phase_meta_empty ^= 1; }
            }
        }
    }
#if __CUDA_ARCH__ == 1030
    // ---- Role: mma_s ----
#else
    // ---- Role: mma ----
#endif
    if (warp == 9) {
#if __CUDA_ARCH__ == 1030
        { // mma_s_main
            unsigned int total_items_s = total_q * num_kv_heads;
            int item_stage_s = 0;
            int s_stage_s = 0;
            int se_phase_s = 1;
            int k_tile_s = 0;
            int s_pairs_s = 0;
            int pv_tile_c = 0;
            int pv_pairs_c = 0;
#else
        { // mma_main
            unsigned int total_items_m = total_q * num_kv_heads;
            int item_stage_m = 0;
            int item_ord_m = 0;
            int s_stage_m = 0;
            int se_phase_m = 1;
            int pf_stage_m = 0;
            int pf_phase_m = 0;
            int have_prev_m = 0;
            int k_tile_m = 0;
            int v_tile_m = 0;
#endif
            unsigned int _phase_meta_full_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_s_empty = 1;
#if __CUDA_ARCH__ == 1030
#else
            unsigned int _phase_p_full = 0;
            unsigned int _phase_decode_done_0 = 0;
#endif
            #pragma unroll 1
#if __CUDA_ARCH__ == 1030
            for (unsigned int work_idx_s = blockIdx.x; work_idx_s < total_items_s; work_idx_s += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_s) * 8, _phase_meta_full_2);
                int npairs_s = smem_npairs[item_stage_s];
                item_stage_s += 1;
                if (item_stage_s == 2) { item_stage_s = 0; _phase_meta_full_2 ^= 1; }
#else
            for (unsigned int work_idx_m = blockIdx.x; work_idx_m < total_items_m; work_idx_m += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_m) * 8, _phase_meta_full_2);
                int npairs_m = smem_npairs[item_stage_m];
                item_stage_m += 1;
                if (item_stage_m == 2) { item_stage_m = 0; _phase_meta_full_2 ^= 1; }
#endif
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                if (npairs_s == 0) {
#else
                if (npairs_m == 0) {
#endif
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(q_empty_addr);
                    }
                }
#if __CUDA_ARCH__ == 1030
#else
                int s_next = 0;
                int pv_next = 0;
                if (npairs_m > 0) {
                    v_tile_m += 2;
                }
                if (npairs_m > 1) {
                    v_tile_m += 2;
                }
                if (npairs_m > 2) {
                    v_tile_m += 2;
                }
                int first_pv = 1;
                int dd_waited = 0;
                unsigned int consumed_m = 0;
                unsigned int two_bits = 3;
#endif
                #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                for (int s_pair = 0; s_pair < npairs_s; s_pair++) {
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr)), "r"(static_cast<uint32_t>((unsigned int)k_tile_s)) : "memory");
#else
                for (int spin = 0; spin < 4194304; spin++) {
                    if (pv_next >= npairs_m) {
                        break;
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    int need_tile_s = k_tile_s + 2 - 11;
                    int need_pairs_s = s_pairs_s + 1 - 3;
                    #pragma unroll 1
                    for (int spin_s = 0; spin_s < 4194304; spin_s++) {
                        int ok_s = 0;
                        if (pv_tile_c >= need_tile_s) {
                            if (pv_pairs_c >= need_pairs_s) {
                                ok_s = 1;
#else
                    if (s_next < npairs_m) {
                        if (s_next - pv_next < 3) {
                            int kl = 2 * s_next;
                            if (s_next >= 2) {
                                kl = 4 * s_next - 4;
                            }
                            int kpred = kl - 11;
                            int k_free = 1;
                            if (kpred >= 0) {
                                if ((consumed_m >> (unsigned int)kpred & two_bits) != two_bits) {
                                    k_free = 0;
                                }
                            }
                            unsigned int s_ready = 0;
                            if (k_free != 0) {
                                int k0 = k_tile_m % 11;
                                int k1 = (k_tile_m + 1) % 11;
                                int _parity_kv_full = (int)((unsigned int)k_tile_m / 11 % 2);
                                uint32_t _mbar_token_0 = mbarrier_test_wait(kv_full_addr + (k0) * 8, _parity_kv_full);
                                unsigned int s_tok = _mbar_token_0;
                                int _parity_kv_full_1 = (int)((unsigned int)(k_tile_m + 1) / 11 % 2);
                                uint32_t _mbar_token_1 = mbarrier_test_wait(kv_full_addr + (k1) * 8, _parity_kv_full_1);
                                s_tok = s_tok & _mbar_token_1;
                                uint32_t _mbar_token_2 = mbarrier_test_wait(s_empty_addr + (s_stage_m) * 8, _phase_s_empty);
                                s_tok = s_tok & _mbar_token_2;
                                unsigned int _warp_vote_0 = __ballot_sync(0xFFFFFFFF, s_tok != 0);
                                s_ready = _warp_vote_0;
                            }
                            if (s_ready == 4294967295u) {
                                int k0_1 = k_tile_m % 11;
                                int k1_1 = (k_tile_m + 1) % 11;
                                int _parity_kv_full_2 = (int)((unsigned int)k_tile_m / 11 % 2);
                                mbarrier_wait(kv_full_addr + (k0_1) * 8, _parity_kv_full_2);
                                int _parity_kv_full_3 = (int)((unsigned int)(k_tile_m + 1) / 11 % 2);
                                mbarrier_wait(kv_full_addr + (k1_1) * 8, _parity_kv_full_3);
                                mbarrier_wait(s_empty_addr + (s_stage_m) * 8, _phase_s_empty);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k0_1) * 1024);
                                int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                                {
                                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 0);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                                    }
                                }
                                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k1_1) * 1024);
                                {
                                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32 + 16)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 0);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32 + 16)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32 + 16)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_m * 32 + 16)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                                    }
                                }
                                if (npairs_m == s_next + 1) {
                                    elect_commit(q_empty_addr);
                                }
                                elect_commit(s_full_addr + (s_stage_m) * 8);
                                elect_commit(kv_empty_addr + (k0_1) * 8);
                                elect_commit(kv_empty_addr + (k1_1) * 8);
                                s_stage_m += 1;
                                if (s_stage_m == 4) { s_stage_m = 0; _phase_s_empty ^= 1; }
                                consumed_m = consumed_m | two_bits << (unsigned int)kl;
                                if (s_next < 2) {
                                    k_tile_m += 2;
                                } else {
                                    k_tile_m += 4;
                                }
                                s_next += 1;
#endif
                            }
                        }
#if __CUDA_ARCH__ == 1030
                        unsigned int ready_s = __ballot_sync(0xFFFFFFFF, ok_s != 0);
                        if (ready_s == 4294967295u) {
                            break;
                        }
                        uint32_t _shared_acquire_0;
                        asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(_shared_acquire_0) : "r"(static_cast<uint32_t>(smem_issue_addr + 4)) : "memory");
                        pv_tile_c = (int)_shared_acquire_0;
                        uint32_t _shared_acquire_1;
                        asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(_shared_acquire_1) : "r"(static_cast<uint32_t>(smem_issue_addr + 8)) : "memory");
                        pv_pairs_c = (int)_shared_acquire_1;
#else
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    int k0 = k_tile_s % 11;
                    int k1 = (k_tile_s + 1) % 11;
                    int _parity_kv_full = (int)((unsigned int)k_tile_s / 11 % 2);
                    mbarrier_wait(kv_full_addr + (k0) * 8, _parity_kv_full);
                    int _parity_kv_full_1 = (int)((unsigned int)(k_tile_s + 1) / 11 % 2);
                    mbarrier_wait(kv_full_addr + (k1) * 8, _parity_kv_full_1);
                    mbarrier_wait(s_empty_addr + (s_stage_s) * 8, _phase_s_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k0) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 0);
#else
                    if (pv_next < s_next) {
                        int kiters = pv_next;
                        if (kiters > npairs_m - 2) {
                            kiters = npairs_m - 2;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
#else
                        if (kiters < 0) {
                            kiters = 0;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
#else
                        int vl = 4 * kiters + 2 * (pv_next - kiters) + 2;
                        if (npairs_m > 1) {
                            vl = vl + 2;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
#else
                        if (npairs_m > pv_next + 2) {
                            vl = vl + 2;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32 + 16)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 0);
#else
                        int vpred = vl - 11;
                        int v_free = 1;
                        if (vpred >= 0) {
                            if ((consumed_m >> (unsigned int)vpred & two_bits) != two_bits) {
                                v_free = 0;
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32 + 16)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
#else
                        unsigned int p_ready = 0;
                        if (v_free != 0) {
                            int _stage_kv_full = (int)((unsigned int)(int)v_tile_m % 11);
                            int _parity_kv_full_4 = (int)((unsigned int)(int)v_tile_m / 11 % 2);
                            uint32_t _mbar_token_3 = mbarrier_test_wait(kv_full_addr + (_stage_kv_full) * 8, _parity_kv_full_4);
                            unsigned int p_tok = _mbar_token_3;
                            int _stage_kv_full_1 = (int)((unsigned int)(int)(v_tile_m + 1) % 11);
                            int _parity_kv_full_5 = (int)((unsigned int)(int)(v_tile_m + 1) / 11 % 2);
                            uint32_t _mbar_token_4 = mbarrier_test_wait(kv_full_addr + (_stage_kv_full_1) * 8, _parity_kv_full_5);
                            p_tok = p_tok & _mbar_token_4;
                            uint32_t _mbar_token_5 = mbarrier_test_wait(p_full_addr + (pf_stage_m) * 8, _phase_p_full);
                            p_tok = p_tok & _mbar_token_5;
                            unsigned int _warp_vote_3 = __ballot_sync(0xFFFFFFFF, p_tok != 0);
                            p_ready = _warp_vote_3;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32 + 16)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 32 + 16)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                        }
                    }
                    if (npairs_s == s_pair + 1) {
                        elect_commit(q_empty_addr);
                    }
                    elect_commit(s_full_addr + (s_stage_s) * 8);
                    elect_commit(kv_empty_addr + (k0) * 8);
                    elect_commit(kv_empty_addr + (k1) * 8);
                    s_stage_s += 1;
                    if (s_stage_s == 4) { s_stage_s = 0; _phase_s_empty ^= 1; }
                    s_pairs_s = s_pairs_s + 1;
                    if (s_pair < 2) {
                        k_tile_s += 2;
                    } else {
                        k_tile_s += 4;
                    }
                }
                if (npairs_s == 1) {
                    k_tile_s += 2;
                }
                if (npairs_s > 1) {
                    k_tile_s += 4;
                }
                if (elect_sync()) {
                    asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr)), "r"(static_cast<uint32_t>((unsigned int)k_tile_s)) : "memory");
                }
            }
        }
    }
    // ---- Role: mma_pv ----
    if (warp == 10) {
        { // mma_pv_main
            unsigned int total_items_v = total_q * num_kv_heads;
            int item_stage_v = 0;
            int item_ord_v = 0;
            int pf_stage_v = 0;
            int pf_phase_v = 0;
            int have_prev_v = 0;
            int v_tile_v = 0;
            int pv_pairs_v = 0;
            int s_tile_c = 0;
            unsigned int _phase_meta_full_3 = 0;
            unsigned int _phase_decode_done_0 = 0;
            unsigned int _phase_p_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_v = blockIdx.x; work_idx_v < total_items_v; work_idx_v += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_v) * 8, _phase_meta_full_3);
                int npairs_v = smem_npairs[item_stage_v];
                item_stage_v += 1;
                if (item_stage_v == 2) { item_stage_v = 0; _phase_meta_full_3 ^= 1; }
                if (npairs_v > 0) {
                    int first_v = 2;
                    v_tile_v += 2;
                    if (npairs_v > 1) {
                        first_v = first_v + 2;
                        v_tile_v += 2;
                    }
                    if (npairs_v > 2) {
                        first_v = first_v + 2;
                        v_tile_v += 2;
                    }
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 4)), "r"(static_cast<uint32_t>((unsigned int)v_tile_v)) : "memory");
                    }
                }
                if (have_prev_v != 0) {
                    mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                    _phase_decode_done_0 ^= 1;
                    int _parity_pv_done_8 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr, _parity_pv_done_8);
                    int _parity_pv_done_9 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 8, _parity_pv_done_9);
                    int _parity_pv_done_10 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 16, _parity_pv_done_10);
                    int _parity_pv_done_11 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 24, _parity_pv_done_11);
                    int _parity_pv_done_12 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 32, _parity_pv_done_12);
                    int _parity_pv_done_13 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 40, _parity_pv_done_13);
                    int _parity_pv_done_14 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 48, _parity_pv_done_14);
                    int _parity_pv_done_15 = (int)((unsigned int)item_ord_v % 2) ^ 1;
                    mbarrier_wait(pv_done_addr + 56, _parity_pv_done_15);
                }
                int first_pv_v = 1;
                #pragma unroll 1
                for (int pv_pair = 0; pv_pair < npairs_v; pv_pair++) {
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 4)), "r"(static_cast<uint32_t>((unsigned int)v_tile_v)) : "memory");
                    }
                    int need_tile_v = v_tile_v + 2 - 11;
                    #pragma unroll 1
                    for (int spin_v = 0; spin_v < 4194304; spin_v++) {
                        int ok_v = 0;
                        if (s_tile_c >= need_tile_v) {
                            ok_v = 1;
                        }
                        unsigned int ready_v = __ballot_sync(0xFFFFFFFF, ok_v != 0);
                        if (ready_v == 4294967295u) {
                            break;
                        }
                        uint32_t _shared_acquire_2;
                        asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(_shared_acquire_2) : "r"(static_cast<uint32_t>(smem_issue_addr)) : "memory");
                        s_tile_c = (int)_shared_acquire_2;
                    }
                    int v0 = (int)v_tile_v % 11;
                    int v1 = (int)(v_tile_v + 1) % 11;
                    int _stage_kv_full = (int)((unsigned int)(int)v_tile_v % 11);
                    int _parity_kv_full_2 = (int)((unsigned int)(int)v_tile_v / 11 % 2);
                    mbarrier_wait(kv_full_addr + (_stage_kv_full) * 8, _parity_kv_full_2);
                    int _stage_kv_full_1 = (int)((unsigned int)(int)(v_tile_v + 1) % 11);
                    int _parity_kv_full_3 = (int)((unsigned int)(int)(v_tile_v + 1) / 11 % 2);
                    mbarrier_wait(kv_full_addr + (_stage_kv_full_1) * 8, _parity_kv_full_3);
                    mbarrier_wait(p_full_addr + (pf_stage_v) * 8, _phase_p_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v0) * 1024);
                    int _mma_b_lo_0_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2) * 256);
                    asm volatile(
#else
                        if (p_ready == 4294967295u) {
                            if (dd_waited == 0) {
                                if (have_prev_m != 0) {
                                    mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                                    _phase_decode_done_0 ^= 1;
                                    int _parity_pv_done_8 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr, _parity_pv_done_8);
                                    int _parity_pv_done_9 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 8, _parity_pv_done_9);
                                    int _parity_pv_done_10 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 16, _parity_pv_done_10);
                                    int _parity_pv_done_11 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 24, _parity_pv_done_11);
                                    int _parity_pv_done_12 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 32, _parity_pv_done_12);
                                    int _parity_pv_done_13 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 40, _parity_pv_done_13);
                                    int _parity_pv_done_14 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 48, _parity_pv_done_14);
                                    int _parity_pv_done_15 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                                    mbarrier_wait(pv_done_addr + 56, _parity_pv_done_15);
                                }
                                dd_waited = 1;
                            }
                            int v0 = (int)v_tile_m % 11;
                            int v1 = (int)(v_tile_m + 1) % 11;
                            int _stage_kv_full_2 = (int)((unsigned int)(int)v_tile_m % 11);
                            int _parity_kv_full_6 = (int)((unsigned int)(int)v_tile_m / 11 % 2);
                            mbarrier_wait(kv_full_addr + (_stage_kv_full_2) * 8, _parity_kv_full_6);
                            int _stage_kv_full_3 = (int)((unsigned int)(int)(v_tile_m + 1) % 11);
                            int _parity_kv_full_7 = (int)((unsigned int)(int)(v_tile_m + 1) / 11 % 2);
                            mbarrier_wait(kv_full_addr + (_stage_kv_full_3) * 8, _parity_kv_full_7);
                            mbarrier_wait(p_full_addr + (pf_stage_m) * 8, _phase_p_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v0) * 1024);
                            int _mma_b_lo_0_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_m * 2) * 256);
                            asm volatile(
#endif
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134840336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(_mma_a_lo_0_1), "r"(_mma_b_lo_0_1), "r"(tmem_tmem_o), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    int _mma_a_lo_1_0 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v0) * 1024);
                    int _mma_b_lo_5_0 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2) * 256);
                    asm volatile(
#else
                    :: "r"(_mma_a_lo_0_1), "r"(_mma_b_lo_0_1), "r"(tmem_tmem_o), "r"(((first_pv) ? 0 : 1)) : "memory");
                            int _mma_a_lo_1_0 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v0) * 1024);
                            int _mma_b_lo_5_0 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_m * 2) * 256);
                            asm volatile(
#endif
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134840336;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(_mma_a_lo_1_0), "r"(_mma_b_lo_5_0), "r"((tmem_tmem_o + (32))), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    int _mma_a_lo_1_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v1) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2 + 1) * 256);
                    asm volatile(
#else
                    :: "r"(_mma_a_lo_1_0), "r"(_mma_b_lo_5_0), "r"((tmem_tmem_o + (32))), "r"(((first_pv) ? 0 : 1)) : "memory");
                            int _mma_a_lo_1_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v1) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_m * 2 + 1) * 256);
                            asm volatile(
#endif
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134840336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(_mma_a_lo_1_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_o + (64))), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    int _mma_a_lo_1_1_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v1) * 1024);
                    int _mma_b_lo_5_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2 + 1) * 256);
                    asm volatile(
#else
                    :: "r"(_mma_a_lo_1_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_o + (64))), "r"(((first_pv) ? 0 : 1)) : "memory");
                            int _mma_a_lo_1_1_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v1) * 1024);
                            int _mma_b_lo_5_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_m * 2 + 1) * 256);
                            asm volatile(
#endif
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134840336;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(_mma_a_lo_1_1_1), "r"(_mma_b_lo_5_1), "r"((tmem_tmem_o + (96))), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    elect_commit(pv_done_addr + (pv_pair) * 8);
                    elect_commit(kv_empty_addr + (v0) * 8);
                    elect_commit(kv_empty_addr + (v1) * 8);
                    pf_stage_v += 1;
                    if (pf_stage_v == 4) { pf_stage_v = 0; _phase_p_full ^= 1; }
                    first_pv_v = 0;
                    pv_pairs_v = pv_pairs_v + 1;
                    if (npairs_v > pv_pair + 3) {
                        v_tile_v += 4;
                    } else {
                        v_tile_v += 2;
#else
                    :: "r"(_mma_a_lo_1_1_1), "r"(_mma_b_lo_5_1), "r"((tmem_tmem_o + (96))), "r"(((first_pv) ? 0 : 1)) : "memory");
                            elect_commit(pv_done_addr + (pv_next) * 8);
                            elect_commit(kv_empty_addr + (v0) * 8);
                            elect_commit(kv_empty_addr + (v1) * 8);
                            pf_stage_m += 1;
                            if (pf_stage_m == 4) { pf_stage_m = 0; _phase_p_full ^= 1; }
                            consumed_m = consumed_m | two_bits << (unsigned int)vl;
                            first_pv = 0;
                            if (npairs_m > pv_next + 3) {
                                v_tile_m += 4;
                            } else {
                                v_tile_m += 2;
                            }
                            pv_next += 1;
                        }
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 8)), "r"(static_cast<uint32_t>((unsigned int)pv_pairs_v)) : "memory");
#else
                }
                if (dd_waited == 0) {
                    if (have_prev_m != 0) {
                        mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                        _phase_decode_done_0 ^= 1;
                        int _parity_pv_done_16 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr, _parity_pv_done_16);
                        int _parity_pv_done_17 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 8, _parity_pv_done_17);
                        int _parity_pv_done_18 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 16, _parity_pv_done_18);
                        int _parity_pv_done_19 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 24, _parity_pv_done_19);
                        int _parity_pv_done_20 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 32, _parity_pv_done_20);
                        int _parity_pv_done_21 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 40, _parity_pv_done_21);
                        int _parity_pv_done_22 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 48, _parity_pv_done_22);
                        int _parity_pv_done_23 = (int)((unsigned int)item_ord_m % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 56, _parity_pv_done_23);
#endif
                    }
                }
                #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                for (int spare_stage_v = npairs_v; spare_stage_v < 8; spare_stage_v++) {
#else
                for (int spare_stage = npairs_m; spare_stage < 8; spare_stage++) {
#endif
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        mbarrier_arrive(pv_done_addr + (spare_stage_v) * 8);
#else
                        mbarrier_arrive(pv_done_addr + (spare_stage) * 8);
#endif
                    }
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                have_prev_v = 1;
                item_ord_v += 1;
                if (elect_sync()) {
                    asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 4)), "r"(static_cast<uint32_t>((unsigned int)(v_tile_v + 2))) : "memory");
#else
                have_prev_m = 1;
                item_ord_m += 1;
                if (npairs_m == 1) {
                    k_tile_m += 2;
                }
                if (npairs_m > 1) {
                    k_tile_m += 4;
#endif
                }
            }
        }
    }
    // ---- Role: idle ----
#if __CUDA_ARCH__ == 1030
    if (warp == 11) {
#else
    if (warp == 10) {
#endif
        { // idle_main
            unsigned int _phase_tmem_dead_0 = 0;
            mbarrier_wait(tmem_dead_addr, _phase_tmem_dead_0);
            _phase_tmem_dead_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
#if __CUDA_ARCH__ == 1030
#else
    // ---- Role: spare ----
    if (warp == 11) {
        // idle — no tasks assigned
    }
#endif

    // Cleanup
}

} // extern "C"
