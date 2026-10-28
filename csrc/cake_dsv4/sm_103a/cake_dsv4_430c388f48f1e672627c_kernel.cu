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
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 1
#define NUM_V_PIPE_STAGES 1
#define NUM_KV_PIPE_STAGES 8
#define NUM_INDEX_PIPE_STAGES 1
#define NUM_WINDOW_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_KV_OFF 66560
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 66560
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_Q_FULL_OFF 1024
#define SMEM_SMEM_Q_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_Q_FULL_STRIDE 32768
#define SMEM_SMEM_K_FULL_OFF 66560
#define SMEM_SMEM_K_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_K_FULL_STRIDE 32768
#define SMEM_SMEM_V_FULL_OFF 132096
#define SMEM_SMEM_V_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_V_FULL_STRIDE 32768
#define SMEM_SMEM_STATS_MAX_OFF 197632
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 198656
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_SMEM_STATS_FINAL_MAX_OFF 199168
#define SMEM_SMEM_STATS_FINAL_MAX_STAGE_BYTES 512
#define SMEM_SMEM_STATS_FINAL_MAX_STRIDE 512
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_OFF 199680
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STAGE_BYTES 1024
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STRIDE 1024
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_OFF 200704
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STRIDE 512
#define SMEM_SMEM_SPARSE_INDICES_OFF 201216
#define SMEM_SMEM_SPARSE_INDICES_STAGE_BYTES 4608
#define SMEM_SMEM_SPARSE_INDICES_STRIDE 4608
#define SMEM_SMEM_SWA_WINDOW_OFF 205824
#define SMEM_SMEM_SWA_WINDOW_STAGE_BYTES 32
#define SMEM_SMEM_SWA_WINDOW_STRIDE 32
#define SMEM_SMEM_P_FP8_OFF 1024
#define SMEM_SMEM_P_FP8_STAGE_BYTES 8192
#define SMEM_SMEM_P_FP8_STRIDE 8192
#define SMEM_SMEM_BF16_SCORE_EXCHANGE_OFF 1024
#define SMEM_SMEM_BF16_SCORE_EXCHANGE_STAGE_BYTES 8192
#define SMEM_SMEM_BF16_SCORE_EXCHANGE_STRIDE 8192
#define SMEM_TOTAL 205952
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



__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}






__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
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



__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_430c388f48f1e672627c(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_lse, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, int ragged_query, int max_q_len, int batch_size)
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
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 24)
    #define v_full_addr (mbar_base + 32)
    #define v_empty_addr (mbar_base + 40)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 112)
    #define index_full_addr (mbar_base + 176)
    #define index_empty_addr (mbar_base + 184)
    #define s_full_addr (mbar_base + 192)
    #define p_full_addr (mbar_base + 208)
    #define corr_done_addr (mbar_base + 224)
    #define stats_addr (mbar_base + 240)
    #define sum_ready_addr (mbar_base + 256)
    #define o_done_addr (mbar_base + 264)
    #define pv_done_addr (mbar_base + 272)
    #define window_full_addr (mbar_base + 280)
    #define window_empty_addr (mbar_base + 296)
    #define s_seeded_addr (mbar_base + 312)
    #define q_pair_ready_addr (mbar_base + 320)
    #define kv_pair_ready_addr (mbar_base + 328)
    #define pv_pair_ready_addr (mbar_base + 392)
    #define tmem_dealloc_addr (mbar_base + 408)
    #define tmem_dealloc_peer_addr (mbar_base + 416)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_kv_addr = smem + 66560;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_v_addr = smem + 66560;
    uint8_t* smem_q_full = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_full_addr = smem + 1024;
    uint8_t* smem_k_full = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_k_full_addr = smem + 66560;
    uint8_t* smem_v_full = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v_full_addr = smem + 132096;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_stats_max_addr = smem + 197632;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 198656);
    const int smem_stats_sum_addr = smem + 198656;
    float* smem_stats_final_max = reinterpret_cast<float*>(smem_raw + 199168);
    const int smem_stats_final_max_addr = smem + 199168;
    float* smem_softmax_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 199680);
    const int smem_softmax_warp_pair_exchange_addr = smem + 199680;
    float* smem_corr_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 200704);
    const int smem_corr_warp_pair_exchange_addr = smem + 200704;
    int* smem_sparse_indices = reinterpret_cast<int*>(smem_raw + 201216);
    const int smem_sparse_indices_addr = smem + 201216;
    int* smem_swa_window = reinterpret_cast<int*>(smem_raw + 205824);
    const int smem_swa_window_addr = smem + 205824;
    uint8_t* smem_p_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_p_fp8_addr = smem + 1024;
    float* smem_bf16_score_exchange = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_bf16_score_exchange_addr = smem + 1024;
    int k1_pre_rows[4];
    #pragma unroll
    for (int k1_pre_row_i = 0; k1_pre_row_i < 4; k1_pre_row_i++) {
        k1_pre_rows[k1_pre_row_i] = -1;
    }
    if (warp >= 9 && warp <= 12 && bid < total_work_items) {
        int k1_pre_load_warp_rank = warp - 9;
        int k1_pre_query_idx = bid >> 1;
        int k1_pre_all_num_kv_tiles = (sparse_topk + 128 - 1) / 128;
        int k1_pre_sparse_extent = k1_pre_all_num_kv_tiles * 128;
        int k1_pre_num_index_passes = k1_pre_sparse_extent / 128;
        if (k1_pre_load_warp_rank < k1_pre_num_index_passes) {
            int k1_pre_tile_col_base = k1_pre_load_warp_rank * 128;
            int* k1_pre_row_tile_ptr = ((k1_pre_tile_col_base < 128) ? (swa_indices + (k1_pre_query_idx * swa_index_stride)) : (compressed_indices + (k1_pre_query_idx * compressed_index_stride + (k1_pre_tile_col_base - 128))));
            int k1_pre_lane_col = k1_pre_tile_col_base + lane * 4;
            int k1_pre_row_stride = ((k1_pre_tile_col_base < 128) ? swa_index_stride : compressed_index_stride);
            int k1_pre_row_vec_misaligned = k1_pre_query_idx * k1_pre_row_stride & 3;
            if (k1_pre_lane_col + 4 <= sparse_topk && k1_pre_row_vec_misaligned == 0) {
                int _vec_load_0[4];
                {
                    const int4* _ivptr_0 = reinterpret_cast<const int4*>(k1_pre_row_tile_ptr + (lane * 4) + 0);
                    int4 _ivld_0;
                    _ivld_0 = *_ivptr_0;
                    _vec_load_0[0 + 0] = _ivld_0.x;
                    _vec_load_0[0 + 1] = _ivld_0.y;
                    _vec_load_0[0 + 2] = _ivld_0.z;
                    _vec_load_0[0 + 3] = _ivld_0.w;
                }
                #pragma unroll
                for (int k1_pre_row_i_1 = 0; k1_pre_row_i_1 < 4; k1_pre_row_i_1++) {
                    k1_pre_rows[k1_pre_row_i_1] = _vec_load_0[k1_pre_row_i_1];
                }
            } else {
                #pragma unroll
                for (int k1_pre_row_i_2 = 0; k1_pre_row_i_2 < 4; k1_pre_row_i_2++) {
                    if (k1_pre_lane_col + k1_pre_row_i_2 < sparse_topk) {
                        k1_pre_rows[k1_pre_row_i_2] = k1_pre_row_tile_ptr[lane * 4 + k1_pre_row_i_2];
                    }
                }
            }
        }
    }

    // Mbarrier init (25 pipeline groups, 0 ordered-sequence groups, 53 barriers)
    // Mbarriers at smem_raw[0..424)

    if (warp == 0) {
        // q_full: 1 barriers, init_count=1
        // q_empty: 1 barriers, init_count=1
        // --- pipeline 'k_pipe' ---
        // k_full: 1 barriers, init_count=1
        // k_empty: 1 barriers, init_count=1
        // --- pipeline 'v_pipe' ---
        // v_full: 1 barriers, init_count=1
        // v_empty: 1 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 8 barriers, init_count=1
        // kv_empty: 8 barriers, init_count=1
        // --- pipeline 'index_pipe' ---
        // index_full: 1 barriers, init_count=1
        // index_empty: 1 barriers, init_count=1
        // s_full: 2 barriers, init_count=1
        // p_full: 2 barriers, init_count=128
        // corr_done: 2 barriers, init_count=128
        // stats: 2 barriers, init_count=128
        // sum_ready: 1 barriers, init_count=128
        // o_done: 1 barriers, init_count=1
        // pv_done: 1 barriers, init_count=1
        // --- pipeline 'window_pipe' ---
        // window_full: 2 barriers, init_count=64
        // window_empty: 2 barriers, init_count=128
        // s_seeded: 1 barriers, init_count=128
        // q_pair_ready: 1 barriers, init_count=64
        // --- pipeline 'kv_pipe' ---
        // kv_pair_ready: 8 barriers, init_count=64
        // pv_pair_ready: 2 barriers, init_count=64
        // tmem_dealloc: 1 barriers, init_count=416
        // tmem_dealloc_peer: 1 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(26), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(20), "r"((uint32_t)(416)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(19), "r"((uint32_t)(64)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(5), "r"((uint32_t)(64)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(3), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(1), "r"((uint32_t)(128)));
        if (lane < 21) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncthreads();

    // Kernel pre-TMEM ops
    if (warp == 9 && bid < total_work_items) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_full_addr, 65536);
            #pragma unroll
            for (int k1_pre_q_stage = 0; k1_pre_q_stage < 4; k1_pre_q_stage++) {
                tma_4d_gmem2smem(smem_q_addr + (unsigned int)(k1_pre_q_stage * 16384), (&tmap_q), 0, 0, k1_pre_q_stage, bid >> 1, q_full_addr);
            }
        }
    }

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 424);
    if (warp == 0) {
        int _tmem_hold = smem + 424;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: index_warp ----
    if (warp == 13) {
        { // index_warp_main
            const int index_dummy = 0;
            unsigned int _phase_q_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work_items; work_idx += num_bids) {
                mbarrier_wait_hint(q_full_addr, _phase_q_full_0, 10000000);
                _phase_q_full_0 ^= 1;
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 9 && warp <= 12) {
        { // load_warp_main
            const int wg2_dummy = 0;
            const int load_warp_rank = warp - 9;
            int all_num_kv_tiles = (sparse_topk + 128 - 1) / 128;
            unsigned int load_kv_stage = 0;
            int k_cta_offset = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_index_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = bid; work_idx_1 < total_work_items; work_idx_1 += num_bids) {
                int split_idx = 0;
                int query_idx = work_idx_1 >> 1;
                int v_chunk = work_idx_1 & 1;
                int tiles_per_split = all_num_kv_tiles;
                int first_tile = split_idx * tiles_per_split;
                int num_kv_tiles = tiles_per_split;
                int sparse_extent = num_kv_tiles * 128;
                mbarrier_wait_hint(q_empty_addr, _phase_q_empty_0, 10000000);
                _phase_q_empty_0 ^= 1;
                if (load_warp_rank == 0 && work_idx_1 != (unsigned int)bid) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr, 65536);
                        #pragma unroll
                        for (int q_stage = 0; q_stage < 4; q_stage++) {
                            tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 16384), (&tmap_q), 0, 0, q_stage, query_idx, q_full_addr);
                        }
                    }
                }
                int num_index_passes = sparse_extent / 128;
                if (work_idx_1 == (unsigned int)bid && load_warp_rank < num_index_passes) {
                    int k1_pre_index_offset = load_warp_rank * 128 + lane * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sparse_indices_addr + (unsigned int)(k1_pre_index_offset * 4)), "r"(k1_pre_rows[0]), "r"(k1_pre_rows[1]), "r"(k1_pre_rows[2]), "r"(k1_pre_rows[3]) : "memory");
                }
                #pragma unroll 1
                for (int index_pass = ((work_idx_1 == (unsigned int)bid) ? load_warp_rank + 4 : load_warp_rank); index_pass < num_index_passes; index_pass += 4) {
                    int index_offset = index_pass * 128 + lane * 4;
                    int tile_col_base = (first_tile + index_pass) * 128;
                    int* row_tile_ptr = ((tile_col_base < 128) ? (swa_indices + (query_idx * swa_index_stride)) : (compressed_indices + (query_idx * compressed_index_stride + (tile_col_base - 128))));
                    int lane_col = tile_col_base + lane * 4;
                    int sparse_rows[4];
                    #pragma unroll
                    for (int row_i = 0; row_i < 4; row_i++) {
                        sparse_rows[row_i] = -1;
                    }
                    int row_stride = ((tile_col_base < 128) ? swa_index_stride : compressed_index_stride);
                    int row_vec_misaligned = query_idx * row_stride & 3;
                    if (lane_col + 4 <= sparse_topk && row_vec_misaligned == 0) {
                        int _vec_load_1[4];
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(row_tile_ptr + (lane * 4) + 0);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            _vec_load_1[0 + 0] = _ivld_0.x;
                            _vec_load_1[0 + 1] = _ivld_0.y;
                            _vec_load_1[0 + 2] = _ivld_0.z;
                            _vec_load_1[0 + 3] = _ivld_0.w;
                        }
                        #pragma unroll
                        for (int row_i_1 = 0; row_i_1 < 4; row_i_1++) {
                            sparse_rows[row_i_1] = _vec_load_1[row_i_1];
                        }
                    } else {
                        #pragma unroll
                        for (int row_i_2 = 0; row_i_2 < 4; row_i_2++) {
                            if (lane_col + row_i_2 < sparse_topk) {
                                sparse_rows[row_i_2] = row_tile_ptr[lane * 4 + row_i_2];
                            }
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sparse_indices_addr + (unsigned int)(index_offset * 4)), "r"(sparse_rows[0]), "r"(sparse_rows[1]), "r"(sparse_rows[2]), "r"(sparse_rows[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                int k_index_stage_base = smem_sparse_indices_addr;
                #pragma unroll
                for (int qk_stage = 0; qk_stage < 4; qk_stage++) {
                    if (qk_stage == load_warp_rank) {
                        mbarrier_wait_hint(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty, 10000000);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                        }
                        int k_dst = smem_kv_addr + load_kv_stage * 16384;
                        int group = lane;
                        int k_group_limit = ((work_idx_1 == (unsigned int)bid) ? 16 : 32);
                        if (group < k_group_limit) {
                            int group_offset = k_cta_offset + group * 4;
                            int raw_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                                : "r"(k_index_stage_base + group_offset * 4));
                            int row0 = ((raw_rows[0] >= 0) ? raw_rows[0] : 0);
                            int row1 = ((raw_rows[1] >= 0) ? raw_rows[1] : 0);
                            int row2 = ((raw_rows[2] >= 0) ? raw_rows[2] : 0);
                            int row3 = ((raw_rows[3] >= 0) ? raw_rows[3] : 0);
                            if (first_tile == 0) {
                                tma_gather4_gmem2smem(k_dst + group * 512, (&tmap_swa_kv), qk_stage * 128, raw_rows[0], raw_rows[1], raw_rows[2], raw_rows[3], kv_full_addr + (load_kv_stage) * 8);
                            } else {
                                tma_gather4_gmem2smem(k_dst + group * 512, (&tmap_compressed_kv), qk_stage * 128, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                            }
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                }
                #pragma unroll 1
                for (int tile = 1; tile < num_kv_tiles; tile++) {
                    if ((0 & (int)((tile & 1) == 0)) != 0) {
                        mbarrier_wait_hint(index_full_addr, _phase_index_full, 10000000);
                    }
                    k_index_stage_base = smem_sparse_indices_addr;
                    #pragma unroll
                    for (int qk_stage_1 = 0; qk_stage_1 < 4; qk_stage_1++) {
                        if (qk_stage_1 == load_warp_rank) {
                            mbarrier_wait_hint(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty, 10000000);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                            }
                            int k_dst_1 = smem_kv_addr + load_kv_stage * 16384;
                            int group_1 = lane;
                            if (group_1 < 32) {
                                int group_offset_1 = tile * 128 + k_cta_offset + group_1 * 4;
                                int raw_rows_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                    : "r"(k_index_stage_base + group_offset_1 * 4));
                                int row0_1 = ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : 0);
                                int row1_1 = ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : 0);
                                int row2_1 = ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : 0);
                                int row3_1 = ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : 0);
                                tma_gather4_gmem2smem(k_dst_1 + group_1 * 512, (&tmap_compressed_kv), qk_stage_1 * 128, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (load_kv_stage) * 8);
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                    int prev_tile = tile - 1;
                    int global_prev_tile = first_tile + prev_tile;
                    int v_index_stage_base = smem_sparse_indices_addr;
                    #pragma unroll
                    for (int pv_stage = 0; pv_stage < 2; pv_stage++) {
                        if (pv_stage == load_warp_rank) {
                            mbarrier_wait_hint(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty, 10000000);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                            }
                            int v_dst = smem_kv_addr + load_kv_stage * 16384;
                            int v_col = v_chunk * 256 + pv_stage * 128;
                            int group_2 = lane;
                            int group_offset_2 = prev_tile * 128 + group_2 * 4;
                            int raw_rows_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 3]))
                                : "r"(v_index_stage_base + group_offset_2 * 4));
                            int row0_2 = ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : 0);
                            int row1_2 = ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : 0);
                            int row2_2 = ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : 0);
                            int row3_2 = ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : 0);
                            if (global_prev_tile == 0) {
                                tma_gather4_gmem2smem(v_dst + group_2 * 512, (&tmap_swa_kv), v_col, row0_2, row1_2, row2_2, row3_2, kv_full_addr + (load_kv_stage) * 8);
                            } else {
                                tma_gather4_gmem2smem(v_dst + group_2 * 512, (&tmap_compressed_kv), v_col, row0_2, row1_2, row2_2, row3_2, kv_full_addr + (load_kv_stage) * 8);
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
                int last_tile = num_kv_tiles - 1;
                int global_last_tile = first_tile + last_tile;
                #pragma unroll
                for (int pv_stage_1 = 0; pv_stage_1 < 2; pv_stage_1++) {
                    if (pv_stage_1 == load_warp_rank) {
                        mbarrier_wait_hint(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty, 10000000);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                        }
                        int v_dst_1 = smem_kv_addr + load_kv_stage * 16384;
                        int v_col_1 = v_chunk * 256 + pv_stage_1 * 128;
                        int group_3 = lane;
                        int group_offset_3 = last_tile * 128 + group_3 * 4;
                        int raw_rows_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 3]))
                            : "r"(smem_sparse_indices_addr + (unsigned int)(group_offset_3 * 4)));
                        int row0_3 = ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : 0);
                        int row1_3 = ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : 0);
                        int row2_3 = ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : 0);
                        int row3_3 = ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : 0);
                        if (global_last_tile == 0) {
                            tma_gather4_gmem2smem(v_dst_1 + group_3 * 512, (&tmap_swa_kv), v_col_1, row0_3, row1_3, row2_3, row3_3, kv_full_addr + (load_kv_stage) * 8);
                        } else {
                            tma_gather4_gmem2smem(v_dst_1 + group_3 * 512, (&tmap_compressed_kv), v_col_1, row0_3, row1_3, row2_3, row3_3, kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.442695f;
            float epi_output_scale = bmm2_scale[0];
            const int wg_dummy_inc = 0;
            int all_num_kv_tiles_1 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base = warp % 4 * 32;
            const int tmem_score_row_base = tmem_row_base;
            const int n_half = 0;
            const int my_row = tmem_row_base + lane;
            const int stats_row = n_half * 64 + my_row;
            int softmax_tile_cursor = 0;
            unsigned int window_cons_stage = 0;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_window_full = 0;
            unsigned int _phase_o_done_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid; work_idx_2 < total_work_items; work_idx_2 += num_bids) {
                int split_idx_1 = 0;
                int query_idx_1 = work_idx_2 >> 1;
                int tiles_per_split_1 = all_num_kv_tiles_1;
                int first_tile_1 = split_idx_1 * tiles_per_split_1;
                int num_kv_tiles_1 = tiles_per_split_1;
                int _max_0 = ((sparse_topk_lens[query_idx_1] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_1] + sparse_topk_lens_offset) : (0));
                int active_topk = ((_max_0) < (sparse_topk) ? (_max_0) : (sparse_topk));
                float seed_zero[4];
                #pragma unroll
                for (int seed_c4 = 0; seed_c4 < 4; seed_c4++) {
                    seed_zero[seed_c4] = 0.0f;
                }
                #pragma unroll
                for (int seed_half = 0; seed_half < 2; seed_half++) {
                    #pragma unroll
                    for (int seed_c = 0; seed_c < 8; seed_c++) {
                        int seed_addr = taddr + (unsigned int)(((seed_half == 0) ? 96 : 224)) + (unsigned int)(seed_c * 4) + (unsigned int)(tmem_row_base << 16);
                        tmem_st_x4_f32(seed_addr, seed_zero);
                    }
                }
                #pragma unroll
                for (int seed_half_1 = 0; seed_half_1 < 2; seed_half_1++) {
                    #pragma unroll
                    for (int seed_c_1 = 16; seed_c_1 < 24; seed_c_1++) {
                        int seed_addr_1 = taddr + (unsigned int)(((seed_half_1 == 0) ? 0 : 128)) + (unsigned int)(seed_c_1 * 4) + (unsigned int)(tmem_row_base << 16);
                        tmem_st_x4_f32(seed_addr_1, seed_zero);
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_seeded_addr);
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                if (work_idx_2 == (unsigned int)bid) {
                    const int helper_stage = warp % 4;
                    int helper_dst = smem_kv_addr + (unsigned int)(helper_stage * 16384);
                    int helper_group = lane;
                    if (helper_group >= 16) {
                        int helper_rows[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&helper_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&helper_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&helper_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&helper_rows[(0) + 3]))
                            : "r"(smem_sparse_indices_addr + (unsigned int)(helper_group * 16)));
                        int hrow0 = ((helper_rows[0] >= 0) ? helper_rows[0] : 0);
                        int hrow1 = ((helper_rows[1] >= 0) ? helper_rows[1] : 0);
                        int hrow2 = ((helper_rows[2] >= 0) ? helper_rows[2] : 0);
                        int hrow3 = ((helper_rows[3] >= 0) ? helper_rows[3] : 0);
                        if (first_tile_1 == 0) {
                            tma_gather4_gmem2smem(helper_dst + helper_group * 512, (&tmap_swa_kv), helper_stage * 128, helper_rows[0], helper_rows[1], helper_rows[2], helper_rows[3], kv_full_addr + ((int)((unsigned int)warp % 4)) * 8);
                        } else {
                            tma_gather4_gmem2smem(helper_dst + helper_group * 512, (&tmap_compressed_kv), helper_stage * 128, hrow0, hrow1, hrow2, hrow3, kv_full_addr + ((int)((unsigned int)warp % 4)) * 8);
                        }
                    }
                }
                mbarrier_wait_hint(q_full_addr, _phase_q_full_0_1, 10000000);
                _phase_q_full_0_1 ^= 1;
                mbarrier_wait_hint(window_full_addr + (window_cons_stage) * 8, _phase_window_full, 10000000);
                int swa_visible = smem_swa_window[window_cons_stage * 4];
                mbarrier_arrive(window_empty_addr + (window_cons_stage) * 8);
                window_cons_stage += 1;
                if (window_cons_stage == 2) { window_cons_stage = 0; _phase_window_full ^= 1; }
                float row_max_val = -CUDART_INF_F;
                float row_sum_val = 0.0f;
                int sink_head = my_row;
                if (has_sinks != 0 && sink_head < num_heads && split_idx_1 == 0) {
                    row_max_val = sinks[sink_head] * 1.442695f / softmax_scale_log2;
                    row_sum_val = 256.0f;
                }
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int pipeline_tile = softmax_tile_cursor;
                    int phase = softmax_tile_cursor % 2;
                    int _stage_s_full = (int)((unsigned int)softmax_tile_cursor % 2);
                    int _parity_s_full = (int)((unsigned int)softmax_tile_cursor / 2 % 2);
                    mbarrier_wait_hint(s_full_addr + (_stage_s_full) * 8, _parity_s_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_off = ((phase != 0) ? 128 : 0);
                    int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_score_row_base << 16);
                    float new_max = row_max_val;
                    int valid_sparse_cols = ((active_topk < sparse_topk) ? active_topk : sparse_topk);
                    valid_sparse_cols = valid_sparse_cols - (first_tile_1 + tile_1) * 128 - n_half * 64;
                    if (valid_sparse_cols < 0) {
                        valid_sparse_cols = 0;
                    }
                    if (valid_sparse_cols > 128) {
                        valid_sparse_cols = 128;
                    }
                    if (first_tile_1 + tile_1 == 0) {
                        int swa_visible_cols = swa_visible - n_half * 64;
                        if (swa_visible_cols < 0) {
                            swa_visible_cols = 0;
                        }
                        if (valid_sparse_cols > swa_visible_cols) {
                            valid_sparse_cols = swa_visible_cols;
                        }
                    }
                    int index_tile_addr = smem_sparse_indices_addr + (unsigned int)(tile_1 * 512);
                    int index_lane_addr = index_tile_addr + (n_half * 64 + lane) * 4;
                    int staged_index[4];
                    unsigned int invalid_cols[4];
                    unsigned int any_invalid = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; w++) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[w])) : "r"(index_lane_addr + w * 128));
                        unsigned int _warp_vote_1 = __ballot_sync(0xFFFFFFFF, staged_index[w] < 0);
                        invalid_cols[w] = _warp_vote_1;
                        any_invalid = any_invalid | invalid_cols[w];
                    }
                    float sv[32];
                    float sv_red0;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv_red0)
                        : "r"(s_base));
                    float sv_f1[32];
                    float sv_red1;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                        : "=f"(sv_f1[0]), "=f"(sv_f1[1]), "=f"(sv_f1[2]), "=f"(sv_f1[3]), "=f"(sv_f1[4]), "=f"(sv_f1[5]), "=f"(sv_f1[6]), "=f"(sv_f1[7]), "=f"(sv_f1[8]), "=f"(sv_f1[9]), "=f"(sv_f1[10]), "=f"(sv_f1[11]), "=f"(sv_f1[12]), "=f"(sv_f1[13]), "=f"(sv_f1[14]), "=f"(sv_f1[15]), "=f"(sv_f1[16]), "=f"(sv_f1[17]), "=f"(sv_f1[18]), "=f"(sv_f1[19]), "=f"(sv_f1[20]), "=f"(sv_f1[21]), "=f"(sv_f1[22]), "=f"(sv_f1[23]), "=f"(sv_f1[24]), "=f"(sv_f1[25]), "=f"(sv_f1[26]), "=f"(sv_f1[27]), "=f"(sv_f1[28]), "=f"(sv_f1[29]), "=f"(sv_f1[30]), "=f"(sv_f1[31]), "=f"(sv_red1)
                        : "r"(s_base + 32));
                    float sv_f2[32];
                    float sv_red2;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                        : "=f"(sv_f2[0]), "=f"(sv_f2[1]), "=f"(sv_f2[2]), "=f"(sv_f2[3]), "=f"(sv_f2[4]), "=f"(sv_f2[5]), "=f"(sv_f2[6]), "=f"(sv_f2[7]), "=f"(sv_f2[8]), "=f"(sv_f2[9]), "=f"(sv_f2[10]), "=f"(sv_f2[11]), "=f"(sv_f2[12]), "=f"(sv_f2[13]), "=f"(sv_f2[14]), "=f"(sv_f2[15]), "=f"(sv_f2[16]), "=f"(sv_f2[17]), "=f"(sv_f2[18]), "=f"(sv_f2[19]), "=f"(sv_f2[20]), "=f"(sv_f2[21]), "=f"(sv_f2[22]), "=f"(sv_f2[23]), "=f"(sv_f2[24]), "=f"(sv_f2[25]), "=f"(sv_f2[26]), "=f"(sv_f2[27]), "=f"(sv_f2[28]), "=f"(sv_f2[29]), "=f"(sv_f2[30]), "=f"(sv_f2[31]), "=f"(sv_red2)
                        : "r"(s_base + 64));
                    float sv_f3[32];
                    float sv_red3;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                        : "=f"(sv_f3[0]), "=f"(sv_f3[1]), "=f"(sv_f3[2]), "=f"(sv_f3[3]), "=f"(sv_f3[4]), "=f"(sv_f3[5]), "=f"(sv_f3[6]), "=f"(sv_f3[7]), "=f"(sv_f3[8]), "=f"(sv_f3[9]), "=f"(sv_f3[10]), "=f"(sv_f3[11]), "=f"(sv_f3[12]), "=f"(sv_f3[13]), "=f"(sv_f3[14]), "=f"(sv_f3[15]), "=f"(sv_f3[16]), "=f"(sv_f3[17]), "=f"(sv_f3[18]), "=f"(sv_f3[19]), "=f"(sv_f3[20]), "=f"(sv_f3[21]), "=f"(sv_f3[22]), "=f"(sv_f3[23]), "=f"(sv_f3[24]), "=f"(sv_f3[25]), "=f"(sv_f3[26]), "=f"(sv_f3[27]), "=f"(sv_f3[28]), "=f"(sv_f3[29]), "=f"(sv_f3[30]), "=f"(sv_f3[31]), "=f"(sv_red3)
                        : "r"(s_base + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], s_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int ldred_full = any_invalid == 0 && valid_sparse_cols >= 128;
                    if (ldred_full != 0) {
                        new_max = max_noftz(new_max, sv_red0);
                        new_max = max_noftz(new_max, sv_red1);
                        new_max = max_noftz(new_max, sv_red2);
                        new_max = max_noftz(new_max, sv_red3);
                    } else {
                        int frag_valid_0 = valid_sparse_cols;
                        uint32_t _thresh_mask_2;
                        {
                            int _lim_3 = frag_valid_0;
                            if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                            else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                            }
                        }
                        if (!(_thresh_mask_2 & (1u << 0))) sv[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 1))) sv[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 2))) sv[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 3))) sv[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 4))) sv[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 5))) sv[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 6))) sv[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 7))) sv[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 8))) sv[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 9))) sv[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 10))) sv[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 11))) sv[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 12))) sv[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 13))) sv[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 14))) sv[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 15))) sv[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 16))) sv[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 17))) sv[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 18))) sv[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 19))) sv[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 20))) sv[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 21))) sv[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 22))) sv[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 23))) sv[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 24))) sv[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 25))) sv[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 26))) sv[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 27))) sv[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 28))) sv[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 29))) sv[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 30))) sv[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 31))) sv[31] = -CUDART_INF_F;
                        if (any_invalid != 0) {
                            unsigned int frag_mask_word_0 = invalid_cols[0];
                            sv[0] = (((frag_mask_word_0 & 1) != 0) ? -CUDART_INF_F : sv[0]);
                            sv[1] = (((frag_mask_word_0 >> 1 & 1) != 0) ? -CUDART_INF_F : sv[1]);
                            sv[2] = (((frag_mask_word_0 >> 2 & 1) != 0) ? -CUDART_INF_F : sv[2]);
                            sv[3] = (((frag_mask_word_0 >> 3 & 1) != 0) ? -CUDART_INF_F : sv[3]);
                            sv[4] = (((frag_mask_word_0 >> 4 & 1) != 0) ? -CUDART_INF_F : sv[4]);
                            sv[5] = (((frag_mask_word_0 >> 5 & 1) != 0) ? -CUDART_INF_F : sv[5]);
                            sv[6] = (((frag_mask_word_0 >> 6 & 1) != 0) ? -CUDART_INF_F : sv[6]);
                            sv[7] = (((frag_mask_word_0 >> 7 & 1) != 0) ? -CUDART_INF_F : sv[7]);
                            sv[8] = (((frag_mask_word_0 >> 8 & 1) != 0) ? -CUDART_INF_F : sv[8]);
                            sv[9] = (((frag_mask_word_0 >> 9 & 1) != 0) ? -CUDART_INF_F : sv[9]);
                            sv[10] = (((frag_mask_word_0 >> 10 & 1) != 0) ? -CUDART_INF_F : sv[10]);
                            sv[11] = (((frag_mask_word_0 >> 11 & 1) != 0) ? -CUDART_INF_F : sv[11]);
                            sv[12] = (((frag_mask_word_0 >> 12 & 1) != 0) ? -CUDART_INF_F : sv[12]);
                            sv[13] = (((frag_mask_word_0 >> 13 & 1) != 0) ? -CUDART_INF_F : sv[13]);
                            sv[14] = (((frag_mask_word_0 >> 14 & 1) != 0) ? -CUDART_INF_F : sv[14]);
                            sv[15] = (((frag_mask_word_0 >> 15 & 1) != 0) ? -CUDART_INF_F : sv[15]);
                            sv[16] = (((frag_mask_word_0 >> 16 & 1) != 0) ? -CUDART_INF_F : sv[16]);
                            sv[17] = (((frag_mask_word_0 >> 17 & 1) != 0) ? -CUDART_INF_F : sv[17]);
                            sv[18] = (((frag_mask_word_0 >> 18 & 1) != 0) ? -CUDART_INF_F : sv[18]);
                            sv[19] = (((frag_mask_word_0 >> 19 & 1) != 0) ? -CUDART_INF_F : sv[19]);
                            sv[20] = (((frag_mask_word_0 >> 20 & 1) != 0) ? -CUDART_INF_F : sv[20]);
                            sv[21] = (((frag_mask_word_0 >> 21 & 1) != 0) ? -CUDART_INF_F : sv[21]);
                            sv[22] = (((frag_mask_word_0 >> 22 & 1) != 0) ? -CUDART_INF_F : sv[22]);
                            sv[23] = (((frag_mask_word_0 >> 23 & 1) != 0) ? -CUDART_INF_F : sv[23]);
                            sv[24] = (((frag_mask_word_0 >> 24 & 1) != 0) ? -CUDART_INF_F : sv[24]);
                            sv[25] = (((frag_mask_word_0 >> 25 & 1) != 0) ? -CUDART_INF_F : sv[25]);
                            sv[26] = (((frag_mask_word_0 >> 26 & 1) != 0) ? -CUDART_INF_F : sv[26]);
                            sv[27] = (((frag_mask_word_0 >> 27 & 1) != 0) ? -CUDART_INF_F : sv[27]);
                            sv[28] = (((frag_mask_word_0 >> 28 & 1) != 0) ? -CUDART_INF_F : sv[28]);
                            sv[29] = (((frag_mask_word_0 >> 29 & 1) != 0) ? -CUDART_INF_F : sv[29]);
                            sv[30] = (((frag_mask_word_0 >> 30 & 1) != 0) ? -CUDART_INF_F : sv[30]);
                            sv[31] = (((frag_mask_word_0 >> 31 & 1) != 0) ? -CUDART_INF_F : sv[31]);
                        }
                        float2 _reg_reduce_max2_5 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[0], sv[1]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[2], sv[3]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[4], sv[5]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[6], sv[7]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[8], sv[9]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[10], sv[11]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[12], sv[13]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[14], sv[15]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[16], sv[17]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[18], sv[19]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[20], sv[21]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[22], sv[23]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[24], sv[25]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[26], sv[27]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[28], sv[29]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[30], sv[31]));
                        float _reg_array_reduce_4 = row_max_reduce(_reg_reduce_max2_5);
                        new_max = max_noftz(new_max, _reg_array_reduce_4);
                        int frag_valid_1 = valid_sparse_cols - 32;
                        uint32_t _thresh_mask_6;
                        {
                            int _lim_7 = frag_valid_1;
                            if (_lim_7 <= 0) { _thresh_mask_6 = 0u; }
                            else if (_lim_7 >= 32) { _thresh_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_6) : "r"(_lim_7));
                            }
                        }
                        if (!(_thresh_mask_6 & (1u << 0))) sv_f1[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 1))) sv_f1[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 2))) sv_f1[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 3))) sv_f1[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 4))) sv_f1[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 5))) sv_f1[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 6))) sv_f1[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 7))) sv_f1[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 8))) sv_f1[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 9))) sv_f1[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 10))) sv_f1[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 11))) sv_f1[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 12))) sv_f1[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 13))) sv_f1[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 14))) sv_f1[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 15))) sv_f1[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 16))) sv_f1[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 17))) sv_f1[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 18))) sv_f1[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 19))) sv_f1[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 20))) sv_f1[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 21))) sv_f1[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 22))) sv_f1[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 23))) sv_f1[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 24))) sv_f1[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 25))) sv_f1[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 26))) sv_f1[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 27))) sv_f1[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 28))) sv_f1[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 29))) sv_f1[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 30))) sv_f1[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 31))) sv_f1[31] = -CUDART_INF_F;
                        if (any_invalid != 0) {
                            unsigned int frag_mask_word_1 = invalid_cols[1];
                            sv_f1[0] = (((frag_mask_word_1 & 1) != 0) ? -CUDART_INF_F : sv_f1[0]);
                            sv_f1[1] = (((frag_mask_word_1 >> 1 & 1) != 0) ? -CUDART_INF_F : sv_f1[1]);
                            sv_f1[2] = (((frag_mask_word_1 >> 2 & 1) != 0) ? -CUDART_INF_F : sv_f1[2]);
                            sv_f1[3] = (((frag_mask_word_1 >> 3 & 1) != 0) ? -CUDART_INF_F : sv_f1[3]);
                            sv_f1[4] = (((frag_mask_word_1 >> 4 & 1) != 0) ? -CUDART_INF_F : sv_f1[4]);
                            sv_f1[5] = (((frag_mask_word_1 >> 5 & 1) != 0) ? -CUDART_INF_F : sv_f1[5]);
                            sv_f1[6] = (((frag_mask_word_1 >> 6 & 1) != 0) ? -CUDART_INF_F : sv_f1[6]);
                            sv_f1[7] = (((frag_mask_word_1 >> 7 & 1) != 0) ? -CUDART_INF_F : sv_f1[7]);
                            sv_f1[8] = (((frag_mask_word_1 >> 8 & 1) != 0) ? -CUDART_INF_F : sv_f1[8]);
                            sv_f1[9] = (((frag_mask_word_1 >> 9 & 1) != 0) ? -CUDART_INF_F : sv_f1[9]);
                            sv_f1[10] = (((frag_mask_word_1 >> 10 & 1) != 0) ? -CUDART_INF_F : sv_f1[10]);
                            sv_f1[11] = (((frag_mask_word_1 >> 11 & 1) != 0) ? -CUDART_INF_F : sv_f1[11]);
                            sv_f1[12] = (((frag_mask_word_1 >> 12 & 1) != 0) ? -CUDART_INF_F : sv_f1[12]);
                            sv_f1[13] = (((frag_mask_word_1 >> 13 & 1) != 0) ? -CUDART_INF_F : sv_f1[13]);
                            sv_f1[14] = (((frag_mask_word_1 >> 14 & 1) != 0) ? -CUDART_INF_F : sv_f1[14]);
                            sv_f1[15] = (((frag_mask_word_1 >> 15 & 1) != 0) ? -CUDART_INF_F : sv_f1[15]);
                            sv_f1[16] = (((frag_mask_word_1 >> 16 & 1) != 0) ? -CUDART_INF_F : sv_f1[16]);
                            sv_f1[17] = (((frag_mask_word_1 >> 17 & 1) != 0) ? -CUDART_INF_F : sv_f1[17]);
                            sv_f1[18] = (((frag_mask_word_1 >> 18 & 1) != 0) ? -CUDART_INF_F : sv_f1[18]);
                            sv_f1[19] = (((frag_mask_word_1 >> 19 & 1) != 0) ? -CUDART_INF_F : sv_f1[19]);
                            sv_f1[20] = (((frag_mask_word_1 >> 20 & 1) != 0) ? -CUDART_INF_F : sv_f1[20]);
                            sv_f1[21] = (((frag_mask_word_1 >> 21 & 1) != 0) ? -CUDART_INF_F : sv_f1[21]);
                            sv_f1[22] = (((frag_mask_word_1 >> 22 & 1) != 0) ? -CUDART_INF_F : sv_f1[22]);
                            sv_f1[23] = (((frag_mask_word_1 >> 23 & 1) != 0) ? -CUDART_INF_F : sv_f1[23]);
                            sv_f1[24] = (((frag_mask_word_1 >> 24 & 1) != 0) ? -CUDART_INF_F : sv_f1[24]);
                            sv_f1[25] = (((frag_mask_word_1 >> 25 & 1) != 0) ? -CUDART_INF_F : sv_f1[25]);
                            sv_f1[26] = (((frag_mask_word_1 >> 26 & 1) != 0) ? -CUDART_INF_F : sv_f1[26]);
                            sv_f1[27] = (((frag_mask_word_1 >> 27 & 1) != 0) ? -CUDART_INF_F : sv_f1[27]);
                            sv_f1[28] = (((frag_mask_word_1 >> 28 & 1) != 0) ? -CUDART_INF_F : sv_f1[28]);
                            sv_f1[29] = (((frag_mask_word_1 >> 29 & 1) != 0) ? -CUDART_INF_F : sv_f1[29]);
                            sv_f1[30] = (((frag_mask_word_1 >> 30 & 1) != 0) ? -CUDART_INF_F : sv_f1[30]);
                            sv_f1[31] = (((frag_mask_word_1 >> 31 & 1) != 0) ? -CUDART_INF_F : sv_f1[31]);
                        }
                        float2 _reg_reduce_max2_9 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[0], sv_f1[1]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[2], sv_f1[3]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[4], sv_f1[5]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[6], sv_f1[7]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[8], sv_f1[9]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[10], sv_f1[11]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[12], sv_f1[13]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[14], sv_f1[15]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[16], sv_f1[17]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[18], sv_f1[19]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[20], sv_f1[21]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[22], sv_f1[23]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[24], sv_f1[25]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[26], sv_f1[27]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv_f1[28], sv_f1[29]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv_f1[30], sv_f1[31]));
                        float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                        new_max = max_noftz(new_max, _reg_array_reduce_8);
                        int frag_valid_2 = valid_sparse_cols - 64;
                        uint32_t _thresh_mask_10;
                        {
                            int _lim_11 = frag_valid_2;
                            if (_lim_11 <= 0) { _thresh_mask_10 = 0u; }
                            else if (_lim_11 >= 32) { _thresh_mask_10 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_10) : "r"(_lim_11));
                            }
                        }
                        if (!(_thresh_mask_10 & (1u << 0))) sv_f2[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 1))) sv_f2[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 2))) sv_f2[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 3))) sv_f2[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 4))) sv_f2[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 5))) sv_f2[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 6))) sv_f2[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 7))) sv_f2[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 8))) sv_f2[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 9))) sv_f2[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 10))) sv_f2[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 11))) sv_f2[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 12))) sv_f2[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 13))) sv_f2[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 14))) sv_f2[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 15))) sv_f2[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 16))) sv_f2[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 17))) sv_f2[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 18))) sv_f2[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 19))) sv_f2[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 20))) sv_f2[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 21))) sv_f2[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 22))) sv_f2[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 23))) sv_f2[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 24))) sv_f2[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 25))) sv_f2[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 26))) sv_f2[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 27))) sv_f2[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 28))) sv_f2[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 29))) sv_f2[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 30))) sv_f2[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_10 & (1u << 31))) sv_f2[31] = -CUDART_INF_F;
                        if (any_invalid != 0) {
                            unsigned int frag_mask_word_2 = invalid_cols[2];
                            sv_f2[0] = (((frag_mask_word_2 & 1) != 0) ? -CUDART_INF_F : sv_f2[0]);
                            sv_f2[1] = (((frag_mask_word_2 >> 1 & 1) != 0) ? -CUDART_INF_F : sv_f2[1]);
                            sv_f2[2] = (((frag_mask_word_2 >> 2 & 1) != 0) ? -CUDART_INF_F : sv_f2[2]);
                            sv_f2[3] = (((frag_mask_word_2 >> 3 & 1) != 0) ? -CUDART_INF_F : sv_f2[3]);
                            sv_f2[4] = (((frag_mask_word_2 >> 4 & 1) != 0) ? -CUDART_INF_F : sv_f2[4]);
                            sv_f2[5] = (((frag_mask_word_2 >> 5 & 1) != 0) ? -CUDART_INF_F : sv_f2[5]);
                            sv_f2[6] = (((frag_mask_word_2 >> 6 & 1) != 0) ? -CUDART_INF_F : sv_f2[6]);
                            sv_f2[7] = (((frag_mask_word_2 >> 7 & 1) != 0) ? -CUDART_INF_F : sv_f2[7]);
                            sv_f2[8] = (((frag_mask_word_2 >> 8 & 1) != 0) ? -CUDART_INF_F : sv_f2[8]);
                            sv_f2[9] = (((frag_mask_word_2 >> 9 & 1) != 0) ? -CUDART_INF_F : sv_f2[9]);
                            sv_f2[10] = (((frag_mask_word_2 >> 10 & 1) != 0) ? -CUDART_INF_F : sv_f2[10]);
                            sv_f2[11] = (((frag_mask_word_2 >> 11 & 1) != 0) ? -CUDART_INF_F : sv_f2[11]);
                            sv_f2[12] = (((frag_mask_word_2 >> 12 & 1) != 0) ? -CUDART_INF_F : sv_f2[12]);
                            sv_f2[13] = (((frag_mask_word_2 >> 13 & 1) != 0) ? -CUDART_INF_F : sv_f2[13]);
                            sv_f2[14] = (((frag_mask_word_2 >> 14 & 1) != 0) ? -CUDART_INF_F : sv_f2[14]);
                            sv_f2[15] = (((frag_mask_word_2 >> 15 & 1) != 0) ? -CUDART_INF_F : sv_f2[15]);
                            sv_f2[16] = (((frag_mask_word_2 >> 16 & 1) != 0) ? -CUDART_INF_F : sv_f2[16]);
                            sv_f2[17] = (((frag_mask_word_2 >> 17 & 1) != 0) ? -CUDART_INF_F : sv_f2[17]);
                            sv_f2[18] = (((frag_mask_word_2 >> 18 & 1) != 0) ? -CUDART_INF_F : sv_f2[18]);
                            sv_f2[19] = (((frag_mask_word_2 >> 19 & 1) != 0) ? -CUDART_INF_F : sv_f2[19]);
                            sv_f2[20] = (((frag_mask_word_2 >> 20 & 1) != 0) ? -CUDART_INF_F : sv_f2[20]);
                            sv_f2[21] = (((frag_mask_word_2 >> 21 & 1) != 0) ? -CUDART_INF_F : sv_f2[21]);
                            sv_f2[22] = (((frag_mask_word_2 >> 22 & 1) != 0) ? -CUDART_INF_F : sv_f2[22]);
                            sv_f2[23] = (((frag_mask_word_2 >> 23 & 1) != 0) ? -CUDART_INF_F : sv_f2[23]);
                            sv_f2[24] = (((frag_mask_word_2 >> 24 & 1) != 0) ? -CUDART_INF_F : sv_f2[24]);
                            sv_f2[25] = (((frag_mask_word_2 >> 25 & 1) != 0) ? -CUDART_INF_F : sv_f2[25]);
                            sv_f2[26] = (((frag_mask_word_2 >> 26 & 1) != 0) ? -CUDART_INF_F : sv_f2[26]);
                            sv_f2[27] = (((frag_mask_word_2 >> 27 & 1) != 0) ? -CUDART_INF_F : sv_f2[27]);
                            sv_f2[28] = (((frag_mask_word_2 >> 28 & 1) != 0) ? -CUDART_INF_F : sv_f2[28]);
                            sv_f2[29] = (((frag_mask_word_2 >> 29 & 1) != 0) ? -CUDART_INF_F : sv_f2[29]);
                            sv_f2[30] = (((frag_mask_word_2 >> 30 & 1) != 0) ? -CUDART_INF_F : sv_f2[30]);
                            sv_f2[31] = (((frag_mask_word_2 >> 31 & 1) != 0) ? -CUDART_INF_F : sv_f2[31]);
                        }
                        float2 _reg_reduce_max2_13 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[0], sv_f2[1]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[2], sv_f2[3]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[4], sv_f2[5]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[6], sv_f2[7]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[8], sv_f2[9]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[10], sv_f2[11]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[12], sv_f2[13]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[14], sv_f2[15]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[16], sv_f2[17]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[18], sv_f2[19]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[20], sv_f2[21]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[22], sv_f2[23]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[24], sv_f2[25]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[26], sv_f2[27]));
                        _reg_reduce_max2_13.x = max_noftz(_reg_reduce_max2_13.x, max_noftz(sv_f2[28], sv_f2[29]));
                        _reg_reduce_max2_13.y = max_noftz(_reg_reduce_max2_13.y, max_noftz(sv_f2[30], sv_f2[31]));
                        float _reg_array_reduce_12 = row_max_reduce(_reg_reduce_max2_13);
                        new_max = max_noftz(new_max, _reg_array_reduce_12);
                        int frag_valid_3 = valid_sparse_cols - 96;
                        uint32_t _thresh_mask_14;
                        {
                            int _lim_15 = frag_valid_3;
                            if (_lim_15 <= 0) { _thresh_mask_14 = 0u; }
                            else if (_lim_15 >= 32) { _thresh_mask_14 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_14) : "r"(_lim_15));
                            }
                        }
                        if (!(_thresh_mask_14 & (1u << 0))) sv_f3[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 1))) sv_f3[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 2))) sv_f3[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 3))) sv_f3[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 4))) sv_f3[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 5))) sv_f3[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 6))) sv_f3[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 7))) sv_f3[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 8))) sv_f3[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 9))) sv_f3[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 10))) sv_f3[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 11))) sv_f3[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 12))) sv_f3[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 13))) sv_f3[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 14))) sv_f3[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 15))) sv_f3[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 16))) sv_f3[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 17))) sv_f3[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 18))) sv_f3[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 19))) sv_f3[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 20))) sv_f3[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 21))) sv_f3[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 22))) sv_f3[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 23))) sv_f3[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 24))) sv_f3[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 25))) sv_f3[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 26))) sv_f3[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 27))) sv_f3[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 28))) sv_f3[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 29))) sv_f3[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 30))) sv_f3[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_14 & (1u << 31))) sv_f3[31] = -CUDART_INF_F;
                        if (any_invalid != 0) {
                            unsigned int frag_mask_word_3 = invalid_cols[3];
                            sv_f3[0] = (((frag_mask_word_3 & 1) != 0) ? -CUDART_INF_F : sv_f3[0]);
                            sv_f3[1] = (((frag_mask_word_3 >> 1 & 1) != 0) ? -CUDART_INF_F : sv_f3[1]);
                            sv_f3[2] = (((frag_mask_word_3 >> 2 & 1) != 0) ? -CUDART_INF_F : sv_f3[2]);
                            sv_f3[3] = (((frag_mask_word_3 >> 3 & 1) != 0) ? -CUDART_INF_F : sv_f3[3]);
                            sv_f3[4] = (((frag_mask_word_3 >> 4 & 1) != 0) ? -CUDART_INF_F : sv_f3[4]);
                            sv_f3[5] = (((frag_mask_word_3 >> 5 & 1) != 0) ? -CUDART_INF_F : sv_f3[5]);
                            sv_f3[6] = (((frag_mask_word_3 >> 6 & 1) != 0) ? -CUDART_INF_F : sv_f3[6]);
                            sv_f3[7] = (((frag_mask_word_3 >> 7 & 1) != 0) ? -CUDART_INF_F : sv_f3[7]);
                            sv_f3[8] = (((frag_mask_word_3 >> 8 & 1) != 0) ? -CUDART_INF_F : sv_f3[8]);
                            sv_f3[9] = (((frag_mask_word_3 >> 9 & 1) != 0) ? -CUDART_INF_F : sv_f3[9]);
                            sv_f3[10] = (((frag_mask_word_3 >> 10 & 1) != 0) ? -CUDART_INF_F : sv_f3[10]);
                            sv_f3[11] = (((frag_mask_word_3 >> 11 & 1) != 0) ? -CUDART_INF_F : sv_f3[11]);
                            sv_f3[12] = (((frag_mask_word_3 >> 12 & 1) != 0) ? -CUDART_INF_F : sv_f3[12]);
                            sv_f3[13] = (((frag_mask_word_3 >> 13 & 1) != 0) ? -CUDART_INF_F : sv_f3[13]);
                            sv_f3[14] = (((frag_mask_word_3 >> 14 & 1) != 0) ? -CUDART_INF_F : sv_f3[14]);
                            sv_f3[15] = (((frag_mask_word_3 >> 15 & 1) != 0) ? -CUDART_INF_F : sv_f3[15]);
                            sv_f3[16] = (((frag_mask_word_3 >> 16 & 1) != 0) ? -CUDART_INF_F : sv_f3[16]);
                            sv_f3[17] = (((frag_mask_word_3 >> 17 & 1) != 0) ? -CUDART_INF_F : sv_f3[17]);
                            sv_f3[18] = (((frag_mask_word_3 >> 18 & 1) != 0) ? -CUDART_INF_F : sv_f3[18]);
                            sv_f3[19] = (((frag_mask_word_3 >> 19 & 1) != 0) ? -CUDART_INF_F : sv_f3[19]);
                            sv_f3[20] = (((frag_mask_word_3 >> 20 & 1) != 0) ? -CUDART_INF_F : sv_f3[20]);
                            sv_f3[21] = (((frag_mask_word_3 >> 21 & 1) != 0) ? -CUDART_INF_F : sv_f3[21]);
                            sv_f3[22] = (((frag_mask_word_3 >> 22 & 1) != 0) ? -CUDART_INF_F : sv_f3[22]);
                            sv_f3[23] = (((frag_mask_word_3 >> 23 & 1) != 0) ? -CUDART_INF_F : sv_f3[23]);
                            sv_f3[24] = (((frag_mask_word_3 >> 24 & 1) != 0) ? -CUDART_INF_F : sv_f3[24]);
                            sv_f3[25] = (((frag_mask_word_3 >> 25 & 1) != 0) ? -CUDART_INF_F : sv_f3[25]);
                            sv_f3[26] = (((frag_mask_word_3 >> 26 & 1) != 0) ? -CUDART_INF_F : sv_f3[26]);
                            sv_f3[27] = (((frag_mask_word_3 >> 27 & 1) != 0) ? -CUDART_INF_F : sv_f3[27]);
                            sv_f3[28] = (((frag_mask_word_3 >> 28 & 1) != 0) ? -CUDART_INF_F : sv_f3[28]);
                            sv_f3[29] = (((frag_mask_word_3 >> 29 & 1) != 0) ? -CUDART_INF_F : sv_f3[29]);
                            sv_f3[30] = (((frag_mask_word_3 >> 30 & 1) != 0) ? -CUDART_INF_F : sv_f3[30]);
                            sv_f3[31] = (((frag_mask_word_3 >> 31 & 1) != 0) ? -CUDART_INF_F : sv_f3[31]);
                        }
                        float2 _reg_reduce_max2_17 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[0], sv_f3[1]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[2], sv_f3[3]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[4], sv_f3[5]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[6], sv_f3[7]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[8], sv_f3[9]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[10], sv_f3[11]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[12], sv_f3[13]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[14], sv_f3[15]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[16], sv_f3[17]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[18], sv_f3[19]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[20], sv_f3[21]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[22], sv_f3[23]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[24], sv_f3[25]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[26], sv_f3[27]));
                        _reg_reduce_max2_17.x = max_noftz(_reg_reduce_max2_17.x, max_noftz(sv_f3[28], sv_f3[29]));
                        _reg_reduce_max2_17.y = max_noftz(_reg_reduce_max2_17.y, max_noftz(sv_f3[30], sv_f3[31]));
                        float _reg_array_reduce_16 = row_max_reduce(_reg_reduce_max2_17);
                        new_max = max_noftz(new_max, _reg_array_reduce_16);
                    }
                    float delta = __fmaf_rn(row_max_val, softmax_scale_log2, (-new_max) * softmax_scale_log2);
                    float exp_delta = approx_exp2(delta);
                    float acc_scale = ((row_max_val > -CUDART_INF_F) ? exp_delta : 1.0f);
                    smem_stats_max[phase * 128 + stats_row] = acc_scale;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(stats_addr + ((int)((unsigned int)softmax_tile_cursor % 2)) * 8);
                    row_max_val = new_max;
                    float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    float block_sum = 0.0f;
                    int p_off_all = ((phase != 0) ? 224 : 96);
                    int p_base_all = taddr + (unsigned int)p_off_all + (unsigned int)(tmem_row_base << 16);
                    const float2 _fma_b2_18 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_19 = {8.0f - max_scaled, 8.0f - max_scaled};
                    float2 _fma_pair_20 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_18, _fma_c2_19);
                    sv[0] = _fma_pair_20.x;
                    sv[1] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_18, _fma_c2_19);
                    sv[2] = _fma_pair_21.x;
                    sv[3] = _fma_pair_21.y;
                    float2 _fma_pair_22 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_18, _fma_c2_19);
                    sv[4] = _fma_pair_22.x;
                    sv[5] = _fma_pair_22.y;
                    float2 _fma_pair_23 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_18, _fma_c2_19);
                    sv[6] = _fma_pair_23.x;
                    sv[7] = _fma_pair_23.y;
                    float2 _fma_pair_24 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_18, _fma_c2_19);
                    sv[8] = _fma_pair_24.x;
                    sv[9] = _fma_pair_24.y;
                    float2 _fma_pair_25 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_18, _fma_c2_19);
                    sv[10] = _fma_pair_25.x;
                    sv[11] = _fma_pair_25.y;
                    float2 _fma_pair_26 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_18, _fma_c2_19);
                    sv[12] = _fma_pair_26.x;
                    sv[13] = _fma_pair_26.y;
                    float2 _fma_pair_27 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_18, _fma_c2_19);
                    sv[14] = _fma_pair_27.x;
                    sv[15] = _fma_pair_27.y;
                    float2 _fma_pair_28 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_18, _fma_c2_19);
                    sv[16] = _fma_pair_28.x;
                    sv[17] = _fma_pair_28.y;
                    float2 _fma_pair_29 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_18, _fma_c2_19);
                    sv[18] = _fma_pair_29.x;
                    sv[19] = _fma_pair_29.y;
                    float2 _fma_pair_30 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_18, _fma_c2_19);
                    sv[20] = _fma_pair_30.x;
                    sv[21] = _fma_pair_30.y;
                    float2 _fma_pair_31 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_18, _fma_c2_19);
                    sv[22] = _fma_pair_31.x;
                    sv[23] = _fma_pair_31.y;
                    float2 _fma_pair_32 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_18, _fma_c2_19);
                    sv[24] = _fma_pair_32.x;
                    sv[25] = _fma_pair_32.y;
                    float2 _fma_pair_33 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_18, _fma_c2_19);
                    sv[26] = _fma_pair_33.x;
                    sv[27] = _fma_pair_33.y;
                    float2 _fma_pair_34 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_18, _fma_c2_19);
                    sv[28] = _fma_pair_34.x;
                    sv[29] = _fma_pair_34.y;
                    float2 _fma_pair_35 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_18, _fma_c2_19);
                    sv[30] = _fma_pair_35.x;
                    sv[31] = _fma_pair_35.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv[_le] = approx_exp2(sv[_le]);
                    }
                    float2 _reg_reduce_sum2_37 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[0], sv[1]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[2], sv[3]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[4], sv[5]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[6], sv[7]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[8], sv[9]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[10], sv[11]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[12], sv[13]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[14], sv[15]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[16], sv[17]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[18], sv[19]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[20], sv[21]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[22], sv[23]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[24], sv[25]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[26], sv[27]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[28], sv[29]));
                    _reg_reduce_sum2_37 = add_f32x2(_reg_reduce_sum2_37, make_float2(sv[30], sv[31]));
                    float _reg_array_reduce_36 = _reg_reduce_sum2_37.x + _reg_reduce_sum2_37.y;
                    block_sum = block_sum + _reg_array_reduce_36;
                    unsigned int _fp8_0[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                               "f"(sv[2]), "f"(sv[3]));
                        _fp8_0[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                               "f"(sv[6]), "f"(sv[7]));
                        _fp8_0[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                               "f"(sv[10]), "f"(sv[11]));
                        _fp8_0[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                               "f"(sv[14]), "f"(sv[15]));
                        _fp8_0[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                               "f"(sv[18]), "f"(sv[19]));
                        _fp8_0[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                               "f"(sv[22]), "f"(sv[23]));
                        _fp8_0[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                               "f"(sv[26]), "f"(sv[27]));
                        _fp8_0[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                               "f"(sv[30]), "f"(sv[31]));
                        _fp8_0[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_all, (const uint32_t*)_fp8_0);
                    const float2 _fma_b2_38 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_39 = {8.0f - max_scaled, 8.0f - max_scaled};
                    float2 _fma_pair_40 = fma_f32x2(make_float2(sv_f1[0], sv_f1[1]), _fma_b2_38, _fma_c2_39);
                    sv_f1[0] = _fma_pair_40.x;
                    sv_f1[1] = _fma_pair_40.y;
                    float2 _fma_pair_41 = fma_f32x2(make_float2(sv_f1[2], sv_f1[3]), _fma_b2_38, _fma_c2_39);
                    sv_f1[2] = _fma_pair_41.x;
                    sv_f1[3] = _fma_pair_41.y;
                    float2 _fma_pair_42 = fma_f32x2(make_float2(sv_f1[4], sv_f1[5]), _fma_b2_38, _fma_c2_39);
                    sv_f1[4] = _fma_pair_42.x;
                    sv_f1[5] = _fma_pair_42.y;
                    float2 _fma_pair_43 = fma_f32x2(make_float2(sv_f1[6], sv_f1[7]), _fma_b2_38, _fma_c2_39);
                    sv_f1[6] = _fma_pair_43.x;
                    sv_f1[7] = _fma_pair_43.y;
                    float2 _fma_pair_44 = fma_f32x2(make_float2(sv_f1[8], sv_f1[9]), _fma_b2_38, _fma_c2_39);
                    sv_f1[8] = _fma_pair_44.x;
                    sv_f1[9] = _fma_pair_44.y;
                    float2 _fma_pair_45 = fma_f32x2(make_float2(sv_f1[10], sv_f1[11]), _fma_b2_38, _fma_c2_39);
                    sv_f1[10] = _fma_pair_45.x;
                    sv_f1[11] = _fma_pair_45.y;
                    float2 _fma_pair_46 = fma_f32x2(make_float2(sv_f1[12], sv_f1[13]), _fma_b2_38, _fma_c2_39);
                    sv_f1[12] = _fma_pair_46.x;
                    sv_f1[13] = _fma_pair_46.y;
                    float2 _fma_pair_47 = fma_f32x2(make_float2(sv_f1[14], sv_f1[15]), _fma_b2_38, _fma_c2_39);
                    sv_f1[14] = _fma_pair_47.x;
                    sv_f1[15] = _fma_pair_47.y;
                    float2 _fma_pair_48 = fma_f32x2(make_float2(sv_f1[16], sv_f1[17]), _fma_b2_38, _fma_c2_39);
                    sv_f1[16] = _fma_pair_48.x;
                    sv_f1[17] = _fma_pair_48.y;
                    float2 _fma_pair_49 = fma_f32x2(make_float2(sv_f1[18], sv_f1[19]), _fma_b2_38, _fma_c2_39);
                    sv_f1[18] = _fma_pair_49.x;
                    sv_f1[19] = _fma_pair_49.y;
                    float2 _fma_pair_50 = fma_f32x2(make_float2(sv_f1[20], sv_f1[21]), _fma_b2_38, _fma_c2_39);
                    sv_f1[20] = _fma_pair_50.x;
                    sv_f1[21] = _fma_pair_50.y;
                    float2 _fma_pair_51 = fma_f32x2(make_float2(sv_f1[22], sv_f1[23]), _fma_b2_38, _fma_c2_39);
                    sv_f1[22] = _fma_pair_51.x;
                    sv_f1[23] = _fma_pair_51.y;
                    float2 _fma_pair_52 = fma_f32x2(make_float2(sv_f1[24], sv_f1[25]), _fma_b2_38, _fma_c2_39);
                    sv_f1[24] = _fma_pair_52.x;
                    sv_f1[25] = _fma_pair_52.y;
                    float2 _fma_pair_53 = fma_f32x2(make_float2(sv_f1[26], sv_f1[27]), _fma_b2_38, _fma_c2_39);
                    sv_f1[26] = _fma_pair_53.x;
                    sv_f1[27] = _fma_pair_53.y;
                    float2 _fma_pair_54 = fma_f32x2(make_float2(sv_f1[28], sv_f1[29]), _fma_b2_38, _fma_c2_39);
                    sv_f1[28] = _fma_pair_54.x;
                    sv_f1[29] = _fma_pair_54.y;
                    float2 _fma_pair_55 = fma_f32x2(make_float2(sv_f1[30], sv_f1[31]), _fma_b2_38, _fma_c2_39);
                    sv_f1[30] = _fma_pair_55.x;
                    sv_f1[31] = _fma_pair_55.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv_f1[_le] = approx_exp2(sv_f1[_le]);
                    }
                    float2 _reg_reduce_sum2_57 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[0], sv_f1[1]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[2], sv_f1[3]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[4], sv_f1[5]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[6], sv_f1[7]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[8], sv_f1[9]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[10], sv_f1[11]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[12], sv_f1[13]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[14], sv_f1[15]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[16], sv_f1[17]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[18], sv_f1[19]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[20], sv_f1[21]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[22], sv_f1[23]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[24], sv_f1[25]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[26], sv_f1[27]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[28], sv_f1[29]));
                    _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(sv_f1[30], sv_f1[31]));
                    float _reg_array_reduce_56 = _reg_reduce_sum2_57.x + _reg_reduce_sum2_57.y;
                    block_sum = block_sum + _reg_array_reduce_56;
                    unsigned int _fp8_1[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_f1[0]), "f"(sv_f1[1]),
                                               "f"(sv_f1[2]), "f"(sv_f1[3]));
                        _fp8_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[4]), "f"(sv_f1[5]),
                                               "f"(sv_f1[6]), "f"(sv_f1[7]));
                        _fp8_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[8]), "f"(sv_f1[9]),
                                               "f"(sv_f1[10]), "f"(sv_f1[11]));
                        _fp8_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[12]), "f"(sv_f1[13]),
                                               "f"(sv_f1[14]), "f"(sv_f1[15]));
                        _fp8_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[16]), "f"(sv_f1[17]),
                                               "f"(sv_f1[18]), "f"(sv_f1[19]));
                        _fp8_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[20]), "f"(sv_f1[21]),
                                               "f"(sv_f1[22]), "f"(sv_f1[23]));
                        _fp8_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[24]), "f"(sv_f1[25]),
                                               "f"(sv_f1[26]), "f"(sv_f1[27]));
                        _fp8_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f1[28]), "f"(sv_f1[29]),
                                               "f"(sv_f1[30]), "f"(sv_f1[31]));
                        _fp8_1[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_all + 8, (const uint32_t*)_fp8_1);
                    const float2 _fma_b2_58 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_59 = {8.0f - max_scaled, 8.0f - max_scaled};
                    float2 _fma_pair_60 = fma_f32x2(make_float2(sv_f2[0], sv_f2[1]), _fma_b2_58, _fma_c2_59);
                    sv_f2[0] = _fma_pair_60.x;
                    sv_f2[1] = _fma_pair_60.y;
                    float2 _fma_pair_61 = fma_f32x2(make_float2(sv_f2[2], sv_f2[3]), _fma_b2_58, _fma_c2_59);
                    sv_f2[2] = _fma_pair_61.x;
                    sv_f2[3] = _fma_pair_61.y;
                    float2 _fma_pair_62 = fma_f32x2(make_float2(sv_f2[4], sv_f2[5]), _fma_b2_58, _fma_c2_59);
                    sv_f2[4] = _fma_pair_62.x;
                    sv_f2[5] = _fma_pair_62.y;
                    float2 _fma_pair_63 = fma_f32x2(make_float2(sv_f2[6], sv_f2[7]), _fma_b2_58, _fma_c2_59);
                    sv_f2[6] = _fma_pair_63.x;
                    sv_f2[7] = _fma_pair_63.y;
                    float2 _fma_pair_64 = fma_f32x2(make_float2(sv_f2[8], sv_f2[9]), _fma_b2_58, _fma_c2_59);
                    sv_f2[8] = _fma_pair_64.x;
                    sv_f2[9] = _fma_pair_64.y;
                    float2 _fma_pair_65 = fma_f32x2(make_float2(sv_f2[10], sv_f2[11]), _fma_b2_58, _fma_c2_59);
                    sv_f2[10] = _fma_pair_65.x;
                    sv_f2[11] = _fma_pair_65.y;
                    float2 _fma_pair_66 = fma_f32x2(make_float2(sv_f2[12], sv_f2[13]), _fma_b2_58, _fma_c2_59);
                    sv_f2[12] = _fma_pair_66.x;
                    sv_f2[13] = _fma_pair_66.y;
                    float2 _fma_pair_67 = fma_f32x2(make_float2(sv_f2[14], sv_f2[15]), _fma_b2_58, _fma_c2_59);
                    sv_f2[14] = _fma_pair_67.x;
                    sv_f2[15] = _fma_pair_67.y;
                    float2 _fma_pair_68 = fma_f32x2(make_float2(sv_f2[16], sv_f2[17]), _fma_b2_58, _fma_c2_59);
                    sv_f2[16] = _fma_pair_68.x;
                    sv_f2[17] = _fma_pair_68.y;
                    float2 _fma_pair_69 = fma_f32x2(make_float2(sv_f2[18], sv_f2[19]), _fma_b2_58, _fma_c2_59);
                    sv_f2[18] = _fma_pair_69.x;
                    sv_f2[19] = _fma_pair_69.y;
                    float2 _fma_pair_70 = fma_f32x2(make_float2(sv_f2[20], sv_f2[21]), _fma_b2_58, _fma_c2_59);
                    sv_f2[20] = _fma_pair_70.x;
                    sv_f2[21] = _fma_pair_70.y;
                    float2 _fma_pair_71 = fma_f32x2(make_float2(sv_f2[22], sv_f2[23]), _fma_b2_58, _fma_c2_59);
                    sv_f2[22] = _fma_pair_71.x;
                    sv_f2[23] = _fma_pair_71.y;
                    float2 _fma_pair_72 = fma_f32x2(make_float2(sv_f2[24], sv_f2[25]), _fma_b2_58, _fma_c2_59);
                    sv_f2[24] = _fma_pair_72.x;
                    sv_f2[25] = _fma_pair_72.y;
                    float2 _fma_pair_73 = fma_f32x2(make_float2(sv_f2[26], sv_f2[27]), _fma_b2_58, _fma_c2_59);
                    sv_f2[26] = _fma_pair_73.x;
                    sv_f2[27] = _fma_pair_73.y;
                    float2 _fma_pair_74 = fma_f32x2(make_float2(sv_f2[28], sv_f2[29]), _fma_b2_58, _fma_c2_59);
                    sv_f2[28] = _fma_pair_74.x;
                    sv_f2[29] = _fma_pair_74.y;
                    float2 _fma_pair_75 = fma_f32x2(make_float2(sv_f2[30], sv_f2[31]), _fma_b2_58, _fma_c2_59);
                    sv_f2[30] = _fma_pair_75.x;
                    sv_f2[31] = _fma_pair_75.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv_f2[_le] = approx_exp2(sv_f2[_le]);
                    }
                    float2 _reg_reduce_sum2_77 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[0], sv_f2[1]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[2], sv_f2[3]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[4], sv_f2[5]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[6], sv_f2[7]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[8], sv_f2[9]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[10], sv_f2[11]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[12], sv_f2[13]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[14], sv_f2[15]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[16], sv_f2[17]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[18], sv_f2[19]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[20], sv_f2[21]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[22], sv_f2[23]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[24], sv_f2[25]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[26], sv_f2[27]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[28], sv_f2[29]));
                    _reg_reduce_sum2_77 = add_f32x2(_reg_reduce_sum2_77, make_float2(sv_f2[30], sv_f2[31]));
                    float _reg_array_reduce_76 = _reg_reduce_sum2_77.x + _reg_reduce_sum2_77.y;
                    block_sum = block_sum + _reg_array_reduce_76;
                    unsigned int _fp8_2[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_f2[0]), "f"(sv_f2[1]),
                                               "f"(sv_f2[2]), "f"(sv_f2[3]));
                        _fp8_2[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[4]), "f"(sv_f2[5]),
                                               "f"(sv_f2[6]), "f"(sv_f2[7]));
                        _fp8_2[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[8]), "f"(sv_f2[9]),
                                               "f"(sv_f2[10]), "f"(sv_f2[11]));
                        _fp8_2[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[12]), "f"(sv_f2[13]),
                                               "f"(sv_f2[14]), "f"(sv_f2[15]));
                        _fp8_2[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[16]), "f"(sv_f2[17]),
                                               "f"(sv_f2[18]), "f"(sv_f2[19]));
                        _fp8_2[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[20]), "f"(sv_f2[21]),
                                               "f"(sv_f2[22]), "f"(sv_f2[23]));
                        _fp8_2[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[24]), "f"(sv_f2[25]),
                                               "f"(sv_f2[26]), "f"(sv_f2[27]));
                        _fp8_2[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f2[28]), "f"(sv_f2[29]),
                                               "f"(sv_f2[30]), "f"(sv_f2[31]));
                        _fp8_2[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_all + 16, (const uint32_t*)_fp8_2);
                    const float2 _fma_b2_78 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_79 = {8.0f - max_scaled, 8.0f - max_scaled};
                    float2 _fma_pair_80 = fma_f32x2(make_float2(sv_f3[0], sv_f3[1]), _fma_b2_78, _fma_c2_79);
                    sv_f3[0] = _fma_pair_80.x;
                    sv_f3[1] = _fma_pair_80.y;
                    float2 _fma_pair_81 = fma_f32x2(make_float2(sv_f3[2], sv_f3[3]), _fma_b2_78, _fma_c2_79);
                    sv_f3[2] = _fma_pair_81.x;
                    sv_f3[3] = _fma_pair_81.y;
                    float2 _fma_pair_82 = fma_f32x2(make_float2(sv_f3[4], sv_f3[5]), _fma_b2_78, _fma_c2_79);
                    sv_f3[4] = _fma_pair_82.x;
                    sv_f3[5] = _fma_pair_82.y;
                    float2 _fma_pair_83 = fma_f32x2(make_float2(sv_f3[6], sv_f3[7]), _fma_b2_78, _fma_c2_79);
                    sv_f3[6] = _fma_pair_83.x;
                    sv_f3[7] = _fma_pair_83.y;
                    float2 _fma_pair_84 = fma_f32x2(make_float2(sv_f3[8], sv_f3[9]), _fma_b2_78, _fma_c2_79);
                    sv_f3[8] = _fma_pair_84.x;
                    sv_f3[9] = _fma_pair_84.y;
                    float2 _fma_pair_85 = fma_f32x2(make_float2(sv_f3[10], sv_f3[11]), _fma_b2_78, _fma_c2_79);
                    sv_f3[10] = _fma_pair_85.x;
                    sv_f3[11] = _fma_pair_85.y;
                    float2 _fma_pair_86 = fma_f32x2(make_float2(sv_f3[12], sv_f3[13]), _fma_b2_78, _fma_c2_79);
                    sv_f3[12] = _fma_pair_86.x;
                    sv_f3[13] = _fma_pair_86.y;
                    float2 _fma_pair_87 = fma_f32x2(make_float2(sv_f3[14], sv_f3[15]), _fma_b2_78, _fma_c2_79);
                    sv_f3[14] = _fma_pair_87.x;
                    sv_f3[15] = _fma_pair_87.y;
                    float2 _fma_pair_88 = fma_f32x2(make_float2(sv_f3[16], sv_f3[17]), _fma_b2_78, _fma_c2_79);
                    sv_f3[16] = _fma_pair_88.x;
                    sv_f3[17] = _fma_pair_88.y;
                    float2 _fma_pair_89 = fma_f32x2(make_float2(sv_f3[18], sv_f3[19]), _fma_b2_78, _fma_c2_79);
                    sv_f3[18] = _fma_pair_89.x;
                    sv_f3[19] = _fma_pair_89.y;
                    float2 _fma_pair_90 = fma_f32x2(make_float2(sv_f3[20], sv_f3[21]), _fma_b2_78, _fma_c2_79);
                    sv_f3[20] = _fma_pair_90.x;
                    sv_f3[21] = _fma_pair_90.y;
                    float2 _fma_pair_91 = fma_f32x2(make_float2(sv_f3[22], sv_f3[23]), _fma_b2_78, _fma_c2_79);
                    sv_f3[22] = _fma_pair_91.x;
                    sv_f3[23] = _fma_pair_91.y;
                    float2 _fma_pair_92 = fma_f32x2(make_float2(sv_f3[24], sv_f3[25]), _fma_b2_78, _fma_c2_79);
                    sv_f3[24] = _fma_pair_92.x;
                    sv_f3[25] = _fma_pair_92.y;
                    float2 _fma_pair_93 = fma_f32x2(make_float2(sv_f3[26], sv_f3[27]), _fma_b2_78, _fma_c2_79);
                    sv_f3[26] = _fma_pair_93.x;
                    sv_f3[27] = _fma_pair_93.y;
                    float2 _fma_pair_94 = fma_f32x2(make_float2(sv_f3[28], sv_f3[29]), _fma_b2_78, _fma_c2_79);
                    sv_f3[28] = _fma_pair_94.x;
                    sv_f3[29] = _fma_pair_94.y;
                    float2 _fma_pair_95 = fma_f32x2(make_float2(sv_f3[30], sv_f3[31]), _fma_b2_78, _fma_c2_79);
                    sv_f3[30] = _fma_pair_95.x;
                    sv_f3[31] = _fma_pair_95.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv_f3[_le] = approx_exp2(sv_f3[_le]);
                    }
                    float2 _reg_reduce_sum2_97 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[0], sv_f3[1]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[2], sv_f3[3]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[4], sv_f3[5]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[6], sv_f3[7]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[8], sv_f3[9]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[10], sv_f3[11]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[12], sv_f3[13]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[14], sv_f3[15]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[16], sv_f3[17]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[18], sv_f3[19]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[20], sv_f3[21]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[22], sv_f3[23]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[24], sv_f3[25]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[26], sv_f3[27]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[28], sv_f3[29]));
                    _reg_reduce_sum2_97 = add_f32x2(_reg_reduce_sum2_97, make_float2(sv_f3[30], sv_f3[31]));
                    float _reg_array_reduce_96 = _reg_reduce_sum2_97.x + _reg_reduce_sum2_97.y;
                    block_sum = block_sum + _reg_array_reduce_96;
                    unsigned int _fp8_3[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_f3[0]), "f"(sv_f3[1]),
                                               "f"(sv_f3[2]), "f"(sv_f3[3]));
                        _fp8_3[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[4]), "f"(sv_f3[5]),
                                               "f"(sv_f3[6]), "f"(sv_f3[7]));
                        _fp8_3[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[8]), "f"(sv_f3[9]),
                                               "f"(sv_f3[10]), "f"(sv_f3[11]));
                        _fp8_3[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[12]), "f"(sv_f3[13]),
                                               "f"(sv_f3[14]), "f"(sv_f3[15]));
                        _fp8_3[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[16]), "f"(sv_f3[17]),
                                               "f"(sv_f3[18]), "f"(sv_f3[19]));
                        _fp8_3[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[20]), "f"(sv_f3[21]),
                                               "f"(sv_f3[22]), "f"(sv_f3[23]));
                        _fp8_3[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[24]), "f"(sv_f3[25]),
                                               "f"(sv_f3[26]), "f"(sv_f3[27]));
                        _fp8_3[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_f3[28]), "f"(sv_f3[29]),
                                               "f"(sv_f3[30]), "f"(sv_f3[31]));
                        _fp8_3[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_all + 24, (const uint32_t*)_fp8_3);
                    row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + ((int)((unsigned int)softmax_tile_cursor % 2)) * 8);
                    softmax_tile_cursor += 1;
                }
                smem_stats_sum[stats_row] = row_sum_val;
                smem_stats_final_max[stats_row] = row_max_val;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(sum_ready_addr);
                mbarrier_wait_hint(o_done_addr, _phase_o_done_0, 10000000);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _rcp_98 = approx_rcp(row_sum_val);
                float epi_inv_sum = ((row_sum_val > 0.0f) ? _rcp_98 : 0.0f);
                int epi_v_chunk = work_idx_2 & 1;
                int epi_o_offset = (query_idx_1 * num_heads + my_row) * 512 + epi_v_chunk * 256;
                #pragma unroll
                for (int epi_vs = 0; epi_vs < 2; epi_vs++) {
                    int epi_o_base = taddr + 256 + (unsigned int)(epi_vs * 128) + (unsigned int)(tmem_row_base << 16);
                    #pragma unroll
                    for (int epi_c = 64; epi_c < 128; epi_c += 32) {
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(epi_o_base + epi_c));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int epi_gmem_base = epi_o_offset + epi_vs * 128 + epi_c;
                        if (my_row < num_heads) {
                            #pragma unroll
                            for (int epi_j = 0; epi_j < 32; epi_j += 16) {
                                float _scaled_0[16];
                                const float2 _scale2_99 = {epi_inv_sum * epi_output_scale, epi_inv_sum * epi_output_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((_tmem_load_1 + epi_j))[_ls], _scale2_99);
                                unsigned int _packed_0[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
                                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 4], _scaled_0[_lp*2+1 + 4]));
                                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 6], _scaled_0[_lp*2+1 + 6]));
                                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 8], _scaled_0[_lp*2+1 + 8]));
                                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 10], _scaled_0[_lp*2+1 + 10]));
                                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 12], _scaled_0[_lp*2+1 + 12]));
                                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 14], _scaled_0[_lp*2+1 + 14]));
                                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                                }
                                {
                                    const unsigned* _raw_stv8_100 = reinterpret_cast<const unsigned*>(_packed_0);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(O + (epi_gmem_base + epi_j))), "r"(_raw_stv8_100[0]), "r"(_raw_stv8_100[1]), "r"(_raw_stv8_100[2]), "r"(_raw_stv8_100[3]), "r"(_raw_stv8_100[4]), "r"(_raw_stv8_100[5]), "r"(_raw_stv8_100[6]), "r"(_raw_stv8_100[7]) : "memory");
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.442695f;
            float output_scale = bmm2_scale[0];
            const int wg_dummy_inc_1 = 0;
            int all_num_kv_tiles_2 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base_1 = warp % 4 * 32;
            const int n_half_1 = 0;
            const int my_row_1 = tmem_row_base_1 + lane;
            const int stats_row_1 = n_half_1 * 64 + my_row_1;
            const int corr_row = tmem_row_base_1 << 16;
            int correction_tile_cursor = 0;
            unsigned int _phase_q_full_0_2 = 0;
            unsigned int _phase_pv_done_0 = 0;
            unsigned int _phase_o_done_0_1 = 0;
            unsigned int _phase_sum_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = bid; work_idx_3 < total_work_items; work_idx_3 += num_bids) {
                int split_idx_2 = 0;
                int query_idx_2 = work_idx_3 >> 1;
                int v_chunk_1 = work_idx_3 & 1;
                int tiles_per_split_2 = all_num_kv_tiles_2;
                int first_tile_2 = split_idx_2 * tiles_per_split_2;
                int num_kv_tiles_2 = tiles_per_split_2;
                mbarrier_wait_hint(q_full_addr, _phase_q_full_0_2, 10000000);
                _phase_q_full_0_2 ^= 1;
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                    int pipeline_tile_1 = correction_tile_cursor;
                    int phase_1 = pipeline_tile_1 % 2;
                    int _stage_stats = (int)((unsigned int)correction_tile_cursor % 2);
                    int _parity_stats = (int)((unsigned int)correction_tile_cursor / 2 % 2);
                    mbarrier_wait_hint(stats_addr + (_stage_stats) * 8, _parity_stats, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc_scale_1 = smem_stats_max[phase_1 * 128 + stats_row_1];
                    if (tile_2 > 0) {
                        mbarrier_wait_hint(pv_done_addr, _phase_pv_done_0, 10000000);
                        _phase_pv_done_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    if (tile_2 > 0) {
                        int any_rescale = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (any_rescale != 0) {
                            #pragma unroll
                            for (int vs = 0; vs < 2; vs++) {
                                int o_base = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                                #pragma unroll
                                for (int c = 0; c < 128; c += 16) {
                                    float _tmem_load_2[16];
                                    tmem_ld_x16(&_tmem_load_2[0], o_base + c);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                                    tmem_st_x16_f32(o_base + c, _tmem_load_2);
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_done_addr + ((int)((unsigned int)correction_tile_cursor % 2)) * 8);
                    correction_tile_cursor += 1;
                }
                mbarrier_wait_hint(o_done_addr, _phase_o_done_0_1, 10000000);
                _phase_o_done_0_1 ^= 1;
                mbarrier_wait_hint(sum_ready_addr, _phase_sum_ready_0, 10000000);
                _phase_sum_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float total_sum = smem_stats_sum[stats_row_1];
                float final_max = smem_stats_final_max[stats_row_1];
                float _rcp_1 = approx_rcp(total_sum);
                float inv_sum = ((total_sum > 0.0f) ? _rcp_1 : 0.0f);
                int head_idx = my_row_1;
                int direct_o_offset = (query_idx_2 * num_heads + head_idx) * 512 + v_chunk_1 * 256;
                int partial_o_offset = (query_idx_2 * num_heads + head_idx + split_idx_2) * 512 + n_half_1 * 256;
                int o_offset = direct_o_offset;
                if ((0 & (int)(n_half_1 == 0) & (int)(head_idx < num_heads)) != 0) {
                    int stat_offset = query_idx_2 * num_heads + head_idx + split_idx_2;
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(total_sum));
                    float _log2_2;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(total_sum));
                    partial_lse[stat_offset] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_2) - 8.0f : -CUDART_INF_F);
                }
                #pragma unroll
                for (int vs_1 = 0; vs_1 < 2; vs_1++) {
                    int o_base_epi = taddr + 256 + (unsigned int)(vs_1 * 128) + (unsigned int)corr_row;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 64; c_1 += 32) {
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                            : "r"(o_base_epi + c_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int gmem_base = o_offset + vs_1 * 128 + c_1;
                        if (head_idx < num_heads) {
                            #pragma unroll
                            for (int j = 0; j < 32; j += 16) {
                                float _scaled_1[16];
                                const float2 _scale2_3 = {inv_sum * output_scale, inv_sum * output_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((_tmem_load_3 + j))[_ls], _scale2_3);
                                unsigned int _packed_1[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 2], _scaled_1[_lp*2+1 + 2]));
                                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 4], _scaled_1[_lp*2+1 + 4]));
                                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 6], _scaled_1[_lp*2+1 + 6]));
                                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 8], _scaled_1[_lp*2+1 + 8]));
                                    _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 10], _scaled_1[_lp*2+1 + 10]));
                                    _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 12], _scaled_1[_lp*2+1 + 12]));
                                    _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 14], _scaled_1[_lp*2+1 + 14]));
                                    _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                                }
                                {
                                    const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_1);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(O + (gmem_base + j))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg2_dummy_1 = 0;
            int all_num_kv_tiles_3 = (sparse_topk + 128 - 1) / 128;
            unsigned int mma_kv_stage = 0;
            int mma_tile_cursor = 0;
            int mma_pv_tile = 0;
            unsigned int mma_pv_pair = 0;
            unsigned int _phase_s_seeded_0 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            unsigned int _phase_kv_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = bid; work_idx_4 < total_work_items; work_idx_4 += num_bids) {
                int split_idx_3 = 0;
                int tiles_per_split_3 = all_num_kv_tiles_3;
                int first_tile_3 = split_idx_3 * tiles_per_split_3;
                int num_kv_tiles_3 = tiles_per_split_3;
                mbarrier_wait_hint(s_seeded_addr, _phase_s_seeded_0, 10000000);
                _phase_s_seeded_0 ^= 1;
                mbarrier_wait_hint(q_full_addr, _phase_q_full_0_3, 10000000);
                _phase_q_full_0_3 ^= 1;
                int first_pv = 1;
                #pragma unroll 1
                for (int tile_3 = 0; tile_3 < num_kv_tiles_3; tile_3++) {
                    int pipeline_tile_2 = mma_tile_cursor;
                    int phase_2 = pipeline_tile_2 & 1;
                    int score_col = ((phase_2 != 0) ? 128 : 0);
                    #pragma unroll 1
                    for (int n = 0; n < 4; n++) {
                        mbarrier_wait_hint(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (n) * 1024);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem_scratch + (score_col)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, ((n == 0) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem_scratch + (score_col)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem_scratch + (score_col)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_tmem_scratch + (score_col)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                            }
                        }
                        if (n == 3) {
                            elect_commit(s_full_addr + ((int)((unsigned int)mma_tile_cursor % 2)) * 8);
                        }
                        elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 8) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    if (tile_3 > 0) {
                        int prev_pipeline_tile = mma_pv_tile;
                        int prev_phase = mma_pv_tile % 2;
                        int _stage_p_full = (int)((unsigned int)mma_pv_tile % 2);
                        int _parity_p_full = (int)((unsigned int)mma_pv_tile / 2 % 2);
                        mbarrier_wait_hint(p_full_addr + (_stage_p_full) * 8, _parity_p_full, 10000000);
                        int _stage_corr_done = (int)((unsigned int)mma_pv_tile % 2);
                        int _parity_corr_done = (int)((unsigned int)mma_pv_tile / 2 % 2);
                        mbarrier_wait_hint(corr_done_addr + (_stage_corr_done) * 8, _parity_corr_done, 10000000);
                        int p_col = ((prev_phase != 0) ? 224 : 96);
                        #pragma unroll
                        for (int vs_2 = 0; vs_2 < 2; vs_2++) {
                            mbarrier_wait_hint(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int output_col = 256 + vs_2 * 128;
                            int _mma_b_lo_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (output_col))), "r"(_mma_b_lo_1), "r"(tmem_tmem_scratch + p_col), "r"(((first_pv) ? 0 : 1)) : "memory");
                            elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                            mma_kv_stage += 1;
                            if (mma_kv_stage == 8) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                        }
                        first_pv = 0;
                        elect_commit(pv_done_addr);
                        mma_pv_tile += 1;
                    }
                    mma_tile_cursor += 1;
                }
                int last_pipeline_tile = mma_pv_tile;
                int last_phase = mma_pv_tile % 2;
                int _stage_p_full_1 = (int)((unsigned int)mma_pv_tile % 2);
                int _parity_p_full_1 = (int)((unsigned int)mma_pv_tile / 2 % 2);
                mbarrier_wait_hint(p_full_addr + (_stage_p_full_1) * 8, _parity_p_full_1, 10000000);
                int _stage_corr_done_1 = (int)((unsigned int)mma_pv_tile % 2);
                int _parity_corr_done_1 = (int)((unsigned int)mma_pv_tile / 2 % 2);
                mbarrier_wait_hint(corr_done_addr + (_stage_corr_done_1) * 8, _parity_corr_done_1, 10000000);
                int p_col_last = ((last_phase != 0) ? 224 : 96);
                #pragma unroll
                for (int vs_3 = 0; vs_3 < 2; vs_3++) {
                    mbarrier_wait_hint(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int output_col_d = 256 + vs_3 * 128;
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (output_col_d))), "r"(_mma_b_lo_2), "r"(tmem_tmem_scratch + p_col_last), "r"(((first_pv) ? 0 : 1)) : "memory");
                    elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 8) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                }
                elect_commit(q_empty_addr);
                elect_commit(o_done_addr);
                mma_pv_tile += 1;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait_hint(tmem_dealloc_addr, _phase_tmem_dealloc_0, 10000000);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: empty1 ----
    if (warp >= 14 && warp <= 15) {
        { // empty1_main
            const int wg2_dummy_2 = 0;
            unsigned int window_prod_stage = 0;
            unsigned int _phase_window_empty = 1;
            unsigned int _phase_q_full_0_4 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_5 = bid; work_idx_5 < total_work_items; work_idx_5 += num_bids) {
                int window_query_idx = work_idx_5 >> 1;
                int query_batch = window_query_idx / max_q_len;
                int query_offset = window_query_idx - query_batch * max_q_len;
                int query_length = max_q_len;
                if (ragged_query != 0) {
                    query_batch = 0;
                    #pragma unroll 2
                    for (int chunk = 0; chunk < (batch_size + 31) / 32; chunk++) {
                        int lane_entry = chunk * 32 + lane + 1;
                        int lane_load = ((lane_entry) < (batch_size) ? (lane_entry) : (batch_size));
                        unsigned int started = __ballot_sync(0xFFFFFFFF, lane_entry <= batch_size && window_query_idx >= cum_seq_lens_q[lane_load]);
                        query_batch = query_batch + (int)(unsigned int)__popc(started);
                    }
                    int query_begin = cum_seq_lens_q[query_batch];
                    query_length = cum_seq_lens_q[query_batch + 1] - query_begin;
                    query_offset = window_query_idx - query_begin;
                }
                int visible = seq_lens[query_batch] - query_length + query_offset + 1;
                if (visible < 0) {
                    visible = 0;
                }
                if (visible > 128) {
                    visible = 128;
                }
                int swa_visible_pub = visible;
                mbarrier_wait_hint(window_empty_addr + (window_prod_stage) * 8, _phase_window_empty, 10000000);
                if (warp == 14 && lane == 0) {
                    smem_swa_window[window_prod_stage * 4] = swa_visible_pub;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(window_full_addr + (window_prod_stage) * 8);
                window_prod_stage += 1;
                if (window_prod_stage == 2) { window_prod_stage = 0; _phase_window_empty ^= 1; }
                mbarrier_wait_hint(q_full_addr, _phase_q_full_0_4, 10000000);
                _phase_q_full_0_4 ^= 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
