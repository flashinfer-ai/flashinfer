/*
 * Copyright (c) 2023 by FlashInfer team.
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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 4
#define NUM_V_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_K_OFF 66560
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_V_LO_OFF 132096
#define SMEM_SMEM_V_LO_STAGE_BYTES 16384
#define SMEM_SMEM_V_LO_STRIDE 32768
#define SMEM_SMEM_V_HI_OFF 148480
#define SMEM_SMEM_V_HI_STAGE_BYTES 16384
#define SMEM_SMEM_V_HI_STRIDE 32768
#define SMEM_SMEM_P_OFF 197632
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_INDICES_OFF 219936
#define SMEM_SMEM_INDICES_STAGE_BYTES 9216
#define SMEM_SMEM_INDICES_STRIDE 9216
#define SMEM_SMEM_SOFTMAX_EXCHANGE_OFF 218624
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STRIDE 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_OFF 219136
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STRIDE 512
#define SMEM_SMEM_INDEX_VALID_OFF 219648
#define SMEM_SMEM_INDEX_VALID_STAGE_BYTES 288
#define SMEM_SMEM_INDEX_VALID_STRIDE 288
#define SMEM_TOTAL 229248
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}




__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)





__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}





__device__ __forceinline__ void tma_gather4_gmem2smem_mc_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast + cta_group::2 variant; see tma_gather4_gmem2smem_mc.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
}


extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_304e0f834046e530860a(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_k, const __grid_constant__ CUtensorMap tmap_compressed_k, const __grid_constant__ CUtensorMap tmap_swa_v, const __grid_constant__ CUtensorMap tmap_compressed_v, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_lse, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int num_query_tokens, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int sparse_topk, int has_sinks, int total_work_items)
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
    #define k_empty_addr (mbar_base + 48)
    #define v_lo_full_addr (mbar_base + 80)
    #define v_lo_empty_addr (mbar_base + 96)
    #define v_hi_full_addr (mbar_base + 112)
    #define v_hi_empty_addr (mbar_base + 128)
    #define s_full_addr (mbar_base + 144)
    #define s_empty_addr (mbar_base + 160)
    #define p_full_addr (mbar_base + 176)
    #define p_empty_addr (mbar_base + 184)
    #define o_lo_empty_addr (mbar_base + 192)
    #define o_hi_empty_addr (mbar_base + 200)
    #define stats_addr (mbar_base + 208)
    #define stats_empty_addr (mbar_base + 224)
    #define o_lo_full_addr (mbar_base + 240)
    #define o_hi_full_addr (mbar_base + 248)
    #define o_first_slice_seeded_addr (mbar_base + 256)
    #define pv_issue_done_addr (mbar_base + 264)
    #define tmem_dealloc_peer_addr (mbar_base + 272)
    #define index_ready_addr (mbar_base + 280)
    #define index_empty_addr (mbar_base + 296)
    #define index_valid_full_addr (mbar_base + 312)
    #define index_valid_empty_addr (mbar_base + 328)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;
    __nv_bfloat16* smem_v_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_v_lo_addr = smem + 132096;
    __nv_bfloat16* smem_v_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_v_hi_addr = smem + 148480;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_p_addr = smem + 197632;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 219936);
    const int smem_indices_addr = smem + 219936;
    float* smem_softmax_exchange = reinterpret_cast<float*>(smem_raw + 218624);
    const int smem_softmax_exchange_addr = smem + 218624;
    float* smem_epilogue_exchange = reinterpret_cast<float*>(smem_raw + 219136);
    const int smem_epilogue_exchange_addr = smem + 219136;
    unsigned int* smem_index_valid = reinterpret_cast<unsigned int*>(smem_raw + 219648);
    const int smem_index_valid_addr = smem + 219648;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (25 pipeline groups, 0 ordered-sequence groups, 43 barriers)
    // Mbarriers at smem_raw[0..344)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'v_pipe' ---
            // v_lo_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // v_lo_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // v_hi_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // v_hi_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            // p_empty: 1 barriers, init_count=2
            mbarrier_init(smem + 184, 2);
            // o_lo_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            // o_hi_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            // o_lo_full: 1 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            // o_hi_full: 1 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            // o_first_slice_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 256, 256);
            // pv_issue_done: 1 barriers, init_count=2
            mbarrier_init(smem + 264, 2);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 272, 32);
            // index_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            // index_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            mbarrier_init(smem + 304, 128);
            // index_valid_full: 2 barriers, init_count=32
            mbarrier_init(smem + 312, 32);
            mbarrier_init(smem + 320, 32);
            // index_valid_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 344);
    if (warp == 0) {
        int _tmem_hold = smem + 344;
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
    const int tmem_tmem_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // softmax_wg_main
            const int softmax_dummy = 0;
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int head_row_base = warp % 2 * 32;
            const int score_row_base = warp % 4 * 32;
            const int n_half = warp % 4 / 2;
            const int my_row = head_row_base + lane;
            const int exchange_idx = n_half * 64 + my_row;
            int tile_cursor = 0;
            int valid_iter = 0;
            float seed_zero[4];
            #pragma unroll
            for (int seed_i = 0; seed_i < 4; seed_i++) {
                seed_zero[seed_i] = 0.0f;
            }
            #pragma unroll
            for (int seed_stage = 0; seed_stage < 2; seed_stage++) {
                #pragma unroll
                for (int seed_col = 128 + seed_stage * 128; seed_col < 128 + (seed_stage + 1) * 128; seed_col += 4) {
                    int seed_addr = taddr + (unsigned int)seed_col + (unsigned int)(score_row_base << 16);
                    tmem_st_x4_f32(seed_addr, seed_zero);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(o_first_slice_seeded_addr);
            int peer_rank = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(o_first_slice_seeded_addr), "r"(peer_rank) : "memory");
            int work_base = (unsigned int)total_work_items / num_clusters;
            int work_extra = (unsigned int)total_work_items % num_clusters;
            int extra_start = num_clusters - (unsigned int)work_extra;
            int extra_prefix = cluster_id - (unsigned int)extra_start;
            if (extra_prefix < 0) {
                extra_prefix = 0;
            }
            int work_begin = cluster_id * (unsigned int)work_base + (unsigned int)extra_prefix;
            int work_end = work_begin + work_base;
            if ((unsigned int)extra_start <= cluster_id) {
                work_end = work_end + 1;
            }
            #pragma unroll 1
            for (unsigned int work_slot = work_begin; work_slot < work_end; work_slot++) {
                int work_idx = work_slot;
                if ((unsigned int)extra_start > cluster_id) {
                    work_idx = (work_slot - (unsigned int)work_begin) * (unsigned int)extra_start + cluster_id;
                }
                int split_idx = work_idx % ((0) ? 3 : 1);
                int query_idx = work_idx / ((0) ? 3 : 1);
                int _max_4 = ((sparse_topk_lens[query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx] + sparse_topk_lens_offset) : (0));
                int _min_4 = ((_max_4) < (sparse_topk) ? (_max_4) : (sparse_topk));
                int active_topk = _min_4;
                int num_kv_tiles = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                {
                    int _max_5 = ((sparse_topk_lens[work_idx / ((0) ? 3 : 1)] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[work_idx / ((0) ? 3 : 1)] + sparse_topk_lens_offset) : (0));
                    int _min_5 = ((_max_5) < (sparse_topk) ? (_max_5) : (sparse_topk));
                    int active_count = _min_5;
                    int active_tile_count = (active_count + 128 - 1) / 128;
                    if (active_tile_count < 1) {
                        active_tile_count = 1;
                    }
                    num_kv_tiles = (active_tile_count + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                }
                int first_kv_tile = split_idx * num_kv_tiles;
                int head_idx = cta_rank * 64 + my_row;
                float row_max_val = -CAKE_INF;
                float row_sum_val = 0.0f;
                if (has_sinks != 0 && head_idx < num_heads && split_idx == 0) {
                    row_max_val = sinks[head_idx] * 1.4426950408889634f / softmax_scale_log2;
                    if (n_half == 0) {
                        row_sum_val = 1.0f;
                    }
                }
                int valid_slot = valid_iter & 1;
                int valid_full_phase = valid_iter >> 1 & 1;
                mbarrier_wait(index_valid_full_addr + (valid_slot) * 8, valid_full_phase);
                int valid_word_base = valid_slot * 36 + n_half * 2;
                #pragma unroll 1
                for (int tile = 0; tile < num_kv_tiles; tile++) {
                    int pipeline_tile = tile_cursor + tile;
                    int score_phase = pipeline_tile & 1;
                    int s_full_phase = pipeline_tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (score_phase) * 8, s_full_phase);
                    int score_offset = ((score_phase != 0) ? 64 : 0);
                    int score_base = taddr + (unsigned int)score_offset + (unsigned int)(score_row_base << 16);
                    int valid_cols = active_topk - (first_kv_tile + tile) * 128 - n_half * 64;
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                    if (valid_cols > 64) {
                        valid_cols = 64;
                    }
                    unsigned int keep_lo = smem_index_valid[valid_word_base + tile * 4];
                    unsigned int keep_hi = smem_index_valid[valid_word_base + tile * 4 + 1];
                    float scores[64];
                    float tile_max = -CAKE_INF;
                    if (valid_cols == 64 && ~(keep_lo & keep_hi) == 0) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(tile_max)
                            : "r"(score_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    } else {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63])
                            : "r"(score_base));
                        uint32_t _thresh_mask_0;
                        {
                            int _lim_0 = valid_cols;
                            if (_lim_0 <= 0) { _thresh_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_0) : "r"(_lim_0));
                            }
                        }
                        uint32_t _thresh_mask_1;
                        {
                            int _lim_1 = valid_cols - 32;
                            if (_lim_1 <= 0) { _thresh_mask_1 = 0u; }
                            else if (_lim_1 >= 32) { _thresh_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_1) : "r"(_lim_1));
                            }
                        }
                        if (!(_thresh_mask_0 & keep_lo & (1u << 0))) scores[0] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 1))) scores[1] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 2))) scores[2] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 3))) scores[3] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 4))) scores[4] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 5))) scores[5] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 6))) scores[6] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 7))) scores[7] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 8))) scores[8] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 9))) scores[9] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 10))) scores[10] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 11))) scores[11] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 12))) scores[12] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 13))) scores[13] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 14))) scores[14] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 15))) scores[15] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 16))) scores[16] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 17))) scores[17] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 18))) scores[18] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 19))) scores[19] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 20))) scores[20] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 21))) scores[21] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 22))) scores[22] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 23))) scores[23] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 24))) scores[24] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 25))) scores[25] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 26))) scores[26] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 27))) scores[27] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 28))) scores[28] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 29))) scores[29] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 30))) scores[30] = -CAKE_INF;
                        if (!(_thresh_mask_0 & keep_lo & (1u << 31))) scores[31] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 0))) scores[32] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 1))) scores[33] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 2))) scores[34] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 3))) scores[35] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 4))) scores[36] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 5))) scores[37] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 6))) scores[38] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 7))) scores[39] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 8))) scores[40] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 9))) scores[41] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 10))) scores[42] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 11))) scores[43] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 12))) scores[44] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 13))) scores[45] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 14))) scores[46] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 15))) scores[47] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 16))) scores[48] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 17))) scores[49] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 18))) scores[50] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 19))) scores[51] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 20))) scores[52] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 21))) scores[53] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 22))) scores[54] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 23))) scores[55] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 24))) scores[56] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 25))) scores[57] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 26))) scores[58] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 27))) scores[59] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 28))) scores[60] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 29))) scores[61] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 30))) scores[62] = -CAKE_INF;
                        if (!(_thresh_mask_1 & keep_hi & (1u << 31))) scores[63] = -CAKE_INF;
                        float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&scores[0], _reg_reduce_max2_2);
                        row_max_x32_accum(&scores[32], _reg_reduce_max2_2);
                        float scores_max = row_max_reduce(_reg_reduce_max2_2);
                        tile_max = scores_max;
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + (unsigned int)(score_phase * 8)), "r"(0) : "memory");
                    float _max_6 = max_noftz(tile_max, row_max_val);
                    float new_max = _max_6;
                    smem_softmax_exchange[exchange_idx] = new_max;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float _max_7 = max_noftz(new_max, smem_softmax_exchange[exchange_idx ^ 64]);
                    new_max = _max_7;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float grow = (new_max - row_max_val) * softmax_scale_log2;
                    float no_correction = ((grow <= 8.0f || row_max_val == -CAKE_INF) ? 1.0f : 0.0f);
                    float delta = softmax_scale_log2 * (row_max_val - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    float exp_delta = _exp2_0;
                    float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                    if (grow <= 8.0f) {
                        acc_scale = 1.0f;
                        new_max = row_max_val;
                    }
                    row_max_val = new_max;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    const float2 _fma_b2_3 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_4 = {-max_scaled, -max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(scores)[_lf], _fma_b2_3, _fma_c2_4);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        scores[_le] = approx_exp2(scores[_le]);
                    }
                    float2 _reg_reduce_sum2_5 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&scores[0], &_reg_reduce_sum2_5);
                    softmax_block_sum(&scores[32], &_reg_reduce_sum2_5);
                    float scores_sum = _reg_reduce_sum2_5.x + _reg_reduce_sum2_5.y;
                    row_sum_val = row_sum_val * acc_scale + scores_sum;
                    int stats_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(stats_empty_addr + (score_phase) * 8, stats_empty_phase);
                    float meta[4];
                    meta[0] = row_sum_val;
                    meta[1] = row_max_val;
                    meta[2] = acc_scale;
                    meta[3] = no_correction;
                    int meta_addr = taddr + 384 + (unsigned int)(score_phase * 8) + (unsigned int)(n_half * 4) + (unsigned int)(head_row_base << 16);
                    tmem_st_x4_f32(meta_addr, meta);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(stats_addr + (score_phase) * 8);
                    int p_stage = 0;
                    int p_empty_phase = pipeline_tile & 1 ^ 1;
                    mbarrier_wait(p_empty_addr + (p_stage) * 8, p_empty_phase);
                    uint32_t scores_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scores[_lp*2 + 0], scores[_lp*2+1 + 0]));
                        scores_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_p_stage = p_stage * 2 + n_half;
                    int p_base_smem = smem_p_addr + (unsigned int)(smem_p_stage * 8192);
                    #pragma unroll
                    for (int vec = 0; vec < 8; vec++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_smem + (my_row * 128 + vec * 16 ^ (my_row * 128 + vec * 16 >> 7 & 7) << 4))), "r"(scores_bf16[vec * 4]), "r"(scores_bf16[vec * 4 + 1]), "r"(scores_bf16[vec * 4 + 2]), "r"(scores_bf16[vec * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (unsigned int)(p_stage * 8)), "r"(0) : "memory");
                }
                mbarrier_arrive(index_valid_empty_addr + (valid_slot) * 8);
                valid_iter = valid_iter + 1;
                tile_cursor = tile_cursor + num_kv_tiles;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // correction_wg_main
            const int correction_dummy = 0;
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int head_row_base_1 = warp % 2 * 32;
            const int output_row_base = warp % 4 * 32;
            const int n_half_1 = warp % 4 / 2;
            const int my_row_1 = head_row_base_1 + lane;
            const int corr_row = head_row_base_1 << 16;
            int tile_cursor_1 = 0;
            int work_base_1 = (unsigned int)total_work_items / num_clusters;
            int work_extra_1 = (unsigned int)total_work_items % num_clusters;
            int extra_start_1 = num_clusters - (unsigned int)work_extra_1;
            int extra_prefix_1 = cluster_id - (unsigned int)extra_start_1;
            if (extra_prefix_1 < 0) {
                extra_prefix_1 = 0;
            }
            int work_begin_1 = cluster_id * (unsigned int)work_base_1 + (unsigned int)extra_prefix_1;
            int work_end_1 = work_begin_1 + work_base_1;
            if ((unsigned int)extra_start_1 <= cluster_id) {
                work_end_1 = work_end_1 + 1;
            }
            unsigned int _phase_o_lo_full_0 = 0;
            unsigned int _phase_o_hi_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_slot_1 = work_begin_1; work_slot_1 < work_end_1; work_slot_1++) {
                int work_idx_1 = work_slot_1;
                if ((unsigned int)extra_start_1 > cluster_id) {
                    work_idx_1 = (work_slot_1 - (unsigned int)work_begin_1) * (unsigned int)extra_start_1 + cluster_id;
                }
                int split_idx_1 = work_idx_1 % ((0) ? 3 : 1);
                int query_idx_1 = work_idx_1 / ((0) ? 3 : 1);
                int num_kv_tiles_1 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                {
                    int _max_8 = ((sparse_topk_lens[work_idx_1 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[work_idx_1 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) : (0));
                    int _min_6 = ((_max_8) < (sparse_topk) ? (_max_8) : (sparse_topk));
                    int active_count_1 = _min_6;
                    int active_tile_count_1 = (active_count_1 + 128 - 1) / 128;
                    if (active_tile_count_1 < 1) {
                        active_tile_count_1 = 1;
                    }
                    num_kv_tiles_1 = (active_tile_count_1 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                }
                float final_sum = 0.0f;
                float final_max = -CAKE_INF;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int pipeline_tile_1 = tile_cursor_1 + tile_1;
                    int score_phase_1 = pipeline_tile_1 & 1;
                    int stats_phase = pipeline_tile_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (score_phase_1) * 8, stats_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int meta_addr_1 = taddr + 384 + (unsigned int)(score_phase_1 * 8) + (unsigned int)(n_half_1 * 4) + (unsigned int)corr_row;
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], meta_addr_1);
                    final_sum = _tmem_load_0[0];
                    final_max = _tmem_load_0[1];
                    float acc_scale_1 = _tmem_load_0[2];
                    float no_correction_1 = _tmem_load_0[3];
                    if (tile_1 > 0) {
                        mbarrier_wait(o_lo_full_addr, _phase_o_lo_full_0);
                        _phase_o_lo_full_0 ^= 1;
                        mbarrier_wait(o_hi_full_addr, _phase_o_hi_full_0);
                        _phase_o_hi_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_0 = __all_sync(0xFFFFFFFF, no_correction_1 == 1.0f);
                        int skip_correction = _vote_0;
                        if (skip_correction == 0) {
                            #pragma unroll
                            for (int local_slice = 0; local_slice < 2; local_slice++) {
                                #pragma unroll
                                for (int acc_stage = 0; acc_stage < 2; acc_stage++) {
                                    int o_base = taddr + 128 + (unsigned int)(acc_stage * 128) + (unsigned int)(local_slice * 64) + (unsigned int)(output_row_base << 16);
                                    #pragma unroll
                                    for (int c = 0; c < 64; c += 16) {
                                        float _tmem_load_1[16];
                                        tmem_ld_x16(&_tmem_load_1[0], o_base + c);
                                        #if __CUDA_ARCH__ >= 1000
                                        const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                                        #else
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 16; _ls++) {
                                            _tmem_load_1[_ls] = _tmem_load_1[_ls] * acc_scale_1;
                                        }
                                        #endif
                                        tmem_st_x16_f32(o_base + c, _tmem_load_1);
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_lo_empty_addr), "r"(0) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_hi_empty_addr), "r"(0) : "memory");
                    }
                    mbarrier_arrive(stats_empty_addr + (score_phase_1) * 8);
                }
                mbarrier_wait(o_lo_full_addr, _phase_o_lo_full_0);
                _phase_o_lo_full_0 ^= 1;
                mbarrier_wait(o_hi_full_addr, _phase_o_hi_full_0);
                _phase_o_hi_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                const int exchange_idx_1 = n_half_1 * 64 + my_row_1;
                smem_epilogue_exchange[exchange_idx_1] = final_sum;
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                final_sum = final_sum + smem_epilogue_exchange[exchange_idx_1 ^ 64];
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = ((final_sum > 0.0f) ? _rcp_0 : 0.0f);
                int head_idx_1 = cta_rank * 64 + my_row_1;
                int output_offset = ((0) ? ((query_idx_1 * num_heads + head_idx_1) * ((0) ? 3 : 1) + split_idx_1) * 512 : (query_idx_1 * num_heads + head_idx_1) * 512);
                #pragma unroll
                for (int acc_stage_1 = 0; acc_stage_1 < 2; acc_stage_1++) {
                    #pragma unroll
                    for (int local_slice_1 = 0; local_slice_1 < 2; local_slice_1++) {
                        int logical_slice = acc_stage_1 * 2 * 2 + n_half_1 * 2 + local_slice_1;
                        int o_base_1 = taddr + 128 + (unsigned int)(acc_stage_1 * 128) + (unsigned int)(local_slice_1 * 64) + (unsigned int)(output_row_base << 16);
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 64; c_1 += 32) {
                            float _tmem_load_2[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                                : "r"(o_base_1 + c_1));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int gmem_base = output_offset + logical_slice * 64 + c_1;
                            #pragma unroll
                            for (int j = 0; j < 32; j += 16) {
                                {
                                    const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[j])[_ps], _prescale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_2[j + _ps] *= inv_sum * output_scale;
                                    #endif
                                    {
                                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_2[j + 0], _tmem_load_2[j + 1]);
                                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_2[j + 2], _tmem_load_2[j + 3]);
                                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_2[j + 4], _tmem_load_2[j + 5]);
                                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_2[j + 6], _tmem_load_2[j + 7]);
                                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_2[j + 8], _tmem_load_2[j + 9]);
                                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_2[j + 10], _tmem_load_2[j + 11]);
                                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_2[j + 12], _tmem_load_2[j + 13]);
                                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_2[j + 14], _tmem_load_2[j + 15]);
                                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(&((__nv_bfloat16*)(O + (gmem_base + j)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_lo_empty_addr), "r"(0) : "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_hi_empty_addr), "r"(0) : "memory");
                tile_cursor_1 = tile_cursor_1 + num_kv_tiles_1;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: qk_mma_warp ----
    if (warp == 8) {
        { // qk_mma_warp_main
            const int qk_mma_dummy = 0;
            unsigned int k_stage = 0;
            int tile_cursor_2 = 0;
            int work_base_2 = (unsigned int)total_work_items / num_clusters;
            int work_extra_2 = (unsigned int)total_work_items % num_clusters;
            int extra_start_2 = num_clusters - (unsigned int)work_extra_2;
            int extra_prefix_2 = cluster_id - (unsigned int)extra_start_2;
            if (extra_prefix_2 < 0) {
                extra_prefix_2 = 0;
            }
            int work_begin_2 = cluster_id * (unsigned int)work_base_2 + (unsigned int)extra_prefix_2;
            int work_end_2 = work_begin_2 + work_base_2;
            if ((unsigned int)extra_start_2 <= cluster_id) {
                work_end_2 = work_end_2 + 1;
            }
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            #pragma unroll 1
            for (unsigned int work_slot_2 = work_begin_2; work_slot_2 < work_end_2; work_slot_2++) {
                int work_idx_2 = work_slot_2;
                if ((unsigned int)extra_start_2 > cluster_id) {
                    work_idx_2 = (work_slot_2 - (unsigned int)work_begin_2) * (unsigned int)extra_start_2 + cluster_id;
                }
                int num_kv_tiles_2 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                {
                    int _max_1 = ((sparse_topk_lens[work_idx_2 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[work_idx_2 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) : (0));
                    int _min_1 = ((_max_1) < (sparse_topk) ? (_max_1) : (sparse_topk));
                    int active_count_2 = _min_1;
                    int active_tile_count_2 = (active_count_2 + 128 - 1) / 128;
                    if (active_tile_count_2 < 1) {
                        active_tile_count_2 = 1;
                    }
                    num_kv_tiles_2 = (active_tile_count_2 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                }
                if (cta_rank == 0) {
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                        int pipeline_tile_2 = tile_cursor_2 + tile_2;
                        int score_phase_2 = pipeline_tile_2 & 1;
                        int s_empty_phase = pipeline_tile_2 >> 1 & 1 ^ 1;
                        mbarrier_wait(s_empty_addr + (score_phase_2) * 8, s_empty_phase);
                        int score_col = ((score_phase_2 != 0) ? 64 : 0);
                        #pragma unroll
                        for (int qk_stage = 0; qk_stage < 4; qk_stage++) {
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (qk_stage * 2) * 512;
                            int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(((qk_stage == 0) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (qk_stage * 2 + 1) * 512;
                            int _mma_b_lo_1 = (((smem_k_addr + 8192) >> 4) & 0x3FFF) + (k_stage) * 1024;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(s_full_addr + (score_phase_2) * 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                }
                tile_cursor_2 = tile_cursor_2 + num_kv_tiles_2;
            }
            unsigned int _phase_pv_issue_done_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll
                for (int tail_offset = 0; tail_offset < 2; tail_offset++) {
                    int tail_tile = tile_cursor_2 + tail_offset;
                    int tail_stage = tail_tile & 1;
                    int tail_phase = tail_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(s_empty_addr + (tail_stage) * 8, tail_phase);
                }
                mbarrier_wait(pv_issue_done_addr, _phase_pv_issue_done_0);
                _phase_pv_issue_done_0 ^= 1;
                int final_o_empty_phase = tile_cursor_2 + 1 & 1;
                mbarrier_wait(o_lo_empty_addr, final_o_empty_phase);
                mbarrier_wait(o_hi_empty_addr, final_o_empty_phase);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            int peer_rank_1 = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_dealloc_peer_addr), "r"(peer_rank_1) : "memory");
            mbarrier_wait(tmem_dealloc_peer_addr, 0);
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_wg ----
    if (warp >= 9 && warp <= 12) {
        { // load_wg_main
            bool _elect_sync_0 = elect_sync();
            int issuer_rows[64];
            const int load_dummy = 0;
            const int load_warp_rank = warp - 9;
            int load_iter = 0;
            unsigned int k_stage_1 = 0;
            unsigned int v_stage = 0;
            int work_base_3 = (unsigned int)total_work_items / num_clusters;
            int work_extra_3 = (unsigned int)total_work_items % num_clusters;
            int extra_start_3 = num_clusters - (unsigned int)work_extra_3;
            int extra_prefix_3 = cluster_id - (unsigned int)extra_start_3;
            if (extra_prefix_3 < 0) {
                extra_prefix_3 = 0;
            }
            int work_begin_3 = cluster_id * (unsigned int)work_base_3 + (unsigned int)extra_prefix_3;
            int work_end_3 = work_begin_3 + work_base_3;
            if ((unsigned int)extra_start_3 <= cluster_id) {
                work_end_3 = work_end_3 + 1;
            }
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_v_lo_empty = 1;
            unsigned int _phase_v_hi_empty = 1;
            #pragma unroll 1
            for (unsigned int work_slot_3 = work_begin_3; work_slot_3 < work_end_3; work_slot_3++) {
                int work_idx_3 = work_slot_3;
                if ((unsigned int)extra_start_3 > cluster_id) {
                    work_idx_3 = (work_slot_3 - (unsigned int)work_begin_3) * (unsigned int)extra_start_3 + cluster_id;
                }
                int split_idx_2 = work_idx_3 % ((0) ? 3 : 1);
                int query_idx_2 = work_idx_3 / ((0) ? 3 : 1);
                int _max_0 = ((sparse_topk_lens[query_idx_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_2] + sparse_topk_lens_offset) : (0));
                int _min_0 = ((_max_0) < (sparse_topk) ? (_max_0) : (sparse_topk));
                int active_count_3 = _min_0;
                int active_tile_count_3 = (active_count_3 + 128 - 1) / 128;
                if (active_tile_count_3 < 1) {
                    active_tile_count_3 = 1;
                }
                int num_kv_tiles_3 = (active_tile_count_3 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                int first_kv_tile_1 = split_idx_2 * num_kv_tiles_3;
                if (load_warp_rank < 2) {
                    mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                    _phase_q_empty_0 ^= 1;
                    if (load_warp_rank == 0) {
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(131072)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            #pragma unroll
                            for (int q_stage = 0; q_stage < 8; q_stage++) {
                                tma_4d_gmem2smem_cta2(smem_q_addr + (unsigned int)(q_stage * 8192), (&tmap_q), 0, cta_rank * 64, q_stage, query_idx_2, ((q_full_addr) & 0xFEFFFFFF));
                            }
                        }
                    }
                    int idx_slot = load_iter & 1;
                    int idx_phase = load_iter >> 1 & 1;
                    int idx_base_bytes = idx_slot * 4608;
                    mbarrier_wait(index_ready_addr + (idx_slot) * 8, idx_phase);
                    #pragma unroll 1
                    for (int tile_3 = 0; tile_3 < num_kv_tiles_3; tile_3++) {
                        if (_elect_sync_0) {
                            #pragma unroll
                            for (int issue_group = 0; issue_group < 8; issue_group++) {
                                int group = load_warp_rank * 8 + issue_group;
                                int raw_rows[4];
                                int rows[4];
                                int index_offset = tile_3 * 128 + cta_rank * 64 + group * 4;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                                    : "r"(smem_indices_addr + (unsigned int)idx_base_bytes + (unsigned int)(index_offset * 4)));
                                #pragma unroll
                                for (int row_id = 0; row_id < 4; row_id++) {
                                    rows[row_id] = ((raw_rows[row_id] >= 0) ? raw_rows[row_id] : 0);
                                }
                                #pragma unroll
                                for (int saved_i = 0; saved_i < 4; saved_i++) {
                                    issuer_rows[issue_group * 4 + saved_i] = rows[saved_i];
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int qk_stage_1 = 0; qk_stage_1 < 4; qk_stage_1++) {
                            mbarrier_wait(k_empty_addr + (k_stage_1) * 8, _phase_k_empty);
                            if (load_warp_rank == 0) {
                                if (cta_rank == 0) {
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                    }
                                }
                            }
                            if (_elect_sync_0) {
                                if (first_kv_tile_1 + tile_3 == 0) {
                                    #pragma unroll
                                    for (int issue_group_1 = 0; issue_group_1 < 8; issue_group_1++) {
                                        int group_1 = load_warp_rank * 8 + issue_group_1;
                                        tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_1 * 512), (&tmap_swa_k), qk_stage_1 * 128, issuer_rows[issue_group_1 * 4], issuer_rows[issue_group_1 * 4 + 1], issuer_rows[issue_group_1 * 4 + 2], issuer_rows[issue_group_1 * 4 + 3], ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_1 * 512) + 8192, (&tmap_swa_k), qk_stage_1 * 128 + 64, issuer_rows[issue_group_1 * 4], issuer_rows[issue_group_1 * 4 + 1], issuer_rows[issue_group_1 * 4 + 2], issuer_rows[issue_group_1 * 4 + 3], ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    }
                                } else {
                                    #pragma unroll
                                    for (int issue_group_2 = 0; issue_group_2 < 8; issue_group_2++) {
                                        int group_2 = load_warp_rank * 8 + issue_group_2;
                                        tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_2 * 512), (&tmap_compressed_k), qk_stage_1 * 128, issuer_rows[issue_group_2 * 4], issuer_rows[issue_group_2 * 4 + 1], issuer_rows[issue_group_2 * 4 + 2], issuer_rows[issue_group_2 * 4 + 3], ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_2 * 512) + 8192, (&tmap_compressed_k), qk_stage_1 * 128 + 64, issuer_rows[issue_group_2 * 4], issuer_rows[issue_group_2 * 4 + 1], issuer_rows[issue_group_2 * 4 + 2], issuer_rows[issue_group_2 * 4 + 3], ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    }
                                }
                            }
                            k_stage_1 += 1;
                            if (k_stage_1 == 4) { k_stage_1 = 0; _phase_k_empty ^= 1; }
                        }
                    }
                } else {
                    int idx_slot_1 = load_iter & 1;
                    int idx_phase_1 = load_iter >> 1 & 1;
                    int idx_base_bytes_1 = idx_slot_1 * 4608;
                    mbarrier_wait(index_ready_addr + (idx_slot_1) * 8, idx_phase_1);
                    #pragma unroll 1
                    for (int tile_4 = 0; tile_4 < num_kv_tiles_3; tile_4++) {
                        #pragma unroll
                        for (int ps = 0; ps < 2; ps++) {
                            if (_elect_sync_0) {
                                if (ps == 0) {
                                    #pragma unroll
                                    for (int issue_group_3 = 0; issue_group_3 < 8; issue_group_3++) {
                                        int group_3 = (load_warp_rank - 2) * 8 + issue_group_3;
                                        int v_rows[4];
                                        int index_offset_1 = tile_4 * 128 + ps * 64 + group_3 * 4;
                                        int raw_rows_1[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                            : "r"(smem_indices_addr + (unsigned int)idx_base_bytes_1 + (unsigned int)(index_offset_1 * 4)));
                                        #pragma unroll
                                        for (int row_id_1 = 0; row_id_1 < 4; row_id_1++) {
                                            v_rows[row_id_1] = ((raw_rows_1[row_id_1] >= 0) ? raw_rows_1[row_id_1] : 0);
                                        }
                                        #pragma unroll
                                        for (int saved_i_1 = 0; saved_i_1 < 4; saved_i_1++) {
                                            issuer_rows[issue_group_3 * 4 + saved_i_1] = v_rows[saved_i_1];
                                        }
                                    }
                                    #pragma unroll
                                    for (int issue_group_4 = 0; issue_group_4 < 8; issue_group_4++) {
                                        int group_ahead = (load_warp_rank - 2) * 8 + issue_group_4;
                                        int index_offset_ahead = tile_4 * 128 + 64 + group_ahead * 4;
                                        int raw_rows_ahead[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_ahead[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_ahead[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_ahead[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_ahead[(0) + 3]))
                                            : "r"(smem_indices_addr + (unsigned int)idx_base_bytes_1 + (unsigned int)(index_offset_ahead * 4)));
                                        #pragma unroll
                                        for (int saved_i_2 = 0; saved_i_2 < 4; saved_i_2++) {
                                            issuer_rows[32 + issue_group_4 * 4 + saved_i_2] = raw_rows_ahead[saved_i_2];
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int saved_i_3 = 0; saved_i_3 < 32; saved_i_3++) {
                                        issuer_rows[saved_i_3] = ((issuer_rows[32 + saved_i_3] >= 0) ? issuer_rows[32 + saved_i_3] : 0);
                                    }
                                }
                            }
                            mbarrier_wait(v_lo_empty_addr + (v_stage) * 8, _phase_v_lo_empty);
                            mbarrier_wait(v_hi_empty_addr + (v_stage) * 8, _phase_v_hi_empty);
                            if (load_warp_rank == 2) {
                                if (cta_rank == 0) {
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((v_lo_full_addr + (v_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((v_hi_full_addr + (v_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                    }
                                }
                            }
                            if (_elect_sync_0) {
                                if (first_kv_tile_1 + tile_4 == 0) {
                                    #pragma unroll
                                    for (int issue_group_5 = 0; issue_group_5 < 8; issue_group_5++) {
                                        int group_4 = (load_warp_rank - 2) * 8 + issue_group_5;
                                        int rank_col = cta_rank * 128;
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_lo_addr + v_stage * 32768 + (unsigned int)(group_4 * 512), (&tmap_swa_v), rank_col, issuer_rows[issue_group_5 * 4], issuer_rows[issue_group_5 * 4 + 1], issuer_rows[issue_group_5 * 4 + 2], issuer_rows[issue_group_5 * 4 + 3], ((v_lo_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_lo_addr + v_stage * 32768 + (unsigned int)(group_4 * 512) + 8192, (&tmap_swa_v), rank_col + 64, issuer_rows[issue_group_5 * 4], issuer_rows[issue_group_5 * 4 + 1], issuer_rows[issue_group_5 * 4 + 2], issuer_rows[issue_group_5 * 4 + 3], ((v_lo_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_hi_addr + v_stage * 32768 + (unsigned int)(group_4 * 512), (&tmap_swa_v), 256 + rank_col, issuer_rows[issue_group_5 * 4], issuer_rows[issue_group_5 * 4 + 1], issuer_rows[issue_group_5 * 4 + 2], issuer_rows[issue_group_5 * 4 + 3], ((v_hi_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_hi_addr + v_stage * 32768 + (unsigned int)(group_4 * 512) + 8192, (&tmap_swa_v), 256 + rank_col + 64, issuer_rows[issue_group_5 * 4], issuer_rows[issue_group_5 * 4 + 1], issuer_rows[issue_group_5 * 4 + 2], issuer_rows[issue_group_5 * 4 + 3], ((v_hi_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    }
                                } else {
                                    #pragma unroll
                                    for (int issue_group_6 = 0; issue_group_6 < 8; issue_group_6++) {
                                        int group_5 = (load_warp_rank - 2) * 8 + issue_group_6;
                                        int rank_col_1 = cta_rank * 128;
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_lo_addr + v_stage * 32768 + (unsigned int)(group_5 * 512), (&tmap_compressed_v), rank_col_1, issuer_rows[issue_group_6 * 4], issuer_rows[issue_group_6 * 4 + 1], issuer_rows[issue_group_6 * 4 + 2], issuer_rows[issue_group_6 * 4 + 3], ((v_lo_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_lo_addr + v_stage * 32768 + (unsigned int)(group_5 * 512) + 8192, (&tmap_compressed_v), rank_col_1 + 64, issuer_rows[issue_group_6 * 4], issuer_rows[issue_group_6 * 4 + 1], issuer_rows[issue_group_6 * 4 + 2], issuer_rows[issue_group_6 * 4 + 3], ((v_lo_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_hi_addr + v_stage * 32768 + (unsigned int)(group_5 * 512), (&tmap_compressed_v), 256 + rank_col_1, issuer_rows[issue_group_6 * 4], issuer_rows[issue_group_6 * 4 + 1], issuer_rows[issue_group_6 * 4 + 2], issuer_rows[issue_group_6 * 4 + 3], ((v_hi_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(smem_v_hi_addr + v_stage * 32768 + (unsigned int)(group_5 * 512) + 8192, (&tmap_compressed_v), 256 + rank_col_1 + 64, issuer_rows[issue_group_6 * 4], issuer_rows[issue_group_6 * 4 + 1], issuer_rows[issue_group_6 * 4 + 2], issuer_rows[issue_group_6 * 4 + 3], ((v_hi_full_addr + (v_stage) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    }
                                }
                            }
                            v_stage += 1;
                            if (v_stage == 2) { v_stage = 0; _phase_v_lo_empty ^= 1; _phase_v_hi_empty ^= 1; }
                        }
                    }
                }
                int idx_done_slot = load_iter & 1;
                mbarrier_arrive(index_empty_addr + (idx_done_slot) * 8);
                load_iter = load_iter + 1;
            }
            if (load_warp_rank < 2) {
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
            }
        }
    }
    // ---- Role: index_warp ----
    if (warp == 13) {
        { // index_warp_main
            const int index_dummy = 0;
            int valid_iter_1 = 0;
            int work_base_4 = (unsigned int)total_work_items / num_clusters;
            int work_extra_4 = (unsigned int)total_work_items % num_clusters;
            int extra_start_4 = num_clusters - (unsigned int)work_extra_4;
            int extra_prefix_4 = cluster_id - (unsigned int)extra_start_4;
            if (extra_prefix_4 < 0) {
                extra_prefix_4 = 0;
            }
            int work_begin_4 = cluster_id * (unsigned int)work_base_4 + (unsigned int)extra_prefix_4;
            int work_end_4 = work_begin_4 + work_base_4;
            if ((unsigned int)extra_start_4 <= cluster_id) {
                work_end_4 = work_end_4 + 1;
            }
            #pragma unroll 1
            for (unsigned int work_slot_4 = work_begin_4; work_slot_4 < work_end_4; work_slot_4++) {
                int work_idx_4 = work_slot_4;
                if ((unsigned int)extra_start_4 > cluster_id) {
                    work_idx_4 = (work_slot_4 - (unsigned int)work_begin_4) * (unsigned int)extra_start_4 + cluster_id;
                }
                int split_idx_3 = work_idx_4 % ((0) ? 3 : 1);
                int query_idx_3 = work_idx_4 / ((0) ? 3 : 1);
                int owner_col_base = split_idx_3 * (9 / ((0) ? 3 : 1)) * 128;
                int* swa_row = swa_indices + (query_idx_3 * swa_index_stride);
                int* compressed_row = compressed_indices + (query_idx_3 * compressed_index_stride);
                int valid_slot_1 = valid_iter_1 & 1;
                int valid_empty_phase = valid_iter_1 >> 1 & 1 ^ 1;
                mbarrier_wait(index_valid_empty_addr + (valid_slot_1) * 8, valid_empty_phase);
                mbarrier_wait(index_empty_addr + (valid_slot_1) * 8, valid_empty_phase);
                int staged[4 * (9 / ((0) ? 3 : 1))];
                #pragma unroll
                for (int k = 0; k < 4 * (9 / ((0) ? 3 : 1)); k++) {
                    staged[k] = -1;
                }
                #pragma unroll
                for (int index_pass = 0; index_pass < 9 / ((0) ? 3 : 1); index_pass++) {
                    int index_offset_2 = index_pass * 128 + lane * 4;
                    int combined_col = owner_col_base + index_offset_2;
                    int* index_src = ((combined_col < 128) ? (swa_row + combined_col) : (compressed_row + (combined_col - 128)));
                    if (combined_col + 4 <= sparse_topk) {
                        int _vec_load_0[4];
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(index_src + 0);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            _vec_load_0[0 + 0] = _ivld_0.x;
                            _vec_load_0[0 + 1] = _ivld_0.y;
                            _vec_load_0[0 + 2] = _ivld_0.z;
                            _vec_load_0[0 + 3] = _ivld_0.w;
                        }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            staged[index_pass * 4 + i] = _vec_load_0[i];
                        }
                    }
                    if (combined_col < sparse_topk && combined_col + 4 > sparse_topk) {
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            if (combined_col + i_1 < sparse_topk) {
                                staged[index_pass * 4 + i_1] = index_src[i_1];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int index_pass_1 = 0; index_pass_1 < 9 / ((0) ? 3 : 1); index_pass_1++) {
                    int index_offset_3 = index_pass_1 * 128 + lane * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)((valid_slot_1 * 1152 + index_offset_3) * 4)), "r"(staged[index_pass_1 * 4]), "r"(staged[index_pass_1 * 4 + 1]), "r"(staged[index_pass_1 * 4 + 2]), "r"(staged[index_pass_1 * 4 + 3]) : "memory");
                    unsigned int valid_word = (((staged[index_pass_1 * 4] >= 0) ? 1 : 0) | ((staged[index_pass_1 * 4 + 1] >= 0) ? 1 : 0) << 1 | ((staged[index_pass_1 * 4 + 2] >= 0) ? 1 : 0) << 2 | ((staged[index_pass_1 * 4 + 3] >= 0) ? 1 : 0) << 3) << (lane & 7) * 4;
                    unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, valid_word, 1);
                    valid_word = valid_word | _shfl_xor_0;
                    unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, valid_word, 2);
                    valid_word = valid_word | _shfl_xor_1;
                    unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, valid_word, 4);
                    valid_word = valid_word | _shfl_xor_2;
                    if ((lane & 7) == 0) {
                        smem_index_valid[valid_slot_1 * 36 + index_pass_1 * 4 + (lane >> 3)] = valid_word;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(index_valid_full_addr + (valid_slot_1) * 8);
                mbarrier_arrive(index_ready_addr + (valid_slot_1) * 8);
                valid_iter_1 = valid_iter_1 + 1;
            }
        }
    }
    // ---- Role: pv_lo_mma_warp ----
    if (warp == 14) {
        { // pv_lo_mma_warp_main
            const int pv_lo_dummy = 0;
            unsigned int v_stage_1 = 0;
            int tile_cursor_3 = 0;
            unsigned int _phase_o_first_slice_seeded_0 = 0;
            mbarrier_wait(o_first_slice_seeded_addr, _phase_o_first_slice_seeded_0);
            _phase_o_first_slice_seeded_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int work_base_5 = (unsigned int)total_work_items / num_clusters;
            int work_extra_5 = (unsigned int)total_work_items % num_clusters;
            int extra_start_5 = num_clusters - (unsigned int)work_extra_5;
            int extra_prefix_5 = cluster_id - (unsigned int)extra_start_5;
            if (extra_prefix_5 < 0) {
                extra_prefix_5 = 0;
            }
            int work_begin_5 = cluster_id * (unsigned int)work_base_5 + (unsigned int)extra_prefix_5;
            int work_end_5 = work_begin_5 + work_base_5;
            if ((unsigned int)extra_start_5 <= cluster_id) {
                work_end_5 = work_end_5 + 1;
            }
            unsigned int _phase_o_lo_empty_0 = 1;
            unsigned int _phase_v_lo_full = 0;
            #pragma unroll 1
            for (unsigned int work_slot_5 = work_begin_5; work_slot_5 < work_end_5; work_slot_5++) {
                int work_idx_5 = work_slot_5;
                if ((unsigned int)extra_start_5 > cluster_id) {
                    work_idx_5 = (work_slot_5 - (unsigned int)work_begin_5) * (unsigned int)extra_start_5 + cluster_id;
                }
                int num_kv_tiles_4 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                {
                    int _max_2 = ((sparse_topk_lens[work_idx_5 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[work_idx_5 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) : (0));
                    int _min_2 = ((_max_2) < (sparse_topk) ? (_max_2) : (sparse_topk));
                    int active_count_4 = _min_2;
                    int active_tile_count_4 = (active_count_4 + 128 - 1) / 128;
                    if (active_tile_count_4 < 1) {
                        active_tile_count_4 = 1;
                    }
                    num_kv_tiles_4 = (active_tile_count_4 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                }
                if (cta_rank == 0) {
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int tile_5 = 0; tile_5 < num_kv_tiles_4; tile_5++) {
                        int pipeline_tile_3 = tile_cursor_3 + tile_5;
                        int p_stage_1 = 0;
                        int p_full_phase = pipeline_tile_3 & 1;
                        mbarrier_wait(o_lo_empty_addr, _phase_o_lo_empty_0);
                        _phase_o_lo_empty_0 ^= 1;
                        mbarrier_wait(p_full_addr + (p_stage_1) * 8, p_full_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int ps_1 = 0; ps_1 < 2; ps_1++) {
                            int smem_p_stage_1 = p_stage_1 * 2 + ps_1;
                            mbarrier_wait(v_lo_full_addr + (v_stage_1) * 8, _phase_v_lo_full);
                            int _mma_a_lo_2 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_1) * 512;
                            int _mma_b_lo_2 = ((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage_1) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (128))), "r"(((((ps_1 == 0) ? first_pv : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(v_lo_empty_addr + (v_stage_1) * 8, (uint16_t)(3));
                            v_stage_1 += 1;
                            if (v_stage_1 == 2) { v_stage_1 = 0; _phase_v_lo_full ^= 1; }
                        }
                        first_pv = 0;
                        elect_commit_cg2_multicast(p_empty_addr + (p_stage_1) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(o_lo_full_addr, (uint16_t)(3));
                    }
                    tile_cursor_3 = tile_cursor_3 + num_kv_tiles_4;
                }
            }
            if (cta_rank == 0) {
                elect_commit_cg2_multicast(pv_issue_done_addr, (uint16_t)(3));
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: pv_hi_mma_warp ----
    if (warp == 15) {
        { // pv_hi_mma_warp_main
            const int pv_hi_dummy = 0;
            unsigned int v_stage_2 = 0;
            int tile_cursor_4 = 0;
            unsigned int _phase_o_first_slice_seeded_0_1 = 0;
            mbarrier_wait(o_first_slice_seeded_addr, _phase_o_first_slice_seeded_0_1);
            _phase_o_first_slice_seeded_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int work_base_6 = (unsigned int)total_work_items / num_clusters;
            int work_extra_6 = (unsigned int)total_work_items % num_clusters;
            int extra_start_6 = num_clusters - (unsigned int)work_extra_6;
            int extra_prefix_6 = cluster_id - (unsigned int)extra_start_6;
            if (extra_prefix_6 < 0) {
                extra_prefix_6 = 0;
            }
            int work_begin_6 = cluster_id * (unsigned int)work_base_6 + (unsigned int)extra_prefix_6;
            int work_end_6 = work_begin_6 + work_base_6;
            if ((unsigned int)extra_start_6 <= cluster_id) {
                work_end_6 = work_end_6 + 1;
            }
            unsigned int _phase_o_hi_empty_0 = 1;
            unsigned int _phase_v_hi_full = 0;
            #pragma unroll 1
            for (unsigned int work_slot_6 = work_begin_6; work_slot_6 < work_end_6; work_slot_6++) {
                int work_idx_6 = work_slot_6;
                if ((unsigned int)extra_start_6 > cluster_id) {
                    work_idx_6 = (work_slot_6 - (unsigned int)work_begin_6) * (unsigned int)extra_start_6 + cluster_id;
                }
                int num_kv_tiles_5 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                {
                    int _max_3 = ((sparse_topk_lens[work_idx_6 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[work_idx_6 / ((0) ? 3 : 1)] + sparse_topk_lens_offset) : (0));
                    int _min_3 = ((_max_3) < (sparse_topk) ? (_max_3) : (sparse_topk));
                    int active_count_5 = _min_3;
                    int active_tile_count_5 = (active_count_5 + 128 - 1) / 128;
                    if (active_tile_count_5 < 1) {
                        active_tile_count_5 = 1;
                    }
                    num_kv_tiles_5 = (active_tile_count_5 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                }
                if (cta_rank == 0) {
                    int first_pv_1 = 1;
                    #pragma unroll 1
                    for (int tile_6 = 0; tile_6 < num_kv_tiles_5; tile_6++) {
                        int pipeline_tile_4 = tile_cursor_4 + tile_6;
                        int p_stage_2 = 0;
                        int p_full_phase_1 = pipeline_tile_4 & 1;
                        mbarrier_wait(o_hi_empty_addr, _phase_o_hi_empty_0);
                        _phase_o_hi_empty_0 ^= 1;
                        mbarrier_wait(p_full_addr + (p_stage_2) * 8, p_full_phase_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int ps_2 = 0; ps_2 < 2; ps_2++) {
                            int smem_p_stage_2 = p_stage_2 * 2 + ps_2;
                            mbarrier_wait(v_hi_full_addr + (v_stage_2) * 8, _phase_v_hi_full);
                            int _mma_a_lo_3 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_2) * 512;
                            int _mma_b_lo_3 = ((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage_2) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_scratch + (256))), "r"(((((ps_2 == 0) ? first_pv_1 : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(v_hi_empty_addr + (v_stage_2) * 8, (uint16_t)(3));
                            v_stage_2 += 1;
                            if (v_stage_2 == 2) { v_stage_2 = 0; _phase_v_hi_full ^= 1; }
                        }
                        first_pv_1 = 0;
                        elect_commit_cg2_multicast(p_empty_addr + (p_stage_2) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(o_hi_full_addr, (uint16_t)(3));
                    }
                    tile_cursor_4 = tile_cursor_4 + num_kv_tiles_5;
                }
            }
            if (cta_rank == 0) {
                elect_commit_cg2_multicast(pv_issue_done_addr, (uint16_t)(3));
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }

    // Cleanup
}

} // extern "C"
