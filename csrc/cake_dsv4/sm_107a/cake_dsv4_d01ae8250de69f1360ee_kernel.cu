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





__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)






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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}





__device__ __forceinline__ void tcgen05_commit_cg1_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_d01ae8250de69f1360ee(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_o, const __grid_constant__ CUtensorMap tmap_partial_o, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_lse, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items, int ragged_query, int max_q_len, int batch_size, int num_whole_items, int num_splits)
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
    #define q_stage_full_addr (mbar_base + 0)
    #define q_stage_empty_addr (mbar_base + 16)
    #define q_tmem_full_addr (mbar_base + 32)
    #define k_full_addr (mbar_base + 40)
    #define k_empty_addr (mbar_base + 56)
    #define v_full_addr (mbar_base + 72)
    #define v_empty_addr (mbar_base + 88)
    #define index_full_addr (mbar_base + 104)
    #define index_empty_addr (mbar_base + 152)
    #define s_full_addr (mbar_base + 200)
    #define s_empty_addr (mbar_base + 216)
    #define row_max_full_addr (mbar_base + 232)
    #define p_full_addr (mbar_base + 248)
    #define p_empty_addr (mbar_base + 264)
    #define stats_addr (mbar_base + 280)
    #define row_sum_full_addr (mbar_base + 296)
    #define sum_ready_addr (mbar_base + 304)
    #define sum_empty_addr (mbar_base + 312)
    #define o_empty_addr (mbar_base + 320)
    #define o_full_addr (mbar_base + 328)
    #define o_lo_full_addr (mbar_base + 336)
    #define tmem_dealloc_addr (mbar_base + 344)
    #define source_work_full_addr (mbar_base + 352)
    #define source_work_empty_addr (mbar_base + 368)
    #define source_throttle_full_addr (mbar_base + 384)
    #define source_throttle_empty_addr (mbar_base + 400)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v_addr = smem + 132096;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_p_addr = smem + 197632;
    __nv_bfloat16* smem_epi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 230400);
    const int smem_epi_addr = smem + 230400;
    int* smem_index = reinterpret_cast<int*>(smem_raw + 263168);
    const int smem_index_addr = smem + 263168;
    float* smem_peer_max = reinterpret_cast<float*>(smem_raw + 269312);
    const int smem_peer_max_addr = smem + 269312;
    float* smem_acc_scale = reinterpret_cast<float*>(smem_raw + 270336);
    const int smem_acc_scale_addr = smem + 270336;
    float* smem_peer_sum = reinterpret_cast<float*>(smem_raw + 271360);
    const int smem_peer_sum_addr = smem + 271360;
    float* smem_row_total = reinterpret_cast<float*>(smem_raw + 271872);
    const int smem_row_total_addr = smem + 271872;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 272384);
    const int work_response_addr = smem + 272384;

    // Mbarrier init (52 barriers at smem_raw[0..416))

    if (warp == 12) {
        // --- pipeline 'q_pipe' ---
        // q_stage_full: 2 barriers, init_count=1
        // q_stage_empty: 2 barriers, init_count=1
        // --- pipeline 'k_pipe' ---
        // k_full: 2 barriers, init_count=1
        // k_empty: 2 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        if (lane < 4 || (lane >= 5 && lane < 9)) {
            mbarrier_init(smem + 0 + lane * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 8) {
        // q_tmem_full: 1 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        if (lane < 1) {
            mbarrier_init(smem + 32 + lane * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 13) {
        // --- pipeline 'v_pipe' ---
        // v_full: 2 barriers, init_count=1
        // v_empty: 2 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        if (lane < 4) {
            mbarrier_init(smem + 72 + lane * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 9) {
        // --- pipeline 'index_pipe' ---
        // index_full: 6 barriers, init_count=32
        // index_empty: 6 barriers, init_count=256
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_9_13 = 256;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_9_13) : "r"(lane), "n"(6), "r"((uint32_t)(32)));
        if (lane < 12) {
            mbarrier_init(smem + 104 + lane * 8, _mbarrier_init_count_9_13);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 1) {
        // s_full: 2 barriers, init_count=1
        // s_empty: 2 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_1_25 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_1_25) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        if (lane < 4) {
            mbarrier_init(smem + 200 + lane * 8, _mbarrier_init_count_1_25);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 2) {
        // row_max_full: 2 barriers, init_count=1
        // p_full: 2 barriers, init_count=4
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_2_29 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_2_29) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        if (lane < 4) {
            mbarrier_init(smem + 232 + lane * 8, _mbarrier_init_count_2_29);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 3) {
        // p_empty: 2 barriers, init_count=2
        // row_sum_full: 1 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_3_33 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_3_33) : "r"(lane), "n"(2), "r"((uint32_t)(2)));
        if (lane < 2 || (lane >= 4 && lane < 5)) {
            mbarrier_init(smem + 264 + lane * 8, _mbarrier_init_count_3_33);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 5) {
        // stats: 2 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        if (lane < 2) {
            mbarrier_init(smem + 280 + lane * 8, 128);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 6) {
        // sum_ready: 1 barriers, init_count=128
        // sum_empty: 1 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        if (lane < 2) {
            mbarrier_init(smem + 304 + lane * 8, 128);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 4) {
        // o_empty: 1 barriers, init_count=128
        // o_full: 1 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_4_40 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_4_40) : "r"(lane), "n"(1), "r"((uint32_t)(128)));
        if (lane < 2) {
            mbarrier_init(smem + 320 + lane * 8, _mbarrier_init_count_4_40);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 7) {
        // o_lo_full: 1 barriers, init_count=1
        // tmem_dealloc: 1 barriers, init_count=448
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_7_42 = 448;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_7_42) : "r"(lane), "n"(1), "r"((uint32_t)(1)));
        if (lane < 2) {
            mbarrier_init(smem + 336 + lane * 8, _mbarrier_init_count_7_42);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 10) {
        // --- pipeline 'source_work_pipe' ---
        // source_work_full: 2 barriers, init_count=1
        // source_work_empty: 2 barriers, init_count=960
        // --- pipeline 'source_throttle_pipe' ---
        // source_throttle_full: 2 barriers, init_count=128
        // source_throttle_empty: 2 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_10_44 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_44) : "r"(lane), "n"(6), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_44) : "r"(lane), "n"(4), "r"((uint32_t)(960)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_44) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        if (lane < 8) {
            mbarrier_init(smem + 352 + lane * 8, _mbarrier_init_count_10_44);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (576 columns, 576 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 416);
    if (warp == 0) {
        int _tmem_hold = smem + 416;
        asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(576) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_s = taddr;
    const unsigned int tmem_o = taddr + 128;
    const unsigned int tmem_q = taddr + 384;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.442695f;
            const int cta_rank_1 = cta_rank;
            const int peer_rank = cta_rank_1 ^ 1;
            const int row = warp % 4 * 32 + lane;
            const int s_lane = warp % 4 * 32 << 16;
            int softmax_tile_cursor = 0;
            unsigned int softmax_index_stage = 0;
            uint32_t peer_max_slot;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(peer_max_slot) : "r"(smem_peer_max_addr + (unsigned int)(row * 4)), "r"(peer_rank));
            uint32_t peer_sum_slot;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(peer_sum_slot) : "r"(smem_peer_sum_addr + (unsigned int)(row * 4)), "r"(peer_rank));
            int p_warp_rows = cta_rank_1 * 8192 + warp % 4 * 2048;
            uint32_t p_peer_rows;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(p_peer_rows) : "r"(smem_p_addr + (unsigned int)p_warp_rows), "r"(peer_rank));
            unsigned int source_work_stage = 0;
            unsigned int source_work_x = blockIdx.x;
            unsigned int source_work_z = blockIdx.z;
            unsigned int _phase_row_sum_full_0 = 0;
            unsigned int _phase_sum_empty_0 = 1;
            unsigned int _phase_index_full = 0;
            unsigned int _phase_source_work_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx = 0; work_idx < num_clusters; work_idx++) {
                unsigned int item_query = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1);
                int item_piece = -1;
                if (item_query >= (unsigned int)num_whole_items) {
                    unsigned int piece_item = item_query - (unsigned int)num_whole_items;
                    item_query = (unsigned int)num_whole_items + piece_item / (unsigned int)num_splits;
                    item_piece = (int)(piece_item % (unsigned int)num_splits);
                }
                int item_valid = item_query < (unsigned int)num_query_tokens;
                int query = (int)item_query;
                if (item_valid != 0) {
                    int _max_0 = ((sparse_topk_lens[query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query] + sparse_topk_lens_offset) : (0));
                    int active_topk = ((_max_0) < (sparse_topk) ? (_max_0) : (sparse_topk));
                    int _max_1 = ((active_topk) > (1) ? (active_topk) : (1));
                    int num_kv_tiles = (_max_1 + 128 - 1) / 128;
                    int _max_2 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int tiles_per_piece = ((_max_2 + 128 - 1) / 128 + num_splits - 1) / num_splits;
                    int _max_3 = ((item_piece) > (0) ? (item_piece) : (0));
                    int _max_4 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int piece_tile0 = _max_3 * (((_max_4 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    int item_partial = 0;
                    if (item_piece >= 0 && num_kv_tiles > tiles_per_piece) {
                        int _min_6 = ((num_kv_tiles - piece_tile0) < (tiles_per_piece) ? (num_kv_tiles - piece_tile0) : (tiles_per_piece));
                        int _max_5 = ((_min_6) > (0) ? (_min_6) : (0));
                        num_kv_tiles = _max_5;
                        item_partial = 1;
                    } else if (item_piece > 0) {
                        num_kv_tiles = 0;
                    }
                    if (num_kv_tiles != 0) {
                        int active_cols = active_topk - piece_tile0 * 128;
                        int swa_tile_cols = active_cols;
                        if (piece_tile0 == 0) {
                            int query_batch = query / max_q_len;
                            int query_offset = query - query_batch * max_q_len;
                            int query_length = max_q_len;
                            if (ragged_query != 0) {
                                query_batch = 0;
                                #pragma unroll 2
                                for (int chunk = 0; chunk < (batch_size + 31) / 32; chunk++) {
                                    int lane_entry = chunk * 32 + lane + 1;
                                    int lane_load = ((lane_entry) < (batch_size) ? (lane_entry) : (batch_size));
                                    unsigned int started = __ballot_sync(0xFFFFFFFF, lane_entry <= batch_size && query >= cum_seq_lens_q[lane_load]);
                                    query_batch = query_batch + (int)(unsigned int)__popc(started);
                                }
                                int query_begin = cum_seq_lens_q[query_batch];
                                query_length = cum_seq_lens_q[query_batch + 1] - query_begin;
                                query_offset = query - query_begin;
                            }
                            int visible = seq_lens[query_batch] - query_length + query_offset + 1;
                            if (visible < 0) {
                                visible = 0;
                            }
                            if (visible > 128) {
                                visible = 128;
                            }
                            int _min_7 = ((active_topk) < (visible) ? (active_topk) : (visible));
                            swa_tile_cols = _min_7;
                        }
                        float row_max_val = -CUDART_INF_F;
                        float row_sum_val = 0.0f;
                        #pragma unroll 1
                        for (int tile = 0; tile < num_kv_tiles; tile++) {
                            int pipeline_tile = softmax_tile_cursor + tile;
                            int phase = (int)((unsigned int)pipeline_tile % 2);
                            int s_base = taddr + (unsigned int)(phase * 64) + (unsigned int)s_lane;
                            int valid_sparse_cols = ((tile == 0) ? swa_tile_cols : active_cols);
                            valid_sparse_cols = valid_sparse_cols - tile * 128 - cta_rank_1 * 64;
                            if (valid_sparse_cols < 0) {
                                valid_sparse_cols = 0;
                            }
                            if (valid_sparse_cols > 64) {
                                valid_sparse_cols = 64;
                            }
                            int _stage_s_full = (int)((unsigned int)(softmax_tile_cursor + tile) % 2);
                            int _parity_s_full = (int)((unsigned int)(softmax_tile_cursor + tile) / 2 % 2);
                            mbarrier_wait(s_full_addr + (_stage_s_full) * 8, _parity_s_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int index_lane_addr = smem_index_addr + softmax_index_stage * 1024 + (unsigned int)((tile & 1) * 512) + (unsigned int)((cta_rank_1 * 64 + lane) * 4);
                            int staged_index[2];
                            unsigned int invalid_cols[2];
                            unsigned int any_invalid = 0;
                            #pragma unroll
                            for (int w = 0; w < 2; w++) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[w])) : "r"(index_lane_addr + w * 128));
                                unsigned int _warp_vote_8 = __ballot_sync(0xFFFFFFFF, staged_index[w] < 0);
                                invalid_cols[w] = _warp_vote_8;
                                any_invalid = any_invalid | invalid_cols[w];
                            }
                            if ((tile & 1) != 0 || tile + 1 == num_kv_tiles) {
                                mbarrier_arrive(index_empty_addr + (softmax_index_stage) * 8);
                                softmax_index_stage += 1;
                                if (softmax_index_stage == 6) { softmax_index_stage = 0; _phase_index_full ^= 1; }
                            }
                            float score_tile_max = -CUDART_INF_F;
                            float score_tile_max_hi = -CUDART_INF_F;
                            float sv[64];
                            if (valid_sparse_cols == 64 && any_invalid == 0) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(score_tile_max)
                                    : "r"(s_base));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(score_tile_max_hi)
                                    : "r"(s_base + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                score_tile_max = max_noftz(score_tile_max, score_tile_max_hi);
                            } else if (valid_sparse_cols != 0) {
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                                    : "r"(s_base));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                                    : "r"(s_base + 32));
                                uint32_t _thresh_mask_9;
                                {
                                    int _lim_10 = valid_sparse_cols;
                                    if (_lim_10 <= 0) { _thresh_mask_9 = 0u; }
                                    else if (_lim_10 >= 32) { _thresh_mask_9 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_thresh_mask_9) : "r"(_lim_10));
                                    }
                                }
                                if (!(_thresh_mask_9 & (1u << 0))) sv[0] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 1))) sv[1] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 2))) sv[2] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 3))) sv[3] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 4))) sv[4] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 5))) sv[5] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 6))) sv[6] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 7))) sv[7] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 8))) sv[8] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 9))) sv[9] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 10))) sv[10] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 11))) sv[11] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 12))) sv[12] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 13))) sv[13] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 14))) sv[14] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 15))) sv[15] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 16))) sv[16] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 17))) sv[17] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 18))) sv[18] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 19))) sv[19] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 20))) sv[20] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 21))) sv[21] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 22))) sv[22] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 23))) sv[23] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 24))) sv[24] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 25))) sv[25] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 26))) sv[26] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 27))) sv[27] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 28))) sv[28] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 29))) sv[29] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 30))) sv[30] = -CUDART_INF_F;
                                if (!(_thresh_mask_9 & (1u << 31))) sv[31] = -CUDART_INF_F;
                                uint32_t _thresh_mask_11;
                                {
                                    int _lim_12 = valid_sparse_cols - 32;
                                    if (_lim_12 <= 0) { _thresh_mask_11 = 0u; }
                                    else if (_lim_12 >= 32) { _thresh_mask_11 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_thresh_mask_11) : "r"(_lim_12));
                                    }
                                }
                                if (!(_thresh_mask_11 & (1u << 0))) sv[32] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 1))) sv[33] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 2))) sv[34] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 3))) sv[35] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 4))) sv[36] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 5))) sv[37] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 6))) sv[38] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 7))) sv[39] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 8))) sv[40] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 9))) sv[41] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 10))) sv[42] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 11))) sv[43] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 12))) sv[44] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 13))) sv[45] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 14))) sv[46] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 15))) sv[47] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 16))) sv[48] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 17))) sv[49] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 18))) sv[50] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 19))) sv[51] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 20))) sv[52] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 21))) sv[53] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 22))) sv[54] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 23))) sv[55] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 24))) sv[56] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 25))) sv[57] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 26))) sv[58] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 27))) sv[59] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 28))) sv[60] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 29))) sv[61] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 30))) sv[62] = -CUDART_INF_F;
                                if (!(_thresh_mask_11 & (1u << 31))) sv[63] = -CUDART_INF_F;
                                if (any_invalid != 0) {
                                    sv[0] = (((invalid_cols[0] & 1) != 0) ? -CUDART_INF_F : sv[0]);
                                    sv[1] = (((invalid_cols[0] >> 1 & 1) != 0) ? -CUDART_INF_F : sv[1]);
                                    sv[2] = (((invalid_cols[0] >> 2 & 1) != 0) ? -CUDART_INF_F : sv[2]);
                                    sv[3] = (((invalid_cols[0] >> 3 & 1) != 0) ? -CUDART_INF_F : sv[3]);
                                    sv[4] = (((invalid_cols[0] >> 4 & 1) != 0) ? -CUDART_INF_F : sv[4]);
                                    sv[5] = (((invalid_cols[0] >> 5 & 1) != 0) ? -CUDART_INF_F : sv[5]);
                                    sv[6] = (((invalid_cols[0] >> 6 & 1) != 0) ? -CUDART_INF_F : sv[6]);
                                    sv[7] = (((invalid_cols[0] >> 7 & 1) != 0) ? -CUDART_INF_F : sv[7]);
                                    sv[8] = (((invalid_cols[0] >> 8 & 1) != 0) ? -CUDART_INF_F : sv[8]);
                                    sv[9] = (((invalid_cols[0] >> 9 & 1) != 0) ? -CUDART_INF_F : sv[9]);
                                    sv[10] = (((invalid_cols[0] >> 10 & 1) != 0) ? -CUDART_INF_F : sv[10]);
                                    sv[11] = (((invalid_cols[0] >> 11 & 1) != 0) ? -CUDART_INF_F : sv[11]);
                                    sv[12] = (((invalid_cols[0] >> 12 & 1) != 0) ? -CUDART_INF_F : sv[12]);
                                    sv[13] = (((invalid_cols[0] >> 13 & 1) != 0) ? -CUDART_INF_F : sv[13]);
                                    sv[14] = (((invalid_cols[0] >> 14 & 1) != 0) ? -CUDART_INF_F : sv[14]);
                                    sv[15] = (((invalid_cols[0] >> 15 & 1) != 0) ? -CUDART_INF_F : sv[15]);
                                    sv[16] = (((invalid_cols[0] >> 16 & 1) != 0) ? -CUDART_INF_F : sv[16]);
                                    sv[17] = (((invalid_cols[0] >> 17 & 1) != 0) ? -CUDART_INF_F : sv[17]);
                                    sv[18] = (((invalid_cols[0] >> 18 & 1) != 0) ? -CUDART_INF_F : sv[18]);
                                    sv[19] = (((invalid_cols[0] >> 19 & 1) != 0) ? -CUDART_INF_F : sv[19]);
                                    sv[20] = (((invalid_cols[0] >> 20 & 1) != 0) ? -CUDART_INF_F : sv[20]);
                                    sv[21] = (((invalid_cols[0] >> 21 & 1) != 0) ? -CUDART_INF_F : sv[21]);
                                    sv[22] = (((invalid_cols[0] >> 22 & 1) != 0) ? -CUDART_INF_F : sv[22]);
                                    sv[23] = (((invalid_cols[0] >> 23 & 1) != 0) ? -CUDART_INF_F : sv[23]);
                                    sv[24] = (((invalid_cols[0] >> 24 & 1) != 0) ? -CUDART_INF_F : sv[24]);
                                    sv[25] = (((invalid_cols[0] >> 25 & 1) != 0) ? -CUDART_INF_F : sv[25]);
                                    sv[26] = (((invalid_cols[0] >> 26 & 1) != 0) ? -CUDART_INF_F : sv[26]);
                                    sv[27] = (((invalid_cols[0] >> 27 & 1) != 0) ? -CUDART_INF_F : sv[27]);
                                    sv[28] = (((invalid_cols[0] >> 28 & 1) != 0) ? -CUDART_INF_F : sv[28]);
                                    sv[29] = (((invalid_cols[0] >> 29 & 1) != 0) ? -CUDART_INF_F : sv[29]);
                                    sv[30] = (((invalid_cols[0] >> 30 & 1) != 0) ? -CUDART_INF_F : sv[30]);
                                    sv[31] = (((invalid_cols[0] >> 31 & 1) != 0) ? -CUDART_INF_F : sv[31]);
                                    sv[32] = (((invalid_cols[1] & 1) != 0) ? -CUDART_INF_F : sv[32]);
                                    sv[33] = (((invalid_cols[1] >> 1 & 1) != 0) ? -CUDART_INF_F : sv[33]);
                                    sv[34] = (((invalid_cols[1] >> 2 & 1) != 0) ? -CUDART_INF_F : sv[34]);
                                    sv[35] = (((invalid_cols[1] >> 3 & 1) != 0) ? -CUDART_INF_F : sv[35]);
                                    sv[36] = (((invalid_cols[1] >> 4 & 1) != 0) ? -CUDART_INF_F : sv[36]);
                                    sv[37] = (((invalid_cols[1] >> 5 & 1) != 0) ? -CUDART_INF_F : sv[37]);
                                    sv[38] = (((invalid_cols[1] >> 6 & 1) != 0) ? -CUDART_INF_F : sv[38]);
                                    sv[39] = (((invalid_cols[1] >> 7 & 1) != 0) ? -CUDART_INF_F : sv[39]);
                                    sv[40] = (((invalid_cols[1] >> 8 & 1) != 0) ? -CUDART_INF_F : sv[40]);
                                    sv[41] = (((invalid_cols[1] >> 9 & 1) != 0) ? -CUDART_INF_F : sv[41]);
                                    sv[42] = (((invalid_cols[1] >> 10 & 1) != 0) ? -CUDART_INF_F : sv[42]);
                                    sv[43] = (((invalid_cols[1] >> 11 & 1) != 0) ? -CUDART_INF_F : sv[43]);
                                    sv[44] = (((invalid_cols[1] >> 12 & 1) != 0) ? -CUDART_INF_F : sv[44]);
                                    sv[45] = (((invalid_cols[1] >> 13 & 1) != 0) ? -CUDART_INF_F : sv[45]);
                                    sv[46] = (((invalid_cols[1] >> 14 & 1) != 0) ? -CUDART_INF_F : sv[46]);
                                    sv[47] = (((invalid_cols[1] >> 15 & 1) != 0) ? -CUDART_INF_F : sv[47]);
                                    sv[48] = (((invalid_cols[1] >> 16 & 1) != 0) ? -CUDART_INF_F : sv[48]);
                                    sv[49] = (((invalid_cols[1] >> 17 & 1) != 0) ? -CUDART_INF_F : sv[49]);
                                    sv[50] = (((invalid_cols[1] >> 18 & 1) != 0) ? -CUDART_INF_F : sv[50]);
                                    sv[51] = (((invalid_cols[1] >> 19 & 1) != 0) ? -CUDART_INF_F : sv[51]);
                                    sv[52] = (((invalid_cols[1] >> 20 & 1) != 0) ? -CUDART_INF_F : sv[52]);
                                    sv[53] = (((invalid_cols[1] >> 21 & 1) != 0) ? -CUDART_INF_F : sv[53]);
                                    sv[54] = (((invalid_cols[1] >> 22 & 1) != 0) ? -CUDART_INF_F : sv[54]);
                                    sv[55] = (((invalid_cols[1] >> 23 & 1) != 0) ? -CUDART_INF_F : sv[55]);
                                    sv[56] = (((invalid_cols[1] >> 24 & 1) != 0) ? -CUDART_INF_F : sv[56]);
                                    sv[57] = (((invalid_cols[1] >> 25 & 1) != 0) ? -CUDART_INF_F : sv[57]);
                                    sv[58] = (((invalid_cols[1] >> 26 & 1) != 0) ? -CUDART_INF_F : sv[58]);
                                    sv[59] = (((invalid_cols[1] >> 27 & 1) != 0) ? -CUDART_INF_F : sv[59]);
                                    sv[60] = (((invalid_cols[1] >> 28 & 1) != 0) ? -CUDART_INF_F : sv[60]);
                                    sv[61] = (((invalid_cols[1] >> 29 & 1) != 0) ? -CUDART_INF_F : sv[61]);
                                    sv[62] = (((invalid_cols[1] >> 30 & 1) != 0) ? -CUDART_INF_F : sv[62]);
                                    sv[63] = (((invalid_cols[1] >> 31 & 1) != 0) ? -CUDART_INF_F : sv[63]);
                                }
                                float2 _reg_reduce_max2_14 = {-CUDART_INF_F, -CUDART_INF_F};
                                row_max_x32_accum(&sv[0], _reg_reduce_max2_14);
                                row_max_x32_accum(&sv[32], _reg_reduce_max2_14);
                                float _reg_array_reduce_13 = row_max_reduce(_reg_reduce_max2_14);
                                score_tile_max = _reg_array_reduce_13;
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(s_empty_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8);
                            float new_max;
                            asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(row_max_val), "f"(score_tile_max));
                            uint32_t _mbar_peer_15;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_15) : "r"(row_max_full_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8), "r"(peer_rank));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(peer_max_slot + (unsigned int)(phase * 512)), "f"(new_max), "r"(_mbar_peer_15) : "memory");
                            if (warp % 4 == 0) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(row_max_full_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8, 512);
                                }
                            }
                            int _stage_row_max_full = (int)((unsigned int)(softmax_tile_cursor + tile) % 2);
                            int _parity_row_max_full = (int)((unsigned int)(softmax_tile_cursor + tile) / 2 % 2);
                            asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(row_max_full_addr + (_stage_row_max_full) * 8), "r"(uint32_t(_parity_row_max_full)), "r"(uint32_t(10000000)) : "memory");
                            asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
                            new_max = max_noftz(new_max, smem_peer_max[phase * 128 + row]);
                            if (row_max_val > -CUDART_INF_F && (new_max - row_max_val) * softmax_scale_log2 <= 4.0f) {
                                new_max = row_max_val;
                            }
                            float delta = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max * softmax_scale_log2);
                            float exp_delta = approx_exp2(delta);
                            float acc_scale = ((row_max_val > -CUDART_INF_F) ? exp_delta : 1.0f);
                            int _stage_p_empty = (int)((unsigned int)(softmax_tile_cursor + tile) % 2);
                            int _parity_p_empty = (int)((unsigned int)(softmax_tile_cursor + tile) / 2 % 2) ^ 1;
                            mbarrier_wait(p_empty_addr + (_stage_p_empty) * 8, _parity_p_empty);
                            smem_acc_scale[phase * 128 + row] = acc_scale;
                            mbarrier_arrive(stats_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8);
                            row_max_val = new_max;
                            float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                            float max_scaled = safe_max * softmax_scale_log2;
                            unsigned int p_fp8[16];
                            float block_sum = 0.0f;
                            if (valid_sparse_cols != 0) {
                                const float2 _fma_b2_16 = {softmax_scale_log2, softmax_scale_log2};
                                const float2 _fma_c2_17 = {4.0f - max_scaled, 4.0f - max_scaled};
                                #pragma unroll
                                for (int _lf = 0; _lf < 32; _lf++)
                                    fma_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_lf], _fma_b2_16, _fma_c2_17);
                                #pragma unroll
                                for (int _le = 0; _le < 64; _le++) {
                                    sv[_le] = approx_exp2(sv[_le]);
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
                                        : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                                           "f"(sv[2]), "f"(sv[3]));
                                    p_fp8[0] = _packed;
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
                                    p_fp8[1] = _packed;
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
                                    p_fp8[2] = _packed;
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
                                    p_fp8[3] = _packed;
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
                                    p_fp8[4] = _packed;
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
                                    p_fp8[5] = _packed;
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
                                    p_fp8[6] = _packed;
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
                                    p_fp8[7] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[32]), "f"(sv[33]),
                                                           "f"(sv[34]), "f"(sv[35]));
                                    p_fp8[8] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[36]), "f"(sv[37]),
                                                           "f"(sv[38]), "f"(sv[39]));
                                    p_fp8[9] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[40]), "f"(sv[41]),
                                                           "f"(sv[42]), "f"(sv[43]));
                                    p_fp8[10] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[44]), "f"(sv[45]),
                                                           "f"(sv[46]), "f"(sv[47]));
                                    p_fp8[11] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[48]), "f"(sv[49]),
                                                           "f"(sv[50]), "f"(sv[51]));
                                    p_fp8[12] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[52]), "f"(sv[53]),
                                                           "f"(sv[54]), "f"(sv[55]));
                                    p_fp8[13] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[56]), "f"(sv[57]),
                                                           "f"(sv[58]), "f"(sv[59]));
                                    p_fp8[14] = _packed;
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
                                        : "=r"(_packed) : "f"(sv[60]), "f"(sv[61]),
                                                           "f"(sv[62]), "f"(sv[63]));
                                    p_fp8[15] = _packed;
                                }
                            } else {
                                #pragma unroll
                                for (int p_word = 0; p_word < 16; p_word++) {
                                    p_fp8[p_word] = 0;
                                }
                            }
                            int p_stage_rows = smem_p_addr + (unsigned int)(phase * 16384 + p_warp_rows);
                            #pragma unroll
                            for (int p_vec = 0; p_vec < 4; p_vec++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)(phase * 16384) + (unsigned int)((cta_rank_1 * 64 + p_vec * 16) / 64 * 8192 + row * 64 + (cta_rank_1 * 64 + p_vec * 16) % 64 ^ ((cta_rank_1 * 64 + p_vec * 16) / 64 * 8192 + row * 64 + (cta_rank_1 * 64 + p_vec * 16) % 64 >> 7 & 3) << 4))), "r"(p_fp8[p_vec * 4]), "r"(p_fp8[p_vec * 4 + 1]), "r"(p_fp8[p_vec * 4 + 2]), "r"(p_fp8[p_vec * 4 + 3]) : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                uint32_t _mbar_peer_18;
                                asm(
                                    "mapa.shared::cluster.u32 %0, %1, %2;"
                                    : "=r"(_mbar_peer_18) : "r"(p_full_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8), "r"(peer_rank));
                                asm volatile(
                                    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                    " [%0], [%1], %2, [%3];"
                                    :: "r"(p_peer_rows + (unsigned int)(phase * 16384)), "r"(p_stage_rows), "r"((uint32_t)(2048)), "r"(_mbar_peer_18)
                                    : "memory");
                                if (warp % 4 == 0) {
                                    mbarrier_arrive_expect_tx(p_full_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8, 8192);
                                } else {
                                    mbarrier_arrive(p_full_addr + ((int)((unsigned int)(softmax_tile_cursor + tile) % 2)) * 8);
                                }
                            }
                            if (valid_sparse_cols != 0) {
                                const float2* _reg_reduce_src2_20 = reinterpret_cast<const float2*>(&sv[0]);
                                float2 _reg_reduce_sum2_20_0 = make_float2(0.0f, 0.0f);
                                float2 _reg_reduce_sum2_20_1 = make_float2(0.0f, 0.0f);
                                float2 _reg_reduce_sum2_20_2 = make_float2(0.0f, 0.0f);
                                float2 _reg_reduce_sum2_20_3 = make_float2(0.0f, 0.0f);
                                #pragma unroll
                                for (int _rr = 0; _rr < 8; _rr++) {
                                    add_f32x2_inplace(&_reg_reduce_sum2_20_0, _reg_reduce_src2_20[_rr * 4 + 0]);
                                    add_f32x2_inplace(&_reg_reduce_sum2_20_1, _reg_reduce_src2_20[_rr * 4 + 1]);
                                    add_f32x2_inplace(&_reg_reduce_sum2_20_2, _reg_reduce_src2_20[_rr * 4 + 2]);
                                    add_f32x2_inplace(&_reg_reduce_sum2_20_3, _reg_reduce_src2_20[_rr * 4 + 3]);
                                }
                                add_f32x2_inplace(&_reg_reduce_sum2_20_0, _reg_reduce_sum2_20_1);
                                add_f32x2_inplace(&_reg_reduce_sum2_20_2, _reg_reduce_sum2_20_3);
                                add_f32x2_inplace(&_reg_reduce_sum2_20_0, _reg_reduce_sum2_20_2);
                                float _reg_array_reduce_19 = _reg_reduce_sum2_20_0.x + _reg_reduce_sum2_20_0.y;
                                block_sum = _reg_array_reduce_19;
                            }
                            row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                        }
                        uint32_t _mbar_peer_21;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_21) : "r"(row_sum_full_addr), "r"(peer_rank));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(peer_sum_slot), "f"(row_sum_val), "r"(_mbar_peer_21) : "memory");
                        if (warp % 4 == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(row_sum_full_addr, 512);
                            }
                        }
                        asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(row_sum_full_addr), "r"(uint32_t(_phase_row_sum_full_0)), "r"(uint32_t(10000000)) : "memory");
                        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
                        _phase_row_sum_full_0 ^= 1;
                        float peer_sum = smem_peer_sum[row];
                        float sum_lo = ((cta_rank_1 == 0) ? row_sum_val : peer_sum);
                        float sum_hi = ((cta_rank_1 == 0) ? peer_sum : row_sum_val);
                        if (has_sinks != 0 && row < num_heads && item_piece <= 0) {
                            float sink_max = sinks[row] * 1.442695f / softmax_scale_log2;
                            if (row_max_val > -CUDART_INF_F) {
                                float sink_delta = __fmaf_rn(sink_max, softmax_scale_log2, __fmaf_rn(-row_max_val, softmax_scale_log2, 4.0f));
                                float _exp2_22 = approx_exp2(sink_delta);
                                sum_lo = sum_lo + _exp2_22;
                            } else {
                                sum_lo = 16.0f;
                            }
                        }
                        mbarrier_wait(sum_empty_addr, _phase_sum_empty_0);
                        _phase_sum_empty_0 ^= 1;
                        smem_row_total[row] = sum_lo + sum_hi;
                        mbarrier_arrive(sum_ready_addr);
                        if (item_piece >= 0 && cta_rank_1 == 0 && row < num_heads) {
                            float piece_lse = CUDART_INF_F;
                            if (item_partial != 0) {
                                float piece_total = sum_lo + sum_hi;
                                piece_lse = -CUDART_INF_F;
                                if (row_max_val > -CUDART_INF_F) {
                                    float _log2_23;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_23) : "f"(piece_total));
                                    piece_lse = __fmaf_rn(row_max_val, softmax_scale_log2, _log2_23 - 4.0f);
                                }
                                if (has_sinks != 0 && item_piece == 0) {
                                    if (row_max_val == -CUDART_INF_F || piece_total == CUDART_INF_F) {
                                        piece_lse = sinks[row] * 1.442695f;
                                    }
                                }
                            }
                            partial_lse[((query - num_whole_items) * num_splits + item_piece) * num_heads + row] = piece_lse;
                        }
                    }
                    softmax_tile_cursor += num_kv_tiles;
                    if (num_kv_tiles == 0) {
                        mbarrier_wait(index_full_addr + (softmax_index_stage) * 8, _phase_index_full);
                        mbarrier_arrive(index_empty_addr + (softmax_index_stage) * 8);
                        softmax_index_stage += 1;
                        if (softmax_index_stage == 6) { softmax_index_stage = 0; _phase_index_full ^= 1; }
                        if (cta_rank_1 == 0 && row < num_heads) {
                            partial_lse[((query - num_whole_items) * num_splits + item_piece) * num_heads + row] = -CUDART_INF_F;
                        }
                    }
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage) * 8, _phase_source_work_full);
                uint32_t clc_valid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid)
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid)
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1)
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2)
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage) * 8), "r"(0) : "memory");
                source_work_stage += 1;
                if (source_work_stage == 2) { source_work_stage = 0; _phase_source_work_full ^= 1; }
                if (clc_valid == 0) {
                    break;
                }
                source_work_x = clc_ctaid + (unsigned int)cta_rank;
                source_work_z = clc_ctaid_2;
            }
            int _stage_p_empty_1 = (int)((unsigned int)softmax_tile_cursor % 2);
            int _parity_p_empty_1 = (int)((unsigned int)softmax_tile_cursor / 2 % 2) ^ 1;
            mbarrier_wait(p_empty_addr + (_stage_p_empty_1) * 8, _parity_p_empty_1);
            int _stage_p_empty_2 = (int)((unsigned int)(softmax_tile_cursor + 1) % 2);
            int _parity_p_empty_2 = (int)((unsigned int)(softmax_tile_cursor + 1) / 2 % 2) ^ 1;
            mbarrier_wait(p_empty_addr + (_stage_p_empty_2) * 8, _parity_p_empty_2);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // correction_wg_main
            float output_scale = bmm2_scale[0];
            const int cta_rank_2 = cta_rank;
            const int row_1 = warp % 4 * 32 + lane;
            const int corr_row = warp % 4 * 32 << 16;
            int correction_tile_cursor = 0;
            int tmem_released = 0;
            const int o_tmem = taddr + 128 + (unsigned int)corr_row;
            const int epi_head0 = warp % 4 * 32;
            const int epi_stage_base = smem_epi_addr + (unsigned int)(warp % 4 * 4096);
            unsigned int source_work_stage_1 = 0;
            unsigned int source_work_x_1 = blockIdx.x;
            unsigned int source_work_z_1 = blockIdx.z;
            unsigned int _phase_sum_ready_0 = 0;
            unsigned int _phase_o_lo_full_0 = 0;
            unsigned int _phase_source_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = 0; work_idx_1 < num_clusters; work_idx_1++) {
                unsigned int item_query_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
                int item_piece_1 = -1;
                if (item_query_1 >= (unsigned int)num_whole_items) {
                    unsigned int piece_item_1 = item_query_1 - (unsigned int)num_whole_items;
                    item_query_1 = (unsigned int)num_whole_items + piece_item_1 / (unsigned int)num_splits;
                    item_piece_1 = (int)(piece_item_1 % (unsigned int)num_splits);
                }
                int item_valid_1 = item_query_1 < (unsigned int)num_query_tokens;
                int query_1 = (int)item_query_1;
                if (item_valid_1 != 0) {
                    int _max_0_1 = ((sparse_topk_lens[query_1] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_1] + sparse_topk_lens_offset) : (0));
                    int active_topk_1 = ((_max_0_1) < (sparse_topk) ? (_max_0_1) : (sparse_topk));
                    int _max_1_1 = ((active_topk_1) > (1) ? (active_topk_1) : (1));
                    int num_kv_tiles_1 = (_max_1_1 + 128 - 1) / 128;
                    int _max_2_1 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int tiles_per_piece_1 = ((_max_2_1 + 128 - 1) / 128 + num_splits - 1) / num_splits;
                    int _max_3_1 = ((item_piece_1) > (0) ? (item_piece_1) : (0));
                    int _max_4_1 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int piece_tile0_1 = _max_3_1 * (((_max_4_1 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    int item_partial_1 = 0;
                    if (item_piece_1 >= 0 && num_kv_tiles_1 > tiles_per_piece_1) {
                        int _min_6_1 = ((num_kv_tiles_1 - piece_tile0_1) < (tiles_per_piece_1) ? (num_kv_tiles_1 - piece_tile0_1) : (tiles_per_piece_1));
                        int _max_5_1 = ((_min_6_1) > (0) ? (_min_6_1) : (0));
                        num_kv_tiles_1 = _max_5_1;
                        item_partial_1 = 1;
                    } else if (item_piece_1 > 0) {
                        num_kv_tiles_1 = 0;
                    }
                    if (num_kv_tiles_1 != 0) {
                        int epi_row = query_1 * num_heads;
                        if (item_partial_1 != 0) {
                            epi_row = ((query_1 - num_whole_items) * num_splits + item_piece_1) * num_heads;
                        }
                        int _stage_stats = (int)((unsigned int)correction_tile_cursor % 2);
                        int _parity_stats = (int)((unsigned int)correction_tile_cursor / 2 % 2);
                        mbarrier_wait(stats_addr + (_stage_stats) * 8, _parity_stats);
                        mbarrier_arrive(o_empty_addr);
                        #pragma unroll 1
                        for (int tile_1 = 1; tile_1 < num_kv_tiles_1; tile_1++) {
                            int pipeline_tile_1 = correction_tile_cursor + tile_1;
                            int _stage_stats_1 = (int)((unsigned int)(correction_tile_cursor + tile_1) % 2);
                            int _parity_stats_1 = (int)((unsigned int)(correction_tile_cursor + tile_1) / 2 % 2);
                            mbarrier_wait(stats_addr + (_stage_stats_1) * 8, _parity_stats_1);
                            float acc_scale_1 = smem_acc_scale[pipeline_tile_1 % 2 * 128 + row_1];
                            int _parity_o_full = (int)((unsigned int)(correction_tile_cursor + tile_1) % 2) ^ 1;
                            mbarrier_wait(o_full_addr, _parity_o_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int any_rescale = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                            if (any_rescale != 0) {
                                #pragma unroll
                                for (int o_col = 0; o_col < 256; o_col += 16) {
                                    float o_part[16];
                                    tmem_ld_x16(&o_part[0], o_tmem + o_col);
                                    const float2 _scale2_7 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_part)[_ls], _scale2_7);
                                    tmem_st_x16_f32(o_tmem + o_col, o_part);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
                        }
                        correction_tile_cursor += num_kv_tiles_1;
                        mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
                        _phase_sum_ready_0 ^= 1;
                        float row_total = smem_row_total[row_1];
                        mbarrier_arrive(sum_empty_addr);
                        float _rcp_8 = approx_rcp(row_total);
                        float inv_sum = ((row_total > 0.0f) ? _rcp_8 : 0.0f);
                        mbarrier_wait(o_lo_full_addr, _phase_o_lo_full_0);
                        _phase_o_lo_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
                        uint32_t next_valid = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %0, 1, 0, p1;\n\t"
                            "}\n"
                            : "=r"(next_valid)
                            : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                            : "memory");
                        float epi_scale = inv_sum * output_scale;
                        float epi_ring[32];
                        tmem_ld_x16(&epi_ring[0], o_tmem);
                        #pragma unroll 1
                        for (int box = 0; box < 4; box++) {
                            int epi_box_base = epi_stage_base + (box & 1) * 16384;
                            if (elect_sync()) {
                                if (box == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                } else if (box >= 2) {
                                    asm volatile("cp.async.bulk.wait_group.read 1;");
                                }
                            }
                            __syncwarp();
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (box * 4 + 1 < 16) {
                                tmem_ld_x16(&epi_ring[16], o_tmem + (box * 64 + 16));
                            }
                            float epi_scaled_0[16];
                            #pragma unroll
                            for (int elem_0 = 0; elem_0 < 16; elem_0++) {
                                epi_scaled_0[elem_0] = epi_ring[elem_0] * epi_scale;
                            }
                            unsigned int epi_packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi_scaled_0[_lp*2 + 0], epi_scaled_0[_lp*2+1 + 0]));
                                epi_packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            int epi_col_0 = 0;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_0 ^ (lane * 128 + epi_col_0 >> 7 & 7) << 4))), "r"(epi_packed_0[0]), "r"(epi_packed_0[1]), "r"(epi_packed_0[2]), "r"(epi_packed_0[3]) : "memory");
                            int epi_col_1 = 16;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_1 ^ (lane * 128 + epi_col_1 >> 7 & 7) << 4))), "r"(epi_packed_0[4]), "r"(epi_packed_0[5]), "r"(epi_packed_0[6]), "r"(epi_packed_0[7]) : "memory");
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (box * 4 + 1 + 1 < 16) {
                                tmem_ld_x16(&epi_ring[0], o_tmem + (box * 64 + 32));
                            }
                            float epi_scaled_1[16];
                            #pragma unroll
                            for (int elem_1 = 0; elem_1 < 16; elem_1++) {
                                epi_scaled_1[elem_1] = epi_ring[16 + elem_1] * epi_scale;
                            }
                            unsigned int epi_packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi_scaled_1[_lp*2 + 0], epi_scaled_1[_lp*2+1 + 0]));
                                epi_packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            int epi_col_2 = 32;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_2 ^ (lane * 128 + epi_col_2 >> 7 & 7) << 4))), "r"(epi_packed_1[0]), "r"(epi_packed_1[1]), "r"(epi_packed_1[2]), "r"(epi_packed_1[3]) : "memory");
                            int epi_col_3 = 48;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_3 ^ (lane * 128 + epi_col_3 >> 7 & 7) << 4))), "r"(epi_packed_1[4]), "r"(epi_packed_1[5]), "r"(epi_packed_1[6]), "r"(epi_packed_1[7]) : "memory");
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (box * 4 + 2 + 1 < 16) {
                                tmem_ld_x16(&epi_ring[16], o_tmem + (box * 64 + 48));
                            }
                            float epi_scaled_2[16];
                            #pragma unroll
                            for (int elem_2 = 0; elem_2 < 16; elem_2++) {
                                epi_scaled_2[elem_2] = epi_ring[elem_2] * epi_scale;
                            }
                            unsigned int epi_packed_2[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi_scaled_2[_lp*2 + 0], epi_scaled_2[_lp*2+1 + 0]));
                                epi_packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            int epi_col_4 = 64;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_4 ^ (lane * 128 + epi_col_4 >> 7 & 7) << 4))), "r"(epi_packed_2[0]), "r"(epi_packed_2[1]), "r"(epi_packed_2[2]), "r"(epi_packed_2[3]) : "memory");
                            int epi_col_5 = 80;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_5 ^ (lane * 128 + epi_col_5 >> 7 & 7) << 4))), "r"(epi_packed_2[4]), "r"(epi_packed_2[5]), "r"(epi_packed_2[6]), "r"(epi_packed_2[7]) : "memory");
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (box == 1) {
                                int _parity_o_full_1 = (int)((unsigned int)correction_tile_cursor % 2) ^ 1;
                                mbarrier_wait(o_full_addr, _parity_o_full_1);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (box == 3 && next_valid == 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(tmem_dealloc_addr);
                                tmem_released = 1;
                            }
                            if (box * 4 + 3 + 1 < 16) {
                                tmem_ld_x16(&epi_ring[0], o_tmem + (box * 64 + 64));
                            }
                            float epi_scaled_3[16];
                            #pragma unroll
                            for (int elem_3 = 0; elem_3 < 16; elem_3++) {
                                epi_scaled_3[elem_3] = epi_ring[16 + elem_3] * epi_scale;
                            }
                            unsigned int epi_packed_3[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi_scaled_3[_lp*2 + 0], epi_scaled_3[_lp*2+1 + 0]));
                                epi_packed_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            int epi_col_6 = 96;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_6 ^ (lane * 128 + epi_col_6 >> 7 & 7) << 4))), "r"(epi_packed_3[0]), "r"(epi_packed_3[1]), "r"(epi_packed_3[2]), "r"(epi_packed_3[3]) : "memory");
                            int epi_col_7 = 112;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_box_base + (lane * 128 + epi_col_7 ^ (lane * 128 + epi_col_7 >> 7 & 7) << 4))), "r"(epi_packed_3[4]), "r"(epi_packed_3[5]), "r"(epi_packed_3[6]), "r"(epi_packed_3[7]) : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            __syncwarp();
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (epi_head0 < num_heads) {
                                if (elect_sync()) {
                                    tma_store_3d(((item_partial_1 != 0) ? ((&tmap_partial_o)) : ((&tmap_o))), 0, epi_row + epi_head0, cta_rank_2 * 4 + box, epi_box_base);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
                uint32_t clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_1)
                    : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_3)
                    : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_1)
                    : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_1)
                    : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_1) * 8), "r"(0) : "memory");
                source_work_stage_1 += 1;
                if (source_work_stage_1 == 2) { source_work_stage_1 = 0; _phase_source_work_full_1 ^= 1; }
                if (clc_valid_1 == 0) {
                    break;
                }
                source_work_x_1 = clc_ctaid_3 + (unsigned int)cta_rank;
                source_work_z_1 = clc_ctaid_2_1;
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tmem_released == 0) {
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int mma_k_stage = 0;
            unsigned int q_copy_stage = 0;
            int mma_tile_cursor = 0;
            int q_item = 0;
            unsigned int source_work_stage_2 = 0;
            unsigned int source_work_x_2 = blockIdx.x;
            unsigned int source_work_z_2 = blockIdx.z;
            unsigned int _phase_q_stage_full = 0;
            unsigned int _phase_q_tmem_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_source_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = 0; work_idx_2 < num_clusters; work_idx_2++) {
                unsigned int item_query_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1);
                int item_piece_2 = -1;
                if (item_query_2 >= (unsigned int)num_whole_items) {
                    unsigned int piece_item_2 = item_query_2 - (unsigned int)num_whole_items;
                    item_query_2 = (unsigned int)num_whole_items + piece_item_2 / (unsigned int)num_splits;
                    item_piece_2 = (int)(piece_item_2 % (unsigned int)num_splits);
                }
                int item_valid_2 = item_query_2 < (unsigned int)num_query_tokens;
                int query_2 = (int)item_query_2;
                if (item_valid_2 != 0) {
                    int _max_0_2 = ((sparse_topk_lens[query_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_2] + sparse_topk_lens_offset) : (0));
                    int active_topk_2 = ((_max_0_2) < (sparse_topk) ? (_max_0_2) : (sparse_topk));
                    int _max_1_2 = ((active_topk_2) > (1) ? (active_topk_2) : (1));
                    int num_kv_tiles_2 = (_max_1_2 + 128 - 1) / 128;
                    int _max_2_2 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int tiles_per_piece_2 = ((_max_2_2 + 128 - 1) / 128 + num_splits - 1) / num_splits;
                    int _max_3_2 = ((item_piece_2) > (0) ? (item_piece_2) : (0));
                    int _max_4_2 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int piece_tile0_2 = _max_3_2 * (((_max_4_2 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    int item_partial_2 = 0;
                    if (item_piece_2 >= 0 && num_kv_tiles_2 > tiles_per_piece_2) {
                        int _min_6_2 = ((num_kv_tiles_2 - piece_tile0_2) < (tiles_per_piece_2) ? (num_kv_tiles_2 - piece_tile0_2) : (tiles_per_piece_2));
                        int _max_5_2 = ((_min_6_2) > (0) ? (_min_6_2) : (0));
                        num_kv_tiles_2 = _max_5_2;
                        item_partial_2 = 1;
                    } else if (item_piece_2 > 0) {
                        num_kv_tiles_2 = 0;
                    }
                    if (num_kv_tiles_2 != 0) {
                        unsigned int q_lo = (unsigned int)(2 * q_item) % 3;
                        unsigned int q_hi = (unsigned int)(2 * q_item + 1) % 3;
                        mbarrier_wait(q_stage_full_addr + (q_copy_stage) * 8, _phase_q_stage_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_7 = elect_sync();
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 0)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64))), "l"(_tcgen05_cp_desc_8), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 8))), "l"(_tcgen05_cp_desc_9), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 16))), "l"(_tcgen05_cp_desc_10), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 24))), "l"(_tcgen05_cp_desc_11), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 32))), "l"(_tcgen05_cp_desc_12), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16416)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 40))), "l"(_tcgen05_cp_desc_13), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16448)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 48))), "l"(_tcgen05_cp_desc_14), "r"(_elected_7)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16480)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_lo * 64 + 56))), "l"(_tcgen05_cp_desc_15), "r"(_elected_7)
                                : "memory");
                        }
                        tcgen05_commit_leader(q_stage_empty_addr + (q_copy_stage) * 8, _elected_7);
                        tcgen05_commit_leader(q_tmem_full_addr, _elected_7);
                        q_copy_stage += 1;
                        if (q_copy_stage == 2) { q_copy_stage = 0; _phase_q_stage_full ^= 1; }
                        mbarrier_wait(q_tmem_full_addr, _phase_q_tmem_full_0);
                        _phase_q_tmem_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(q_stage_full_addr + (q_copy_stage) * 8, _phase_q_stage_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_16 = elect_sync();
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 0)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64))), "l"(_tcgen05_cp_desc_17), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 8))), "l"(_tcgen05_cp_desc_18), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 16))), "l"(_tcgen05_cp_desc_19), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_20 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 24))), "l"(_tcgen05_cp_desc_20), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_21 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 32))), "l"(_tcgen05_cp_desc_21), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_22 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16416)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 40))), "l"(_tcgen05_cp_desc_22), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_23 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16448)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 48))), "l"(_tcgen05_cp_desc_23), "r"(_elected_16)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_24 = ((((uint64_t)(smem_q_addr + q_copy_stage * 32768 + 16480)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_q + (unsigned int)(q_hi * 64 + 56))), "l"(_tcgen05_cp_desc_24), "r"(_elected_16)
                                : "memory");
                        }
                        tcgen05_commit_leader(q_stage_empty_addr + (q_copy_stage) * 8, _elected_16);
                        tcgen05_commit_leader(q_tmem_full_addr, _elected_16);
                        q_copy_stage += 1;
                        if (q_copy_stage == 2) { q_copy_stage = 0; _phase_q_stage_full ^= 1; }
                        mbarrier_wait(q_tmem_full_addr, _phase_q_tmem_full_0);
                        _phase_q_tmem_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll 1
                        for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                            int s_tile = mma_tile_cursor + tile_2;
                            mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                            int _stage_s_empty = (int)((unsigned int)(mma_tile_cursor + tile_2) % 2);
                            int _parity_s_empty = (int)((unsigned int)(mma_tile_cursor + tile_2) / 2 % 2) ^ 1;
                            mbarrier_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int k_lo_0 = (((smem_k_addr) >> 4) & 0x7FFF) + (mma_k_stage) * 2048;
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
                    "mov.b32 id, 672137232;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 508;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_s + ((int)((unsigned int)s_tile % 2) * 64))), "r"(k_lo_0), "r"(tmem_q + (unsigned int)(q_lo * 64)), "r"(0) : "memory");
                            int k_lo_1 = (((smem_k_addr + 16384) >> 4) & 0x7FFF) + (mma_k_stage) * 2048;
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
                    "mov.b32 id, 672137232;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 508;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_s + ((int)((unsigned int)s_tile % 2) * 64))), "r"(k_lo_1), "r"(tmem_q + (unsigned int)(q_hi * 64)), "r"(1) : "memory");
                            uint32_t _elected_25 = elect_sync();
                            tcgen05_commit_leader(k_empty_addr + (mma_k_stage) * 8, _elected_25);
                            tcgen05_commit_leader(s_full_addr + ((int)((unsigned int)(mma_tile_cursor + tile_2) % 2)) * 8, _elected_25);
                            mma_k_stage += 1;
                            if (mma_k_stage == 2) { mma_k_stage = 0; _phase_k_full ^= 1; }
                        }
                        mma_tile_cursor += num_kv_tiles_2;
                        q_item = q_item + 1;
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_wait(source_work_full_addr + (source_work_stage_2) * 8, _phase_source_work_full_2);
                uint32_t clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_2)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_4)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_2)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_2)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_2) * 8), "r"(0) : "memory");
                source_work_stage_2 += 1;
                if (source_work_stage_2 == 2) { source_work_stage_2 = 0; _phase_source_work_full_2 ^= 1; }
                if (clc_valid_2 == 0) {
                    break;
                }
                source_work_x_2 = clc_ctaid_4 + (unsigned int)cta_rank;
                source_work_z_2 = clc_ctaid_2_2;
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(576));
        }
    }
    // ---- Role: index_warp ----
    if (warp == 9) {
        { // index_warp_main
            unsigned int index_prod_stage = 0;
            unsigned int source_work_stage_3 = 0;
            unsigned int source_work_x_3 = blockIdx.x;
            unsigned int source_work_z_3 = blockIdx.z;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_source_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = 0; work_idx_3 < num_clusters; work_idx_3++) {
                unsigned int item_query_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1);
                int item_piece_3 = -1;
                if (item_query_3 >= (unsigned int)num_whole_items) {
                    unsigned int piece_item_3 = item_query_3 - (unsigned int)num_whole_items;
                    item_query_3 = (unsigned int)num_whole_items + piece_item_3 / (unsigned int)num_splits;
                    item_piece_3 = (int)(piece_item_3 % (unsigned int)num_splits);
                }
                int item_valid_3 = item_query_3 < (unsigned int)num_query_tokens;
                int query_3 = (int)item_query_3;
                if (item_valid_3 != 0) {
                    int _max_0_3 = ((item_piece_3) > (0) ? (item_piece_3) : (0));
                    int _max_1_3 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int index_tile0 = _max_0_3 * (((_max_1_3 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    mbarrier_wait(index_empty_addr + (index_prod_stage) * 8, _phase_index_empty);
                    int index_stage_base = smem_index_addr + index_prod_stage * 1024;
                    #pragma unroll
                    for (int index_half = 0; index_half < 2; index_half++) {
                        int index_offset = index_half * 128 + lane * 4;
                        int tile_col_base = index_tile0 * 128 + index_half * 128;
                        int* row_tile_ptr = ((tile_col_base < 128) ? (swa_indices + query_3 * swa_index_stride) : (compressed_indices + (query_3 * compressed_index_stride + (tile_col_base - 128))));
                        int lane_col = tile_col_base + lane * 4;
                        if (lane_col + 4 <= sparse_topk) {
                            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;"
                                :: "r"(index_stage_base + index_offset * 4), "l"(row_tile_ptr + lane * 4));
                        } else {
                            int tail_rows[4];
                            int tail_fill = -1;
                            if (tile_col_base < sparse_topk) {
                                tail_fill = row_tile_ptr[0];
                            }
                            #pragma unroll
                            for (int row_i = 0; row_i < 4; row_i++) {
                                tail_rows[row_i] = tail_fill;
                            }
                            #pragma unroll
                            for (int row_i_1 = 0; row_i_1 < 4; row_i_1++) {
                                if (lane_col + row_i_1 < sparse_topk) {
                                    tail_rows[row_i_1] = row_tile_ptr[lane * 4 + row_i_1];
                                }
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(index_stage_base + index_offset * 4), "r"(tail_rows[0]), "r"(tail_rows[1]), "r"(tail_rows[2]), "r"(tail_rows[3]) : "memory");
                        }
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(index_full_addr + (index_prod_stage) * 8) : "memory");
                    mbarrier_arrive(index_full_addr + (index_prod_stage) * 8);
                    index_prod_stage += 1;
                    if (index_prod_stage == 6) { index_prod_stage = 0; _phase_index_empty ^= 1; }
                    int _max_2_3 = ((sparse_topk_lens[query_3] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_3] + sparse_topk_lens_offset) : (0));
                    int active_topk_3 = ((_max_2_3) < (sparse_topk) ? (_max_2_3) : (sparse_topk));
                    int _max_3_3 = ((active_topk_3) > (1) ? (active_topk_3) : (1));
                    int num_kv_tiles_3 = (_max_3_3 + 128 - 1) / 128;
                    int _max_4_3 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int tiles_per_piece_3 = ((_max_4_3 + 128 - 1) / 128 + num_splits - 1) / num_splits;
                    int _max_5_3 = ((item_piece_3) > (0) ? (item_piece_3) : (0));
                    int _max_6 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int piece_tile0_3 = _max_5_3 * (((_max_6 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    int item_partial_3 = 0;
                    if (item_piece_3 >= 0 && num_kv_tiles_3 > tiles_per_piece_3) {
                        int _min_8 = ((num_kv_tiles_3 - piece_tile0_3) < (tiles_per_piece_3) ? (num_kv_tiles_3 - piece_tile0_3) : (tiles_per_piece_3));
                        int _max_7 = ((_min_8) > (0) ? (_min_8) : (0));
                        num_kv_tiles_3 = _max_7;
                        item_partial_3 = 1;
                    } else if (item_piece_3 > 0) {
                        num_kv_tiles_3 = 0;
                    }
                    int _max_9 = (((num_kv_tiles_3 + 1) / 2) > (1) ? ((num_kv_tiles_3 + 1) / 2) : (1));
                    #pragma unroll 1
                    for (int index_pass = 1; index_pass < _max_9; index_pass++) {
                        mbarrier_wait(index_empty_addr + (index_prod_stage) * 8, _phase_index_empty);
                        int index_stage_base_0 = smem_index_addr + index_prod_stage * 1024;
                        #pragma unroll
                        for (int index_half_1 = 0; index_half_1 < 2; index_half_1++) {
                            int index_offset_1 = index_half_1 * 128 + lane * 4;
                            int tile_col_base_1 = index_tile0 * 128 + index_pass * 256 + index_half_1 * 128;
                            int* row_tile_ptr_1 = ((tile_col_base_1 < 128) ? (swa_indices + query_3 * swa_index_stride) : (compressed_indices + (query_3 * compressed_index_stride + (tile_col_base_1 - 128))));
                            int lane_col_1 = tile_col_base_1 + lane * 4;
                            if (lane_col_1 + 4 <= sparse_topk) {
                                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;"
                                    :: "r"(index_stage_base_0 + index_offset_1 * 4), "l"(row_tile_ptr_1 + lane * 4));
                            } else {
                                int tail_rows_1[4];
                                int tail_fill_1 = -1;
                                if (tile_col_base_1 < sparse_topk) {
                                    tail_fill_1 = row_tile_ptr_1[0];
                                }
                                #pragma unroll
                                for (int row_i_2 = 0; row_i_2 < 4; row_i_2++) {
                                    tail_rows_1[row_i_2] = tail_fill_1;
                                }
                                #pragma unroll
                                for (int row_i_3 = 0; row_i_3 < 4; row_i_3++) {
                                    if (lane_col_1 + row_i_3 < sparse_topk) {
                                        tail_rows_1[row_i_3] = row_tile_ptr_1[lane * 4 + row_i_3];
                                    }
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(index_stage_base_0 + index_offset_1 * 4), "r"(tail_rows_1[0]), "r"(tail_rows_1[1]), "r"(tail_rows_1[2]), "r"(tail_rows_1[3]) : "memory");
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(index_full_addr + (index_prod_stage) * 8) : "memory");
                        mbarrier_arrive(index_full_addr + (index_prod_stage) * 8);
                        index_prod_stage += 1;
                        if (index_prod_stage == 6) { index_prod_stage = 0; _phase_index_empty ^= 1; }
                    }
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_3) * 8, _phase_source_work_full_3);
                uint32_t clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_3)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_5)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_3)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_3)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_3) * 8), "r"(0) : "memory");
                source_work_stage_3 += 1;
                if (source_work_stage_3 == 2) { source_work_stage_3 = 0; _phase_source_work_full_3 ^= 1; }
                if (clc_valid_3 == 0) {
                    break;
                }
                source_work_x_3 = clc_ctaid_5 + (unsigned int)cta_rank;
                source_work_z_3 = clc_ctaid_2_3;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            unsigned int source_work_stage_4 = 0;
            unsigned int source_throttle_stage = 0;
            unsigned int _phase_source_throttle_full = 0;
            unsigned int _phase_source_work_empty = 1;
            unsigned int _phase_source_work_full_4 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _source_work = 0; _source_work < num_clusters; _source_work++) {
                    mbarrier_wait(source_throttle_full_addr + (source_throttle_stage) * 8, _phase_source_throttle_full);
                    mbarrier_arrive(source_throttle_empty_addr + (source_throttle_stage) * 8);
                    source_throttle_stage += 1;
                    if (source_throttle_stage == 2) { source_throttle_stage = 0; _phase_source_throttle_full ^= 1; }
                    mbarrier_wait(source_work_empty_addr + (source_work_stage_4) * 8, _phase_source_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(source_work_full_addr + (source_work_stage_4) * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16), "r"(source_work_full_addr + source_work_stage_4 * 8)
                            : "memory");
                    }
                    mbarrier_wait(source_work_full_addr + (source_work_stage_4) * 8, _phase_source_work_full_4);
                    uint32_t clc_valid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid_4)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_6 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_6)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_1_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_1_4)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_2_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_2_4)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    source_work_stage_4 += 1;
                    if (source_work_stage_4 == 2) { source_work_stage_4 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_4 ^= 1; }
                    if (clc_valid_4 == 0) {
                        break;
                    }
                }
                #pragma unroll
                for (unsigned int _source_tail = 0; _source_tail < 2; _source_tail++) {
                    mbarrier_wait(source_work_empty_addr + (source_work_stage_4) * 8, _phase_source_work_empty);
                    source_work_stage_4 += 1;
                    if (source_work_stage_4 == 2) { source_work_stage_4 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_4 ^= 1; }
                }
            }
        }
    }
    // ---- Role: pv_warp ----
    if (warp == 11) {
        { // pv_warp_main
            unsigned int pv_v_stage = 0;
            int pv_tile_cursor = 0;
            unsigned int source_work_stage_5 = 0;
            unsigned int source_work_x_4 = blockIdx.x;
            unsigned int source_work_z_4 = blockIdx.z;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_source_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = 0; work_idx_4 < num_clusters; work_idx_4++) {
                unsigned int item_query_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1);
                int item_piece_4 = -1;
                if (item_query_4 >= (unsigned int)num_whole_items) {
                    unsigned int piece_item_4 = item_query_4 - (unsigned int)num_whole_items;
                    item_query_4 = (unsigned int)num_whole_items + piece_item_4 / (unsigned int)num_splits;
                    item_piece_4 = (int)(piece_item_4 % (unsigned int)num_splits);
                }
                int item_valid_4 = item_query_4 < (unsigned int)num_query_tokens;
                int query_4 = (int)item_query_4;
                if (item_valid_4 != 0) {
                    int _max_0_4 = ((sparse_topk_lens[query_4] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_4] + sparse_topk_lens_offset) : (0));
                    int active_topk_4 = ((_max_0_4) < (sparse_topk) ? (_max_0_4) : (sparse_topk));
                    int _max_1_4 = ((active_topk_4) > (1) ? (active_topk_4) : (1));
                    int num_kv_tiles_4 = (_max_1_4 + 128 - 1) / 128;
                    int _max_2_4 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int tiles_per_piece_4 = ((_max_2_4 + 128 - 1) / 128 + num_splits - 1) / num_splits;
                    int _max_3_4 = ((item_piece_4) > (0) ? (item_piece_4) : (0));
                    int _max_4_4 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int piece_tile0_4 = _max_3_4 * (((_max_4_4 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    int item_partial_4 = 0;
                    if (item_piece_4 >= 0 && num_kv_tiles_4 > tiles_per_piece_4) {
                        int _min_6_3 = ((num_kv_tiles_4 - piece_tile0_4) < (tiles_per_piece_4) ? (num_kv_tiles_4 - piece_tile0_4) : (tiles_per_piece_4));
                        int _max_5_4 = ((_min_6_3) > (0) ? (_min_6_3) : (0));
                        num_kv_tiles_4 = _max_5_4;
                        item_partial_4 = 1;
                    } else if (item_piece_4 > 0) {
                        num_kv_tiles_4 = 0;
                    }
                    if (num_kv_tiles_4 != 0) {
                        #pragma unroll 1
                        for (int tile_3 = 0; tile_3 < num_kv_tiles_4; tile_3++) {
                            int pipeline_tile_2 = pv_tile_cursor + tile_3;
                            int _stage_p_full = (int)((unsigned int)(pv_tile_cursor + tile_3) % 2);
                            int _parity_p_full = (int)((unsigned int)(pv_tile_cursor + tile_3) / 2 % 2);
                            mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                            int _parity_o_empty = (int)((unsigned int)(pv_tile_cursor + tile_3) % 2);
                            mbarrier_wait(o_empty_addr, _parity_o_empty);
                            mbarrier_wait(v_full_addr + (pv_v_stage) * 8, _phase_v_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int pv_init = tile_3 == 0;
                            int p_lo = (((smem_p_addr) >> 4) & 0x7FFF) + ((int)((unsigned int)pipeline_tile_2 % 2)) * 1024;
                            if (tile_3 + 1 == num_kv_tiles_4) {
                                int v_lo_half_0 = ((((smem_v_addr) >> 4) & 0x7FFF) | 0x4000000) + (pv_v_stage) * 2048;
                                uint32_t _elected_7 = elect_sync();
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 673251344;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 512;\n\t"
                    "add.u32 blo, blo, 512;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(p_lo), "r"(v_lo_half_0), "r"(tmem_o), "r"(((pv_init) ? 0 : 1)), "r"(_elected_7) : "memory");
                                tcgen05_commit_leader(o_lo_full_addr, _elected_7);
                                int v_lo_half_1 = ((((smem_v_addr + 16384) >> 4) & 0x7FFF) | 0x4000000) + (pv_v_stage) * 2048;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 673251344;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 512;\n\t"
                    "add.u32 blo, blo, 512;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(p_lo), "r"(v_lo_half_1), "r"((tmem_o + (128))), "r"(((pv_init) ? 0 : 1)) : "memory");
                            } else {
                                int v_lo = ((((smem_v_addr) >> 4) & 0x7FFF) | 0x4000000) + (pv_v_stage) * 2048;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 675348496;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 512;\n\t"
                    "add.u32 blo, blo, 512;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(p_lo), "r"(v_lo), "r"(tmem_o), "r"(((pv_init) ? 0 : 1)) : "memory");
                            }
                            uint32_t _elected_8 = elect_sync();
                            tcgen05_commit_leader(v_empty_addr + (pv_v_stage) * 8, _elected_8);
                            tcgen05_commit_cg1_multicast_leader(p_empty_addr + ((int)((unsigned int)(pv_tile_cursor + tile_3) % 2)) * 8, (uint16_t)(3), _elected_8);
                            tcgen05_commit_leader(o_full_addr, _elected_8);
                            pv_v_stage += 1;
                            if (pv_v_stage == 2) { pv_v_stage = 0; _phase_v_full ^= 1; }
                        }
                        pv_tile_cursor += num_kv_tiles_4;
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_wait(source_work_full_addr + (source_work_stage_5) * 8, _phase_source_work_full_5);
                uint32_t clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_5)
                    : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_7)
                    : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_5)
                    : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_5)
                    : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_5) * 8), "r"(0) : "memory");
                source_work_stage_5 += 1;
                if (source_work_stage_5 == 2) { source_work_stage_5 = 0; _phase_source_work_full_5 ^= 1; }
                if (clc_valid_5 == 0) {
                    break;
                }
                source_work_x_4 = clc_ctaid_7 + (unsigned int)cta_rank;
                source_work_z_4 = clc_ctaid_2_5;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_warp_main
            const int load_rank = warp - 12;
            const int cta_rank_3 = cta_rank;
            unsigned int load_k_stage = 0;
            unsigned int load_v_stage = 0;
            unsigned int load_k_index_stage = 0;
            unsigned int load_v_index_stage = 0;
            unsigned int q_load_stage = 0;
            unsigned int source_throttle_stage_1 = 0;
            unsigned int source_work_stage_6 = 0;
            unsigned int source_work_x_5 = blockIdx.x;
            unsigned int source_work_z_5 = blockIdx.z;
            unsigned int _phase_source_throttle_empty = 1;
            unsigned int _phase_q_stage_empty = 1;
            unsigned int _phase_index_full_1 = 0;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_index_full_2 = 0;
            unsigned int _phase_v_empty = 1;
            unsigned int _phase_source_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_5 = 0; work_idx_5 < num_clusters; work_idx_5++) {
                unsigned int item_query_5 = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1);
                int item_piece_5 = -1;
                if (item_query_5 >= (unsigned int)num_whole_items) {
                    unsigned int piece_item_5 = item_query_5 - (unsigned int)num_whole_items;
                    item_query_5 = (unsigned int)num_whole_items + piece_item_5 / (unsigned int)num_splits;
                    item_piece_5 = (int)(piece_item_5 % (unsigned int)num_splits);
                }
                int item_valid_5 = item_query_5 < (unsigned int)num_query_tokens;
                int query_5 = (int)item_query_5;
                if (bid % 2 == 0) {
                    mbarrier_wait(source_throttle_empty_addr + (source_throttle_stage_1) * 8, _phase_source_throttle_empty);
                    mbarrier_arrive(source_throttle_full_addr + (source_throttle_stage_1) * 8);
                    source_throttle_stage_1 += 1;
                    if (source_throttle_stage_1 == 2) { source_throttle_stage_1 = 0; _phase_source_throttle_empty ^= 1; }
                }
                if (item_valid_5 != 0) {
                    int _max_0_5 = ((sparse_topk_lens[query_5] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_5] + sparse_topk_lens_offset) : (0));
                    int active_topk_5 = ((_max_0_5) < (sparse_topk) ? (_max_0_5) : (sparse_topk));
                    int _max_1_5 = ((active_topk_5) > (1) ? (active_topk_5) : (1));
                    int num_kv_tiles_5 = (_max_1_5 + 128 - 1) / 128;
                    int _max_2_5 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int tiles_per_piece_5 = ((_max_2_5 + 128 - 1) / 128 + num_splits - 1) / num_splits;
                    int _max_3_5 = ((item_piece_5) > (0) ? (item_piece_5) : (0));
                    int _max_4_5 = ((sparse_topk) > (1) ? (sparse_topk) : (1));
                    int piece_tile0_5 = _max_3_5 * (((_max_4_5 + 128 - 1) / 128 + num_splits - 1) / num_splits);
                    int item_partial_5 = 0;
                    if (item_piece_5 >= 0 && num_kv_tiles_5 > tiles_per_piece_5) {
                        int _min_6_4 = ((num_kv_tiles_5 - piece_tile0_5) < (tiles_per_piece_5) ? (num_kv_tiles_5 - piece_tile0_5) : (tiles_per_piece_5));
                        int _max_5_5 = ((_min_6_4) > (0) ? (_min_6_4) : (0));
                        num_kv_tiles_5 = _max_5_5;
                        item_partial_5 = 1;
                    } else if (item_piece_5 > 0) {
                        num_kv_tiles_5 = 0;
                    }
                    if (load_rank < 2) {
                        if (load_rank == 0) {
                            if (num_kv_tiles_5 != 0) {
                                mbarrier_wait(q_stage_empty_addr + (q_load_stage) * 8, _phase_q_stage_empty);
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(q_stage_full_addr + (q_load_stage) * 8, 32768);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 0, (&tmap_q), 0, 0, 0, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 8192, (&tmap_q), 0, 64, 0, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 16384, (&tmap_q), 0, 0, 1, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 24576, (&tmap_q), 0, 64, 1, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                }
                                q_load_stage += 1;
                                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_stage_empty ^= 1; }
                                mbarrier_wait(q_stage_empty_addr + (q_load_stage) * 8, _phase_q_stage_empty);
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(q_stage_full_addr + (q_load_stage) * 8, 32768);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 0, (&tmap_q), 0, 0, 2, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 8192, (&tmap_q), 0, 64, 2, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 16384, (&tmap_q), 0, 0, 3, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                    tma_4d_gmem2smem(smem_q_addr + q_load_stage * 32768 + 24576, (&tmap_q), 0, 64, 3, query_5, q_stage_full_addr + (q_load_stage) * 8);
                                }
                                q_load_stage += 1;
                                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_stage_empty ^= 1; }
                            }
                        }
                        int _max_7_1 = (((num_kv_tiles_5 + 1) / 2) > (1) ? ((num_kv_tiles_5 + 1) / 2) : (1));
                        #pragma unroll 1
                        for (int index_pass_1 = 0; index_pass_1 < _max_7_1; index_pass_1++) {
                            mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full_1);
                            int _min_8_1 = ((2 * index_pass_1 + 2) < (num_kv_tiles_5) ? (2 * index_pass_1 + 2) : (num_kv_tiles_5));
                            #pragma unroll 1
                            for (int tile_4 = 2 * index_pass_1; tile_4 < _min_8_1; tile_4++) {
                                int k_index_base = smem_index_addr + load_k_index_stage * 1024 + (unsigned int)((tile_4 & 1) * 512);
                                mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                                if (load_rank == 0) {
                                    if (elect_sync()) {
                                        mbarrier_arrive_expect_tx(k_full_addr + (load_k_stage) * 8, 32768);
                                    }
                                }
                                int k_stage_base = smem_k_addr + load_k_stage * 32768;
                                if (elect_sync()) {
                                    int first_rows[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&first_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows[(0) + 3]))
                                        : "r"(k_index_base));
                                    int fill = ((first_rows[0]) > (0) ? (first_rows[0]) : (0));
                                    int k_fill = fill;
                                    #pragma unroll
                                    for (int k_group_local = 0; k_group_local < 8; k_group_local++) {
                                        int k_group = load_rank * 8 + k_group_local;
                                        int k_rows[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&k_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_rows[(0) + 3]))
                                            : "r"(k_index_base + (cta_rank_3 * 64 + k_group * 4) * 4));
                                        #pragma unroll
                                        for (int k_chunk = 0; k_chunk < 4; k_chunk++) {
                                            tma_gather4_gmem2smem(k_stage_base + (k_chunk * 8192 + k_group * 512), ((piece_tile0_5 + tile_4 == 0) ? ((&tmap_swa_kv)) : ((&tmap_compressed_kv))), k_chunk * 128, ((k_rows[0] >= 0) ? k_rows[0] : k_fill), ((k_rows[1] >= 0) ? k_rows[1] : k_fill), ((k_rows[2] >= 0) ? k_rows[2] : k_fill), ((k_rows[3] >= 0) ? k_rows[3] : k_fill), k_full_addr + (load_k_stage) * 8);
                                        }
                                    }
                                }
                                load_k_stage += 1;
                                if (load_k_stage == 2) { load_k_stage = 0; _phase_k_empty ^= 1; }
                            }
                            mbarrier_arrive(index_empty_addr + (load_k_index_stage) * 8);
                            load_k_index_stage += 1;
                            if (load_k_index_stage == 6) { load_k_index_stage = 0; _phase_index_full_1 ^= 1; }
                        }
                    } else {
                        int v_rank = load_rank - 2;
                        int _max_9_1 = (((num_kv_tiles_5 + 1) / 2) > (1) ? ((num_kv_tiles_5 + 1) / 2) : (1));
                        #pragma unroll 1
                        for (int index_pass_2 = 0; index_pass_2 < _max_9_1; index_pass_2++) {
                            mbarrier_wait(index_full_addr + (load_v_index_stage) * 8, _phase_index_full_2);
                            int _min_10 = ((2 * index_pass_2 + 2) < (num_kv_tiles_5) ? (2 * index_pass_2 + 2) : (num_kv_tiles_5));
                            #pragma unroll 1
                            for (int tile_5 = 2 * index_pass_2; tile_5 < _min_10; tile_5++) {
                                int v_index_base = smem_index_addr + load_v_index_stage * 1024 + (unsigned int)((tile_5 & 1) * 512);
                                mbarrier_wait(v_empty_addr + (load_v_stage) * 8, _phase_v_empty);
                                if (v_rank == 0) {
                                    if (elect_sync()) {
                                        mbarrier_arrive_expect_tx(v_full_addr + (load_v_stage) * 8, 32768);
                                    }
                                }
                                int v_stage_base = smem_v_addr + load_v_stage * 32768;
                                if (elect_sync()) {
                                    int first_rows_1[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[(0) + 3]))
                                        : "r"(v_index_base));
                                    int fill_1 = ((first_rows_1[0]) > (0) ? (first_rows_1[0]) : (0));
                                    int v_fill = fill_1;
                                    #pragma unroll
                                    for (int v_group_local = 0; v_group_local < 16; v_group_local++) {
                                        int v_group = v_rank * 16 + v_group_local;
                                        int v_rows[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&v_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_rows[(0) + 3]))
                                            : "r"(v_index_base + v_group * 16));
                                        #pragma unroll
                                        for (int v_chunk = 0; v_chunk < 2; v_chunk++) {
                                            tma_gather4_gmem2smem(v_stage_base + (v_chunk * 16384 + v_group * 512), ((piece_tile0_5 + tile_5 == 0) ? ((&tmap_swa_kv)) : ((&tmap_compressed_kv))), cta_rank_3 * 256 + v_chunk * 128, ((v_rows[0] >= 0) ? v_rows[0] : v_fill), ((v_rows[1] >= 0) ? v_rows[1] : v_fill), ((v_rows[2] >= 0) ? v_rows[2] : v_fill), ((v_rows[3] >= 0) ? v_rows[3] : v_fill), v_full_addr + (load_v_stage) * 8);
                                        }
                                    }
                                }
                                load_v_stage += 1;
                                if (load_v_stage == 2) { load_v_stage = 0; _phase_v_empty ^= 1; }
                            }
                            mbarrier_arrive(index_empty_addr + (load_v_index_stage) * 8);
                            load_v_index_stage += 1;
                            if (load_v_index_stage == 6) { load_v_index_stage = 0; _phase_index_full_2 ^= 1; }
                        }
                    }
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_6) * 8, _phase_source_work_full_6);
                uint32_t clc_valid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_6)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_8)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_6)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_6)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_6) * 8), "r"(0) : "memory");
                source_work_stage_6 += 1;
                if (source_work_stage_6 == 2) { source_work_stage_6 = 0; _phase_source_work_full_6 ^= 1; }
                if (clc_valid_6 == 0) {
                    break;
                }
                source_work_x_5 = clc_ctaid_8 + (unsigned int)cta_rank;
                source_work_z_5 = clc_ctaid_2_6;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }

    // Cleanup
}

} // extern "C"
