/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 160
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 16
#define TMEM_TMEM_STATS0_OFFSET 32
#define TMEM_TMEM_STATS1_OFFSET 64
#define TMEM_TMEM_O0_HI_OFFSET 96
#define TMEM_TMEM_O0_LO_OFFSET 112
#define TMEM_TMEM_O1_HI_OFFSET 128
#define TMEM_TMEM_O1_LO_OFFSET 144
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_Q_RAW_HI_OFF 2048
#define SMEM_SMEM_Q_RAW_HI_STAGE_BYTES 4096
#define SMEM_SMEM_Q_RAW_HI_STRIDE 4096
#define SMEM_SMEM_Q_RAW_LO_OFF 6144
#define SMEM_SMEM_Q_RAW_LO_STAGE_BYTES 4096
#define SMEM_SMEM_Q_RAW_LO_STRIDE 4096
#define SMEM_SMEM_Q_HI_OFF 10240
#define SMEM_SMEM_Q_HI_STAGE_BYTES 2048
#define SMEM_SMEM_Q_HI_STRIDE 2048
#define SMEM_SMEM_Q_LO_OFF 12288
#define SMEM_SMEM_Q_LO_STAGE_BYTES 2048
#define SMEM_SMEM_Q_LO_STRIDE 2048
#define SMEM_SMEM_KV_HI_OFF 14336
#define SMEM_SMEM_KV_HI_STAGE_BYTES 16384
#define SMEM_SMEM_KV_HI_STRIDE 16384
#define SMEM_SMEM_KV_LO_OFF 79872
#define SMEM_SMEM_KV_LO_STAGE_BYTES 16384
#define SMEM_SMEM_KV_LO_STRIDE 16384
#define SMEM_SMEM_V_HI_OFF 14336
#define SMEM_SMEM_V_HI_STAGE_BYTES 16384
#define SMEM_SMEM_V_HI_STRIDE 16384
#define SMEM_SMEM_V_LO_OFF 79872
#define SMEM_SMEM_V_LO_STAGE_BYTES 16384
#define SMEM_SMEM_V_LO_STRIDE 16384
#define SMEM_SMEM_P0_OFF 145408
#define SMEM_SMEM_P0_STAGE_BYTES 2048
#define SMEM_SMEM_P0_STRIDE 2048
#define SMEM_SMEM_P1_OFF 147456
#define SMEM_SMEM_P1_STAGE_BYTES 2048
#define SMEM_SMEM_P1_STRIDE 2048
#define SMEM_SPLIT_REDUCE_FLAG_OFF 149504
#define SMEM_SPLIT_REDUCE_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_REDUCE_FLAG_STRIDE 4
#define SMEM_SPLIT_MERGED_LSE_OFF 150532
#define SMEM_SPLIT_MERGED_LSE_STAGE_BYTES 64
#define SMEM_SPLIT_MERGED_LSE_STRIDE 64
#define SMEM_SPLIT_WEIGHTS_OFF 149508
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 1024
#define SMEM_SPLIT_WEIGHTS_STRIDE 1024
#define SMEM_TOTAL 151552
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define TILE_Q 16
#define PAGE_SIZE 64
#define NUM_KV_STAGES 4
#ifndef BATCH_SIZE
#define BATCH_SIZE 1
#endif
#ifndef NUM_Q_HEADS
#define NUM_Q_HEADS 16
#endif
#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 1
#endif
#ifndef Q_LEN
#define Q_LEN 4
#endif
#ifndef CP_WORLD
#define CP_WORLD 4
#endif
#define NUM_SPLIT 8
#define RETAIN_KV_L2 0

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
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
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_fmha_dcp_spec_bf16_fp8_d256(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O_ptr, float* __restrict__ partial_LSE_ptr, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ split_completion, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ causal_seqlens_kv_global, int max_pages_per_seq, int max_local_seq_len, float softmax_scale_log2, float output_scale, int cp_rank, int num_q_heads, int num_kv_heads, int batch_size)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Qt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr0_addr = smem + 1024;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_corr1_addr = smem + 1088;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    __nv_bfloat16* smem_q_raw_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_q_raw_hi_addr = smem + 2048;
    __nv_bfloat16* smem_q_raw_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_q_raw_lo_addr = smem + 6144;
    uint8_t* smem_q_hi = reinterpret_cast<uint8_t*>(smem_raw + 10240);
    const int smem_q_hi_addr = smem + 10240;
    uint8_t* smem_q_lo = reinterpret_cast<uint8_t*>(smem_raw + 12288);
    const int smem_q_lo_addr = smem + 12288;
    uint8_t* smem_kv_hi = reinterpret_cast<uint8_t*>(smem_raw + 14336);
    const int smem_kv_hi_addr = smem + 14336;
    uint8_t* smem_kv_lo = reinterpret_cast<uint8_t*>(smem_raw + 79872);
    const int smem_kv_lo_addr = smem + 79872;
    uint8_t* smem_v_hi = reinterpret_cast<uint8_t*>(smem_raw + 14336);
    const int smem_v_hi_addr = smem + 14336;
    uint8_t* smem_v_lo = reinterpret_cast<uint8_t*>(smem_raw + 79872);
    const int smem_v_lo_addr = smem + 79872;
    uint8_t* smem_p0 = reinterpret_cast<uint8_t*>(smem_raw + 145408);
    const int smem_p0_addr = smem + 145408;
    uint8_t* smem_p1 = reinterpret_cast<uint8_t*>(smem_raw + 147456);
    const int smem_p1_addr = smem + 147456;
    int* split_reduce_flag = reinterpret_cast<int*>(smem_raw + 149504);
    const int split_reduce_flag_addr = smem + 149504;
    float* split_merged_lse = reinterpret_cast<float*>(smem_raw + 150532);
    const int split_merged_lse_addr = smem + 150532;
    float* split_weights = reinterpret_cast<float*>(smem_raw + 149508);
    const int split_weights_addr = smem + 149508;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory"); }

    // Mbarrier init (18 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_tma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_full: 1 barriers, init_count=4
            mbarrier_init(smem + 8, 4);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // p_full_0: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            // p_full_1: 1 barriers, init_count=256
            mbarrier_init(smem + 112, 256);
            // corr_scale_0: 1 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            // corr_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // corr_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 144, 4);
            // o_ready_0: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // o_ready_1: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // o_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            // o_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 160 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_tma_full_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 8)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 24)
    #define kv_empty_addr (mbar_base + 56)
    #define s_full_0_addr (mbar_base + 88)
    #define s_full_1_addr (mbar_base + 96)
    #define p_full_0_addr (mbar_base + 104)
    #define p_full_1_addr (mbar_base + 112)
    #define corr_scale_0_addr (mbar_base + 120)
    #define corr_empty_0_addr (mbar_base + 128)
    #define corr_empty_1_addr (mbar_base + 136)
    #define stats_empty_addr (mbar_base + 144)
    #define o_ready_0_addr (mbar_base + 152)
    #define o_ready_1_addr (mbar_base + 160)
    #define o_empty_0_addr (mbar_base + 168)
    #define o_empty_1_addr (mbar_base + 176)
    #define tmem_dealloc_addr (mbar_base + 184)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 16;
    const int tmem_tmem_stats0 = taddr + 32;
    const int tmem_tmem_stats1 = taddr + 64;
    const int tmem_tmem_o0_hi = taddr + 96;
    const int tmem_tmem_o0_lo = taddr + 112;
    const int tmem_tmem_o1_hi = taddr + 128;
    const int tmem_tmem_o1_lo = taddr + 144;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int is_stream1 = ((warp >= 4) ? 1 : 0);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            const int tmem_row_base = warp_in_wg * 32;
            int my_tmem_s = taddr + (unsigned int)(((is_stream1 != 0) ? 16 : 0)) + (unsigned int)(tmem_row_base << 16);
            int my_tmem_stats = taddr + (unsigned int)(((is_stream1 != 0) ? 64 : 32)) + (unsigned int)(tmem_row_base << 16);
            float* my_exch_ptr = ((is_stream1 != 0) ? smem_exch1 : smem_exch0);
            float* my_corr_ptr = ((is_stream1 != 0) ? smem_corr1 : smem_corr0);
            uint8_t* my_p_base = ((is_stream1 != 0) ? smem_p1 : smem_p0);
            unsigned int total_tiles_s = BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT;
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_s = blockIdx.x; tile_idx_s < total_tiles_s; tile_idx_s += gridDim.x) {
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                unsigned int tile_idx = tile_idx_s / (unsigned int)NUM_SPLIT;
                int split_idx = tile_idx_s % (unsigned int)NUM_SPLIT;
                int kv_head_idx = tile_idx % (unsigned int)NUM_KV_HEADS;
                int request_row = tile_idx / (unsigned int)NUM_KV_HEADS;
                int q_row_idx = request_row % Q_LEN;
                int batch_idx = request_row / Q_LEN;
                int last_global_position = causal_seqlens_kv_global[batch_idx] + q_row_idx;
                int visible_local_keys = 0;
                if (last_global_position >= cp_rank) {
                    visible_local_keys = (last_global_position - cp_rank) / CP_WORLD + 1;
                }
                int num_n_blocks_total = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_total < 1) {
                    num_n_blocks_total = 1;
                }
                int total_pairs = (num_n_blocks_total + 1) / 2;
                int base_pairs = total_pairs / NUM_SPLIT;
                int extra_pairs = total_pairs % NUM_SPLIT;
                int num_pairs = base_pairs;
                int split_start_pair = extra_pairs * (base_pairs + 1) + (split_idx - extra_pairs) * base_pairs;
                if (split_idx < extra_pairs) {
                    num_pairs = base_pairs + 1;
                    split_start_pair = split_idx * (base_pairs + 1);
                }
                float owned_row_max = -CAKE_INF;
                float owned_row_sum = 0.0f;
                #pragma unroll 1
                for (int pair = 0; pair < num_pairs; pair++) {
                    if (is_stream1 != 0) {
                        mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                        _phase_s_full_1_0 ^= 1;
                    } else {
                        mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                        _phase_s_full_0_0 ^= 1;
                    }
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], my_tmem_s);
                    int my_block = split_start_pair * 2 + num_pairs * 2 - 1 - 2 * pair - is_stream1;
                    int kv_pos = my_block * BLOCK_N + warp_in_wg * 32 + lane;
                    int key_is_visible = ((kv_pos < visible_local_keys) ? 1 : 0);
                    if (kv_pos >= visible_local_keys) {
                        #pragma unroll
                        for (int h = 0; h < 16; h++) {
                            _tmem_load_0[h] = -3.4028235e+38f;
                        }
                    }
                    #pragma unroll
                    for (int h_1 = 0; h_1 < 16; h_1++) {
                        float _warp_reduce_0 = _tmem_load_0[h_1];
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                        float partial_max_h = _warp_reduce_0;
                        if (lane == 0) {
                            my_exch_ptr[warp_in_wg * 16 + h_1] = partial_max_h;
                        }
                    }
                    if (is_stream1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    float my_tile_max = 0.0f;
                    if (lane < 16) {
                        float _max_0 = max_noftz(my_exch_ptr[lane], my_exch_ptr[16 + lane]);
                        float _max_1 = max_noftz(my_exch_ptr[32 + lane], my_exch_ptr[48 + lane]);
                        float _max_2 = max_noftz(_max_0, _max_1);
                        my_tile_max = _max_2;
                    }
                    if (is_stream1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    float _max_3 = max_noftz(owned_row_max, my_tile_max);
                    float owned_new_max = _max_3;
                    float owned_delta = softmax_scale_log2 * (owned_row_max - owned_new_max);
                    float _exp2_0 = approx_exp2(owned_delta);
                    float owned_acc_scale = ((owned_row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    float acc_scale[16];
                    #pragma unroll
                    for (int h_2 = 0; h_2 < 16; h_2++) {
                        float _shfl_0 = __shfl_sync(0xFFFFFFFF, owned_acc_scale, h_2);
                        acc_scale[h_2] = _shfl_0;
                    }
                    if (is_stream1 != 0) {
                        mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                        _phase_corr_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                        _phase_corr_empty_0_0 ^= 1;
                    }
                    tmem_st_x16_f32(my_tmem_stats, acc_scale);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_stream1 != 0) {
                        asm volatile("barrier.arrive 11, 256;" ::: "memory");
                    } else {
                        mbarrier_arrive(corr_scale_0_addr);
                    }
                    #pragma unroll
                    for (int h_3 = 0; h_3 < 16; h_3++) {
                        float _shfl_1 = __shfl_sync(0xFFFFFFFF, owned_new_max, h_3);
                        float new_max_h = _shfl_1;
                        float safe_max = ((new_max_h == -CAKE_INF) ? 0.0f : new_max_h);
                        float _exp2_1 = approx_exp2(_tmem_load_0[h_3] * softmax_scale_log2 - safe_max * softmax_scale_log2 + 8.8073549f);
                        float exp_value = ((key_is_visible != 0) ? _exp2_1 : 0.0f);
                        float _warp_reduce_1 = exp_value;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                        float warp_sum_h = _warp_reduce_1;
                        if (lane == h_3) {
                            owned_row_sum = owned_row_sum * owned_acc_scale + warp_sum_h;
                        }
                        {
                            uint16_t _fp8_pair_0;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_0) : "f"(0.0f), "f"(exp_value));
                            uint32_t _byte_0 = (uint32_t)(_fp8_pair_0 & 0xFF);
                            const void* _ptr_0 = reinterpret_cast<const void*>((reinterpret_cast<uint8_t*>(my_p_base) + (h_3 * 128 + wg_tid ^ (h_3 * 128 + wg_tid >> 7 & 7) << 4)));
                            uint64_t _addr64_0;
                            asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_addr64_0) : "l"(_ptr_0));
                            uint32_t _addr_0;
                            asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_addr_0) : "l"(_addr64_0));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_0), "r"(_byte_0) : "memory");
                        }
                    }
                    owned_row_max = owned_new_max;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_stream1 != 0) {
                        mbarrier_arrive(p_full_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
                    }
                }
                if (is_stream1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (lane < 16) {
                    my_exch_ptr[warp_in_wg * 16 + lane] = owned_row_sum;
                }
                if (is_stream1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                float my_total_sum = 0.0f;
                if (lane < 16) {
                    my_total_sum = my_exch_ptr[lane] + my_exch_ptr[16 + lane] + my_exch_ptr[32 + lane] + my_exch_ptr[48 + lane];
                }
                if (is_stream1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (warp_in_wg == 0 && lane < 16) {
                    my_corr_ptr[lane] = my_total_sum;
                    my_exch_ptr[lane] = owned_row_max;
                }
                if (is_stream1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                    _phase_corr_empty_1_0 ^= 1;
                    asm volatile("barrier.arrive 11, 256;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                    _phase_corr_empty_0_0 ^= 1;
                    mbarrier_arrive(corr_scale_0_addr);
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int warp_in_wg_c = warp % 4;
            const int wg_tid_c = warp_in_wg_c * 32 + lane;
            const int corr_row = warp_in_wg_c * 32 << 16;
            const int d_idx = warp_in_wg_c * 32 + lane;
            const int group_ratio_rt = NUM_Q_HEADS / NUM_KV_HEADS;
            unsigned int total_tiles_c = BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT;
            unsigned int _phase_q_tma_full_0 = 0;
            unsigned int _phase_corr_scale_0_0 = 0;
            unsigned int _phase_o_ready_0_0 = 0;
            unsigned int _phase_o_ready_1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_c = blockIdx.x; tile_idx_c < total_tiles_c; tile_idx_c += gridDim.x) {
                unsigned int tile_idx_1 = tile_idx_c / (unsigned int)NUM_SPLIT;
                int split_idx_1 = tile_idx_c % (unsigned int)NUM_SPLIT;
                int kv_head_idx_1 = tile_idx_1 % (unsigned int)NUM_KV_HEADS;
                int request_row_1 = tile_idx_1 / (unsigned int)NUM_KV_HEADS;
                int q_row_idx_1 = request_row_1 % Q_LEN;
                int batch_idx_1 = request_row_1 / Q_LEN;
                int last_global_position_1 = causal_seqlens_kv_global[batch_idx_1] + q_row_idx_1;
                int visible_local_keys_1 = 0;
                if (last_global_position_1 >= cp_rank) {
                    visible_local_keys_1 = (last_global_position_1 - cp_rank) / CP_WORLD + 1;
                }
                int num_n_blocks_total_1 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_total_1 < 1) {
                    num_n_blocks_total_1 = 1;
                }
                int total_pairs_1 = (num_n_blocks_total_1 + 1) / 2;
                int base_pairs_1 = total_pairs_1 / NUM_SPLIT;
                int extra_pairs_1 = total_pairs_1 % NUM_SPLIT;
                int num_pairs_1 = base_pairs_1;
                int split_start_pair_1 = extra_pairs_1 * (base_pairs_1 + 1) + (split_idx_1 - extra_pairs_1) * base_pairs_1;
                if (split_idx_1 < extra_pairs_1) {
                    num_pairs_1 = base_pairs_1 + 1;
                    split_start_pair_1 = split_idx_1 * (base_pairs_1 + 1);
                }
                mbarrier_wait(q_tma_full_addr, _phase_q_tma_full_0);
                _phase_q_tma_full_0 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (wg_tid_c < HEAD_DIM_HALF / 2) {
                    int q_pair_col = wg_tid_c * 2;
                    int q_group = q_pair_col / 64;
                    int q_group_col = q_pair_col % 64;
                    #pragma unroll
                    for (int q_head = 0; q_head < TILE_Q; q_head++) {
                        int q_raw_row = q_group * TILE_Q + q_head;
                        unsigned int q_hi_packed[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_hi_packed[0])) : "r"((smem_q_raw_hi_addr + (unsigned int)(q_raw_row * 128 + q_group_col * 2 ^ (q_raw_row * 128 + q_group_col * 2 >> 7 & 7) << 4))));
                        float q_hi_packed_f32[2];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&q_hi_packed_f32[_pair * 2])[0]), "=f"((&q_hi_packed_f32[_pair * 2])[1])
                                : "r"(q_hi_packed[_pair]));
                        }
                        #pragma unroll
                        for (int pair_elem = 0; pair_elem < 2; pair_elem++) {
                            {
                                uint16_t _fp8_pair_0;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_0) : "f"(0.0f), "f"(q_hi_packed_f32[pair_elem]));
                                uint32_t _byte_0 = (uint32_t)(_fp8_pair_0 & 0xFF);
                                uint32_t _addr_0 = static_cast<uint32_t>((smem_q_hi_addr + (unsigned int)(q_head * 128 + (q_pair_col + pair_elem) ^ (q_head * 128 + (q_pair_col + pair_elem) >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_0), "r"(_byte_0) : "memory");
                            }
                        }
                    }
                } else {
                    int q_pair_col_1 = (wg_tid_c - HEAD_DIM_HALF / 2) * 2;
                    int q_group_1 = q_pair_col_1 / 64;
                    int q_group_col_1 = q_pair_col_1 % 64;
                    #pragma unroll
                    for (int q_head_1 = 0; q_head_1 < TILE_Q; q_head_1++) {
                        int q_raw_row_1 = q_group_1 * TILE_Q + q_head_1;
                        unsigned int q_lo_packed[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_lo_packed[0])) : "r"((smem_q_raw_lo_addr + (unsigned int)(q_raw_row_1 * 128 + q_group_col_1 * 2 ^ (q_raw_row_1 * 128 + q_group_col_1 * 2 >> 7 & 7) << 4))));
                        float q_lo_packed_f32[2];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&q_lo_packed_f32[_pair * 2])[0]), "=f"((&q_lo_packed_f32[_pair * 2])[1])
                                : "r"(q_lo_packed[_pair]));
                        }
                        #pragma unroll
                        for (int pair_elem_1 = 0; pair_elem_1 < 2; pair_elem_1++) {
                            {
                                uint16_t _fp8_pair_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                    : "=h"(_fp8_pair_1) : "f"(0.0f), "f"(q_lo_packed_f32[pair_elem_1]));
                                uint32_t _byte_1 = (uint32_t)(_fp8_pair_1 & 0xFF);
                                uint32_t _addr_1 = static_cast<uint32_t>((smem_q_lo_addr + (unsigned int)(q_head_1 * 128 + (q_pair_col_1 + pair_elem_1) ^ (q_head_1 * 128 + (q_pair_col_1 + pair_elem_1) >> 7 & 7) << 4)));
                                asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_1), "r"(_byte_1) : "memory");
                            }
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(q_full_addr);
                }
                if (num_pairs_1 > 0) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    asm volatile("barrier.sync 11, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < num_pairs_1; pair_1++) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + 32 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                    _phase_o_ready_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + 96 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_4 = 0; h_4 < 16; h_4++) {
                        _tmem_load_2[h_4] = _tmem_load_2[h_4] * _tmem_load_1[h_4];
                    }
                    tmem_st_x16_f32(taddr + 96 + (unsigned int)corr_row, _tmem_load_2);
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 112 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_5 = 0; h_5 < 16; h_5++) {
                        _tmem_load_3[h_5] = _tmem_load_3[h_5] * _tmem_load_1[h_5];
                    }
                    tmem_st_x16_f32(taddr + 112 + (unsigned int)corr_row, _tmem_load_3);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(o_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    asm volatile("barrier.sync 11, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 64 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                    _phase_o_ready_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 128 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_6 = 0; h_6 < 16; h_6++) {
                        _tmem_load_5[h_6] = _tmem_load_5[h_6] * _tmem_load_4[h_6];
                    }
                    tmem_st_x16_f32(taddr + 128 + (unsigned int)corr_row, _tmem_load_5);
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 144 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_7 = 0; h_7 < 16; h_7++) {
                        _tmem_load_6[h_7] = _tmem_load_6[h_7] * _tmem_load_4[h_7];
                    }
                    tmem_st_x16_f32(taddr + 144 + (unsigned int)corr_row, _tmem_load_6);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(o_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                _phase_corr_scale_0_0 ^= 1;
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float owner_scale0 = 0.0f;
                float owner_scale1 = 0.0f;
                float owner_inv_sum = 0.0f;
                float owner_lse = -CAKE_INF;
                int owner_valid = 0;
                if (group_ratio_rt > lane) {
                    float owner_m0 = smem_exch0[lane];
                    float owner_m1 = smem_exch1[lane];
                    float owner_s0 = smem_corr0[lane];
                    float owner_s1 = smem_corr1[lane];
                    float _max_4 = max_noftz(owner_m0, owner_m1);
                    float owner_max = _max_4;
                    float _exp2_2 = approx_exp2(softmax_scale_log2 * (owner_m0 - owner_max));
                    owner_scale0 = ((owner_m0 == -CAKE_INF) ? 0.0f : _exp2_2);
                    float _exp2_3 = approx_exp2(softmax_scale_log2 * (owner_m1 - owner_max));
                    owner_scale1 = ((owner_m1 == -CAKE_INF) ? 0.0f : _exp2_3);
                    float owner_sum = owner_s0 * owner_scale0 + owner_s1 * owner_scale1;
                    if (owner_sum > 0.0f && owner_sum == owner_sum) {
                        float _rcp_0 = approx_rcp(owner_sum);
                        owner_inv_sum = _rcp_0;
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(owner_sum));
                        owner_lse = owner_max * softmax_scale_log2 + _log2_0 - 8.8073549f;
                        owner_valid = 1;
                    }
                }
                if (warp_in_wg_c == 0 && group_ratio_rt > lane) {
                    int lse_q_head = kv_head_idx_1 * group_ratio_rt + lane;
                    {
                        int partial_lse_idx = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + lse_q_head) * NUM_SPLIT + split_idx_1;
                        *(reinterpret_cast<float*>(partial_LSE_ptr + partial_lse_idx) + (0)) = owner_lse;
                    }
                }
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                _phase_o_ready_0_0 ^= 1;
                mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                _phase_o_ready_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int head_base = 0; head_base < 16; head_base += 8) {
                    float _tmem_load_7[8];
                    tmem_ld_x8(&_tmem_load_7[0], taddr + 96 + (unsigned int)head_base + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_8[8];
                    tmem_ld_x8(&_tmem_load_8[0], taddr + 128 + (unsigned int)head_base + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int head_lane = 0; head_lane < 8; head_lane++) {
                        const int h_8 = head_base + head_lane;
                        float _shfl_2 = __shfl_sync(0xFFFFFFFF, owner_scale0, h_8);
                        float head_scale0 = _shfl_2;
                        float _shfl_3 = __shfl_sync(0xFFFFFFFF, owner_scale1, h_8);
                        float head_scale1 = _shfl_3;
                        float _shfl_4 = __shfl_sync(0xFFFFFFFF, owner_inv_sum, h_8);
                        float head_inv_sum = _shfl_4;
                        float _shfl_5 = __shfl_sync(0xFFFFFFFF, owner_valid, h_8);
                        int head_valid = _shfl_5;
                        float final_o_hi = 0.0f;
                        if (head_valid != 0) {
                            final_o_hi = (_tmem_load_7[head_lane] * head_scale0 + _tmem_load_8[head_lane] * head_scale1) * head_inv_sum * output_scale;
                        }
                        if (h_8 < group_ratio_rt) {
                            int q_head_2 = kv_head_idx_1 * group_ratio_rt + h_8;
                            {
                                int partial_o_idx = (((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head_2) * NUM_SPLIT + split_idx_1) * HEAD_DIM + d_idx;
                                *(reinterpret_cast<__nv_bfloat16*>(partial_O_ptr + partial_o_idx) + (0)) = __float2bfloat16_rn(final_o_hi);
                            }
                        }
                    }
                    float _tmem_load_9[8];
                    tmem_ld_x8(&_tmem_load_9[0], taddr + 112 + (unsigned int)head_base + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_10[8];
                    tmem_ld_x8(&_tmem_load_10[0], taddr + 144 + (unsigned int)head_base + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int head_lane_1 = 0; head_lane_1 < 8; head_lane_1++) {
                        const int h_9 = head_base + head_lane_1;
                        float _shfl_6 = __shfl_sync(0xFFFFFFFF, owner_scale0, h_9);
                        float head_scale0_1 = _shfl_6;
                        float _shfl_7 = __shfl_sync(0xFFFFFFFF, owner_scale1, h_9);
                        float head_scale1_1 = _shfl_7;
                        float _shfl_8 = __shfl_sync(0xFFFFFFFF, owner_inv_sum, h_9);
                        float head_inv_sum_1 = _shfl_8;
                        float _shfl_9 = __shfl_sync(0xFFFFFFFF, owner_valid, h_9);
                        int head_valid_1 = _shfl_9;
                        float final_o_lo = 0.0f;
                        if (head_valid_1 != 0) {
                            final_o_lo = (_tmem_load_9[head_lane_1] * head_scale0_1 + _tmem_load_10[head_lane_1] * head_scale1_1) * head_inv_sum_1 * output_scale;
                        }
                        if (h_9 < group_ratio_rt) {
                            int q_head_3 = kv_head_idx_1 * group_ratio_rt + h_9;
                            {
                                int partial_o_idx_1 = (((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head_3) * NUM_SPLIT + split_idx_1) * HEAD_DIM + HEAD_DIM_HALF + d_idx;
                                *(reinterpret_cast<__nv_bfloat16*>(partial_O_ptr + partial_o_idx_1) + (0)) = __float2bfloat16_rn(final_o_lo);
                            }
                        }
                    }
                }
                mbarrier_arrive(o_empty_0_addr);
                mbarrier_arrive(o_empty_1_addr);
                if (elect_sync()) {
                    mbarrier_arrive(stats_empty_addr);
                }
                {
                    int base_tile_idx = (batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_KV_HEADS + kv_head_idx_1;
                    __threadfence();
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                        int _atomic_old_0 = atomicAdd(&split_completion[base_tile_idx], 1);
                        int old_count = _atomic_old_0;
                        split_reduce_flag[0] = ((old_count + 1 == NUM_SPLIT) ? 1 : 0);
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    if (split_reduce_flag[0] != 0) {
                        __threadfence();
                        int reduce_head_in_group = wg_tid_c / 8;
                        int reduce_lane = wg_tid_c % 8;
                        int reduce_q_head = kv_head_idx_1 * group_ratio_rt + reduce_head_in_group;
                        int reduce_stat_base = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + reduce_q_head) * NUM_SPLIT;
                        int split0 = reduce_lane;
                        int split1 = reduce_lane + 8;
                        float lse0 = -CAKE_INF;
                        float lse1 = -CAKE_INF;
                        if (split0 < NUM_SPLIT) {
                            lse0 = partial_LSE_ptr[reduce_stat_base + split0];
                        }
                        if (split1 < NUM_SPLIT) {
                            lse1 = partial_LSE_ptr[reduce_stat_base + split1];
                        }
                        float _max_5 = max_noftz(lse0, lse1);
                        float lane_max = _max_5;
                        int subgroup_lane_base = lane / 8 * 8;
                        float merged_max = -CAKE_INF;
                        #pragma unroll
                        for (int source_lane = 0; source_lane < 8; source_lane++) {
                            float _shfl_10 = __shfl_sync(0xFFFFFFFF, lane_max, subgroup_lane_base + source_lane);
                            float source_max = _shfl_10;
                            float _max_6 = max_noftz(merged_max, source_max);
                            merged_max = _max_6;
                        }
                        float weight0 = 0.0f;
                        float weight1 = 0.0f;
                        if (lse0 != -CAKE_INF) {
                            float _exp2_4 = approx_exp2(lse0 - merged_max);
                            weight0 = _exp2_4;
                        }
                        if (lse1 != -CAKE_INF) {
                            float _exp2_5 = approx_exp2(lse1 - merged_max);
                            weight1 = _exp2_5;
                        }
                        float lane_weight_sum = weight0 + weight1;
                        float weight_sum = 0.0f;
                        #pragma unroll
                        for (int source_lane_1 = 0; source_lane_1 < 8; source_lane_1++) {
                            float _shfl_11 = __shfl_sync(0xFFFFFFFF, lane_weight_sum, subgroup_lane_base + source_lane_1);
                            weight_sum = weight_sum + _shfl_11;
                        }
                        float _rcp_1 = approx_rcp(weight_sum);
                        float inv_weight_sum = ((weight_sum > 0.0f) ? _rcp_1 : 0.0f);
                        if (split0 < NUM_SPLIT) {
                            split_weights[split0 * TILE_Q + reduce_head_in_group] = weight0 * inv_weight_sum;
                        }
                        if (split1 < NUM_SPLIT) {
                            split_weights[split1 * TILE_Q + reduce_head_in_group] = weight1 * inv_weight_sum;
                        }
                        if (reduce_lane == 0) {
                            float merged_lse = -CAKE_INF;
                            if (weight_sum > 0.0f) {
                                float _log2_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(weight_sum));
                                merged_lse = merged_max + _log2_1;
                            }
                            int final_lse_idx = (batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + reduce_q_head;
                            *(reinterpret_cast<float*>(LSE_ptr + final_lse_idx) + (0)) = merged_lse;
                        }
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        int merge_head_in_group = wg_tid_c / 8;
                        int merge_d_base = wg_tid_c % 8 * 32;
                        if (merge_head_in_group < group_ratio_rt) {
                            int reduce_q_head_0 = kv_head_idx_1 * group_ratio_rt + merge_head_in_group;
                            #pragma unroll
                            for (int vec_chunk = 0; vec_chunk < 4; vec_chunk++) {
                                int elem_base = merge_d_base + vec_chunk * 8;
                                int partial_o_base = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + reduce_q_head_0) * NUM_SPLIT * HEAD_DIM + elem_base;
                                int final_o_idx = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + reduce_q_head_0) * HEAD_DIM + elem_base;
                                float _vec_load_0[8];
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O_ptr + partial_o_base + 0);
                                    uint4 _vld_2[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_2[_blk] = _vptr_2[_blk];
                                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                : "r"(_vpairs_2[_pair]));
                                        }
                                    }
                                }
                                float weight0_0 = split_weights[merge_head_in_group];
                                #pragma unroll
                                for (int elem = 0; elem < 8; elem++) {
                                    _vec_load_0[elem] = _vec_load_0[elem] * weight0_0;
                                }
                                #pragma unroll 2
                                for (int reduce_split = 1; reduce_split < NUM_SPLIT; reduce_split++) {
                                    float _vec_load_1[8];
                                    {
                                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_O_ptr + (partial_o_base + reduce_split * HEAD_DIM) + 0);
                                        uint4 _vld_3[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_3[_blk] = _vptr_3[_blk];
                                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_3[_pair]));
                                            }
                                        }
                                    }
                                    float reduce_weight = split_weights[reduce_split * TILE_Q + merge_head_in_group];
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                        float _fma_0 = __fmaf_rn(_vec_load_1[elem_1], reduce_weight, _vec_load_0[elem_1]);
                                        _vec_load_0[elem_1] = _fma_0;
                                    }
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + final_o_idx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                        if (wg_tid_c == 0) {
                            *(reinterpret_cast<int*>(split_completion + base_tile_idx) + (0)) = 0;
                        }
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int total_tiles_m = BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_o_empty_0_0 = 1;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_o_empty_1_0 = 1;
            unsigned int _phase_p_full_1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_m = blockIdx.x; tile_idx_m < total_tiles_m; tile_idx_m += gridDim.x) {
                unsigned int tile_idx_2 = tile_idx_m / (unsigned int)NUM_SPLIT;
                int split_idx_2 = tile_idx_m % (unsigned int)NUM_SPLIT;
                int kv_head_idx_2 = tile_idx_2 % (unsigned int)NUM_KV_HEADS;
                int request_row_2 = tile_idx_2 / (unsigned int)NUM_KV_HEADS;
                int q_row_idx_2 = request_row_2 % Q_LEN;
                int batch_idx_2 = request_row_2 / Q_LEN;
                int last_global_position_2 = causal_seqlens_kv_global[batch_idx_2] + q_row_idx_2;
                int visible_local_keys_2 = 0;
                if (last_global_position_2 >= cp_rank) {
                    visible_local_keys_2 = (last_global_position_2 - cp_rank) / CP_WORLD + 1;
                }
                int num_n_blocks_total_2 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_total_2 < 1) {
                    num_n_blocks_total_2 = 1;
                }
                int total_pairs_2 = (num_n_blocks_total_2 + 1) / 2;
                int base_pairs_2 = total_pairs_2 / NUM_SPLIT;
                int extra_pairs_2 = total_pairs_2 % NUM_SPLIT;
                int num_pairs_2 = base_pairs_2;
                int split_start_pair_2 = extra_pairs_2 * (base_pairs_2 + 1) + (split_idx_2 - extra_pairs_2) * base_pairs_2;
                if (split_idx_2 < extra_pairs_2) {
                    num_pairs_2 = base_pairs_2 + 1;
                    split_start_pair_2 = split_idx_2 * (base_pairs_2 + 1);
                }
                int inst0_stage = 0;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (0) * 1024);
                int _mma_b_lo_0 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_s0), "r"(0));
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (0) * 1024);
                int _mma_b_lo_1 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_tmem_s0), "r"(1));
                elect_commit(s_full_0_addr);
                elect_commit(kv_empty_addr);
                mbarrier_wait(kv_full_addr + 8, 0);
                int _mma_a_lo_2 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (1) * 1024);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_0), "r"(tmem_tmem_s1), "r"(0));
                int _mma_a_lo_3 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (1) * 1024);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_1), "r"(tmem_tmem_s1), "r"(1));
                elect_commit(s_full_1_addr);
                elect_commit(kv_empty_addr + 8);
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < num_pairs_2 - 1; pair_2++) {
                    int s0 = inst0_stage;
                    int s1 = (inst0_stage + 1) % NUM_KV_STAGES;
                    int s0_next = (inst0_stage + 2) % NUM_KV_STAGES;
                    int s1_next = (inst0_stage + 3) % NUM_KV_STAGES;
                    mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                    _phase_o_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 1024);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o0_hi), "r"(((first_pv0) ? 0 : 1)));
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_4), "r"(tmem_tmem_o0_lo), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (s0) * 8, o_ready_0_addr);
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_6 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (s0_next) * 1024);
                    int _mma_b_lo_6 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_s0), "r"(0));
                    int _mma_a_lo_7 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (s0_next) * 1024);
                    int _mma_b_lo_7 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_s0), "r"(1));
                    elect_commit(s_full_0_addr);
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                    _phase_o_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_8 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 1024);
                    int _mma_b_lo_8 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_tmem_o1_hi), "r"(((first_pv1) ? 0 : 1)));
                    int _mma_a_lo_9 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_8), "r"(tmem_tmem_o1_lo), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (s1) * 8, o_ready_1_addr);
                    mbarrier_wait(kv_full_addr + (s1_next) * 8, 0);
                    int _mma_a_lo_10 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (s1_next) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_6), "r"(tmem_tmem_s1), "r"(0));
                    int _mma_a_lo_11 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (s1_next) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_7), "r"(tmem_tmem_s1), "r"(1));
                    elect_commit(s_full_1_addr);
                    elect_commit(kv_empty_addr + (s1_next) * 8);
                    inst0_stage = s0_next;
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                elect_commit(q_empty_addr);
                int s0_last = inst0_stage;
                int s1_last = (inst0_stage + 1) % NUM_KV_STAGES;
                mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                _phase_o_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                _phase_p_full_0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_12 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 1024);
                int _mma_b_lo_12 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"(tmem_tmem_o0_hi), "r"(((first_pv0) ? 0 : 1)));
                int _mma_a_lo_13 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 1024);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_12), "r"(tmem_tmem_o0_lo), "r"(((first_pv0) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (s0_last) * 8, o_ready_0_addr);
                mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                _phase_o_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_14 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 1024);
                int _mma_b_lo_14 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"(tmem_tmem_o1_hi), "r"(((first_pv1) ? 0 : 1)));
                int _mma_a_lo_15 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 1024);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134512656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_14), "r"(tmem_tmem_o1_lo), "r"(((first_pv1) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (s1_last) * 8, o_ready_1_addr);
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 13) {
        // idle — no tasks assigned
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            unsigned int total_tiles_l = BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_l = blockIdx.x; tile_idx_l < total_tiles_l; tile_idx_l += gridDim.x) {
                unsigned int tile_idx_3 = tile_idx_l / (unsigned int)NUM_SPLIT;
                int split_idx_3 = tile_idx_l % (unsigned int)NUM_SPLIT;
                int kv_head_idx_3 = tile_idx_3 % (unsigned int)NUM_KV_HEADS;
                int request_row_3 = tile_idx_3 / (unsigned int)NUM_KV_HEADS;
                int q_row_idx_3 = request_row_3 % Q_LEN;
                int batch_idx_3 = request_row_3 / Q_LEN;
                int last_global_position_3 = causal_seqlens_kv_global[batch_idx_3] + q_row_idx_3;
                int visible_local_keys_3 = 0;
                if (last_global_position_3 >= cp_rank) {
                    visible_local_keys_3 = (last_global_position_3 - cp_rank) / CP_WORLD + 1;
                }
                int seqlen_kv = seq_lens_kv[batch_idx_3];
                int num_n_blocks_total_3 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_total_3 < 1) {
                    num_n_blocks_total_3 = 1;
                }
                int total_pairs_3 = (num_n_blocks_total_3 + 1) / 2;
                int base_pairs_3 = total_pairs_3 / NUM_SPLIT;
                int extra_pairs_3 = total_pairs_3 % NUM_SPLIT;
                int num_pairs_3 = base_pairs_3;
                int split_start_pair_3 = extra_pairs_3 * (base_pairs_3 + 1) + (split_idx_3 - extra_pairs_3) * base_pairs_3;
                if (split_idx_3 < extra_pairs_3) {
                    num_pairs_3 = base_pairs_3 + 1;
                    split_start_pair_3 = split_idx_3 * (base_pairs_3 + 1);
                }
                int pages_per_seq_v = (seqlen_kv + PAGE_SIZE - 1) / PAGE_SIZE;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    const int group_ratio_l = NUM_Q_HEADS / NUM_KV_HEADS;
                    int off_qt = (batch_idx_3 * Q_LEN + q_row_idx_3) * NUM_Q_HEADS + kv_head_idx_3 * group_ratio_l;
                    mbarrier_arrive_expect_tx(q_tma_full_addr, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_q_raw_hi_addr, Qt, 0, off_qt, 0, q_tma_full_addr);
                    tma_3d_gmem2smem(smem_q_raw_lo_addr, Qt, 0, off_qt, 2, q_tma_full_addr);
                    int pt_base = batch_idx_3 * max_pages_per_seq;
                    int kv_stage = 0;
                    int kv_phase = 1;
                    int prefill = ((num_pairs_3 * 2 < 2) ? num_pairs_3 * 2 : 2);
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        int n_block = split_start_pair_3 * 2 + num_pairs_3 * 2 - 1 - ni;
                        int pg_base = n_block * 2;
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                        int hdst = smem_kv_hi_addr + (unsigned int)(kv_stage * 16384);
                        int ldst = smem_kv_lo_addr + (unsigned int)(kv_stage * 16384);
                        int pg_k[2];
                        {
                            pg_k[0] = *reinterpret_cast<const int*>(page_table + pt_base + pg_base);
                        }
                        {
                            pg_k[1] = *reinterpret_cast<const int*>(page_table + pt_base + pg_base + 1);
                        }
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 2; pg_i++) {
                            int toff = pg_i * PAGE_SIZE * HEAD_DIM_HALF;
                            {
                                tma_5d_gmem2smem(hdst + toff, K, 0, 0, 0, kv_head_idx_3, pg_k[pg_i], kv_full_addr + (kv_stage) * 8);
                                tma_5d_gmem2smem(ldst + toff, K, 0, 0, 1, kv_head_idx_3, pg_k[pg_i], kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < num_pairs_3 * 2; ni_1++) {
                        int stage = ni_1 % NUM_KV_STAGES;
                        int n_block_1 = split_start_pair_3 * 2 + num_pairs_3 * 2 - 1 - ni_1;
                        int vpg_base = n_block_1 * 2;
                        mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
                        int vhdst = smem_kv_hi_addr + (unsigned int)(stage * 16384);
                        int vldst = smem_kv_lo_addr + (unsigned int)(stage * 16384);
                        int pg_v[2];
                        {
                            pg_v[0] = *reinterpret_cast<const int*>(page_table + pt_base + vpg_base);
                        }
                        {
                            pg_v[1] = *reinterpret_cast<const int*>(page_table + pt_base + vpg_base + 1);
                        }
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 2; pg_i_1++) {
                            int vtoff = pg_i_1 * PAGE_SIZE * HEAD_DIM_HALF;
                            {
                                tma_5d_gmem2smem(vhdst + vtoff, V, 0, 0, 0, kv_head_idx_3, pg_v[pg_i_1], kv_full_addr + (stage) * 8);
                                tma_5d_gmem2smem(vldst + vtoff, V, 0, 0, 1, kv_head_idx_3, pg_v[pg_i_1], kv_full_addr + (stage) * 8);
                            }
                        }
                        int next_ni = ni_1 + 2;
                        if (next_ni < num_pairs_3 * 2) {
                            int k_stage = next_ni % NUM_KV_STAGES;
                            int next_n = split_start_pair_3 * 2 + num_pairs_3 * 2 - 1 - next_ni;
                            int npg_base = next_n * 2;
                            mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 32768);
                            int khdst = smem_kv_hi_addr + (unsigned int)(k_stage * 16384);
                            int kldst = smem_kv_lo_addr + (unsigned int)(k_stage * 16384);
                            int pg_nk[2];
                            {
                                pg_nk[0] = *reinterpret_cast<const int*>(page_table + pt_base + npg_base);
                            }
                            {
                                pg_nk[1] = *reinterpret_cast<const int*>(page_table + pt_base + npg_base + 1);
                            }
                            #pragma unroll
                            for (int pg_i_2 = 0; pg_i_2 < 2; pg_i_2++) {
                                int ntoff = pg_i_2 * PAGE_SIZE * HEAD_DIM_HALF;
                                {
                                    tma_5d_gmem2smem(khdst + ntoff, K, 0, 0, 0, kv_head_idx_3, pg_nk[pg_i_2], kv_full_addr + (k_stage) * 8);
                                    tma_5d_gmem2smem(kldst + ntoff, K, 0, 0, 1, kv_head_idx_3, pg_nk[pg_i_2], kv_full_addr + (k_stage) * 8);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
