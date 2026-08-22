/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeFmhaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeFmhaTensorMapPack { CakeFmhaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
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
#define SMEM_SMEM_EXCH0_U32_OFF 1152
#define SMEM_SMEM_EXCH0_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_U32_STRIDE 256
#define SMEM_SMEM_EXCH1_U32_OFF 1408
#define SMEM_SMEM_EXCH1_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_U32_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 2048
#define SMEM_SMEM_QT_STRIDE 2048
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 2048
#define SMEM_SMEM_P0_STRIDE 2048
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 2048
#define SMEM_SMEM_P1_STRIDE 2048
#define SMEM_TOTAL 145408
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 16
#define NUM_KV_STAGES 4
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif
#ifndef NUM_Q_HEADS
#define NUM_Q_HEADS 8
#endif
#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 2
#endif
#ifndef Q_LEN
#define Q_LEN 1
#endif
#define HAS_SINK 0
#define HAS_WINDOW 0
#define USE_SCALE_PTR 1
#define RETAIN_KV_L2 1
#define FOUR_LANE_MAX 0

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
        :: "r"(mbar_addr), "r"(count));
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

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
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


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
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

__global__ __launch_bounds__(512) void
kernel_cake_fmha_decode_native_bf16(CakeFmhaTensorMap const* Qt, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ page_table, int* __restrict__ causal_seqlens_kv_global, float* __restrict__ scale_log2_ptr, float* __restrict__ sinks_ptr, int max_pages_per_seq, int max_local_seq_len, float softmax_scale_log2, int window_left, int num_q_heads, int num_kv_heads, int batch_size)
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
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_p0_addr = smem + 137216;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (17 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // kv_full: 4 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // p_full_0: 1 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            // p_full_1: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            // corr_scale: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
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
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 48)
    #define s_full_0_addr (mbar_base + 80)
    #define s_full_1_addr (mbar_base + 88)
    #define p_full_0_addr (mbar_base + 96)
    #define p_full_1_addr (mbar_base + 104)
    #define corr_scale_addr (mbar_base + 112)
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
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0 = taddr + 80;
    const int tmem_tmem_o1 = taddr + 88;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 8 : 0));
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
            unsigned int* my_exch_u32_ptr = ((is_wg1 != 0) ? smem_exch1_u32 : smem_exch0_u32);
            float* my_corr_ptr = ((is_wg1 != 0) ? smem_corr1 : smem_corr0);
            __nv_bfloat16* my_p_base = ((is_wg1 != 0) ? smem_p1 : smem_p0);
            float smx_scale = softmax_scale_log2;
            {
                smx_scale = scale_log2_ptr[0];
            }
            float sv[8];
            float sv_lo[4];
            float sv_hi[4];
            unsigned int total_tiles_s = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_s = blockIdx.x; tile_idx_s < total_tiles_s; tile_idx_s += gridDim.x) {
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                const int tiles_per_batch = Q_LEN * NUM_KV_HEADS;
                int batch_idx = tile_idx_s / (unsigned int)tiles_per_batch;
                int tile_in_batch = tile_idx_s % (unsigned int)tiles_per_batch;
                int q_row_idx = tile_in_batch / NUM_KV_HEADS;
                int kv_head_idx = tile_in_batch % NUM_KV_HEADS;
                int global_q_pos = causal_seqlens_kv_global[batch_idx] + q_row_idx;
                int visible_local_keys = global_q_pos + 1;
                int win_start = 0;
                int num_n_blocks_total = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks = num_n_blocks_total + num_n_blocks_total % 2;
                if (cta_n_blocks < 4) {
                    cta_n_blocks = 4;
                }
                int split_start_block = 0;
                float row_max_pair[2];
                float row_sum_pair[2];
                row_max_pair[0] = -LOOM_INF;
                row_max_pair[1] = -LOOM_INF;
                row_sum_pair[0] = 0.0f;
                row_sum_pair[1] = 0.0f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    my_exch_u32_ptr[wg_tid] = _amf_enc_0;
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                    _phase_corr_empty_1_0 ^= 1;
                } else {
                    mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                    _phase_corr_empty_0_0 ^= 1;
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                    _phase_s_full_1_0 ^= 1;
                } else {
                    mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                    _phase_s_full_0_0 ^= 1;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                    : "r"(my_tmem_s_base)
                    : "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                    : "r"(my_tmem_s_base + 1048576)
                    : "memory");
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    sv[c] = sv_lo[c];
                    sv[c + 4] = sv_hi[c];
                }
                #pragma unroll 1
                for (int pair = 0; pair < cta_n_blocks / 2; pair++) {
                    int my_block = split_start_block + cta_n_blocks - 1 - 2 * pair - is_wg1;
                    int ldtm_row_base = warp_in_wg * 32 + lane / 4;
                    int kv_pos0 = my_block * BLOCK_N + ldtm_row_base;
                    int kv_pos1 = kv_pos0 + 8;
                    int kv_pos2 = kv_pos0 + 16;
                    int kv_pos3 = kv_pos0 + 24;
                    if (kv_pos0 >= visible_local_keys) {
                        sv[0] = -3.4028235e+38f;
                        sv[1] = -3.4028235e+38f;
                    }
                    if (kv_pos1 >= visible_local_keys) {
                        sv[2] = -3.4028235e+38f;
                        sv[3] = -3.4028235e+38f;
                    }
                    if (kv_pos2 >= visible_local_keys) {
                        sv[4] = -3.4028235e+38f;
                        sv[5] = -3.4028235e+38f;
                    }
                    if (kv_pos3 >= visible_local_keys) {
                        sv[6] = -3.4028235e+38f;
                        sv[7] = -3.4028235e+38f;
                    }
                    float pair_max[2];
                    float _max_0 = max_noftz(sv[0], sv[2]);
                    float _max_1 = max_noftz(sv[4], sv[6]);
                    float _max_2 = max_noftz(_max_0, _max_1);
                    pair_max[0] = _max_2;
                    float _max_3 = max_noftz(sv[1], sv[3]);
                    float _max_4 = max_noftz(sv[5], sv[7]);
                    float _max_5 = max_noftz(_max_3, _max_4);
                    pair_max[1] = _max_5;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        float _max_6 = max_noftz(pair_max[c_1], _shfl_xor_0);
                        pair_max[c_1] = _max_6;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        float _max_7 = max_noftz(pair_max[c_1], _shfl_xor_1);
                        pair_max[c_1] = _max_7;
                    }
                    float old_max_pair[2];
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        old_max_pair[c_2] = row_max_pair[c_2];
                        float _max_9 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_9;
                    }
                    const int max_owner_lanes = ((FOUR_LANE_MAX != 0) ? 4 : 8);
                    if (max_owner_lanes > lane) {
                        uint32_t _amf_u_1 = __float_as_uint(new_max_pair[0]);
                        uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                        uint32_t _amf_u_2 = __float_as_uint(new_max_pair[1]);
                        uint32_t _amf_mask_2 = -int32_t(_amf_u_2 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_2 ^ _amf_mask_2;
                        atomicMax(&my_exch_u32_ptr[col_pair_base], _amf_enc_1);
                        atomicMax(&my_exch_u32_ptr[col_pair_base + 1], _amf_enc_2);
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    uint32_t _amf_u_3 = my_exch_u32_ptr[col_pair_base];
                    uint32_t _amf_mask_3 = ((_amf_u_3 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_3 ^ _amf_mask_3);
                    new_max_pair[0] = _amf_dec_0;
                    uint32_t _amf_u_4 = my_exch_u32_ptr[col_pair_base + 1];
                    uint32_t _amf_mask_4 = ((_amf_u_4 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_4 ^ _amf_mask_4);
                    new_max_pair[1] = _amf_dec_1;
                    float acc_scale_pair[2];
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 2; c_3++) {
                        float delta = smx_scale * (row_max_pair[c_3] - new_max_pair[c_3]);
                        float _exp2_0 = approx_exp2(delta);
                        acc_scale_pair[c_3] = ((row_max_pair[c_3] > -LOOM_INF) ? _exp2_0 : 1.0f);
                    }
                    float stats_pair[4];
                    stats_pair[0] = old_max_pair[0];
                    stats_pair[1] = old_max_pair[1];
                    stats_pair[2] = new_max_pair[0];
                    stats_pair[3] = new_max_pair[1];
                    tmem_st_x4_f32(my_tmem_stats, stats_pair);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(corr_scale_addr + (is_wg1) * 8);
                    float exp_vals[8];
                    #pragma unroll
                    for (int c_4 = 0; c_4 < 8; c_4++) {
                        float safe_max = ((new_max_pair[c_4 % 2] == -LOOM_INF) ? 0.0f : new_max_pair[c_4 % 2]);
                        float max_scaled = safe_max * smx_scale;
                        float _exp2_1 = approx_exp2(sv[c_4] * smx_scale - max_scaled);
                        exp_vals[c_4] = _exp2_1;
                    }
                    if (kv_pos0 >= visible_local_keys) {
                        exp_vals[0] = 0.0f;
                        exp_vals[1] = 0.0f;
                    }
                    if (kv_pos1 >= visible_local_keys) {
                        exp_vals[2] = 0.0f;
                        exp_vals[3] = 0.0f;
                    }
                    if (kv_pos2 >= visible_local_keys) {
                        exp_vals[4] = 0.0f;
                        exp_vals[5] = 0.0f;
                    }
                    if (kv_pos3 >= visible_local_keys) {
                        exp_vals[6] = 0.0f;
                        exp_vals[7] = 0.0f;
                    }
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 2; c_5++) {
                        row_max_pair[c_5] = new_max_pair[c_5];
                    }
                    float pair_sum[2];
                    pair_sum[0] = exp_vals[0] + exp_vals[2] + exp_vals[4] + exp_vals[6];
                    pair_sum[1] = exp_vals[1] + exp_vals[3] + exp_vals[5] + exp_vals[7];
                    #pragma unroll
                    for (int c_6 = 0; c_6 < 2; c_6++) {
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_6], 16);
                        pair_sum[c_6] = pair_sum[c_6] + _shfl_xor_3;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_6], 8);
                        pair_sum[c_6] = pair_sum[c_6] + _shfl_xor_4;
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_6], 4);
                        pair_sum[c_6] = pair_sum[c_6] + _shfl_xor_5;
                    }
                    #pragma unroll
                    for (int c_7 = 0; c_7 < 2; c_7++) {
                        row_sum_pair[c_7] = row_sum_pair[c_7] * acc_scale_pair[c_7] + pair_sum[c_7];
                    }
                    unsigned int regs_p[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(exp_vals[_lp*2 + 0], exp_vals[_lp*2+1 + 0]));
                        regs_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    int slice_idx = warp_in_wg / 2;
                    int warp_idx_in_slice = warp_in_wg % 2;
                    int mtx_idx = lane / 8;
                    int thr_row_idx = lane % 8;
                    int mtx_col_idx = warp_idx_in_slice * 4 + mtx_idx;
                    int seg_col_idx = mtx_col_idx ^ thr_row_idx;
                    int stsm_offset = slice_idx * 8 * 128 + thr_row_idx * 128 + seg_col_idx * 16;
                    const void* _stmatrix_ptr_5 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset);
                    uint64_t _stmatrix_addr64_5;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_5) : "l"(_stmatrix_ptr_5));
                    uint32_t _stmatrix_addr_5;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_5) : "l"(_stmatrix_addr64_5));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
                    }
                    if (pair < cta_n_blocks / 2 - 1) {
                        if (is_wg1 != 0) {
                            mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                            _phase_corr_empty_1_0 ^= 1;
                        } else {
                            mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                            _phase_corr_empty_0_0 ^= 1;
                        }
                        if (is_wg1 != 0) {
                            mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                            _phase_s_full_1_0 ^= 1;
                        } else {
                            mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                            _phase_s_full_0_0 ^= 1;
                        }
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                            : "r"(my_tmem_s_base)
                            : "memory");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                            : "r"(my_tmem_s_base + 1048576)
                            : "memory");
                        #pragma unroll
                        for (int c_8 = 0; c_8 < 4; c_8++) {
                            sv[c_8] = sv_lo[c_8];
                            sv[c_8 + 4] = sv_hi[c_8];
                        }
                    }
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                    _phase_corr_empty_1_0 ^= 1;
                } else {
                    mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                    _phase_corr_empty_0_0 ^= 1;
                }
                if (lane < 4) {
                    my_exch_ptr[warp_in_wg * 8 + col_pair_base] = row_sum_pair[0];
                    my_exch_ptr[warp_in_wg * 8 + col_pair_base + 1] = row_sum_pair[1];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (wg_tid < 4) {
                    my_corr_ptr[col_pair_base] = my_exch_ptr[col_pair_base] + my_exch_ptr[8 + col_pair_base] + my_exch_ptr[16 + col_pair_base] + my_exch_ptr[24 + col_pair_base];
                    my_corr_ptr[col_pair_base + 1] = my_exch_ptr[col_pair_base + 1] + my_exch_ptr[8 + col_pair_base + 1] + my_exch_ptr[16 + col_pair_base + 1] + my_exch_ptr[24 + col_pair_base + 1];
                    my_exch_ptr[col_pair_base] = row_max_pair[0];
                    my_exch_ptr[col_pair_base + 1] = row_max_pair[1];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                mbarrier_arrive(corr_scale_addr + (is_wg1) * 8);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int warp_in_wg_c = warp % 4;
            const int wg_tid_c = warp_in_wg_c * 32 + lane;
            int d_idx = warp % 4 * 32 + lane;
            const int group_ratio_rt = NUM_Q_HEADS / NUM_KV_HEADS;
            unsigned int total_tiles_c = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            float smx_scale_c = softmax_scale_log2;
            {
                smx_scale_c = scale_log2_ptr[0];
            }
            unsigned int _phase_corr_scale_0 = 0;
            unsigned int _phase_corr_scale_1 = 0;
            unsigned int _phase_o_ready_0_0 = 0;
            unsigned int _phase_o_ready_1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_c = blockIdx.x; tile_idx_c < total_tiles_c; tile_idx_c += gridDim.x) {
                const int tiles_per_batch_1 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_1 = tile_idx_c / (unsigned int)tiles_per_batch_1;
                int tile_in_batch_1 = tile_idx_c % (unsigned int)tiles_per_batch_1;
                int q_row_idx_1 = tile_in_batch_1 / NUM_KV_HEADS;
                int kv_head_idx_1 = tile_in_batch_1 % NUM_KV_HEADS;
                int global_q_pos_1 = causal_seqlens_kv_global[batch_idx_1] + q_row_idx_1;
                int visible_local_keys_1 = global_q_pos_1 + 1;
                int win_start_1 = 0;
                int num_n_blocks_total_1 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_1 = num_n_blocks_total_1 + num_n_blocks_total_1 % 2;
                if (cta_n_blocks_1 < 4) {
                    cta_n_blocks_1 = 4;
                }
                if (cta_n_blocks_1 / 2 > 0) {
                    mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                    _phase_corr_scale_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                    _phase_corr_scale_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < cta_n_blocks_1 / 2; pair_1++) {
                    mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                    _phase_corr_scale_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float acc0_pair[2];
                    #pragma unroll
                    for (int c_9 = 0; c_9 < 2; c_9++) {
                        float max_diff0 = _tmem_load_0[c_9] - _tmem_load_0[c_9 + 2];
                        float scaled_diff0 = smx_scale_c * max_diff0;
                        float _exp2_2 = approx_exp2(scaled_diff0);
                        acc0_pair[c_9] = ((max_diff0 != 0.0f) ? _exp2_2 : 1.0f);
                    }
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                    _phase_o_ready_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_0 = ((acc0_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_0 = rescale_pred_0 | ((acc0_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_0 = __any_sync(0xFFFFFFFF, rescale_pred_0 != 0);
                    if (_vote_0 != 0) {
                        float o0_lo[4];
                        float o0_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[3]))
                            : "r"(taddr + 80)
                            : "memory");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[3]))
                            : "r"(taddr + 80 + 1048576)
                            : "memory");
                        float o0[8];
                        #pragma unroll
                        for (int h = 0; h < 4; h++) {
                            o0[h] = o0_lo[h];
                            o0[h + 4] = o0_hi[h];
                        }
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 8; h_1++) {
                            o0[h_1] = o0[h_1] * acc0_pair[h_1 % 2];
                        }
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 4; h_2++) {
                            o0_lo[h_2] = o0[h_2];
                            o0_hi[h_2] = o0[h_2 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[3]))
                            : "memory");
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[3]))
                            : "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                    _phase_corr_scale_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float acc1_pair[2];
                    #pragma unroll
                    for (int c_10 = 0; c_10 < 2; c_10++) {
                        float max_diff1 = _tmem_load_1[c_10] - _tmem_load_1[c_10 + 2];
                        float scaled_diff1 = smx_scale_c * max_diff1;
                        float _exp2_3 = approx_exp2(scaled_diff1);
                        acc1_pair[c_10] = ((max_diff1 != 0.0f) ? _exp2_3 : 1.0f);
                    }
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                    _phase_o_ready_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_1 = ((acc1_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_1 = rescale_pred_1 | ((acc1_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_1 = __any_sync(0xFFFFFFFF, rescale_pred_1 != 0);
                    if (_vote_1 != 0) {
                        float o1_lo[4];
                        float o1_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[3]))
                            : "r"(taddr + 88)
                            : "memory");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[3]))
                            : "r"(taddr + 88 + 1048576)
                            : "memory");
                        float o1[8];
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 4; h_3++) {
                            o1[h_3] = o1_lo[h_3];
                            o1[h_3 + 4] = o1_hi[h_3];
                        }
                        #pragma unroll
                        for (int h_4 = 0; h_4 < 8; h_4++) {
                            o1[h_4] = o1[h_4] * acc1_pair[h_4 % 2];
                        }
                        #pragma unroll
                        for (int h_5 = 0; h_5 < 4; h_5++) {
                            o1_lo[h_5] = o1[h_5];
                            o1_hi[h_5] = o1[h_5 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[3]))
                            : "memory");
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[3]))
                            : "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                _phase_corr_scale_0 ^= 1;
                mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                _phase_corr_scale_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scale0[8];
                float scale1[8];
                float local_sum[8];
                float local_max[8];
                #pragma unroll
                for (int c_11 = 0; c_11 < 8; c_11++) {
                    float m0 = smem_exch0[c_11];
                    float m1 = smem_exch1[c_11];
                    float s0 = smem_corr0[c_11];
                    float s1 = smem_corr1[c_11];
                    float _max_10 = max_noftz(m0, m1);
                    float fm = _max_10;
                    local_max[c_11] = fm;
                    float d0 = smx_scale_c * (m0 - fm);
                    float d1 = smx_scale_c * (m1 - fm);
                    float _exp2_4 = approx_exp2(d0);
                    scale0[c_11] = ((m0 == -LOOM_INF) ? 0.0f : _exp2_4);
                    float _exp2_5 = approx_exp2(d1);
                    scale1[c_11] = ((m1 == -LOOM_INF) ? 0.0f : _exp2_5);
                    local_sum[c_11] = s0 * scale0[c_11] + s1 * scale1[c_11];
                }
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                _phase_o_ready_0_0 ^= 1;
                mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                _phase_o_ready_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], taddr + 80 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], taddr + 88 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int h_6 = 0; h_6 < 8; h_6++) {
                    float final_o = 0.0f;
                    float local_lse = -LOOM_INF;
                    {
                        if (local_sum[h_6] > 0.0f && local_sum[h_6] == local_sum[h_6]) {
                            float merged = _tmem_load_2[h_6] * scale0[h_6] + _tmem_load_3[h_6] * scale1[h_6];
                            float _rcp_1 = approx_rcp(local_sum[h_6]);
                            final_o = merged * _rcp_1;
                            {
                                float _log2_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(local_sum[h_6]));
                                local_lse = local_max[h_6] * smx_scale_c + _log2_1;
                            }
                        }
                    }
                    if (group_ratio_rt > h_6) {
                        int q_head = kv_head_idx_1 * group_ratio_rt + h_6;
                        int o_idx = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head) * HEAD_DIM + d_idx;
                        *(reinterpret_cast<__nv_bfloat16*>(O_ptr + o_idx) + (0)) = __float2bfloat16_rn(final_o);
                        {
                            if (d_idx == 0) {
                                int lse_idx = (batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head;
                                *(reinterpret_cast<float*>(LSE_ptr + lse_idx) + (0)) = local_lse;
                            }
                        }
                    }
                }
                mbarrier_arrive(o_empty_0_addr);
                mbarrier_arrive(o_empty_1_addr);
                if (elect_sync()) {
                    mbarrier_arrive(stats_empty_addr);
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            const int tmem_s0v = taddr;
            const int tmem_s1v = taddr + 8;
            const int tmem_o0v = taddr + 80;
            const int tmem_o1v = taddr + 88;
            unsigned int total_tiles_m = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_o_empty_0_0 = 1;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_o_empty_1_0 = 1;
            unsigned int _phase_p_full_1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_m = blockIdx.x; tile_idx_m < total_tiles_m; tile_idx_m += gridDim.x) {
                const int tiles_per_batch_2 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_2 = tile_idx_m / (unsigned int)tiles_per_batch_2;
                int tile_in_batch_2 = tile_idx_m % (unsigned int)tiles_per_batch_2;
                int q_row_idx_2 = tile_in_batch_2 / NUM_KV_HEADS;
                int kv_head_idx_2 = tile_in_batch_2 % NUM_KV_HEADS;
                int global_q_pos_2 = causal_seqlens_kv_global[batch_idx_2] + q_row_idx_2;
                int visible_local_keys_2 = global_q_pos_2 + 1;
                int win_start_2 = 0;
                int num_n_blocks_total_2 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_2 = num_n_blocks_total_2 + num_n_blocks_total_2 % 2;
                if (cta_n_blocks_2 < 4) {
                    cta_n_blocks_2 = 4;
                }
                int inst0_stage = 0;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_s0), "r"(0));
                elect_commit(s_full_0_addr);
                elect_commit(kv_empty_addr);
                mbarrier_wait(kv_full_addr + 8, 0);
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 2048);
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
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_0), "r"(tmem_tmem_s1), "r"(0));
                elect_commit(s_full_1_addr);
                elect_commit(kv_empty_addr + 8);
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < cta_n_blocks_2 / 2 - 1; pair_2++) {
                    int s0_1 = inst0_stage;
                    int s1_1 = (inst0_stage + 1) % 4;
                    int s0_next = (inst0_stage + 2) % 4;
                    int s1_next = (inst0_stage + 3) % 4;
                    mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                    _phase_o_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0_1) * 8, 1);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_1) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (s0_1) * 8, o_ready_0_addr);
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_s0), "r"(0));
                    elect_commit(s_full_0_addr);
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                    _phase_o_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1_1) * 8, 1);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_1) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (s1_1) * 8, o_ready_1_addr);
                    mbarrier_wait(kv_full_addr + (s1_next) * 8, 0);
                    int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_next) * 2048);
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
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_3), "r"(tmem_tmem_s1), "r"(0));
                    elect_commit(s_full_1_addr);
                    elect_commit(kv_empty_addr + (s1_next) * 8);
                    inst0_stage = s0_next;
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                elect_commit(q_empty_addr);
                int s0_last = inst0_stage;
                int s1_last = (inst0_stage + 1) % 4;
                mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                _phase_o_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                _phase_p_full_0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
                int _mma_b_lo_6 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (s0_last) * 8, o_ready_0_addr);
                mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                _phase_o_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
                int _mma_b_lo_7 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (s1_last) * 8, o_ready_1_addr);
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
        }
    }
    // ---- Role: load_pgoff ----
    if (warp == 13) {
        { // load_pgoff_main
            unsigned int total_tiles_la = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            #pragma unroll 1
            for (unsigned int tile_idx_la = blockIdx.x; tile_idx_la < total_tiles_la; tile_idx_la += gridDim.x) {
                const int tiles_per_batch_3 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_3 = tile_idx_la / (unsigned int)tiles_per_batch_3;
                int tile_in_batch_3 = tile_idx_la % (unsigned int)tiles_per_batch_3;
                int q_row_idx_3 = tile_in_batch_3 / NUM_KV_HEADS;
                int kv_head_idx_3 = tile_in_batch_3 % NUM_KV_HEADS;
                int global_q_pos_3 = causal_seqlens_kv_global[batch_idx_3] + q_row_idx_3;
                int visible_local_keys_3 = global_q_pos_3 + 1;
                int win_start_3 = 0;
                int num_n_blocks_total_3 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_3 = num_n_blocks_total_3 + num_n_blocks_total_3 % 2;
                if (cta_n_blocks_3 < 4) {
                    cta_n_blocks_3 = 4;
                }
                int split_start_block_la = 0;
                if (elect_sync()) {
                    int pt_base_la = batch_idx_3 * max_pages_per_seq;
                    int kv_stage_la = 0;
                    int kv_phase_la = 1;
                    int prefill_la = ((cta_n_blocks_3 < 2) ? cta_n_blocks_3 : 2);
                    #pragma unroll 1
                    for (int ni_la = 0; ni_la < prefill_la; ni_la++) {
                        int n_block_la = split_start_block_la + cta_n_blocks_3 - 1 - ni_la;
                        int pg_base_la = n_block_la * 8;
                        mbarrier_wait(kv_empty_addr + (kv_stage_la) * 8, kv_phase_la);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_la) * 8, 16384);
                        int ldst_la = smem_kv_addr + (unsigned int)(kv_stage_la * 32768);
                        int pg_k_la[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base_la + pg_base_la);
                            pg_k_la[0 + 0] = _iv4.x;
                            pg_k_la[0 + 1] = _iv4.y;
                            pg_k_la[0 + 2] = _iv4.z;
                            pg_k_la[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_la = 0; pg_half_la < 4; pg_half_la++) {
                            int pg0_la = pg_k_la[pg_half_la];
                            #pragma unroll
                            for (int hg_la = 0; hg_la < 2; hg_la++) {
                                int toff_la = hg_la * 16384 + pg_half_la * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(ldst_la + toff_la), "l"(K), "r"(0), "r"(0), "r"(hg_la), "r"(kv_head_idx_3), "r"(pg0_la),
                                           "r"(kv_full_addr + (kv_stage_la) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        kv_stage_la += 1;
                        if (kv_stage_la == 4) { kv_stage_la = 0; kv_phase_la ^= 1; }
                    }
                    int main_ni_la = cta_n_blocks_3 - 2;
                    #pragma unroll 1
                    for (int ni_la_1 = 0; ni_la_1 < main_ni_la; ni_la_1++) {
                        int stage_la = ni_la_1 % 4;
                        int n_block_la_1 = split_start_block_la + cta_n_blocks_3 - 1 - ni_la_1;
                        int vpg_base_la = n_block_la_1 * 8;
                        mbarrier_wait(kv_empty_addr + (stage_la) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage_la) * 8, 16384);
                        int vdst_la = smem_kv_addr + (unsigned int)(stage_la * 32768);
                        int pg_v_la[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base_la + vpg_base_la);
                            pg_v_la[0 + 0] = _iv4.x;
                            pg_v_la[0 + 1] = _iv4.y;
                            pg_v_la[0 + 2] = _iv4.z;
                            pg_v_la[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_la_1 = 0; pg_half_la_1 < 4; pg_half_la_1++) {
                            int vpg0_la = pg_v_la[pg_half_la_1];
                            #pragma unroll
                            for (int hg_la_1 = 0; hg_la_1 < 2; hg_la_1++) {
                                int vtoff_la = hg_la_1 * 16384 + pg_half_la_1 * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(vdst_la + vtoff_la), "l"(V), "r"(0), "r"(0), "r"(hg_la_1), "r"(kv_head_idx_3), "r"(vpg0_la),
                                           "r"(kv_full_addr + (stage_la) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        int next_ni_la = ni_la_1 + 2;
                        int k_stage_la = next_ni_la % NUM_KV_STAGES;
                        int next_n_la = split_start_block_la + cta_n_blocks_3 - 1 - next_ni_la;
                        int npg_base_la = next_n_la * 8;
                        mbarrier_wait(kv_empty_addr + (k_stage_la) * 8, 1);
                        mbarrier_arrive_expect_tx(kv_full_addr + (k_stage_la) * 8, 16384);
                        int kdst_la = smem_kv_addr + (unsigned int)(k_stage_la * 32768);
                        int pg_nk_la[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base_la + npg_base_la);
                            pg_nk_la[0 + 0] = _iv4.x;
                            pg_nk_la[0 + 1] = _iv4.y;
                            pg_nk_la[0 + 2] = _iv4.z;
                            pg_nk_la[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_la_2 = 0; pg_half_la_2 < 4; pg_half_la_2++) {
                            int npg0_la = pg_nk_la[pg_half_la_2];
                            #pragma unroll
                            for (int hg_la_2 = 0; hg_la_2 < 2; hg_la_2++) {
                                int ntoff_la = hg_la_2 * 16384 + pg_half_la_2 * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(kdst_la + ntoff_la), "l"(K), "r"(0), "r"(0), "r"(hg_la_2), "r"(kv_head_idx_3), "r"(npg0_la),
                                           "r"(kv_full_addr + (k_stage_la) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int tail_ni_la = main_ni_la; tail_ni_la < cta_n_blocks_3; tail_ni_la++) {
                        int tstage_la = tail_ni_la % 4;
                        int tn_block_la = split_start_block_la + cta_n_blocks_3 - 1 - tail_ni_la;
                        int tvpg_base_la = tn_block_la * 8;
                        mbarrier_wait(kv_empty_addr + (tstage_la) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (tstage_la) * 8, 16384);
                        int tvdst_la = smem_kv_addr + (unsigned int)(tstage_la * 32768);
                        int tpg_v_la[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base_la + tvpg_base_la);
                            tpg_v_la[0 + 0] = _iv4.x;
                            tpg_v_la[0 + 1] = _iv4.y;
                            tpg_v_la[0 + 2] = _iv4.z;
                            tpg_v_la[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_la_3 = 0; pg_half_la_3 < 4; pg_half_la_3++) {
                            int tvpg0_la = tpg_v_la[pg_half_la_3];
                            #pragma unroll
                            for (int hg_la_3 = 0; hg_la_3 < 2; hg_la_3++) {
                                int tvtoff_la = hg_la_3 * 16384 + pg_half_la_3 * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(tvdst_la + tvtoff_la), "l"(V), "r"(0), "r"(0), "r"(hg_la_3), "r"(kv_head_idx_3), "r"(tvpg0_la),
                                           "r"(kv_full_addr + (tstage_la) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            unsigned int total_tiles_l = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_l = blockIdx.x; tile_idx_l < total_tiles_l; tile_idx_l += gridDim.x) {
                const int tiles_per_batch_4 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_4 = tile_idx_l / (unsigned int)tiles_per_batch_4;
                int tile_in_batch_4 = tile_idx_l % (unsigned int)tiles_per_batch_4;
                int q_row_idx_4 = tile_in_batch_4 / NUM_KV_HEADS;
                int kv_head_idx_4 = tile_in_batch_4 % NUM_KV_HEADS;
                int global_q_pos_4 = causal_seqlens_kv_global[batch_idx_4] + q_row_idx_4;
                int visible_local_keys_4 = global_q_pos_4 + 1;
                int win_start_4 = 0;
                int num_n_blocks_total_4 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_4 = num_n_blocks_total_4 + num_n_blocks_total_4 % 2;
                if (cta_n_blocks_4 < 4) {
                    cta_n_blocks_4 = 4;
                }
                int split_start_block_1 = 0;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    const int group_ratio_l = NUM_Q_HEADS / NUM_KV_HEADS;
                    int off_qt = (batch_idx_4 * Q_LEN + q_row_idx_4) * NUM_Q_HEADS + kv_head_idx_4 * group_ratio_l;
                    mbarrier_arrive_expect_tx(q_full_addr, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_qt_addr, Qt, 0, off_qt, 0, q_full_addr);
                    int pt_base = batch_idx_4 * max_pages_per_seq;
                    int kv_stage = 0;
                    int kv_phase = 1;
                    int prefill = ((cta_n_blocks_4 < 2) ? cta_n_blocks_4 : 2);
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        int n_block = split_start_block_1 + cta_n_blocks_4 - 1 - ni;
                        int pg_base = n_block * 8;
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                        int ldst = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                        int pg_k[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base + pg_base + 4);
                            pg_k[0 + 0] = _iv4.x;
                            pg_k[0 + 1] = _iv4.y;
                            pg_k[0 + 2] = _iv4.z;
                            pg_k[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half = 0; pg_half < 4; pg_half++) {
                            int pg0 = pg_k[pg_half];
                            const int pg_i = pg_half + 4;
                            #pragma unroll
                            for (int hg = 0; hg < 2; hg++) {
                                int toff = hg * 16384 + pg_i * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(ldst + toff), "l"(K), "r"(0), "r"(0), "r"(hg), "r"(kv_head_idx_4), "r"(pg0),
                                           "r"(kv_full_addr + (kv_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    int main_ni = cta_n_blocks_4 - 2;
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < main_ni; ni_1++) {
                        int stage = ni_1 % 4;
                        int n_block_1 = split_start_block_1 + cta_n_blocks_4 - 1 - ni_1;
                        int vpg_base = n_block_1 * 8;
                        mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 16384);
                        int vdst = smem_kv_addr + (unsigned int)(stage * 32768);
                        int pg_v[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base + vpg_base + 4);
                            pg_v[0 + 0] = _iv4.x;
                            pg_v[0 + 1] = _iv4.y;
                            pg_v[0 + 2] = _iv4.z;
                            pg_v[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_1 = 0; pg_half_1 < 4; pg_half_1++) {
                            int vpg0 = pg_v[pg_half_1];
                            const int pg_i_1 = pg_half_1 + 4;
                            #pragma unroll
                            for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                int vtoff = hg_1 * 16384 + pg_i_1 * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(vdst + vtoff), "l"(V), "r"(0), "r"(0), "r"(hg_1), "r"(kv_head_idx_4), "r"(vpg0),
                                           "r"(kv_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        int next_ni = ni_1 + 2;
                        int k_stage = next_ni % NUM_KV_STAGES;
                        int next_n = split_start_block_1 + cta_n_blocks_4 - 1 - next_ni;
                        int npg_base = next_n * 8;
                        mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                        mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 16384);
                        int kdst = smem_kv_addr + (unsigned int)(k_stage * 32768);
                        int pg_nk[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base + npg_base + 4);
                            pg_nk[0 + 0] = _iv4.x;
                            pg_nk[0 + 1] = _iv4.y;
                            pg_nk[0 + 2] = _iv4.z;
                            pg_nk[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_2 = 0; pg_half_2 < 4; pg_half_2++) {
                            int npg0 = pg_nk[pg_half_2];
                            const int pg_i_2 = pg_half_2 + 4;
                            #pragma unroll
                            for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                int ntoff = hg_2 * 16384 + pg_i_2 * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(kdst + ntoff), "l"(K), "r"(0), "r"(0), "r"(hg_2), "r"(kv_head_idx_4), "r"(npg0),
                                           "r"(kv_full_addr + (k_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int tail_ni = main_ni; tail_ni < cta_n_blocks_4; tail_ni++) {
                        int tstage = tail_ni % 4;
                        int tn_block = split_start_block_1 + cta_n_blocks_4 - 1 - tail_ni;
                        int tvpg_base = tn_block * 8;
                        mbarrier_wait(kv_empty_addr + (tstage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (tstage) * 8, 16384);
                        int tvdst = smem_kv_addr + (unsigned int)(tstage * 32768);
                        int tpg_v[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base + tvpg_base + 4);
                            tpg_v[0 + 0] = _iv4.x;
                            tpg_v[0 + 1] = _iv4.y;
                            tpg_v[0 + 2] = _iv4.z;
                            tpg_v[0 + 3] = _iv4.w;
                        }
                        #pragma unroll
                        for (int pg_half_3 = 0; pg_half_3 < 4; pg_half_3++) {
                            int tvpg0 = tpg_v[pg_half_3];
                            const int pg_i_3 = pg_half_3 + 4;
                            #pragma unroll
                            for (int hg_3 = 0; hg_3 < 2; hg_3++) {
                                int tvtoff = hg_3 * 16384 + pg_i_3 * 2048;
                                {
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(tvdst + tvtoff), "l"(V), "r"(0), "r"(0), "r"(hg_3), "r"(kv_head_idx_4), "r"(tvpg0),
                                           "r"(kv_full_addr + (tstage) * 8), "l"(0x14F0000000000000ULL) : "memory");
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
