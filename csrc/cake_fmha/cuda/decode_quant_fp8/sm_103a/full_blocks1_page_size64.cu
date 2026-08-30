/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef signed char        int8_t;
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

#define CAKE_FMHA_INF CUDART_INF_F
#define TMEM_NCOLS 96
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 9
#define NUM_PG_PIPE_STAGES 6
#define NUM_OP01_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_CORR_REDUCE_OFF 1024
#define SMEM_SMEM_CORR_REDUCE_STAGE_BYTES 128
#define SMEM_SMEM_CORR_REDUCE_STRIDE 128
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
#define SMEM_SMEM_QT_STAGE_BYTES 1024
#define SMEM_SMEM_QT_STRIDE 1024
#define SMEM_SMEM_KV_OFF 3712
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 3712
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P0_OFF 151168
#define SMEM_SMEM_P0_STAGE_BYTES 1024
#define SMEM_SMEM_P0_STRIDE 1024
#define SMEM_SMEM_P1_OFF 153216
#define SMEM_SMEM_P1_STAGE_BYTES 1024
#define SMEM_SMEM_P1_STRIDE 1024
#define SMEM_SMEM_O_OFF 155264
#define SMEM_SMEM_O_STAGE_BYTES 1024
#define SMEM_SMEM_O_STRIDE 1024
#define SMEM_SMEM_PG_OFF 156288
#define SMEM_SMEM_PG_STAGE_BYTES 256
#define SMEM_SMEM_PG_STRIDE 256
#define SMEM_WORK_RESPONSE_OFF 157824
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 157952
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 64
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif
#ifndef NUM_Q_HEADS
#define NUM_Q_HEADS 8
#endif
#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 2
#endif
#define NUM_SPLITS 0
#define BLOCKS_PER_SPLIT 0
#define FULL_BLOCKS 1
#define ENABLE_PDL 0

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


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_fmha_decode_quant_fp8(CakeFmhaTensorMap const* Qt, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, uint8_t* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int pt_batch_stride, int pt_v_offset, int bmm1_is_log2, int num_splits, int blocks_per_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 104)
    #define pg_full_addr (mbar_base + 176)
    #define pg_empty_addr (mbar_base + 224)
    #define s_full_0_addr (mbar_base + 272)
    #define s_full_1_addr (mbar_base + 280)
    #define s_empty_0_addr (mbar_base + 288)
    #define s_empty_1_addr (mbar_base + 296)
    #define o_free_0_addr (mbar_base + 304)
    #define o_free_1_addr (mbar_base + 312)
    #define o_done_0_addr (mbar_base + 320)
    #define o_done_1_addr (mbar_base + 328)
    #define corr_scale_addr (mbar_base + 336)
    #define corr_empty_0_addr (mbar_base + 352)
    #define corr_empty_1_addr (mbar_base + 360)
    #define stats_empty_addr (mbar_base + 368)
    #define tmem_dealloc_addr (mbar_base + 376)
    #define order_p01_0_addr (mbar_base + 384)
    #define order_p01_1_addr (mbar_base + 392)
    #define work_full_addr (mbar_base + 400)
    #define work_empty_addr (mbar_base + 416)
    #define throttle_full_addr (mbar_base + 432)
    #define throttle_empty_addr (mbar_base + 448)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Qt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* smem_corr_reduce = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr_reduce_addr = smem + 1024;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 3712);
    const int smem_kv_addr = smem + 3712;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 3712);
    const int smem_v_addr = smem + 3712;
    uint8_t* smem_p0 = reinterpret_cast<uint8_t*>(smem_raw + 151168);
    const int smem_p0_addr = smem + 151168;
    uint8_t* smem_p1 = reinterpret_cast<uint8_t*>(smem_raw + 153216);
    const int smem_p1_addr = smem + 153216;
    uint8_t* smem_o = reinterpret_cast<uint8_t*>(smem_raw + 155264);
    const int smem_o_addr = smem + 155264;
    int* smem_pg = reinterpret_cast<int*>(smem_raw + 156288);
    const int smem_pg_addr = smem + 156288;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 157824);
    const int work_response_addr = smem + 157824;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (25 groups, 58 barriers)
    // Mbarriers at smem_raw[0..464)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // kv_full: 9 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // kv_empty: 9 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // pg_full: 6 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            mbarrier_init(smem + 192, 32);
            mbarrier_init(smem + 200, 32);
            mbarrier_init(smem + 208, 32);
            mbarrier_init(smem + 216, 32);
            // pg_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            // s_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 288, 128);
            // s_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            // o_free_0: 1 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            // o_free_1: 1 barriers, init_count=128
            mbarrier_init(smem + 312, 128);
            // o_done_0: 1 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            // o_done_1: 1 barriers, init_count=1
            mbarrier_init(smem + 328, 1);
            // corr_scale: 2 barriers, init_count=128
            mbarrier_init(smem + 336, 128);
            mbarrier_init(smem + 344, 128);
            // corr_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 352, 128);
            // corr_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 368, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 376, 128);
            // order_p01_0: 1 barriers, init_count=128
            mbarrier_init(smem + 384, 128);
            // order_p01_1: 1 barriers, init_count=128
            mbarrier_init(smem + 392, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 416, 512);
            mbarrier_init(smem + 424, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=32
            mbarrier_init(smem + 432, 32);
            mbarrier_init(smem + 440, 32);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 448, 32);
            mbarrier_init(smem + 456, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 464);
    if (warp == 0) {
        int _tmem_hold = smem + 464;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0 = taddr + 80;
    const int tmem_tmem_o1 = taddr + 88;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            float log2e_mult_s = ((bmm1_is_log2 != 0) ? 1.0f : 1.4426950408889634f);
            float softmax_scale_log2 = bmm1_scale_ptr[0] * log2e_mult_s;
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
            uint8_t* my_p_base = ((is_wg1 != 0) ? smem_p1 : smem_p0);
            float sv[8];
            float sv_lo[4];
            float sv_hi[4];
            int op01_phase = ((is_wg1 != 0) ? 0 : 1);
            int op01_stage = 0;
            unsigned int work_stage_s = 0;
            unsigned int tile_idx_s = blockIdx.x;
            unsigned int total_tiles_s = BATCH_SIZE * NUM_KV_HEADS * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits);
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < total_tiles_s; _tile_iter_s++) {
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                int split_s = tile_idx_s % (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int bh_s = tile_idx_s / (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int batch_idx = bh_s / NUM_KV_HEADS;
                int seqlen_kv = seq_lens_kv[batch_idx];
                int num_n_blocks_total = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks = num_n_blocks_total + num_n_blocks_total % 2;
                int split_start_block = split_s * ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split);
                int rem_blocks = even_n_blocks - split_start_block;
                int capped_blocks = ((rem_blocks > ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split)) ? ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split) : rem_blocks);
                int cta_n_blocks = ((capped_blocks < 2) ? 2 : capped_blocks);
                int num_pairs = cta_n_blocks / 2;
                float row_max_pair[2];
                float row_sum_pair[2];
                row_max_pair[0] = -CAKE_FMHA_INF;
                row_max_pair[1] = -CAKE_FMHA_INF;
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
                    : "r"(my_tmem_s_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                    : "r"(my_tmem_s_base + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    sv[c] = sv_lo[c];
                    sv[c + 4] = sv_hi[c];
                }
                #pragma unroll 1
                for (int pair = 0; pair < num_pairs; pair++) {
                    int my_block = split_start_block + cta_n_blocks - 1 - 2 * pair - is_wg1;
                    int ldtm_row_base = warp_in_wg * 32 + lane / 4;
                    int mask_x_odd = ((lane % 2 == 0) ? 112 : 48);
                    int block_base = my_block * BLOCK_N;
                    int kv_row0 = ldtm_row_base;
                    int kv_row1 = ldtm_row_base + 8;
                    int kv_row2 = ldtm_row_base + 16;
                    int kv_row3 = ldtm_row_base + 24;
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
                        float _max_8 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_8;
                    }
                    if (lane < 8) {
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
                    float2 _f2_0 = make_float2(row_max_pair[0], row_max_pair[1]);
                    float2 _f2_1 = make_float2(new_max_pair[0], new_max_pair[1]);
                    float2 acc_delta_pair_f2 = sub_f32x2(_f2_0, _f2_1);
                    float2 _f2_2 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 acc_scaled_delta_pair_f2 = mul_f32x2(_f2_2, acc_delta_pair_f2);
                    acc_scale_pair[0] = 1.0f;
                    acc_scale_pair[1] = 1.0f;
                    int needs_acc_rescale = ((acc_delta_pair_f2.x != 0.0f) ? 1 : 0);
                    needs_acc_rescale = needs_acc_rescale | ((acc_delta_pair_f2.y != 0.0f) ? 1 : 0);
                    if (needs_acc_rescale != 0) {
                        float _exp2_0 = approx_exp2(acc_scaled_delta_pair_f2.x);
                        acc_scale_pair[0] = _exp2_0;
                        float _exp2_1 = approx_exp2(acc_scaled_delta_pair_f2.y);
                        acc_scale_pair[1] = _exp2_1;
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
                    row_max_pair[0] = new_max_pair[0];
                    row_max_pair[1] = new_max_pair[1];
                    float safe_max0 = ((new_max_pair[0] == -CAKE_FMHA_INF) ? 0.0f : new_max_pair[0]);
                    float safe_max1 = ((new_max_pair[1] == -CAKE_FMHA_INF) ? 0.0f : new_max_pair[1]);
                    float2 _f2_3 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 _f2_4 = make_float2(-softmax_scale_log2, -softmax_scale_log2);
                    float2 _f2_5 = make_float2(8.8073549f, 8.8073549f);
                    float2 _f2_6 = make_float2(safe_max0, safe_max1);
                    float2 neg_scaled_pair_f2 = fma_f32x2(_f2_6, _f2_4, _f2_5);
                    float2 _f2_7 = make_float2(sv[0], sv[1]);
                    float2 scaled01_pair_f2 = mul_f32x2(_f2_3, _f2_7);
                    float2 affine01_pair_f2 = add_f32x2(scaled01_pair_f2, neg_scaled_pair_f2);
                    exp_vals[0] = affine01_pair_f2.x;
                    exp_vals[1] = affine01_pair_f2.y;
                    float2 _f2_8 = make_float2(sv[2], sv[3]);
                    float2 affine23_pair_f2 = fma_f32x2(_f2_3, _f2_8, neg_scaled_pair_f2);
                    exp_vals[2] = affine23_pair_f2.x;
                    exp_vals[3] = affine23_pair_f2.y;
                    if (is_wg1 != 0) {
                        mbarrier_wait(order_p01_1_addr, op01_phase);
                    } else {
                        mbarrier_wait(order_p01_0_addr, op01_phase);
                    }
                    float _exp2_2 = approx_exp2(exp_vals[0]);
                    exp_vals[0] = _exp2_2;
                    float2 _f2_9 = make_float2(sv[4], sv[5]);
                    float2 affine45_pair_f2 = fma_f32x2(_f2_3, _f2_9, neg_scaled_pair_f2);
                    exp_vals[4] = affine45_pair_f2.x;
                    exp_vals[5] = affine45_pair_f2.y;
                    float _exp2_3 = approx_exp2(exp_vals[1]);
                    exp_vals[1] = _exp2_3;
                    if (is_wg1 != 0) {
                        mbarrier_arrive(order_p01_0_addr);
                    } else {
                        mbarrier_arrive(order_p01_1_addr);
                    }
                    op01_phase ^= 1;
                    float _exp2_4 = approx_exp2(exp_vals[2]);
                    exp_vals[2] = _exp2_4;
                    float2 _f2_10 = make_float2(sv[6], sv[7]);
                    float2 affine67_pair_f2 = fma_f32x2(_f2_3, _f2_10, neg_scaled_pair_f2);
                    exp_vals[6] = affine67_pair_f2.x;
                    exp_vals[7] = affine67_pair_f2.y;
                    float _exp2_5 = approx_exp2(exp_vals[3]);
                    exp_vals[3] = _exp2_5;
                    float _exp2_6 = approx_exp2(exp_vals[4]);
                    exp_vals[4] = _exp2_6;
                    float _exp2_7 = approx_exp2(exp_vals[5]);
                    exp_vals[5] = _exp2_7;
                    float _exp2_8 = approx_exp2(exp_vals[6]);
                    exp_vals[6] = _exp2_8;
                    float _exp2_9 = approx_exp2(exp_vals[7]);
                    exp_vals[7] = _exp2_9;
                    unsigned int regs_p[2];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(exp_vals[0]), "f"(exp_vals[1]),
                                               "f"(exp_vals[2]), "f"(exp_vals[3]));
                        regs_p[0] = _packed;
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
                            : "=r"(_packed) : "f"(exp_vals[4]), "f"(exp_vals[5]),
                                               "f"(exp_vals[6]), "f"(exp_vals[7]));
                        regs_p[1] = _packed;
                    }
                    int mtx_idx = lane / 8;
                    int thr_row_idx = lane % 8;
                    int seg_col_idx = warp_in_wg * 2 + mtx_idx ^ thr_row_idx;
                    int stsm_offset = thr_row_idx * 128 + seg_col_idx * 16;
                    const void* _stmatrix_b8_ptr_5 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset);
                    uint64_t _stmatrix_b8_addr64_5;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_5) : "l"(_stmatrix_b8_ptr_5));
                    uint32_t _stmatrix_b8_addr_5;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_5) : "l"(_stmatrix_b8_addr64_5));
                    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                        :: "r"(_stmatrix_b8_addr_5), "r"(regs_p[0]), "r"(regs_p[1])
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                        _phase_corr_empty_1_0 ^= 1;
                        mbarrier_arrive(s_empty_1_addr);
                    } else {
                        mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                        _phase_corr_empty_0_0 ^= 1;
                        mbarrier_arrive(s_empty_0_addr);
                    }
                    float2 _f2_11 = make_float2(exp_vals[0], exp_vals[1]);
                    float2 _f2_12 = make_float2(exp_vals[2], exp_vals[3]);
                    float2 _f2_13 = make_float2(exp_vals[4], exp_vals[5]);
                    float2 _f2_14 = make_float2(exp_vals[6], exp_vals[7]);
                    float2 _f2_15 = make_float2(row_sum_pair[0], row_sum_pair[1]);
                    float2 _f2_16 = make_float2(acc_scale_pair[0], acc_scale_pair[1]);
                    float2 row_sum_p0_f2 = fma_f32x2(_f2_15, _f2_16, _f2_11);
                    float2 row_sum_p1_f2 = add_f32x2(row_sum_p0_f2, _f2_12);
                    float2 row_sum_p2_f2 = add_f32x2(row_sum_p1_f2, _f2_13);
                    float2 row_sum_next_f2 = add_f32x2(row_sum_p2_f2, _f2_14);
                    row_sum_pair[0] = row_sum_next_f2.x;
                    row_sum_pair[1] = row_sum_next_f2.y;
                    if (pair < num_pairs - 1) {
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
                            : "r"(my_tmem_s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                            : "r"(my_tmem_s_base + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 4; c_3++) {
                            sv[c_3] = sv_lo[c_3];
                            sv[c_3 + 4] = sv_hi[c_3];
                        }
                    }
                }
                float final_stats_pair[4];
                final_stats_pair[0] = row_sum_pair[0];
                final_stats_pair[1] = row_sum_pair[1];
                final_stats_pair[2] = row_max_pair[0];
                final_stats_pair[3] = row_max_pair[1];
                tmem_st_x4_f32(my_tmem_stats, final_stats_pair);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(corr_scale_addr + (is_wg1) * 8);
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(work_response_addr + work_stage_s * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_s * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 2) { work_stage_s = 0; _phase_work_full ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                tile_idx_s = _clc_ctaid_4;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            float log2e_mult_c = ((bmm1_is_log2 != 0) ? 1.0f : 1.4426950408889634f);
            float softmax_scale_log2_1 = bmm1_scale_ptr[0] * log2e_mult_c;
            float output_scale = bmm2_scale_ptr[0];
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int corr_tid = warp % 4 * 32 + lane;
            const int col_pair_base_c = corr_tid % 4 * 2;
            const int group_ratio_rt = NUM_Q_HEADS / NUM_KV_HEADS;
            unsigned int total_tiles_c = BATCH_SIZE * NUM_KV_HEADS * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits);
            unsigned int work_stage_c = 0;
            unsigned int tile_idx_c = blockIdx.x;
            unsigned int _phase_corr_scale_0 = 0;
            unsigned int _phase_corr_scale_1 = 0;
            unsigned int _phase_o_done_0_0 = 0;
            unsigned int _phase_o_done_1_0 = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < total_tiles_c; _tile_iter_c++) {
                int split_c = tile_idx_c % (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int bh_c = tile_idx_c / (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int batch_idx_1 = bh_c / NUM_KV_HEADS;
                int kv_head_idx = bh_c % NUM_KV_HEADS;
                int seqlen_kv_1 = seq_lens_kv[batch_idx_1];
                int num_n_blocks_total_1 = (seqlen_kv_1 + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_1 = num_n_blocks_total_1 + num_n_blocks_total_1 % 2;
                int split_start_block_1 = split_c * ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split);
                int rem_blocks_1 = even_n_blocks_1 - split_start_block_1;
                int capped_blocks_1 = ((rem_blocks_1 > ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split)) ? ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split) : rem_blocks_1);
                int cta_n_blocks_1 = ((capped_blocks_1 < 2) ? 2 : capped_blocks_1);
                int num_pairs_1 = cta_n_blocks_1 / 2;
                if (num_pairs_1 > 0) {
                    mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                    _phase_corr_scale_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(o_free_0_addr);
                    mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                    _phase_corr_scale_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(o_free_1_addr);
                }
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < num_pairs_1; pair_1++) {
                    mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                    _phase_corr_scale_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_done_0_addr, _phase_o_done_0_0);
                    _phase_o_done_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc0_pair[2];
                    float2 _f2_17 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                    float2 _f2_18 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
                    float2 max_diff0_pair_f2 = sub_f32x2(_f2_17, _f2_18);
                    float2 _f2_19 = make_float2(softmax_scale_log2_1, softmax_scale_log2_1);
                    float2 scaled_diff0_pair_f2 = mul_f32x2(_f2_19, max_diff0_pair_f2);
                    float _exp2_10 = approx_exp2(scaled_diff0_pair_f2.x);
                    acc0_pair[0] = ((max_diff0_pair_f2.x != 0.0f) ? _exp2_10 : 1.0f);
                    float _exp2_11 = approx_exp2(scaled_diff0_pair_f2.y);
                    acc0_pair[1] = ((max_diff0_pair_f2.y != 0.0f) ? _exp2_11 : 1.0f);
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
                            : "r"(taddr + 80));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[3]))
                            : "r"(taddr + 80 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float o0[8];
                        #pragma unroll
                        for (int h = 0; h < 4; h++) {
                            o0[h] = o0_lo[h];
                            o0[h + 4] = o0_hi[h];
                        }
                        float2 _f2_20 = make_float2(acc0_pair[0], acc0_pair[1]);
                        float2 _f2_21 = make_float2(o0[0], o0[1]);
                        float2 _f2_22 = make_float2(o0[2], o0[3]);
                        float2 _f2_23 = make_float2(o0[4], o0[5]);
                        float2 _f2_24 = make_float2(o0[6], o0[7]);
                        float2 o0_lo01_scaled_f2 = mul_f32x2(_f2_21, _f2_20);
                        float2 o0_lo23_scaled_f2 = mul_f32x2(_f2_22, _f2_20);
                        float2 o0_hi01_scaled_f2 = mul_f32x2(_f2_23, _f2_20);
                        float2 o0_hi23_scaled_f2 = mul_f32x2(_f2_24, _f2_20);
                        o0[0] = o0_lo01_scaled_f2.x;
                        o0[1] = o0_lo01_scaled_f2.y;
                        o0[2] = o0_lo23_scaled_f2.x;
                        o0[3] = o0_lo23_scaled_f2.y;
                        o0[4] = o0_hi01_scaled_f2.x;
                        o0[5] = o0_hi01_scaled_f2.y;
                        o0[6] = o0_hi23_scaled_f2.x;
                        o0[7] = o0_hi23_scaled_f2.y;
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 4; h_1++) {
                            o0_lo[h_1] = o0[h_1];
                            o0_hi[h_1] = o0[h_1 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_free_0_addr);
                    mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                    _phase_corr_scale_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_done_1_addr, _phase_o_done_1_0);
                    _phase_o_done_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc1_pair[2];
                    float2 _f2_25 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                    float2 _f2_26 = make_float2(_tmem_load_1[2], _tmem_load_1[3]);
                    float2 max_diff1_pair_f2 = sub_f32x2(_f2_25, _f2_26);
                    float2 _f2_27 = make_float2(softmax_scale_log2_1, softmax_scale_log2_1);
                    float2 scaled_diff1_pair_f2 = mul_f32x2(_f2_27, max_diff1_pair_f2);
                    float _exp2_12 = approx_exp2(scaled_diff1_pair_f2.x);
                    acc1_pair[0] = ((max_diff1_pair_f2.x != 0.0f) ? _exp2_12 : 1.0f);
                    float _exp2_13 = approx_exp2(scaled_diff1_pair_f2.y);
                    acc1_pair[1] = ((max_diff1_pair_f2.y != 0.0f) ? _exp2_13 : 1.0f);
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
                            : "r"(taddr + 88));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[3]))
                            : "r"(taddr + 88 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float o1[8];
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 4; h_2++) {
                            o1[h_2] = o1_lo[h_2];
                            o1[h_2 + 4] = o1_hi[h_2];
                        }
                        float2 _f2_28 = make_float2(acc1_pair[0], acc1_pair[1]);
                        float2 _f2_29 = make_float2(o1[0], o1[1]);
                        float2 _f2_30 = make_float2(o1[2], o1[3]);
                        float2 _f2_31 = make_float2(o1[4], o1[5]);
                        float2 _f2_32 = make_float2(o1[6], o1[7]);
                        float2 o1_lo01_scaled_f2 = mul_f32x2(_f2_29, _f2_28);
                        float2 o1_lo23_scaled_f2 = mul_f32x2(_f2_30, _f2_28);
                        float2 o1_hi01_scaled_f2 = mul_f32x2(_f2_31, _f2_28);
                        float2 o1_hi23_scaled_f2 = mul_f32x2(_f2_32, _f2_28);
                        o1[0] = o1_lo01_scaled_f2.x;
                        o1[1] = o1_lo01_scaled_f2.y;
                        o1[2] = o1_lo23_scaled_f2.x;
                        o1[3] = o1_lo23_scaled_f2.y;
                        o1[4] = o1_hi01_scaled_f2.x;
                        o1[5] = o1_hi01_scaled_f2.y;
                        o1[6] = o1_hi23_scaled_f2.x;
                        o1[7] = o1_hi23_scaled_f2.y;
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 4; h_3++) {
                            o1_lo[h_3] = o1[h_3];
                            o1_hi[h_3] = o1[h_3 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_free_1_addr);
                }
                mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                _phase_corr_scale_0 ^= 1;
                mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                _phase_corr_scale_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scale0_pair[2];
                float scale1_pair[2];
                float local_sum_pair[2];
                const int idx0_c = col_pair_base_c;
                const int idx1_c = col_pair_base_c + 1;
                float _tmem_load_2[4];
                tmem_ld_x4(&_tmem_load_2[0], taddr + 16 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[4];
                tmem_ld_x4(&_tmem_load_3[0], taddr + 48 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                float2 _f2_33 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                float2 _f2_34 = make_float2(_tmem_load_3[0], _tmem_load_3[1]);
                float m0_0 = _tmem_load_2[2];
                float m0_1 = _tmem_load_2[3];
                float m1_0 = _tmem_load_3[2];
                float m1_1 = _tmem_load_3[3];
                float _max_9 = max_noftz(m0_0, m1_0);
                float fm0 = _max_9;
                float _max_10 = max_noftz(m0_1, m1_1);
                float fm1 = _max_10;
                float2 _f2_35 = make_float2(m0_0, m0_1);
                float2 _f2_36 = make_float2(m1_0, m1_1);
                float2 _f2_37 = make_float2(fm0, fm1);
                float2 max_diff_i0_f2 = sub_f32x2(_f2_35, _f2_37);
                float2 max_diff_i1_f2 = sub_f32x2(_f2_36, _f2_37);
                float2 _f2_38 = make_float2(softmax_scale_log2_1, softmax_scale_log2_1);
                float2 d0_pair_f2 = mul_f32x2(_f2_38, max_diff_i0_f2);
                float2 d1_pair_f2 = mul_f32x2(_f2_38, max_diff_i1_f2);
                float _exp2_14 = approx_exp2(d0_pair_f2.x);
                scale0_pair[0] = ((m0_0 == -CAKE_FMHA_INF) ? 0.0f : _exp2_14);
                float _exp2_15 = approx_exp2(d0_pair_f2.y);
                scale0_pair[1] = ((m0_1 == -CAKE_FMHA_INF) ? 0.0f : _exp2_15);
                float _exp2_16 = approx_exp2(d1_pair_f2.x);
                scale1_pair[0] = ((m1_0 == -CAKE_FMHA_INF) ? 0.0f : _exp2_16);
                float _exp2_17 = approx_exp2(d1_pair_f2.y);
                scale1_pair[1] = ((m1_1 == -CAKE_FMHA_INF) ? 0.0f : _exp2_17);
                float2 _f2_39 = make_float2(scale0_pair[0], scale0_pair[1]);
                float2 _f2_40 = make_float2(scale1_pair[0], scale1_pair[1]);
                float2 s1_scaled_pair_f2 = mul_f32x2(_f2_34, _f2_40);
                float2 local_sum_pair_f2 = fma_f32x2(_f2_33, _f2_39, s1_scaled_pair_f2);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, local_sum_pair_f2.x, 16);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, local_sum_pair_f2.y, 16);
                float2 _f2_41 = make_float2(_shfl_xor_2, _shfl_xor_3);
                float2 sum16_pair_f2 = add_f32x2(local_sum_pair_f2, _f2_41);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum16_pair_f2.x, 8);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sum16_pair_f2.y, 8);
                float2 _f2_42 = make_float2(_shfl_xor_4, _shfl_xor_5);
                float2 sum8_pair_f2 = add_f32x2(sum16_pair_f2, _f2_42);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, sum8_pair_f2.x, 4);
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, sum8_pair_f2.y, 4);
                float2 _f2_43 = make_float2(_shfl_xor_6, _shfl_xor_7);
                float2 reduced_sum_pair_f2 = add_f32x2(sum8_pair_f2, _f2_43);
                if (lane < 4) {
                    const int warp_sum_base_c = warp % 4 * 8 + col_pair_base_c;
                    smem_corr_reduce[warp_sum_base_c] = reduced_sum_pair_f2.x;
                    smem_corr_reduce[warp_sum_base_c + 1] = reduced_sum_pair_f2.y;
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float2 _f2_44 = make_float2(smem_corr_reduce[idx0_c], smem_corr_reduce[idx1_c]);
                float2 _f2_45 = make_float2(smem_corr_reduce[idx0_c + 8], smem_corr_reduce[idx1_c + 8]);
                float2 _f2_46 = make_float2(smem_corr_reduce[idx0_c + 16], smem_corr_reduce[idx1_c + 16]);
                float2 _f2_47 = make_float2(smem_corr_reduce[idx0_c + 24], smem_corr_reduce[idx1_c + 24]);
                float2 sum_w01_pair_f2 = add_f32x2(_f2_44, _f2_45);
                float2 sum_w23_pair_f2 = add_f32x2(_f2_46, _f2_47);
                float2 sum_reduced_pair_f2 = add_f32x2(sum_w01_pair_f2, sum_w23_pair_f2);
                local_sum_pair[0] = sum_reduced_pair_f2.x;
                local_sum_pair[1] = sum_reduced_pair_f2.y;
                mbarrier_wait(o_done_0_addr, _phase_o_done_0_0);
                _phase_o_done_0_0 ^= 1;
                mbarrier_wait(o_done_1_addr, _phase_o_done_1_0);
                _phase_o_done_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float inv_sum_pair[2];
                #pragma unroll
                for (int c_4 = 0; c_4 < 2; c_4++) {
                    float _rcp_0 = approx_rcp(local_sum_pair[c_4]);
                    inv_sum_pair[c_4] = ((local_sum_pair[c_4] == 0.0f) ? 0.0f : _rcp_0);
                }
                float o0_lo_epi[4];
                float o0_hi_epi[4];
                float o1_lo_epi[4];
                float o1_hi_epi[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[3]))
                    : "r"(taddr + 80));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[3]))
                    : "r"(taddr + 80 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[3]))
                    : "r"(taddr + 88));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[3]))
                    : "r"(taddr + 88 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits) > 1) {
                    float part_vals[8];
                    float pscale0_e = scale0_pair[0] * inv_sum_pair[0];
                    float pscale0_o = scale0_pair[1] * inv_sum_pair[1];
                    float pscale1_e = scale1_pair[0] * inv_sum_pair[0];
                    float pscale1_o = scale1_pair[1] * inv_sum_pair[1];
                    part_vals[0] = o0_lo_epi[0] * pscale0_e + o1_lo_epi[0] * pscale1_e;
                    part_vals[1] = o0_lo_epi[1] * pscale0_o + o1_lo_epi[1] * pscale1_o;
                    part_vals[2] = o0_lo_epi[2] * pscale0_e + o1_lo_epi[2] * pscale1_e;
                    part_vals[3] = o0_lo_epi[3] * pscale0_o + o1_lo_epi[3] * pscale1_o;
                    part_vals[4] = o0_hi_epi[0] * pscale0_e + o1_hi_epi[0] * pscale1_e;
                    part_vals[5] = o0_hi_epi[1] * pscale0_o + o1_hi_epi[1] * pscale1_o;
                    part_vals[6] = o0_hi_epi[2] * pscale0_e + o1_hi_epi[2] * pscale1_e;
                    part_vals[7] = o0_hi_epi[3] * pscale0_o + o1_hi_epi[3] * pscale1_o;
                    int corr_x_odd = ((lane % 2 == 0) ? 112 : 48);
                    int corr_row_e = tmem_row_base_v_1 + lane / 4;
                    int q_even = col_pair_base_c;
                    int q_odd = col_pair_base_c + 1;
                    int head_base = batch_idx_1 * NUM_Q_HEADS + kv_head_idx * group_ratio_rt;
                    int po_even = ((head_base + q_even) * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits) + split_c) * HEAD_DIM;
                    int po_odd = ((head_base + q_odd) * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits) + split_c) * HEAD_DIM;
                    if (q_even < group_ratio_rt) {
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e ^ 80))) + (0)) = part_vals[0];
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e + 8 ^ 80))) + (0)) = part_vals[2];
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e + 16 ^ 80))) + (0)) = part_vals[4];
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e + 24 ^ 80))) + (0)) = part_vals[6];
                    }
                    if (q_odd < group_ratio_rt) {
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e ^ corr_x_odd))) + (0)) = part_vals[1];
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e + 8 ^ corr_x_odd))) + (0)) = part_vals[3];
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e + 16 ^ corr_x_odd))) + (0)) = part_vals[5];
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e + 24 ^ corr_x_odd))) + (0)) = part_vals[7];
                    }
                    if (corr_tid < 4) {
                        int st_even = (head_base + q_even) * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits) + split_c;
                        int st_odd = (head_base + q_odd) * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits) + split_c;
                        if (q_even < group_ratio_rt) {
                            *(reinterpret_cast<float*>(partial_max + st_even) + (0)) = fm0 * softmax_scale_log2_1;
                            *(reinterpret_cast<float*>(partial_sum + st_even) + (0)) = local_sum_pair[0];
                        }
                        if (q_odd < group_ratio_rt) {
                            *(reinterpret_cast<float*>(partial_max + st_odd) + (0)) = fm1 * softmax_scale_log2_1;
                            *(reinterpret_cast<float*>(partial_sum + st_odd) + (0)) = local_sum_pair[1];
                        }
                    }
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(stats_empty_addr);
                    }
                } else {
                    float out_vals[8];
                    float2 _f2_48 = make_float2(inv_sum_pair[0], inv_sum_pair[1]);
                    float2 _f2_49 = make_float2(output_scale, output_scale);
                    float2 norm_pair_f2 = mul_f32x2(_f2_48, _f2_49);
                    float2 final_scale0_pair_f2 = mul_f32x2(_f2_39, norm_pair_f2);
                    float2 final_scale1_pair_f2 = mul_f32x2(_f2_40, norm_pair_f2);
                    float2 _f2_50 = make_float2(o0_lo_epi[0], o0_lo_epi[1]);
                    float2 _f2_51 = make_float2(o1_lo_epi[0], o1_lo_epi[1]);
                    float2 o1_lo01_scaled_f2_1 = mul_f32x2(_f2_51, final_scale1_pair_f2);
                    float2 out_lo01_f2 = fma_f32x2(_f2_50, final_scale0_pair_f2, o1_lo01_scaled_f2_1);
                    out_vals[0] = out_lo01_f2.x;
                    out_vals[1] = out_lo01_f2.y;
                    float2 _f2_52 = make_float2(o0_lo_epi[2], o0_lo_epi[3]);
                    float2 _f2_53 = make_float2(o1_lo_epi[2], o1_lo_epi[3]);
                    float2 o1_lo23_scaled_f2_1 = mul_f32x2(_f2_53, final_scale1_pair_f2);
                    float2 out_lo23_f2 = fma_f32x2(_f2_52, final_scale0_pair_f2, o1_lo23_scaled_f2_1);
                    out_vals[2] = out_lo23_f2.x;
                    out_vals[3] = out_lo23_f2.y;
                    float2 _f2_54 = make_float2(o0_hi_epi[0], o0_hi_epi[1]);
                    float2 _f2_55 = make_float2(o1_hi_epi[0], o1_hi_epi[1]);
                    float2 o1_hi01_scaled_f2_1 = mul_f32x2(_f2_55, final_scale1_pair_f2);
                    float2 out_hi01_f2 = fma_f32x2(_f2_54, final_scale0_pair_f2, o1_hi01_scaled_f2_1);
                    out_vals[4] = out_hi01_f2.x;
                    out_vals[5] = out_hi01_f2.y;
                    float2 _f2_56 = make_float2(o0_hi_epi[2], o0_hi_epi[3]);
                    float2 _f2_57 = make_float2(o1_hi_epi[2], o1_hi_epi[3]);
                    float2 o1_hi23_scaled_f2_1 = mul_f32x2(_f2_57, final_scale1_pair_f2);
                    float2 out_hi23_f2 = fma_f32x2(_f2_56, final_scale0_pair_f2, o1_hi23_scaled_f2_1);
                    out_vals[6] = out_hi23_f2.x;
                    out_vals[7] = out_hi23_f2.y;
                    unsigned int regs_o[2];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(out_vals[0]), "f"(out_vals[1]),
                                               "f"(out_vals[2]), "f"(out_vals[3]));
                        regs_o[0] = _packed;
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
                            : "=r"(_packed) : "f"(out_vals[4]), "f"(out_vals[5]),
                                               "f"(out_vals[6]), "f"(out_vals[7]));
                        regs_o[1] = _packed;
                    }
                    int o_mtx_idx = lane / 8;
                    int o_thr_row_idx = lane % 8;
                    int o_seg_col_idx = warp % 4 * 2 + o_mtx_idx ^ o_thr_row_idx;
                    int o_stsm_offset = o_thr_row_idx * 128 + o_seg_col_idx * 16;
                    const void* _stmatrix_b8_ptr_0 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_o) + o_stsm_offset);
                    uint64_t _stmatrix_b8_addr64_0;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_0) : "l"(_stmatrix_b8_ptr_0));
                    uint32_t _stmatrix_b8_addr_0;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_0) : "l"(_stmatrix_b8_addr64_0));
                    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                        :: "r"(_stmatrix_b8_addr_0), "r"(regs_o[0]), "r"(regs_o[1])
                        : "memory");
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                    int copy_base = corr_tid * 16;
                    int copy_row = copy_base / 128;
                    int copy_col = copy_base % 128;
                    int copy_smem_offset = copy_base ^ copy_row % 8 * 16;
                    if (copy_row < group_ratio_rt) {
                        unsigned int copy_vec[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 3]))
                            : "r"(smem_o_addr + (unsigned int)copy_smem_offset));
                        int o_base = (batch_idx_1 * NUM_Q_HEADS + kv_head_idx * group_ratio_rt) * HEAD_DIM;
                        reinterpret_cast<int4*>(O + (o_base + copy_row * HEAD_DIM + copy_col))[0] = reinterpret_cast<int4*>(copy_vec)[0];
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(stats_empty_addr);
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                uint32_t _clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_5)
                    : "r"(work_response_addr + work_stage_c * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_c * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 2) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
                tile_idx_c = _clc_ctaid_5;
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
            unsigned int total_tiles_m = BATCH_SIZE * NUM_KV_HEADS * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits);
            int q_slot_m = 0;
            int q_phase_m = 0;
            int kv_slot_m = 0;
            int kv_phase_m = 0;
            unsigned int work_stage_m = 0;
            unsigned int tile_idx_m = blockIdx.x;
            unsigned int _phase_s_empty_0_0 = 1;
            unsigned int _phase_s_empty_1_0 = 1;
            unsigned int _phase_o_free_0_0 = 0;
            unsigned int _phase_o_free_1_0 = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < total_tiles_m; _tile_iter_m++) {
                int split_m = tile_idx_m % (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int bh_m = tile_idx_m / (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int batch_idx_2 = bh_m / NUM_KV_HEADS;
                int seqlen_kv_2 = seq_lens_kv[batch_idx_2];
                int num_n_blocks_total_2 = (seqlen_kv_2 + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_2 = num_n_blocks_total_2 + num_n_blocks_total_2 % 2;
                int split_start_block_2 = split_m * ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split);
                int rem_blocks_2 = even_n_blocks_2 - split_start_block_2;
                int capped_blocks_2 = ((rem_blocks_2 > ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split)) ? ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split) : rem_blocks_2);
                int cta_n_blocks_2 = ((capped_blocks_2 < 2) ? 2 : capped_blocks_2);
                int num_pairs_2 = cta_n_blocks_2 / 2;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr + (q_slot_m) * 8, q_phase_m);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
                int _mma_b_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
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
                    "mov.b32 id, 134348816;\n\t"
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
                elect_commit2(kv_empty_addr + (kv_slot_m) * 8, s_full_0_addr);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
                int _mma_b_lo_1 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
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
                    "mov.b32 id, 134348816;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_tmem_s1), "r"(0));
                elect_commit2(kv_empty_addr + (kv_slot_m) * 8, s_full_1_addr);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < num_pairs_2 - 1; pair_2++) {
                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                    _phase_s_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
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
                    "mov.b32 id, 134348816;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_tmem_s0), "r"(0));
                    elect_commit2(kv_empty_addr + (kv_slot_m) * 8, s_full_0_addr);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(o_free_0_addr, _phase_o_free_0_0);
                    _phase_o_free_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134381584;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (kv_slot_m) * 8, o_done_0_addr);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    int _mma_a_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
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
                    "mov.b32 id, 134348816;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_s1), "r"(0));
                    elect_commit2(kv_empty_addr + (kv_slot_m) * 8, s_full_1_addr);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(o_free_1_addr, _phase_o_free_1_0);
                    _phase_o_free_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134381584;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (kv_slot_m) * 8, o_done_1_addr);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                elect_commit(q_empty_addr + (q_slot_m) * 8);
                q_slot_m += 1;
                if (q_slot_m == 2) { q_slot_m = 0; q_phase_m ^= 1; }
                mbarrier_wait(o_free_0_addr, _phase_o_free_0_0);
                _phase_o_free_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 134381584;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (kv_slot_m) * 8, o_done_0_addr);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(o_free_1_addr, _phase_o_free_1_0);
                _phase_o_free_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 134381584;\n\t"
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (kv_slot_m) * 8, o_done_1_addr);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage_m * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_m * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 2) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                tile_idx_m = _clc_ctaid_3;
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
            int pg_slot_p = 0;
            int pg_phase_p = 1;
            unsigned int work_stage_p = 0;
            unsigned int tile_idx_p = blockIdx.x;
            unsigned int total_tiles_p = BATCH_SIZE * NUM_KV_HEADS * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits);
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < total_tiles_p; _tile_iter_p++) {
                int split_p = tile_idx_p % (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int bh_p = tile_idx_p / (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int batch_idx_p = bh_p / NUM_KV_HEADS;
                int seqlen_kv_p = seq_lens_kv[batch_idx_p];
                int num_n_blocks_total_p = (seqlen_kv_p + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_p = num_n_blocks_total_p + num_n_blocks_total_p % 2;
                int split_start_block_p = split_p * ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split);
                int rem_blocks_p = even_n_blocks_p - split_start_block_p;
                int capped_blocks_p = ((rem_blocks_p > ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split)) ? ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split) : rem_blocks_p);
                int cta_n_blocks_p = ((capped_blocks_p < 2) ? 2 : capped_blocks_p);
                int pt_base_k_p = batch_idx_p * pt_batch_stride;
                int pt_base_v_p = pt_base_k_p + pt_v_offset;
                const int pages_per_block_p = BLOCK_N / PAGE_SIZE;
                #pragma unroll 1
                for (int group_base_p = 0; group_base_p < cta_n_blocks_p; group_base_p += 4) {
                    int group_rem_p = cta_n_blocks_p - group_base_p;
                    int group_blocks_p = ((group_rem_p > 4) ? 4 : group_rem_p);
                    int group_block_p = lane / pages_per_block_p;
                    int active_pg_p = ((group_block_p < group_blocks_p) ? 1 : 0);
                    int safe_group_block_p = ((active_pg_p != 0) ? group_block_p : 0);
                    int n_block_p = split_start_block_p + cta_n_blocks_p - 1 - group_base_p - safe_group_block_p;
                    int page_i_p = lane % pages_per_block_p;
                    int page_base_p = n_block_p * pages_per_block_p;
                    mbarrier_wait(pg_empty_addr + (pg_slot_p) * 8, pg_phase_p);
                    int pg_stage_addr_p = smem_pg_addr + (unsigned int)(pg_slot_p * 256);
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((active_pg_p) ? 1 : 0), "r"(pg_stage_addr_p + lane * 4), "l"(page_table + (pt_base_k_p + page_base_p + page_i_p)));
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((active_pg_p) ? 1 : 0), "r"(pg_stage_addr_p + (32 + lane) * 4), "l"(page_table + (pt_base_v_p + page_base_p + page_i_p)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(pg_full_addr + (pg_slot_p) * 8) : "memory");
                    mbarrier_arrive(pg_full_addr + (pg_slot_p) * 8);
                    pg_slot_p += 1;
                    if (pg_slot_p == 6) { pg_slot_p = 0; pg_phase_p ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + work_stage_p * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_p * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 2) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                tile_idx_p = _clc_ctaid_0;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        { // scheduler_main
            unsigned int work_stage_h = 0;
            unsigned int throttle_stage_h = 0;
            unsigned int tile_idx_h = blockIdx.x;
            unsigned int total_tiles_h = BATCH_SIZE * NUM_KV_HEADS * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits);
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_h = 0; _tile_iter_h < total_tiles_h; _tile_iter_h++) {
                mbarrier_wait(throttle_full_addr + (throttle_stage_h) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_h) * 8);
                throttle_stage_h += 1;
                if (throttle_stage_h == 2) { throttle_stage_h = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_h) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_h) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_h * 16), "r"(work_full_addr + work_stage_h * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (work_stage_h) * 8, _phase_work_full_4);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_h * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_h * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_h) * 8);
                work_stage_h += 1;
                if (work_stage_h == 2) { work_stage_h = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                tile_idx_h = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            int q_slot_l = 0;
            int q_phase_l = 1;
            int pg_slot_l = 0;
            int pg_phase_l = 0;
            int kv_slot_l = 0;
            int kv_phase_l = 1;
            unsigned int work_stage_l = 0;
            unsigned int throttle_stage_l = 0;
            unsigned int tile_idx_l = blockIdx.x;
            unsigned int total_tiles_l = BATCH_SIZE * NUM_KV_HEADS * ((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits);
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < total_tiles_l; _tile_iter_l++) {
                mbarrier_wait(throttle_empty_addr + (throttle_stage_l) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage_l) * 8);
                throttle_stage_l += 1;
                if (throttle_stage_l == 2) { throttle_stage_l = 0; _phase_throttle_empty ^= 1; }
                int split_l = tile_idx_l % (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int bh_l = tile_idx_l / (unsigned int)(((NUM_SPLITS != 0) ? NUM_SPLITS : num_splits));
                int batch_idx_3 = bh_l / NUM_KV_HEADS;
                int kv_head_idx_1 = bh_l % NUM_KV_HEADS;
                int seqlen_kv_3 = seq_lens_kv[batch_idx_3];
                int num_n_blocks_total_3 = (seqlen_kv_3 + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_3 = num_n_blocks_total_3 + num_n_blocks_total_3 % 2;
                int split_start_block_3 = split_l * ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split);
                int rem_blocks_3 = even_n_blocks_3 - split_start_block_3;
                int capped_blocks_3 = ((rem_blocks_3 > ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split)) ? ((BLOCKS_PER_SPLIT != 0) ? BLOCKS_PER_SPLIT : blocks_per_split) : rem_blocks_3);
                int cta_n_blocks_3 = ((capped_blocks_3 < 2) ? 2 : capped_blocks_3);
                mbarrier_wait(q_empty_addr + (q_slot_l) * 8, q_phase_l);
                if (elect_sync()) {
                    int off_qt = (batch_idx_3 * NUM_KV_HEADS + kv_head_idx_1) * TILE_Q;
                    mbarrier_arrive_expect_tx(q_full_addr + (q_slot_l) * 8, TILE_Q * HEAD_DIM);
                    tma_3d_gmem2smem(smem_qt_addr + (unsigned int)(q_slot_l * 1024), Qt, 0, off_qt, 0, q_full_addr + (q_slot_l) * 8);
                    int pg_tile_base_l = pg_slot_l;
                    int prefill = ((cta_n_blocks_3 < 2) ? cta_n_blocks_3 : 2);
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        mbarrier_wait(kv_empty_addr + (kv_slot_l) * 8, kv_phase_l);
                        if (ni % 4 == 0) {
                            mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, pg_phase_l);
                            pg_slot_l += 1;
                            if (pg_slot_l == 6) { pg_slot_l = 0; pg_phase_l ^= 1; }
                        }
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_slot_l) * 8, 16384);
                        int ldst = smem_kv_addr + (unsigned int)(kv_slot_l * 16384);
                        int pg_k[8];
                        int pg_k_stage = (pg_tile_base_l + ni / 4) % 6;
                        int pg_k_base = smem_pg_addr + (unsigned int)(pg_k_stage * 256) + (unsigned int)(ni % 4 * 8 * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 3]))
                            : "r"(pg_k_base));
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 8; pg_i++) {
                            int pg0 = pg_k[pg_i / (PAGE_SIZE / 16)];
                            int toff = pg_i * 2048;
                            tma_5d_gmem2smem(ldst + toff, K, 0, pg_i % (PAGE_SIZE / 16) * 16, 0, kv_head_idx_1, pg0, kv_full_addr + (kv_slot_l) * 8);
                        }
                        kv_slot_l += 1;
                        if (kv_slot_l == 9) { kv_slot_l = 0; kv_phase_l ^= 1; }
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < cta_n_blocks_3; ni_1++) {
                        int pg_v_stage = (pg_tile_base_l + ni_1 / 4) % 6;
                        int next_ni = ni_1 + 2;
                        if (next_ni < cta_n_blocks_3) {
                            mbarrier_wait(kv_empty_addr + (kv_slot_l) * 8, kv_phase_l);
                            if (next_ni % 4 == 0) {
                                mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, pg_phase_l);
                                pg_slot_l += 1;
                                if (pg_slot_l == 6) { pg_slot_l = 0; pg_phase_l ^= 1; }
                            }
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_slot_l) * 8, 16384);
                            int kdst = smem_kv_addr + (unsigned int)(kv_slot_l * 16384);
                            int pg_nk[8];
                            int pg_nk_stage = (pg_tile_base_l + next_ni / 4) % 6;
                            int pg_nk_base = smem_pg_addr + (unsigned int)(pg_nk_stage * 256) + (unsigned int)(next_ni % 4 * 8 * 4);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 3]))
                                : "r"(pg_nk_base));
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                                int npg0 = pg_nk[pg_i_1 / (PAGE_SIZE / 16)];
                                int ntoff = pg_i_1 * 2048;
                                tma_5d_gmem2smem(kdst + ntoff, K, 0, pg_i_1 % (PAGE_SIZE / 16) * 16, 0, kv_head_idx_1, npg0, kv_full_addr + (kv_slot_l) * 8);
                            }
                            kv_slot_l += 1;
                            if (kv_slot_l == 9) { kv_slot_l = 0; kv_phase_l ^= 1; }
                        }
                        int pg_v[8];
                        int pg_v_base = smem_pg_addr + (unsigned int)(pg_v_stage * 256) + (unsigned int)((32 + ni_1 % 4 * 8) * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 3]))
                            : "r"(pg_v_base));
                        int pg_group_done = ((ni_1 % 4 == 3) ? 1 : 0);
                        pg_group_done = pg_group_done | ((ni_1 + 1 == cta_n_blocks_3) ? 1 : 0);
                        if (pg_group_done != 0) {
                            mbarrier_arrive(pg_empty_addr + (pg_v_stage) * 8);
                        }
                        mbarrier_wait(kv_empty_addr + (kv_slot_l) * 8, kv_phase_l);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_slot_l) * 8, 16384);
                        int vdst = smem_kv_addr + (unsigned int)(kv_slot_l * 16384);
                        #pragma unroll
                        for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                            int vpg0 = pg_v[pg_i_2 / (PAGE_SIZE / 16)];
                            int vtoff = pg_i_2 * 2048;
                            tma_5d_gmem2smem(vdst + vtoff, V, 0, pg_i_2 % (PAGE_SIZE / 16) * 16, 0, kv_head_idx_1, vpg0, kv_full_addr + (kv_slot_l) * 8);
                        }
                        kv_slot_l += 1;
                        if (kv_slot_l == 9) { kv_slot_l = 0; kv_phase_l ^= 1; }
                    }
                }
                q_slot_l += 1;
                if (q_slot_l == 2) { q_slot_l = 0; q_phase_l ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_5);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_l * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_l * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 2) { work_stage_l = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                tile_idx_l = _clc_ctaid_2;
            }
        }
    }

    // Cleanup
}

} // extern "C"
