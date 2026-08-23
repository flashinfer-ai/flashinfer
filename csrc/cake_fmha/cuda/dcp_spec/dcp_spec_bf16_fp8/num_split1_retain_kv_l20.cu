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
#define SMEM_SMEM_Q_RAW_OFF 1664
#define SMEM_SMEM_Q_RAW_STAGE_BYTES 2048
#define SMEM_SMEM_Q_RAW_STRIDE 2048
#define SMEM_SMEM_QT_OFF 71680
#define SMEM_SMEM_QT_STAGE_BYTES 1024
#define SMEM_SMEM_QT_STRIDE 1024
#define SMEM_SMEM_KV_FP8_OFF 6144
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_KV_TRANSFORM_SRC_OFF 6144
#define SMEM_SMEM_KV_TRANSFORM_SRC_STAGE_BYTES 16384
#define SMEM_SMEM_KV_TRANSFORM_SRC_STRIDE 16384
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_KV_TRANSFORM_DST_OFF 71680
#define SMEM_SMEM_KV_TRANSFORM_DST_STAGE_BYTES 32768
#define SMEM_SMEM_KV_TRANSFORM_DST_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P0_OFF 72704
#define SMEM_SMEM_P0_STAGE_BYTES 1024
#define SMEM_SMEM_P0_STRIDE 1024
#define SMEM_SMEM_P1_OFF 73728
#define SMEM_SMEM_P1_STAGE_BYTES 1024
#define SMEM_SMEM_P1_STRIDE 1024
#define SMEM_SPLIT_REDUCE_FLAG_OFF 74752
#define SMEM_SPLIT_REDUCE_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_REDUCE_FLAG_STRIDE 4
#define SMEM_SPLIT_WEIGHT0_OFF 74756
#define SMEM_SPLIT_WEIGHT0_STAGE_BYTES 32
#define SMEM_SPLIT_WEIGHT0_STRIDE 32
#define SMEM_SPLIT_WEIGHT1_OFF 74788
#define SMEM_SPLIT_WEIGHT1_STAGE_BYTES 32
#define SMEM_SPLIT_WEIGHT1_STRIDE 32
#define SMEM_SPLIT_MERGED_LSE_OFF 74820
#define SMEM_SPLIT_MERGED_LSE_STAGE_BYTES 32
#define SMEM_SPLIT_MERGED_LSE_STRIDE 32
#define SMEM_SPLIT_WEIGHTS_OFF 74756
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 128
#define SMEM_SPLIT_WEIGHTS_STRIDE 128
#define SMEM_TOTAL 208896
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 64
#define NUM_KV_STAGES 4
#ifndef BATCH_SIZE
#define BATCH_SIZE 128
#endif
#ifndef NUM_Q_HEADS
#define NUM_Q_HEADS 64
#endif
#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 8
#endif
#ifndef Q_LEN
#define Q_LEN 1
#endif
#ifndef CP_WORLD
#define CP_WORLD 1
#endif
#define NUM_SPLIT 1
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
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
kernel_cake_fmha_dcp_spec_bf16_fp8(CakeFmhaTensorMap const* Qt, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* __restrict__ partial_O_ptr, float* __restrict__ partial_LSE_ptr, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ split_completion, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ causal_seqlens_kv_global, int max_pages_per_seq, int max_local_seq_len, float softmax_scale_log2, float output_scale, int cp_rank, int num_q_heads, int num_kv_heads, int batch_size)
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
    __nv_bfloat16* smem_q_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_q_raw_addr = smem + 1664;
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 71680);
    const int smem_qt_addr = smem + 71680;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int smem_kv_fp8_addr = smem + 6144;
    uint8_t* smem_kv_transform_src = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int smem_kv_transform_src_addr = smem + 6144;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_kv_transform_dst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 71680);
    const int smem_kv_transform_dst_addr = smem + 71680;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    uint8_t* smem_p0 = reinterpret_cast<uint8_t*>(smem_raw + 72704);
    const int smem_p0_addr = smem + 72704;
    uint8_t* smem_p1 = reinterpret_cast<uint8_t*>(smem_raw + 73728);
    const int smem_p1_addr = smem + 73728;
    int* split_reduce_flag = reinterpret_cast<int*>(smem_raw + 74752);
    const int split_reduce_flag_addr = smem + 74752;
    float* split_weight0 = reinterpret_cast<float*>(smem_raw + 74756);
    const int split_weight0_addr = smem + 74756;
    float* split_weight1 = reinterpret_cast<float*>(smem_raw + 74788);
    const int split_weight1_addr = smem + 74788;
    float* split_merged_lse = reinterpret_cast<float*>(smem_raw + 74820);
    const int split_merged_lse_addr = smem + 74820;
    float* split_weights = reinterpret_cast<float*>(smem_raw + 74756);
    const int split_weights_addr = smem + 74756;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory"); }

    // Mbarrier init (20 groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_tma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_full: 1 barriers, init_count=4
            mbarrier_init(smem + 8, 4);
            // kv_transform_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // p_full_0: 1 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            // p_full_1: 1 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            // corr_scale_0: 1 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            // corr_scale_1: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // corr_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            // corr_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 184, 4);
            // o_ready_0: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // o_ready_1: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // o_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // o_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 216, 128);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_tma_full_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 8)
    #define kv_transform_full_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 48)
    #define q_empty_addr (mbar_base + 80)
    #define kv_empty_addr (mbar_base + 88)
    #define s_full_0_addr (mbar_base + 120)
    #define s_full_1_addr (mbar_base + 128)
    #define p_full_0_addr (mbar_base + 136)
    #define p_full_1_addr (mbar_base + 144)
    #define corr_scale_0_addr (mbar_base + 152)
    #define corr_scale_1_addr (mbar_base + 160)
    #define corr_empty_0_addr (mbar_base + 168)
    #define corr_empty_1_addr (mbar_base + 176)
    #define stats_empty_addr (mbar_base + 184)
    #define o_ready_0_addr (mbar_base + 192)
    #define o_ready_1_addr (mbar_base + 200)
    #define o_empty_0_addr (mbar_base + 208)
    #define o_empty_1_addr (mbar_base + 216)
    #define tmem_dealloc_addr (mbar_base + 224)
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
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            const int tmem_row_base_v = warp % 4 * 32;
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
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
                int visible_local_keys = seq_lens_kv[batch_idx];
                {
                    int last_global_position = causal_seqlens_kv_global[batch_idx] + q_row_idx;
                    visible_local_keys = 0;
                    if (last_global_position >= cp_rank) {
                        visible_local_keys = (last_global_position - cp_rank) / CP_WORLD + 1;
                    }
                }
                int loop_seq_len = seq_lens_kv[batch_idx];
                {
                    loop_seq_len = max_local_seq_len;
                }
                int num_n_blocks_total = (loop_seq_len + BLOCK_N - 1) / BLOCK_N;
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
                float row_max_pair0[2];
                float row_sum_pair0[2];
                float row_max_pair1[2];
                float row_sum_pair1[2];
                row_max_pair0[0] = -LOOM_INF;
                row_max_pair0[1] = -LOOM_INF;
                row_sum_pair0[0] = 0.0f;
                row_sum_pair0[1] = 0.0f;
                row_max_pair1[0] = -LOOM_INF;
                row_max_pair1[1] = -LOOM_INF;
                row_sum_pair1[0] = 0.0f;
                row_sum_pair1[1] = 0.0f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    smem_exch0_u32[wg_tid] = _amf_enc_0;
                    smem_exch1_u32[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                #pragma unroll 1
                for (int pair = 0; pair < num_pairs; pair++) {
                    #pragma unroll
                    for (int inst_s = 0; inst_s < 2; inst_s++) {
                        int is_wg1 = inst_s;
                        int my_tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 8 : 0));
                        int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
                        float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
                        unsigned int* my_exch_u32_ptr = ((is_wg1 != 0) ? smem_exch1_u32 : smem_exch0_u32);
                        if (is_wg1 != 0) {
                            mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                            _phase_s_full_1_0 ^= 1;
                        } else {
                            mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                            _phase_s_full_0_0 ^= 1;
                        }
                        float sv[8];
                        float sv_lo[4];
                        float sv_hi[4];
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
                        int my_block = split_start_pair * 2 + num_pairs * 2 - 1 - 2 * pair - is_wg1;
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
                            old_max_pair[c_2] = ((is_wg1 != 0) ? row_max_pair1[c_2] : row_max_pair0[c_2]);
                            float _max_8 = max_noftz(old_max_pair[c_2], pair_max[c_2]);
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
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 2; c_3++) {
                            float delta = softmax_scale_log2 * (old_max_pair[c_3] - new_max_pair[c_3]);
                            float _exp2_0 = approx_exp2(delta);
                            acc_scale_pair[c_3] = ((old_max_pair[c_3] > -LOOM_INF) ? _exp2_0 : 1.0f);
                        }
                        if (is_wg1 != 0) {
                            mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                            _phase_corr_empty_1_0 ^= 1;
                        } else {
                            mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                            _phase_corr_empty_0_0 ^= 1;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x2.b32"
                            " [%0], {%1, %2};"
                            :: "r"(my_tmem_stats), "f"(acc_scale_pair[0]), "f"(acc_scale_pair[1])
                            : "memory");
                        float exp_vals[8];
                        float p_log2_bias = ((1) ? 8.8073549f : 0.0f);
                        #pragma unroll
                        for (int c_4 = 0; c_4 < 8; c_4++) {
                            float safe_max = ((new_max_pair[c_4 % 2] == -LOOM_INF) ? 0.0f : new_max_pair[c_4 % 2]);
                            float max_scaled = safe_max * softmax_scale_log2;
                            float _exp2_1 = approx_exp2(sv[c_4] * softmax_scale_log2 - max_scaled + p_log2_bias);
                            exp_vals[c_4] = _exp2_1;
                        }
                        #pragma unroll
                        for (int c_5 = 0; c_5 < 2; c_5++) {
                            if (is_wg1 != 0) {
                                row_max_pair1[c_5] = new_max_pair[c_5];
                            } else {
                                row_max_pair0[c_5] = new_max_pair[c_5];
                            }
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (is_wg1 != 0) {
                            mbarrier_arrive(corr_scale_1_addr);
                        } else {
                            mbarrier_arrive(corr_scale_0_addr);
                        }
                        float pair_sum[2];
                        pair_sum[0] = exp_vals[0] + exp_vals[2] + exp_vals[4] + exp_vals[6];
                        pair_sum[1] = exp_vals[1] + exp_vals[3] + exp_vals[5] + exp_vals[7];
                        #pragma unroll
                        for (int c_6 = 0; c_6 < 2; c_6++) {
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_6], 16);
                            pair_sum[c_6] = pair_sum[c_6] + _shfl_xor_2;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_6], 8);
                            pair_sum[c_6] = pair_sum[c_6] + _shfl_xor_3;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_6], 4);
                            pair_sum[c_6] = pair_sum[c_6] + _shfl_xor_4;
                        }
                        #pragma unroll
                        for (int c_7 = 0; c_7 < 2; c_7++) {
                            float new_sum = ((is_wg1 != 0) ? row_sum_pair1[c_7] : row_sum_pair0[c_7]);
                            new_sum = new_sum * acc_scale_pair[c_7] + pair_sum[c_7];
                            if (is_wg1 != 0) {
                                row_sum_pair1[c_7] = new_sum;
                            } else {
                                row_sum_pair0[c_7] = new_sum;
                            }
                        }
                        int mtx_idx = lane / 8;
                        int thr_row_idx = lane % 8;
                        {
                            unsigned int regs_p_fp8[2];
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
                                regs_p_fp8[0] = _packed;
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
                                regs_p_fp8[1] = _packed;
                            }
                            int seg_col_idx_fp8 = warp_in_wg * 2 + mtx_idx ^ thr_row_idx;
                            int stsm_offset_fp8 = thr_row_idx * 128 + seg_col_idx_fp8 * 16;
                            const void* _stmatrix_b8_ptr_5 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(((is_wg1 != 0) ? smem_p1 : smem_p0)) + stsm_offset_fp8);
                            uint64_t _stmatrix_b8_addr64_5;
                            asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_5) : "l"(_stmatrix_b8_ptr_5));
                            uint32_t _stmatrix_b8_addr_5;
                            asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_5) : "l"(_stmatrix_b8_addr64_5));
                            asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_b8_addr_5), "r"(regs_p_fp8[0]), "r"(regs_p_fp8[1])
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (is_wg1 != 0) {
                            mbarrier_arrive(p_full_1_addr);
                        } else {
                            mbarrier_arrive(p_full_0_addr);
                        }
                    }
                }
                #pragma unroll
                for (int inst_s_1 = 0; inst_s_1 < 2; inst_s_1++) {
                    int is_wg1_1 = inst_s_1;
                    float* my_exch_ptr_1 = ((is_wg1_1 != 0) ? smem_exch1 : smem_exch0);
                    float* my_corr_ptr = ((is_wg1_1 != 0) ? smem_corr1 : smem_corr0);
                    if (is_wg1_1 != 0) {
                        mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                        _phase_corr_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                        _phase_corr_empty_0_0 ^= 1;
                    }
                    if (lane < 4) {
                        float sum0 = ((is_wg1_1 != 0) ? row_sum_pair1[0] : row_sum_pair0[0]);
                        float sum1 = ((is_wg1_1 != 0) ? row_sum_pair1[1] : row_sum_pair0[1]);
                        my_exch_ptr_1[warp_in_wg * 8 + col_pair_base] = sum0;
                        my_exch_ptr_1[warp_in_wg * 8 + col_pair_base + 1] = sum1;
                    }
                    if (is_wg1_1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    if (wg_tid < 4) {
                        my_corr_ptr[col_pair_base] = my_exch_ptr_1[col_pair_base] + my_exch_ptr_1[8 + col_pair_base] + my_exch_ptr_1[16 + col_pair_base] + my_exch_ptr_1[24 + col_pair_base];
                        my_corr_ptr[col_pair_base + 1] = my_exch_ptr_1[col_pair_base + 1] + my_exch_ptr_1[8 + col_pair_base + 1] + my_exch_ptr_1[16 + col_pair_base + 1] + my_exch_ptr_1[24 + col_pair_base + 1];
                        float max0 = ((is_wg1_1 != 0) ? row_max_pair1[0] : row_max_pair0[0]);
                        float max1 = ((is_wg1_1 != 0) ? row_max_pair1[1] : row_max_pair0[1]);
                        my_exch_ptr_1[col_pair_base] = max0;
                        my_exch_ptr_1[col_pair_base + 1] = max1;
                    }
                    if (is_wg1_1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    if (is_wg1_1 != 0) {
                        mbarrier_arrive(corr_scale_1_addr);
                    } else {
                        mbarrier_arrive(corr_scale_0_addr);
                    }
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int warp_in_wg_c = warp % 4;
            const int wg_tid_c = warp_in_wg_c * 32 + lane;
            int d_idx = warp % 4 * 32 + lane;
            const int group_ratio_rt = NUM_Q_HEADS / NUM_KV_HEADS;
            unsigned int total_tiles_c = BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT;
            unsigned int _phase_corr_scale_0_0 = 0;
            unsigned int _phase_corr_scale_1_0 = 0;
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
                int visible_local_keys_1 = seq_lens_kv[batch_idx_1];
                {
                    int last_global_position_1 = causal_seqlens_kv_global[batch_idx_1] + q_row_idx_1;
                    visible_local_keys_1 = 0;
                    if (last_global_position_1 >= cp_rank) {
                        visible_local_keys_1 = (last_global_position_1 - cp_rank) / CP_WORLD + 1;
                    }
                }
                int loop_seq_len_1 = seq_lens_kv[batch_idx_1];
                {
                    loop_seq_len_1 = max_local_seq_len;
                }
                int num_n_blocks_total_1 = (loop_seq_len_1 + BLOCK_N - 1) / BLOCK_N;
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
                if (num_pairs_1 > 0) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < num_pairs_1; pair_1++) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[2];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                        " {%0, %1}, [%2];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1])
                        : "r"(taddr + 16 + (unsigned int)corr_row)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                    _phase_o_ready_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_0 = ((_tmem_load_0[0] != 1.0f) ? 1 : 0);
                    rescale_pred_0 = rescale_pred_0 | ((_tmem_load_0[1] != 1.0f) ? 1 : 0);
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
                            o0[h_1] = o0[h_1] * _tmem_load_0[h_1 % 2];
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
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[2];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                        " {%0, %1}, [%2];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1])
                        : "r"(taddr + 48 + (unsigned int)corr_row)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                    _phase_o_ready_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_1 = ((_tmem_load_1[0] != 1.0f) ? 1 : 0);
                    rescale_pred_1 = rescale_pred_1 | ((_tmem_load_1[1] != 1.0f) ? 1 : 0);
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
                            o1[h_4] = o1[h_4] * _tmem_load_1[h_4 % 2];
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
                mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                _phase_corr_scale_0_0 ^= 1;
                mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                _phase_corr_scale_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float owner_scale0 = 0.0f;
                float owner_scale1 = 0.0f;
                float owner_inv_sum = 0.0f;
                float owner_lse = -LOOM_INF;
                int owner_valid = 0;
                if (group_ratio_rt > lane) {
                    float owner_m0 = smem_exch0[lane];
                    float owner_m1 = smem_exch1[lane];
                    float owner_s0 = smem_corr0[lane];
                    float owner_s1 = smem_corr1[lane];
                    float _max_9 = max_noftz(owner_m0, owner_m1);
                    float owner_max = _max_9;
                    float owner_d0 = softmax_scale_log2 * (owner_m0 - owner_max);
                    float owner_d1 = softmax_scale_log2 * (owner_m1 - owner_max);
                    float _exp2_2 = approx_exp2(owner_d0);
                    owner_scale0 = ((owner_m0 == -LOOM_INF) ? 0.0f : _exp2_2);
                    float _exp2_3 = approx_exp2(owner_d1);
                    owner_scale1 = ((owner_m1 == -LOOM_INF) ? 0.0f : _exp2_3);
                    float owner_sum = owner_s0 * owner_scale0 + owner_s1 * owner_scale1;
                    if (owner_sum > 0.0f && owner_sum == owner_sum) {
                        float _rcp_0 = approx_rcp(owner_sum);
                        owner_inv_sum = _rcp_0;
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(owner_sum));
                        owner_lse = owner_max * softmax_scale_log2 + _log2_0 - ((1) ? 8.8073549f : 0.0f);
                        owner_valid = 1;
                    }
                }
                {
                    if (warp_in_wg_c == 0 && group_ratio_rt > lane) {
                        int lse_q_head = kv_head_idx_1 * group_ratio_rt + lane;
                        {
                            int lse_idx = (batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + lse_q_head;
                            *(reinterpret_cast<float*>(LSE_ptr + lse_idx) + (0)) = owner_lse;
                        }
                    }
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
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, owner_scale0, h_6);
                    float head_scale0 = _shfl_0;
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, owner_scale1, h_6);
                    float head_scale1 = _shfl_1;
                    float _shfl_2 = __shfl_sync(0xFFFFFFFF, owner_inv_sum, h_6);
                    float head_inv_sum = _shfl_2;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, owner_valid, h_6);
                    int head_valid = _shfl_3;
                    float final_o = 0.0f;
                    if (head_valid != 0) {
                        float merged = _tmem_load_2[h_6] * head_scale0 + _tmem_load_3[h_6] * head_scale1;
                        final_o = merged * head_inv_sum * output_scale;
                    }
                    if (group_ratio_rt > h_6) {
                        int q_head = kv_head_idx_1 * group_ratio_rt + h_6;
                        {
                            int o_idx = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head) * HEAD_DIM + d_idx;
                            *(reinterpret_cast<__nv_bfloat16*>(O_ptr + o_idx) + (0)) = __float2bfloat16_rn(final_o);
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
    if (warp == 8) {
        { // mma_warp_main
            const int tmem_s0v = taddr;
            const int tmem_s1v = taddr + 8;
            const int tmem_o0v = taddr + 80;
            const int tmem_o1v = taddr + 88;
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
                int visible_local_keys_2 = seq_lens_kv[batch_idx_2];
                {
                    int last_global_position_2 = causal_seqlens_kv_global[batch_idx_2] + q_row_idx_2;
                    visible_local_keys_2 = 0;
                    if (last_global_position_2 >= cp_rank) {
                        visible_local_keys_2 = (last_global_position_2 - cp_rank) / CP_WORLD + 1;
                    }
                }
                int loop_seq_len_2 = seq_lens_kv[batch_idx_2];
                {
                    loop_seq_len_2 = max_local_seq_len;
                }
                int num_n_blocks_total_2 = (loop_seq_len_2 + BLOCK_N - 1) / BLOCK_N;
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
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 1024);
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
                elect_commit(s_full_0_addr);
                elect_commit(kv_empty_addr);
                mbarrier_wait(kv_full_addr + 8, 0);
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 1024);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_0), "r"(tmem_tmem_s1), "r"(0));
                elect_commit(s_full_1_addr);
                elect_commit(kv_empty_addr + 8);
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < num_pairs_2 - 1; pair_2++) {
                    int s0 = inst0_stage;
                    int s1 = (inst0_stage + 1) % 4;
                    int s0_next = (inst0_stage + 2) % 4;
                    int s1_next = (inst0_stage + 3) % 4;
                    mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                    _phase_o_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 1024);
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (s0) * 8, o_ready_0_addr);
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 1024);
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_s0), "r"(0));
                    elect_commit(s_full_0_addr);
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                    _phase_o_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 1024);
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (s1) * 8, o_ready_1_addr);
                    mbarrier_wait(kv_full_addr + (s1_next) * 8, 0);
                    int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_next) * 1024);
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
                int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 1024);
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
                elect_commit2(kv_empty_addr + (s0_last) * 8, o_ready_0_addr);
                mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                _phase_o_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 1024);
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
                elect_commit2(kv_empty_addr + (s1_last) * 8, o_ready_1_addr);
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 9) {
        // idle — no tasks assigned
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
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
                int visible_local_keys_3 = seq_lens_kv[batch_idx_3];
                {
                    int last_global_position_3 = causal_seqlens_kv_global[batch_idx_3] + q_row_idx_3;
                    visible_local_keys_3 = 0;
                    if (last_global_position_3 >= cp_rank) {
                        visible_local_keys_3 = (last_global_position_3 - cp_rank) / CP_WORLD + 1;
                    }
                }
                int seqlen_kv = seq_lens_kv[batch_idx_3];
                int loop_seq_len_3 = seq_lens_kv[batch_idx_3];
                {
                    loop_seq_len_3 = max_local_seq_len;
                }
                int num_n_blocks_total_3 = (loop_seq_len_3 + BLOCK_N - 1) / BLOCK_N;
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
                    int off_qt = ((batch_idx_3 * Q_LEN + q_row_idx_3) * NUM_KV_HEADS + kv_head_idx_3) * TILE_Q;
                    {
                        const int group_ratio_l = NUM_Q_HEADS / NUM_KV_HEADS;
                        off_qt = (batch_idx_3 * Q_LEN + q_row_idx_3) * NUM_Q_HEADS + kv_head_idx_3 * group_ratio_l;
                    }
                    mbarrier_arrive_expect_tx(q_tma_full_addr, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_q_raw_addr, Qt, 0, off_qt, 0, q_tma_full_addr);
                    int pt_base = batch_idx_3 * max_pages_per_seq;
                    int kv_stage = 0;
                    int kv_phase = 1;
                    int prefill = ((num_pairs_3 * 2 < 2) ? num_pairs_3 * 2 : 2);
                    int max_pg = pages_per_seq_v - 1;
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        int n_block = split_start_pair_3 * 2 + num_pairs_3 * 2 - 1 - ni;
                        int pg_base = n_block * 2;
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                        int ldst = smem_kv_fp8_addr + (unsigned int)(kv_stage * 16384);
                        int pb_k = pg_base;
                        int pg_k[2];
                        {
                            {
                                pg_k[0] = *reinterpret_cast<const int*>(page_table + pt_base + pb_k);
                            }
                            {
                                pg_k[1] = *reinterpret_cast<const int*>(page_table + pt_base + pb_k + 1);
                            }
                        }
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 2; pg_i++) {
                            int toff = pg_i * PAGE_SIZE * HEAD_DIM;
                            {
                                {
                                    tma_4d_gmem2smem(ldst + toff, K, 0, 0, kv_head_idx_3, pg_k[pg_i], kv_full_addr + (kv_stage) * 8);
                                }
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < num_pairs_3 * 2; ni_1++) {
                        int stage = ni_1 % 4;
                        int n_block_1 = split_start_pair_3 * 2 + num_pairs_3 * 2 - 1 - ni_1;
                        int vpg_base = n_block_1 * 2;
                        mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 16384);
                        int vdst = smem_kv_fp8_addr + (unsigned int)(stage * 16384);
                        int pb_v = vpg_base;
                        int pg_v[2];
                        {
                            {
                                pg_v[0] = *reinterpret_cast<const int*>(page_table + pt_base + pb_v);
                            }
                            {
                                pg_v[1] = *reinterpret_cast<const int*>(page_table + pt_base + pb_v + 1);
                            }
                        }
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 2; pg_i_1++) {
                            int vtoff = pg_i_1 * PAGE_SIZE * HEAD_DIM;
                            {
                                {
                                    tma_4d_gmem2smem(vdst + vtoff, V, 0, 0, kv_head_idx_3, pg_v[pg_i_1], kv_full_addr + (stage) * 8);
                                }
                            }
                        }
                        int next_ni = ni_1 + 2;
                        if (next_ni < num_pairs_3 * 2) {
                            int k_stage = next_ni % NUM_KV_STAGES;
                            int next_n = split_start_pair_3 * 2 + num_pairs_3 * 2 - 1 - next_ni;
                            int npg_base = next_n * 2;
                            mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 16384);
                            int kdst = smem_kv_fp8_addr + (unsigned int)(k_stage * 16384);
                            int pb_nk = npg_base;
                            int pg_nk[2];
                            {
                                {
                                    pg_nk[0] = *reinterpret_cast<const int*>(page_table + pt_base + pb_nk);
                                }
                                {
                                    pg_nk[1] = *reinterpret_cast<const int*>(page_table + pt_base + pb_nk + 1);
                                }
                            }
                            #pragma unroll
                            for (int pg_i_2 = 0; pg_i_2 < 2; pg_i_2++) {
                                int ntoff = pg_i_2 * PAGE_SIZE * HEAD_DIM;
                                {
                                    {
                                        tma_4d_gmem2smem(kdst + ntoff, K, 0, 0, kv_head_idx_3, pg_nk[pg_i_2], kv_full_addr + (k_stage) * 8);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // transform_main
            unsigned int direct_tiles_t = ((1) ? BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT : 0);
            int q_xform_tid = tid - 384;
            unsigned int _phase_q_tma_full_0 = 0;
            #pragma unroll 1
            for (unsigned int direct_tile_idx_t = blockIdx.x; direct_tile_idx_t < direct_tiles_t; direct_tile_idx_t += gridDim.x) {
                mbarrier_wait(q_tma_full_addr, _phase_q_tma_full_0);
                _phase_q_tma_full_0 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                #pragma unroll
                for (int q_xform_row = 0; q_xform_row < TILE_Q; q_xform_row++) {
                    float q_value = smem_q_raw[q_xform_row * HEAD_DIM + q_xform_tid];
                    {
                        uint16_t _fp8_pair_4010709856;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                            : "=h"(_fp8_pair_4010709856) : "f"(0.0f), "f"(q_value));
                        uint32_t _byte_4010709856 = (uint32_t)(_fp8_pair_4010709856 & 0xFF);
                        uint32_t _addr_4010709856 = static_cast<uint32_t>((smem_qt_addr + (unsigned int)(q_xform_row * 128 + q_xform_tid ^ (q_xform_row * 128 + q_xform_tid >> 7 & 7) << 4)));
                        asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_4010709856), "r"(_byte_4010709856) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(q_full_addr);
                }
            }
            unsigned int total_tiles_t = ((1) ? 0 : BATCH_SIZE * Q_LEN * NUM_KV_HEADS * NUM_SPLIT);
            #pragma unroll 1
            for (unsigned int tile_idx_t = blockIdx.x; tile_idx_t < total_tiles_t; tile_idx_t += gridDim.x) {
                unsigned int tile_idx_4 = tile_idx_t / (unsigned int)NUM_SPLIT;
                int split_idx_4 = tile_idx_t % (unsigned int)NUM_SPLIT;
                int kv_head_idx_4 = tile_idx_4 % (unsigned int)NUM_KV_HEADS;
                int request_row_4 = tile_idx_4 / (unsigned int)NUM_KV_HEADS;
                int q_row_idx_4 = request_row_4 % Q_LEN;
                int batch_idx_4 = request_row_4 / Q_LEN;
                int visible_local_keys_4 = seq_lens_kv[batch_idx_4];
                {
                    int last_global_position_4 = causal_seqlens_kv_global[batch_idx_4] + q_row_idx_4;
                    visible_local_keys_4 = 0;
                    if (last_global_position_4 >= cp_rank) {
                        visible_local_keys_4 = (last_global_position_4 - cp_rank) / CP_WORLD + 1;
                    }
                }
                int loop_seq_len_4 = seq_lens_kv[batch_idx_4];
                {
                    loop_seq_len_4 = max_local_seq_len;
                }
                int num_n_blocks_total_4 = (loop_seq_len_4 + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_total_4 < 1) {
                    num_n_blocks_total_4 = 1;
                }
                int total_pairs_4 = (num_n_blocks_total_4 + 1) / 2;
                int base_pairs_4 = total_pairs_4 / NUM_SPLIT;
                int extra_pairs_4 = total_pairs_4 % NUM_SPLIT;
                int num_pairs_4 = base_pairs_4;
                int split_start_pair_4 = extra_pairs_4 * (base_pairs_4 + 1) + (split_idx_4 - extra_pairs_4) * base_pairs_4;
                if (split_idx_4 < extra_pairs_4) {
                    num_pairs_4 = base_pairs_4 + 1;
                    split_start_pair_4 = split_idx_4 * (base_pairs_4 + 1);
                }
                int inst0_stage_t = 0;
                mbarrier_wait(kv_full_addr, 0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr - smem);
                    const int _tid = (int)threadIdx.x - (12) * 32;
                    uint64_t _src_buf[16];
                    #pragma unroll
                    for (int _outer = 0; _outer < 2; ++_outer) {
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        }
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            uint64_t _src64 = _src_buf[_base >> 7];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_full_addr);
                }
                mbarrier_wait(kv_full_addr + 8, 0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + 16384 - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + 32768 - smem);
                    const int _tid = (int)threadIdx.x - (12) * 32;
                    uint64_t _src_buf[16];
                    #pragma unroll
                    for (int _outer = 0; _outer < 2; ++_outer) {
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        }
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            uint64_t _src64 = _src_buf[_base >> 7];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_full_addr + 8);
                }
                #pragma unroll 1
                for (int pair_3 = 0; pair_3 < num_pairs_4 - 1; pair_3++) {
                    int s0_1 = inst0_stage_t;
                    int s1_1 = (inst0_stage_t + 1) % NUM_KV_STAGES;
                    int s0_next_1 = (inst0_stage_t + 2) % NUM_KV_STAGES;
                    int s1_next_1 = (inst0_stage_t + 3) % NUM_KV_STAGES;
                    mbarrier_wait(kv_full_addr + (s0_1) * 8, 1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + (unsigned int)(s0_1 * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + (unsigned int)(s0_1 * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (12) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[_base >> 7];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (s0_1) * 8);
                    }
                    mbarrier_wait(kv_full_addr + (s0_next_1) * 8, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + (unsigned int)(s0_next_1 * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + (unsigned int)(s0_next_1 * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (12) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[_base >> 7];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (s0_next_1) * 8);
                    }
                    mbarrier_wait(kv_full_addr + (s1_1) * 8, 1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + (unsigned int)(s1_1 * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + (unsigned int)(s1_1 * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (12) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[_base >> 7];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (s1_1) * 8);
                    }
                    mbarrier_wait(kv_full_addr + (s1_next_1) * 8, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + (unsigned int)(s1_next_1 * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + (unsigned int)(s1_next_1 * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (12) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[_base >> 7];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (s1_next_1) * 8);
                    }
                    inst0_stage_t = s0_next_1;
                }
                int s0_last_t = inst0_stage_t;
                int s1_last_t = (inst0_stage_t + 1) % NUM_KV_STAGES;
                mbarrier_wait(kv_full_addr + (s0_last_t) * 8, 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + (unsigned int)(s0_last_t * 16384) - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + (unsigned int)(s0_last_t * 32768) - smem);
                    const int _tid = (int)threadIdx.x - (12) * 32;
                    uint64_t _src_buf[16];
                    #pragma unroll
                    for (int _outer = 0; _outer < 2; ++_outer) {
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        }
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            uint64_t _src64 = _src_buf[_base >> 7];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_full_addr + (s0_last_t) * 8);
                }
                mbarrier_wait(kv_full_addr + (s1_last_t) * 8, 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    const char* _src_ptr = smem_raw + (smem_kv_transform_src_addr + (unsigned int)(s1_last_t * 16384) - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_transform_dst_addr + (unsigned int)(s1_last_t * 32768) - smem);
                    const int _tid = (int)threadIdx.x - (12) * 32;
                    uint64_t _src_buf[16];
                    #pragma unroll
                    for (int _outer = 0; _outer < 2; ++_outer) {
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        }
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                            int _off = _base + _tid;
                            uint64_t _src64 = _src_buf[_base >> 7];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(kv_full_addr + (s1_last_t) * 8);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
