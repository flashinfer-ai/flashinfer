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
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 192
#define TMEM_TMEM_OFFSET 0
#define NUM_KV_PIPE_STAGES 9
#define NUM_ROPE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 4096
#define SMEM_SMEM_Q_STRIDE 4096
#define SMEM_SMEM_QR_OFF 17408
#define SMEM_SMEM_QR_STAGE_BYTES 2048
#define SMEM_SMEM_QR_STRIDE 2048
#define SMEM_SMEM_K_OFF 19456
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_VT_OFF 19456
#define SMEM_SMEM_VT_STAGE_BYTES 16384
#define SMEM_SMEM_VT_STRIDE 16384
#define SMEM_SMEM_KR_OFF 166912
#define SMEM_SMEM_KR_STAGE_BYTES 8192
#define SMEM_SMEM_KR_STRIDE 8192
#define SMEM_SMEM_PT_OFF 183296
#define SMEM_SMEM_PT_STAGE_BYTES 16384
#define SMEM_SMEM_PT_STRIDE 16384
#define SMEM_SMEM_PT_WORDS_OFF 183296
#define SMEM_SMEM_PT_WORDS_STAGE_BYTES 32768
#define SMEM_SMEM_PT_WORDS_STRIDE 32768
#define SMEM_SMEM_STATS_OFF 216064
#define SMEM_SMEM_STATS_STAGE_BYTES 1152
#define SMEM_SMEM_STATS_STRIDE 1152
#define SMEM_SMEM_VIS_OFF 217216
#define SMEM_SMEM_VIS_STAGE_BYTES 128
#define SMEM_SMEM_VIS_STRIDE 128
#define SMEM_SMEM_FLAGS_OFF 217344
#define SMEM_SMEM_FLAGS_STAGE_BYTES 128
#define SMEM_SMEM_FLAGS_STRIDE 128
#define SMEM_SMEM_STATS_W_OFF 216064
#define SMEM_SMEM_STATS_W_STAGE_BYTES 1152
#define SMEM_SMEM_STATS_W_STRIDE 1152
#define SMEM_SMEM_VIS_W_OFF 217216
#define SMEM_SMEM_VIS_W_STAGE_BYTES 128
#define SMEM_SMEM_VIS_W_STRIDE 128
#define SMEM_SMEM_FLAGS_W_OFF 217344
#define SMEM_SMEM_FLAGS_W_STAGE_BYTES 128
#define SMEM_SMEM_FLAGS_W_STRIDE 128
#define SMEM_TOTAL 217472
#define THREADS 384

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

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
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

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
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
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(384, 1) void
kernel_cake_kimi_k3_mla_fp8_paged_attention_a5d38bf637ec40bc8083(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_kr, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 80)
    #define rope_full_addr (mbar_base + 152)
    #define rope_empty_addr (mbar_base + 168)
    #define s_full_addr (mbar_base + 184)
    #define s_free_addr (mbar_base + 200)
    #define p_full_addr (mbar_base + 216)
    #define pt_free_addr (mbar_base + 232)
    #define pv_done_addr (mbar_base + 248)
    #define o_done_addr (mbar_base + 256)
    #define tmem_dealloc_addr (mbar_base + 264)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_qr_addr = smem + 17408;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_k_addr = smem + 19456;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_vt_addr = smem + 19456;
    uint8_t* smem_kr = reinterpret_cast<uint8_t*>(smem_raw + 166912);
    const int smem_kr_addr = smem + 166912;
    uint8_t* smem_pt = reinterpret_cast<uint8_t*>(smem_raw + 183296);
    const int smem_pt_addr = smem + 183296;
    unsigned int* smem_pt_words = reinterpret_cast<unsigned int*>(smem_raw + 183296);
    const int smem_pt_words_addr = smem + 183296;
    float* smem_stats = reinterpret_cast<float*>(smem_raw + 216064);
    const int smem_stats_addr = smem + 216064;
    int* smem_vis = reinterpret_cast<int*>(smem_raw + 217216);
    const int smem_vis_addr = smem + 217216;
    int* smem_flags = reinterpret_cast<int*>(smem_raw + 217344);
    const int smem_flags_addr = smem + 217344;
    unsigned int* smem_stats_w = reinterpret_cast<unsigned int*>(smem_raw + 216064);
    const int smem_stats_w_addr = smem + 216064;
    unsigned int* smem_vis_w = reinterpret_cast<unsigned int*>(smem_raw + 217216);
    const int smem_vis_w_addr = smem + 217216;
    unsigned int* smem_flags_w = reinterpret_cast<unsigned int*>(smem_raw + 217344);
    const int smem_flags_w_addr = smem + 217344;

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 9 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty: 9 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // --- pipeline 'rope_pipe' ---
            // rope_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // rope_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // s_free: 2 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            mbarrier_init(smem + 208, 256);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            // pt_free: 2 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            // tmem_dealloc: 1 barriers, init_count=352
            mbarrier_init(smem + 264, 352);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 272);
    if (warp == 0) {
        int _tmem_hold = smem + 272;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_wg0_main
            float psum0[8];
            float psum1[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                psum0[j] = 0.0f;
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                psum1[j_1] = 0.0f;
            }
            int split_idx = blockIdx.x;
            int m_tile = gridDim.y - 1 - blockIdx.y;
            int b = blockIdx.z;
            int q_start = cum_seq_lens_q[b];
            int q_len_b = cum_seq_lens_q[b + 1] - q_start;
            int kv_len = seq_lens[b];
            int rows_b = q_len_b * num_heads;
            int row0 = m_tile * 32;
            int rows_left = rows_b - row0;
            int rows_pos = ((rows_left < 0) ? 0 : rows_left);
            int rows_valid = ((rows_pos > 32) ? 32 : rows_pos);
            int row_base_global = q_start * num_heads + row0;
            int last_row = row0 + rows_valid - 1;
            int t_last = last_row / num_heads;
            int kv_end_raw = kv_len - q_len_b + t_last + 1;
            int kv_end = ((rows_valid == 0) ? 0 : kv_end_raw);
            int n_pages = (kv_end + 64 - 1) / 64;
            int n_tiles_total = (kv_end + 128 - 1) / 128;
            int tiles_per_split = (n_tiles_total + num_split - 1) / num_split;
            int my_start = split_idx * tiles_per_split;
            int my_end_raw = my_start + tiles_per_split;
            int my_end = ((my_end_raw > n_tiles_total) ? n_tiles_total : my_end_raw);
            int my_n_raw = my_end - my_start;
            int my_n_tiles = ((my_n_raw < 0) ? 0 : my_n_raw);
            int pt_base = b * max_pages_per_seq;
            const int warp_in_wg = warp % 4;
            const int lane_base = warp_in_wg * 32;
            const int my_tok = lane_base + lane;
            int red_col = warp_in_wg * 4 + lane;
            int is_reducer = lane < 4;
            int row_stride = num_split * 512;
            int out_base = (row_base_global * num_split + split_idx) * 512 + my_tok;
            if (is_reducer != 0) {
                int q_tok = (row0 + red_col) / num_heads;
                smem_vis[red_col] = kv_len - q_len_b + q_tok + 1;
                smem_stats[192 + red_col] = 1031.8073549220576f;
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int vis_min = smem_vis[0];
            #pragma unroll 1
            for (int tile = 0; tile < my_n_tiles; tile++) {
                int pphase = tile & 1;
                int aphase = tile & 1;
                int sphase = tile & 1;
                int s_wait = tile >> 1 & 1;
                mbarrier_wait(s_full_addr + (sphase) * 8, s_wait);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs = (my_start + tile) * 128 + my_tok;
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)(sphase * 32) + (unsigned int)(lane_base << 16));
                float _tmem_load_1[8];
                tmem_ld_x8(&_tmem_load_1[0], taddr + (unsigned int)(sphase * 32) + 8 + (unsigned int)(lane_base << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase) * 8);
                int tile_last = (my_start + tile) * 128 + 127;
                if (tile_last >= vis_min) {
                    #pragma unroll
                    for (int k = 0; k < 8; k += 4) {
                        uint32_t _smem_vis_w_reg_0[4];
                        __int128_t _smem_b128_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_0) : "r"(smem_vis_w_addr + (k) * 4));
                        _smem_vis_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[0];
                        _smem_vis_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[1];
                        _smem_vis_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[2];
                        _smem_vis_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[3];
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            int vis_e = 0;
                            vis_e = reinterpret_cast<int*>(&_smem_vis_w_reg_0[e])[0];
                            if (t_abs >= vis_e) {
                                _tmem_load_0[k + e] = -CAKE_INF;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_1 = 0; k_1 < 8; k_1 += 4) {
                        uint32_t _smem_vis_w_reg_1[4];
                        __int128_t _smem_b128_1;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_1) : "r"(smem_vis_w_addr + (8 + k_1) * 4));
                        _smem_vis_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[0];
                        _smem_vis_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[1];
                        _smem_vis_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[2];
                        _smem_vis_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[3];
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 4; e_1++) {
                            int vis_e_1 = 0;
                            vis_e_1 = reinterpret_cast<int*>(&_smem_vis_w_reg_1[e_1])[0];
                            if (t_abs >= vis_e_1) {
                                _tmem_load_1[k_1 + e_1] = -CAKE_INF;
                            }
                        }
                    }
                }
                float pm = -CAKE_INF;
                #pragma unroll
                for (int k_2 = 0; k_2 < 8; k_2 += 4) {
                    uint32_t _smem_stats_w_reg_0[4];
                    __int128_t _smem_b128_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_2) : "r"(smem_stats_w_addr + (192 + k_2) * 4));
                    _smem_stats_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[0];
                    _smem_stats_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[1];
                    _smem_stats_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[2];
                    _smem_stats_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[3];
                    #pragma unroll
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        _tmem_load_0[k_2 + e_2] = _tmem_load_0[k_2 + e_2] * softmax_scale_log2 + __uint_as_float(_smem_stats_w_reg_0[e_2]);
                        float _max_0 = max_noftz(pm, _tmem_load_0[k_2 + e_2]);
                        pm = _max_0;
                    }
                }
                float pm_0 = -CAKE_INF;
                #pragma unroll
                for (int k_3 = 0; k_3 < 8; k_3 += 4) {
                    uint32_t _smem_stats_w_reg_1[4];
                    __int128_t _smem_b128_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_3) : "r"(smem_stats_w_addr + (200 + k_3) * 4));
                    _smem_stats_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[0];
                    _smem_stats_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[1];
                    _smem_stats_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[2];
                    _smem_stats_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[3];
                    #pragma unroll
                    for (int e_3 = 0; e_3 < 4; e_3++) {
                        _tmem_load_1[k_3 + e_3] = _tmem_load_1[k_3 + e_3] * softmax_scale_log2 + __uint_as_float(_smem_stats_w_reg_1[e_3]);
                        float _max_1 = max_noftz(pm_0, _tmem_load_1[k_3 + e_3]);
                        pm_0 = _max_1;
                    }
                }
                float _max_2 = max_noftz(pm, pm_0);
                float pm_1 = _max_2;
                int _vote_0 = __any_sync(0xFFFFFFFF, pm_1 > 8.8073549220576f);
                int warp_exceed = _vote_0;
                if (lane == 0) {
                    smem_flags[aphase * 8 + warp_in_wg] = warp_exceed;
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                uint32_t _smem_flags_w_reg_0[4];
                __int128_t _smem_b128_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_4) : "r"(smem_flags_w_addr + (aphase * 8) * 4));
                _smem_flags_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[0];
                _smem_flags_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[1];
                _smem_flags_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[2];
                _smem_flags_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[3];
                unsigned int f_or = _smem_flags_w_reg_0[0] | _smem_flags_w_reg_0[1] | (_smem_flags_w_reg_0[2] | _smem_flags_w_reg_0[3]);
                int wg_exceed = 0;
                wg_exceed = reinterpret_cast<int*>(&f_or)[0];
                if (wg_exceed != 0) {
                    float red[8];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        red[j_2] = _tmem_load_0[j_2];
                    }
                    int upper = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 4; j_3++) {
                        float send = ((upper != 0) ? red[j_3] : red[j_3 + 4]);
                        float keep = ((upper != 0) ? red[j_3 + 4] : red[j_3]);
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
                        float recv = _shfl_xor_0;
                        float _max_3 = max_noftz(keep, recv);
                        red[j_3] = _max_3;
                    }
                    int upper_0 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 2; j_4++) {
                        float send_1 = ((upper_0 != 0) ? red[j_4] : red[j_4 + 2]);
                        float keep_1 = ((upper_0 != 0) ? red[j_4 + 2] : red[j_4]);
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1, 8);
                        float recv_1 = _shfl_xor_1;
                        float _max_4 = max_noftz(keep_1, recv_1);
                        red[j_4] = _max_4;
                    }
                    int upper_1 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 1; j_5++) {
                        float send_2 = ((upper_1 != 0) ? red[j_5] : red[j_5 + 1]);
                        float keep_2 = ((upper_1 != 0) ? red[j_5 + 1] : red[j_5]);
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_2, 4);
                        float recv_2 = _shfl_xor_2;
                        float _max_5 = max_noftz(keep_2, recv_2);
                        red[j_5] = _max_5;
                    }
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, red[0], 2);
                    float other = _shfl_xor_3;
                    float _max_6 = max_noftz(red[0], other);
                    red[0] = _max_6;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, red[0], 1);
                    float other_2 = _shfl_xor_4;
                    float _max_7 = max_noftz(red[0], other_2);
                    red[0] = _max_7;
                    if ((lane & 3) == 0) {
                        smem_stats[64 + warp_in_wg * 16 + (lane >> 2)] = red[0];
                    }
                    float red_3[8];
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        red_3[j_6] = _tmem_load_1[j_6];
                    }
                    int upper_4 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 4; j_7++) {
                        float send_3 = ((upper_4 != 0) ? red_3[j_7] : red_3[j_7 + 4]);
                        float keep_3 = ((upper_4 != 0) ? red_3[j_7 + 4] : red_3[j_7]);
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
                        float recv_3 = _shfl_xor_5;
                        float _max_8 = max_noftz(keep_3, recv_3);
                        red_3[j_7] = _max_8;
                    }
                    int upper_5 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 2; j_8++) {
                        float send_4 = ((upper_5 != 0) ? red_3[j_8] : red_3[j_8 + 2]);
                        float keep_4 = ((upper_5 != 0) ? red_3[j_8 + 2] : red_3[j_8]);
                        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_4, 8);
                        float recv_4 = _shfl_xor_6;
                        float _max_9 = max_noftz(keep_4, recv_4);
                        red_3[j_8] = _max_9;
                    }
                    int upper_6 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 1; j_9++) {
                        float send_5 = ((upper_6 != 0) ? red_3[j_9] : red_3[j_9 + 1]);
                        float keep_5 = ((upper_6 != 0) ? red_3[j_9 + 1] : red_3[j_9]);
                        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_5, 4);
                        float recv_5 = _shfl_xor_7;
                        float _max_10 = max_noftz(keep_5, recv_5);
                        red_3[j_9] = _max_10;
                    }
                    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, red_3[0], 2);
                    float other_7 = _shfl_xor_8;
                    float _max_11 = max_noftz(red_3[0], other_7);
                    red_3[0] = _max_11;
                    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, red_3[0], 1);
                    float other_8 = _shfl_xor_9;
                    float _max_12 = max_noftz(red_3[0], other_8);
                    red_3[0] = _max_12;
                    if ((lane & 3) == 0) {
                        smem_stats[64 + warp_in_wg * 16 + 8 + (lane >> 2)] = red_3[0];
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (is_reducer != 0) {
                        float m0 = smem_stats[64 + red_col];
                        float m1 = smem_stats[80 + red_col];
                        float m2 = smem_stats[96 + red_col];
                        float m3 = smem_stats[112 + red_col];
                        float _max_13 = max_noftz(m0, m1);
                        float _max_14 = max_noftz(m2, m3);
                        float _max_15 = max_noftz(_max_13, _max_14);
                        float tm = _max_15;
                        float c_old = smem_stats[192 + red_col];
                        float c_upd = 7.8073549220576f - tm + c_old;
                        float c_new = ((tm > -CAKE_INF) ? c_upd : c_old);
                        float delta = c_new - c_old;
                        float _exp2_0 = approx_exp2(delta);
                        float alpha = _exp2_0;
                        smem_stats[192 + red_col] = c_new;
                        smem_stats[32 + red_col] = delta;
                        smem_stats[red_col] = alpha;
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 8; k_4 += 4) {
                        uint32_t _smem_stats_w_reg_2[4];
                        __int128_t _smem_b128_5;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_5) : "r"(smem_stats_w_addr + (32 + k_4) * 4));
                        _smem_stats_w_reg_2[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[0];
                        _smem_stats_w_reg_2[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[1];
                        _smem_stats_w_reg_2[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[2];
                        _smem_stats_w_reg_2[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[3];
                        uint32_t _smem_stats_w_reg_3[4];
                        __int128_t _smem_b128_6;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_6) : "r"(smem_stats_w_addr + (k_4) * 4));
                        _smem_stats_w_reg_3[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[0];
                        _smem_stats_w_reg_3[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[1];
                        _smem_stats_w_reg_3[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[2];
                        _smem_stats_w_reg_3[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[3];
                        #pragma unroll
                        for (int e_4 = 0; e_4 < 4; e_4++) {
                            _tmem_load_0[k_4 + e_4] = _tmem_load_0[k_4 + e_4] + __uint_as_float(_smem_stats_w_reg_2[e_4]);
                            psum0[k_4 + e_4] = psum0[k_4 + e_4] * __uint_as_float(_smem_stats_w_reg_3[e_4]);
                        }
                    }
                    #pragma unroll
                    for (int k_5 = 0; k_5 < 8; k_5 += 4) {
                        uint32_t _smem_stats_w_reg_4[4];
                        __int128_t _smem_b128_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_7) : "r"(smem_stats_w_addr + (40 + k_5) * 4));
                        _smem_stats_w_reg_4[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[0];
                        _smem_stats_w_reg_4[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[1];
                        _smem_stats_w_reg_4[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[2];
                        _smem_stats_w_reg_4[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[3];
                        uint32_t _smem_stats_w_reg_5[4];
                        __int128_t _smem_b128_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_8) : "r"(smem_stats_w_addr + (8 + k_5) * 4));
                        _smem_stats_w_reg_5[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[0];
                        _smem_stats_w_reg_5[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[1];
                        _smem_stats_w_reg_5[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[2];
                        _smem_stats_w_reg_5[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[3];
                        #pragma unroll
                        for (int e_5 = 0; e_5 < 4; e_5++) {
                            _tmem_load_1[k_5 + e_5] = _tmem_load_1[k_5 + e_5] + __uint_as_float(_smem_stats_w_reg_4[e_5]);
                            psum1[k_5 + e_5] = psum1[k_5 + e_5] * __uint_as_float(_smem_stats_w_reg_5[e_5]);
                        }
                    }
                }
                if (tile >= 2) {
                    mbarrier_wait(pt_free_addr + (pphase) * 8, (tile >> 1) - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    float _exp2_1 = approx_exp2(_tmem_load_0[j_10]);
                    _tmem_load_0[j_10] = _exp2_1;
                    psum0[j_10] = psum0[j_10] + _tmem_load_0[j_10];
                }
                unsigned int packed[2];
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
                    packed[0] = _packed;
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
                    packed[1] = _packed;
                }
                int row_addr = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel = pphase * 16384 + my_tok * 128;
                int dst = row_addr + ((0 ^ my_tok & 7) << 4);
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst), "r"(packed[0]), "r"(packed[1]) : "memory");
                #pragma unroll
                for (int j_11 = 0; j_11 < 8; j_11++) {
                    float _exp2_2 = approx_exp2(_tmem_load_1[j_11]);
                    _tmem_load_1[j_11] = _exp2_2;
                    psum1[j_11] = psum1[j_11] + _tmem_load_1[j_11];
                }
                unsigned int packed_2[2];
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
                    packed_2[0] = _packed;
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
                    packed_2[1] = _packed;
                }
                int row_addr_3 = smem_pt_addr + (unsigned int)(pphase * 16384) + (unsigned int)(my_tok * 128);
                int row_rel_4 = pphase * 16384 + my_tok * 128;
                int dst_5 = row_addr_3 + ((0 ^ my_tok & 7) << 4) + 8;
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst_5), "r"(packed_2[0]), "r"(packed_2[1]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (wg_exceed != 0) {
                    if (tile > 0) {
                        mbarrier_wait(pv_done_addr, tile - 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int vs = 0; vs < 4; vs++) {
                            int o_a = taddr + 64 + (unsigned int)(vs * 32) + (unsigned int)(lane_base << 16);
                            int o_b = taddr + 64 + (unsigned int)(vs * 32) + 8 + (unsigned int)(lane_base << 16);
                            float _tmem_load_2[8];
                            tmem_ld_x8(&_tmem_load_2[0], o_a);
                            float _tmem_load_3[8];
                            tmem_ld_x8(&_tmem_load_3[0], o_b);
                            float vals[8];
                            #pragma unroll
                            for (int k_6 = 0; k_6 < 8; k_6 += 4) {
                                uint32_t _smem_stats_w_reg_6[4];
                                __int128_t _smem_b128_9;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_9) : "r"(smem_stats_w_addr + (k_6) * 4));
                                _smem_stats_w_reg_6[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[0];
                                _smem_stats_w_reg_6[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[1];
                                _smem_stats_w_reg_6[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[2];
                                _smem_stats_w_reg_6[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[3];
                                #pragma unroll
                                for (int e_6 = 0; e_6 < 4; e_6++) {
                                    vals[k_6 + e_6] = __uint_as_float(_smem_stats_w_reg_6[e_6]);
                                }
                            }
                            float vals_0[8];
                            #pragma unroll
                            for (int k_7 = 0; k_7 < 8; k_7 += 4) {
                                uint32_t _smem_stats_w_reg_7[4];
                                __int128_t _smem_b128_10;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_10) : "r"(smem_stats_w_addr + (8 + k_7) * 4));
                                _smem_stats_w_reg_7[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[0];
                                _smem_stats_w_reg_7[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[1];
                                _smem_stats_w_reg_7[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[2];
                                _smem_stats_w_reg_7[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[3];
                                #pragma unroll
                                for (int e_7 = 0; e_7 < 4; e_7++) {
                                    vals_0[k_7 + e_7] = __uint_as_float(_smem_stats_w_reg_7[e_7]);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int j_12 = 0; j_12 < 8; j_12++) {
                                _tmem_load_2[j_12] = _tmem_load_2[j_12] * vals[j_12];
                            }
                            #pragma unroll
                            for (int j_13 = 0; j_13 < 8; j_13++) {
                                _tmem_load_3[j_13] = _tmem_load_3[j_13] * vals_0[j_13];
                            }
                            tmem_st_x8_f32(o_a, _tmem_load_2);
                            tmem_st_x8_f32(o_b, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                mbarrier_arrive(p_full_addr + (pphase) * 8);
            }
            #pragma unroll
            for (int j_14 = 0; j_14 < 8; j_14++) {
                float _warp_reduce_0 = psum0[j_14];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                float wsum_j = _warp_reduce_0;
                if (lane == j_14 % 32) {
                    smem_stats[64 + warp_in_wg * 16 + j_14] = wsum_j;
                }
            }
            #pragma unroll
            for (int j_15 = 0; j_15 < 8; j_15++) {
                float _warp_reduce_1 = psum1[j_15];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                float wsum_j_1 = _warp_reduce_1;
                if (lane == (8 + j_15) % 32) {
                    smem_stats[64 + warp_in_wg * 16 + 8 + j_15] = wsum_j_1;
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (is_reducer != 0) {
                float s0 = smem_stats[64 + red_col];
                float s1 = smem_stats[80 + red_col];
                float s2 = smem_stats[96 + red_col];
                float s3 = smem_stats[112 + red_col];
                float csum = s0 + s1 + (s2 + s3);
                smem_stats[224 + red_col] = csum;
                float safe_sum = ((csum > 0.0f) ? csum : 1.0f);
                float out_scale = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_0 = approx_rcp(safe_sum);
                smem_stats[256 + red_col] = _rcp_0 * out_scale;
                if (red_col < rows_valid) {
                    int stat_off = (row_base_global + red_col) * num_split + split_idx;
                    float c_fin = smem_stats[192 + red_col];
                    float stored_max = 7.8073549220576f - c_fin;
                    *(reinterpret_cast<float*>(partial_max + stat_off) + (0)) = stored_max;
                    *(reinterpret_cast<float*>(partial_sum + stat_off) + (0)) = csum;
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            unsigned int _phase_o_done_0 = 0;
            if (my_n_tiles > 0) {
                float vals_1[16];
                #pragma unroll
                for (int k_8 = 0; k_8 < 16; k_8 += 4) {
                    uint32_t _smem_stats_w_reg_8[4];
                    __int128_t _smem_b128_11;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_11) : "r"(smem_stats_w_addr + (256 + k_8) * 4));
                    _smem_stats_w_reg_8[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[0];
                    _smem_stats_w_reg_8[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[1];
                    _smem_stats_w_reg_8[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[2];
                    _smem_stats_w_reg_8[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[3];
                    #pragma unroll
                    for (int e_8 = 0; e_8 < 4; e_8++) {
                        vals_1[k_8 + e_8] = __uint_as_float(_smem_stats_w_reg_8[e_8]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_1 = 0; vs_1 < 4; vs_1++) {
                    float _tmem_load_4[8];
                    tmem_ld_x8(&_tmem_load_4[0], taddr + 64 + (unsigned int)(vs_1 * 32) + (unsigned int)(lane_base << 16));
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 8; j_16++) {
                        int c = j_16;
                        if (c < rows_valid) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c * row_stride + vs_1 * 128)) + (0)) = __float2bfloat16_rn(_tmem_load_4[j_16] * vals_1[j_16]);
                        }
                    }
                    float _tmem_load_5[8];
                    tmem_ld_x8(&_tmem_load_5[0], taddr + 64 + (unsigned int)(vs_1 * 32) + 8 + (unsigned int)(lane_base << 16));
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 8; j_17++) {
                        int c_1 = 8 + j_17;
                        if (c_1 < rows_valid) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base + c_1 * row_stride + vs_1 * 128)) + (0)) = __float2bfloat16_rn(_tmem_load_5[j_17] * vals_1[8 + j_17]);
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_wg1_main
            float psum0_1[8];
            float psum1_1[8];
            #pragma unroll
            for (int j_18 = 0; j_18 < 8; j_18++) {
                psum0_1[j_18] = 0.0f;
            }
            #pragma unroll
            for (int j_19 = 0; j_19 < 8; j_19++) {
                psum1_1[j_19] = 0.0f;
            }
            int split_idx_1 = blockIdx.x;
            int m_tile_1 = gridDim.y - 1 - blockIdx.y;
            int b_1 = blockIdx.z;
            int q_start_1 = cum_seq_lens_q[b_1];
            int q_len_b_1 = cum_seq_lens_q[b_1 + 1] - q_start_1;
            int kv_len_1 = seq_lens[b_1];
            int rows_b_1 = q_len_b_1 * num_heads;
            int row0_1 = m_tile_1 * 32;
            int rows_left_1 = rows_b_1 - row0_1;
            int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
            int rows_valid_1 = ((rows_pos_1 > 32) ? 32 : rows_pos_1);
            int row_base_global_1 = q_start_1 * num_heads + row0_1;
            int last_row_1 = row0_1 + rows_valid_1 - 1;
            int t_last_1 = last_row_1 / num_heads;
            int kv_end_raw_1 = kv_len_1 - q_len_b_1 + t_last_1 + 1;
            int kv_end_1 = ((rows_valid_1 == 0) ? 0 : kv_end_raw_1);
            int n_pages_1 = (kv_end_1 + 64 - 1) / 64;
            int n_tiles_total_1 = (kv_end_1 + 128 - 1) / 128;
            int tiles_per_split_1 = (n_tiles_total_1 + num_split - 1) / num_split;
            int my_start_1 = split_idx_1 * tiles_per_split_1;
            int my_end_raw_1 = my_start_1 + tiles_per_split_1;
            int my_end_1 = ((my_end_raw_1 > n_tiles_total_1) ? n_tiles_total_1 : my_end_raw_1);
            int my_n_raw_1 = my_end_1 - my_start_1;
            int my_n_tiles_1 = ((my_n_raw_1 < 0) ? 0 : my_n_raw_1);
            int pt_base_1 = b_1 * max_pages_per_seq;
            const int warp_in_wg_1 = warp % 4;
            const int lane_base_1 = warp_in_wg_1 * 32;
            const int my_tok_1 = lane_base_1 + lane;
            int red_col_1 = 16 + warp_in_wg_1 * 4 + lane;
            int is_reducer_1 = lane < 4;
            int row_stride_1 = num_split * 512;
            int out_base_1 = (row_base_global_1 * num_split + split_idx_1) * 512 + my_tok_1;
            if (is_reducer_1 != 0) {
                int q_tok_1 = (row0_1 + red_col_1) / num_heads;
                smem_vis[red_col_1] = kv_len_1 - q_len_b_1 + q_tok_1 + 1;
                smem_stats[192 + red_col_1] = 1031.8073549220576f;
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            int vis_min_1 = smem_vis[16];
            #pragma unroll 1
            for (int tile_1 = 0; tile_1 < my_n_tiles_1; tile_1++) {
                int pphase_1 = tile_1 & 1;
                int aphase_1 = tile_1 & 1;
                int sphase_1 = tile_1 & 1;
                int s_wait_1 = tile_1 >> 1 & 1;
                mbarrier_wait(s_full_addr + (sphase_1) * 8, s_wait_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int t_abs_1 = (my_start_1 + tile_1) * 128 + my_tok_1;
                float _tmem_load_6[8];
                tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)(sphase_1 * 32) + 16 + (unsigned int)(lane_base_1 << 16));
                float _tmem_load_7[8];
                tmem_ld_x8(&_tmem_load_7[0], taddr + (unsigned int)(sphase_1 * 32) + 16 + 8 + (unsigned int)(lane_base_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr + (sphase_1) * 8);
                int tile_last_1 = (my_start_1 + tile_1) * 128 + 127;
                if (tile_last_1 >= vis_min_1) {
                    #pragma unroll
                    for (int k_9 = 0; k_9 < 8; k_9 += 4) {
                        uint32_t _smem_vis_w_reg_2[4];
                        __int128_t _smem_b128_0;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_0) : "r"(smem_vis_w_addr + (16 + k_9) * 4));
                        _smem_vis_w_reg_2[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[0];
                        _smem_vis_w_reg_2[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[1];
                        _smem_vis_w_reg_2[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[2];
                        _smem_vis_w_reg_2[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[3];
                        #pragma unroll
                        for (int e_9 = 0; e_9 < 4; e_9++) {
                            int vis_e_2 = 0;
                            vis_e_2 = reinterpret_cast<int*>(&_smem_vis_w_reg_2[e_9])[0];
                            if (t_abs_1 >= vis_e_2) {
                                _tmem_load_6[k_9 + e_9] = -CAKE_INF;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_10 = 0; k_10 < 8; k_10 += 4) {
                        uint32_t _smem_vis_w_reg_3[4];
                        __int128_t _smem_b128_1;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_1) : "r"(smem_vis_w_addr + (24 + k_10) * 4));
                        _smem_vis_w_reg_3[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[0];
                        _smem_vis_w_reg_3[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[1];
                        _smem_vis_w_reg_3[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[2];
                        _smem_vis_w_reg_3[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[3];
                        #pragma unroll
                        for (int e_10 = 0; e_10 < 4; e_10++) {
                            int vis_e_3 = 0;
                            vis_e_3 = reinterpret_cast<int*>(&_smem_vis_w_reg_3[e_10])[0];
                            if (t_abs_1 >= vis_e_3) {
                                _tmem_load_7[k_10 + e_10] = -CAKE_INF;
                            }
                        }
                    }
                }
                float pm_2 = -CAKE_INF;
                #pragma unroll
                for (int k_11 = 0; k_11 < 8; k_11 += 4) {
                    uint32_t _smem_stats_w_reg_9[4];
                    __int128_t _smem_b128_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_2) : "r"(smem_stats_w_addr + (208 + k_11) * 4));
                    _smem_stats_w_reg_9[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[0];
                    _smem_stats_w_reg_9[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[1];
                    _smem_stats_w_reg_9[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[2];
                    _smem_stats_w_reg_9[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[3];
                    #pragma unroll
                    for (int e_11 = 0; e_11 < 4; e_11++) {
                        _tmem_load_6[k_11 + e_11] = _tmem_load_6[k_11 + e_11] * softmax_scale_log2 + __uint_as_float(_smem_stats_w_reg_9[e_11]);
                        float _max_16 = max_noftz(pm_2, _tmem_load_6[k_11 + e_11]);
                        pm_2 = _max_16;
                    }
                }
                float pm_0_1 = -CAKE_INF;
                #pragma unroll
                for (int k_12 = 0; k_12 < 8; k_12 += 4) {
                    uint32_t _smem_stats_w_reg_10[4];
                    __int128_t _smem_b128_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_3) : "r"(smem_stats_w_addr + (216 + k_12) * 4));
                    _smem_stats_w_reg_10[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[0];
                    _smem_stats_w_reg_10[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[1];
                    _smem_stats_w_reg_10[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[2];
                    _smem_stats_w_reg_10[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[3];
                    #pragma unroll
                    for (int e_12 = 0; e_12 < 4; e_12++) {
                        _tmem_load_7[k_12 + e_12] = _tmem_load_7[k_12 + e_12] * softmax_scale_log2 + __uint_as_float(_smem_stats_w_reg_10[e_12]);
                        float _max_17 = max_noftz(pm_0_1, _tmem_load_7[k_12 + e_12]);
                        pm_0_1 = _max_17;
                    }
                }
                float _max_18 = max_noftz(pm_2, pm_0_1);
                float pm_1_1 = _max_18;
                int _vote_1 = __any_sync(0xFFFFFFFF, pm_1_1 > 8.8073549220576f);
                int warp_exceed_1 = _vote_1;
                if (lane == 0) {
                    smem_flags[aphase_1 * 8 + 4 + warp_in_wg_1] = warp_exceed_1;
                }
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                uint32_t _smem_flags_w_reg_1[4];
                __int128_t _smem_b128_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_4) : "r"(smem_flags_w_addr + (aphase_1 * 8 + 4) * 4));
                _smem_flags_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[0];
                _smem_flags_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[1];
                _smem_flags_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[2];
                _smem_flags_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_4)[3];
                unsigned int f_or_1 = _smem_flags_w_reg_1[0] | _smem_flags_w_reg_1[1] | (_smem_flags_w_reg_1[2] | _smem_flags_w_reg_1[3]);
                int wg_exceed_1 = 0;
                wg_exceed_1 = reinterpret_cast<int*>(&f_or_1)[0];
                if (wg_exceed_1 != 0) {
                    float red_1[8];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 8; j_20++) {
                        red_1[j_20] = _tmem_load_6[j_20];
                    }
                    int upper_2 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 4; j_21++) {
                        float send_6 = ((upper_2 != 0) ? red_1[j_21] : red_1[j_21 + 4]);
                        float keep_6 = ((upper_2 != 0) ? red_1[j_21 + 4] : red_1[j_21]);
                        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
                        float recv_6 = _shfl_xor_10;
                        float _max_19 = max_noftz(keep_6, recv_6);
                        red_1[j_21] = _max_19;
                    }
                    int upper_0_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 2; j_22++) {
                        float send_7 = ((upper_0_1 != 0) ? red_1[j_22] : red_1[j_22 + 2]);
                        float keep_7 = ((upper_0_1 != 0) ? red_1[j_22 + 2] : red_1[j_22]);
                        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_7, 8);
                        float recv_7 = _shfl_xor_11;
                        float _max_20 = max_noftz(keep_7, recv_7);
                        red_1[j_22] = _max_20;
                    }
                    int upper_1_1 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_23 = 0; j_23 < 1; j_23++) {
                        float send_8 = ((upper_1_1 != 0) ? red_1[j_23] : red_1[j_23 + 1]);
                        float keep_8 = ((upper_1_1 != 0) ? red_1[j_23 + 1] : red_1[j_23]);
                        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_8, 4);
                        float recv_8 = _shfl_xor_12;
                        float _max_21 = max_noftz(keep_8, recv_8);
                        red_1[j_23] = _max_21;
                    }
                    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 2);
                    float other_1 = _shfl_xor_13;
                    float _max_22 = max_noftz(red_1[0], other_1);
                    red_1[0] = _max_22;
                    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, red_1[0], 1);
                    float other_2_1 = _shfl_xor_14;
                    float _max_23 = max_noftz(red_1[0], other_2_1);
                    red_1[0] = _max_23;
                    if ((lane & 3) == 0) {
                        smem_stats[128 + warp_in_wg_1 * 16 + (lane >> 2)] = red_1[0];
                    }
                    float red_3_1[8];
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 8; j_24++) {
                        red_3_1[j_24] = _tmem_load_7[j_24];
                    }
                    int upper_4_1 = (lane & 16) != 0;
                    #pragma unroll
                    for (int j_25 = 0; j_25 < 4; j_25++) {
                        float send_9 = ((upper_4_1 != 0) ? red_3_1[j_25] : red_3_1[j_25 + 4]);
                        float keep_9 = ((upper_4_1 != 0) ? red_3_1[j_25 + 4] : red_3_1[j_25]);
                        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_9, 16);
                        float recv_9 = _shfl_xor_15;
                        float _max_24 = max_noftz(keep_9, recv_9);
                        red_3_1[j_25] = _max_24;
                    }
                    int upper_5_1 = (lane & 8) != 0;
                    #pragma unroll
                    for (int j_26 = 0; j_26 < 2; j_26++) {
                        float send_10 = ((upper_5_1 != 0) ? red_3_1[j_26] : red_3_1[j_26 + 2]);
                        float keep_10 = ((upper_5_1 != 0) ? red_3_1[j_26 + 2] : red_3_1[j_26]);
                        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, send_10, 8);
                        float recv_10 = _shfl_xor_16;
                        float _max_25 = max_noftz(keep_10, recv_10);
                        red_3_1[j_26] = _max_25;
                    }
                    int upper_6_1 = (lane & 4) != 0;
                    #pragma unroll
                    for (int j_27 = 0; j_27 < 1; j_27++) {
                        float send_11 = ((upper_6_1 != 0) ? red_3_1[j_27] : red_3_1[j_27 + 1]);
                        float keep_11 = ((upper_6_1 != 0) ? red_3_1[j_27 + 1] : red_3_1[j_27]);
                        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, send_11, 4);
                        float recv_11 = _shfl_xor_17;
                        float _max_26 = max_noftz(keep_11, recv_11);
                        red_3_1[j_27] = _max_26;
                    }
                    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, red_3_1[0], 2);
                    float other_7_1 = _shfl_xor_18;
                    float _max_27 = max_noftz(red_3_1[0], other_7_1);
                    red_3_1[0] = _max_27;
                    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, red_3_1[0], 1);
                    float other_8_1 = _shfl_xor_19;
                    float _max_28 = max_noftz(red_3_1[0], other_8_1);
                    red_3_1[0] = _max_28;
                    if ((lane & 3) == 0) {
                        smem_stats[128 + warp_in_wg_1 * 16 + 8 + (lane >> 2)] = red_3_1[0];
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (is_reducer_1 != 0) {
                        float m0_1 = smem_stats[128 + (red_col_1 - 16)];
                        float m1_1 = smem_stats[144 + (red_col_1 - 16)];
                        float m2_1 = smem_stats[160 + (red_col_1 - 16)];
                        float m3_1 = smem_stats[176 + (red_col_1 - 16)];
                        float _max_29 = max_noftz(m0_1, m1_1);
                        float _max_30 = max_noftz(m2_1, m3_1);
                        float _max_31 = max_noftz(_max_29, _max_30);
                        float tm_1 = _max_31;
                        float c_old_1 = smem_stats[192 + red_col_1];
                        float c_upd_1 = 7.8073549220576f - tm_1 + c_old_1;
                        float c_new_1 = ((tm_1 > -CAKE_INF) ? c_upd_1 : c_old_1);
                        float delta_1 = c_new_1 - c_old_1;
                        float _exp2_3 = approx_exp2(delta_1);
                        float alpha_1 = _exp2_3;
                        smem_stats[192 + red_col_1] = c_new_1;
                        smem_stats[32 + red_col_1] = delta_1;
                        smem_stats[red_col_1] = alpha_1;
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    #pragma unroll
                    for (int k_13 = 0; k_13 < 8; k_13 += 4) {
                        uint32_t _smem_stats_w_reg_11[4];
                        __int128_t _smem_b128_5;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_5) : "r"(smem_stats_w_addr + (48 + k_13) * 4));
                        _smem_stats_w_reg_11[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[0];
                        _smem_stats_w_reg_11[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[1];
                        _smem_stats_w_reg_11[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[2];
                        _smem_stats_w_reg_11[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[3];
                        uint32_t _smem_stats_w_reg_12[4];
                        __int128_t _smem_b128_6;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_6) : "r"(smem_stats_w_addr + (16 + k_13) * 4));
                        _smem_stats_w_reg_12[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[0];
                        _smem_stats_w_reg_12[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[1];
                        _smem_stats_w_reg_12[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[2];
                        _smem_stats_w_reg_12[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[3];
                        #pragma unroll
                        for (int e_13 = 0; e_13 < 4; e_13++) {
                            _tmem_load_6[k_13 + e_13] = _tmem_load_6[k_13 + e_13] + __uint_as_float(_smem_stats_w_reg_11[e_13]);
                            psum0_1[k_13 + e_13] = psum0_1[k_13 + e_13] * __uint_as_float(_smem_stats_w_reg_12[e_13]);
                        }
                    }
                    #pragma unroll
                    for (int k_14 = 0; k_14 < 8; k_14 += 4) {
                        uint32_t _smem_stats_w_reg_13[4];
                        __int128_t _smem_b128_7;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_7) : "r"(smem_stats_w_addr + (56 + k_14) * 4));
                        _smem_stats_w_reg_13[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[0];
                        _smem_stats_w_reg_13[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[1];
                        _smem_stats_w_reg_13[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[2];
                        _smem_stats_w_reg_13[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[3];
                        uint32_t _smem_stats_w_reg_14[4];
                        __int128_t _smem_b128_8;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_8) : "r"(smem_stats_w_addr + (24 + k_14) * 4));
                        _smem_stats_w_reg_14[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[0];
                        _smem_stats_w_reg_14[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[1];
                        _smem_stats_w_reg_14[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[2];
                        _smem_stats_w_reg_14[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[3];
                        #pragma unroll
                        for (int e_14 = 0; e_14 < 4; e_14++) {
                            _tmem_load_7[k_14 + e_14] = _tmem_load_7[k_14 + e_14] + __uint_as_float(_smem_stats_w_reg_13[e_14]);
                            psum1_1[k_14 + e_14] = psum1_1[k_14 + e_14] * __uint_as_float(_smem_stats_w_reg_14[e_14]);
                        }
                    }
                }
                if (tile_1 >= 2) {
                    mbarrier_wait(pt_free_addr + (pphase_1) * 8, (tile_1 >> 1) - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                #pragma unroll
                for (int j_28 = 0; j_28 < 8; j_28++) {
                    float _exp2_4 = approx_exp2(_tmem_load_6[j_28]);
                    _tmem_load_6[j_28] = _exp2_4;
                    psum0_1[j_28] = psum0_1[j_28] + _tmem_load_6[j_28];
                }
                unsigned int packed_1[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(_tmem_load_6[0]), "f"(_tmem_load_6[1]),
                                           "f"(_tmem_load_6[2]), "f"(_tmem_load_6[3]));
                    packed_1[0] = _packed;
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
                        : "=r"(_packed) : "f"(_tmem_load_6[4]), "f"(_tmem_load_6[5]),
                                           "f"(_tmem_load_6[6]), "f"(_tmem_load_6[7]));
                    packed_1[1] = _packed;
                }
                int row_addr_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int dst_1 = row_addr_1 + ((1 ^ my_tok_1 & 7) << 4);
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst_1), "r"(packed_1[0]), "r"(packed_1[1]) : "memory");
                #pragma unroll
                for (int j_29 = 0; j_29 < 8; j_29++) {
                    float _exp2_5 = approx_exp2(_tmem_load_7[j_29]);
                    _tmem_load_7[j_29] = _exp2_5;
                    psum1_1[j_29] = psum1_1[j_29] + _tmem_load_7[j_29];
                }
                unsigned int packed_2_1[2];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(_tmem_load_7[0]), "f"(_tmem_load_7[1]),
                                           "f"(_tmem_load_7[2]), "f"(_tmem_load_7[3]));
                    packed_2_1[0] = _packed;
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
                        : "=r"(_packed) : "f"(_tmem_load_7[4]), "f"(_tmem_load_7[5]),
                                           "f"(_tmem_load_7[6]), "f"(_tmem_load_7[7]));
                    packed_2_1[1] = _packed;
                }
                int row_addr_3_1 = smem_pt_addr + (unsigned int)(pphase_1 * 16384) + (unsigned int)(my_tok_1 * 128);
                int row_rel_4_1 = pphase_1 * 16384 + my_tok_1 * 128;
                int dst_5_1 = row_addr_3_1 + ((1 ^ my_tok_1 & 7) << 4) + 8;
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(dst_5_1), "r"(packed_2_1[0]), "r"(packed_2_1[1]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (wg_exceed_1 != 0) {
                    if (tile_1 > 0) {
                        mbarrier_wait(pv_done_addr, tile_1 - 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int vs_2 = 0; vs_2 < 4; vs_2++) {
                            int o_a_1 = taddr + 64 + (unsigned int)(vs_2 * 32) + 16 + (unsigned int)(lane_base_1 << 16);
                            int o_b_1 = taddr + 64 + (unsigned int)(vs_2 * 32) + 16 + 8 + (unsigned int)(lane_base_1 << 16);
                            float _tmem_load_8[8];
                            tmem_ld_x8(&_tmem_load_8[0], o_a_1);
                            float _tmem_load_9[8];
                            tmem_ld_x8(&_tmem_load_9[0], o_b_1);
                            float vals_2[8];
                            #pragma unroll
                            for (int k_15 = 0; k_15 < 8; k_15 += 4) {
                                uint32_t _smem_stats_w_reg_15[4];
                                __int128_t _smem_b128_9;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_9) : "r"(smem_stats_w_addr + (16 + k_15) * 4));
                                _smem_stats_w_reg_15[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[0];
                                _smem_stats_w_reg_15[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[1];
                                _smem_stats_w_reg_15[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[2];
                                _smem_stats_w_reg_15[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_9)[3];
                                #pragma unroll
                                for (int e_15 = 0; e_15 < 4; e_15++) {
                                    vals_2[k_15 + e_15] = __uint_as_float(_smem_stats_w_reg_15[e_15]);
                                }
                            }
                            float vals_0_1[8];
                            #pragma unroll
                            for (int k_16 = 0; k_16 < 8; k_16 += 4) {
                                uint32_t _smem_stats_w_reg_16[4];
                                __int128_t _smem_b128_10;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_10) : "r"(smem_stats_w_addr + (24 + k_16) * 4));
                                _smem_stats_w_reg_16[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[0];
                                _smem_stats_w_reg_16[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[1];
                                _smem_stats_w_reg_16[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[2];
                                _smem_stats_w_reg_16[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_10)[3];
                                #pragma unroll
                                for (int e_16 = 0; e_16 < 4; e_16++) {
                                    vals_0_1[k_16 + e_16] = __uint_as_float(_smem_stats_w_reg_16[e_16]);
                                }
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int j_30 = 0; j_30 < 8; j_30++) {
                                _tmem_load_8[j_30] = _tmem_load_8[j_30] * vals_2[j_30];
                            }
                            #pragma unroll
                            for (int j_31 = 0; j_31 < 8; j_31++) {
                                _tmem_load_9[j_31] = _tmem_load_9[j_31] * vals_0_1[j_31];
                            }
                            tmem_st_x8_f32(o_a_1, _tmem_load_8);
                            tmem_st_x8_f32(o_b_1, _tmem_load_9);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                mbarrier_arrive(p_full_addr + (pphase_1) * 8);
            }
            #pragma unroll
            for (int j_32 = 0; j_32 < 8; j_32++) {
                float _warp_reduce_2 = psum0_1[j_32];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                float wsum_j_2 = _warp_reduce_2;
                if (lane == j_32 % 32) {
                    smem_stats[128 + warp_in_wg_1 * 16 + j_32] = wsum_j_2;
                }
            }
            #pragma unroll
            for (int j_33 = 0; j_33 < 8; j_33++) {
                float _warp_reduce_3 = psum1_1[j_33];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                float wsum_j_3 = _warp_reduce_3;
                if (lane == (8 + j_33) % 32) {
                    smem_stats[128 + warp_in_wg_1 * 16 + 8 + j_33] = wsum_j_3;
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (is_reducer_1 != 0) {
                float s0_1 = smem_stats[128 + (red_col_1 - 16)];
                float s1_1 = smem_stats[144 + (red_col_1 - 16)];
                float s2_1 = smem_stats[160 + (red_col_1 - 16)];
                float s3_1 = smem_stats[176 + (red_col_1 - 16)];
                float csum_1 = s0_1 + s1_1 + (s2_1 + s3_1);
                smem_stats[224 + red_col_1] = csum_1;
                float safe_sum_1 = ((csum_1 > 0.0f) ? csum_1 : 1.0f);
                float out_scale_1 = ((num_split == 1) ? bmm2_scale : 1.0f);
                float _rcp_1 = approx_rcp(safe_sum_1);
                smem_stats[256 + red_col_1] = _rcp_1 * out_scale_1;
                if (red_col_1 < rows_valid_1) {
                    int stat_off_1 = (row_base_global_1 + red_col_1) * num_split + split_idx_1;
                    float c_fin_1 = smem_stats[192 + red_col_1];
                    float stored_max_1 = 7.8073549220576f - c_fin_1;
                    *(reinterpret_cast<float*>(partial_max + stat_off_1) + (0)) = stored_max_1;
                    *(reinterpret_cast<float*>(partial_sum + stat_off_1) + (0)) = csum_1;
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            unsigned int _phase_o_done_0_1 = 0;
            if (my_n_tiles_1 > 0) {
                float vals_3[16];
                #pragma unroll
                for (int k_17 = 0; k_17 < 16; k_17 += 4) {
                    uint32_t _smem_stats_w_reg_17[4];
                    __int128_t _smem_b128_11;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_11) : "r"(smem_stats_w_addr + (272 + k_17) * 4));
                    _smem_stats_w_reg_17[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[0];
                    _smem_stats_w_reg_17[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[1];
                    _smem_stats_w_reg_17[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[2];
                    _smem_stats_w_reg_17[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_11)[3];
                    #pragma unroll
                    for (int e_17 = 0; e_17 < 4; e_17++) {
                        vals_3[k_17 + e_17] = __uint_as_float(_smem_stats_w_reg_17[e_17]);
                    }
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0_1);
                _phase_o_done_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_3 = 0; vs_3 < 4; vs_3++) {
                    float _tmem_load_10[8];
                    tmem_ld_x8(&_tmem_load_10[0], taddr + 64 + (unsigned int)(vs_3 * 32) + 16 + (unsigned int)(lane_base_1 << 16));
                    #pragma unroll
                    for (int j_34 = 0; j_34 < 8; j_34++) {
                        int c_2 = 16 + j_34;
                        if (c_2 < rows_valid_1) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_2 * row_stride_1 + vs_3 * 128)) + (0)) = __float2bfloat16_rn(_tmem_load_10[j_34] * vals_3[j_34]);
                        }
                    }
                    float _tmem_load_11[8];
                    tmem_ld_x8(&_tmem_load_11[0], taddr + 64 + (unsigned int)(vs_3 * 32) + 16 + 8 + (unsigned int)(lane_base_1 << 16));
                    #pragma unroll
                    for (int j_35 = 0; j_35 < 8; j_35++) {
                        int c_3 = 24 + j_35;
                        if (c_3 < rows_valid_1) {
                            *(reinterpret_cast<__nv_bfloat16*>(partial_O + (out_base_1 + c_3 * row_stride_1 + vs_3 * 128)) + (0)) = __float2bfloat16_rn(_tmem_load_11[j_35] * vals_3[8 + j_35]);
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
            int split_idx_2 = blockIdx.x;
            int m_tile_2 = gridDim.y - 1 - blockIdx.y;
            int b_2 = blockIdx.z;
            int q_start_2 = cum_seq_lens_q[b_2];
            int q_len_b_2 = cum_seq_lens_q[b_2 + 1] - q_start_2;
            int kv_len_2 = seq_lens[b_2];
            int rows_b_2 = q_len_b_2 * num_heads;
            int row0_2 = m_tile_2 * 32;
            int rows_left_2 = rows_b_2 - row0_2;
            int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
            int rows_valid_2 = ((rows_pos_2 > 32) ? 32 : rows_pos_2);
            int row_base_global_2 = q_start_2 * num_heads + row0_2;
            int last_row_2 = row0_2 + rows_valid_2 - 1;
            int t_last_2 = last_row_2 / num_heads;
            int kv_end_raw_2 = kv_len_2 - q_len_b_2 + t_last_2 + 1;
            int kv_end_2 = ((rows_valid_2 == 0) ? 0 : kv_end_raw_2);
            int n_pages_2 = (kv_end_2 + 64 - 1) / 64;
            int n_tiles_total_2 = (kv_end_2 + 128 - 1) / 128;
            int tiles_per_split_2 = (n_tiles_total_2 + num_split - 1) / num_split;
            int my_start_2 = split_idx_2 * tiles_per_split_2;
            int my_end_raw_2 = my_start_2 + tiles_per_split_2;
            int my_end_2 = ((my_end_raw_2 > n_tiles_total_2) ? n_tiles_total_2 : my_end_raw_2);
            int my_n_raw_2 = my_end_2 - my_start_2;
            int my_n_tiles_2 = ((my_n_raw_2 < 0) ? 0 : my_n_raw_2);
            int pt_base_2 = b_2 * max_pages_per_seq;
            unsigned int mma_stage = 0;
            unsigned int mma_phase = 0;
            unsigned int rmma_stage = 0;
            unsigned int rmma_phase = 0;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll 1
            for (int tile_2 = 0; tile_2 < my_n_tiles_2; tile_2++) {
                int sphase_2 = tile_2 & 1;
                if (tile_2 >= 2) {
                    mbarrier_wait(s_free_addr + (sphase_2) * 8, (tile_2 >> 1) - 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                #pragma unroll
                for (int st = 0; st < 4; st++) {
                    mbarrier_wait(kv_full_addr + (mma_stage) * 8, mma_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (st) * 256);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134742032, ((st == 0) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134742032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134742032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134742032, 1);
                        }
                    }
                    mma_stage += 1;
                    if (mma_stage == 9) { mma_stage = 0; mma_phase ^= 1; }
                }
                mbarrier_wait(rope_full_addr + (rmma_stage) * 8, rmma_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_1 = make_warp_uniform((((smem_kr_addr) >> 4) & 0x3FFF) + (rmma_stage) * 512);
                int _mma_b_lo_1 = make_warp_uniform(((smem_qr_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134742032, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (sphase_2 * 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134742032, 1);
                    }
                }
                elect_commit(s_full_addr + (sphase_2) * 8);
                elect_commit(rope_empty_addr + (rmma_stage) * 8);
                rmma_stage += 1;
                if (rmma_stage == 2) { rmma_stage = 0; rmma_phase ^= 1; }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: pv_warp ----
    if (warp == 9) {
        { // pv_warp_main
            int split_idx_3 = blockIdx.x;
            int m_tile_3 = gridDim.y - 1 - blockIdx.y;
            int b_3 = blockIdx.z;
            int q_start_3 = cum_seq_lens_q[b_3];
            int q_len_b_3 = cum_seq_lens_q[b_3 + 1] - q_start_3;
            int kv_len_3 = seq_lens[b_3];
            int rows_b_3 = q_len_b_3 * num_heads;
            int row0_3 = m_tile_3 * 32;
            int rows_left_3 = rows_b_3 - row0_3;
            int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
            int rows_valid_3 = ((rows_pos_3 > 32) ? 32 : rows_pos_3);
            int row_base_global_3 = q_start_3 * num_heads + row0_3;
            int last_row_3 = row0_3 + rows_valid_3 - 1;
            int t_last_3 = last_row_3 / num_heads;
            int kv_end_raw_3 = kv_len_3 - q_len_b_3 + t_last_3 + 1;
            int kv_end_3 = ((rows_valid_3 == 0) ? 0 : kv_end_raw_3);
            int n_pages_3 = (kv_end_3 + 64 - 1) / 64;
            int n_tiles_total_3 = (kv_end_3 + 128 - 1) / 128;
            int tiles_per_split_3 = (n_tiles_total_3 + num_split - 1) / num_split;
            int my_start_3 = split_idx_3 * tiles_per_split_3;
            int my_end_raw_3 = my_start_3 + tiles_per_split_3;
            int my_end_3 = ((my_end_raw_3 > n_tiles_total_3) ? n_tiles_total_3 : my_end_raw_3);
            int my_n_raw_3 = my_end_3 - my_start_3;
            int my_n_tiles_3 = ((my_n_raw_3 < 0) ? 0 : my_n_raw_3);
            int pt_base_3 = b_3 * max_pages_per_seq;
            int first_pv = 1;
            unsigned int base = 0;
            #pragma unroll 1
            for (int tile_3 = 0; tile_3 < my_n_tiles_3; tile_3++) {
                int pv_pp = tile_3 & 1;
                int pv_wait = tile_3 >> 1 & 1;
                mbarrier_wait(p_full_addr + (pv_pp) * 8, pv_wait);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int vs_4 = 0; vs_4 < 4; vs_4++) {
                    unsigned int slot_raw = base + (unsigned int)vs_4;
                    unsigned int slot = ((slot_raw >= 9) ? slot_raw - 9 : slot_raw);
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_vt_addr) >> 4) & 0x3FFF) | 0x4000000) + (slot) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_pt_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_pp) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134840336, ((first_pv) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134840336, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134840336, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem + (64 + vs_4 * 32)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134840336, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (slot) * 8);
                }
                elect_commit(pt_free_addr + (pv_pp) * 8);
                elect_commit(pv_done_addr);
                first_pv = 0;
                unsigned int base_raw = base + 4;
                base = ((base_raw >= 9) ? base_raw - 9 : base_raw);
            }
            if (my_n_tiles_3 > 0) {
                elect_commit(o_done_addr);
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp == 10) {
        { // load_warp_main
            int split_idx_4 = blockIdx.x;
            int m_tile_4 = gridDim.y - 1 - blockIdx.y;
            int b_4 = blockIdx.z;
            int q_start_4 = cum_seq_lens_q[b_4];
            int q_len_b_4 = cum_seq_lens_q[b_4 + 1] - q_start_4;
            int kv_len_4 = seq_lens[b_4];
            int rows_b_4 = q_len_b_4 * num_heads;
            int row0_4 = m_tile_4 * 32;
            int rows_left_4 = rows_b_4 - row0_4;
            int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
            int rows_valid_4 = ((rows_pos_4 > 32) ? 32 : rows_pos_4);
            int row_base_global_4 = q_start_4 * num_heads + row0_4;
            int last_row_4 = row0_4 + rows_valid_4 - 1;
            int t_last_4 = last_row_4 / num_heads;
            int kv_end_raw_4 = kv_len_4 - q_len_b_4 + t_last_4 + 1;
            int kv_end_4 = ((rows_valid_4 == 0) ? 0 : kv_end_raw_4);
            int n_pages_4 = (kv_end_4 + 64 - 1) / 64;
            int n_tiles_total_4 = (kv_end_4 + 128 - 1) / 128;
            int tiles_per_split_4 = (n_tiles_total_4 + num_split - 1) / num_split;
            int my_start_4 = split_idx_4 * tiles_per_split_4;
            int my_end_raw_4 = my_start_4 + tiles_per_split_4;
            int my_end_4 = ((my_end_raw_4 > n_tiles_total_4) ? n_tiles_total_4 : my_end_raw_4);
            int my_n_raw_4 = my_end_4 - my_start_4;
            int my_n_tiles_4 = ((my_n_raw_4 < 0) ? 0 : my_n_raw_4);
            int pt_base_4 = b_4 * max_pages_per_seq;
            unsigned int load_stage = 0;
            unsigned int load_phase = 1;
            unsigned int rope_stage = 0;
            unsigned int rope_phase = 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 18432);
                #pragma unroll
                for (int st_1 = 0; st_1 < 4; st_1++) {
                    tma_2d_gmem2smem(smem_q_addr + (unsigned int)(st_1 * 4096), (&tmap_q), st_1 * 128, row_base_global_4, q_full_addr);
                }
                tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 512, row_base_global_4, q_full_addr);
            }
            if (my_n_tiles_4 > 0) {
                if (elect_sync()) {
                    int p0 = my_start_4 * 2;
                    int p1_raw = p0 + 1;
                    int p1 = ((p1_raw >= n_pages_4) ? p0 : p1_raw);
                    int g0 = page_table[pt_base_4 + p0];
                    int g1 = page_table[pt_base_4 + p1];
                    #pragma unroll
                    for (int st_2 = 0; st_2 < 4; st_2++) {
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_2 * 128)), "r"((int)(g0 * 64)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_2 * 128)), "r"((int)(g1 * 64)) : "memory");
                    }
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g0 * 64)) : "memory");
                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g1 * 64)) : "memory");
                }
            }
            #pragma unroll 1
            for (int tile_4 = 0; tile_4 < my_n_tiles_4; tile_4++) {
                int p0_1 = (my_start_4 + tile_4) * 2;
                int p1_raw_1 = p0_1 + 1;
                int p1_1 = ((p1_raw_1 >= n_pages_4) ? p0_1 : p1_raw_1);
                int g0_1 = page_table[pt_base_4 + p0_1];
                int g1_1 = page_table[pt_base_4 + p1_1];
                #pragma unroll
                for (int st_3 = 0; st_3 < 4; st_3++) {
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        tma_2d_gmem2smem(smem_k_addr + load_stage * 16384, (&tmap_k), st_3 * 128, g0_1 * 64, kv_full_addr + (load_stage) * 8);
                        tma_2d_gmem2smem(smem_k_addr + load_stage * 16384 + 8192, (&tmap_k), st_3 * 128, g1_1 * 64, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 9) { load_stage = 0; load_phase ^= 1; }
                }
                mbarrier_wait(rope_empty_addr + (rope_stage) * 8, rope_phase);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(rope_full_addr + (rope_stage) * 8, 8192);
                    tma_2d_gmem2smem(smem_kr_addr + rope_stage * 8192, (&tmap_kr), 512, g0_1 * 64, rope_full_addr + (rope_stage) * 8);
                    tma_2d_gmem2smem(smem_kr_addr + rope_stage * 8192 + 4096, (&tmap_kr), 512, g1_1 * 64, rope_full_addr + (rope_stage) * 8);
                }
                rope_stage += 1;
                if (rope_stage == 2) { rope_stage = 0; rope_phase ^= 1; }
                int pf_tile = tile_4 + 1;
                if (pf_tile < my_n_tiles_4) {
                    if (elect_sync()) {
                        int p0_0 = (my_start_4 + pf_tile) * 2;
                        int p1_raw_1_1 = p0_0 + 1;
                        int p1_2 = ((p1_raw_1_1 >= n_pages_4) ? p0_0 : p1_raw_1_1);
                        int g0_3 = page_table[pt_base_4 + p0_0];
                        int g1_4 = page_table[pt_base_4 + p1_2];
                        #pragma unroll
                        for (int st_4 = 0; st_4 < 4; st_4++) {
                            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_4 * 128)), "r"((int)(g0_3 * 64)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_k))), "r"((int)(st_4 * 128)), "r"((int)(g1_4 * 64)) : "memory");
                        }
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g0_3 * 64)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&tmap_kr))), "r"((int)(512)), "r"((int)(g1_4 * 64)) : "memory");
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
