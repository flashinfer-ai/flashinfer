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
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBS_0_OFFSET 160
#define TMEM_PROBS_1_OFFSET 32
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_KV_STAGES 8
#define SMEM_SCALES_OFF 1024
#define SMEM_SCALES_STAGE_BYTES 4096
#define SMEM_SCALES_STRIDE 4096
#define SMEM_SMEM_Q0_OFF 5120
#define SMEM_SMEM_Q0_STAGE_BYTES 16384
#define SMEM_SMEM_Q0_STRIDE 16384
#define SMEM_SMEM_Q1_OFF 21504
#define SMEM_SMEM_Q1_STAGE_BYTES 16384
#define SMEM_SMEM_Q1_STRIDE 16384
#define SMEM_SMEM_O_OFF 103424
#define SMEM_SMEM_O_STAGE_BYTES 16384
#define SMEM_SMEM_O_STRIDE 16384
#define SMEM_SMEM_KV_OFF 37888
#define SMEM_SMEM_KV_STAGE_BYTES 8192
#define SMEM_SMEM_KV_STRIDE 8192
#define SMEM_SMEM_V_OFF 37888
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 8192
#define SMEM_TOTAL 168960
#if __CUDA_ARCH__ == 1030
#define USE_TMEM_LD_RED 1
#else
#define USE_TMEM_LD_RED 0
#endif
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
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
#endif
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

#if __CUDA_ARCH__ == 1030
#else

__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}
#endif



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

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_varlen_fp8_attention_529a83afa416443c4fb4(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, __nv_bfloat16* __restrict__ O_raw, int* __restrict__ unit_table, __half* __restrict__ partial_O, float* __restrict__ partial_ML, float* __restrict__ q_descale, float* __restrict__ k_descale, float* __restrict__ v_descale, unsigned int total_tiles, int num_heads, float softmax_scale_log2)
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
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 160)
    #define p_full_2_addr (mbar_base + 176)
    #define scale_full_addr (mbar_base + 192)
    #define scale_empty_addr (mbar_base + 208)
    #define s_read_addr (mbar_base + 224)
    #define o_full_addr (mbar_base + 240)
    #define pv0_done_addr (mbar_base + 256)
    #define pv1_done_addr (mbar_base + 264)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* scales = reinterpret_cast<float*>(smem_raw + 1024);
    const int scales_addr = smem + 1024;
    uint8_t* smem_q0 = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_q0_addr = smem + 5120;
    uint8_t* smem_q1 = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_q1_addr = smem + 21504;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int smem_o_addr = smem + 103424;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_kv_addr = smem + 37888;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v_addr = smem + 37888;

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv' ---
            // kv_full: 8 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            // kv_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 160, 512);
            mbarrier_init(smem + 168, 512);
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            // scale_full: 2 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // scale_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 208, 256);
            mbarrier_init(smem + 216, 256);
            // s_read: 2 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // pv0_done: 1 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            // pv1_done: 1 barriers, init_count=128
            mbarrier_init(smem + 264, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 272);
    if (warp == 0) {
        int _tmem_hold = smem + 272;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_probs_0 = taddr + 160;
    const int tmem_probs_1 = taddr + 32;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            float exp_coef = softmax_scale_log2 * q_descale[0] * k_descale[0];
            unsigned int stage = make_warp_uniform(warp / 4);
            unsigned int other_stage = make_warp_uniform(1 - stage);
            uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
            int sync_group = make_warp_uniform(warp_group_idx);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(other_stage * 128 + 32);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            int nxt_doc_begin;
            int nxt_doc_len;
            int nxt_packed;
            int nxt_kv_words;
            int nxt_ws_slot;
            int nxt_q_half;
            int rec = cluster_id * 8;
            int doc_begin = unit_table[rec];
            int doc_len = unit_table[rec + 1];
            int packed = unit_table[rec + 2];
            int kv_words = unit_table[rec + 3];
            int ws_slot = unit_table[rec + 4];
            int q_half = unit_table[rec + 5];
            nxt_doc_begin = doc_begin;
            nxt_doc_len = doc_len;
            nxt_packed = packed;
            nxt_kv_words = kv_words;
            nxt_ws_slot = ws_slot;
            nxt_q_half = q_half;
            int scale_buf = 0;
            unsigned int _phase_s_read = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_pv0_done_0 = 0;
            unsigned int _phase_pv1_done_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int doc_begin_0 = nxt_doc_begin;
                int doc_len_1 = nxt_doc_len;
                int ws_slot_2 = nxt_ws_slot;
                int head = nxt_packed >> 16;
                int c = nxt_packed & 65535;
                int n_begin = nxt_kv_words >> 16;
                int num_n_blocks = nxt_kv_words & 65535;
                int unit_row0 = c * 512 + (nxt_q_half >> 1) * 256;
                int full_rows = ((doc_len_1 - unit_row0 > 256) ? 1 : 0);
                int two_stages = ((nxt_q_half == 0) ? full_rows : 0);
                unsigned int nxt_tile = tile_idx + num_clusters;
                int rec_3 = nxt_tile * 8;
                int doc_begin_4 = unit_table[rec_3];
                int doc_len_5 = unit_table[rec_3 + 1];
                int packed_6 = unit_table[rec_3 + 2];
                int kv_words_7 = unit_table[rec_3 + 3];
                int ws_slot_8 = unit_table[rec_3 + 4];
                int q_half_9 = unit_table[rec_3 + 5];
                nxt_doc_begin = doc_begin_4;
                nxt_doc_len = doc_len_5;
                nxt_packed = packed_6;
                nxt_kv_words = kv_words_7;
                nxt_ws_slot = ws_slot_8;
                nxt_q_half = q_half_9;
                int active = ((stage == 0) ? 1 : two_stages);
                if (active == 1) {
                    int tail_base = doc_len_1 - n_begin * BLOCK_N;
                    unsigned int last_n_iter = (unsigned int)(num_n_blocks - 1);
                    float row_max = -CUDART_INF_F;
                    float row_sum = 0.0f;
                    if (stage == 1) {
                        mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                        _phase_s_read ^= 1;
                    }
                    #pragma unroll 1
                    for (unsigned int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                        int n_block = (unsigned int)(num_n_blocks - 1) - n_iter;
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
#else
                        mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 1000000);
#endif
                        _phase_s_full ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int s_addr = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                        float sv[128];
                        float tile_max = -CUDART_INF_F;
                        {
#if __CUDA_ARCH__ == 1030
                            float tile_max_lo = -CUDART_INF_F;
                            float tile_max_hi = -CUDART_INF_F;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#else
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1030
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(tile_max_lo)
                                : "r"((unsigned int)tmem_scores + stage * 128 + (unsigned int)(warp % 4 * 32 << 16)));
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#else
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                                : "r"(s_addr));
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1030
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(tile_max_hi)
                                : "r"((unsigned int)tmem_scores + stage * 128 + 64 + (unsigned int)(warp % 4 * 32 << 16)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            tile_max = max_noftz(tile_max_lo, tile_max_hi);
#else
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                                : "r"(s_addr + 32));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95])
                                : "r"(s_addr + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127])
                                : "r"(s_addr + 96));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _reg_reduce_max2_1 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_1);
                            float _reg_array_reduce_0 = row_max_reduce(_reg_reduce_max2_1);
                            tile_max = _reg_array_reduce_0;
#endif
                        }
                        int tail_valid = tail_base - n_block * BLOCK_N;
                        if (tail_valid < BLOCK_N) {
#if __CUDA_ARCH__ == 1030
                            uint32_t _thresh_mask_0;
                            {
                                int _lim_1 = tail_valid;
                                if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                                else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                                }
                            }
                            if (!(_thresh_mask_0 & (1u << 0))) sv[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 1))) sv[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 2))) sv[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 3))) sv[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 4))) sv[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 5))) sv[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 6))) sv[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 7))) sv[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 8))) sv[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 9))) sv[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 10))) sv[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 11))) sv[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 12))) sv[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 13))) sv[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 14))) sv[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 15))) sv[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 16))) sv[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 17))) sv[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 18))) sv[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 19))) sv[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 20))) sv[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 21))) sv[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 22))) sv[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 23))) sv[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 24))) sv[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 25))) sv[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 26))) sv[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 27))) sv[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 28))) sv[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 29))) sv[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 30))) sv[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 31))) sv[31] = -CUDART_INF_F;
#else
#endif
                            uint32_t _thresh_mask_2;
                            {
#if __CUDA_ARCH__ == 1030
                                int _lim_3 = tail_valid - 32;
#else
                                int _lim_3 = tail_valid;
#endif
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
#if __CUDA_ARCH__ == 1030
                            if (!(_thresh_mask_2 & (1u << 0))) sv[32] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 1))) sv[33] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 2))) sv[34] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 3))) sv[35] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 4))) sv[36] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 5))) sv[37] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 6))) sv[38] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 7))) sv[39] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 8))) sv[40] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 9))) sv[41] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 10))) sv[42] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 11))) sv[43] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 12))) sv[44] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 13))) sv[45] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 14))) sv[46] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 15))) sv[47] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 16))) sv[48] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 17))) sv[49] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 18))) sv[50] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 19))) sv[51] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 20))) sv[52] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 21))) sv[53] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 22))) sv[54] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 23))) sv[55] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 24))) sv[56] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 25))) sv[57] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 26))) sv[58] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 27))) sv[59] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 28))) sv[60] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 29))) sv[61] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 30))) sv[62] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 31))) sv[63] = -CUDART_INF_F;
#else
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
#endif
                            uint32_t _thresh_mask_4;
                            {
#if __CUDA_ARCH__ == 1030
                                int _lim_5 = tail_valid - 64;
#else
                                int _lim_5 = tail_valid - 32;
#endif
                                if (_lim_5 <= 0) { _thresh_mask_4 = 0u; }
                                else if (_lim_5 >= 32) { _thresh_mask_4 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_4) : "r"(_lim_5));
                                }
                            }
#if __CUDA_ARCH__ == 1030
                            if (!(_thresh_mask_4 & (1u << 0))) sv[64] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 1))) sv[65] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 2))) sv[66] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 3))) sv[67] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 4))) sv[68] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 5))) sv[69] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 6))) sv[70] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 7))) sv[71] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 8))) sv[72] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 9))) sv[73] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 10))) sv[74] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 11))) sv[75] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 12))) sv[76] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 13))) sv[77] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 14))) sv[78] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 15))) sv[79] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 16))) sv[80] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 17))) sv[81] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 18))) sv[82] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 19))) sv[83] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 20))) sv[84] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 21))) sv[85] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 22))) sv[86] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 23))) sv[87] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 24))) sv[88] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 25))) sv[89] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 26))) sv[90] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 27))) sv[91] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 28))) sv[92] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 29))) sv[93] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 30))) sv[94] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 31))) sv[95] = -CUDART_INF_F;
#else
                            if (!(_thresh_mask_4 & (1u << 0))) sv[32] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 1))) sv[33] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 2))) sv[34] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 3))) sv[35] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 4))) sv[36] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 5))) sv[37] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 6))) sv[38] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 7))) sv[39] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 8))) sv[40] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 9))) sv[41] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 10))) sv[42] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 11))) sv[43] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 12))) sv[44] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 13))) sv[45] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 14))) sv[46] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 15))) sv[47] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 16))) sv[48] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 17))) sv[49] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 18))) sv[50] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 19))) sv[51] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 20))) sv[52] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 21))) sv[53] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 22))) sv[54] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 23))) sv[55] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 24))) sv[56] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 25))) sv[57] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 26))) sv[58] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 27))) sv[59] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 28))) sv[60] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 29))) sv[61] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 30))) sv[62] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 31))) sv[63] = -CUDART_INF_F;
#endif
                            uint32_t _thresh_mask_6;
                            {
#if __CUDA_ARCH__ == 1030
                                int _lim_7 = tail_valid - 96;
#else
                                int _lim_7 = tail_valid - 64;
#endif
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
#if __CUDA_ARCH__ == 1030
                            if (!(_thresh_mask_6 & (1u << 0))) sv[96] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 1))) sv[97] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 2))) sv[98] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 3))) sv[99] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 4))) sv[100] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 5))) sv[101] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 6))) sv[102] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 7))) sv[103] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 8))) sv[104] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 9))) sv[105] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 10))) sv[106] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 11))) sv[107] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 12))) sv[108] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 13))) sv[109] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 14))) sv[110] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 15))) sv[111] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 16))) sv[112] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 17))) sv[113] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 18))) sv[114] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 19))) sv[115] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 20))) sv[116] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 21))) sv[117] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 22))) sv[118] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 23))) sv[119] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 24))) sv[120] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 25))) sv[121] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 26))) sv[122] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 27))) sv[123] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 28))) sv[124] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 29))) sv[125] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 30))) sv[126] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 31))) sv[127] = -CUDART_INF_F;
                            float2 _reg_reduce_max2_9 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_9);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_9);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_9);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_9);
                            float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                            tile_max = _reg_array_reduce_8;
#else
                            if (!(_thresh_mask_6 & (1u << 0))) sv[64] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 1))) sv[65] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 2))) sv[66] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 3))) sv[67] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 4))) sv[68] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 5))) sv[69] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 6))) sv[70] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 7))) sv[71] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 8))) sv[72] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 9))) sv[73] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 10))) sv[74] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 11))) sv[75] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 12))) sv[76] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 13))) sv[77] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 14))) sv[78] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 15))) sv[79] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 16))) sv[80] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 17))) sv[81] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 18))) sv[82] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 19))) sv[83] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 20))) sv[84] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 21))) sv[85] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 22))) sv[86] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 23))) sv[87] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 24))) sv[88] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 25))) sv[89] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 26))) sv[90] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 27))) sv[91] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 28))) sv[92] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 29))) sv[93] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 30))) sv[94] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 31))) sv[95] = -CUDART_INF_F;
                            uint32_t _thresh_mask_8;
                            {
                                int _lim_9 = tail_valid - 96;
                                if (_lim_9 <= 0) { _thresh_mask_8 = 0u; }
                                else if (_lim_9 >= 32) { _thresh_mask_8 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_8) : "r"(_lim_9));
                                }
                            }
                            if (!(_thresh_mask_8 & (1u << 0))) sv[96] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 1))) sv[97] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 2))) sv[98] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 3))) sv[99] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 4))) sv[100] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 5))) sv[101] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 6))) sv[102] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 7))) sv[103] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 8))) sv[104] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 9))) sv[105] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 10))) sv[106] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 11))) sv[107] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 12))) sv[108] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 13))) sv[109] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 14))) sv[110] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 15))) sv[111] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 16))) sv[112] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 17))) sv[113] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 18))) sv[114] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 19))) sv[115] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 20))) sv[116] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 21))) sv[117] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 22))) sv[118] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 23))) sv[119] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 24))) sv[120] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 25))) sv[121] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 26))) sv[122] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 27))) sv[123] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 28))) sv[124] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 29))) sv[125] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 30))) sv[126] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 31))) sv[127] = -CUDART_INF_F;
                            float2 _reg_reduce_max2_11 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_11);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_11);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_11);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_11);
                            float _reg_array_reduce_10 = row_max_reduce(_reg_reduce_max2_11);
                            tile_max = _reg_array_reduce_10;
#endif
                        }
                        if (two_stages == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(s_read_addr + (stage) * 8);
                        }
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * exp_coef;
                        float acc_scale_log2 = __fmaf_rn(row_max, exp_coef, -new_max_scaled);
                        float acc_scale;
                        if (acc_scale_log2 >= -8.0f) {
                            safe_max = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                            acc_scale = 1.0f;
                            new_max_scaled = safe_max * exp_coef;
                        } else {
#if __CUDA_ARCH__ == 1030
                            float _exp2_10 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_10 : 1.0f);
#else
                            float _exp2_12 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_12 : 1.0f);
#endif
                            row_max = new_max;
                        }
                        float neg_max_scaled = -new_max_scaled;
                        float scale_value[1];
                        scale_value[0] = acc_scale;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(s_addr), "f"(scale_value[0]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#else
#endif
                        mbarrier_arrive(scale_full_addr + (stage) * 8);
                        if (tail_valid <= 32) {
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_11 = {exp_coef, exp_coef};
                            const float2 _fma_c2_12 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_11, _fma_c2_12);
#else
#endif
                            const float2 _fma_b2_13 = {exp_coef, exp_coef};
                            const float2 _fma_c2_14 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_13, _fma_c2_14);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_13, _fma_c2_14);
#endif
                            const float2 _fma_b2_15 = {exp_coef, exp_coef};
                            const float2 _fma_c2_16 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_15, _fma_c2_16);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_15, _fma_c2_16);
#endif
                            const float2 _fma_b2_17 = {exp_coef, exp_coef};
                            const float2 _fma_c2_18 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_17, _fma_c2_18);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_17, _fma_c2_18);
#endif
                            const float2 _fma_b2_19 = {exp_coef, exp_coef};
                            const float2 _fma_c2_20 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_19, _fma_c2_20);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_19, _fma_c2_20);
#endif
                            const float2 _fma_b2_21 = {exp_coef, exp_coef};
                            const float2 _fma_c2_22 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_21, _fma_c2_22);
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
                            }
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_21, _fma_c2_22);
#endif
                            const float2 _fma_b2_23 = {exp_coef, exp_coef};
                            const float2 _fma_c2_24 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_23, _fma_c2_24);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_23, _fma_c2_24);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 2] = approx_exp2(sv[_le + 2]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_25 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                                    sv[_le*2] = _exp2_pair_25.x;
                                    sv[_le*2 + 1] = _exp2_pair_25.y;
                                } else {
                                    sv[_le*2] = approx_exp2(sv[_le*2]);
                                    sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_25 = {exp_coef, exp_coef};
                            const float2 _fma_c2_26 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_26 = {exp_coef, exp_coef};
                            const float2 _fma_c2_27 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_25, _fma_c2_26);
                            float2 psum0s = make_float2(sv[0], sv[1]);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_26, _fma_c2_27);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 4] = approx_exp2(sv[_le + 4]);
                            }
                            const float2 _fma_b2_27 = {exp_coef, exp_coef};
                            const float2 _fma_c2_28 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_27, _fma_c2_28);
                            psum0s = add_f32x2(psum0s, make_float2(sv[2], sv[3]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 6] = approx_exp2(sv[_le + 6]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_28 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 2], sv[_le*2 + 1 + 2]));
                                    sv[_le*2 + 2] = _exp2_pair_28.x;
                                    sv[_le*2 + 1 + 2] = _exp2_pair_28.y;
                                } else {
                                    sv[_le*2 + 2] = approx_exp2(sv[_le*2 + 2]);
                                    sv[_le*2 + 1 + 2] = approx_exp2(sv[_le*2 + 1 + 2]);
                                }
#endif
                            }
                            const float2 _fma_b2_29 = {exp_coef, exp_coef};
                            const float2 _fma_c2_30 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_29, _fma_c2_30);
                            psum0s = add_f32x2(psum0s, make_float2(sv[4], sv[5]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_29, _fma_c2_30);
                            float2 psum0s = make_float2(sv[0], sv[1]);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 8] = approx_exp2(sv[_le + 8]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_31 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 4], sv[_le*2 + 1 + 4]));
                                    sv[_le*2 + 4] = _exp2_pair_31.x;
                                    sv[_le*2 + 1 + 4] = _exp2_pair_31.y;
                                } else {
                                    sv[_le*2 + 4] = approx_exp2(sv[_le*2 + 4]);
                                    sv[_le*2 + 1 + 4] = approx_exp2(sv[_le*2 + 1 + 4]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_31 = {exp_coef, exp_coef};
                            const float2 _fma_c2_32 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_32 = {exp_coef, exp_coef};
                            const float2 _fma_c2_33 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_31, _fma_c2_32);
                            psum0s = add_f32x2(psum0s, make_float2(sv[6], sv[7]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_32, _fma_c2_33);
                            psum0s = add_f32x2(psum0s, make_float2(sv[2], sv[3]));
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 10] = approx_exp2(sv[_le + 10]);
                            }
                            const float2 _fma_b2_33 = {exp_coef, exp_coef};
                            const float2 _fma_c2_34 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_33, _fma_c2_34);
                            psum0s = add_f32x2(psum0s, make_float2(sv[8], sv[9]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 12] = approx_exp2(sv[_le + 12]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_34 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 6], sv[_le*2 + 1 + 6]));
                                    sv[_le*2 + 6] = _exp2_pair_34.x;
                                    sv[_le*2 + 1 + 6] = _exp2_pair_34.y;
                                } else {
                                    sv[_le*2 + 6] = approx_exp2(sv[_le*2 + 6]);
                                    sv[_le*2 + 1 + 6] = approx_exp2(sv[_le*2 + 1 + 6]);
                                }
#endif
                            }
                            const float2 _fma_b2_35 = {exp_coef, exp_coef};
                            const float2 _fma_c2_36 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_35, _fma_c2_36);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_35, _fma_c2_36);
                            psum0s = add_f32x2(psum0s, make_float2(sv[4], sv[5]));
                            #pragma unroll
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_37 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 8], sv[_le*2 + 1 + 8]));
                                    sv[_le*2 + 8] = _exp2_pair_37.x;
                                    sv[_le*2 + 1 + 8] = _exp2_pair_37.y;
                                } else {
                                    sv[_le*2 + 8] = approx_exp2(sv[_le*2 + 8]);
                                    sv[_le*2 + 1 + 8] = approx_exp2(sv[_le*2 + 1 + 8]);
                                }
                            }
                            const float2 _fma_b2_38 = {exp_coef, exp_coef};
                            const float2 _fma_c2_39 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_38, _fma_c2_39);
                            psum0s = add_f32x2(psum0s, make_float2(sv[6], sv[7]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 10] = approx_exp2(sv[_le + 10]);
                            }
                            const float2 _fma_b2_40 = {exp_coef, exp_coef};
                            const float2 _fma_c2_41 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_40, _fma_c2_41);
                            psum0s = add_f32x2(psum0s, make_float2(sv[8], sv[9]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 12] = approx_exp2(sv[_le + 12]);
                            }
                            const float2 _fma_b2_42 = {exp_coef, exp_coef};
                            const float2 _fma_c2_43 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_42, _fma_c2_43);
#endif
                            psum0s = add_f32x2(psum0s, make_float2(sv[10], sv[11]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 14] = approx_exp2(sv[_le + 14]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_37 = {exp_coef, exp_coef};
                            const float2 _fma_c2_38 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_44 = {exp_coef, exp_coef};
                            const float2 _fma_c2_45 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_37, _fma_c2_38);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_44, _fma_c2_45);
#endif
                            psum0s = add_f32x2(psum0s, make_float2(sv[12], sv[13]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 16] = approx_exp2(sv[_le + 16]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_39 = {exp_coef, exp_coef};
                            const float2 _fma_c2_40 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_46 = {exp_coef, exp_coef};
                            const float2 _fma_c2_47 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_39, _fma_c2_40);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_46, _fma_c2_47);
#endif
                            psum0s = add_f32x2(psum0s, make_float2(sv[14], sv[15]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 18] = approx_exp2(sv[_le + 18]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_41 = {exp_coef, exp_coef};
                            const float2 _fma_c2_42 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_48 = {exp_coef, exp_coef};
                            const float2 _fma_c2_49 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_41, _fma_c2_42);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_48, _fma_c2_49);
#endif
                            psum0s = add_f32x2(psum0s, make_float2(sv[16], sv[17]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 20] = approx_exp2(sv[_le + 20]);
                            }
                            psum0s = add_f32x2(psum0s, make_float2(sv[18], sv[19]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 22] = approx_exp2(sv[_le + 22]);
                            }
                            psum0s = add_f32x2(psum0s, make_float2(sv[20], sv[21]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 24] = approx_exp2(sv[_le + 24]);
                            }
                            psum0s = add_f32x2(psum0s, make_float2(sv[22], sv[23]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 26] = approx_exp2(sv[_le + 26]);
                            }
                            psum0s = add_f32x2(psum0s, make_float2(sv[24], sv[25]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 28] = approx_exp2(sv[_le + 28]);
                            }
                            psum0s = add_f32x2(psum0s, make_float2(sv[26], sv[27]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 30] = approx_exp2(sv[_le + 30]);
                            }
                            psum0s = add_f32x2(psum0s, make_float2(sv[28], sv[29]));
                            psum0s = add_f32x2(psum0s, make_float2(sv[30], sv[31]));
                            unsigned int pk0s[8];
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
                                pk0s[0] = _packed;
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
                                pk0s[1] = _packed;
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
                                pk0s[2] = _packed;
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
                                pk0s[3] = _packed;
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
                                pk0s[4] = _packed;
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
                                pk0s[5] = _packed;
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
                                pk0s[6] = _packed;
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
                                pk0s[7] = _packed;
                            }
                            unsigned int pkz1[8] = {0};
                            unsigned int pkz2[8] = {0};
                            unsigned int pkz3[8] = {0};
                            int p_addr_s = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                            if (two_stages == 1) {
                                mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                                _phase_s_read ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (stage == 0 && two_stages == 0 && n_iter >= 1) {
                                mbarrier_wait(pv0_done_addr, _phase_pv0_done_0);
                                _phase_pv0_done_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (stage == 1 && two_stages == 1 && n_iter >= 1 && n_iter == last_n_iter) {
                                mbarrier_wait(pv1_done_addr, _phase_pv1_done_0);
                                _phase_pv1_done_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            tmem_st_x8_u32(p_addr_s, (const uint32_t*)pk0s);
                            tmem_st_x8_u32(p_addr_s + 8, (const uint32_t*)pkz1);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                            tmem_st_x8_u32(p_addr_s + 16, (const uint32_t*)pkz2);
                            tmem_st_x8_u32(p_addr_s + 24, (const uint32_t*)pkz3);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                            float2 total_s = psum0s;
                            row_sum = __fmaf_rn(row_sum, acc_scale, total_s.x) + total_s.y;
                        } else {
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_43 = {exp_coef, exp_coef};
                            const float2 _fma_c2_44 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_50 = {exp_coef, exp_coef};
                            const float2 _fma_c2_51 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_43, _fma_c2_44);
                            const float2 _fma_b2_45 = {exp_coef, exp_coef};
                            const float2 _fma_c2_46 = {neg_max_scaled, neg_max_scaled};
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_50, _fma_c2_51);
                            const float2 _fma_b2_52 = {exp_coef, exp_coef};
                            const float2 _fma_c2_53 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_45, _fma_c2_46);
                            const float2 _fma_b2_47 = {exp_coef, exp_coef};
                            const float2 _fma_c2_48 = {neg_max_scaled, neg_max_scaled};
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_52, _fma_c2_53);
                            const float2 _fma_b2_54 = {exp_coef, exp_coef};
                            const float2 _fma_c2_55 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_47, _fma_c2_48);
                            const float2 _fma_b2_49 = {exp_coef, exp_coef};
                            const float2 _fma_c2_50 = {neg_max_scaled, neg_max_scaled};
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_54, _fma_c2_55);
                            const float2 _fma_b2_56 = {exp_coef, exp_coef};
                            const float2 _fma_c2_57 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_49, _fma_c2_50);
                            const float2 _fma_b2_51 = {exp_coef, exp_coef};
                            const float2 _fma_c2_52 = {neg_max_scaled, neg_max_scaled};
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_56, _fma_c2_57);
                            const float2 _fma_b2_58 = {exp_coef, exp_coef};
                            const float2 _fma_c2_59 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_51, _fma_c2_52);
                            const float2 _fma_b2_53 = {exp_coef, exp_coef};
                            const float2 _fma_c2_54 = {neg_max_scaled, neg_max_scaled};
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_58, _fma_c2_59);
                            const float2 _fma_b2_60 = {exp_coef, exp_coef};
                            const float2 _fma_c2_61 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_53, _fma_c2_54);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_60, _fma_c2_61);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
                            }
                            const float2 _fma_b2_55 = {exp_coef, exp_coef};
                            const float2 _fma_c2_56 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_55, _fma_c2_56);
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 2] = approx_exp2(sv[_le + 2]);
                            }
                            const float2 _fma_b2_57 = {exp_coef, exp_coef};
                            const float2 _fma_c2_58 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_57, _fma_c2_58);
                            float2 psum0f = make_float2(sv[0], sv[1]);
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 4] = approx_exp2(sv[_le + 4]);
                            }
                            const float2 _fma_b2_59 = {exp_coef, exp_coef};
                            const float2 _fma_c2_60 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_59, _fma_c2_60);
                            psum0f = add_f32x2(psum0f, make_float2(sv[2], sv[3]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 6] = approx_exp2(sv[_le + 6]);
                            }
                            const float2 _fma_b2_61 = {exp_coef, exp_coef};
                            const float2 _fma_c2_62 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_61, _fma_c2_62);
                            psum0f = add_f32x2(psum0f, make_float2(sv[4], sv[5]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 8] = approx_exp2(sv[_le + 8]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_62 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                                    sv[_le*2] = _exp2_pair_62.x;
                                    sv[_le*2 + 1] = _exp2_pair_62.y;
                                } else {
                                    sv[_le*2] = approx_exp2(sv[_le*2]);
                                    sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                                }
#endif
                            }
                            const float2 _fma_b2_63 = {exp_coef, exp_coef};
                            const float2 _fma_c2_64 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_63, _fma_c2_64);
                            psum0f = add_f32x2(psum0f, make_float2(sv[6], sv[7]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_63, _fma_c2_64);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 10] = approx_exp2(sv[_le + 10]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_65 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 2], sv[_le*2 + 1 + 2]));
                                    sv[_le*2 + 2] = _exp2_pair_65.x;
                                    sv[_le*2 + 1 + 2] = _exp2_pair_65.y;
                                } else {
                                    sv[_le*2 + 2] = approx_exp2(sv[_le*2 + 2]);
                                    sv[_le*2 + 1 + 2] = approx_exp2(sv[_le*2 + 1 + 2]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_65 = {exp_coef, exp_coef};
                            const float2 _fma_c2_66 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_66 = {exp_coef, exp_coef};
                            const float2 _fma_c2_67 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_65, _fma_c2_66);
                            psum0f = add_f32x2(psum0f, make_float2(sv[8], sv[9]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_66, _fma_c2_67);
                            float2 psum0f = make_float2(sv[0], sv[1]);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 12] = approx_exp2(sv[_le + 12]);
                            }
                            const float2 _fma_b2_67 = {exp_coef, exp_coef};
                            const float2 _fma_c2_68 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_67, _fma_c2_68);
                            psum0f = add_f32x2(psum0f, make_float2(sv[10], sv[11]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 14] = approx_exp2(sv[_le + 14]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_68 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 4], sv[_le*2 + 1 + 4]));
                                    sv[_le*2 + 4] = _exp2_pair_68.x;
                                    sv[_le*2 + 1 + 4] = _exp2_pair_68.y;
                                } else {
                                    sv[_le*2 + 4] = approx_exp2(sv[_le*2 + 4]);
                                    sv[_le*2 + 1 + 4] = approx_exp2(sv[_le*2 + 1 + 4]);
                                }
#endif
                            }
                            const float2 _fma_b2_69 = {exp_coef, exp_coef};
                            const float2 _fma_c2_70 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_69, _fma_c2_70);
                            psum0f = add_f32x2(psum0f, make_float2(sv[12], sv[13]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_69, _fma_c2_70);
                            psum0f = add_f32x2(psum0f, make_float2(sv[2], sv[3]));
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 16] = approx_exp2(sv[_le + 16]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_71 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 6], sv[_le*2 + 1 + 6]));
                                    sv[_le*2 + 6] = _exp2_pair_71.x;
                                    sv[_le*2 + 1 + 6] = _exp2_pair_71.y;
                                } else {
                                    sv[_le*2 + 6] = approx_exp2(sv[_le*2 + 6]);
                                    sv[_le*2 + 1 + 6] = approx_exp2(sv[_le*2 + 1 + 6]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_71 = {exp_coef, exp_coef};
                            const float2 _fma_c2_72 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_72 = {exp_coef, exp_coef};
                            const float2 _fma_c2_73 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_71, _fma_c2_72);
                            psum0f = add_f32x2(psum0f, make_float2(sv[14], sv[15]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_72, _fma_c2_73);
                            psum0f = add_f32x2(psum0f, make_float2(sv[4], sv[5]));
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 18] = approx_exp2(sv[_le + 18]);
                            }
                            const float2 _fma_b2_73 = {exp_coef, exp_coef};
                            const float2 _fma_c2_74 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_73, _fma_c2_74);
                            psum0f = add_f32x2(psum0f, make_float2(sv[16], sv[17]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 20] = approx_exp2(sv[_le + 20]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_74 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 8], sv[_le*2 + 1 + 8]));
                                    sv[_le*2 + 8] = _exp2_pair_74.x;
                                    sv[_le*2 + 1 + 8] = _exp2_pair_74.y;
                                } else {
                                    sv[_le*2 + 8] = approx_exp2(sv[_le*2 + 8]);
                                    sv[_le*2 + 1 + 8] = approx_exp2(sv[_le*2 + 1 + 8]);
                                }
#endif
                            }
                            const float2 _fma_b2_75 = {exp_coef, exp_coef};
                            const float2 _fma_c2_76 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_75, _fma_c2_76);
                            psum0f = add_f32x2(psum0f, make_float2(sv[18], sv[19]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_75, _fma_c2_76);
                            psum0f = add_f32x2(psum0f, make_float2(sv[6], sv[7]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 22] = approx_exp2(sv[_le + 22]);
#else
                                sv[_le + 10] = approx_exp2(sv[_le + 10]);
#endif
                            }
                            const float2 _fma_b2_77 = {exp_coef, exp_coef};
                            const float2 _fma_c2_78 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 34))[_lf], _fma_b2_77, _fma_c2_78);
                            psum0f = add_f32x2(psum0f, make_float2(sv[20], sv[21]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_77, _fma_c2_78);
                            psum0f = add_f32x2(psum0f, make_float2(sv[8], sv[9]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 24] = approx_exp2(sv[_le + 24]);
#else
                                sv[_le + 12] = approx_exp2(sv[_le + 12]);
#endif
                            }
                            const float2 _fma_b2_79 = {exp_coef, exp_coef};
                            const float2 _fma_c2_80 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 36))[_lf], _fma_b2_79, _fma_c2_80);
                            psum0f = add_f32x2(psum0f, make_float2(sv[22], sv[23]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_79, _fma_c2_80);
                            psum0f = add_f32x2(psum0f, make_float2(sv[10], sv[11]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 26] = approx_exp2(sv[_le + 26]);
#else
                                sv[_le + 14] = approx_exp2(sv[_le + 14]);
#endif
                            }
                            const float2 _fma_b2_81 = {exp_coef, exp_coef};
                            const float2 _fma_c2_82 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 38))[_lf], _fma_b2_81, _fma_c2_82);
                            psum0f = add_f32x2(psum0f, make_float2(sv[24], sv[25]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_81, _fma_c2_82);
                            psum0f = add_f32x2(psum0f, make_float2(sv[12], sv[13]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 28] = approx_exp2(sv[_le + 28]);
#else
                                sv[_le + 16] = approx_exp2(sv[_le + 16]);
#endif
                            }
                            const float2 _fma_b2_83 = {exp_coef, exp_coef};
                            const float2 _fma_c2_84 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 40))[_lf], _fma_b2_83, _fma_c2_84);
                            psum0f = add_f32x2(psum0f, make_float2(sv[26], sv[27]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_83, _fma_c2_84);
                            psum0f = add_f32x2(psum0f, make_float2(sv[14], sv[15]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 30] = approx_exp2(sv[_le + 30]);
#else
                                sv[_le + 18] = approx_exp2(sv[_le + 18]);
#endif
                            }
                            const float2 _fma_b2_85 = {exp_coef, exp_coef};
                            const float2 _fma_c2_86 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 42))[_lf], _fma_b2_85, _fma_c2_86);
                            psum0f = add_f32x2(psum0f, make_float2(sv[28], sv[29]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_85, _fma_c2_86);
                            psum0f = add_f32x2(psum0f, make_float2(sv[16], sv[17]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 32] = approx_exp2(sv[_le + 32]);
#else
                                sv[_le + 20] = approx_exp2(sv[_le + 20]);
#endif
                            }
                            const float2 _fma_b2_87 = {exp_coef, exp_coef};
                            const float2 _fma_c2_88 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 44))[_lf], _fma_b2_87, _fma_c2_88);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_87, _fma_c2_88);
                            psum0f = add_f32x2(psum0f, make_float2(sv[18], sv[19]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 22] = approx_exp2(sv[_le + 22]);
                            }
                            const float2 _fma_b2_89 = {exp_coef, exp_coef};
                            const float2 _fma_c2_90 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 34))[_lf], _fma_b2_89, _fma_c2_90);
                            psum0f = add_f32x2(psum0f, make_float2(sv[20], sv[21]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 24] = approx_exp2(sv[_le + 24]);
                            }
                            const float2 _fma_b2_91 = {exp_coef, exp_coef};
                            const float2 _fma_c2_92 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 36))[_lf], _fma_b2_91, _fma_c2_92);
                            psum0f = add_f32x2(psum0f, make_float2(sv[22], sv[23]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 26] = approx_exp2(sv[_le + 26]);
                            }
                            const float2 _fma_b2_93 = {exp_coef, exp_coef};
                            const float2 _fma_c2_94 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 38))[_lf], _fma_b2_93, _fma_c2_94);
                            psum0f = add_f32x2(psum0f, make_float2(sv[24], sv[25]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 28] = approx_exp2(sv[_le + 28]);
                            }
                            const float2 _fma_b2_95 = {exp_coef, exp_coef};
                            const float2 _fma_c2_96 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 40))[_lf], _fma_b2_95, _fma_c2_96);
                            psum0f = add_f32x2(psum0f, make_float2(sv[26], sv[27]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 30] = approx_exp2(sv[_le + 30]);
                            }
                            const float2 _fma_b2_97 = {exp_coef, exp_coef};
                            const float2 _fma_c2_98 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 42))[_lf], _fma_b2_97, _fma_c2_98);
                            psum0f = add_f32x2(psum0f, make_float2(sv[28], sv[29]));
                            #pragma unroll
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_99 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 32], sv[_le*2 + 1 + 32]));
                                    sv[_le*2 + 32] = _exp2_pair_99.x;
                                    sv[_le*2 + 1 + 32] = _exp2_pair_99.y;
                                } else {
                                    sv[_le*2 + 32] = approx_exp2(sv[_le*2 + 32]);
                                    sv[_le*2 + 1 + 32] = approx_exp2(sv[_le*2 + 1 + 32]);
                                }
                            }
                            const float2 _fma_b2_100 = {exp_coef, exp_coef};
                            const float2 _fma_c2_101 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 44))[_lf], _fma_b2_100, _fma_c2_101);
#endif
                            psum0f = add_f32x2(psum0f, make_float2(sv[30], sv[31]));
                            unsigned int pk0f[8];
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
                                pk0f[0] = _packed;
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
                                pk0f[1] = _packed;
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
                                pk0f[2] = _packed;
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
                                pk0f[3] = _packed;
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
                                pk0f[4] = _packed;
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
                                pk0f[5] = _packed;
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
                                pk0f[6] = _packed;
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
                                pk0f[7] = _packed;
                            }
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 34] = approx_exp2(sv[_le + 34]);
                            }
                            const float2 _fma_b2_89 = {exp_coef, exp_coef};
                            const float2 _fma_c2_90 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 46))[_lf], _fma_b2_89, _fma_c2_90);
                            float2 psum1f = make_float2(sv[32], sv[33]);
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 36] = approx_exp2(sv[_le + 36]);
                            }
                            const float2 _fma_b2_91 = {exp_coef, exp_coef};
                            const float2 _fma_c2_92 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_91, _fma_c2_92);
                            psum1f = add_f32x2(psum1f, make_float2(sv[34], sv[35]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 38] = approx_exp2(sv[_le + 38]);
                            }
                            const float2 _fma_b2_93 = {exp_coef, exp_coef};
                            const float2 _fma_c2_94 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 50))[_lf], _fma_b2_93, _fma_c2_94);
                            psum1f = add_f32x2(psum1f, make_float2(sv[36], sv[37]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 40] = approx_exp2(sv[_le + 40]);
                            }
                            const float2 _fma_b2_95 = {exp_coef, exp_coef};
                            const float2 _fma_c2_96 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 52))[_lf], _fma_b2_95, _fma_c2_96);
                            psum1f = add_f32x2(psum1f, make_float2(sv[38], sv[39]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 42] = approx_exp2(sv[_le + 42]);
                            }
                            const float2 _fma_b2_97 = {exp_coef, exp_coef};
                            const float2 _fma_c2_98 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 54))[_lf], _fma_b2_97, _fma_c2_98);
                            psum1f = add_f32x2(psum1f, make_float2(sv[40], sv[41]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 44] = approx_exp2(sv[_le + 44]);
                            }
                            const float2 _fma_b2_99 = {exp_coef, exp_coef};
                            const float2 _fma_c2_100 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 56))[_lf], _fma_b2_99, _fma_c2_100);
                            psum1f = add_f32x2(psum1f, make_float2(sv[42], sv[43]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 46] = approx_exp2(sv[_le + 46]);
                            }
                            const float2 _fma_b2_101 = {exp_coef, exp_coef};
                            const float2 _fma_c2_102 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 58))[_lf], _fma_b2_101, _fma_c2_102);
                            psum1f = add_f32x2(psum1f, make_float2(sv[44], sv[45]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 48] = approx_exp2(sv[_le + 48]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_102 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 34], sv[_le*2 + 1 + 34]));
                                    sv[_le*2 + 34] = _exp2_pair_102.x;
                                    sv[_le*2 + 1 + 34] = _exp2_pair_102.y;
                                } else {
                                    sv[_le*2 + 34] = approx_exp2(sv[_le*2 + 34]);
                                    sv[_le*2 + 1 + 34] = approx_exp2(sv[_le*2 + 1 + 34]);
                                }
#endif
                            }
                            const float2 _fma_b2_103 = {exp_coef, exp_coef};
                            const float2 _fma_c2_104 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 60))[_lf], _fma_b2_103, _fma_c2_104);
                            psum1f = add_f32x2(psum1f, make_float2(sv[46], sv[47]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 46))[_lf], _fma_b2_103, _fma_c2_104);
                            float2 psum1f = make_float2(sv[32], sv[33]);
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 50] = approx_exp2(sv[_le + 50]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_105 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 36], sv[_le*2 + 1 + 36]));
                                    sv[_le*2 + 36] = _exp2_pair_105.x;
                                    sv[_le*2 + 1 + 36] = _exp2_pair_105.y;
                                } else {
                                    sv[_le*2 + 36] = approx_exp2(sv[_le*2 + 36]);
                                    sv[_le*2 + 1 + 36] = approx_exp2(sv[_le*2 + 1 + 36]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_105 = {exp_coef, exp_coef};
                            const float2 _fma_c2_106 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_106 = {exp_coef, exp_coef};
                            const float2 _fma_c2_107 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 62))[_lf], _fma_b2_105, _fma_c2_106);
                            psum1f = add_f32x2(psum1f, make_float2(sv[48], sv[49]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_106, _fma_c2_107);
                            psum1f = add_f32x2(psum1f, make_float2(sv[34], sv[35]));
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 52] = approx_exp2(sv[_le + 52]);
                            }
                            const float2 _fma_b2_107 = {exp_coef, exp_coef};
                            const float2 _fma_c2_108 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_107, _fma_c2_108);
                            psum1f = add_f32x2(psum1f, make_float2(sv[50], sv[51]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 54] = approx_exp2(sv[_le + 54]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_108 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 38], sv[_le*2 + 1 + 38]));
                                    sv[_le*2 + 38] = _exp2_pair_108.x;
                                    sv[_le*2 + 1 + 38] = _exp2_pair_108.y;
                                } else {
                                    sv[_le*2 + 38] = approx_exp2(sv[_le*2 + 38]);
                                    sv[_le*2 + 1 + 38] = approx_exp2(sv[_le*2 + 1 + 38]);
                                }
#endif
                            }
                            const float2 _fma_b2_109 = {exp_coef, exp_coef};
                            const float2 _fma_c2_110 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 66))[_lf], _fma_b2_109, _fma_c2_110);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 50))[_lf], _fma_b2_109, _fma_c2_110);
                            psum1f = add_f32x2(psum1f, make_float2(sv[36], sv[37]));
                            #pragma unroll
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_111 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 40], sv[_le*2 + 1 + 40]));
                                    sv[_le*2 + 40] = _exp2_pair_111.x;
                                    sv[_le*2 + 1 + 40] = _exp2_pair_111.y;
                                } else {
                                    sv[_le*2 + 40] = approx_exp2(sv[_le*2 + 40]);
                                    sv[_le*2 + 1 + 40] = approx_exp2(sv[_le*2 + 1 + 40]);
                                }
                            }
                            const float2 _fma_b2_112 = {exp_coef, exp_coef};
                            const float2 _fma_c2_113 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 52))[_lf], _fma_b2_112, _fma_c2_113);
                            psum1f = add_f32x2(psum1f, make_float2(sv[38], sv[39]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 42] = approx_exp2(sv[_le + 42]);
                            }
                            const float2 _fma_b2_114 = {exp_coef, exp_coef};
                            const float2 _fma_c2_115 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 54))[_lf], _fma_b2_114, _fma_c2_115);
                            psum1f = add_f32x2(psum1f, make_float2(sv[40], sv[41]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 44] = approx_exp2(sv[_le + 44]);
                            }
                            const float2 _fma_b2_116 = {exp_coef, exp_coef};
                            const float2 _fma_c2_117 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 56))[_lf], _fma_b2_116, _fma_c2_117);
                            psum1f = add_f32x2(psum1f, make_float2(sv[42], sv[43]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 46] = approx_exp2(sv[_le + 46]);
                            }
                            const float2 _fma_b2_118 = {exp_coef, exp_coef};
                            const float2 _fma_c2_119 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 58))[_lf], _fma_b2_118, _fma_c2_119);
                            psum1f = add_f32x2(psum1f, make_float2(sv[44], sv[45]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 48] = approx_exp2(sv[_le + 48]);
                            }
                            const float2 _fma_b2_120 = {exp_coef, exp_coef};
                            const float2 _fma_c2_121 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 60))[_lf], _fma_b2_120, _fma_c2_121);
                            psum1f = add_f32x2(psum1f, make_float2(sv[46], sv[47]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 50] = approx_exp2(sv[_le + 50]);
                            }
                            const float2 _fma_b2_122 = {exp_coef, exp_coef};
                            const float2 _fma_c2_123 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 62))[_lf], _fma_b2_122, _fma_c2_123);
                            psum1f = add_f32x2(psum1f, make_float2(sv[48], sv[49]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 52] = approx_exp2(sv[_le + 52]);
                            }
                            const float2 _fma_b2_124 = {exp_coef, exp_coef};
                            const float2 _fma_c2_125 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_124, _fma_c2_125);
                            psum1f = add_f32x2(psum1f, make_float2(sv[50], sv[51]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 54] = approx_exp2(sv[_le + 54]);
                            }
                            const float2 _fma_b2_126 = {exp_coef, exp_coef};
                            const float2 _fma_c2_127 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 66))[_lf], _fma_b2_126, _fma_c2_127);
#endif
                            psum1f = add_f32x2(psum1f, make_float2(sv[52], sv[53]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 56] = approx_exp2(sv[_le + 56]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_111 = {exp_coef, exp_coef};
                            const float2 _fma_c2_112 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_128 = {exp_coef, exp_coef};
                            const float2 _fma_c2_129 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 68))[_lf], _fma_b2_111, _fma_c2_112);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 68))[_lf], _fma_b2_128, _fma_c2_129);
#endif
                            psum1f = add_f32x2(psum1f, make_float2(sv[54], sv[55]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 58] = approx_exp2(sv[_le + 58]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_113 = {exp_coef, exp_coef};
                            const float2 _fma_c2_114 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_130 = {exp_coef, exp_coef};
                            const float2 _fma_c2_131 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 70))[_lf], _fma_b2_113, _fma_c2_114);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 70))[_lf], _fma_b2_130, _fma_c2_131);
#endif
                            psum1f = add_f32x2(psum1f, make_float2(sv[56], sv[57]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 60] = approx_exp2(sv[_le + 60]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_115 = {exp_coef, exp_coef};
                            const float2 _fma_c2_116 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_132 = {exp_coef, exp_coef};
                            const float2 _fma_c2_133 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 72))[_lf], _fma_b2_115, _fma_c2_116);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 72))[_lf], _fma_b2_132, _fma_c2_133);
#endif
                            psum1f = add_f32x2(psum1f, make_float2(sv[58], sv[59]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 62] = approx_exp2(sv[_le + 62]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_117 = {exp_coef, exp_coef};
                            const float2 _fma_c2_118 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_134 = {exp_coef, exp_coef};
                            const float2 _fma_c2_135 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 74))[_lf], _fma_b2_117, _fma_c2_118);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 74))[_lf], _fma_b2_134, _fma_c2_135);
#endif
                            psum1f = add_f32x2(psum1f, make_float2(sv[60], sv[61]));
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 64] = approx_exp2(sv[_le + 64]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_136 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 64], sv[_le*2 + 1 + 64]));
                                    sv[_le*2 + 64] = _exp2_pair_136.x;
                                    sv[_le*2 + 1 + 64] = _exp2_pair_136.y;
                                } else {
                                    sv[_le*2 + 64] = approx_exp2(sv[_le*2 + 64]);
                                    sv[_le*2 + 1 + 64] = approx_exp2(sv[_le*2 + 1 + 64]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_119 = {exp_coef, exp_coef};
                            const float2 _fma_c2_120 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_137 = {exp_coef, exp_coef};
                            const float2 _fma_c2_138 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 76))[_lf], _fma_b2_119, _fma_c2_120);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 76))[_lf], _fma_b2_137, _fma_c2_138);
#endif
                            psum1f = add_f32x2(psum1f, make_float2(sv[62], sv[63]));
                            unsigned int pk1f[8];
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
                                pk1f[0] = _packed;
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
                                pk1f[1] = _packed;
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
                                pk1f[2] = _packed;
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
                                pk1f[3] = _packed;
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
                                pk1f[4] = _packed;
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
                                pk1f[5] = _packed;
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
                                pk1f[6] = _packed;
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
                                pk1f[7] = _packed;
                            }
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 66] = approx_exp2(sv[_le + 66]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_139 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 66], sv[_le*2 + 1 + 66]));
                                    sv[_le*2 + 66] = _exp2_pair_139.x;
                                    sv[_le*2 + 1 + 66] = _exp2_pair_139.y;
                                } else {
                                    sv[_le*2 + 66] = approx_exp2(sv[_le*2 + 66]);
                                    sv[_le*2 + 1 + 66] = approx_exp2(sv[_le*2 + 1 + 66]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_121 = {exp_coef, exp_coef};
                            const float2 _fma_c2_122 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_140 = {exp_coef, exp_coef};
                            const float2 _fma_c2_141 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 78))[_lf], _fma_b2_121, _fma_c2_122);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 78))[_lf], _fma_b2_140, _fma_c2_141);
#endif
                            float2 psum2f = make_float2(sv[64], sv[65]);
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 68] = approx_exp2(sv[_le + 68]);
                            }
                            const float2 _fma_b2_123 = {exp_coef, exp_coef};
                            const float2 _fma_c2_124 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_123, _fma_c2_124);
                            psum2f = add_f32x2(psum2f, make_float2(sv[66], sv[67]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 70] = approx_exp2(sv[_le + 70]);
                            }
                            const float2 _fma_b2_125 = {exp_coef, exp_coef};
                            const float2 _fma_c2_126 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 82))[_lf], _fma_b2_125, _fma_c2_126);
                            psum2f = add_f32x2(psum2f, make_float2(sv[68], sv[69]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 72] = approx_exp2(sv[_le + 72]);
                            }
                            const float2 _fma_b2_127 = {exp_coef, exp_coef};
                            const float2 _fma_c2_128 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 84))[_lf], _fma_b2_127, _fma_c2_128);
                            psum2f = add_f32x2(psum2f, make_float2(sv[70], sv[71]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 74] = approx_exp2(sv[_le + 74]);
                            }
                            const float2 _fma_b2_129 = {exp_coef, exp_coef};
                            const float2 _fma_c2_130 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 86))[_lf], _fma_b2_129, _fma_c2_130);
                            psum2f = add_f32x2(psum2f, make_float2(sv[72], sv[73]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 76] = approx_exp2(sv[_le + 76]);
                            }
                            const float2 _fma_b2_131 = {exp_coef, exp_coef};
                            const float2 _fma_c2_132 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 88))[_lf], _fma_b2_131, _fma_c2_132);
                            psum2f = add_f32x2(psum2f, make_float2(sv[74], sv[75]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 78] = approx_exp2(sv[_le + 78]);
                            }
                            const float2 _fma_b2_133 = {exp_coef, exp_coef};
                            const float2 _fma_c2_134 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 90))[_lf], _fma_b2_133, _fma_c2_134);
                            psum2f = add_f32x2(psum2f, make_float2(sv[76], sv[77]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 80] = approx_exp2(sv[_le + 80]);
                            }
                            const float2 _fma_b2_135 = {exp_coef, exp_coef};
                            const float2 _fma_c2_136 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 92))[_lf], _fma_b2_135, _fma_c2_136);
                            psum2f = add_f32x2(psum2f, make_float2(sv[78], sv[79]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 82] = approx_exp2(sv[_le + 82]);
                            }
                            const float2 _fma_b2_137 = {exp_coef, exp_coef};
                            const float2 _fma_c2_138 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 94))[_lf], _fma_b2_137, _fma_c2_138);
                            psum2f = add_f32x2(psum2f, make_float2(sv[80], sv[81]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 84] = approx_exp2(sv[_le + 84]);
                            }
                            const float2 _fma_b2_139 = {exp_coef, exp_coef};
                            const float2 _fma_c2_140 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_139, _fma_c2_140);
                            psum2f = add_f32x2(psum2f, make_float2(sv[82], sv[83]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 86] = approx_exp2(sv[_le + 86]);
                            }
                            const float2 _fma_b2_141 = {exp_coef, exp_coef};
                            const float2 _fma_c2_142 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 98))[_lf], _fma_b2_141, _fma_c2_142);
                            psum2f = add_f32x2(psum2f, make_float2(sv[84], sv[85]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 88] = approx_exp2(sv[_le + 88]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_142 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 68], sv[_le*2 + 1 + 68]));
                                    sv[_le*2 + 68] = _exp2_pair_142.x;
                                    sv[_le*2 + 1 + 68] = _exp2_pair_142.y;
                                } else {
                                    sv[_le*2 + 68] = approx_exp2(sv[_le*2 + 68]);
                                    sv[_le*2 + 1 + 68] = approx_exp2(sv[_le*2 + 1 + 68]);
                                }
#endif
                            }
                            const float2 _fma_b2_143 = {exp_coef, exp_coef};
                            const float2 _fma_c2_144 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 100))[_lf], _fma_b2_143, _fma_c2_144);
                            psum2f = add_f32x2(psum2f, make_float2(sv[86], sv[87]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_143, _fma_c2_144);
                            psum2f = add_f32x2(psum2f, make_float2(sv[66], sv[67]));
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 90] = approx_exp2(sv[_le + 90]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_145 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 70], sv[_le*2 + 1 + 70]));
                                    sv[_le*2 + 70] = _exp2_pair_145.x;
                                    sv[_le*2 + 1 + 70] = _exp2_pair_145.y;
                                } else {
                                    sv[_le*2 + 70] = approx_exp2(sv[_le*2 + 70]);
                                    sv[_le*2 + 1 + 70] = approx_exp2(sv[_le*2 + 1 + 70]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_145 = {exp_coef, exp_coef};
                            const float2 _fma_c2_146 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_146 = {exp_coef, exp_coef};
                            const float2 _fma_c2_147 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 102))[_lf], _fma_b2_145, _fma_c2_146);
                            psum2f = add_f32x2(psum2f, make_float2(sv[88], sv[89]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 82))[_lf], _fma_b2_146, _fma_c2_147);
                            psum2f = add_f32x2(psum2f, make_float2(sv[68], sv[69]));
#endif
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 92] = approx_exp2(sv[_le + 92]);
                            }
                            const float2 _fma_b2_147 = {exp_coef, exp_coef};
                            const float2 _fma_c2_148 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 104))[_lf], _fma_b2_147, _fma_c2_148);
                            psum2f = add_f32x2(psum2f, make_float2(sv[90], sv[91]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 94] = approx_exp2(sv[_le + 94]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_148 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 72], sv[_le*2 + 1 + 72]));
                                    sv[_le*2 + 72] = _exp2_pair_148.x;
                                    sv[_le*2 + 1 + 72] = _exp2_pair_148.y;
                                } else {
                                    sv[_le*2 + 72] = approx_exp2(sv[_le*2 + 72]);
                                    sv[_le*2 + 1 + 72] = approx_exp2(sv[_le*2 + 1 + 72]);
                                }
#endif
                            }
                            const float2 _fma_b2_149 = {exp_coef, exp_coef};
                            const float2 _fma_c2_150 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 106))[_lf], _fma_b2_149, _fma_c2_150);
                            psum2f = add_f32x2(psum2f, make_float2(sv[92], sv[93]));
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 84))[_lf], _fma_b2_149, _fma_c2_150);
                            psum2f = add_f32x2(psum2f, make_float2(sv[70], sv[71]));
#endif
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                                sv[_le + 96] = approx_exp2(sv[_le + 96]);
#else
                                sv[_le + 74] = approx_exp2(sv[_le + 74]);
#endif
                            }
                            const float2 _fma_b2_151 = {exp_coef, exp_coef};
                            const float2 _fma_c2_152 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 108))[_lf], _fma_b2_151, _fma_c2_152);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 86))[_lf], _fma_b2_151, _fma_c2_152);
                            psum2f = add_f32x2(psum2f, make_float2(sv[72], sv[73]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 76] = approx_exp2(sv[_le + 76]);
                            }
                            const float2 _fma_b2_153 = {exp_coef, exp_coef};
                            const float2 _fma_c2_154 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 88))[_lf], _fma_b2_153, _fma_c2_154);
                            psum2f = add_f32x2(psum2f, make_float2(sv[74], sv[75]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 78] = approx_exp2(sv[_le + 78]);
                            }
                            const float2 _fma_b2_155 = {exp_coef, exp_coef};
                            const float2 _fma_c2_156 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 90))[_lf], _fma_b2_155, _fma_c2_156);
                            psum2f = add_f32x2(psum2f, make_float2(sv[76], sv[77]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 80] = approx_exp2(sv[_le + 80]);
                            }
                            const float2 _fma_b2_157 = {exp_coef, exp_coef};
                            const float2 _fma_c2_158 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 92))[_lf], _fma_b2_157, _fma_c2_158);
                            psum2f = add_f32x2(psum2f, make_float2(sv[78], sv[79]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 82] = approx_exp2(sv[_le + 82]);
                            }
                            const float2 _fma_b2_159 = {exp_coef, exp_coef};
                            const float2 _fma_c2_160 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 94))[_lf], _fma_b2_159, _fma_c2_160);
                            psum2f = add_f32x2(psum2f, make_float2(sv[80], sv[81]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 84] = approx_exp2(sv[_le + 84]);
                            }
                            const float2 _fma_b2_161 = {exp_coef, exp_coef};
                            const float2 _fma_c2_162 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_161, _fma_c2_162);
                            psum2f = add_f32x2(psum2f, make_float2(sv[82], sv[83]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 86] = approx_exp2(sv[_le + 86]);
                            }
                            const float2 _fma_b2_163 = {exp_coef, exp_coef};
                            const float2 _fma_c2_164 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 98))[_lf], _fma_b2_163, _fma_c2_164);
                            psum2f = add_f32x2(psum2f, make_float2(sv[84], sv[85]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 88] = approx_exp2(sv[_le + 88]);
                            }
                            const float2 _fma_b2_165 = {exp_coef, exp_coef};
                            const float2 _fma_c2_166 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 100))[_lf], _fma_b2_165, _fma_c2_166);
                            psum2f = add_f32x2(psum2f, make_float2(sv[86], sv[87]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 90] = approx_exp2(sv[_le + 90]);
                            }
                            const float2 _fma_b2_167 = {exp_coef, exp_coef};
                            const float2 _fma_c2_168 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 102))[_lf], _fma_b2_167, _fma_c2_168);
                            psum2f = add_f32x2(psum2f, make_float2(sv[88], sv[89]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 92] = approx_exp2(sv[_le + 92]);
                            }
                            const float2 _fma_b2_169 = {exp_coef, exp_coef};
                            const float2 _fma_c2_170 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 104))[_lf], _fma_b2_169, _fma_c2_170);
                            psum2f = add_f32x2(psum2f, make_float2(sv[90], sv[91]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 94] = approx_exp2(sv[_le + 94]);
                            }
                            const float2 _fma_b2_171 = {exp_coef, exp_coef};
                            const float2 _fma_c2_172 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 106))[_lf], _fma_b2_171, _fma_c2_172);
                            psum2f = add_f32x2(psum2f, make_float2(sv[92], sv[93]));
                            #pragma unroll
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_173 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 96], sv[_le*2 + 1 + 96]));
                                    sv[_le*2 + 96] = _exp2_pair_173.x;
                                    sv[_le*2 + 1 + 96] = _exp2_pair_173.y;
                                } else {
                                    sv[_le*2 + 96] = approx_exp2(sv[_le*2 + 96]);
                                    sv[_le*2 + 1 + 96] = approx_exp2(sv[_le*2 + 1 + 96]);
                                }
                            }
                            const float2 _fma_b2_174 = {exp_coef, exp_coef};
                            const float2 _fma_c2_175 = {neg_max_scaled, neg_max_scaled};
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 108))[_lf], _fma_b2_174, _fma_c2_175);
#endif
                            psum2f = add_f32x2(psum2f, make_float2(sv[94], sv[95]));
                            unsigned int pk2f[8];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[64]), "f"(sv[65]),
                                                       "f"(sv[66]), "f"(sv[67]));
                                pk2f[0] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[68]), "f"(sv[69]),
                                                       "f"(sv[70]), "f"(sv[71]));
                                pk2f[1] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[72]), "f"(sv[73]),
                                                       "f"(sv[74]), "f"(sv[75]));
                                pk2f[2] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[76]), "f"(sv[77]),
                                                       "f"(sv[78]), "f"(sv[79]));
                                pk2f[3] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[80]), "f"(sv[81]),
                                                       "f"(sv[82]), "f"(sv[83]));
                                pk2f[4] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[84]), "f"(sv[85]),
                                                       "f"(sv[86]), "f"(sv[87]));
                                pk2f[5] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[88]), "f"(sv[89]),
                                                       "f"(sv[90]), "f"(sv[91]));
                                pk2f[6] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[92]), "f"(sv[93]),
                                                       "f"(sv[94]), "f"(sv[95]));
                                pk2f[7] = _packed;
                            }
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 98] = approx_exp2(sv[_le + 98]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_176 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 98], sv[_le*2 + 1 + 98]));
                                    sv[_le*2 + 98] = _exp2_pair_176.x;
                                    sv[_le*2 + 1 + 98] = _exp2_pair_176.y;
                                } else {
                                    sv[_le*2 + 98] = approx_exp2(sv[_le*2 + 98]);
                                    sv[_le*2 + 1 + 98] = approx_exp2(sv[_le*2 + 1 + 98]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_153 = {exp_coef, exp_coef};
                            const float2 _fma_c2_154 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_177 = {exp_coef, exp_coef};
                            const float2 _fma_c2_178 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 110))[_lf], _fma_b2_153, _fma_c2_154);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 110))[_lf], _fma_b2_177, _fma_c2_178);
#endif
                            float2 psum3f = make_float2(sv[96], sv[97]);
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 100] = approx_exp2(sv[_le + 100]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_179 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 100], sv[_le*2 + 1 + 100]));
                                    sv[_le*2 + 100] = _exp2_pair_179.x;
                                    sv[_le*2 + 1 + 100] = _exp2_pair_179.y;
                                } else {
                                    sv[_le*2 + 100] = approx_exp2(sv[_le*2 + 100]);
                                    sv[_le*2 + 1 + 100] = approx_exp2(sv[_le*2 + 1 + 100]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_155 = {exp_coef, exp_coef};
                            const float2 _fma_c2_156 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_180 = {exp_coef, exp_coef};
                            const float2 _fma_c2_181 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_155, _fma_c2_156);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_180, _fma_c2_181);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[98], sv[99]));
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 102] = approx_exp2(sv[_le + 102]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_182 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 102], sv[_le*2 + 1 + 102]));
                                    sv[_le*2 + 102] = _exp2_pair_182.x;
                                    sv[_le*2 + 1 + 102] = _exp2_pair_182.y;
                                } else {
                                    sv[_le*2 + 102] = approx_exp2(sv[_le*2 + 102]);
                                    sv[_le*2 + 1 + 102] = approx_exp2(sv[_le*2 + 1 + 102]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_157 = {exp_coef, exp_coef};
                            const float2 _fma_c2_158 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_183 = {exp_coef, exp_coef};
                            const float2 _fma_c2_184 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 114))[_lf], _fma_b2_157, _fma_c2_158);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 114))[_lf], _fma_b2_183, _fma_c2_184);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[100], sv[101]));
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 104] = approx_exp2(sv[_le + 104]);
#else
                            for (int _le = 0; _le < 1; _le++) {
                                if (1 && _le >= 0) {
                                    float2 _exp2_pair_185 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 104], sv[_le*2 + 1 + 104]));
                                    sv[_le*2 + 104] = _exp2_pair_185.x;
                                    sv[_le*2 + 1 + 104] = _exp2_pair_185.y;
                                } else {
                                    sv[_le*2 + 104] = approx_exp2(sv[_le*2 + 104]);
                                    sv[_le*2 + 1 + 104] = approx_exp2(sv[_le*2 + 1 + 104]);
                                }
#endif
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_159 = {exp_coef, exp_coef};
                            const float2 _fma_c2_160 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_186 = {exp_coef, exp_coef};
                            const float2 _fma_c2_187 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 116))[_lf], _fma_b2_159, _fma_c2_160);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 116))[_lf], _fma_b2_186, _fma_c2_187);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[102], sv[103]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 106] = approx_exp2(sv[_le + 106]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_161 = {exp_coef, exp_coef};
                            const float2 _fma_c2_162 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_188 = {exp_coef, exp_coef};
                            const float2 _fma_c2_189 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 118))[_lf], _fma_b2_161, _fma_c2_162);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 118))[_lf], _fma_b2_188, _fma_c2_189);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[104], sv[105]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 108] = approx_exp2(sv[_le + 108]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_163 = {exp_coef, exp_coef};
                            const float2 _fma_c2_164 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_190 = {exp_coef, exp_coef};
                            const float2 _fma_c2_191 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 120))[_lf], _fma_b2_163, _fma_c2_164);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 120))[_lf], _fma_b2_190, _fma_c2_191);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[106], sv[107]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 110] = approx_exp2(sv[_le + 110]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_165 = {exp_coef, exp_coef};
                            const float2 _fma_c2_166 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_192 = {exp_coef, exp_coef};
                            const float2 _fma_c2_193 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 122))[_lf], _fma_b2_165, _fma_c2_166);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 122))[_lf], _fma_b2_192, _fma_c2_193);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[108], sv[109]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 112] = approx_exp2(sv[_le + 112]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_167 = {exp_coef, exp_coef};
                            const float2 _fma_c2_168 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_194 = {exp_coef, exp_coef};
                            const float2 _fma_c2_195 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 124))[_lf], _fma_b2_167, _fma_c2_168);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 124))[_lf], _fma_b2_194, _fma_c2_195);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[110], sv[111]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 114] = approx_exp2(sv[_le + 114]);
                            }
#if __CUDA_ARCH__ == 1030
                            const float2 _fma_b2_169 = {exp_coef, exp_coef};
                            const float2 _fma_c2_170 = {neg_max_scaled, neg_max_scaled};
#else
                            const float2 _fma_b2_196 = {exp_coef, exp_coef};
                            const float2 _fma_c2_197 = {neg_max_scaled, neg_max_scaled};
#endif
                            #pragma unroll
                            for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 126))[_lf], _fma_b2_169, _fma_c2_170);
#else
                                fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 126))[_lf], _fma_b2_196, _fma_c2_197);
#endif
                            psum3f = add_f32x2(psum3f, make_float2(sv[112], sv[113]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 116] = approx_exp2(sv[_le + 116]);
                            }
                            psum3f = add_f32x2(psum3f, make_float2(sv[114], sv[115]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 118] = approx_exp2(sv[_le + 118]);
                            }
                            psum3f = add_f32x2(psum3f, make_float2(sv[116], sv[117]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 120] = approx_exp2(sv[_le + 120]);
                            }
                            psum3f = add_f32x2(psum3f, make_float2(sv[118], sv[119]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 122] = approx_exp2(sv[_le + 122]);
                            }
                            psum3f = add_f32x2(psum3f, make_float2(sv[120], sv[121]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 124] = approx_exp2(sv[_le + 124]);
                            }
                            psum3f = add_f32x2(psum3f, make_float2(sv[122], sv[123]));
                            #pragma unroll
                            for (int _le = 0; _le < 2; _le++) {
                                sv[_le + 126] = approx_exp2(sv[_le + 126]);
                            }
                            psum3f = add_f32x2(psum3f, make_float2(sv[124], sv[125]));
                            psum3f = add_f32x2(psum3f, make_float2(sv[126], sv[127]));
                            unsigned int pk3f[8];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[96]), "f"(sv[97]),
                                                       "f"(sv[98]), "f"(sv[99]));
                                pk3f[0] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[100]), "f"(sv[101]),
                                                       "f"(sv[102]), "f"(sv[103]));
                                pk3f[1] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[104]), "f"(sv[105]),
                                                       "f"(sv[106]), "f"(sv[107]));
                                pk3f[2] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[108]), "f"(sv[109]),
                                                       "f"(sv[110]), "f"(sv[111]));
                                pk3f[3] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[112]), "f"(sv[113]),
                                                       "f"(sv[114]), "f"(sv[115]));
                                pk3f[4] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[116]), "f"(sv[117]),
                                                       "f"(sv[118]), "f"(sv[119]));
                                pk3f[5] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[120]), "f"(sv[121]),
                                                       "f"(sv[122]), "f"(sv[123]));
                                pk3f[6] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[124]), "f"(sv[125]),
                                                       "f"(sv[126]), "f"(sv[127]));
                                pk3f[7] = _packed;
                            }
                            int p_addr_f = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                            if (two_stages == 1) {
                                mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                                _phase_s_read ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (stage == 0 && two_stages == 0 && n_iter >= 1) {
                                mbarrier_wait(pv0_done_addr, _phase_pv0_done_0);
                                _phase_pv0_done_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (stage == 1 && two_stages == 1 && n_iter >= 1 && n_iter == last_n_iter) {
                                mbarrier_wait(pv1_done_addr, _phase_pv1_done_0);
                                _phase_pv1_done_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            tmem_st_x8_u32(p_addr_f, (const uint32_t*)pk0f);
                            tmem_st_x8_u32(p_addr_f + 8, (const uint32_t*)pk1f);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                            tmem_st_x8_u32(p_addr_f + 16, (const uint32_t*)pk2f);
                            tmem_st_x8_u32(p_addr_f + 24, (const uint32_t*)pk3f);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                            float2 sum01_f = add_f32x2(psum0f, psum1f);
                            float2 sum23_f = add_f32x2(psum2f, psum3f);
                            float2 total_f = add_f32x2(sum01_f, sum23_f);
                            row_sum = __fmaf_rn(row_sum, acc_scale, total_f.x) + total_f.y;
                        }
                    }
                    if (stage == 0 && two_stages == 1) {
                        mbarrier_arrive(s_read_addr + (stage) * 8);
                    }
                    scales[warp % 4 * 32 + lane + scale_off + scale_buf] = row_max;
                    scales[warp % 4 * 32 + lane + scale_off + scale_buf + 2 * BLOCK_M] = row_sum;
                    if (sync_group == 0) {
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 2, 256;" ::: "memory");
                    }
                }
                scale_buf = 512 - scale_buf;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            float exp_coef_1 = softmax_scale_log2 * q_descale[0] * k_descale[0];
            float v_d = v_descale[0];
            int nxt_doc_begin_1;
            int nxt_doc_len_1;
            int nxt_packed_1;
            int nxt_kv_words_1;
            int nxt_ws_slot_1;
            int nxt_q_half_1;
            int rec_1 = cluster_id * 8;
            int doc_begin_1 = unit_table[rec_1];
            int doc_len_2 = unit_table[rec_1 + 1];
            int packed_1 = unit_table[rec_1 + 2];
            int kv_words_1 = unit_table[rec_1 + 3];
            int ws_slot_1 = unit_table[rec_1 + 4];
            int q_half_1 = unit_table[rec_1 + 5];
            nxt_doc_begin_1 = doc_begin_1;
            nxt_doc_len_1 = doc_len_2;
            nxt_packed_1 = packed_1;
            nxt_kv_words_1 = kv_words_1;
            nxt_ws_slot_1 = ws_slot_1;
            nxt_q_half_1 = q_half_1;
            int scale_buf_1 = 0;
            unsigned int _phase_scale_full_0 = 0;
            unsigned int _phase_scale_full_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles; tile_idx_1 += num_clusters) {
                int doc_begin_0_1 = nxt_doc_begin_1;
                int doc_len_1_1 = nxt_doc_len_1;
                int ws_slot_2_1 = nxt_ws_slot_1;
                int head_1 = nxt_packed_1 >> 16;
                int c_1 = nxt_packed_1 & 65535;
                int n_begin_1 = nxt_kv_words_1 >> 16;
                int num_n_blocks_1 = nxt_kv_words_1 & 65535;
                int unit_row0_1 = c_1 * 512 + (nxt_q_half_1 >> 1) * 256;
                int full_rows_1 = ((doc_len_1_1 - unit_row0_1 > 256) ? 1 : 0);
                int two_stages_1 = ((nxt_q_half_1 == 0) ? full_rows_1 : 0);
                unsigned int nxt_tile_1 = tile_idx_1 + num_clusters;
                int rec_3_1 = nxt_tile_1 * 8;
                int doc_begin_4_1 = unit_table[rec_3_1];
                int doc_len_5_1 = unit_table[rec_3_1 + 1];
                int packed_6_1 = unit_table[rec_3_1 + 2];
                int kv_words_7_1 = unit_table[rec_3_1 + 3];
                int ws_slot_8_1 = unit_table[rec_3_1 + 4];
                int q_half_9_1 = unit_table[rec_3_1 + 5];
                nxt_doc_begin_1 = doc_begin_4_1;
                nxt_doc_len_1 = doc_len_5_1;
                nxt_packed_1 = packed_6_1;
                nxt_kv_words_1 = kv_words_7_1;
                nxt_ws_slot_1 = ws_slot_8_1;
                nxt_q_half_1 = q_half_9_1;
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                if (two_stages_1 == 1) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                }
#if __CUDA_ARCH__ == 1030
                mbarrier_wait(scale_full_addr, _phase_scale_full_0);
#else
                mbarrier_wait_hint(scale_full_addr, _phase_scale_full_0, 1000000);
#endif
                _phase_scale_full_0 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((scale_empty_addr) & 0xFEFFFFFF) : "memory");
                if (two_stages_1 == 1) {
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
#else
                    mbarrier_wait_hint(scale_full_addr + 8, _phase_scale_full_1, 1000000);
#endif
                    _phase_scale_full_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_empty_addr + 8) & 0xFEFFFFFF) : "memory");
                }
                unsigned int last_n_iter_c = (unsigned int)(num_n_blocks_1 - 1);
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(scale_full_addr, _phase_scale_full_0);
#else
                    mbarrier_wait_hint(scale_full_addr, _phase_scale_full_0, 1000000);
#endif
                    _phase_scale_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[1];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(_tmem_load_0[0])
                        : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float scale0 = _tmem_load_0[0];
                    int skip_rescale0 = __all_sync(0xFFFFFFFF, scale0 == 1.0f);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_empty_addr) & 0xFEFFFFFF) : "memory");
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
#else
                    mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 1000000);
#endif
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (two_stages_1 == 0) {
                        mbarrier_arrive(pv0_done_addr);
                    }
                    if (skip_rescale0 == 0) {
                        #pragma unroll
                        for (int col = 0; col < HEAD_DIM / 16; col++) {
                            int addr0 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 16);
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], addr0);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {scale0, scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_1[_ls] = _tmem_load_1[_ls] * scale0;
                            }
                            #endif
                            tmem_st_x16_f32(addr0, _tmem_load_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    if (two_stages_1 == 1) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
#else
                        mbarrier_wait_hint(scale_full_addr + 8, _phase_scale_full_1, 1000000);
#endif
                        _phase_scale_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_2[1];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                            " {%0}, [%1];"
                            : "=f"(_tmem_load_2[0])
                            : "r"(taddr + (unsigned int)BLOCK_M + (unsigned int)(warp % 4 * 32 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float scale1 = _tmem_load_2[0];
                        int skip_rescale1 = __all_sync(0xFFFFFFFF, scale1 == 1.0f);
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((scale_empty_addr + 8) & 0xFEFFFFFF) : "memory");
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
#else
                        mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 1000000);
#endif
                        _phase_o_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (n_iter_1 == last_n_iter_c) {
                            mbarrier_arrive(pv1_done_addr);
                        }
                        if (skip_rescale1 == 0) {
                            #pragma unroll
                            for (int col_1 = 0; col_1 < HEAD_DIM / 16; col_1++) {
                                int addr1 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_1 * 16);
                                float _tmem_load_3[16];
                                tmem_ld_x16(&_tmem_load_3[0], addr1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_1 = {scale1, scale1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                                #else
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++) {
                                    _tmem_load_3[_ls] = _tmem_load_3[_ls] * scale1;
                                }
                                #endif
                                tmem_st_x16_f32(addr1, _tmem_load_3);
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    }
                }
#if __CUDA_ARCH__ == 1030
                mbarrier_wait(o_full_addr, _phase_o_full_0);
#else
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 1000000);
#endif
                _phase_o_full_0 ^= 1;
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_sum = scales[warp % 4 * 32 + lane + scale_buf_1 + 2 * BLOCK_M];
                float final_scale;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_2 = approx_rcp(final_sum);
                    final_scale = _rcp_2 * v_d;
                } else {
                    final_scale = 0.0f;
                }
                int local_row = unit_row0_1 + cta_rank * BLOCK_M + (warp % 4 * 32 + lane);
                int out_row = (doc_begin_0_1 + local_row) * num_heads + head_1;
                if (ws_slot_2_1 >= 0) {
                    int partial_row = ws_slot_2_1 * 512 + cta_rank * BLOCK_M + (warp % 4 * 32 + lane);
                    if (local_row < doc_len_1_1) {
                        partial_ML[partial_row * 2] = scales[warp % 4 * 32 + lane + scale_buf_1] * exp_coef_1;
                        partial_ML[partial_row * 2 + 1] = final_sum;
                    }
                    #pragma unroll
                    for (int col_2 = 0; col_2 < HEAD_DIM / 16; col_2++) {
                        int addr = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 16);
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (local_row < doc_len_1_1) {
                            {
                                const float2 _prescale2_3 = {final_scale, final_scale};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_3);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_4[0 + _ps] *= final_scale;
                                #endif
                                __half2 _pk[8];
                                _pk[0] = __floats2half2_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                                _pk[1] = __floats2half2_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                                _pk[2] = __floats2half2_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                                _pk[3] = __floats2half2_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                                _pk[4] = __floats2half2_rn(_tmem_load_4[0 + 8], _tmem_load_4[0 + 9]);
                                _pk[5] = __floats2half2_rn(_tmem_load_4[0 + 10], _tmem_load_4[0 + 11]);
                                _pk[6] = __floats2half2_rn(_tmem_load_4[0 + 12], _tmem_load_4[0 + 13]);
                                _pk[7] = __floats2half2_rn(_tmem_load_4[0 + 14], _tmem_load_4[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__half*)(partial_O + (partial_row * HEAD_DIM + col_2 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__half*)(partial_O + (partial_row * HEAD_DIM + col_2 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        }
                    }
                } else {
                    int tile_row0 = unit_row0_1 + cta_rank * BLOCK_M;
                    if (doc_len_1_1 >= tile_row0 + BLOCK_M) {
                        if (warp == 8) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 3, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int col_3 = 0; col_3 < HEAD_DIM / 16; col_3++) {
                            int addr_1 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_3 * 16);
                            float _tmem_load_5[16];
                            tmem_ld_x16(&_tmem_load_5[0], addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_4 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_4);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_5[_ls] = _tmem_load_5[_ls] * final_scale;
                            }
                            #endif
                            unsigned int _tmem_load_5_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int slab = col_3 / 4;
                            int col_bytes = col_3 % 4 * 32;
                            int slab_addr = smem_o_addr + (unsigned int)(slab * 16384);
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr + ((warp % 4 * 32 + lane) * 128 + col_bytes ^ ((warp % 4 * 32 + lane) * 128 + col_bytes >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[0]), "r"(_tmem_load_5_bf16[1]), "r"(_tmem_load_5_bf16[2]), "r"(_tmem_load_5_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr + ((warp % 4 * 32 + lane) * 128 + (col_bytes + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_bytes + 16) >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[4]), "r"(_tmem_load_5_bf16[5]), "r"(_tmem_load_5_bf16[6]), "r"(_tmem_load_5_bf16[7]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 3, 128;" ::: "memory");
                        if (warp == 8) {
                            if (elect_sync()) {
                                tma_store_4d((&O), 0, doc_begin_0_1 + tile_row0, head_1, 0, smem_o_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int col_4 = 0; col_4 < HEAD_DIM / 16; col_4++) {
                            int addr_2 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_4 * 16);
                            float _tmem_load_6[16];
                            tmem_ld_x16(&_tmem_load_6[0], addr_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (local_row < doc_len_1_1) {
                                {
                                    const float2 _prescale2_5 = {final_scale, final_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&_tmem_load_6[0])[_ps], _prescale2_5);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_6[0 + _ps] *= final_scale;
                                    #endif
                                    __nv_bfloat162 _pk[8];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_6[0 + 0], _tmem_load_6[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_6[0 + 2], _tmem_load_6[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_6[0 + 4], _tmem_load_6[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_6[0 + 6], _tmem_load_6[0 + 7]);
                                    _pk[4] = __floats2bfloat162_rn(_tmem_load_6[0 + 8], _tmem_load_6[0 + 9]);
                                    _pk[5] = __floats2bfloat162_rn(_tmem_load_6[0 + 10], _tmem_load_6[0 + 11]);
                                    _pk[6] = __floats2bfloat162_rn(_tmem_load_6[0 + 12], _tmem_load_6[0 + 13]);
                                    _pk[7] = __floats2bfloat162_rn(_tmem_load_6[0 + 14], _tmem_load_6[0 + 15]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_raw + (out_row * HEAD_DIM + col_4 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_raw + (out_row * HEAD_DIM + col_4 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                }
                            }
                        }
                    }
                }
                if (two_stages_1 == 1) {
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
#else
                    mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 1000000);
#endif
                    _phase_o_full_1 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_sum_0 = scales[warp % 4 * 32 + lane + (scale_buf_1 + BLOCK_M) + 2 * BLOCK_M];
                    float final_scale_1;
                    if (final_sum_0 != 0.0f && final_sum_0 == final_sum_0) {
                        float _rcp_6 = approx_rcp(final_sum_0);
                        final_scale_1 = _rcp_6 * v_d;
                    } else {
                        final_scale_1 = 0.0f;
                    }
                    int local_row_2 = unit_row0_1 + (2 + cta_rank) * BLOCK_M + (warp % 4 * 32 + lane);
                    int out_row_3 = (doc_begin_0_1 + local_row_2) * num_heads + head_1;
                    if (ws_slot_2_1 >= 0) {
                        int partial_row_1 = ws_slot_2_1 * 512 + (2 + cta_rank) * BLOCK_M + (warp % 4 * 32 + lane);
                        if (local_row_2 < doc_len_1_1) {
                            partial_ML[partial_row_1 * 2] = scales[warp % 4 * 32 + lane + (scale_buf_1 + BLOCK_M)] * exp_coef_1;
                            partial_ML[partial_row_1 * 2 + 1] = final_sum_0;
                        }
                        #pragma unroll
                        for (int col_5 = 0; col_5 < HEAD_DIM / 16; col_5++) {
                            int addr_3 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_5 * 16);
                            float _tmem_load_7[16];
                            tmem_ld_x16(&_tmem_load_7[0], addr_3);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (local_row_2 < doc_len_1_1) {
                                {
                                    const float2 _prescale2_7 = {final_scale_1, final_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[0])[_ps], _prescale2_7);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_7[0 + _ps] *= final_scale_1;
                                    #endif
                                    __half2 _pk[8];
                                    _pk[0] = __floats2half2_rn(_tmem_load_7[0 + 0], _tmem_load_7[0 + 1]);
                                    _pk[1] = __floats2half2_rn(_tmem_load_7[0 + 2], _tmem_load_7[0 + 3]);
                                    _pk[2] = __floats2half2_rn(_tmem_load_7[0 + 4], _tmem_load_7[0 + 5]);
                                    _pk[3] = __floats2half2_rn(_tmem_load_7[0 + 6], _tmem_load_7[0 + 7]);
                                    _pk[4] = __floats2half2_rn(_tmem_load_7[0 + 8], _tmem_load_7[0 + 9]);
                                    _pk[5] = __floats2half2_rn(_tmem_load_7[0 + 10], _tmem_load_7[0 + 11]);
                                    _pk[6] = __floats2half2_rn(_tmem_load_7[0 + 12], _tmem_load_7[0 + 13]);
                                    _pk[7] = __floats2half2_rn(_tmem_load_7[0 + 14], _tmem_load_7[0 + 15]);
                                    *reinterpret_cast<uint4*>(&((__half*)(partial_O + (partial_row_1 * HEAD_DIM + col_5 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    *reinterpret_cast<uint4*>(&((__half*)(partial_O + (partial_row_1 * HEAD_DIM + col_5 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                }
                            }
                        }
                    } else {
                        int tile_row0_1 = unit_row0_1 + (2 + cta_rank) * BLOCK_M;
                        if (doc_len_1_1 >= tile_row0_1 + BLOCK_M) {
                            if (warp == 8) {
                                if (elect_sync()) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            #pragma unroll
                            for (int col_6 = 0; col_6 < HEAD_DIM / 16; col_6++) {
                                int addr_4 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_6 * 16);
                                float _tmem_load_8[16];
                                tmem_ld_x16(&_tmem_load_8[0], addr_4);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_8 = {final_scale_1, final_scale_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_8);
                                #else
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++) {
                                    _tmem_load_8[_ls] = _tmem_load_8[_ls] * final_scale_1;
                                }
                                #endif
                                unsigned int _tmem_load_8_bf16[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                    _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int slab_1 = 2 + col_6 / 4;
                                int col_bytes_1 = col_6 % 4 * 32;
                                int slab_addr_1 = smem_o_addr + (unsigned int)(slab_1 * 16384);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_1 + ((warp % 4 * 32 + lane) * 128 + col_bytes_1 ^ ((warp % 4 * 32 + lane) * 128 + col_bytes_1 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_1 + ((warp % 4 * 32 + lane) * 128 + (col_bytes_1 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_bytes_1 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (warp == 8) {
                                if (elect_sync()) {
                                    tma_store_4d((&O), 0, doc_begin_0_1 + tile_row0_1, head_1, 0, smem_o_addr + 32768);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int col_7 = 0; col_7 < HEAD_DIM / 16; col_7++) {
                                int addr_5 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_7 * 16);
                                float _tmem_load_9[16];
                                tmem_ld_x16(&_tmem_load_9[0], addr_5);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                if (local_row_2 < doc_len_1_1) {
                                    {
                                        const float2 _prescale2_9 = {final_scale_1, final_scale_1};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[0])[_ps], _prescale2_9);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 16; _ps++)
                                            _tmem_load_9[0 + _ps] *= final_scale_1;
                                        #endif
                                        __nv_bfloat162 _pk[8];
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_9[0 + 0], _tmem_load_9[0 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_9[0 + 2], _tmem_load_9[0 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_9[0 + 4], _tmem_load_9[0 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_9[0 + 6], _tmem_load_9[0 + 7]);
                                        _pk[4] = __floats2bfloat162_rn(_tmem_load_9[0 + 8], _tmem_load_9[0 + 9]);
                                        _pk[5] = __floats2bfloat162_rn(_tmem_load_9[0 + 10], _tmem_load_9[0 + 11]);
                                        _pk[6] = __floats2bfloat162_rn(_tmem_load_9[0 + 12], _tmem_load_9[0 + 13]);
                                        _pk[7] = __floats2bfloat162_rn(_tmem_load_9[0 + 14], _tmem_load_9[0 + 15]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_raw + (out_row_3 * HEAD_DIM + col_7 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_raw + (out_row_3 * HEAD_DIM + col_7 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                    }
                                }
                            }
                        }
                    }
                }
                scale_buf_1 = 512 - scale_buf_1;
            }
            if (warp == 8) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_scale_empty_0 = 1;
            unsigned int _phase_scale_empty_1 = 1;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            if (cta_rank == 0) {
                unsigned int k_stage = 0;
                unsigned int k_phase = 0;
                unsigned int v_stage = 1;
                unsigned int v_phase = 0;
                int nxt_doc_begin_2;
                int nxt_doc_len_2;
                int nxt_packed_2;
                int nxt_kv_words_2;
                int nxt_ws_slot_2;
                int nxt_q_half_2;
                int rec_2 = cluster_id * 8;
                int doc_begin_2 = unit_table[rec_2];
                int doc_len_3 = unit_table[rec_2 + 1];
                int packed_2 = unit_table[rec_2 + 2];
                int kv_words_2 = unit_table[rec_2 + 3];
                int ws_slot_3 = unit_table[rec_2 + 4];
                int q_half_2 = unit_table[rec_2 + 5];
                nxt_doc_begin_2 = doc_begin_2;
                nxt_doc_len_2 = doc_len_3;
                nxt_packed_2 = packed_2;
                nxt_kv_words_2 = kv_words_2;
                nxt_ws_slot_2 = ws_slot_3;
                nxt_q_half_2 = q_half_2;
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles; tile_idx_2 += num_clusters) {
                    int doc_begin_0_2 = nxt_doc_begin_2;
                    int doc_len_1_2 = nxt_doc_len_2;
                    int ws_slot_2_2 = nxt_ws_slot_2;
                    int head_2 = nxt_packed_2 >> 16;
                    int c_2 = nxt_packed_2 & 65535;
                    int n_begin_2 = nxt_kv_words_2 >> 16;
                    int num_n_blocks_2 = nxt_kv_words_2 & 65535;
                    int unit_row0_2 = c_2 * 512 + (nxt_q_half_2 >> 1) * 256;
                    int full_rows_2 = ((doc_len_1_2 - unit_row0_2 > 256) ? 1 : 0);
                    int two_stages_2 = ((nxt_q_half_2 == 0) ? full_rows_2 : 0);
                    unsigned int nxt_tile_2 = tile_idx_2 + num_clusters;
                    int rec_3_2 = nxt_tile_2 * 8;
                    int doc_begin_4_2 = unit_table[rec_3_2];
                    int doc_len_5_2 = unit_table[rec_3_2 + 1];
                    int packed_6_2 = unit_table[rec_3_2 + 2];
                    int kv_words_7_2 = unit_table[rec_3_2 + 3];
                    int ws_slot_8_2 = unit_table[rec_3_2 + 4];
                    int q_half_9_2 = unit_table[rec_3_2 + 5];
                    nxt_doc_begin_2 = doc_begin_4_2;
                    nxt_doc_len_2 = doc_len_5_2;
                    nxt_packed_2 = packed_6_2;
                    nxt_kv_words_2 = kv_words_7_2;
                    nxt_ws_slot_2 = ws_slot_8_2;
                    nxt_q_half_2 = q_half_9_2;
                    mbarrier_wait_cluster_hint(q_full_addr, _phase_q_full_0, 10000000);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(scale_empty_addr, _phase_scale_empty_0);
#else
                    mbarrier_wait_hint(scale_empty_addr, _phase_scale_empty_0, 1000000);
#endif
                    _phase_scale_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_0 = ((smem_q0_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_1 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    "mov.b32 id, 270532624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_scores), "r"(0) : "memory");
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
#else
                        mbarrier_wait_hint(scale_empty_addr + 8, _phase_scale_empty_1, 1000000);
#endif
                        _phase_scale_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_2 = ((smem_q1_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_3 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    "mov.b32 id, 270532624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_scores + (128))), "r"(0) : "memory");
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 8) { k_stage = 0; k_phase ^= 1; }
                    k_stage += 1;
                    if (k_stage == 8) { k_stage = 0; k_phase ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 0; n_iter_2 < num_n_blocks_2 - 1; n_iter_2++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(scale_empty_addr, _phase_scale_empty_0);
#else
                        mbarrier_wait_hint(scale_empty_addr, _phase_scale_empty_0, 1000000);
#endif
                        _phase_scale_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_4 = ((smem_q0_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_5 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    "mov.b32 id, 270532624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_scores), "r"(0) : "memory");
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
#else
                        mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 1000000);
#endif
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_smem_desc_lo_6), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
#else
                        mbarrier_wait_hint(p_full_2_addr, _phase_p_full_2_0, 1000000);
#endif
                        _phase_p_full_2_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_7 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_smem_desc_lo_7), "r"(tmem_probs_0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                            mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
#else
                            mbarrier_wait_hint(scale_empty_addr + 8, _phase_scale_empty_1, 1000000);
#endif
                            _phase_scale_empty_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _smem_desc_lo_8 = ((smem_q1_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_9 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    "mov.b32 id, 270532624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"((tmem_scores + (128))), "r"(0) : "memory");
                            elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 8) { k_stage = 0; k_phase ^= 1; }
                        k_stage += 1;
                        if (k_stage == 8) { k_stage = 0; k_phase ^= 1; }
                        if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
#else
                            mbarrier_wait_hint(p_full_addr + 8, _phase_p_full_1, 1000000);
#endif
                            _phase_p_full_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _smem_desc_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_smem_desc_lo_10), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1030
                            mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
#else
                            mbarrier_wait_hint(p_full_2_addr + 8, _phase_p_full_2_1, 1000000);
#endif
                            _phase_p_full_2_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _smem_desc_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_smem_desc_lo_11), "r"(tmem_probs_1), "r"(1) : "memory");
                            elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 8) { v_stage = 0; v_phase ^= 1; }
                        v_stage += 1;
                        if (v_stage == 8) { v_stage = 0; v_phase ^= 1; }
                        first_pv = 0;
                    }
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    int first_pv_flag_1 = first_pv;
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
#else
                    mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 1000000);
#endif
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_12 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_smem_desc_lo_12), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
#else
                    mbarrier_wait_hint(p_full_2_addr, _phase_p_full_2_0, 1000000);
#endif
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_13 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_smem_desc_lo_13), "r"(tmem_probs_0), "r"(1) : "memory");
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
#else
                        mbarrier_wait_hint(p_full_addr + 8, _phase_p_full_1, 1000000);
#endif
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_14 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_smem_desc_lo_14), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
#else
                        mbarrier_wait_hint(p_full_2_addr + 8, _phase_p_full_2_1, 1000000);
#endif
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_15 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 270598160;\n\t"
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_smem_desc_lo_15), "r"(tmem_probs_1), "r"(1) : "memory");
                        elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                    v_stage += 1;
                    if (v_stage == 8) { v_stage = 0; v_phase ^= 1; }
                    v_stage += 1;
                    if (v_stage == 8) { v_stage = 0; v_phase ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int load_stage = 0;
            int nxt_doc_begin_3;
            int nxt_doc_len_3;
            int nxt_packed_3;
            int nxt_kv_words_3;
            int nxt_ws_slot_3;
            int nxt_q_half_3;
            int rec_4 = cluster_id * 8;
            int doc_begin_3 = unit_table[rec_4];
            int doc_len_4 = unit_table[rec_4 + 1];
            int packed_3 = unit_table[rec_4 + 2];
            int kv_words_3 = unit_table[rec_4 + 3];
            int ws_slot_4 = unit_table[rec_4 + 4];
            int q_half_3 = unit_table[rec_4 + 5];
            nxt_doc_begin_3 = doc_begin_3;
            nxt_doc_len_3 = doc_len_4;
            nxt_packed_3 = packed_3;
            nxt_kv_words_3 = kv_words_3;
            nxt_ws_slot_3 = ws_slot_4;
            nxt_q_half_3 = q_half_3;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles; tile_idx_3 += num_clusters) {
                int doc_begin_0_3 = nxt_doc_begin_3;
                int doc_len_1_3 = nxt_doc_len_3;
                int ws_slot_2_3 = nxt_ws_slot_3;
                int head_3 = nxt_packed_3 >> 16;
                int c_3 = nxt_packed_3 & 65535;
                int n_begin_3 = nxt_kv_words_3 >> 16;
                int num_n_blocks_3 = nxt_kv_words_3 & 65535;
                int unit_row0_3 = c_3 * 512 + (nxt_q_half_3 >> 1) * 256;
                int full_rows_3 = ((doc_len_1_3 - unit_row0_3 > 256) ? 1 : 0);
                int two_stages_3 = ((nxt_q_half_3 == 0) ? full_rows_3 : 0);
                unsigned int nxt_tile_3 = tile_idx_3 + num_clusters;
                int rec_3_3 = nxt_tile_3 * 8;
                int doc_begin_4_3 = unit_table[rec_3_3];
                int doc_len_5_3 = unit_table[rec_3_3 + 1];
                int packed_6_3 = unit_table[rec_3_3 + 2];
                int kv_words_7_3 = unit_table[rec_3_3 + 3];
                int ws_slot_8_3 = unit_table[rec_3_3 + 4];
                int q_half_9_3 = unit_table[rec_3_3 + 5];
                nxt_doc_begin_3 = doc_begin_4_3;
                nxt_doc_len_3 = doc_len_5_3;
                nxt_packed_3 = packed_6_3;
                nxt_kv_words_3 = kv_words_7_3;
                nxt_ws_slot_3 = ws_slot_8_3;
                nxt_q_half_3 = q_half_9_3;
                int q_row0 = doc_begin_0_3 + unit_row0_3 + cta_rank * BLOCK_M;
                int q_row1 = q_row0 + 256;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    if (two_stages_3 == 1) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_3d_gmem2smem_cta2(smem_q0_addr, (&Q), 0, q_row0, head_3, ((q_full_addr) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_q1_addr, (&Q), 0, q_row1, head_3, ((q_full_addr) & 0xFEFFFFFF));
                    } else {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_3d_gmem2smem_cta2(smem_q0_addr, (&Q), 0, q_row0, head_3, ((q_full_addr) & 0xFEFFFFFF));
                    }
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_3; ni++) {
                    unsigned int n = (unsigned int)(num_n_blocks_3 - 1) - ni;
                    int kv_row = (unsigned int)(doc_begin_0_3 + n_begin_3 * BLOCK_N) + n * (unsigned int)BLOCK_N;
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                        tma_3d_gmem2smem_cta2(smem_kv_addr + load_stage * 8192, (&K), 0, kv_row + cta_rank * 64, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 8) { load_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                        tma_3d_gmem2smem_cta2(smem_v_addr + load_stage * 8192, (&V), cta_rank * 64, kv_row, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 8) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
