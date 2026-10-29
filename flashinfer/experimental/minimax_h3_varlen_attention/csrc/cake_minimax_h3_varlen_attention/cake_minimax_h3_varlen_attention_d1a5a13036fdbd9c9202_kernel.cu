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
#define NUM_KV_STAGES 5
#define SMEM_SCALES_OFF 1024
#define SMEM_SCALES_STAGE_BYTES 2048
#define SMEM_SCALES_STRIDE 2048
#define SMEM_TICKETS_OFF 216064
#define SMEM_TICKETS_STAGE_BYTES 8
#define SMEM_TICKETS_STRIDE 8
#define SMEM_SMEM_Q0_OFF 3072
#define SMEM_SMEM_Q0_STAGE_BYTES 32768
#define SMEM_SMEM_Q0_STRIDE 32768
#define SMEM_SMEM_Q1_OFF 35840
#define SMEM_SMEM_Q1_STAGE_BYTES 32768
#define SMEM_SMEM_Q1_STRIDE 32768
#define SMEM_SMEM_O_OFF 150528
#define SMEM_SMEM_O_STAGE_BYTES 16384
#define SMEM_SMEM_O_STRIDE 16384
#define SMEM_SMEM_KV_OFF 68608
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 68608
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_TOTAL 216192
#define USE_TMEM_LD_RED 0
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

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}




__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_varlen_attention_d1a5a13036fdbd9c9202(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, __nv_bfloat16* __restrict__ O_raw, int* __restrict__ unit_table, __half* __restrict__ partial_O, float* __restrict__ partial_ML, unsigned int total_tiles, int num_heads, float softmax_scale_log2)
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
    #define kv_empty_addr (mbar_base + 56)
    #define s_full_addr (mbar_base + 96)
    #define p_full_addr (mbar_base + 112)
    #define scale_full_addr (mbar_base + 128)
    #define scale_empty_addr (mbar_base + 144)
    #define s_read_addr (mbar_base + 160)
    #define o_full_addr (mbar_base + 176)
    #define pv1_done_addr (mbar_base + 192)
    #define scales_free_addr (mbar_base + 200)
    #define tk_full_addr (mbar_base + 216)
    #define tk_empty_addr (mbar_base + 232)
    #define tk_go_addr (mbar_base + 248)

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
    int* tickets = reinterpret_cast<int*>(smem_raw + 216064);
    const int tickets_addr = smem + 216064;
    __nv_bfloat16* smem_q0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_q0_addr = smem + 3072;
    __nv_bfloat16* smem_q1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 35840);
    const int smem_q1_addr = smem + 35840;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 150528);
    const int smem_o_addr = smem + 150528;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_kv_addr = smem + 68608;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_v_addr = smem + 68608;
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
    if (warp == 13) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&O))) : "memory"); }

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 33 barriers)
    // Mbarriers at smem_raw[0..264)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv' ---
            // kv_full: 5 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // kv_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 112, 512);
            mbarrier_init(smem + 120, 512);
            // scale_full: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // scale_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // s_read: 2 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // pv1_done: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // scales_free: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            // tk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // tk_empty: 2 barriers, init_count=768
            mbarrier_init(smem + 232, 768);
            mbarrier_init(smem + 240, 768);
            // tk_go: 2 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Kernel pre-TMEM ops
    int rec = cluster_id * 8;
    int doc_begin = unit_table[rec];
    int doc_len = unit_table[rec + 1];
    int packed = unit_table[rec + 2];
    int kv_words = unit_table[rec + 3];
    int ws_slot = unit_table[rec + 4];
    int q_half = unit_table[rec + 5];

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 0) {
        int _tmem_hold = smem + 264;
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
            unsigned int stage = make_warp_uniform(warp / 4);
            unsigned int other_stage = make_warp_uniform(1 - stage);
            unsigned int warp_group_idx = ((unsigned int)tid - 0) / 128;
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
            nxt_doc_begin = doc_begin;
            nxt_doc_len = doc_len;
            nxt_packed = packed;
            nxt_kv_words = kv_words;
            nxt_ws_slot = ws_slot;
            nxt_q_half = q_half;
            unsigned int _phase_s_read = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_pv1_done_0 = 0;
            unsigned int _phase_scales_free = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx++) {
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
                unsigned int nxt_tile;
                int active = ((stage == 0) ? 1 : two_stages);
                if (active == 1) {
                    int tail_base = doc_len_1 - n_begin * BLOCK_N;
                    float row_max = -CUDART_INF_F;
                    float row_sum = 0.0f;
                    if (stage == 1) {
                        mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                        _phase_s_read ^= 1;
                    }
                    #pragma unroll 1
                    for (unsigned int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                        int n_block = (unsigned int)(num_n_blocks - 1) - n_iter;
                        mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                        _phase_s_full ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int s_addr = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                        float sv[128];
                        float tile_max = -CUDART_INF_F;
                        {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                                : "r"(s_addr));
                            asm volatile(
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
                        }
                        int tail_valid = tail_base - n_block * BLOCK_N;
                        if (tail_valid < BLOCK_N) {
                            uint32_t _thresh_mask_2;
                            {
                                int _lim_3 = tail_valid;
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
                            uint32_t _thresh_mask_4;
                            {
                                int _lim_5 = tail_valid - 32;
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
                            uint32_t _thresh_mask_6;
                            {
                                int _lim_7 = tail_valid - 64;
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
                        }
                        if (two_stages == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(s_read_addr + (stage) * 8);
                        }
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * softmax_scale_log2;
                        float acc_scale_log2 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                        float acc_scale;
                        if (acc_scale_log2 >= -8.0f) {
                            safe_max = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                            acc_scale = 1.0f;
                            new_max_scaled = safe_max * softmax_scale_log2;
                        } else {
                            float _exp2_12 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_12 : 1.0f);
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
                        mbarrier_arrive(scale_full_addr + (stage) * 8);
                        const float2 _fma_b2_13 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_14 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_13, _fma_c2_14);
                        const float2 _fma_b2_15 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_16 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_15, _fma_c2_16);
                        const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_18 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_17, _fma_c2_18);
                        const float2 _fma_b2_19 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_20 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_19, _fma_c2_20);
                        const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_22 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_21, _fma_c2_22);
                        const float2 _fma_b2_23 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_24 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_23, _fma_c2_24);
                        float2 _exp2_pair_25 = ex2_emulation_f32x2_value(make_float2(sv[0], sv[1]));
                        sv[0] = _exp2_pair_25.x;
                        sv[1] = _exp2_pair_25.y;
                        const float2 _fma_b2_26 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_27 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_26, _fma_c2_27);
                        float2 _exp2_pair_28 = ex2_emulation_f32x2_value(make_float2(sv[2], sv[3]));
                        sv[2] = _exp2_pair_28.x;
                        sv[3] = _exp2_pair_28.y;
                        const float2 _fma_b2_29 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_30 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_29, _fma_c2_30);
                        float2 psum0 = make_float2(sv[0], sv[1]);
                        float2 _exp2_pair_31 = ex2_emulation_f32x2_value(make_float2(sv[4], sv[5]));
                        sv[4] = _exp2_pair_31.x;
                        sv[5] = _exp2_pair_31.y;
                        const float2 _fma_b2_32 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_33 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_32, _fma_c2_33);
                        psum0 = add_f32x2(psum0, make_float2(sv[2], sv[3]));
                        float2 _exp2_pair_34 = ex2_emulation_f32x2_value(make_float2(sv[6], sv[7]));
                        sv[6] = _exp2_pair_34.x;
                        sv[7] = _exp2_pair_34.y;
                        const float2 _fma_b2_35 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_36 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_35, _fma_c2_36);
                        psum0 = add_f32x2(psum0, make_float2(sv[4], sv[5]));
                        float2 _exp2_pair_37 = ex2_emulation_f32x2_value(make_float2(sv[8], sv[9]));
                        sv[8] = _exp2_pair_37.x;
                        sv[9] = _exp2_pair_37.y;
                        const float2 _fma_b2_38 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_39 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_38, _fma_c2_39);
                        psum0 = add_f32x2(psum0, make_float2(sv[6], sv[7]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 10] = approx_exp2(sv[_le + 10]);
                        }
                        const float2 _fma_b2_40 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_41 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_40, _fma_c2_41);
                        psum0 = add_f32x2(psum0, make_float2(sv[8], sv[9]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 12] = approx_exp2(sv[_le + 12]);
                        }
                        const float2 _fma_b2_42 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_43 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_42, _fma_c2_43);
                        psum0 = add_f32x2(psum0, make_float2(sv[10], sv[11]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 14] = approx_exp2(sv[_le + 14]);
                        }
                        const float2 _fma_b2_44 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_45 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_44, _fma_c2_45);
                        psum0 = add_f32x2(psum0, make_float2(sv[12], sv[13]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 16] = approx_exp2(sv[_le + 16]);
                        }
                        const float2 _fma_b2_46 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_47 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_46, _fma_c2_47);
                        psum0 = add_f32x2(psum0, make_float2(sv[14], sv[15]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 18] = approx_exp2(sv[_le + 18]);
                        }
                        const float2 _fma_b2_48 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_49 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_48, _fma_c2_49);
                        psum0 = add_f32x2(psum0, make_float2(sv[16], sv[17]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 20] = approx_exp2(sv[_le + 20]);
                        }
                        const float2 _fma_b2_50 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_51 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_50, _fma_c2_51);
                        psum0 = add_f32x2(psum0, make_float2(sv[18], sv[19]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 22] = approx_exp2(sv[_le + 22]);
                        }
                        const float2 _fma_b2_52 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_53 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 34))[_lf], _fma_b2_52, _fma_c2_53);
                        psum0 = add_f32x2(psum0, make_float2(sv[20], sv[21]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 24] = approx_exp2(sv[_le + 24]);
                        }
                        const float2 _fma_b2_54 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_55 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 36))[_lf], _fma_b2_54, _fma_c2_55);
                        psum0 = add_f32x2(psum0, make_float2(sv[22], sv[23]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 26] = approx_exp2(sv[_le + 26]);
                        }
                        const float2 _fma_b2_56 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_57 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 38))[_lf], _fma_b2_56, _fma_c2_57);
                        psum0 = add_f32x2(psum0, make_float2(sv[24], sv[25]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 28] = approx_exp2(sv[_le + 28]);
                        }
                        const float2 _fma_b2_58 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_59 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 40))[_lf], _fma_b2_58, _fma_c2_59);
                        psum0 = add_f32x2(psum0, make_float2(sv[26], sv[27]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 30] = approx_exp2(sv[_le + 30]);
                        }
                        const float2 _fma_b2_60 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_61 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 42))[_lf], _fma_b2_60, _fma_c2_61);
                        psum0 = add_f32x2(psum0, make_float2(sv[28], sv[29]));
                        float2 _exp2_pair_62 = ex2_emulation_f32x2_value(make_float2(sv[32], sv[33]));
                        sv[32] = _exp2_pair_62.x;
                        sv[33] = _exp2_pair_62.y;
                        const float2 _fma_b2_63 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_64 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 44))[_lf], _fma_b2_63, _fma_c2_64);
                        psum0 = add_f32x2(psum0, make_float2(sv[30], sv[31]));
                        unsigned int sv_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                            sv_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        float2 _exp2_pair_65 = ex2_emulation_f32x2_value(make_float2(sv[34], sv[35]));
                        sv[34] = _exp2_pair_65.x;
                        sv[35] = _exp2_pair_65.y;
                        const float2 _fma_b2_66 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_67 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 46))[_lf], _fma_b2_66, _fma_c2_67);
                        float2 psum1 = make_float2(sv[32], sv[33]);
                        float2 _exp2_pair_68 = ex2_emulation_f32x2_value(make_float2(sv[36], sv[37]));
                        sv[36] = _exp2_pair_68.x;
                        sv[37] = _exp2_pair_68.y;
                        const float2 _fma_b2_69 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_70 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_69, _fma_c2_70);
                        psum1 = add_f32x2(psum1, make_float2(sv[34], sv[35]));
                        float2 _exp2_pair_71 = ex2_emulation_f32x2_value(make_float2(sv[38], sv[39]));
                        sv[38] = _exp2_pair_71.x;
                        sv[39] = _exp2_pair_71.y;
                        const float2 _fma_b2_72 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_73 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 50))[_lf], _fma_b2_72, _fma_c2_73);
                        psum1 = add_f32x2(psum1, make_float2(sv[36], sv[37]));
                        float2 _exp2_pair_74 = ex2_emulation_f32x2_value(make_float2(sv[40], sv[41]));
                        sv[40] = _exp2_pair_74.x;
                        sv[41] = _exp2_pair_74.y;
                        const float2 _fma_b2_75 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_76 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 52))[_lf], _fma_b2_75, _fma_c2_76);
                        psum1 = add_f32x2(psum1, make_float2(sv[38], sv[39]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 42] = approx_exp2(sv[_le + 42]);
                        }
                        const float2 _fma_b2_77 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_78 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 54))[_lf], _fma_b2_77, _fma_c2_78);
                        psum1 = add_f32x2(psum1, make_float2(sv[40], sv[41]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 44] = approx_exp2(sv[_le + 44]);
                        }
                        const float2 _fma_b2_79 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_80 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 56))[_lf], _fma_b2_79, _fma_c2_80);
                        psum1 = add_f32x2(psum1, make_float2(sv[42], sv[43]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 46] = approx_exp2(sv[_le + 46]);
                        }
                        const float2 _fma_b2_81 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_82 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 58))[_lf], _fma_b2_81, _fma_c2_82);
                        psum1 = add_f32x2(psum1, make_float2(sv[44], sv[45]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 48] = approx_exp2(sv[_le + 48]);
                        }
                        const float2 _fma_b2_83 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_84 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 60))[_lf], _fma_b2_83, _fma_c2_84);
                        psum1 = add_f32x2(psum1, make_float2(sv[46], sv[47]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 50] = approx_exp2(sv[_le + 50]);
                        }
                        const float2 _fma_b2_85 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_86 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 62))[_lf], _fma_b2_85, _fma_c2_86);
                        psum1 = add_f32x2(psum1, make_float2(sv[48], sv[49]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 52] = approx_exp2(sv[_le + 52]);
                        }
                        const float2 _fma_b2_87 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_88 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_87, _fma_c2_88);
                        psum1 = add_f32x2(psum1, make_float2(sv[50], sv[51]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 54] = approx_exp2(sv[_le + 54]);
                        }
                        const float2 _fma_b2_89 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_90 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 66))[_lf], _fma_b2_89, _fma_c2_90);
                        psum1 = add_f32x2(psum1, make_float2(sv[52], sv[53]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 56] = approx_exp2(sv[_le + 56]);
                        }
                        const float2 _fma_b2_91 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_92 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 68))[_lf], _fma_b2_91, _fma_c2_92);
                        psum1 = add_f32x2(psum1, make_float2(sv[54], sv[55]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 58] = approx_exp2(sv[_le + 58]);
                        }
                        const float2 _fma_b2_93 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_94 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 70))[_lf], _fma_b2_93, _fma_c2_94);
                        psum1 = add_f32x2(psum1, make_float2(sv[56], sv[57]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 60] = approx_exp2(sv[_le + 60]);
                        }
                        const float2 _fma_b2_95 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_96 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 72))[_lf], _fma_b2_95, _fma_c2_96);
                        psum1 = add_f32x2(psum1, make_float2(sv[58], sv[59]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 62] = approx_exp2(sv[_le + 62]);
                        }
                        const float2 _fma_b2_97 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_98 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 74))[_lf], _fma_b2_97, _fma_c2_98);
                        psum1 = add_f32x2(psum1, make_float2(sv[60], sv[61]));
                        float2 _exp2_pair_99 = ex2_emulation_f32x2_value(make_float2(sv[64], sv[65]));
                        sv[64] = _exp2_pair_99.x;
                        sv[65] = _exp2_pair_99.y;
                        const float2 _fma_b2_100 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_101 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 76))[_lf], _fma_b2_100, _fma_c2_101);
                        psum1 = add_f32x2(psum1, make_float2(sv[62], sv[63]));
                        unsigned int sv_bf16_0[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 32], sv[_lp*2+1 + 32]));
                            sv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        float2 _exp2_pair_102 = ex2_emulation_f32x2_value(make_float2(sv[66], sv[67]));
                        sv[66] = _exp2_pair_102.x;
                        sv[67] = _exp2_pair_102.y;
                        const float2 _fma_b2_103 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_104 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 78))[_lf], _fma_b2_103, _fma_c2_104);
                        float2 psum2 = make_float2(sv[64], sv[65]);
                        float2 _exp2_pair_105 = ex2_emulation_f32x2_value(make_float2(sv[68], sv[69]));
                        sv[68] = _exp2_pair_105.x;
                        sv[69] = _exp2_pair_105.y;
                        const float2 _fma_b2_106 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_107 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_106, _fma_c2_107);
                        psum2 = add_f32x2(psum2, make_float2(sv[66], sv[67]));
                        float2 _exp2_pair_108 = ex2_emulation_f32x2_value(make_float2(sv[70], sv[71]));
                        sv[70] = _exp2_pair_108.x;
                        sv[71] = _exp2_pair_108.y;
                        const float2 _fma_b2_109 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_110 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 82))[_lf], _fma_b2_109, _fma_c2_110);
                        psum2 = add_f32x2(psum2, make_float2(sv[68], sv[69]));
                        float2 _exp2_pair_111 = ex2_emulation_f32x2_value(make_float2(sv[72], sv[73]));
                        sv[72] = _exp2_pair_111.x;
                        sv[73] = _exp2_pair_111.y;
                        const float2 _fma_b2_112 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_113 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 84))[_lf], _fma_b2_112, _fma_c2_113);
                        psum2 = add_f32x2(psum2, make_float2(sv[70], sv[71]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 74] = approx_exp2(sv[_le + 74]);
                        }
                        const float2 _fma_b2_114 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_115 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 86))[_lf], _fma_b2_114, _fma_c2_115);
                        psum2 = add_f32x2(psum2, make_float2(sv[72], sv[73]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 76] = approx_exp2(sv[_le + 76]);
                        }
                        const float2 _fma_b2_116 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_117 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 88))[_lf], _fma_b2_116, _fma_c2_117);
                        psum2 = add_f32x2(psum2, make_float2(sv[74], sv[75]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 78] = approx_exp2(sv[_le + 78]);
                        }
                        const float2 _fma_b2_118 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_119 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 90))[_lf], _fma_b2_118, _fma_c2_119);
                        psum2 = add_f32x2(psum2, make_float2(sv[76], sv[77]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 80] = approx_exp2(sv[_le + 80]);
                        }
                        const float2 _fma_b2_120 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_121 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 92))[_lf], _fma_b2_120, _fma_c2_121);
                        psum2 = add_f32x2(psum2, make_float2(sv[78], sv[79]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 82] = approx_exp2(sv[_le + 82]);
                        }
                        const float2 _fma_b2_122 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_123 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 94))[_lf], _fma_b2_122, _fma_c2_123);
                        psum2 = add_f32x2(psum2, make_float2(sv[80], sv[81]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 84] = approx_exp2(sv[_le + 84]);
                        }
                        const float2 _fma_b2_124 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_125 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_124, _fma_c2_125);
                        psum2 = add_f32x2(psum2, make_float2(sv[82], sv[83]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 86] = approx_exp2(sv[_le + 86]);
                        }
                        const float2 _fma_b2_126 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_127 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 98))[_lf], _fma_b2_126, _fma_c2_127);
                        psum2 = add_f32x2(psum2, make_float2(sv[84], sv[85]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 88] = approx_exp2(sv[_le + 88]);
                        }
                        const float2 _fma_b2_128 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_129 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 100))[_lf], _fma_b2_128, _fma_c2_129);
                        psum2 = add_f32x2(psum2, make_float2(sv[86], sv[87]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 90] = approx_exp2(sv[_le + 90]);
                        }
                        const float2 _fma_b2_130 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_131 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 102))[_lf], _fma_b2_130, _fma_c2_131);
                        psum2 = add_f32x2(psum2, make_float2(sv[88], sv[89]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 92] = approx_exp2(sv[_le + 92]);
                        }
                        const float2 _fma_b2_132 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_133 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 104))[_lf], _fma_b2_132, _fma_c2_133);
                        psum2 = add_f32x2(psum2, make_float2(sv[90], sv[91]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 94] = approx_exp2(sv[_le + 94]);
                        }
                        const float2 _fma_b2_134 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_135 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 106))[_lf], _fma_b2_134, _fma_c2_135);
                        psum2 = add_f32x2(psum2, make_float2(sv[92], sv[93]));
                        float2 _exp2_pair_136 = ex2_emulation_f32x2_value(make_float2(sv[96], sv[97]));
                        sv[96] = _exp2_pair_136.x;
                        sv[97] = _exp2_pair_136.y;
                        const float2 _fma_b2_137 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_138 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 108))[_lf], _fma_b2_137, _fma_c2_138);
                        psum2 = add_f32x2(psum2, make_float2(sv[94], sv[95]));
                        unsigned int sv_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 64], sv[_lp*2+1 + 64]));
                            sv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        float2 _exp2_pair_139 = ex2_emulation_f32x2_value(make_float2(sv[98], sv[99]));
                        sv[98] = _exp2_pair_139.x;
                        sv[99] = _exp2_pair_139.y;
                        const float2 _fma_b2_140 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_141 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 110))[_lf], _fma_b2_140, _fma_c2_141);
                        float2 psum3 = make_float2(sv[96], sv[97]);
                        float2 _exp2_pair_142 = ex2_emulation_f32x2_value(make_float2(sv[100], sv[101]));
                        sv[100] = _exp2_pair_142.x;
                        sv[101] = _exp2_pair_142.y;
                        const float2 _fma_b2_143 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_144 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_143, _fma_c2_144);
                        psum3 = add_f32x2(psum3, make_float2(sv[98], sv[99]));
                        float2 _exp2_pair_145 = ex2_emulation_f32x2_value(make_float2(sv[102], sv[103]));
                        sv[102] = _exp2_pair_145.x;
                        sv[103] = _exp2_pair_145.y;
                        const float2 _fma_b2_146 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_147 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 114))[_lf], _fma_b2_146, _fma_c2_147);
                        psum3 = add_f32x2(psum3, make_float2(sv[100], sv[101]));
                        float2 _exp2_pair_148 = ex2_emulation_f32x2_value(make_float2(sv[104], sv[105]));
                        sv[104] = _exp2_pair_148.x;
                        sv[105] = _exp2_pair_148.y;
                        const float2 _fma_b2_149 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_150 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 116))[_lf], _fma_b2_149, _fma_c2_150);
                        psum3 = add_f32x2(psum3, make_float2(sv[102], sv[103]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 106] = approx_exp2(sv[_le + 106]);
                        }
                        const float2 _fma_b2_151 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_152 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 118))[_lf], _fma_b2_151, _fma_c2_152);
                        psum3 = add_f32x2(psum3, make_float2(sv[104], sv[105]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 108] = approx_exp2(sv[_le + 108]);
                        }
                        const float2 _fma_b2_153 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_154 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 120))[_lf], _fma_b2_153, _fma_c2_154);
                        psum3 = add_f32x2(psum3, make_float2(sv[106], sv[107]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 110] = approx_exp2(sv[_le + 110]);
                        }
                        const float2 _fma_b2_155 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_156 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 122))[_lf], _fma_b2_155, _fma_c2_156);
                        psum3 = add_f32x2(psum3, make_float2(sv[108], sv[109]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 112] = approx_exp2(sv[_le + 112]);
                        }
                        const float2 _fma_b2_157 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_158 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 124))[_lf], _fma_b2_157, _fma_c2_158);
                        psum3 = add_f32x2(psum3, make_float2(sv[110], sv[111]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 114] = approx_exp2(sv[_le + 114]);
                        }
                        const float2 _fma_b2_159 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_160 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 126))[_lf], _fma_b2_159, _fma_c2_160);
                        psum3 = add_f32x2(psum3, make_float2(sv[112], sv[113]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 116] = approx_exp2(sv[_le + 116]);
                        }
                        psum3 = add_f32x2(psum3, make_float2(sv[114], sv[115]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 118] = approx_exp2(sv[_le + 118]);
                        }
                        psum3 = add_f32x2(psum3, make_float2(sv[116], sv[117]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 120] = approx_exp2(sv[_le + 120]);
                        }
                        psum3 = add_f32x2(psum3, make_float2(sv[118], sv[119]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 122] = approx_exp2(sv[_le + 122]);
                        }
                        psum3 = add_f32x2(psum3, make_float2(sv[120], sv[121]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 124] = approx_exp2(sv[_le + 124]);
                        }
                        psum3 = add_f32x2(psum3, make_float2(sv[122], sv[123]));
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 126] = approx_exp2(sv[_le + 126]);
                        }
                        psum3 = add_f32x2(psum3, make_float2(sv[124], sv[125]));
                        psum3 = add_f32x2(psum3, make_float2(sv[126], sv[127]));
                        unsigned int sv_bf16_2[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 96], sv[_lp*2+1 + 96]));
                            sv_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        int p_addr = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                        mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                        _phase_s_read ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (stage == 1 && two_stages == 1 && n_iter >= 1 && n_block == 0) {
                            mbarrier_wait(pv1_done_addr, _phase_pv1_done_0);
                            _phase_pv1_done_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_addr + 48), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        float2 sum01 = add_f32x2(psum0, psum1);
                        float2 sum23 = add_f32x2(psum2, psum3);
                        float2 total_sum = add_f32x2(sum01, sum23);
                        row_sum = __fmaf_rn(row_sum, acc_scale, total_sum.x) + total_sum.y;
                    }
                    unsigned int tk_slot = (tile_idx - cluster_id) / 1 % 2;
                    unsigned int tk_parity = (tile_idx - cluster_id) / 1 / 2 % 2;
                    mbarrier_wait(tk_full_addr + (tk_slot) * 8, tk_parity);
                    unsigned int ticket = (unsigned int)tickets[tk_slot];
                    asm volatile(
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                        :: "r"((tk_empty_addr + (tk_slot) * 8) & 0xFEFFFFFF) : "memory");
                    nxt_tile = ticket;
                    int rec_0 = nxt_tile * 8;
                    int doc_begin_1 = unit_table[rec_0];
                    int doc_len_2 = unit_table[rec_0 + 1];
                    int packed_3 = unit_table[rec_0 + 2];
                    int kv_words_4 = unit_table[rec_0 + 3];
                    int ws_slot_5 = unit_table[rec_0 + 4];
                    int q_half_6 = unit_table[rec_0 + 5];
                    nxt_doc_begin = doc_begin_1;
                    nxt_doc_len = doc_len_2;
                    nxt_packed = packed_3;
                    nxt_kv_words = kv_words_4;
                    nxt_ws_slot = ws_slot_5;
                    nxt_q_half = q_half_6;
                    if (stage == 0 && two_stages == 1) {
                        mbarrier_arrive(s_read_addr + (stage) * 8);
                    }
                    mbarrier_wait(scales_free_addr + (stage) * 8, _phase_scales_free);
                    _phase_scales_free ^= 1;
                    scales[warp % 4 * 32 + lane + scale_off] = row_max;
                    scales[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum;
                    if (sync_group == 0) {
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 2, 256;" ::: "memory");
                    }
                }
                if (active == 0) {
                    unsigned int tk_slot_1 = (tile_idx - cluster_id) / 1 % 2;
                    unsigned int tk_parity_1 = (tile_idx - cluster_id) / 1 / 2 % 2;
                    mbarrier_wait(tk_full_addr + (tk_slot_1) * 8, tk_parity_1);
                    unsigned int ticket_1 = (unsigned int)tickets[tk_slot_1];
                    asm volatile(
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                        :: "r"((tk_empty_addr + (tk_slot_1) * 8) & 0xFEFFFFFF) : "memory");
                    nxt_tile = ticket_1;
                    int rec_0_1 = nxt_tile * 8;
                    int doc_begin_1_1 = unit_table[rec_0_1];
                    int doc_len_2_1 = unit_table[rec_0_1 + 1];
                    int packed_3_1 = unit_table[rec_0_1 + 2];
                    int kv_words_4_1 = unit_table[rec_0_1 + 3];
                    int ws_slot_5_1 = unit_table[rec_0_1 + 4];
                    int q_half_6_1 = unit_table[rec_0_1 + 5];
                    nxt_doc_begin = doc_begin_1_1;
                    nxt_doc_len = doc_len_2_1;
                    nxt_packed = packed_3_1;
                    nxt_kv_words = kv_words_4_1;
                    nxt_ws_slot = ws_slot_5_1;
                    nxt_q_half = q_half_6_1;
                }
                if (nxt_tile >= total_tiles) {
                    break;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            int nxt_doc_begin_1;
            int nxt_doc_len_1;
            int nxt_packed_1;
            int nxt_kv_words_1;
            int nxt_ws_slot_1;
            int nxt_q_half_1;
            nxt_doc_begin_1 = doc_begin;
            nxt_doc_len_1 = doc_len;
            nxt_packed_1 = packed;
            nxt_kv_words_1 = kv_words;
            nxt_ws_slot_1 = ws_slot;
            nxt_q_half_1 = q_half;
            unsigned int _phase_scale_full_0 = 0;
            unsigned int _phase_scale_full_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles; tile_idx_1++) {
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
                unsigned int nxt_tile_1;
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                if (two_stages_1 == 1) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                }
                mbarrier_wait(scale_full_addr, _phase_scale_full_0);
                _phase_scale_full_0 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((scale_empty_addr) & 0xFEFFFFFF) : "memory");
                if (two_stages_1 == 1) {
                    mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
                    _phase_scale_full_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_empty_addr + 8) & 0xFEFFFFFF) : "memory");
                }
                if (two_stages_1 == 0) {
                    mbarrier_arrive(s_read_addr + 8);
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
                    mbarrier_wait(scale_full_addr, _phase_scale_full_0);
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
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (two_stages_1 == 0) {
                        mbarrier_arrive(s_read_addr + 8);
                    }
                    if (skip_rescale0 == 0) {
                        #pragma unroll
                        for (int group = 0; group < HEAD_DIM / 32; group++) {
                            int addr = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group * 2 * 16);
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], addr);
                            int addr_0 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group * 2 + 1) * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], addr_0);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            const float2 _scale2_0 = {scale0, scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            tmem_st_x16_f32(addr, _tmem_load_1);
                            const float2 _scale2_1 = {scale0, scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_1);
                            tmem_st_x16_f32(addr_0, _tmem_load_2);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    if (two_stages_1 == 1) {
                        mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
                        _phase_scale_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_3[1];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                            " {%0}, [%1];"
                            : "=f"(_tmem_load_3[0])
                            : "r"(taddr + (unsigned int)BLOCK_M + (unsigned int)(warp % 4 * 32 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float scale1 = _tmem_load_3[0];
                        int skip_rescale1 = __all_sync(0xFFFFFFFF, scale1 == 1.0f);
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((scale_empty_addr + 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                        _phase_o_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (skip_rescale1 == 0) {
                            #pragma unroll
                            for (int group_1 = 0; group_1 < HEAD_DIM / 32; group_1++) {
                                int addr_1 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_1 * 2 * 16);
                                float _tmem_load_4[16];
                                tmem_ld_x16(&_tmem_load_4[0], addr_1);
                                int addr_0_1 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_1 * 2 + 1) * 16);
                                float _tmem_load_5[16];
                                tmem_ld_x16(&_tmem_load_5[0], addr_0_1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_2 = {scale1, scale1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_2);
                                tmem_st_x16_f32(addr_1, _tmem_load_4);
                                const float2 _scale2_3 = {scale1, scale1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_3);
                                tmem_st_x16_f32(addr_0_1, _tmem_load_5);
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    }
                }
                unsigned int tk_slot_2 = (tile_idx_1 - cluster_id) / 1 % 2;
                unsigned int tk_parity_2 = (tile_idx_1 - cluster_id) / 1 / 2 % 2;
                mbarrier_wait(tk_full_addr + (tk_slot_2) * 8, tk_parity_2);
                unsigned int ticket_2 = (unsigned int)tickets[tk_slot_2];
                asm volatile(
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                    :: "r"((tk_empty_addr + (tk_slot_2) * 8) & 0xFEFFFFFF) : "memory");
                nxt_tile_1 = ticket_2;
                int rec_3 = nxt_tile_1 * 8;
                int doc_begin_4 = unit_table[rec_3];
                int doc_len_5 = unit_table[rec_3 + 1];
                int packed_6 = unit_table[rec_3 + 2];
                int kv_words_7 = unit_table[rec_3 + 3];
                int ws_slot_8 = unit_table[rec_3 + 4];
                int q_half_9 = unit_table[rec_3 + 5];
                nxt_doc_begin_1 = doc_begin_4;
                nxt_doc_len_1 = doc_len_5;
                nxt_packed_1 = packed_6;
                nxt_kv_words_1 = kv_words_7;
                nxt_ws_slot_1 = ws_slot_8;
                nxt_q_half_1 = q_half_9;
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_sum = scales[warp % 4 * 32 + lane + 2 * BLOCK_M];
                float ref_max = scales[warp % 4 * 32 + lane];
                mbarrier_arrive(scales_free_addr);
                float final_scale;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_4 = approx_rcp(final_sum);
                    final_scale = _rcp_4;
                } else {
                    final_scale = 0.0f;
                }
                int local_row = unit_row0_1 + cta_rank * BLOCK_M + (warp % 4 * 32 + lane);
                int out_row = (doc_begin_0_1 + local_row) * num_heads + head_1;
                if (ws_slot_2_1 >= 0) {
                    int partial_row = ws_slot_2_1 * 512 + cta_rank * BLOCK_M + (warp % 4 * 32 + lane);
                    if (local_row < doc_len_1_1) {
                        partial_ML[partial_row * 2] = ref_max * softmax_scale_log2;
                        partial_ML[partial_row * 2 + 1] = final_sum;
                    }
                    #pragma unroll
                    for (int group_2 = 0; group_2 < HEAD_DIM / 32; group_2++) {
                        int addr_2 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_2 * 2 * 16);
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], addr_2);
                        int addr_0_2 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_2 * 2 + 1) * 16);
                        float _tmem_load_7[16];
                        tmem_ld_x16(&_tmem_load_7[0], addr_0_2);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (local_row < doc_len_1_1) {
                            float _scaled_0[16];
                            const float2 _scale2_5 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_5);
                            unsigned int _packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 4], _scaled_0[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 6], _scaled_0[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 8], _scaled_0[_lp*2+1 + 8]));
                                _packed_0[_lp + 4] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 10], _scaled_0[_lp*2+1 + 10]));
                                _packed_0[_lp + 5] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 12], _scaled_0[_lp*2+1 + 12]));
                                _packed_0[_lp + 6] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_0[_lp*2 + 14], _scaled_0[_lp*2+1 + 14]));
                                _packed_0[_lp + 7] = *(uint32_t*)&_h2;
                            }
                            reinterpret_cast<int4*>(partial_O + ((long long)partial_row * (long long)HEAD_DIM + (long long)(group_2 * 2 * 16)))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                            reinterpret_cast<int4*>(partial_O + ((long long)partial_row * (long long)HEAD_DIM + (long long)(group_2 * 2 * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_0 + (4))[0];
                        }
                        if (local_row < doc_len_1_1) {
                            float _scaled_1[16];
                            const float2 _scale2_6 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_6);
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 2], _scaled_1[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 4], _scaled_1[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 6], _scaled_1[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 8], _scaled_1[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 10], _scaled_1[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 12], _scaled_1[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_scaled_1[_lp*2 + 14], _scaled_1[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_h2;
                            }
                            reinterpret_cast<int4*>(partial_O + ((long long)partial_row * (long long)HEAD_DIM + (long long)((group_2 * 2 + 1) * 16)))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                            reinterpret_cast<int4*>(partial_O + ((long long)partial_row * (long long)HEAD_DIM + (long long)((group_2 * 2 + 1) * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_1 + (4))[0];
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
                        for (int group_3 = 0; group_3 < HEAD_DIM / 32; group_3++) {
                            int addr_3 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_3 * 2 * 16);
                            float _tmem_load_8[16];
                            tmem_ld_x16(&_tmem_load_8[0], addr_3);
                            int addr_0_3 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_3 * 2 + 1) * 16);
                            float _tmem_load_9[16];
                            tmem_ld_x16(&_tmem_load_9[0], addr_0_3);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            const float2 _scale2_7 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_7);
                            unsigned int _tmem_load_8_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int slab = group_3 * 2 / 4;
                            int col_bytes = group_3 * 2 % 4 * 32;
                            int slab_addr = smem_o_addr + (unsigned int)(slab * 16384);
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr + ((warp % 4 * 32 + lane) * 128 + col_bytes ^ ((warp % 4 * 32 + lane) * 128 + col_bytes >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr + ((warp % 4 * 32 + lane) * 128 + (col_bytes + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_bytes + 16) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                            const float2 _scale2_8 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], _scale2_8);
                            unsigned int _tmem_load_9_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                                _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int slab_1 = (group_3 * 2 + 1) / 4;
                            int col_bytes_2 = (group_3 * 2 + 1) % 4 * 32;
                            int slab_addr_3 = smem_o_addr + (unsigned int)(slab_1 * 16384);
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_3 + ((warp % 4 * 32 + lane) * 128 + col_bytes_2 ^ ((warp % 4 * 32 + lane) * 128 + col_bytes_2 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[0]), "r"(_tmem_load_9_bf16[1]), "r"(_tmem_load_9_bf16[2]), "r"(_tmem_load_9_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_3 + ((warp % 4 * 32 + lane) * 128 + (col_bytes_2 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_bytes_2 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[4]), "r"(_tmem_load_9_bf16[5]), "r"(_tmem_load_9_bf16[6]), "r"(_tmem_load_9_bf16[7]) : "memory");
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
                        for (int group_4 = 0; group_4 < HEAD_DIM / 32; group_4++) {
                            int addr_4 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_4 * 2 * 16);
                            float _tmem_load_10[16];
                            tmem_ld_x16(&_tmem_load_10[0], addr_4);
                            int addr_0_4 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_4 * 2 + 1) * 16);
                            float _tmem_load_11[16];
                            tmem_ld_x16(&_tmem_load_11[0], addr_0_4);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (local_row < doc_len_1_1) {
                                float _scaled_2[16];
                                const float2 _scale2_9 = {final_scale, final_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_10)[_ls], _scale2_9);
                                unsigned int _packed_2[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 2], _scaled_2[_lp*2+1 + 2]));
                                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 4], _scaled_2[_lp*2+1 + 4]));
                                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 6], _scaled_2[_lp*2+1 + 6]));
                                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 8], _scaled_2[_lp*2+1 + 8]));
                                    _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 10], _scaled_2[_lp*2+1 + 10]));
                                    _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 12], _scaled_2[_lp*2+1 + 12]));
                                    _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 14], _scaled_2[_lp*2+1 + 14]));
                                    _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O_raw + ((long long)out_row * (long long)HEAD_DIM + (long long)(group_4 * 2 * 16)))[0] = reinterpret_cast<int4*>(_packed_2)[0];
                                reinterpret_cast<int4*>(O_raw + ((long long)out_row * (long long)HEAD_DIM + (long long)(group_4 * 2 * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_2 + (4))[0];
                            }
                            if (local_row < doc_len_1_1) {
                                float _scaled_3[16];
                                const float2 _scale2_10 = {final_scale, final_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_11)[_ls], _scale2_10);
                                unsigned int _packed_3[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 0], _scaled_3[_lp*2+1 + 0]));
                                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 2], _scaled_3[_lp*2+1 + 2]));
                                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 4], _scaled_3[_lp*2+1 + 4]));
                                    _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 6], _scaled_3[_lp*2+1 + 6]));
                                    _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 8], _scaled_3[_lp*2+1 + 8]));
                                    _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 10], _scaled_3[_lp*2+1 + 10]));
                                    _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 12], _scaled_3[_lp*2+1 + 12]));
                                    _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 14], _scaled_3[_lp*2+1 + 14]));
                                    _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O_raw + ((long long)out_row * (long long)HEAD_DIM + (long long)((group_4 * 2 + 1) * 16)))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                                reinterpret_cast<int4*>(O_raw + ((long long)out_row * (long long)HEAD_DIM + (long long)((group_4 * 2 + 1) * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_3 + (4))[0];
                            }
                        }
                    }
                }
                if (two_stages_1 == 1) {
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                    _phase_o_full_1 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_sum_0 = scales[warp % 4 * 32 + lane + BLOCK_M + 2 * BLOCK_M];
                    float ref_max_1 = scales[warp % 4 * 32 + lane + BLOCK_M];
                    mbarrier_arrive(scales_free_addr + 8);
                    float final_scale_2;
                    if (final_sum_0 != 0.0f && final_sum_0 == final_sum_0) {
                        float _rcp_11 = approx_rcp(final_sum_0);
                        final_scale_2 = _rcp_11;
                    } else {
                        final_scale_2 = 0.0f;
                    }
                    int local_row_3 = unit_row0_1 + (2 + cta_rank) * BLOCK_M + (warp % 4 * 32 + lane);
                    int out_row_4 = (doc_begin_0_1 + local_row_3) * num_heads + head_1;
                    if (ws_slot_2_1 >= 0) {
                        int partial_row_1 = ws_slot_2_1 * 512 + (2 + cta_rank) * BLOCK_M + (warp % 4 * 32 + lane);
                        if (local_row_3 < doc_len_1_1) {
                            partial_ML[partial_row_1 * 2] = ref_max_1 * softmax_scale_log2;
                            partial_ML[partial_row_1 * 2 + 1] = final_sum_0;
                        }
                        #pragma unroll
                        for (int group_5 = 0; group_5 < HEAD_DIM / 32; group_5++) {
                            int addr_5 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_5 * 2 * 16);
                            float _tmem_load_12[16];
                            tmem_ld_x16(&_tmem_load_12[0], addr_5);
                            int addr_0_5 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_5 * 2 + 1) * 16);
                            float _tmem_load_13[16];
                            tmem_ld_x16(&_tmem_load_13[0], addr_0_5);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (local_row_3 < doc_len_1_1) {
                                float _scaled_4[16];
                                const float2 _scale2_12 = {final_scale_2, final_scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_12);
                                unsigned int _packed_4[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 0], _scaled_4[_lp*2+1 + 0]));
                                    _packed_4[_lp] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 2], _scaled_4[_lp*2+1 + 2]));
                                    _packed_4[_lp + 1] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 4], _scaled_4[_lp*2+1 + 4]));
                                    _packed_4[_lp + 2] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 6], _scaled_4[_lp*2+1 + 6]));
                                    _packed_4[_lp + 3] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 8], _scaled_4[_lp*2+1 + 8]));
                                    _packed_4[_lp + 4] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 10], _scaled_4[_lp*2+1 + 10]));
                                    _packed_4[_lp + 5] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 12], _scaled_4[_lp*2+1 + 12]));
                                    _packed_4[_lp + 6] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_4[_lp*2 + 14], _scaled_4[_lp*2+1 + 14]));
                                    _packed_4[_lp + 7] = *(uint32_t*)&_h2;
                                }
                                reinterpret_cast<int4*>(partial_O + ((long long)partial_row_1 * (long long)HEAD_DIM + (long long)(group_5 * 2 * 16)))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                                reinterpret_cast<int4*>(partial_O + ((long long)partial_row_1 * (long long)HEAD_DIM + (long long)(group_5 * 2 * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_4 + (4))[0];
                            }
                            if (local_row_3 < doc_len_1_1) {
                                float _scaled_5[16];
                                const float2 _scale2_13 = {final_scale_2, final_scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_13);
                                unsigned int _packed_5[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 0], _scaled_5[_lp*2+1 + 0]));
                                    _packed_5[_lp] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 2], _scaled_5[_lp*2+1 + 2]));
                                    _packed_5[_lp + 1] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 4], _scaled_5[_lp*2+1 + 4]));
                                    _packed_5[_lp + 2] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 6], _scaled_5[_lp*2+1 + 6]));
                                    _packed_5[_lp + 3] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 8], _scaled_5[_lp*2+1 + 8]));
                                    _packed_5[_lp + 4] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 10], _scaled_5[_lp*2+1 + 10]));
                                    _packed_5[_lp + 5] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 12], _scaled_5[_lp*2+1 + 12]));
                                    _packed_5[_lp + 6] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_5[_lp*2 + 14], _scaled_5[_lp*2+1 + 14]));
                                    _packed_5[_lp + 7] = *(uint32_t*)&_h2;
                                }
                                reinterpret_cast<int4*>(partial_O + ((long long)partial_row_1 * (long long)HEAD_DIM + (long long)((group_5 * 2 + 1) * 16)))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                                reinterpret_cast<int4*>(partial_O + ((long long)partial_row_1 * (long long)HEAD_DIM + (long long)((group_5 * 2 + 1) * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_5 + (4))[0];
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
                            for (int group_6 = 0; group_6 < HEAD_DIM / 32; group_6++) {
                                int addr_6 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_6 * 2 * 16);
                                float _tmem_load_14[16];
                                tmem_ld_x16(&_tmem_load_14[0], addr_6);
                                int addr_0_6 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_6 * 2 + 1) * 16);
                                float _tmem_load_15[16];
                                tmem_ld_x16(&_tmem_load_15[0], addr_0_6);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_14 = {final_scale_2, final_scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_14)[_ls], _scale2_14);
                                unsigned int _tmem_load_14_bf16[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 0], _tmem_load_14[_lp*2+1 + 0]));
                                    _tmem_load_14_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int slab_2 = 2 + group_6 * 2 / 4;
                                int col_bytes_1 = group_6 * 2 % 4 * 32;
                                int slab_addr_1 = smem_o_addr + (unsigned int)(slab_2 * 16384);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_1 + ((warp % 4 * 32 + lane) * 128 + col_bytes_1 ^ ((warp % 4 * 32 + lane) * 128 + col_bytes_1 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[0]), "r"(_tmem_load_14_bf16[1]), "r"(_tmem_load_14_bf16[2]), "r"(_tmem_load_14_bf16[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_1 + ((warp % 4 * 32 + lane) * 128 + (col_bytes_1 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_bytes_1 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[4]), "r"(_tmem_load_14_bf16[5]), "r"(_tmem_load_14_bf16[6]), "r"(_tmem_load_14_bf16[7]) : "memory");
                                const float2 _scale2_15 = {final_scale_2, final_scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_15)[_ls], _scale2_15);
                                unsigned int _tmem_load_15_bf16[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 0], _tmem_load_15[_lp*2+1 + 0]));
                                    _tmem_load_15_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int slab_1_1 = 2 + (group_6 * 2 + 1) / 4;
                                int col_bytes_2_1 = (group_6 * 2 + 1) % 4 * 32;
                                int slab_addr_3_1 = smem_o_addr + (unsigned int)(slab_1_1 * 16384);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_3_1 + ((warp % 4 * 32 + lane) * 128 + col_bytes_2_1 ^ ((warp % 4 * 32 + lane) * 128 + col_bytes_2_1 >> 7 & 7) << 4))), "r"(_tmem_load_15_bf16[0]), "r"(_tmem_load_15_bf16[1]), "r"(_tmem_load_15_bf16[2]), "r"(_tmem_load_15_bf16[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((slab_addr_3_1 + ((warp % 4 * 32 + lane) * 128 + (col_bytes_2_1 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_bytes_2_1 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_15_bf16[4]), "r"(_tmem_load_15_bf16[5]), "r"(_tmem_load_15_bf16[6]), "r"(_tmem_load_15_bf16[7]) : "memory");
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
                            for (int group_7 = 0; group_7 < HEAD_DIM / 32; group_7++) {
                                int addr_7 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(group_7 * 2 * 16);
                                float _tmem_load_16[16];
                                tmem_ld_x16(&_tmem_load_16[0], addr_7);
                                int addr_0_7 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)((group_7 * 2 + 1) * 16);
                                float _tmem_load_17[16];
                                tmem_ld_x16(&_tmem_load_17[0], addr_0_7);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                if (local_row_3 < doc_len_1_1) {
                                    float _scaled_6[16];
                                    const float2 _scale2_16 = {final_scale_2, final_scale_2};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_16)[_ls], _scale2_16);
                                    unsigned int _packed_6[8];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 0], _scaled_6[_lp*2+1 + 0]));
                                        _packed_6[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 2], _scaled_6[_lp*2+1 + 2]));
                                        _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 4], _scaled_6[_lp*2+1 + 4]));
                                        _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 6], _scaled_6[_lp*2+1 + 6]));
                                        _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 8], _scaled_6[_lp*2+1 + 8]));
                                        _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 10], _scaled_6[_lp*2+1 + 10]));
                                        _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 12], _scaled_6[_lp*2+1 + 12]));
                                        _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 14], _scaled_6[_lp*2+1 + 14]));
                                        _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(O_raw + ((long long)out_row_4 * (long long)HEAD_DIM + (long long)(group_7 * 2 * 16)))[0] = reinterpret_cast<int4*>(_packed_6)[0];
                                    reinterpret_cast<int4*>(O_raw + ((long long)out_row_4 * (long long)HEAD_DIM + (long long)(group_7 * 2 * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_6 + (4))[0];
                                }
                                if (local_row_3 < doc_len_1_1) {
                                    float _scaled_7[16];
                                    const float2 _scale2_17 = {final_scale_2, final_scale_2};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_17)[_ls], _scale2_17);
                                    unsigned int _packed_7[8];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 0], _scaled_7[_lp*2+1 + 0]));
                                        _packed_7[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 2], _scaled_7[_lp*2+1 + 2]));
                                        _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 4], _scaled_7[_lp*2+1 + 4]));
                                        _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 6], _scaled_7[_lp*2+1 + 6]));
                                        _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 8], _scaled_7[_lp*2+1 + 8]));
                                        _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 10], _scaled_7[_lp*2+1 + 10]));
                                        _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 12], _scaled_7[_lp*2+1 + 12]));
                                        _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 14], _scaled_7[_lp*2+1 + 14]));
                                        _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(O_raw + ((long long)out_row_4 * (long long)HEAD_DIM + (long long)((group_7 * 2 + 1) * 16)))[0] = reinterpret_cast<int4*>(_packed_7)[0];
                                    reinterpret_cast<int4*>(O_raw + ((long long)out_row_4 * (long long)HEAD_DIM + (long long)((group_7 * 2 + 1) * 16) + 8))[0] = reinterpret_cast<int4*>(_packed_7 + (4))[0];
                                }
                            }
                        }
                    }
                }
                if (nxt_tile_1 >= total_tiles) {
                    break;
                }
            }
            if (warp == 8) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 9, 384;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_scale_empty_0 = 1;
            unsigned int _phase_scale_empty_1 = 1;
            unsigned int _phase_kv_full_1 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int k_stage = 0;
                unsigned int v_stage = 1;
                int nxt_doc_begin_2;
                int nxt_doc_len_2;
                int nxt_packed_2;
                int nxt_kv_words_2;
                int nxt_ws_slot_2;
                int nxt_q_half_2;
                nxt_doc_begin_2 = doc_begin;
                nxt_doc_len_2 = doc_len;
                nxt_packed_2 = packed;
                nxt_kv_words_2 = kv_words;
                nxt_ws_slot_2 = ws_slot;
                nxt_q_half_2 = q_half;
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles; tile_idx_2++) {
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
                    unsigned int nxt_tile_2;
                    mbarrier_wait_cluster_hint(q_full_addr, _phase_q_full_0, 10000000);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, _phase_kv_full);
                    mbarrier_wait(scale_empty_addr, _phase_scale_empty_0);
                    _phase_scale_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = ((smem_q0_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0) : "memory");
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
                        mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
                        _phase_scale_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = ((smem_q1_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_1 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (128))), "r"(0) : "memory");
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 5) { k_stage = 0; _phase_kv_full ^= 1; }
                    k_stage += 1;
                    if (k_stage == 5) { k_stage = 0; _phase_kv_full ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 0; n_iter_2 < num_n_blocks_2 - 1; n_iter_2++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, _phase_kv_full);
                        mbarrier_wait(scale_empty_addr, _phase_scale_empty_0);
                        _phase_scale_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = ((smem_q0_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_2 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_scores), "r"(0) : "memory");
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, _phase_kv_full_1);
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        if (two_stages_2 == 1) {
                            mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
                            _phase_scale_empty_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = ((smem_q1_addr) >> 4) & 0x3FFF;
                            int _mma_b_lo_4 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_4), "r"((tmem_scores + (128))), "r"(0) : "memory");
                            elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 5) { k_stage = 0; _phase_kv_full ^= 1; }
                        k_stage += 1;
                        if (k_stage == 5) { k_stage = 0; _phase_kv_full ^= 1; }
                        if (two_stages_2 == 1) {
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                            _phase_p_full_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                            elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; _phase_kv_full_1 ^= 1; }
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; _phase_kv_full_1 ^= 1; }
                        first_pv = 0;
                    }
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    if (two_stages_2 == 1 && num_n_blocks_2 > 1) {
                        elect_commit_cg2_multicast(pv1_done_addr, (uint16_t)(3));
                    }
                    int first_pv_flag_1 = first_pv;
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, _phase_kv_full_1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_6), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_7 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_7), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                    v_stage += 1;
                    if (v_stage == 5) { v_stage = 0; _phase_kv_full_1 ^= 1; }
                    v_stage += 1;
                    if (v_stage == 5) { v_stage = 0; _phase_kv_full_1 ^= 1; }
                    unsigned int tk_slot_3 = (tile_idx_2 - cluster_id) / 1 % 2;
                    unsigned int tk_parity_3 = (tile_idx_2 - cluster_id) / 1 / 2 % 2;
                    mbarrier_wait(tk_full_addr + (tk_slot_3) * 8, tk_parity_3);
                    unsigned int ticket_3 = (unsigned int)tickets[tk_slot_3];
                    nxt_tile_2 = ticket_3;
                    int rec_3_1 = nxt_tile_2 * 8;
                    int doc_begin_4_1 = unit_table[rec_3_1];
                    int doc_len_5_1 = unit_table[rec_3_1 + 1];
                    int packed_6_1 = unit_table[rec_3_1 + 2];
                    int kv_words_7_1 = unit_table[rec_3_1 + 3];
                    int ws_slot_8_1 = unit_table[rec_3_1 + 4];
                    int q_half_9_1 = unit_table[rec_3_1 + 5];
                    nxt_doc_begin_2 = doc_begin_4_1;
                    nxt_doc_len_2 = doc_len_5_1;
                    nxt_packed_2 = packed_6_1;
                    nxt_kv_words_2 = kv_words_7_1;
                    nxt_ws_slot_2 = ws_slot_8_1;
                    nxt_q_half_2 = q_half_9_1;
                    if (nxt_tile_2 >= total_tiles) {
                        break;
                    }
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
            nxt_doc_begin_3 = doc_begin;
            nxt_doc_len_3 = doc_len;
            nxt_packed_3 = packed;
            nxt_kv_words_3 = kv_words;
            nxt_ws_slot_3 = ws_slot;
            nxt_q_half_3 = q_half;
            int ticket_idx = (total_tiles + num_clusters) * 8;
            int drawn;
            int ticket_4;
            int mailbox = (total_tiles + cluster_id) * 8 + 6 + 1;
            if (cta_rank == 0) {
                mbarrier_wait_cluster(tk_empty_addr, 1);
                if (lane == 0) {
                    int _atomic_fetch_add_0 = atomicAdd(&unit_table[ticket_idx], 1);
                    drawn = _atomic_fetch_add_0;
                }
            }
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles; tile_idx_3++) {
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
                int q_row0 = doc_begin_0_3 + unit_row0_3 + cta_rank * BLOCK_M;
                int q_row1 = q_row0 + 256;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    if (two_stages_3 == 1) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_q0_addr, (&Q), 0, q_row0, head_3, 0, ((q_full_addr) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_q1_addr, (&Q), 0, q_row1, head_3, 0, ((q_full_addr) & 0xFEFFFFFF));
                    } else {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_q0_addr, (&Q), 0, q_row0, head_3, 0, ((q_full_addr) & 0xFEFFFFFF));
                    }
                }
                unsigned int tk_slot_4 = (tile_idx_3 - cluster_id) / 1 % 2;
                unsigned int tk_parity_4 = (tile_idx_3 - cluster_id) / 1 / 2 % 2;
                if (cta_rank == 0) {
                    if (lane == 0) {
                        int grid_clusters = num_clusters;
                        ticket_4 = drawn + grid_clusters;
                        unit_table[mailbox] = ticket_4;
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(tk_go_addr + (tk_slot_4) * 8), "r"(1) : "memory");
                        int last_draw = total_tiles - 1;
                        if (drawn == last_draw) {
                            unit_table[ticket_idx] = 0;
                        }
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
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, (&K), 0, kv_row + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_v_addr + load_stage * 16384, (&V), 0, kv_row, cta_rank, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                if (cta_rank == 1) {
                    unsigned int tk_slot_go = (tile_idx_3 - cluster_id) / 1 % 2;
                    unsigned int tk_parity_go = (tile_idx_3 - cluster_id) / 1 / 2 % 2;
                    mbarrier_wait_cluster(tk_go_addr + (tk_slot_go) * 8, tk_parity_go);
                    if (lane == 0) {
                        int _atomic_fetch_add_1 = atomicAdd(&unit_table[mailbox], 0);
                        ticket_4 = _atomic_fetch_add_1;
                    }
                }
                if (lane == 0) {
                    tickets[tk_slot_4] = ticket_4;
                    mbarrier_arrive(tk_full_addr + (tk_slot_4) * 8);
                }
                mailbox = mailbox ^ 1;
                int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, ticket_4, 0);
                unsigned int nxt_tile_3 = (unsigned int)_shfl_sync_2;
                int rec_3_2 = nxt_tile_3 * 8;
                int doc_begin_4_2 = unit_table[rec_3_2];
                int doc_len_5_2 = unit_table[rec_3_2 + 1];
                int packed_6_2 = unit_table[rec_3_2 + 2];
                int kv_words_7_2 = unit_table[rec_3_2 + 3];
                int ws_slot_8_2 = unit_table[rec_3_2 + 4];
                int q_half_9_2 = unit_table[rec_3_2 + 5];
                nxt_doc_begin_3 = doc_begin_4_2;
                nxt_doc_len_3 = doc_len_5_2;
                nxt_packed_3 = packed_6_2;
                nxt_kv_words_3 = kv_words_7_2;
                nxt_ws_slot_3 = ws_slot_8_2;
                nxt_q_half_3 = q_half_9_2;
                if (nxt_tile_3 < total_tiles) {
                    if (cta_rank == 0) {
                        unsigned int tk_slot2 = ((tile_idx_3 - cluster_id) / 1 + 1) % 2;
                        unsigned int tk_parity2 = ((tile_idx_3 - cluster_id) / 1 + 1) / 2 % 2 ^ 1;
                        mbarrier_wait_cluster(tk_empty_addr + (tk_slot2) * 8, tk_parity2);
                        if (lane == 0) {
                            int _atomic_fetch_add_3 = atomicAdd(&unit_table[ticket_idx], 1);
                            drawn = _atomic_fetch_add_3;
                        }
                    }
                }
                if (nxt_tile_3 >= total_tiles) {
                    if (cta_rank == 0) {
                        unsigned int tk_parity0 = ((tile_idx_3 - cluster_id) / 1 + 1) / 2 % 2 ^ 1;
                        unsigned int tk_slot0 = ((tile_idx_3 - cluster_id) / 1 + 1) % 2;
                        mbarrier_wait_cluster(tk_empty_addr + (tk_slot0) * 8, tk_parity0);
                        unsigned int tk_slot1 = (tile_idx_3 - cluster_id) / 1 % 2;
                        unsigned int tk_parity1 = (tile_idx_3 - cluster_id) / 1 / 2 % 2;
                        mbarrier_wait_cluster(tk_empty_addr + (tk_slot1) * 8, tk_parity1);
                    }
                    break;
                }
            }
            #pragma unroll
            for (int _ = 0; _ < 5; _++) {
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                load_stage += 1;
                if (load_stage == 5) { load_stage = 0; _phase_kv_empty ^= 1; }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
        }
    }
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
