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
#define SMEM_TOTAL 216064
#if __CUDA_ARCH__ == 1000
#define USE_TMEM_LD_RED 0
#else
#define USE_TMEM_LD_RED 1
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
#if __CUDA_ARCH__ == 1000
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

#if __CUDA_ARCH__ == 1000

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

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
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
kernel_cake_minimax_h3_varlen_attention_729b06c00352d9363c74(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, __nv_bfloat16* __restrict__ O_raw, int* __restrict__ unit_table, __half* __restrict__ partial_O, float* __restrict__ partial_ML, unsigned int total_tiles, int num_heads, float softmax_scale_log2)
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
    #define p_full_2_addr (mbar_base + 128)
    #define scale_full_addr (mbar_base + 144)
    #define scale_empty_addr (mbar_base + 160)
    #define s_read_addr (mbar_base + 176)
    #define o_full_addr (mbar_base + 192)
    #define pv0_done_addr (mbar_base + 208)

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

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[0..216)

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
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            // scale_full: 2 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            // scale_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // s_read: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // pv0_done: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 0) {
        int _tmem_hold = smem + 216;
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
    const int tmem_scores = taddr;
    const int tmem_probs_0 = taddr + 160;
    const int tmem_probs_1 = taddr + 32;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
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
            int rec = cluster_id * 8;
            int doc_begin = unit_table[rec];
            int doc_len = unit_table[rec + 1];
            int packed = unit_table[rec + 2];
            int kv_words = unit_table[rec + 3];
            int ws_slot = unit_table[rec + 4];
            nxt_doc_begin = doc_begin;
            nxt_doc_len = doc_len;
            nxt_packed = packed;
            nxt_kv_words = kv_words;
            nxt_ws_slot = ws_slot;
            unsigned int _phase_s_read = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_pv0_done_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int doc_begin_0 = nxt_doc_begin;
                int doc_len_1 = nxt_doc_len;
                int ws_slot_2 = nxt_ws_slot;
                int head = nxt_packed >> 16;
                int c = nxt_packed & 65535;
                int n_begin = nxt_kv_words >> 16;
                int num_n_blocks = nxt_kv_words & 65535;
                int unit_row0 = c * 512;
                int two_stages = ((doc_len_1 - unit_row0 > 256) ? 1 : 0);
                unsigned int nxt_tile = tile_idx + num_clusters;
                int rec_3 = nxt_tile * 8;
                int doc_begin_4 = unit_table[rec_3];
                int doc_len_5 = unit_table[rec_3 + 1];
                int packed_6 = unit_table[rec_3 + 2];
                int kv_words_7 = unit_table[rec_3 + 3];
                int ws_slot_8 = unit_table[rec_3 + 4];
                nxt_doc_begin = doc_begin_4;
                nxt_doc_len = doc_len_5;
                nxt_packed = packed_6;
                nxt_kv_words = kv_words_7;
                nxt_ws_slot = ws_slot_8;
                int active = ((stage == 0) ? 1 : two_stages);
                if (active == 1) {
                    int tail_base = doc_len_1 - n_begin * BLOCK_N;
                    float row_max = -CAKE_INF;
                    float row_sum = 0.0f;
                    if (stage == 1) {
                        mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                        _phase_s_read ^= 1;
                    }
                    #pragma unroll 1
                    for (unsigned int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                        int n_block = (unsigned int)(num_n_blocks - 1) - n_iter;
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 1000000);
#else
                        mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
#endif
                        _phase_s_full ^= 1;
                        int s_addr = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                        float sv[128];
                        float tile_max = -CAKE_INF;
                        {
#if !(__CUDA_ARCH__ == 1000)
                            float tile_max_lo = -CAKE_INF;
                            float tile_max_hi = -CAKE_INF;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1000
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                                : "r"(s_addr));
#else
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(tile_max_lo)
                                : "r"((unsigned int)tmem_scores + stage * 128 + (unsigned int)(warp % 4 * 32 << 16)));
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1000
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
                            float2 _reg_reduce_max2_0 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_0);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_0);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_0);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_0);
                            float sv_max = row_max_reduce(_reg_reduce_max2_0);
                            tile_max = sv_max;
#else
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(tile_max_hi)
                                : "r"((unsigned int)tmem_scores + (stage * 128 + 64) + (unsigned int)(warp % 4 * 32 << 16)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _max_0 = max_noftz(tile_max_lo, tile_max_hi);
                            tile_max = _max_0;
#endif
                        }
                        int tail_valid = tail_base - n_block * BLOCK_N;
                        if (tail_valid < BLOCK_N) {
                            uint32_t _slice_lo_mask_0;
                            {
#if __CUDA_ARCH__ == 1000
                                int _lim_1 = tail_valid;
                                if (_lim_1 <= 0) { _slice_lo_mask_0 = 0u; }
                                else if (_lim_1 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
#else
                                int _lim_0 = tail_valid;
                                if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                                else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
#endif
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1000
                                        "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_1));
#else
                                        "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
#endif
                                }
                            }
                            if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_INF;
                            uint32_t _slice_lo_mask_1;
                            {
#if __CUDA_ARCH__ == 1000
                                int _lim_2 = tail_valid - 32;
                                if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
#else
                                int _lim_1 = tail_valid - 32;
                                if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
#endif
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1000
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
#else
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
#endif
                                }
                            }
                            if (!(_slice_lo_mask_1 & (1u << 0))) sv[32] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 1))) sv[33] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 2))) sv[34] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 3))) sv[35] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 4))) sv[36] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 5))) sv[37] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 6))) sv[38] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 7))) sv[39] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 8))) sv[40] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 9))) sv[41] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 10))) sv[42] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 11))) sv[43] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 12))) sv[44] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 13))) sv[45] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 14))) sv[46] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 15))) sv[47] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 16))) sv[48] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 17))) sv[49] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 18))) sv[50] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 19))) sv[51] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 20))) sv[52] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 21))) sv[53] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 22))) sv[54] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 23))) sv[55] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 24))) sv[56] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 25))) sv[57] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 26))) sv[58] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 27))) sv[59] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 28))) sv[60] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 29))) sv[61] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 30))) sv[62] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 31))) sv[63] = -CAKE_INF;
                            uint32_t _slice_lo_mask_2;
                            {
#if __CUDA_ARCH__ == 1000
                                int _lim_3 = tail_valid - 64;
                                if (_lim_3 <= 0) { _slice_lo_mask_2 = 0u; }
                                else if (_lim_3 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
#else
                                int _lim_2 = tail_valid - 64;
                                if (_lim_2 <= 0) { _slice_lo_mask_2 = 0u; }
                                else if (_lim_2 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
#endif
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1000
                                        "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_3));
#else
                                        "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_2));
#endif
                                }
                            }
                            if (!(_slice_lo_mask_2 & (1u << 0))) sv[64] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 1))) sv[65] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 2))) sv[66] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 3))) sv[67] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 4))) sv[68] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 5))) sv[69] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 6))) sv[70] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 7))) sv[71] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 8))) sv[72] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 9))) sv[73] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 10))) sv[74] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 11))) sv[75] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 12))) sv[76] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 13))) sv[77] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 14))) sv[78] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 15))) sv[79] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 16))) sv[80] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 17))) sv[81] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 18))) sv[82] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 19))) sv[83] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 20))) sv[84] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 21))) sv[85] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 22))) sv[86] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 23))) sv[87] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 24))) sv[88] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 25))) sv[89] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 26))) sv[90] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 27))) sv[91] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 28))) sv[92] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 29))) sv[93] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 30))) sv[94] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 31))) sv[95] = -CAKE_INF;
                            uint32_t _slice_lo_mask_3;
                            {
#if __CUDA_ARCH__ == 1000
                                int _lim_4 = tail_valid - 96;
                                if (_lim_4 <= 0) { _slice_lo_mask_3 = 0u; }
                                else if (_lim_4 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
#else
                                int _lim_3 = tail_valid - 96;
                                if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                                else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
#endif
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1000
                                        "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_4));
#else
                                        "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
#endif
                                }
                            }
                            if (!(_slice_lo_mask_3 & (1u << 0))) sv[96] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 1))) sv[97] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 2))) sv[98] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 3))) sv[99] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 4))) sv[100] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 5))) sv[101] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 6))) sv[102] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 7))) sv[103] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 8))) sv[104] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 9))) sv[105] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 10))) sv[106] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 11))) sv[107] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 12))) sv[108] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 13))) sv[109] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 14))) sv[110] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 15))) sv[111] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 16))) sv[112] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 17))) sv[113] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 18))) sv[114] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 19))) sv[115] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 20))) sv[116] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 21))) sv[117] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 22))) sv[118] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 23))) sv[119] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 24))) sv[120] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 25))) sv[121] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 26))) sv[122] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 27))) sv[123] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 28))) sv[124] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 29))) sv[125] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 30))) sv[126] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 31))) sv[127] = -CAKE_INF;
#if __CUDA_ARCH__ == 1000
                            float2 _reg_reduce_max2_5 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_5);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_5);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_5);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_5);
                            float sv_max_1 = row_max_reduce(_reg_reduce_max2_5);
                            tile_max = sv_max_1;
#else
                            float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_4);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_4);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_4);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_4);
                            float sv_max = row_max_reduce(_reg_reduce_max2_4);
                            tile_max = sv_max;
#endif
                        }
                        if (two_stages == 1) {
                            mbarrier_arrive(s_read_addr + (stage) * 8);
                        }
                        float _max_1 = max_noftz(tile_max, row_max);
                        float new_max = _max_1;
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * softmax_scale_log2;
                        float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                        float acc_scale_log2 = _fma_0;
                        float acc_scale;
                        if (acc_scale_log2 >= -8.0f) {
                            safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                            acc_scale = 1.0f;
                            new_max_scaled = safe_max * softmax_scale_log2;
                        } else {
                            float _exp2_0 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
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
#if !(__CUDA_ARCH__ == 1000)
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
                        mbarrier_arrive(scale_full_addr + (stage) * 8);
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_6 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_7 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_6 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_6, _fma_c2_7);
                        const float2 _fma_b2_8 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_9 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_5, _fma_c2_6);
                        const float2 _fma_b2_7 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_8 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_8, _fma_c2_9);
                        const float2 _fma_b2_10 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_11 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_7, _fma_c2_8);
                        const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_10 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_10, _fma_c2_11);
                        const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_13 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_9, _fma_c2_10);
                        const float2 _fma_b2_11 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_12 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_12, _fma_c2_13);
                        const float2 _fma_b2_14 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_15 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_11, _fma_c2_12);
                        const float2 _fma_b2_13 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_14 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_14, _fma_c2_15);
                        const float2 _fma_b2_16 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_17 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_13, _fma_c2_14);
                        const float2 _fma_b2_15 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_16 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_16, _fma_c2_17);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_15, _fma_c2_16);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_18 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                                sv[_le*2] = _exp2_pair_18.x;
                                sv[_le*2 + 1] = _exp2_pair_18.y;
                            } else {
                                sv[_le*2] = approx_exp2(sv[_le*2]);
                                sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le] = approx_exp2(sv[_le]);
                        }
                        const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_18 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_17, _fma_c2_18);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 2] = approx_exp2(sv[_le + 2]);
#endif
                        }
                        const float2 _fma_b2_19 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_20 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_19, _fma_c2_20);
                        #pragma unroll
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_21 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 2], sv[_le*2 + 1 + 2]));
                                sv[_le*2 + 2] = _exp2_pair_21.x;
                                sv[_le*2 + 1 + 2] = _exp2_pair_21.y;
                            } else {
                                sv[_le*2 + 2] = approx_exp2(sv[_le*2 + 2]);
                                sv[_le*2 + 1 + 2] = approx_exp2(sv[_le*2 + 1 + 2]);
                            }
                        }
                        const float2 _fma_b2_22 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_23 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_22, _fma_c2_23);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_19, _fma_c2_20);
#endif
                        float2 _f2_0 = make_float2(sv[0], sv[1]);
                        float2 psum0 = _f2_0;
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_24 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 4], sv[_le*2 + 1 + 4]));
                                sv[_le*2 + 4] = _exp2_pair_24.x;
                                sv[_le*2 + 1 + 4] = _exp2_pair_24.y;
                            } else {
                                sv[_le*2 + 4] = approx_exp2(sv[_le*2 + 4]);
                                sv[_le*2 + 1 + 4] = approx_exp2(sv[_le*2 + 1 + 4]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 4] = approx_exp2(sv[_le + 4]);
                        }
                        const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_22 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_21, _fma_c2_22);
                        float2 _f2_1 = make_float2(sv[2], sv[3]);
                        psum0 = add_f32x2(psum0, _f2_1);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 6] = approx_exp2(sv[_le + 6]);
                        }
                        const float2 _fma_b2_23 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_24 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_23, _fma_c2_24);
                        float2 _f2_2 = make_float2(sv[4], sv[5]);
                        psum0 = add_f32x2(psum0, _f2_2);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 8] = approx_exp2(sv[_le + 8]);
#endif
                        }
                        const float2 _fma_b2_25 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_26 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_25, _fma_c2_26);
                        float2 _f2_1 = make_float2(sv[2], sv[3]);
                        psum0 = add_f32x2(psum0, _f2_1);
                        #pragma unroll
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_27 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 6], sv[_le*2 + 1 + 6]));
                                sv[_le*2 + 6] = _exp2_pair_27.x;
                                sv[_le*2 + 1 + 6] = _exp2_pair_27.y;
                            } else {
                                sv[_le*2 + 6] = approx_exp2(sv[_le*2 + 6]);
                                sv[_le*2 + 1 + 6] = approx_exp2(sv[_le*2 + 1 + 6]);
                            }
                        }
                        const float2 _fma_b2_28 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_29 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_28, _fma_c2_29);
                        float2 _f2_2 = make_float2(sv[4], sv[5]);
                        psum0 = add_f32x2(psum0, _f2_2);
                        #pragma unroll
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_30 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 8], sv[_le*2 + 1 + 8]));
                                sv[_le*2 + 8] = _exp2_pair_30.x;
                                sv[_le*2 + 1 + 8] = _exp2_pair_30.y;
                            } else {
                                sv[_le*2 + 8] = approx_exp2(sv[_le*2 + 8]);
                                sv[_le*2 + 1 + 8] = approx_exp2(sv[_le*2 + 1 + 8]);
                            }
                        }
                        const float2 _fma_b2_31 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_32 = {neg_max_scaled, neg_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_31, _fma_c2_32);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_25, _fma_c2_26);
#endif
                        float2 _f2_3 = make_float2(sv[6], sv[7]);
                        psum0 = add_f32x2(psum0, _f2_3);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 10] = approx_exp2(sv[_le + 10]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_33 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_34 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_27 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_28 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_33, _fma_c2_34);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_27, _fma_c2_28);
#endif
                        float2 _f2_4 = make_float2(sv[8], sv[9]);
                        psum0 = add_f32x2(psum0, _f2_4);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 12] = approx_exp2(sv[_le + 12]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_35 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_36 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_29 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_30 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_35, _fma_c2_36);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_29, _fma_c2_30);
#endif
                        float2 _f2_5 = make_float2(sv[10], sv[11]);
                        psum0 = add_f32x2(psum0, _f2_5);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 14] = approx_exp2(sv[_le + 14]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_37 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_38 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_31 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_32 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_37, _fma_c2_38);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_31, _fma_c2_32);
#endif
                        float2 _f2_6 = make_float2(sv[12], sv[13]);
                        psum0 = add_f32x2(psum0, _f2_6);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 16] = approx_exp2(sv[_le + 16]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_39 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_40 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_33 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_34 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_39, _fma_c2_40);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_33, _fma_c2_34);
#endif
                        float2 _f2_7 = make_float2(sv[14], sv[15]);
                        psum0 = add_f32x2(psum0, _f2_7);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 18] = approx_exp2(sv[_le + 18]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_41 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_42 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_35 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_36 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_41, _fma_c2_42);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_35, _fma_c2_36);
#endif
                        float2 _f2_8 = make_float2(sv[16], sv[17]);
                        psum0 = add_f32x2(psum0, _f2_8);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 20] = approx_exp2(sv[_le + 20]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_43 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_44 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_37 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_38 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_43, _fma_c2_44);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_37, _fma_c2_38);
#endif
                        float2 _f2_9 = make_float2(sv[18], sv[19]);
                        psum0 = add_f32x2(psum0, _f2_9);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 22] = approx_exp2(sv[_le + 22]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_45 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_46 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_39 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_40 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 34))[_lf], _fma_b2_45, _fma_c2_46);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 34))[_lf], _fma_b2_39, _fma_c2_40);
#endif
                        float2 _f2_10 = make_float2(sv[20], sv[21]);
                        psum0 = add_f32x2(psum0, _f2_10);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 24] = approx_exp2(sv[_le + 24]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_47 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_48 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_41 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_42 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 36))[_lf], _fma_b2_47, _fma_c2_48);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 36))[_lf], _fma_b2_41, _fma_c2_42);
#endif
                        float2 _f2_11 = make_float2(sv[22], sv[23]);
                        psum0 = add_f32x2(psum0, _f2_11);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 26] = approx_exp2(sv[_le + 26]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_49 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_50 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_43 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_44 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 38))[_lf], _fma_b2_49, _fma_c2_50);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 38))[_lf], _fma_b2_43, _fma_c2_44);
#endif
                        float2 _f2_12 = make_float2(sv[24], sv[25]);
                        psum0 = add_f32x2(psum0, _f2_12);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 28] = approx_exp2(sv[_le + 28]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_51 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_52 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_45 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_46 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 40))[_lf], _fma_b2_51, _fma_c2_52);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 40))[_lf], _fma_b2_45, _fma_c2_46);
#endif
                        float2 _f2_13 = make_float2(sv[26], sv[27]);
                        psum0 = add_f32x2(psum0, _f2_13);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 30] = approx_exp2(sv[_le + 30]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_53 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_54 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_47 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_48 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 42))[_lf], _fma_b2_53, _fma_c2_54);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 42))[_lf], _fma_b2_47, _fma_c2_48);
#endif
                        float2 _f2_14 = make_float2(sv[28], sv[29]);
                        psum0 = add_f32x2(psum0, _f2_14);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_55 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 32], sv[_le*2 + 1 + 32]));
                                sv[_le*2 + 32] = _exp2_pair_55.x;
                                sv[_le*2 + 1 + 32] = _exp2_pair_55.y;
                            } else {
                                sv[_le*2 + 32] = approx_exp2(sv[_le*2 + 32]);
                                sv[_le*2 + 1 + 32] = approx_exp2(sv[_le*2 + 1 + 32]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 32] = approx_exp2(sv[_le + 32]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_56 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_57 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_49 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_50 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 44))[_lf], _fma_b2_56, _fma_c2_57);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 44))[_lf], _fma_b2_49, _fma_c2_50);
#endif
                        float2 _f2_15 = make_float2(sv[30], sv[31]);
                        psum0 = add_f32x2(psum0, _f2_15);
                        uint32_t sv_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                            sv_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_58 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 34], sv[_le*2 + 1 + 34]));
                                sv[_le*2 + 34] = _exp2_pair_58.x;
                                sv[_le*2 + 1 + 34] = _exp2_pair_58.y;
                            } else {
                                sv[_le*2 + 34] = approx_exp2(sv[_le*2 + 34]);
                                sv[_le*2 + 1 + 34] = approx_exp2(sv[_le*2 + 1 + 34]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 34] = approx_exp2(sv[_le + 34]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_59 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_60 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_51 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_52 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 46))[_lf], _fma_b2_59, _fma_c2_60);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 46))[_lf], _fma_b2_51, _fma_c2_52);
#endif
                        float2 _f2_16 = make_float2(sv[32], sv[33]);
                        float2 psum1 = _f2_16;
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_61 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 36], sv[_le*2 + 1 + 36]));
                                sv[_le*2 + 36] = _exp2_pair_61.x;
                                sv[_le*2 + 1 + 36] = _exp2_pair_61.y;
                            } else {
                                sv[_le*2 + 36] = approx_exp2(sv[_le*2 + 36]);
                                sv[_le*2 + 1 + 36] = approx_exp2(sv[_le*2 + 1 + 36]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 36] = approx_exp2(sv[_le + 36]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_62 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_63 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_53 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_54 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_62, _fma_c2_63);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_53, _fma_c2_54);
#endif
                        float2 _f2_17 = make_float2(sv[34], sv[35]);
                        psum1 = add_f32x2(psum1, _f2_17);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_64 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 38], sv[_le*2 + 1 + 38]));
                                sv[_le*2 + 38] = _exp2_pair_64.x;
                                sv[_le*2 + 1 + 38] = _exp2_pair_64.y;
                            } else {
                                sv[_le*2 + 38] = approx_exp2(sv[_le*2 + 38]);
                                sv[_le*2 + 1 + 38] = approx_exp2(sv[_le*2 + 1 + 38]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 38] = approx_exp2(sv[_le + 38]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_65 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_66 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_55 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_56 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 50))[_lf], _fma_b2_65, _fma_c2_66);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 50))[_lf], _fma_b2_55, _fma_c2_56);
#endif
                        float2 _f2_18 = make_float2(sv[36], sv[37]);
                        psum1 = add_f32x2(psum1, _f2_18);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_67 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 40], sv[_le*2 + 1 + 40]));
                                sv[_le*2 + 40] = _exp2_pair_67.x;
                                sv[_le*2 + 1 + 40] = _exp2_pair_67.y;
                            } else {
                                sv[_le*2 + 40] = approx_exp2(sv[_le*2 + 40]);
                                sv[_le*2 + 1 + 40] = approx_exp2(sv[_le*2 + 1 + 40]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 40] = approx_exp2(sv[_le + 40]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_68 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_69 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_57 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_58 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 52))[_lf], _fma_b2_68, _fma_c2_69);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 52))[_lf], _fma_b2_57, _fma_c2_58);
#endif
                        float2 _f2_19 = make_float2(sv[38], sv[39]);
                        psum1 = add_f32x2(psum1, _f2_19);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 42] = approx_exp2(sv[_le + 42]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_70 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_71 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_59 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_60 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 54))[_lf], _fma_b2_70, _fma_c2_71);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 54))[_lf], _fma_b2_59, _fma_c2_60);
#endif
                        float2 _f2_20 = make_float2(sv[40], sv[41]);
                        psum1 = add_f32x2(psum1, _f2_20);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 44] = approx_exp2(sv[_le + 44]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_72 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_73 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_61 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_62 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 56))[_lf], _fma_b2_72, _fma_c2_73);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 56))[_lf], _fma_b2_61, _fma_c2_62);
#endif
                        float2 _f2_21 = make_float2(sv[42], sv[43]);
                        psum1 = add_f32x2(psum1, _f2_21);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 46] = approx_exp2(sv[_le + 46]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_74 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_75 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_63 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_64 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 58))[_lf], _fma_b2_74, _fma_c2_75);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 58))[_lf], _fma_b2_63, _fma_c2_64);
#endif
                        float2 _f2_22 = make_float2(sv[44], sv[45]);
                        psum1 = add_f32x2(psum1, _f2_22);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 48] = approx_exp2(sv[_le + 48]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_76 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_77 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_65 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_66 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 60))[_lf], _fma_b2_76, _fma_c2_77);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 60))[_lf], _fma_b2_65, _fma_c2_66);
#endif
                        float2 _f2_23 = make_float2(sv[46], sv[47]);
                        psum1 = add_f32x2(psum1, _f2_23);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 50] = approx_exp2(sv[_le + 50]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_78 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_79 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_67 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_68 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 62))[_lf], _fma_b2_78, _fma_c2_79);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 62))[_lf], _fma_b2_67, _fma_c2_68);
#endif
                        float2 _f2_24 = make_float2(sv[48], sv[49]);
                        psum1 = add_f32x2(psum1, _f2_24);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 52] = approx_exp2(sv[_le + 52]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_80 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_81 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_69 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_70 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_80, _fma_c2_81);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_69, _fma_c2_70);
#endif
                        float2 _f2_25 = make_float2(sv[50], sv[51]);
                        psum1 = add_f32x2(psum1, _f2_25);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 54] = approx_exp2(sv[_le + 54]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_82 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_83 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_71 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_72 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 66))[_lf], _fma_b2_82, _fma_c2_83);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 66))[_lf], _fma_b2_71, _fma_c2_72);
#endif
                        float2 _f2_26 = make_float2(sv[52], sv[53]);
                        psum1 = add_f32x2(psum1, _f2_26);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 56] = approx_exp2(sv[_le + 56]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_84 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_85 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_73 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_74 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 68))[_lf], _fma_b2_84, _fma_c2_85);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 68))[_lf], _fma_b2_73, _fma_c2_74);
#endif
                        float2 _f2_27 = make_float2(sv[54], sv[55]);
                        psum1 = add_f32x2(psum1, _f2_27);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 58] = approx_exp2(sv[_le + 58]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_86 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_87 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_75 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_76 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 70))[_lf], _fma_b2_86, _fma_c2_87);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 70))[_lf], _fma_b2_75, _fma_c2_76);
#endif
                        float2 _f2_28 = make_float2(sv[56], sv[57]);
                        psum1 = add_f32x2(psum1, _f2_28);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 60] = approx_exp2(sv[_le + 60]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_88 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_89 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_77 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_78 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 72))[_lf], _fma_b2_88, _fma_c2_89);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 72))[_lf], _fma_b2_77, _fma_c2_78);
#endif
                        float2 _f2_29 = make_float2(sv[58], sv[59]);
                        psum1 = add_f32x2(psum1, _f2_29);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 62] = approx_exp2(sv[_le + 62]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_90 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_91 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_79 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_80 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 74))[_lf], _fma_b2_90, _fma_c2_91);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 74))[_lf], _fma_b2_79, _fma_c2_80);
#endif
                        float2 _f2_30 = make_float2(sv[60], sv[61]);
                        psum1 = add_f32x2(psum1, _f2_30);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_92 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 64], sv[_le*2 + 1 + 64]));
                                sv[_le*2 + 64] = _exp2_pair_92.x;
                                sv[_le*2 + 1 + 64] = _exp2_pair_92.y;
                            } else {
                                sv[_le*2 + 64] = approx_exp2(sv[_le*2 + 64]);
                                sv[_le*2 + 1 + 64] = approx_exp2(sv[_le*2 + 1 + 64]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 64] = approx_exp2(sv[_le + 64]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_93 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_94 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_81 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_82 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 76))[_lf], _fma_b2_93, _fma_c2_94);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 76))[_lf], _fma_b2_81, _fma_c2_82);
#endif
                        float2 _f2_31 = make_float2(sv[62], sv[63]);
                        psum1 = add_f32x2(psum1, _f2_31);
                        uint32_t sv_bf16_0[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 32], sv[_lp*2+1 + 32]));
                            sv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_95 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 66], sv[_le*2 + 1 + 66]));
                                sv[_le*2 + 66] = _exp2_pair_95.x;
                                sv[_le*2 + 1 + 66] = _exp2_pair_95.y;
                            } else {
                                sv[_le*2 + 66] = approx_exp2(sv[_le*2 + 66]);
                                sv[_le*2 + 1 + 66] = approx_exp2(sv[_le*2 + 1 + 66]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 66] = approx_exp2(sv[_le + 66]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_96 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_97 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_83 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_84 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 78))[_lf], _fma_b2_96, _fma_c2_97);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 78))[_lf], _fma_b2_83, _fma_c2_84);
#endif
                        float2 _f2_32 = make_float2(sv[64], sv[65]);
                        float2 psum2 = _f2_32;
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_98 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 68], sv[_le*2 + 1 + 68]));
                                sv[_le*2 + 68] = _exp2_pair_98.x;
                                sv[_le*2 + 1 + 68] = _exp2_pair_98.y;
                            } else {
                                sv[_le*2 + 68] = approx_exp2(sv[_le*2 + 68]);
                                sv[_le*2 + 1 + 68] = approx_exp2(sv[_le*2 + 1 + 68]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 68] = approx_exp2(sv[_le + 68]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_99 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_100 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_85 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_86 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_99, _fma_c2_100);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_85, _fma_c2_86);
#endif
                        float2 _f2_33 = make_float2(sv[66], sv[67]);
                        psum2 = add_f32x2(psum2, _f2_33);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_101 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 70], sv[_le*2 + 1 + 70]));
                                sv[_le*2 + 70] = _exp2_pair_101.x;
                                sv[_le*2 + 1 + 70] = _exp2_pair_101.y;
                            } else {
                                sv[_le*2 + 70] = approx_exp2(sv[_le*2 + 70]);
                                sv[_le*2 + 1 + 70] = approx_exp2(sv[_le*2 + 1 + 70]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 70] = approx_exp2(sv[_le + 70]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_102 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_103 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_87 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_88 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 82))[_lf], _fma_b2_102, _fma_c2_103);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 82))[_lf], _fma_b2_87, _fma_c2_88);
#endif
                        float2 _f2_34 = make_float2(sv[68], sv[69]);
                        psum2 = add_f32x2(psum2, _f2_34);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_104 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 72], sv[_le*2 + 1 + 72]));
                                sv[_le*2 + 72] = _exp2_pair_104.x;
                                sv[_le*2 + 1 + 72] = _exp2_pair_104.y;
                            } else {
                                sv[_le*2 + 72] = approx_exp2(sv[_le*2 + 72]);
                                sv[_le*2 + 1 + 72] = approx_exp2(sv[_le*2 + 1 + 72]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 72] = approx_exp2(sv[_le + 72]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_105 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_106 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_89 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_90 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 84))[_lf], _fma_b2_105, _fma_c2_106);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 84))[_lf], _fma_b2_89, _fma_c2_90);
#endif
                        float2 _f2_35 = make_float2(sv[70], sv[71]);
                        psum2 = add_f32x2(psum2, _f2_35);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 74] = approx_exp2(sv[_le + 74]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_107 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_108 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_91 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_92 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 86))[_lf], _fma_b2_107, _fma_c2_108);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 86))[_lf], _fma_b2_91, _fma_c2_92);
#endif
                        float2 _f2_36 = make_float2(sv[72], sv[73]);
                        psum2 = add_f32x2(psum2, _f2_36);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 76] = approx_exp2(sv[_le + 76]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_109 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_110 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_93 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_94 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 88))[_lf], _fma_b2_109, _fma_c2_110);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 88))[_lf], _fma_b2_93, _fma_c2_94);
#endif
                        float2 _f2_37 = make_float2(sv[74], sv[75]);
                        psum2 = add_f32x2(psum2, _f2_37);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 78] = approx_exp2(sv[_le + 78]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_111 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_112 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_95 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_96 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 90))[_lf], _fma_b2_111, _fma_c2_112);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 90))[_lf], _fma_b2_95, _fma_c2_96);
#endif
                        float2 _f2_38 = make_float2(sv[76], sv[77]);
                        psum2 = add_f32x2(psum2, _f2_38);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 80] = approx_exp2(sv[_le + 80]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_113 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_114 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_97 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_98 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 92))[_lf], _fma_b2_113, _fma_c2_114);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 92))[_lf], _fma_b2_97, _fma_c2_98);
#endif
                        float2 _f2_39 = make_float2(sv[78], sv[79]);
                        psum2 = add_f32x2(psum2, _f2_39);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 82] = approx_exp2(sv[_le + 82]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_115 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_116 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_99 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_100 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 94))[_lf], _fma_b2_115, _fma_c2_116);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 94))[_lf], _fma_b2_99, _fma_c2_100);
#endif
                        float2 _f2_40 = make_float2(sv[80], sv[81]);
                        psum2 = add_f32x2(psum2, _f2_40);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 84] = approx_exp2(sv[_le + 84]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_117 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_118 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_101 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_102 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_117, _fma_c2_118);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_101, _fma_c2_102);
#endif
                        float2 _f2_41 = make_float2(sv[82], sv[83]);
                        psum2 = add_f32x2(psum2, _f2_41);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 86] = approx_exp2(sv[_le + 86]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_119 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_120 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_103 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_104 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 98))[_lf], _fma_b2_119, _fma_c2_120);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 98))[_lf], _fma_b2_103, _fma_c2_104);
#endif
                        float2 _f2_42 = make_float2(sv[84], sv[85]);
                        psum2 = add_f32x2(psum2, _f2_42);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 88] = approx_exp2(sv[_le + 88]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_121 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_122 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_105 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_106 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 100))[_lf], _fma_b2_121, _fma_c2_122);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 100))[_lf], _fma_b2_105, _fma_c2_106);
#endif
                        float2 _f2_43 = make_float2(sv[86], sv[87]);
                        psum2 = add_f32x2(psum2, _f2_43);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 90] = approx_exp2(sv[_le + 90]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_123 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_124 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_107 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_108 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 102))[_lf], _fma_b2_123, _fma_c2_124);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 102))[_lf], _fma_b2_107, _fma_c2_108);
#endif
                        float2 _f2_44 = make_float2(sv[88], sv[89]);
                        psum2 = add_f32x2(psum2, _f2_44);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 92] = approx_exp2(sv[_le + 92]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_125 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_126 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_109 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_110 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 104))[_lf], _fma_b2_125, _fma_c2_126);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 104))[_lf], _fma_b2_109, _fma_c2_110);
#endif
                        float2 _f2_45 = make_float2(sv[90], sv[91]);
                        psum2 = add_f32x2(psum2, _f2_45);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 94] = approx_exp2(sv[_le + 94]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_127 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_128 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_111 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_112 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 106))[_lf], _fma_b2_127, _fma_c2_128);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 106))[_lf], _fma_b2_111, _fma_c2_112);
#endif
                        float2 _f2_46 = make_float2(sv[92], sv[93]);
                        psum2 = add_f32x2(psum2, _f2_46);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_129 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 96], sv[_le*2 + 1 + 96]));
                                sv[_le*2 + 96] = _exp2_pair_129.x;
                                sv[_le*2 + 1 + 96] = _exp2_pair_129.y;
                            } else {
                                sv[_le*2 + 96] = approx_exp2(sv[_le*2 + 96]);
                                sv[_le*2 + 1 + 96] = approx_exp2(sv[_le*2 + 1 + 96]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 96] = approx_exp2(sv[_le + 96]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_130 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_131 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_113 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_114 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 108))[_lf], _fma_b2_130, _fma_c2_131);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 108))[_lf], _fma_b2_113, _fma_c2_114);
#endif
                        float2 _f2_47 = make_float2(sv[94], sv[95]);
                        psum2 = add_f32x2(psum2, _f2_47);
                        uint32_t sv_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 64], sv[_lp*2+1 + 64]));
                            sv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_132 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 98], sv[_le*2 + 1 + 98]));
                                sv[_le*2 + 98] = _exp2_pair_132.x;
                                sv[_le*2 + 1 + 98] = _exp2_pair_132.y;
                            } else {
                                sv[_le*2 + 98] = approx_exp2(sv[_le*2 + 98]);
                                sv[_le*2 + 1 + 98] = approx_exp2(sv[_le*2 + 1 + 98]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 98] = approx_exp2(sv[_le + 98]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_133 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_134 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_115 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_116 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 110))[_lf], _fma_b2_133, _fma_c2_134);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 110))[_lf], _fma_b2_115, _fma_c2_116);
#endif
                        float2 _f2_48 = make_float2(sv[96], sv[97]);
                        float2 psum3 = _f2_48;
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_135 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 100], sv[_le*2 + 1 + 100]));
                                sv[_le*2 + 100] = _exp2_pair_135.x;
                                sv[_le*2 + 1 + 100] = _exp2_pair_135.y;
                            } else {
                                sv[_le*2 + 100] = approx_exp2(sv[_le*2 + 100]);
                                sv[_le*2 + 1 + 100] = approx_exp2(sv[_le*2 + 1 + 100]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 100] = approx_exp2(sv[_le + 100]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_136 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_137 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_117 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_118 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_136, _fma_c2_137);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_117, _fma_c2_118);
#endif
                        float2 _f2_49 = make_float2(sv[98], sv[99]);
                        psum3 = add_f32x2(psum3, _f2_49);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_138 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 102], sv[_le*2 + 1 + 102]));
                                sv[_le*2 + 102] = _exp2_pair_138.x;
                                sv[_le*2 + 1 + 102] = _exp2_pair_138.y;
                            } else {
                                sv[_le*2 + 102] = approx_exp2(sv[_le*2 + 102]);
                                sv[_le*2 + 1 + 102] = approx_exp2(sv[_le*2 + 1 + 102]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 102] = approx_exp2(sv[_le + 102]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_139 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_140 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_119 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_120 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 114))[_lf], _fma_b2_139, _fma_c2_140);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 114))[_lf], _fma_b2_119, _fma_c2_120);
#endif
                        float2 _f2_50 = make_float2(sv[100], sv[101]);
                        psum3 = add_f32x2(psum3, _f2_50);
                        #pragma unroll
#if __CUDA_ARCH__ == 1000
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_141 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 104], sv[_le*2 + 1 + 104]));
                                sv[_le*2 + 104] = _exp2_pair_141.x;
                                sv[_le*2 + 1 + 104] = _exp2_pair_141.y;
                            } else {
                                sv[_le*2 + 104] = approx_exp2(sv[_le*2 + 104]);
                                sv[_le*2 + 1 + 104] = approx_exp2(sv[_le*2 + 1 + 104]);
                            }
#else
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 104] = approx_exp2(sv[_le + 104]);
#endif
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_142 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_143 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_121 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_122 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 116))[_lf], _fma_b2_142, _fma_c2_143);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 116))[_lf], _fma_b2_121, _fma_c2_122);
#endif
                        float2 _f2_51 = make_float2(sv[102], sv[103]);
                        psum3 = add_f32x2(psum3, _f2_51);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 106] = approx_exp2(sv[_le + 106]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_144 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_145 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_123 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_124 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 118))[_lf], _fma_b2_144, _fma_c2_145);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 118))[_lf], _fma_b2_123, _fma_c2_124);
#endif
                        float2 _f2_52 = make_float2(sv[104], sv[105]);
                        psum3 = add_f32x2(psum3, _f2_52);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 108] = approx_exp2(sv[_le + 108]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_146 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_147 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_125 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_126 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 120))[_lf], _fma_b2_146, _fma_c2_147);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 120))[_lf], _fma_b2_125, _fma_c2_126);
#endif
                        float2 _f2_53 = make_float2(sv[106], sv[107]);
                        psum3 = add_f32x2(psum3, _f2_53);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 110] = approx_exp2(sv[_le + 110]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_148 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_149 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_127 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_128 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 122))[_lf], _fma_b2_148, _fma_c2_149);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 122))[_lf], _fma_b2_127, _fma_c2_128);
#endif
                        float2 _f2_54 = make_float2(sv[108], sv[109]);
                        psum3 = add_f32x2(psum3, _f2_54);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 112] = approx_exp2(sv[_le + 112]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_150 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_151 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_129 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_130 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 124))[_lf], _fma_b2_150, _fma_c2_151);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 124))[_lf], _fma_b2_129, _fma_c2_130);
#endif
                        float2 _f2_55 = make_float2(sv[110], sv[111]);
                        psum3 = add_f32x2(psum3, _f2_55);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 114] = approx_exp2(sv[_le + 114]);
                        }
#if __CUDA_ARCH__ == 1000
                        const float2 _fma_b2_152 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_153 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_131 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_132 = {neg_max_scaled, neg_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1000
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 126))[_lf], _fma_b2_152, _fma_c2_153);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 126))[_lf], _fma_b2_131, _fma_c2_132);
#endif
                        float2 _f2_56 = make_float2(sv[112], sv[113]);
                        psum3 = add_f32x2(psum3, _f2_56);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 116] = approx_exp2(sv[_le + 116]);
                        }
                        float2 _f2_57 = make_float2(sv[114], sv[115]);
                        psum3 = add_f32x2(psum3, _f2_57);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 118] = approx_exp2(sv[_le + 118]);
                        }
                        float2 _f2_58 = make_float2(sv[116], sv[117]);
                        psum3 = add_f32x2(psum3, _f2_58);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 120] = approx_exp2(sv[_le + 120]);
                        }
                        float2 _f2_59 = make_float2(sv[118], sv[119]);
                        psum3 = add_f32x2(psum3, _f2_59);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 122] = approx_exp2(sv[_le + 122]);
                        }
                        float2 _f2_60 = make_float2(sv[120], sv[121]);
                        psum3 = add_f32x2(psum3, _f2_60);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 124] = approx_exp2(sv[_le + 124]);
                        }
                        float2 _f2_61 = make_float2(sv[122], sv[123]);
                        psum3 = add_f32x2(psum3, _f2_61);
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 126] = approx_exp2(sv[_le + 126]);
                        }
                        float2 _f2_62 = make_float2(sv[124], sv[125]);
                        psum3 = add_f32x2(psum3, _f2_62);
                        float2 _f2_63 = make_float2(sv[126], sv[127]);
                        psum3 = add_f32x2(psum3, _f2_63);
                        uint32_t sv_bf16_2[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 96], sv[_lp*2+1 + 96]));
                            sv_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        int p_addr = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                        if (two_stages == 1) {
                            mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                            _phase_s_read ^= 1;
                        }
                        if (stage == 0 && two_stages == 0 && n_iter >= 1) {
                            mbarrier_wait(pv0_done_addr, _phase_pv0_done_0);
                            _phase_pv0_done_0 ^= 1;
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
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_addr + 48), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        float2 sum01 = add_f32x2(psum0, psum1);
                        float2 sum23 = add_f32x2(psum2, psum3);
                        float2 total_sum = add_f32x2(sum01, sum23);
                        row_sum = row_sum * acc_scale + total_sum.x + total_sum.y;
                    }
                    if (stage == 0 && two_stages == 1) {
                        mbarrier_arrive(s_read_addr + (stage) * 8);
                    }
                    scales[warp % 4 * 32 + lane + scale_off] = row_max;
                    scales[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum;
                    if (sync_group == 0) {
                        asm volatile("barrier.sync 1, 256;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 2, 256;" ::: "memory");
                    }
                }
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
            int rec_1 = cluster_id * 8;
            int doc_begin_1 = unit_table[rec_1];
            int doc_len_2 = unit_table[rec_1 + 1];
            int packed_1 = unit_table[rec_1 + 2];
            int kv_words_1 = unit_table[rec_1 + 3];
            int ws_slot_1 = unit_table[rec_1 + 4];
            nxt_doc_begin_1 = doc_begin_1;
            nxt_doc_len_1 = doc_len_2;
            nxt_packed_1 = packed_1;
            nxt_kv_words_1 = kv_words_1;
            nxt_ws_slot_1 = ws_slot_1;
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
                int unit_row0_1 = c_1 * 512;
                int two_stages_1 = ((doc_len_1_1 - unit_row0_1 > 256) ? 1 : 0);
                unsigned int nxt_tile_1 = tile_idx_1 + num_clusters;
                int rec_3_1 = nxt_tile_1 * 8;
                int doc_begin_4_1 = unit_table[rec_3_1];
                int doc_len_5_1 = unit_table[rec_3_1 + 1];
                int packed_6_1 = unit_table[rec_3_1 + 2];
                int kv_words_7_1 = unit_table[rec_3_1 + 3];
                int ws_slot_8_1 = unit_table[rec_3_1 + 4];
                nxt_doc_begin_1 = doc_begin_4_1;
                nxt_doc_len_1 = doc_len_5_1;
                nxt_packed_1 = packed_6_1;
                nxt_kv_words_1 = kv_words_7_1;
                nxt_ws_slot_1 = ws_slot_8_1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                if (two_stages_1 == 1) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                }
#if __CUDA_ARCH__ == 1000
                mbarrier_wait_hint(scale_full_addr, _phase_scale_full_0, 1000000);
#else
                mbarrier_wait(scale_full_addr, _phase_scale_full_0);
#endif
                _phase_scale_full_0 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((scale_empty_addr) & 0xFEFFFFFF) : "memory");
                if (two_stages_1 == 1) {
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(scale_full_addr + 8, _phase_scale_full_1, 1000000);
#else
                    mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
#endif
                    _phase_scale_full_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_empty_addr + 8) & 0xFEFFFFFF) : "memory");
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(scale_full_addr, _phase_scale_full_0, 1000000);
#else
                    mbarrier_wait(scale_full_addr, _phase_scale_full_0);
#endif
                    _phase_scale_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[1];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(_tmem_load_0[0])
                        : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16)));
                    float scale0 = _tmem_load_0[0];
                    int _vote_0 = __all_sync(0xFFFFFFFF, scale0 == 1.0f);
                    int skip_rescale0 = _vote_0;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_empty_addr) & 0xFEFFFFFF) : "memory");
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 1000000);
#else
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
#endif
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (two_stages_1 == 0) {
                        mbarrier_arrive(pv0_done_addr);
                    }
                    if (skip_rescale0 == 0) {
                        #pragma unroll
                        for (int col = 0; col < HEAD_DIM / 16; col++) {
                            int addr0 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 16);
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], addr0);
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {scale0, scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
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
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    if (two_stages_1 == 1) {
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(scale_full_addr + 8, _phase_scale_full_1, 1000000);
#else
                        mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
#endif
                        _phase_scale_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_2[1];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                            " {%0}, [%1];"
                            : "=f"(_tmem_load_2[0])
                            : "r"(taddr + (unsigned int)BLOCK_M + (unsigned int)(warp % 4 * 32 << 16)));
                        float scale1 = _tmem_load_2[0];
                        int _vote_1 = __all_sync(0xFFFFFFFF, scale1 == 1.0f);
                        int skip_rescale1 = _vote_1;
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((scale_empty_addr + 8) & 0xFEFFFFFF) : "memory");
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 1000000);
#else
                        mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
#endif
                        _phase_o_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (skip_rescale1 == 0) {
                            #pragma unroll
                            for (int col_1 = 0; col_1 < HEAD_DIM / 16; col_1++) {
                                int addr1 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_1 * 16);
                                float _tmem_load_3[16];
                                tmem_ld_x16(&_tmem_load_3[0], addr1);
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_1 = {scale1, scale1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
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
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    }
                }
#if __CUDA_ARCH__ == 1000
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 1000000);
#else
                mbarrier_wait(o_full_addr, _phase_o_full_0);
#endif
                _phase_o_full_0 ^= 1;
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                float final_sum = scales[warp % 4 * 32 + lane + 2 * BLOCK_M];
                float final_scale;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_0 = approx_rcp(final_sum);
                    final_scale = _rcp_0;
                } else {
                    final_scale = 0.0f;
                }
                int local_row = unit_row0_1 + cta_rank * BLOCK_M + (warp % 4 * 32 + lane);
                int out_row = (doc_begin_0_1 + local_row) * num_heads + head_1;
                if (ws_slot_2_1 >= 0) {
                    int partial_row = ws_slot_2_1 * 512 + cta_rank * BLOCK_M + (warp % 4 * 32 + lane);
                    if (local_row < doc_len_1_1) {
                        partial_ML[partial_row * 2] = scales[warp % 4 * 32 + lane] * softmax_scale_log2;
                        partial_ML[partial_row * 2 + 1] = final_sum;
                    }
                    #pragma unroll
                    for (int col_2 = 0; col_2 < HEAD_DIM / 16; col_2++) {
                        int addr = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 16);
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], addr);
                        if (local_row < doc_len_1_1) {
                            {
                                const float2 _prescale2_2 = {final_scale, final_scale};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_2);
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
                        #pragma unroll
                        for (int col_3 = 0; col_3 < HEAD_DIM / 16; col_3++) {
                            int addr_1 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_3 * 16);
                            float _tmem_load_5[16];
                            tmem_ld_x16(&_tmem_load_5[0], addr_1);
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_3 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_3);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_5[_ls] = _tmem_load_5[_ls] * final_scale;
                            }
                            #endif
                            uint32_t _tmem_load_5_bf16[8];
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
                            int addr_2 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_4 * 16);
                            float _tmem_load_6[16];
                            tmem_ld_x16(&_tmem_load_6[0], addr_2);
                            if (local_row < doc_len_1_1) {
                                {
                                    const float2 _prescale2_4 = {final_scale, final_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_6[0])[_ps], _prescale2_4);
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
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 1000000);
#else
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
#endif
                    _phase_o_full_1 ^= 1;
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    float final_sum_0 = scales[warp % 4 * 32 + lane + BLOCK_M + 2 * BLOCK_M];
                    float final_scale_1;
                    if (final_sum_0 != 0.0f && final_sum_0 == final_sum_0) {
                        float _rcp_1 = approx_rcp(final_sum_0);
                        final_scale_1 = _rcp_1;
                    } else {
                        final_scale_1 = 0.0f;
                    }
                    int local_row_2 = unit_row0_1 + (2 + cta_rank) * BLOCK_M + (warp % 4 * 32 + lane);
                    int out_row_3 = (doc_begin_0_1 + local_row_2) * num_heads + head_1;
                    if (ws_slot_2_1 >= 0) {
                        int partial_row_1 = ws_slot_2_1 * 512 + (2 + cta_rank) * BLOCK_M + (warp % 4 * 32 + lane);
                        if (local_row_2 < doc_len_1_1) {
                            partial_ML[partial_row_1 * 2] = scales[warp % 4 * 32 + lane + BLOCK_M] * softmax_scale_log2;
                            partial_ML[partial_row_1 * 2 + 1] = final_sum_0;
                        }
                        #pragma unroll
                        for (int col_5 = 0; col_5 < HEAD_DIM / 16; col_5++) {
                            int addr_3 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_5 * 16);
                            float _tmem_load_7[16];
                            tmem_ld_x16(&_tmem_load_7[0], addr_3);
                            if (local_row_2 < doc_len_1_1) {
                                {
                                    const float2 _prescale2_5 = {final_scale_1, final_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[0])[_ps], _prescale2_5);
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
                            #pragma unroll
                            for (int col_6 = 0; col_6 < HEAD_DIM / 16; col_6++) {
                                int addr_4 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_6 * 16);
                                float _tmem_load_8[16];
                                tmem_ld_x16(&_tmem_load_8[0], addr_4);
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_6 = {final_scale_1, final_scale_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_6);
                                #else
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++) {
                                    _tmem_load_8[_ls] = _tmem_load_8[_ls] * final_scale_1;
                                }
                                #endif
                                uint32_t _tmem_load_8_bf16[8];
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
                                int addr_5 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_7 * 16);
                                float _tmem_load_9[16];
                                tmem_ld_x16(&_tmem_load_9[0], addr_5);
                                if (local_row_2 < doc_len_1_1) {
                                    {
                                        const float2 _prescale2_7 = {final_scale_1, final_scale_1};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[0])[_ps], _prescale2_7);
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
                int rec_2 = cluster_id * 8;
                int doc_begin_2 = unit_table[rec_2];
                int doc_len_3 = unit_table[rec_2 + 1];
                int packed_2 = unit_table[rec_2 + 2];
                int kv_words_2 = unit_table[rec_2 + 3];
                int ws_slot_3 = unit_table[rec_2 + 4];
                nxt_doc_begin_2 = doc_begin_2;
                nxt_doc_len_2 = doc_len_3;
                nxt_packed_2 = packed_2;
                nxt_kv_words_2 = kv_words_2;
                nxt_ws_slot_2 = ws_slot_3;
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles; tile_idx_2 += num_clusters) {
                    int doc_begin_0_2 = nxt_doc_begin_2;
                    int doc_len_1_2 = nxt_doc_len_2;
                    int ws_slot_2_2 = nxt_ws_slot_2;
                    int head_2 = nxt_packed_2 >> 16;
                    int c_2 = nxt_packed_2 & 65535;
                    int n_begin_2 = nxt_kv_words_2 >> 16;
                    int num_n_blocks_2 = nxt_kv_words_2 & 65535;
                    int unit_row0_2 = c_2 * 512;
                    int two_stages_2 = ((doc_len_1_2 - unit_row0_2 > 256) ? 1 : 0);
                    unsigned int nxt_tile_2 = tile_idx_2 + num_clusters;
                    int rec_3_2 = nxt_tile_2 * 8;
                    int doc_begin_4_2 = unit_table[rec_3_2];
                    int doc_len_5_2 = unit_table[rec_3_2 + 1];
                    int packed_6_2 = unit_table[rec_3_2 + 2];
                    int kv_words_7_2 = unit_table[rec_3_2 + 3];
                    int ws_slot_8_2 = unit_table[rec_3_2 + 4];
                    nxt_doc_begin_2 = doc_begin_4_2;
                    nxt_doc_len_2 = doc_len_5_2;
                    nxt_packed_2 = packed_6_2;
                    nxt_kv_words_2 = kv_words_7_2;
                    nxt_ws_slot_2 = ws_slot_8_2;
                    mbarrier_wait_cluster_hint(q_full_addr, _phase_q_full_0, 10000000);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(scale_empty_addr, _phase_scale_empty_0, 1000000);
#else
                    mbarrier_wait(scale_empty_addr, _phase_scale_empty_0);
#endif
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(scale_empty_addr + 8, _phase_scale_empty_1, 1000000);
#else
                        mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
#endif
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (128))), "r"(0));
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 5) { k_stage = 0; k_phase ^= 1; }
                    k_stage += 1;
                    if (k_stage == 5) { k_stage = 0; k_phase ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 0; n_iter_2 < num_n_blocks_2 - 1; n_iter_2++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(scale_empty_addr, _phase_scale_empty_0, 1000000);
#else
                        mbarrier_wait(scale_empty_addr, _phase_scale_empty_0);
#endif
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_scores), "r"(0));
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 1000000);
#else
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
#endif
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
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)));
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(p_full_2_addr, _phase_p_full_2_0, 1000000);
#else
                        mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
#endif
                        _phase_p_full_2_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_4), "r"(tmem_probs_0), "r"(1));
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1000
                            mbarrier_wait_hint(scale_empty_addr + 8, _phase_scale_empty_1, 1000000);
#else
                            mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
#endif
                            _phase_scale_empty_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_5 = ((smem_q1_addr) >> 4) & 0x3FFF;
                            int _mma_b_lo_5 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_scores + (128))), "r"(0));
                            elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 5) { k_stage = 0; k_phase ^= 1; }
                        k_stage += 1;
                        if (k_stage == 5) { k_stage = 0; k_phase ^= 1; }
                        if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1000
                            mbarrier_wait_hint(p_full_addr + 8, _phase_p_full_1, 1000000);
#else
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
#endif
                            _phase_p_full_1 ^= 1;
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
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_6), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)));
#if __CUDA_ARCH__ == 1000
                            mbarrier_wait_hint(p_full_2_addr + 8, _phase_p_full_2_1, 1000000);
#else
                            mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
#endif
                            _phase_p_full_2_1 ^= 1;
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_7), "r"(tmem_probs_1), "r"(1));
                            elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                        first_pv = 0;
                    }
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    int first_pv_flag_1 = first_pv;
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 1000000);
#else
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
#endif
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_8 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
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
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)));
#if __CUDA_ARCH__ == 1000
                    mbarrier_wait_hint(p_full_2_addr, _phase_p_full_2_0, 1000000);
#else
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
#endif
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_9 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_9), "r"(tmem_probs_0), "r"(1));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(p_full_addr + 8, _phase_p_full_1, 1000000);
#else
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
#endif
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
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
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)));
#if __CUDA_ARCH__ == 1000
                        mbarrier_wait_hint(p_full_2_addr + 8, _phase_p_full_2_1, 1000000);
#else
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
#endif
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_11), "r"(tmem_probs_1), "r"(1));
                        elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                    v_stage += 1;
                    if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                    v_stage += 1;
                    if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
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
            int rec_4 = cluster_id * 8;
            int doc_begin_3 = unit_table[rec_4];
            int doc_len_4 = unit_table[rec_4 + 1];
            int packed_3 = unit_table[rec_4 + 2];
            int kv_words_3 = unit_table[rec_4 + 3];
            int ws_slot_4 = unit_table[rec_4 + 4];
            nxt_doc_begin_3 = doc_begin_3;
            nxt_doc_len_3 = doc_len_4;
            nxt_packed_3 = packed_3;
            nxt_kv_words_3 = kv_words_3;
            nxt_ws_slot_3 = ws_slot_4;
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
                int unit_row0_3 = c_3 * 512;
                int two_stages_3 = ((doc_len_1_3 - unit_row0_3 > 256) ? 1 : 0);
                unsigned int nxt_tile_3 = tile_idx_3 + num_clusters;
                int rec_3_3 = nxt_tile_3 * 8;
                int doc_begin_4_3 = unit_table[rec_3_3];
                int doc_len_5_3 = unit_table[rec_3_3 + 1];
                int packed_6_3 = unit_table[rec_3_3 + 2];
                int kv_words_7_3 = unit_table[rec_3_3 + 3];
                int ws_slot_8_3 = unit_table[rec_3_3 + 4];
                nxt_doc_begin_3 = doc_begin_4_3;
                nxt_doc_len_3 = doc_len_5_3;
                nxt_packed_3 = packed_6_3;
                nxt_kv_words_3 = kv_words_7_3;
                nxt_ws_slot_3 = ws_slot_8_3;
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
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
