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
kernel_cake_minimax_h3_varlen_fp8_attention_7fad085e4e9ddbed0a98(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, __nv_bfloat16* __restrict__ O_raw, int* __restrict__ unit_table, __half* __restrict__ partial_O, float* __restrict__ partial_ML, float* __restrict__ q_descale, float* __restrict__ k_descale, float* __restrict__ v_descale, unsigned int total_tiles, int num_heads, float softmax_scale_log2)
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
#if __CUDA_ARCH__ == 1030
    #define p_full_2_addr (mbar_base + 176)
    #define scale_full_addr (mbar_base + 192)
    #define scale_empty_addr (mbar_base + 208)
    #define s_read_addr (mbar_base + 224)
    #define o_full_addr (mbar_base + 240)
    #define pv0_done_addr (mbar_base + 256)
    #define pv1_done_addr (mbar_base + 264)
#else
    #define scale_full_addr (mbar_base + 176)
    #define scale_empty_addr (mbar_base + 192)
    #define s_read_addr (mbar_base + 208)
    #define o_full_addr (mbar_base + 224)
    #define pv0_done_addr (mbar_base + 240)
    #define pv1_done_addr (mbar_base + 248)
#endif

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

#if __CUDA_ARCH__ == 1030
    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)
#else
    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)
#endif

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
#if __CUDA_ARCH__ == 1030
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
#else
#endif
            // scale_full: 2 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
#else
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
#endif
            // scale_empty: 2 barriers, init_count=256
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 208, 256);
            mbarrier_init(smem + 216, 256);
#else
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
#endif
            // s_read: 2 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
#else
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
#endif
            // o_full: 2 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
#else
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
#endif
            // pv0_done: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 256, 128);
#else
            mbarrier_init(smem + 240, 128);
#endif
            // pv1_done: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
            mbarrier_init(smem + 264, 128);
#else
            mbarrier_init(smem + 248, 128);
#endif
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
#if __CUDA_ARCH__ == 1030
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 272);
#else
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
#endif
    if (warp == 0) {
#if __CUDA_ARCH__ == 1030
        int _tmem_hold = smem + 272;
#else
        int _tmem_hold = smem + 256;
#endif
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
#if __CUDA_ARCH__ == 1030
#else
            float neg_coef = -exp_coef;
#endif
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
#if __CUDA_ARCH__ == 1030
#else
                    float nms_cache = 0.0f;
#endif
                    if (stage == 1) {
                        mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                        _phase_s_read ^= 1;
                    }
#if __CUDA_ARCH__ == 1030
#else
                    int s_addr_u = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    int p_addr_u = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    unsigned int n_iter_a = 0;
                    int n_block_a = (unsigned int)(num_n_blocks - 1) - n_iter_a;
                    mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 1000000);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float sv_a[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_a[0]), "=f"(sv_a[1]), "=f"(sv_a[2]), "=f"(sv_a[3]), "=f"(sv_a[4]), "=f"(sv_a[5]), "=f"(sv_a[6]), "=f"(sv_a[7]), "=f"(sv_a[8]), "=f"(sv_a[9]), "=f"(sv_a[10]), "=f"(sv_a[11]), "=f"(sv_a[12]), "=f"(sv_a[13]), "=f"(sv_a[14]), "=f"(sv_a[15]), "=f"(sv_a[16]), "=f"(sv_a[17]), "=f"(sv_a[18]), "=f"(sv_a[19]), "=f"(sv_a[20]), "=f"(sv_a[21]), "=f"(sv_a[22]), "=f"(sv_a[23]), "=f"(sv_a[24]), "=f"(sv_a[25]), "=f"(sv_a[26]), "=f"(sv_a[27]), "=f"(sv_a[28]), "=f"(sv_a[29]), "=f"(sv_a[30]), "=f"(sv_a[31])
                        : "r"(s_addr_u));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_a[32]), "=f"(sv_a[33]), "=f"(sv_a[34]), "=f"(sv_a[35]), "=f"(sv_a[36]), "=f"(sv_a[37]), "=f"(sv_a[38]), "=f"(sv_a[39]), "=f"(sv_a[40]), "=f"(sv_a[41]), "=f"(sv_a[42]), "=f"(sv_a[43]), "=f"(sv_a[44]), "=f"(sv_a[45]), "=f"(sv_a[46]), "=f"(sv_a[47]), "=f"(sv_a[48]), "=f"(sv_a[49]), "=f"(sv_a[50]), "=f"(sv_a[51]), "=f"(sv_a[52]), "=f"(sv_a[53]), "=f"(sv_a[54]), "=f"(sv_a[55]), "=f"(sv_a[56]), "=f"(sv_a[57]), "=f"(sv_a[58]), "=f"(sv_a[59]), "=f"(sv_a[60]), "=f"(sv_a[61]), "=f"(sv_a[62]), "=f"(sv_a[63])
                        : "r"(s_addr_u + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_a[64]), "=f"(sv_a[65]), "=f"(sv_a[66]), "=f"(sv_a[67]), "=f"(sv_a[68]), "=f"(sv_a[69]), "=f"(sv_a[70]), "=f"(sv_a[71]), "=f"(sv_a[72]), "=f"(sv_a[73]), "=f"(sv_a[74]), "=f"(sv_a[75]), "=f"(sv_a[76]), "=f"(sv_a[77]), "=f"(sv_a[78]), "=f"(sv_a[79]), "=f"(sv_a[80]), "=f"(sv_a[81]), "=f"(sv_a[82]), "=f"(sv_a[83]), "=f"(sv_a[84]), "=f"(sv_a[85]), "=f"(sv_a[86]), "=f"(sv_a[87]), "=f"(sv_a[88]), "=f"(sv_a[89]), "=f"(sv_a[90]), "=f"(sv_a[91]), "=f"(sv_a[92]), "=f"(sv_a[93]), "=f"(sv_a[94]), "=f"(sv_a[95])
                        : "r"(s_addr_u + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_a[96]), "=f"(sv_a[97]), "=f"(sv_a[98]), "=f"(sv_a[99]), "=f"(sv_a[100]), "=f"(sv_a[101]), "=f"(sv_a[102]), "=f"(sv_a[103]), "=f"(sv_a[104]), "=f"(sv_a[105]), "=f"(sv_a[106]), "=f"(sv_a[107]), "=f"(sv_a[108]), "=f"(sv_a[109]), "=f"(sv_a[110]), "=f"(sv_a[111]), "=f"(sv_a[112]), "=f"(sv_a[113]), "=f"(sv_a[114]), "=f"(sv_a[115]), "=f"(sv_a[116]), "=f"(sv_a[117]), "=f"(sv_a[118]), "=f"(sv_a[119]), "=f"(sv_a[120]), "=f"(sv_a[121]), "=f"(sv_a[122]), "=f"(sv_a[123]), "=f"(sv_a[124]), "=f"(sv_a[125]), "=f"(sv_a[126]), "=f"(sv_a[127])
                        : "r"(s_addr_u + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float2 _reg_reduce_max2_0 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&sv_a[0], _reg_reduce_max2_0);
                    row_max_x32_accum(&sv_a[32], _reg_reduce_max2_0);
                    row_max_x32_accum(&sv_a[64], _reg_reduce_max2_0);
                    row_max_x32_accum(&sv_a[96], _reg_reduce_max2_0);
                    float tile_max_a = row_max_reduce(_reg_reduce_max2_0);
                    int tail_valid_a = tail_base - n_block_a * 128;
                    if (tail_valid_a < 128) {
                        uint32_t _thresh_mask_1;
                        {
                            int _lim_2 = tail_valid_a;
                            if (_lim_2 <= 0) { _thresh_mask_1 = 0u; }
                            else if (_lim_2 >= 32) { _thresh_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_1) : "r"(_lim_2));
                            }
                        }
                        if (!(_thresh_mask_1 & (1u << 0))) sv_a[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 1))) sv_a[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 2))) sv_a[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 3))) sv_a[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 4))) sv_a[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 5))) sv_a[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 6))) sv_a[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 7))) sv_a[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 8))) sv_a[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 9))) sv_a[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 10))) sv_a[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 11))) sv_a[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 12))) sv_a[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 13))) sv_a[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 14))) sv_a[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 15))) sv_a[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 16))) sv_a[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 17))) sv_a[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 18))) sv_a[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 19))) sv_a[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 20))) sv_a[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 21))) sv_a[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 22))) sv_a[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 23))) sv_a[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 24))) sv_a[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 25))) sv_a[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 26))) sv_a[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 27))) sv_a[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 28))) sv_a[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 29))) sv_a[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 30))) sv_a[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 31))) sv_a[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_3;
                        {
                            int _lim_4 = tail_valid_a - 32;
                            if (_lim_4 <= 0) { _thresh_mask_3 = 0u; }
                            else if (_lim_4 >= 32) { _thresh_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_3) : "r"(_lim_4));
                            }
                        }
                        if (!(_thresh_mask_3 & (1u << 0))) sv_a[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 1))) sv_a[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 2))) sv_a[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 3))) sv_a[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 4))) sv_a[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 5))) sv_a[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 6))) sv_a[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 7))) sv_a[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 8))) sv_a[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 9))) sv_a[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 10))) sv_a[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 11))) sv_a[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 12))) sv_a[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 13))) sv_a[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 14))) sv_a[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 15))) sv_a[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 16))) sv_a[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 17))) sv_a[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 18))) sv_a[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 19))) sv_a[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 20))) sv_a[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 21))) sv_a[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 22))) sv_a[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 23))) sv_a[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 24))) sv_a[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 25))) sv_a[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 26))) sv_a[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 27))) sv_a[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 28))) sv_a[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 29))) sv_a[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 30))) sv_a[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 31))) sv_a[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_5;
                        {
                            int _lim_6 = tail_valid_a - 64;
                            if (_lim_6 <= 0) { _thresh_mask_5 = 0u; }
                            else if (_lim_6 >= 32) { _thresh_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_5) : "r"(_lim_6));
                            }
                        }
                        if (!(_thresh_mask_5 & (1u << 0))) sv_a[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 1))) sv_a[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 2))) sv_a[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 3))) sv_a[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 4))) sv_a[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 5))) sv_a[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 6))) sv_a[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 7))) sv_a[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 8))) sv_a[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 9))) sv_a[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 10))) sv_a[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 11))) sv_a[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 12))) sv_a[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 13))) sv_a[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 14))) sv_a[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 15))) sv_a[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 16))) sv_a[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 17))) sv_a[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 18))) sv_a[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 19))) sv_a[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 20))) sv_a[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 21))) sv_a[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 22))) sv_a[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 23))) sv_a[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 24))) sv_a[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 25))) sv_a[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 26))) sv_a[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 27))) sv_a[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 28))) sv_a[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 29))) sv_a[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 30))) sv_a[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 31))) sv_a[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_7;
                        {
                            int _lim_8 = tail_valid_a - 96;
                            if (_lim_8 <= 0) { _thresh_mask_7 = 0u; }
                            else if (_lim_8 >= 32) { _thresh_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_7) : "r"(_lim_8));
                            }
                        }
                        if (!(_thresh_mask_7 & (1u << 0))) sv_a[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 1))) sv_a[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 2))) sv_a[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 3))) sv_a[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 4))) sv_a[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 5))) sv_a[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 6))) sv_a[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 7))) sv_a[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 8))) sv_a[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 9))) sv_a[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 10))) sv_a[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 11))) sv_a[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 12))) sv_a[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 13))) sv_a[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 14))) sv_a[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 15))) sv_a[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 16))) sv_a[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 17))) sv_a[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 18))) sv_a[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 19))) sv_a[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 20))) sv_a[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 21))) sv_a[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 22))) sv_a[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 23))) sv_a[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 24))) sv_a[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 25))) sv_a[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 26))) sv_a[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 27))) sv_a[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 28))) sv_a[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 29))) sv_a[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 30))) sv_a[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 31))) sv_a[127] = -CUDART_INF_F;
                        float2 _reg_reduce_max2_10 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv_a[0], _reg_reduce_max2_10);
                        row_max_x32_accum(&sv_a[32], _reg_reduce_max2_10);
                        row_max_x32_accum(&sv_a[64], _reg_reduce_max2_10);
                        row_max_x32_accum(&sv_a[96], _reg_reduce_max2_10);
                        float _reg_array_reduce_9 = row_max_reduce(_reg_reduce_max2_10);
                        tile_max_a = _reg_array_reduce_9;
                    }
                    if (two_stages == 1) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(s_read_addr + (stage) * 8);
                    }
                    float new_max_a;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max_a) : "f"(tile_max_a), "f"(row_max));
                    float safe_max_a = ((new_max_a == -CUDART_INF_F) ? 0.0f : new_max_a);
                    float neg_max_scaled_a = safe_max_a * neg_coef;
                    float acc_scale_log2_a = __fmaf_rn(row_max, exp_coef, neg_max_scaled_a);
                    float acc_scale_a;
                    if (acc_scale_log2_a >= -8.0f) {
                        acc_scale_a = 1.0f;
                        safe_max_a = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                        neg_max_scaled_a = safe_max_a * neg_coef;
                        nms_cache = neg_max_scaled_a;
                    } else {
                        float _exp2_11 = approx_exp2(acc_scale_log2_a);
                        acc_scale_a = ((row_max > -CUDART_INF_F) ? _exp2_11 : 1.0f);
                        row_max = new_max_a;
                        nms_cache = neg_max_scaled_a;
                    }
                    float scale_value_a[1];
                    scale_value_a[0] = acc_scale_a;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(s_addr_u), "f"(scale_value_a[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(scale_full_addr + (stage) * 8);
                    const float2 _fma_b2_12 = {exp_coef, exp_coef};
                    const float2 _fma_c2_13 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 0))[_lf], _fma_b2_12, _fma_c2_13);
                    const float2 _fma_b2_14 = {exp_coef, exp_coef};
                    const float2 _fma_c2_15 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 2))[_lf], _fma_b2_14, _fma_c2_15);
                    const float2 _fma_b2_16 = {exp_coef, exp_coef};
                    const float2 _fma_c2_17 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 4))[_lf], _fma_b2_16, _fma_c2_17);
                    const float2 _fma_b2_18 = {exp_coef, exp_coef};
                    const float2 _fma_c2_19 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 6))[_lf], _fma_b2_18, _fma_c2_19);
                    const float2 _fma_b2_20 = {exp_coef, exp_coef};
                    const float2 _fma_c2_21 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 8))[_lf], _fma_b2_20, _fma_c2_21);
                    const float2 _fma_b2_22 = {exp_coef, exp_coef};
                    const float2 _fma_c2_23 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 10))[_lf], _fma_b2_22, _fma_c2_23);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_24 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2], sv_a[_le*2 + 1]));
                            sv_a[_le*2] = _exp2_pair_24.x;
                            sv_a[_le*2 + 1] = _exp2_pair_24.y;
                        } else {
                            sv_a[_le*2] = approx_exp2(sv_a[_le*2]);
                            sv_a[_le*2 + 1] = approx_exp2(sv_a[_le*2 + 1]);
                        }
                    }
                    const float2 _fma_b2_25 = {exp_coef, exp_coef};
                    const float2 _fma_c2_26 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 12))[_lf], _fma_b2_25, _fma_c2_26);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_27 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 2], sv_a[_le*2 + 1 + 2]));
                            sv_a[_le*2 + 2] = _exp2_pair_27.x;
                            sv_a[_le*2 + 1 + 2] = _exp2_pair_27.y;
                        } else {
                            sv_a[_le*2 + 2] = approx_exp2(sv_a[_le*2 + 2]);
                            sv_a[_le*2 + 1 + 2] = approx_exp2(sv_a[_le*2 + 1 + 2]);
                        }
                    }
                    const float2 _fma_b2_28 = {exp_coef, exp_coef};
                    const float2 _fma_c2_29 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 14))[_lf], _fma_b2_28, _fma_c2_29);
                    float2 psum0a = make_float2(sv_a[0], sv_a[1]);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_30 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 4], sv_a[_le*2 + 1 + 4]));
                            sv_a[_le*2 + 4] = _exp2_pair_30.x;
                            sv_a[_le*2 + 1 + 4] = _exp2_pair_30.y;
                        } else {
                            sv_a[_le*2 + 4] = approx_exp2(sv_a[_le*2 + 4]);
                            sv_a[_le*2 + 1 + 4] = approx_exp2(sv_a[_le*2 + 1 + 4]);
                        }
                    }
                    const float2 _fma_b2_31 = {exp_coef, exp_coef};
                    const float2 _fma_c2_32 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 16))[_lf], _fma_b2_31, _fma_c2_32);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[2], sv_a[3]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_33 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 6], sv_a[_le*2 + 1 + 6]));
                            sv_a[_le*2 + 6] = _exp2_pair_33.x;
                            sv_a[_le*2 + 1 + 6] = _exp2_pair_33.y;
                        } else {
                            sv_a[_le*2 + 6] = approx_exp2(sv_a[_le*2 + 6]);
                            sv_a[_le*2 + 1 + 6] = approx_exp2(sv_a[_le*2 + 1 + 6]);
                        }
                    }
                    const float2 _fma_b2_34 = {exp_coef, exp_coef};
                    const float2 _fma_c2_35 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 18))[_lf], _fma_b2_34, _fma_c2_35);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[4], sv_a[5]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_36 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 8], sv_a[_le*2 + 1 + 8]));
                            sv_a[_le*2 + 8] = _exp2_pair_36.x;
                            sv_a[_le*2 + 1 + 8] = _exp2_pair_36.y;
                        } else {
                            sv_a[_le*2 + 8] = approx_exp2(sv_a[_le*2 + 8]);
                            sv_a[_le*2 + 1 + 8] = approx_exp2(sv_a[_le*2 + 1 + 8]);
                        }
                    }
                    const float2 _fma_b2_37 = {exp_coef, exp_coef};
                    const float2 _fma_c2_38 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 20))[_lf], _fma_b2_37, _fma_c2_38);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[6], sv_a[7]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 10] = approx_exp2(sv_a[_le + 10]);
                    }
                    const float2 _fma_b2_39 = {exp_coef, exp_coef};
                    const float2 _fma_c2_40 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 22))[_lf], _fma_b2_39, _fma_c2_40);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[8], sv_a[9]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 12] = approx_exp2(sv_a[_le + 12]);
                    }
                    const float2 _fma_b2_41 = {exp_coef, exp_coef};
                    const float2 _fma_c2_42 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 24))[_lf], _fma_b2_41, _fma_c2_42);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[10], sv_a[11]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 14] = approx_exp2(sv_a[_le + 14]);
                    }
                    const float2 _fma_b2_43 = {exp_coef, exp_coef};
                    const float2 _fma_c2_44 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 26))[_lf], _fma_b2_43, _fma_c2_44);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[12], sv_a[13]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 16] = approx_exp2(sv_a[_le + 16]);
                    }
                    const float2 _fma_b2_45 = {exp_coef, exp_coef};
                    const float2 _fma_c2_46 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 28))[_lf], _fma_b2_45, _fma_c2_46);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[14], sv_a[15]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 18] = approx_exp2(sv_a[_le + 18]);
                    }
                    const float2 _fma_b2_47 = {exp_coef, exp_coef};
                    const float2 _fma_c2_48 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 30))[_lf], _fma_b2_47, _fma_c2_48);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[16], sv_a[17]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 20] = approx_exp2(sv_a[_le + 20]);
                    }
                    const float2 _fma_b2_49 = {exp_coef, exp_coef};
                    const float2 _fma_c2_50 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 32))[_lf], _fma_b2_49, _fma_c2_50);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[18], sv_a[19]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 22] = approx_exp2(sv_a[_le + 22]);
                    }
                    const float2 _fma_b2_51 = {exp_coef, exp_coef};
                    const float2 _fma_c2_52 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 34))[_lf], _fma_b2_51, _fma_c2_52);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[20], sv_a[21]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 24] = approx_exp2(sv_a[_le + 24]);
                    }
                    const float2 _fma_b2_53 = {exp_coef, exp_coef};
                    const float2 _fma_c2_54 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 36))[_lf], _fma_b2_53, _fma_c2_54);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[22], sv_a[23]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 26] = approx_exp2(sv_a[_le + 26]);
                    }
                    const float2 _fma_b2_55 = {exp_coef, exp_coef};
                    const float2 _fma_c2_56 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 38))[_lf], _fma_b2_55, _fma_c2_56);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[24], sv_a[25]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 28] = approx_exp2(sv_a[_le + 28]);
                    }
                    const float2 _fma_b2_57 = {exp_coef, exp_coef};
                    const float2 _fma_c2_58 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 40))[_lf], _fma_b2_57, _fma_c2_58);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[26], sv_a[27]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 30] = approx_exp2(sv_a[_le + 30]);
                    }
                    const float2 _fma_b2_59 = {exp_coef, exp_coef};
                    const float2 _fma_c2_60 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 42))[_lf], _fma_b2_59, _fma_c2_60);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[28], sv_a[29]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_61 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 32], sv_a[_le*2 + 1 + 32]));
                            sv_a[_le*2 + 32] = _exp2_pair_61.x;
                            sv_a[_le*2 + 1 + 32] = _exp2_pair_61.y;
                        } else {
                            sv_a[_le*2 + 32] = approx_exp2(sv_a[_le*2 + 32]);
                            sv_a[_le*2 + 1 + 32] = approx_exp2(sv_a[_le*2 + 1 + 32]);
                        }
                    }
                    const float2 _fma_b2_62 = {exp_coef, exp_coef};
                    const float2 _fma_c2_63 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 44))[_lf], _fma_b2_62, _fma_c2_63);
                    psum0a = add_f32x2(psum0a, make_float2(sv_a[30], sv_a[31]));
                    unsigned int pk0a[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_a[0]), "f"(sv_a[1]),
                                               "f"(sv_a[2]), "f"(sv_a[3]));
                        pk0a[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[4]), "f"(sv_a[5]),
                                               "f"(sv_a[6]), "f"(sv_a[7]));
                        pk0a[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[8]), "f"(sv_a[9]),
                                               "f"(sv_a[10]), "f"(sv_a[11]));
                        pk0a[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[12]), "f"(sv_a[13]),
                                               "f"(sv_a[14]), "f"(sv_a[15]));
                        pk0a[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[16]), "f"(sv_a[17]),
                                               "f"(sv_a[18]), "f"(sv_a[19]));
                        pk0a[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[20]), "f"(sv_a[21]),
                                               "f"(sv_a[22]), "f"(sv_a[23]));
                        pk0a[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[24]), "f"(sv_a[25]),
                                               "f"(sv_a[26]), "f"(sv_a[27]));
                        pk0a[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[28]), "f"(sv_a[29]),
                                               "f"(sv_a[30]), "f"(sv_a[31]));
                        pk0a[7] = _packed;
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_64 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 34], sv_a[_le*2 + 1 + 34]));
                            sv_a[_le*2 + 34] = _exp2_pair_64.x;
                            sv_a[_le*2 + 1 + 34] = _exp2_pair_64.y;
                        } else {
                            sv_a[_le*2 + 34] = approx_exp2(sv_a[_le*2 + 34]);
                            sv_a[_le*2 + 1 + 34] = approx_exp2(sv_a[_le*2 + 1 + 34]);
                        }
                    }
                    const float2 _fma_b2_65 = {exp_coef, exp_coef};
                    const float2 _fma_c2_66 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 46))[_lf], _fma_b2_65, _fma_c2_66);
                    float2 psum1a = make_float2(sv_a[32], sv_a[33]);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_67 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 36], sv_a[_le*2 + 1 + 36]));
                            sv_a[_le*2 + 36] = _exp2_pair_67.x;
                            sv_a[_le*2 + 1 + 36] = _exp2_pair_67.y;
                        } else {
                            sv_a[_le*2 + 36] = approx_exp2(sv_a[_le*2 + 36]);
                            sv_a[_le*2 + 1 + 36] = approx_exp2(sv_a[_le*2 + 1 + 36]);
                        }
                    }
                    const float2 _fma_b2_68 = {exp_coef, exp_coef};
                    const float2 _fma_c2_69 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 48))[_lf], _fma_b2_68, _fma_c2_69);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[34], sv_a[35]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_70 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 38], sv_a[_le*2 + 1 + 38]));
                            sv_a[_le*2 + 38] = _exp2_pair_70.x;
                            sv_a[_le*2 + 1 + 38] = _exp2_pair_70.y;
                        } else {
                            sv_a[_le*2 + 38] = approx_exp2(sv_a[_le*2 + 38]);
                            sv_a[_le*2 + 1 + 38] = approx_exp2(sv_a[_le*2 + 1 + 38]);
                        }
                    }
                    const float2 _fma_b2_71 = {exp_coef, exp_coef};
                    const float2 _fma_c2_72 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 50))[_lf], _fma_b2_71, _fma_c2_72);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[36], sv_a[37]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_73 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 40], sv_a[_le*2 + 1 + 40]));
                            sv_a[_le*2 + 40] = _exp2_pair_73.x;
                            sv_a[_le*2 + 1 + 40] = _exp2_pair_73.y;
                        } else {
                            sv_a[_le*2 + 40] = approx_exp2(sv_a[_le*2 + 40]);
                            sv_a[_le*2 + 1 + 40] = approx_exp2(sv_a[_le*2 + 1 + 40]);
                        }
                    }
                    const float2 _fma_b2_74 = {exp_coef, exp_coef};
                    const float2 _fma_c2_75 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 52))[_lf], _fma_b2_74, _fma_c2_75);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[38], sv_a[39]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 42] = approx_exp2(sv_a[_le + 42]);
                    }
                    const float2 _fma_b2_76 = {exp_coef, exp_coef};
                    const float2 _fma_c2_77 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 54))[_lf], _fma_b2_76, _fma_c2_77);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[40], sv_a[41]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 44] = approx_exp2(sv_a[_le + 44]);
                    }
                    const float2 _fma_b2_78 = {exp_coef, exp_coef};
                    const float2 _fma_c2_79 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 56))[_lf], _fma_b2_78, _fma_c2_79);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[42], sv_a[43]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 46] = approx_exp2(sv_a[_le + 46]);
                    }
                    const float2 _fma_b2_80 = {exp_coef, exp_coef};
                    const float2 _fma_c2_81 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 58))[_lf], _fma_b2_80, _fma_c2_81);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[44], sv_a[45]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 48] = approx_exp2(sv_a[_le + 48]);
                    }
                    const float2 _fma_b2_82 = {exp_coef, exp_coef};
                    const float2 _fma_c2_83 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 60))[_lf], _fma_b2_82, _fma_c2_83);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[46], sv_a[47]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 50] = approx_exp2(sv_a[_le + 50]);
                    }
                    const float2 _fma_b2_84 = {exp_coef, exp_coef};
                    const float2 _fma_c2_85 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 62))[_lf], _fma_b2_84, _fma_c2_85);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[48], sv_a[49]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 52] = approx_exp2(sv_a[_le + 52]);
                    }
                    const float2 _fma_b2_86 = {exp_coef, exp_coef};
                    const float2 _fma_c2_87 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 64))[_lf], _fma_b2_86, _fma_c2_87);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[50], sv_a[51]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 54] = approx_exp2(sv_a[_le + 54]);
                    }
                    const float2 _fma_b2_88 = {exp_coef, exp_coef};
                    const float2 _fma_c2_89 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 66))[_lf], _fma_b2_88, _fma_c2_89);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[52], sv_a[53]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 56] = approx_exp2(sv_a[_le + 56]);
                    }
                    const float2 _fma_b2_90 = {exp_coef, exp_coef};
                    const float2 _fma_c2_91 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 68))[_lf], _fma_b2_90, _fma_c2_91);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[54], sv_a[55]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 58] = approx_exp2(sv_a[_le + 58]);
                    }
                    const float2 _fma_b2_92 = {exp_coef, exp_coef};
                    const float2 _fma_c2_93 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 70))[_lf], _fma_b2_92, _fma_c2_93);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[56], sv_a[57]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 60] = approx_exp2(sv_a[_le + 60]);
                    }
                    const float2 _fma_b2_94 = {exp_coef, exp_coef};
                    const float2 _fma_c2_95 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 72))[_lf], _fma_b2_94, _fma_c2_95);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[58], sv_a[59]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 62] = approx_exp2(sv_a[_le + 62]);
                    }
                    const float2 _fma_b2_96 = {exp_coef, exp_coef};
                    const float2 _fma_c2_97 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 74))[_lf], _fma_b2_96, _fma_c2_97);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[60], sv_a[61]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_98 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 64], sv_a[_le*2 + 1 + 64]));
                            sv_a[_le*2 + 64] = _exp2_pair_98.x;
                            sv_a[_le*2 + 1 + 64] = _exp2_pair_98.y;
                        } else {
                            sv_a[_le*2 + 64] = approx_exp2(sv_a[_le*2 + 64]);
                            sv_a[_le*2 + 1 + 64] = approx_exp2(sv_a[_le*2 + 1 + 64]);
                        }
                    }
                    const float2 _fma_b2_99 = {exp_coef, exp_coef};
                    const float2 _fma_c2_100 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 76))[_lf], _fma_b2_99, _fma_c2_100);
                    psum1a = add_f32x2(psum1a, make_float2(sv_a[62], sv_a[63]));
                    unsigned int pk1a[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_a[32]), "f"(sv_a[33]),
                                               "f"(sv_a[34]), "f"(sv_a[35]));
                        pk1a[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[36]), "f"(sv_a[37]),
                                               "f"(sv_a[38]), "f"(sv_a[39]));
                        pk1a[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[40]), "f"(sv_a[41]),
                                               "f"(sv_a[42]), "f"(sv_a[43]));
                        pk1a[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[44]), "f"(sv_a[45]),
                                               "f"(sv_a[46]), "f"(sv_a[47]));
                        pk1a[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[48]), "f"(sv_a[49]),
                                               "f"(sv_a[50]), "f"(sv_a[51]));
                        pk1a[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[52]), "f"(sv_a[53]),
                                               "f"(sv_a[54]), "f"(sv_a[55]));
                        pk1a[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[56]), "f"(sv_a[57]),
                                               "f"(sv_a[58]), "f"(sv_a[59]));
                        pk1a[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[60]), "f"(sv_a[61]),
                                               "f"(sv_a[62]), "f"(sv_a[63]));
                        pk1a[7] = _packed;
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_101 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 66], sv_a[_le*2 + 1 + 66]));
                            sv_a[_le*2 + 66] = _exp2_pair_101.x;
                            sv_a[_le*2 + 1 + 66] = _exp2_pair_101.y;
                        } else {
                            sv_a[_le*2 + 66] = approx_exp2(sv_a[_le*2 + 66]);
                            sv_a[_le*2 + 1 + 66] = approx_exp2(sv_a[_le*2 + 1 + 66]);
                        }
                    }
                    const float2 _fma_b2_102 = {exp_coef, exp_coef};
                    const float2 _fma_c2_103 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 78))[_lf], _fma_b2_102, _fma_c2_103);
                    float2 psum2a = make_float2(sv_a[64], sv_a[65]);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_104 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 68], sv_a[_le*2 + 1 + 68]));
                            sv_a[_le*2 + 68] = _exp2_pair_104.x;
                            sv_a[_le*2 + 1 + 68] = _exp2_pair_104.y;
                        } else {
                            sv_a[_le*2 + 68] = approx_exp2(sv_a[_le*2 + 68]);
                            sv_a[_le*2 + 1 + 68] = approx_exp2(sv_a[_le*2 + 1 + 68]);
                        }
                    }
                    const float2 _fma_b2_105 = {exp_coef, exp_coef};
                    const float2 _fma_c2_106 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 80))[_lf], _fma_b2_105, _fma_c2_106);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[66], sv_a[67]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_107 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 70], sv_a[_le*2 + 1 + 70]));
                            sv_a[_le*2 + 70] = _exp2_pair_107.x;
                            sv_a[_le*2 + 1 + 70] = _exp2_pair_107.y;
                        } else {
                            sv_a[_le*2 + 70] = approx_exp2(sv_a[_le*2 + 70]);
                            sv_a[_le*2 + 1 + 70] = approx_exp2(sv_a[_le*2 + 1 + 70]);
                        }
                    }
                    const float2 _fma_b2_108 = {exp_coef, exp_coef};
                    const float2 _fma_c2_109 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 82))[_lf], _fma_b2_108, _fma_c2_109);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[68], sv_a[69]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_110 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 72], sv_a[_le*2 + 1 + 72]));
                            sv_a[_le*2 + 72] = _exp2_pair_110.x;
                            sv_a[_le*2 + 1 + 72] = _exp2_pair_110.y;
                        } else {
                            sv_a[_le*2 + 72] = approx_exp2(sv_a[_le*2 + 72]);
                            sv_a[_le*2 + 1 + 72] = approx_exp2(sv_a[_le*2 + 1 + 72]);
                        }
                    }
                    const float2 _fma_b2_111 = {exp_coef, exp_coef};
                    const float2 _fma_c2_112 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 84))[_lf], _fma_b2_111, _fma_c2_112);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[70], sv_a[71]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 74] = approx_exp2(sv_a[_le + 74]);
                    }
                    const float2 _fma_b2_113 = {exp_coef, exp_coef};
                    const float2 _fma_c2_114 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 86))[_lf], _fma_b2_113, _fma_c2_114);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[72], sv_a[73]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 76] = approx_exp2(sv_a[_le + 76]);
                    }
                    const float2 _fma_b2_115 = {exp_coef, exp_coef};
                    const float2 _fma_c2_116 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 88))[_lf], _fma_b2_115, _fma_c2_116);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[74], sv_a[75]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 78] = approx_exp2(sv_a[_le + 78]);
                    }
                    const float2 _fma_b2_117 = {exp_coef, exp_coef};
                    const float2 _fma_c2_118 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 90))[_lf], _fma_b2_117, _fma_c2_118);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[76], sv_a[77]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 80] = approx_exp2(sv_a[_le + 80]);
                    }
                    const float2 _fma_b2_119 = {exp_coef, exp_coef};
                    const float2 _fma_c2_120 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 92))[_lf], _fma_b2_119, _fma_c2_120);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[78], sv_a[79]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 82] = approx_exp2(sv_a[_le + 82]);
                    }
                    const float2 _fma_b2_121 = {exp_coef, exp_coef};
                    const float2 _fma_c2_122 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 94))[_lf], _fma_b2_121, _fma_c2_122);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[80], sv_a[81]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 84] = approx_exp2(sv_a[_le + 84]);
                    }
                    const float2 _fma_b2_123 = {exp_coef, exp_coef};
                    const float2 _fma_c2_124 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 96))[_lf], _fma_b2_123, _fma_c2_124);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[82], sv_a[83]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 86] = approx_exp2(sv_a[_le + 86]);
                    }
                    const float2 _fma_b2_125 = {exp_coef, exp_coef};
                    const float2 _fma_c2_126 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 98))[_lf], _fma_b2_125, _fma_c2_126);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[84], sv_a[85]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 88] = approx_exp2(sv_a[_le + 88]);
                    }
                    const float2 _fma_b2_127 = {exp_coef, exp_coef};
                    const float2 _fma_c2_128 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 100))[_lf], _fma_b2_127, _fma_c2_128);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[86], sv_a[87]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 90] = approx_exp2(sv_a[_le + 90]);
                    }
                    const float2 _fma_b2_129 = {exp_coef, exp_coef};
                    const float2 _fma_c2_130 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 102))[_lf], _fma_b2_129, _fma_c2_130);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[88], sv_a[89]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 92] = approx_exp2(sv_a[_le + 92]);
                    }
                    const float2 _fma_b2_131 = {exp_coef, exp_coef};
                    const float2 _fma_c2_132 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 104))[_lf], _fma_b2_131, _fma_c2_132);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[90], sv_a[91]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 94] = approx_exp2(sv_a[_le + 94]);
                    }
                    const float2 _fma_b2_133 = {exp_coef, exp_coef};
                    const float2 _fma_c2_134 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 106))[_lf], _fma_b2_133, _fma_c2_134);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[92], sv_a[93]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_135 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 96], sv_a[_le*2 + 1 + 96]));
                            sv_a[_le*2 + 96] = _exp2_pair_135.x;
                            sv_a[_le*2 + 1 + 96] = _exp2_pair_135.y;
                        } else {
                            sv_a[_le*2 + 96] = approx_exp2(sv_a[_le*2 + 96]);
                            sv_a[_le*2 + 1 + 96] = approx_exp2(sv_a[_le*2 + 1 + 96]);
                        }
                    }
                    const float2 _fma_b2_136 = {exp_coef, exp_coef};
                    const float2 _fma_c2_137 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 108))[_lf], _fma_b2_136, _fma_c2_137);
                    psum2a = add_f32x2(psum2a, make_float2(sv_a[94], sv_a[95]));
                    unsigned int pk2a[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_a[64]), "f"(sv_a[65]),
                                               "f"(sv_a[66]), "f"(sv_a[67]));
                        pk2a[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[68]), "f"(sv_a[69]),
                                               "f"(sv_a[70]), "f"(sv_a[71]));
                        pk2a[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[72]), "f"(sv_a[73]),
                                               "f"(sv_a[74]), "f"(sv_a[75]));
                        pk2a[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[76]), "f"(sv_a[77]),
                                               "f"(sv_a[78]), "f"(sv_a[79]));
                        pk2a[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[80]), "f"(sv_a[81]),
                                               "f"(sv_a[82]), "f"(sv_a[83]));
                        pk2a[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[84]), "f"(sv_a[85]),
                                               "f"(sv_a[86]), "f"(sv_a[87]));
                        pk2a[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[88]), "f"(sv_a[89]),
                                               "f"(sv_a[90]), "f"(sv_a[91]));
                        pk2a[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[92]), "f"(sv_a[93]),
                                               "f"(sv_a[94]), "f"(sv_a[95]));
                        pk2a[7] = _packed;
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_138 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 98], sv_a[_le*2 + 1 + 98]));
                            sv_a[_le*2 + 98] = _exp2_pair_138.x;
                            sv_a[_le*2 + 1 + 98] = _exp2_pair_138.y;
                        } else {
                            sv_a[_le*2 + 98] = approx_exp2(sv_a[_le*2 + 98]);
                            sv_a[_le*2 + 1 + 98] = approx_exp2(sv_a[_le*2 + 1 + 98]);
                        }
                    }
                    const float2 _fma_b2_139 = {exp_coef, exp_coef};
                    const float2 _fma_c2_140 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 110))[_lf], _fma_b2_139, _fma_c2_140);
                    float2 psum3a = make_float2(sv_a[96], sv_a[97]);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_141 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 100], sv_a[_le*2 + 1 + 100]));
                            sv_a[_le*2 + 100] = _exp2_pair_141.x;
                            sv_a[_le*2 + 1 + 100] = _exp2_pair_141.y;
                        } else {
                            sv_a[_le*2 + 100] = approx_exp2(sv_a[_le*2 + 100]);
                            sv_a[_le*2 + 1 + 100] = approx_exp2(sv_a[_le*2 + 1 + 100]);
                        }
                    }
                    const float2 _fma_b2_142 = {exp_coef, exp_coef};
                    const float2 _fma_c2_143 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 112))[_lf], _fma_b2_142, _fma_c2_143);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[98], sv_a[99]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_144 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 102], sv_a[_le*2 + 1 + 102]));
                            sv_a[_le*2 + 102] = _exp2_pair_144.x;
                            sv_a[_le*2 + 1 + 102] = _exp2_pair_144.y;
                        } else {
                            sv_a[_le*2 + 102] = approx_exp2(sv_a[_le*2 + 102]);
                            sv_a[_le*2 + 1 + 102] = approx_exp2(sv_a[_le*2 + 1 + 102]);
                        }
                    }
                    const float2 _fma_b2_145 = {exp_coef, exp_coef};
                    const float2 _fma_c2_146 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 114))[_lf], _fma_b2_145, _fma_c2_146);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[100], sv_a[101]));
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_147 = ex2_emulation_f32x2_value(make_float2(sv_a[_le*2 + 104], sv_a[_le*2 + 1 + 104]));
                            sv_a[_le*2 + 104] = _exp2_pair_147.x;
                            sv_a[_le*2 + 1 + 104] = _exp2_pair_147.y;
                        } else {
                            sv_a[_le*2 + 104] = approx_exp2(sv_a[_le*2 + 104]);
                            sv_a[_le*2 + 1 + 104] = approx_exp2(sv_a[_le*2 + 1 + 104]);
                        }
                    }
                    const float2 _fma_b2_148 = {exp_coef, exp_coef};
                    const float2 _fma_c2_149 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 116))[_lf], _fma_b2_148, _fma_c2_149);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[102], sv_a[103]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 106] = approx_exp2(sv_a[_le + 106]);
                    }
                    const float2 _fma_b2_150 = {exp_coef, exp_coef};
                    const float2 _fma_c2_151 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 118))[_lf], _fma_b2_150, _fma_c2_151);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[104], sv_a[105]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 108] = approx_exp2(sv_a[_le + 108]);
                    }
                    const float2 _fma_b2_152 = {exp_coef, exp_coef};
                    const float2 _fma_c2_153 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 120))[_lf], _fma_b2_152, _fma_c2_153);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[106], sv_a[107]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 110] = approx_exp2(sv_a[_le + 110]);
                    }
                    const float2 _fma_b2_154 = {exp_coef, exp_coef};
                    const float2 _fma_c2_155 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 122))[_lf], _fma_b2_154, _fma_c2_155);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[108], sv_a[109]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 112] = approx_exp2(sv_a[_le + 112]);
                    }
                    const float2 _fma_b2_156 = {exp_coef, exp_coef};
                    const float2 _fma_c2_157 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 124))[_lf], _fma_b2_156, _fma_c2_157);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[110], sv_a[111]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 114] = approx_exp2(sv_a[_le + 114]);
                    }
                    const float2 _fma_b2_158 = {exp_coef, exp_coef};
                    const float2 _fma_c2_159 = {neg_max_scaled_a, neg_max_scaled_a};
                    #pragma unroll
                    for (int _lf = 0; _lf < 1; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_a + 126))[_lf], _fma_b2_158, _fma_c2_159);
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[112], sv_a[113]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 116] = approx_exp2(sv_a[_le + 116]);
                    }
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[114], sv_a[115]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 118] = approx_exp2(sv_a[_le + 118]);
                    }
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[116], sv_a[117]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 120] = approx_exp2(sv_a[_le + 120]);
                    }
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[118], sv_a[119]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 122] = approx_exp2(sv_a[_le + 122]);
                    }
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[120], sv_a[121]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 124] = approx_exp2(sv_a[_le + 124]);
                    }
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[122], sv_a[123]));
                    #pragma unroll
                    for (int _le = 0; _le < 2; _le++) {
                        sv_a[_le + 126] = approx_exp2(sv_a[_le + 126]);
                    }
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[124], sv_a[125]));
                    psum3a = add_f32x2(psum3a, make_float2(sv_a[126], sv_a[127]));
                    unsigned int pk3a[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv_a[96]), "f"(sv_a[97]),
                                               "f"(sv_a[98]), "f"(sv_a[99]));
                        pk3a[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[100]), "f"(sv_a[101]),
                                               "f"(sv_a[102]), "f"(sv_a[103]));
                        pk3a[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[104]), "f"(sv_a[105]),
                                               "f"(sv_a[106]), "f"(sv_a[107]));
                        pk3a[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[108]), "f"(sv_a[109]),
                                               "f"(sv_a[110]), "f"(sv_a[111]));
                        pk3a[3] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[112]), "f"(sv_a[113]),
                                               "f"(sv_a[114]), "f"(sv_a[115]));
                        pk3a[4] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[116]), "f"(sv_a[117]),
                                               "f"(sv_a[118]), "f"(sv_a[119]));
                        pk3a[5] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[120]), "f"(sv_a[121]),
                                               "f"(sv_a[122]), "f"(sv_a[123]));
                        pk3a[6] = _packed;
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
                            : "=r"(_packed) : "f"(sv_a[124]), "f"(sv_a[125]),
                                               "f"(sv_a[126]), "f"(sv_a[127]));
                        pk3a[7] = _packed;
                    }
                    if (two_stages == 1) {
                        mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                        _phase_s_read ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    tmem_st_x8_u32(p_addr_u, (const uint32_t*)pk0a);
                    tmem_st_x8_u32(p_addr_u + 8, (const uint32_t*)pk1a);
                    tmem_st_x8_u32(p_addr_u + 16, (const uint32_t*)pk2a);
                    tmem_st_x8_u32(p_addr_u + 24, (const uint32_t*)pk3a);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    float2 sum01_a = add_f32x2(psum0a, psum1a);
                    float2 sum23_a = add_f32x2(psum2a, psum3a);
                    float2 total_sum_a = add_f32x2(sum01_a, sum23_a);
                    row_sum = __fmaf_rn(row_sum, acc_scale_a, total_sum_a.x) + total_sum_a.y;
#endif
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (unsigned int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                        int n_block = (unsigned int)(num_n_blocks - 1) - n_iter;
                        mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
#else
                    for (unsigned int n_iter = 1; n_iter < num_n_blocks; n_iter++) {
                        mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 1000000);
#endif
                        _phase_s_full ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        int s_addr = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                        float sv[128];
                        float tile_max = -CUDART_INF_F;
                        {
                            float tile_max_lo = -CUDART_INF_F;
                            float tile_max_hi = -CUDART_INF_F;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
                            asm volatile(
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(tile_max_lo)
                                : "r"((unsigned int)tmem_scores + stage * 128 + (unsigned int)(warp % 4 * 32 << 16)));
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
                            asm volatile(
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(tile_max_hi)
                                : "r"((unsigned int)tmem_scores + stage * 128 + 64 + (unsigned int)(warp % 4 * 32 << 16)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            tile_max = max_noftz(tile_max_lo, tile_max_hi);
                        }
                        int tail_valid = tail_base - n_block * BLOCK_N;
                        if (tail_valid < BLOCK_N) {
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
                            uint32_t _thresh_mask_2;
                            {
                                int _lim_3 = tail_valid - 32;
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
                            uint32_t _thresh_mask_4;
                            {
                                int _lim_5 = tail_valid - 64;
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
                            uint32_t _thresh_mask_6;
                            {
                                int _lim_7 = tail_valid - 96;
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
                        }
#else
                        float sv_b[128];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv_b[0]), "=f"(sv_b[1]), "=f"(sv_b[2]), "=f"(sv_b[3]), "=f"(sv_b[4]), "=f"(sv_b[5]), "=f"(sv_b[6]), "=f"(sv_b[7]), "=f"(sv_b[8]), "=f"(sv_b[9]), "=f"(sv_b[10]), "=f"(sv_b[11]), "=f"(sv_b[12]), "=f"(sv_b[13]), "=f"(sv_b[14]), "=f"(sv_b[15]), "=f"(sv_b[16]), "=f"(sv_b[17]), "=f"(sv_b[18]), "=f"(sv_b[19]), "=f"(sv_b[20]), "=f"(sv_b[21]), "=f"(sv_b[22]), "=f"(sv_b[23]), "=f"(sv_b[24]), "=f"(sv_b[25]), "=f"(sv_b[26]), "=f"(sv_b[27]), "=f"(sv_b[28]), "=f"(sv_b[29]), "=f"(sv_b[30]), "=f"(sv_b[31])
                            : "r"(s_addr_u));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv_b[32]), "=f"(sv_b[33]), "=f"(sv_b[34]), "=f"(sv_b[35]), "=f"(sv_b[36]), "=f"(sv_b[37]), "=f"(sv_b[38]), "=f"(sv_b[39]), "=f"(sv_b[40]), "=f"(sv_b[41]), "=f"(sv_b[42]), "=f"(sv_b[43]), "=f"(sv_b[44]), "=f"(sv_b[45]), "=f"(sv_b[46]), "=f"(sv_b[47]), "=f"(sv_b[48]), "=f"(sv_b[49]), "=f"(sv_b[50]), "=f"(sv_b[51]), "=f"(sv_b[52]), "=f"(sv_b[53]), "=f"(sv_b[54]), "=f"(sv_b[55]), "=f"(sv_b[56]), "=f"(sv_b[57]), "=f"(sv_b[58]), "=f"(sv_b[59]), "=f"(sv_b[60]), "=f"(sv_b[61]), "=f"(sv_b[62]), "=f"(sv_b[63])
                            : "r"(s_addr_u + 32));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv_b[64]), "=f"(sv_b[65]), "=f"(sv_b[66]), "=f"(sv_b[67]), "=f"(sv_b[68]), "=f"(sv_b[69]), "=f"(sv_b[70]), "=f"(sv_b[71]), "=f"(sv_b[72]), "=f"(sv_b[73]), "=f"(sv_b[74]), "=f"(sv_b[75]), "=f"(sv_b[76]), "=f"(sv_b[77]), "=f"(sv_b[78]), "=f"(sv_b[79]), "=f"(sv_b[80]), "=f"(sv_b[81]), "=f"(sv_b[82]), "=f"(sv_b[83]), "=f"(sv_b[84]), "=f"(sv_b[85]), "=f"(sv_b[86]), "=f"(sv_b[87]), "=f"(sv_b[88]), "=f"(sv_b[89]), "=f"(sv_b[90]), "=f"(sv_b[91]), "=f"(sv_b[92]), "=f"(sv_b[93]), "=f"(sv_b[94]), "=f"(sv_b[95])
                            : "r"(s_addr_u + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv_b[96]), "=f"(sv_b[97]), "=f"(sv_b[98]), "=f"(sv_b[99]), "=f"(sv_b[100]), "=f"(sv_b[101]), "=f"(sv_b[102]), "=f"(sv_b[103]), "=f"(sv_b[104]), "=f"(sv_b[105]), "=f"(sv_b[106]), "=f"(sv_b[107]), "=f"(sv_b[108]), "=f"(sv_b[109]), "=f"(sv_b[110]), "=f"(sv_b[111]), "=f"(sv_b[112]), "=f"(sv_b[113]), "=f"(sv_b[114]), "=f"(sv_b[115]), "=f"(sv_b[116]), "=f"(sv_b[117]), "=f"(sv_b[118]), "=f"(sv_b[119]), "=f"(sv_b[120]), "=f"(sv_b[121]), "=f"(sv_b[122]), "=f"(sv_b[123]), "=f"(sv_b[124]), "=f"(sv_b[125]), "=f"(sv_b[126]), "=f"(sv_b[127])
                            : "r"(s_addr_u + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _reg_reduce_max2_160 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv_b[0], _reg_reduce_max2_160);
                        row_max_x32_accum(&sv_b[32], _reg_reduce_max2_160);
                        row_max_x32_accum(&sv_b[64], _reg_reduce_max2_160);
                        row_max_x32_accum(&sv_b[96], _reg_reduce_max2_160);
                        float tile_max_b = row_max_reduce(_reg_reduce_max2_160);
#endif
                        if (two_stages == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(s_read_addr + (stage) * 8);
                        }
#if __CUDA_ARCH__ == 1030
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
#else
                        float new_max_b;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max_b) : "f"(tile_max_b), "f"(row_max));
                        float safe_max_b = new_max_b;
                        float neg_max_scaled_b = safe_max_b * neg_coef;
                        float acc_scale_log2_b = __fmaf_rn(row_max, exp_coef, neg_max_scaled_b);
                        float acc_scale_b;
                        if (acc_scale_log2_b >= -8.0f) {
                            acc_scale_b = 1.0f;
                            neg_max_scaled_b = nms_cache;
#endif
                        } else {
#if __CUDA_ARCH__ == 1030
                            float _exp2_10 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_10 : 1.0f);
                            row_max = new_max;
#else
                            float _exp2_161 = approx_exp2(acc_scale_log2_b);
                            acc_scale_b = _exp2_161;
                            row_max = new_max_b;
                            nms_cache = neg_max_scaled_b;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        float neg_max_scaled = -new_max_scaled;
                        float scale_value[1];
                        scale_value[0] = acc_scale;
#else
                        float scale_value_b[1];
                        scale_value_b[0] = acc_scale_b;
#endif
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
#if __CUDA_ARCH__ == 1030
                            :: "r"(s_addr), "f"(scale_value[0]));
#else
                            :: "r"(s_addr_u), "f"(scale_value_b[0]));
#endif
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#else
#endif
                        mbarrier_arrive(scale_full_addr + (stage) * 8);
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_11 = {exp_coef, exp_coef};
                        const float2 _fma_c2_12 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_162 = {exp_coef, exp_coef};
                        const float2 _fma_c2_163 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_11, _fma_c2_12);
                        const float2 _fma_b2_13 = {exp_coef, exp_coef};
                        const float2 _fma_c2_14 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 0))[_lf], _fma_b2_162, _fma_c2_163);
                        const float2 _fma_b2_164 = {exp_coef, exp_coef};
                        const float2 _fma_c2_165 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_13, _fma_c2_14);
                        const float2 _fma_b2_15 = {exp_coef, exp_coef};
                        const float2 _fma_c2_16 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 2))[_lf], _fma_b2_164, _fma_c2_165);
                        const float2 _fma_b2_166 = {exp_coef, exp_coef};
                        const float2 _fma_c2_167 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_15, _fma_c2_16);
                        const float2 _fma_b2_17 = {exp_coef, exp_coef};
                        const float2 _fma_c2_18 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 4))[_lf], _fma_b2_166, _fma_c2_167);
                        const float2 _fma_b2_168 = {exp_coef, exp_coef};
                        const float2 _fma_c2_169 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_17, _fma_c2_18);
                        const float2 _fma_b2_19 = {exp_coef, exp_coef};
                        const float2 _fma_c2_20 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 6))[_lf], _fma_b2_168, _fma_c2_169);
                        const float2 _fma_b2_170 = {exp_coef, exp_coef};
                        const float2 _fma_c2_171 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_19, _fma_c2_20);
                        const float2 _fma_b2_21 = {exp_coef, exp_coef};
                        const float2 _fma_c2_22 = {neg_max_scaled, neg_max_scaled};
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 8))[_lf], _fma_b2_170, _fma_c2_171);
                        const float2 _fma_b2_172 = {exp_coef, exp_coef};
                        const float2 _fma_c2_173 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_21, _fma_c2_22);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 10))[_lf], _fma_b2_172, _fma_c2_173);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le] = approx_exp2(sv[_le]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_174 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2], sv_b[_le*2 + 1]));
                                sv_b[_le*2] = _exp2_pair_174.x;
                                sv_b[_le*2 + 1] = _exp2_pair_174.y;
                            } else {
                                sv_b[_le*2] = approx_exp2(sv_b[_le*2]);
                                sv_b[_le*2 + 1] = approx_exp2(sv_b[_le*2 + 1]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_23 = {exp_coef, exp_coef};
                        const float2 _fma_c2_24 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_175 = {exp_coef, exp_coef};
                        const float2 _fma_c2_176 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_23, _fma_c2_24);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 12))[_lf], _fma_b2_175, _fma_c2_176);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 2] = approx_exp2(sv[_le + 2]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_177 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 2], sv_b[_le*2 + 1 + 2]));
                                sv_b[_le*2 + 2] = _exp2_pair_177.x;
                                sv_b[_le*2 + 1 + 2] = _exp2_pair_177.y;
                            } else {
                                sv_b[_le*2 + 2] = approx_exp2(sv_b[_le*2 + 2]);
                                sv_b[_le*2 + 1 + 2] = approx_exp2(sv_b[_le*2 + 1 + 2]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_25 = {exp_coef, exp_coef};
                        const float2 _fma_c2_26 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_178 = {exp_coef, exp_coef};
                        const float2 _fma_c2_179 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_25, _fma_c2_26);
                        float2 psum0 = make_float2(sv[0], sv[1]);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 14))[_lf], _fma_b2_178, _fma_c2_179);
                        float2 psum0b = make_float2(sv_b[0], sv_b[1]);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 4] = approx_exp2(sv[_le + 4]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_180 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 4], sv_b[_le*2 + 1 + 4]));
                                sv_b[_le*2 + 4] = _exp2_pair_180.x;
                                sv_b[_le*2 + 1 + 4] = _exp2_pair_180.y;
                            } else {
                                sv_b[_le*2 + 4] = approx_exp2(sv_b[_le*2 + 4]);
                                sv_b[_le*2 + 1 + 4] = approx_exp2(sv_b[_le*2 + 1 + 4]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_27 = {exp_coef, exp_coef};
                        const float2 _fma_c2_28 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_181 = {exp_coef, exp_coef};
                        const float2 _fma_c2_182 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_27, _fma_c2_28);
                        psum0 = add_f32x2(psum0, make_float2(sv[2], sv[3]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 16))[_lf], _fma_b2_181, _fma_c2_182);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[2], sv_b[3]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 6] = approx_exp2(sv[_le + 6]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_183 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 6], sv_b[_le*2 + 1 + 6]));
                                sv_b[_le*2 + 6] = _exp2_pair_183.x;
                                sv_b[_le*2 + 1 + 6] = _exp2_pair_183.y;
                            } else {
                                sv_b[_le*2 + 6] = approx_exp2(sv_b[_le*2 + 6]);
                                sv_b[_le*2 + 1 + 6] = approx_exp2(sv_b[_le*2 + 1 + 6]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_29 = {exp_coef, exp_coef};
                        const float2 _fma_c2_30 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_184 = {exp_coef, exp_coef};
                        const float2 _fma_c2_185 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_29, _fma_c2_30);
                        psum0 = add_f32x2(psum0, make_float2(sv[4], sv[5]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 18))[_lf], _fma_b2_184, _fma_c2_185);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[4], sv_b[5]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 8] = approx_exp2(sv[_le + 8]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_186 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 8], sv_b[_le*2 + 1 + 8]));
                                sv_b[_le*2 + 8] = _exp2_pair_186.x;
                                sv_b[_le*2 + 1 + 8] = _exp2_pair_186.y;
                            } else {
                                sv_b[_le*2 + 8] = approx_exp2(sv_b[_le*2 + 8]);
                                sv_b[_le*2 + 1 + 8] = approx_exp2(sv_b[_le*2 + 1 + 8]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_31 = {exp_coef, exp_coef};
                        const float2 _fma_c2_32 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_187 = {exp_coef, exp_coef};
                        const float2 _fma_c2_188 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_31, _fma_c2_32);
                        psum0 = add_f32x2(psum0, make_float2(sv[6], sv[7]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 20))[_lf], _fma_b2_187, _fma_c2_188);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[6], sv_b[7]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 10] = approx_exp2(sv[_le + 10]);
#else
                            sv_b[_le + 10] = approx_exp2(sv_b[_le + 10]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_33 = {exp_coef, exp_coef};
                        const float2 _fma_c2_34 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_189 = {exp_coef, exp_coef};
                        const float2 _fma_c2_190 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_33, _fma_c2_34);
                        psum0 = add_f32x2(psum0, make_float2(sv[8], sv[9]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 22))[_lf], _fma_b2_189, _fma_c2_190);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[8], sv_b[9]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 12] = approx_exp2(sv[_le + 12]);
#else
                            sv_b[_le + 12] = approx_exp2(sv_b[_le + 12]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_35 = {exp_coef, exp_coef};
                        const float2 _fma_c2_36 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_191 = {exp_coef, exp_coef};
                        const float2 _fma_c2_192 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_35, _fma_c2_36);
                        psum0 = add_f32x2(psum0, make_float2(sv[10], sv[11]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 24))[_lf], _fma_b2_191, _fma_c2_192);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[10], sv_b[11]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 14] = approx_exp2(sv[_le + 14]);
#else
                            sv_b[_le + 14] = approx_exp2(sv_b[_le + 14]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_37 = {exp_coef, exp_coef};
                        const float2 _fma_c2_38 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_193 = {exp_coef, exp_coef};
                        const float2 _fma_c2_194 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_37, _fma_c2_38);
                        psum0 = add_f32x2(psum0, make_float2(sv[12], sv[13]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 26))[_lf], _fma_b2_193, _fma_c2_194);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[12], sv_b[13]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 16] = approx_exp2(sv[_le + 16]);
#else
                            sv_b[_le + 16] = approx_exp2(sv_b[_le + 16]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_39 = {exp_coef, exp_coef};
                        const float2 _fma_c2_40 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_195 = {exp_coef, exp_coef};
                        const float2 _fma_c2_196 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_39, _fma_c2_40);
                        psum0 = add_f32x2(psum0, make_float2(sv[14], sv[15]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 28))[_lf], _fma_b2_195, _fma_c2_196);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[14], sv_b[15]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 18] = approx_exp2(sv[_le + 18]);
#else
                            sv_b[_le + 18] = approx_exp2(sv_b[_le + 18]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_41 = {exp_coef, exp_coef};
                        const float2 _fma_c2_42 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_197 = {exp_coef, exp_coef};
                        const float2 _fma_c2_198 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_41, _fma_c2_42);
                        psum0 = add_f32x2(psum0, make_float2(sv[16], sv[17]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 30))[_lf], _fma_b2_197, _fma_c2_198);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[16], sv_b[17]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 20] = approx_exp2(sv[_le + 20]);
#else
                            sv_b[_le + 20] = approx_exp2(sv_b[_le + 20]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_43 = {exp_coef, exp_coef};
                        const float2 _fma_c2_44 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_199 = {exp_coef, exp_coef};
                        const float2 _fma_c2_200 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_43, _fma_c2_44);
                        psum0 = add_f32x2(psum0, make_float2(sv[18], sv[19]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 32))[_lf], _fma_b2_199, _fma_c2_200);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[18], sv_b[19]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 22] = approx_exp2(sv[_le + 22]);
#else
                            sv_b[_le + 22] = approx_exp2(sv_b[_le + 22]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_45 = {exp_coef, exp_coef};
                        const float2 _fma_c2_46 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_201 = {exp_coef, exp_coef};
                        const float2 _fma_c2_202 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 34))[_lf], _fma_b2_45, _fma_c2_46);
                        psum0 = add_f32x2(psum0, make_float2(sv[20], sv[21]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 34))[_lf], _fma_b2_201, _fma_c2_202);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[20], sv_b[21]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 24] = approx_exp2(sv[_le + 24]);
#else
                            sv_b[_le + 24] = approx_exp2(sv_b[_le + 24]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_47 = {exp_coef, exp_coef};
                        const float2 _fma_c2_48 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_203 = {exp_coef, exp_coef};
                        const float2 _fma_c2_204 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 36))[_lf], _fma_b2_47, _fma_c2_48);
                        psum0 = add_f32x2(psum0, make_float2(sv[22], sv[23]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 36))[_lf], _fma_b2_203, _fma_c2_204);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[22], sv_b[23]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 26] = approx_exp2(sv[_le + 26]);
#else
                            sv_b[_le + 26] = approx_exp2(sv_b[_le + 26]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_49 = {exp_coef, exp_coef};
                        const float2 _fma_c2_50 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_205 = {exp_coef, exp_coef};
                        const float2 _fma_c2_206 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 38))[_lf], _fma_b2_49, _fma_c2_50);
                        psum0 = add_f32x2(psum0, make_float2(sv[24], sv[25]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 38))[_lf], _fma_b2_205, _fma_c2_206);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[24], sv_b[25]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 28] = approx_exp2(sv[_le + 28]);
#else
                            sv_b[_le + 28] = approx_exp2(sv_b[_le + 28]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_51 = {exp_coef, exp_coef};
                        const float2 _fma_c2_52 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_207 = {exp_coef, exp_coef};
                        const float2 _fma_c2_208 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 40))[_lf], _fma_b2_51, _fma_c2_52);
                        psum0 = add_f32x2(psum0, make_float2(sv[26], sv[27]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 40))[_lf], _fma_b2_207, _fma_c2_208);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[26], sv_b[27]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 30] = approx_exp2(sv[_le + 30]);
#else
                            sv_b[_le + 30] = approx_exp2(sv_b[_le + 30]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_53 = {exp_coef, exp_coef};
                        const float2 _fma_c2_54 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_209 = {exp_coef, exp_coef};
                        const float2 _fma_c2_210 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 42))[_lf], _fma_b2_53, _fma_c2_54);
                        psum0 = add_f32x2(psum0, make_float2(sv[28], sv[29]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 42))[_lf], _fma_b2_209, _fma_c2_210);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[28], sv_b[29]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 32] = approx_exp2(sv[_le + 32]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_211 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 32], sv_b[_le*2 + 1 + 32]));
                                sv_b[_le*2 + 32] = _exp2_pair_211.x;
                                sv_b[_le*2 + 1 + 32] = _exp2_pair_211.y;
                            } else {
                                sv_b[_le*2 + 32] = approx_exp2(sv_b[_le*2 + 32]);
                                sv_b[_le*2 + 1 + 32] = approx_exp2(sv_b[_le*2 + 1 + 32]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_55 = {exp_coef, exp_coef};
                        const float2 _fma_c2_56 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_212 = {exp_coef, exp_coef};
                        const float2 _fma_c2_213 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 44))[_lf], _fma_b2_55, _fma_c2_56);
                        psum0 = add_f32x2(psum0, make_float2(sv[30], sv[31]));
                        unsigned int pk0[8];
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 44))[_lf], _fma_b2_212, _fma_c2_213);
                        psum0b = add_f32x2(psum0b, make_float2(sv_b[30], sv_b[31]));
                        unsigned int pk0b[8];
#endif
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                                   "f"(sv[2]), "f"(sv[3]));
                            pk0[0] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[0]), "f"(sv_b[1]),
                                                   "f"(sv_b[2]), "f"(sv_b[3]));
                            pk0b[0] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                                   "f"(sv[6]), "f"(sv[7]));
                            pk0[1] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[4]), "f"(sv_b[5]),
                                                   "f"(sv_b[6]), "f"(sv_b[7]));
                            pk0b[1] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                                   "f"(sv[10]), "f"(sv[11]));
                            pk0[2] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[8]), "f"(sv_b[9]),
                                                   "f"(sv_b[10]), "f"(sv_b[11]));
                            pk0b[2] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                                   "f"(sv[14]), "f"(sv[15]));
                            pk0[3] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[12]), "f"(sv_b[13]),
                                                   "f"(sv_b[14]), "f"(sv_b[15]));
                            pk0b[3] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                                   "f"(sv[18]), "f"(sv[19]));
                            pk0[4] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[16]), "f"(sv_b[17]),
                                                   "f"(sv_b[18]), "f"(sv_b[19]));
                            pk0b[4] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                                   "f"(sv[22]), "f"(sv[23]));
                            pk0[5] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[20]), "f"(sv_b[21]),
                                                   "f"(sv_b[22]), "f"(sv_b[23]));
                            pk0b[5] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                                   "f"(sv[26]), "f"(sv[27]));
                            pk0[6] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[24]), "f"(sv_b[25]),
                                                   "f"(sv_b[26]), "f"(sv_b[27]));
                            pk0b[6] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                                   "f"(sv[30]), "f"(sv[31]));
                            pk0[7] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[28]), "f"(sv_b[29]),
                                                   "f"(sv_b[30]), "f"(sv_b[31]));
                            pk0b[7] = _packed;
#endif
                        }
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 34] = approx_exp2(sv[_le + 34]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_214 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 34], sv_b[_le*2 + 1 + 34]));
                                sv_b[_le*2 + 34] = _exp2_pair_214.x;
                                sv_b[_le*2 + 1 + 34] = _exp2_pair_214.y;
                            } else {
                                sv_b[_le*2 + 34] = approx_exp2(sv_b[_le*2 + 34]);
                                sv_b[_le*2 + 1 + 34] = approx_exp2(sv_b[_le*2 + 1 + 34]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_57 = {exp_coef, exp_coef};
                        const float2 _fma_c2_58 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_215 = {exp_coef, exp_coef};
                        const float2 _fma_c2_216 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 46))[_lf], _fma_b2_57, _fma_c2_58);
                        float2 psum1 = make_float2(sv[32], sv[33]);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 46))[_lf], _fma_b2_215, _fma_c2_216);
                        float2 psum1b = make_float2(sv_b[32], sv_b[33]);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 36] = approx_exp2(sv[_le + 36]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_217 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 36], sv_b[_le*2 + 1 + 36]));
                                sv_b[_le*2 + 36] = _exp2_pair_217.x;
                                sv_b[_le*2 + 1 + 36] = _exp2_pair_217.y;
                            } else {
                                sv_b[_le*2 + 36] = approx_exp2(sv_b[_le*2 + 36]);
                                sv_b[_le*2 + 1 + 36] = approx_exp2(sv_b[_le*2 + 1 + 36]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_59 = {exp_coef, exp_coef};
                        const float2 _fma_c2_60 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_218 = {exp_coef, exp_coef};
                        const float2 _fma_c2_219 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_59, _fma_c2_60);
                        psum1 = add_f32x2(psum1, make_float2(sv[34], sv[35]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 48))[_lf], _fma_b2_218, _fma_c2_219);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[34], sv_b[35]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 38] = approx_exp2(sv[_le + 38]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_220 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 38], sv_b[_le*2 + 1 + 38]));
                                sv_b[_le*2 + 38] = _exp2_pair_220.x;
                                sv_b[_le*2 + 1 + 38] = _exp2_pair_220.y;
                            } else {
                                sv_b[_le*2 + 38] = approx_exp2(sv_b[_le*2 + 38]);
                                sv_b[_le*2 + 1 + 38] = approx_exp2(sv_b[_le*2 + 1 + 38]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_61 = {exp_coef, exp_coef};
                        const float2 _fma_c2_62 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_221 = {exp_coef, exp_coef};
                        const float2 _fma_c2_222 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 50))[_lf], _fma_b2_61, _fma_c2_62);
                        psum1 = add_f32x2(psum1, make_float2(sv[36], sv[37]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 50))[_lf], _fma_b2_221, _fma_c2_222);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[36], sv_b[37]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 40] = approx_exp2(sv[_le + 40]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_223 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 40], sv_b[_le*2 + 1 + 40]));
                                sv_b[_le*2 + 40] = _exp2_pair_223.x;
                                sv_b[_le*2 + 1 + 40] = _exp2_pair_223.y;
                            } else {
                                sv_b[_le*2 + 40] = approx_exp2(sv_b[_le*2 + 40]);
                                sv_b[_le*2 + 1 + 40] = approx_exp2(sv_b[_le*2 + 1 + 40]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_63 = {exp_coef, exp_coef};
                        const float2 _fma_c2_64 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_224 = {exp_coef, exp_coef};
                        const float2 _fma_c2_225 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 52))[_lf], _fma_b2_63, _fma_c2_64);
                        psum1 = add_f32x2(psum1, make_float2(sv[38], sv[39]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 52))[_lf], _fma_b2_224, _fma_c2_225);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[38], sv_b[39]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 42] = approx_exp2(sv[_le + 42]);
#else
                            sv_b[_le + 42] = approx_exp2(sv_b[_le + 42]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_65 = {exp_coef, exp_coef};
                        const float2 _fma_c2_66 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_226 = {exp_coef, exp_coef};
                        const float2 _fma_c2_227 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 54))[_lf], _fma_b2_65, _fma_c2_66);
                        psum1 = add_f32x2(psum1, make_float2(sv[40], sv[41]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 54))[_lf], _fma_b2_226, _fma_c2_227);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[40], sv_b[41]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 44] = approx_exp2(sv[_le + 44]);
#else
                            sv_b[_le + 44] = approx_exp2(sv_b[_le + 44]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_67 = {exp_coef, exp_coef};
                        const float2 _fma_c2_68 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_228 = {exp_coef, exp_coef};
                        const float2 _fma_c2_229 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 56))[_lf], _fma_b2_67, _fma_c2_68);
                        psum1 = add_f32x2(psum1, make_float2(sv[42], sv[43]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 56))[_lf], _fma_b2_228, _fma_c2_229);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[42], sv_b[43]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 46] = approx_exp2(sv[_le + 46]);
#else
                            sv_b[_le + 46] = approx_exp2(sv_b[_le + 46]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_69 = {exp_coef, exp_coef};
                        const float2 _fma_c2_70 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_230 = {exp_coef, exp_coef};
                        const float2 _fma_c2_231 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 58))[_lf], _fma_b2_69, _fma_c2_70);
                        psum1 = add_f32x2(psum1, make_float2(sv[44], sv[45]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 58))[_lf], _fma_b2_230, _fma_c2_231);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[44], sv_b[45]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 48] = approx_exp2(sv[_le + 48]);
#else
                            sv_b[_le + 48] = approx_exp2(sv_b[_le + 48]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_71 = {exp_coef, exp_coef};
                        const float2 _fma_c2_72 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_232 = {exp_coef, exp_coef};
                        const float2 _fma_c2_233 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 60))[_lf], _fma_b2_71, _fma_c2_72);
                        psum1 = add_f32x2(psum1, make_float2(sv[46], sv[47]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 60))[_lf], _fma_b2_232, _fma_c2_233);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[46], sv_b[47]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 50] = approx_exp2(sv[_le + 50]);
#else
                            sv_b[_le + 50] = approx_exp2(sv_b[_le + 50]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_73 = {exp_coef, exp_coef};
                        const float2 _fma_c2_74 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_234 = {exp_coef, exp_coef};
                        const float2 _fma_c2_235 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 62))[_lf], _fma_b2_73, _fma_c2_74);
                        psum1 = add_f32x2(psum1, make_float2(sv[48], sv[49]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 62))[_lf], _fma_b2_234, _fma_c2_235);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[48], sv_b[49]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 52] = approx_exp2(sv[_le + 52]);
#else
                            sv_b[_le + 52] = approx_exp2(sv_b[_le + 52]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_75 = {exp_coef, exp_coef};
                        const float2 _fma_c2_76 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_236 = {exp_coef, exp_coef};
                        const float2 _fma_c2_237 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_75, _fma_c2_76);
                        psum1 = add_f32x2(psum1, make_float2(sv[50], sv[51]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 64))[_lf], _fma_b2_236, _fma_c2_237);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[50], sv_b[51]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 54] = approx_exp2(sv[_le + 54]);
#else
                            sv_b[_le + 54] = approx_exp2(sv_b[_le + 54]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_77 = {exp_coef, exp_coef};
                        const float2 _fma_c2_78 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_238 = {exp_coef, exp_coef};
                        const float2 _fma_c2_239 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 66))[_lf], _fma_b2_77, _fma_c2_78);
                        psum1 = add_f32x2(psum1, make_float2(sv[52], sv[53]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 66))[_lf], _fma_b2_238, _fma_c2_239);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[52], sv_b[53]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 56] = approx_exp2(sv[_le + 56]);
#else
                            sv_b[_le + 56] = approx_exp2(sv_b[_le + 56]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_79 = {exp_coef, exp_coef};
                        const float2 _fma_c2_80 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_240 = {exp_coef, exp_coef};
                        const float2 _fma_c2_241 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 68))[_lf], _fma_b2_79, _fma_c2_80);
                        psum1 = add_f32x2(psum1, make_float2(sv[54], sv[55]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 68))[_lf], _fma_b2_240, _fma_c2_241);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[54], sv_b[55]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 58] = approx_exp2(sv[_le + 58]);
#else
                            sv_b[_le + 58] = approx_exp2(sv_b[_le + 58]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_81 = {exp_coef, exp_coef};
                        const float2 _fma_c2_82 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_242 = {exp_coef, exp_coef};
                        const float2 _fma_c2_243 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 70))[_lf], _fma_b2_81, _fma_c2_82);
                        psum1 = add_f32x2(psum1, make_float2(sv[56], sv[57]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 70))[_lf], _fma_b2_242, _fma_c2_243);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[56], sv_b[57]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 60] = approx_exp2(sv[_le + 60]);
#else
                            sv_b[_le + 60] = approx_exp2(sv_b[_le + 60]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_83 = {exp_coef, exp_coef};
                        const float2 _fma_c2_84 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_244 = {exp_coef, exp_coef};
                        const float2 _fma_c2_245 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 72))[_lf], _fma_b2_83, _fma_c2_84);
                        psum1 = add_f32x2(psum1, make_float2(sv[58], sv[59]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 72))[_lf], _fma_b2_244, _fma_c2_245);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[58], sv_b[59]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 62] = approx_exp2(sv[_le + 62]);
#else
                            sv_b[_le + 62] = approx_exp2(sv_b[_le + 62]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_85 = {exp_coef, exp_coef};
                        const float2 _fma_c2_86 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_246 = {exp_coef, exp_coef};
                        const float2 _fma_c2_247 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 74))[_lf], _fma_b2_85, _fma_c2_86);
                        psum1 = add_f32x2(psum1, make_float2(sv[60], sv[61]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 74))[_lf], _fma_b2_246, _fma_c2_247);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[60], sv_b[61]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 64] = approx_exp2(sv[_le + 64]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_248 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 64], sv_b[_le*2 + 1 + 64]));
                                sv_b[_le*2 + 64] = _exp2_pair_248.x;
                                sv_b[_le*2 + 1 + 64] = _exp2_pair_248.y;
                            } else {
                                sv_b[_le*2 + 64] = approx_exp2(sv_b[_le*2 + 64]);
                                sv_b[_le*2 + 1 + 64] = approx_exp2(sv_b[_le*2 + 1 + 64]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_87 = {exp_coef, exp_coef};
                        const float2 _fma_c2_88 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_249 = {exp_coef, exp_coef};
                        const float2 _fma_c2_250 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 76))[_lf], _fma_b2_87, _fma_c2_88);
                        psum1 = add_f32x2(psum1, make_float2(sv[62], sv[63]));
                        unsigned int pk1[8];
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 76))[_lf], _fma_b2_249, _fma_c2_250);
                        psum1b = add_f32x2(psum1b, make_float2(sv_b[62], sv_b[63]));
                        unsigned int pk1b[8];
#endif
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[32]), "f"(sv[33]),
                                                   "f"(sv[34]), "f"(sv[35]));
                            pk1[0] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[32]), "f"(sv_b[33]),
                                                   "f"(sv_b[34]), "f"(sv_b[35]));
                            pk1b[0] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[36]), "f"(sv[37]),
                                                   "f"(sv[38]), "f"(sv[39]));
                            pk1[1] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[36]), "f"(sv_b[37]),
                                                   "f"(sv_b[38]), "f"(sv_b[39]));
                            pk1b[1] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[40]), "f"(sv[41]),
                                                   "f"(sv[42]), "f"(sv[43]));
                            pk1[2] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[40]), "f"(sv_b[41]),
                                                   "f"(sv_b[42]), "f"(sv_b[43]));
                            pk1b[2] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[44]), "f"(sv[45]),
                                                   "f"(sv[46]), "f"(sv[47]));
                            pk1[3] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[44]), "f"(sv_b[45]),
                                                   "f"(sv_b[46]), "f"(sv_b[47]));
                            pk1b[3] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[48]), "f"(sv[49]),
                                                   "f"(sv[50]), "f"(sv[51]));
                            pk1[4] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[48]), "f"(sv_b[49]),
                                                   "f"(sv_b[50]), "f"(sv_b[51]));
                            pk1b[4] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[52]), "f"(sv[53]),
                                                   "f"(sv[54]), "f"(sv[55]));
                            pk1[5] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[52]), "f"(sv_b[53]),
                                                   "f"(sv_b[54]), "f"(sv_b[55]));
                            pk1b[5] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[56]), "f"(sv[57]),
                                                   "f"(sv[58]), "f"(sv[59]));
                            pk1[6] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[56]), "f"(sv_b[57]),
                                                   "f"(sv_b[58]), "f"(sv_b[59]));
                            pk1b[6] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[60]), "f"(sv[61]),
                                                   "f"(sv[62]), "f"(sv[63]));
                            pk1[7] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[60]), "f"(sv_b[61]),
                                                   "f"(sv_b[62]), "f"(sv_b[63]));
                            pk1b[7] = _packed;
#endif
                        }
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 66] = approx_exp2(sv[_le + 66]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_251 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 66], sv_b[_le*2 + 1 + 66]));
                                sv_b[_le*2 + 66] = _exp2_pair_251.x;
                                sv_b[_le*2 + 1 + 66] = _exp2_pair_251.y;
                            } else {
                                sv_b[_le*2 + 66] = approx_exp2(sv_b[_le*2 + 66]);
                                sv_b[_le*2 + 1 + 66] = approx_exp2(sv_b[_le*2 + 1 + 66]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_89 = {exp_coef, exp_coef};
                        const float2 _fma_c2_90 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_252 = {exp_coef, exp_coef};
                        const float2 _fma_c2_253 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 78))[_lf], _fma_b2_89, _fma_c2_90);
                        float2 psum2 = make_float2(sv[64], sv[65]);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 78))[_lf], _fma_b2_252, _fma_c2_253);
                        float2 psum2b = make_float2(sv_b[64], sv_b[65]);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 68] = approx_exp2(sv[_le + 68]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_254 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 68], sv_b[_le*2 + 1 + 68]));
                                sv_b[_le*2 + 68] = _exp2_pair_254.x;
                                sv_b[_le*2 + 1 + 68] = _exp2_pair_254.y;
                            } else {
                                sv_b[_le*2 + 68] = approx_exp2(sv_b[_le*2 + 68]);
                                sv_b[_le*2 + 1 + 68] = approx_exp2(sv_b[_le*2 + 1 + 68]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_91 = {exp_coef, exp_coef};
                        const float2 _fma_c2_92 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_255 = {exp_coef, exp_coef};
                        const float2 _fma_c2_256 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_91, _fma_c2_92);
                        psum2 = add_f32x2(psum2, make_float2(sv[66], sv[67]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 80))[_lf], _fma_b2_255, _fma_c2_256);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[66], sv_b[67]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 70] = approx_exp2(sv[_le + 70]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_257 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 70], sv_b[_le*2 + 1 + 70]));
                                sv_b[_le*2 + 70] = _exp2_pair_257.x;
                                sv_b[_le*2 + 1 + 70] = _exp2_pair_257.y;
                            } else {
                                sv_b[_le*2 + 70] = approx_exp2(sv_b[_le*2 + 70]);
                                sv_b[_le*2 + 1 + 70] = approx_exp2(sv_b[_le*2 + 1 + 70]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_93 = {exp_coef, exp_coef};
                        const float2 _fma_c2_94 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_258 = {exp_coef, exp_coef};
                        const float2 _fma_c2_259 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 82))[_lf], _fma_b2_93, _fma_c2_94);
                        psum2 = add_f32x2(psum2, make_float2(sv[68], sv[69]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 82))[_lf], _fma_b2_258, _fma_c2_259);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[68], sv_b[69]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 72] = approx_exp2(sv[_le + 72]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_260 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 72], sv_b[_le*2 + 1 + 72]));
                                sv_b[_le*2 + 72] = _exp2_pair_260.x;
                                sv_b[_le*2 + 1 + 72] = _exp2_pair_260.y;
                            } else {
                                sv_b[_le*2 + 72] = approx_exp2(sv_b[_le*2 + 72]);
                                sv_b[_le*2 + 1 + 72] = approx_exp2(sv_b[_le*2 + 1 + 72]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_95 = {exp_coef, exp_coef};
                        const float2 _fma_c2_96 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_261 = {exp_coef, exp_coef};
                        const float2 _fma_c2_262 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 84))[_lf], _fma_b2_95, _fma_c2_96);
                        psum2 = add_f32x2(psum2, make_float2(sv[70], sv[71]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 84))[_lf], _fma_b2_261, _fma_c2_262);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[70], sv_b[71]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 74] = approx_exp2(sv[_le + 74]);
#else
                            sv_b[_le + 74] = approx_exp2(sv_b[_le + 74]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_97 = {exp_coef, exp_coef};
                        const float2 _fma_c2_98 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_263 = {exp_coef, exp_coef};
                        const float2 _fma_c2_264 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 86))[_lf], _fma_b2_97, _fma_c2_98);
                        psum2 = add_f32x2(psum2, make_float2(sv[72], sv[73]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 86))[_lf], _fma_b2_263, _fma_c2_264);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[72], sv_b[73]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 76] = approx_exp2(sv[_le + 76]);
#else
                            sv_b[_le + 76] = approx_exp2(sv_b[_le + 76]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_99 = {exp_coef, exp_coef};
                        const float2 _fma_c2_100 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_265 = {exp_coef, exp_coef};
                        const float2 _fma_c2_266 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 88))[_lf], _fma_b2_99, _fma_c2_100);
                        psum2 = add_f32x2(psum2, make_float2(sv[74], sv[75]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 88))[_lf], _fma_b2_265, _fma_c2_266);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[74], sv_b[75]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 78] = approx_exp2(sv[_le + 78]);
#else
                            sv_b[_le + 78] = approx_exp2(sv_b[_le + 78]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_101 = {exp_coef, exp_coef};
                        const float2 _fma_c2_102 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_267 = {exp_coef, exp_coef};
                        const float2 _fma_c2_268 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 90))[_lf], _fma_b2_101, _fma_c2_102);
                        psum2 = add_f32x2(psum2, make_float2(sv[76], sv[77]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 90))[_lf], _fma_b2_267, _fma_c2_268);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[76], sv_b[77]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 80] = approx_exp2(sv[_le + 80]);
#else
                            sv_b[_le + 80] = approx_exp2(sv_b[_le + 80]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_103 = {exp_coef, exp_coef};
                        const float2 _fma_c2_104 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_269 = {exp_coef, exp_coef};
                        const float2 _fma_c2_270 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 92))[_lf], _fma_b2_103, _fma_c2_104);
                        psum2 = add_f32x2(psum2, make_float2(sv[78], sv[79]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 92))[_lf], _fma_b2_269, _fma_c2_270);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[78], sv_b[79]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 82] = approx_exp2(sv[_le + 82]);
#else
                            sv_b[_le + 82] = approx_exp2(sv_b[_le + 82]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_105 = {exp_coef, exp_coef};
                        const float2 _fma_c2_106 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_271 = {exp_coef, exp_coef};
                        const float2 _fma_c2_272 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 94))[_lf], _fma_b2_105, _fma_c2_106);
                        psum2 = add_f32x2(psum2, make_float2(sv[80], sv[81]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 94))[_lf], _fma_b2_271, _fma_c2_272);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[80], sv_b[81]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 84] = approx_exp2(sv[_le + 84]);
#else
                            sv_b[_le + 84] = approx_exp2(sv_b[_le + 84]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_107 = {exp_coef, exp_coef};
                        const float2 _fma_c2_108 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_273 = {exp_coef, exp_coef};
                        const float2 _fma_c2_274 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_107, _fma_c2_108);
                        psum2 = add_f32x2(psum2, make_float2(sv[82], sv[83]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 96))[_lf], _fma_b2_273, _fma_c2_274);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[82], sv_b[83]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 86] = approx_exp2(sv[_le + 86]);
#else
                            sv_b[_le + 86] = approx_exp2(sv_b[_le + 86]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_109 = {exp_coef, exp_coef};
                        const float2 _fma_c2_110 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_275 = {exp_coef, exp_coef};
                        const float2 _fma_c2_276 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 98))[_lf], _fma_b2_109, _fma_c2_110);
                        psum2 = add_f32x2(psum2, make_float2(sv[84], sv[85]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 98))[_lf], _fma_b2_275, _fma_c2_276);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[84], sv_b[85]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 88] = approx_exp2(sv[_le + 88]);
#else
                            sv_b[_le + 88] = approx_exp2(sv_b[_le + 88]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_111 = {exp_coef, exp_coef};
                        const float2 _fma_c2_112 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_277 = {exp_coef, exp_coef};
                        const float2 _fma_c2_278 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 100))[_lf], _fma_b2_111, _fma_c2_112);
                        psum2 = add_f32x2(psum2, make_float2(sv[86], sv[87]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 100))[_lf], _fma_b2_277, _fma_c2_278);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[86], sv_b[87]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 90] = approx_exp2(sv[_le + 90]);
#else
                            sv_b[_le + 90] = approx_exp2(sv_b[_le + 90]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_113 = {exp_coef, exp_coef};
                        const float2 _fma_c2_114 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_279 = {exp_coef, exp_coef};
                        const float2 _fma_c2_280 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 102))[_lf], _fma_b2_113, _fma_c2_114);
                        psum2 = add_f32x2(psum2, make_float2(sv[88], sv[89]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 102))[_lf], _fma_b2_279, _fma_c2_280);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[88], sv_b[89]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 92] = approx_exp2(sv[_le + 92]);
#else
                            sv_b[_le + 92] = approx_exp2(sv_b[_le + 92]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_115 = {exp_coef, exp_coef};
                        const float2 _fma_c2_116 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_281 = {exp_coef, exp_coef};
                        const float2 _fma_c2_282 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 104))[_lf], _fma_b2_115, _fma_c2_116);
                        psum2 = add_f32x2(psum2, make_float2(sv[90], sv[91]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 104))[_lf], _fma_b2_281, _fma_c2_282);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[90], sv_b[91]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 94] = approx_exp2(sv[_le + 94]);
#else
                            sv_b[_le + 94] = approx_exp2(sv_b[_le + 94]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_117 = {exp_coef, exp_coef};
                        const float2 _fma_c2_118 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_283 = {exp_coef, exp_coef};
                        const float2 _fma_c2_284 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 106))[_lf], _fma_b2_117, _fma_c2_118);
                        psum2 = add_f32x2(psum2, make_float2(sv[92], sv[93]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 106))[_lf], _fma_b2_283, _fma_c2_284);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[92], sv_b[93]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 96] = approx_exp2(sv[_le + 96]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_285 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 96], sv_b[_le*2 + 1 + 96]));
                                sv_b[_le*2 + 96] = _exp2_pair_285.x;
                                sv_b[_le*2 + 1 + 96] = _exp2_pair_285.y;
                            } else {
                                sv_b[_le*2 + 96] = approx_exp2(sv_b[_le*2 + 96]);
                                sv_b[_le*2 + 1 + 96] = approx_exp2(sv_b[_le*2 + 1 + 96]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_119 = {exp_coef, exp_coef};
                        const float2 _fma_c2_120 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_286 = {exp_coef, exp_coef};
                        const float2 _fma_c2_287 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 108))[_lf], _fma_b2_119, _fma_c2_120);
                        psum2 = add_f32x2(psum2, make_float2(sv[94], sv[95]));
                        unsigned int pk2[8];
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 108))[_lf], _fma_b2_286, _fma_c2_287);
                        psum2b = add_f32x2(psum2b, make_float2(sv_b[94], sv_b[95]));
                        unsigned int pk2b[8];
#endif
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[64]), "f"(sv[65]),
                                                   "f"(sv[66]), "f"(sv[67]));
                            pk2[0] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[64]), "f"(sv_b[65]),
                                                   "f"(sv_b[66]), "f"(sv_b[67]));
                            pk2b[0] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[68]), "f"(sv[69]),
                                                   "f"(sv[70]), "f"(sv[71]));
                            pk2[1] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[68]), "f"(sv_b[69]),
                                                   "f"(sv_b[70]), "f"(sv_b[71]));
                            pk2b[1] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[72]), "f"(sv[73]),
                                                   "f"(sv[74]), "f"(sv[75]));
                            pk2[2] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[72]), "f"(sv_b[73]),
                                                   "f"(sv_b[74]), "f"(sv_b[75]));
                            pk2b[2] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[76]), "f"(sv[77]),
                                                   "f"(sv[78]), "f"(sv[79]));
                            pk2[3] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[76]), "f"(sv_b[77]),
                                                   "f"(sv_b[78]), "f"(sv_b[79]));
                            pk2b[3] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[80]), "f"(sv[81]),
                                                   "f"(sv[82]), "f"(sv[83]));
                            pk2[4] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[80]), "f"(sv_b[81]),
                                                   "f"(sv_b[82]), "f"(sv_b[83]));
                            pk2b[4] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[84]), "f"(sv[85]),
                                                   "f"(sv[86]), "f"(sv[87]));
                            pk2[5] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[84]), "f"(sv_b[85]),
                                                   "f"(sv_b[86]), "f"(sv_b[87]));
                            pk2b[5] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[88]), "f"(sv[89]),
                                                   "f"(sv[90]), "f"(sv[91]));
                            pk2[6] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[88]), "f"(sv_b[89]),
                                                   "f"(sv_b[90]), "f"(sv_b[91]));
                            pk2b[6] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[92]), "f"(sv[93]),
                                                   "f"(sv[94]), "f"(sv[95]));
                            pk2[7] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[92]), "f"(sv_b[93]),
                                                   "f"(sv_b[94]), "f"(sv_b[95]));
                            pk2b[7] = _packed;
#endif
                        }
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 98] = approx_exp2(sv[_le + 98]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_288 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 98], sv_b[_le*2 + 1 + 98]));
                                sv_b[_le*2 + 98] = _exp2_pair_288.x;
                                sv_b[_le*2 + 1 + 98] = _exp2_pair_288.y;
                            } else {
                                sv_b[_le*2 + 98] = approx_exp2(sv_b[_le*2 + 98]);
                                sv_b[_le*2 + 1 + 98] = approx_exp2(sv_b[_le*2 + 1 + 98]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_121 = {exp_coef, exp_coef};
                        const float2 _fma_c2_122 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_289 = {exp_coef, exp_coef};
                        const float2 _fma_c2_290 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 110))[_lf], _fma_b2_121, _fma_c2_122);
                        float2 psum3 = make_float2(sv[96], sv[97]);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 110))[_lf], _fma_b2_289, _fma_c2_290);
                        float2 psum3b = make_float2(sv_b[96], sv_b[97]);
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 100] = approx_exp2(sv[_le + 100]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_291 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 100], sv_b[_le*2 + 1 + 100]));
                                sv_b[_le*2 + 100] = _exp2_pair_291.x;
                                sv_b[_le*2 + 1 + 100] = _exp2_pair_291.y;
                            } else {
                                sv_b[_le*2 + 100] = approx_exp2(sv_b[_le*2 + 100]);
                                sv_b[_le*2 + 1 + 100] = approx_exp2(sv_b[_le*2 + 1 + 100]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_123 = {exp_coef, exp_coef};
                        const float2 _fma_c2_124 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_292 = {exp_coef, exp_coef};
                        const float2 _fma_c2_293 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_123, _fma_c2_124);
                        psum3 = add_f32x2(psum3, make_float2(sv[98], sv[99]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 112))[_lf], _fma_b2_292, _fma_c2_293);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[98], sv_b[99]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 102] = approx_exp2(sv[_le + 102]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_294 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 102], sv_b[_le*2 + 1 + 102]));
                                sv_b[_le*2 + 102] = _exp2_pair_294.x;
                                sv_b[_le*2 + 1 + 102] = _exp2_pair_294.y;
                            } else {
                                sv_b[_le*2 + 102] = approx_exp2(sv_b[_le*2 + 102]);
                                sv_b[_le*2 + 1 + 102] = approx_exp2(sv_b[_le*2 + 1 + 102]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_125 = {exp_coef, exp_coef};
                        const float2 _fma_c2_126 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_295 = {exp_coef, exp_coef};
                        const float2 _fma_c2_296 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 114))[_lf], _fma_b2_125, _fma_c2_126);
                        psum3 = add_f32x2(psum3, make_float2(sv[100], sv[101]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 114))[_lf], _fma_b2_295, _fma_c2_296);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[100], sv_b[101]));
#endif
                        #pragma unroll
#if __CUDA_ARCH__ == 1030
                        for (int _le = 0; _le < 2; _le++) {
                            sv[_le + 104] = approx_exp2(sv[_le + 104]);
#else
                        for (int _le = 0; _le < 1; _le++) {
                            if (1 && _le >= 0) {
                                float2 _exp2_pair_297 = ex2_emulation_f32x2_value(make_float2(sv_b[_le*2 + 104], sv_b[_le*2 + 1 + 104]));
                                sv_b[_le*2 + 104] = _exp2_pair_297.x;
                                sv_b[_le*2 + 1 + 104] = _exp2_pair_297.y;
                            } else {
                                sv_b[_le*2 + 104] = approx_exp2(sv_b[_le*2 + 104]);
                                sv_b[_le*2 + 1 + 104] = approx_exp2(sv_b[_le*2 + 1 + 104]);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_127 = {exp_coef, exp_coef};
                        const float2 _fma_c2_128 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_298 = {exp_coef, exp_coef};
                        const float2 _fma_c2_299 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 116))[_lf], _fma_b2_127, _fma_c2_128);
                        psum3 = add_f32x2(psum3, make_float2(sv[102], sv[103]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 116))[_lf], _fma_b2_298, _fma_c2_299);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[102], sv_b[103]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 106] = approx_exp2(sv[_le + 106]);
#else
                            sv_b[_le + 106] = approx_exp2(sv_b[_le + 106]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_129 = {exp_coef, exp_coef};
                        const float2 _fma_c2_130 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_300 = {exp_coef, exp_coef};
                        const float2 _fma_c2_301 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 118))[_lf], _fma_b2_129, _fma_c2_130);
                        psum3 = add_f32x2(psum3, make_float2(sv[104], sv[105]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 118))[_lf], _fma_b2_300, _fma_c2_301);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[104], sv_b[105]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 108] = approx_exp2(sv[_le + 108]);
#else
                            sv_b[_le + 108] = approx_exp2(sv_b[_le + 108]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_131 = {exp_coef, exp_coef};
                        const float2 _fma_c2_132 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_302 = {exp_coef, exp_coef};
                        const float2 _fma_c2_303 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 120))[_lf], _fma_b2_131, _fma_c2_132);
                        psum3 = add_f32x2(psum3, make_float2(sv[106], sv[107]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 120))[_lf], _fma_b2_302, _fma_c2_303);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[106], sv_b[107]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 110] = approx_exp2(sv[_le + 110]);
#else
                            sv_b[_le + 110] = approx_exp2(sv_b[_le + 110]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_133 = {exp_coef, exp_coef};
                        const float2 _fma_c2_134 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_304 = {exp_coef, exp_coef};
                        const float2 _fma_c2_305 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 122))[_lf], _fma_b2_133, _fma_c2_134);
                        psum3 = add_f32x2(psum3, make_float2(sv[108], sv[109]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 122))[_lf], _fma_b2_304, _fma_c2_305);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[108], sv_b[109]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 112] = approx_exp2(sv[_le + 112]);
#else
                            sv_b[_le + 112] = approx_exp2(sv_b[_le + 112]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_135 = {exp_coef, exp_coef};
                        const float2 _fma_c2_136 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_306 = {exp_coef, exp_coef};
                        const float2 _fma_c2_307 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 124))[_lf], _fma_b2_135, _fma_c2_136);
                        psum3 = add_f32x2(psum3, make_float2(sv[110], sv[111]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 124))[_lf], _fma_b2_306, _fma_c2_307);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[110], sv_b[111]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 114] = approx_exp2(sv[_le + 114]);
#else
                            sv_b[_le + 114] = approx_exp2(sv_b[_le + 114]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_137 = {exp_coef, exp_coef};
                        const float2 _fma_c2_138 = {neg_max_scaled, neg_max_scaled};
#else
                        const float2 _fma_b2_308 = {exp_coef, exp_coef};
                        const float2 _fma_c2_309 = {neg_max_scaled_b, neg_max_scaled_b};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 126))[_lf], _fma_b2_137, _fma_c2_138);
                        psum3 = add_f32x2(psum3, make_float2(sv[112], sv[113]));
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv_b + 126))[_lf], _fma_b2_308, _fma_c2_309);
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[112], sv_b[113]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 116] = approx_exp2(sv[_le + 116]);
#else
                            sv_b[_le + 116] = approx_exp2(sv_b[_le + 116]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        psum3 = add_f32x2(psum3, make_float2(sv[114], sv[115]));
#else
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[114], sv_b[115]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 118] = approx_exp2(sv[_le + 118]);
#else
                            sv_b[_le + 118] = approx_exp2(sv_b[_le + 118]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        psum3 = add_f32x2(psum3, make_float2(sv[116], sv[117]));
#else
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[116], sv_b[117]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 120] = approx_exp2(sv[_le + 120]);
#else
                            sv_b[_le + 120] = approx_exp2(sv_b[_le + 120]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        psum3 = add_f32x2(psum3, make_float2(sv[118], sv[119]));
#else
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[118], sv_b[119]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 122] = approx_exp2(sv[_le + 122]);
#else
                            sv_b[_le + 122] = approx_exp2(sv_b[_le + 122]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        psum3 = add_f32x2(psum3, make_float2(sv[120], sv[121]));
#else
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[120], sv_b[121]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 124] = approx_exp2(sv[_le + 124]);
#else
                            sv_b[_le + 124] = approx_exp2(sv_b[_le + 124]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        psum3 = add_f32x2(psum3, make_float2(sv[122], sv[123]));
#else
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[122], sv_b[123]));
#endif
                        #pragma unroll
                        for (int _le = 0; _le < 2; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 126] = approx_exp2(sv[_le + 126]);
#else
                            sv_b[_le + 126] = approx_exp2(sv_b[_le + 126]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        psum3 = add_f32x2(psum3, make_float2(sv[124], sv[125]));
                        psum3 = add_f32x2(psum3, make_float2(sv[126], sv[127]));
                        unsigned int pk3[8];
#else
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[124], sv_b[125]));
                        psum3b = add_f32x2(psum3b, make_float2(sv_b[126], sv_b[127]));
                        unsigned int pk3b[8];
#endif
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[96]), "f"(sv[97]),
                                                   "f"(sv[98]), "f"(sv[99]));
                            pk3[0] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[96]), "f"(sv_b[97]),
                                                   "f"(sv_b[98]), "f"(sv_b[99]));
                            pk3b[0] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[100]), "f"(sv[101]),
                                                   "f"(sv[102]), "f"(sv[103]));
                            pk3[1] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[100]), "f"(sv_b[101]),
                                                   "f"(sv_b[102]), "f"(sv_b[103]));
                            pk3b[1] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[104]), "f"(sv[105]),
                                                   "f"(sv[106]), "f"(sv[107]));
                            pk3[2] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[104]), "f"(sv_b[105]),
                                                   "f"(sv_b[106]), "f"(sv_b[107]));
                            pk3b[2] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[108]), "f"(sv[109]),
                                                   "f"(sv[110]), "f"(sv[111]));
                            pk3[3] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[108]), "f"(sv_b[109]),
                                                   "f"(sv_b[110]), "f"(sv_b[111]));
                            pk3b[3] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[112]), "f"(sv[113]),
                                                   "f"(sv[114]), "f"(sv[115]));
                            pk3[4] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[112]), "f"(sv_b[113]),
                                                   "f"(sv_b[114]), "f"(sv_b[115]));
                            pk3b[4] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[116]), "f"(sv[117]),
                                                   "f"(sv[118]), "f"(sv[119]));
                            pk3[5] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[116]), "f"(sv_b[117]),
                                                   "f"(sv_b[118]), "f"(sv_b[119]));
                            pk3b[5] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[120]), "f"(sv[121]),
                                                   "f"(sv[122]), "f"(sv[123]));
                            pk3[6] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[120]), "f"(sv_b[121]),
                                                   "f"(sv_b[122]), "f"(sv_b[123]));
                            pk3b[6] = _packed;
#endif
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
#if __CUDA_ARCH__ == 1030
                                : "=r"(_packed) : "f"(sv[124]), "f"(sv[125]),
                                                   "f"(sv[126]), "f"(sv[127]));
                            pk3[7] = _packed;
#else
                                : "=r"(_packed) : "f"(sv_b[124]), "f"(sv_b[125]),
                                                   "f"(sv_b[126]), "f"(sv_b[127]));
                            pk3b[7] = _packed;
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        int p_addr = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
#else
#endif
                        if (two_stages == 1) {
                            mbarrier_wait(s_read_addr + (other_stage) * 8, _phase_s_read);
                            _phase_s_read ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
#if __CUDA_ARCH__ == 1030
                        if (stage == 0 && two_stages == 0 && n_iter >= 1) {
#else
                        if (stage == 0 && two_stages == 0) {
#endif
                            mbarrier_wait(pv0_done_addr, _phase_pv0_done_0);
                            _phase_pv0_done_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
#if __CUDA_ARCH__ == 1030
                        if (stage == 1 && two_stages == 1 && n_iter >= 1 && n_iter == last_n_iter) {
#else
                        if (stage == 1 && two_stages == 1 && n_iter == last_n_iter) {
#endif
                            mbarrier_wait(pv1_done_addr, _phase_pv1_done_0);
                            _phase_pv1_done_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
#if __CUDA_ARCH__ == 1030
                        tmem_st_x8_u32(p_addr, (const uint32_t*)pk0);
                        tmem_st_x8_u32(p_addr + 8, (const uint32_t*)pk1);
#else
                        tmem_st_x8_u32(p_addr_u, (const uint32_t*)pk0b);
                        tmem_st_x8_u32(p_addr_u + 8, (const uint32_t*)pk1b);
                        tmem_st_x8_u32(p_addr_u + 16, (const uint32_t*)pk2b);
                        tmem_st_x8_u32(p_addr_u + 24, (const uint32_t*)pk3b);
#endif
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
#if __CUDA_ARCH__ == 1030
                        tmem_st_x8_u32(p_addr + 16, (const uint32_t*)pk2);
                        tmem_st_x8_u32(p_addr + 24, (const uint32_t*)pk3);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        float2 sum01 = add_f32x2(psum0, psum1);
                        float2 sum23 = add_f32x2(psum2, psum3);
                        float2 total_sum = add_f32x2(sum01, sum23);
                        row_sum = __fmaf_rn(row_sum, acc_scale, total_sum.x) + total_sum.y;
#else
                        float2 sum01_b = add_f32x2(psum0b, psum1b);
                        float2 sum23_b = add_f32x2(psum2b, psum3b);
                        float2 total_sum_b = add_f32x2(sum01_b, sum23_b);
                        row_sum = __fmaf_rn(row_sum, acc_scale_b, total_sum_b.x) + total_sum_b.y;
#endif
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
#if __CUDA_ARCH__ == 1030
            unsigned int _phase_p_full_2_0 = 0;
#else
#endif
            unsigned int _phase_p_full_1 = 0;
#if __CUDA_ARCH__ == 1030
            unsigned int _phase_p_full_2_1 = 0;
#else
#endif
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
                    int _mma_a_lo_0 = ((smem_q0_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0) : "memory");
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
#else
                        mbarrier_wait_hint(scale_empty_addr + 8, _phase_scale_empty_1, 1000000);
#endif
                        _phase_scale_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = ((smem_q1_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_1 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (128))), "r"(0) : "memory");
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
                        int _mma_a_lo_2 = ((smem_q0_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_2 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_scores), "r"(0) : "memory");
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
#else
                        mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 1000000);
#endif
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
#if __CUDA_ARCH__ == 1030
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                        mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                        _phase_p_full_2_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
#else
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
#endif
                    "mov.b64 db, {blo, dhi};\n\t"
#if __CUDA_ARCH__ == 1030
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
#else
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
#endif
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_4), "r"(tmem_probs_0), "r"(1) : "memory");
#else
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
#endif
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                            mbarrier_wait(scale_empty_addr + 8, _phase_scale_empty_1);
#else
                            mbarrier_wait_hint(scale_empty_addr + 8, _phase_scale_empty_1, 1000000);
#endif
                            _phase_scale_empty_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = ((smem_q1_addr) >> 4) & 0x3FFF;
#if __CUDA_ARCH__ == 1030
                            int _mma_b_lo_5 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
#else
                            int _mma_b_lo_4 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
#endif
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
#if __CUDA_ARCH__ == 1030
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_5), "r"((tmem_scores + (128))), "r"(0) : "memory");
#else
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_4), "r"((tmem_scores + (128))), "r"(0) : "memory");
#endif
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
#if __CUDA_ARCH__ == 1030
                            int _mma_b_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
#else
                            int _mma_b_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
#endif
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
#if __CUDA_ARCH__ == 1030
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_6), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                            mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                            _phase_p_full_2_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_7 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
#else
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
#endif
                    "mov.b64 db, {blo, dhi};\n\t"
#if __CUDA_ARCH__ == 1030
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
#else
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
#endif
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_7), "r"(tmem_probs_1), "r"(1) : "memory");
#else
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
#endif
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
#if __CUDA_ARCH__ == 1030
                    int _mma_b_lo_8 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
#else
                    int _mma_b_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
#endif
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
#if __CUDA_ARCH__ == 1030
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_9 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
#else
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
#endif
                    "mov.b64 db, {blo, dhi};\n\t"
#if __CUDA_ARCH__ == 1030
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
#else
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
#endif
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_9), "r"(tmem_probs_0), "r"(1) : "memory");
#else
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_6), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
#endif
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    if (two_stages_2 == 1) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
#else
                        mbarrier_wait_hint(p_full_addr + 8, _phase_p_full_1, 1000000);
#endif
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        int _mma_b_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
#else
                        int _mma_b_lo_7 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
#endif
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
#if __CUDA_ARCH__ == 1030
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
#else
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
#endif
                    "mov.b64 db, {blo, dhi};\n\t"
#if __CUDA_ARCH__ == 1030
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
#else
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
#endif
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
#if __CUDA_ARCH__ == 1030
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_11), "r"(tmem_probs_1), "r"(1) : "memory");
#else
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_7), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
#endif
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
