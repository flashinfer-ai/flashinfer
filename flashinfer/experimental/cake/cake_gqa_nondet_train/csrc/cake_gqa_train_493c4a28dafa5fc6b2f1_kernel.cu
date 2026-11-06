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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_S_OFFSET 0
#define TMEM_P0_OFFSET 64
#define TMEM_P1_OFFSET 192
#define TMEM_O_HI_OFFSET 256
#define TMEM_O_LO_OFFSET 384
#define NUM_KV_PIPE_STAGES 5
#define NUM_Q_PIPE_STAGES 1
#define NUM_S_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 2
#define NUM_OF_PIPE_STAGES 1
#define NUM_SC_PIPE_STAGES 1
#define NUM_ST_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 2
#define SMEM_S_SCALE_OFF 230400
#define SMEM_S_SCALE_STAGE_BYTES 512
#define SMEM_S_SCALE_STRIDE 512
#define SMEM_S_STATS_OFF 230912
#define SMEM_S_STATS_STAGE_BYTES 1024
#define SMEM_S_STATS_STRIDE 1024
#define SMEM_SMEM_Q_HI_OFF 1024
#define SMEM_SMEM_Q_HI_STAGE_BYTES 32768
#define SMEM_SMEM_Q_HI_STRIDE 32768
#define SMEM_SMEM_Q_LO_OFF 33792
#define SMEM_SMEM_Q_LO_STAGE_BYTES 32768
#define SMEM_SMEM_Q_LO_STRIDE 32768
#define SMEM_SMEM_K_HI_OFF 66560
#define SMEM_SMEM_K_HI_STAGE_BYTES 16384
#define SMEM_SMEM_K_HI_STRIDE 16384
#define SMEM_SMEM_K_LO_OFF 148480
#define SMEM_SMEM_K_LO_STAGE_BYTES 16384
#define SMEM_SMEM_K_LO_STRIDE 16384
#define SMEM_SMEM_V_HI_OFF 66560
#define SMEM_SMEM_V_HI_STAGE_BYTES 16384
#define SMEM_SMEM_V_HI_STRIDE 16384
#define SMEM_SMEM_V_LO_OFF 148480
#define SMEM_SMEM_V_LO_STAGE_BYTES 16384
#define SMEM_SMEM_V_LO_STRIDE 16384
#define SMEM_CELLS_OFF 231936
#define SMEM_CELLS_STAGE_BYTES 8
#define SMEM_CELLS_STRIDE 8
#define SMEM_TOTAL 232064
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





__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.rn.ftz.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
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


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_gqa_train_493c4a28dafa5fc6b2f1(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, unsigned int* __restrict__ tile_counter, int num_segments, int num_heads_q, int group, int total_q, float softmax_scale_log2)
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
    #define o_full_addr (mbar_base + 144)
    #define o_free_addr (mbar_base + 160)
    #define scale_full_addr (mbar_base + 168)
    #define scale_empty_addr (mbar_base + 176)
    #define stats_full_addr (mbar_base + 184)
    #define stats_empty_addr (mbar_base + 192)
    #define work_full_addr (mbar_base + 200)
    #define work_empty_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* s_scale = reinterpret_cast<float*>(smem_raw + 230400);
    const int s_scale_addr = smem + 230400;
    float* s_stats = reinterpret_cast<float*>(smem_raw + 230912);
    const int s_stats_addr = smem + 230912;
    __nv_bfloat16* smem_q_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_hi_addr = smem + 1024;
    __nv_bfloat16* smem_q_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_q_lo_addr = smem + 33792;
    __nv_bfloat16* smem_k_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_k_hi_addr = smem + 66560;
    __nv_bfloat16* smem_k_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_k_lo_addr = smem + 148480;
    __nv_bfloat16* smem_v_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v_hi_addr = smem + 66560;
    __nv_bfloat16* smem_v_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_v_lo_addr = smem + 148480;
    unsigned int* cells = reinterpret_cast<unsigned int*>(smem_raw + 231936);
    const int cells_addr = smem + 231936;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv_pipe' ---
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
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 112, 512);
            mbarrier_init(smem + 120, 512);
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            // --- pipeline 'o_pipe' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'of_pipe' ---
            // o_free: 1 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            // --- pipeline 'sc_pipe' ---
            // scale_full: 1 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            // scale_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // --- pipeline 'st_pipe' ---
            // stats_full: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            // stats_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // work_empty: 2 barriers, init_count=640
            mbarrier_init(smem + 216, 640);
            mbarrier_init(smem + 224, 640);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_s = taddr;
    const int tmem_p0 = taddr + 64;
    const int tmem_p1 = taddr + 192;
    const int tmem_o_hi = taddr + 256;
    const int tmem_o_lo = taddr + 384;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        { // softmax_main
            unsigned int wk = 0;
            unsigned int sq = 0;
            unsigned int pq = 0;
            const int tr = warp % 4 * 32 + lane;
            unsigned int head_tiles = 0;
            #pragma unroll 1
            for (int s = 0; s < num_segments; s++) {
                head_tiles = head_tiles + ((unsigned int)(cu_seqlens_q[s + 1] - cu_seqlens_q[s] + 255) >> 8);
            }
            int wmax = 0;
            #pragma unroll 1
            for (int s_1 = 0; s_1 < num_segments; s_1++) {
                int q_len = cu_seqlens_q[s_1 + 1] - cu_seqlens_q[s_1];
                int k_len = cu_seqlens_k[s_1 + 1] - cu_seqlens_k[s_1];
                int row0 = ((q_len + 255 >> 8) - 1) * 256;
                int diag = k_len - q_len;
                int n_k = k_len + 127 >> 7;
                int n_c = row0 + 256 + diag + 127 >> 7;
                int n_seen = ((n_c < n_k) ? n_c : n_k);
                int n_blocks = ((n_seen > 0) ? n_seen : 0);
                int first_seen = row0 + diag + 1;
                int n_first = ((first_seen > 0) ? first_seen >> 7 : 0);
                int k_full = k_len >> 7;
                int n_kf = ((n_first < k_full) ? n_first : k_full);
                int n_full = ((n_kf < n_blocks) ? n_kf : n_blocks);
                int n_masked = n_blocks - n_full;
                int w = n_blocks;
                wmax = ((w > wmax) ? w : wmax);
            }
            unsigned int t0 = 0;
            unsigned int t1 = 0;
            unsigned int t2 = 0;
            #pragma unroll 1
            for (int s_2 = 0; s_2 < num_segments; s_2++) {
                unsigned int tiles = (unsigned int)(cu_seqlens_q[s_2 + 1] - cu_seqlens_q[s_2] + 255) >> 8;
                int q_len_1 = cu_seqlens_q[s_2 + 1] - cu_seqlens_q[s_2];
                int k_len_1 = cu_seqlens_k[s_2 + 1] - cu_seqlens_k[s_2];
                int row0_1 = ((q_len_1 + 255 >> 8) - 1) * 256;
                int diag_1 = k_len_1 - q_len_1;
                int n_k_1 = k_len_1 + 127 >> 7;
                int n_c_1 = row0_1 + 256 + diag_1 + 127 >> 7;
                int n_seen_1 = ((n_c_1 < n_k_1) ? n_c_1 : n_k_1);
                int n_blocks_1 = ((n_seen_1 > 0) ? n_seen_1 : 0);
                int first_seen_1 = row0_1 + diag_1 + 1;
                int n_first_1 = ((first_seen_1 > 0) ? first_seen_1 >> 7 : 0);
                int k_full_1 = k_len_1 >> 7;
                int n_kf_1 = ((n_first_1 < k_full_1) ? n_first_1 : k_full_1);
                int n_full_1 = ((n_kf_1 < n_blocks_1) ? n_kf_1 : n_blocks_1);
                int n_masked_1 = n_blocks_1 - n_full_1;
                int c = ((2 * n_blocks_1 >= wmax) ? 0 : ((4 * n_blocks_1 >= wmax) ? 1 : ((8 * n_blocks_1 >= wmax) ? 2 : 3)));
                t0 = t0 + ((c == 0) ? tiles : 0);
                t1 = t1 + ((c == 1) ? tiles : 0);
                t2 = t2 + ((c == 2) ? tiles : 0);
            }
            unsigned int total = (unsigned int)num_heads_q * head_tiles;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_scale_empty = 1;
            unsigned int _phase_stats_empty = 1;
            #pragma unroll 1
            for (unsigned int it = 0; it < (unsigned int)num_heads_q * ((unsigned int)(total_q + 255 >> 8) + (unsigned int)num_segments) + 1; it++) {
                mbarrier_wait_cluster_hint(work_full_addr + (wk) * 8, _phase_work_full, 10000000);
                unsigned int ticket = cells[wk];
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wk) * 8), "r"(0) : "memory");
                wk += 1;
                if (wk == 2) { wk = 0; _phase_work_full ^= 1; }
                if (ticket >= total) {
                    break;
                }
                unsigned int hq = (unsigned int)num_heads_q;
                unsigned int p1 = hq * t0;
                unsigned int p2 = p1 + hq * t1;
                unsigned int p3 = p2 + hq * t2;
                int c_1 = ((ticket < p1) ? 0 : ((ticket < p2) ? 1 : ((ticket < p3) ? 2 : 3)));
                unsigned int base = ((c_1 == 0) ? 0 : ((c_1 == 1) ? p1 : ((c_1 == 2) ? p2 : p3)));
                unsigned int tc = ((c_1 == 0) ? t0 : ((c_1 == 1) ? t1 : ((c_1 == 2) ? t2 : head_tiles - t0 - t1 - t2)));
                unsigned int r = ticket - base;
                unsigned int per_kv = tc * (unsigned int)group;
                unsigned int kvh_u = r / per_kv;
                unsigned int rem = r - kvh_u * per_kv;
                unsigned int j = rem / (unsigned int)group;
                unsigned int head_u = kvh_u * (unsigned int)group + (rem - j * (unsigned int)group);
                int seg = 0;
                unsigned int before = 0;
                unsigned int ends = 0;
                #pragma unroll 1
                for (int s_3 = 0; s_3 < num_segments; s_3++) {
                    int q_len_2 = cu_seqlens_q[s_3 + 1] - cu_seqlens_q[s_3];
                    int k_len_2 = cu_seqlens_k[s_3 + 1] - cu_seqlens_k[s_3];
                    int row0_2 = ((q_len_2 + 255 >> 8) - 1) * 256;
                    int diag_2 = k_len_2 - q_len_2;
                    int n_k_2 = k_len_2 + 127 >> 7;
                    int n_c_2 = row0_2 + 256 + diag_2 + 127 >> 7;
                    int n_seen_2 = ((n_c_2 < n_k_2) ? n_c_2 : n_k_2);
                    int n_blocks_2 = ((n_seen_2 > 0) ? n_seen_2 : 0);
                    int first_seen_2 = row0_2 + diag_2 + 1;
                    int n_first_2 = ((first_seen_2 > 0) ? first_seen_2 >> 7 : 0);
                    int k_full_2 = k_len_2 >> 7;
                    int n_kf_2 = ((n_first_2 < k_full_2) ? n_first_2 : k_full_2);
                    int n_full_2 = ((n_kf_2 < n_blocks_2) ? n_kf_2 : n_blocks_2);
                    int n_masked_2 = n_blocks_2 - n_full_2;
                    bool in_class = ((2 * n_blocks_2 >= wmax) ? 0 : ((4 * n_blocks_2 >= wmax) ? 1 : ((8 * n_blocks_2 >= wmax) ? 2 : 3))) == c_1;
                    ends = ends + ((in_class) ? (unsigned int)(cu_seqlens_q[s_3 + 1] - cu_seqlens_q[s_3] + 255) >> 8 : 0);
                    seg = seg + ((ends <= j) ? 1 : 0);
                    before = ((ends <= j) ? ends : before);
                }
                unsigned int m_rel = j - before;
                int q0 = cu_seqlens_q[seg];
                int q_len_3 = cu_seqlens_q[seg + 1] - q0;
                int k0 = cu_seqlens_k[seg];
                int k_len_3 = cu_seqlens_k[seg + 1] - k0;
                int m = (int)(((unsigned int)(q_len_3 + 255) >> 8) - 1 - m_rel);
                int head = (int)head_u;
                int kvh = (int)kvh_u;
                int row0_3 = m * 256;
                int diag_3 = k_len_3 - q_len_3;
                int n_k_3 = k_len_3 + 127 >> 7;
                int n_c_3 = row0_3 + 256 + diag_3 + 127 >> 7;
                int n_seen_3 = ((n_c_3 < n_k_3) ? n_c_3 : n_k_3);
                int n_blocks_3 = ((n_seen_3 > 0) ? n_seen_3 : 0);
                int first_seen_3 = row0_3 + diag_3 + 1;
                int n_first_3 = ((first_seen_3 > 0) ? first_seen_3 >> 7 : 0);
                int k_full_3 = k_len_3 >> 7;
                int n_kf_3 = ((n_first_3 < k_full_3) ? n_first_3 : k_full_3);
                int n_full_3 = ((n_kf_3 < n_blocks_3) ? n_kf_3 : n_blocks_3);
                int n_masked_3 = n_blocks_3 - n_full_3;
                int row = m * 256 + cta_rank * 128 + tr;
                int seen = row + (k_len_3 - q_len_3) + 1;
                int vis = ((seen < k_len_3) ? seen : k_len_3);
                float row_max = -CUDART_INF_F;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (int i = 0; i < n_masked_3; i++) {
                    mbarrier_wait(s_full_addr + (sq) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int s_addr = taddr + (unsigned int)(sq * 128) + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -CUDART_INF_F;
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
                    int lim = vis - (n_blocks_3 - 1 - i) * 128;
                    uint32_t _thresh_mask_0;
                    {
                        int _lim_1 = lim;
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
                        int _lim_3 = lim - 32;
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
                        int _lim_5 = lim - 64;
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
                        int _lim_7 = lim - 96;
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
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                    float new_max_scaled = ((new_max == -CUDART_INF_F) ? 0.0f : new_max) * softmax_scale_log2;
                    float log2_ratio = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float kept_max = row_max;
                    float acc_scale = 1.0f;
                    if (log2_ratio >= -8.0f) {
                        new_max_scaled = ((row_max == -CUDART_INF_F) ? 0.0f : row_max) * softmax_scale_log2;
                    } else {
                        kept_max = new_max;
                        if (row_max > -CUDART_INF_F) {
                            float _exp2_10 = approx_exp2(log2_ratio);
                            acc_scale = _exp2_10;
                        }
                    }
                    mbarrier_wait(scale_empty_addr, _phase_scale_empty);
                    s_scale[(int)0 * 128 + tr] = acc_scale;
                    mbarrier_arrive(scale_full_addr);
                    _phase_scale_empty ^= 1;
                    const float2 _fma_b2_11 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_12 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_lf], _fma_b2_11, _fma_c2_12);
                    unsigned int p_addr = s_addr + 64;
                    unsigned int pk_0[16];
                    softmax_frag_exp2_cast(&sv[0], pk_0, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 0), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[15])));
                    unsigned int pk_1[16];
                    softmax_frag_exp2_cast(&sv[32], pk_1, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[15])));
                    unsigned int pk_2[16];
                    softmax_frag_exp2_cast(&sv[64], pk_2, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (pq) * 8) & 0xFEFFFFFF) : "memory");
                    unsigned int pk_3[16];
                    softmax_frag_exp2_cast(&sv[96], pk_3, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 48), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (pq) * 8) & 0xFEFFFFFF) : "memory");
                    pq += 1;
                    if (pq == 2) { pq = 0; }
                    sq += 1;
                    if (sq == 2) { sq = 0; _phase_s_full ^= 1; }
                    row_max = kept_max;
                    float2 _reg_reduce_sum2_14 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&sv[0], &_reg_reduce_sum2_14);
                    softmax_block_sum(&sv[32], &_reg_reduce_sum2_14);
                    softmax_block_sum(&sv[64], &_reg_reduce_sum2_14);
                    softmax_block_sum(&sv[96], &_reg_reduce_sum2_14);
                    float _reg_array_reduce_13 = _reg_reduce_sum2_14.x + _reg_reduce_sum2_14.y;
                    row_sum = __fmaf_rn(row_sum, acc_scale, _reg_array_reduce_13);
                }
                #pragma unroll 1
                for (int i_1 = n_masked_3; i_1 < n_blocks_3; i_1++) {
                    mbarrier_wait(s_full_addr + (sq) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int s_addr_1 = taddr + (unsigned int)(sq * 128) + (unsigned int)(warp % 4 * 32 << 16);
                    float sv_1[128];
                    float tile_max_1 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                    float lo_max = -CUDART_INF_F;
                    float hi_max = -CUDART_INF_F;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
#endif
                    asm volatile(
#if __CUDA_ARCH__ == 1030
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31]), "=f"(sv_1[32]), "=f"(sv_1[33]), "=f"(sv_1[34]), "=f"(sv_1[35]), "=f"(sv_1[36]), "=f"(sv_1[37]), "=f"(sv_1[38]), "=f"(sv_1[39]), "=f"(sv_1[40]), "=f"(sv_1[41]), "=f"(sv_1[42]), "=f"(sv_1[43]), "=f"(sv_1[44]), "=f"(sv_1[45]), "=f"(sv_1[46]), "=f"(sv_1[47]), "=f"(sv_1[48]), "=f"(sv_1[49]), "=f"(sv_1[50]), "=f"(sv_1[51]), "=f"(sv_1[52]), "=f"(sv_1[53]), "=f"(sv_1[54]), "=f"(sv_1[55]), "=f"(sv_1[56]), "=f"(sv_1[57]), "=f"(sv_1[58]), "=f"(sv_1[59]), "=f"(sv_1[60]), "=f"(sv_1[61]), "=f"(sv_1[62]), "=f"(sv_1[63]), "=f"(lo_max)
#else
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31])
#endif
                        : "r"(s_addr_1));
#if __CUDA_ARCH__ == 1030
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
#endif
                    asm volatile(
#if __CUDA_ARCH__ == 1030
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv_1[64]), "=f"(sv_1[65]), "=f"(sv_1[66]), "=f"(sv_1[67]), "=f"(sv_1[68]), "=f"(sv_1[69]), "=f"(sv_1[70]), "=f"(sv_1[71]), "=f"(sv_1[72]), "=f"(sv_1[73]), "=f"(sv_1[74]), "=f"(sv_1[75]), "=f"(sv_1[76]), "=f"(sv_1[77]), "=f"(sv_1[78]), "=f"(sv_1[79]), "=f"(sv_1[80]), "=f"(sv_1[81]), "=f"(sv_1[82]), "=f"(sv_1[83]), "=f"(sv_1[84]), "=f"(sv_1[85]), "=f"(sv_1[86]), "=f"(sv_1[87]), "=f"(sv_1[88]), "=f"(sv_1[89]), "=f"(sv_1[90]), "=f"(sv_1[91]), "=f"(sv_1[92]), "=f"(sv_1[93]), "=f"(sv_1[94]), "=f"(sv_1[95]), "=f"(sv_1[96]), "=f"(sv_1[97]), "=f"(sv_1[98]), "=f"(sv_1[99]), "=f"(sv_1[100]), "=f"(sv_1[101]), "=f"(sv_1[102]), "=f"(sv_1[103]), "=f"(sv_1[104]), "=f"(sv_1[105]), "=f"(sv_1[106]), "=f"(sv_1[107]), "=f"(sv_1[108]), "=f"(sv_1[109]), "=f"(sv_1[110]), "=f"(sv_1[111]), "=f"(sv_1[112]), "=f"(sv_1[113]), "=f"(sv_1[114]), "=f"(sv_1[115]), "=f"(sv_1[116]), "=f"(sv_1[117]), "=f"(sv_1[118]), "=f"(sv_1[119]), "=f"(sv_1[120]), "=f"(sv_1[121]), "=f"(sv_1[122]), "=f"(sv_1[123]), "=f"(sv_1[124]), "=f"(sv_1[125]), "=f"(sv_1[126]), "=f"(sv_1[127]), "=f"(hi_max)
#else
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_1[32]), "=f"(sv_1[33]), "=f"(sv_1[34]), "=f"(sv_1[35]), "=f"(sv_1[36]), "=f"(sv_1[37]), "=f"(sv_1[38]), "=f"(sv_1[39]), "=f"(sv_1[40]), "=f"(sv_1[41]), "=f"(sv_1[42]), "=f"(sv_1[43]), "=f"(sv_1[44]), "=f"(sv_1[45]), "=f"(sv_1[46]), "=f"(sv_1[47]), "=f"(sv_1[48]), "=f"(sv_1[49]), "=f"(sv_1[50]), "=f"(sv_1[51]), "=f"(sv_1[52]), "=f"(sv_1[53]), "=f"(sv_1[54]), "=f"(sv_1[55]), "=f"(sv_1[56]), "=f"(sv_1[57]), "=f"(sv_1[58]), "=f"(sv_1[59]), "=f"(sv_1[60]), "=f"(sv_1[61]), "=f"(sv_1[62]), "=f"(sv_1[63])
                        : "r"(s_addr_1 + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_1[64]), "=f"(sv_1[65]), "=f"(sv_1[66]), "=f"(sv_1[67]), "=f"(sv_1[68]), "=f"(sv_1[69]), "=f"(sv_1[70]), "=f"(sv_1[71]), "=f"(sv_1[72]), "=f"(sv_1[73]), "=f"(sv_1[74]), "=f"(sv_1[75]), "=f"(sv_1[76]), "=f"(sv_1[77]), "=f"(sv_1[78]), "=f"(sv_1[79]), "=f"(sv_1[80]), "=f"(sv_1[81]), "=f"(sv_1[82]), "=f"(sv_1[83]), "=f"(sv_1[84]), "=f"(sv_1[85]), "=f"(sv_1[86]), "=f"(sv_1[87]), "=f"(sv_1[88]), "=f"(sv_1[89]), "=f"(sv_1[90]), "=f"(sv_1[91]), "=f"(sv_1[92]), "=f"(sv_1[93]), "=f"(sv_1[94]), "=f"(sv_1[95])
#endif
                        : "r"(s_addr_1 + 64));
#if __CUDA_ARCH__ == 1030
#else
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv_1[96]), "=f"(sv_1[97]), "=f"(sv_1[98]), "=f"(sv_1[99]), "=f"(sv_1[100]), "=f"(sv_1[101]), "=f"(sv_1[102]), "=f"(sv_1[103]), "=f"(sv_1[104]), "=f"(sv_1[105]), "=f"(sv_1[106]), "=f"(sv_1[107]), "=f"(sv_1[108]), "=f"(sv_1[109]), "=f"(sv_1[110]), "=f"(sv_1[111]), "=f"(sv_1[112]), "=f"(sv_1[113]), "=f"(sv_1[114]), "=f"(sv_1[115]), "=f"(sv_1[116]), "=f"(sv_1[117]), "=f"(sv_1[118]), "=f"(sv_1[119]), "=f"(sv_1[120]), "=f"(sv_1[121]), "=f"(sv_1[122]), "=f"(sv_1[123]), "=f"(sv_1[124]), "=f"(sv_1[125]), "=f"(sv_1[126]), "=f"(sv_1[127])
                        : "r"(s_addr_1 + 96));
#endif
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                    tile_max_1 = max_noftz(lo_max, hi_max);
#else
                    float2 _reg_reduce_max2_16 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&sv_1[0], _reg_reduce_max2_16);
                    row_max_x32_accum(&sv_1[32], _reg_reduce_max2_16);
                    row_max_x32_accum(&sv_1[64], _reg_reduce_max2_16);
                    row_max_x32_accum(&sv_1[96], _reg_reduce_max2_16);
                    float _reg_array_reduce_15 = row_max_reduce(_reg_reduce_max2_16);
                    tile_max_1 = _reg_array_reduce_15;
#endif
                    float new_max_1;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max_1) : "f"(tile_max_1), "f"(row_max));
                    float new_max_scaled_1 = ((new_max_1 == -CUDART_INF_F) ? 0.0f : new_max_1) * softmax_scale_log2;
                    float log2_ratio_1 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled_1);
                    float kept_max_1 = row_max;
                    float acc_scale_1 = 1.0f;
                    if (log2_ratio_1 >= -8.0f) {
                        new_max_scaled_1 = ((row_max == -CUDART_INF_F) ? 0.0f : row_max) * softmax_scale_log2;
                    } else {
                        kept_max_1 = new_max_1;
                        if (row_max > -CUDART_INF_F) {
#if __CUDA_ARCH__ == 1030
                            float _exp2_15 = approx_exp2(log2_ratio_1);
                            acc_scale_1 = _exp2_15;
#else
                            float _exp2_17 = approx_exp2(log2_ratio_1);
                            acc_scale_1 = _exp2_17;
#endif
                        }
                    }
                    mbarrier_wait(scale_empty_addr, _phase_scale_empty);
                    s_scale[(int)0 * 128 + tr] = acc_scale_1;
                    mbarrier_arrive(scale_full_addr);
                    _phase_scale_empty ^= 1;
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_16 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_17 = {-new_max_scaled_1, -new_max_scaled_1};
#else
                    const float2 _fma_b2_18 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_19 = {-new_max_scaled_1, -new_max_scaled_1};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv_1)[_lf], _fma_b2_16, _fma_c2_17);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv_1)[_lf], _fma_b2_18, _fma_c2_19);
#endif
                    unsigned int p_addr_1 = s_addr_1 + 64;
                    unsigned int pk_4[16];
                    softmax_frag_exp2_cast(&sv_1[0], pk_4, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_1 + 0), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[15])));
                    unsigned int pk_5[16];
                    softmax_frag_exp2_cast(&sv_1[32], pk_5, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[15])));
                    unsigned int pk_6[16];
                    softmax_frag_exp2_cast(&sv_1[64], pk_6, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (pq) * 8) & 0xFEFFFFFF) : "memory");
                    unsigned int pk_7[16];
                    softmax_frag_exp2_cast(&sv_1[96], pk_7, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr_1 + 48), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[3])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[4])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[5])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[6])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[7])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[8])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[9])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[10])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[11])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[12])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[13])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[14])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (pq) * 8) & 0xFEFFFFFF) : "memory");
                    pq += 1;
                    if (pq == 2) { pq = 0; }
                    sq += 1;
                    if (sq == 2) { sq = 0; _phase_s_full ^= 1; }
                    row_max = kept_max_1;
#if __CUDA_ARCH__ == 1030
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&sv_1[0], &_reg_reduce_sum2_19);
                    softmax_block_sum(&sv_1[32], &_reg_reduce_sum2_19);
                    softmax_block_sum(&sv_1[64], &_reg_reduce_sum2_19);
                    softmax_block_sum(&sv_1[96], &_reg_reduce_sum2_19);
                    float _reg_array_reduce_18 = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
                    row_sum = __fmaf_rn(row_sum, acc_scale_1, _reg_array_reduce_18);
#else
                    float2 _reg_reduce_sum2_21 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&sv_1[0], &_reg_reduce_sum2_21);
                    softmax_block_sum(&sv_1[32], &_reg_reduce_sum2_21);
                    softmax_block_sum(&sv_1[64], &_reg_reduce_sum2_21);
                    softmax_block_sum(&sv_1[96], &_reg_reduce_sum2_21);
                    float _reg_array_reduce_20 = _reg_reduce_sum2_21.x + _reg_reduce_sum2_21.y;
                    row_sum = __fmaf_rn(row_sum, acc_scale_1, _reg_array_reduce_20);
#endif
                }
                mbarrier_wait(stats_empty_addr, _phase_stats_empty);
                s_stats[tr] = row_sum;
                s_stats[128 + tr] = row_max;
                mbarrier_arrive(stats_full_addr);
                _phase_stats_empty ^= 1;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 160;");
        { // correction_main
            unsigned int wk_1 = 0;
            unsigned int pq_1 = 0;
            unsigned int oq = 0;
            const int tr_1 = warp % 4 * 32 + lane;
            unsigned int head_tiles_1 = 0;
            #pragma unroll 1
            for (int s_4 = 0; s_4 < num_segments; s_4++) {
                head_tiles_1 = head_tiles_1 + ((unsigned int)(cu_seqlens_q[s_4 + 1] - cu_seqlens_q[s_4] + 255) >> 8);
            }
            int wmax_1 = 0;
            #pragma unroll 1
            for (int s_5 = 0; s_5 < num_segments; s_5++) {
                int q_len_4 = cu_seqlens_q[s_5 + 1] - cu_seqlens_q[s_5];
                int k_len_4 = cu_seqlens_k[s_5 + 1] - cu_seqlens_k[s_5];
                int row0_4 = ((q_len_4 + 255 >> 8) - 1) * 256;
                int diag_4 = k_len_4 - q_len_4;
                int n_k_4 = k_len_4 + 127 >> 7;
                int n_c_4 = row0_4 + 256 + diag_4 + 127 >> 7;
                int n_seen_4 = ((n_c_4 < n_k_4) ? n_c_4 : n_k_4);
                int n_blocks_4 = ((n_seen_4 > 0) ? n_seen_4 : 0);
                int first_seen_4 = row0_4 + diag_4 + 1;
                int n_first_4 = ((first_seen_4 > 0) ? first_seen_4 >> 7 : 0);
                int k_full_4 = k_len_4 >> 7;
                int n_kf_4 = ((n_first_4 < k_full_4) ? n_first_4 : k_full_4);
                int n_full_4 = ((n_kf_4 < n_blocks_4) ? n_kf_4 : n_blocks_4);
                int n_masked_4 = n_blocks_4 - n_full_4;
                int w_1 = n_blocks_4;
                wmax_1 = ((w_1 > wmax_1) ? w_1 : wmax_1);
            }
            unsigned int t0_1 = 0;
            unsigned int t1_1 = 0;
            unsigned int t2_1 = 0;
            #pragma unroll 1
            for (int s_6 = 0; s_6 < num_segments; s_6++) {
                unsigned int tiles_1 = (unsigned int)(cu_seqlens_q[s_6 + 1] - cu_seqlens_q[s_6] + 255) >> 8;
                int q_len_5 = cu_seqlens_q[s_6 + 1] - cu_seqlens_q[s_6];
                int k_len_5 = cu_seqlens_k[s_6 + 1] - cu_seqlens_k[s_6];
                int row0_5 = ((q_len_5 + 255 >> 8) - 1) * 256;
                int diag_5 = k_len_5 - q_len_5;
                int n_k_5 = k_len_5 + 127 >> 7;
                int n_c_5 = row0_5 + 256 + diag_5 + 127 >> 7;
                int n_seen_5 = ((n_c_5 < n_k_5) ? n_c_5 : n_k_5);
                int n_blocks_5 = ((n_seen_5 > 0) ? n_seen_5 : 0);
                int first_seen_5 = row0_5 + diag_5 + 1;
                int n_first_5 = ((first_seen_5 > 0) ? first_seen_5 >> 7 : 0);
                int k_full_5 = k_len_5 >> 7;
                int n_kf_5 = ((n_first_5 < k_full_5) ? n_first_5 : k_full_5);
                int n_full_5 = ((n_kf_5 < n_blocks_5) ? n_kf_5 : n_blocks_5);
                int n_masked_5 = n_blocks_5 - n_full_5;
                int c_2 = ((2 * n_blocks_5 >= wmax_1) ? 0 : ((4 * n_blocks_5 >= wmax_1) ? 1 : ((8 * n_blocks_5 >= wmax_1) ? 2 : 3)));
                t0_1 = t0_1 + ((c_2 == 0) ? tiles_1 : 0);
                t1_1 = t1_1 + ((c_2 == 1) ? tiles_1 : 0);
                t2_1 = t2_1 + ((c_2 == 2) ? tiles_1 : 0);
            }
            unsigned int total_1 = (unsigned int)num_heads_q * head_tiles_1;
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((o_free_addr) & 0xFEFFFFFF) : "memory");
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_scale_full = 0;
            unsigned int _phase_o_full = 0;
            unsigned int _phase_stats_full = 0;
            #pragma unroll 1
            for (unsigned int it_1 = 0; it_1 < (unsigned int)num_heads_q * ((unsigned int)(total_q + 255 >> 8) + (unsigned int)num_segments) + 1; it_1++) {
                mbarrier_wait_cluster_hint(work_full_addr + (wk_1) * 8, _phase_work_full_1, 10000000);
                unsigned int ticket_1 = cells[wk_1];
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wk_1) * 8), "r"(0) : "memory");
                wk_1 += 1;
                if (wk_1 == 2) { wk_1 = 0; _phase_work_full_1 ^= 1; }
                if (ticket_1 >= total_1) {
                    break;
                }
                unsigned int hq_1 = (unsigned int)num_heads_q;
                unsigned int p1_1 = hq_1 * t0_1;
                unsigned int p2_1 = p1_1 + hq_1 * t1_1;
                unsigned int p3_1 = p2_1 + hq_1 * t2_1;
                int c_3 = ((ticket_1 < p1_1) ? 0 : ((ticket_1 < p2_1) ? 1 : ((ticket_1 < p3_1) ? 2 : 3)));
                unsigned int base_1 = ((c_3 == 0) ? 0 : ((c_3 == 1) ? p1_1 : ((c_3 == 2) ? p2_1 : p3_1)));
                unsigned int tc_1 = ((c_3 == 0) ? t0_1 : ((c_3 == 1) ? t1_1 : ((c_3 == 2) ? t2_1 : head_tiles_1 - t0_1 - t1_1 - t2_1)));
                unsigned int r_1 = ticket_1 - base_1;
                unsigned int per_kv_1 = tc_1 * (unsigned int)group;
                unsigned int kvh_u_1 = r_1 / per_kv_1;
                unsigned int rem_1 = r_1 - kvh_u_1 * per_kv_1;
                unsigned int j_1 = rem_1 / (unsigned int)group;
                unsigned int head_u_1 = kvh_u_1 * (unsigned int)group + (rem_1 - j_1 * (unsigned int)group);
                int seg_1 = 0;
                unsigned int before_1 = 0;
                unsigned int ends_1 = 0;
                #pragma unroll 1
                for (int s_7 = 0; s_7 < num_segments; s_7++) {
                    int q_len_6 = cu_seqlens_q[s_7 + 1] - cu_seqlens_q[s_7];
                    int k_len_6 = cu_seqlens_k[s_7 + 1] - cu_seqlens_k[s_7];
                    int row0_6 = ((q_len_6 + 255 >> 8) - 1) * 256;
                    int diag_6 = k_len_6 - q_len_6;
                    int n_k_6 = k_len_6 + 127 >> 7;
                    int n_c_6 = row0_6 + 256 + diag_6 + 127 >> 7;
                    int n_seen_6 = ((n_c_6 < n_k_6) ? n_c_6 : n_k_6);
                    int n_blocks_6 = ((n_seen_6 > 0) ? n_seen_6 : 0);
                    int first_seen_6 = row0_6 + diag_6 + 1;
                    int n_first_6 = ((first_seen_6 > 0) ? first_seen_6 >> 7 : 0);
                    int k_full_6 = k_len_6 >> 7;
                    int n_kf_6 = ((n_first_6 < k_full_6) ? n_first_6 : k_full_6);
                    int n_full_6 = ((n_kf_6 < n_blocks_6) ? n_kf_6 : n_blocks_6);
                    int n_masked_6 = n_blocks_6 - n_full_6;
                    bool in_class_1 = ((2 * n_blocks_6 >= wmax_1) ? 0 : ((4 * n_blocks_6 >= wmax_1) ? 1 : ((8 * n_blocks_6 >= wmax_1) ? 2 : 3))) == c_3;
                    ends_1 = ends_1 + ((in_class_1) ? (unsigned int)(cu_seqlens_q[s_7 + 1] - cu_seqlens_q[s_7] + 255) >> 8 : 0);
                    seg_1 = seg_1 + ((ends_1 <= j_1) ? 1 : 0);
                    before_1 = ((ends_1 <= j_1) ? ends_1 : before_1);
                }
                unsigned int m_rel_1 = j_1 - before_1;
                int q0_1 = cu_seqlens_q[seg_1];
                int q_len_7 = cu_seqlens_q[seg_1 + 1] - q0_1;
                int k0_1 = cu_seqlens_k[seg_1];
                int k_len_7 = cu_seqlens_k[seg_1 + 1] - k0_1;
                int m_1 = (int)(((unsigned int)(q_len_7 + 255) >> 8) - 1 - m_rel_1);
                int head_1 = (int)head_u_1;
                int kvh_1 = (int)kvh_u_1;
                int row0_7 = m_1 * 256;
                int diag_7 = k_len_7 - q_len_7;
                int n_k_7 = k_len_7 + 127 >> 7;
                int n_c_7 = row0_7 + 256 + diag_7 + 127 >> 7;
                int n_seen_7 = ((n_c_7 < n_k_7) ? n_c_7 : n_k_7);
                int n_blocks_7 = ((n_seen_7 > 0) ? n_seen_7 : 0);
                int first_seen_7 = row0_7 + diag_7 + 1;
                int n_first_7 = ((first_seen_7 > 0) ? first_seen_7 >> 7 : 0);
                int k_full_7 = k_len_7 >> 7;
                int n_kf_7 = ((n_first_7 < k_full_7) ? n_first_7 : k_full_7);
                int n_full_7 = ((n_kf_7 < n_blocks_7) ? n_kf_7 : n_blocks_7);
                int n_masked_7 = n_blocks_7 - n_full_7;
                int row_1 = m_1 * 256 + cta_rank * 128 + tr_1;
                bool row_valid = row_1 < q_len_7;
                long long o_base = ((long long)(q0_1 + row_1) * (long long)num_heads_q + (long long)head_1) * 256;
                long long lse_index = (long long)head_1 * (long long)total_q + (long long)(q0_1 + row_1);
                int has_block = ((n_blocks_7 > 0) ? 1 : 0);
                #pragma unroll 1
                for (int _ = 0; _ < has_block; _++) {
                    mbarrier_wait(scale_full_addr, _phase_scale_full);
                    mbarrier_arrive(scale_empty_addr);
                    _phase_scale_full ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (pq_1) * 8) & 0xFEFFFFFF) : "memory");
                    pq_1 += 1;
                    if (pq_1 == 2) { pq_1 = 0; }
                }
                #pragma unroll 1
                for (int i_2 = 1; i_2 < n_blocks_7; i_2++) {
                    mbarrier_wait(scale_full_addr, _phase_scale_full);
                    float acc_scale_2 = s_scale[(int)0 * 128 + tr_1];
                    mbarrier_arrive(scale_empty_addr);
                    _phase_scale_full ^= 1;
                    mbarrier_wait(o_full_addr + (oq) * 8, _phase_o_full);
                    oq += 1;
                    if (oq == 2) { oq = 0; _phase_o_full ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _warp_vote_0 = __any_sync(0xFFFFFFFF, acc_scale_2 < 1.0f);
                    if (_warp_vote_0 != 0) {
                        unsigned int addr_0 = taddr + 256 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_0[0]), "=f"(ov_0[1]), "=f"(ov_0[2]), "=f"(ov_0[3]), "=f"(ov_0[4]), "=f"(ov_0[5]), "=f"(ov_0[6]), "=f"(ov_0[7]), "=f"(ov_0[8]), "=f"(ov_0[9]), "=f"(ov_0[10]), "=f"(ov_0[11]), "=f"(ov_0[12]), "=f"(ov_0[13]), "=f"(ov_0[14]), "=f"(ov_0[15]), "=f"(ov_0[16]), "=f"(ov_0[17]), "=f"(ov_0[18]), "=f"(ov_0[19]), "=f"(ov_0[20]), "=f"(ov_0[21]), "=f"(ov_0[22]), "=f"(ov_0[23]), "=f"(ov_0[24]), "=f"(ov_0[25]), "=f"(ov_0[26]), "=f"(ov_0[27]), "=f"(ov_0[28]), "=f"(ov_0[29]), "=f"(ov_0[30]), "=f"(ov_0[31])
                            : "r"(addr_0));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_1 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_0)[_ls], _scale2_1);
                        tmem_st_x32_f32(addr_0, ov_0);
                        unsigned int addr_1 = taddr + 288 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_1[0]), "=f"(ov_1[1]), "=f"(ov_1[2]), "=f"(ov_1[3]), "=f"(ov_1[4]), "=f"(ov_1[5]), "=f"(ov_1[6]), "=f"(ov_1[7]), "=f"(ov_1[8]), "=f"(ov_1[9]), "=f"(ov_1[10]), "=f"(ov_1[11]), "=f"(ov_1[12]), "=f"(ov_1[13]), "=f"(ov_1[14]), "=f"(ov_1[15]), "=f"(ov_1[16]), "=f"(ov_1[17]), "=f"(ov_1[18]), "=f"(ov_1[19]), "=f"(ov_1[20]), "=f"(ov_1[21]), "=f"(ov_1[22]), "=f"(ov_1[23]), "=f"(ov_1[24]), "=f"(ov_1[25]), "=f"(ov_1[26]), "=f"(ov_1[27]), "=f"(ov_1[28]), "=f"(ov_1[29]), "=f"(ov_1[30]), "=f"(ov_1[31])
                            : "r"(addr_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_2 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_2);
                        tmem_st_x32_f32(addr_1, ov_1);
                        unsigned int addr_2 = taddr + 320 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_2[0]), "=f"(ov_2[1]), "=f"(ov_2[2]), "=f"(ov_2[3]), "=f"(ov_2[4]), "=f"(ov_2[5]), "=f"(ov_2[6]), "=f"(ov_2[7]), "=f"(ov_2[8]), "=f"(ov_2[9]), "=f"(ov_2[10]), "=f"(ov_2[11]), "=f"(ov_2[12]), "=f"(ov_2[13]), "=f"(ov_2[14]), "=f"(ov_2[15]), "=f"(ov_2[16]), "=f"(ov_2[17]), "=f"(ov_2[18]), "=f"(ov_2[19]), "=f"(ov_2[20]), "=f"(ov_2[21]), "=f"(ov_2[22]), "=f"(ov_2[23]), "=f"(ov_2[24]), "=f"(ov_2[25]), "=f"(ov_2[26]), "=f"(ov_2[27]), "=f"(ov_2[28]), "=f"(ov_2[29]), "=f"(ov_2[30]), "=f"(ov_2[31])
                            : "r"(addr_2));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_3 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_3);
                        tmem_st_x32_f32(addr_2, ov_2);
                        unsigned int addr_3 = taddr + 352 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_3[0]), "=f"(ov_3[1]), "=f"(ov_3[2]), "=f"(ov_3[3]), "=f"(ov_3[4]), "=f"(ov_3[5]), "=f"(ov_3[6]), "=f"(ov_3[7]), "=f"(ov_3[8]), "=f"(ov_3[9]), "=f"(ov_3[10]), "=f"(ov_3[11]), "=f"(ov_3[12]), "=f"(ov_3[13]), "=f"(ov_3[14]), "=f"(ov_3[15]), "=f"(ov_3[16]), "=f"(ov_3[17]), "=f"(ov_3[18]), "=f"(ov_3[19]), "=f"(ov_3[20]), "=f"(ov_3[21]), "=f"(ov_3[22]), "=f"(ov_3[23]), "=f"(ov_3[24]), "=f"(ov_3[25]), "=f"(ov_3[26]), "=f"(ov_3[27]), "=f"(ov_3[28]), "=f"(ov_3[29]), "=f"(ov_3[30]), "=f"(ov_3[31])
                            : "r"(addr_3));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_4 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_3)[_ls], _scale2_4);
                        tmem_st_x32_f32(addr_3, ov_3);
                        unsigned int addr_4 = taddr + 384 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_4[0]), "=f"(ov_4[1]), "=f"(ov_4[2]), "=f"(ov_4[3]), "=f"(ov_4[4]), "=f"(ov_4[5]), "=f"(ov_4[6]), "=f"(ov_4[7]), "=f"(ov_4[8]), "=f"(ov_4[9]), "=f"(ov_4[10]), "=f"(ov_4[11]), "=f"(ov_4[12]), "=f"(ov_4[13]), "=f"(ov_4[14]), "=f"(ov_4[15]), "=f"(ov_4[16]), "=f"(ov_4[17]), "=f"(ov_4[18]), "=f"(ov_4[19]), "=f"(ov_4[20]), "=f"(ov_4[21]), "=f"(ov_4[22]), "=f"(ov_4[23]), "=f"(ov_4[24]), "=f"(ov_4[25]), "=f"(ov_4[26]), "=f"(ov_4[27]), "=f"(ov_4[28]), "=f"(ov_4[29]), "=f"(ov_4[30]), "=f"(ov_4[31])
                            : "r"(addr_4));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_5 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_4)[_ls], _scale2_5);
                        tmem_st_x32_f32(addr_4, ov_4);
                        unsigned int addr_5 = taddr + 416 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_5[0]), "=f"(ov_5[1]), "=f"(ov_5[2]), "=f"(ov_5[3]), "=f"(ov_5[4]), "=f"(ov_5[5]), "=f"(ov_5[6]), "=f"(ov_5[7]), "=f"(ov_5[8]), "=f"(ov_5[9]), "=f"(ov_5[10]), "=f"(ov_5[11]), "=f"(ov_5[12]), "=f"(ov_5[13]), "=f"(ov_5[14]), "=f"(ov_5[15]), "=f"(ov_5[16]), "=f"(ov_5[17]), "=f"(ov_5[18]), "=f"(ov_5[19]), "=f"(ov_5[20]), "=f"(ov_5[21]), "=f"(ov_5[22]), "=f"(ov_5[23]), "=f"(ov_5[24]), "=f"(ov_5[25]), "=f"(ov_5[26]), "=f"(ov_5[27]), "=f"(ov_5[28]), "=f"(ov_5[29]), "=f"(ov_5[30]), "=f"(ov_5[31])
                            : "r"(addr_5));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_6 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_5)[_ls], _scale2_6);
                        tmem_st_x32_f32(addr_5, ov_5);
                        unsigned int addr_6 = taddr + 448 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_6[0]), "=f"(ov_6[1]), "=f"(ov_6[2]), "=f"(ov_6[3]), "=f"(ov_6[4]), "=f"(ov_6[5]), "=f"(ov_6[6]), "=f"(ov_6[7]), "=f"(ov_6[8]), "=f"(ov_6[9]), "=f"(ov_6[10]), "=f"(ov_6[11]), "=f"(ov_6[12]), "=f"(ov_6[13]), "=f"(ov_6[14]), "=f"(ov_6[15]), "=f"(ov_6[16]), "=f"(ov_6[17]), "=f"(ov_6[18]), "=f"(ov_6[19]), "=f"(ov_6[20]), "=f"(ov_6[21]), "=f"(ov_6[22]), "=f"(ov_6[23]), "=f"(ov_6[24]), "=f"(ov_6[25]), "=f"(ov_6[26]), "=f"(ov_6[27]), "=f"(ov_6[28]), "=f"(ov_6[29]), "=f"(ov_6[30]), "=f"(ov_6[31])
                            : "r"(addr_6));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_7 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_6)[_ls], _scale2_7);
                        tmem_st_x32_f32(addr_6, ov_6);
                        unsigned int addr_7 = taddr + 480 + (unsigned int)(warp % 4 * 32 << 16);
                        float ov_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov_7[0]), "=f"(ov_7[1]), "=f"(ov_7[2]), "=f"(ov_7[3]), "=f"(ov_7[4]), "=f"(ov_7[5]), "=f"(ov_7[6]), "=f"(ov_7[7]), "=f"(ov_7[8]), "=f"(ov_7[9]), "=f"(ov_7[10]), "=f"(ov_7[11]), "=f"(ov_7[12]), "=f"(ov_7[13]), "=f"(ov_7[14]), "=f"(ov_7[15]), "=f"(ov_7[16]), "=f"(ov_7[17]), "=f"(ov_7[18]), "=f"(ov_7[19]), "=f"(ov_7[20]), "=f"(ov_7[21]), "=f"(ov_7[22]), "=f"(ov_7[23]), "=f"(ov_7[24]), "=f"(ov_7[25]), "=f"(ov_7[26]), "=f"(ov_7[27]), "=f"(ov_7[28]), "=f"(ov_7[29]), "=f"(ov_7[30]), "=f"(ov_7[31])
                            : "r"(addr_7));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_8 = {acc_scale_2, acc_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_7)[_ls], _scale2_8);
                        tmem_st_x32_f32(addr_7, ov_7);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (pq_1) * 8) & 0xFEFFFFFF) : "memory");
                    pq_1 += 1;
                    if (pq_1 == 2) { pq_1 = 0; }
                }
                mbarrier_wait(stats_full_addr, _phase_stats_full);
                float final_sum = s_stats[tr_1];
                float final_max = s_stats[128 + tr_1];
                mbarrier_arrive(stats_empty_addr);
                _phase_stats_full ^= 1;
                mbarrier_wait(o_full_addr + (oq) * 8, _phase_o_full);
                oq += 1;
                if (oq == 2) { oq = 0; _phase_o_full ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((o_free_addr) & 0xFEFFFFFF) : "memory");
                float final_scale = 0.0f;
                float lse = -CUDART_INF_F;
                if (final_sum > 0.0f) {
                    float _rcp_9 = approx_rcp(final_sum);
                    final_scale = _rcp_9;
                    float _log2_10;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_10) : "f"(final_sum));
                    lse = (final_max * softmax_scale_log2 + _log2_10) * 0.6931472f;
                }
                if (n_blocks_7 > 0) {
                    unsigned int addr_8 = taddr + 256 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_8[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_8[0]), "=f"(ov_8[1]), "=f"(ov_8[2]), "=f"(ov_8[3]), "=f"(ov_8[4]), "=f"(ov_8[5]), "=f"(ov_8[6]), "=f"(ov_8[7]), "=f"(ov_8[8]), "=f"(ov_8[9]), "=f"(ov_8[10]), "=f"(ov_8[11]), "=f"(ov_8[12]), "=f"(ov_8[13]), "=f"(ov_8[14]), "=f"(ov_8[15]), "=f"(ov_8[16]), "=f"(ov_8[17]), "=f"(ov_8[18]), "=f"(ov_8[19]), "=f"(ov_8[20]), "=f"(ov_8[21]), "=f"(ov_8[22]), "=f"(ov_8[23]), "=f"(ov_8[24]), "=f"(ov_8[25]), "=f"(ov_8[26]), "=f"(ov_8[27]), "=f"(ov_8[28]), "=f"(ov_8[29]), "=f"(ov_8[30]), "=f"(ov_8[31])
                        : "r"(addr_8));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_11 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_8)[_ls], _scale2_11);
                    unsigned int pk_8[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 0], ov_8[_lp*2+1 + 0]));
                        pk_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 0)), "r"(pk_8[(0) + 0]), "r"(pk_8[(0) + 1]), "r"(pk_8[(0) + 2]), "r"(pk_8[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_9[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 8], ov_8[_lp*2+1 + 8]));
                        pk_9[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 8)), "r"(pk_9[(0) + 0]), "r"(pk_9[(0) + 1]), "r"(pk_9[(0) + 2]), "r"(pk_9[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_10[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 16], ov_8[_lp*2+1 + 16]));
                        pk_10[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 16)), "r"(pk_10[(0) + 0]), "r"(pk_10[(0) + 1]), "r"(pk_10[(0) + 2]), "r"(pk_10[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_11[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 24], ov_8[_lp*2+1 + 24]));
                        pk_11[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 24)), "r"(pk_11[(0) + 0]), "r"(pk_11[(0) + 1]), "r"(pk_11[(0) + 2]), "r"(pk_11[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_9 = taddr + 288 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_9[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_9[0]), "=f"(ov_9[1]), "=f"(ov_9[2]), "=f"(ov_9[3]), "=f"(ov_9[4]), "=f"(ov_9[5]), "=f"(ov_9[6]), "=f"(ov_9[7]), "=f"(ov_9[8]), "=f"(ov_9[9]), "=f"(ov_9[10]), "=f"(ov_9[11]), "=f"(ov_9[12]), "=f"(ov_9[13]), "=f"(ov_9[14]), "=f"(ov_9[15]), "=f"(ov_9[16]), "=f"(ov_9[17]), "=f"(ov_9[18]), "=f"(ov_9[19]), "=f"(ov_9[20]), "=f"(ov_9[21]), "=f"(ov_9[22]), "=f"(ov_9[23]), "=f"(ov_9[24]), "=f"(ov_9[25]), "=f"(ov_9[26]), "=f"(ov_9[27]), "=f"(ov_9[28]), "=f"(ov_9[29]), "=f"(ov_9[30]), "=f"(ov_9[31])
                        : "r"(addr_9));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_12 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_9)[_ls], _scale2_12);
                    unsigned int pk_12[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_9[_lp*2 + 0], ov_9[_lp*2+1 + 0]));
                        pk_12[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 32)), "r"(pk_12[(0) + 0]), "r"(pk_12[(0) + 1]), "r"(pk_12[(0) + 2]), "r"(pk_12[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_13[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_9[_lp*2 + 8], ov_9[_lp*2+1 + 8]));
                        pk_13[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 40)), "r"(pk_13[(0) + 0]), "r"(pk_13[(0) + 1]), "r"(pk_13[(0) + 2]), "r"(pk_13[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_14[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_9[_lp*2 + 16], ov_9[_lp*2+1 + 16]));
                        pk_14[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 48)), "r"(pk_14[(0) + 0]), "r"(pk_14[(0) + 1]), "r"(pk_14[(0) + 2]), "r"(pk_14[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_15[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_9[_lp*2 + 24], ov_9[_lp*2+1 + 24]));
                        pk_15[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 56)), "r"(pk_15[(0) + 0]), "r"(pk_15[(0) + 1]), "r"(pk_15[(0) + 2]), "r"(pk_15[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_10 = taddr + 320 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_10[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_10[0]), "=f"(ov_10[1]), "=f"(ov_10[2]), "=f"(ov_10[3]), "=f"(ov_10[4]), "=f"(ov_10[5]), "=f"(ov_10[6]), "=f"(ov_10[7]), "=f"(ov_10[8]), "=f"(ov_10[9]), "=f"(ov_10[10]), "=f"(ov_10[11]), "=f"(ov_10[12]), "=f"(ov_10[13]), "=f"(ov_10[14]), "=f"(ov_10[15]), "=f"(ov_10[16]), "=f"(ov_10[17]), "=f"(ov_10[18]), "=f"(ov_10[19]), "=f"(ov_10[20]), "=f"(ov_10[21]), "=f"(ov_10[22]), "=f"(ov_10[23]), "=f"(ov_10[24]), "=f"(ov_10[25]), "=f"(ov_10[26]), "=f"(ov_10[27]), "=f"(ov_10[28]), "=f"(ov_10[29]), "=f"(ov_10[30]), "=f"(ov_10[31])
                        : "r"(addr_10));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_13 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_10)[_ls], _scale2_13);
                    unsigned int pk_16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 0], ov_10[_lp*2+1 + 0]));
                        pk_16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 64)), "r"(pk_16[(0) + 0]), "r"(pk_16[(0) + 1]), "r"(pk_16[(0) + 2]), "r"(pk_16[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_17[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 8], ov_10[_lp*2+1 + 8]));
                        pk_17[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 72)), "r"(pk_17[(0) + 0]), "r"(pk_17[(0) + 1]), "r"(pk_17[(0) + 2]), "r"(pk_17[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_18[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 16], ov_10[_lp*2+1 + 16]));
                        pk_18[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 80)), "r"(pk_18[(0) + 0]), "r"(pk_18[(0) + 1]), "r"(pk_18[(0) + 2]), "r"(pk_18[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_19[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 24], ov_10[_lp*2+1 + 24]));
                        pk_19[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 88)), "r"(pk_19[(0) + 0]), "r"(pk_19[(0) + 1]), "r"(pk_19[(0) + 2]), "r"(pk_19[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_11 = taddr + 352 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_11[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_11[0]), "=f"(ov_11[1]), "=f"(ov_11[2]), "=f"(ov_11[3]), "=f"(ov_11[4]), "=f"(ov_11[5]), "=f"(ov_11[6]), "=f"(ov_11[7]), "=f"(ov_11[8]), "=f"(ov_11[9]), "=f"(ov_11[10]), "=f"(ov_11[11]), "=f"(ov_11[12]), "=f"(ov_11[13]), "=f"(ov_11[14]), "=f"(ov_11[15]), "=f"(ov_11[16]), "=f"(ov_11[17]), "=f"(ov_11[18]), "=f"(ov_11[19]), "=f"(ov_11[20]), "=f"(ov_11[21]), "=f"(ov_11[22]), "=f"(ov_11[23]), "=f"(ov_11[24]), "=f"(ov_11[25]), "=f"(ov_11[26]), "=f"(ov_11[27]), "=f"(ov_11[28]), "=f"(ov_11[29]), "=f"(ov_11[30]), "=f"(ov_11[31])
                        : "r"(addr_11));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_14 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_11)[_ls], _scale2_14);
                    unsigned int pk_20[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_11[_lp*2 + 0], ov_11[_lp*2+1 + 0]));
                        pk_20[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 96)), "r"(pk_20[(0) + 0]), "r"(pk_20[(0) + 1]), "r"(pk_20[(0) + 2]), "r"(pk_20[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_21[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_11[_lp*2 + 8], ov_11[_lp*2+1 + 8]));
                        pk_21[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 104)), "r"(pk_21[(0) + 0]), "r"(pk_21[(0) + 1]), "r"(pk_21[(0) + 2]), "r"(pk_21[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_22[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_11[_lp*2 + 16], ov_11[_lp*2+1 + 16]));
                        pk_22[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 112)), "r"(pk_22[(0) + 0]), "r"(pk_22[(0) + 1]), "r"(pk_22[(0) + 2]), "r"(pk_22[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_23[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_11[_lp*2 + 24], ov_11[_lp*2+1 + 24]));
                        pk_23[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 120)), "r"(pk_23[(0) + 0]), "r"(pk_23[(0) + 1]), "r"(pk_23[(0) + 2]), "r"(pk_23[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_12 = taddr + 384 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_12[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_12[0]), "=f"(ov_12[1]), "=f"(ov_12[2]), "=f"(ov_12[3]), "=f"(ov_12[4]), "=f"(ov_12[5]), "=f"(ov_12[6]), "=f"(ov_12[7]), "=f"(ov_12[8]), "=f"(ov_12[9]), "=f"(ov_12[10]), "=f"(ov_12[11]), "=f"(ov_12[12]), "=f"(ov_12[13]), "=f"(ov_12[14]), "=f"(ov_12[15]), "=f"(ov_12[16]), "=f"(ov_12[17]), "=f"(ov_12[18]), "=f"(ov_12[19]), "=f"(ov_12[20]), "=f"(ov_12[21]), "=f"(ov_12[22]), "=f"(ov_12[23]), "=f"(ov_12[24]), "=f"(ov_12[25]), "=f"(ov_12[26]), "=f"(ov_12[27]), "=f"(ov_12[28]), "=f"(ov_12[29]), "=f"(ov_12[30]), "=f"(ov_12[31])
                        : "r"(addr_12));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_15 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_12)[_ls], _scale2_15);
                    unsigned int pk_24[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 0], ov_12[_lp*2+1 + 0]));
                        pk_24[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 128)), "r"(pk_24[(0) + 0]), "r"(pk_24[(0) + 1]), "r"(pk_24[(0) + 2]), "r"(pk_24[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_25[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 8], ov_12[_lp*2+1 + 8]));
                        pk_25[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 136)), "r"(pk_25[(0) + 0]), "r"(pk_25[(0) + 1]), "r"(pk_25[(0) + 2]), "r"(pk_25[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_26[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 16], ov_12[_lp*2+1 + 16]));
                        pk_26[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 144)), "r"(pk_26[(0) + 0]), "r"(pk_26[(0) + 1]), "r"(pk_26[(0) + 2]), "r"(pk_26[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_27[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 24], ov_12[_lp*2+1 + 24]));
                        pk_27[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 152)), "r"(pk_27[(0) + 0]), "r"(pk_27[(0) + 1]), "r"(pk_27[(0) + 2]), "r"(pk_27[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_13 = taddr + 416 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_13[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_13[0]), "=f"(ov_13[1]), "=f"(ov_13[2]), "=f"(ov_13[3]), "=f"(ov_13[4]), "=f"(ov_13[5]), "=f"(ov_13[6]), "=f"(ov_13[7]), "=f"(ov_13[8]), "=f"(ov_13[9]), "=f"(ov_13[10]), "=f"(ov_13[11]), "=f"(ov_13[12]), "=f"(ov_13[13]), "=f"(ov_13[14]), "=f"(ov_13[15]), "=f"(ov_13[16]), "=f"(ov_13[17]), "=f"(ov_13[18]), "=f"(ov_13[19]), "=f"(ov_13[20]), "=f"(ov_13[21]), "=f"(ov_13[22]), "=f"(ov_13[23]), "=f"(ov_13[24]), "=f"(ov_13[25]), "=f"(ov_13[26]), "=f"(ov_13[27]), "=f"(ov_13[28]), "=f"(ov_13[29]), "=f"(ov_13[30]), "=f"(ov_13[31])
                        : "r"(addr_13));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_16 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_13)[_ls], _scale2_16);
                    unsigned int pk_28[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_13[_lp*2 + 0], ov_13[_lp*2+1 + 0]));
                        pk_28[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 160)), "r"(pk_28[(0) + 0]), "r"(pk_28[(0) + 1]), "r"(pk_28[(0) + 2]), "r"(pk_28[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_29[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_13[_lp*2 + 8], ov_13[_lp*2+1 + 8]));
                        pk_29[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 168)), "r"(pk_29[(0) + 0]), "r"(pk_29[(0) + 1]), "r"(pk_29[(0) + 2]), "r"(pk_29[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_30[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_13[_lp*2 + 16], ov_13[_lp*2+1 + 16]));
                        pk_30[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 176)), "r"(pk_30[(0) + 0]), "r"(pk_30[(0) + 1]), "r"(pk_30[(0) + 2]), "r"(pk_30[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_31[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_13[_lp*2 + 24], ov_13[_lp*2+1 + 24]));
                        pk_31[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 184)), "r"(pk_31[(0) + 0]), "r"(pk_31[(0) + 1]), "r"(pk_31[(0) + 2]), "r"(pk_31[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_14 = taddr + 448 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_14[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_14[0]), "=f"(ov_14[1]), "=f"(ov_14[2]), "=f"(ov_14[3]), "=f"(ov_14[4]), "=f"(ov_14[5]), "=f"(ov_14[6]), "=f"(ov_14[7]), "=f"(ov_14[8]), "=f"(ov_14[9]), "=f"(ov_14[10]), "=f"(ov_14[11]), "=f"(ov_14[12]), "=f"(ov_14[13]), "=f"(ov_14[14]), "=f"(ov_14[15]), "=f"(ov_14[16]), "=f"(ov_14[17]), "=f"(ov_14[18]), "=f"(ov_14[19]), "=f"(ov_14[20]), "=f"(ov_14[21]), "=f"(ov_14[22]), "=f"(ov_14[23]), "=f"(ov_14[24]), "=f"(ov_14[25]), "=f"(ov_14[26]), "=f"(ov_14[27]), "=f"(ov_14[28]), "=f"(ov_14[29]), "=f"(ov_14[30]), "=f"(ov_14[31])
                        : "r"(addr_14));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_17 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_14)[_ls], _scale2_17);
                    unsigned int pk_32[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_14[_lp*2 + 0], ov_14[_lp*2+1 + 0]));
                        pk_32[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 192)), "r"(pk_32[(0) + 0]), "r"(pk_32[(0) + 1]), "r"(pk_32[(0) + 2]), "r"(pk_32[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_33[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_14[_lp*2 + 8], ov_14[_lp*2+1 + 8]));
                        pk_33[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 200)), "r"(pk_33[(0) + 0]), "r"(pk_33[(0) + 1]), "r"(pk_33[(0) + 2]), "r"(pk_33[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_34[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_14[_lp*2 + 16], ov_14[_lp*2+1 + 16]));
                        pk_34[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 208)), "r"(pk_34[(0) + 0]), "r"(pk_34[(0) + 1]), "r"(pk_34[(0) + 2]), "r"(pk_34[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_35[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_14[_lp*2 + 24], ov_14[_lp*2+1 + 24]));
                        pk_35[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 216)), "r"(pk_35[(0) + 0]), "r"(pk_35[(0) + 1]), "r"(pk_35[(0) + 2]), "r"(pk_35[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int addr_15 = taddr + 480 + (unsigned int)(warp % 4 * 32 << 16);
                    float ov_15[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(ov_15[0]), "=f"(ov_15[1]), "=f"(ov_15[2]), "=f"(ov_15[3]), "=f"(ov_15[4]), "=f"(ov_15[5]), "=f"(ov_15[6]), "=f"(ov_15[7]), "=f"(ov_15[8]), "=f"(ov_15[9]), "=f"(ov_15[10]), "=f"(ov_15[11]), "=f"(ov_15[12]), "=f"(ov_15[13]), "=f"(ov_15[14]), "=f"(ov_15[15]), "=f"(ov_15[16]), "=f"(ov_15[17]), "=f"(ov_15[18]), "=f"(ov_15[19]), "=f"(ov_15[20]), "=f"(ov_15[21]), "=f"(ov_15[22]), "=f"(ov_15[23]), "=f"(ov_15[24]), "=f"(ov_15[25]), "=f"(ov_15[26]), "=f"(ov_15[27]), "=f"(ov_15[28]), "=f"(ov_15[29]), "=f"(ov_15[30]), "=f"(ov_15[31])
                        : "r"(addr_15));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_18 = {final_scale, final_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_15)[_ls], _scale2_18);
                    unsigned int pk_36[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_15[_lp*2 + 0], ov_15[_lp*2+1 + 0]));
                        pk_36[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 224)), "r"(pk_36[(0) + 0]), "r"(pk_36[(0) + 1]), "r"(pk_36[(0) + 2]), "r"(pk_36[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_37[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_15[_lp*2 + 8], ov_15[_lp*2+1 + 8]));
                        pk_37[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 232)), "r"(pk_37[(0) + 0]), "r"(pk_37[(0) + 1]), "r"(pk_37[(0) + 2]), "r"(pk_37[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_38[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_15[_lp*2 + 16], ov_15[_lp*2+1 + 16]));
                        pk_38[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 240)), "r"(pk_38[(0) + 0]), "r"(pk_38[(0) + 1]), "r"(pk_38[(0) + 2]), "r"(pk_38[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    unsigned int pk_39[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_15[_lp*2 + 24], ov_15[_lp*2+1 + 24]));
                        pk_39[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 248)), "r"(pk_39[(0) + 0]), "r"(pk_39[(0) + 1]), "r"(pk_39[(0) + 2]), "r"(pk_39[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                } else {
                    unsigned int zero[4];
                    zero[0] = 0;
                    zero[1] = 0;
                    zero[2] = 0;
                    zero[3] = 0;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 0)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 8)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 16)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 24)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 32)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 40)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 48)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 56)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 64)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 72)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 80)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 88)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 96)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 104)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 112)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 120)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 128)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 136)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 144)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 152)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 160)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 168)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 176)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 184)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 192)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 200)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 208)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 216)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 224)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 232)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 240)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(O + (o_base + 248)), "r"(zero[(0) + 0]), "r"(zero[(0) + 1]), "r"(zero[(0) + 2]), "r"(zero[(0) + 3]), "r"((unsigned int)(row_valid)) : "memory");
                }
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(LSE + lse_index)), "f"(lse), "r"((unsigned int)(row_valid)) : "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            unsigned int wk_2 = 0;
            unsigned int kv = 0;
            unsigned int sq_1 = 0;
            unsigned int pq_2 = 0;
            unsigned int oq_1 = 0;
            unsigned int head_tiles_2 = 0;
            #pragma unroll 1
            for (int s_8 = 0; s_8 < num_segments; s_8++) {
                head_tiles_2 = head_tiles_2 + ((unsigned int)(cu_seqlens_q[s_8 + 1] - cu_seqlens_q[s_8] + 255) >> 8);
            }
            int wmax_2 = 0;
            #pragma unroll 1
            for (int s_9 = 0; s_9 < num_segments; s_9++) {
                int q_len_8 = cu_seqlens_q[s_9 + 1] - cu_seqlens_q[s_9];
                int k_len_8 = cu_seqlens_k[s_9 + 1] - cu_seqlens_k[s_9];
                int row0_8 = ((q_len_8 + 255 >> 8) - 1) * 256;
                int diag_8 = k_len_8 - q_len_8;
                int n_k_8 = k_len_8 + 127 >> 7;
                int n_c_8 = row0_8 + 256 + diag_8 + 127 >> 7;
                int n_seen_8 = ((n_c_8 < n_k_8) ? n_c_8 : n_k_8);
                int n_blocks_8 = ((n_seen_8 > 0) ? n_seen_8 : 0);
                int first_seen_8 = row0_8 + diag_8 + 1;
                int n_first_8 = ((first_seen_8 > 0) ? first_seen_8 >> 7 : 0);
                int k_full_8 = k_len_8 >> 7;
                int n_kf_8 = ((n_first_8 < k_full_8) ? n_first_8 : k_full_8);
                int n_full_8 = ((n_kf_8 < n_blocks_8) ? n_kf_8 : n_blocks_8);
                int n_masked_8 = n_blocks_8 - n_full_8;
                int w_2 = n_blocks_8;
                wmax_2 = ((w_2 > wmax_2) ? w_2 : wmax_2);
            }
            unsigned int t0_2 = 0;
            unsigned int t1_2 = 0;
            unsigned int t2_2 = 0;
            #pragma unroll 1
            for (int s_10 = 0; s_10 < num_segments; s_10++) {
                unsigned int tiles_2 = (unsigned int)(cu_seqlens_q[s_10 + 1] - cu_seqlens_q[s_10] + 255) >> 8;
                int q_len_9 = cu_seqlens_q[s_10 + 1] - cu_seqlens_q[s_10];
                int k_len_9 = cu_seqlens_k[s_10 + 1] - cu_seqlens_k[s_10];
                int row0_9 = ((q_len_9 + 255 >> 8) - 1) * 256;
                int diag_9 = k_len_9 - q_len_9;
                int n_k_9 = k_len_9 + 127 >> 7;
                int n_c_9 = row0_9 + 256 + diag_9 + 127 >> 7;
                int n_seen_9 = ((n_c_9 < n_k_9) ? n_c_9 : n_k_9);
                int n_blocks_9 = ((n_seen_9 > 0) ? n_seen_9 : 0);
                int first_seen_9 = row0_9 + diag_9 + 1;
                int n_first_9 = ((first_seen_9 > 0) ? first_seen_9 >> 7 : 0);
                int k_full_9 = k_len_9 >> 7;
                int n_kf_9 = ((n_first_9 < k_full_9) ? n_first_9 : k_full_9);
                int n_full_9 = ((n_kf_9 < n_blocks_9) ? n_kf_9 : n_blocks_9);
                int n_masked_9 = n_blocks_9 - n_full_9;
                int c_4 = ((2 * n_blocks_9 >= wmax_2) ? 0 : ((4 * n_blocks_9 >= wmax_2) ? 1 : ((8 * n_blocks_9 >= wmax_2) ? 2 : 3)));
                t0_2 = t0_2 + ((c_4 == 0) ? tiles_2 : 0);
                t1_2 = t1_2 + ((c_4 == 1) ? tiles_2 : 0);
                t2_2 = t2_2 + ((c_4 == 2) ? tiles_2 : 0);
            }
            unsigned int total_2 = (unsigned int)num_heads_q * head_tiles_2;
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_o_free = 0;
            #pragma unroll 1
            for (unsigned int it_2 = 0; it_2 < (unsigned int)num_heads_q * ((unsigned int)(total_q + 255 >> 8) + (unsigned int)num_segments) + 1; it_2++) {
                mbarrier_wait_cluster_hint(work_full_addr + (wk_2) * 8, _phase_work_full_2, 10000000);
                unsigned int ticket_2 = cells[wk_2];
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wk_2) * 8), "r"(0) : "memory");
                wk_2 += 1;
                if (wk_2 == 2) { wk_2 = 0; _phase_work_full_2 ^= 1; }
                if (ticket_2 >= total_2) {
                    break;
                }
                unsigned int hq_2 = (unsigned int)num_heads_q;
                unsigned int p1_2 = hq_2 * t0_2;
                unsigned int p2_2 = p1_2 + hq_2 * t1_2;
                unsigned int p3_2 = p2_2 + hq_2 * t2_2;
                int c_5 = ((ticket_2 < p1_2) ? 0 : ((ticket_2 < p2_2) ? 1 : ((ticket_2 < p3_2) ? 2 : 3)));
                unsigned int base_2 = ((c_5 == 0) ? 0 : ((c_5 == 1) ? p1_2 : ((c_5 == 2) ? p2_2 : p3_2)));
                unsigned int tc_2 = ((c_5 == 0) ? t0_2 : ((c_5 == 1) ? t1_2 : ((c_5 == 2) ? t2_2 : head_tiles_2 - t0_2 - t1_2 - t2_2)));
                unsigned int r_2 = ticket_2 - base_2;
                unsigned int per_kv_2 = tc_2 * (unsigned int)group;
                unsigned int kvh_u_2 = r_2 / per_kv_2;
                unsigned int rem_2 = r_2 - kvh_u_2 * per_kv_2;
                unsigned int j_2 = rem_2 / (unsigned int)group;
                unsigned int head_u_2 = kvh_u_2 * (unsigned int)group + (rem_2 - j_2 * (unsigned int)group);
                int seg_2 = 0;
                unsigned int before_2 = 0;
                unsigned int ends_2 = 0;
                #pragma unroll 1
                for (int s_11 = 0; s_11 < num_segments; s_11++) {
                    int q_len_10 = cu_seqlens_q[s_11 + 1] - cu_seqlens_q[s_11];
                    int k_len_10 = cu_seqlens_k[s_11 + 1] - cu_seqlens_k[s_11];
                    int row0_10 = ((q_len_10 + 255 >> 8) - 1) * 256;
                    int diag_10 = k_len_10 - q_len_10;
                    int n_k_10 = k_len_10 + 127 >> 7;
                    int n_c_10 = row0_10 + 256 + diag_10 + 127 >> 7;
                    int n_seen_10 = ((n_c_10 < n_k_10) ? n_c_10 : n_k_10);
                    int n_blocks_10 = ((n_seen_10 > 0) ? n_seen_10 : 0);
                    int first_seen_10 = row0_10 + diag_10 + 1;
                    int n_first_10 = ((first_seen_10 > 0) ? first_seen_10 >> 7 : 0);
                    int k_full_10 = k_len_10 >> 7;
                    int n_kf_10 = ((n_first_10 < k_full_10) ? n_first_10 : k_full_10);
                    int n_full_10 = ((n_kf_10 < n_blocks_10) ? n_kf_10 : n_blocks_10);
                    int n_masked_10 = n_blocks_10 - n_full_10;
                    bool in_class_2 = ((2 * n_blocks_10 >= wmax_2) ? 0 : ((4 * n_blocks_10 >= wmax_2) ? 1 : ((8 * n_blocks_10 >= wmax_2) ? 2 : 3))) == c_5;
                    ends_2 = ends_2 + ((in_class_2) ? (unsigned int)(cu_seqlens_q[s_11 + 1] - cu_seqlens_q[s_11] + 255) >> 8 : 0);
                    seg_2 = seg_2 + ((ends_2 <= j_2) ? 1 : 0);
                    before_2 = ((ends_2 <= j_2) ? ends_2 : before_2);
                }
                unsigned int m_rel_2 = j_2 - before_2;
                int q0_2 = cu_seqlens_q[seg_2];
                int q_len_11 = cu_seqlens_q[seg_2 + 1] - q0_2;
                int k0_2 = cu_seqlens_k[seg_2];
                int k_len_11 = cu_seqlens_k[seg_2 + 1] - k0_2;
                int m_2 = (int)(((unsigned int)(q_len_11 + 255) >> 8) - 1 - m_rel_2);
                int head_2 = (int)head_u_2;
                int kvh_2 = (int)kvh_u_2;
                int row0_11 = m_2 * 256;
                int diag_11 = k_len_11 - q_len_11;
                int n_k_11 = k_len_11 + 127 >> 7;
                int n_c_11 = row0_11 + 256 + diag_11 + 127 >> 7;
                int n_seen_11 = ((n_c_11 < n_k_11) ? n_c_11 : n_k_11);
                int n_blocks_11 = ((n_seen_11 > 0) ? n_seen_11 : 0);
                int first_seen_11 = row0_11 + diag_11 + 1;
                int n_first_11 = ((first_seen_11 > 0) ? first_seen_11 >> 7 : 0);
                int k_full_11 = k_len_11 >> 7;
                int n_kf_11 = ((n_first_11 < k_full_11) ? n_first_11 : k_full_11);
                int n_full_11 = ((n_kf_11 < n_blocks_11) ? n_kf_11 : n_blocks_11);
                int n_masked_11 = n_blocks_11 - n_full_11;
                if (cta_rank == 0) {
                    int has_block_1 = ((n_blocks_11 > 0) ? 1 : 0);
                    mbarrier_wait_cluster_hint(q_full_addr, _phase_q_full, 10000000);
                    #pragma unroll 1
                    for (int __1 = 0; __1 < has_block_1; __1++) {
                        mbarrier_wait(kv_full_addr + (kv) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                        int a_hi = ((smem_q_hi_addr) >> 4) & 0x3FFF;
                        int a_lo = ((smem_q_lo_addr) >> 4) & 0x3FFF;
                        int b_hi = (((smem_k_hi_addr) >> 4) & 0x3FFF) + (kv) * 1024;
                        int b_lo = (((smem_k_lo_addr) >> 4) & 0x3FFF) + (kv) * 1024;
#else
                        int a_hi = ((smem_q_hi_addr) >> 4) & 0x7FFF;
                        int a_lo = ((smem_q_lo_addr) >> 4) & 0x7FFF;
                        int b_hi = (((smem_k_hi_addr) >> 4) & 0x7FFF) + (kv) * 1024;
                        int b_lo = (((smem_k_lo_addr) >> 4) & 0x7FFF) + (kv) * 1024;
#endif
                        uint32_t _elected_0 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
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
                    :: "r"(a_hi), "r"(b_hi), "r"((tmem_s + (sq_1 * 128))), "r"(0), "r"(_elected_0) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
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
                    :: "r"(a_lo), "r"(b_lo), "r"((tmem_s + (sq_1 * 128))), "r"(1), "r"(_elected_0) : "memory");
                        tcgen05_commit_cg2_multicast_leader(s_full_addr + (sq_1) * 8, (uint16_t)(3), _elected_0);
                        tcgen05_commit_cg2_multicast_leader(kv_empty_addr + (kv) * 8, (uint16_t)(3), _elected_0);
                        kv += 1;
                        if (kv == 5) { kv = 0; _phase_kv_full ^= 1; }
                        sq_1 += 1;
                        if (sq_1 == 2) { sq_1 = 0; }
                    }
                    #pragma unroll 1
                    for (int i_3 = 1; i_3 < n_blocks_11; i_3++) {
                        mbarrier_wait(kv_full_addr + (kv) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                        int a_hi_1 = ((smem_q_hi_addr) >> 4) & 0x3FFF;
                        int a_lo_1 = ((smem_q_lo_addr) >> 4) & 0x3FFF;
                        int b_hi_1 = (((smem_k_hi_addr) >> 4) & 0x3FFF) + (kv) * 1024;
                        int b_lo_1 = (((smem_k_lo_addr) >> 4) & 0x3FFF) + (kv) * 1024;
#else
                        int a_hi_1 = ((smem_q_hi_addr) >> 4) & 0x7FFF;
                        int a_lo_1 = ((smem_q_lo_addr) >> 4) & 0x7FFF;
                        int b_hi_1 = (((smem_k_hi_addr) >> 4) & 0x7FFF) + (kv) * 1024;
                        int b_lo_1 = (((smem_k_lo_addr) >> 4) & 0x7FFF) + (kv) * 1024;
#endif
                        uint32_t _elected_1 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
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
                    :: "r"(a_hi_1), "r"(b_hi_1), "r"((tmem_s + (sq_1 * 128))), "r"(0), "r"(_elected_1) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
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
                    :: "r"(a_lo_1), "r"(b_lo_1), "r"((tmem_s + (sq_1 * 128))), "r"(1), "r"(_elected_1) : "memory");
                        tcgen05_commit_cg2_multicast_leader(s_full_addr + (sq_1) * 8, (uint16_t)(3), _elected_1);
                        tcgen05_commit_cg2_multicast_leader(kv_empty_addr + (kv) * 8, (uint16_t)(3), _elected_1);
                        kv += 1;
                        if (kv == 5) { kv = 0; _phase_kv_full ^= 1; }
                        sq_1 += 1;
                        if (sq_1 == 2) { sq_1 = 0; }
                        int first = ((i_3 == 1) ? 1 : 0);
                        mbarrier_wait(kv_full_addr + (kv) * 8, _phase_kv_full);
                        mbarrier_wait(p_full_addr + (pq_2) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                        int b_hi_0 = ((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv) * 1024;
                        int b_lo_0 = ((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv) * 1024;
#else
                        int b_hi_0 = ((((smem_v_hi_addr) >> 4) & 0x7FFF) | 0x4000000) + (kv) * 1024;
                        int b_lo_0 = ((((smem_v_lo_addr) >> 4) & 0x7FFF) | 0x4000000) + (kv) * 1024;
#endif
                        if (elect_sync()) {
                            if (pq_2 == 0) {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_0), "r"(tmem_p0), "r"(((first) ? 0 : 1)) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_0), "r"(tmem_p0), "r"(((first) ? 0 : 1)) : "memory");
                            } else {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_0), "r"(tmem_p1), "r"(((first) ? 0 : 1)) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_0), "r"(tmem_p1), "r"(((first) ? 0 : 1)) : "memory");
                            }
                        }
                        mbarrier_wait(p_full_2_addr + (pq_2) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (pq_2 == 0) {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_0), "r"(tmem_p0), "r"(1) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_0), "r"(tmem_p0), "r"(1) : "memory");
                            } else {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_0), "r"(tmem_p1), "r"(1) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_0), "r"(tmem_p1), "r"(1) : "memory");
                            }
                            tcgen05_commit_cg2_multicast(o_full_addr + (oq_1) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(kv_empty_addr + (kv) * 8, (uint16_t)(3));
                        }
                        kv += 1;
                        if (kv == 5) { kv = 0; _phase_kv_full ^= 1; }
                        pq_2 += 1;
                        if (pq_2 == 2) { pq_2 = 0; _phase_p_full ^= 1; }
                        oq_1 += 1;
                        if (oq_1 == 2) { oq_1 = 0; }
                    }
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    _phase_q_full ^= 1;
                    int last_first = ((n_blocks_11 == 1) ? 1 : 0);
                    #pragma unroll 1
                    for (int __2 = 0; __2 < has_block_1; __2++) {
                        mbarrier_wait(kv_full_addr + (kv) * 8, _phase_kv_full);
                        mbarrier_wait(p_full_addr + (pq_2) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
                        int b_hi_2 = ((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv) * 1024;
                        int b_lo_2 = ((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv) * 1024;
#else
                        int b_hi_2 = ((((smem_v_hi_addr) >> 4) & 0x7FFF) | 0x4000000) + (kv) * 1024;
                        int b_lo_2 = ((((smem_v_lo_addr) >> 4) & 0x7FFF) | 0x4000000) + (kv) * 1024;
#endif
                        if (elect_sync()) {
                            if (pq_2 == 0) {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_2), "r"(tmem_p0), "r"(((last_first) ? 0 : 1)) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_2), "r"(tmem_p0), "r"(((last_first) ? 0 : 1)) : "memory");
                            } else {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_2), "r"(tmem_p1), "r"(((last_first) ? 0 : 1)) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_2), "r"(tmem_p1), "r"(((last_first) ? 0 : 1)) : "memory");
                            }
                        }
                        mbarrier_wait(p_full_2_addr + (pq_2) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (pq_2 == 0) {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_2), "r"(tmem_p0), "r"(1) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_2), "r"(tmem_p0), "r"(1) : "memory");
                            } else {
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_hi), "r"(b_hi_2), "r"(tmem_p1), "r"(1) : "memory");
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_o_lo), "r"(b_lo_2), "r"(tmem_p1), "r"(1) : "memory");
                            }
                            tcgen05_commit_cg2_multicast(kv_empty_addr + (kv) * 8, (uint16_t)(3));
                        }
                        kv += 1;
                        if (kv == 5) { kv = 0; _phase_kv_full ^= 1; }
                        pq_2 += 1;
                        if (pq_2 == 2) { pq_2 = 0; _phase_p_full ^= 1; }
                    }
                    mbarrier_wait(o_free_addr, _phase_o_free);
                    elect_commit_cg2_multicast(o_full_addr + (oq_1) * 8, (uint16_t)(3));
                    oq_1 += 1;
                    if (oq_1 == 2) { oq_1 = 0; }
                    _phase_o_free ^= 1;
                }
            }
            if (cta_rank == 0) {
                mbarrier_wait(o_free_addr, _phase_o_free);
            }
        }
    }
    // ---- Role: load ----
    if (warp == 9) {
        { // load_main
            unsigned int wk_3 = 0;
            unsigned int kv_1 = 0;
            unsigned int head_tiles_3 = 0;
            #pragma unroll 1
            for (int s_12 = 0; s_12 < num_segments; s_12++) {
                head_tiles_3 = head_tiles_3 + ((unsigned int)(cu_seqlens_q[s_12 + 1] - cu_seqlens_q[s_12] + 255) >> 8);
            }
            int wmax_3 = 0;
            #pragma unroll 1
            for (int s_13 = 0; s_13 < num_segments; s_13++) {
                int q_len_12 = cu_seqlens_q[s_13 + 1] - cu_seqlens_q[s_13];
                int k_len_12 = cu_seqlens_k[s_13 + 1] - cu_seqlens_k[s_13];
                int row0_12 = ((q_len_12 + 255 >> 8) - 1) * 256;
                int diag_12 = k_len_12 - q_len_12;
                int n_k_12 = k_len_12 + 127 >> 7;
                int n_c_12 = row0_12 + 256 + diag_12 + 127 >> 7;
                int n_seen_12 = ((n_c_12 < n_k_12) ? n_c_12 : n_k_12);
                int n_blocks_12 = ((n_seen_12 > 0) ? n_seen_12 : 0);
                int first_seen_12 = row0_12 + diag_12 + 1;
                int n_first_12 = ((first_seen_12 > 0) ? first_seen_12 >> 7 : 0);
                int k_full_12 = k_len_12 >> 7;
                int n_kf_12 = ((n_first_12 < k_full_12) ? n_first_12 : k_full_12);
                int n_full_12 = ((n_kf_12 < n_blocks_12) ? n_kf_12 : n_blocks_12);
                int n_masked_12 = n_blocks_12 - n_full_12;
                int w_3 = n_blocks_12;
                wmax_3 = ((w_3 > wmax_3) ? w_3 : wmax_3);
            }
            unsigned int t0_3 = 0;
            unsigned int t1_3 = 0;
            unsigned int t2_3 = 0;
            #pragma unroll 1
            for (int s_14 = 0; s_14 < num_segments; s_14++) {
                unsigned int tiles_3 = (unsigned int)(cu_seqlens_q[s_14 + 1] - cu_seqlens_q[s_14] + 255) >> 8;
                int q_len_13 = cu_seqlens_q[s_14 + 1] - cu_seqlens_q[s_14];
                int k_len_13 = cu_seqlens_k[s_14 + 1] - cu_seqlens_k[s_14];
                int row0_13 = ((q_len_13 + 255 >> 8) - 1) * 256;
                int diag_13 = k_len_13 - q_len_13;
                int n_k_13 = k_len_13 + 127 >> 7;
                int n_c_13 = row0_13 + 256 + diag_13 + 127 >> 7;
                int n_seen_13 = ((n_c_13 < n_k_13) ? n_c_13 : n_k_13);
                int n_blocks_13 = ((n_seen_13 > 0) ? n_seen_13 : 0);
                int first_seen_13 = row0_13 + diag_13 + 1;
                int n_first_13 = ((first_seen_13 > 0) ? first_seen_13 >> 7 : 0);
                int k_full_13 = k_len_13 >> 7;
                int n_kf_13 = ((n_first_13 < k_full_13) ? n_first_13 : k_full_13);
                int n_full_13 = ((n_kf_13 < n_blocks_13) ? n_kf_13 : n_blocks_13);
                int n_masked_13 = n_blocks_13 - n_full_13;
                int c_6 = ((2 * n_blocks_13 >= wmax_3) ? 0 : ((4 * n_blocks_13 >= wmax_3) ? 1 : ((8 * n_blocks_13 >= wmax_3) ? 2 : 3)));
                t0_3 = t0_3 + ((c_6 == 0) ? tiles_3 : 0);
                t1_3 = t1_3 + ((c_6 == 1) ? tiles_3 : 0);
                t2_3 = t2_3 + ((c_6 == 2) ? tiles_3 : 0);
            }
            unsigned int total_3 = (unsigned int)num_heads_q * head_tiles_3;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int it_3 = 0; it_3 < (unsigned int)num_heads_q * ((unsigned int)(total_q + 255 >> 8) + (unsigned int)num_segments) + 1; it_3++) {
                mbarrier_wait_cluster_hint(work_full_addr + (wk_3) * 8, _phase_work_full_3, 10000000);
                unsigned int ticket_3 = cells[wk_3];
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wk_3) * 8), "r"(0) : "memory");
                wk_3 += 1;
                if (wk_3 == 2) { wk_3 = 0; _phase_work_full_3 ^= 1; }
                if (ticket_3 >= total_3) {
                    break;
                }
                unsigned int hq_3 = (unsigned int)num_heads_q;
                unsigned int p1_3 = hq_3 * t0_3;
                unsigned int p2_3 = p1_3 + hq_3 * t1_3;
                unsigned int p3_3 = p2_3 + hq_3 * t2_3;
                int c_7 = ((ticket_3 < p1_3) ? 0 : ((ticket_3 < p2_3) ? 1 : ((ticket_3 < p3_3) ? 2 : 3)));
                unsigned int base_3 = ((c_7 == 0) ? 0 : ((c_7 == 1) ? p1_3 : ((c_7 == 2) ? p2_3 : p3_3)));
                unsigned int tc_3 = ((c_7 == 0) ? t0_3 : ((c_7 == 1) ? t1_3 : ((c_7 == 2) ? t2_3 : head_tiles_3 - t0_3 - t1_3 - t2_3)));
                unsigned int r_3 = ticket_3 - base_3;
                unsigned int per_kv_3 = tc_3 * (unsigned int)group;
                unsigned int kvh_u_3 = r_3 / per_kv_3;
                unsigned int rem_3 = r_3 - kvh_u_3 * per_kv_3;
                unsigned int j_3 = rem_3 / (unsigned int)group;
                unsigned int head_u_3 = kvh_u_3 * (unsigned int)group + (rem_3 - j_3 * (unsigned int)group);
                int seg_3 = 0;
                unsigned int before_3 = 0;
                unsigned int ends_3 = 0;
                #pragma unroll 1
                for (int s_15 = 0; s_15 < num_segments; s_15++) {
                    int q_len_14 = cu_seqlens_q[s_15 + 1] - cu_seqlens_q[s_15];
                    int k_len_14 = cu_seqlens_k[s_15 + 1] - cu_seqlens_k[s_15];
                    int row0_14 = ((q_len_14 + 255 >> 8) - 1) * 256;
                    int diag_14 = k_len_14 - q_len_14;
                    int n_k_14 = k_len_14 + 127 >> 7;
                    int n_c_14 = row0_14 + 256 + diag_14 + 127 >> 7;
                    int n_seen_14 = ((n_c_14 < n_k_14) ? n_c_14 : n_k_14);
                    int n_blocks_14 = ((n_seen_14 > 0) ? n_seen_14 : 0);
                    int first_seen_14 = row0_14 + diag_14 + 1;
                    int n_first_14 = ((first_seen_14 > 0) ? first_seen_14 >> 7 : 0);
                    int k_full_14 = k_len_14 >> 7;
                    int n_kf_14 = ((n_first_14 < k_full_14) ? n_first_14 : k_full_14);
                    int n_full_14 = ((n_kf_14 < n_blocks_14) ? n_kf_14 : n_blocks_14);
                    int n_masked_14 = n_blocks_14 - n_full_14;
                    bool in_class_3 = ((2 * n_blocks_14 >= wmax_3) ? 0 : ((4 * n_blocks_14 >= wmax_3) ? 1 : ((8 * n_blocks_14 >= wmax_3) ? 2 : 3))) == c_7;
                    ends_3 = ends_3 + ((in_class_3) ? (unsigned int)(cu_seqlens_q[s_15 + 1] - cu_seqlens_q[s_15] + 255) >> 8 : 0);
                    seg_3 = seg_3 + ((ends_3 <= j_3) ? 1 : 0);
                    before_3 = ((ends_3 <= j_3) ? ends_3 : before_3);
                }
                unsigned int m_rel_3 = j_3 - before_3;
                int q0_3 = cu_seqlens_q[seg_3];
                int q_len_15 = cu_seqlens_q[seg_3 + 1] - q0_3;
                int k0_3 = cu_seqlens_k[seg_3];
                int k_len_15 = cu_seqlens_k[seg_3 + 1] - k0_3;
                int m_3 = (int)(((unsigned int)(q_len_15 + 255) >> 8) - 1 - m_rel_3);
                int head_3 = (int)head_u_3;
                int kvh_3 = (int)kvh_u_3;
                int row0_15 = m_3 * 256;
                int diag_15 = k_len_15 - q_len_15;
                int n_k_15 = k_len_15 + 127 >> 7;
                int n_c_15 = row0_15 + 256 + diag_15 + 127 >> 7;
                int n_seen_15 = ((n_c_15 < n_k_15) ? n_c_15 : n_k_15);
                int n_blocks_15 = ((n_seen_15 > 0) ? n_seen_15 : 0);
                int first_seen_15 = row0_15 + diag_15 + 1;
                int n_first_15 = ((first_seen_15 > 0) ? first_seen_15 >> 7 : 0);
                int k_full_15 = k_len_15 >> 7;
                int n_kf_15 = ((n_first_15 < k_full_15) ? n_first_15 : k_full_15);
                int n_full_15 = ((n_kf_15 < n_blocks_15) ? n_kf_15 : n_blocks_15);
                int n_masked_15 = n_blocks_15 - n_full_15;
                int has_block_2 = ((n_blocks_15 > 0) ? 1 : 0);
                int q_row = q0_3 + m_3 * 256 + cta_rank * 128;
                mbarrier_wait(q_empty_addr, _phase_q_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_q_hi_addr, (&Q), 0, q_row, head_3, 0, ((q_full_addr) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(smem_q_lo_addr, (&Q), 0, q_row, head_3, 2, ((q_full_addr) & 0xFEFFFFFF));
                }
                _phase_q_empty ^= 1;
                int k_row = k0_3 + cta_rank * 64;
                #pragma unroll 1
                for (int __3 = 0; __3 < has_block_2; __3++) {
                    mbarrier_wait(kv_empty_addr + (kv_1) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_k_hi_addr + kv_1 * 16384, (&K), 0, k_row + (n_blocks_15 - 1) * 128, kvh_3, 0, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_k_lo_addr + kv_1 * 16384, (&K), 0, k_row + (n_blocks_15 - 1) * 128, kvh_3, 2, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                    }
                    kv_1 += 1;
                    if (kv_1 == 5) { kv_1 = 0; _phase_kv_empty ^= 1; }
                }
                #pragma unroll 1
                for (int i_4 = 1; i_4 < n_blocks_15; i_4++) {
                    mbarrier_wait(kv_empty_addr + (kv_1) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_k_hi_addr + kv_1 * 16384, (&K), 0, k_row + (n_blocks_15 - 1 - i_4) * 128, kvh_3, 0, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_k_lo_addr + kv_1 * 16384, (&K), 0, k_row + (n_blocks_15 - 1 - i_4) * 128, kvh_3, 2, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                    }
                    kv_1 += 1;
                    if (kv_1 == 5) { kv_1 = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (kv_1) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_v_hi_addr + kv_1 * 16384, (&V), 0, k0_3 + (n_blocks_15 - i_4) * 128, cta_rank, kvh_3, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_v_lo_addr + kv_1 * 16384, (&V), 0, k0_3 + (n_blocks_15 - i_4) * 128, cta_rank + 2, kvh_3, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                    }
                    kv_1 += 1;
                    if (kv_1 == 5) { kv_1 = 0; _phase_kv_empty ^= 1; }
                }
                #pragma unroll 1
                for (int __4 = 0; __4 < has_block_2; __4++) {
                    mbarrier_wait(kv_empty_addr + (kv_1) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_v_hi_addr + kv_1 * 16384, (&V), 0, k0_3, cta_rank, kvh_3, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_v_lo_addr + kv_1 * 16384, (&V), 0, k0_3, cta_rank + 2, kvh_3, ((kv_full_addr + (kv_1) * 8) & 0xFEFFFFFF));
                    }
                    kv_1 += 1;
                    if (kv_1 == 5) { kv_1 = 0; _phase_kv_empty ^= 1; }
                }
            }
            #pragma unroll 1
            for (int __5 = 0; __5 < 5; __5++) {
                mbarrier_wait(kv_empty_addr + (kv_1) * 8, _phase_kv_empty);
                kv_1 += 1;
                if (kv_1 == 5) { kv_1 = 0; _phase_kv_empty ^= 1; }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty);
        }
    }
    // ---- Role: sched ----
    if (warp == 10) {
        { // sched_main
            unsigned int wk_4 = 0;
            unsigned int head_tiles_4 = 0;
            #pragma unroll 1
            for (int s_16 = 0; s_16 < num_segments; s_16++) {
                head_tiles_4 = head_tiles_4 + ((unsigned int)(cu_seqlens_q[s_16 + 1] - cu_seqlens_q[s_16] + 255) >> 8);
            }
            unsigned int total_4 = (unsigned int)num_heads_q * head_tiles_4;
            unsigned int _phase_work_empty = 1;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int it_4 = 0; it_4 < (unsigned int)num_heads_q * ((unsigned int)(total_q + 255 >> 8) + (unsigned int)num_segments) + 1; it_4++) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (wk_4) * 8, _phase_work_empty, 10000000);
                        unsigned int ticket_4 = atomicAdd(tile_counter, 1);
                        cells[wk_4] = ticket_4;
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (wk_4) * 8), "r"(1), "r"((uint32_t)(4)) : "memory");
                        uint32_t _cluster_mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_0) : "r"(cells_addr + wk_4 * 4), "r"(1));
                        uint32_t _mbar_peer_1;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_1) : "r"(work_full_addr + (wk_4) * 8), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_0), "r"(ticket_4), "r"(_mbar_peer_1) : "memory");
                        mbarrier_arrive(work_full_addr + (wk_4) * 8);
                        wk_4 += 1;
                        if (wk_4 == 2) { wk_4 = 0; _phase_work_empty ^= 1; }
                        if (ticket_4 >= total_4) {
                            if (ticket_4 == total_4 + (unsigned int)num_clusters - 1) {
                                *(reinterpret_cast<unsigned int*>(tile_counter) + (0)) = 0;
                            }
                            break;
                        }
                    }
                }
            }
        }
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
