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
#define TMEM_NCOLS 480
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_0_OFFSET 128
#define TMEM_OUTPUT_1_OFFSET 256
#define TMEM_TMEM_SFA_QK0_OFFSET 384
#define TMEM_TMEM_SFB_QK0_OFFSET 392
#define TMEM_TMEM_SFA_QK1_OFFSET 400
#define TMEM_PROBS_0_OFFSET 416
#define TMEM_PROBS_1_OFFSET 448
#define NUM_V_PIPE_STAGES 4
#define NUM_K_PIPE_STAGES 4
#define SMEM_O_STAGE_OFF 74752
#define SMEM_O_STAGE_STAGE_BYTES 69632
#define SMEM_O_STAGE_STRIDE 69632
#define SMEM_ROW_STATE_OFF 1024
#define SMEM_ROW_STATE_STAGE_BYTES 2048
#define SMEM_ROW_STATE_STRIDE 2048
#define SMEM_SMEM_Q_OFF 3072
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_SFQ_OFF 19456
#define SMEM_SMEM_SFQ_STAGE_BYTES 1024
#define SMEM_SMEM_SFQ_STRIDE 1024
#define SMEM_SMEM_K_OFF 21504
#define SMEM_SMEM_K_STAGE_BYTES 4096
#define SMEM_SMEM_K_STRIDE 4096
#define SMEM_SMEM_SFK_OFF 37888
#define SMEM_SMEM_SFK_STAGE_BYTES 1024
#define SMEM_SMEM_SFK_STRIDE 1024
#define SMEM_SMEM_V_OFF 41984
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 8192
#define SMEM_TOTAL 144384
#define THREADS 512
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}



__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo256(int lo) {
    const int SBO = 256;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}



__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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
kernel_cake_minimax_h3_varlen_attention_89f32e4786d93ee3b7bb(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap SFQ, const __grid_constant__ CUtensorMap SFK, __nv_bfloat16* __restrict__ O, float* __restrict__ v_amax, int* __restrict__ cl_head, int* __restrict__ cl_seg_begin, int* __restrict__ cl_seg_len, int* __restrict__ cl_kv_base, int* __restrict__ cl_q_block, int* __restrict__ cl_kv_begin, int* __restrict__ cl_kv_blocks, int* __restrict__ cl_ws_slot, __half* __restrict__ partial_O, float* __restrict__ partial_ML, unsigned int num_tiles, int heads, int PB, float softmax_scale_log2)
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
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 56)
    #define v_full_addr (mbar_base + 88)
    #define v_empty_addr (mbar_base + 120)
    #define s_full_addr (mbar_base + 152)
    #define s_empty_addr (mbar_base + 168)
    #define p_full_addr (mbar_base + 184)
    #define p_empty_addr (mbar_base + 200)
    #define corr_sig_lo_addr (mbar_base + 216)
    #define corr_sig_hi_addr (mbar_base + 232)
    #define corr_done_addr (mbar_base + 248)
    #define o_full_addr (mbar_base + 264)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    unsigned int* o_stage = reinterpret_cast<unsigned int*>(smem_raw + 74752);
    const int o_stage_addr = smem + 74752;
    float* row_state = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_state_addr = smem + 1024;
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_q_addr = smem + 3072;
    uint8_t* smem_sfq = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_sfq_addr = smem + 19456;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_k_addr = smem + 21504;
    uint8_t* smem_sfk = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_sfk_addr = smem + 37888;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_v_addr = smem + 41984;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[0..280)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 4 barriers, init_count=2
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // v_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            mbarrier_init(smem + 176, 256);
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 184, 512);
            mbarrier_init(smem + 192, 512);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // corr_sig_lo: 2 barriers, init_count=64
            mbarrier_init(smem + 216, 64);
            mbarrier_init(smem + 224, 64);
            // corr_sig_hi: 2 barriers, init_count=64
            mbarrier_init(smem + 232, 64);
            mbarrier_init(smem + 240, 64);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 248, 128);
            mbarrier_init(smem + 256, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 280);
    if (warp == 0) {
        int _tmem_hold = smem + 280;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output_0 = taddr + 128;
    const int tmem_output_1 = taddr + 256;
    const int tmem_tmem_sfa_qk0 = taddr + 384;
    const int tmem_tmem_sfb_qk0 = taddr + 392;
    const int tmem_tmem_sfa_qk1 = taddr + 400;
    const int tmem_probs_0 = taddr + 416;
    const int tmem_probs_1 = taddr + 448;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            unsigned int total_tiles = num_tiles;
            unsigned int stage = make_warp_uniform(warp / 4);
            int scale_off = make_warp_uniform(stage * 128);
            int p_col = make_warp_uniform(416 + stage * 32);
            int _phase_corr_done = 1;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty = 1;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int head = cl_head[tile_idx];
                int seg_begin = cl_seg_begin[tile_idx];
                int seg_len = cl_seg_len[tile_idx];
                int kv_base = cl_kv_base[tile_idx];
                int q_word = cl_q_block[tile_idx];
                int q_base = q_word & 65535;
                int q_half = q_word >> 16;
                int two_stages = ((q_half == 0) ? 1 : 0);
                int m_block = q_base + (q_half >> 1) * 2 + cta_rank * (1 + two_stages);
                unsigned int num_n_blocks = cl_kv_blocks[tile_idx];
                int kv_begin = cl_kv_begin[tile_idx];
                int ws_slot = cl_ws_slot[tile_idx];
                int active = ((stage == 0) ? 1 : two_stages);
                if (active == 1) {
                    int tail_base = seg_len - kv_begin * 128;
                    int kv_len = num_n_blocks * 128;
                    unsigned int n_count = (unsigned int)((kv_len + 128 - 1) / 128);
                    float row_max = -CUDART_INF_F;
                    float row_max_scaled = 0.0f;
                    float row_sum = 0.0f;
                    #pragma unroll 1
                    for (unsigned int n_iter = 0; n_iter < n_count; n_iter++) {
                        int n_block = n_count - 1 - n_iter;
                        mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 10000000);
                        _phase_s_full ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int s_addr = taddr + (unsigned int)0 + (unsigned int)(warp % 4 * 32 << 16);
                        float sv[128];
                        float tile_max = -CUDART_INF_F;
                        {
#if __CUDA_ARCH__ == 1030
                            float lo_max = -CUDART_INF_F;
                            float hi_max = -CUDART_INF_F;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#else
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1030
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(lo_max)
                                : "r"(tmem_scores + (warp % 4 * 32 << 16)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
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
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(hi_max)
                                : "r"(tmem_scores + 64 + (warp % 4 * 32 << 16)));
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
#endif
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                            tile_max = max_noftz(lo_max, hi_max);
#else
                            float2 _reg_reduce_max2_1 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_1);
                            float _reg_array_reduce_0 = row_max_reduce(_reg_reduce_max2_1);
                            tile_max = _reg_array_reduce_0;
#endif
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((s_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        int tail_valid = tail_base - n_block * 128;
                        if (tail_valid < 128) {
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
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * softmax_scale_log2;
                        float acc_scale_log2 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                        float acc_scale;
                        if (acc_scale_log2 >= -8.0f) {
                            acc_scale = 1.0f;
                            new_max_scaled = row_max_scaled;
                        } else {
#if __CUDA_ARCH__ == 1030
                            float _exp2_10 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_10 : 1.0f);
#else
                            float _exp2_12 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_12 : 1.0f);
#endif
                            row_max = new_max;
                            row_max_scaled = new_max_scaled;
                        }
                        if (n_iter > 0) {
                            mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                        }
                        row_state[warp % 4 * 32 + lane + scale_off] = acc_scale;
                        if (warp % 4 < 2) {
                            mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                        } else {
                            mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                        }
                        float block_sum = 0.0f;
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_11 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_12 = {-new_max_scaled, -new_max_scaled};
#else
                        const float2 _fma_b2_13 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_14 = {-new_max_scaled, -new_max_scaled};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 64; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_lf], _fma_b2_11, _fma_c2_12);
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_lf], _fma_b2_13, _fma_c2_14);
#endif
                        {
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 128; _le++) {
#else
                            for (int _le = 0; _le < 96; _le++) {
#endif
                                sv[_le] = approx_exp2(sv[_le]);
                            }
#if __CUDA_ARCH__ == 1030
#else
                            float2 _exp2_pair_15 = ex2_emulation_f32x2_value(make_float2(sv[96], sv[97]));
                            sv[96] = _exp2_pair_15.x;
                            sv[97] = _exp2_pair_15.y;
                            float2 _exp2_pair_16 = ex2_emulation_f32x2_value(make_float2(sv[98], sv[99]));
                            sv[98] = _exp2_pair_16.x;
                            sv[99] = _exp2_pair_16.y;
                            float2 _exp2_pair_17 = ex2_emulation_f32x2_value(make_float2(sv[100], sv[101]));
                            sv[100] = _exp2_pair_17.x;
                            sv[101] = _exp2_pair_17.y;
                            float2 _exp2_pair_18 = ex2_emulation_f32x2_value(make_float2(sv[102], sv[103]));
                            sv[102] = _exp2_pair_18.x;
                            sv[103] = _exp2_pair_18.y;
                            float2 _exp2_pair_19 = ex2_emulation_f32x2_value(make_float2(sv[104], sv[105]));
                            sv[104] = _exp2_pair_19.x;
                            sv[105] = _exp2_pair_19.y;
                            float2 _exp2_pair_20 = ex2_emulation_f32x2_value(make_float2(sv[106], sv[107]));
                            sv[106] = _exp2_pair_20.x;
                            sv[107] = _exp2_pair_20.y;
                            float2 _exp2_pair_21 = ex2_emulation_f32x2_value(make_float2(sv[108], sv[109]));
                            sv[108] = _exp2_pair_21.x;
                            sv[109] = _exp2_pair_21.y;
                            float2 _exp2_pair_22 = ex2_emulation_f32x2_value(make_float2(sv[110], sv[111]));
                            sv[110] = _exp2_pair_22.x;
                            sv[111] = _exp2_pair_22.y;
                            float2 _exp2_pair_23 = ex2_emulation_f32x2_value(make_float2(sv[112], sv[113]));
                            sv[112] = _exp2_pair_23.x;
                            sv[113] = _exp2_pair_23.y;
                            float2 _exp2_pair_24 = ex2_emulation_f32x2_value(make_float2(sv[114], sv[115]));
                            sv[114] = _exp2_pair_24.x;
                            sv[115] = _exp2_pair_24.y;
                            float2 _exp2_pair_25 = ex2_emulation_f32x2_value(make_float2(sv[116], sv[117]));
                            sv[116] = _exp2_pair_25.x;
                            sv[117] = _exp2_pair_25.y;
                            float2 _exp2_pair_26 = ex2_emulation_f32x2_value(make_float2(sv[118], sv[119]));
                            sv[118] = _exp2_pair_26.x;
                            sv[119] = _exp2_pair_26.y;
                            float2 _exp2_pair_27 = ex2_emulation_f32x2_value(make_float2(sv[120], sv[121]));
                            sv[120] = _exp2_pair_27.x;
                            sv[121] = _exp2_pair_27.y;
                            float2 _exp2_pair_28 = ex2_emulation_f32x2_value(make_float2(sv[122], sv[123]));
                            sv[122] = _exp2_pair_28.x;
                            sv[123] = _exp2_pair_28.y;
                            float2 _exp2_pair_29 = ex2_emulation_f32x2_value(make_float2(sv[124], sv[125]));
                            sv[124] = _exp2_pair_29.x;
                            sv[125] = _exp2_pair_29.y;
                            float2 _exp2_pair_30 = ex2_emulation_f32x2_value(make_float2(sv[126], sv[127]));
                            sv[126] = _exp2_pair_30.x;
                            sv[127] = _exp2_pair_30.y;
#endif
                        }
                        int p_addr = taddr + (unsigned int)p_col + (unsigned int)(warp % 4 * 32 << 16);
                        unsigned int p0[8];
                        unsigned int p1[8];
                        unsigned int p2[8];
                        unsigned int p3[8];
                        float2 partial = make_float2(sv[0], sv[1]);
                        #pragma unroll
                        for (int pair = 2; pair < 32; pair += 2) {
                            partial = add_f32x2(partial, make_float2(sv[pair], sv[pair + 1]));
                        }
                        float2 sum0 = partial;
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
                            p0[0] = _packed;
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
                            p0[1] = _packed;
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
                            p0[2] = _packed;
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
                            p0[3] = _packed;
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
                            p0[4] = _packed;
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
                            p0[5] = _packed;
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
                            p0[6] = _packed;
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
                            p0[7] = _packed;
                        }
                        mbarrier_wait(p_empty_addr + (stage) * 8, _phase_p_empty);
                        _phase_p_empty ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        tmem_st_x8_u32(p_addr, (const uint32_t*)p0);
                        float2 partial_0 = make_float2(sv[32], sv[33]);
                        #pragma unroll
                        for (int pair_1 = 34; pair_1 < 64; pair_1 += 2) {
                            partial_0 = add_f32x2(partial_0, make_float2(sv[pair_1], sv[pair_1 + 1]));
                        }
                        float2 sum1 = partial_0;
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
                            p1[0] = _packed;
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
                            p1[1] = _packed;
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
                            p1[2] = _packed;
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
                            p1[3] = _packed;
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
                            p1[4] = _packed;
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
                            p1[5] = _packed;
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
                            p1[6] = _packed;
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
                            p1[7] = _packed;
                        }
                        tmem_st_x8_u32(p_addr + 8, (const uint32_t*)p1);
                        float2 partial_1 = make_float2(sv[64], sv[65]);
                        #pragma unroll
                        for (int pair_2 = 66; pair_2 < 96; pair_2 += 2) {
                            partial_1 = add_f32x2(partial_1, make_float2(sv[pair_2], sv[pair_2 + 1]));
                        }
                        float2 sum2 = partial_1;
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
                            p2[0] = _packed;
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
                            p2[1] = _packed;
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
                            p2[2] = _packed;
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
                            p2[3] = _packed;
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
                            p2[4] = _packed;
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
                            p2[5] = _packed;
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
                            p2[6] = _packed;
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
                            p2[7] = _packed;
                        }
                        tmem_st_x8_u32(p_addr + 16, (const uint32_t*)p2);
                        float2 partial_2 = make_float2(sv[96], sv[97]);
                        #pragma unroll
                        for (int pair_3 = 98; pair_3 < 128; pair_3 += 2) {
                            partial_2 = add_f32x2(partial_2, make_float2(sv[pair_3], sv[pair_3 + 1]));
                        }
                        float2 sum3 = partial_2;
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
                            p3[0] = _packed;
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
                            p3[1] = _packed;
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
                            p3[2] = _packed;
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
                            p3[3] = _packed;
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
                            p3[4] = _packed;
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
                            p3[5] = _packed;
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
                            p3[6] = _packed;
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
                            p3[7] = _packed;
                        }
                        tmem_st_x8_u32(p_addr + 24, (const uint32_t*)p3);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        float2 sum01 = add_f32x2(sum0, sum1);
                        float2 sum23 = add_f32x2(sum2, sum3);
                        float2 total_sum = add_f32x2(sum01, sum23);
                        block_sum = total_sum.x + total_sum.y;
                        row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
                        _phase_corr_done ^= 1;
                        if (n_iter == n_count - 1) {
                            mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (warp % 4 < 2) {
                        mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                    } else {
                        mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                    }
                    float v_amax_reg[1];
                    {
#if __CUDA_ARCH__ == 1030
                        uint32_t _scalar_bits_13;
#else
                        uint32_t _scalar_bits_31;
#endif
                        asm volatile("ld.volatile.global.b32 %0, [%1];"
#if __CUDA_ARCH__ == 1030
                            : "=r"(_scalar_bits_13) : "l"((const void*)(v_amax + (0))) : "memory");
                        v_amax_reg[0] = __uint_as_float(_scalar_bits_13);
#else
                            : "=r"(_scalar_bits_31) : "l"((const void*)(v_amax + (0))) : "memory");
                        v_amax_reg[0] = __uint_as_float(_scalar_bits_31);
#endif
                    }
                    mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                    _phase_o_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_scale = 0.0f;
                    float v_dequant = v_amax_reg[0] * 0.002232143f;
#if __CUDA_ARCH__ == 1030
                    float _rcp_14 = approx_rcp(row_sum);
                    final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_14 * v_dequant : 0.0f);
#else
                    float _rcp_32 = approx_rcp(row_sum);
                    final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_32 * v_dequant : 0.0f);
#endif
                    int seg_len_out = cl_seg_len[tile_idx];
                    int ws_slot_out = cl_ws_slot[tile_idx];
                    int local_row = ((unsigned int)m_block + stage) * 128 + (unsigned int)(warp % 4 * 32 + lane);
                    int token = seg_begin + local_row;
                    long long out_off = ((long long)token * (long long)heads + (long long)head) * 128;
                    int tmem_o_off = make_warp_uniform((unsigned int)128 + stage * 128);
                    int partial_row = ws_slot_out * 512 + local_row - (cl_q_block[tile_idx] & 65535) * 128;
                    if (ws_slot_out >= 0) {
                        if (local_row < seg_len_out) {
                            partial_ML[partial_row * 2] = row_max_scaled;
                            partial_ML[partial_row * 2 + 1] = row_sum;
                        }
                    }
                    unsigned int row_addr = o_stage_addr + ((unsigned int)scale_off + (unsigned int)(warp % 4 * 32 + lane)) * 272;
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    #pragma unroll
                    for (int part = 0; part < 2; part++) {
                        int o_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(part * 64);
                        float ov[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31])
                            : "r"(o_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[32]), "=f"(ov[33]), "=f"(ov[34]), "=f"(ov[35]), "=f"(ov[36]), "=f"(ov[37]), "=f"(ov[38]), "=f"(ov[39]), "=f"(ov[40]), "=f"(ov[41]), "=f"(ov[42]), "=f"(ov[43]), "=f"(ov[44]), "=f"(ov[45]), "=f"(ov[46]), "=f"(ov[47]), "=f"(ov[48]), "=f"(ov[49]), "=f"(ov[50]), "=f"(ov[51]), "=f"(ov[52]), "=f"(ov[53]), "=f"(ov[54]), "=f"(ov[55]), "=f"(ov[56]), "=f"(ov[57]), "=f"(ov[58]), "=f"(ov[59]), "=f"(ov[60]), "=f"(ov[61]), "=f"(ov[62]), "=f"(ov[63])
                            : "r"(o_addr + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (local_row < seg_len_out) {
                            if (ws_slot_out >= 0) {
                                float _scaled_0[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_15 = {final_scale, final_scale};
#else
                                const float2 _scale2_33 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(ov)[_ls], _scale2_15);
#else
                                    reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(ov)[_ls], _scale2_33);
#endif
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
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)(part * 4 * 32)), "r"(_packed_0[0]), "r"(_packed_0[1]), "r"(_packed_0[2]), "r"(_packed_0[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)(part * 4 * 32 + 16)), "r"(_packed_0[4]), "r"(_packed_0[5]), "r"(_packed_0[6]), "r"(_packed_0[7]) : "memory");
                            } else {
                                float _scaled_1[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_16 = {final_scale, final_scale};
#else
                                const float2 _scale2_34 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(ov)[_ls], _scale2_16);
#else
                                    reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(ov)[_ls], _scale2_34);
#endif
                                unsigned int _packed_1[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 2], _scaled_1[_lp*2+1 + 2]));
                                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 4], _scaled_1[_lp*2+1 + 4]));
                                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 6], _scaled_1[_lp*2+1 + 6]));
                                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 8], _scaled_1[_lp*2+1 + 8]));
                                    _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 10], _scaled_1[_lp*2+1 + 10]));
                                    _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 12], _scaled_1[_lp*2+1 + 12]));
                                    _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 14], _scaled_1[_lp*2+1 + 14]));
                                    _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)(part * 4 * 32)), "r"(_packed_1[0]), "r"(_packed_1[1]), "r"(_packed_1[2]), "r"(_packed_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)(part * 4 * 32 + 16)), "r"(_packed_1[4]), "r"(_packed_1[5]), "r"(_packed_1[6]), "r"(_packed_1[7]) : "memory");
                            }
                            if (ws_slot_out >= 0) {
                                float _scaled_2[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_17 = {final_scale, final_scale};
#else
                                const float2 _scale2_35 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 16))[_ls], _scale2_17);
#else
                                    reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 16))[_ls], _scale2_35);
#endif
                                unsigned int _packed_2[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                                    _packed_2[_lp] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 2], _scaled_2[_lp*2+1 + 2]));
                                    _packed_2[_lp + 1] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 4], _scaled_2[_lp*2+1 + 4]));
                                    _packed_2[_lp + 2] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 6], _scaled_2[_lp*2+1 + 6]));
                                    _packed_2[_lp + 3] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 8], _scaled_2[_lp*2+1 + 8]));
                                    _packed_2[_lp + 4] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 10], _scaled_2[_lp*2+1 + 10]));
                                    _packed_2[_lp + 5] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 12], _scaled_2[_lp*2+1 + 12]));
                                    _packed_2[_lp + 6] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_2[_lp*2 + 14], _scaled_2[_lp*2+1 + 14]));
                                    _packed_2[_lp + 7] = *(uint32_t*)&_h2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 1) * 32)), "r"(_packed_2[0]), "r"(_packed_2[1]), "r"(_packed_2[2]), "r"(_packed_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 1) * 32 + 16)), "r"(_packed_2[4]), "r"(_packed_2[5]), "r"(_packed_2[6]), "r"(_packed_2[7]) : "memory");
                            } else {
                                float _scaled_3[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_18 = {final_scale, final_scale};
#else
                                const float2 _scale2_36 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 16))[_ls], _scale2_18);
#else
                                    reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 16))[_ls], _scale2_36);
#endif
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
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 1) * 32)), "r"(_packed_3[0]), "r"(_packed_3[1]), "r"(_packed_3[2]), "r"(_packed_3[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 1) * 32 + 16)), "r"(_packed_3[4]), "r"(_packed_3[5]), "r"(_packed_3[6]), "r"(_packed_3[7]) : "memory");
                            }
                            if (ws_slot_out >= 0) {
                                float _scaled_4[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_19 = {final_scale, final_scale};
#else
                                const float2 _scale2_37 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 32))[_ls], _scale2_19);
#else
                                    reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 32))[_ls], _scale2_37);
#endif
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
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 2) * 32)), "r"(_packed_4[0]), "r"(_packed_4[1]), "r"(_packed_4[2]), "r"(_packed_4[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 2) * 32 + 16)), "r"(_packed_4[4]), "r"(_packed_4[5]), "r"(_packed_4[6]), "r"(_packed_4[7]) : "memory");
                            } else {
                                float _scaled_5[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_20 = {final_scale, final_scale};
#else
                                const float2 _scale2_38 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 32))[_ls], _scale2_20);
#else
                                    reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 32))[_ls], _scale2_38);
#endif
                                unsigned int _packed_5[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 0], _scaled_5[_lp*2+1 + 0]));
                                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 2], _scaled_5[_lp*2+1 + 2]));
                                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 4], _scaled_5[_lp*2+1 + 4]));
                                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 6], _scaled_5[_lp*2+1 + 6]));
                                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 8], _scaled_5[_lp*2+1 + 8]));
                                    _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 10], _scaled_5[_lp*2+1 + 10]));
                                    _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 12], _scaled_5[_lp*2+1 + 12]));
                                    _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 14], _scaled_5[_lp*2+1 + 14]));
                                    _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 2) * 32)), "r"(_packed_5[0]), "r"(_packed_5[1]), "r"(_packed_5[2]), "r"(_packed_5[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 2) * 32 + 16)), "r"(_packed_5[4]), "r"(_packed_5[5]), "r"(_packed_5[6]), "r"(_packed_5[7]) : "memory");
                            }
                            if (ws_slot_out >= 0) {
                                float _scaled_6[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_21 = {final_scale, final_scale};
#else
                                const float2 _scale2_39 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 48))[_ls], _scale2_21);
#else
                                    reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 48))[_ls], _scale2_39);
#endif
                                unsigned int _packed_6[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 0], _scaled_6[_lp*2+1 + 0]));
                                    _packed_6[_lp] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 2], _scaled_6[_lp*2+1 + 2]));
                                    _packed_6[_lp + 1] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 4], _scaled_6[_lp*2+1 + 4]));
                                    _packed_6[_lp + 2] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 6], _scaled_6[_lp*2+1 + 6]));
                                    _packed_6[_lp + 3] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 8], _scaled_6[_lp*2+1 + 8]));
                                    _packed_6[_lp + 4] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 10], _scaled_6[_lp*2+1 + 10]));
                                    _packed_6[_lp + 5] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 12], _scaled_6[_lp*2+1 + 12]));
                                    _packed_6[_lp + 6] = *(uint32_t*)&_h2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2(_scaled_6[_lp*2 + 14], _scaled_6[_lp*2+1 + 14]));
                                    _packed_6[_lp + 7] = *(uint32_t*)&_h2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 3) * 32)), "r"(_packed_6[0]), "r"(_packed_6[1]), "r"(_packed_6[2]), "r"(_packed_6[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 3) * 32 + 16)), "r"(_packed_6[4]), "r"(_packed_6[5]), "r"(_packed_6[6]), "r"(_packed_6[7]) : "memory");
                            } else {
                                float _scaled_7[16];
#if __CUDA_ARCH__ == 1030
                                const float2 _scale2_22 = {final_scale, final_scale};
#else
                                const float2 _scale2_40 = {final_scale, final_scale};
#endif
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                    reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 48))[_ls], _scale2_22);
#else
                                    reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 48))[_ls], _scale2_40);
#endif
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
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 3) * 32)), "r"(_packed_7[0]), "r"(_packed_7[1]), "r"(_packed_7[2]), "r"(_packed_7[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 3) * 32 + 16)), "r"(_packed_7[4]), "r"(_packed_7[5]), "r"(_packed_7[6]), "r"(_packed_7[7]) : "memory");
                            }
                        }
                    }
                    if (ws_slot_out >= 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (local_row < seg_len_out) {
                            {
#if __CUDA_ARCH__ == 1030
                                void* _cpbulk_dst_23 = reinterpret_cast<void*>(partial_O + ((long long)partial_row * 128));
#else
                                void* _cpbulk_dst_41 = reinterpret_cast<void*>(partial_O + ((long long)partial_row * 128));
#endif
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
#if __CUDA_ARCH__ == 1030
                                    :: "l"(_cpbulk_dst_23), "r"(row_addr), "r"((uint32_t)(256))
#else
                                    :: "l"(_cpbulk_dst_41), "r"(row_addr), "r"((uint32_t)(256))
#endif
                                    : "memory");
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    } else {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (local_row < seg_len_out) {
                            {
#if __CUDA_ARCH__ == 1030
                                void* _cpbulk_dst_24 = reinterpret_cast<void*>(O + out_off);
#else
                                void* _cpbulk_dst_42 = reinterpret_cast<void*>(O + out_off);
#endif
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
#if __CUDA_ARCH__ == 1030
                                    :: "l"(_cpbulk_dst_24), "r"(row_addr), "r"((uint32_t)(256))
#else
                                    :: "l"(_cpbulk_dst_42), "r"(row_addr), "r"((uint32_t)(256))
#endif
                                    : "memory");
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    _phase_corr_done ^= 1;
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 8 && warp <= 9) {
        { // correction_lo_main
            unsigned int total_tiles_1 = num_tiles;
            int owned_row_base = make_warp_uniform(warp % 2 * 32);
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_p_empty_1 = 1;
            unsigned int _phase_corr_sig_lo_0 = 0;
            unsigned int _phase_corr_sig_lo_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles_1; tile_idx_1 += num_clusters) {
                int head_1 = cl_head[tile_idx_1];
                int seg_begin_1 = cl_seg_begin[tile_idx_1];
                int seg_len_1 = cl_seg_len[tile_idx_1];
                int kv_base_1 = cl_kv_base[tile_idx_1];
                int q_word_1 = cl_q_block[tile_idx_1];
                int q_base_1 = q_word_1 & 65535;
                int q_half_1 = q_word_1 >> 16;
                int two_stages_1 = ((q_half_1 == 0) ? 1 : 0);
                int m_block_1 = q_base_1 + (q_half_1 >> 1) * 2 + cta_rank * (1 + two_stages_1);
                unsigned int num_n_blocks_1 = cl_kv_blocks[tile_idx_1];
                int kv_begin_1 = cl_kv_begin[tile_idx_1];
                int ws_slot_1 = cl_ws_slot[tile_idx_1];
                if (two_stages_1 == 1) {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                    _phase_p_empty_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                    _phase_corr_sig_lo_1 ^= 1;
                    mbarrier_arrive(corr_done_addr + 8);
                    #pragma unroll 1
                    for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
                        mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                        _phase_corr_sig_lo_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                        _phase_p_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row = owned_row_base + lane;
                        int tmem_row_base = make_warp_uniform(owned_row_base << 16);
                        float scale = row_state[logical_row];
                        int skip_rescale = __all_sync(0xFFFFFFFF, scale == 1.0f);
                        if (skip_rescale == 0) {
                            #pragma unroll
                            for (int col = 0; col < 8; col++) {
                                int addr = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base + (unsigned int)(col * 16));
                                float _tmem_load_0[16];
                                tmem_ld_x16(&_tmem_load_0[0], addr);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_0 = {scale, scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                                tmem_st_x16_f32(addr, _tmem_load_0);
                            }
                        }
                        int skip_rescale_0 = skip_rescale;
                        if (skip_rescale_0 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(corr_done_addr);
                        mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                        _phase_corr_sig_lo_1 ^= 1;
                        mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                        _phase_p_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_1 = owned_row_base + lane;
                        int tmem_row_base_2 = make_warp_uniform(owned_row_base << 16);
                        float scale_3 = row_state[logical_row_1 + 128];
                        int skip_rescale_4 = __all_sync(0xFFFFFFFF, scale_3 == 1.0f);
                        if (skip_rescale_4 == 0) {
                            #pragma unroll
                            for (int col_1 = 0; col_1 < 8; col_1++) {
                                int addr_1 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_2 + (unsigned int)(col_1 * 16));
                                float _tmem_load_1[16];
                                tmem_ld_x16(&_tmem_load_1[0], addr_1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_1 = {scale_3, scale_3};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                                tmem_st_x16_f32(addr_1, _tmem_load_1);
                            }
                        }
                        skip_rescale_0 = skip_rescale_4;
                        if (skip_rescale_0 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(corr_done_addr + 8);
                    }
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                    _phase_corr_sig_lo_1 ^= 1;
                    mbarrier_arrive(corr_done_addr + 8);
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                    _phase_o_full_1 ^= 1;
                } else {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
                        mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                        _phase_corr_sig_lo_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                        _phase_p_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_2 = owned_row_base + lane;
                        int tmem_row_base_1 = make_warp_uniform(owned_row_base << 16);
                        float scale_1 = row_state[logical_row_2];
                        int skip_rescale_1 = __all_sync(0xFFFFFFFF, scale_1 == 1.0f);
                        if (skip_rescale_1 == 0) {
                            #pragma unroll
                            for (int col_2 = 0; col_2 < 8; col_2++) {
                                int addr_2 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_1 + (unsigned int)(col_2 * 16));
                                float _tmem_load_2[16];
                                tmem_ld_x16(&_tmem_load_2[0], addr_2);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_2 = {scale_1, scale_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_2);
                                tmem_st_x16_f32(addr_2, _tmem_load_2);
                            }
                        }
                        int skip_rescale_0_1 = skip_rescale_1;
                        if (skip_rescale_0_1 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(corr_done_addr);
                    }
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                }
            }
            mbarrier_wait(p_empty_addr, _phase_p_empty_0);
            _phase_p_empty_0 ^= 1;
            mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
            _phase_p_empty_1 ^= 1;
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 10 && warp <= 11) {
        { // correction_hi_main
            unsigned int total_tiles_2 = num_tiles;
            int owned_row_base_1 = make_warp_uniform(64 + warp % 2 * 32);
            unsigned int _phase_p_empty_0_1 = 1;
            unsigned int _phase_p_empty_1_1 = 1;
            unsigned int _phase_corr_sig_hi_0 = 0;
            unsigned int _phase_corr_sig_hi_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            unsigned int _phase_o_full_1_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles_2; tile_idx_2 += num_clusters) {
                int head_2 = cl_head[tile_idx_2];
                int seg_begin_2 = cl_seg_begin[tile_idx_2];
                int seg_len_2 = cl_seg_len[tile_idx_2];
                int kv_base_2 = cl_kv_base[tile_idx_2];
                int q_word_2 = cl_q_block[tile_idx_2];
                int q_base_2 = q_word_2 & 65535;
                int q_half_2 = q_word_2 >> 16;
                int two_stages_2 = ((q_half_2 == 0) ? 1 : 0);
                int m_block_2 = q_base_2 + (q_half_2 >> 1) * 2 + cta_rank * (1 + two_stages_2);
                unsigned int num_n_blocks_2 = cl_kv_blocks[tile_idx_2];
                int kv_begin_2 = cl_kv_begin[tile_idx_2];
                int ws_slot_2 = cl_ws_slot[tile_idx_2];
                if (two_stages_2 == 1) {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                    _phase_p_empty_1_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                    _phase_corr_sig_hi_1 ^= 1;
                    mbarrier_arrive(corr_done_addr + 8);
                    #pragma unroll 1
                    for (unsigned int n_iter_3 = 1; n_iter_3 < num_n_blocks_2; n_iter_3++) {
                        mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                        _phase_corr_sig_hi_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                        _phase_p_empty_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_3 = owned_row_base_1 + lane;
                        int tmem_row_base_3 = make_warp_uniform(owned_row_base_1 << 16);
                        float scale_2 = row_state[logical_row_3];
                        int skip_rescale_2 = __all_sync(0xFFFFFFFF, scale_2 == 1.0f);
                        if (skip_rescale_2 == 0) {
                            #pragma unroll
                            for (int col_3 = 0; col_3 < 8; col_3++) {
                                int addr_3 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_3 + (unsigned int)(col_3 * 16));
                                float _tmem_load_3[16];
                                tmem_ld_x16(&_tmem_load_3[0], addr_3);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_0 = {scale_2, scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                                tmem_st_x16_f32(addr_3, _tmem_load_3);
                            }
                        }
                        int skip_rescale_0_2 = skip_rescale_2;
                        if (skip_rescale_0_2 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(corr_done_addr);
                        mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                        _phase_corr_sig_hi_1 ^= 1;
                        mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                        _phase_p_empty_1_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_1_1 = owned_row_base_1 + lane;
                        int tmem_row_base_2_1 = make_warp_uniform(owned_row_base_1 << 16);
                        float scale_3_1 = row_state[logical_row_1_1 + 128];
                        int skip_rescale_4_1 = __all_sync(0xFFFFFFFF, scale_3_1 == 1.0f);
                        if (skip_rescale_4_1 == 0) {
                            #pragma unroll
                            for (int col_4 = 0; col_4 < 8; col_4++) {
                                int addr_4 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_2_1 + (unsigned int)(col_4 * 16));
                                float _tmem_load_4[16];
                                tmem_ld_x16(&_tmem_load_4[0], addr_4);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_1 = {scale_3_1, scale_3_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_1);
                                tmem_st_x16_f32(addr_4, _tmem_load_4);
                            }
                        }
                        skip_rescale_0_2 = skip_rescale_4_1;
                        if (skip_rescale_0_2 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(corr_done_addr + 8);
                    }
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                    _phase_corr_sig_hi_1 ^= 1;
                    mbarrier_arrive(corr_done_addr + 8);
                    mbarrier_wait(o_full_addr, _phase_o_full_0_1);
                    _phase_o_full_0_1 ^= 1;
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1_1);
                    _phase_o_full_1_1 ^= 1;
                } else {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    #pragma unroll 1
                    for (unsigned int n_iter_4 = 1; n_iter_4 < num_n_blocks_2; n_iter_4++) {
                        mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                        _phase_corr_sig_hi_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                        _phase_p_empty_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_4 = owned_row_base_1 + lane;
                        int tmem_row_base_4 = make_warp_uniform(owned_row_base_1 << 16);
                        float scale_4 = row_state[logical_row_4];
                        int skip_rescale_3 = __all_sync(0xFFFFFFFF, scale_4 == 1.0f);
                        if (skip_rescale_3 == 0) {
                            #pragma unroll
                            for (int col_5 = 0; col_5 < 8; col_5++) {
                                int addr_5 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_4 + (unsigned int)(col_5 * 16));
                                float _tmem_load_5[16];
                                tmem_ld_x16(&_tmem_load_5[0], addr_5);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_2 = {scale_4, scale_4};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_2);
                                tmem_st_x16_f32(addr_5, _tmem_load_5);
                            }
                        }
                        int skip_rescale_0_3 = skip_rescale_3;
                        if (skip_rescale_0_3 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(corr_done_addr);
                    }
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(o_full_addr, _phase_o_full_0_1);
                    _phase_o_full_0_1 ^= 1;
                }
            }
            mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
            _phase_p_empty_0_1 ^= 1;
            mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
            _phase_p_empty_1_1 ^= 1;
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_s_empty_1 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int total_tiles_3 = num_tiles;
                unsigned int k_stage = 0;
                unsigned int v_stage = 0;
                #pragma unroll 1
                for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles_3; tile_idx_3 += num_clusters) {
                    int head_3 = cl_head[tile_idx_3];
                    int seg_begin_3 = cl_seg_begin[tile_idx_3];
                    int seg_len_3 = cl_seg_len[tile_idx_3];
                    int kv_base_3 = cl_kv_base[tile_idx_3];
                    int q_word_3 = cl_q_block[tile_idx_3];
                    int q_base_3 = q_word_3 & 65535;
                    int q_half_3 = q_word_3 >> 16;
                    int two_stages_3 = ((q_half_3 == 0) ? 1 : 0);
                    int m_block_3 = q_base_3 + (q_half_3 >> 1) * 2 + cta_rank * (1 + two_stages_3);
                    unsigned int num_n_blocks_3 = cl_kv_blocks[tile_idx_3];
                    int kv_begin_3 = cl_kv_begin[tile_idx_3];
                    int ws_slot_3 = cl_ws_slot[tile_idx_3];
                    if (two_stages_3 == 1) {
                        mbarrier_wait(q_full_addr, _phase_q_full_0);
                        _phase_q_full_0 ^= 1;
                        mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                        _phase_q_full_1 ^= 1;
                        mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4))));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 8)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk1, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk1 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 64 + 8)));
                        }
                        int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                        unsigned int first_pv = 1;
                        #pragma unroll 1
                        for (unsigned int n_iter_5 = 0; n_iter_5 < num_n_blocks_3 - 1; n_iter_5++) {
                            int first_pv_flag = first_pv;
                            mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                            }
                            mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                            _phase_s_empty_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_2 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                            mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                            _phase_s_empty_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                            int _mma_b_lo_3 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                        0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                            mbarrier_wait(p_full_addr, _phase_p_full_0);
                            _phase_p_full_0 ^= 1;
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
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_4), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                            elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                            _phase_p_full_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                            elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                            v_stage += 1;
                            if (v_stage == 4) { v_stage = 0; _phase_v_full ^= 1; }
                            first_pv = 0;
                        }
                        elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        int first_pv_flag_1 = first_pv;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_6), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
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
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_7), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 4) { v_stage = 0; _phase_v_full ^= 1; }
                        mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                        _phase_s_empty_1 ^= 1;
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                    } else {
                        mbarrier_wait(q_full_addr, _phase_q_full_0);
                        _phase_q_full_0 ^= 1;
                        mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4))));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 8)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        int _mma_a_lo_4 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_8 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                        unsigned int first_pv_1 = 1;
                        #pragma unroll 1
                        for (unsigned int n_iter_6 = 0; n_iter_6 < num_n_blocks_3 - 1; n_iter_6++) {
                            int first_pv_flag_2 = first_pv_1;
                            mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                            }
                            mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                            _phase_s_empty_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_5 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_9 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                            mbarrier_wait(p_full_addr, _phase_p_full_0);
                            _phase_p_full_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 512;
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_10), "r"(tmem_probs_0), "r"(((first_pv_flag_2) ? 0 : 1)) : "memory");
                            elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                            elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                            v_stage += 1;
                            if (v_stage == 4) { v_stage = 0; _phase_v_full ^= 1; }
                            first_pv_1 = 0;
                        }
                        elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        int first_pv_flag_3 = first_pv_1;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
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
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_11), "r"(tmem_probs_0), "r"(((first_pv_flag_3) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 4) { v_stage = 0; _phase_v_full ^= 1; }
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int total_tiles_4 = num_tiles;
            unsigned int k_load_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < total_tiles_4; tile_idx_4 += num_clusters) {
                int head_4 = cl_head[tile_idx_4];
                int seg_begin_4 = cl_seg_begin[tile_idx_4];
                int seg_len_4 = cl_seg_len[tile_idx_4];
                int kv_base_4 = cl_kv_base[tile_idx_4];
                int q_word_4 = cl_q_block[tile_idx_4];
                int q_base_4 = q_word_4 & 65535;
                int q_half_4 = q_word_4 >> 16;
                int two_stages_4 = ((q_half_4 == 0) ? 1 : 0);
                int m_block_4 = q_base_4 + (q_half_4 >> 1) * 2 + cta_rank * (1 + two_stages_4);
                unsigned int num_n_blocks_4 = cl_kv_blocks[tile_idx_4];
                int kv_begin_4 = cl_kv_begin[tile_idx_4];
                int ws_slot_4 = cl_ws_slot[tile_idx_4];
                int q_tile = kv_base_4 + m_block_4;
                int q_sf_tile = head_4 * PB + q_tile;
                int q_row = q_sf_tile * 128;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    tma_2d_gmem2smem_cta2(smem_q_addr, (&Q), 0, q_row, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr, (&SFQ), 0, q_sf_tile * 32, ((q_full_addr) & 0xFEFFFFFF));
                    if (two_stages_4 == 1) {
                        tma_2d_gmem2smem_cta2(smem_q_addr + 8192, (&Q), 0, q_row + 128, ((q_full_addr + 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfq_addr + 1024, (&SFQ), 0, (q_sf_tile + 1) * 32, ((q_full_addr + 8) & 0xFEFFFFFF));
                    }
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                    if (two_stages_4 == 1) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr + 8) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                    }
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_4; ni++) {
                    unsigned int n = num_n_blocks_4 - 1 - ni;
                    int kv_sf_tile = (unsigned int)(head_4 * PB + kv_base_4 + kv_begin_4) + n;
                    int kv_row = kv_sf_tile * 128;
                    mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_k_addr + k_load_stage * 4096, (&K), 0, kv_row + cta_rank * 64, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfk_addr + k_load_stage * 1024, (&SFK), 0, kv_sf_tile * 32, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(5120)) : "memory");
                    }
                    k_load_stage += 1;
                    if (k_load_stage == 4) { k_load_stage = 0; _phase_k_empty ^= 1; }
                }
            }
            #pragma unroll
            for (int _ = 0; _ < 4; _++) {
                mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                k_load_stage += 1;
                if (k_load_stage == 4) { k_load_stage = 0; _phase_k_empty ^= 1; }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
        }
    }
    // ---- Role: v_load ----
    if (warp == 14) {
        { // v_load_main
            unsigned int total_tiles_5 = num_tiles;
            unsigned int v_load_stage = 0;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_5 = cluster_id; tile_idx_5 < total_tiles_5; tile_idx_5 += num_clusters) {
                int head_5 = cl_head[tile_idx_5];
                int seg_begin_5 = cl_seg_begin[tile_idx_5];
                int seg_len_5 = cl_seg_len[tile_idx_5];
                int kv_base_5 = cl_kv_base[tile_idx_5];
                int q_word_5 = cl_q_block[tile_idx_5];
                int q_base_5 = q_word_5 & 65535;
                int q_half_5 = q_word_5 >> 16;
                int two_stages_5 = ((q_half_5 == 0) ? 1 : 0);
                int m_block_5 = q_base_5 + (q_half_5 >> 1) * 2 + cta_rank * (1 + two_stages_5);
                unsigned int num_n_blocks_5 = cl_kv_blocks[tile_idx_5];
                int kv_begin_5 = cl_kv_begin[tile_idx_5];
                int ws_slot_5 = cl_ws_slot[tile_idx_5];
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_5; ni_1++) {
                    unsigned int n_1 = num_n_blocks_5 - 1 - ni_1;
                    int kv_tile = (unsigned int)(kv_base_5 + kv_begin_5) + n_1;
                    int kv_sf_tile_1 = head_5 * PB + kv_tile;
                    mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_v_addr + v_load_stage * 8192, (&V), cta_rank * 64, kv_sf_tile_1 * 128, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                    }
                    v_load_stage += 1;
                    if (v_load_stage == 4) { v_load_stage = 0; _phase_v_empty ^= 1; }
                }
            }
            #pragma unroll
            for (int __1 = 0; __1 < 4; __1++) {
                mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                v_load_stage += 1;
                if (v_load_stage == 4) { v_load_stage = 0; _phase_v_empty ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        { // idle_main
            __syncwarp();
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
