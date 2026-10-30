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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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
#define TMEM_SCORES_0_OFFSET 0
#define TMEM_SOFTMAX_0_OFFSET 64
#define TMEM_SCORES_1_OFFSET 128
#define TMEM_SOFTMAX_1_OFFSET 192
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_Q_STAGES 2
#define NUM_KV_STAGES 3
#define NUM_PO_STAGES 6
#define NUM_ACC_STAGES 1
#define NUM_ORDER_P01_STAGES 1
#define SMEM_SSCALE_OFF 1024
#define SMEM_SSCALE_STAGE_BYTES 3072
#define SMEM_SSCALE_STRIDE 3072
#define SMEM_SMEM_Q0_OFF 4096
#define SMEM_SMEM_Q0_STAGE_BYTES 16384
#define SMEM_SMEM_Q0_STRIDE 16384
#define SMEM_SMEM_Q1_OFF 20480
#define SMEM_SMEM_Q1_STAGE_BYTES 16384
#define SMEM_SMEM_Q1_STRIDE 16384
#define SMEM_SMEM_KV_OFF 36864
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 36864
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_PAGES_OFF 86016
#define SMEM_SMEM_PAGES_STAGE_BYTES 384
#define SMEM_SMEM_PAGES_STRIDE 384
#define SMEM_WORK_ID_SLOT_OFF 86400
#define SMEM_WORK_ID_SLOT_STAGE_BYTES 4
#define SMEM_WORK_ID_SLOT_STRIDE 4
#define SMEM_MERGE_FLAG_OFF 86404
#define SMEM_MERGE_FLAG_STAGE_BYTES 4
#define SMEM_MERGE_FLAG_STRIDE 4
#define SMEM_MERGE_ML_OFF 86528
#define SMEM_MERGE_ML_STAGE_BYTES 65536
#define SMEM_MERGE_ML_STRIDE 65536
#define SMEM_MERGE_RING_OFF 152064
#define SMEM_MERGE_RING_STAGE_BYTES 65536
#define SMEM_MERGE_RING_STRIDE 65536
#define SMEM_TOTAL 217600
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define MMA_K 32
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


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
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





__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_context_fp8_074b1da06353ee6d95c6(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, uint8_t* __restrict__ O_ptr, float* __restrict__ LSE_ptr, float* __restrict__ sinks, int* __restrict__ page_table_k, int* __restrict__ page_table_v, int* __restrict__ seq_lens_q, int* __restrict__ seq_lens_kv, int* __restrict__ cu_seq_lens_q, float softmax_scale_log2, float output_scale, int total_bh, int page_row_stride, int num_ctas, unsigned int* __restrict__ dynamic_counter, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int scale_source, CakeFastDivmod num_m_blocks, unsigned int full_sections, CakeFastDivmod num_bh_units, int num_q_heads, CakeFastDivmod heads_per_group, int page_shift, CakeFastDivmod split_tiles, unsigned int split_blocks, unsigned int num_splits, float* __restrict__ partial_o, float* __restrict__ partial_ml, unsigned int* __restrict__ tile_counters)
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
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 64)
    #define p_full_addr (mbar_base + 80)
    #define p_full_2_addr (mbar_base + 96)
    #define corr_sig_addr (mbar_base + 112)
    #define corr_done_addr (mbar_base + 128)
    #define stats_full_addr (mbar_base + 144)
    #define order_p01_0_addr (mbar_base + 160)
    #define order_p01_1_addr (mbar_base + 168)
    #define o_full_addr (mbar_base + 176)
    #define q_empty_addr (mbar_base + 192)
    #define page_full_addr (mbar_base + 200)
    #define page_empty_addr (mbar_base + 248)
    #define work_id_full_addr (mbar_base + 296)
    #define work_id_empty_addr (mbar_base + 304)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* sScale = reinterpret_cast<float*>(smem_raw + 1024);
    const int sScale_addr = smem + 1024;
    uint8_t* smem_q0 = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int smem_q0_addr = smem + 4096;
    uint8_t* smem_q1 = reinterpret_cast<uint8_t*>(smem_raw + 20480);
    const int smem_q1_addr = smem + 20480;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_kv_addr = smem + 36864;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v_addr = smem + 36864;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 86016);
    const int smem_pages_addr = smem + 86016;
    unsigned int* work_id_slot = reinterpret_cast<unsigned int*>(smem_raw + 86400);
    const int work_id_slot_addr = smem + 86400;
    unsigned int* merge_flag = reinterpret_cast<unsigned int*>(smem_raw + 86404);
    const int merge_flag_addr = smem + 86404;
    float* merge_ml = reinterpret_cast<float*>(smem_raw + 86528);
    const int merge_ml_addr = smem + 86528;
    float* merge_ring = reinterpret_cast<float*>(smem_raw + 152064);
    const int merge_ring_addr = smem + 152064;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (17 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv' ---
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            mbarrier_init(smem + 88, 256);
            // p_full_2: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            // --- pipeline 'order_p01' ---
            // order_p01_0: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // order_p01_1: 1 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // --- pipeline 'po' ---
            // page_full: 6 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // page_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            // work_id_full: 1 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            // work_id_empty: 1 barriers, init_count=480
            mbarrier_init(smem + 304, 480);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores_0 = taddr;
    const int tmem_softmax_0 = taddr + 64;
    const int tmem_scores_1 = taddr + 128;
    const int tmem_softmax_1 = taddr + 192;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            float smx_scale = softmax_scale_log2;
            if (scale_source != 0) {
                smx_scale = bmm1_scale_ptr[0];
                unsigned int raw_bits = __as_u32(smx_scale);
                if (bmm1_is_log2 == 0) {
                    smx_scale = (float)((double)smx_scale * 1.4426950216293335);
                }
                smx_scale = (float)((double)smx_scale / 1.4426950408889634 * 1.4426950408889634);
                if (bmm1_is_log2 == 0 && (raw_bits & 2147483647) - 1 < 8388607) {
                    double wide = (double)(int)(raw_bits & 8388607) * 1.401298464324817e-45;
                    if (raw_bits >= 2147483648u) {
                        wide = -wide;
                    }
                    smx_scale = (float)(wide * 1.4426950216293335);
                    smx_scale = (float)((double)smx_scale / 1.4426950408889634 * 1.4426950408889634);
                }
            }
            unsigned int stage = make_warp_uniform(warp / 4);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(stage * 128 + 64);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int total_tiles = ((uint32_t)(num_m_blocks.divisor)) * (unsigned int)total_bh;
            unsigned int total_units = total_tiles + (num_splits - 1) * ((uint32_t)(split_tiles.divisor));
            unsigned int max_rounds = total_units + 1;
            int order_p01_start = ((stage == 0) ? -1 : 0);
            int order_p01_seen = order_p01_start;
            unsigned int _phase_work_id_full_0 = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (unsigned int tile_iter = 0; tile_iter < max_rounds; tile_iter++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0);
                _phase_work_id_full_0 ^= 1;
                unsigned int ticket = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (ticket >= total_units) {
                    break;
                }
                unsigned int ut_units = num_splits * ((uint32_t)(split_tiles.divisor));
                unsigned int ut_tile = ticket - ut_units + ((uint32_t)(split_tiles.divisor));
                unsigned int ut_split = 0;
                unsigned int ut_in_split = 0;
                if (ticket < ut_units) {
                    uint32_t ut_s = (split_tiles.divisor != 1) ? (__umulhi((uint32_t)(ticket), split_tiles.multiplier) >> split_tiles.shift_right) : (uint32_t)(ticket);
                    uint32_t ut_j = (uint32_t)(ticket) - ut_s * (uint32_t)(split_tiles.divisor);
                    ut_tile = ut_j;
                    ut_split = ut_s;
                    ut_in_split = 1;
                }
                unsigned int m_block;
                unsigned int bh;
                uint32_t tile_section = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(ut_tile >> 3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(ut_tile >> 3);
                uint32_t tile_sw_m = (uint32_t)(ut_tile >> 3) - tile_section * (uint32_t)(num_m_blocks.divisor);
                if (tile_section >= full_sections) {
                    unsigned int tile_tail = ut_tile - tile_section * 8 * ((uint32_t)(num_m_blocks.divisor));
                    uint32_t tail_unit = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(tile_tail), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(tile_tail);
                    uint32_t tail_m = (uint32_t)(tile_tail) - tail_unit * (uint32_t)(num_m_blocks.divisor);
                    m_block = tail_m;
                    bh = tile_section * 8 + tail_unit;
                } else {
                    m_block = tile_sw_m;
                    bh = tile_section * 8 + (ut_tile & 7);
                }
                m_block = ((uint32_t)(num_m_blocks.divisor)) - 1 - m_block;
                uint32_t b_idx = (num_bh_units.divisor != 1) ? (__umulhi((uint32_t)(bh), num_bh_units.multiplier) >> num_bh_units.shift_right) : (uint32_t)(bh);
                uint32_t unit = (uint32_t)(bh) - b_idx * (uint32_t)(num_bh_units.divisor);
                int seqlen_kv_bh = seq_lens_kv[b_idx];
                int seqlen_q_bh = cu_seq_lens_q[b_idx + 1] - cu_seq_lens_q[b_idx];
                unsigned int num_n_blocks = (seqlen_kv_bh + BLOCK_N - 1) / BLOCK_N;
                int tile_kv_shift = seqlen_kv_bh - seqlen_q_bh;
                unsigned int tile_max_n = (m_block * 2 + 2) * 32 + (unsigned int)tile_kv_shift;
                if (tile_max_n < (unsigned int)seqlen_kv_bh) {
                    num_n_blocks = (tile_max_n + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                }
                unsigned int _max_0 = ((num_n_blocks) > (1) ? (num_n_blocks) : (1));
                num_n_blocks = _max_0;
                unsigned int us_count = 1;
                unsigned int us_hi = num_n_blocks;
                unsigned int us_blocks = num_n_blocks;
                if (ut_in_split != 0) {
                    unsigned int _min_1 = (((num_n_blocks + split_blocks - 1) / split_blocks) < (num_splits) ? ((num_n_blocks + split_blocks - 1) / split_blocks) : (num_splits));
                    us_count = _min_1;
                    us_hi = num_n_blocks - ut_split * split_blocks;
                    if (ut_split + 1 < us_count) {
                        us_blocks = split_blocks;
                    } else {
                        us_blocks = us_hi;
                    }
                }
                int causal_row;
                unsigned int num_masked_iters;
                int kv_shift = seqlen_kv_bh - seqlen_q_bh;
                causal_row = (m_block * 2 + stage) * 32 + (unsigned int)((warp % 4 * 32 + lane) / 4) + (unsigned int)kv_shift;
                int causal_row_min = (m_block * 2 + stage) * 32 + (unsigned int)kv_shift;
                unsigned int n_block_no_mask_limit = (causal_row_min + 1) / BLOCK_N;
                int masked_blocks = (int)us_hi - (int)n_block_no_mask_limit;
                if (masked_blocks < 0) {
                    masked_blocks = 0;
                }
                num_masked_iters = masked_blocks;
                if (num_masked_iters > us_blocks) {
                    num_masked_iters = us_blocks;
                }
                float row_max_val = -CUDART_INF_F;
                float row_sum_val = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < num_masked_iters; n_iter++) {
                    int n_block = us_hi - 1 - n_iter;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_base = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_0[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(s_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(s_base + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                        : "r"(s_base + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                        : "r"(s_base + 96));
                    int tail_valid = seqlen_kv_bh - n_block * BLOCK_N;
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
                        if (!(_thresh_mask_2 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
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
                        if (!(_thresh_mask_4 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
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
                        if (!(_thresh_mask_6 & (1u << 0))) _tmem_load_0[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 1))) _tmem_load_0[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 2))) _tmem_load_0[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 3))) _tmem_load_0[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 4))) _tmem_load_0[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 5))) _tmem_load_0[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 6))) _tmem_load_0[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 7))) _tmem_load_0[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 8))) _tmem_load_0[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 9))) _tmem_load_0[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 10))) _tmem_load_0[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 11))) _tmem_load_0[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 12))) _tmem_load_0[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 13))) _tmem_load_0[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 14))) _tmem_load_0[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 15))) _tmem_load_0[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 16))) _tmem_load_0[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 17))) _tmem_load_0[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 18))) _tmem_load_0[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 19))) _tmem_load_0[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 20))) _tmem_load_0[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 21))) _tmem_load_0[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 22))) _tmem_load_0[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 23))) _tmem_load_0[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 24))) _tmem_load_0[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 25))) _tmem_load_0[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 26))) _tmem_load_0[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 27))) _tmem_load_0[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 28))) _tmem_load_0[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 29))) _tmem_load_0[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 30))) _tmem_load_0[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 31))) _tmem_load_0[95] = -CUDART_INF_F;
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
                        if (!(_thresh_mask_8 & (1u << 0))) _tmem_load_0[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 1))) _tmem_load_0[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 2))) _tmem_load_0[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 3))) _tmem_load_0[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 4))) _tmem_load_0[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 5))) _tmem_load_0[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 6))) _tmem_load_0[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 7))) _tmem_load_0[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 8))) _tmem_load_0[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 9))) _tmem_load_0[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 10))) _tmem_load_0[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 11))) _tmem_load_0[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 12))) _tmem_load_0[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 13))) _tmem_load_0[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 14))) _tmem_load_0[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 15))) _tmem_load_0[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 16))) _tmem_load_0[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 17))) _tmem_load_0[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 18))) _tmem_load_0[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 19))) _tmem_load_0[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 20))) _tmem_load_0[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 21))) _tmem_load_0[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 22))) _tmem_load_0[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 23))) _tmem_load_0[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 24))) _tmem_load_0[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 25))) _tmem_load_0[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 26))) _tmem_load_0[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 27))) _tmem_load_0[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 28))) _tmem_load_0[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 29))) _tmem_load_0[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 30))) _tmem_load_0[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_8 & (1u << 31))) _tmem_load_0[127] = -CUDART_INF_F;
                    }
                    int valid_count = causal_row - n_block * BLOCK_N + 1;
                    uint32_t _thresh_mask_10;
                    {
                        int _lim_11 = valid_count;
                        if (_lim_11 <= 0) { _thresh_mask_10 = 0u; }
                        else if (_lim_11 >= 32) { _thresh_mask_10 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_10) : "r"(_lim_11));
                        }
                    }
                    if (!(_thresh_mask_10 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(_thresh_mask_10 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                    uint32_t _thresh_mask_12;
                    {
                        int _lim_13 = valid_count - 32;
                        if (_lim_13 <= 0) { _thresh_mask_12 = 0u; }
                        else if (_lim_13 >= 32) { _thresh_mask_12 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_12) : "r"(_lim_13));
                        }
                    }
                    if (!(_thresh_mask_12 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                    if (!(_thresh_mask_12 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
                    uint32_t _thresh_mask_14;
                    {
                        int _lim_15 = valid_count - 64;
                        if (_lim_15 <= 0) { _thresh_mask_14 = 0u; }
                        else if (_lim_15 >= 32) { _thresh_mask_14 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_14) : "r"(_lim_15));
                        }
                    }
                    if (!(_thresh_mask_14 & (1u << 0))) _tmem_load_0[64] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 1))) _tmem_load_0[65] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 2))) _tmem_load_0[66] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 3))) _tmem_load_0[67] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 4))) _tmem_load_0[68] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 5))) _tmem_load_0[69] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 6))) _tmem_load_0[70] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 7))) _tmem_load_0[71] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 8))) _tmem_load_0[72] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 9))) _tmem_load_0[73] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 10))) _tmem_load_0[74] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 11))) _tmem_load_0[75] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 12))) _tmem_load_0[76] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 13))) _tmem_load_0[77] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 14))) _tmem_load_0[78] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 15))) _tmem_load_0[79] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 16))) _tmem_load_0[80] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 17))) _tmem_load_0[81] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 18))) _tmem_load_0[82] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 19))) _tmem_load_0[83] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 20))) _tmem_load_0[84] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 21))) _tmem_load_0[85] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 22))) _tmem_load_0[86] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 23))) _tmem_load_0[87] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 24))) _tmem_load_0[88] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 25))) _tmem_load_0[89] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 26))) _tmem_load_0[90] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 27))) _tmem_load_0[91] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 28))) _tmem_load_0[92] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 29))) _tmem_load_0[93] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 30))) _tmem_load_0[94] = -CUDART_INF_F;
                    if (!(_thresh_mask_14 & (1u << 31))) _tmem_load_0[95] = -CUDART_INF_F;
                    uint32_t _thresh_mask_16;
                    {
                        int _lim_17 = valid_count - 96;
                        if (_lim_17 <= 0) { _thresh_mask_16 = 0u; }
                        else if (_lim_17 >= 32) { _thresh_mask_16 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_16) : "r"(_lim_17));
                        }
                    }
                    if (!(_thresh_mask_16 & (1u << 0))) _tmem_load_0[96] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 1))) _tmem_load_0[97] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 2))) _tmem_load_0[98] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 3))) _tmem_load_0[99] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 4))) _tmem_load_0[100] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 5))) _tmem_load_0[101] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 6))) _tmem_load_0[102] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 7))) _tmem_load_0[103] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 8))) _tmem_load_0[104] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 9))) _tmem_load_0[105] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 10))) _tmem_load_0[106] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 11))) _tmem_load_0[107] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 12))) _tmem_load_0[108] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 13))) _tmem_load_0[109] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 14))) _tmem_load_0[110] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 15))) _tmem_load_0[111] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 16))) _tmem_load_0[112] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 17))) _tmem_load_0[113] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 18))) _tmem_load_0[114] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 19))) _tmem_load_0[115] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 20))) _tmem_load_0[116] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 21))) _tmem_load_0[117] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 22))) _tmem_load_0[118] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 23))) _tmem_load_0[119] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 24))) _tmem_load_0[120] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 25))) _tmem_load_0[121] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 26))) _tmem_load_0[122] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 27))) _tmem_load_0[123] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 28))) _tmem_load_0[124] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 29))) _tmem_load_0[125] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 30))) _tmem_load_0[126] = -CUDART_INF_F;
                    if (!(_thresh_mask_16 & (1u << 31))) _tmem_load_0[127] = -CUDART_INF_F;
                    float2 _reg_reduce_max2_18 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_18);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_18);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_18);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_18);
                    float new_max = row_max_reduce(_reg_reduce_max2_18);
                    new_max = max_noftz(new_max, row_max_val);
                    float new_max_scaled = ((new_max == -CUDART_INF_F) ? 0.0f : new_max) * smx_scale;
                    float acc_scale;
                    float _fma_0 = __fmaf_rn(row_max_val, smx_scale, -new_max_scaled);
                    if (row_max_val > -CUDART_INF_F) {
                        float _exp2_19 = approx_exp2(_fma_0);
                        acc_scale = _exp2_19;
                    } else {
                        acc_scale = 1.0f;
                    }
                    row_max_val = new_max;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    int p_base = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    float p_log2_bias = 8.807355f - new_max_scaled;
                    unsigned int pv0[16];
                    unsigned int pv1[8];
                    unsigned int pv2[8];
                    float block_sum0;
                    float block_sum1;
                    float block_sum2;
                    const bool _selected_wait_cond_20 = stage == 0;
                    mbarrier_wait((_selected_wait_cond_20 ? (order_p01_0_addr) : (order_p01_1_addr)), (_selected_wait_cond_20 ? ((int)((unsigned int)order_p01_seen % 2)) : ((int)((unsigned int)order_p01_seen % 2))));
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    const float2 _fma_b2_21 = {smx_scale, smx_scale};
                    const float2 _fma_c2_22 = {p_log2_bias, p_log2_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_21, _fma_c2_22);
                    #pragma unroll
                    for (int _le = 0; _le < 128; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_24 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[32], _tmem_load_0[33]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[34], _tmem_load_0[35]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[36], _tmem_load_0[37]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[38], _tmem_load_0[39]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[40], _tmem_load_0[41]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[42], _tmem_load_0[43]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[44], _tmem_load_0[45]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[46], _tmem_load_0[47]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[48], _tmem_load_0[49]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[50], _tmem_load_0[51]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[52], _tmem_load_0[53]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[54], _tmem_load_0[55]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[56], _tmem_load_0[57]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[58], _tmem_load_0[59]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[60], _tmem_load_0[61]));
                    _reg_reduce_sum2_24 = add_f32x2_rn_noftz(_reg_reduce_sum2_24, make_float2(_tmem_load_0[62], _tmem_load_0[63]));
                    float _reg_array_reduce_23 = _reg_reduce_sum2_24.x + _reg_reduce_sum2_24.y;
                    block_sum0 = _reg_array_reduce_23;
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
                        pv0[0] = _packed;
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
                        pv0[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[8]), "f"(_tmem_load_0[9]),
                                               "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]));
                        pv0[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]),
                                               "f"(_tmem_load_0[14]), "f"(_tmem_load_0[15]));
                        pv0[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[16]), "f"(_tmem_load_0[17]),
                                               "f"(_tmem_load_0[18]), "f"(_tmem_load_0[19]));
                        pv0[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[20]), "f"(_tmem_load_0[21]),
                                               "f"(_tmem_load_0[22]), "f"(_tmem_load_0[23]));
                        pv0[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[24]), "f"(_tmem_load_0[25]),
                                               "f"(_tmem_load_0[26]), "f"(_tmem_load_0[27]));
                        pv0[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[28]), "f"(_tmem_load_0[29]),
                                               "f"(_tmem_load_0[30]), "f"(_tmem_load_0[31]));
                        pv0[7] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[32]), "f"(_tmem_load_0[33]),
                                               "f"(_tmem_load_0[34]), "f"(_tmem_load_0[35]));
                        pv0[8] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[36]), "f"(_tmem_load_0[37]),
                                               "f"(_tmem_load_0[38]), "f"(_tmem_load_0[39]));
                        pv0[9] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[40]), "f"(_tmem_load_0[41]),
                                               "f"(_tmem_load_0[42]), "f"(_tmem_load_0[43]));
                        pv0[10] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[44]), "f"(_tmem_load_0[45]),
                                               "f"(_tmem_load_0[46]), "f"(_tmem_load_0[47]));
                        pv0[11] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[48]), "f"(_tmem_load_0[49]),
                                               "f"(_tmem_load_0[50]), "f"(_tmem_load_0[51]));
                        pv0[12] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[52]), "f"(_tmem_load_0[53]),
                                               "f"(_tmem_load_0[54]), "f"(_tmem_load_0[55]));
                        pv0[13] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[56]), "f"(_tmem_load_0[57]),
                                               "f"(_tmem_load_0[58]), "f"(_tmem_load_0[59]));
                        pv0[14] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[60]), "f"(_tmem_load_0[61]),
                                               "f"(_tmem_load_0[62]), "f"(_tmem_load_0[63]));
                        pv0[15] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&pv0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[3])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[4])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[5])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[6])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[7])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[8])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[9])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[10])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[11])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[12])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[13])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[14])), "r"(*reinterpret_cast<const uint32_t*>(&pv0[15])));
                    float2 _reg_reduce_sum2_26 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 0], _tmem_load_0[(64) + 1]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 2], _tmem_load_0[(64) + 3]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 4], _tmem_load_0[(64) + 5]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 6], _tmem_load_0[(64) + 7]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 8], _tmem_load_0[(64) + 9]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 10], _tmem_load_0[(64) + 11]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 12], _tmem_load_0[(64) + 13]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 14], _tmem_load_0[(64) + 15]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 16], _tmem_load_0[(64) + 17]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 18], _tmem_load_0[(64) + 19]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 20], _tmem_load_0[(64) + 21]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 22], _tmem_load_0[(64) + 23]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 24], _tmem_load_0[(64) + 25]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 26], _tmem_load_0[(64) + 27]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 28], _tmem_load_0[(64) + 29]));
                    _reg_reduce_sum2_26 = add_f32x2_rn_noftz(_reg_reduce_sum2_26, make_float2(_tmem_load_0[(64) + 30], _tmem_load_0[(64) + 31]));
                    float _reg_array_reduce_25 = _reg_reduce_sum2_26.x + _reg_reduce_sum2_26.y;
                    block_sum1 = _reg_array_reduce_25;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_0[64]), "f"(_tmem_load_0[65]),
                                               "f"(_tmem_load_0[66]), "f"(_tmem_load_0[67]));
                        pv1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[68]), "f"(_tmem_load_0[69]),
                                               "f"(_tmem_load_0[70]), "f"(_tmem_load_0[71]));
                        pv1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[72]), "f"(_tmem_load_0[73]),
                                               "f"(_tmem_load_0[74]), "f"(_tmem_load_0[75]));
                        pv1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[76]), "f"(_tmem_load_0[77]),
                                               "f"(_tmem_load_0[78]), "f"(_tmem_load_0[79]));
                        pv1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[80]), "f"(_tmem_load_0[81]),
                                               "f"(_tmem_load_0[82]), "f"(_tmem_load_0[83]));
                        pv1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[84]), "f"(_tmem_load_0[85]),
                                               "f"(_tmem_load_0[86]), "f"(_tmem_load_0[87]));
                        pv1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[88]), "f"(_tmem_load_0[89]),
                                               "f"(_tmem_load_0[90]), "f"(_tmem_load_0[91]));
                        pv1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[92]), "f"(_tmem_load_0[93]),
                                               "f"(_tmem_load_0[94]), "f"(_tmem_load_0[95]));
                        pv1[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base + 16, (const uint32_t*)pv1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    float2 _reg_reduce_sum2_28 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 0], _tmem_load_0[(96) + 1]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 2], _tmem_load_0[(96) + 3]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 4], _tmem_load_0[(96) + 5]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 6], _tmem_load_0[(96) + 7]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 8], _tmem_load_0[(96) + 9]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 10], _tmem_load_0[(96) + 11]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 12], _tmem_load_0[(96) + 13]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 14], _tmem_load_0[(96) + 15]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 16], _tmem_load_0[(96) + 17]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 18], _tmem_load_0[(96) + 19]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 20], _tmem_load_0[(96) + 21]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 22], _tmem_load_0[(96) + 23]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 24], _tmem_load_0[(96) + 25]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 26], _tmem_load_0[(96) + 27]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 28], _tmem_load_0[(96) + 29]));
                    _reg_reduce_sum2_28 = add_f32x2_rn_noftz(_reg_reduce_sum2_28, make_float2(_tmem_load_0[(96) + 30], _tmem_load_0[(96) + 31]));
                    float _reg_array_reduce_27 = _reg_reduce_sum2_28.x + _reg_reduce_sum2_28.y;
                    block_sum2 = _reg_array_reduce_27;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_0[96]), "f"(_tmem_load_0[97]),
                                               "f"(_tmem_load_0[98]), "f"(_tmem_load_0[99]));
                        pv2[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[100]), "f"(_tmem_load_0[101]),
                                               "f"(_tmem_load_0[102]), "f"(_tmem_load_0[103]));
                        pv2[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[104]), "f"(_tmem_load_0[105]),
                                               "f"(_tmem_load_0[106]), "f"(_tmem_load_0[107]));
                        pv2[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[108]), "f"(_tmem_load_0[109]),
                                               "f"(_tmem_load_0[110]), "f"(_tmem_load_0[111]));
                        pv2[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[112]), "f"(_tmem_load_0[113]),
                                               "f"(_tmem_load_0[114]), "f"(_tmem_load_0[115]));
                        pv2[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[116]), "f"(_tmem_load_0[117]),
                                               "f"(_tmem_load_0[118]), "f"(_tmem_load_0[119]));
                        pv2[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[120]), "f"(_tmem_load_0[121]),
                                               "f"(_tmem_load_0[122]), "f"(_tmem_load_0[123]));
                        pv2[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_0[124]), "f"(_tmem_load_0[125]),
                                               "f"(_tmem_load_0[126]), "f"(_tmem_load_0[127]));
                        pv2[7] = _packed;
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_seen += 1;
                    tmem_st_x8_u32(p_base + 24, (const uint32_t*)pv2);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum0 + block_sum1 + block_sum2);
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = num_masked_iters; n_iter_1 < us_blocks; n_iter_1++) {
                    int n_block_1 = us_hi - 1 - n_iter_1;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_base_1 = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_1[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(s_base_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(s_base_1 + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95])
                        : "r"(s_base_1 + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                        : "r"(s_base_1 + 96));
                    int tail_valid_1 = seqlen_kv_bh - n_block_1 * BLOCK_N;
                    if (tail_valid_1 < BLOCK_N) {
                        uint32_t _thresh_mask_29;
                        {
                            int _lim_30 = tail_valid_1;
                            if (_lim_30 <= 0) { _thresh_mask_29 = 0u; }
                            else if (_lim_30 >= 32) { _thresh_mask_29 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_29) : "r"(_lim_30));
                            }
                        }
                        if (!(_thresh_mask_29 & (1u << 0))) _tmem_load_1[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 1))) _tmem_load_1[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 2))) _tmem_load_1[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 3))) _tmem_load_1[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 4))) _tmem_load_1[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 5))) _tmem_load_1[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 6))) _tmem_load_1[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 7))) _tmem_load_1[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 8))) _tmem_load_1[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 9))) _tmem_load_1[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 10))) _tmem_load_1[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 11))) _tmem_load_1[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 12))) _tmem_load_1[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 13))) _tmem_load_1[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 14))) _tmem_load_1[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 15))) _tmem_load_1[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 16))) _tmem_load_1[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 17))) _tmem_load_1[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 18))) _tmem_load_1[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 19))) _tmem_load_1[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 20))) _tmem_load_1[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 21))) _tmem_load_1[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 22))) _tmem_load_1[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 23))) _tmem_load_1[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 24))) _tmem_load_1[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 25))) _tmem_load_1[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 26))) _tmem_load_1[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 27))) _tmem_load_1[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 28))) _tmem_load_1[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 29))) _tmem_load_1[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 30))) _tmem_load_1[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 31))) _tmem_load_1[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_31;
                        {
                            int _lim_32 = tail_valid_1 - 32;
                            if (_lim_32 <= 0) { _thresh_mask_31 = 0u; }
                            else if (_lim_32 >= 32) { _thresh_mask_31 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_31) : "r"(_lim_32));
                            }
                        }
                        if (!(_thresh_mask_31 & (1u << 0))) _tmem_load_1[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 1))) _tmem_load_1[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 2))) _tmem_load_1[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 3))) _tmem_load_1[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 4))) _tmem_load_1[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 5))) _tmem_load_1[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 6))) _tmem_load_1[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 7))) _tmem_load_1[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 8))) _tmem_load_1[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 9))) _tmem_load_1[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 10))) _tmem_load_1[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 11))) _tmem_load_1[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 12))) _tmem_load_1[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 13))) _tmem_load_1[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 14))) _tmem_load_1[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 15))) _tmem_load_1[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 16))) _tmem_load_1[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 17))) _tmem_load_1[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 18))) _tmem_load_1[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 19))) _tmem_load_1[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 20))) _tmem_load_1[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 21))) _tmem_load_1[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 22))) _tmem_load_1[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 23))) _tmem_load_1[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 24))) _tmem_load_1[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 25))) _tmem_load_1[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 26))) _tmem_load_1[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 27))) _tmem_load_1[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 28))) _tmem_load_1[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 29))) _tmem_load_1[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 30))) _tmem_load_1[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 31))) _tmem_load_1[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_33;
                        {
                            int _lim_34 = tail_valid_1 - 64;
                            if (_lim_34 <= 0) { _thresh_mask_33 = 0u; }
                            else if (_lim_34 >= 32) { _thresh_mask_33 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_33) : "r"(_lim_34));
                            }
                        }
                        if (!(_thresh_mask_33 & (1u << 0))) _tmem_load_1[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 1))) _tmem_load_1[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 2))) _tmem_load_1[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 3))) _tmem_load_1[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 4))) _tmem_load_1[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 5))) _tmem_load_1[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 6))) _tmem_load_1[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 7))) _tmem_load_1[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 8))) _tmem_load_1[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 9))) _tmem_load_1[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 10))) _tmem_load_1[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 11))) _tmem_load_1[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 12))) _tmem_load_1[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 13))) _tmem_load_1[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 14))) _tmem_load_1[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 15))) _tmem_load_1[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 16))) _tmem_load_1[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 17))) _tmem_load_1[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 18))) _tmem_load_1[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 19))) _tmem_load_1[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 20))) _tmem_load_1[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 21))) _tmem_load_1[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 22))) _tmem_load_1[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 23))) _tmem_load_1[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 24))) _tmem_load_1[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 25))) _tmem_load_1[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 26))) _tmem_load_1[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 27))) _tmem_load_1[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 28))) _tmem_load_1[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 29))) _tmem_load_1[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 30))) _tmem_load_1[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 31))) _tmem_load_1[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_35;
                        {
                            int _lim_36 = tail_valid_1 - 96;
                            if (_lim_36 <= 0) { _thresh_mask_35 = 0u; }
                            else if (_lim_36 >= 32) { _thresh_mask_35 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_35) : "r"(_lim_36));
                            }
                        }
                        if (!(_thresh_mask_35 & (1u << 0))) _tmem_load_1[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 1))) _tmem_load_1[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 2))) _tmem_load_1[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 3))) _tmem_load_1[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 4))) _tmem_load_1[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 5))) _tmem_load_1[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 6))) _tmem_load_1[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 7))) _tmem_load_1[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 8))) _tmem_load_1[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 9))) _tmem_load_1[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 10))) _tmem_load_1[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 11))) _tmem_load_1[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 12))) _tmem_load_1[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 13))) _tmem_load_1[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 14))) _tmem_load_1[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 15))) _tmem_load_1[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 16))) _tmem_load_1[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 17))) _tmem_load_1[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 18))) _tmem_load_1[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 19))) _tmem_load_1[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 20))) _tmem_load_1[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 21))) _tmem_load_1[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 22))) _tmem_load_1[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 23))) _tmem_load_1[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 24))) _tmem_load_1[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 25))) _tmem_load_1[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 26))) _tmem_load_1[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 27))) _tmem_load_1[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 28))) _tmem_load_1[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 29))) _tmem_load_1[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 30))) _tmem_load_1[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 31))) _tmem_load_1[127] = -CUDART_INF_F;
                    }
                    float2 _reg_reduce_max2_37 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_37);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_37);
                    row_max_x32_accum(&_tmem_load_1[64], _reg_reduce_max2_37);
                    row_max_x32_accum(&_tmem_load_1[96], _reg_reduce_max2_37);
                    float new_max_1 = row_max_reduce(_reg_reduce_max2_37);
                    new_max_1 = max_noftz(new_max_1, row_max_val);
                    float new_max_scaled_1 = ((new_max_1 == -CUDART_INF_F) ? 0.0f : new_max_1) * smx_scale;
                    float acc_scale_1;
                    float _fma_1 = __fmaf_rn(row_max_val, smx_scale, -new_max_scaled_1);
                    if (row_max_val > -CUDART_INF_F) {
                        float _exp2_38 = approx_exp2(_fma_1);
                        acc_scale_1 = _exp2_38;
                    } else {
                        acc_scale_1 = 1.0f;
                    }
                    row_max_val = new_max_1;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale_1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    int p_base_1 = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    float p_log2_bias_1 = 8.807355f - new_max_scaled_1;
                    unsigned int pv0_1[16];
                    unsigned int pv1_1[8];
                    unsigned int pv2_1[8];
                    float block_sum0_1;
                    float block_sum1_1;
                    float block_sum2_1;
                    const bool _selected_wait_cond_39 = stage == 0;
                    mbarrier_wait((_selected_wait_cond_39 ? (order_p01_0_addr) : (order_p01_1_addr)), (_selected_wait_cond_39 ? ((int)((unsigned int)order_p01_seen % 2)) : ((int)((unsigned int)order_p01_seen % 2))));
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    const float2 _fma_b2_40 = {smx_scale, smx_scale};
                    const float2 _fma_c2_41 = {p_log2_bias_1, p_log2_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_40, _fma_c2_41);
                    #pragma unroll
                    for (int _le = 0; _le < 128; _le++) {
                        _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                    }
                    float2 _reg_reduce_sum2_43 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[4], _tmem_load_1[5]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[6], _tmem_load_1[7]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[8], _tmem_load_1[9]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[10], _tmem_load_1[11]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[12], _tmem_load_1[13]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[14], _tmem_load_1[15]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[16], _tmem_load_1[17]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[18], _tmem_load_1[19]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[20], _tmem_load_1[21]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[22], _tmem_load_1[23]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[24], _tmem_load_1[25]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[26], _tmem_load_1[27]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[28], _tmem_load_1[29]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[30], _tmem_load_1[31]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[32], _tmem_load_1[33]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[34], _tmem_load_1[35]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[36], _tmem_load_1[37]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[38], _tmem_load_1[39]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[40], _tmem_load_1[41]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[42], _tmem_load_1[43]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[44], _tmem_load_1[45]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[46], _tmem_load_1[47]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[48], _tmem_load_1[49]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[50], _tmem_load_1[51]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[52], _tmem_load_1[53]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[54], _tmem_load_1[55]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[56], _tmem_load_1[57]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[58], _tmem_load_1[59]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[60], _tmem_load_1[61]));
                    _reg_reduce_sum2_43 = add_f32x2_rn_noftz(_reg_reduce_sum2_43, make_float2(_tmem_load_1[62], _tmem_load_1[63]));
                    float _reg_array_reduce_42 = _reg_reduce_sum2_43.x + _reg_reduce_sum2_43.y;
                    block_sum0_1 = _reg_array_reduce_42;
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
                        pv0_1[0] = _packed;
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
                        pv0_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]),
                                               "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]));
                        pv0_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]),
                                               "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]));
                        pv0_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]),
                                               "f"(_tmem_load_1[18]), "f"(_tmem_load_1[19]));
                        pv0_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]),
                                               "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]));
                        pv0_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[24]), "f"(_tmem_load_1[25]),
                                               "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]));
                        pv0_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]),
                                               "f"(_tmem_load_1[30]), "f"(_tmem_load_1[31]));
                        pv0_1[7] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]),
                                               "f"(_tmem_load_1[34]), "f"(_tmem_load_1[35]));
                        pv0_1[8] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]),
                                               "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]));
                        pv0_1[9] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[40]), "f"(_tmem_load_1[41]),
                                               "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]));
                        pv0_1[10] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]),
                                               "f"(_tmem_load_1[46]), "f"(_tmem_load_1[47]));
                        pv0_1[11] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]),
                                               "f"(_tmem_load_1[50]), "f"(_tmem_load_1[51]));
                        pv0_1[12] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]),
                                               "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]));
                        pv0_1[13] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[56]), "f"(_tmem_load_1[57]),
                                               "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]));
                        pv0_1[14] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]),
                                               "f"(_tmem_load_1[62]), "f"(_tmem_load_1[63]));
                        pv0_1[15] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base_1), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&pv0_1[15])));
                    float2 _reg_reduce_sum2_45 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 0], _tmem_load_1[(64) + 1]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 2], _tmem_load_1[(64) + 3]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 4], _tmem_load_1[(64) + 5]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 6], _tmem_load_1[(64) + 7]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 8], _tmem_load_1[(64) + 9]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 10], _tmem_load_1[(64) + 11]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 12], _tmem_load_1[(64) + 13]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 14], _tmem_load_1[(64) + 15]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 16], _tmem_load_1[(64) + 17]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 18], _tmem_load_1[(64) + 19]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 20], _tmem_load_1[(64) + 21]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 22], _tmem_load_1[(64) + 23]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 24], _tmem_load_1[(64) + 25]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 26], _tmem_load_1[(64) + 27]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 28], _tmem_load_1[(64) + 29]));
                    _reg_reduce_sum2_45 = add_f32x2_rn_noftz(_reg_reduce_sum2_45, make_float2(_tmem_load_1[(64) + 30], _tmem_load_1[(64) + 31]));
                    float _reg_array_reduce_44 = _reg_reduce_sum2_45.x + _reg_reduce_sum2_45.y;
                    block_sum1_1 = _reg_array_reduce_44;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]),
                                               "f"(_tmem_load_1[66]), "f"(_tmem_load_1[67]));
                        pv1_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]),
                                               "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]));
                        pv1_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[72]), "f"(_tmem_load_1[73]),
                                               "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]));
                        pv1_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]),
                                               "f"(_tmem_load_1[78]), "f"(_tmem_load_1[79]));
                        pv1_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]),
                                               "f"(_tmem_load_1[82]), "f"(_tmem_load_1[83]));
                        pv1_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]),
                                               "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]));
                        pv1_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[88]), "f"(_tmem_load_1[89]),
                                               "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]));
                        pv1_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]),
                                               "f"(_tmem_load_1[94]), "f"(_tmem_load_1[95]));
                        pv1_1[7] = _packed;
                    }
                    tmem_st_x8_u32(p_base_1 + 16, (const uint32_t*)pv1_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    float2 _reg_reduce_sum2_47 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 0], _tmem_load_1[(96) + 1]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 2], _tmem_load_1[(96) + 3]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 4], _tmem_load_1[(96) + 5]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 6], _tmem_load_1[(96) + 7]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 8], _tmem_load_1[(96) + 9]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 10], _tmem_load_1[(96) + 11]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 12], _tmem_load_1[(96) + 13]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 14], _tmem_load_1[(96) + 15]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 16], _tmem_load_1[(96) + 17]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 18], _tmem_load_1[(96) + 19]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 20], _tmem_load_1[(96) + 21]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 22], _tmem_load_1[(96) + 23]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 24], _tmem_load_1[(96) + 25]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 26], _tmem_load_1[(96) + 27]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 28], _tmem_load_1[(96) + 29]));
                    _reg_reduce_sum2_47 = add_f32x2_rn_noftz(_reg_reduce_sum2_47, make_float2(_tmem_load_1[(96) + 30], _tmem_load_1[(96) + 31]));
                    float _reg_array_reduce_46 = _reg_reduce_sum2_47.x + _reg_reduce_sum2_47.y;
                    block_sum2_1 = _reg_array_reduce_46;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]),
                                               "f"(_tmem_load_1[98]), "f"(_tmem_load_1[99]));
                        pv2_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]),
                                               "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]));
                        pv2_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[104]), "f"(_tmem_load_1[105]),
                                               "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]));
                        pv2_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]),
                                               "f"(_tmem_load_1[110]), "f"(_tmem_load_1[111]));
                        pv2_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]),
                                               "f"(_tmem_load_1[114]), "f"(_tmem_load_1[115]));
                        pv2_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]),
                                               "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]));
                        pv2_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[120]), "f"(_tmem_load_1[121]),
                                               "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]));
                        pv2_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]),
                                               "f"(_tmem_load_1[126]), "f"(_tmem_load_1[127]));
                        pv2_1[7] = _packed;
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_seen += 1;
                    tmem_st_x8_u32(p_base_1 + 24, (const uint32_t*)pv2_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    row_sum_val = __fmaf_rn(row_sum_val, acc_scale_1, block_sum0_1 + block_sum1_1 + block_sum2_1);
                }
                sScale[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum_val;
                sScale[warp % 4 * 32 + lane + scale_off + 4 * BLOCK_M] = row_max_val;
                mbarrier_arrive(stats_full_addr + (stage) * 8);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            float smx_scale_c = softmax_scale_log2;
            float out_scale_c = output_scale;
            if (scale_source != 0) {
                smx_scale_c = bmm1_scale_ptr[0];
                unsigned int raw_bits_c = __as_u32(smx_scale_c);
                if (bmm1_is_log2 == 0) {
                    smx_scale_c = (float)((double)smx_scale_c * 1.4426950216293335);
                }
                smx_scale_c = (float)((double)smx_scale_c / 1.4426950408889634 * 1.4426950408889634);
                if (bmm1_is_log2 == 0 && (raw_bits_c & 2147483647) - 1 < 8388607) {
                    double wide_c = (double)(int)(raw_bits_c & 8388607) * 1.401298464324817e-45;
                    if (raw_bits_c >= 2147483648u) {
                        wide_c = -wide_c;
                    }
                    smx_scale_c = (float)(wide_c * 1.4426950216293335);
                    smx_scale_c = (float)((double)smx_scale_c / 1.4426950408889634 * 1.4426950408889634);
                }
                out_scale_c = bmm2_scale_ptr[0];
            }
            unsigned int total_tiles_1 = ((uint32_t)(num_m_blocks.divisor)) * (unsigned int)total_bh;
            unsigned int total_units_1 = total_tiles_1 + (num_splits - 1) * ((uint32_t)(split_tiles.divisor));
            unsigned int max_rounds_1 = total_units_1 + 1;
            int p_full_block = 0;
            unsigned int _phase_work_id_full_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            unsigned int _phase_stats_full_0 = 0;
            unsigned int _phase_stats_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_1 = 0; tile_iter_1 < max_rounds_1; tile_iter_1++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_1);
                _phase_work_id_full_0_1 ^= 1;
                unsigned int ticket_1 = work_id_slot[0];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_id_empty_addr);
                if (ticket_1 >= total_units_1) {
                    break;
                }
                unsigned int ut_units_1 = num_splits * ((uint32_t)(split_tiles.divisor));
                unsigned int ut_tile_1 = ticket_1 - ut_units_1 + ((uint32_t)(split_tiles.divisor));
                unsigned int ut_split_1 = 0;
                unsigned int ut_in_split_1 = 0;
                if (ticket_1 < ut_units_1) {
                    uint32_t ut_s_1 = (split_tiles.divisor != 1) ? (__umulhi((uint32_t)(ticket_1), split_tiles.multiplier) >> split_tiles.shift_right) : (uint32_t)(ticket_1);
                    uint32_t ut_j_1 = (uint32_t)(ticket_1) - ut_s_1 * (uint32_t)(split_tiles.divisor);
                    ut_tile_1 = ut_j_1;
                    ut_split_1 = ut_s_1;
                    ut_in_split_1 = 1;
                }
                unsigned int m_block_1;
                unsigned int bh_1;
                uint32_t tile_section_1 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(ut_tile_1 >> 3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(ut_tile_1 >> 3);
                uint32_t tile_sw_m_1 = (uint32_t)(ut_tile_1 >> 3) - tile_section_1 * (uint32_t)(num_m_blocks.divisor);
                if (tile_section_1 >= full_sections) {
                    unsigned int tile_tail_1 = ut_tile_1 - tile_section_1 * 8 * ((uint32_t)(num_m_blocks.divisor));
                    uint32_t tail_unit_1 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(tile_tail_1), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(tile_tail_1);
                    uint32_t tail_m_1 = (uint32_t)(tile_tail_1) - tail_unit_1 * (uint32_t)(num_m_blocks.divisor);
                    m_block_1 = tail_m_1;
                    bh_1 = tile_section_1 * 8 + tail_unit_1;
                } else {
                    m_block_1 = tile_sw_m_1;
                    bh_1 = tile_section_1 * 8 + (ut_tile_1 & 7);
                }
                m_block_1 = ((uint32_t)(num_m_blocks.divisor)) - 1 - m_block_1;
                uint32_t b_idx_1 = (num_bh_units.divisor != 1) ? (__umulhi((uint32_t)(bh_1), num_bh_units.multiplier) >> num_bh_units.shift_right) : (uint32_t)(bh_1);
                uint32_t unit_1 = (uint32_t)(bh_1) - b_idx_1 * (uint32_t)(num_bh_units.divisor);
                int seqlen_kv_bh_1 = seq_lens_kv[b_idx_1];
                int seqlen_q_bh_1 = cu_seq_lens_q[b_idx_1 + 1] - cu_seq_lens_q[b_idx_1];
                unsigned int num_n_blocks_1 = (seqlen_kv_bh_1 + BLOCK_N - 1) / BLOCK_N;
                int tile_kv_shift_1 = seqlen_kv_bh_1 - seqlen_q_bh_1;
                unsigned int tile_max_n_1 = (m_block_1 * 2 + 2) * 32 + (unsigned int)tile_kv_shift_1;
                if (tile_max_n_1 < (unsigned int)seqlen_kv_bh_1) {
                    num_n_blocks_1 = (tile_max_n_1 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                }
                unsigned int _max_0_1 = ((num_n_blocks_1) > (1) ? (num_n_blocks_1) : (1));
                num_n_blocks_1 = _max_0_1;
                unsigned int us_count_1 = 1;
                unsigned int us_hi_1 = num_n_blocks_1;
                unsigned int us_blocks_1 = num_n_blocks_1;
                if (ut_in_split_1 != 0) {
                    unsigned int _min_1_1 = (((num_n_blocks_1 + split_blocks - 1) / split_blocks) < (num_splits) ? ((num_n_blocks_1 + split_blocks - 1) / split_blocks) : (num_splits));
                    us_count_1 = _min_1_1;
                    us_hi_1 = num_n_blocks_1 - ut_split_1 * split_blocks;
                    if (ut_split_1 + 1 < us_count_1) {
                        us_blocks_1 = split_blocks;
                    } else {
                        us_blocks_1 = us_hi_1;
                    }
                }
                int q_head_base = unit_1 * 4;
                int row_tok = (warp % 4 * 32 + lane) / 4;
                int row_head = q_head_base + (warp % 4 * 32 + lane) % 4;
                int tok_base = cu_seq_lens_q[b_idx_1];
                mbarrier_arrive(p_full_addr);
                mbarrier_arrive(p_full_addr + 8);
                p_full_block += 1;
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < us_blocks_1; n_iter_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale = sScale[warp % 4 * 32 + lane];
                    int _warp_vote_2 = __any_sync(0xFFFFFFFF, scale < 1.0f);
                    if (_warp_vote_2 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < HEAD_DIM / 16; cr_col++) {
                            int cr_addr = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr);
                            const float2 _scale2_3 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_3);
                            tmem_st_x16_f32(cr_addr, _tmem_load_2);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr + 8);
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale1 = sScale[warp % 4 * 32 + lane + BLOCK_M];
                    int _warp_vote_4 = __any_sync(0xFFFFFFFF, scale1 < 1.0f);
                    if (_warp_vote_4 != 0) {
                        #pragma unroll
                        for (int cr_col_1 = 0; cr_col_1 < HEAD_DIM / 16; cr_col_1++) {
                            int cr_addr_1 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col_1 * 16);
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], cr_addr_1);
                            const float2 _scale2_5 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_5);
                            tmem_st_x16_f32(cr_addr_1, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + 8);
                    p_full_block += 1;
                    mbarrier_arrive(corr_done_addr);
                }
                mbarrier_arrive(corr_done_addr + 8);
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                _phase_o_full_1 ^= 1;
                mbarrier_wait(stats_full_addr, _phase_stats_full_0);
                _phase_stats_full_0 ^= 1;
                mbarrier_wait(stats_full_addr + 8, _phase_stats_full_1);
                _phase_stats_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (us_count_1 == 1) {
                    #pragma unroll
                    for (int stage_1 = 0; stage_1 < 2; stage_1++) {
                        int tmem_o_off = ((stage_1 == 0) ? 256 : 384);
                        int s_off = stage_1 * BLOCK_M;
                        float final_sum = sScale[warp % 4 * 32 + lane + s_off + 2 * BLOCK_M];
                        float final_max = sScale[warp % 4 * 32 + lane + s_off + 4 * BLOCK_M];
                        float final_scale;
                        if (final_sum != 0.0f && final_sum == final_sum) {
                            float _rcp_6 = approx_rcp(final_sum);
                            final_scale = _rcp_6;
                            final_scale = final_scale * out_scale_c;
                        } else {
                            final_scale = 0.0f;
                        }
                        int q_tok0 = (m_block_1 * 2 + (unsigned int)stage_1) * 32;
                        if (q_tok0 < seqlen_q_bh_1) {
                            int row_valid = warp % 4 * 32 + lane < 128 && seqlen_q_bh_1 > q_tok0 + row_tok;
                            long long o_tok = tok_base + q_tok0 + row_tok;
                            long long o_row = o_tok * (long long)num_q_heads + (long long)row_head;
                            #pragma unroll
                            for (int ce_col = 0; ce_col < HEAD_DIM / 16; ce_col++) {
                                int ce_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(ce_col * 16);
                                float _tmem_load_4[16];
                                tmem_ld_x16(&_tmem_load_4[0], ce_addr);
                                if (row_valid != 0) {
                                    long long o_elem = o_row * (long long)HEAD_DIM + (long long)(ce_col * 16);
                                    const float2 _scale2_7 = {final_scale, final_scale};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_7);
                                    unsigned int _packed_0[4];
                                    {
                                        uint32_t _packed;
                                        asm volatile("{\n\t"
                                            ".reg .b16 _lo;\n\t"
                                            ".reg .b16 _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}"
                                            : "=r"(_packed) : "f"(_tmem_load_4[0]), "f"(_tmem_load_4[1]),
                                                               "f"(_tmem_load_4[2]), "f"(_tmem_load_4[3]));
                                        _packed_0[0] = _packed;
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
                                            : "=r"(_packed) : "f"(_tmem_load_4[4]), "f"(_tmem_load_4[5]),
                                                               "f"(_tmem_load_4[6]), "f"(_tmem_load_4[7]));
                                        _packed_0[1] = _packed;
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
                                            : "=r"(_packed) : "f"(_tmem_load_4[8]), "f"(_tmem_load_4[9]),
                                                               "f"(_tmem_load_4[10]), "f"(_tmem_load_4[11]));
                                        _packed_0[2] = _packed;
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
                                            : "=r"(_packed) : "f"(_tmem_load_4[12]), "f"(_tmem_load_4[13]),
                                                               "f"(_tmem_load_4[14]), "f"(_tmem_load_4[15]));
                                        _packed_0[3] = _packed;
                                    }
                                    reinterpret_cast<int4*>(O_ptr + o_elem)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                } else {
                    int crow = warp % 4 * 32 + lane;
                    #pragma unroll
                    for (int pstage = 0; pstage < 2; pstage++) {
                        int p_tmem_off = ((pstage == 0) ? 256 : 384);
                        int p_soff = pstage * BLOCK_M;
                        float p_sum = sScale[crow + p_soff + 2 * BLOCK_M];
                        float p_max = sScale[crow + p_soff + 4 * BLOCK_M];
                        int p_tok0 = (m_block_1 * 2 + (unsigned int)pstage) * 32;
                        if (p_tok0 < seqlen_q_bh_1) {
                            int p_valid = crow < 128 && seqlen_q_bh_1 > p_tok0 + row_tok;
                            unsigned int p_base_2 = (ticket_1 * 2 + (unsigned int)pstage) * 16384 + (unsigned int)crow * 4;
                            #pragma unroll
                            for (int pc_col = 0; pc_col < HEAD_DIM / 16; pc_col++) {
                                int pc_addr = taddr + (unsigned int)p_tmem_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(pc_col * 16);
                                float _tmem_load_5[16];
                                tmem_ld_x16(&_tmem_load_5[0], pc_addr);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                if (p_valid != 0) {
                                    #pragma unroll
                                    for (int pq = 0; pq < 4; pq++) {
                                        {
                                            float4 _v4 = make_float4(_tmem_load_5[pq * 4 + 0], _tmem_load_5[pq * 4 + 1], _tmem_load_5[pq * 4 + 2], _tmem_load_5[pq * 4 + 3]);
                                            *reinterpret_cast<float4*>((partial_o + (p_base_2 + (unsigned int)((pc_col * 4 + pq) * 512))) + 0) = _v4;
                                        }
                                    }
                                }
                            }
                            if (p_valid != 0) {
                                float p_ml[2];
                                p_ml[0] = p_max;
                                p_ml[1] = p_sum;
                                {
                                    float2 _v2 = make_float2(p_ml[0 + 0], p_ml[0 + 1]);
                                    *reinterpret_cast<float2*>((partial_ml + (((ticket_1 * 2 + (unsigned int)pstage) * 128 + (unsigned int)crow) * 4)) + 0) = _v2;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (crow == 0) {
                        uint32_t done_old;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(done_old) : "l"(&tile_counters[ut_tile_1]), "r"(static_cast<uint32_t>(us_count_1 - 1)) : "memory");
                        merge_flag[0] = ((done_old == us_count_1 - 1) ? 1 : 0);
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    unsigned int merge_now = merge_flag[0];
                    if (merge_now != 0) {
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        int mv_tok0 = m_block_1 * 2 * 32;
                        int mv_tok1 = mv_tok0 + 32;
                        unsigned int mv_ok0 = 0;
                        if (mv_tok0 < seqlen_q_bh_1) {
                            if (crow < 128 && seqlen_q_bh_1 > mv_tok0 + row_tok) {
                                mv_ok0 = 1;
                            }
                        }
                        unsigned int mv_ok1 = 0;
                        if (mv_tok1 < seqlen_q_bh_1) {
                            if (crow < 128 && seqlen_q_bh_1 > mv_tok1 + row_tok) {
                                mv_ok1 = 1;
                            }
                        }
                        #pragma unroll
                        for (int ml_st = 0; ml_st < 2; ml_st++) {
                            unsigned int ml_ok = ((ml_st == 0) ? mv_ok0 : mv_ok1);
                            #pragma unroll 1
                            for (unsigned int ml_s = 0; ml_s < us_count_1; ml_s++) {
                                unsigned int ml_slot = ml_s * ((uint32_t)(split_tiles.divisor)) + ut_tile_1;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p;\n\t"
                                    "setp.ne.b32 p, %0, 0;\n\t"
                                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                    "}"
                                    :: "r"((ml_ok != 0) ? 1 : 0), "r"(merge_ml_addr + (((unsigned int)ml_st * 16 + ml_s) * 128 + (unsigned int)crow) * 16), "l"(partial_ml + (((ml_slot * 2 + (unsigned int)ml_st) * 128 + (unsigned int)crow) * 4)));
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                        unsigned int iss_st = 0;
                        unsigned int iss_cg = 0;
                        unsigned int iss_s = 0;
                        #pragma unroll
                        for (int pro_i = 0; pro_i < 8; pro_i++) {
                            unsigned int pro_ok = ((iss_st == 0) ? mv_ok0 : mv_ok1);
                            if (iss_st >= 2) {
                                pro_ok = 0;
                            }
                            unsigned int mi_slot = iss_s * ((uint32_t)(split_tiles.divisor)) + ut_tile_1;
                            unsigned int mi_src = (mi_slot * 2 + iss_st) * 16384 + (iss_cg * 512 + (unsigned int)crow) * 4;
                            #pragma unroll
                            for (int mi_q = 0; mi_q < 4; mi_q++) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p;\n\t"
                                    "setp.ne.b32 p, %0, 0;\n\t"
                                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                    "}"
                                    :: "r"((pro_ok != 0) ? 1 : 0), "r"(merge_ring_addr + (((unsigned int)pro_i * 4 + (unsigned int)mi_q) * 128 + (unsigned int)crow) * 16), "l"(partial_o + (mi_src + (unsigned int)(mi_q * 512))));
                            }
                            asm volatile("cp.async.commit_group;");
                            iss_s = iss_s + 1;
                            if (iss_s == us_count_1) {
                                iss_s = 0;
                                iss_cg = iss_cg + 1;
                                if (iss_cg == 8) {
                                    iss_cg = 0;
                                    iss_st = iss_st + 1;
                                }
                            }
                        }
                        asm volatile("cp.async.wait_group 8;");
                        unsigned int cons_n = 0;
                        #pragma unroll
                        for (int cst = 0; cst < 2; cst++) {
                            int c_tok0 = (m_block_1 * 2 + (unsigned int)cst) * 32;
                            unsigned int c_ok = ((cst == 0) ? mv_ok0 : mv_ok1);
                            float sm_max = -CUDART_INF_F;
                            #pragma unroll 1
                            for (unsigned int msa = 0; msa < us_count_1; msa++) {
                                float sm_m = merge_ml[(((unsigned int)cst * 16 + msa) * 128 + (unsigned int)crow) * 4];
                                sm_max = max_noftz(sm_max, sm_m);
                            }
                            float sm_scaled = ((sm_max == -CUDART_INF_F) ? 0.0f : sm_max) * smx_scale_c;
                            float sm_sum = 0.0f;
                            #pragma unroll 1
                            for (unsigned int msb = 0; msb < us_count_1; msb++) {
                                unsigned int sb_idx = (((unsigned int)cst * 16 + msb) * 128 + (unsigned int)crow) * 4;
                                float sb_m = merge_ml[sb_idx];
                                float sb_l = merge_ml[sb_idx + 1];
                                float _exp2_8 = approx_exp2(__fmaf_rn(sb_m, smx_scale_c, -sm_scaled));
                                sm_sum = __fmaf_rn(_exp2_8, sb_l, sm_sum);
                            }
                            float sm_fscale;
                            if (sm_sum != 0.0f && sm_sum == sm_sum) {
                                float _rcp_9 = approx_rcp(sm_sum);
                                sm_fscale = _rcp_9;
                                sm_fscale = sm_fscale * out_scale_c;
                            } else {
                                sm_fscale = 0.0f;
                            }
                            #pragma unroll 1
                            for (int ccg = 0; ccg < HEAD_DIM / 16; ccg++) {
                                float c_acc[16];
                                #pragma unroll
                                for (int ca = 0; ca < 16; ca++) {
                                    c_acc[ca] = 0.0f;
                                }
                                #pragma unroll 1
                                for (unsigned int cs = 0; cs < us_count_1; cs++) {
                                    asm volatile("cp.async.wait_group 7;");
                                    unsigned int c_slot = cons_n % 8;
                                    if (c_ok != 0) {
                                        float c_m = merge_ml[(((unsigned int)cst * 16 + cs) * 128 + (unsigned int)crow) * 4];
                                        float c_w = approx_exp2(__fmaf_rn(c_m, smx_scale_c, -sm_scaled));
                                        #pragma unroll
                                        for (int cq = 0; cq < 4; cq++) {
                                            float c_v[4];
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&c_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&c_v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c_v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c_v[(0) + 3]))
                                                : "r"(merge_ring_addr + ((c_slot * 4 + (unsigned int)cq) * 128 + (unsigned int)crow) * 16));
                                            #pragma unroll
                                            for (int cb = 0; cb < 4; cb++) {
                                                c_acc[cq * 4 + cb] = __fmaf_rn(c_v[cb], c_w, c_acc[cq * 4 + cb]);
                                            }
                                        }
                                    }
                                    unsigned int nx_ok = ((iss_st == 0) ? mv_ok0 : mv_ok1);
                                    if (iss_st >= 2) {
                                        nx_ok = 0;
                                    }
                                    unsigned int mi_slot_1 = iss_s * ((uint32_t)(split_tiles.divisor)) + ut_tile_1;
                                    unsigned int mi_src_1 = (mi_slot_1 * 2 + iss_st) * 16384 + (iss_cg * 512 + (unsigned int)crow) * 4;
                                    #pragma unroll
                                    for (int mi_q_1 = 0; mi_q_1 < 4; mi_q_1++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .pred p;\n\t"
                                            "setp.ne.b32 p, %0, 0;\n\t"
                                            "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                            "}"
                                            :: "r"((nx_ok != 0) ? 1 : 0), "r"(merge_ring_addr + (((unsigned int)c_slot * 4 + (unsigned int)mi_q_1) * 128 + (unsigned int)crow) * 16), "l"(partial_o + (mi_src_1 + (unsigned int)(mi_q_1 * 512))));
                                    }
                                    asm volatile("cp.async.commit_group;");
                                    iss_s = iss_s + 1;
                                    if (iss_s == us_count_1) {
                                        iss_s = 0;
                                        iss_cg = iss_cg + 1;
                                        if (iss_cg == 8) {
                                            iss_cg = 0;
                                            iss_st = iss_st + 1;
                                        }
                                    }
                                    cons_n = cons_n + 1;
                                }
                                if (c_ok != 0) {
                                    long long c_row = (long long)(tok_base + c_tok0 + row_tok) * (long long)num_q_heads + (long long)row_head;
                                    const float2 _scale2_10 = {sm_fscale, sm_fscale};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(c_acc)[_ls], _scale2_10);
                                    unsigned int c_packed[4];
                                    {
                                        uint32_t _packed;
                                        asm volatile("{\n\t"
                                            ".reg .b16 _lo;\n\t"
                                            ".reg .b16 _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}"
                                            : "=r"(_packed) : "f"(c_acc[0]), "f"(c_acc[1]),
                                                               "f"(c_acc[2]), "f"(c_acc[3]));
                                        c_packed[0] = _packed;
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
                                            : "=r"(_packed) : "f"(c_acc[4]), "f"(c_acc[5]),
                                                               "f"(c_acc[6]), "f"(c_acc[7]));
                                        c_packed[1] = _packed;
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
                                            : "=r"(_packed) : "f"(c_acc[8]), "f"(c_acc[9]),
                                                               "f"(c_acc[10]), "f"(c_acc[11]));
                                        c_packed[2] = _packed;
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
                                            : "=r"(_packed) : "f"(c_acc[12]), "f"(c_acc[13]),
                                                               "f"(c_acc[14]), "f"(c_acc[15]));
                                        c_packed[3] = _packed;
                                    }
                                    reinterpret_cast<int4*>(O_ptr + (c_row * (long long)HEAD_DIM + (long long)(ccg * 16)))[0] = reinterpret_cast<int4*>(c_packed)[0];
                                }
                            }
                        }
                        asm volatile("cp.async.wait_group 0;");
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int total_tiles_2 = ((uint32_t)(num_m_blocks.divisor)) * (unsigned int)total_bh;
            unsigned int total_units_2 = total_tiles_2 + (num_splits - 1) * ((uint32_t)(split_tiles.divisor));
            unsigned int max_rounds_2 = total_units_2 + 1;
            unsigned int mma_kv_stage = 0;
            unsigned int _phase_work_id_full_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_2 = 0; tile_iter_2 < max_rounds_2; tile_iter_2++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_2);
                _phase_work_id_full_0_2 ^= 1;
                unsigned int ticket_2 = work_id_slot[0];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_id_empty_addr);
                if (ticket_2 >= total_units_2) {
                    break;
                }
                unsigned int ut_units_2 = num_splits * ((uint32_t)(split_tiles.divisor));
                unsigned int ut_tile_2 = ticket_2 - ut_units_2 + ((uint32_t)(split_tiles.divisor));
                unsigned int ut_split_2 = 0;
                unsigned int ut_in_split_2 = 0;
                if (ticket_2 < ut_units_2) {
                    uint32_t ut_s_2 = (split_tiles.divisor != 1) ? (__umulhi((uint32_t)(ticket_2), split_tiles.multiplier) >> split_tiles.shift_right) : (uint32_t)(ticket_2);
                    uint32_t ut_j_2 = (uint32_t)(ticket_2) - ut_s_2 * (uint32_t)(split_tiles.divisor);
                    ut_tile_2 = ut_j_2;
                    ut_split_2 = ut_s_2;
                    ut_in_split_2 = 1;
                }
                unsigned int m_block_2;
                unsigned int bh_2;
                uint32_t tile_section_2 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(ut_tile_2 >> 3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(ut_tile_2 >> 3);
                uint32_t tile_sw_m_2 = (uint32_t)(ut_tile_2 >> 3) - tile_section_2 * (uint32_t)(num_m_blocks.divisor);
                if (tile_section_2 >= full_sections) {
                    unsigned int tile_tail_2 = ut_tile_2 - tile_section_2 * 8 * ((uint32_t)(num_m_blocks.divisor));
                    uint32_t tail_unit_2 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(tile_tail_2), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(tile_tail_2);
                    uint32_t tail_m_2 = (uint32_t)(tile_tail_2) - tail_unit_2 * (uint32_t)(num_m_blocks.divisor);
                    m_block_2 = tail_m_2;
                    bh_2 = tile_section_2 * 8 + tail_unit_2;
                } else {
                    m_block_2 = tile_sw_m_2;
                    bh_2 = tile_section_2 * 8 + (ut_tile_2 & 7);
                }
                m_block_2 = ((uint32_t)(num_m_blocks.divisor)) - 1 - m_block_2;
                uint32_t b_idx_2 = (num_bh_units.divisor != 1) ? (__umulhi((uint32_t)(bh_2), num_bh_units.multiplier) >> num_bh_units.shift_right) : (uint32_t)(bh_2);
                uint32_t unit_2 = (uint32_t)(bh_2) - b_idx_2 * (uint32_t)(num_bh_units.divisor);
                int seqlen_kv_bh_2 = seq_lens_kv[b_idx_2];
                int seqlen_q_bh_2 = cu_seq_lens_q[b_idx_2 + 1] - cu_seq_lens_q[b_idx_2];
                unsigned int num_n_blocks_2 = (seqlen_kv_bh_2 + BLOCK_N - 1) / BLOCK_N;
                int tile_kv_shift_2 = seqlen_kv_bh_2 - seqlen_q_bh_2;
                unsigned int tile_max_n_2 = (m_block_2 * 2 + 2) * 32 + (unsigned int)tile_kv_shift_2;
                if (tile_max_n_2 < (unsigned int)seqlen_kv_bh_2) {
                    num_n_blocks_2 = (tile_max_n_2 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                }
                unsigned int _max_0_2 = ((num_n_blocks_2) > (1) ? (num_n_blocks_2) : (1));
                num_n_blocks_2 = _max_0_2;
                unsigned int us_count_2 = 1;
                unsigned int us_hi_2 = num_n_blocks_2;
                unsigned int us_blocks_2 = num_n_blocks_2;
                if (ut_in_split_2 != 0) {
                    unsigned int _min_1_2 = (((num_n_blocks_2 + split_blocks - 1) / split_blocks) < (num_splits) ? ((num_n_blocks_2 + split_blocks - 1) / split_blocks) : (num_splits));
                    us_count_2 = _min_1_2;
                    us_hi_2 = num_n_blocks_2 - ut_split_2 * split_blocks;
                    if (ut_split_2 + 1 < us_count_2) {
                        us_blocks_2 = split_blocks;
                    } else {
                        us_blocks_2 = us_hi_2;
                    }
                }
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                _phase_q_full_1 ^= 1;
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 1);
                    }
                }
                elect_commit(s_full_addr);
                int _mma_a_lo_1 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024);
                {
                    uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 1);
                    }
                }
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                unsigned int first_pv = 1;
                #pragma unroll 1
                for (unsigned int n_iter_3 = 0; n_iter_3 < us_blocks_2 - 1; n_iter_3++) {
                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                    unsigned int v_stage = mma_kv_stage;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    int first_pv_flag = first_pv;
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_2), "r"(tmem_softmax_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
                    mma_ts_step(tmem_output_0, tmem_softmax_0 + 24, _mma_b_lo_3 + 768, 0x40004040, 136380432, 1);
                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int k_stage = mma_kv_stage;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    int _mma_a_lo_2 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136314896, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136314896, 1);
                        }
                    }
                    elect_commit(s_full_addr);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_softmax_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024);
                    mma_ts_step(tmem_output_1, tmem_softmax_1 + 24, _mma_b_lo_6 + 768, 0x40004040, 136380432, 1);
                    elect_commit(kv_empty_addr + (((mma_kv_stage + 1 < 3) ? mma_kv_stage + 1 : (unsigned int)0)) * 8);
                    int _mma_a_lo_3 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 136314896, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores_1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 136314896, 1);
                        }
                    }
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (((mma_kv_stage + 2 >= 3) ? mma_kv_stage + 2 - 3 : mma_kv_stage + 2)) * 8);
                    first_pv = 0;
                }
                elect_commit(q_empty_addr);
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                int first_pv_flag_1 = first_pv;
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_softmax_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                _phase_p_full_2_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                mma_ts_step(tmem_output_0, tmem_softmax_0 + 24, _mma_b_lo_9 + 768, 0x40004040, 136380432, 1);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_softmax_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                _phase_p_full_2_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024);
                mma_ts_step(tmem_output_1, tmem_softmax_1 + 24, _mma_b_lo_11 + 768, 0x40004040, 136380432, 1);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                elect_commit2(o_full_addr, o_full_addr + 8);
            }
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 13) {
        { // page_offsets_main
            unsigned int total_tiles_3 = ((uint32_t)(num_m_blocks.divisor)) * (unsigned int)total_bh;
            unsigned int total_units_3 = total_tiles_3 + (num_splits - 1) * ((uint32_t)(split_tiles.divisor));
            unsigned int max_rounds_3 = total_units_3 + 1;
            unsigned int po_stage = 0;
            unsigned int po_shift;
            po_shift = (unsigned int)page_shift;
            unsigned int _phase_work_id_full_0_3 = 0;
            unsigned int _phase_page_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_3 = 0; tile_iter_3 < max_rounds_3; tile_iter_3++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_3);
                _phase_work_id_full_0_3 ^= 1;
                unsigned int ticket_3 = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (ticket_3 >= total_units_3) {
                    break;
                }
                unsigned int ut_units_3 = num_splits * ((uint32_t)(split_tiles.divisor));
                unsigned int ut_tile_3 = ticket_3 - ut_units_3 + ((uint32_t)(split_tiles.divisor));
                unsigned int ut_split_3 = 0;
                unsigned int ut_in_split_3 = 0;
                if (ticket_3 < ut_units_3) {
                    uint32_t ut_s_3 = (split_tiles.divisor != 1) ? (__umulhi((uint32_t)(ticket_3), split_tiles.multiplier) >> split_tiles.shift_right) : (uint32_t)(ticket_3);
                    uint32_t ut_j_3 = (uint32_t)(ticket_3) - ut_s_3 * (uint32_t)(split_tiles.divisor);
                    ut_tile_3 = ut_j_3;
                    ut_split_3 = ut_s_3;
                    ut_in_split_3 = 1;
                }
                unsigned int m_block_3;
                unsigned int bh_3;
                uint32_t tile_section_3 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(ut_tile_3 >> 3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(ut_tile_3 >> 3);
                uint32_t tile_sw_m_3 = (uint32_t)(ut_tile_3 >> 3) - tile_section_3 * (uint32_t)(num_m_blocks.divisor);
                if (tile_section_3 >= full_sections) {
                    unsigned int tile_tail_3 = ut_tile_3 - tile_section_3 * 8 * ((uint32_t)(num_m_blocks.divisor));
                    uint32_t tail_unit_3 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(tile_tail_3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(tile_tail_3);
                    uint32_t tail_m_3 = (uint32_t)(tile_tail_3) - tail_unit_3 * (uint32_t)(num_m_blocks.divisor);
                    m_block_3 = tail_m_3;
                    bh_3 = tile_section_3 * 8 + tail_unit_3;
                } else {
                    m_block_3 = tile_sw_m_3;
                    bh_3 = tile_section_3 * 8 + (ut_tile_3 & 7);
                }
                m_block_3 = ((uint32_t)(num_m_blocks.divisor)) - 1 - m_block_3;
                uint32_t b_idx_3 = (num_bh_units.divisor != 1) ? (__umulhi((uint32_t)(bh_3), num_bh_units.multiplier) >> num_bh_units.shift_right) : (uint32_t)(bh_3);
                uint32_t unit_3 = (uint32_t)(bh_3) - b_idx_3 * (uint32_t)(num_bh_units.divisor);
                int seqlen_kv_bh_3 = seq_lens_kv[b_idx_3];
                int seqlen_q_bh_3 = cu_seq_lens_q[b_idx_3 + 1] - cu_seq_lens_q[b_idx_3];
                unsigned int num_n_blocks_3 = (seqlen_kv_bh_3 + BLOCK_N - 1) / BLOCK_N;
                int tile_kv_shift_3 = seqlen_kv_bh_3 - seqlen_q_bh_3;
                unsigned int tile_max_n_3 = (m_block_3 * 2 + 2) * 32 + (unsigned int)tile_kv_shift_3;
                if (tile_max_n_3 < (unsigned int)seqlen_kv_bh_3) {
                    num_n_blocks_3 = (tile_max_n_3 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                }
                unsigned int _max_0_3 = ((num_n_blocks_3) > (1) ? (num_n_blocks_3) : (1));
                num_n_blocks_3 = _max_0_3;
                unsigned int us_count_3 = 1;
                unsigned int us_hi_3 = num_n_blocks_3;
                unsigned int us_blocks_3 = num_n_blocks_3;
                if (ut_in_split_3 != 0) {
                    unsigned int _min_1_3 = (((num_n_blocks_3 + split_blocks - 1) / split_blocks) < (num_splits) ? ((num_n_blocks_3 + split_blocks - 1) / split_blocks) : (num_splits));
                    us_count_3 = _min_1_3;
                    us_hi_3 = num_n_blocks_3 - ut_split_3 * split_blocks;
                    if (ut_split_3 + 1 < us_count_3) {
                        us_blocks_3 = split_blocks;
                    } else {
                        us_blocks_3 = us_hi_3;
                    }
                }
                int pt_base = b_idx_3 * (unsigned int)page_row_stride;
                int _max_2 = ((seqlen_kv_bh_3 - 1) > (0) ? (seqlen_kv_bh_3 - 1) : (0));
                unsigned int last_slot = (unsigned int)_max_2 >> po_shift;
                unsigned int block_pages = (((unsigned int)BLOCK_N >> po_shift) > (1) ? ((unsigned int)BLOCK_N >> po_shift) : (1));
                #pragma unroll 1
                for (unsigned int ni = 0; ni < us_blocks_3; ni++) {
                    unsigned int n = us_hi_3 - 1 - ni;
                    unsigned int first_slot = n * (unsigned int)BLOCK_N >> po_shift;
                    mbarrier_wait(page_empty_addr + (po_stage) * 8, _phase_page_empty);
                    if (elect_sync()) {
                        int pg_idx = po_stage * 16;
                        if (seqlen_kv_bh_3 == 0) {
                            #pragma unroll
                            for (int slot = 0; slot < 16; slot++) {
                                smem_pages[pg_idx + slot] = 1073741824;
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                if ((unsigned int)j < block_pages) {
                                    unsigned int _min_3 = ((first_slot + (unsigned int)j) < (last_slot) ? (first_slot + (unsigned int)j) : (last_slot));
                                    int s_j = _min_3;
                                    smem_pages[pg_idx + j] = page_table_k[pt_base + s_j];
                                    smem_pages[pg_idx + 8 + j] = page_table_v[pt_base + s_j];
                                }
                            }
                        }
                        mbarrier_arrive(page_full_addr + (po_stage) * 8);
                    }
                    po_stage += 1;
                    if (po_stage == 6) { po_stage = 0; _phase_page_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        { // scheduler_main
            unsigned int total_tiles_4 = ((uint32_t)(num_m_blocks.divisor)) * (unsigned int)total_bh;
            unsigned int total_units_4 = total_tiles_4 + (num_splits - 1) * ((uint32_t)(split_tiles.divisor));
            unsigned int max_rounds_4 = total_units_4 + 1;
            bool static_first = (unsigned int)num_ctas <= total_units_4;
            unsigned int split_units = num_splits * ((uint32_t)(split_tiles.divisor));
            unsigned int _phase_work_id_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_iter_4 = 0; tile_iter_4 < max_rounds_4; tile_iter_4++) {
                    mbarrier_wait(work_id_empty_addr, _phase_work_id_empty_0);
                    _phase_work_id_empty_0 ^= 1;
                    unsigned int _atomic_old_0;
                    if (static_first) {
                        if (tile_iter_4 == 0) {
                            _atomic_old_0 = (unsigned int)blockIdx.x;
                        } else {
                            unsigned int _atomic_fetch_add_0 = atomicAdd(dynamic_counter, 1);
                            _atomic_old_0 = _atomic_fetch_add_0 + (unsigned int)num_ctas;
                        }
                    } else {
                        unsigned int _atomic_fetch_add_1 = atomicAdd(dynamic_counter, 1);
                        _atomic_old_0 = _atomic_fetch_add_1;
                    }
                    #pragma unroll 1
                    for (unsigned int skip_iter = 0; skip_iter < max_rounds_4; skip_iter++) {
                        if (_atomic_old_0 >= split_units) {
                            break;
                        }
                        uint32_t skip_s = (split_tiles.divisor != 1) ? (__umulhi((uint32_t)(_atomic_old_0), split_tiles.multiplier) >> split_tiles.shift_right) : (uint32_t)(_atomic_old_0);
                        uint32_t skip_j = (uint32_t)(_atomic_old_0) - skip_s * (uint32_t)(split_tiles.divisor);
                        if (skip_s == 0) {
                            break;
                        }
                        unsigned int m_block_4;
                        unsigned int bh_4;
                        uint32_t tile_section_4 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(skip_j >> 3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(skip_j >> 3);
                        uint32_t tile_sw_m_4 = (uint32_t)(skip_j >> 3) - tile_section_4 * (uint32_t)(num_m_blocks.divisor);
                        if (tile_section_4 >= full_sections) {
                            unsigned int tile_tail_4 = skip_j - tile_section_4 * 8 * ((uint32_t)(num_m_blocks.divisor));
                            uint32_t tail_unit_4 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(tile_tail_4), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(tile_tail_4);
                            uint32_t tail_m_4 = (uint32_t)(tile_tail_4) - tail_unit_4 * (uint32_t)(num_m_blocks.divisor);
                            m_block_4 = tail_m_4;
                            bh_4 = tile_section_4 * 8 + tail_unit_4;
                        } else {
                            m_block_4 = tile_sw_m_4;
                            bh_4 = tile_section_4 * 8 + (skip_j & 7);
                        }
                        m_block_4 = ((uint32_t)(num_m_blocks.divisor)) - 1 - m_block_4;
                        uint32_t b_idx_4 = (num_bh_units.divisor != 1) ? (__umulhi((uint32_t)(bh_4), num_bh_units.multiplier) >> num_bh_units.shift_right) : (uint32_t)(bh_4);
                        uint32_t unit_4 = (uint32_t)(bh_4) - b_idx_4 * (uint32_t)(num_bh_units.divisor);
                        int seqlen_kv_bh_4 = seq_lens_kv[b_idx_4];
                        int seqlen_q_bh_4 = cu_seq_lens_q[b_idx_4 + 1] - cu_seq_lens_q[b_idx_4];
                        unsigned int num_n_blocks_4 = (seqlen_kv_bh_4 + BLOCK_N - 1) / BLOCK_N;
                        int tile_kv_shift_4 = seqlen_kv_bh_4 - seqlen_q_bh_4;
                        unsigned int tile_max_n_4 = (m_block_4 * 2 + 2) * 32 + (unsigned int)tile_kv_shift_4;
                        if (tile_max_n_4 < (unsigned int)seqlen_kv_bh_4) {
                            num_n_blocks_4 = (tile_max_n_4 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                        }
                        unsigned int _max_2_1 = ((num_n_blocks_4) > (1) ? (num_n_blocks_4) : (1));
                        num_n_blocks_4 = _max_2_1;
                        if (skip_s * split_blocks < num_n_blocks_4) {
                            break;
                        }
                        if (static_first) {
                            unsigned int _atomic_fetch_add_3 = atomicAdd(dynamic_counter, 1);
                            _atomic_old_0 = _atomic_fetch_add_3 + (unsigned int)num_ctas;
                        } else {
                            unsigned int _atomic_fetch_add_4 = atomicAdd(dynamic_counter, 1);
                            _atomic_old_0 = _atomic_fetch_add_4;
                        }
                    }
                    work_id_slot[0] = _atomic_old_0;
                    mbarrier_arrive(work_id_full_addr);
                    if (_atomic_old_0 >= total_units_4) {
                        unsigned int last_fetch = total_units_4 + (unsigned int)num_ctas - 1;
                        if (_atomic_old_0 == last_fetch) {
                            unsigned int counter_draws = last_fetch + 1;
                            if (static_first) {
                                counter_draws = total_units_4;
                            }
                            unsigned int reset_add = -counter_draws;
                            unsigned int _atomic_old_1 = atomicAdd(dynamic_counter, reset_add);
                        }
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 15) {
        { // load_main
            unsigned int total_tiles_5 = ((uint32_t)(num_m_blocks.divisor)) * (unsigned int)total_bh;
            unsigned int total_units_5 = total_tiles_5 + (num_splits - 1) * ((uint32_t)(split_tiles.divisor));
            unsigned int max_rounds_5 = total_units_5 + 1;
            unsigned int load_kv_stage = 0;
            unsigned int load_po_stage = 0;
            unsigned int ld_shift;
            ld_shift = (unsigned int)page_shift;
            unsigned int page_mask = (1 << ld_shift) - 1;
            int chunk_page0 = 0 >> ld_shift;
            int chunk_page1 = 16 >> ld_shift;
            int chunk_page2 = 32 >> ld_shift;
            int chunk_page3 = 48 >> ld_shift;
            int chunk_page4 = 64 >> ld_shift;
            int chunk_page5 = 80 >> ld_shift;
            int chunk_page6 = 96 >> ld_shift;
            int chunk_page7 = 112 >> ld_shift;
            unsigned int _phase_work_id_full_0_4 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_5 = 0; tile_iter_5 < max_rounds_5; tile_iter_5++) {
                mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_4);
                _phase_work_id_full_0_4 ^= 1;
                unsigned int ticket_4 = work_id_slot[0];
                mbarrier_arrive(work_id_empty_addr);
                if (ticket_4 >= total_units_5) {
                    break;
                }
                unsigned int ut_units_4 = num_splits * ((uint32_t)(split_tiles.divisor));
                unsigned int ut_tile_4 = ticket_4 - ut_units_4 + ((uint32_t)(split_tiles.divisor));
                unsigned int ut_split_4 = 0;
                unsigned int ut_in_split_4 = 0;
                if (ticket_4 < ut_units_4) {
                    uint32_t ut_s_4 = (split_tiles.divisor != 1) ? (__umulhi((uint32_t)(ticket_4), split_tiles.multiplier) >> split_tiles.shift_right) : (uint32_t)(ticket_4);
                    uint32_t ut_j_4 = (uint32_t)(ticket_4) - ut_s_4 * (uint32_t)(split_tiles.divisor);
                    ut_tile_4 = ut_j_4;
                    ut_split_4 = ut_s_4;
                    ut_in_split_4 = 1;
                }
                unsigned int m_block_5;
                unsigned int bh_5;
                uint32_t tile_section_5 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(ut_tile_4 >> 3), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(ut_tile_4 >> 3);
                uint32_t tile_sw_m_5 = (uint32_t)(ut_tile_4 >> 3) - tile_section_5 * (uint32_t)(num_m_blocks.divisor);
                if (tile_section_5 >= full_sections) {
                    unsigned int tile_tail_5 = ut_tile_4 - tile_section_5 * 8 * ((uint32_t)(num_m_blocks.divisor));
                    uint32_t tail_unit_5 = (num_m_blocks.divisor != 1) ? (__umulhi((uint32_t)(tile_tail_5), num_m_blocks.multiplier) >> num_m_blocks.shift_right) : (uint32_t)(tile_tail_5);
                    uint32_t tail_m_5 = (uint32_t)(tile_tail_5) - tail_unit_5 * (uint32_t)(num_m_blocks.divisor);
                    m_block_5 = tail_m_5;
                    bh_5 = tile_section_5 * 8 + tail_unit_5;
                } else {
                    m_block_5 = tile_sw_m_5;
                    bh_5 = tile_section_5 * 8 + (ut_tile_4 & 7);
                }
                m_block_5 = ((uint32_t)(num_m_blocks.divisor)) - 1 - m_block_5;
                uint32_t b_idx_5 = (num_bh_units.divisor != 1) ? (__umulhi((uint32_t)(bh_5), num_bh_units.multiplier) >> num_bh_units.shift_right) : (uint32_t)(bh_5);
                uint32_t unit_5 = (uint32_t)(bh_5) - b_idx_5 * (uint32_t)(num_bh_units.divisor);
                int seqlen_kv_bh_5 = seq_lens_kv[b_idx_5];
                int seqlen_q_bh_5 = cu_seq_lens_q[b_idx_5 + 1] - cu_seq_lens_q[b_idx_5];
                unsigned int num_n_blocks_5 = (seqlen_kv_bh_5 + BLOCK_N - 1) / BLOCK_N;
                int tile_kv_shift_5 = seqlen_kv_bh_5 - seqlen_q_bh_5;
                unsigned int tile_max_n_5 = (m_block_5 * 2 + 2) * 32 + (unsigned int)tile_kv_shift_5;
                if (tile_max_n_5 < (unsigned int)seqlen_kv_bh_5) {
                    num_n_blocks_5 = (tile_max_n_5 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                }
                unsigned int _max_0_4 = ((num_n_blocks_5) > (1) ? (num_n_blocks_5) : (1));
                num_n_blocks_5 = _max_0_4;
                unsigned int us_count_4 = 1;
                unsigned int us_hi_4 = num_n_blocks_5;
                unsigned int us_blocks_4 = num_n_blocks_5;
                if (ut_in_split_4 != 0) {
                    unsigned int _min_1_4 = (((num_n_blocks_5 + split_blocks - 1) / split_blocks) < (num_splits) ? ((num_n_blocks_5 + split_blocks - 1) / split_blocks) : (num_splits));
                    us_count_4 = _min_1_4;
                    us_hi_4 = num_n_blocks_5 - ut_split_4 * split_blocks;
                    if (ut_split_4 + 1 < us_count_4) {
                        us_blocks_4 = split_blocks;
                    } else {
                        us_blocks_4 = us_hi_4;
                    }
                }
                uint32_t group_head = (heads_per_group.divisor != 1) ? (__umulhi((uint32_t)(unit_5 * 4), heads_per_group.multiplier) >> heads_per_group.shift_right) : (uint32_t)(unit_5 * 4);
                uint32_t group_rank = (uint32_t)(unit_5 * 4) - group_head * (uint32_t)(heads_per_group.divisor);
                int kv_head = group_head;
                int tok_q0 = (unsigned int)cu_seq_lens_q[b_idx_5] + m_block_5 * 2 * 32;
                int tok_q1 = tok_q0 + 32;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    tma_5d_gmem2smem(smem_q0_addr, (&Q), 0, 0, tok_q0, 0, unit_5, q_full_addr);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + 8, 16384);
                    tma_5d_gmem2smem(smem_q1_addr, (&Q), 0, 0, tok_q1, 0, unit_5, q_full_addr + 8);
                }
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < us_blocks_4; ni_1++) {
                    unsigned int n_1 = us_hi_4 - 1 - ni_1;
                    int base_tok = n_1 * (unsigned int)BLOCK_N;
                    int w0 = (unsigned int)(base_tok + 0) & page_mask;
                    int w1 = (unsigned int)(base_tok + 16) & page_mask;
                    int w2 = (unsigned int)(base_tok + 32) & page_mask;
                    int w3 = (unsigned int)(base_tok + 48) & page_mask;
                    int w4 = (unsigned int)(base_tok + 64) & page_mask;
                    int w5 = (unsigned int)(base_tok + 80) & page_mask;
                    int w6 = (unsigned int)(base_tok + 96) & page_mask;
                    int w7 = (unsigned int)(base_tok + 112) & page_mask;
                    mbarrier_wait(page_full_addr + (load_po_stage) * 8, _phase_page_full);
                    int pg_idx_1 = load_po_stage * 16;
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384, (&K), 0, w0, 0, kv_head, smem_pages[pg_idx_1 + chunk_page0], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 2048, (&K), 0, w1, 0, kv_head, smem_pages[pg_idx_1 + chunk_page1], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 4096, (&K), 0, w2, 0, kv_head, smem_pages[pg_idx_1 + chunk_page2], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 6144, (&K), 0, w3, 0, kv_head, smem_pages[pg_idx_1 + chunk_page3], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 8192, (&K), 0, w4, 0, kv_head, smem_pages[pg_idx_1 + chunk_page4], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 10240, (&K), 0, w5, 0, kv_head, smem_pages[pg_idx_1 + chunk_page5], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 12288, (&K), 0, w6, 0, kv_head, smem_pages[pg_idx_1 + chunk_page6], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 14336, (&K), 0, w7, 0, kv_head, smem_pages[pg_idx_1 + chunk_page7], kv_full_addr + (load_kv_stage) * 8);
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384, (&V), 0, w0, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page0], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 2048, (&V), 0, w1, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page1], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 4096, (&V), 0, w2, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page2], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 6144, (&V), 0, w3, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page3], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 8192, (&V), 0, w4, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page4], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 10240, (&V), 0, w5, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page5], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 12288, (&V), 0, w6, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page6], kv_full_addr + (load_kv_stage) * 8);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 16384 + 14336, (&V), 0, w7, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_page7], kv_full_addr + (load_kv_stage) * 8);
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    if (elect_sync()) {
                        mbarrier_arrive(page_empty_addr + (load_po_stage) * 8);
                    }
                    load_po_stage += 1;
                    if (load_po_stage == 6) { load_po_stage = 0; _phase_page_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
