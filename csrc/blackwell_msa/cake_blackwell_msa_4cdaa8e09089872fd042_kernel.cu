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
#define TMEM_SCORES0_OFFSET 0
#define TMEM_SCORES1_OFFSET 128
#define TMEM_OUTPUT0_OFFSET 256
#define TMEM_OUTPUT1_OFFSET 384
#define NUM_DECODE_KV_STAGES 4
#define NUM_P_STORE_ORDER_STAGES 1
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_KV_OFF 17408
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 17408
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_NVFP4_DATA_OFF 82944
#define SMEM_SMEM_NVFP4_DATA_STAGE_BYTES 8192
#define SMEM_SMEM_NVFP4_DATA_STRIDE 8192
#define SMEM_SMEM_NVFP4_SCALE_OFF 115712
#define SMEM_SMEM_NVFP4_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_NVFP4_SCALE_STRIDE 1024
#define SMEM_SMEM_PAGE_INDICES_OFF 148480
#define SMEM_SMEM_PAGE_INDICES_STAGE_BYTES 2048
#define SMEM_SMEM_PAGE_INDICES_STRIDE 2048
#define SMEM_SMEM_ACC_SCALE_OFF 152576
#define SMEM_SMEM_ACC_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_ACC_SCALE_STRIDE 1024
#define SMEM_SMEM_ROW_SUM_OFF 153600
#define SMEM_SMEM_ROW_SUM_STAGE_BYTES 1024
#define SMEM_SMEM_ROW_SUM_STRIDE 1024
#define SMEM_SMEM_ROW_MAX_OFF 154624
#define SMEM_SMEM_ROW_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_ROW_MAX_STRIDE 1024
#define SMEM_TOTAL 156672
#define THREADS 512

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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.

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




__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_blackwell_msa_4cdaa8e09089872fd042(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages)
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
    #define kv_raw_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define corr_sig_addr (mbar_base + 144)
    #define p_store_turn0_addr (mbar_base + 160)
    #define p_store_turn1_addr (mbar_base + 168)
    #define o_full_addr (mbar_base + 176)
    #define decode_done_addr (mbar_base + 184)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_kv_addr = smem + 17408;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v_addr = smem + 17408;
    uint8_t* smem_nvfp4_data = reinterpret_cast<uint8_t*>(smem_raw + 82944);
    const int smem_nvfp4_data_addr = smem + 82944;
    uint8_t* smem_nvfp4_scale = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int smem_nvfp4_scale_addr = smem + 115712;
    int* smem_page_indices = reinterpret_cast<int*>(smem_raw + 148480);
    const int smem_page_indices_addr = smem + 148480;
    float* smem_acc_scale = reinterpret_cast<float*>(smem_raw + 152576);
    const int smem_acc_scale_addr = smem + 152576;
    float* smem_row_sum = reinterpret_cast<float*>(smem_raw + 153600);
    const int smem_row_sum_addr = smem + 153600;
    float* smem_row_max = reinterpret_cast<float*>(smem_raw + 154624);
    const int smem_row_max_addr = smem + 154624;

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'decode_kv' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_raw_full: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // p_full: 2 barriers, init_count=64
            mbarrier_init(smem + 128, 64);
            mbarrier_init(smem + 136, 64);
            // corr_sig: 2 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            // p_store_turn0: 1 barriers, init_count=32
            mbarrier_init(smem + 160, 32);
            // p_store_turn1: 1 barriers, init_count=32
            mbarrier_init(smem + 168, 32);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // decode_done: 1 barriers, init_count=32
            mbarrier_init(smem + 184, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores0 = taddr;
    const int tmem_scores1 = taddr + 128;
    const int tmem_output0 = taddr + 256;
    const int tmem_output1 = taddr + 384;

    // ---- Role: softmax ----
    if (warp == 0 || warp == 4) {
        { // softmax_main
            unsigned int total_work_items_s = total_q * num_kv_heads;
            const int stage = warp / 4;
            int p_store_phase = ((stage == 0) ? 1 : 0);
            int p_store_order_stage = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_s_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_s = blockIdx.x; work_idx_s < total_work_items_s; work_idx_s += gridDim.x) {
                int my_row = lane;
                int state_idx = stage * 128 + my_row;
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (int pair = 0; pair < 8; pair++) {
                    if (stage == 0) {
                        mbarrier_wait(s_full_addr, _phase_s_full_0);
                        _phase_s_full_0 ^= 1;
                    } else {
                        mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                        _phase_s_full_1 ^= 1;
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int valid_cols = smem_page_indices[pair * 2 + stage];
                    int s_base = taddr + (unsigned int)(stage * 128);
                    int _min_0 = ((valid_cols) < (64) ? (valid_cols) : (64));
                    int valid_lo = _min_0;
                    int _max_0 = ((valid_cols - 64) > (0) ? (valid_cols - 64) : (0));
                    int valid_hi = _max_0;
                    float _tmem_load_0[64];
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
                    if (valid_lo < 64) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = valid_lo;
                            if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                            }
                        }
                        if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) _tmem_load_0[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) _tmem_load_0[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) _tmem_load_0[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) _tmem_load_0[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) _tmem_load_0[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) _tmem_load_0[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) _tmem_load_0[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) _tmem_load_0[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) _tmem_load_0[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) _tmem_load_0[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) _tmem_load_0[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) _tmem_load_0[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) _tmem_load_0[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) _tmem_load_0[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) _tmem_load_0[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) _tmem_load_0[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = valid_lo - 32;
                            if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                            }
                        }
                        if (!(_slice_lo_mask_1 & (1u << 0))) _tmem_load_0[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) _tmem_load_0[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) _tmem_load_0[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) _tmem_load_0[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) _tmem_load_0[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) _tmem_load_0[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) _tmem_load_0[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) _tmem_load_0[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) _tmem_load_0[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) _tmem_load_0[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) _tmem_load_0[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) _tmem_load_0[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) _tmem_load_0[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) _tmem_load_0[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) _tmem_load_0[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) _tmem_load_0[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) _tmem_load_0[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) _tmem_load_0[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) _tmem_load_0[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) _tmem_load_0[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) _tmem_load_0[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) _tmem_load_0[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) _tmem_load_0[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) _tmem_load_0[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) _tmem_load_0[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) _tmem_load_0[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) _tmem_load_0[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) _tmem_load_0[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) _tmem_load_0[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) _tmem_load_0[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) _tmem_load_0[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) _tmem_load_0[63] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_2);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_2);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_2);
                    float tile_max = _tmem_load_0_max;
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(s_base + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(s_base + 64 + 32));
                    if (valid_hi < 64) {
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_3 = valid_hi;
                            if (_lim_3 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_3 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_3));
                            }
                        }
                        if (!(_slice_lo_mask_2 & (1u << 0))) _tmem_load_1[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 1))) _tmem_load_1[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 2))) _tmem_load_1[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 3))) _tmem_load_1[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 4))) _tmem_load_1[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 5))) _tmem_load_1[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 6))) _tmem_load_1[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 7))) _tmem_load_1[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 8))) _tmem_load_1[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 9))) _tmem_load_1[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 10))) _tmem_load_1[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 11))) _tmem_load_1[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 12))) _tmem_load_1[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 13))) _tmem_load_1[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 14))) _tmem_load_1[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 15))) _tmem_load_1[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 16))) _tmem_load_1[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 17))) _tmem_load_1[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 18))) _tmem_load_1[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 19))) _tmem_load_1[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 20))) _tmem_load_1[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 21))) _tmem_load_1[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 22))) _tmem_load_1[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 23))) _tmem_load_1[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 24))) _tmem_load_1[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 25))) _tmem_load_1[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 26))) _tmem_load_1[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 27))) _tmem_load_1[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 28))) _tmem_load_1[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 29))) _tmem_load_1[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 30))) _tmem_load_1[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 31))) _tmem_load_1[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_4 = valid_hi - 32;
                            if (_lim_4 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_4 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_4));
                            }
                        }
                        if (!(_slice_lo_mask_3 & (1u << 0))) _tmem_load_1[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 1))) _tmem_load_1[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 2))) _tmem_load_1[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 3))) _tmem_load_1[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 4))) _tmem_load_1[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 5))) _tmem_load_1[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 6))) _tmem_load_1[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 7))) _tmem_load_1[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 8))) _tmem_load_1[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 9))) _tmem_load_1[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 10))) _tmem_load_1[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 11))) _tmem_load_1[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 12))) _tmem_load_1[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 13))) _tmem_load_1[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 14))) _tmem_load_1[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 15))) _tmem_load_1[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 16))) _tmem_load_1[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 17))) _tmem_load_1[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 18))) _tmem_load_1[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 19))) _tmem_load_1[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 20))) _tmem_load_1[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 21))) _tmem_load_1[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 22))) _tmem_load_1[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 23))) _tmem_load_1[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 24))) _tmem_load_1[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 25))) _tmem_load_1[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 26))) _tmem_load_1[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 27))) _tmem_load_1[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 28))) _tmem_load_1[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 29))) _tmem_load_1[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 30))) _tmem_load_1[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 31))) _tmem_load_1[63] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_5 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_5);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_5);
                    float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_5);
                    float _max_1 = max_noftz(tile_max, _tmem_load_1_max);
                    tile_max = _max_1;
                    float _max_2 = max_noftz(row_max, tile_max);
                    float new_max = _max_2;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -max_scaled);
                    float delta = _fma_0;
                    float _exp2_0 = approx_exp2(delta);
                    float acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    row_max = new_max;
                    smem_acc_scale[state_idx] = acc_scale;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (stage == 0) {
                        mbarrier_arrive(corr_sig_addr);
                    } else {
                        mbarrier_arrive(corr_sig_addr + 8);
                    }
                    int p_base = taddr + (unsigned int)(stage * 128) + 64;
                    float _tmem_load_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(s_base + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                        : "r"(s_base + 64 + 32));
                    if (valid_hi < 64) {
                        uint32_t _slice_lo_mask_4;
                        {
                            int _lim_6 = valid_hi;
                            if (_lim_6 <= 0) { _slice_lo_mask_4 = 0u; }
                            else if (_lim_6 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_6));
                            }
                        }
                        if (!(_slice_lo_mask_4 & (1u << 0))) _tmem_load_2[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 1))) _tmem_load_2[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 2))) _tmem_load_2[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 3))) _tmem_load_2[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 4))) _tmem_load_2[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 5))) _tmem_load_2[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 6))) _tmem_load_2[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 7))) _tmem_load_2[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 8))) _tmem_load_2[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 9))) _tmem_load_2[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 10))) _tmem_load_2[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 11))) _tmem_load_2[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 12))) _tmem_load_2[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 13))) _tmem_load_2[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 14))) _tmem_load_2[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 15))) _tmem_load_2[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 16))) _tmem_load_2[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 17))) _tmem_load_2[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 18))) _tmem_load_2[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 19))) _tmem_load_2[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 20))) _tmem_load_2[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 21))) _tmem_load_2[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 22))) _tmem_load_2[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 23))) _tmem_load_2[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 24))) _tmem_load_2[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 25))) _tmem_load_2[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 26))) _tmem_load_2[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 27))) _tmem_load_2[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 28))) _tmem_load_2[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 29))) _tmem_load_2[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 30))) _tmem_load_2[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 31))) _tmem_load_2[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_5;
                        {
                            int _lim_7 = valid_hi - 32;
                            if (_lim_7 <= 0) { _slice_lo_mask_5 = 0u; }
                            else if (_lim_7 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_7));
                            }
                        }
                        if (!(_slice_lo_mask_5 & (1u << 0))) _tmem_load_2[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 1))) _tmem_load_2[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 2))) _tmem_load_2[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 3))) _tmem_load_2[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 4))) _tmem_load_2[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 5))) _tmem_load_2[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 6))) _tmem_load_2[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 7))) _tmem_load_2[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 8))) _tmem_load_2[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 9))) _tmem_load_2[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 10))) _tmem_load_2[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 11))) _tmem_load_2[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 12))) _tmem_load_2[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 13))) _tmem_load_2[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 14))) _tmem_load_2[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 15))) _tmem_load_2[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 16))) _tmem_load_2[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 17))) _tmem_load_2[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 18))) _tmem_load_2[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 19))) _tmem_load_2[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 20))) _tmem_load_2[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 21))) _tmem_load_2[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 22))) _tmem_load_2[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 23))) _tmem_load_2[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 24))) _tmem_load_2[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 25))) _tmem_load_2[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 26))) _tmem_load_2[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 27))) _tmem_load_2[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 28))) _tmem_load_2[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 29))) _tmem_load_2[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 30))) _tmem_load_2[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 31))) _tmem_load_2[63] = -CAKE_INF;
                    }
                    const float2 _fma_b2_8 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_9 = {-max_scaled, -max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_lf], _fma_b2_8, _fma_c2_9);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_2[_le] = approx_exp2(_tmem_load_2[_le]);
                    }
                    float2 _reg_reduce_sum2_10 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_2[0], &_reg_reduce_sum2_10);
                    softmax_block_sum(&_tmem_load_2[32], &_reg_reduce_sum2_10);
                    float _tmem_load_2_sum = _reg_reduce_sum2_10.x + _reg_reduce_sum2_10.y;
                    float block_sum = _tmem_load_2_sum;
                    if (pair == 0) {
                        if (stage == 0) {
                            mbarrier_wait(p_store_turn0_addr, p_store_phase);
                        } else {
                            mbarrier_wait(p_store_turn1_addr, p_store_phase);
                        }
                    }
                    uint32_t _fp8_0[16];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_2[0]), "f"(_tmem_load_2[1]),
                                               "f"(_tmem_load_2[2]), "f"(_tmem_load_2[3]));
                        _fp8_0[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[4]), "f"(_tmem_load_2[5]),
                                               "f"(_tmem_load_2[6]), "f"(_tmem_load_2[7]));
                        _fp8_0[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[8]), "f"(_tmem_load_2[9]),
                                               "f"(_tmem_load_2[10]), "f"(_tmem_load_2[11]));
                        _fp8_0[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[12]), "f"(_tmem_load_2[13]),
                                               "f"(_tmem_load_2[14]), "f"(_tmem_load_2[15]));
                        _fp8_0[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[16]), "f"(_tmem_load_2[17]),
                                               "f"(_tmem_load_2[18]), "f"(_tmem_load_2[19]));
                        _fp8_0[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[20]), "f"(_tmem_load_2[21]),
                                               "f"(_tmem_load_2[22]), "f"(_tmem_load_2[23]));
                        _fp8_0[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[24]), "f"(_tmem_load_2[25]),
                                               "f"(_tmem_load_2[26]), "f"(_tmem_load_2[27]));
                        _fp8_0[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[28]), "f"(_tmem_load_2[29]),
                                               "f"(_tmem_load_2[30]), "f"(_tmem_load_2[31]));
                        _fp8_0[7] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[32]), "f"(_tmem_load_2[33]),
                                               "f"(_tmem_load_2[34]), "f"(_tmem_load_2[35]));
                        _fp8_0[8] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[36]), "f"(_tmem_load_2[37]),
                                               "f"(_tmem_load_2[38]), "f"(_tmem_load_2[39]));
                        _fp8_0[9] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[40]), "f"(_tmem_load_2[41]),
                                               "f"(_tmem_load_2[42]), "f"(_tmem_load_2[43]));
                        _fp8_0[10] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[44]), "f"(_tmem_load_2[45]),
                                               "f"(_tmem_load_2[46]), "f"(_tmem_load_2[47]));
                        _fp8_0[11] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[48]), "f"(_tmem_load_2[49]),
                                               "f"(_tmem_load_2[50]), "f"(_tmem_load_2[51]));
                        _fp8_0[12] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[52]), "f"(_tmem_load_2[53]),
                                               "f"(_tmem_load_2[54]), "f"(_tmem_load_2[55]));
                        _fp8_0[13] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[56]), "f"(_tmem_load_2[57]),
                                               "f"(_tmem_load_2[58]), "f"(_tmem_load_2[59]));
                        _fp8_0[14] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_2[60]), "f"(_tmem_load_2[61]),
                                               "f"(_tmem_load_2[62]), "f"(_tmem_load_2[63]));
                        _fp8_0[15] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 16), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[15])));
                    float _tmem_load_3[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(s_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                        : "r"(s_base + 32));
                    if (valid_lo < 64) {
                        uint32_t _slice_lo_mask_6;
                        {
                            int _lim_11 = valid_lo;
                            if (_lim_11 <= 0) { _slice_lo_mask_6 = 0u; }
                            else if (_lim_11 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_11));
                            }
                        }
                        if (!(_slice_lo_mask_6 & (1u << 0))) _tmem_load_3[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 1))) _tmem_load_3[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 2))) _tmem_load_3[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 3))) _tmem_load_3[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 4))) _tmem_load_3[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 5))) _tmem_load_3[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 6))) _tmem_load_3[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 7))) _tmem_load_3[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 8))) _tmem_load_3[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 9))) _tmem_load_3[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 10))) _tmem_load_3[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 11))) _tmem_load_3[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 12))) _tmem_load_3[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 13))) _tmem_load_3[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 14))) _tmem_load_3[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 15))) _tmem_load_3[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 16))) _tmem_load_3[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 17))) _tmem_load_3[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 18))) _tmem_load_3[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 19))) _tmem_load_3[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 20))) _tmem_load_3[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 21))) _tmem_load_3[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 22))) _tmem_load_3[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 23))) _tmem_load_3[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 24))) _tmem_load_3[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 25))) _tmem_load_3[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 26))) _tmem_load_3[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 27))) _tmem_load_3[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 28))) _tmem_load_3[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 29))) _tmem_load_3[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 30))) _tmem_load_3[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 31))) _tmem_load_3[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_7;
                        {
                            int _lim_12 = valid_lo - 32;
                            if (_lim_12 <= 0) { _slice_lo_mask_7 = 0u; }
                            else if (_lim_12 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_12));
                            }
                        }
                        if (!(_slice_lo_mask_7 & (1u << 0))) _tmem_load_3[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 1))) _tmem_load_3[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 2))) _tmem_load_3[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 3))) _tmem_load_3[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 4))) _tmem_load_3[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 5))) _tmem_load_3[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 6))) _tmem_load_3[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 7))) _tmem_load_3[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 8))) _tmem_load_3[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 9))) _tmem_load_3[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 10))) _tmem_load_3[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 11))) _tmem_load_3[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 12))) _tmem_load_3[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 13))) _tmem_load_3[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 14))) _tmem_load_3[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 15))) _tmem_load_3[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 16))) _tmem_load_3[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 17))) _tmem_load_3[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 18))) _tmem_load_3[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 19))) _tmem_load_3[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 20))) _tmem_load_3[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 21))) _tmem_load_3[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 22))) _tmem_load_3[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 23))) _tmem_load_3[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 24))) _tmem_load_3[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 25))) _tmem_load_3[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 26))) _tmem_load_3[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 27))) _tmem_load_3[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 28))) _tmem_load_3[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 29))) _tmem_load_3[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 30))) _tmem_load_3[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 31))) _tmem_load_3[63] = -CAKE_INF;
                    }
                    const float2 _fma_b2_13 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_14 = {-max_scaled, -max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_lf], _fma_b2_13, _fma_c2_14);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_3[_le] = approx_exp2(_tmem_load_3[_le]);
                    }
                    float2 _reg_reduce_sum2_15 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_3[0], &_reg_reduce_sum2_15);
                    softmax_block_sum(&_tmem_load_3[32], &_reg_reduce_sum2_15);
                    float _tmem_load_3_sum = _reg_reduce_sum2_15.x + _reg_reduce_sum2_15.y;
                    block_sum = block_sum + _tmem_load_3_sum;
                    uint32_t _fp8_1[16];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(_tmem_load_3[0]), "f"(_tmem_load_3[1]),
                                               "f"(_tmem_load_3[2]), "f"(_tmem_load_3[3]));
                        _fp8_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[4]), "f"(_tmem_load_3[5]),
                                               "f"(_tmem_load_3[6]), "f"(_tmem_load_3[7]));
                        _fp8_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[8]), "f"(_tmem_load_3[9]),
                                               "f"(_tmem_load_3[10]), "f"(_tmem_load_3[11]));
                        _fp8_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[12]), "f"(_tmem_load_3[13]),
                                               "f"(_tmem_load_3[14]), "f"(_tmem_load_3[15]));
                        _fp8_1[3] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[16]), "f"(_tmem_load_3[17]),
                                               "f"(_tmem_load_3[18]), "f"(_tmem_load_3[19]));
                        _fp8_1[4] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[20]), "f"(_tmem_load_3[21]),
                                               "f"(_tmem_load_3[22]), "f"(_tmem_load_3[23]));
                        _fp8_1[5] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[24]), "f"(_tmem_load_3[25]),
                                               "f"(_tmem_load_3[26]), "f"(_tmem_load_3[27]));
                        _fp8_1[6] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[28]), "f"(_tmem_load_3[29]),
                                               "f"(_tmem_load_3[30]), "f"(_tmem_load_3[31]));
                        _fp8_1[7] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[32]), "f"(_tmem_load_3[33]),
                                               "f"(_tmem_load_3[34]), "f"(_tmem_load_3[35]));
                        _fp8_1[8] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[36]), "f"(_tmem_load_3[37]),
                                               "f"(_tmem_load_3[38]), "f"(_tmem_load_3[39]));
                        _fp8_1[9] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[40]), "f"(_tmem_load_3[41]),
                                               "f"(_tmem_load_3[42]), "f"(_tmem_load_3[43]));
                        _fp8_1[10] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[44]), "f"(_tmem_load_3[45]),
                                               "f"(_tmem_load_3[46]), "f"(_tmem_load_3[47]));
                        _fp8_1[11] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[48]), "f"(_tmem_load_3[49]),
                                               "f"(_tmem_load_3[50]), "f"(_tmem_load_3[51]));
                        _fp8_1[12] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[52]), "f"(_tmem_load_3[53]),
                                               "f"(_tmem_load_3[54]), "f"(_tmem_load_3[55]));
                        _fp8_1[13] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[56]), "f"(_tmem_load_3[57]),
                                               "f"(_tmem_load_3[58]), "f"(_tmem_load_3[59]));
                        _fp8_1[14] = _packed;
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
                            : "=r"(_packed) : "f"(_tmem_load_3[60]), "f"(_tmem_load_3[61]),
                                               "f"(_tmem_load_3[62]), "f"(_tmem_load_3[63]));
                        _fp8_1[15] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[15])));
                    float _fma_1 = __fmaf_rn(row_sum, acc_scale, block_sum);
                    row_sum = _fma_1;
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (pair == 0) {
                        if (stage == 0) {
                            mbarrier_arrive(p_store_turn1_addr);
                        } else {
                            mbarrier_arrive(p_store_turn0_addr);
                        }
                        p_store_phase ^= 1;
                    }
                    if (stage == 0) {
                        mbarrier_arrive(p_full_addr);
                    } else {
                        mbarrier_arrive(p_full_addr + 8);
                    }
                }
                smem_row_sum[state_idx] = row_sum;
                smem_row_max[state_idx] = row_max;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (stage == 0) {
                    mbarrier_arrive(corr_sig_addr);
                } else {
                    mbarrier_arrive(corr_sig_addr + 8);
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 5 && warp <= 11) {
        { // transform_main

        }
    }
    // ---- Role: front_idle ----
    if (warp == 1 || warp == 13 || warp == 14 || warp == 15) {
        // idle — no tasks assigned
    }
    // ---- Role: producer ----
    if (warp == 2) {
        { // producer_main
            unsigned int total_work_items_l = total_q * num_kv_heads;
            int load_stage = 0;
            int load_phase = 1;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int work_idx_l = blockIdx.x; work_idx_l < total_work_items_l; work_idx_l += gridDim.x) {
                int logical_work_l = work_idx_l;
                int split_l = 0;
                int query = logical_work_l / num_kv_heads;
                int kv_head = logical_work_l % num_kv_heads;
                int group_size = num_q_heads / num_kv_heads;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    int q_row = query * num_q_heads + kv_head * group_size;
                    mbarrier_arrive_expect_tx(q_full_addr, 2048);
                    tma_3d_gmem2smem(smem_q_addr, (&Q), 0, q_row, 0, q_full_addr);
                }
                if (elect_sync()) {
                    #pragma unroll
                    for (int k_tile = 0; k_tile < 2; k_tile++) {
                        int selected_position = 15 - split_l * 16 - k_tile;
                        int token_base = 0;
                        int page_head = 0;
                        int valid_cols_1 = 128;
                        int batch = query / seqlen_q;
                        int query_in_batch = query - batch * seqlen_q;
                        int selected_block = task_kind[(kv_head * total_q + query) * 16 + selected_position];
                        int kv_len = task_kv_head[batch];
                        int valid_cols_0 = 0;
                        if (selected_block >= 0) {
                            int block_start = selected_block * 128;
                            valid_cols_0 = kv_len - block_start;
                            if (valid_cols_0 > 128) {
                                valid_cols_0 = 128;
                            }
                            if (valid_cols_0 < 0) {
                                valid_cols_0 = 0;
                            }
                            {
                                int query_position = kv_len - seqlen_q + query_in_batch;
                                int causal_cols = query_position - block_start + 1;
                                if (valid_cols_0 > causal_cols) {
                                    valid_cols_0 = causal_cols;
                                }
                                if (valid_cols_0 < 0) {
                                    valid_cols_0 = 0;
                                }
                            }
                        }
                        int token_base_1 = 0;
                        int page_head_2 = 0;
                        {
                            int physical_page = 0;
                            if (selected_block >= 0) {
                                physical_page = kv_indices[batch * msa_max_pages + selected_block];
                                if (physical_page < 0) {
                                    valid_cols_0 = 0;
                                    physical_page = 0;
                                }
                            }
                            page_head_2 = physical_page * num_kv_heads + kv_head;
                        }
                        token_base = token_base_1;
                        page_head = page_head_2;
                        valid_cols_1 = valid_cols_0;
                        smem_page_indices[k_tile] = valid_cols_1;
                        smem_page_indices[16 + k_tile] = page_head;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        {
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, page_head, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; load_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int next_k_tile = 2; next_k_tile < 16; next_k_tile++) {
                        int v_tile = next_k_tile - 2;
                        int v_page_head = smem_page_indices[16 + v_tile];
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        {
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&V), 0, 0, v_page_head, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; load_phase ^= 1; }
                        int selected_position_1 = 15 - split_l * 16 - next_k_tile;
                        int next_token_base = 0;
                        int next_page_head = 0;
                        int next_valid_cols = 128;
                        int batch_1 = query / seqlen_q;
                        int query_in_batch_1 = query - batch_1 * seqlen_q;
                        int selected_block_1 = task_kind[(kv_head * total_q + query) * 16 + selected_position_1];
                        int kv_len_1 = task_kv_head[batch_1];
                        int valid_cols_2 = 0;
                        if (selected_block_1 >= 0) {
                            int block_start_1 = selected_block_1 * 128;
                            valid_cols_2 = kv_len_1 - block_start_1;
                            if (valid_cols_2 > 128) {
                                valid_cols_2 = 128;
                            }
                            if (valid_cols_2 < 0) {
                                valid_cols_2 = 0;
                            }
                            {
                                int query_position_1 = kv_len_1 - seqlen_q + query_in_batch_1;
                                int causal_cols_1 = query_position_1 - block_start_1 + 1;
                                if (valid_cols_2 > causal_cols_1) {
                                    valid_cols_2 = causal_cols_1;
                                }
                                if (valid_cols_2 < 0) {
                                    valid_cols_2 = 0;
                                }
                            }
                        }
                        int token_base_2 = 0;
                        int page_head_1 = 0;
                        {
                            int physical_page_1 = 0;
                            if (selected_block_1 >= 0) {
                                physical_page_1 = kv_indices[batch_1 * msa_max_pages + selected_block_1];
                                if (physical_page_1 < 0) {
                                    valid_cols_2 = 0;
                                    physical_page_1 = 0;
                                }
                            }
                            page_head_1 = physical_page_1 * num_kv_heads + kv_head;
                        }
                        next_token_base = token_base_2;
                        next_page_head = page_head_1;
                        next_valid_cols = valid_cols_2;
                        smem_page_indices[next_k_tile] = next_valid_cols;
                        smem_page_indices[16 + next_k_tile] = next_page_head;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        {
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, next_page_head, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; load_phase ^= 1; }
                    }
                    #pragma unroll
                    for (int v_tile_1 = 14; v_tile_1 < 16; v_tile_1++) {
                        int v_page_head_1 = smem_page_indices[16 + v_tile_1];
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                        {
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&V), 0, 0, v_page_head_1, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; load_phase ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 3) {
        { // mma_main
            unsigned int total_work_items_m = total_q * num_kv_heads;
            int kv_stage_m = 0;
            int kv_phase_m = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_decode_done_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_m = blockIdx.x; work_idx_m < total_work_items_m; work_idx_m += gridDim.x) {
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                int _mma_a_lo_0 = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 1);
                    }
                }
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 7; pair_1++) {
                    mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
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
                    :: "r"(tmem_output0), "r"(_mma_b_lo_2), "r"(tmem_scores0 + 64), "r"(((first_pv0) ? 0 : 1)));
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
                    mma_ts_step(tmem_output0, tmem_scores0 + 64 + 24, _mma_b_lo_3 + 768, 0x40004040, 136380432, 1);
                    elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                    int _mma_a_lo_4 = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136314896, 1);
                        }
                    }
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
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
                    :: "r"(tmem_output1), "r"(_mma_b_lo_5), "r"(tmem_scores1 + 64), "r"(((first_pv1) ? 0 : 1)));
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
                    mma_ts_step(tmem_output1, tmem_scores1 + 64 + 24, _mma_b_lo_6 + 768, 0x40004040, 136380432, 1);
                    elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_scores1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136314896, 1);
                        }
                    }
                    if (pair_1 == 6) {
                        elect_commit2(s_full_addr + 8, q_empty_addr);
                    } else {
                        elect_commit(s_full_addr + 8);
                    }
                    elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                    kv_stage_m += 1;
                    if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
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
                    :: "r"(tmem_output0), "r"(_mma_b_lo_8), "r"(tmem_scores0 + 64), "r"(((first_pv0) ? 0 : 1)));
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
                mma_ts_step(tmem_output0, tmem_scores0 + 64 + 24, _mma_b_lo_9 + 768, 0x40004040, 136380432, 1);
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, kv_phase_m);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
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
                    :: "r"(tmem_output1), "r"(_mma_b_lo_10), "r"(tmem_scores1 + 64), "r"(((first_pv1) ? 0 : 1)));
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
                mma_ts_step(tmem_output1, tmem_scores1 + 64 + 24, _mma_b_lo_11 + 768, 0x40004040, 136380432, 1);
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 4) { kv_stage_m = 0; kv_phase_m ^= 1; }
                elect_commit(o_full_addr);
                mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                _phase_decode_done_0 ^= 1;
            }
        }
    }
    // ---- Role: correction ----
    if (warp == 12) {
        { // correction_main
            unsigned int total_work_items_c = total_q * num_kv_heads;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_c = blockIdx.x; work_idx_c < total_work_items_c; work_idx_c += gridDim.x) {
                int logical_work_c = work_idx_c;
                int split_c = 0;
                int query_1 = logical_work_c / num_kv_heads;
                int kv_head_1 = logical_work_c % num_kv_heads;
                int group_size_1 = num_q_heads / num_kv_heads;
                const int warp_in_role = warp - 12;
                const int tmem_row_base = warp_in_role * 32;
                int my_row_1 = tmem_row_base + lane;
                const int row_addr = tmem_row_base << 16;
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale0 = smem_acc_scale[my_row_1];
                    if (pair_2 > 0 && warp == 12) {
                        #pragma unroll
                        for (int col = 0; col < 128; col += 16) {
                            float _tmem_load_4[16];
                            tmem_ld_x16(&_tmem_load_4[0], taddr + 256 + (unsigned int)row_addr + (unsigned int)col);
                            const float2 _scale2_0 = {scale0, scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_0);
                            tmem_st_x16_f32(taddr + 256 + (unsigned int)row_addr + (unsigned int)col, _tmem_load_4);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale1 = smem_acc_scale[128 + my_row_1];
                    if (pair_2 > 0 && warp == 12) {
                        #pragma unroll
                        for (int col_1 = 0; col_1 < 128; col_1 += 16) {
                            float _tmem_load_5[16];
                            tmem_ld_x16(&_tmem_load_5[0], taddr + 384 + (unsigned int)row_addr + (unsigned int)col_1);
                            const float2 _scale2_1 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_1);
                            tmem_st_x16_f32(taddr + 384 + (unsigned int)row_addr + (unsigned int)col_1, _tmem_load_5);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp == 12) {
                    float sum0 = smem_row_sum[my_row_1];
                    float sum1 = smem_row_sum[128 + my_row_1];
                    float max0 = smem_row_max[my_row_1];
                    float max1 = smem_row_max[128 + my_row_1];
                    float _max_3 = max_noftz(max0, max1);
                    float final_max = _max_3;
                    float d0 = ((max0 == -CAKE_INF) ? 0.0f : softmax_scale_log2 * (max0 - final_max));
                    float d1 = ((max1 == -CAKE_INF) ? 0.0f : softmax_scale_log2 * (max1 - final_max));
                    float _exp2_1 = approx_exp2(d0);
                    float merge_scale0 = _exp2_1;
                    float _exp2_2 = approx_exp2(d1);
                    float merge_scale1 = _exp2_2;
                    float final_sum = sum0 * merge_scale0 + sum1 * merge_scale1;
                    float _rcp_0 = approx_rcp(final_sum);
                    float inv_sum = ((final_sum > 0.0f) ? _rcp_0 : 0.0f);
                    int q_head_row = query_1 * num_q_heads + kv_head_1 * group_size_1 + my_row_1;
                    int output_row = q_head_row * 128;
                    int logical_output = query_1 * num_kv_heads + kv_head_1;
                    int partial_slot = logical_output + split_c;
                    int partial_stat_idx = partial_slot * 128 + my_row_1;
                    #pragma unroll
                    for (int col_2 = 0; col_2 < 128; col_2 += 16) {
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], taddr + 256 + (unsigned int)row_addr + (unsigned int)col_2);
                        float _tmem_load_7[16];
                        tmem_ld_x16(&_tmem_load_7[0], taddr + 384 + (unsigned int)row_addr + (unsigned int)col_2);
                        #pragma unroll
                        for (int elem = 0; elem < 16; elem++) {
                            _tmem_load_6[elem] = _tmem_load_6[elem] * merge_scale0 + _tmem_load_7[elem] * merge_scale1;
                        }
                        if (my_row_1 < group_size_1) {
                            {
                                {
                                    const float2 _prescale2_2 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_6[0])[_ps], _prescale2_2);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_6[0 + _ps] *= inv_sum * output_scale;
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
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_row + col_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_row + col_2)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                }
                            }
                        }
                    }
                    if (my_row_1 < group_size_1) {
                        {
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum));
                            float lse_value = ((final_sum > 0.0f) ? final_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_INF);
                            msa_lse[q_head_row] = lse_value;
                        }
                    }
                }
                mbarrier_arrive(decode_done_addr);
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
