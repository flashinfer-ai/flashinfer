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
#define TMEM_SCORES_0_OFFSET 0
#define TMEM_SOFTMAX_0_OFFSET 64
#define TMEM_SCORES_1_OFFSET 128
#define TMEM_SOFTMAX_1_OFFSET 192
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_Q_STAGES 2
#define NUM_KV_STAGES 4
#define NUM_PO_STAGES 6
#define NUM_ACC_STAGES 1
#define NUM_ORDER_P01_STAGES 1
#define SMEM_SSCALE_OFF 1024
#define SMEM_SSCALE_STAGE_BYTES 3072
#define SMEM_SSCALE_STRIDE 3072
#define SMEM_SMEM_Q0_OFF 4096
#define SMEM_SMEM_Q0_STAGE_BYTES 32768
#define SMEM_SMEM_Q0_STRIDE 32768
#define SMEM_SMEM_Q1_OFF 36864
#define SMEM_SMEM_Q1_STAGE_BYTES 32768
#define SMEM_SMEM_Q1_STRIDE 32768
#define SMEM_SMEM_KV_OFF 69632
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 69632
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_PAGES_OFF 200704
#define SMEM_SMEM_PAGES_STAGE_BYTES 384
#define SMEM_SMEM_PAGES_STRIDE 384
#define SMEM_WORK_ID_SLOT_OFF 201088
#define SMEM_WORK_ID_SLOT_STAGE_BYTES 4
#define SMEM_WORK_ID_SLOT_STRIDE 4
#define SMEM_TOTAL 201216
#define IS_CAUSAL 1
#define NUM_M_BLOCKS 2
#define NUM_Q_HEADS 32
#define HEADS_PER_GROUP 8
#define PACK_G 8
#define TOK_PER_STAGE 16
#define L2_SWIZZLE 8
#define STATIC_ONE_TILE 0
#define RETURN_LSE 0
#define ENABLE_SINK 0
#define UNIFORM_Q_LEN -1
#define UNIFORM_KV_LEN -1
#define SINGLE_MASK_LOOP 0
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define MMA_K 16
#define PAGE_SIZE 16
#define KV_TOK_CHUNK 16
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

// Source-faithful plain CTA wait: no .sem/.scope qualifier, which PTX reads as
// .acquire.cta, for a typed protocol whose source spells none (FA4's interior
// P-ready edge).
__device__ __forceinline__ void mbarrier_wait_plain(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_PLAIN:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_PLAIN;\n\t"
        "bra.uni LAB_WAIT_PLAIN;\n\t"
        "DONE_PLAIN:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
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

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
#if __CUDA_ARCH__ == 1030
#else
    return r;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
#endif
    return r;
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




__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_context_bf16_31373706c9fcb1a2b1db(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, float* __restrict__ sinks, int* __restrict__ page_table_k, int* __restrict__ page_table_v, int* __restrict__ seq_lens_q, int* __restrict__ seq_lens_kv, int* __restrict__ cu_seq_lens_q, float softmax_scale_log2, int total_bh, int page_row_stride, int num_ctas, unsigned int* __restrict__ dynamic_counter)
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
    #define kv_empty_addr (mbar_base + 48)
    #define s_full_addr (mbar_base + 80)
    #define p_full_addr (mbar_base + 96)
    #define p_full_2_addr (mbar_base + 112)
    #define corr_sig_addr (mbar_base + 128)
    #define corr_done_addr (mbar_base + 144)
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
    __nv_bfloat16* smem_q0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int smem_q0_addr = smem + 4096;
    __nv_bfloat16* smem_q1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 36864);
    const int smem_q1_addr = smem + 36864;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int smem_kv_addr = smem + 69632;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int smem_v_addr = smem + 69632;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 200704);
    const int smem_pages_addr = smem + 200704;
    unsigned int* work_id_slot = reinterpret_cast<unsigned int*>(smem_raw + 201088);
    const int work_id_slot_addr = smem + 201088;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            // p_full_2: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // corr_done: 2 barriers, init_count=128
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
            unsigned int stage = make_warp_uniform(warp / 4);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(stage * 128 + 64);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int total_tiles = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds = total_tiles + 1;
            int order_p01_start = ((stage == 0) ? -1 : 0);
            int order_p01_seen = order_p01_start;
            unsigned int _phase_work_id_full_0 = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (unsigned int tile_iter = 0; tile_iter < max_rounds; tile_iter++) {
                unsigned int tile_idx;
                {
                    mbarrier_wait(work_id_full_addr, _phase_work_id_full_0);
                    _phase_work_id_full_0 ^= 1;
                    tile_idx = work_id_slot[0];
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx >= total_tiles) {
                    break;
                }
                unsigned int m_block;
                unsigned int bh;
                {
                    unsigned int l2section = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section = tile_idx / l2section;
                    unsigned int l2_mod = tile_idx % l2section;
                    {
                        bool tail = section >= (unsigned int)total_bh / (unsigned int)L2_SWIZZLE;
                        m_block = ((tail) ? l2_mod % (unsigned int)NUM_M_BLOCKS : l2_mod / (unsigned int)L2_SWIZZLE);
                        bh = section * (unsigned int)L2_SWIZZLE + ((tail) ? l2_mod / (unsigned int)NUM_M_BLOCKS : l2_mod % (unsigned int)L2_SWIZZLE);
                    }
                    m_block = (unsigned int)(NUM_M_BLOCKS - 1) - m_block;
                }
                int seqlen_kv_bh = UNIFORM_KV_LEN;
                int seqlen_q_bh = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh = seq_lens_kv[bh / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                {
                    seqlen_q_bh = cu_seq_lens_q[bh / (unsigned int)(NUM_Q_HEADS / PACK_G) + 1] - cu_seq_lens_q[bh / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                unsigned int num_n_blocks = (seqlen_kv_bh + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift = seqlen_kv_bh - seqlen_q_bh;
                    unsigned int max_n = (m_block * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift;
                    if (max_n < (unsigned int)seqlen_kv_bh) {
                        num_n_blocks = (max_n + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                unsigned int _max_0 = ((num_n_blocks) > (1) ? (num_n_blocks) : (1));
                num_n_blocks = _max_0;
                int causal_row;
                unsigned int num_masked_iters;
                {
                    int kv_shift_1 = seqlen_kv_bh - seqlen_q_bh;
                    causal_row = (m_block * 2 + stage) * (unsigned int)TOK_PER_STAGE + (unsigned int)((warp % 4 * 32 + lane) / PACK_G) + (unsigned int)kv_shift_1;
                    int causal_row_min = (m_block * 2 + stage) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_1;
                    unsigned int n_block_no_mask_limit = (causal_row_min + 1) / BLOCK_N;
                    num_masked_iters = num_n_blocks - n_block_no_mask_limit;
                    if (num_masked_iters > num_n_blocks) {
                        num_masked_iters = num_n_blocks;
                    }
                }
                float row_max_val = -CUDART_INF_F;
                float row_sum_val = 0.0f;
                unsigned int masked_loop_end = num_masked_iters;
                unsigned int unmasked_loop_start = num_masked_iters;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < masked_loop_end; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
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
                        uint32_t _thresh_mask_1;
                        {
                            int _lim_2 = tail_valid;
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
                        if (!(_thresh_mask_1 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_1 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_3;
                        {
                            int _lim_4 = tail_valid - 32;
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
                        if (!(_thresh_mask_3 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_5;
                        {
                            int _lim_6 = tail_valid - 64;
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
                        if (!(_thresh_mask_5 & (1u << 0))) _tmem_load_0[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 1))) _tmem_load_0[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 2))) _tmem_load_0[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 3))) _tmem_load_0[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 4))) _tmem_load_0[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 5))) _tmem_load_0[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 6))) _tmem_load_0[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 7))) _tmem_load_0[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 8))) _tmem_load_0[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 9))) _tmem_load_0[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 10))) _tmem_load_0[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 11))) _tmem_load_0[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 12))) _tmem_load_0[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 13))) _tmem_load_0[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 14))) _tmem_load_0[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 15))) _tmem_load_0[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 16))) _tmem_load_0[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 17))) _tmem_load_0[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 18))) _tmem_load_0[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 19))) _tmem_load_0[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 20))) _tmem_load_0[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 21))) _tmem_load_0[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 22))) _tmem_load_0[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 23))) _tmem_load_0[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 24))) _tmem_load_0[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 25))) _tmem_load_0[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 26))) _tmem_load_0[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 27))) _tmem_load_0[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 28))) _tmem_load_0[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 29))) _tmem_load_0[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 30))) _tmem_load_0[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 31))) _tmem_load_0[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_7;
                        {
                            int _lim_8 = tail_valid - 96;
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
                        if (!(_thresh_mask_7 & (1u << 0))) _tmem_load_0[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 1))) _tmem_load_0[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 2))) _tmem_load_0[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 3))) _tmem_load_0[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 4))) _tmem_load_0[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 5))) _tmem_load_0[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 6))) _tmem_load_0[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 7))) _tmem_load_0[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 8))) _tmem_load_0[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 9))) _tmem_load_0[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 10))) _tmem_load_0[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 11))) _tmem_load_0[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 12))) _tmem_load_0[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 13))) _tmem_load_0[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 14))) _tmem_load_0[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 15))) _tmem_load_0[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 16))) _tmem_load_0[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 17))) _tmem_load_0[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 18))) _tmem_load_0[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 19))) _tmem_load_0[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 20))) _tmem_load_0[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 21))) _tmem_load_0[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 22))) _tmem_load_0[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 23))) _tmem_load_0[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 24))) _tmem_load_0[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 25))) _tmem_load_0[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 26))) _tmem_load_0[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 27))) _tmem_load_0[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 28))) _tmem_load_0[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 29))) _tmem_load_0[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 30))) _tmem_load_0[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 31))) _tmem_load_0[127] = -CUDART_INF_F;
                    }
                    int valid_count = causal_row - n_block * BLOCK_N + 1;
                    uint32_t _thresh_mask_9;
                    {
                        int _lim_10 = valid_count;
                        if (_lim_10 <= 0) { _thresh_mask_9 = 0u; }
                        else if (_lim_10 >= 32) { _thresh_mask_9 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_9) : "r"(_lim_10));
                        }
                    }
                    if (!(_thresh_mask_9 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(_thresh_mask_9 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                    uint32_t _thresh_mask_11;
                    {
                        int _lim_12 = valid_count - 32;
                        if (_lim_12 <= 0) { _thresh_mask_11 = 0u; }
                        else if (_lim_12 >= 32) { _thresh_mask_11 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_11) : "r"(_lim_12));
                        }
                    }
                    if (!(_thresh_mask_11 & (1u << 0))) _tmem_load_0[32] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 1))) _tmem_load_0[33] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 2))) _tmem_load_0[34] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 3))) _tmem_load_0[35] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 4))) _tmem_load_0[36] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 5))) _tmem_load_0[37] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 6))) _tmem_load_0[38] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 7))) _tmem_load_0[39] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 8))) _tmem_load_0[40] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 9))) _tmem_load_0[41] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 10))) _tmem_load_0[42] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 11))) _tmem_load_0[43] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 12))) _tmem_load_0[44] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 13))) _tmem_load_0[45] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 14))) _tmem_load_0[46] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 15))) _tmem_load_0[47] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 16))) _tmem_load_0[48] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 17))) _tmem_load_0[49] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 18))) _tmem_load_0[50] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 19))) _tmem_load_0[51] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 20))) _tmem_load_0[52] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 21))) _tmem_load_0[53] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 22))) _tmem_load_0[54] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 23))) _tmem_load_0[55] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 24))) _tmem_load_0[56] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 25))) _tmem_load_0[57] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 26))) _tmem_load_0[58] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 27))) _tmem_load_0[59] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 28))) _tmem_load_0[60] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 29))) _tmem_load_0[61] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 30))) _tmem_load_0[62] = -CUDART_INF_F;
                    if (!(_thresh_mask_11 & (1u << 31))) _tmem_load_0[63] = -CUDART_INF_F;
                    uint32_t _thresh_mask_13;
                    {
                        int _lim_14 = valid_count - 64;
                        if (_lim_14 <= 0) { _thresh_mask_13 = 0u; }
                        else if (_lim_14 >= 32) { _thresh_mask_13 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_13) : "r"(_lim_14));
                        }
                    }
                    if (!(_thresh_mask_13 & (1u << 0))) _tmem_load_0[64] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 1))) _tmem_load_0[65] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 2))) _tmem_load_0[66] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 3))) _tmem_load_0[67] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 4))) _tmem_load_0[68] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 5))) _tmem_load_0[69] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 6))) _tmem_load_0[70] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 7))) _tmem_load_0[71] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 8))) _tmem_load_0[72] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 9))) _tmem_load_0[73] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 10))) _tmem_load_0[74] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 11))) _tmem_load_0[75] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 12))) _tmem_load_0[76] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 13))) _tmem_load_0[77] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 14))) _tmem_load_0[78] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 15))) _tmem_load_0[79] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 16))) _tmem_load_0[80] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 17))) _tmem_load_0[81] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 18))) _tmem_load_0[82] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 19))) _tmem_load_0[83] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 20))) _tmem_load_0[84] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 21))) _tmem_load_0[85] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 22))) _tmem_load_0[86] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 23))) _tmem_load_0[87] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 24))) _tmem_load_0[88] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 25))) _tmem_load_0[89] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 26))) _tmem_load_0[90] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 27))) _tmem_load_0[91] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 28))) _tmem_load_0[92] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 29))) _tmem_load_0[93] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 30))) _tmem_load_0[94] = -CUDART_INF_F;
                    if (!(_thresh_mask_13 & (1u << 31))) _tmem_load_0[95] = -CUDART_INF_F;
                    uint32_t _thresh_mask_15;
                    {
                        int _lim_16 = valid_count - 96;
                        if (_lim_16 <= 0) { _thresh_mask_15 = 0u; }
                        else if (_lim_16 >= 32) { _thresh_mask_15 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_15) : "r"(_lim_16));
                        }
                    }
                    if (!(_thresh_mask_15 & (1u << 0))) _tmem_load_0[96] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 1))) _tmem_load_0[97] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 2))) _tmem_load_0[98] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 3))) _tmem_load_0[99] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 4))) _tmem_load_0[100] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 5))) _tmem_load_0[101] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 6))) _tmem_load_0[102] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 7))) _tmem_load_0[103] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 8))) _tmem_load_0[104] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 9))) _tmem_load_0[105] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 10))) _tmem_load_0[106] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 11))) _tmem_load_0[107] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 12))) _tmem_load_0[108] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 13))) _tmem_load_0[109] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 14))) _tmem_load_0[110] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 15))) _tmem_load_0[111] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 16))) _tmem_load_0[112] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 17))) _tmem_load_0[113] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 18))) _tmem_load_0[114] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 19))) _tmem_load_0[115] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 20))) _tmem_load_0[116] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 21))) _tmem_load_0[117] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 22))) _tmem_load_0[118] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 23))) _tmem_load_0[119] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 24))) _tmem_load_0[120] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 25))) _tmem_load_0[121] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 26))) _tmem_load_0[122] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 27))) _tmem_load_0[123] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 28))) _tmem_load_0[124] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 29))) _tmem_load_0[125] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 30))) _tmem_load_0[126] = -CUDART_INF_F;
                    if (!(_thresh_mask_15 & (1u << 31))) _tmem_load_0[127] = -CUDART_INF_F;
                    float2 _reg_reduce_max2_17 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_17);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_17);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_17);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_17);
                    float new_max = row_max_reduce(_reg_reduce_max2_17);
                    new_max = max_noftz(new_max, row_max_val);
                    float new_max_scaled = ((new_max == -CUDART_INF_F) ? 0.0f : new_max) * softmax_scale_log2;
                    float acc_scale;
                    float selected_max;
                    float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled);
                    if (_fma_0 >= -8.0f) {
                        selected_max = row_max_val;
                        acc_scale = 1.0f;
                        new_max_scaled = ((row_max_val == -CUDART_INF_F) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        if (row_max_val > -CUDART_INF_F) {
                            float _exp2_18 = approx_exp2(_fma_0);
                            acc_scale = _exp2_18;
                        } else {
                            acc_scale = 1.0f;
                        }
                    }
                    row_max_val = selected_max;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    const float2 _fma_b2_19 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_20 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_19, _fma_c2_20);
                    int p_base = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    if (stage == 0) {
                        int _parity_order_p01_0 = (int)((unsigned int)order_p01_seen % 2);
                        mbarrier_wait_plain(order_p01_0_addr, _parity_order_p01_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        int _parity_order_p01_1 = (int)((unsigned int)order_p01_seen % 2);
                        mbarrier_wait_plain(order_p01_1_addr, _parity_order_p01_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    unsigned int _tmem_load_0_bf16[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[0], _tmem_load_0_bf16, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[15])));
                    unsigned int _tmem_load_0_bf16_0[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[32], _tmem_load_0_bf16_0, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 16), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_0[15])));
                    unsigned int _tmem_load_0_bf16_1[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[64], _tmem_load_0_bf16_1, 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_1[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_0[_le + 96] = approx_exp2(_tmem_load_0[_le + 96]);
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_seen += 1;
                    unsigned int _tmem_load_0_bf16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 96], _tmem_load_0[_lp*2+1 + 96]));
                        _tmem_load_0_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16_2[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    float2 _reg_reduce_sum2_22 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[(96) + 0], &_reg_reduce_sum2_22);
                    float _reg_array_reduce_21 = _reg_reduce_sum2_22.x + _reg_reduce_sum2_22.y;
                    float2 _reg_reduce_sum2_24 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[(64) + 0], &_reg_reduce_sum2_24);
                    float _reg_array_reduce_23 = _reg_reduce_sum2_24.x + _reg_reduce_sum2_24.y;
                    float2 _reg_reduce_sum2_26 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[(32) + 0], &_reg_reduce_sum2_26);
                    float _reg_array_reduce_25 = _reg_reduce_sum2_26.x + _reg_reduce_sum2_26.y;
                    float2 _reg_reduce_sum2_28 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                    _reg_reduce_sum2_28 = add_f32x2(_reg_reduce_sum2_28, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                    float _reg_array_reduce_27 = _reg_reduce_sum2_28.x + _reg_reduce_sum2_28.y;
                    row_sum_val = __fmaf_rn(row_sum_val, acc_scale, _reg_array_reduce_21 + _reg_array_reduce_23 + _reg_array_reduce_25 + _reg_array_reduce_27);
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = unmasked_loop_start; n_iter_1 < num_n_blocks; n_iter_1++) {
                    int n_block_1 = num_n_blocks - 1 - n_iter_1;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_base_1 = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
#if __CUDA_ARCH__ == 1030
                    float sv[128];
                    float tile_max = -CUDART_INF_F;
                    float tile_max_lo = -CUDART_INF_F;
                    float tile_max_hi = -CUDART_INF_F;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
#else
                    float _tmem_load_1[128];
#endif
                    asm volatile(
#if __CUDA_ARCH__ == 1030
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(tile_max_lo)
#else
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
#endif
                        : "r"(s_base_1));
#if __CUDA_ARCH__ == 1030
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
#else
#endif
                    asm volatile(
#if __CUDA_ARCH__ == 1030
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(tile_max_hi)
#else
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(s_base_1 + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95])
#endif
                        : "r"(s_base_1 + 64));
#if __CUDA_ARCH__ == 1030
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    tile_max = max_noftz(tile_max_lo, tile_max_hi);
#else
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                        : "r"(s_base_1 + 96));
#endif
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
#if __CUDA_ARCH__ == 1030
                        if (!(_thresh_mask_29 & (1u << 0))) sv[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 1))) sv[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 2))) sv[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 3))) sv[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 4))) sv[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 5))) sv[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 6))) sv[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 7))) sv[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 8))) sv[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 9))) sv[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 10))) sv[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 11))) sv[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 12))) sv[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 13))) sv[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 14))) sv[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 15))) sv[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 16))) sv[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 17))) sv[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 18))) sv[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 19))) sv[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 20))) sv[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 21))) sv[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 22))) sv[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 23))) sv[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 24))) sv[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 25))) sv[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 26))) sv[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 27))) sv[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 28))) sv[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 29))) sv[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 30))) sv[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_29 & (1u << 31))) sv[31] = -CUDART_INF_F;
#else
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
#endif
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
#if __CUDA_ARCH__ == 1030
                        if (!(_thresh_mask_31 & (1u << 0))) sv[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 1))) sv[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 2))) sv[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 3))) sv[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 4))) sv[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 5))) sv[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 6))) sv[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 7))) sv[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 8))) sv[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 9))) sv[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 10))) sv[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 11))) sv[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 12))) sv[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 13))) sv[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 14))) sv[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 15))) sv[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 16))) sv[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 17))) sv[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 18))) sv[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 19))) sv[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 20))) sv[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 21))) sv[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 22))) sv[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 23))) sv[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 24))) sv[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 25))) sv[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 26))) sv[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 27))) sv[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 28))) sv[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 29))) sv[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 30))) sv[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_31 & (1u << 31))) sv[63] = -CUDART_INF_F;
#else
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
#endif
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
#if __CUDA_ARCH__ == 1030
                        if (!(_thresh_mask_33 & (1u << 0))) sv[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 1))) sv[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 2))) sv[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 3))) sv[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 4))) sv[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 5))) sv[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 6))) sv[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 7))) sv[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 8))) sv[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 9))) sv[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 10))) sv[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 11))) sv[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 12))) sv[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 13))) sv[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 14))) sv[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 15))) sv[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 16))) sv[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 17))) sv[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 18))) sv[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 19))) sv[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 20))) sv[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 21))) sv[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 22))) sv[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 23))) sv[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 24))) sv[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 25))) sv[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 26))) sv[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 27))) sv[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 28))) sv[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 29))) sv[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 30))) sv[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_33 & (1u << 31))) sv[95] = -CUDART_INF_F;
#else
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
#endif
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
#if __CUDA_ARCH__ == 1030
                        if (!(_thresh_mask_35 & (1u << 0))) sv[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 1))) sv[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 2))) sv[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 3))) sv[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 4))) sv[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 5))) sv[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 6))) sv[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 7))) sv[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 8))) sv[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 9))) sv[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 10))) sv[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 11))) sv[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 12))) sv[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 13))) sv[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 14))) sv[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 15))) sv[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 16))) sv[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 17))) sv[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 18))) sv[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 19))) sv[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 20))) sv[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 21))) sv[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 22))) sv[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 23))) sv[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 24))) sv[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 25))) sv[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 26))) sv[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 27))) sv[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 28))) sv[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 29))) sv[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 30))) sv[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_35 & (1u << 31))) sv[127] = -CUDART_INF_F;
                        float2 _reg_reduce_max2_38 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_38);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_38);
                        row_max_x32_accum(&sv[64], _reg_reduce_max2_38);
                        row_max_x32_accum(&sv[96], _reg_reduce_max2_38);
                        float _reg_array_reduce_37 = row_max_reduce(_reg_reduce_max2_38);
                        tile_max = _reg_array_reduce_37;
#else
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
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    float new_max_1 = tile_max;
#else
                    float2 _reg_reduce_max2_37 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_37);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_37);
                    row_max_x32_accum(&_tmem_load_1[64], _reg_reduce_max2_37);
                    row_max_x32_accum(&_tmem_load_1[96], _reg_reduce_max2_37);
                    float new_max_1 = row_max_reduce(_reg_reduce_max2_37);
#endif
                    new_max_1 = max_noftz(new_max_1, row_max_val);
                    float new_max_scaled_1 = ((new_max_1 == -CUDART_INF_F) ? 0.0f : new_max_1) * softmax_scale_log2;
                    float acc_scale_1;
                    float selected_max_1;
                    float _fma_1 = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max_scaled_1);
                    if (_fma_1 >= -8.0f) {
                        selected_max_1 = row_max_val;
                        acc_scale_1 = 1.0f;
                        new_max_scaled_1 = ((row_max_val == -CUDART_INF_F) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max_1 = new_max_1;
                        if (row_max_val > -CUDART_INF_F) {
#if __CUDA_ARCH__ == 1030
                            float _exp2_39 = approx_exp2(_fma_1);
                            acc_scale_1 = _exp2_39;
#else
                            float _exp2_38 = approx_exp2(_fma_1);
                            acc_scale_1 = _exp2_38;
#endif
                        } else {
                            acc_scale_1 = 1.0f;
                        }
                    }
                    row_max_val = selected_max_1;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale_1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_40 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_41 = {-new_max_scaled_1, -new_max_scaled_1};
#else
                    const float2 _fma_b2_39 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_40 = {-new_max_scaled_1, -new_max_scaled_1};
#endif
                    #pragma unroll
#if __CUDA_ARCH__ == 1030
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_lf], _fma_b2_40, _fma_c2_41);
#else
                    for (int _lf = 0; _lf < 48; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 0))[_lf], _fma_b2_39, _fma_c2_40);
#endif
                    int p_base_1 = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    if (stage == 0) {
                        int _parity_order_p01_0_1 = (int)((unsigned int)order_p01_seen % 2);
                        mbarrier_wait_plain(order_p01_0_addr, _parity_order_p01_0_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        int _parity_order_p01_1_1 = (int)((unsigned int)order_p01_seen % 2);
                        mbarrier_wait_plain(order_p01_1_addr, _parity_order_p01_1_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    {
                        #pragma unroll
                        for (int _le = 0; _le < 24; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le] = approx_exp2(sv[_le]);
#else
                            _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        float2 _exp2_pair_42 = ex2_emulation_f32x2_value(make_float2(sv[24], sv[25]));
                        sv[24] = _exp2_pair_42.x;
                        sv[25] = _exp2_pair_42.y;
                        float2 _exp2_pair_43 = ex2_emulation_f32x2_value(make_float2(sv[26], sv[27]));
                        sv[26] = _exp2_pair_43.x;
                        sv[27] = _exp2_pair_43.y;
                        float2 _exp2_pair_44 = ex2_emulation_f32x2_value(make_float2(sv[28], sv[29]));
                        sv[28] = _exp2_pair_44.x;
                        sv[29] = _exp2_pair_44.y;
                        float2 _exp2_pair_45 = ex2_emulation_f32x2_value(make_float2(sv[30], sv[31]));
                        sv[30] = _exp2_pair_45.x;
                        sv[31] = _exp2_pair_45.y;
#else
                        float2 _exp2_pair_41 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[24], _tmem_load_1[25]));
                        _tmem_load_1[24] = _exp2_pair_41.x;
                        _tmem_load_1[25] = _exp2_pair_41.y;
                        float2 _exp2_pair_42 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[26], _tmem_load_1[27]));
                        _tmem_load_1[26] = _exp2_pair_42.x;
                        _tmem_load_1[27] = _exp2_pair_42.y;
                        float2 _exp2_pair_43 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[28], _tmem_load_1[29]));
                        _tmem_load_1[28] = _exp2_pair_43.x;
                        _tmem_load_1[29] = _exp2_pair_43.y;
                        float2 _exp2_pair_44 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[30], _tmem_load_1[31]));
                        _tmem_load_1[30] = _exp2_pair_44.x;
                        _tmem_load_1[31] = _exp2_pair_44.y;
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    unsigned int sv_bf16[16];
#else
                    unsigned int _tmem_load_1_bf16[16];
#endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
#if __CUDA_ARCH__ == 1030
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                        sv_bf16[_lp] = *(uint32_t*)&_bf2;
#else
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
#endif
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
#if __CUDA_ARCH__ == 1030
                        :: "r"(p_base_1), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[15])));
#else
                        :: "r"(p_base_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[15])));
#endif
                    {
                        #pragma unroll
                        for (int _le = 0; _le < 24; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 32] = approx_exp2(sv[_le + 32]);
#else
                            _tmem_load_1[_le + 32] = approx_exp2(_tmem_load_1[_le + 32]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        float2 _exp2_pair_46 = ex2_emulation_f32x2_value(make_float2(sv[56], sv[57]));
                        sv[56] = _exp2_pair_46.x;
                        sv[57] = _exp2_pair_46.y;
                        float2 _exp2_pair_47 = ex2_emulation_f32x2_value(make_float2(sv[58], sv[59]));
                        sv[58] = _exp2_pair_47.x;
                        sv[59] = _exp2_pair_47.y;
                        float2 _exp2_pair_48 = ex2_emulation_f32x2_value(make_float2(sv[60], sv[61]));
                        sv[60] = _exp2_pair_48.x;
                        sv[61] = _exp2_pair_48.y;
                        float2 _exp2_pair_49 = ex2_emulation_f32x2_value(make_float2(sv[62], sv[63]));
                        sv[62] = _exp2_pair_49.x;
                        sv[63] = _exp2_pair_49.y;
#else
                        float2 _exp2_pair_45 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[56], _tmem_load_1[57]));
                        _tmem_load_1[56] = _exp2_pair_45.x;
                        _tmem_load_1[57] = _exp2_pair_45.y;
                        float2 _exp2_pair_46 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[58], _tmem_load_1[59]));
                        _tmem_load_1[58] = _exp2_pair_46.x;
                        _tmem_load_1[59] = _exp2_pair_46.y;
                        float2 _exp2_pair_47 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[60], _tmem_load_1[61]));
                        _tmem_load_1[60] = _exp2_pair_47.x;
                        _tmem_load_1[61] = _exp2_pair_47.y;
                        float2 _exp2_pair_48 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[62], _tmem_load_1[63]));
                        _tmem_load_1[62] = _exp2_pair_48.x;
                        _tmem_load_1[63] = _exp2_pair_48.y;
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    unsigned int sv_bf16_0[16];
#else
                    unsigned int _tmem_load_1_bf16_0[16];
#endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
#if __CUDA_ARCH__ == 1030
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 32], sv[_lp*2+1 + 32]));
                        sv_bf16_0[_lp] = *(uint32_t*)&_bf2;
#else
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 32], _tmem_load_1[_lp*2+1 + 32]));
                        _tmem_load_1_bf16_0[_lp] = *(uint32_t*)&_bf2;
#endif
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
#if __CUDA_ARCH__ == 1030
                        :: "r"(p_base_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_0[15])));
#else
                        :: "r"(p_base_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_0[15])));
#endif
                    {
                        #pragma unroll
                        for (int _le = 0; _le < 24; _le++) {
#if __CUDA_ARCH__ == 1030
                            sv[_le + 64] = approx_exp2(sv[_le + 64]);
#else
                            _tmem_load_1[_le + 64] = approx_exp2(_tmem_load_1[_le + 64]);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        float2 _exp2_pair_50 = ex2_emulation_f32x2_value(make_float2(sv[88], sv[89]));
                        sv[88] = _exp2_pair_50.x;
                        sv[89] = _exp2_pair_50.y;
                        float2 _exp2_pair_51 = ex2_emulation_f32x2_value(make_float2(sv[90], sv[91]));
                        sv[90] = _exp2_pair_51.x;
                        sv[91] = _exp2_pair_51.y;
                        float2 _exp2_pair_52 = ex2_emulation_f32x2_value(make_float2(sv[92], sv[93]));
                        sv[92] = _exp2_pair_52.x;
                        sv[93] = _exp2_pair_52.y;
                        float2 _exp2_pair_53 = ex2_emulation_f32x2_value(make_float2(sv[94], sv[95]));
                        sv[94] = _exp2_pair_53.x;
                        sv[95] = _exp2_pair_53.y;
#else
                        float2 _exp2_pair_49 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[88], _tmem_load_1[89]));
                        _tmem_load_1[88] = _exp2_pair_49.x;
                        _tmem_load_1[89] = _exp2_pair_49.y;
                        float2 _exp2_pair_50 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[90], _tmem_load_1[91]));
                        _tmem_load_1[90] = _exp2_pair_50.x;
                        _tmem_load_1[91] = _exp2_pair_50.y;
                        float2 _exp2_pair_51 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[92], _tmem_load_1[93]));
                        _tmem_load_1[92] = _exp2_pair_51.x;
                        _tmem_load_1[93] = _exp2_pair_51.y;
                        float2 _exp2_pair_52 = ex2_emulation_f32x2_value(make_float2(_tmem_load_1[94], _tmem_load_1[95]));
                        _tmem_load_1[94] = _exp2_pair_52.x;
                        _tmem_load_1[95] = _exp2_pair_52.y;
#endif
                    }
#if __CUDA_ARCH__ == 1030
                    unsigned int sv_bf16_1[16];
#else
                    unsigned int _tmem_load_1_bf16_1[16];
#endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
#if __CUDA_ARCH__ == 1030
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 64], sv[_lp*2+1 + 64]));
                        sv_bf16_1[_lp] = *(uint32_t*)&_bf2;
#else
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 64], _tmem_load_1[_lp*2+1 + 64]));
                        _tmem_load_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
#endif
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
#if __CUDA_ARCH__ == 1030
                        :: "r"(p_base_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[15])));
#else
                        :: "r"(p_base_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_1[15])));
#endif
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
#if __CUDA_ARCH__ == 1030
#else
                    const float2 _fma_b2_53 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_54 = {-new_max_scaled_1, -new_max_scaled_1};
                    float2 _fma_pair_55 = fma_f32x2(make_float2((_tmem_load_1 + 96)[0], (_tmem_load_1 + 96)[1]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[0] = _fma_pair_55.x;
                    (_tmem_load_1 + 96)[1] = _fma_pair_55.y;
                    float2 _fma_pair_56 = fma_f32x2(make_float2((_tmem_load_1 + 96)[2], (_tmem_load_1 + 96)[3]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[2] = _fma_pair_56.x;
                    (_tmem_load_1 + 96)[3] = _fma_pair_56.y;
                    float2 _fma_pair_57 = fma_f32x2(make_float2((_tmem_load_1 + 96)[4], (_tmem_load_1 + 96)[5]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[4] = _fma_pair_57.x;
                    (_tmem_load_1 + 96)[5] = _fma_pair_57.y;
                    float2 _fma_pair_58 = fma_f32x2(make_float2((_tmem_load_1 + 96)[6], (_tmem_load_1 + 96)[7]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[6] = _fma_pair_58.x;
                    (_tmem_load_1 + 96)[7] = _fma_pair_58.y;
                    float2 _fma_pair_59 = fma_f32x2(make_float2((_tmem_load_1 + 96)[8], (_tmem_load_1 + 96)[9]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[8] = _fma_pair_59.x;
                    (_tmem_load_1 + 96)[9] = _fma_pair_59.y;
                    float2 _fma_pair_60 = fma_f32x2(make_float2((_tmem_load_1 + 96)[10], (_tmem_load_1 + 96)[11]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[10] = _fma_pair_60.x;
                    (_tmem_load_1 + 96)[11] = _fma_pair_60.y;
                    float2 _fma_pair_61 = fma_f32x2(make_float2((_tmem_load_1 + 96)[12], (_tmem_load_1 + 96)[13]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[12] = _fma_pair_61.x;
                    (_tmem_load_1 + 96)[13] = _fma_pair_61.y;
                    float2 _fma_pair_62 = fma_f32x2(make_float2((_tmem_load_1 + 96)[14], (_tmem_load_1 + 96)[15]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[14] = _fma_pair_62.x;
                    (_tmem_load_1 + 96)[15] = _fma_pair_62.y;
                    float2 _fma_pair_63 = fma_f32x2(make_float2((_tmem_load_1 + 96)[16], (_tmem_load_1 + 96)[17]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[16] = _fma_pair_63.x;
                    (_tmem_load_1 + 96)[17] = _fma_pair_63.y;
                    float2 _fma_pair_64 = fma_f32x2(make_float2((_tmem_load_1 + 96)[18], (_tmem_load_1 + 96)[19]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[18] = _fma_pair_64.x;
                    (_tmem_load_1 + 96)[19] = _fma_pair_64.y;
                    float2 _fma_pair_65 = fma_f32x2(make_float2((_tmem_load_1 + 96)[20], (_tmem_load_1 + 96)[21]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[20] = _fma_pair_65.x;
                    (_tmem_load_1 + 96)[21] = _fma_pair_65.y;
                    float2 _fma_pair_66 = fma_f32x2(make_float2((_tmem_load_1 + 96)[22], (_tmem_load_1 + 96)[23]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[22] = _fma_pair_66.x;
                    (_tmem_load_1 + 96)[23] = _fma_pair_66.y;
                    float2 _fma_pair_67 = fma_f32x2(make_float2((_tmem_load_1 + 96)[24], (_tmem_load_1 + 96)[25]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[24] = _fma_pair_67.x;
                    (_tmem_load_1 + 96)[25] = _fma_pair_67.y;
                    float2 _fma_pair_68 = fma_f32x2(make_float2((_tmem_load_1 + 96)[26], (_tmem_load_1 + 96)[27]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[26] = _fma_pair_68.x;
                    (_tmem_load_1 + 96)[27] = _fma_pair_68.y;
                    float2 _fma_pair_69 = fma_f32x2(make_float2((_tmem_load_1 + 96)[28], (_tmem_load_1 + 96)[29]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[28] = _fma_pair_69.x;
                    (_tmem_load_1 + 96)[29] = _fma_pair_69.y;
                    float2 _fma_pair_70 = fma_f32x2(make_float2((_tmem_load_1 + 96)[30], (_tmem_load_1 + 96)[31]), _fma_b2_53, _fma_c2_54);
                    (_tmem_load_1 + 96)[30] = _fma_pair_70.x;
                    (_tmem_load_1 + 96)[31] = _fma_pair_70.y;
#endif
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
#if __CUDA_ARCH__ == 1030
                        sv[_le + 96] = approx_exp2(sv[_le + 96]);
#else
                        _tmem_load_1[_le + 96] = approx_exp2(_tmem_load_1[_le + 96]);
#endif
                    }
                    if (stage == 0) {
                        mbarrier_arrive(order_p01_1_addr);
                    } else {
                        mbarrier_arrive(order_p01_0_addr);
                    }
                    order_p01_seen += 1;
#if __CUDA_ARCH__ == 1030
                    unsigned int sv_bf16_2[16];
#else
                    unsigned int _tmem_load_1_bf16_2[16];
#endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
#if __CUDA_ARCH__ == 1030
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 96], sv[_lp*2+1 + 96]));
                        sv_bf16_2[_lp] = *(uint32_t*)&_bf2;
#else
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 96], _tmem_load_1[_lp*2+1 + 96]));
                        _tmem_load_1_bf16_2[_lp] = *(uint32_t*)&_bf2;
#endif
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
#if __CUDA_ARCH__ == 1030
                        :: "r"(p_base_1 + 48), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_2[15])));
#else
                        :: "r"(p_base_1 + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16_2[15])));
#endif
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
#if __CUDA_ARCH__ == 1030
                    float2 _reg_reduce_sum2_55 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[0], sv[1]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[2], sv[3]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[4], sv[5]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[6], sv[7]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[8], sv[9]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[10], sv[11]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[12], sv[13]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[14], sv[15]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[16], sv[17]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[18], sv[19]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[20], sv[21]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[22], sv[23]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[24], sv[25]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[26], sv[27]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[28], sv[29]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[30], sv[31]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[32], sv[33]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[34], sv[35]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[36], sv[37]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[38], sv[39]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[40], sv[41]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[42], sv[43]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[44], sv[45]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[46], sv[47]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[48], sv[49]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[50], sv[51]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[52], sv[53]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[54], sv[55]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[56], sv[57]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[58], sv[59]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[60], sv[61]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[62], sv[63]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[64], sv[65]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[66], sv[67]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[68], sv[69]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[70], sv[71]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[72], sv[73]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[74], sv[75]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[76], sv[77]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[78], sv[79]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[80], sv[81]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[82], sv[83]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[84], sv[85]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[86], sv[87]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[88], sv[89]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[90], sv[91]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[92], sv[93]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[94], sv[95]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[96], sv[97]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[98], sv[99]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[100], sv[101]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[102], sv[103]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[104], sv[105]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[106], sv[107]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[108], sv[109]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[110], sv[111]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[112], sv[113]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[114], sv[115]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[116], sv[117]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[118], sv[119]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[120], sv[121]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[122], sv[123]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[124], sv[125]));
                    _reg_reduce_sum2_55 = add_f32x2_rn_noftz(_reg_reduce_sum2_55, make_float2(sv[126], sv[127]));
                    float _reg_array_reduce_54 = _reg_reduce_sum2_55.x + _reg_reduce_sum2_55.y;
                    row_sum_val = __fmaf_rn(row_sum_val, acc_scale_1, _reg_array_reduce_54);
#else
                    float2 _reg_reduce_sum2_72 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[4], _tmem_load_1[5]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[6], _tmem_load_1[7]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[8], _tmem_load_1[9]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[10], _tmem_load_1[11]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[12], _tmem_load_1[13]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[14], _tmem_load_1[15]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[16], _tmem_load_1[17]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[18], _tmem_load_1[19]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[20], _tmem_load_1[21]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[22], _tmem_load_1[23]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[24], _tmem_load_1[25]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[26], _tmem_load_1[27]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[28], _tmem_load_1[29]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[30], _tmem_load_1[31]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[32], _tmem_load_1[33]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[34], _tmem_load_1[35]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[36], _tmem_load_1[37]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[38], _tmem_load_1[39]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[40], _tmem_load_1[41]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[42], _tmem_load_1[43]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[44], _tmem_load_1[45]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[46], _tmem_load_1[47]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[48], _tmem_load_1[49]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[50], _tmem_load_1[51]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[52], _tmem_load_1[53]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[54], _tmem_load_1[55]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[56], _tmem_load_1[57]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[58], _tmem_load_1[59]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[60], _tmem_load_1[61]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[62], _tmem_load_1[63]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[64], _tmem_load_1[65]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[66], _tmem_load_1[67]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[68], _tmem_load_1[69]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[70], _tmem_load_1[71]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[72], _tmem_load_1[73]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[74], _tmem_load_1[75]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[76], _tmem_load_1[77]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[78], _tmem_load_1[79]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[80], _tmem_load_1[81]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[82], _tmem_load_1[83]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[84], _tmem_load_1[85]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[86], _tmem_load_1[87]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[88], _tmem_load_1[89]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[90], _tmem_load_1[91]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[92], _tmem_load_1[93]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[94], _tmem_load_1[95]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[96], _tmem_load_1[97]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[98], _tmem_load_1[99]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[100], _tmem_load_1[101]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[102], _tmem_load_1[103]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[104], _tmem_load_1[105]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[106], _tmem_load_1[107]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[108], _tmem_load_1[109]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[110], _tmem_load_1[111]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[112], _tmem_load_1[113]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[114], _tmem_load_1[115]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[116], _tmem_load_1[117]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[118], _tmem_load_1[119]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[120], _tmem_load_1[121]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[122], _tmem_load_1[123]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[124], _tmem_load_1[125]));
                    _reg_reduce_sum2_72 = add_f32x2_rn_noftz(_reg_reduce_sum2_72, make_float2(_tmem_load_1[126], _tmem_load_1[127]));
                    float _reg_array_reduce_71 = _reg_reduce_sum2_72.x + _reg_reduce_sum2_72.y;
                    row_sum_val = __fmaf_rn(row_sum_val, acc_scale_1, _reg_array_reduce_71);
#endif
                }
                sScale[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum_val;
                sScale[warp % 4 * 32 + lane + scale_off + 4 * BLOCK_M] = row_max_val;
                mbarrier_arrive(corr_sig_addr + (stage) * 8);
                mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                _phase_corr_done ^= 1;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            unsigned int total_tiles_1 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_1 = total_tiles_1 + 1;
            int corr_seq = 0;
            unsigned int _phase_work_id_full_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_iter_1 = 0; tile_iter_1 < max_rounds_1; tile_iter_1++) {
                unsigned int tile_idx_1;
                {
                    mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_1);
                    _phase_work_id_full_0_1 ^= 1;
                    tile_idx_1 = work_id_slot[0];
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_1 >= total_tiles_1) {
                    break;
                }
                unsigned int m_block_1;
                unsigned int bh_1;
                {
                    unsigned int l2section_1 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_1 = tile_idx_1 / l2section_1;
                    unsigned int l2_mod_1 = tile_idx_1 % l2section_1;
                    {
                        bool tail_1 = section_1 >= (unsigned int)total_bh / (unsigned int)L2_SWIZZLE;
                        m_block_1 = ((tail_1) ? l2_mod_1 % (unsigned int)NUM_M_BLOCKS : l2_mod_1 / (unsigned int)L2_SWIZZLE);
                        bh_1 = section_1 * (unsigned int)L2_SWIZZLE + ((tail_1) ? l2_mod_1 / (unsigned int)NUM_M_BLOCKS : l2_mod_1 % (unsigned int)L2_SWIZZLE);
                    }
                    m_block_1 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_1;
                }
                int seqlen_kv_bh_1 = UNIFORM_KV_LEN;
                int seqlen_q_bh_1 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_1 = seq_lens_kv[bh_1 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                {
                    seqlen_q_bh_1 = cu_seq_lens_q[bh_1 / (unsigned int)(NUM_Q_HEADS / PACK_G) + 1] - cu_seq_lens_q[bh_1 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                unsigned int num_n_blocks_1 = (seqlen_kv_bh_1 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_2 = seqlen_kv_bh_1 - seqlen_q_bh_1;
                    unsigned int max_n_1 = (m_block_1 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_2;
                    if (max_n_1 < (unsigned int)seqlen_kv_bh_1) {
                        num_n_blocks_1 = (max_n_1 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                unsigned int _max_0_1 = ((num_n_blocks_1) > (1) ? (num_n_blocks_1) : (1));
                num_n_blocks_1 = _max_0_1;
                int unit = bh_1 % (unsigned int)(NUM_Q_HEADS / PACK_G);
                int q_head_base = unit * PACK_G;
                int row_tok = (warp % 4 * 32 + lane) / PACK_G;
                int row_head = q_head_base + (warp % 4 * 32 + lane) % PACK_G;
                int tok_base = cu_seq_lens_q[bh_1 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                if (tile_iter_1 == 0) {
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(p_full_addr + 8);
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale = sScale[warp % 4 * 32 + lane];
                    int _warp_vote_1 = __any_sync(0xFFFFFFFF, scale < 1.0f);
                    if (_warp_vote_1 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < HEAD_DIM / 16; cr_col++) {
                            int cr_addr = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
#if __CUDA_ARCH__ == 1030
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], cr_addr);
#else
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr);
#endif
                            const float2 _scale2_2 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_2);
                            tmem_st_x16_f32(cr_addr, _tmem_load_1);
#else
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_2);
                            tmem_st_x16_f32(cr_addr, _tmem_load_2);
#endif
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
                    int _warp_vote_3 = __any_sync(0xFFFFFFFF, scale1 < 1.0f);
                    if (_warp_vote_3 != 0) {
                        #pragma unroll
                        for (int cr_col_1 = 0; cr_col_1 < HEAD_DIM / 16; cr_col_1++) {
                            int cr_addr_1 = taddr + (unsigned int)384 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col_1 * 16);
#if __CUDA_ARCH__ == 1030
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr_1);
#else
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], cr_addr_1);
#endif
                            const float2 _scale2_4 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_4);
                            tmem_st_x16_f32(cr_addr_1, _tmem_load_2);
#else
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_4);
                            tmem_st_x16_f32(cr_addr_1, _tmem_load_3);
#endif
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + 8);
                    mbarrier_arrive(corr_done_addr);
                    corr_seq += 1;
                }
                mbarrier_arrive(corr_done_addr + 8);
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                float final_sum_0 = sScale[warp % 4 * 32 + lane + 2 * BLOCK_M];
                float final_max_0 = sScale[warp % 4 * 32 + lane + 4 * BLOCK_M];
                float final_sum_1 = sScale[warp % 4 * 32 + lane + BLOCK_M + 2 * BLOCK_M];
                float final_max_1 = sScale[warp % 4 * 32 + lane + BLOCK_M + 4 * BLOCK_M];
                mbarrier_arrive(corr_done_addr);
                mbarrier_arrive(corr_done_addr + 8);
                bool o_wide = ((unsigned long long)O_ptr & 31) == 0;
                #pragma unroll
                for (int stage_1 = 0; stage_1 < 2; stage_1++) {
                    if (stage_1 == 0) {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                    } else {
                        mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                        _phase_o_full_1 ^= 1;
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int tmem_o_off = ((stage_1 == 0) ? 256 : 384);
                    float final_sum = ((stage_1 == 0) ? final_sum_0 : final_sum_1);
                    float final_max = ((stage_1 == 0) ? final_max_0 : final_max_1);
                    float final_scale;
                    if (final_sum != 0.0f && final_sum == final_sum) {
                        float _rcp_5 = approx_rcp(final_sum);
                        final_scale = _rcp_5;
                    } else {
                        final_scale = 0.0f;
                    }
                    int q_tok0 = (m_block_1 * 2 + (unsigned int)stage_1) * (unsigned int)TOK_PER_STAGE;
                    if (q_tok0 < seqlen_q_bh_1) {
                        int row_valid = warp % 4 * 32 + lane < TOK_PER_STAGE * PACK_G && seqlen_q_bh_1 > q_tok0 + row_tok;
                        long long o_tok = tok_base + q_tok0 + row_tok;
                        long long o_row = o_tok * (long long)NUM_Q_HEADS + (long long)row_head;
                        #pragma unroll
                        for (int ce_col = 0; ce_col < HEAD_DIM / 32; ce_col++) {
                            int ce_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(ce_col * 32);
#if __CUDA_ARCH__ == 1030
                            float _tmem_load_3[32];
#else
                            float _tmem_load_4[32];
#endif
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
#if __CUDA_ARCH__ == 1030
                                : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
#else
                                : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
#endif
                                : "r"(ce_addr));
                            if (row_valid != 0) {
                                long long o_elem = o_row * (long long)HEAD_DIM + (long long)(ce_col * 32);
                                const float2 _scale2_6 = {final_scale, final_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_6);
#else
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_6);
#endif
                                if (o_wide) {
                                    unsigned int _packed_0[16];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
#endif
                                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 2], _tmem_load_3[_lp*2+1 + 2]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 2], _tmem_load_4[_lp*2+1 + 2]));
#endif
                                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 4], _tmem_load_3[_lp*2+1 + 4]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 4], _tmem_load_4[_lp*2+1 + 4]));
#endif
                                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 6], _tmem_load_3[_lp*2+1 + 6]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 6], _tmem_load_4[_lp*2+1 + 6]));
#endif
                                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 8], _tmem_load_3[_lp*2+1 + 8]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 8], _tmem_load_4[_lp*2+1 + 8]));
#endif
                                        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 10], _tmem_load_3[_lp*2+1 + 10]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 10], _tmem_load_4[_lp*2+1 + 10]));
#endif
                                        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 12], _tmem_load_3[_lp*2+1 + 12]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 12], _tmem_load_4[_lp*2+1 + 12]));
#endif
                                        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 14], _tmem_load_3[_lp*2+1 + 14]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 14], _tmem_load_4[_lp*2+1 + 14]));
#endif
                                        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 16], _tmem_load_3[_lp*2+1 + 16]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 16], _tmem_load_4[_lp*2+1 + 16]));
#endif
                                        _packed_0[_lp + 8] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 18], _tmem_load_3[_lp*2+1 + 18]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 18], _tmem_load_4[_lp*2+1 + 18]));
#endif
                                        _packed_0[_lp + 9] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 20], _tmem_load_3[_lp*2+1 + 20]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 20], _tmem_load_4[_lp*2+1 + 20]));
#endif
                                        _packed_0[_lp + 10] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 22], _tmem_load_3[_lp*2+1 + 22]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 22], _tmem_load_4[_lp*2+1 + 22]));
#endif
                                        _packed_0[_lp + 11] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 24], _tmem_load_3[_lp*2+1 + 24]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 24], _tmem_load_4[_lp*2+1 + 24]));
#endif
                                        _packed_0[_lp + 12] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 26], _tmem_load_3[_lp*2+1 + 26]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 26], _tmem_load_4[_lp*2+1 + 26]));
#endif
                                        _packed_0[_lp + 13] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 28], _tmem_load_3[_lp*2+1 + 28]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 28], _tmem_load_4[_lp*2+1 + 28]));
#endif
                                        _packed_0[_lp + 14] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 30], _tmem_load_3[_lp*2+1 + 30]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 30], _tmem_load_4[_lp*2+1 + 30]));
#endif
                                        _packed_0[_lp + 15] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(O_ptr + o_elem), "r"(_packed_0[(0) + 0]), "r"(_packed_0[(0) + 1]), "r"(_packed_0[(0) + 2]), "r"(_packed_0[(0) + 3]), "r"(_packed_0[(0) + 4]), "r"(_packed_0[(0) + 5]), "r"(_packed_0[(0) + 6]), "r"(_packed_0[(0) + 7]) : "memory");
                                    asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(O_ptr + (o_elem + 16)), "r"(_packed_0[(8) + 0]), "r"(_packed_0[(8) + 1]), "r"(_packed_0[(8) + 2]), "r"(_packed_0[(8) + 3]), "r"(_packed_0[(8) + 4]), "r"(_packed_0[(8) + 5]), "r"(_packed_0[(8) + 6]), "r"(_packed_0[(8) + 7]) : "memory");
                                } else {
                                    unsigned int _packed_1[16];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
#endif
                                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 2], _tmem_load_3[_lp*2+1 + 2]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 2], _tmem_load_4[_lp*2+1 + 2]));
#endif
                                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 4], _tmem_load_3[_lp*2+1 + 4]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 4], _tmem_load_4[_lp*2+1 + 4]));
#endif
                                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 6], _tmem_load_3[_lp*2+1 + 6]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 6], _tmem_load_4[_lp*2+1 + 6]));
#endif
                                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 8], _tmem_load_3[_lp*2+1 + 8]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 8], _tmem_load_4[_lp*2+1 + 8]));
#endif
                                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 10], _tmem_load_3[_lp*2+1 + 10]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 10], _tmem_load_4[_lp*2+1 + 10]));
#endif
                                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 12], _tmem_load_3[_lp*2+1 + 12]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 12], _tmem_load_4[_lp*2+1 + 12]));
#endif
                                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 14], _tmem_load_3[_lp*2+1 + 14]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 14], _tmem_load_4[_lp*2+1 + 14]));
#endif
                                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 16], _tmem_load_3[_lp*2+1 + 16]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 16], _tmem_load_4[_lp*2+1 + 16]));
#endif
                                        _packed_1[_lp + 8] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 18], _tmem_load_3[_lp*2+1 + 18]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 18], _tmem_load_4[_lp*2+1 + 18]));
#endif
                                        _packed_1[_lp + 9] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 20], _tmem_load_3[_lp*2+1 + 20]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 20], _tmem_load_4[_lp*2+1 + 20]));
#endif
                                        _packed_1[_lp + 10] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 22], _tmem_load_3[_lp*2+1 + 22]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 22], _tmem_load_4[_lp*2+1 + 22]));
#endif
                                        _packed_1[_lp + 11] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 24], _tmem_load_3[_lp*2+1 + 24]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 24], _tmem_load_4[_lp*2+1 + 24]));
#endif
                                        _packed_1[_lp + 12] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 26], _tmem_load_3[_lp*2+1 + 26]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 26], _tmem_load_4[_lp*2+1 + 26]));
#endif
                                        _packed_1[_lp + 13] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 28], _tmem_load_3[_lp*2+1 + 28]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 28], _tmem_load_4[_lp*2+1 + 28]));
#endif
                                        _packed_1[_lp + 14] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
#if __CUDA_ARCH__ == 1030
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 30], _tmem_load_3[_lp*2+1 + 30]));
#else
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 30], _tmem_load_4[_lp*2+1 + 30]));
#endif
                                        _packed_1[_lp + 15] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(O_ptr + o_elem)[0] = reinterpret_cast<int4*>(_packed_1)[0];
                                    reinterpret_cast<int4*>(O_ptr + (o_elem + 8))[0] = reinterpret_cast<int4*>(_packed_1 + (4))[0];
                                    reinterpret_cast<int4*>(O_ptr + (o_elem + 16))[0] = reinterpret_cast<int4*>(_packed_1 + (8))[0];
                                    reinterpret_cast<int4*>(O_ptr + (o_elem + 24))[0] = reinterpret_cast<int4*>(_packed_1 + (12))[0];
                                }
                            }
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (stage_1) * 8);
                }
                corr_seq += 2;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int total_tiles_2 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_2 = total_tiles_2 + 1;
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
                unsigned int tile_idx_2;
                {
                    mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_2);
                    _phase_work_id_full_0_2 ^= 1;
                    tile_idx_2 = work_id_slot[0];
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_2 >= total_tiles_2) {
                    break;
                }
                unsigned int m_block_2;
                unsigned int bh_2;
                {
                    unsigned int l2section_2 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_2 = tile_idx_2 / l2section_2;
                    unsigned int l2_mod_2 = tile_idx_2 % l2section_2;
                    {
                        bool tail_2 = section_2 >= (unsigned int)total_bh / (unsigned int)L2_SWIZZLE;
                        m_block_2 = ((tail_2) ? l2_mod_2 % (unsigned int)NUM_M_BLOCKS : l2_mod_2 / (unsigned int)L2_SWIZZLE);
                        bh_2 = section_2 * (unsigned int)L2_SWIZZLE + ((tail_2) ? l2_mod_2 / (unsigned int)NUM_M_BLOCKS : l2_mod_2 % (unsigned int)L2_SWIZZLE);
                    }
                    m_block_2 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_2;
                }
                int seqlen_kv_bh_2 = UNIFORM_KV_LEN;
                int seqlen_q_bh_2 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_2 = seq_lens_kv[bh_2 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                {
                    seqlen_q_bh_2 = cu_seq_lens_q[bh_2 / (unsigned int)(NUM_Q_HEADS / PACK_G) + 1] - cu_seq_lens_q[bh_2 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                unsigned int num_n_blocks_2 = (seqlen_kv_bh_2 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_3 = seqlen_kv_bh_2 - seqlen_q_bh_2;
                    unsigned int max_n_2 = (m_block_2 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_3;
                    if (max_n_2 < (unsigned int)seqlen_kv_bh_2) {
                        num_n_blocks_2 = (max_n_2 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                unsigned int _max_0_2 = ((num_n_blocks_2) > (1) ? (num_n_blocks_2) : (1));
                num_n_blocks_2 = _max_0_2;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                _phase_q_full_1 ^= 1;
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 1018U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                }
                elect_commit(s_full_addr);
                int _mma_a_lo_1 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 1018U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                }
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 4) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                unsigned int first_pv = 1;
                #pragma unroll 1
                for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_2 - 1; n_iter_3++) {
                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                    unsigned int v_stage = mma_kv_stage;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 4) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    int first_pv_flag = first_pv;
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_2), "r"(tmem_softmax_0), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_softmax_0), "r"(1) : "memory");
                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int k_stage = mma_kv_stage;
                    int _mma_a_lo_2 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 1018U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136316048, 1);
                        }
                    }
                    elect_commit(s_full_addr);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_softmax_1), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_6), "r"(tmem_softmax_1), "r"(1) : "memory");
                    elect_commit(kv_empty_addr + (((mma_kv_stage + 3 >= 4) ? mma_kv_stage + 3 - 4 : mma_kv_stage + 3)) * 8);
                    int _mma_a_lo_3 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 1018U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_scores_1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 136316048, 1);
                        }
                    }
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 4) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    first_pv = 0;
                }
                elect_commit(q_empty_addr);
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                int first_pv_flag_1 = first_pv;
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_softmax_0), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                _phase_p_full_2_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_9 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_9), "r"(tmem_softmax_0), "r"(1) : "memory");
                elect_commit(o_full_addr);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_softmax_1), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                _phase_p_full_2_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 48;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_11), "r"(tmem_softmax_1), "r"(1) : "memory");
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 4) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                elect_commit(o_full_addr + 8);
            }
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 13) {
        { // page_offsets_main
            unsigned int total_tiles_3 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_3 = total_tiles_3 + 1;
            unsigned int po_stage = 0;
            unsigned int _phase_work_id_full_0_3 = 0;
            unsigned int _phase_page_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_3 = 0; tile_iter_3 < max_rounds_3; tile_iter_3++) {
                unsigned int tile_idx_3;
                {
                    mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_3);
                    _phase_work_id_full_0_3 ^= 1;
                    tile_idx_3 = work_id_slot[0];
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_3 >= total_tiles_3) {
                    break;
                }
                unsigned int m_block_3;
                unsigned int bh_3;
                {
                    unsigned int l2section_3 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_3 = tile_idx_3 / l2section_3;
                    unsigned int l2_mod_3 = tile_idx_3 % l2section_3;
                    {
                        bool tail_3 = section_3 >= (unsigned int)total_bh / (unsigned int)L2_SWIZZLE;
                        m_block_3 = ((tail_3) ? l2_mod_3 % (unsigned int)NUM_M_BLOCKS : l2_mod_3 / (unsigned int)L2_SWIZZLE);
                        bh_3 = section_3 * (unsigned int)L2_SWIZZLE + ((tail_3) ? l2_mod_3 / (unsigned int)NUM_M_BLOCKS : l2_mod_3 % (unsigned int)L2_SWIZZLE);
                    }
                    m_block_3 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_3;
                }
                int seqlen_kv_bh_3 = UNIFORM_KV_LEN;
                int seqlen_q_bh_3 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_3 = seq_lens_kv[bh_3 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                {
                    seqlen_q_bh_3 = cu_seq_lens_q[bh_3 / (unsigned int)(NUM_Q_HEADS / PACK_G) + 1] - cu_seq_lens_q[bh_3 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                unsigned int num_n_blocks_3 = (seqlen_kv_bh_3 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_4 = seqlen_kv_bh_3 - seqlen_q_bh_3;
                    unsigned int max_n_3 = (m_block_3 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_4;
                    if (max_n_3 < (unsigned int)seqlen_kv_bh_3) {
                        num_n_blocks_3 = (max_n_3 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                unsigned int _max_0_3 = ((num_n_blocks_3) > (1) ? (num_n_blocks_3) : (1));
                num_n_blocks_3 = _max_0_3;
                unsigned int b_idx = bh_3 / (unsigned int)(NUM_Q_HEADS / PACK_G);
                int pt_base = b_idx * (unsigned int)page_row_stride;
                int last_slot = (seqlen_kv_bh_3 - 1) / PAGE_SIZE;
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_3; ni++) {
                    unsigned int n = num_n_blocks_3 - 1 - ni;
                    int base_tok = n * (unsigned int)BLOCK_N;
                    mbarrier_wait(page_empty_addr + (po_stage) * 8, _phase_page_empty);
                    if (elect_sync()) {
                        int pg_idx = po_stage * 16;
                        if (seqlen_kv_bh_3 == 0) {
                            #pragma unroll
                            for (int chunk = 0; chunk < BLOCK_N / KV_TOK_CHUNK; chunk++) {
                                smem_pages[pg_idx + chunk] = 1073741824;
                                smem_pages[pg_idx + 8 + chunk] = 1073741824;
                            }
                        } else {
                            #pragma unroll
                            for (int chunk_1 = 0; chunk_1 < BLOCK_N / KV_TOK_CHUNK; chunk_1++) {
                                int slot = (((base_tok + chunk_1 * KV_TOK_CHUNK) / PAGE_SIZE) < (last_slot) ? ((base_tok + chunk_1 * KV_TOK_CHUNK) / PAGE_SIZE) : (last_slot));
                                smem_pages[pg_idx + chunk_1] = page_table_k[pt_base + slot];
                                smem_pages[pg_idx + 8 + chunk_1] = page_table_v[pt_base + slot];
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
            unsigned int _phase_work_id_empty_0 = 1;
            {
                unsigned int total_tiles_4 = NUM_M_BLOCKS * total_bh;
                unsigned int max_rounds_4 = total_tiles_4 + 1;
                bool static_first = (unsigned int)num_ctas <= total_tiles_4;
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
                        work_id_slot[0] = _atomic_old_0;
                        mbarrier_arrive(work_id_full_addr);
                        if (_atomic_old_0 >= total_tiles_4) {
                            unsigned int last_fetch = total_tiles_4 + (unsigned int)num_ctas - 1;
                            if (_atomic_old_0 == last_fetch) {
                                unsigned int counter_draws = last_fetch + 1;
                                if (static_first) {
                                    counter_draws = total_tiles_4;
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
    }
    // ---- Role: load ----
    if (warp == 15) {
        { // load_main
            unsigned int total_tiles_5 = NUM_M_BLOCKS * total_bh;
            unsigned int max_rounds_5 = total_tiles_5 + 1;
            unsigned int load_kv_stage = 0;
            unsigned int load_po_stage = 0;
            unsigned int _phase_work_id_full_0_4 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_iter_5 = 0; tile_iter_5 < max_rounds_5; tile_iter_5++) {
                unsigned int tile_idx_4;
                {
                    mbarrier_wait(work_id_full_addr, _phase_work_id_full_0_4);
                    _phase_work_id_full_0_4 ^= 1;
                    tile_idx_4 = work_id_slot[0];
                    mbarrier_arrive(work_id_empty_addr);
                }
                if (tile_idx_4 >= total_tiles_5) {
                    break;
                }
                unsigned int m_block_4;
                unsigned int bh_4;
                {
                    unsigned int l2section_4 = L2_SWIZZLE * NUM_M_BLOCKS;
                    unsigned int section_4 = tile_idx_4 / l2section_4;
                    unsigned int l2_mod_4 = tile_idx_4 % l2section_4;
                    {
                        bool tail_4 = section_4 >= (unsigned int)total_bh / (unsigned int)L2_SWIZZLE;
                        m_block_4 = ((tail_4) ? l2_mod_4 % (unsigned int)NUM_M_BLOCKS : l2_mod_4 / (unsigned int)L2_SWIZZLE);
                        bh_4 = section_4 * (unsigned int)L2_SWIZZLE + ((tail_4) ? l2_mod_4 / (unsigned int)NUM_M_BLOCKS : l2_mod_4 % (unsigned int)L2_SWIZZLE);
                    }
                    m_block_4 = (unsigned int)(NUM_M_BLOCKS - 1) - m_block_4;
                }
                int seqlen_kv_bh_4 = UNIFORM_KV_LEN;
                int seqlen_q_bh_4 = UNIFORM_Q_LEN;
                {
                    seqlen_kv_bh_4 = seq_lens_kv[bh_4 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                {
                    seqlen_q_bh_4 = cu_seq_lens_q[bh_4 / (unsigned int)(NUM_Q_HEADS / PACK_G) + 1] - cu_seq_lens_q[bh_4 / (unsigned int)(NUM_Q_HEADS / PACK_G)];
                }
                unsigned int num_n_blocks_4 = (seqlen_kv_bh_4 + BLOCK_N - 1) / BLOCK_N;
                {
                    int kv_shift_5 = seqlen_kv_bh_4 - seqlen_q_bh_4;
                    unsigned int max_n_4 = (m_block_4 * 2 + 2) * (unsigned int)TOK_PER_STAGE + (unsigned int)kv_shift_5;
                    if (max_n_4 < (unsigned int)seqlen_kv_bh_4) {
                        num_n_blocks_4 = (max_n_4 + (unsigned int)BLOCK_N - 1) / (unsigned int)BLOCK_N;
                    }
                }
                unsigned int _max_0_4 = ((num_n_blocks_4) > (1) ? (num_n_blocks_4) : (1));
                num_n_blocks_4 = _max_0_4;
                int unit_1 = bh_4 % (unsigned int)(NUM_Q_HEADS / PACK_G);
                int kv_head = unit_1 * PACK_G / HEADS_PER_GROUP;
                int tok_q0 = (unsigned int)cu_seq_lens_q[bh_4 / (unsigned int)(NUM_Q_HEADS / PACK_G)] + m_block_4 * 2 * (unsigned int)TOK_PER_STAGE;
                int tok_q1 = tok_q0 + TOK_PER_STAGE;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, TOK_PER_STAGE * PACK_G * 256);
                    tma_5d_gmem2smem(smem_q0_addr, (&Q), 0, 0, tok_q0, 0, unit_1, q_full_addr);
                    tma_5d_gmem2smem(smem_q0_addr + 16384, (&Q), 0, 0, tok_q0, 1, unit_1, q_full_addr);
                }
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + 8, TOK_PER_STAGE * PACK_G * 256);
                    tma_5d_gmem2smem(smem_q1_addr, (&Q), 0, 0, tok_q1, 0, unit_1, q_full_addr + 8);
                    tma_5d_gmem2smem(smem_q1_addr + 16384, (&Q), 0, 0, tok_q1, 1, unit_1, q_full_addr + 8);
                }
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_4; ni_1++) {
                    unsigned int n_1 = num_n_blocks_4 - 1 - ni_1;
                    int base_tok_1 = n_1 * (unsigned int)BLOCK_N;
                    mbarrier_wait(page_full_addr + (load_po_stage) * 8, _phase_page_full);
                    int pg_idx_1 = load_po_stage * 16;
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 32768);
                        #pragma unroll
                        for (int chunk_2 = 0; chunk_2 < BLOCK_N / KV_TOK_CHUNK; chunk_2++) {
                            int tok_off = (base_tok_1 + chunk_2 * KV_TOK_CHUNK) % PAGE_SIZE;
                            const int chunk_bytes = KV_TOK_CHUNK * 64 * 2;
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(chunk_2 * chunk_bytes), (&K), 0, tok_off, 0, kv_head, smem_pages[pg_idx_1 + chunk_2], kv_full_addr + (load_kv_stage) * 8);
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(16384 + chunk_2 * chunk_bytes), (&K), 0, tok_off, 1, kv_head, smem_pages[pg_idx_1 + chunk_2], kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 4) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 32768);
                        #pragma unroll
                        for (int chunk_3 = 0; chunk_3 < BLOCK_N / KV_TOK_CHUNK; chunk_3++) {
                            int tok_off_1 = (base_tok_1 + chunk_3 * KV_TOK_CHUNK) % PAGE_SIZE;
                            const int chunk_bytes_1 = KV_TOK_CHUNK * 64 * 2;
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(chunk_3 * chunk_bytes_1), (&V), 0, tok_off_1, 0, kv_head, smem_pages[pg_idx_1 + 8 + chunk_3], kv_full_addr + (load_kv_stage) * 8);
                            tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768 + (unsigned int)(16384 + chunk_3 * chunk_bytes_1), (&V), 0, tok_off_1, 1, kv_head, smem_pages[pg_idx_1 + 8 + chunk_3], kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 4) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
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
