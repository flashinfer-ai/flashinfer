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
#define TMEM_NCOLS 128
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_EXCH0_U32_OFF 1152
#define SMEM_SMEM_EXCH0_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_U32_STRIDE 256
#define SMEM_SMEM_EXCH1_U32_OFF 1408
#define SMEM_SMEM_EXCH1_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_U32_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 2048
#define SMEM_SMEM_QT_STRIDE 2048
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#if __CUDA_ARCH__ == 1030
#define SMEM_SMEM_P0_OFF 202752
#else
#define SMEM_SMEM_P0_OFF 169984
#endif
#define SMEM_SMEM_P0_STAGE_BYTES 2048
#define SMEM_SMEM_P0_STRIDE 2048
#if __CUDA_ARCH__ == 1030
#define SMEM_SMEM_P1_OFF 206848
#else
#define SMEM_SMEM_P1_OFF 174080
#endif
#define SMEM_SMEM_P1_STAGE_BYTES 2048
#define SMEM_SMEM_P1_STRIDE 2048
#if __CUDA_ARCH__ == 1030
#define SMEM_SMEM_PAGE_OFFSETS_OFF 208896
#else
#define SMEM_SMEM_PAGE_OFFSETS_OFF 176128
#endif
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 128
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 128
#if __CUDA_ARCH__ == 1030
#define SMEM_WORK_RESPONSE_OFF 209920
#else
#define SMEM_WORK_RESPONSE_OFF 177152
#endif
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#if __CUDA_ARCH__ == 1030
#define SMEM_TOTAL 210944
#else
#define SMEM_TOTAL 178176
#endif
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 16
#define BATCH_SIZE 256
#define NUM_Q_HEADS 32
#define NUM_KV_HEADS 4
#define Q_LEN 1
#define HAS_SINK 1
#define HAS_WINDOW 0
#define USE_SCALE_PTR 1
#define RETAIN_KV_L2 0
#if __CUDA_ARCH__ == 1030
#define NUM_KV_STAGES 6
#else
#define NUM_KV_STAGES 5
#endif
#define K_LOOKAHEAD (NUM_KV_STAGES - 2)
#define KV_SUFFIX (5120 + (NUM_KV_STAGES * 32768))
#define SMEM_BYTES (KV_SUFFIX + 8192)
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>
#include <cuda_awbarrier_primitives.h>

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


__device__ __forceinline__ void mbarrier_init_owner_lane(
    void* mbar_ptr, uint32_t count) {
    __mbarrier_init(reinterpret_cast<__mbarrier_t*>(mbar_ptr), count);
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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


__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_decode_bf16_394b0caa8e0362fe9988(const __grid_constant__ CUtensorMap Qt, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ page_table, int* __restrict__ causal_seqlens_kv_global, float* __restrict__ scale_log2_ptr, float* __restrict__ bmm2_ptr, float* __restrict__ sinks_ptr, int max_pages_per_seq, int max_local_seq_len, float softmax_scale_log2, int window_left, int num_q_heads, int num_kv_heads, int batch_size)
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
#if __CUDA_ARCH__ == 1030
    #define kv_empty_addr (mbar_base + 64)
    #define page_full_addr (mbar_base + 112)
    #define page_empty_addr (mbar_base + 176)
    #define s_full_0_addr (mbar_base + 240)
    #define s_full_1_addr (mbar_base + 248)
    #define s_empty_0_addr (mbar_base + 256)
    #define s_empty_1_addr (mbar_base + 264)
    #define p_full_0_addr (mbar_base + 272)
    #define p_full_1_addr (mbar_base + 280)
    #define corr_scale_addr (mbar_base + 288)
    #define corr_empty_0_addr (mbar_base + 304)
    #define corr_empty_1_addr (mbar_base + 312)
    #define stats_empty_addr (mbar_base + 320)
    #define o_ready_0_addr (mbar_base + 328)
    #define o_ready_1_addr (mbar_base + 336)
    #define o_empty_0_addr (mbar_base + 344)
    #define o_empty_1_addr (mbar_base + 352)
    #define tmem_dealloc_addr (mbar_base + 360)
    #define work_full_addr (mbar_base + 368)
    #define work_empty_addr (mbar_base + 384)
    #define throttle_full_addr (mbar_base + 400)
    #define throttle_empty_addr (mbar_base + 416)
#else
    #define kv_empty_addr (mbar_base + 56)
    #define page_full_addr (mbar_base + 96)
    #define page_empty_addr (mbar_base + 160)
    #define s_full_0_addr (mbar_base + 224)
    #define s_full_1_addr (mbar_base + 232)
    #define s_empty_0_addr (mbar_base + 240)
    #define s_empty_1_addr (mbar_base + 248)
    #define p_full_0_addr (mbar_base + 256)
    #define p_full_1_addr (mbar_base + 264)
    #define corr_scale_addr (mbar_base + 272)
    #define corr_empty_0_addr (mbar_base + 288)
    #define corr_empty_1_addr (mbar_base + 296)
    #define stats_empty_addr (mbar_base + 304)
    #define o_ready_0_addr (mbar_base + 312)
    #define o_ready_1_addr (mbar_base + 320)
    #define o_empty_0_addr (mbar_base + 328)
    #define o_empty_1_addr (mbar_base + 336)
    #define tmem_dealloc_addr (mbar_base + 344)
    #define work_full_addr (mbar_base + 352)
    #define work_empty_addr (mbar_base + 368)
    #define throttle_full_addr (mbar_base + 384)
    #define throttle_empty_addr (mbar_base + 400)
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr0_addr = smem + 1024;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_corr1_addr = smem + 1088;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
#if __CUDA_ARCH__ == 1030
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 202752);
    const int smem_p0_addr = smem + 202752;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int smem_p1_addr = smem + 206848;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 208896);
    const int smem_page_offsets_addr = smem + 208896;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 209920);
    const int work_response_addr = smem + 209920;
#else
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 169984);
    const int smem_p0_addr = smem + 169984;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 174080);
    const int smem_p1_addr = smem + 174080;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 176128);
    const int smem_page_offsets_addr = smem + 176128;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 177152);
    const int work_response_addr = smem + 177152;
#endif
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Qt))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

#if __CUDA_ARCH__ == 1030
    // Mbarrier init (25 pipeline groups, 0 ordered-sequence groups, 54 barriers)
    // Mbarriers at smem_raw[0..432)
#else
    // Mbarrier init (25 pipeline groups, 0 ordered-sequence groups, 52 barriers)
    // Mbarriers at smem_raw[0..416)
#endif

    if (tid == 0) {
        // q_full: 1 barriers, init_count=1
        mbarrier_init_owner_lane(smem_raw + 0, 1);
        // q_empty: 1 barriers, init_count=1
        mbarrier_init_owner_lane(smem_raw + 8, 1);
#if __CUDA_ARCH__ == 1030
        // kv_full: 6 barriers, init_count=1
#else
        // kv_full: 5 barriers, init_count=1
#endif
        mbarrier_init_owner_lane(smem_raw + 16, 1);
        mbarrier_init_owner_lane(smem_raw + 24, 1);
        mbarrier_init_owner_lane(smem_raw + 32, 1);
        mbarrier_init_owner_lane(smem_raw + 40, 1);
        mbarrier_init_owner_lane(smem_raw + 48, 1);
#if __CUDA_ARCH__ == 1030
#else
        // kv_empty: 5 barriers, init_count=1
#endif
        mbarrier_init_owner_lane(smem_raw + 56, 1);
#if __CUDA_ARCH__ == 1030
        // kv_empty: 6 barriers, init_count=1
#else
#endif
        mbarrier_init_owner_lane(smem_raw + 64, 1);
        mbarrier_init_owner_lane(smem_raw + 72, 1);
        mbarrier_init_owner_lane(smem_raw + 80, 1);
        mbarrier_init_owner_lane(smem_raw + 88, 1);
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 96, 1);
        mbarrier_init_owner_lane(smem_raw + 104, 1);
#else
#endif
        // page_full: 8 barriers, init_count=32
#if __CUDA_ARCH__ == 1030
#else
        mbarrier_init_owner_lane(smem_raw + 96, 32);
        mbarrier_init_owner_lane(smem_raw + 104, 32);
#endif
        mbarrier_init_owner_lane(smem_raw + 112, 32);
        mbarrier_init_owner_lane(smem_raw + 120, 32);
        mbarrier_init_owner_lane(smem_raw + 128, 32);
        mbarrier_init_owner_lane(smem_raw + 136, 32);
        mbarrier_init_owner_lane(smem_raw + 144, 32);
        mbarrier_init_owner_lane(smem_raw + 152, 32);
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 160, 32);
        mbarrier_init_owner_lane(smem_raw + 168, 32);
#else
#endif
        // page_empty: 8 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
#else
        mbarrier_init_owner_lane(smem_raw + 160, 1);
        mbarrier_init_owner_lane(smem_raw + 168, 1);
#endif
        mbarrier_init_owner_lane(smem_raw + 176, 1);
        mbarrier_init_owner_lane(smem_raw + 184, 1);
        mbarrier_init_owner_lane(smem_raw + 192, 1);
        mbarrier_init_owner_lane(smem_raw + 200, 1);
        mbarrier_init_owner_lane(smem_raw + 208, 1);
        mbarrier_init_owner_lane(smem_raw + 216, 1);
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 224, 1);
        mbarrier_init_owner_lane(smem_raw + 232, 1);
#else
#endif
        // s_full_0: 1 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 240, 1);
#else
        mbarrier_init_owner_lane(smem_raw + 224, 1);
#endif
        // s_full_1: 1 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 248, 1);
#else
        mbarrier_init_owner_lane(smem_raw + 232, 1);
#endif
        // s_empty_0: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 256, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 240, 128);
#endif
        // s_empty_1: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 264, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 248, 128);
#endif
        // p_full_0: 1 barriers, init_count=256
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 272, 256);
#else
        mbarrier_init_owner_lane(smem_raw + 256, 256);
#endif
        // p_full_1: 1 barriers, init_count=256
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 280, 256);
#else
        mbarrier_init_owner_lane(smem_raw + 264, 256);
#endif
        // corr_scale: 2 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 288, 128);
        mbarrier_init_owner_lane(smem_raw + 296, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 272, 128);
        mbarrier_init_owner_lane(smem_raw + 280, 128);
#endif
        // corr_empty_0: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 304, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 288, 128);
#endif
        // corr_empty_1: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 312, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 296, 128);
#endif
        // stats_empty: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 320, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 304, 128);
#endif
        // o_ready_0: 1 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 328, 1);
#else
        mbarrier_init_owner_lane(smem_raw + 312, 1);
#endif
        // o_ready_1: 1 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 336, 1);
#else
        mbarrier_init_owner_lane(smem_raw + 320, 1);
#endif
        // o_empty_0: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 344, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 328, 128);
#endif
        // o_empty_1: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 352, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 336, 128);
#endif
        // tmem_dealloc: 1 barriers, init_count=128
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 360, 128);
#else
        mbarrier_init_owner_lane(smem_raw + 344, 128);
#endif
        // --- pipeline 'work_pipe' ---
        // work_full: 2 barriers, init_count=1
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 368, 1);
        mbarrier_init_owner_lane(smem_raw + 376, 1);
#else
        mbarrier_init_owner_lane(smem_raw + 352, 1);
        mbarrier_init_owner_lane(smem_raw + 360, 1);
#endif
        // work_empty: 2 barriers, init_count=512
#if __CUDA_ARCH__ == 1030
        mbarrier_init_owner_lane(smem_raw + 384, 512);
        mbarrier_init_owner_lane(smem_raw + 392, 512);
#else
        mbarrier_init_owner_lane(smem_raw + 368, 512);
        mbarrier_init_owner_lane(smem_raw + 376, 512);
#endif
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 2 barriers, init_count=32
#if __CUDA_ARCH__ == 1030
#else
        mbarrier_init_owner_lane(smem_raw + 384, 32);
        mbarrier_init_owner_lane(smem_raw + 392, 32);
        // throttle_empty: 2 barriers, init_count=32
#endif
        mbarrier_init_owner_lane(smem_raw + 400, 32);
        mbarrier_init_owner_lane(smem_raw + 408, 32);
#if __CUDA_ARCH__ == 1030
        // throttle_empty: 2 barriers, init_count=32
        mbarrier_init_owner_lane(smem_raw + 416, 32);
        mbarrier_init_owner_lane(smem_raw + 424, 32);
#else
#endif
    }

    // CUTLASS owner-lane publication sequence
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
#if __CUDA_ARCH__ == 1030
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 432);
#else
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 416);
#endif
    if (warp == 0) {
#if __CUDA_ARCH__ == 1030
        int _tmem_hold = smem + 432;
#else
        int _tmem_hold = smem + 416;
#endif
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0 = taddr + 80;
    const int tmem_tmem_o1 = taddr + 88;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int instance_id = (warp - 0) / 4;
            int is_wg1 = instance_id;
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 8 : 0)) + (unsigned int)(tmem_row_base_v << 16);
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            float* my_exch_ptr = ((is_wg1 != 0) ? (smem_exch1) : (smem_exch0));
            unsigned int* my_exch_u32_ptr = ((is_wg1 != 0) ? (smem_exch1_u32) : (smem_exch0_u32));
            float* my_corr_ptr = ((is_wg1 != 0) ? (smem_corr1) : (smem_corr0));
            __nv_bfloat16* my_p_base = ((is_wg1 != 0) ? (smem_p1) : (smem_p0));
            float smx_scale = softmax_scale_log2;
            {
                smx_scale = scale_log2_ptr[0];
            }
            float sv[8];
            float sv_lo[4];
            float sv_hi[4];
            unsigned int work_stage_s = 0;
            unsigned int total_tiles_s = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int tile_idx_s = (blockIdx.z * Q_LEN + blockIdx.x) * NUM_KV_HEADS + blockIdx.y;
            int corr_signals = 0;
            int pairs_seen = 0;
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < total_tiles_s; _tile_iter_s++) {
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                const int tiles_per_batch = Q_LEN * NUM_KV_HEADS;
                int batch_idx = tile_idx_s / (unsigned int)tiles_per_batch;
                int tile_in_batch = tile_idx_s % (unsigned int)tiles_per_batch;
                int q_row_idx = tile_in_batch / NUM_KV_HEADS;
                int kv_head_idx = tile_in_batch % NUM_KV_HEADS;
                int kv_len_b = causal_seqlens_kv_global[batch_idx];
                int causal_prefix = 0;
                if (kv_len_b > Q_LEN) {
                    causal_prefix = kv_len_b - Q_LEN;
                }
                int global_q_pos = causal_prefix + q_row_idx;
                int visible_local_keys = global_q_pos + 1;
                int win_start = 0;
                int num_n_blocks_total = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks = ((num_n_blocks_total + num_n_blocks_total % 2) > (4) ? (num_n_blocks_total + num_n_blocks_total % 2) : (4));
                int split_start_block = 0;
                float row_max_pair[2];
                float row_sum_pair[2];
                row_max_pair[0] = -3.4028235e+38f;
                row_max_pair[1] = -3.4028235e+38f;
                row_sum_pair[0] = 0.0f;
                row_sum_pair[1] = 0.0f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    my_exch_u32_ptr[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + instance_id) : "memory");
                const bool _selected_wait_cond_1 = is_wg1 != 0;
                mbarrier_wait((_selected_wait_cond_1 ? (s_full_1_addr) : (s_full_0_addr)), (_selected_wait_cond_1 ? (_phase_s_full_1_0) : (_phase_s_full_0_0)));
                if (_selected_wait_cond_1) {
                    _phase_s_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                } else {
                    _phase_s_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                    : "r"(my_tmem_s_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                    : "r"(my_tmem_s_base + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    sv[c] = sv_lo[c];
                    sv[c + 4] = sv_hi[c];
                }
                #pragma unroll 1
                for (int pair = 0; pair < cta_n_blocks / 2; pair++) {
                    int my_block = split_start_block + 2 * pair + is_wg1;
                    int ldtm_row_base = warp_in_wg * 32 + lane / 4;
                    int kv_pos0 = my_block * BLOCK_N + ldtm_row_base;
                    int kv_pos1 = kv_pos0 + 8;
                    int kv_pos2 = kv_pos0 + 16;
                    int kv_pos3 = kv_pos0 + 24;
                    if (kv_pos0 >= visible_local_keys) {
                        sv[0] = -3.4028235e+38f;
                        sv[1] = -3.4028235e+38f;
                    }
                    if (kv_pos1 >= visible_local_keys) {
                        sv[2] = -3.4028235e+38f;
                        sv[3] = -3.4028235e+38f;
                    }
                    if (kv_pos2 >= visible_local_keys) {
                        sv[4] = -3.4028235e+38f;
                        sv[5] = -3.4028235e+38f;
                    }
                    if (kv_pos3 >= visible_local_keys) {
                        sv[6] = -3.4028235e+38f;
                        sv[7] = -3.4028235e+38f;
                    }
                    float pair_max[2];
                    pair_max[0] = max_noftz(max_noftz(sv[0], sv[2]), max_noftz(sv[4], sv[6]));
                    pair_max[1] = max_noftz(max_noftz(sv[1], sv[3]), max_noftz(sv[5], sv[7]));
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        pair_max[c_1] = max_noftz(pair_max[c_1], _shfl_xor_sync_2);
                        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        pair_max[c_1] = max_noftz(pair_max[c_1], _shfl_xor_sync_3);
                    }
                    float old_max_pair[2];
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        old_max_pair[c_2] = row_max_pair[c_2];
                        new_max_pair[c_2] = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                    }
                    if (lane < 8) {
                        uint32_t _amf_u_4 = __float_as_uint(new_max_pair[0]);
                        uint32_t _amf_mask_4 = -int32_t(_amf_u_4 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_4 ^ _amf_mask_4;
                        uint32_t _amf_u_5 = __float_as_uint(new_max_pair[1]);
                        uint32_t _amf_mask_5 = -int32_t(_amf_u_5 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_5 ^ _amf_mask_5;
                        atomicMax(&my_exch_u32_ptr[col_pair_base], _amf_enc_1);
                        atomicMax(&my_exch_u32_ptr[col_pair_base + 1], _amf_enc_2);
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(8 + instance_id) : "memory");
                    uint32_t _amf_u_7 = my_exch_u32_ptr[col_pair_base];
                    uint32_t _amf_mask_7 = ((_amf_u_7 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_6 = __uint_as_float(_amf_u_7 ^ _amf_mask_7);
                    new_max_pair[0] = _atomic_max_float_decode_6;
                    uint32_t _amf_u_9 = my_exch_u32_ptr[col_pair_base + 1];
                    uint32_t _amf_mask_9 = ((_amf_u_9 >> 31) - 1u) | 0x80000000u;
                    float _atomic_max_float_decode_8 = __uint_as_float(_amf_u_9 ^ _amf_mask_9);
                    new_max_pair[1] = _atomic_max_float_decode_8;
                    float acc_scale_pair[2];
                    float2 acc_delta_pair_f2 = sub_f32x2(make_float2(row_max_pair[0], row_max_pair[1]), make_float2(new_max_pair[0], new_max_pair[1]));
                    float2 _f2_0 = make_float2(smx_scale, smx_scale);
                    float2 acc_scaled_delta_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&acc_scaled_delta_pair_f2) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&acc_delta_pair_f2));
                    float _exp2_10 = approx_exp2(acc_scaled_delta_pair_f2.x);
                    acc_scale_pair[0] = ((row_max_pair[0] > -CUDART_INF_F) ? _exp2_10 : 1.0f);
                    float _exp2_11 = approx_exp2(acc_scaled_delta_pair_f2.y);
                    acc_scale_pair[1] = ((row_max_pair[1] > -CUDART_INF_F) ? _exp2_11 : 1.0f);
                    float stats_pair[4];
                    stats_pair[0] = old_max_pair[0];
                    stats_pair[1] = old_max_pair[1];
                    stats_pair[2] = new_max_pair[0];
                    stats_pair[3] = new_max_pair[1];
                    const bool _selected_wait_cond_12 = is_wg1 != 0;
                    mbarrier_wait((_selected_wait_cond_12 ? (corr_empty_1_addr) : (corr_empty_0_addr)), (_selected_wait_cond_12 ? (_phase_corr_empty_1_0) : (_phase_corr_empty_0_0)));
                    if (_selected_wait_cond_12) {
                        _phase_corr_empty_1_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        _phase_corr_empty_0_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    tmem_st_x4_f32(my_tmem_stats, stats_pair);
                    float exp_vals[8];
                    float safe_max0 = ((new_max_pair[0] == -3.4028235e+38f) ? 0.0f : new_max_pair[0]);
                    float safe_max1 = ((new_max_pair[1] == -3.4028235e+38f) ? 0.0f : new_max_pair[1]);
                    float2 _f32x2_mul_a_13 = make_float2(safe_max0, safe_max1);
                    float2 _f32x2_mul_b_14 = make_float2(-smx_scale, -smx_scale);
                    float2 neg_max_scaled_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&neg_max_scaled_pair_f2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
                    float2 _f2_1 = make_float2(sv[0], sv[1]);
                    float2 _f2_2 = make_float2(sv[2], sv[3]);
                    float2 _f2_3 = make_float2(sv[4], sv[5]);
                    float2 _f2_4 = make_float2(sv[6], sv[7]);
                    float2 affine01_pair_f2 = fma_f32x2_rn_ftz(_f2_1, _f2_0, neg_max_scaled_pair_f2);
                    float2 affine23_pair_f2 = fma_f32x2_rn_ftz(_f2_2, _f2_0, neg_max_scaled_pair_f2);
                    float2 affine45_pair_f2 = fma_f32x2_rn_ftz(_f2_3, _f2_0, neg_max_scaled_pair_f2);
                    float2 affine67_pair_f2 = fma_f32x2_rn_ftz(_f2_4, _f2_0, neg_max_scaled_pair_f2);
                    float _exp2_15 = approx_exp2(affine01_pair_f2.x);
                    exp_vals[0] = _exp2_15;
                    float _exp2_16 = approx_exp2(affine01_pair_f2.y);
                    exp_vals[1] = _exp2_16;
                    float _exp2_17 = approx_exp2(affine23_pair_f2.x);
                    exp_vals[2] = _exp2_17;
                    float _exp2_18 = approx_exp2(affine23_pair_f2.y);
                    exp_vals[3] = _exp2_18;
                    float _exp2_19 = approx_exp2(affine45_pair_f2.x);
                    exp_vals[4] = _exp2_19;
                    float _exp2_20 = approx_exp2(affine45_pair_f2.y);
                    exp_vals[5] = _exp2_20;
                    float _exp2_21 = approx_exp2(affine67_pair_f2.x);
                    exp_vals[6] = _exp2_21;
                    float _exp2_22 = approx_exp2(affine67_pair_f2.y);
                    exp_vals[7] = _exp2_22;
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 2; c_3++) {
                        row_max_pair[c_3] = new_max_pair[c_3];
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_scale_addr + (is_wg1) * 8);
                    corr_signals += 1;
                    float2 _f2_5 = make_float2(exp_vals[0], exp_vals[1]);
                    float2 _f2_6 = make_float2(exp_vals[2], exp_vals[3]);
                    float2 _f2_7 = make_float2(exp_vals[4], exp_vals[5]);
                    float2 _f2_8 = make_float2(exp_vals[6], exp_vals[7]);
                    float2 pair_sum01_f2 = add_f32x2(_f2_5, _f2_6);
                    float2 pair_sum012_f2 = add_f32x2(pair_sum01_f2, _f2_7);
                    float2 pair_sum_pre_reduce_f2 = add_f32x2(pair_sum012_f2, _f2_8);
                    float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, pair_sum_pre_reduce_f2.x, 16);
                    float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, pair_sum_pre_reduce_f2.y, 16);
                    float2 pair_sum16_f2 = add_f32x2(pair_sum_pre_reduce_f2, make_float2(_shfl_xor_sync_23, _shfl_xor_sync_24));
                    float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, pair_sum16_f2.x, 8);
                    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, pair_sum16_f2.y, 8);
                    float2 pair_sum8_f2 = add_f32x2(pair_sum16_f2, make_float2(_shfl_xor_sync_25, _shfl_xor_sync_26));
                    float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, pair_sum8_f2.x, 4);
                    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, pair_sum8_f2.y, 4);
                    float2 pair_sum_reduced_f2 = add_f32x2(pair_sum8_f2, make_float2(_shfl_xor_sync_27, _shfl_xor_sync_28));
                    float2 row_sum_next_f2 = fma_f32x2_rn_ftz(make_float2(row_sum_pair[0], row_sum_pair[1]), make_float2(acc_scale_pair[0], acc_scale_pair[1]), pair_sum_reduced_f2);
                    row_sum_pair[0] = row_sum_next_f2.x;
                    row_sum_pair[1] = row_sum_next_f2.y;
                    unsigned int regs_p[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(exp_vals[_lp*2 + 0], exp_vals[_lp*2+1 + 0]));
                        regs_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    int slice_idx = warp_in_wg / 2;
                    int warp_idx_in_slice = warp_in_wg % 2;
                    int mtx_idx = lane / 8;
                    int thr_row_idx = lane % 8;
                    int mtx_col_idx = warp_idx_in_slice * 4 + mtx_idx;
                    int seg_col_idx = mtx_col_idx ^ thr_row_idx;
                    int stsm_offset = slice_idx * 8 * 128 + thr_row_idx * 128 + seg_col_idx * 16;
                    const bool _selected_wait_cond_29 = is_wg1 != 0;
                    mbarrier_wait((_selected_wait_cond_29 ? (o_ready_1_addr) : (o_ready_0_addr)), (_selected_wait_cond_29 ? ((int)((unsigned int)pairs_seen % 2) ^ 1) : ((int)((unsigned int)pairs_seen % 2) ^ 1)));
                    pairs_seen += 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "l"(__cvta_generic_to_shared(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset)), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_1_addr);
                        mbarrier_arrive(s_empty_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
                        mbarrier_arrive(s_empty_0_addr);
                    }
                    if (pair < cta_n_blocks / 2 - 1) {
                        const bool _selected_wait_cond_30 = is_wg1 != 0;
                        mbarrier_wait((_selected_wait_cond_30 ? (s_full_1_addr) : (s_full_0_addr)), (_selected_wait_cond_30 ? (_phase_s_full_1_0) : (_phase_s_full_0_0)));
                        if (_selected_wait_cond_30) {
                            _phase_s_full_1_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        } else {
                            _phase_s_full_0_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                            : "r"(my_tmem_s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                            : "r"(my_tmem_s_base + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int c_4 = 0; c_4 < 4; c_4++) {
                            sv[c_4] = sv_lo[c_4];
                            sv[c_4 + 4] = sv_hi[c_4];
                        }
                    }
                }
                const bool _selected_wait_cond_31 = is_wg1 != 0;
                mbarrier_wait((_selected_wait_cond_31 ? (corr_empty_1_addr) : (corr_empty_0_addr)), (_selected_wait_cond_31 ? (_phase_corr_empty_1_0) : (_phase_corr_empty_0_0)));
                if (_selected_wait_cond_31) {
                    _phase_corr_empty_1_0 ^= 1;
                } else {
                    _phase_corr_empty_0_0 ^= 1;
                }
                if (lane < 4) {
                    my_exch_ptr[warp_in_wg * 8 + col_pair_base] = row_sum_pair[0];
                    my_exch_ptr[warp_in_wg * 8 + col_pair_base + 1] = row_sum_pair[1];
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + instance_id) : "memory");
                if (wg_tid < 4) {
                    my_corr_ptr[col_pair_base] = my_exch_ptr[col_pair_base] + my_exch_ptr[8 + col_pair_base] + my_exch_ptr[16 + col_pair_base] + my_exch_ptr[24 + col_pair_base];
                    my_corr_ptr[col_pair_base + 1] = my_exch_ptr[col_pair_base + 1] + my_exch_ptr[8 + col_pair_base + 1] + my_exch_ptr[16 + col_pair_base + 1] + my_exch_ptr[24 + col_pair_base + 1];
                    my_exch_ptr[col_pair_base] = row_max_pair[0];
                    my_exch_ptr[col_pair_base + 1] = row_max_pair[1];
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + instance_id) : "memory");
                mbarrier_arrive(corr_scale_addr + (is_wg1) * 8);
                corr_signals += 1;
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(work_response_addr + work_stage_s * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_12 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_12)
                    : "r"(work_response_addr + work_stage_s * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_13 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_13)
                    : "r"(work_response_addr + work_stage_s * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage_s * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 2) { work_stage_s = 0; _phase_work_full ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                tile_idx_s = (_clc_ctaid_14 * (unsigned int)Q_LEN + _clc_ctaid_12) * (unsigned int)NUM_KV_HEADS + _clc_ctaid_13;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int warp_in_wg_c = warp % 4;
            const int wg_tid_c = warp_in_wg_c * 32 + lane;
            int d_idx = warp % 4 * 32 + lane;
            const int group_ratio_rt = NUM_Q_HEADS / NUM_KV_HEADS;
            unsigned int work_stage_c = 0;
            unsigned int total_tiles_c = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            float smx_scale_c = softmax_scale_log2;
            {
                smx_scale_c = scale_log2_ptr[0];
            }
            float bmm2_c = 1.0f;
            bmm2_c = bmm2_ptr[0];
            unsigned int tile_idx_c = (blockIdx.z * Q_LEN + blockIdx.x) * NUM_KV_HEADS + blockIdx.y;
            unsigned int _phase_corr_scale_0 = 0;
            unsigned int _phase_corr_scale_1 = 0;
            unsigned int _phase_o_ready_0_0 = 0;
            unsigned int _phase_o_ready_1_0 = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < total_tiles_c; _tile_iter_c++) {
                const int tiles_per_batch_1 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_1 = tile_idx_c / (unsigned int)tiles_per_batch_1;
                int tile_in_batch_1 = tile_idx_c % (unsigned int)tiles_per_batch_1;
                int q_row_idx_1 = tile_in_batch_1 / NUM_KV_HEADS;
                int kv_head_idx_1 = tile_in_batch_1 % NUM_KV_HEADS;
                int kv_len_b_1 = causal_seqlens_kv_global[batch_idx_1];
                int causal_prefix_1 = 0;
                if (kv_len_b_1 > Q_LEN) {
                    causal_prefix_1 = kv_len_b_1 - Q_LEN;
                }
                int global_q_pos_1 = causal_prefix_1 + q_row_idx_1;
                int visible_local_keys_1 = global_q_pos_1 + 1;
                int win_start_1 = 0;
                int num_n_blocks_total_1 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_1 = ((num_n_blocks_total_1 + num_n_blocks_total_1 % 2) > (4) ? (num_n_blocks_total_1 + num_n_blocks_total_1 % 2) : (4));
                if (cta_n_blocks_1 / 2 > 0) {
                    mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                    _phase_corr_scale_0 ^= 1;
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                    _phase_corr_scale_1 ^= 1;
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < cta_n_blocks_1 / 2; pair_1++) {
                    mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                    _phase_corr_scale_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float acc0_pair[2];
                    float2 max_diff0_pair_f2 = sub_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                    float2 _f2_9 = make_float2(smx_scale_c, smx_scale_c);
                    float2 scaled_diff0_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_diff0_pair_f2) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&max_diff0_pair_f2));
                    float _exp2_0 = approx_exp2(scaled_diff0_pair_f2.x);
                    acc0_pair[0] = ((max_diff0_pair_f2.x != 0.0f) ? _exp2_0 : 1.0f);
                    float _exp2_1 = approx_exp2(scaled_diff0_pair_f2.y);
                    acc0_pair[1] = ((max_diff0_pair_f2.y != 0.0f) ? _exp2_1 : 1.0f);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                    _phase_o_ready_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_0 = ((acc0_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_0 = rescale_pred_0 | ((acc0_pair[1] != 1.0f) ? 1 : 0);
                    int _warp_vote_2 = __any_sync(0xFFFFFFFF, rescale_pred_0 != 0);
                    if (_warp_vote_2 != 0) {
                        float o0_lo[4];
                        float o0_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[3]))
                            : "r"(taddr + 80 + (unsigned int)corr_row));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[3]))
                            : "r"(taddr + 80 + (unsigned int)corr_row + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _f2_10 = make_float2(acc0_pair[0], acc0_pair[1]);
                        float2 _f2_11 = make_float2(o0_lo[0], o0_lo[1]);
                        float2 _f2_12 = make_float2(o0_lo[2], o0_lo[3]);
                        float2 _f2_13 = make_float2(o0_hi[0], o0_hi[1]);
                        float2 _f2_14 = make_float2(o0_hi[2], o0_hi[3]);
                        float2 o0_lo01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_lo01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_10));
                        float2 o0_lo23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_lo23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_10));
                        float2 o0_hi01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_hi01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&_f2_10));
                        float2 o0_hi23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_hi23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_10));
                        o0_lo[0] = o0_lo01_scaled_f2.x;
                        o0_lo[1] = o0_lo01_scaled_f2.y;
                        o0_lo[2] = o0_lo23_scaled_f2.x;
                        o0_lo[3] = o0_lo23_scaled_f2.y;
                        o0_hi[0] = o0_hi01_scaled_f2.x;
                        o0_hi[1] = o0_hi01_scaled_f2.y;
                        o0_hi[2] = o0_hi23_scaled_f2.x;
                        o0_hi[3] = o0_hi23_scaled_f2.y;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + (unsigned int)corr_row + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                    _phase_corr_scale_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float acc1_pair[2];
                    float2 max_diff1_pair_f2 = sub_f32x2(make_float2(_tmem_load_1[0], _tmem_load_1[1]), make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                    float2 scaled_diff1_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_diff1_pair_f2) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&max_diff1_pair_f2));
                    float _exp2_3 = approx_exp2(scaled_diff1_pair_f2.x);
                    acc1_pair[0] = ((max_diff1_pair_f2.x != 0.0f) ? _exp2_3 : 1.0f);
                    float _exp2_4 = approx_exp2(scaled_diff1_pair_f2.y);
                    acc1_pair[1] = ((max_diff1_pair_f2.y != 0.0f) ? _exp2_4 : 1.0f);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                    _phase_o_ready_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_1 = ((acc1_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_1 = rescale_pred_1 | ((acc1_pair[1] != 1.0f) ? 1 : 0);
                    int _warp_vote_5 = __any_sync(0xFFFFFFFF, rescale_pred_1 != 0);
                    if (_warp_vote_5 != 0) {
                        float o1_lo[4];
                        float o1_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[3]))
                            : "r"(taddr + 88 + (unsigned int)corr_row));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[3]))
                            : "r"(taddr + 88 + (unsigned int)corr_row + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float2 _f2_15 = make_float2(acc1_pair[0], acc1_pair[1]);
                        float2 _f2_16 = make_float2(o1_lo[0], o1_lo[1]);
                        float2 _f2_17 = make_float2(o1_lo[2], o1_lo[3]);
                        float2 _f2_18 = make_float2(o1_hi[0], o1_hi[1]);
                        float2 _f2_19 = make_float2(o1_hi[2], o1_hi[3]);
                        float2 o1_lo01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_lo01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_15));
                        float2 o1_lo23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_lo23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&_f2_15));
                        float2 o1_hi01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_hi01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_15));
                        float2 o1_hi23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_hi23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&_f2_15));
                        o1_lo[0] = o1_lo01_scaled_f2.x;
                        o1_lo[1] = o1_lo01_scaled_f2.y;
                        o1_lo[2] = o1_lo23_scaled_f2.x;
                        o1_lo[3] = o1_lo23_scaled_f2.y;
                        o1_hi[0] = o1_hi01_scaled_f2.x;
                        o1_hi[1] = o1_hi01_scaled_f2.y;
                        o1_hi[2] = o1_hi23_scaled_f2.x;
                        o1_hi[3] = o1_hi23_scaled_f2.y;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + (unsigned int)corr_row), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + (unsigned int)corr_row + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                float sink_b2_lane = 0.0f;
                if (group_ratio_rt > lane) {
                    int sink_head_lane = kv_head_idx_1 * group_ratio_rt + lane;
                    sink_b2_lane = sinks_ptr[sink_head_lane] * 1.442695f;
                }
                mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                _phase_corr_scale_0 ^= 1;
                mbarrier_wait(corr_scale_addr + 8, _phase_corr_scale_1);
                _phase_corr_scale_1 ^= 1;
                float scale0[8];
                float scale1[8];
                float local_sum[8];
                float local_max[8];
                #pragma unroll
                for (int c_5 = 0; c_5 < 8; c_5++) {
                    float m0 = smem_exch0[c_5];
                    float m1 = smem_exch1[c_5];
                    float s0 = smem_corr0[c_5];
                    float s1 = smem_corr1[c_5];
                    float fm;
                    asm("max.f32 %0, %1, %2;" : "=f"(fm) : "f"(m0), "f"(m1));
                    local_max[c_5] = fm;
                    float d0 = smx_scale_c * (m0 - fm);
                    float d1 = smx_scale_c * (m1 - fm);
                    float _exp2_6 = approx_exp2(d0);
                    scale0[c_5] = ((m0 == -CUDART_INF_F) ? 0.0f : _exp2_6);
                    float _exp2_7 = approx_exp2(d1);
                    scale1[c_5] = ((m1 == -CUDART_INF_F) ? 0.0f : _exp2_7);
                    local_sum[c_5] = __fmaf_rn(s0, scale0[c_5], s1 * scale1[c_5]);
                }
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                _phase_o_ready_0_0 ^= 1;
                mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                _phase_o_ready_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], taddr + 80 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], taddr + 88 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float merged_pair_vals[8];
                #pragma unroll
                for (int hp = 0; hp < 4; hp++) {
                    const int h0 = hp * 2;
                    const int h1 = h0 + 1;
                    float2 _f2_20 = make_float2(_tmem_load_2[h0], _tmem_load_2[h1]);
                    float2 _f2_21 = make_float2(_tmem_load_3[h0], _tmem_load_3[h1]);
                    float2 _f2_22 = make_float2(scale0[h0], scale0[h1]);
                    float2 _f32x2_mul_b_8 = make_float2(scale1[h0], scale1[h1]);
                    float2 o1_scaled_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_scaled_pair_f2) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                    float2 merged_pair_f2 = fma_f32x2_rn_ftz(_f2_20, _f2_22, o1_scaled_pair_f2);
                    merged_pair_vals[h0] = merged_pair_f2.x;
                    merged_pair_vals[h1] = merged_pair_f2.y;
                }
                #pragma unroll
                for (int h = 0; h < 8; h++) {
                    float final_o = 0.0f;
                    float local_lse = -CUDART_INF_F;
                    {
                        float merged_n = 0.0f;
                        if (local_sum[h] > 0.0f && local_sum[h] == local_sum[h]) {
                            merged_n = merged_pair_vals[h];
                            {
                                float _log2_9;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_9) : "f"(local_sum[h]));
                                local_lse = __fmaf_rn(local_max[h], smx_scale_c, _log2_9);
                            }
                        }
                        if (group_ratio_rt > h) {
                            float sink_b2 = __shfl_sync(0xFFFFFFFF, sink_b2_lane, h);
                            float m2 = local_max[h] * smx_scale_c;
                            float big2;
                            asm("max.f32 %0, %1, %2;" : "=f"(big2) : "f"(m2), "f"(sink_b2));
                            float num_scale = 0.0f;
                            if (local_sum[h] > 0.0f && local_sum[h] == local_sum[h]) {
                                float _exp2_10_1 = approx_exp2(m2 - big2);
                                num_scale = _exp2_10_1;
                            }
                            float _exp2_11_1 = approx_exp2(sink_b2 - big2);
                            float denom = __fmaf_rn(local_sum[h], num_scale, _exp2_11_1);
                            float _rcp_12 = approx_rcp(denom);
                            final_o = merged_n * num_scale * _rcp_12;
                        }
                    }
                    if (group_ratio_rt > h) {
                        int q_head = kv_head_idx_1 * group_ratio_rt + h;
                        int o_idx = ((batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head) * HEAD_DIM + d_idx;
                        final_o = mul_f32_rn_noftz(final_o, bmm2_c);
                        *(reinterpret_cast<__nv_bfloat16*>(O_ptr + o_idx) + (0)) = (__nv_bfloat16)final_o;
                        if (d_idx == 0) {
                            int lse_idx = (batch_idx_1 * Q_LEN + q_row_idx_1) * NUM_Q_HEADS + q_head;
                            *(reinterpret_cast<float*>(LSE_ptr + lse_idx) + (0)) = local_lse;
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_empty_0_addr);
                mbarrier_arrive(o_empty_1_addr);
                mbarrier_arrive(stats_empty_addr);
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                uint32_t _clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_5)
                    : "r"(work_response_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_15)
                    : "r"(work_response_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_16 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_16)
                    : "r"(work_response_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_17 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_17)
                    : "r"(work_response_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 2) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
                tile_idx_c = (_clc_ctaid_17 * (unsigned int)Q_LEN + _clc_ctaid_15) * (unsigned int)NUM_KV_HEADS + _clc_ctaid_16;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            const int tmem_s0v = taddr;
            const int tmem_s1v = taddr + 8;
            const int tmem_o0v = taddr + 80;
            const int tmem_o1v = taddr + 88;
            unsigned int work_stage_m = 0;
            unsigned int total_tiles_m = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int tile_idx_m = (blockIdx.z * Q_LEN + blockIdx.x) * NUM_KV_HEADS + blockIdx.y;
            unsigned int _phase_s_empty_0_0 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_s_empty_1_0 = 1;
            unsigned int _phase_o_empty_0_0 = 1;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_o_empty_1_0 = 1;
            unsigned int _phase_p_full_1_0 = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < total_tiles_m; _tile_iter_m++) {
                const int tiles_per_batch_2 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_2 = tile_idx_m / (unsigned int)tiles_per_batch_2;
                int tile_in_batch_2 = tile_idx_m % (unsigned int)tiles_per_batch_2;
                int q_row_idx_2 = tile_in_batch_2 / NUM_KV_HEADS;
                int kv_head_idx_2 = tile_in_batch_2 % NUM_KV_HEADS;
                int kv_len_b_2 = causal_seqlens_kv_global[batch_idx_2];
                int causal_prefix_2 = 0;
                if (kv_len_b_2 > Q_LEN) {
                    causal_prefix_2 = kv_len_b_2 - Q_LEN;
                }
                int global_q_pos_2 = causal_prefix_2 + q_row_idx_2;
                int visible_local_keys_2 = global_q_pos_2 + 1;
                int win_start_2 = 0;
                int num_n_blocks_total_2 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_2 = ((num_n_blocks_total_2 + num_n_blocks_total_2 % 2) > (4) ? (num_n_blocks_total_2 + num_n_blocks_total_2 % 2) : (4));
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                }
                elect_commit(s_full_0_addr);
                elect_commit(kv_empty_addr);
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + 8, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 2048);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                }
                elect_commit(s_full_1_addr);
                elect_commit(kv_empty_addr + 8);
                int kv_pos = 0;
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < cta_n_blocks_2 / 2 - 1; pair_2++) {
                    int s0_1 = 2 * pair_2 % NUM_KV_STAGES;
                    int s1_1 = (2 * pair_2 + 1) % NUM_KV_STAGES;
                    int s0_next = (2 * pair_2 + 2) % NUM_KV_STAGES;
                    int s1_next = (2 * pair_2 + 3) % NUM_KV_STAGES;
                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                    _phase_s_empty_0_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_full = (int)((unsigned int)(2 * pair_2 + 2) % 6);
#else
                    int _stage_kv_full = (int)((unsigned int)(2 * pair_2 + 2) % 5);
#endif
                    mbarrier_wait(kv_full_addr + (_stage_kv_full) * 8, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                    }
                    elect_commit(s_full_0_addr);
#if __CUDA_ARCH__ == 1030
                    elect_commit(kv_empty_addr + ((int)((unsigned int)(2 * pair_2 + 2) % 6)) * 8);
#else
                    elect_commit(kv_empty_addr + ((int)((unsigned int)(2 * pair_2 + 2) % 5)) * 8);
#endif
                    mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                    _phase_o_empty_0_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_full_1 = (int)((unsigned int)(2 * pair_2) % 6);
#else
                    int _stage_kv_full_1 = (int)((unsigned int)(2 * pair_2) % 5);
#endif
                    mbarrier_wait(kv_full_addr + (_stage_kv_full_1) * 8, 1);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_1) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, ((first_pv0) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                    }
#if __CUDA_ARCH__ == 1030
                    elect_commit2(kv_empty_addr + ((int)((unsigned int)(2 * pair_2) % 6)) * 8, o_ready_0_addr);
#else
                    elect_commit2(kv_empty_addr + ((int)((unsigned int)(2 * pair_2) % 5)) * 8, o_ready_0_addr);
#endif
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_full_2 = (int)((unsigned int)(2 * pair_2 + 3) % 6);
#else
                    int _stage_kv_full_2 = (int)((unsigned int)(2 * pair_2 + 3) % 5);
#endif
                    mbarrier_wait(kv_full_addr + (_stage_kv_full_2) * 8, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_next) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                    }
                    elect_commit(s_full_1_addr);
#if __CUDA_ARCH__ == 1030
                    elect_commit(kv_empty_addr + ((int)((unsigned int)(2 * pair_2 + 3) % 6)) * 8);
#else
                    elect_commit(kv_empty_addr + ((int)((unsigned int)(2 * pair_2 + 3) % 5)) * 8);
#endif
                    mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                    _phase_o_empty_1_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_full_3 = (int)((unsigned int)(2 * pair_2 + 1) % 6);
#else
                    int _stage_kv_full_3 = (int)((unsigned int)(2 * pair_2 + 1) % 5);
#endif
                    mbarrier_wait(kv_full_addr + (_stage_kv_full_3) * 8, 1);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_1) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, ((first_pv1) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                    }
#if __CUDA_ARCH__ == 1030
                    elect_commit2(kv_empty_addr + ((int)((unsigned int)(2 * pair_2 + 1) % 6)) * 8, o_ready_1_addr);
#else
                    elect_commit2(kv_empty_addr + ((int)((unsigned int)(2 * pair_2 + 1) % 5)) * 8, o_ready_1_addr);
#endif
                    first_pv0 = 0;
                    first_pv1 = 0;
                    kv_pos += 2;
                }
                elect_commit(q_empty_addr);
                int last = cta_n_blocks_2 / 2 - 1;
                int s0_last = kv_pos % NUM_KV_STAGES;
                int s1_last = (kv_pos + 1) % NUM_KV_STAGES;
                mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                _phase_o_empty_0_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                int _stage_kv_full_4 = (int)((unsigned int)kv_pos % 6);
#else
                int _stage_kv_full_4 = (int)((unsigned int)kv_pos % 5);
#endif
                mbarrier_wait(kv_full_addr + (_stage_kv_full_4) * 8, 1);
                mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                _phase_p_full_0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
                int _mma_b_lo_4 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                    uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, ((first_pv0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                }
#if __CUDA_ARCH__ == 1030
                elect_commit2(kv_empty_addr + ((int)((unsigned int)kv_pos % 6)) * 8, o_ready_0_addr);
#else
                elect_commit2(kv_empty_addr + ((int)((unsigned int)kv_pos % 5)) * 8, o_ready_0_addr);
#endif
                mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                _phase_o_empty_1_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                int _stage_kv_full_5 = (int)((unsigned int)(kv_pos + 1) % 6);
#else
                int _stage_kv_full_5 = (int)((unsigned int)(kv_pos + 1) % 5);
#endif
                mbarrier_wait(kv_full_addr + (_stage_kv_full_5) * 8, 1);
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
                int _mma_b_lo_5 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                    uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, ((first_pv1) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                }
#if __CUDA_ARCH__ == 1030
                elect_commit2(kv_empty_addr + ((int)((unsigned int)(kv_pos + 1) % 6)) * 8, o_ready_1_addr);
#else
                elect_commit2(kv_empty_addr + ((int)((unsigned int)(kv_pos + 1) % 5)) * 8, o_ready_1_addr);
#endif
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_9)
                    : "r"(work_response_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 2) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                tile_idx_m = (_clc_ctaid_11 * (unsigned int)Q_LEN + _clc_ctaid_9) * (unsigned int)NUM_KV_HEADS + _clc_ctaid_10;
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        }
    }
    // ---- Role: load_pgoff ----
    if (warp == 13) {
        { // load_pgoff_main
            int page_slot_p = 0;
            unsigned int work_stage_p = 0;
            unsigned int total_tiles_p = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int tile_idx_p = (blockIdx.z * Q_LEN + blockIdx.x) * NUM_KV_HEADS + blockIdx.y;
            unsigned int _phase_page_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < total_tiles_p; _tile_iter_p++) {
                const int tiles_per_batch_3 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_3 = tile_idx_p / (unsigned int)tiles_per_batch_3;
                int tile_in_batch_3 = tile_idx_p % (unsigned int)tiles_per_batch_3;
                int q_row_idx_3 = tile_in_batch_3 / NUM_KV_HEADS;
                int kv_head_idx_3 = tile_in_batch_3 % NUM_KV_HEADS;
                int kv_len_b_3 = causal_seqlens_kv_global[batch_idx_3];
                int causal_prefix_3 = 0;
                if (kv_len_b_3 > Q_LEN) {
                    causal_prefix_3 = kv_len_b_3 - Q_LEN;
                }
                int global_q_pos_3 = causal_prefix_3 + q_row_idx_3;
                int visible_local_keys_3 = global_q_pos_3 + 1;
                int win_start_3 = 0;
                int pt_base_p = batch_idx_3 * max_pages_per_seq;
                int num_n_blocks_total_3 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_3 = ((num_n_blocks_total_3 + num_n_blocks_total_3 % 2) > (4) ? (num_n_blocks_total_3 + num_n_blocks_total_3 % 2) : (4));
                int last_block_p = cta_n_blocks_3 - 1;
                int n_groups_p = (cta_n_blocks_3 + 3) / 4;
                int pair_blocks_end_p = n_groups_p / 2 * 8;
                #pragma unroll 1
                for (int group_pair_base_p = 0; group_pair_base_p < pair_blocks_end_p; group_pair_base_p += 8) {
                    int group_block_p = lane / 8;
                    int page_i_p = lane % 8;
                    int n_block_p0 = group_pair_base_p + group_block_p;
                    if (n_block_p0 > last_block_p) {
                        n_block_p0 = last_block_p;
                    }
                    mbarrier_wait(page_empty_addr + (page_slot_p) * 8, _phase_page_empty);
                    int page_stage_addr_p0 = smem_page_offsets_addr + (unsigned int)(page_slot_p * 128);
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(page_stage_addr_p0 + lane * 4), "l"(page_table + (pt_base_p + n_block_p0 * 8 + page_i_p)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(page_full_addr + (page_slot_p) * 8) : "memory");
                    mbarrier_arrive(page_full_addr + (page_slot_p) * 8);
                    page_slot_p += 1;
                    if (page_slot_p == 8) { page_slot_p = 0; _phase_page_empty ^= 1; }
                    int n_block_p1 = group_pair_base_p + 4 + group_block_p;
                    if (n_block_p1 > last_block_p) {
                        n_block_p1 = last_block_p;
                    }
                    mbarrier_wait(page_empty_addr + (page_slot_p) * 8, _phase_page_empty);
                    int page_stage_addr_p1 = smem_page_offsets_addr + (unsigned int)(page_slot_p * 128);
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(page_stage_addr_p1 + lane * 4), "l"(page_table + (pt_base_p + n_block_p1 * 8 + page_i_p)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(page_full_addr + (page_slot_p) * 8) : "memory");
                    mbarrier_arrive(page_full_addr + (page_slot_p) * 8);
                    page_slot_p += 1;
                    if (page_slot_p == 8) { page_slot_p = 0; _phase_page_empty ^= 1; }
                }
                if (n_groups_p % 2 == 1) {
                    int group_block_pt = lane / 8;
                    int page_i_pt = lane % 8;
                    int n_block_pt = (n_groups_p - 1) * 4 + group_block_pt;
                    if (n_block_pt > last_block_p) {
                        n_block_pt = last_block_p;
                    }
                    mbarrier_wait(page_empty_addr + (page_slot_p) * 8, _phase_page_empty);
                    int page_stage_addr_pt = smem_page_offsets_addr + (unsigned int)(page_slot_p * 128);
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(page_stage_addr_pt + lane * 4), "l"(page_table + (pt_base_p + n_block_pt * 8 + page_i_pt)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(page_full_addr + (page_slot_p) * 8) : "memory");
                    mbarrier_arrive(page_full_addr + (page_slot_p) * 8);
                    page_slot_p += 1;
                    if (page_slot_p == 8) { page_slot_p = 0; _phase_page_empty ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + work_stage_p * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_p * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_p * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_p * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 2) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                tile_idx_p = (_clc_ctaid_2 * (unsigned int)Q_LEN + _clc_ctaid_0) * (unsigned int)NUM_KV_HEADS + _clc_ctaid_1;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        { // scheduler_main
            unsigned int work_stage_sched = 0;
            unsigned int throttle_stage_sched = 0;
            unsigned int total_tiles_sched = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_sched = 0; _tile_iter_sched < total_tiles_sched; _tile_iter_sched++) {
                mbarrier_wait(throttle_full_addr + (throttle_stage_sched) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_sched) * 8);
                throttle_stage_sched += 1;
                if (throttle_stage_sched == 2) { throttle_stage_sched = 0; _phase_throttle_full ^= 1; }
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_sched) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_sched * 16 + 0 * 16), "r"(work_full_addr + work_stage_sched * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (work_stage_sched) * 8, _phase_work_full_4);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_sched * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_sched * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_sched * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_sched * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_sched) * 8);
                work_stage_sched += 1;
                if (work_stage_sched == 2) { work_stage_sched = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int page_slot_l = 0;
            unsigned int work_stage_l = 0;
            unsigned int throttle_stage_l = 0;
            unsigned int total_tiles_l = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
            unsigned int tile_idx_l = (blockIdx.z * Q_LEN + blockIdx.x) * NUM_KV_HEADS + blockIdx.y;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_page_full = 0;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < total_tiles_l; _tile_iter_l++) {
                mbarrier_wait(throttle_empty_addr + (throttle_stage_l) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage_l) * 8);
                throttle_stage_l += 1;
                if (throttle_stage_l == 2) { throttle_stage_l = 0; _phase_throttle_empty ^= 1; }
                const int tiles_per_batch_4 = Q_LEN * NUM_KV_HEADS;
                int batch_idx_4 = tile_idx_l / (unsigned int)tiles_per_batch_4;
                int tile_in_batch_4 = tile_idx_l % (unsigned int)tiles_per_batch_4;
                int q_row_idx_4 = tile_in_batch_4 / NUM_KV_HEADS;
                int kv_head_idx_4 = tile_in_batch_4 % NUM_KV_HEADS;
                int kv_len_b_4 = causal_seqlens_kv_global[batch_idx_4];
                int causal_prefix_4 = 0;
                if (kv_len_b_4 > Q_LEN) {
                    causal_prefix_4 = kv_len_b_4 - Q_LEN;
                }
                int global_q_pos_4 = causal_prefix_4 + q_row_idx_4;
                int visible_local_keys_4 = global_q_pos_4 + 1;
                int win_start_4 = 0;
                int num_n_blocks_total_4 = (max_local_seq_len + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks_4 = ((num_n_blocks_total_4 + num_n_blocks_total_4 % 2) > (4) ? (num_n_blocks_total_4 + num_n_blocks_total_4 % 2) : (4));
                int split_start_block_1 = 0;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                const int group_ratio_l = NUM_Q_HEADS / NUM_KV_HEADS;
                int off_qt = (batch_idx_4 * Q_LEN + q_row_idx_4) * NUM_Q_HEADS + kv_head_idx_4 * group_ratio_l;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_qt_addr, (&Qt), 0, off_qt, 0, q_full_addr);
                }
                int page_tile_base_l = page_slot_l;
                int prefill = K_LOOKAHEAD;
                #pragma unroll 1
                for (int ni = 0; ni < prefill; ni++) {
                    int kv_stage = ni % NUM_KV_STAGES;
                    int n_block = split_start_block_1 + cta_n_blocks_4 - 1 - ni;
                    if (ni % 4 == 0) {
                        mbarrier_wait(page_full_addr + (page_slot_l) * 8, _phase_page_full);
                        page_slot_l += 1;
                        if (page_slot_l == 8) { page_slot_l = 0; _phase_page_full ^= 1; }
                    }
                    int pg_k[4];
                    int page_stage_k = (page_tile_base_l + ni / 4) % 8;
                    if (elect_sync()) {
                        int page_addr = smem_page_offsets_addr + (unsigned int)(page_stage_k * 128) + (unsigned int)((ni % 4 * 8 + 4) * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 3]))
                            : "r"(page_addr));
                    }
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_empty = (int)((unsigned int)ni % 6);
#else
                    int _stage_kv_empty = (int)((unsigned int)ni % 5);
#endif
                    mbarrier_wait(kv_empty_addr + (_stage_kv_empty) * 8, 1);
                    if (elect_sync()) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)ni % 6)) * 8, 32768);
#else
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)ni % 5)) * 8, 32768);
#endif
                        int ldst = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                        int page_ids[4];
                        int page_addr_1 = smem_page_offsets_addr + (unsigned int)(page_stage_k * 128) + (unsigned int)(ni % 4 * 8 * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&page_ids[0])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids[(0) + 3]))
                            : "r"(page_addr_1));
                        #pragma unroll
                        for (int page_i = 0; page_i < 4; page_i++) {
                            int page_id = page_ids[page_i];
                            #pragma unroll
                            for (int head_dim_half = 0; head_dim_half < 2; head_dim_half++) {
                                int byte_offset = head_dim_half * 16384 + page_i * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(ldst + byte_offset, (&K), 0, 0, head_dim_half, kv_head_idx_4, page_id, kv_full_addr + ((int)((unsigned int)ni % 6)) * 8);
#else
                                    tma_5d_gmem2smem(ldst + byte_offset, (&K), 0, 0, head_dim_half, kv_head_idx_4, page_id, kv_full_addr + ((int)((unsigned int)ni % 5)) * 8);
#endif
                                }
                            }
                        }
                        #pragma unroll
                        for (int pg_half = 0; pg_half < 4; pg_half++) {
                            int pg0 = pg_k[pg_half];
                            const int pg_i = pg_half + 4;
                            #pragma unroll
                            for (int hg = 0; hg < 2; hg++) {
                                int toff = hg * 16384 + pg_i * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(ldst + toff, (&K), 0, 0, hg, kv_head_idx_4, pg0, kv_full_addr + ((int)((unsigned int)ni % 6)) * 8);
#else
                                    tma_5d_gmem2smem(ldst + toff, (&K), 0, 0, hg, kv_head_idx_4, pg0, kv_full_addr + ((int)((unsigned int)ni % 5)) * 8);
#endif
                                }
                            }
                        }
                    }
                }
                int main_ni = cta_n_blocks_4 - K_LOOKAHEAD;
                #pragma unroll 1
                for (int ni_1 = 0; ni_1 < main_ni; ni_1++) {
                    int stage = ni_1 % NUM_KV_STAGES;
                    int n_block_1 = split_start_block_1 + cta_n_blocks_4 - 1 - ni_1;
                    int pg_v[4];
                    int page_stage_v = (page_tile_base_l + ni_1 / 4) % 8;
                    if (elect_sync()) {
                        int page_addr_2 = smem_page_offsets_addr + (unsigned int)(page_stage_v * 128) + (unsigned int)((ni_1 % 4 * 8 + 4) * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 3]))
                            : "r"(page_addr_2));
                    }
                    int page_group_done = ((ni_1 % 4 == 3) ? 1 : 0);
                    page_group_done = page_group_done | ((ni_1 + 1 == cta_n_blocks_4) ? 1 : 0);
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_empty_1 = (int)((unsigned int)ni_1 % 6);
#else
                    int _stage_kv_empty_1 = (int)((unsigned int)ni_1 % 5);
#endif
                    mbarrier_wait(kv_empty_addr + (_stage_kv_empty_1) * 8, 0);
                    if (elect_sync()) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)ni_1 % 6)) * 8, 32768);
#else
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)ni_1 % 5)) * 8, 32768);
#endif
                        int vdst = smem_kv_addr + (unsigned int)(stage * 32768);
                        int page_ids_1[4];
                        int page_addr_3 = smem_page_offsets_addr + (unsigned int)(page_stage_v * 128) + (unsigned int)(ni_1 % 4 * 8 * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&page_ids_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_1[(0) + 3]))
                            : "r"(page_addr_3));
                        #pragma unroll
                        for (int page_i_1 = 0; page_i_1 < 4; page_i_1++) {
                            int page_id_1 = page_ids_1[page_i_1];
                            #pragma unroll
                            for (int head_dim_half_1 = 0; head_dim_half_1 < 2; head_dim_half_1++) {
                                int byte_offset_1 = head_dim_half_1 * 16384 + page_i_1 * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(vdst + byte_offset_1, (&V), 0, 0, head_dim_half_1, kv_head_idx_4, page_id_1, kv_full_addr + ((int)((unsigned int)ni_1 % 6)) * 8);
#else
                                    tma_5d_gmem2smem(vdst + byte_offset_1, (&V), 0, 0, head_dim_half_1, kv_head_idx_4, page_id_1, kv_full_addr + ((int)((unsigned int)ni_1 % 5)) * 8);
#endif
                                }
                            }
                        }
                        if (page_group_done != 0) {
                            mbarrier_arrive(page_empty_addr + (((page_slot_l + 6 >= 8) ? page_slot_l + 6 - 8 : page_slot_l + 6)) * 8);
                        }
                        #pragma unroll
                        for (int pg_half_1 = 0; pg_half_1 < 4; pg_half_1++) {
                            int vpg0 = pg_v[pg_half_1];
                            const int pg_i_1 = pg_half_1 + 4;
                            #pragma unroll
                            for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                int vtoff = hg_1 * 16384 + pg_i_1 * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, hg_1, kv_head_idx_4, vpg0, kv_full_addr + ((int)((unsigned int)ni_1 % 6)) * 8);
#else
                                    tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, hg_1, kv_head_idx_4, vpg0, kv_full_addr + ((int)((unsigned int)ni_1 % 5)) * 8);
#endif
                                }
                            }
                        }
                    }
                    int next_ni = ni_1 + K_LOOKAHEAD;
                    int k_stage = next_ni % NUM_KV_STAGES;
                    int next_n = split_start_block_1 + cta_n_blocks_4 - 1 - next_ni;
                    if (next_ni % 4 == 0) {
                        mbarrier_wait(page_full_addr + (page_slot_l) * 8, _phase_page_full);
                        page_slot_l += 1;
                        if (page_slot_l == 8) { page_slot_l = 0; _phase_page_full ^= 1; }
                    }
                    int pg_nk[4];
                    int page_stage_nk = (page_tile_base_l + next_ni / 4) % 8;
                    if (elect_sync()) {
                        int page_addr_4 = smem_page_offsets_addr + (unsigned int)(page_stage_nk * 128) + (unsigned int)((next_ni % 4 * 8 + 4) * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 3]))
                            : "r"(page_addr_4));
                    }
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_empty_2 = (int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 6);
#else
                    int _stage_kv_empty_2 = (int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 5);
#endif
                    mbarrier_wait(kv_empty_addr + (_stage_kv_empty_2) * 8, 1);
                    if (elect_sync()) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 6)) * 8, 32768);
#else
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 5)) * 8, 32768);
#endif
                        int kdst = smem_kv_addr + (unsigned int)(k_stage * 32768);
                        int page_ids_2[4];
                        int page_addr_5 = smem_page_offsets_addr + (unsigned int)(page_stage_nk * 128) + (unsigned int)(next_ni % 4 * 8 * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&page_ids_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_2[(0) + 3]))
                            : "r"(page_addr_5));
                        #pragma unroll
                        for (int page_i_2 = 0; page_i_2 < 4; page_i_2++) {
                            int page_id_2 = page_ids_2[page_i_2];
                            #pragma unroll
                            for (int head_dim_half_2 = 0; head_dim_half_2 < 2; head_dim_half_2++) {
                                int byte_offset_2 = head_dim_half_2 * 16384 + page_i_2 * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(kdst + byte_offset_2, (&K), 0, 0, head_dim_half_2, kv_head_idx_4, page_id_2, kv_full_addr + ((int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 6)) * 8);
#else
                                    tma_5d_gmem2smem(kdst + byte_offset_2, (&K), 0, 0, head_dim_half_2, kv_head_idx_4, page_id_2, kv_full_addr + ((int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 5)) * 8);
#endif
                                }
                            }
                        }
                        #pragma unroll
                        for (int pg_half_2 = 0; pg_half_2 < 4; pg_half_2++) {
                            int npg0 = pg_nk[pg_half_2];
                            const int pg_i_2 = pg_half_2 + 4;
                            #pragma unroll
                            for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                int ntoff = hg_2 * 16384 + pg_i_2 * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, hg_2, kv_head_idx_4, npg0, kv_full_addr + ((int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 6)) * 8);
#else
                                    tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, hg_2, kv_head_idx_4, npg0, kv_full_addr + ((int)((unsigned int)(ni_1 + K_LOOKAHEAD) % 5)) * 8);
#endif
                                }
                            }
                        }
                    }
                }
                #pragma unroll 1
                for (int tail_ni = main_ni; tail_ni < cta_n_blocks_4; tail_ni++) {
                    int tstage = tail_ni % NUM_KV_STAGES;
                    int tn_block = split_start_block_1 + cta_n_blocks_4 - 1 - tail_ni;
                    int tpg_v[4];
                    int page_stage_tv = (page_tile_base_l + tail_ni / 4) % 8;
                    if (elect_sync()) {
                        int page_addr_6 = smem_page_offsets_addr + (unsigned int)(page_stage_tv * 128) + (unsigned int)((tail_ni % 4 * 8 + 4) * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tpg_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&tpg_v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tpg_v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tpg_v[(0) + 3]))
                            : "r"(page_addr_6));
                    }
                    int page_group_done_tail = ((tail_ni % 4 == 3) ? 1 : 0);
                    page_group_done_tail = page_group_done_tail | ((tail_ni + 1 == cta_n_blocks_4) ? 1 : 0);
#if __CUDA_ARCH__ == 1030
                    int _stage_kv_empty_3 = (int)((unsigned int)tail_ni % 6);
#else
                    int _stage_kv_empty_3 = (int)((unsigned int)tail_ni % 5);
#endif
                    mbarrier_wait(kv_empty_addr + (_stage_kv_empty_3) * 8, 0);
                    if (elect_sync()) {
#if __CUDA_ARCH__ == 1030
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)tail_ni % 6)) * 8, 32768);
#else
                        mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)tail_ni % 5)) * 8, 32768);
#endif
                        int tvdst = smem_kv_addr + (unsigned int)(tstage * 32768);
                        int page_ids_3[4];
                        int page_addr_7 = smem_page_offsets_addr + (unsigned int)(page_stage_tv * 128) + (unsigned int)(tail_ni % 4 * 8 * 4);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&page_ids_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&page_ids_3[(0) + 3]))
                            : "r"(page_addr_7));
                        #pragma unroll
                        for (int page_i_3 = 0; page_i_3 < 4; page_i_3++) {
                            int page_id_3 = page_ids_3[page_i_3];
                            #pragma unroll
                            for (int head_dim_half_3 = 0; head_dim_half_3 < 2; head_dim_half_3++) {
                                int byte_offset_3 = head_dim_half_3 * 16384 + page_i_3 * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(tvdst + byte_offset_3, (&V), 0, 0, head_dim_half_3, kv_head_idx_4, page_id_3, kv_full_addr + ((int)((unsigned int)tail_ni % 6)) * 8);
#else
                                    tma_5d_gmem2smem(tvdst + byte_offset_3, (&V), 0, 0, head_dim_half_3, kv_head_idx_4, page_id_3, kv_full_addr + ((int)((unsigned int)tail_ni % 5)) * 8);
#endif
                                }
                            }
                        }
                        if (page_group_done_tail != 0) {
                            mbarrier_arrive(page_empty_addr + (((page_slot_l + (8 - ((tail_ni + 1 == cta_n_blocks_4) ? 1 : 2)) >= 8) ? page_slot_l + (8 - ((tail_ni + 1 == cta_n_blocks_4) ? 1 : 2)) - 8 : page_slot_l + (8 - ((tail_ni + 1 == cta_n_blocks_4) ? 1 : 2)))) * 8);
                        }
                        #pragma unroll
                        for (int pg_half_3 = 0; pg_half_3 < 4; pg_half_3++) {
                            int tvpg0 = tpg_v[pg_half_3];
                            const int pg_i_3 = pg_half_3 + 4;
                            #pragma unroll
                            for (int hg_3 = 0; hg_3 < 2; hg_3++) {
                                int tvtoff = hg_3 * 16384 + pg_i_3 * 2048;
                                {
#if __CUDA_ARCH__ == 1030
                                    tma_5d_gmem2smem(tvdst + tvtoff, (&V), 0, 0, hg_3, kv_head_idx_4, tvpg0, kv_full_addr + ((int)((unsigned int)tail_ni % 6)) * 8);
#else
                                    tma_5d_gmem2smem(tvdst + tvtoff, (&V), 0, 0, hg_3, kv_head_idx_4, tvpg0, kv_full_addr + ((int)((unsigned int)tail_ni % 5)) * 8);
#endif
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_5);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_l * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage_l * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage_l * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_8)
                    : "r"(work_response_addr + work_stage_l * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 2) { work_stage_l = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                tile_idx_l = (_clc_ctaid_8 * (unsigned int)Q_LEN + _clc_ctaid_6) * (unsigned int)NUM_KV_HEADS + _clc_ctaid_7;
            }
        }
    }

    // Cleanup
}

} // extern "C"
