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
#define TMEM_NCOLS 224
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
#define TMEM_TMEM_KV_OFFSET 96
#define NUM_PK_PIPE_STAGES 4
#define NUM_PG_PIPE_STAGES 4
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_CORR_REDUCE_OFF 1024
#define SMEM_SMEM_CORR_REDUCE_STAGE_BYTES 128
#define SMEM_SMEM_CORR_REDUCE_STRIDE 128
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
#define SMEM_SMEM_QT_OFF 2048
#define SMEM_SMEM_QT_STAGE_BYTES 1024
#define SMEM_SMEM_QT_STRIDE 1024
#define SMEM_SMEM_P0_OFF 3072
#define SMEM_SMEM_P0_STAGE_BYTES 1024
#define SMEM_SMEM_P0_STRIDE 1024
#define SMEM_SMEM_P1_OFF 5120
#define SMEM_SMEM_P1_STAGE_BYTES 1024
#define SMEM_SMEM_P1_STRIDE 1024
#define SMEM_SMEM_PG_OFF 6784
#define SMEM_SMEM_PG_STAGE_BYTES 64
#define SMEM_SMEM_PG_STRIDE 64
#define SMEM_SMEM_PK_OFF 7168
#define SMEM_SMEM_PK_STAGE_BYTES 34816
#define SMEM_SMEM_PK_STRIDE 34816
#define SMEM_SMEM_O_OFF 146432
#define SMEM_SMEM_O_STAGE_BYTES 1024
#define SMEM_SMEM_O_STRIDE 1024
#define SMEM_TOTAL 147456
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 16
#define BATCH_SIZE 128
#define NUM_Q_HEADS 32
#define NUM_KV_HEADS 4
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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
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





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


// Qmul4 (QMUL4.E4M3x4.E2M1x4.E4M3x4.SATFINITE) as the instruction: byte k of the result is the E2M1 value in one
// nibble of src times the E4M3 value in one byte of scale, rounded once to E4M3 (to nearest even, saturating to
// +-448). mul .e2m1p4x4 reads nibble 2k of a word, mul .e2m1x4 nibble k of a 16-bit half; both times byte k of
// their scale word.
//   qmul4_direct   nibble 2k      byte k
//   qmul4_b<j>     nibble 2k      byte j
//   qmul4_lower4   nibble k       byte 0
//   qmul4_higher4  nibble k + 4   byte 0

// Byte kByte of scale in all four bytes.
template <int kByte>
__device__ __forceinline__ uint32_t qmul4_broadcast(uint32_t scale) {
  uint32_t y;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(y) : "r"(scale), "n"(kByte * 0x1111));
  return y;
}

__device__ __forceinline__ uint32_t qmul4_direct(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 %0, %1, %2;" : "=r"(y) : "r"(src), "r"(scale));
  return y;
}

template <int kByte>
__device__ __forceinline__ uint32_t qmul4_b(uint32_t src, uint32_t scale) {
  return qmul4_direct(src, qmul4_broadcast<kByte>(scale));
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_decode_nvfp4_bcfe92828a999ce45ff4(const __grid_constant__ CUtensorMap Qt, const __grid_constant__ CUtensorMap Kp, const __grid_constant__ CUtensorMap Vp, const __grid_constant__ CUtensorMap Ksf, const __grid_constant__ CUtensorMap Vsf, uint8_t* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int pt_batch_stride, int pt_v_offset, int bmm1_is_log2, int num_splits, int blocks_per_split, float bmm1_scalar, float bmm2_scalar)
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
    #define pk_full_addr (mbar_base + 16)
    #define pk_empty_addr (mbar_base + 48)
    #define pg_full_addr (mbar_base + 80)
    #define pg_empty_addr (mbar_base + 112)
    #define kv_full_addr (mbar_base + 144)
    #define kv_empty_addr (mbar_base + 176)
    #define s_full_0_addr (mbar_base + 208)
    #define s_full_1_addr (mbar_base + 216)
    #define s_empty_0_addr (mbar_base + 224)
    #define s_empty_1_addr (mbar_base + 232)
    #define o_free_0_addr (mbar_base + 240)
    #define o_free_1_addr (mbar_base + 248)
    #define o_done_0_addr (mbar_base + 256)
    #define o_done_1_addr (mbar_base + 264)
    #define corr_scale_0_addr (mbar_base + 272)
    #define corr_scale_1_addr (mbar_base + 280)
    #define corr_empty_0_addr (mbar_base + 288)
    #define corr_empty_1_addr (mbar_base + 296)
    #define stats_empty_addr (mbar_base + 304)
    #define tmem_dealloc_addr (mbar_base + 312)
    #define p_empty_0_addr (mbar_base + 320)
    #define p_empty_1_addr (mbar_base + 328)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_corr_reduce = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr_reduce_addr = smem + 1024;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int smem_qt_addr = smem + 2048;
    uint8_t* smem_p0 = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_p0_addr = smem + 3072;
    uint8_t* smem_p1 = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_p1_addr = smem + 5120;
    int* smem_pg = reinterpret_cast<int*>(smem_raw + 6784);
    const int smem_pg_addr = smem + 6784;
    uint8_t* smem_pk = reinterpret_cast<uint8_t*>(smem_raw + 7168);
    const int smem_pk_addr = smem + 7168;
    uint8_t* smem_o = reinterpret_cast<uint8_t*>(smem_raw + 146432);
    const int smem_o_addr = smem + 146432;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Qt))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Kp))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vp))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Ksf))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vsf))) : "memory");

    // Mbarrier init (24 pipeline groups, 0 ordered-sequence groups, 42 barriers)
    // Mbarriers at smem_raw[0..336)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'pk_pipe' ---
            // pk_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // pk_empty: 4 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            mbarrier_init(smem + 64, 128);
            mbarrier_init(smem + 72, 128);
            // --- pipeline 'pg_pipe' ---
            // pg_full: 4 barriers, init_count=32
            mbarrier_init(smem + 80, 32);
            mbarrier_init(smem + 88, 32);
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            // pg_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // s_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            // s_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            // o_free_0: 1 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            // o_free_1: 1 barriers, init_count=128
            mbarrier_init(smem + 248, 128);
            // o_done_0: 1 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            // o_done_1: 1 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            // corr_scale_0: 1 barriers, init_count=128
            mbarrier_init(smem + 272, 128);
            // corr_scale_1: 1 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            // corr_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 288, 128);
            // corr_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 304, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 312, 128);
            // p_empty_0: 1 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            // p_empty_1: 1 barriers, init_count=1
            mbarrier_init(smem + 328, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 224 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 336);
    if (warp == 0) {
        int _tmem_hold = smem + 336;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
    const int tmem_tmem_kv = taddr + 96;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            float log2e_mult_s = ((bmm1_is_log2 != 0) ? 1.0f : 1.442695f);
            float bmm1_scale_s = bmm1_scalar;
            if ((unsigned long long)bmm1_scale_ptr != 0) {
                bmm1_scale_s = bmm1_scale_ptr[0];
            }
            float softmax_scale_log2 = bmm1_scale_s * log2e_mult_s;
            const int tmem_row_base_v = warp % 4 * 32;
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            unsigned int total_tiles_s = BATCH_SIZE * NUM_KV_HEADS * num_splits;
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            unsigned int _phase_p_empty_1_0 = 1;
            unsigned int _phase_p_empty_0_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_s = blockIdx.x; tile_idx_s < total_tiles_s; tile_idx_s += gridDim.x) {
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                int split_s = tile_idx_s % (unsigned int)num_splits;
                int bh_s = tile_idx_s / (unsigned int)num_splits;
                int batch_idx_s = bh_s / NUM_KV_HEADS;
                int seqlen_kv_s = seq_lens_kv[batch_idx_s];
                int num_n_blocks_total_s = (seqlen_kv_s + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_s = num_n_blocks_total_s + num_n_blocks_total_s % 2;
                int split_start_block_s = split_s * blocks_per_split;
                int rem_blocks_s = even_n_blocks_s - split_start_block_s;
                int capped_blocks_s = ((rem_blocks_s > blocks_per_split) ? blocks_per_split : rem_blocks_s);
                int cta_n_blocks_s = ((capped_blocks_s < 2) ? 2 : capped_blocks_s);
                int num_pairs_s = cta_n_blocks_s / 2;
                float row_max_pair0[2];
                float row_sum_pair0[2];
                float row_max_pair1[2];
                float row_sum_pair1[2];
                row_max_pair0[0] = -CUDART_INF_F;
                row_max_pair0[1] = -CUDART_INF_F;
                row_sum_pair0[0] = 0.0f;
                row_sum_pair0[1] = 0.0f;
                row_max_pair1[0] = -CUDART_INF_F;
                row_max_pair1[1] = -CUDART_INF_F;
                row_sum_pair1[0] = 0.0f;
                row_sum_pair1[1] = 0.0f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    smem_exch0_u32[wg_tid] = _amf_enc_0;
                    smem_exch1_u32[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll 1
                for (int pair_s = 0; pair_s < num_pairs_s; pair_s++) {
                    #pragma unroll
                    for (int inst_s = 0; inst_s < 2; inst_s++) {
                        int is_inst1 = inst_s;
                        int my_tmem_s_base = taddr + (unsigned int)(((is_inst1 != 0) ? 8 : 0)) + (unsigned int)(tmem_row_base_v << 16);
                        int my_tmem_stats = taddr + (unsigned int)(((is_inst1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
                        unsigned int* my_exch_u32_ptr = ((is_inst1 != 0) ? (smem_exch1_u32) : (smem_exch0_u32));
                        uint8_t* my_p_base = ((is_inst1 != 0) ? (smem_p1) : (smem_p0));
                        if (is_inst1 != 0) {
                            mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                            _phase_s_full_1_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        } else {
                            mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                            _phase_s_full_0_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        float sv[8];
                        float sv_lo[4];
                        float sv_hi[4];
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
                        int my_block_s = split_start_block_s + cta_n_blocks_s - 1 - 2 * pair_s - is_inst1;
                        int ldtm_row_base_s = warp_in_wg * 32 + lane / 4;
                        int block_base_s = my_block_s * BLOCK_N;
                        int kv_row0_s = ldtm_row_base_s;
                        int kv_row1_s = ldtm_row_base_s + 8;
                        int kv_row2_s = ldtm_row_base_s + 16;
                        int kv_row3_s = ldtm_row_base_s + 24;
                        if (seqlen_kv_s <= block_base_s + kv_row0_s) {
                            sv[0] = -3.4028235e+38f;
                            sv[1] = -3.4028235e+38f;
                        }
                        if (seqlen_kv_s <= block_base_s + kv_row1_s) {
                            sv[2] = -3.4028235e+38f;
                            sv[3] = -3.4028235e+38f;
                        }
                        if (seqlen_kv_s <= block_base_s + kv_row2_s) {
                            sv[4] = -3.4028235e+38f;
                            sv[5] = -3.4028235e+38f;
                        }
                        if (seqlen_kv_s <= block_base_s + kv_row3_s) {
                            sv[6] = -3.4028235e+38f;
                            sv[7] = -3.4028235e+38f;
                        }
                        float pair_max_s[2];
                        pair_max_s[0] = max_noftz(max_noftz(sv[0], sv[2]), max_noftz(sv[4], sv[6]));
                        pair_max_s[1] = max_noftz(max_noftz(sv[1], sv[3]), max_noftz(sv[5], sv[7]));
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 2; c_1++) {
                            float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max_s[c_1], 16);
                            pair_max_s[c_1] = max_noftz(pair_max_s[c_1], _shfl_xor_sync_1);
                            float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, pair_max_s[c_1], 8);
                            pair_max_s[c_1] = max_noftz(pair_max_s[c_1], _shfl_xor_sync_2);
                            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, pair_max_s[c_1], 4);
                            pair_max_s[c_1] = max_noftz(pair_max_s[c_1], _shfl_xor_sync_3);
                        }
                        float old_max_pair_s[2];
                        float new_max_pair_s[2];
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 2; c_2++) {
                            old_max_pair_s[c_2] = ((is_inst1 != 0) ? row_max_pair1[c_2] : row_max_pair0[c_2]);
                            new_max_pair_s[c_2] = max_noftz(old_max_pair_s[c_2], pair_max_s[c_2]);
                        }
                        if (lane < 4) {
                            uint32_t _amf_u_4 = __float_as_uint(new_max_pair_s[0]);
                            uint32_t _amf_mask_4 = -int32_t(_amf_u_4 >> 31) | 0x80000000u;
                            unsigned int _amf_enc_1 = _amf_u_4 ^ _amf_mask_4;
                            uint32_t _amf_u_5 = __float_as_uint(new_max_pair_s[1]);
                            uint32_t _amf_mask_5 = -int32_t(_amf_u_5 >> 31) | 0x80000000u;
                            unsigned int _amf_enc_2 = _amf_u_5 ^ _amf_mask_5;
                            atomicMax(&my_exch_u32_ptr[col_pair_base], _amf_enc_1);
                            atomicMax(&my_exch_u32_ptr[col_pair_base + 1], _amf_enc_2);
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        uint32_t _amf_u_7 = my_exch_u32_ptr[col_pair_base];
                        uint32_t _amf_mask_7 = ((_amf_u_7 >> 31) - 1u) | 0x80000000u;
                        float _atomic_max_float_decode_6 = __uint_as_float(_amf_u_7 ^ _amf_mask_7);
                        new_max_pair_s[0] = _atomic_max_float_decode_6;
                        uint32_t _amf_u_9 = my_exch_u32_ptr[col_pair_base + 1];
                        uint32_t _amf_mask_9 = ((_amf_u_9 >> 31) - 1u) | 0x80000000u;
                        float _atomic_max_float_decode_8 = __uint_as_float(_amf_u_9 ^ _amf_mask_9);
                        new_max_pair_s[1] = _atomic_max_float_decode_8;
                        float2 acc_delta_pair_f2_s = sub_f32x2(make_float2(old_max_pair_s[0], old_max_pair_s[1]), make_float2(new_max_pair_s[0], new_max_pair_s[1]));
                        float2 _f32x2_mul_a_10 = make_float2(softmax_scale_log2, softmax_scale_log2);
                        float2 acc_scaled_delta_pair_f2_s;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&acc_scaled_delta_pair_f2_s) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&acc_delta_pair_f2_s));
                        float acc_scale_pair_s[2];
                        acc_scale_pair_s[0] = 1.0f;
                        acc_scale_pair_s[1] = 1.0f;
                        int needs_acc_rescale_s = ((acc_delta_pair_f2_s.x != 0.0f) ? 1 : 0);
                        needs_acc_rescale_s = needs_acc_rescale_s | ((acc_delta_pair_f2_s.y != 0.0f) ? 1 : 0);
                        if (needs_acc_rescale_s != 0) {
                            float _exp2_11 = approx_exp2(acc_scaled_delta_pair_f2_s.x);
                            acc_scale_pair_s[0] = _exp2_11;
                            float _exp2_12 = approx_exp2(acc_scaled_delta_pair_f2_s.y);
                            acc_scale_pair_s[1] = _exp2_12;
                        }
                        float stats_pair_s[4];
                        stats_pair_s[0] = old_max_pair_s[0];
                        stats_pair_s[1] = old_max_pair_s[1];
                        stats_pair_s[2] = new_max_pair_s[0];
                        stats_pair_s[3] = new_max_pair_s[1];
                        if (is_inst1 != 0) {
                            mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                            _phase_corr_empty_1_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        } else {
                            mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                            _phase_corr_empty_0_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        tmem_st_x4_f32(my_tmem_stats, stats_pair_s);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (is_inst1 != 0) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(corr_scale_1_addr);
                        } else {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(corr_scale_0_addr);
                        }
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 2; c_3++) {
                            if (is_inst1 != 0) {
                                row_max_pair1[c_3] = new_max_pair_s[c_3];
                            } else {
                                row_max_pair0[c_3] = new_max_pair_s[c_3];
                            }
                        }
                        float safe_max0_s = ((new_max_pair_s[0] == -CUDART_INF_F) ? 0.0f : new_max_pair_s[0]);
                        float safe_max1_s = ((new_max_pair_s[1] == -CUDART_INF_F) ? 0.0f : new_max_pair_s[1]);
                        float2 _f2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
                        float2 neg_scaled_pair_f2_s = fma_f32x2_rn_ftz(make_float2(safe_max0_s, safe_max1_s), make_float2(-softmax_scale_log2, -softmax_scale_log2), make_float2(8.807355f, 8.807355f));
                        float exp_vals_s[8];
                        float2 affine01_pair_f2_s = fma_f32x2_rn_ftz(_f2_0, make_float2(sv[0], sv[1]), neg_scaled_pair_f2_s);
                        float _exp2_13 = approx_exp2(affine01_pair_f2_s.x);
                        exp_vals_s[0] = _exp2_13;
                        float _exp2_14 = approx_exp2(affine01_pair_f2_s.y);
                        exp_vals_s[1] = _exp2_14;
                        float2 affine23_pair_f2_s = fma_f32x2_rn_ftz(_f2_0, make_float2(sv[2], sv[3]), neg_scaled_pair_f2_s);
                        float _exp2_15 = approx_exp2(affine23_pair_f2_s.x);
                        exp_vals_s[2] = _exp2_15;
                        float _exp2_16 = approx_exp2(affine23_pair_f2_s.y);
                        exp_vals_s[3] = _exp2_16;
                        float2 affine45_pair_f2_s = fma_f32x2_rn_ftz(_f2_0, make_float2(sv[4], sv[5]), neg_scaled_pair_f2_s);
                        float _exp2_17 = approx_exp2(affine45_pair_f2_s.x);
                        exp_vals_s[4] = _exp2_17;
                        float _exp2_18 = approx_exp2(affine45_pair_f2_s.y);
                        exp_vals_s[5] = _exp2_18;
                        float2 affine67_pair_f2_s = fma_f32x2_rn_ftz(_f2_0, make_float2(sv[6], sv[7]), neg_scaled_pair_f2_s);
                        float _exp2_19 = approx_exp2(affine67_pair_f2_s.x);
                        exp_vals_s[6] = _exp2_19;
                        float _exp2_20 = approx_exp2(affine67_pair_f2_s.y);
                        exp_vals_s[7] = _exp2_20;
                        unsigned int regs_p_s[2];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(exp_vals_s[0]), "f"(exp_vals_s[1]),
                                                   "f"(exp_vals_s[2]), "f"(exp_vals_s[3]));
                            regs_p_s[0] = _packed;
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
                                : "=r"(_packed) : "f"(exp_vals_s[4]), "f"(exp_vals_s[5]),
                                                   "f"(exp_vals_s[6]), "f"(exp_vals_s[7]));
                            regs_p_s[1] = _packed;
                        }
                        int mtx_idx_s = lane / 8;
                        int thr_row_idx_s = lane % 8;
                        int seg_col_idx_s = warp_in_wg * 2 + mtx_idx_s ^ thr_row_idx_s;
                        int stsm_offset_s = thr_row_idx_s * 128 + seg_col_idx_s * 16;
                        if (is_inst1 != 0) {
                            mbarrier_wait(p_empty_1_addr, _phase_p_empty_1_0);
                            _phase_p_empty_1_0 ^= 1;
                        } else {
                            mbarrier_wait(p_empty_0_addr, _phase_p_empty_0_0);
                            _phase_p_empty_0_0 ^= 1;
                        }
                        const void* _stmatrix_b8_ptr_21 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset_s);
                        uint64_t _stmatrix_b8_addr64_21;
                        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_21) : "l"(_stmatrix_b8_ptr_21));
                        uint32_t _stmatrix_b8_addr_21;
                        asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_21) : "l"(_stmatrix_b8_addr64_21));
                        asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_b8_addr_21), "r"(regs_p_s[0]), "r"(regs_p_s[1])
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (is_inst1 != 0) {
                            mbarrier_arrive(s_empty_1_addr);
                        } else {
                            mbarrier_arrive(s_empty_0_addr);
                        }
                        float prior_sum0_s = ((is_inst1 != 0) ? row_sum_pair1[0] : row_sum_pair0[0]);
                        float prior_sum1_s = ((is_inst1 != 0) ? row_sum_pair1[1] : row_sum_pair0[1]);
                        float2 _f2_1 = make_float2(exp_vals_s[0], exp_vals_s[1]);
                        float2 _f2_2 = make_float2(exp_vals_s[2], exp_vals_s[3]);
                        float2 _f2_3 = make_float2(exp_vals_s[4], exp_vals_s[5]);
                        float2 _f2_4 = make_float2(exp_vals_s[6], exp_vals_s[7]);
                        float2 row_sum_p0_f2_s = fma_f32x2_rn_ftz(make_float2(prior_sum0_s, prior_sum1_s), make_float2(acc_scale_pair_s[0], acc_scale_pair_s[1]), _f2_1);
                        float2 row_sum_p1_f2_s = add_f32x2(row_sum_p0_f2_s, _f2_2);
                        float2 row_sum_p2_f2_s = add_f32x2(row_sum_p1_f2_s, _f2_3);
                        float2 row_sum_next_f2_s = add_f32x2(row_sum_p2_f2_s, _f2_4);
                        if (is_inst1 != 0) {
                            row_sum_pair1[0] = row_sum_next_f2_s.x;
                            row_sum_pair1[1] = row_sum_next_f2_s.y;
                        } else {
                            row_sum_pair0[0] = row_sum_next_f2_s.x;
                            row_sum_pair0[1] = row_sum_next_f2_s.y;
                        }
                    }
                }
                #pragma unroll
                for (int inst_s_1 = 0; inst_s_1 < 2; inst_s_1++) {
                    int is_inst1_1 = inst_s_1;
                    int my_tmem_stats_1 = taddr + (unsigned int)(((is_inst1_1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
                    if (is_inst1_1 != 0) {
                        mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                        _phase_corr_empty_1_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                        _phase_corr_empty_0_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    float final_stats_pair_s[4];
                    final_stats_pair_s[0] = ((is_inst1_1 != 0) ? row_sum_pair1[0] : row_sum_pair0[0]);
                    final_stats_pair_s[1] = ((is_inst1_1 != 0) ? row_sum_pair1[1] : row_sum_pair0[1]);
                    final_stats_pair_s[2] = ((is_inst1_1 != 0) ? row_max_pair1[0] : row_max_pair0[0]);
                    final_stats_pair_s[3] = ((is_inst1_1 != 0) ? row_max_pair1[1] : row_max_pair0[1]);
                    tmem_st_x4_f32(my_tmem_stats_1, final_stats_pair_s);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_inst1_1 != 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(corr_scale_1_addr);
                    } else {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(corr_scale_0_addr);
                    }
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            float log2e_mult_c = ((bmm1_is_log2 != 0) ? 1.0f : 1.442695f);
            float bmm1_scale_c = bmm1_scalar;
            if ((unsigned long long)bmm1_scale_ptr != 0) {
                bmm1_scale_c = bmm1_scale_ptr[0];
            }
            float softmax_scale_log2_1 = bmm1_scale_c * log2e_mult_c;
            float output_scale = bmm2_scalar;
            if ((unsigned long long)bmm2_scale_ptr != 0) {
                output_scale = bmm2_scale_ptr[0];
            }
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int corr_tid = warp % 4 * 32 + lane;
            const int col_pair_base_c = corr_tid % 4 * 2;
            const int group_ratio_rt = NUM_Q_HEADS / NUM_KV_HEADS;
            unsigned int total_tiles_c = BATCH_SIZE * NUM_KV_HEADS * num_splits;
            unsigned int _phase_corr_scale_0_0 = 0;
            unsigned int _phase_corr_scale_1_0 = 0;
            unsigned int _phase_o_done_0_0 = 0;
            unsigned int _phase_o_done_1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_c = blockIdx.x; tile_idx_c < total_tiles_c; tile_idx_c += gridDim.x) {
                int split_c = tile_idx_c % (unsigned int)num_splits;
                int bh_c = tile_idx_c / (unsigned int)num_splits;
                int batch_idx = bh_c / NUM_KV_HEADS;
                int kv_head_idx = bh_c % NUM_KV_HEADS;
                int seqlen_kv = seq_lens_kv[batch_idx];
                int num_n_blocks_total = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks = num_n_blocks_total + num_n_blocks_total % 2;
                int split_start_block = split_c * blocks_per_split;
                int rem_blocks = even_n_blocks - split_start_block;
                int capped_blocks = ((rem_blocks > blocks_per_split) ? blocks_per_split : rem_blocks);
                int cta_n_blocks = ((capped_blocks < 2) ? 2 : capped_blocks);
                int num_pairs = cta_n_blocks / 2;
                if (num_pairs > 0) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(o_free_0_addr);
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(o_free_1_addr);
                }
                #pragma unroll 1
                for (int pair = 1; pair < num_pairs; pair++) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_done_0_addr, _phase_o_done_0_0);
                    _phase_o_done_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc0_pair[2];
                    float2 max_diff0_pair_f2 = sub_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                    float2 _f32x2_mul_a_0 = make_float2(softmax_scale_log2_1, softmax_scale_log2_1);
                    float2 scaled_diff0_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_diff0_pair_f2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&max_diff0_pair_f2));
                    float _exp2_1 = approx_exp2(scaled_diff0_pair_f2.x);
                    acc0_pair[0] = ((max_diff0_pair_f2.x != 0.0f) ? _exp2_1 : 1.0f);
                    float _exp2_2 = approx_exp2(scaled_diff0_pair_f2.y);
                    acc0_pair[1] = ((max_diff0_pair_f2.y != 0.0f) ? _exp2_2 : 1.0f);
                    int rescale_pred_0 = ((acc0_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_0 = rescale_pred_0 | ((acc0_pair[1] != 1.0f) ? 1 : 0);
                    int _warp_vote_3 = __any_sync(0xFFFFFFFF, rescale_pred_0 != 0);
                    if (_warp_vote_3 != 0) {
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
                        float o0[8];
                        #pragma unroll
                        for (int h = 0; h < 4; h++) {
                            o0[h] = o0_lo[h];
                            o0[h + 4] = o0_hi[h];
                        }
                        float2 _f2_5 = make_float2(acc0_pair[0], acc0_pair[1]);
                        float2 _f2_6 = make_float2(o0[0], o0[1]);
                        float2 _f2_7 = make_float2(o0[2], o0[3]);
                        float2 _f2_8 = make_float2(o0[4], o0[5]);
                        float2 _f2_9 = make_float2(o0[6], o0[7]);
                        float2 o0_lo01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_lo01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_5));
                        float2 o0_lo23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_lo23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_7), "l"(*(const unsigned long long*)&_f2_5));
                        float2 o0_hi01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_hi01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_5));
                        float2 o0_hi23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o0_hi23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&_f2_5));
                        o0[0] = o0_lo01_scaled_f2.x;
                        o0[1] = o0_lo01_scaled_f2.y;
                        o0[2] = o0_lo23_scaled_f2.x;
                        o0[3] = o0_lo23_scaled_f2.y;
                        o0[4] = o0_hi01_scaled_f2.x;
                        o0[5] = o0_hi01_scaled_f2.y;
                        o0[6] = o0_hi23_scaled_f2.x;
                        o0[7] = o0_hi23_scaled_f2.y;
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 4; h_1++) {
                            o0_lo[h_1] = o0[h_1];
                            o0_hi[h_1] = o0[h_1 + 4];
                        }
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
                    mbarrier_arrive(o_free_0_addr);
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_done_1_addr, _phase_o_done_1_0);
                    _phase_o_done_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc1_pair[2];
                    float2 max_diff1_pair_f2 = sub_f32x2(make_float2(_tmem_load_1[0], _tmem_load_1[1]), make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                    float2 _f32x2_mul_a_4 = make_float2(softmax_scale_log2_1, softmax_scale_log2_1);
                    float2 scaled_diff1_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_diff1_pair_f2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&max_diff1_pair_f2));
                    float _exp2_5 = approx_exp2(scaled_diff1_pair_f2.x);
                    acc1_pair[0] = ((max_diff1_pair_f2.x != 0.0f) ? _exp2_5 : 1.0f);
                    float _exp2_6 = approx_exp2(scaled_diff1_pair_f2.y);
                    acc1_pair[1] = ((max_diff1_pair_f2.y != 0.0f) ? _exp2_6 : 1.0f);
                    int rescale_pred_1 = ((acc1_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_1 = rescale_pred_1 | ((acc1_pair[1] != 1.0f) ? 1 : 0);
                    int _warp_vote_7 = __any_sync(0xFFFFFFFF, rescale_pred_1 != 0);
                    if (_warp_vote_7 != 0) {
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
                        float o1[8];
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 4; h_2++) {
                            o1[h_2] = o1_lo[h_2];
                            o1[h_2 + 4] = o1_hi[h_2];
                        }
                        float2 _f2_10 = make_float2(acc1_pair[0], acc1_pair[1]);
                        float2 _f2_11 = make_float2(o1[0], o1[1]);
                        float2 _f2_12 = make_float2(o1[2], o1[3]);
                        float2 _f2_13 = make_float2(o1[4], o1[5]);
                        float2 _f2_14 = make_float2(o1[6], o1[7]);
                        float2 o1_lo01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_lo01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_10));
                        float2 o1_lo23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_lo23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_10));
                        float2 o1_hi01_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_hi01_scaled_f2) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&_f2_10));
                        float2 o1_hi23_scaled_f2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_hi23_scaled_f2) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_10));
                        o1[0] = o1_lo01_scaled_f2.x;
                        o1[1] = o1_lo01_scaled_f2.y;
                        o1[2] = o1_lo23_scaled_f2.x;
                        o1[3] = o1_lo23_scaled_f2.y;
                        o1[4] = o1_hi01_scaled_f2.x;
                        o1[5] = o1_hi01_scaled_f2.y;
                        o1[6] = o1_hi23_scaled_f2.x;
                        o1[7] = o1_hi23_scaled_f2.y;
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 4; h_3++) {
                            o1_lo[h_3] = o1[h_3];
                            o1_hi[h_3] = o1[h_3 + 4];
                        }
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
                    mbarrier_arrive(o_free_1_addr);
                }
                mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                _phase_corr_scale_0_0 ^= 1;
                mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                _phase_corr_scale_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scale0_pair[2];
                float scale1_pair[2];
                float local_sum_pair[2];
                const int idx0_c = col_pair_base_c;
                const int idx1_c = col_pair_base_c + 1;
                float _tmem_load_2[4];
                tmem_ld_x4(&_tmem_load_2[0], taddr + 16 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[4];
                tmem_ld_x4(&_tmem_load_3[0], taddr + 48 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                float2 _f2_15 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                float2 _f2_16 = make_float2(_tmem_load_3[0], _tmem_load_3[1]);
                float m0_0 = _tmem_load_2[2];
                float m0_1 = _tmem_load_2[3];
                float m1_0 = _tmem_load_3[2];
                float m1_1 = _tmem_load_3[3];
                float fm0;
                asm("max.f32 %0, %1, %2;" : "=f"(fm0) : "f"(m0_0), "f"(m1_0));
                float fm1;
                asm("max.f32 %0, %1, %2;" : "=f"(fm1) : "f"(m0_1), "f"(m1_1));
                float2 _f2_17 = make_float2(m0_0, m0_1);
                float2 _f2_18 = make_float2(m1_0, m1_1);
                float2 _f2_19 = make_float2(fm0, fm1);
                float2 max_diff_i0_f2 = sub_f32x2(_f2_17, _f2_19);
                float2 max_diff_i1_f2 = sub_f32x2(_f2_18, _f2_19);
                float2 _f2_20 = make_float2(softmax_scale_log2_1, softmax_scale_log2_1);
                float2 d0_pair_f2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d0_pair_f2) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&max_diff_i0_f2));
                float2 d1_pair_f2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d1_pair_f2) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&max_diff_i1_f2));
                float _exp2_8 = approx_exp2(d0_pair_f2.x);
                scale0_pair[0] = ((m0_0 == -CUDART_INF_F) ? 0.0f : _exp2_8);
                float _exp2_9 = approx_exp2(d0_pair_f2.y);
                scale0_pair[1] = ((m0_1 == -CUDART_INF_F) ? 0.0f : _exp2_9);
                float _exp2_10 = approx_exp2(d1_pair_f2.x);
                scale1_pair[0] = ((m1_0 == -CUDART_INF_F) ? 0.0f : _exp2_10);
                float _exp2_11_1 = approx_exp2(d1_pair_f2.y);
                scale1_pair[1] = ((m1_1 == -CUDART_INF_F) ? 0.0f : _exp2_11_1);
                float2 _f2_21 = make_float2(scale0_pair[0], scale0_pair[1]);
                float2 _f2_22 = make_float2(scale1_pair[0], scale1_pair[1]);
                float2 s1_scaled_pair_f2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&s1_scaled_pair_f2) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_22));
                float2 local_sum_pair_f2 = fma_f32x2_rn_ftz(_f2_15, _f2_21, s1_scaled_pair_f2);
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, local_sum_pair_f2.x, 16);
                float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, local_sum_pair_f2.y, 16);
                float2 sum16_pair_f2 = add_f32x2(local_sum_pair_f2, make_float2(_shfl_xor_sync_12, _shfl_xor_sync_13));
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum16_pair_f2.x, 8);
                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum16_pair_f2.y, 8);
                float2 sum8_pair_f2 = add_f32x2(sum16_pair_f2, make_float2(_shfl_xor_sync_14, _shfl_xor_sync_15));
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sum8_pair_f2.x, 4);
                float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, sum8_pair_f2.y, 4);
                float2 reduced_sum_pair_f2 = add_f32x2(sum8_pair_f2, make_float2(_shfl_xor_sync_16, _shfl_xor_sync_17));
                if (lane < 4) {
                    const int warp_sum_base_c = warp % 4 * 8 + col_pair_base_c;
                    smem_corr_reduce[warp_sum_base_c] = reduced_sum_pair_f2.x;
                    smem_corr_reduce[warp_sum_base_c + 1] = reduced_sum_pair_f2.y;
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                float2 _f2_23 = make_float2(smem_corr_reduce[idx0_c], smem_corr_reduce[idx1_c]);
                float2 _f2_24 = make_float2(smem_corr_reduce[idx0_c + 8], smem_corr_reduce[idx1_c + 8]);
                float2 _f2_25 = make_float2(smem_corr_reduce[idx0_c + 16], smem_corr_reduce[idx1_c + 16]);
                float2 _f2_26 = make_float2(smem_corr_reduce[idx0_c + 24], smem_corr_reduce[idx1_c + 24]);
                float2 sum_w01_pair_f2 = add_f32x2(_f2_23, _f2_24);
                float2 sum_w23_pair_f2 = add_f32x2(_f2_25, _f2_26);
                float2 sum_reduced_pair_f2 = add_f32x2(sum_w01_pair_f2, sum_w23_pair_f2);
                local_sum_pair[0] = sum_reduced_pair_f2.x;
                local_sum_pair[1] = sum_reduced_pair_f2.y;
                mbarrier_wait(o_done_0_addr, _phase_o_done_0_0);
                _phase_o_done_0_0 ^= 1;
                mbarrier_wait(o_done_1_addr, _phase_o_done_1_0);
                _phase_o_done_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float inv_sum_pair[2];
                #pragma unroll
                for (int c_4 = 0; c_4 < 2; c_4++) {
                    float _rcp_18 = approx_rcp(local_sum_pair[c_4]);
                    inv_sum_pair[c_4] = ((local_sum_pair[c_4] == 0.0f) ? 0.0f : _rcp_18);
                }
                float o0_lo_epi[4];
                float o0_hi_epi[4];
                float o1_lo_epi[4];
                float o1_hi_epi[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[3]))
                    : "r"(taddr + 80 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[3]))
                    : "r"(taddr + 80 + (unsigned int)corr_row + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[3]))
                    : "r"(taddr + 88 + (unsigned int)corr_row));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[3]))
                    : "r"(taddr + 88 + (unsigned int)corr_row + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (num_splits > 1) {
                    float part_vals[8];
                    float pscale0_e = scale0_pair[0] * inv_sum_pair[0];
                    float pscale0_o = scale0_pair[1] * inv_sum_pair[1];
                    float pscale1_e = scale1_pair[0] * inv_sum_pair[0];
                    float pscale1_o = scale1_pair[1] * inv_sum_pair[1];
                    part_vals[0] = __fmaf_rn(o0_lo_epi[0], pscale0_e, o1_lo_epi[0] * pscale1_e);
                    part_vals[1] = __fmaf_rn(o0_lo_epi[1], pscale0_o, o1_lo_epi[1] * pscale1_o);
                    part_vals[2] = __fmaf_rn(o0_lo_epi[2], pscale0_e, o1_lo_epi[2] * pscale1_e);
                    part_vals[3] = __fmaf_rn(o0_lo_epi[3], pscale0_o, o1_lo_epi[3] * pscale1_o);
                    part_vals[4] = __fmaf_rn(o0_hi_epi[0], pscale0_e, o1_hi_epi[0] * pscale1_e);
                    part_vals[5] = __fmaf_rn(o0_hi_epi[1], pscale0_o, o1_hi_epi[1] * pscale1_o);
                    part_vals[6] = __fmaf_rn(o0_hi_epi[2], pscale0_e, o1_hi_epi[2] * pscale1_e);
                    part_vals[7] = __fmaf_rn(o0_hi_epi[3], pscale0_o, o1_hi_epi[3] * pscale1_o);
                    int tile_empty_c = ((seqlen_kv <= split_start_block * BLOCK_N) ? 1 : 0);
                    if (tile_empty_c != 0) {
                        part_vals[0] = 0.0f;
                        part_vals[1] = 0.0f;
                        part_vals[2] = 0.0f;
                        part_vals[3] = 0.0f;
                        part_vals[4] = 0.0f;
                        part_vals[5] = 0.0f;
                        part_vals[6] = 0.0f;
                        part_vals[7] = 0.0f;
                    }
                    int corr_row_e = tmem_row_base_v_1 + lane / 4;
                    int q_even = col_pair_base_c;
                    int q_odd = col_pair_base_c + 1;
                    int head_base = batch_idx * NUM_Q_HEADS + kv_head_idx * group_ratio_rt;
                    int po_even = ((head_base + q_even) * num_splits + split_c) * HEAD_DIM;
                    int po_odd = ((head_base + q_odd) * num_splits + split_c) * HEAD_DIM;
                    if (q_even < group_ratio_rt) {
                        *(reinterpret_cast<float*>(partial_O + (po_even + corr_row_e)) + (0)) = part_vals[0];
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e + 8))) + (0)) = part_vals[2];
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e + 16))) + (0)) = part_vals[4];
                        *(reinterpret_cast<float*>(partial_O + (po_even + (corr_row_e + 24))) + (0)) = part_vals[6];
                    }
                    if (q_odd < group_ratio_rt) {
                        *(reinterpret_cast<float*>(partial_O + (po_odd + corr_row_e)) + (0)) = part_vals[1];
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e + 8))) + (0)) = part_vals[3];
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e + 16))) + (0)) = part_vals[5];
                        *(reinterpret_cast<float*>(partial_O + (po_odd + (corr_row_e + 24))) + (0)) = part_vals[7];
                    }
                    if (corr_tid < 4) {
                        int st_even = (head_base + q_even) * num_splits + split_c;
                        int st_odd = (head_base + q_odd) * num_splits + split_c;
                        if (q_even < group_ratio_rt) {
                            *(reinterpret_cast<float*>(partial_max + st_even) + (0)) = ((tile_empty_c != 0) ? -CUDART_INF_F : fm0 * softmax_scale_log2_1);
                            *(reinterpret_cast<float*>(partial_sum + st_even) + (0)) = ((tile_empty_c != 0) ? 0.0f : local_sum_pair[0]);
                        }
                        if (q_odd < group_ratio_rt) {
                            *(reinterpret_cast<float*>(partial_max + st_odd) + (0)) = ((tile_empty_c != 0) ? -CUDART_INF_F : fm1 * softmax_scale_log2_1);
                            *(reinterpret_cast<float*>(partial_sum + st_odd) + (0)) = ((tile_empty_c != 0) ? 0.0f : local_sum_pair[1]);
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(stats_empty_addr);
                    }
                } else {
                    float out_vals[8];
                    float2 _f32x2_mul_a_19 = make_float2(inv_sum_pair[0], inv_sum_pair[1]);
                    float2 _f32x2_mul_b_20 = make_float2(output_scale, output_scale);
                    float2 norm_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&norm_pair_f2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
                    float2 final_scale0_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&final_scale0_pair_f2) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&norm_pair_f2));
                    float2 final_scale1_pair_f2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&final_scale1_pair_f2) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&norm_pair_f2));
                    float2 _f2_27 = make_float2(o0_lo_epi[0], o0_lo_epi[1]);
                    float2 _f32x2_mul_a_21 = make_float2(o1_lo_epi[0], o1_lo_epi[1]);
                    float2 o1_lo01_scaled_f2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_lo01_scaled_f2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                    float2 out_lo01_f2 = fma_f32x2_rn_ftz(_f2_27, final_scale0_pair_f2, o1_lo01_scaled_f2_1);
                    out_vals[0] = out_lo01_f2.x;
                    out_vals[1] = out_lo01_f2.y;
                    float2 _f2_28 = make_float2(o0_lo_epi[2], o0_lo_epi[3]);
                    float2 _f32x2_mul_a_22 = make_float2(o1_lo_epi[2], o1_lo_epi[3]);
                    float2 o1_lo23_scaled_f2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_lo23_scaled_f2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                    float2 out_lo23_f2 = fma_f32x2_rn_ftz(_f2_28, final_scale0_pair_f2, o1_lo23_scaled_f2_1);
                    out_vals[2] = out_lo23_f2.x;
                    out_vals[3] = out_lo23_f2.y;
                    float2 _f2_29 = make_float2(o0_hi_epi[0], o0_hi_epi[1]);
                    float2 _f32x2_mul_a_23 = make_float2(o1_hi_epi[0], o1_hi_epi[1]);
                    float2 o1_hi01_scaled_f2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_hi01_scaled_f2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                    float2 out_hi01_f2 = fma_f32x2_rn_ftz(_f2_29, final_scale0_pair_f2, o1_hi01_scaled_f2_1);
                    out_vals[4] = out_hi01_f2.x;
                    out_vals[5] = out_hi01_f2.y;
                    float2 _f2_30 = make_float2(o0_hi_epi[2], o0_hi_epi[3]);
                    float2 _f32x2_mul_a_24 = make_float2(o1_hi_epi[2], o1_hi_epi[3]);
                    float2 o1_hi23_scaled_f2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o1_hi23_scaled_f2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                    float2 out_hi23_f2 = fma_f32x2_rn_ftz(_f2_30, final_scale0_pair_f2, o1_hi23_scaled_f2_1);
                    out_vals[6] = out_hi23_f2.x;
                    out_vals[7] = out_hi23_f2.y;
                    if (seqlen_kv <= split_start_block * BLOCK_N) {
                        out_vals[0] = 0.0f;
                        out_vals[1] = 0.0f;
                        out_vals[2] = 0.0f;
                        out_vals[3] = 0.0f;
                        out_vals[4] = 0.0f;
                        out_vals[5] = 0.0f;
                        out_vals[6] = 0.0f;
                        out_vals[7] = 0.0f;
                    }
                    unsigned int regs_o[2];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(out_vals[0]), "f"(out_vals[1]),
                                               "f"(out_vals[2]), "f"(out_vals[3]));
                        regs_o[0] = _packed;
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
                            : "=r"(_packed) : "f"(out_vals[4]), "f"(out_vals[5]),
                                               "f"(out_vals[6]), "f"(out_vals[7]));
                        regs_o[1] = _packed;
                    }
                    int o_mtx_idx = lane / 8;
                    int o_thr_row_idx = lane % 8;
                    int o_seg_col_idx = warp % 4 * 2 + o_mtx_idx ^ o_thr_row_idx;
                    int o_stsm_offset = o_thr_row_idx * 128 + o_seg_col_idx * 16;
                    const void* _stmatrix_b8_ptr_25 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_o) + o_stsm_offset);
                    uint64_t _stmatrix_b8_addr64_25;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_25) : "l"(_stmatrix_b8_ptr_25));
                    uint32_t _stmatrix_b8_addr_25;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_25) : "l"(_stmatrix_b8_addr64_25));
                    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                        :: "r"(_stmatrix_b8_addr_25), "r"(regs_o[0]), "r"(regs_o[1])
                        : "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    int copy_base = corr_tid * 16;
                    int copy_row = copy_base / 128;
                    int copy_col = copy_base % 128;
                    int copy_smem_offset = copy_base ^ copy_row % 8 * 16;
                    if (copy_row < group_ratio_rt) {
                        unsigned int copy_vec[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 3]))
                            : "r"(smem_o_addr + (unsigned int)copy_smem_offset));
                        int o_base = (batch_idx * NUM_Q_HEADS + kv_head_idx * group_ratio_rt) * HEAD_DIM;
                        reinterpret_cast<int4*>(O + (o_base + copy_row * HEAD_DIM + copy_col))[0] = reinterpret_cast<int4*>(copy_vec)[0];
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(stats_empty_addr);
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int total_tiles_m = BATCH_SIZE * NUM_KV_HEADS * num_splits;
            int kv_k_m = 0;
            int kv_v_m = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_s_empty_0_0 = 0;
            unsigned int _phase_s_empty_1_0 = 0;
            unsigned int _phase_o_free_0_0 = 0;
            unsigned int _phase_o_free_1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_m = blockIdx.x; tile_idx_m < total_tiles_m; tile_idx_m += gridDim.x) {
                int first_pv0_m = 1;
                int first_pv1_m = 1;
                int split_m = tile_idx_m % (unsigned int)num_splits;
                int bh_m = tile_idx_m / (unsigned int)num_splits;
                int batch_idx_m = bh_m / NUM_KV_HEADS;
                int seqlen_kv_m = seq_lens_kv[batch_idx_m];
                int num_n_blocks_total_m = (seqlen_kv_m + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_m = num_n_blocks_total_m + num_n_blocks_total_m % 2;
                int split_start_block_m = split_m * blocks_per_split;
                int rem_blocks_m = even_n_blocks_m - split_start_block_m;
                int capped_blocks_m = ((rem_blocks_m > blocks_per_split) ? blocks_per_split : rem_blocks_m);
                int cta_n_blocks_m = ((capped_blocks_m < 2) ? 2 : capped_blocks_m);
                #pragma unroll 1
                for (int round_idx_m = 0; round_idx_m < cta_n_blocks_m; round_idx_m++) {
                    int a_is_v_m = 0;
                    int a_blk_m = 0;
                    int b_is_v_m = 0;
                    int b_blk_m = 1;
                    if (round_idx_m > 0) {
                        if (round_idx_m < cta_n_blocks_m - 1) {
                            a_blk_m = round_idx_m + 1;
                            b_is_v_m = 1;
                            b_blk_m = round_idx_m - 1;
                        } else {
                            a_is_v_m = 1;
                            a_blk_m = cta_n_blocks_m - 2;
                            b_is_v_m = 1;
                            b_blk_m = cta_n_blocks_m - 1;
                        }
                    }
                    #pragma unroll
                    for (int issue_i_m = 0; issue_i_m < 2; issue_i_m++) {
                        int issue_is_v_m = a_is_v_m;
                        int issue_blk_m = a_blk_m;
                        if (issue_i_m == 1) {
                            issue_is_v_m = b_is_v_m;
                            issue_blk_m = b_blk_m;
                        }
                        int issue_stage_m = ((issue_is_v_m == 1) ? kv_v_m : kv_k_m);
                        if (issue_is_v_m == 0) {
                            mbarrier_wait(kv_full_addr + (kv_k_m) * 8, 0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (issue_blk_m == 0) {
                                mbarrier_wait(q_full_addr, _phase_q_full_0);
                                _phase_q_full_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (issue_blk_m % 2 == 0) {
                                if (issue_blk_m >= 2) {
                                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                                    _phase_s_empty_0_0 ^= 1;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                }
                                int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 134348816;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_s0), "r"(_mma_b_lo_0), "r"(tmem_tmem_kv + issue_stage_m * 32), "r"(0) : "memory");
                                elect_commit(kv_empty_addr + (kv_k_m) * 8);
                                elect_commit(s_full_0_addr);
                            } else {
                                if (issue_blk_m >= 2) {
                                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                                    _phase_s_empty_1_0 ^= 1;
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                }
                                int _mma_b_lo_1 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 134348816;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_s1), "r"(_mma_b_lo_1), "r"(tmem_tmem_kv + issue_stage_m * 32), "r"(0) : "memory");
                                elect_commit(kv_empty_addr + (kv_k_m) * 8);
                                elect_commit(s_full_1_addr);
                            }
                            if (issue_blk_m == cta_n_blocks_m - 1) {
                                elect_commit(q_empty_addr);
                            }
                            kv_k_m += 1;
                            if (kv_k_m == 4) { kv_k_m = 0; }
                        } else {
                            mbarrier_wait(kv_full_addr + (kv_v_m) * 8, 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (issue_blk_m % 2 == 0) {
                                if (issue_blk_m >= cta_n_blocks_m - 2) {
                                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                                    _phase_s_empty_0_0 ^= 1;
                                }
                                mbarrier_wait(o_free_0_addr, _phase_o_free_0_0);
                                _phase_o_free_0_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134348816;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_o0), "r"(_mma_b_lo_2), "r"(tmem_tmem_kv + issue_stage_m * 32), "r"(((first_pv0_m) ? 0 : 1)) : "memory");
                                elect_commit2(kv_empty_addr + (kv_v_m) * 8, o_done_0_addr);
                                elect_commit(p_empty_0_addr);
                                first_pv0_m = 0;
                            } else {
                                if (issue_blk_m >= cta_n_blocks_m - 2) {
                                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                                    _phase_s_empty_1_0 ^= 1;
                                }
                                mbarrier_wait(o_free_1_addr, _phase_o_free_1_0);
                                _phase_o_free_1_0 ^= 1;
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_b_lo_3 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
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
                    "mov.b32 id, 134348816;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_o1), "r"(_mma_b_lo_3), "r"(tmem_tmem_kv + issue_stage_m * 32), "r"(((first_pv1_m) ? 0 : 1)) : "memory");
                                elect_commit2(kv_empty_addr + (kv_v_m) * 8, o_done_1_addr);
                                elect_commit(p_empty_1_addr);
                                first_pv1_m = 0;
                            }
                            kv_v_m += 1;
                            if (kv_v_m == 4) { kv_v_m = 0; }
                        }
                    }
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 9) {
        { // page_offsets_main
            int pg_slot_p = 0;
            unsigned int total_tiles_p = BATCH_SIZE * NUM_KV_HEADS * num_splits;
            unsigned int _phase_pg_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_p = blockIdx.x; tile_idx_p < total_tiles_p; tile_idx_p += gridDim.x) {
                int split_p = tile_idx_p % (unsigned int)num_splits;
                int bh_p = tile_idx_p / (unsigned int)num_splits;
                int batch_idx_p = bh_p / NUM_KV_HEADS;
                int seqlen_kv_p = seq_lens_kv[batch_idx_p];
                int num_n_blocks_total_p = (seqlen_kv_p + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_p = num_n_blocks_total_p + num_n_blocks_total_p % 2;
                int split_start_p = split_p * blocks_per_split;
                int rem_blocks_p = even_n_blocks_p - split_start_p;
                int capped_blocks_p = ((rem_blocks_p > blocks_per_split) ? blocks_per_split : rem_blocks_p);
                int cta_n_blocks_p = ((capped_blocks_p < 2) ? 2 : capped_blocks_p);
                int pt_base_k_p = batch_idx_p * pt_batch_stride;
                int pt_base_v_p = pt_base_k_p + pt_v_offset;
                #pragma unroll 1
                for (int round_p = 0; round_p < cta_n_blocks_p; round_p++) {
                    int a_is_v_p = 0;
                    int a_blk_p = 0;
                    int b_is_v_p = 0;
                    int b_blk_p = 1;
                    if (round_p > 0) {
                        if (round_p < cta_n_blocks_p - 1) {
                            a_blk_p = round_p + 1;
                            b_is_v_p = 1;
                            b_blk_p = round_p - 1;
                        } else {
                            a_is_v_p = 1;
                            a_blk_p = cta_n_blocks_p - 2;
                            b_is_v_p = 1;
                            b_blk_p = cta_n_blocks_p - 1;
                        }
                    }
                    mbarrier_wait(pg_empty_addr + (pg_slot_p) * 8, _phase_pg_empty);
                    const int pages_per_block_p = BLOCK_N / PAGE_SIZE;
                    if (lane < 2 * pages_per_block_p) {
                        int task_i_p = lane / pages_per_block_p;
                        int page_i_p = lane % pages_per_block_p;
                        int task_is_v_p = a_is_v_p;
                        int task_blk_p = a_blk_p;
                        if (task_i_p == 1) {
                            task_is_v_p = b_is_v_p;
                            task_blk_p = b_blk_p;
                        }
                        int n_block_p = split_start_p + cta_n_blocks_p - 1 - task_blk_p;
                        int pt_row_p = ((task_is_v_p == 1) ? pt_base_v_p : pt_base_k_p);
                        int pg_value_p = page_table[pt_row_p + n_block_p * pages_per_block_p + page_i_p];
                        int pg_dst_p = smem_pg_addr + (unsigned int)(pg_slot_p * 64) + (unsigned int)((task_i_p * 8 + page_i_p) * 4);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(pg_dst_p), "r"(pg_value_p));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(pg_full_addr + (pg_slot_p) * 8);
                    pg_slot_p += 1;
                    if (pg_slot_p == 4) { pg_slot_p = 0; _phase_pg_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
        { // load_warp_main
            int pk_slot_l = 0;
            int pg_slot_l = 0;
            unsigned int total_tiles_l = BATCH_SIZE * NUM_KV_HEADS * num_splits;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_pk_empty = 1;
            unsigned int _phase_pg_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_l = blockIdx.x; tile_idx_l < total_tiles_l; tile_idx_l += gridDim.x) {
                int split_l = tile_idx_l % (unsigned int)num_splits;
                int bh_l = tile_idx_l / (unsigned int)num_splits;
                int batch_idx_l = bh_l / NUM_KV_HEADS;
                int kv_head_idx_l = bh_l % NUM_KV_HEADS;
                int seqlen_kv_l = seq_lens_kv[batch_idx_l];
                int num_n_blocks_total_l = (seqlen_kv_l + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_l = num_n_blocks_total_l + num_n_blocks_total_l % 2;
                int split_start_l = split_l * blocks_per_split;
                int rem_blocks_l = even_n_blocks_l - split_start_l;
                int capped_blocks_l = ((rem_blocks_l > blocks_per_split) ? blocks_per_split : rem_blocks_l);
                int cta_n_blocks_l = ((capped_blocks_l < 2) ? 2 : capped_blocks_l);
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    int off_qt_l = (batch_idx_l * NUM_KV_HEADS + kv_head_idx_l) * TILE_Q;
                    mbarrier_arrive_expect_tx(q_full_addr, TILE_Q * HEAD_DIM);
                    tma_3d_gmem2smem(smem_qt_addr, (&Qt), 0, off_qt_l, 0, q_full_addr);
                    #pragma unroll 1
                    for (int round_l = 0; round_l < cta_n_blocks_l; round_l++) {
                        int a_is_v_l = 0;
                        int a_blk_l = 0;
                        int b_is_v_l = 0;
                        int b_blk_l = 1;
                        if (round_l > 0) {
                            if (round_l < cta_n_blocks_l - 1) {
                                a_blk_l = round_l + 1;
                                b_is_v_l = 1;
                                b_blk_l = round_l - 1;
                            } else {
                                a_is_v_l = 1;
                                a_blk_l = cta_n_blocks_l - 2;
                                b_is_v_l = 1;
                                b_blk_l = cta_n_blocks_l - 1;
                            }
                        }
                        mbarrier_wait(pk_empty_addr + (pk_slot_l) * 8, _phase_pk_empty);
                        mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, _phase_pg_full);
                        mbarrier_arrive_expect_tx(pk_full_addr + (pk_slot_l) * 8, 18432);
                        int round_dst_l = smem_pk_addr + (unsigned int)(pk_slot_l * 34816);
                        #pragma unroll
                        for (int task_i_l = 0; task_i_l < 2; task_i_l++) {
                            int task_is_v_l = a_is_v_l;
                            int task_blk_l = a_blk_l;
                            if (task_i_l == 1) {
                                task_is_v_l = b_is_v_l;
                                task_blk_l = b_blk_l;
                            }
                            int pg_l[8];
                            int task_pg_base_l = smem_pg_addr + (unsigned int)(pg_slot_l * 64) + (unsigned int)(task_i_l * 8 * 4);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pg_l[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_l[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_l[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_l[(0) + 3]))
                                : "r"(task_pg_base_l));
                            {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&pg_l[4])), "=r"(*reinterpret_cast<uint32_t*>(&pg_l[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_l[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_l[(4) + 3]))
                                    : "r"(task_pg_base_l + 16));
                            }
                            int task_dst_l = round_dst_l + task_i_l * 17408;
                            if (task_is_v_l == 1) {
                                #pragma unroll
                                for (int pg_i_l = 0; pg_i_l < 8; pg_i_l++) {
                                    int vpg_l = pg_l[pg_i_l / (PAGE_SIZE / 16)];
                                    int vsub_l = pg_i_l % (PAGE_SIZE / 16) * 16;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(task_dst_l + pg_i_l * 2048), "l"((&Vp)), "r"(0), "r"(vsub_l), "r"(0), "r"(kv_head_idx_l), "r"(vpg_l),
                                           "r"(pk_full_addr + (pk_slot_l) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(task_dst_l + 16384 + pg_i_l * 128), "l"((&Vsf)), "r"(vsub_l * 8), "r"(kv_head_idx_l), "r"(vpg_l),
                                           "r"(pk_full_addr + (pk_slot_l) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            } else {
                                #pragma unroll
                                for (int pg_i_l_1 = 0; pg_i_l_1 < 8; pg_i_l_1++) {
                                    int kpg_l = pg_l[pg_i_l_1 / (PAGE_SIZE / 16)];
                                    int ksub_l = pg_i_l_1 % (PAGE_SIZE / 16) * 16;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(task_dst_l + pg_i_l_1 * 2048), "l"((&Kp)), "r"(0), "r"(ksub_l), "r"(0), "r"(kv_head_idx_l), "r"(kpg_l),
                                           "r"(pk_full_addr + (pk_slot_l) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(task_dst_l + 16384 + pg_i_l_1 * 128), "l"((&Ksf)), "r"(ksub_l * 8), "r"(kv_head_idx_l), "r"(kpg_l),
                                           "r"(pk_full_addr + (pk_slot_l) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                        mbarrier_arrive(pg_empty_addr + (pg_slot_l) * 8);
                        pk_slot_l += 1;
                        if (pk_slot_l == 4) { pk_slot_l = 0; _phase_pk_empty ^= 1; }
                        pg_slot_l += 1;
                        if (pg_slot_l == 4) { pg_slot_l = 0; _phase_pg_full ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // compute_main
            int pk_slot_c = 0;
            int kv_k_c = 0;
            int kv_v_c = 0;
            unsigned int total_tiles_w = BATCH_SIZE * NUM_KV_HEADS * num_splits;
            unsigned int _phase_pk_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_w = blockIdx.x; tile_idx_w < total_tiles_w; tile_idx_w += gridDim.x) {
                int split_w = tile_idx_w % (unsigned int)num_splits;
                int bh_w = tile_idx_w / (unsigned int)num_splits;
                int batch_idx_1 = bh_w / NUM_KV_HEADS;
                int kv_head_idx_1 = bh_w % NUM_KV_HEADS;
                int seqlen_kv_1 = seq_lens_kv[batch_idx_1];
                int num_n_blocks_total_1 = (seqlen_kv_1 + BLOCK_N - 1) / BLOCK_N;
                int even_n_blocks_1 = num_n_blocks_total_1 + num_n_blocks_total_1 % 2;
                int split_start_block_1 = split_w * blocks_per_split;
                int rem_blocks_1 = even_n_blocks_1 - split_start_block_1;
                int capped_blocks_1 = ((rem_blocks_1 > blocks_per_split) ? blocks_per_split : rem_blocks_1);
                int cta_n_blocks_1 = ((capped_blocks_1 < 2) ? 2 : capped_blocks_1);
                #pragma unroll 1
                for (int round_idx = 0; round_idx < cta_n_blocks_1; round_idx++) {
                    int a_is_v = 0;
                    int a_blk = 0;
                    int b_is_v = 0;
                    int b_blk = 1;
                    if (round_idx > 0) {
                        if (round_idx < cta_n_blocks_1 - 1) {
                            a_blk = round_idx + 1;
                            b_is_v = 1;
                            b_blk = round_idx - 1;
                        } else {
                            a_is_v = 1;
                            a_blk = cta_n_blocks_1 - 2;
                            b_is_v = 1;
                            b_blk = cta_n_blocks_1 - 1;
                        }
                    }
                    mbarrier_wait(pk_full_addr + (pk_slot_c) * 8, _phase_pk_full);
                    int work_block0 = (warp - 12) * 2;
                    #pragma unroll
                    for (int work_task_index = 0; work_task_index < 2; work_task_index++) {
                        int work_is_v = a_is_v;
                        if (work_task_index == 1) {
                            work_is_v = b_is_v;
                        }
                        int work_stage = ((work_is_v == 1) ? kv_v_c : kv_k_c);
                        if (work_is_v == 1) {
                            uint32_t _mbar_token_0 = mbarrier_try_wait(kv_full_addr + (kv_v_c) * 8, 0);
                            unsigned int v_reuse_ready = _mbar_token_0;
                            mbarrier_wait(kv_empty_addr + (kv_v_c) * 8, 0);
                            mbarrier_wait_token(kv_full_addr + (kv_v_c) * 8, 0, v_reuse_ready);
                        } else {
                            uint32_t _mbar_token_1 = mbarrier_try_wait(kv_full_addr + (kv_k_c) * 8, 1);
                            unsigned int k_reuse_ready = _mbar_token_1;
                            mbarrier_wait(kv_empty_addr + (kv_k_c) * 8, 1);
                            mbarrier_wait_token(kv_full_addr + (kv_k_c) * 8, 1, k_reuse_ready);
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int work_pk_base = smem_pk_addr + (unsigned int)(pk_slot_c * 34816) + (unsigned int)(work_task_index * 17408);
                        int work_sf_base = work_pk_base + 16384;
                        if (work_is_v == 0) {
                            unsigned int k_src_word0[16];
                            unsigned int k_src_word1[16];
                            unsigned int ksf_words0[4];
                            unsigned int ksf_words1[4];
                            #pragma unroll
                            for (int atom_i = 0; atom_i < 2; atom_i++) {
                                int atom_base = ((atom_i == 0) ? work_block0 * 16 : (work_block0 + 1) * 16);
                                int ksf_row0 = atom_base + lane / 4;
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words0[atom_i * 2])) : "r"(work_sf_base + ksf_row0 * 8));
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words0[atom_i * 2 + 1])) : "r"(work_sf_base + ksf_row0 * 8 + 4));
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words1[atom_i * 2])) : "r"(work_sf_base + (ksf_row0 + 8) * 8));
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ksf_words1[atom_i * 2 + 1])) : "r"(work_sf_base + (ksf_row0 + 8) * 8 + 4));
                                #pragma unroll
                                for (int ldsm_stage = 0; ldsm_stage < 8; ldsm_stage++) {
                                    const int slot = atom_i * 8 + ldsm_stage;
                                    int krow = atom_base + lane;
                                    int kbyte = krow * 128 + ldsm_stage * 16;
                                    kbyte = kbyte ^ krow % 8 * 16;
                                    asm volatile("ldmatrix.sync.aligned.shared::cta.m8n16.x2.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                                        : "=r"(k_src_word0[slot]), "=r"(k_src_word1[slot])
                                        : "r"(work_pk_base + kbyte)
                                        : "memory");
                                }
                            }
                            #pragma unroll
                            for (int atom_i_1 = 0; atom_i_1 < 2; atom_i_1++) {
                                int atom_base_1 = ((atom_i_1 == 0) ? work_block0 * 16 : (work_block0 + 1) * 16);
                                #pragma unroll
                                for (int ldsm_stage_1 = 0; ldsm_stage_1 < 8; ldsm_stage_1++) {
                                    const int slot_1 = atom_i_1 * 8 + ldsm_stage_1;
                                    const int scale_slot = atom_i_1 * 2 + ldsm_stage_1 / 4;
                                    unsigned int dst_word0 = 0;
                                    unsigned int dst_word1 = 0;
                                    if (ldsm_stage_1 % 4 == 0) {
                                        dst_word0 = qmul4_b<0>(k_src_word0[slot_1], ksf_words0[scale_slot]);
                                        dst_word1 = qmul4_b<0>(k_src_word1[slot_1], ksf_words1[scale_slot]);
                                    } else if (ldsm_stage_1 % 4 == 1) {
                                        dst_word0 = qmul4_b<1>(k_src_word0[slot_1], ksf_words0[scale_slot]);
                                        dst_word1 = qmul4_b<1>(k_src_word1[slot_1], ksf_words1[scale_slot]);
                                    } else {
                                        if (ldsm_stage_1 % 4 == 2) {
                                            dst_word0 = qmul4_b<2>(k_src_word0[slot_1], ksf_words0[scale_slot]);
                                            dst_word1 = qmul4_b<2>(k_src_word1[slot_1], ksf_words1[scale_slot]);
                                        } else {
                                            dst_word0 = qmul4_b<3>(k_src_word0[slot_1], ksf_words0[scale_slot]);
                                            dst_word1 = qmul4_b<3>(k_src_word1[slot_1], ksf_words1[scale_slot]);
                                        }
                                    }
                                    unsigned int store_words[2];
                                    store_words[0] = dst_word0;
                                    store_words[1] = dst_word1;
                                    int taddr_atom = taddr + 96 + (unsigned int)(warp % 4 * 32 + atom_i_1 * 16 << 16) + (unsigned int)(work_stage * 32 + ldsm_stage_1 * 4);
                                    asm volatile(
                                        "tcgen05.st.sync.aligned.16x128b.x1.b32"
                                        " [%0], {%1, %2};"
                                        :: "r"(taddr_atom), "r"(*reinterpret_cast<const uint32_t*>(&store_words[0])), "r"(*reinterpret_cast<const uint32_t*>(&store_words[1])));
                                }
                            }
                        } else {
                            unsigned int v_src_word0[16];
                            unsigned int v_src_word1[16];
                            unsigned int vsf_words[16];
                            int vsf_group0 = work_block0 * 16 / 16;
                            #pragma unroll
                            for (int ldsm_stage_2 = 0; ldsm_stage_2 < 8; ldsm_stage_2++) {
                                int vrow = ldsm_stage_2 * 16 + lane;
                                #pragma unroll
                                for (int atom_i_2 = 0; atom_i_2 < 2; atom_i_2++) {
                                    const int slot_2 = atom_i_2 * 8 + ldsm_stage_2;
                                    int atom_base_2 = ((atom_i_2 == 0) ? work_block0 * 16 : (work_block0 + 1) * 16);
                                    int vbyte = vrow * 128 + atom_base_2;
                                    vbyte = vbyte ^ vrow % 8 * 16;
                                    asm volatile("ldmatrix.sync.aligned.shared::cta.m16n16.x1.trans.b8x16.b4x16_p64 {%0, %1}, [%2];\n"
                                        : "=r"(v_src_word0[slot_2]), "=r"(v_src_word1[slot_2])
                                        : "r"(work_pk_base + vbyte)
                                        : "memory");
                                }
                                int vsf_addr = work_sf_base + ldsm_stage_2 * 128 + lane % 4 * 32 + vsf_group0 * 4;
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[ldsm_stage_2 * 2])) : "r"(vsf_addr));
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&vsf_words[ldsm_stage_2 * 2 + 1])) : "r"(vsf_addr + 4));
                            }
                            #pragma unroll
                            for (int atom_i_3 = 0; atom_i_3 < 2; atom_i_3++) {
                                int atom_base_3 = ((atom_i_3 == 0) ? work_block0 * 16 : (work_block0 + 1) * 16);
                                #pragma unroll
                                for (int ldsm_stage_3 = 0; ldsm_stage_3 < 8; ldsm_stage_3++) {
                                    const int slot_3 = atom_i_3 * 8 + ldsm_stage_3;
                                    const int scale_slot_1 = ldsm_stage_3 * 2 + atom_i_3;
                                    unsigned int dst_word0_1 = 0;
                                    unsigned int dst_word1_1 = 0;
                                    dst_word0_1 = qmul4_direct(v_src_word0[slot_3], vsf_words[scale_slot_1]);
                                    dst_word1_1 = qmul4_direct(v_src_word1[slot_3], vsf_words[scale_slot_1]);
                                    unsigned int store_words_1[2];
                                    store_words_1[0] = dst_word0_1;
                                    store_words_1[1] = dst_word1_1;
                                    int taddr_atom_1 = taddr + 96 + (unsigned int)(warp % 4 * 32 + atom_i_3 * 16 << 16) + (unsigned int)(work_stage * 32 + ldsm_stage_3 * 4);
                                    asm volatile(
                                        "tcgen05.st.sync.aligned.16x128b.x1.b32"
                                        " [%0], {%1, %2};"
                                        :: "r"(taddr_atom_1), "r"(*reinterpret_cast<const uint32_t*>(&store_words_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&store_words_1[1])));
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (work_is_v == 1) {
                            mbarrier_arrive(kv_full_addr + (kv_v_c) * 8);
                            kv_v_c += 1;
                            if (kv_v_c == 4) { kv_v_c = 0; }
                        } else {
                            mbarrier_arrive(kv_full_addr + (kv_k_c) * 8);
                            kv_k_c += 1;
                            if (kv_k_c == 4) { kv_k_c = 0; }
                        }
                    }
                    mbarrier_arrive(pk_empty_addr + (pk_slot_c) * 8);
                    pk_slot_c += 1;
                    if (pk_slot_c == 4) { pk_slot_c = 0; _phase_pk_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
