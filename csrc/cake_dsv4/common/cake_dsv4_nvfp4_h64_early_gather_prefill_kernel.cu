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

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
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

__device__ __forceinline__ uint32_t qmul4_lower4(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("{\n.reg .b16 h0, h1;\nmov.b32 {h0, h1}, %1;\nmul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, h0, %2;\n}\n"
      : "=r"(y)
      : "r"(src), "r"(qmul4_broadcast<0>(scale)));
  return y;
}

__device__ __forceinline__ uint32_t qmul4_higher4(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("{\n.reg .b16 h0, h1;\nmov.b32 {h0, h1}, %1;\nmul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, h1, %2;\n}\n"
      : "=r"(y)
      : "r"(src), "r"(qmul4_broadcast<0>(scale)));
  return y;
}

extern "C" {

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_h64_early_gather_prefill(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int has_sinks, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens, const __grid_constant__ CUtensorMap tmap_o)
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
    #define q_load_full_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 8)
    #define q_empty_addr (mbar_base + 16)
    #define index_full_addr (mbar_base + 24)
    #define index_empty_addr (mbar_base + 72)
    #define k_full_addr (mbar_base + 120)
    #define k_empty_addr (mbar_base + 144)
    #define kstage_free_addr (mbar_base + 168)
    #define sf_full_addr (mbar_base + 192)
    #define kexp_full_addr (mbar_base + 216)
    #define v_full_addr (mbar_base + 232)
    #define v_empty_addr (mbar_base + 248)
    #define s_full_addr (mbar_base + 264)
    #define s_empty_addr (mbar_base + 280)
    #define p_full_addr (mbar_base + 296)
    #define p_recycle_addr (mbar_base + 312)
    #define stats_addr (mbar_base + 328)
    #define o_empty_addr (mbar_base + 344)
    #define o_full_addr (mbar_base + 360)
    #define sum_ready_addr (mbar_base + 376)
    #define sum_empty_addr (mbar_base + 384)
    #define tmem_dealloc_addr (mbar_base + 392)
    #define work_full_addr (mbar_base + 400)
    #define work_empty_addr (mbar_base + 416)
    #define throttle_full_addr (mbar_base + 432)
    #define throttle_empty_addr (mbar_base + 448)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* smem_q_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_a_addr = smem + 1024;
    uint8_t* smem_q_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_q_b_addr = smem + 17408;
    __nv_bfloat16* smem_q_rope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_q_rope_addr = smem + 33792;
    unsigned int* smem_q_sf = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int smem_q_sf_addr = smem + 50176;
    uint8_t* smem_k_a = reinterpret_cast<uint8_t*>(smem_raw + 52224);
    const int smem_k_a_addr = smem + 52224;
    uint8_t* smem_k_b = reinterpret_cast<uint8_t*>(smem_raw + 60416);
    const int smem_k_b_addr = smem + 60416;
    __nv_bfloat16* smem_k_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_k_c_addr = smem + 68608;
    unsigned int* smem_k_sf = reinterpret_cast<unsigned int*>(smem_raw + 76800);
    const int smem_k_sf_addr = smem + 76800;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_v_addr = smem + 132096;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_p_addr = smem + 197632;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 205824);
    const int smem_indices_addr = smem + 205824;
    float* smem_scale = reinterpret_cast<float*>(smem_raw + 211968);
    const int smem_scale_addr = smem + 211968;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 212992);
    const int smem_exch_addr = smem + 212992;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 214016);
    const int smem_sum_addr = smem + 214016;
    float* smem_fmax = reinterpret_cast<float*>(smem_raw + 214528);
    const int smem_fmax_addr = smem + 214528;
    unsigned int* smem_kexp = reinterpret_cast<unsigned int*>(smem_raw + 215040);
    const int smem_kexp_addr = smem + 215040;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 215072);
    const int work_response_addr = smem + 215072;

    // Kernel setup ops
    if (warp == 16 && lane < 5) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(((lane == 4) ? sparse_topk_lens + blockIdx.x : swa_indices + ((int)blockIdx.x * swa_index_stride + lane * 32)))));
    }

    // Mbarrier init (58 barriers at smem_raw[0..464))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_load_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'index_pipe' ---
            // index_full: 6 barriers, init_count=32
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            mbarrier_init(smem + 64, 32);
            // index_empty: 6 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            mbarrier_init(smem + 80, 256);
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // kstage_free: 3 barriers, init_count=2
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            mbarrier_init(smem + 184, 2);
            // sf_full: 3 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 232, 2);
            mbarrier_init(smem + 240, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            mbarrier_init(smem + 304, 128);
            // p_recycle: 2 barriers, init_count=1
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 376, 128);
            // sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 384, 128);
            // tmem_dealloc: 1 barriers, init_count=448
            mbarrier_init(smem + 392, 448);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            // work_empty: 2 barriers, init_count=736
            mbarrier_init(smem + 416, 736);
            mbarrier_init(smem + 424, 736);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 432, 128);
            mbarrier_init(smem + 440, 128);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 448, 32);
            mbarrier_init(smem + 456, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 464);
    if (warp == 0) {
        int _tmem_hold = smem + 464;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem_scratch = taddr;
    const unsigned int tmem_tmem_sfa = taddr + 384;
    const unsigned int tmem_tmem_sfb = taddr + 416;
    const unsigned int tmem_tmem_qa0 = taddr + 448;
    const unsigned int tmem_tmem_qa1 = taddr + 480;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 120;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.442695f;
            const int sm_row_base = warp % 4 * 32;
            const int sm_thread = sm_row_base + lane;
            const int sm_head = sm_thread & 63;
            const int sm_half = sm_thread >> 6;
            int sm_tile_cursor = 0;
            int sm_item = 0;
            unsigned int sm_index_stage = 0;
            unsigned int sm_work_stage = 0;
            int sm_query = blockIdx.x;
            int _max_0 = ((sparse_topk_lens[sm_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[sm_query] + sparse_topk_lens_offset) : (0));
            int l_main = ((_max_0) < (swa_width) ? (_max_0) : (swa_width));
            int l_extra = 0;
            if (compressed_width > 0) {
                int _max_2 = ((extra_topk_lens[sm_query]) > (0) ? (extra_topk_lens[sm_query]) : (0));
                int _min_1 = ((_max_2) < (compressed_width) ? (_max_2) : (compressed_width));
                l_extra = _min_1;
            }
            unsigned int _phase_index_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int sm_work = 0; sm_work < 1048576; sm_work++) {
                if (sm_query < num_query_tokens) {
                    int _max_3 = ((sparse_topk_lens[sm_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[sm_query] + sparse_topk_lens_offset) : (0));
                    int l_main_6 = ((_max_3) < (swa_width) ? (_max_3) : (swa_width));
                    int l_extra_6 = 0;
                    if (compressed_width > 0) {
                        int _max_5 = ((extra_topk_lens[sm_query]) > (0) ? (extra_topk_lens[sm_query]) : (0));
                        int _min_4 = ((_max_5) < (compressed_width) ? (_max_5) : (compressed_width));
                        l_extra_6 = _min_4;
                    }
                    int sm_n_main = (l_main_6 + 64 - 1) / 64;
                    int sm_n_extra = (l_extra_6 + 64 - 1) / 64;
                    int sm_tiles = ((sm_n_main + sm_n_extra) > (1) ? (sm_n_main + sm_n_extra) : (1));
                    int _parity_q_load_full = (int)((unsigned int)sm_item % 2);
                    mbarrier_wait(q_load_full_addr, _parity_q_load_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sfa_regs[32];
                    #pragma unroll
                    for (int sfa_s = 0; sfa_s < 7; sfa_s++) {
                        sfa_regs[4 * sfa_s] = smem_q_sf[lane * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 1] = smem_q_sf[(lane + 32) * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 2] = sfa_regs[4 * sfa_s];
                        sfa_regs[4 * sfa_s + 3] = sfa_regs[4 * sfa_s + 1];
                    }
                    #pragma unroll
                    for (int sfa_pad = 28; sfa_pad < 32; sfa_pad++) {
                        sfa_regs[sfa_pad] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 384 + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 384 + 16 + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(q_full_addr);
                        }
                    }
                    float row_max_val = -CUDART_INF_F;
                    float row_sum_val = 0.0f;
                    if (has_sinks != 0 && sm_head < num_heads) {
                        row_max_val = sinks[sm_head] * 1.442695f / softmax_scale_log2;
                        row_sum_val = ((sm_half == 0) ? 1.0f : 0.0f);
                    }
                    #pragma unroll 1
                    for (int sm_tile = 0; sm_tile < sm_tiles; sm_tile++) {
                        int s_stage = (sm_tile_cursor + sm_tile) % 2;
                        int sm_slot_tile = sm_tile & 3;
                        if (sm_slot_tile == 0) {
                            mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                        }
                        int seg_in_main = sm_n_main > sm_tile || sm_n_extra == 0;
                        int seg_active = ((seg_in_main != 0) ? l_main_6 : l_extra_6);
                        int seg_tile = ((seg_in_main != 0) ? sm_tile : sm_tile - sm_n_main);
                        int _max_6 = ((seg_active - seg_tile * 64 - sm_half * 32) > (0) ? (seg_active - seg_tile * 64 - sm_half * 32) : (0));
                        int valid_cols = ((_max_6) < (32) ? (_max_6) : (32));
                        int staged_index[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[0])) : "r"(smem_indices_addr + sm_index_stage * 1024 + (unsigned int)((sm_slot_tile * 64 + sm_half * 32 + lane) * 4)));
                        unsigned int invalid_cols = __ballot_sync(0xFFFFFFFF, staged_index[0] < 0);
                        if (sm_slot_tile == 3 || sm_tile + 1 == sm_tiles) {
                            mbarrier_arrive(index_empty_addr + (sm_index_stage) * 8);
                            sm_index_stage += 1;
                            if (sm_index_stage == 6) { sm_index_stage = 0; _phase_index_full ^= 1; }
                        }
                        int _stage_s_full = (int)((unsigned int)(sm_tile_cursor + sm_tile) % 2);
                        int _parity_s_full = (int)((unsigned int)(sm_tile_cursor + sm_tile) / 2 % 2);
                        mbarrier_wait(s_full_addr + (_stage_s_full) * 8, _parity_s_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float sv[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                            : "r"(taddr + 256 + (unsigned int)(s_stage * 64) + (unsigned int)(sm_half * 32) + (unsigned int)(sm_row_base << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(s_empty_addr + ((int)((unsigned int)(sm_tile_cursor + sm_tile) % 2)) * 8);
                        if (valid_cols < 32) {
                            uint32_t _thresh_mask_7;
                            {
                                int _lim_8 = valid_cols;
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
                            if (!(_thresh_mask_7 & (1u << 0))) sv[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 1))) sv[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 2))) sv[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 3))) sv[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 4))) sv[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 5))) sv[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 6))) sv[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 7))) sv[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 8))) sv[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 9))) sv[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 10))) sv[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 11))) sv[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 12))) sv[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 13))) sv[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 14))) sv[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 15))) sv[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 16))) sv[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 17))) sv[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 18))) sv[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 19))) sv[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 20))) sv[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 21))) sv[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 22))) sv[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 23))) sv[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 24))) sv[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 25))) sv[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 26))) sv[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 27))) sv[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 28))) sv[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 29))) sv[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 30))) sv[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_7 & (1u << 31))) sv[31] = -CUDART_INF_F;
                        }
                        if (invalid_cols != 0) {
                            sv[0] = (((invalid_cols & 1) != 0) ? -CUDART_INF_F : sv[0]);
                            sv[1] = (((invalid_cols >> 1 & 1) != 0) ? -CUDART_INF_F : sv[1]);
                            sv[2] = (((invalid_cols >> 2 & 1) != 0) ? -CUDART_INF_F : sv[2]);
                            sv[3] = (((invalid_cols >> 3 & 1) != 0) ? -CUDART_INF_F : sv[3]);
                            sv[4] = (((invalid_cols >> 4 & 1) != 0) ? -CUDART_INF_F : sv[4]);
                            sv[5] = (((invalid_cols >> 5 & 1) != 0) ? -CUDART_INF_F : sv[5]);
                            sv[6] = (((invalid_cols >> 6 & 1) != 0) ? -CUDART_INF_F : sv[6]);
                            sv[7] = (((invalid_cols >> 7 & 1) != 0) ? -CUDART_INF_F : sv[7]);
                            sv[8] = (((invalid_cols >> 8 & 1) != 0) ? -CUDART_INF_F : sv[8]);
                            sv[9] = (((invalid_cols >> 9 & 1) != 0) ? -CUDART_INF_F : sv[9]);
                            sv[10] = (((invalid_cols >> 10 & 1) != 0) ? -CUDART_INF_F : sv[10]);
                            sv[11] = (((invalid_cols >> 11 & 1) != 0) ? -CUDART_INF_F : sv[11]);
                            sv[12] = (((invalid_cols >> 12 & 1) != 0) ? -CUDART_INF_F : sv[12]);
                            sv[13] = (((invalid_cols >> 13 & 1) != 0) ? -CUDART_INF_F : sv[13]);
                            sv[14] = (((invalid_cols >> 14 & 1) != 0) ? -CUDART_INF_F : sv[14]);
                            sv[15] = (((invalid_cols >> 15 & 1) != 0) ? -CUDART_INF_F : sv[15]);
                            sv[16] = (((invalid_cols >> 16 & 1) != 0) ? -CUDART_INF_F : sv[16]);
                            sv[17] = (((invalid_cols >> 17 & 1) != 0) ? -CUDART_INF_F : sv[17]);
                            sv[18] = (((invalid_cols >> 18 & 1) != 0) ? -CUDART_INF_F : sv[18]);
                            sv[19] = (((invalid_cols >> 19 & 1) != 0) ? -CUDART_INF_F : sv[19]);
                            sv[20] = (((invalid_cols >> 20 & 1) != 0) ? -CUDART_INF_F : sv[20]);
                            sv[21] = (((invalid_cols >> 21 & 1) != 0) ? -CUDART_INF_F : sv[21]);
                            sv[22] = (((invalid_cols >> 22 & 1) != 0) ? -CUDART_INF_F : sv[22]);
                            sv[23] = (((invalid_cols >> 23 & 1) != 0) ? -CUDART_INF_F : sv[23]);
                            sv[24] = (((invalid_cols >> 24 & 1) != 0) ? -CUDART_INF_F : sv[24]);
                            sv[25] = (((invalid_cols >> 25 & 1) != 0) ? -CUDART_INF_F : sv[25]);
                            sv[26] = (((invalid_cols >> 26 & 1) != 0) ? -CUDART_INF_F : sv[26]);
                            sv[27] = (((invalid_cols >> 27 & 1) != 0) ? -CUDART_INF_F : sv[27]);
                            sv[28] = (((invalid_cols >> 28 & 1) != 0) ? -CUDART_INF_F : sv[28]);
                            sv[29] = (((invalid_cols >> 29 & 1) != 0) ? -CUDART_INF_F : sv[29]);
                            sv[30] = (((invalid_cols >> 30 & 1) != 0) ? -CUDART_INF_F : sv[30]);
                            sv[31] = (((invalid_cols >> 31 & 1) != 0) ? -CUDART_INF_F : sv[31]);
                        }
                        float2 _reg_reduce_max2_9 = {-CUDART_INF_F, -CUDART_INF_F};
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[0], sv[1]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[2], sv[3]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[4], sv[5]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[6], sv[7]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[8], sv[9]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[10], sv[11]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[12], sv[13]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[14], sv[15]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[16], sv[17]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[18], sv[19]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[20], sv[21]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[22], sv[23]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[24], sv[25]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[26], sv[27]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv[28], sv[29]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv[30], sv[31]));
                        float local_max = row_max_reduce(_reg_reduce_max2_9);
                        smem_exch[s_stage * 128 + sm_thread] = local_max;
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(max_noftz(local_max, smem_exch[s_stage * 128 + (sm_thread ^ 64)])), "f"(row_max_val));
                        if (row_max_val > -CUDART_INF_F && (new_max - row_max_val) * softmax_scale_log2 <= 5.0f) {
                            new_max = row_max_val;
                        }
                        float delta = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max * softmax_scale_log2);
                        float _exp2_10 = approx_exp2(delta);
                        float acc_scale = ((row_max_val > -CUDART_INF_F && new_max != row_max_val) ? _exp2_10 : 1.0f);
                        int _stage_p_recycle = (int)((unsigned int)(sm_tile_cursor + sm_tile) % 2);
                        int _parity_p_recycle = (int)((unsigned int)(sm_tile_cursor + sm_tile) / 2 % 2) ^ 1;
                        mbarrier_wait(p_recycle_addr + (_stage_p_recycle) * 8, _parity_p_recycle);
                        smem_scale[s_stage * 128 + sm_thread] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(stats_addr + ((int)((unsigned int)(sm_tile_cursor + sm_tile) % 2)) * 8);
                        row_max_val = new_max;
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float max_scaled = safe_max * softmax_scale_log2;
                        int _stage_kexp_full = (int)((unsigned int)(sm_tile_cursor + sm_tile) % 2);
                        int _parity_kexp_full = (int)((unsigned int)(sm_tile_cursor + sm_tile) / 2 % 2);
                        mbarrier_wait(kexp_full_addr + (_stage_kexp_full) * 8, _parity_kexp_full);
                        unsigned int kexp_bits = smem_kexp[s_stage * 4];
                        float kexp_f = (float)kexp_bits;
                        float kexp_inv = __uint_as_float(127 - kexp_bits << 23);
                        const float2 _fma_b2_11 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_12 = {kexp_f - max_scaled, kexp_f - max_scaled};
                        float2 _fma_pair_13 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_11, _fma_c2_12);
                        sv[0] = _fma_pair_13.x;
                        sv[1] = _fma_pair_13.y;
                        float2 _fma_pair_14 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_11, _fma_c2_12);
                        sv[2] = _fma_pair_14.x;
                        sv[3] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_11, _fma_c2_12);
                        sv[4] = _fma_pair_15.x;
                        sv[5] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_11, _fma_c2_12);
                        sv[6] = _fma_pair_16.x;
                        sv[7] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_11, _fma_c2_12);
                        sv[8] = _fma_pair_17.x;
                        sv[9] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_11, _fma_c2_12);
                        sv[10] = _fma_pair_18.x;
                        sv[11] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_11, _fma_c2_12);
                        sv[12] = _fma_pair_19.x;
                        sv[13] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_11, _fma_c2_12);
                        sv[14] = _fma_pair_20.x;
                        sv[15] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_11, _fma_c2_12);
                        sv[16] = _fma_pair_21.x;
                        sv[17] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_11, _fma_c2_12);
                        sv[18] = _fma_pair_22.x;
                        sv[19] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_11, _fma_c2_12);
                        sv[20] = _fma_pair_23.x;
                        sv[21] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_11, _fma_c2_12);
                        sv[22] = _fma_pair_24.x;
                        sv[23] = _fma_pair_24.y;
                        float2 _fma_pair_25 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_11, _fma_c2_12);
                        sv[24] = _fma_pair_25.x;
                        sv[25] = _fma_pair_25.y;
                        float2 _fma_pair_26 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_11, _fma_c2_12);
                        sv[26] = _fma_pair_26.x;
                        sv[27] = _fma_pair_26.y;
                        float2 _fma_pair_27 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_11, _fma_c2_12);
                        sv[28] = _fma_pair_27.x;
                        sv[29] = _fma_pair_27.y;
                        float2 _fma_pair_28 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_11, _fma_c2_12);
                        sv[30] = _fma_pair_28.x;
                        sv[31] = _fma_pair_28.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv[_le] = approx_exp2(sv[_le]);
                        }
                        unsigned int _fp8_0[8];
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
                                : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                                   "f"(sv[6]), "f"(sv[7]));
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
                                : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                                   "f"(sv[10]), "f"(sv[11]));
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
                                : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                                   "f"(sv[14]), "f"(sv[15]));
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
                                : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                                   "f"(sv[18]), "f"(sv[19]));
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
                                : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                                   "f"(sv[22]), "f"(sv[23]));
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
                                : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                                   "f"(sv[26]), "f"(sv[27]));
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
                                : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                                   "f"(sv[30]), "f"(sv[31]));
                            _fp8_0[7] = _packed;
                        }
                        int p_stage_base = smem_p_addr + (unsigned int)(s_stage * 4096);
                        #pragma unroll
                        for (int p_vec = 0; p_vec < 2; p_vec++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_stage_base + (sm_head * 64 + (sm_half * 32 + p_vec * 16) ^ (sm_head * 64 + (sm_half * 32 + p_vec * 16) >> 7 & 3) << 4))), "r"(_fp8_0[p_vec * 4]), "r"(_fp8_0[p_vec * 4 + 1]), "r"(_fp8_0[p_vec * 4 + 2]), "r"(_fp8_0[p_vec * 4 + 3]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(p_full_addr + ((int)((unsigned int)(sm_tile_cursor + sm_tile) % 2)) * 8);
                        float2 _reg_reduce_sum2_30 = make_float2(0.0f, 0.0f);
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[0], sv[1]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[2], sv[3]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[4], sv[5]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[6], sv[7]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[8], sv[9]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[10], sv[11]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[12], sv[13]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[14], sv[15]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[16], sv[17]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[18], sv[19]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[20], sv[21]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[22], sv[23]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[24], sv[25]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[26], sv[27]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[28], sv[29]));
                        _reg_reduce_sum2_30 = add_f32x2(_reg_reduce_sum2_30, make_float2(sv[30], sv[31]));
                        float _reg_array_reduce_29 = _reg_reduce_sum2_30.x + _reg_reduce_sum2_30.y;
                        float block_sum = _reg_array_reduce_29 * kexp_inv;
                        row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                    }
                    int _parity_sum_empty = (int)((unsigned int)sm_item % 2) ^ 1;
                    mbarrier_wait(sum_empty_addr, _parity_sum_empty);
                    smem_sum[sm_thread] = row_sum_val;
                    smem_fmax[sm_thread] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    sm_tile_cursor += sm_tiles;
                    sm_item += 1;
                }
                mbarrier_wait(work_full_addr + (sm_work_stage) * 8, _phase_work_full);
                uint32_t w_valid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid)
                    : "r"(work_response_addr + sm_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query)
                    : "r"(work_response_addr + sm_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (sm_work_stage) * 8);
                sm_work_stage += 1;
                if (sm_work_stage == 2) { sm_work_stage = 0; _phase_work_full ^= 1; }
                if (w_valid == 0) {
                    break;
                }
                sm_query = w_query;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.442695f;
            float output_scale = bmm2_scale[0];
            const int co_row_base = warp % 4 * 32;
            const int co_thread = co_row_base + lane;
            const int co_head = co_thread & 63;
            const int co_half = co_thread >> 6;
            const int co_lane_addr = co_row_base << 16;
            int co_tile_cursor = 0;
            int co_item = 0;
            unsigned int co_work_stage = 0;
            int co_query = blockIdx.x;
            int _max_0_1 = ((sparse_topk_lens[co_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[co_query] + sparse_topk_lens_offset) : (0));
            int l_main_1 = ((_max_0_1) < (swa_width) ? (_max_0_1) : (swa_width));
            int l_extra_1 = 0;
            if (compressed_width > 0) {
                int _max_2_1 = ((extra_topk_lens[co_query]) > (0) ? (extra_topk_lens[co_query]) : (0));
                int _min_1_1 = ((_max_2_1) < (compressed_width) ? (_max_2_1) : (compressed_width));
                l_extra_1 = _min_1_1;
            }
            mbarrier_arrive(o_empty_addr);
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int co_work = 0; co_work < 1048576; co_work++) {
                if (co_query < num_query_tokens) {
                    int _max_3_1 = ((sparse_topk_lens[co_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[co_query] + sparse_topk_lens_offset) : (0));
                    int l_main_7 = ((_max_3_1) < (swa_width) ? (_max_3_1) : (swa_width));
                    int l_extra_7 = 0;
                    if (compressed_width > 0) {
                        int _max_5_1 = ((extra_topk_lens[co_query]) > (0) ? (extra_topk_lens[co_query]) : (0));
                        int _min_4_1 = ((_max_5_1) < (compressed_width) ? (_max_5_1) : (compressed_width));
                        l_extra_7 = _min_4_1;
                    }
                    int co_n_main = (l_main_7 + 64 - 1) / 64;
                    int co_n_extra = (l_extra_7 + 64 - 1) / 64;
                    int co_tiles = ((co_n_main + co_n_extra) > (1) ? (co_n_main + co_n_extra) : (1));
                    #pragma unroll 1
                    for (int co_tile = 0; co_tile < co_tiles; co_tile++) {
                        int c_stage = (co_tile_cursor + co_tile) % 2;
                        int _stage_stats = (int)((unsigned int)(co_tile_cursor + co_tile) % 2);
                        int _parity_stats = (int)((unsigned int)(co_tile_cursor + co_tile) / 2 % 2);
                        mbarrier_wait(stats_addr + (_stage_stats) * 8, _parity_stats);
                        float acc_scale_1 = smem_scale[c_stage * 128 + co_thread];
                        int any_rescale = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (co_tile > 0) {
                            int _parity_o_full = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                            mbarrier_wait(o_full_addr, _parity_o_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (any_rescale != 0) {
                                int o_base = taddr + (unsigned int)co_lane_addr;
                                #pragma unroll 2
                                for (int c = 0; c < 128; c += 16) {
                                    float _tmem_load_0[16];
                                    tmem_ld_x16(&_tmem_load_0[0], o_base + c);
                                    const float2 _scale2_6 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_6);
                                    tmem_st_x16_f32(o_base + c, _tmem_load_0);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
                        }
                        if (co_tile > 0) {
                            int _parity_o_full_1 = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                            mbarrier_wait(o_full_addr + 8, _parity_o_full_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (any_rescale != 0) {
                                int o_base_1 = taddr + 128 + (unsigned int)co_lane_addr;
                                #pragma unroll 2
                                for (int c_1 = 0; c_1 < 128; c_1 += 16) {
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], o_base_1 + c_1);
                                    const float2 _scale2_7 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_7);
                                    tmem_st_x16_f32(o_base_1 + c_1, _tmem_load_1);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                        }
                        mbarrier_arrive(o_empty_addr + 8);
                    }
                    co_tile_cursor += co_tiles;
                    int _parity_o_full_2 = (int)((unsigned int)co_tile_cursor % 2) ^ 1;
                    mbarrier_wait(o_full_addr, _parity_o_full_2);
                    int _parity_o_full_3 = (int)((unsigned int)co_tile_cursor % 2) ^ 1;
                    mbarrier_wait(o_full_addr + 8, _parity_o_full_3);
                    int _parity_sum_ready = (int)((unsigned int)co_item % 2);
                    mbarrier_wait(sum_ready_addr, _parity_sum_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = smem_sum[co_thread] + smem_sum[co_thread ^ 64];
                    float final_max = smem_fmax[co_thread];
                    const int ep_lane0 = co_row_base + lane % 16;
                    const int ep_lane1 = ep_lane0 + 16;
                    float ep_sum0 = smem_sum[ep_lane0] + smem_sum[ep_lane0 ^ 64];
                    float ep_sum1 = smem_sum[ep_lane1] + smem_sum[ep_lane1 ^ 64];
                    mbarrier_arrive(sum_empty_addr);
                    if (co_half == 0 && co_head < num_heads) {
                        float _log2_8;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_8) : "f"(total_sum));
                        LSE[co_query * num_heads + co_head] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_8) : -CUDART_INF_F);
                    }
                    float _rcp_9 = approx_rcp(ep_sum0);
                    float ep_scale0 = ((ep_sum0 > 0.0f) ? _rcp_9 : 0.0f) * output_scale;
                    float _rcp_10 = approx_rcp(ep_sum1);
                    float ep_scale1 = ((ep_sum1 > 0.0f) ? _rcp_10 : 0.0f) * output_scale;
                    int ep_shuffle = num_heads > 32;
                    const int ep_src_lane = (lane >> 1) + (lane & 1) * 16;
                    int ep_dst_sub = ((ep_shuffle != 0) ? lane >> 1 : lane % 16);
                    int ep_dst_half = ((ep_shuffle != 0) ? lane & 1 : lane / 16);
                    int ep_head0 = co_row_base + ep_dst_sub & 63;
                    int ep_head1 = ep_head0 + 16;
                    int ep_col = co_half * 128 + ep_dst_half * 16;
                    long long ep_row0 = ((long long)co_query * (long long)num_heads + (long long)ep_head0) * 512ll + (long long)ep_col;
                    long long ep_row1 = ((long long)co_query * (long long)num_heads + (long long)ep_head1) * 512ll + (long long)ep_col;
                    const int ep_group0 = co_row_base & 63;
                    float ep_buf0[16];
                    float ep_buf1[16];
                    float ep_buf2[16];
                    float ep_buf3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576 + 32));
                    const float2 _scale2_11 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_11);
                    unsigned int ep_pack0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack0[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w = 0; w < 8; w++) {
                            unsigned int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, ep_pack0[w], ep_src_lane);
                            ep_pack0[w] = _shfl_sync_12;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_13 = reinterpret_cast<const unsigned*>(ep_pack0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + ep_row0)), "r"(_raw_stv8_13[0]), "r"(_raw_stv8_13[1]), "r"(_raw_stv8_13[2]), "r"(_raw_stv8_13[3]), "r"(_raw_stv8_13[4]), "r"(_raw_stv8_13[5]), "r"(_raw_stv8_13[6]), "r"(_raw_stv8_13[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 64));
                    const float2 _scale2_14 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_14);
                    unsigned int ep_pack1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack1[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_1 = 0; w_1 < 8; w_1++) {
                            unsigned int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, ep_pack1[w_1], ep_src_lane);
                            ep_pack1[w_1] = _shfl_sync_15;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_16 = reinterpret_cast<const unsigned*>(ep_pack1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + ep_row1)), "r"(_raw_stv8_16[0]), "r"(_raw_stv8_16[1]), "r"(_raw_stv8_16[2]), "r"(_raw_stv8_16[3]), "r"(_raw_stv8_16[4]), "r"(_raw_stv8_16[5]), "r"(_raw_stv8_16[6]), "r"(_raw_stv8_16[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576 + 64));
                    const float2 _scale2_17 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_17);
                    unsigned int ep_pack2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_2 = 0; w_2 < 8; w_2++) {
                            unsigned int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, ep_pack2[w_2], ep_src_lane);
                            ep_pack2[w_2] = _shfl_sync_18;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_19 = reinterpret_cast<const unsigned*>(ep_pack2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 32ll))), "r"(_raw_stv8_19[0]), "r"(_raw_stv8_19[1]), "r"(_raw_stv8_19[2]), "r"(_raw_stv8_19[3]), "r"(_raw_stv8_19[4]), "r"(_raw_stv8_19[5]), "r"(_raw_stv8_19[6]), "r"(_raw_stv8_19[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 96));
                    const float2 _scale2_20 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_20);
                    unsigned int ep_pack3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack3[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_3 = 0; w_3 < 8; w_3++) {
                            unsigned int _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, ep_pack3[w_3], ep_src_lane);
                            ep_pack3[w_3] = _shfl_sync_21;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_22 = reinterpret_cast<const unsigned*>(ep_pack3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 32ll))), "r"(_raw_stv8_22[0]), "r"(_raw_stv8_22[1]), "r"(_raw_stv8_22[2]), "r"(_raw_stv8_22[3]), "r"(_raw_stv8_22[4]), "r"(_raw_stv8_22[5]), "r"(_raw_stv8_22[6]), "r"(_raw_stv8_22[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 1048576 + 96));
                    const float2 _scale2_23 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_23);
                    unsigned int ep_pack4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_4 = 0; w_4 < 8; w_4++) {
                            unsigned int _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, ep_pack4[w_4], ep_src_lane);
                            ep_pack4[w_4] = _shfl_sync_24;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_25 = reinterpret_cast<const unsigned*>(ep_pack4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 64ll))), "r"(_raw_stv8_25[0]), "r"(_raw_stv8_25[1]), "r"(_raw_stv8_25[2]), "r"(_raw_stv8_25[3]), "r"(_raw_stv8_25[4]), "r"(_raw_stv8_25[5]), "r"(_raw_stv8_25[6]), "r"(_raw_stv8_25[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128));
                    const float2 _scale2_26 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_26);
                    unsigned int ep_pack5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack5[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_5 = 0; w_5 < 8; w_5++) {
                            unsigned int _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, ep_pack5[w_5], ep_src_lane);
                            ep_pack5[w_5] = _shfl_sync_27;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_28 = reinterpret_cast<const unsigned*>(ep_pack5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 64ll))), "r"(_raw_stv8_28[0]), "r"(_raw_stv8_28[1]), "r"(_raw_stv8_28[2]), "r"(_raw_stv8_28[3]), "r"(_raw_stv8_28[4]), "r"(_raw_stv8_28[5]), "r"(_raw_stv8_28[6]), "r"(_raw_stv8_28[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576));
                    const float2 _scale2_29 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_29);
                    unsigned int ep_pack6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_6 = 0; w_6 < 8; w_6++) {
                            unsigned int _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, ep_pack6[w_6], ep_src_lane);
                            ep_pack6[w_6] = _shfl_sync_30;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_31 = reinterpret_cast<const unsigned*>(ep_pack6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 96ll))), "r"(_raw_stv8_31[0]), "r"(_raw_stv8_31[1]), "r"(_raw_stv8_31[2]), "r"(_raw_stv8_31[3]), "r"(_raw_stv8_31[4]), "r"(_raw_stv8_31[5]), "r"(_raw_stv8_31[6]), "r"(_raw_stv8_31[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 32));
                    const float2 _scale2_32 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_32);
                    unsigned int ep_pack7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack7[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_7 = 0; w_7 < 8; w_7++) {
                            unsigned int _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, ep_pack7[w_7], ep_src_lane);
                            ep_pack7[w_7] = _shfl_sync_33;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_34 = reinterpret_cast<const unsigned*>(ep_pack7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 96ll))), "r"(_raw_stv8_34[0]), "r"(_raw_stv8_34[1]), "r"(_raw_stv8_34[2]), "r"(_raw_stv8_34[3]), "r"(_raw_stv8_34[4]), "r"(_raw_stv8_34[5]), "r"(_raw_stv8_34[6]), "r"(_raw_stv8_34[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576 + 32));
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr);
                    const float2 _scale2_35 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_35);
                    unsigned int ep_pack8[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack8[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_8 = 0; w_8 < 8; w_8++) {
                            unsigned int _shfl_sync_36 = __shfl_sync(0xFFFFFFFF, ep_pack8[w_8], ep_src_lane);
                            ep_pack8[w_8] = _shfl_sync_36;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_37 = reinterpret_cast<const unsigned*>(ep_pack8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256ll))), "r"(_raw_stv8_37[0]), "r"(_raw_stv8_37[1]), "r"(_raw_stv8_37[2]), "r"(_raw_stv8_37[3]), "r"(_raw_stv8_37[4]), "r"(_raw_stv8_37[5]), "r"(_raw_stv8_37[6]), "r"(_raw_stv8_37[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf0[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 64));
                    const float2 _scale2_38 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_38);
                    unsigned int ep_pack9[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack9[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_9 = 0; w_9 < 8; w_9++) {
                            unsigned int _shfl_sync_39 = __shfl_sync(0xFFFFFFFF, ep_pack9[w_9], ep_src_lane);
                            ep_pack9[w_9] = _shfl_sync_39;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_40 = reinterpret_cast<const unsigned*>(ep_pack9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256ll))), "r"(_raw_stv8_40[0]), "r"(_raw_stv8_40[1]), "r"(_raw_stv8_40[2]), "r"(_raw_stv8_40[3]), "r"(_raw_stv8_40[4]), "r"(_raw_stv8_40[5]), "r"(_raw_stv8_40[6]), "r"(_raw_stv8_40[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf1[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576 + 64));
                    const float2 _scale2_41 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_41);
                    unsigned int ep_pack10[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack10[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_10 = 0; w_10 < 8; w_10++) {
                            unsigned int _shfl_sync_42 = __shfl_sync(0xFFFFFFFF, ep_pack10[w_10], ep_src_lane);
                            ep_pack10[w_10] = _shfl_sync_42;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_43 = reinterpret_cast<const unsigned*>(ep_pack10);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256ll + 32ll))), "r"(_raw_stv8_43[0]), "r"(_raw_stv8_43[1]), "r"(_raw_stv8_43[2]), "r"(_raw_stv8_43[3]), "r"(_raw_stv8_43[4]), "r"(_raw_stv8_43[5]), "r"(_raw_stv8_43[6]), "r"(_raw_stv8_43[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf2[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 96));
                    const float2 _scale2_44 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_44);
                    unsigned int ep_pack11[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack11[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_11 = 0; w_11 < 8; w_11++) {
                            unsigned int _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, ep_pack11[w_11], ep_src_lane);
                            ep_pack11[w_11] = _shfl_sync_45;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_46 = reinterpret_cast<const unsigned*>(ep_pack11);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256ll + 32ll))), "r"(_raw_stv8_46[0]), "r"(_raw_stv8_46[1]), "r"(_raw_stv8_46[2]), "r"(_raw_stv8_46[3]), "r"(_raw_stv8_46[4]), "r"(_raw_stv8_46[5]), "r"(_raw_stv8_46[6]), "r"(_raw_stv8_46[7]) : "memory");
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf3[15]))
                        : "r"(taddr + (unsigned int)co_lane_addr + 128 + 1048576 + 96));
                    const float2 _scale2_47 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf0)[_ls], _scale2_47);
                    unsigned int ep_pack12[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf0[_lp*2 + 0], ep_buf0[_lp*2+1 + 0]));
                        ep_pack12[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_12 = 0; w_12 < 8; w_12++) {
                            unsigned int _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, ep_pack12[w_12], ep_src_lane);
                            ep_pack12[w_12] = _shfl_sync_48;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_49 = reinterpret_cast<const unsigned*>(ep_pack12);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256ll + 64ll))), "r"(_raw_stv8_49[0]), "r"(_raw_stv8_49[1]), "r"(_raw_stv8_49[2]), "r"(_raw_stv8_49[3]), "r"(_raw_stv8_49[4]), "r"(_raw_stv8_49[5]), "r"(_raw_stv8_49[6]), "r"(_raw_stv8_49[7]) : "memory");
                        }
                    }
                    const float2 _scale2_50 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf1)[_ls], _scale2_50);
                    unsigned int ep_pack13[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf1[_lp*2 + 0], ep_buf1[_lp*2+1 + 0]));
                        ep_pack13[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_13 = 0; w_13 < 8; w_13++) {
                            unsigned int _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, ep_pack13[w_13], ep_src_lane);
                            ep_pack13[w_13] = _shfl_sync_51;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_52 = reinterpret_cast<const unsigned*>(ep_pack13);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256ll + 64ll))), "r"(_raw_stv8_52[0]), "r"(_raw_stv8_52[1]), "r"(_raw_stv8_52[2]), "r"(_raw_stv8_52[3]), "r"(_raw_stv8_52[4]), "r"(_raw_stv8_52[5]), "r"(_raw_stv8_52[6]), "r"(_raw_stv8_52[7]) : "memory");
                        }
                    }
                    const float2 _scale2_53 = {ep_scale0, ep_scale0};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf2)[_ls], _scale2_53);
                    unsigned int ep_pack14[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf2[_lp*2 + 0], ep_buf2[_lp*2+1 + 0]));
                        ep_pack14[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_14 = 0; w_14 < 8; w_14++) {
                            unsigned int _shfl_sync_54 = __shfl_sync(0xFFFFFFFF, ep_pack14[w_14], ep_src_lane);
                            ep_pack14[w_14] = _shfl_sync_54;
                        }
                    }
                    if (ep_head0 < num_heads) {
                        {
                            const unsigned* _raw_stv8_55 = reinterpret_cast<const unsigned*>(ep_pack14);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row0 + 256ll + 96ll))), "r"(_raw_stv8_55[0]), "r"(_raw_stv8_55[1]), "r"(_raw_stv8_55[2]), "r"(_raw_stv8_55[3]), "r"(_raw_stv8_55[4]), "r"(_raw_stv8_55[5]), "r"(_raw_stv8_55[6]), "r"(_raw_stv8_55[7]) : "memory");
                        }
                    }
                    const float2 _scale2_56 = {ep_scale1, ep_scale1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf3)[_ls], _scale2_56);
                    unsigned int ep_pack15[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf3[_lp*2 + 0], ep_buf3[_lp*2+1 + 0]));
                        ep_pack15[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (ep_shuffle != 0) {
                        #pragma unroll
                        for (int w_15 = 0; w_15 < 8; w_15++) {
                            unsigned int _shfl_sync_57 = __shfl_sync(0xFFFFFFFF, ep_pack15[w_15], ep_src_lane);
                            ep_pack15[w_15] = _shfl_sync_57;
                        }
                    }
                    if (ep_head1 < num_heads) {
                        {
                            const unsigned* _raw_stv8_58 = reinterpret_cast<const unsigned*>(ep_pack15);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(O + (ep_row1 + 256ll + 96ll))), "r"(_raw_stv8_58[0]), "r"(_raw_stv8_58[1]), "r"(_raw_stv8_58[2]), "r"(_raw_stv8_58[3]), "r"(_raw_stv8_58[4]), "r"(_raw_stv8_58[5]), "r"(_raw_stv8_58[6]), "r"(_raw_stv8_58[7]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    co_item += 1;
                }
                mbarrier_wait(work_full_addr + (co_work_stage) * 8, _phase_work_full_1);
                uint32_t w_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_1)
                    : "r"(work_response_addr + co_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_1)
                    : "r"(work_response_addr + co_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (co_work_stage) * 8);
                co_work_stage += 1;
                if (co_work_stage == 2) { co_work_stage = 0; _phase_work_full_1 ^= 1; }
                if (w_valid_1 == 0) {
                    break;
                }
                co_query = w_query_1;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            int mma_tile_cursor = 0;
            int mma_item = 0;
            unsigned int mma_work_stage = 0;
            int mma_query = blockIdx.x;
            int _max_0_2 = ((sparse_topk_lens[mma_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mma_query] + sparse_topk_lens_offset) : (0));
            int l_main_2 = ((_max_0_2) < (swa_width) ? (_max_0_2) : (swa_width));
            int l_extra_2 = 0;
            if (compressed_width > 0) {
                int _max_2_2 = ((extra_topk_lens[mma_query]) > (0) ? (extra_topk_lens[mma_query]) : (0));
                int _min_1_2 = ((_max_2_2) < (compressed_width) ? (_max_2_2) : (compressed_width));
                l_extra_2 = _min_1_2;
            }
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int mma_work = 0; mma_work < 1048576; mma_work++) {
                if (mma_query < num_query_tokens) {
                    int _max_3_2 = ((sparse_topk_lens[mma_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mma_query] + sparse_topk_lens_offset) : (0));
                    int l_main_4 = ((_max_3_2) < (swa_width) ? (_max_3_2) : (swa_width));
                    int l_extra_4 = 0;
                    if (compressed_width > 0) {
                        int _max_5_2 = ((extra_topk_lens[mma_query]) > (0) ? (extra_topk_lens[mma_query]) : (0));
                        int _min_4_2 = ((_max_5_2) < (compressed_width) ? (_max_5_2) : (compressed_width));
                        l_extra_4 = _min_4_2;
                    }
                    int mma_n_main = (l_main_4 + 64 - 1) / 64;
                    int mma_n_extra = (l_extra_4 + 64 - 1) / 64;
                    int mma_tiles = ((mma_n_main + mma_n_extra) > (1) ? (mma_n_main + mma_n_extra) : (1));
                    int _parity_q_empty = (int)((unsigned int)mma_item % 2) ^ 1;
                    mbarrier_wait(q_empty_addr, _parity_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_load_full_addr, 51200);
                        #pragma unroll
                        for (int q_copy = 0; q_copy < 2; q_copy++) {
                            tma_3d_gmem2smem(smem_q_a_addr + (unsigned int)(q_copy * 8192), (&tmap_q), 0, 0, mma_query, q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_b_addr + (unsigned int)(q_copy * 8192), (&tmap_q), 128, 0, mma_query, q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_rope_addr + (unsigned int)(q_copy * 8192), (&tmap_q), 224, 0, mma_query, q_load_full_addr);
                        }
                        tma_3d_gmem2smem(smem_q_sf_addr, (&tmap_q_sf), 352, 0, mma_query, q_load_full_addr);
                    }
                    int _parity_q_full = (int)((unsigned int)mma_item % 2);
                    mbarrier_wait(q_full_addr, _parity_q_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int q_cp_0 = 0; q_cp_0 < 4; q_cp_0++) {
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_q_a_addr + (unsigned int)(q_cp_0 * 32))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                    :: "r"((uint32_t)(tmem_tmem_qa0 + q_cp_0 * 8)), "l"(_tcgen05_cp_desc_6)
                                    : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int q_cp_1 = 0; q_cp_1 < 3; q_cp_1++) {
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_q_b_addr + (unsigned int)(q_cp_1 * 32))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                                    :: "r"((uint32_t)(tmem_tmem_qa1 + q_cp_1 * 8)), "l"(_tcgen05_cp_desc_7)
                                    : "memory");
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int mma_tile = 0; mma_tile < mma_tiles; mma_tile++) {
                        int m_stage = mma_tile_cursor + mma_tile & 1;
                        int mk_stage = (mma_tile_cursor + mma_tile) % 3;
                        int _stage_sf_full = (int)((unsigned int)(mma_tile_cursor + mma_tile) % 3);
                        int _parity_sf_full = (int)((unsigned int)(mma_tile_cursor + mma_tile) / 3 % 2);
                        mbarrier_wait(sf_full_addr + (_stage_sf_full) * 8, _parity_sf_full);
                        int _stage_s_empty = (int)((unsigned int)(mma_tile_cursor + mma_tile) % 2);
                        int _parity_s_empty = (int)((unsigned int)(mma_tile_cursor + mma_tile) / 2 % 2) ^ 1;
                        mbarrier_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_0_0 = make_warp_uniform((((smem_k_a_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"(tmem_tmem_qa0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa + 0), "r"(tmem_tmem_sfb + m_stage * 16), "r"(0) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa0 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa + 0 + 4)), "r"((tmem_tmem_sfb + (m_stage * 16 - 2) + 4)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa0 + 16)), "l"(((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa + 0 + 8)), "r"((tmem_tmem_sfb + (m_stage * 16 - 4) + 8)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa0 + 24)), "l"(((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa + 0 + 12)), "r"((tmem_tmem_sfb + (m_stage * 16 - 6) + 12)), "r"(1) : "memory");
                            }
                        }
                        int _mma_b_lo_0_1 = make_warp_uniform((((smem_k_b_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"(tmem_tmem_qa1), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa + 16), "r"(tmem_tmem_sfb + (m_stage * 16 + 8)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa1 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa + 16 + 4)), "r"((tmem_tmem_sfb + (m_stage * 16 + 8 - 2) + 4)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa1 + 16)), "l"(((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa + 16 + 8)), "r"((tmem_tmem_sfb + (m_stage * 16 + 8 - 4) + 8)), "r"(1) : "memory");
                            }
                        }
                        int _mma_a_lo_2 = make_warp_uniform((((smem_q_rope_addr) >> 4) & 0x3FFF) + (((0) ? mma_item & 1 : 0)) * 1024);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_k_c_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        {
                            uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135267472, 1);
                            }
                        }
                        elect_commit(k_empty_addr + ((int)((unsigned int)(mma_tile_cursor + mma_tile) % 3)) * 8);
                        elect_commit(s_full_addr + ((int)((unsigned int)(mma_tile_cursor + mma_tile) % 2)) * 8);
                    }
                    elect_commit(q_empty_addr);
                    mma_tile_cursor += mma_tiles;
                    mma_item += 1;
                }
                mbarrier_wait(work_full_addr + (mma_work_stage) * 8, _phase_work_full_2);
                uint32_t w_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_2)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_2)
                    : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (mma_work_stage) * 8);
                mma_work_stage += 1;
                if (mma_work_stage == 2) { mma_work_stage = 0; _phase_work_full_2 ^= 1; }
                if (w_valid_2 == 0) {
                    break;
                }
                mma_query = w_query_2;
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: scheduler ----
    if (warp == 9) {
        { // scheduler_main
            unsigned int sc_work_stage = 0;
            unsigned int sc_throttle_stage = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int sc_work = 0; sc_work < 1048576; sc_work++) {
                mbarrier_wait(throttle_full_addr + (sc_throttle_stage) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (sc_throttle_stage) * 8);
                sc_throttle_stage += 1;
                if (sc_throttle_stage == 2) { sc_throttle_stage = 0; _phase_throttle_full ^= 1; }
                mbarrier_wait(work_empty_addr + (sc_work_stage) * 8, _phase_work_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(work_full_addr + (sc_work_stage) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + sc_work_stage * 16 + 0 * 16), "r"(work_full_addr + sc_work_stage * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (sc_work_stage) * 8, _phase_work_full_3);
                uint32_t clc_valid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid)
                    : "r"(work_response_addr + sc_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid)
                    : "r"(work_response_addr + sc_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                sc_work_stage += 1;
                if (sc_work_stage == 2) { sc_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full_3 ^= 1; }
                if (clc_valid == 0) {
                    break;
                }
            }
            #pragma unroll
            for (unsigned int sc_tail = 0; sc_tail < 2; sc_tail++) {
                mbarrier_wait(work_empty_addr + (sc_work_stage) * 8, _phase_work_empty);
                sc_work_stage += 1;
                if (sc_work_stage == 2) { sc_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full_3 ^= 1; }
            }
        }
    }
    // ---- Role: index_warp ----
    if (warp == 10) {
        { // index_warp_main
            unsigned int ix_stage = 0;
            unsigned int ix_work_stage = 0;
            int ix_query = blockIdx.x;
            int _max_0_3 = ((sparse_topk_lens[ix_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[ix_query] + sparse_topk_lens_offset) : (0));
            int l_main_3 = ((_max_0_3) < (swa_width) ? (_max_0_3) : (swa_width));
            int l_extra_3 = 0;
            if (compressed_width > 0) {
                int _max_2_3 = ((extra_topk_lens[ix_query]) > (0) ? (extra_topk_lens[ix_query]) : (0));
                int _min_1_3 = ((_max_2_3) < (compressed_width) ? (_max_2_3) : (compressed_width));
                l_extra_3 = _min_1_3;
            }
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int ix_work = 0; ix_work < 1048576; ix_work++) {
                if (ix_query < num_query_tokens) {
                    int _max_3_3 = ((sparse_topk_lens[ix_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[ix_query] + sparse_topk_lens_offset) : (0));
                    int l_main_0 = ((_max_3_3) < (swa_width) ? (_max_3_3) : (swa_width));
                    int l_extra_0 = 0;
                    if (compressed_width > 0) {
                        int _max_5_3 = ((extra_topk_lens[ix_query]) > (0) ? (extra_topk_lens[ix_query]) : (0));
                        int _min_4_3 = ((_max_5_3) < (compressed_width) ? (_max_5_3) : (compressed_width));
                        l_extra_0 = _min_4_3;
                    }
                    int ix_n_main = (l_main_0 + 64 - 1) / 64;
                    int ix_n_extra = (l_extra_0 + 64 - 1) / 64;
                    int ix_tiles = ((ix_n_main + ix_n_extra) > (1) ? (ix_n_main + ix_n_extra) : (1));
                    int ix_passes = (ix_tiles + 4 - 1) / 4;
                    #pragma unroll 1
                    for (int ix_pass = 0; ix_pass < ix_passes; ix_pass++) {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        int ix_base = smem_indices_addr + ix_stage * 1024;
                        #pragma unroll
                        for (int ix_half = 0; ix_half < 2; ix_half++) {
                            int ix_slot = ix_half * 128 + lane * 4;
                            int ix_tile = ix_pass * 4 + (ix_slot >> 6);
                            int ix_in_main = ix_tile < ix_n_main || ix_n_extra == 0;
                            int ix_seg_tile = ((ix_in_main != 0) ? ix_tile : ix_tile - ix_n_main);
                            int ix_width = ((ix_in_main != 0) ? swa_width : compressed_width);
                            int ix_col = ix_seg_tile * 64 + (ix_slot & 63);
                            int* ix_row_ptr = ((ix_in_main != 0) ? (swa_indices + ix_query * swa_index_stride) : (compressed_indices + ix_query * compressed_index_stride));
                            int ix_vals[4];
                            #pragma unroll
                            for (int ix_j = 0; ix_j < 4; ix_j++) {
                                ix_vals[ix_j] = -1;
                            }
                            #pragma unroll
                            for (int ix_j_1 = 0; ix_j_1 < 4; ix_j_1++) {
                                if (ix_width > ix_col + ix_j_1) {
                                    ix_vals[ix_j_1] = ix_row_ptr[ix_col + ix_j_1];
                                }
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + ((unsigned int)ix_base - smem_indices_addr + (unsigned int)(ix_slot * 4))), "r"(ix_vals[0]), "r"(ix_vals[1]), "r"(ix_vals[2]), "r"(ix_vals[3]) : "memory");
                        }
                        mbarrier_arrive(index_full_addr + (ix_stage) * 8);
                        ix_stage += 1;
                        if (ix_stage == 6) { ix_stage = 0; _phase_index_empty ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (ix_work_stage) * 8, _phase_work_full_4);
                uint32_t w_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_3)
                    : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_3)
                    : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (ix_work_stage) * 8);
                ix_work_stage += 1;
                if (ix_work_stage == 2) { ix_work_stage = 0; _phase_work_full_4 ^= 1; }
                if (w_valid_3 == 0) {
                    break;
                }
                ix_query = w_query_3;
            }
        }
    }
    // ---- Role: pv_issuer ----
    if (warp == 11) {
        { // pv_issuer_main
            int pv_tile_cursor = 0;
            unsigned int pv_work_stage = 0;
            int pv_query = blockIdx.x;
            int _max_0_4 = ((sparse_topk_lens[pv_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[pv_query] + sparse_topk_lens_offset) : (0));
            int l_main_5 = ((_max_0_4) < (swa_width) ? (_max_0_4) : (swa_width));
            int l_extra_5 = 0;
            if (compressed_width > 0) {
                int _max_2_4 = ((extra_topk_lens[pv_query]) > (0) ? (extra_topk_lens[pv_query]) : (0));
                int _min_1_4 = ((_max_2_4) < (compressed_width) ? (_max_2_4) : (compressed_width));
                l_extra_5 = _min_1_4;
            }
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int pv_work = 0; pv_work < 1048576; pv_work++) {
                if (pv_query < num_query_tokens) {
                    int _max_3_4 = ((sparse_topk_lens[pv_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[pv_query] + sparse_topk_lens_offset) : (0));
                    int l_main_5_1 = ((_max_3_4) < (swa_width) ? (_max_3_4) : (swa_width));
                    int l_extra_5_1 = 0;
                    if (compressed_width > 0) {
                        int _max_5_4 = ((extra_topk_lens[pv_query]) > (0) ? (extra_topk_lens[pv_query]) : (0));
                        int _min_4_4 = ((_max_5_4) < (compressed_width) ? (_max_5_4) : (compressed_width));
                        l_extra_5_1 = _min_4_4;
                    }
                    int pv_n_main = (l_main_5_1 + 64 - 1) / 64;
                    int pv_n_extra = (l_extra_5_1 + 64 - 1) / 64;
                    int pv_tiles = ((pv_n_main + pv_n_extra) > (1) ? (pv_n_main + pv_n_extra) : (1));
                    int pv_first = 1;
                    #pragma unroll 1
                    for (int pv_tile = 0; pv_tile < pv_tiles; pv_tile++) {
                        int pv_stage = (pv_tile_cursor + pv_tile) % 2;
                        int _stage_p_full = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                        int _parity_p_full = (int)((unsigned int)(pv_tile_cursor + pv_tile) / 2 % 2);
                        mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                        int _stage_v_full = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                        int _parity_v_full = (int)((unsigned int)(pv_tile_cursor + pv_tile) / 2 % 2);
                        mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                        int _parity_o_empty = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                        mbarrier_wait(o_empty_addr, _parity_o_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_3 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                        int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71368720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_scratch), "r"(((pv_first) ? 0 : 1)) : "memory");
                        elect_commit(o_full_addr);
                        int _parity_o_empty_1 = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                        mbarrier_wait(o_empty_addr + 8, _parity_o_empty_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                        int _mma_b_lo_4 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71368720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (128))), "r"(((pv_first) ? 0 : 1)) : "memory");
                        elect_commit(o_full_addr + 8);
                        elect_commit(v_empty_addr + ((int)((unsigned int)(pv_tile_cursor + pv_tile) % 2)) * 8);
                        elect_commit(p_recycle_addr + ((int)((unsigned int)(pv_tile_cursor + pv_tile) % 2)) * 8);
                        pv_first = 0;
                    }
                    pv_tile_cursor += pv_tiles;
                }
                mbarrier_wait(work_full_addr + (pv_work_stage) * 8, _phase_work_full_5);
                uint32_t w_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_4)
                    : "r"(work_response_addr + pv_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_4)
                    : "r"(work_response_addr + pv_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(work_empty_addr + (pv_work_stage) * 8);
                pv_work_stage += 1;
                if (pv_work_stage == 2) { pv_work_stage = 0; _phase_work_full_5 ^= 1; }
                if (w_valid_4 == 0) {
                    break;
                }
                pv_query = w_query_4;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_warp_main
            const int ld_warp = warp - 12;
            const int ld_thread = ld_warp * 32 + lane;
            const int ld_row_base = ld_warp * 32;
            const int v_key = ld_thread & 63;
            const int v_half = ld_thread >> 6;
            int ld_tile_cursor = 0;
            unsigned int ld_work_stage = 0;
            unsigned int ld_throttle_stage = 0;
            int ld_query = blockIdx.x;
            int _max_0_5 = ((sparse_topk_lens[ld_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[ld_query] + sparse_topk_lens_offset) : (0));
            int l_main_8 = ((_max_0_5) < (swa_width) ? (_max_0_5) : (swa_width));
            int l_extra_8 = 0;
            if (compressed_width > 0) {
                int _max_2_5 = ((extra_topk_lens[ld_query]) > (0) ? (extra_topk_lens[ld_query]) : (0));
                int _min_1_5 = ((_max_2_5) < (compressed_width) ? (_max_2_5) : (compressed_width));
                l_extra_8 = _min_1_5;
            }
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int ld_work = 0; ld_work < 1048576; ld_work++) {
                mbarrier_wait(throttle_empty_addr + (ld_throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (ld_throttle_stage) * 8);
                ld_throttle_stage += 1;
                if (ld_throttle_stage == 2) { ld_throttle_stage = 0; _phase_throttle_empty ^= 1; }
                if (ld_query < num_query_tokens) {
                    int _max_3_5 = ((sparse_topk_lens[ld_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[ld_query] + sparse_topk_lens_offset) : (0));
                    int l_main_2_1 = ((_max_3_5) < (swa_width) ? (_max_3_5) : (swa_width));
                    int l_extra_2_1 = 0;
                    if (compressed_width > 0) {
                        int _max_5_5 = ((extra_topk_lens[ld_query]) > (0) ? (extra_topk_lens[ld_query]) : (0));
                        int _min_4_5 = ((_max_5_5) < (compressed_width) ? (_max_5_5) : (compressed_width));
                        l_extra_2_1 = _min_4_5;
                    }
                    int ld_n_main = (l_main_2_1 + 64 - 1) / 64;
                    int ld_n_extra = (l_extra_2_1 + 64 - 1) / 64;
                    int ld_tiles = ((ld_n_main + ld_n_extra) > (1) ? (ld_n_main + ld_n_extra) : (1));
                    int ld_item_on = ld_work < 48;
                    #pragma unroll 1
                    for (int ld_tile = 0; ld_tile < ld_tiles; ld_tile++) {
                        int p_stage = ld_tile_cursor + ld_tile & 1;
                        int k_stage = (ld_tile_cursor + ld_tile) % 3;
                        int lt_on = ld_tile_cursor + ld_tile < 320;
                        int _stage_k_full = (int)((unsigned int)(ld_tile_cursor + ld_tile) % 3);
                        int _parity_k_full = (int)((unsigned int)(ld_tile_cursor + ld_tile) / 3 % 2);
                        mbarrier_wait(k_full_addr + (_stage_k_full) * 8, _parity_k_full);
                        int sf_base = smem_k_sf_addr + (unsigned int)(k_stage * 26624);
                        unsigned int sf_a[8];
                        unsigned int sf_b[8];
                        int sf_par = lane >> 2 & 1;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 3]))
                            : "r"(sf_base + lane * 32 + 16 * sf_par));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 3]))
                            : "r"(sf_base + lane * 32 + (16 - 16 * sf_par)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 3]))
                            : "r"(sf_base + (lane + 32) * 32 + 16 * sf_par));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 3]))
                            : "r"(sf_base + (lane + 32) * 32 + (16 - 16 * sf_par)));
                        unsigned int sf_a_lo_0 = sf_a[0];
                        unsigned int sf_a_hi_0 = sf_a[4];
                        sf_a[0] = ((sf_par != 0) ? sf_a_hi_0 : sf_a_lo_0);
                        sf_a[4] = ((sf_par != 0) ? sf_a_lo_0 : sf_a_hi_0);
                        unsigned int sf_b_lo_0 = sf_b[0];
                        unsigned int sf_b_hi_0 = sf_b[4];
                        sf_b[0] = ((sf_par != 0) ? sf_b_hi_0 : sf_b_lo_0);
                        sf_b[4] = ((sf_par != 0) ? sf_b_lo_0 : sf_b_hi_0);
                        unsigned int sf_a_lo_1 = sf_a[1];
                        unsigned int sf_a_hi_1 = sf_a[5];
                        sf_a[1] = ((sf_par != 0) ? sf_a_hi_1 : sf_a_lo_1);
                        sf_a[5] = ((sf_par != 0) ? sf_a_lo_1 : sf_a_hi_1);
                        unsigned int sf_b_lo_1 = sf_b[1];
                        unsigned int sf_b_hi_1 = sf_b[5];
                        sf_b[1] = ((sf_par != 0) ? sf_b_hi_1 : sf_b_lo_1);
                        sf_b[5] = ((sf_par != 0) ? sf_b_lo_1 : sf_b_hi_1);
                        unsigned int sf_a_lo_2 = sf_a[2];
                        unsigned int sf_a_hi_2 = sf_a[6];
                        sf_a[2] = ((sf_par != 0) ? sf_a_hi_2 : sf_a_lo_2);
                        sf_a[6] = ((sf_par != 0) ? sf_a_lo_2 : sf_a_hi_2);
                        unsigned int sf_b_lo_2 = sf_b[2];
                        unsigned int sf_b_hi_2 = sf_b[6];
                        sf_b[2] = ((sf_par != 0) ? sf_b_hi_2 : sf_b_lo_2);
                        sf_b[6] = ((sf_par != 0) ? sf_b_lo_2 : sf_b_hi_2);
                        unsigned int sf_a_lo_3 = sf_a[3];
                        unsigned int sf_a_hi_3 = sf_a[7];
                        sf_a[3] = ((sf_par != 0) ? sf_a_hi_3 : sf_a_lo_3);
                        sf_a[7] = ((sf_par != 0) ? sf_a_lo_3 : sf_a_hi_3);
                        unsigned int sf_b_lo_3 = sf_b[3];
                        unsigned int sf_b_hi_3 = sf_b[7];
                        sf_b[3] = ((sf_par != 0) ? sf_b_hi_3 : sf_b_lo_3);
                        sf_b[7] = ((sf_par != 0) ? sf_b_lo_3 : sf_b_hi_3);
                        int _stage_k_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile + 1) % 3);
                        int _parity_k_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile + 1) / 3 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty) * 8, _parity_k_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int sfb_regs[16];
                        #pragma unroll
                        for (int sf_s = 0; sf_s < 7; sf_s++) {
                            sfb_regs[2 * sf_s] = sf_a[sf_s];
                            sfb_regs[2 * sf_s + 1] = sf_b[sf_s];
                        }
                        sfb_regs[14] = 0;
                        sfb_regs[15] = 0;
                        int sfb_col = 416 + p_stage * 16;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)sfb_col + (unsigned int)(ld_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[15])));
                        unsigned int k68 = 0;
                        #pragma unroll
                        for (int sf_s_1 = 0; sf_s_1 < 7; sf_s_1++) {
                            k68 = k68 | sf_a[sf_s_1] + 404232216 | sf_b[sf_s_1] + 404232216;
                        }
                        int _warp_vote_6 = __any_sync(0xFFFFFFFF, (k68 & 2155905152u) != 0);
                        unsigned int kexp_u = (unsigned int)(_warp_vote_6 != 0);
                        if (kexp_u != 0) {
                            unsigned int k70 = 0;
                            unsigned int k78 = 0;
                            #pragma unroll
                            for (int sf_s_2 = 0; sf_s_2 < 7; sf_s_2++) {
                                k70 = k70 | sf_a[sf_s_2] + 269488144 | sf_b[sf_s_2] + 269488144;
                                k78 = k78 | sf_a[sf_s_2] + 134744072 | sf_b[sf_s_2] + 134744072;
                            }
                            int _warp_vote_7 = __any_sync(0xFFFFFFFF, (k70 & 2155905152u) != 0);
                            int _warp_vote_8 = __any_sync(0xFFFFFFFF, (k78 & 2155905152u) != 0);
                            kexp_u = 1 + (unsigned int)(_warp_vote_7 != 0) + (unsigned int)(_warp_vote_8 != 0);
                        }
                        unsigned int kexp_f16x2 = (15 - kexp_u << 10) * 65537;
                        int l_swz16 = (v_key & 7) << 4;
                        int l_codes_src = smem_k_a_addr + (unsigned int)(k_stage * 26624) + (unsigned int)(v_half * 8192) + (unsigned int)(v_key * 128);
                        unsigned int l_sc[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&l_sc[0])), "=r"(*reinterpret_cast<uint32_t*>(&l_sc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_sc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_sc[(0) + 3]))
                            : "r"(sf_base + v_key * 32 + v_half * 16));
                        unsigned int l_pre[24];
                        if (v_half == 0) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[0])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(0) + 3]))
                                : "r"(l_codes_src + (0 ^ l_swz16)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[4])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(4) + 3]))
                                : "r"(l_codes_src + (16 ^ l_swz16)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[8])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(8) + 3]))
                                : "r"(l_codes_src + (32 ^ l_swz16)));
                        } else {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[0])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(0) + 3]))
                                : "r"(l_codes_src + (0 ^ l_swz16)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[4])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(4) + 3]))
                                : "r"(l_codes_src + (16 ^ l_swz16)));
                            int l_rope_src = smem_k_a_addr + (unsigned int)(k_stage * 26624) + 16384 + (unsigned int)(v_key * 128);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[8])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(8) + 3]))
                                : "r"(l_rope_src + (0 ^ l_swz16)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[12])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(12) + 3]))
                                : "r"(l_rope_src + (16 ^ l_swz16)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[16])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(16) + 3]))
                                : "r"(l_rope_src + (32 ^ l_swz16)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&l_pre[20])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&l_pre[(20) + 3]))
                                : "r"(l_rope_src + (48 ^ l_swz16)));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(sf_full_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 3)) * 8);
                            }
                        }
                        int _stage_v_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile) % 2);
                        int _parity_v_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile) / 2 % 2) ^ 1;
                        mbarrier_wait(v_empty_addr + (_stage_v_empty) * 8, _parity_v_empty);
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                smem_kexp[p_stage * 4] = kexp_u;
                                mbarrier_arrive(kexp_full_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 2)) * 8);
                            }
                        }
                        int swz16 = (v_key & 7) << 4;
                        int v_row_off = v_key * 128;
                        int v_codes_src = smem_k_a_addr + (unsigned int)(k_stage * 26624) + (unsigned int)(v_half * 8192) + (unsigned int)v_row_off;
                        int v_dst_base0 = p_stage * 32768 + v_key * 128 + 2 * v_half * 8192;
                        int v_dst_base1 = v_dst_base0 + 8192;
                        if ((int)(kexp_u == 0) != 0) {
                            if (v_half == 0) {
                                unsigned int p_lo_0 = l_sc[0];
                                unsigned int p_hi_0 = l_sc[0] >> 8;
                                unsigned int p_out_0[8];
                                p_out_0[0] = qmul4_lower4(l_pre[0], ((1) ? p_lo_0 : p_hi_0));
                                p_out_0[1] = qmul4_higher4(l_pre[0], ((1) ? p_lo_0 : p_hi_0));
                                p_out_0[2] = qmul4_lower4(l_pre[1], ((1) ? p_lo_0 : p_hi_0));
                                p_out_0[3] = qmul4_higher4(l_pre[1], ((1) ? p_lo_0 : p_hi_0));
                                p_out_0[4] = qmul4_lower4(l_pre[2], ((0) ? p_lo_0 : p_hi_0));
                                p_out_0[5] = qmul4_higher4(l_pre[2], ((0) ? p_lo_0 : p_hi_0));
                                p_out_0[6] = qmul4_lower4(l_pre[3], ((0) ? p_lo_0 : p_hi_0));
                                p_out_0[7] = qmul4_higher4(l_pre[3], ((0) ? p_lo_0 : p_hi_0));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (0 ^ swz16))), "r"(p_out_0[0]), "r"(p_out_0[1]), "r"(p_out_0[2]), "r"(p_out_0[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (16 ^ swz16))), "r"(p_out_0[4]), "r"(p_out_0[5]), "r"(p_out_0[6]), "r"(p_out_0[7]) : "memory");
                                unsigned int p_lo_1 = l_sc[0] >> 16;
                                unsigned int p_hi_1 = l_sc[0] >> 24;
                                unsigned int p_out_1[8];
                                p_out_1[0] = qmul4_lower4(l_pre[4], ((1) ? p_lo_1 : p_hi_1));
                                p_out_1[1] = qmul4_higher4(l_pre[4], ((1) ? p_lo_1 : p_hi_1));
                                p_out_1[2] = qmul4_lower4(l_pre[5], ((1) ? p_lo_1 : p_hi_1));
                                p_out_1[3] = qmul4_higher4(l_pre[5], ((1) ? p_lo_1 : p_hi_1));
                                p_out_1[4] = qmul4_lower4(l_pre[6], ((0) ? p_lo_1 : p_hi_1));
                                p_out_1[5] = qmul4_higher4(l_pre[6], ((0) ? p_lo_1 : p_hi_1));
                                p_out_1[6] = qmul4_lower4(l_pre[7], ((0) ? p_lo_1 : p_hi_1));
                                p_out_1[7] = qmul4_higher4(l_pre[7], ((0) ? p_lo_1 : p_hi_1));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (32 ^ swz16))), "r"(p_out_1[0]), "r"(p_out_1[1]), "r"(p_out_1[2]), "r"(p_out_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (48 ^ swz16))), "r"(p_out_1[4]), "r"(p_out_1[5]), "r"(p_out_1[6]), "r"(p_out_1[7]) : "memory");
                                unsigned int p_lo_2 = l_sc[1];
                                unsigned int p_hi_2 = l_sc[1] >> 8;
                                unsigned int p_out_2[8];
                                p_out_2[0] = qmul4_lower4(l_pre[8], ((1) ? p_lo_2 : p_hi_2));
                                p_out_2[1] = qmul4_higher4(l_pre[8], ((1) ? p_lo_2 : p_hi_2));
                                p_out_2[2] = qmul4_lower4(l_pre[9], ((1) ? p_lo_2 : p_hi_2));
                                p_out_2[3] = qmul4_higher4(l_pre[9], ((1) ? p_lo_2 : p_hi_2));
                                p_out_2[4] = qmul4_lower4(l_pre[10], ((0) ? p_lo_2 : p_hi_2));
                                p_out_2[5] = qmul4_higher4(l_pre[10], ((0) ? p_lo_2 : p_hi_2));
                                p_out_2[6] = qmul4_lower4(l_pre[11], ((0) ? p_lo_2 : p_hi_2));
                                p_out_2[7] = qmul4_higher4(l_pre[11], ((0) ? p_lo_2 : p_hi_2));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (64 ^ swz16))), "r"(p_out_2[0]), "r"(p_out_2[1]), "r"(p_out_2[2]), "r"(p_out_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (80 ^ swz16))), "r"(p_out_2[4]), "r"(p_out_2[5]), "r"(p_out_2[6]), "r"(p_out_2[7]) : "memory");
                            }
                            if (v_half == 1) {
                                unsigned int p_lo_3 = l_sc[0];
                                unsigned int p_hi_3 = l_sc[0] >> 8;
                                unsigned int p_out_3[8];
                                p_out_3[0] = qmul4_lower4(l_pre[0], ((1) ? p_lo_3 : p_hi_3));
                                p_out_3[1] = qmul4_higher4(l_pre[0], ((1) ? p_lo_3 : p_hi_3));
                                p_out_3[2] = qmul4_lower4(l_pre[1], ((1) ? p_lo_3 : p_hi_3));
                                p_out_3[3] = qmul4_higher4(l_pre[1], ((1) ? p_lo_3 : p_hi_3));
                                p_out_3[4] = qmul4_lower4(l_pre[2], ((0) ? p_lo_3 : p_hi_3));
                                p_out_3[5] = qmul4_higher4(l_pre[2], ((0) ? p_lo_3 : p_hi_3));
                                p_out_3[6] = qmul4_lower4(l_pre[3], ((0) ? p_lo_3 : p_hi_3));
                                p_out_3[7] = qmul4_higher4(l_pre[3], ((0) ? p_lo_3 : p_hi_3));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (0 ^ swz16))), "r"(p_out_3[0]), "r"(p_out_3[1]), "r"(p_out_3[2]), "r"(p_out_3[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (16 ^ swz16))), "r"(p_out_3[4]), "r"(p_out_3[5]), "r"(p_out_3[6]), "r"(p_out_3[7]) : "memory");
                                unsigned int p_lo_4 = l_sc[0] >> 16;
                                unsigned int p_hi_4 = l_sc[0] >> 24;
                                unsigned int p_out_4[8];
                                p_out_4[0] = qmul4_lower4(l_pre[4], ((1) ? p_lo_4 : p_hi_4));
                                p_out_4[1] = qmul4_higher4(l_pre[4], ((1) ? p_lo_4 : p_hi_4));
                                p_out_4[2] = qmul4_lower4(l_pre[5], ((1) ? p_lo_4 : p_hi_4));
                                p_out_4[3] = qmul4_higher4(l_pre[5], ((1) ? p_lo_4 : p_hi_4));
                                p_out_4[4] = qmul4_lower4(l_pre[6], ((0) ? p_lo_4 : p_hi_4));
                                p_out_4[5] = qmul4_higher4(l_pre[6], ((0) ? p_lo_4 : p_hi_4));
                                p_out_4[6] = qmul4_lower4(l_pre[7], ((0) ? p_lo_4 : p_hi_4));
                                p_out_4[7] = qmul4_higher4(l_pre[7], ((0) ? p_lo_4 : p_hi_4));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (32 ^ swz16))), "r"(p_out_4[0]), "r"(p_out_4[1]), "r"(p_out_4[2]), "r"(p_out_4[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0 + (48 ^ swz16))), "r"(p_out_4[4]), "r"(p_out_4[5]), "r"(p_out_4[6]), "r"(p_out_4[7]) : "memory");
                            }
                        } else {
                            #pragma unroll 1
                            for (int u = 0; u < 3 - v_half; u++) {
                                unsigned int v_in[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&v_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 3]))
                                    : "r"(v_codes_src + (u << 4 ^ swz16)));
                                int v_w_idx = u >> 1;
                                unsigned int v_sc_lo = ((v_w_idx == 0) ? l_sc[0] : l_sc[1]);
                                unsigned int v_sc_hi = ((v_w_idx == 2) ? l_sc[2] : l_sc[3]);
                                unsigned int v_sc_word = ((v_w_idx < 2) ? v_sc_lo : v_sc_hi);
                                uint32_t v_sc_pair;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(v_sc_pair) : "h"((uint16_t)(v_sc_word >> (unsigned int)((u & 1) * 16))));
                                uint32_t _f16x2_mul_0;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(v_sc_pair), "r"(kexp_f16x2));
                                v_sc_pair = _f16x2_mul_0;
                                uint32_t v_sc0;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(v_sc0) : "r"(v_sc_pair), "r"(v_sc_pair));
                                uint32_t v_sc1;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(v_sc1) : "r"(v_sc_pair), "r"(v_sc_pair));
                                unsigned int v_out[8];
                                #pragma unroll
                                for (int v_w = 0; v_w < 4; v_w++) {
                                    unsigned int v_sc_w = ((v_w < 2) ? v_sc0 : v_sc1);
                                    unsigned int v_word = v_in[v_w];
                                    uint32_t _cvt_f16x2_e2m1x2_9;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_9) : "r"((uint32_t)(v_word)));
                                    uint32_t f16x2_mul;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul) : "r"(_cvt_f16x2_e2m1x2_9), "r"(v_sc_w));
                                    uint16_t e4m3x2;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2) : "r"(f16x2_mul));
                                    uint32_t _cvt_f16x2_e2m1x2_10;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_10) : "r"((uint32_t)(v_word >> 8)));
                                    uint32_t f16x2_mul_0;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_0) : "r"(_cvt_f16x2_e2m1x2_10), "r"(v_sc_w));
                                    uint16_t e4m3x2_0;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_0) : "r"(f16x2_mul_0));
                                    uint32_t _cvt_f16x2_e2m1x2_11;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_11) : "r"((uint32_t)(v_word >> 16)));
                                    uint32_t f16x2_mul_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_1) : "r"(_cvt_f16x2_e2m1x2_11), "r"(v_sc_w));
                                    uint16_t e4m3x2_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_1) : "r"(f16x2_mul_1));
                                    uint32_t _cvt_f16x2_e2m1x2_12;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_12) : "r"((uint32_t)(v_word >> 24)));
                                    uint32_t f16x2_mul_2;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_2) : "r"(_cvt_f16x2_e2m1x2_12), "r"(v_sc_w));
                                    uint16_t e4m3x2_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_2) : "r"(f16x2_mul_2));
                                    uint32_t _mov_b32_pack_u16x2_13;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_13) : "h"(e4m3x2), "h"(e4m3x2_0));
                                    v_out[2 * v_w] = _mov_b32_pack_u16x2_13;
                                    uint32_t _mov_b32_pack_u16x2_14;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_14) : "h"(e4m3x2_1), "h"(e4m3x2_2));
                                    v_out[2 * v_w + 1] = _mov_b32_pack_u16x2_14;
                                }
                                int v_dst = ((u >> 2 == 0) ? v_dst_base0 : v_dst_base1);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst + ((u & 3) * 2 << 4 ^ swz16))), "r"(v_out[0]), "r"(v_out[1]), "r"(v_out[2]), "r"(v_out[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst + ((u & 3) * 2 + 1 << 4 ^ swz16))), "r"(v_out[4]), "r"(v_out[5]), "r"(v_out[6]), "r"(v_out[7]) : "memory");
                            }
                        }
                        if (v_half != 0) {
                            unsigned int r_kexp_bf16x2 = (127 - kexp_u << 7) * 65537;
                            if (kexp_u != 0) {
                                uint32_t r_scaled_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_0) : "r"(l_pre[8]), "r"(r_kexp_bf16x2));
                                l_pre[8] = r_scaled_0;
                                uint32_t r_scaled_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_1) : "r"(l_pre[9]), "r"(r_kexp_bf16x2));
                                l_pre[9] = r_scaled_1;
                                uint32_t r_scaled_2;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_2) : "r"(l_pre[10]), "r"(r_kexp_bf16x2));
                                l_pre[10] = r_scaled_2;
                                uint32_t r_scaled_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_3) : "r"(l_pre[11]), "r"(r_kexp_bf16x2));
                                l_pre[11] = r_scaled_3;
                                uint32_t r_scaled_4;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_4) : "r"(l_pre[12]), "r"(r_kexp_bf16x2));
                                l_pre[12] = r_scaled_4;
                                uint32_t r_scaled_5;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_5) : "r"(l_pre[13]), "r"(r_kexp_bf16x2));
                                l_pre[13] = r_scaled_5;
                                uint32_t r_scaled_6;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_6) : "r"(l_pre[14]), "r"(r_kexp_bf16x2));
                                l_pre[14] = r_scaled_6;
                                uint32_t r_scaled_7;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_7) : "r"(l_pre[15]), "r"(r_kexp_bf16x2));
                                l_pre[15] = r_scaled_7;
                                uint32_t r_scaled_8;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_8) : "r"(l_pre[16]), "r"(r_kexp_bf16x2));
                                l_pre[16] = r_scaled_8;
                                uint32_t r_scaled_9;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_9) : "r"(l_pre[17]), "r"(r_kexp_bf16x2));
                                l_pre[17] = r_scaled_9;
                                uint32_t r_scaled_10;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_10) : "r"(l_pre[18]), "r"(r_kexp_bf16x2));
                                l_pre[18] = r_scaled_10;
                                uint32_t r_scaled_11;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_11) : "r"(l_pre[19]), "r"(r_kexp_bf16x2));
                                l_pre[19] = r_scaled_11;
                                uint32_t r_scaled_12;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_12) : "r"(l_pre[20]), "r"(r_kexp_bf16x2));
                                l_pre[20] = r_scaled_12;
                                uint32_t r_scaled_13;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_13) : "r"(l_pre[21]), "r"(r_kexp_bf16x2));
                                l_pre[21] = r_scaled_13;
                                uint32_t r_scaled_14;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_14) : "r"(l_pre[22]), "r"(r_kexp_bf16x2));
                                l_pre[22] = r_scaled_14;
                                uint32_t r_scaled_15;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_15) : "r"(l_pre[23]), "r"(r_kexp_bf16x2));
                                l_pre[23] = r_scaled_15;
                            }
                            unsigned int r_out_0[4];
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_16;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_16) : "r"(l_pre[8]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[8]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_17;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_17) : "r"(l_pre[9]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[9]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_15;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_15) : "h"(_cvt_satfinite_e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_17));
                            r_out_0[0] = _mov_b32_pack_u16x2_15;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_19;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "r"(l_pre[10]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[10]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_20;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_20) : "r"(l_pre[11]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[11]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_18;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_18) : "h"(_cvt_satfinite_e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_20));
                            r_out_0[1] = _mov_b32_pack_u16x2_18;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_22;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_22) : "r"(l_pre[12]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[12]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_23;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_23) : "r"(l_pre[13]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[13]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_21;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_21) : "h"(_cvt_satfinite_e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_23));
                            r_out_0[2] = _mov_b32_pack_u16x2_21;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_25;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_25) : "r"(l_pre[14]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[14]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_26;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_26) : "r"(l_pre[15]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[15]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_24;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_24) : "h"(_cvt_satfinite_e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_26));
                            r_out_0[3] = _mov_b32_pack_u16x2_24;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((unsigned int)(smem_v_addr + (unsigned int)(p_stage * 32768 + v_key * 128 + 24576)) + (unsigned int)(64 ^ swz16)), "r"(r_out_0[0]), "r"(r_out_0[1]), "r"(r_out_0[2]), "r"(r_out_0[3]) : "memory");
                            unsigned int r_out_1[4];
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_28;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_28) : "r"(l_pre[16]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[16]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_29;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_29) : "r"(l_pre[17]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[17]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_27;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_27) : "h"(_cvt_satfinite_e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_29));
                            r_out_1[0] = _mov_b32_pack_u16x2_27;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_31;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_31) : "r"(l_pre[18]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[18]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_31) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_32;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_32) : "r"(l_pre[19]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[19]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_32) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_30;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_30) : "h"(_cvt_satfinite_e4m3x2_bf16x2_31), "h"(_cvt_satfinite_e4m3x2_bf16x2_32));
                            r_out_1[1] = _mov_b32_pack_u16x2_30;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_34;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_34) : "r"(l_pre[20]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[20]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_34) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_35;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_35) : "r"(l_pre[21]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[21]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_35) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_33;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_33) : "h"(_cvt_satfinite_e4m3x2_bf16x2_34), "h"(_cvt_satfinite_e4m3x2_bf16x2_35));
                            r_out_1[2] = _mov_b32_pack_u16x2_33;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_37;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_37) : "r"(l_pre[22]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[22]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_37) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_38;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_38) : "r"(l_pre[23]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(l_pre[23]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_38) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_36;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_36) : "h"(_cvt_satfinite_e4m3x2_bf16x2_37), "h"(_cvt_satfinite_e4m3x2_bf16x2_38));
                            r_out_1[3] = _mov_b32_pack_u16x2_36;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((unsigned int)(smem_v_addr + (unsigned int)(p_stage * 32768 + v_key * 128 + 24576)) + (unsigned int)(80 ^ swz16)), "r"(r_out_1[0]), "r"(r_out_1[1]), "r"(r_out_1[2]), "r"(r_out_1[3]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(kstage_free_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 3)) * 8);
                                mbarrier_arrive(v_full_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 2)) * 8);
                            }
                        }
                    }
                    ld_tile_cursor += ld_tiles;
                }
                mbarrier_wait(work_full_addr + (ld_work_stage) * 8, _phase_work_full_6);
                uint32_t w_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_5)
                    : "r"(work_response_addr + ld_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_5)
                    : "r"(work_response_addr + ld_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (ld_work_stage) * 8);
                ld_work_stage += 1;
                if (ld_work_stage == 2) { ld_work_stage = 0; _phase_work_full_6 ^= 1; }
                if (w_valid_5 == 0) {
                    break;
                }
                ld_query = w_query_5;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: gather ----
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // gather_main
            const int g_warp = warp - 16;
            int g_tile_cursor = 0;
            unsigned int g_index_stage = 0;
            unsigned int g_work_stage = 0;
            int g_query = blockIdx.x;
            int _max_0_6 = ((sparse_topk_lens[g_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[g_query] + sparse_topk_lens_offset) : (0));
            int l_main_9 = ((_max_0_6) < (swa_width) ? (_max_0_6) : (swa_width));
            int l_extra_9 = 0;
            if (compressed_width > 0) {
                int _max_2_6 = ((extra_topk_lens[g_query]) > (0) ? (extra_topk_lens[g_query]) : (0));
                int _min_1_6 = ((_max_2_6) < (compressed_width) ? (_max_2_6) : (compressed_width));
                l_extra_9 = _min_1_6;
            }
            int g_start = 0;
            int pi_rows[8];
            int pi_len = sparse_topk_lens[blockIdx.x];
            #pragma unroll
            for (int pi_j = 0; pi_j < 8; pi_j++) {
                int pi_col = (pi_j >> 2) * 64 + (lane & 15) * 4 + (pi_j & 3);
                pi_rows[pi_j] = -1;
                if (swa_width > pi_col) {
                    pi_rows[pi_j] = swa_indices[(int)blockIdx.x * swa_index_stride + pi_col];
                }
            }
            int _max_3_6 = ((pi_len + sparse_topk_lens_offset) > (0) ? (pi_len + sparse_topk_lens_offset) : (0));
            int g_first_main = ((_max_3_6) < (swa_width) ? (_max_3_6) : (swa_width));
            g_start = ((g_first_main > 64 && g_query < num_query_tokens) ? 2 : 0);
            if (g_start > 0) {
                int _stage_k_empty_1 = (int)((unsigned int)g_tile_cursor % 3);
                int _parity_k_empty_1 = (int)((unsigned int)g_tile_cursor / 3 % 2) ^ 1;
                mbarrier_wait(k_empty_addr + (_stage_k_empty_1) * 8, _parity_k_empty_1);
                int _stage_kstage_free = (int)((unsigned int)g_tile_cursor % 3);
                int _parity_kstage_free = (int)((unsigned int)g_tile_cursor / 3 % 2) ^ 1;
                mbarrier_wait(kstage_free_addr + (_stage_kstage_free) * 8, _parity_kstage_free);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (g_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr, 26624);
                    }
                }
                if (lane < 16) {
                    int gp_data_rows_0[4];
                    int gp_foot_rows_0[4];
                    #pragma unroll
                    for (int g_j_0 = 0; g_j_0 < 4; g_j_0++) {
                        int gp_row_0 = pi_rows[g_j_0];
                        int gp_tok_0 = ((gp_row_0 >= 0) ? gp_row_0 : 0);
                        int gp_page_0 = gp_tok_0 >> swa_page_log2;
                        int gp_pslot_0 = gp_tok_0 - (gp_page_0 << swa_page_log2);
                        gp_data_rows_0[g_j_0] = gp_page_0 * swa_pitch_units + gp_pslot_0 * 11;
                        gp_foot_rows_0[g_j_0] = gp_page_0 * swa_pitch_units + swa_footer_units + gp_pslot_0;
                    }
                    int gp_dst_0 = smem_k_a_addr + 0;
                    if (g_warp < 3) {
                        tma_gather4_gmem2smem(gp_dst_0 + g_warp * 8192 + lane * 512, (&tmap_swa_kv), ((g_warp == 2) ? 224 : g_warp * 128), gp_data_rows_0[0], gp_data_rows_0[1], gp_data_rows_0[2], gp_data_rows_0[3], k_full_addr);
                    } else {
                        tma_gather4_gmem2smem(gp_dst_0 + 24576 + lane * 128, (&tmap_swa_sf), 0, gp_foot_rows_0[0], gp_foot_rows_0[1], gp_foot_rows_0[2], gp_foot_rows_0[3], k_full_addr);
                    }
                }
                int _stage_k_empty_2 = (int)((unsigned int)(g_tile_cursor + 1) % 3);
                int _parity_k_empty_2 = (int)((unsigned int)(g_tile_cursor + 1) / 3 % 2) ^ 1;
                mbarrier_wait(k_empty_addr + (_stage_k_empty_2) * 8, _parity_k_empty_2);
                int _stage_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + 1) % 3);
                int _parity_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + 1) / 3 % 2) ^ 1;
                mbarrier_wait(kstage_free_addr + (_stage_kstage_free_1) * 8, _parity_kstage_free_1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (g_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + 8, 26624);
                    }
                }
                if (lane < 16) {
                    int gp_data_rows_1[4];
                    int gp_foot_rows_1[4];
                    #pragma unroll
                    for (int g_j_1 = 0; g_j_1 < 4; g_j_1++) {
                        int gp_row_1 = pi_rows[4 + g_j_1];
                        int gp_tok_1 = ((gp_row_1 >= 0) ? gp_row_1 : 0);
                        int gp_page_1 = gp_tok_1 >> swa_page_log2;
                        int gp_pslot_1 = gp_tok_1 - (gp_page_1 << swa_page_log2);
                        gp_data_rows_1[g_j_1] = gp_page_1 * swa_pitch_units + gp_pslot_1 * 11;
                        gp_foot_rows_1[g_j_1] = gp_page_1 * swa_pitch_units + swa_footer_units + gp_pslot_1;
                    }
                    int gp_dst_1 = smem_k_a_addr + 26624;
                    if (g_warp < 3) {
                        tma_gather4_gmem2smem(gp_dst_1 + g_warp * 8192 + lane * 512, (&tmap_swa_kv), ((g_warp == 2) ? 224 : g_warp * 128), gp_data_rows_1[0], gp_data_rows_1[1], gp_data_rows_1[2], gp_data_rows_1[3], k_full_addr + 8);
                    } else {
                        tma_gather4_gmem2smem(gp_dst_1 + 24576 + lane * 128, (&tmap_swa_sf), 0, gp_foot_rows_1[0], gp_foot_rows_1[1], gp_foot_rows_1[2], gp_foot_rows_1[3], k_full_addr + 8);
                    }
                }
            }
            unsigned int _phase_index_full_1 = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int g_work = 0; g_work < 1048576; g_work++) {
                if (g_query < num_query_tokens) {
                    int _max_4 = ((sparse_topk_lens[g_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[g_query] + sparse_topk_lens_offset) : (0));
                    int l_main_1_1 = ((_max_4) < (swa_width) ? (_max_4) : (swa_width));
                    int l_extra_1_1 = 0;
                    if (compressed_width > 0) {
                        int _max_6_1 = ((extra_topk_lens[g_query]) > (0) ? (extra_topk_lens[g_query]) : (0));
                        int _min_5 = ((_max_6_1) < (compressed_width) ? (_max_6_1) : (compressed_width));
                        l_extra_1_1 = _min_5;
                    }
                    int g_n_main = (l_main_1_1 + 64 - 1) / 64;
                    int g_n_extra = (l_extra_1_1 + 64 - 1) / 64;
                    int g_tiles = ((g_n_main + g_n_extra) > (1) ? (g_n_main + g_n_extra) : (1));
                    if (g_start > 0) {
                        mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_1);
                        if (g_tiles == 2) {
                            mbarrier_arrive(index_empty_addr + (g_index_stage) * 8);
                            g_index_stage += 1;
                            if (g_index_stage == 6) { g_index_stage = 0; _phase_index_full_1 ^= 1; }
                        }
                    }
                    #pragma unroll 1
                    for (int g_tile = g_start; g_tile < g_tiles; g_tile++) {
                        int g_stage = (g_tile_cursor + g_tile) % 3;
                        int g_slot_tile = g_tile & 3;
                        if (g_slot_tile == 0) {
                            mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_1);
                        }
                        int _stage_k_empty_3 = (int)((unsigned int)(g_tile_cursor + g_tile) % 3);
                        int _parity_k_empty_3 = (int)((unsigned int)(g_tile_cursor + g_tile) / 3 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty_3) * 8, _parity_k_empty_3);
                        int _stage_kstage_free_2 = (int)((unsigned int)(g_tile_cursor + g_tile) % 3);
                        int _parity_kstage_free_2 = (int)((unsigned int)(g_tile_cursor + g_tile) / 3 % 2) ^ 1;
                        mbarrier_wait(kstage_free_addr + (_stage_kstage_free_2) * 8, _parity_kstage_free_2);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (g_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8, 26624);
                            }
                        }
                        if (lane < 16) {
                            int g_group = lane;
                            int g_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&g_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 3]))
                                : "r"(smem_indices_addr + g_index_stage * 1024 + (unsigned int)((g_slot_tile * 64 + g_group * 4) * 4)));
                            int g_use_main = g_n_main > g_tile || g_n_extra == 0;
                            int g_page_log2 = ((g_use_main != 0) ? swa_page_log2 : compressed_page_log2);
                            int g_pitch = ((g_use_main != 0) ? swa_pitch_units : compressed_pitch_units);
                            int g_footer_units = ((g_use_main != 0) ? swa_footer_units : compressed_footer_units);
                            int g_data_rows[4];
                            int g_foot_rows[4];
                            #pragma unroll
                            for (int g_j = 0; g_j < 4; g_j++) {
                                int g_tok = ((g_rows[g_j] >= 0) ? g_rows[g_j] : 0);
                                int g_page = g_tok >> g_page_log2;
                                int g_pslot = g_tok - (g_page << g_page_log2);
                                g_data_rows[g_j] = g_page * g_pitch + g_pslot * 11;
                                g_foot_rows[g_j] = g_page * g_pitch + g_footer_units + g_pslot;
                            }
                            int g_stage_base = smem_k_a_addr + (unsigned int)(g_stage * 26624);
                            if (g_warp < 3) {
                                int g_col = ((g_warp == 2) ? 224 : g_warp * 128);
                                int g_dst = g_stage_base + g_warp * 8192 + g_group * 512;
                                if (g_use_main != 0) {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_swa_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_compressed_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8);
                                }
                            } else {
                                int g_fdst = g_stage_base + 24576 + g_group * 128;
                                if (g_use_main != 0) {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_swa_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_compressed_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8);
                                }
                            }
                        }
                        if (g_slot_tile == 3 || g_tile + 1 == g_tiles) {
                            mbarrier_arrive(index_empty_addr + (g_index_stage) * 8);
                            g_index_stage += 1;
                            if (g_index_stage == 6) { g_index_stage = 0; _phase_index_full_1 ^= 1; }
                        }
                    }
                    g_tile_cursor += g_tiles;
                }
                mbarrier_wait(work_full_addr + (g_work_stage) * 8, _phase_work_full_7);
                uint32_t w_valid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_6)
                    : "r"(work_response_addr + g_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_6)
                    : "r"(work_response_addr + g_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                g_start = 0;
                mbarrier_arrive(work_empty_addr + (g_work_stage) * 8);
                g_work_stage += 1;
                if (g_work_stage == 2) { g_work_stage = 0; _phase_work_full_7 ^= 1; }
                if (w_valid_6 == 0) {
                    break;
                }
                g_query = w_query_6;
            }
        }
    }
    // ---- Role: vx ----
    if (warp >= 20 && warp <= 23) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // vx_main
            const int vx_thread = (warp - 20) * 32 + lane;
            const int vx_key = vx_thread & 63;
            const int vx_half = vx_thread >> 6;
            int vx_tile_cursor = 0;
            unsigned int vx_work_stage = 0;
            int vx_query = blockIdx.x;
            int _max_0_7 = ((sparse_topk_lens[vx_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[vx_query] + sparse_topk_lens_offset) : (0));
            int l_main_10 = ((_max_0_7) < (swa_width) ? (_max_0_7) : (swa_width));
            int l_extra_10 = 0;
            if (compressed_width > 0) {
                int _max_2_7 = ((extra_topk_lens[vx_query]) > (0) ? (extra_topk_lens[vx_query]) : (0));
                int _min_1_7 = ((_max_2_7) < (compressed_width) ? (_max_2_7) : (compressed_width));
                l_extra_10 = _min_1_7;
            }
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int vx_work = 0; vx_work < 1048576; vx_work++) {
                if (vx_query < num_query_tokens) {
                    int _max_3_7 = ((sparse_topk_lens[vx_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[vx_query] + sparse_topk_lens_offset) : (0));
                    int l_main_3_1 = ((_max_3_7) < (swa_width) ? (_max_3_7) : (swa_width));
                    int l_extra_3_1 = 0;
                    if (compressed_width > 0) {
                        int _max_5_6 = ((extra_topk_lens[vx_query]) > (0) ? (extra_topk_lens[vx_query]) : (0));
                        int _min_4_6 = ((_max_5_6) < (compressed_width) ? (_max_5_6) : (compressed_width));
                        l_extra_3_1 = _min_4_6;
                    }
                    int vx_tiles = (((l_main_3_1 + 64 - 1) / 64 + (l_extra_3_1 + 64 - 1) / 64) > (1) ? ((l_main_3_1 + 64 - 1) / 64 + (l_extra_3_1 + 64 - 1) / 64) : (1));
                    #pragma unroll 1
                    for (int vx_tile = 0; vx_tile < vx_tiles; vx_tile++) {
                        int x_stage = (vx_tile_cursor + vx_tile) % 2;
                        int xk_stage = (vx_tile_cursor + vx_tile) % 3;
                        int _stage_k_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) % 3);
                        int _parity_k_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) / 3 % 2);
                        mbarrier_wait(k_full_addr + (_stage_k_full_1) * 8, _parity_k_full_1);
                        int _stage_kexp_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) % 2);
                        int _parity_kexp_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) / 2 % 2);
                        mbarrier_wait(kexp_full_addr + (_stage_kexp_full_1) * 8, _parity_kexp_full_1);
                        unsigned int x_kexp = smem_kexp[x_stage * 4];
                        unsigned int x_kexp_f16x2 = (15 - x_kexp << 10) * 65537;
                        int swz16_1 = (vx_key & 7) << 4;
                        int v_row_off_1 = vx_key * 128;
                        int v_sf_addr = smem_k_sf_addr + (unsigned int)(xk_stage * 26624) + (unsigned int)(vx_key * 32) + (unsigned int)(vx_half * 16);
                        int v_codes_src_1 = smem_k_a_addr + (unsigned int)(xk_stage * 26624) + (unsigned int)(vx_half * 8192) + (unsigned int)v_row_off_1;
                        int v_dst_base0_1 = x_stage * 32768 + vx_key * 128 + 2 * vx_half * 8192;
                        int v_dst_base1_1 = v_dst_base0_1 + 8192;
                        unsigned int v_sc[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v_sc[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 3]))
                            : "r"(v_sf_addr));
                        if ((int)(x_kexp == 0) != 0) {
                            if (vx_half == 0) {
                                unsigned int codes3[4];
                                unsigned int codes4[4];
                                unsigned int codes5[4];
                                unsigned int codes6[4];
                                unsigned int codes7[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes3[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes3[(0) + 3]))
                                    : "r"(v_codes_src_1 + (48 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes4[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes4[(0) + 3]))
                                    : "r"(v_codes_src_1 + (64 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes5[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes5[(0) + 3]))
                                    : "r"(v_codes_src_1 + (80 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes6[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes6[(0) + 3]))
                                    : "r"(v_codes_src_1 + (96 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes7[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes7[(0) + 3]))
                                    : "r"(v_codes_src_1 + (112 ^ swz16_1)));
                                unsigned int v_sc_lo_1 = v_sc[1] >> 16;
                                unsigned int v_sc_hi_1 = v_sc[1] >> 24;
                                unsigned int v_out_1[8];
                                #pragma unroll
                                for (int v_w_1 = 0; v_w_1 < 4; v_w_1++) {
                                    unsigned int v_sc_w_1 = ((v_w_1 < 2) ? v_sc_lo_1 : v_sc_hi_1);
                                    v_out_1[2 * v_w_1] = qmul4_lower4(codes3[v_w_1], v_sc_w_1);
                                    v_out_1[2 * v_w_1 + 1] = qmul4_higher4(codes3[v_w_1], v_sc_w_1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (96 ^ swz16_1))), "r"(v_out_1[0]), "r"(v_out_1[1]), "r"(v_out_1[2]), "r"(v_out_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (112 ^ swz16_1))), "r"(v_out_1[4]), "r"(v_out_1[5]), "r"(v_out_1[6]), "r"(v_out_1[7]) : "memory");
                                unsigned int v_sc_lo_0 = v_sc[2];
                                unsigned int v_sc_hi_1_1 = v_sc[2] >> 8;
                                unsigned int v_out_2[8];
                                #pragma unroll
                                for (int v_w_2 = 0; v_w_2 < 4; v_w_2++) {
                                    unsigned int v_sc_w_2 = ((v_w_2 < 2) ? v_sc_lo_0 : v_sc_hi_1_1);
                                    v_out_2[2 * v_w_2] = qmul4_lower4(codes4[v_w_2], v_sc_w_2);
                                    v_out_2[2 * v_w_2 + 1] = qmul4_higher4(codes4[v_w_2], v_sc_w_2);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (0 ^ swz16_1))), "r"(v_out_2[0]), "r"(v_out_2[1]), "r"(v_out_2[2]), "r"(v_out_2[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (16 ^ swz16_1))), "r"(v_out_2[4]), "r"(v_out_2[5]), "r"(v_out_2[6]), "r"(v_out_2[7]) : "memory");
                                unsigned int v_sc_lo_3 = v_sc[2] >> 16;
                                unsigned int v_sc_hi_4 = v_sc[2] >> 24;
                                unsigned int v_out_5[8];
                                #pragma unroll
                                for (int v_w_3 = 0; v_w_3 < 4; v_w_3++) {
                                    unsigned int v_sc_w_3 = ((v_w_3 < 2) ? v_sc_lo_3 : v_sc_hi_4);
                                    v_out_5[2 * v_w_3] = qmul4_lower4(codes5[v_w_3], v_sc_w_3);
                                    v_out_5[2 * v_w_3 + 1] = qmul4_higher4(codes5[v_w_3], v_sc_w_3);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (32 ^ swz16_1))), "r"(v_out_5[0]), "r"(v_out_5[1]), "r"(v_out_5[2]), "r"(v_out_5[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (48 ^ swz16_1))), "r"(v_out_5[4]), "r"(v_out_5[5]), "r"(v_out_5[6]), "r"(v_out_5[7]) : "memory");
                                unsigned int v_sc_lo_6 = v_sc[3];
                                unsigned int v_sc_hi_7 = v_sc[3] >> 8;
                                unsigned int v_out_8[8];
                                #pragma unroll
                                for (int v_w_4 = 0; v_w_4 < 4; v_w_4++) {
                                    unsigned int v_sc_w_4 = ((v_w_4 < 2) ? v_sc_lo_6 : v_sc_hi_7);
                                    v_out_8[2 * v_w_4] = qmul4_lower4(codes6[v_w_4], v_sc_w_4);
                                    v_out_8[2 * v_w_4 + 1] = qmul4_higher4(codes6[v_w_4], v_sc_w_4);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (64 ^ swz16_1))), "r"(v_out_8[0]), "r"(v_out_8[1]), "r"(v_out_8[2]), "r"(v_out_8[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (80 ^ swz16_1))), "r"(v_out_8[4]), "r"(v_out_8[5]), "r"(v_out_8[6]), "r"(v_out_8[7]) : "memory");
                                unsigned int v_sc_lo_9 = v_sc[3] >> 16;
                                unsigned int v_sc_hi_10 = v_sc[3] >> 24;
                                unsigned int v_out_11[8];
                                #pragma unroll
                                for (int v_w_5 = 0; v_w_5 < 4; v_w_5++) {
                                    unsigned int v_sc_w_5 = ((v_w_5 < 2) ? v_sc_lo_9 : v_sc_hi_10);
                                    v_out_11[2 * v_w_5] = qmul4_lower4(codes7[v_w_5], v_sc_w_5);
                                    v_out_11[2 * v_w_5 + 1] = qmul4_higher4(codes7[v_w_5], v_sc_w_5);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (96 ^ swz16_1))), "r"(v_out_11[0]), "r"(v_out_11[1]), "r"(v_out_11[2]), "r"(v_out_11[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (112 ^ swz16_1))), "r"(v_out_11[4]), "r"(v_out_11[5]), "r"(v_out_11[6]), "r"(v_out_11[7]) : "memory");
                            }
                            if (vx_half == 1) {
                                unsigned int codes2[4];
                                unsigned int codes3_1[4];
                                unsigned int codes4_1[4];
                                unsigned int codes5_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes2[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes2[(0) + 3]))
                                    : "r"(v_codes_src_1 + (32 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes3_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (48 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes4_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (64 ^ swz16_1)));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&codes5_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (80 ^ swz16_1)));
                                unsigned int v_sc_lo_2 = v_sc[1];
                                unsigned int v_sc_hi_2 = v_sc[1] >> 8;
                                unsigned int v_out_3[8];
                                #pragma unroll
                                for (int v_w_6 = 0; v_w_6 < 4; v_w_6++) {
                                    unsigned int v_sc_w_6 = ((v_w_6 < 2) ? v_sc_lo_2 : v_sc_hi_2);
                                    v_out_3[2 * v_w_6] = qmul4_lower4(codes2[v_w_6], v_sc_w_6);
                                    v_out_3[2 * v_w_6 + 1] = qmul4_higher4(codes2[v_w_6], v_sc_w_6);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (64 ^ swz16_1))), "r"(v_out_3[0]), "r"(v_out_3[1]), "r"(v_out_3[2]), "r"(v_out_3[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (80 ^ swz16_1))), "r"(v_out_3[4]), "r"(v_out_3[5]), "r"(v_out_3[6]), "r"(v_out_3[7]) : "memory");
                                unsigned int v_sc_lo_0_1 = v_sc[1] >> 16;
                                unsigned int v_sc_hi_1_2 = v_sc[1] >> 24;
                                unsigned int v_out_2_1[8];
                                #pragma unroll
                                for (int v_w_7 = 0; v_w_7 < 4; v_w_7++) {
                                    unsigned int v_sc_w_7 = ((v_w_7 < 2) ? v_sc_lo_0_1 : v_sc_hi_1_2);
                                    v_out_2_1[2 * v_w_7] = qmul4_lower4(codes3_1[v_w_7], v_sc_w_7);
                                    v_out_2_1[2 * v_w_7 + 1] = qmul4_higher4(codes3_1[v_w_7], v_sc_w_7);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (96 ^ swz16_1))), "r"(v_out_2_1[0]), "r"(v_out_2_1[1]), "r"(v_out_2_1[2]), "r"(v_out_2_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base0_1 + (112 ^ swz16_1))), "r"(v_out_2_1[4]), "r"(v_out_2_1[5]), "r"(v_out_2_1[6]), "r"(v_out_2_1[7]) : "memory");
                                unsigned int v_sc_lo_3_1 = v_sc[2];
                                unsigned int v_sc_hi_4_1 = v_sc[2] >> 8;
                                unsigned int v_out_5_1[8];
                                #pragma unroll
                                for (int v_w_8 = 0; v_w_8 < 4; v_w_8++) {
                                    unsigned int v_sc_w_8 = ((v_w_8 < 2) ? v_sc_lo_3_1 : v_sc_hi_4_1);
                                    v_out_5_1[2 * v_w_8] = qmul4_lower4(codes4_1[v_w_8], v_sc_w_8);
                                    v_out_5_1[2 * v_w_8 + 1] = qmul4_higher4(codes4_1[v_w_8], v_sc_w_8);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (0 ^ swz16_1))), "r"(v_out_5_1[0]), "r"(v_out_5_1[1]), "r"(v_out_5_1[2]), "r"(v_out_5_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (16 ^ swz16_1))), "r"(v_out_5_1[4]), "r"(v_out_5_1[5]), "r"(v_out_5_1[6]), "r"(v_out_5_1[7]) : "memory");
                                unsigned int v_sc_lo_6_1 = v_sc[2] >> 16;
                                unsigned int v_sc_hi_7_1 = v_sc[2] >> 24;
                                unsigned int v_out_8_1[8];
                                #pragma unroll
                                for (int v_w_9 = 0; v_w_9 < 4; v_w_9++) {
                                    unsigned int v_sc_w_9 = ((v_w_9 < 2) ? v_sc_lo_6_1 : v_sc_hi_7_1);
                                    v_out_8_1[2 * v_w_9] = qmul4_lower4(codes5_1[v_w_9], v_sc_w_9);
                                    v_out_8_1[2 * v_w_9 + 1] = qmul4_higher4(codes5_1[v_w_9], v_sc_w_9);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (32 ^ swz16_1))), "r"(v_out_8_1[0]), "r"(v_out_8_1[1]), "r"(v_out_8_1[2]), "r"(v_out_8_1[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_base1_1 + (48 ^ swz16_1))), "r"(v_out_8_1[4]), "r"(v_out_8_1[5]), "r"(v_out_8_1[6]), "r"(v_out_8_1[7]) : "memory");
                            }
                        } else {
                            #pragma unroll 1
                            for (int u_1 = 3 - vx_half; u_1 < 8 - 2 * vx_half; u_1++) {
                                unsigned int v_in_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 3]))
                                    : "r"(v_codes_src_1 + (u_1 << 4 ^ swz16_1)));
                                int v_w_idx_1 = u_1 >> 1;
                                unsigned int v_sc_lo_4 = ((v_w_idx_1 == 0) ? v_sc[0] : v_sc[1]);
                                unsigned int v_sc_hi_3 = ((v_w_idx_1 == 2) ? v_sc[2] : v_sc[3]);
                                unsigned int v_sc_word_1 = ((v_w_idx_1 < 2) ? v_sc_lo_4 : v_sc_hi_3);
                                uint32_t v_sc_pair_1;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(v_sc_pair_1) : "h"((uint16_t)(v_sc_word_1 >> (unsigned int)((u_1 & 1) * 16))));
                                uint32_t _f16x2_mul_5;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_5) : "r"(v_sc_pair_1), "r"(x_kexp_f16x2));
                                v_sc_pair_1 = _f16x2_mul_5;
                                uint32_t v_sc0_1;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(v_sc0_1) : "r"(v_sc_pair_1), "r"(v_sc_pair_1));
                                uint32_t v_sc1_1;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(v_sc1_1) : "r"(v_sc_pair_1), "r"(v_sc_pair_1));
                                unsigned int v_out_4[8];
                                #pragma unroll
                                for (int v_w_10 = 0; v_w_10 < 4; v_w_10++) {
                                    unsigned int v_sc_w_10 = ((v_w_10 < 2) ? v_sc0_1 : v_sc1_1);
                                    unsigned int v_word_1 = v_in_1[v_w_10];
                                    uint32_t _cvt_f16x2_e2m1x2_6;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_6) : "r"((uint32_t)(v_word_1)));
                                    uint32_t f16x2_mul_3;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_3) : "r"(_cvt_f16x2_e2m1x2_6), "r"(v_sc_w_10));
                                    uint16_t e4m3x2_3;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_3) : "r"(f16x2_mul_3));
                                    uint32_t _cvt_f16x2_e2m1x2_7;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_7) : "r"((uint32_t)(v_word_1 >> 8)));
                                    uint32_t f16x2_mul_3_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_3_1) : "r"(_cvt_f16x2_e2m1x2_7), "r"(v_sc_w_10));
                                    uint16_t e4m3x2_3_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_3_1) : "r"(f16x2_mul_3_1));
                                    uint32_t _cvt_f16x2_e2m1x2_8;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_8) : "r"((uint32_t)(v_word_1 >> 16)));
                                    uint32_t f16x2_mul_4;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_4) : "r"(_cvt_f16x2_e2m1x2_8), "r"(v_sc_w_10));
                                    uint16_t e4m3x2_4;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_4) : "r"(f16x2_mul_4));
                                    uint32_t _cvt_f16x2_e2m1x2_9_1;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_9_1) : "r"((uint32_t)(v_word_1 >> 24)));
                                    uint32_t f16x2_mul_5;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_5) : "r"(_cvt_f16x2_e2m1x2_9_1), "r"(v_sc_w_10));
                                    uint16_t e4m3x2_5;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_5) : "r"(f16x2_mul_5));
                                    uint32_t _mov_b32_pack_u16x2_10;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_10) : "h"(e4m3x2_3), "h"(e4m3x2_3_1));
                                    v_out_4[2 * v_w_10] = _mov_b32_pack_u16x2_10;
                                    uint32_t _mov_b32_pack_u16x2_11;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_11) : "h"(e4m3x2_4), "h"(e4m3x2_5));
                                    v_out_4[2 * v_w_10 + 1] = _mov_b32_pack_u16x2_11;
                                }
                                int v_dst_1 = ((u_1 >> 2 == 0) ? v_dst_base0_1 : v_dst_base1_1);
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_1 + ((u_1 & 3) * 2 << 4 ^ swz16_1))), "r"(v_out_4[0]), "r"(v_out_4[1]), "r"(v_out_4[2]), "r"(v_out_4[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_1 + ((u_1 & 3) * 2 + 1 << 4 ^ swz16_1))), "r"(v_out_4[4]), "r"(v_out_4[5]), "r"(v_out_4[6]), "r"(v_out_4[7]) : "memory");
                            }
                        }
                        if (vx_half != 0) {
                            int r_src_row = smem_k_a_addr + (unsigned int)(xk_stage * 26624) + 16384 + (unsigned int)v_row_off_1;
                            int r_dst = x_stage * 32768 + vx_key * 128 + 24576;
                            #pragma unroll
                            for (int r_u = 2; r_u < 4; r_u++) {
                                unsigned int r_out[4];
                                unsigned int r_kexp_bf16x2_1 = (127 - x_kexp << 7) * 65537;
                                #pragma unroll
                                for (int r_half = 0; r_half < 2; r_half++) {
                                    unsigned int r_in[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&r_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 3]))
                                        : "r"(r_src_row + (2 * r_u + r_half << 4 ^ swz16_1)));
                                    if (x_kexp != 0) {
                                        uint32_t r_scaled_16;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_16) : "r"(r_in[0]), "r"(r_kexp_bf16x2_1));
                                        r_in[0] = r_scaled_16;
                                        uint32_t r_scaled_17;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_17) : "r"(r_in[1]), "r"(r_kexp_bf16x2_1));
                                        r_in[1] = r_scaled_17;
                                        uint32_t r_scaled_18;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_18) : "r"(r_in[2]), "r"(r_kexp_bf16x2_1));
                                        r_in[2] = r_scaled_18;
                                        uint32_t r_scaled_19;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_19) : "r"(r_in[3]), "r"(r_kexp_bf16x2_1));
                                        r_in[3] = r_scaled_19;
                                    }
                                    #pragma unroll
                                    for (int r_w = 0; r_w < 2; r_w++) {
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_13;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_13) : "r"(r_in[2 * r_w]));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(r_in[2 * r_w]);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_14;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_14) : "r"(r_in[2 * r_w + 1]));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(r_in[2 * r_w + 1]);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _mov_b32_pack_u16x2_12;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_12) : "h"(_cvt_satfinite_e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_14));
                                        r_out[2 * r_half + r_w] = _mov_b32_pack_u16x2_12;
                                    }
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(r_dst + (4 + r_u << 4 ^ swz16_1))), "r"(r_out[0]), "r"(r_out[1]), "r"(r_out[2]), "r"(r_out[3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (vx_thread == 0) {
                            mbarrier_arrive(kstage_free_addr + ((int)((unsigned int)(vx_tile_cursor + vx_tile) % 3)) * 8);
                            mbarrier_arrive(v_full_addr + ((int)((unsigned int)(vx_tile_cursor + vx_tile) % 2)) * 8);
                        }
                    }
                    vx_tile_cursor += vx_tiles;
                }
                mbarrier_wait(work_full_addr + (vx_work_stage) * 8, _phase_work_full_8);
                uint32_t w_valid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(w_valid_7)
                    : "r"(work_response_addr + vx_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t w_query_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(w_query_7)
                    : "r"(work_response_addr + vx_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (vx_work_stage) * 8);
                vx_work_stage += 1;
                if (vx_work_stage == 2) { vx_work_stage = 0; _phase_work_full_8 ^= 1; }
                if (w_valid_7 == 0) {
                    break;
                }
                vx_query = w_query_7;
            }
        }
    }

    // Cleanup
}

} // extern "C"
