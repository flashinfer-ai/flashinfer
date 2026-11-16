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

__device__ __forceinline__ uint32_t qmul4_direct(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 %0, %1, %2;" : "=r"(y) : "r"(src), "r"(scale));
  return y;
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

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_h64_tmem_v_early_gather_prefill(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int has_sinks, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens, const __grid_constant__ CUtensorMap tmap_o)
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
    #define k_empty_addr (mbar_base + 160)
    #define kstage_free_addr (mbar_base + 200)
    #define sf_full_addr (mbar_base + 240)
    #define kexp_full_addr (mbar_base + 264)
    #define v_full_addr (mbar_base + 280)
    #define v_empty_addr (mbar_base + 296)
    #define s_full_addr (mbar_base + 312)
    #define s_empty_addr (mbar_base + 328)
    #define p_full_addr (mbar_base + 344)
    #define p_recycle_addr (mbar_base + 360)
    #define stats_addr (mbar_base + 376)
    #define o_empty_addr (mbar_base + 392)
    #define o_full_addr (mbar_base + 408)
    #define sum_ready_addr (mbar_base + 424)
    #define sum_empty_addr (mbar_base + 432)
    #define tmem_dealloc_addr (mbar_base + 440)
    #define work_full_addr (mbar_base + 448)
    #define work_empty_addr (mbar_base + 464)
    #define throttle_full_addr (mbar_base + 480)
    #define throttle_empty_addr (mbar_base + 496)

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

    // Mbarrier init (64 barriers at smem_raw[0..512))

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
            // k_full: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // k_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // kstage_free: 5 barriers, init_count=2
            mbarrier_init(smem + 200, 2);
            mbarrier_init(smem + 208, 2);
            mbarrier_init(smem + 216, 2);
            mbarrier_init(smem + 224, 2);
            mbarrier_init(smem + 232, 2);
            // sf_full: 3 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            // kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 280, 2);
            mbarrier_init(smem + 288, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // p_recycle: 2 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 376, 128);
            mbarrier_init(smem + 384, 128);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 392, 128);
            mbarrier_init(smem + 400, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 408, 1);
            mbarrier_init(smem + 416, 1);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 424, 128);
            // sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 432, 128);
            // tmem_dealloc: 1 barriers, init_count=448
            mbarrier_init(smem + 440, 448);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 448, 1);
            mbarrier_init(smem + 456, 1);
            // work_empty: 2 barriers, init_count=736
            mbarrier_init(smem + 464, 736);
            mbarrier_init(smem + 472, 736);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 480, 128);
            mbarrier_init(smem + 488, 128);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 496, 32);
            mbarrier_init(smem + 504, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 512);
    if (warp == 0) {
        int _tmem_hold = smem + 512;
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
    const unsigned int tmem_tmem_vt = taddr + 128;

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
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_stage_base + (sm_head * 64 + (sm_half * 32 + p_vec * 16) ^ (sm_head * 64 + (sm_half * 32 + p_vec * 16) >> 7 & 3) << 4))), "r"(_fp8_0[2 * p_vec]), "r"(_fp8_0[2 * p_vec + 4]), "r"(_fp8_0[2 * p_vec + 1]), "r"(_fp8_0[2 * p_vec + 5]) : "memory");
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
                        int any_rescale = __any_sync(0xFFFFFFFF, smem_scale[c_stage * 128 + lane] < 1.0f);
                        #pragma unroll 1
                        for (int co_hf = 0; co_hf < 2; co_hf++) {
                            if (co_tile > 0) {
                                if (co_hf == 0) {
                                    int _parity_o_full = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                                    mbarrier_wait(o_full_addr, _parity_o_full);
                                } else {
                                    int _parity_o_full_1 = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                                    mbarrier_wait(o_full_addr + 8, _parity_o_full_1);
                                }
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                if (any_rescale != 0) {
                                    float sc_lo[16];
                                    float sc_hi[16];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(0) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(0) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 64)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(4) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 16)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(4) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 64 + 16)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[8])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(8) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 32)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[8])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(8) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 64 + 32)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[12])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_lo[(12) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 48)));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[12])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sc_hi[(12) + 3]))
                                        : "r"(smem_scale_addr + (unsigned int)(c_stage * 512 + 64 + 48)));
                                    #pragma unroll 1
                                    for (int k2 = 0; k2 < 2; k2++) {
                                        unsigned int o_addr_0 = taddr + (unsigned int)co_lane_addr + (unsigned int)(64 * co_hf + 32 * k2);
                                        float o_buf_0[16];
                                        tmem_ld_x16(&o_buf_0[0], o_addr_0);
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_buf_0)[_ls], reinterpret_cast<const float2*>(sc_lo)[_ls]);
                                        tmem_st_x16_f32(o_addr_0, o_buf_0);
                                        unsigned int o_addr_1 = taddr + (unsigned int)co_lane_addr + (unsigned int)(64 * co_hf + 32 * k2 + 16);
                                        float o_buf_1[16];
                                        tmem_ld_x16(&o_buf_1[0], o_addr_1);
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_buf_1)[_ls], reinterpret_cast<const float2*>(sc_hi)[_ls]);
                                        tmem_st_x16_f32(o_addr_1, o_buf_1);
                                    }
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
                                asm volatile("tcgen05.fence::before_thread_sync;");
                            }
                            if (co_hf == 0) {
                                if (co_tile > 0) {
                                    mbarrier_arrive(o_empty_addr);
                                }
                            } else {
                                mbarrier_arrive(o_empty_addr + 8);
                            }
                        }
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
                    float ep_lsm = smem_sum[lane] + smem_sum[lane + 64];
                    mbarrier_arrive(sum_empty_addr);
                    if (co_half == 0 && co_head < num_heads) {
                        float _log2_6;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(total_sum));
                        LSE[co_query * num_heads + co_head] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_6) : -CUDART_INF_F);
                    }
                    int ep_c = lane & 3;
                    int ep_a = lane >> 2;
                    int ep_odd = ep_a & 1;
                    float _rcp_7 = approx_rcp(ep_lsm);
                    float ep_lf = ((ep_lsm > 0.0f) ? _rcp_7 : 0.0f) * output_scale;
                    float ep_f[16];
                    unsigned int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 2 * ep_c);
                    ep_f[0] = __uint_as_float(_shfl_sync_8);
                    ep_f[2] = ep_f[0];
                    unsigned int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 2 * ep_c + 1);
                    ep_f[1] = __uint_as_float(_shfl_sync_9);
                    ep_f[3] = ep_f[1];
                    unsigned int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 8 + 2 * ep_c);
                    ep_f[4] = __uint_as_float(_shfl_sync_10);
                    ep_f[6] = ep_f[4];
                    unsigned int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 8 + 2 * ep_c + 1);
                    ep_f[5] = __uint_as_float(_shfl_sync_11);
                    ep_f[7] = ep_f[5];
                    unsigned int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 16 + 2 * ep_c);
                    ep_f[8] = __uint_as_float(_shfl_sync_12);
                    ep_f[10] = ep_f[8];
                    unsigned int _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 16 + 2 * ep_c + 1);
                    ep_f[9] = __uint_as_float(_shfl_sync_13);
                    ep_f[11] = ep_f[9];
                    unsigned int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 24 + 2 * ep_c);
                    ep_f[12] = __uint_as_float(_shfl_sync_14);
                    ep_f[14] = ep_f[12];
                    unsigned int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, __as_u32(ep_lf), 24 + 2 * ep_c + 1);
                    ep_f[13] = __uint_as_float(_shfl_sync_15);
                    ep_f[15] = ep_f[13];
                    #pragma unroll 1
                    for (int ph = 0; ph < 8; ph++) {
                        int ep_panel = ph >> 1;
                        int ep_h = ph & 1;
                        float ep_buf[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[3])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[5])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[6])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[7])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[9])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[10])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[11])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[13])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[14])), "=r"(*reinterpret_cast<uint32_t*>(&ep_buf[15]))
                            : "r"(taddr + (unsigned int)co_lane_addr + (unsigned int)(ep_h << 20) + (unsigned int)(32 * ep_panel)));
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf)[_ls], reinterpret_cast<const float2*>(ep_f)[_ls]);
                        float ep_pr[16];
                        int ep_dim = 32 * (4 * ep_panel + co_row_base / 32) + 16 * ep_h;
                        if (ep_panel == 3 && co_row_base >= 64) {
                            float ep_v0_0 = ep_buf[0];
                            float ep_v1_0 = ep_buf[2];
                            unsigned int _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_0 : ep_v1_0)), 4);
                            float ep_recv_0 = __uint_as_float(_shfl_xor_sync_16);
                            ep_pr[0] = ((ep_odd != 0) ? ep_recv_0 : ep_v0_0);
                            ep_pr[1] = ((ep_odd != 0) ? ep_v1_0 : ep_recv_0);
                            float ep_v0_1 = ep_buf[1];
                            float ep_v1_1 = ep_buf[3];
                            unsigned int _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_1 : ep_v1_1)), 4);
                            float ep_recv_1 = __uint_as_float(_shfl_xor_sync_17);
                            ep_pr[2] = ((ep_odd != 0) ? ep_recv_1 : ep_v0_1);
                            ep_pr[3] = ((ep_odd != 0) ? ep_v1_1 : ep_recv_1);
                            float ep_v0_2 = ep_buf[4];
                            float ep_v1_2 = ep_buf[6];
                            unsigned int _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_2 : ep_v1_2)), 4);
                            float ep_recv_2 = __uint_as_float(_shfl_xor_sync_18);
                            ep_pr[4] = ((ep_odd != 0) ? ep_recv_2 : ep_v0_2);
                            ep_pr[5] = ((ep_odd != 0) ? ep_v1_2 : ep_recv_2);
                            float ep_v0_3 = ep_buf[5];
                            float ep_v1_3 = ep_buf[7];
                            unsigned int _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_3 : ep_v1_3)), 4);
                            float ep_recv_3 = __uint_as_float(_shfl_xor_sync_19);
                            ep_pr[6] = ((ep_odd != 0) ? ep_recv_3 : ep_v0_3);
                            ep_pr[7] = ((ep_odd != 0) ? ep_v1_3 : ep_recv_3);
                            float ep_v0_4 = ep_buf[8];
                            float ep_v1_4 = ep_buf[10];
                            unsigned int _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_4 : ep_v1_4)), 4);
                            float ep_recv_4 = __uint_as_float(_shfl_xor_sync_20);
                            ep_pr[8] = ((ep_odd != 0) ? ep_recv_4 : ep_v0_4);
                            ep_pr[9] = ((ep_odd != 0) ? ep_v1_4 : ep_recv_4);
                            float ep_v0_5 = ep_buf[9];
                            float ep_v1_5 = ep_buf[11];
                            unsigned int _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_5 : ep_v1_5)), 4);
                            float ep_recv_5 = __uint_as_float(_shfl_xor_sync_21);
                            ep_pr[10] = ((ep_odd != 0) ? ep_recv_5 : ep_v0_5);
                            ep_pr[11] = ((ep_odd != 0) ? ep_v1_5 : ep_recv_5);
                            float ep_v0_6 = ep_buf[12];
                            float ep_v1_6 = ep_buf[14];
                            unsigned int _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_6 : ep_v1_6)), 4);
                            float ep_recv_6 = __uint_as_float(_shfl_xor_sync_22);
                            ep_pr[12] = ((ep_odd != 0) ? ep_recv_6 : ep_v0_6);
                            ep_pr[13] = ((ep_odd != 0) ? ep_v1_6 : ep_recv_6);
                            float ep_v0_7 = ep_buf[13];
                            float ep_v1_7 = ep_buf[15];
                            unsigned int _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, __as_u32(((ep_odd != 0) ? ep_v0_7 : ep_v1_7)), 4);
                            float ep_recv_7 = __uint_as_float(_shfl_xor_sync_23);
                            ep_pr[14] = ((ep_odd != 0) ? ep_recv_7 : ep_v0_7);
                            ep_pr[15] = ((ep_odd != 0) ? ep_v1_7 : ep_recv_7);
                            ep_dim = ep_dim + ep_a + 7 * ep_odd;
                        } else {
                            ep_pr[0] = ep_buf[0];
                            ep_pr[1] = ep_buf[2];
                            ep_pr[2] = ep_buf[1];
                            ep_pr[3] = ep_buf[3];
                            ep_pr[4] = ep_buf[4];
                            ep_pr[5] = ep_buf[6];
                            ep_pr[6] = ep_buf[5];
                            ep_pr[7] = ep_buf[7];
                            ep_pr[8] = ep_buf[8];
                            ep_pr[9] = ep_buf[10];
                            ep_pr[10] = ep_buf[9];
                            ep_pr[11] = ep_buf[11];
                            ep_pr[12] = ep_buf[12];
                            ep_pr[13] = ep_buf[14];
                            ep_pr[14] = ep_buf[13];
                            ep_pr[15] = ep_buf[15];
                            ep_dim = ep_dim + 2 * ep_a;
                        }
                        unsigned int ep_pack[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_pr[_lp*2 + 0], ep_pr[_lp*2+1 + 0]));
                            ep_pack[_lp] = *(uint32_t*)&_bf2;
                        }
                        int ep_head_0 = 2 * ep_c;
                        if (ep_head_0 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_0) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[0];
                        }
                        int ep_head_1 = 2 * ep_c + 1;
                        if (ep_head_1 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_1) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[1];
                        }
                        int ep_head_2 = 8 + 2 * ep_c;
                        if (ep_head_2 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_2) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[2];
                        }
                        int ep_head_3 = 8 + 2 * ep_c + 1;
                        if (ep_head_3 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_3) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[3];
                        }
                        int ep_head_4 = 16 + 2 * ep_c;
                        if (ep_head_4 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_4) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[4];
                        }
                        int ep_head_5 = 16 + 2 * ep_c + 1;
                        if (ep_head_5 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_5) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[5];
                        }
                        int ep_head_6 = 24 + 2 * ep_c;
                        if (ep_head_6 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_6) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[6];
                        }
                        int ep_head_7 = 24 + 2 * ep_c + 1;
                        if (ep_head_7 < num_heads) {
                            *(reinterpret_cast<unsigned int*>(O + (((long long)co_query * (long long)num_heads + (long long)ep_head_7) * 512ll + (long long)ep_dim)) + (0)) = ep_pack[7];
                        }
                        if (ph == 3) {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
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
                    uint32_t _elected_6 = elect_sync();
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_q_a_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 0)), "l"(_tcgen05_cp_desc_7), "r"(_elected_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_q_a_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 8)), "l"(_tcgen05_cp_desc_8), "r"(_elected_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_q_a_addr + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 16)), "l"(_tcgen05_cp_desc_9), "r"(_elected_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_q_a_addr + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 24)), "l"(_tcgen05_cp_desc_10), "r"(_elected_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_q_b_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 0)), "l"(_tcgen05_cp_desc_11), "r"(_elected_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(smem_q_b_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 8)), "l"(_tcgen05_cp_desc_12), "r"(_elected_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(smem_q_b_addr + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 16)), "l"(_tcgen05_cp_desc_13), "r"(_elected_6)
                            : "memory");
                    }
                    #pragma unroll 1
                    for (int mma_tile = 0; mma_tile < mma_tiles; mma_tile++) {
                        int m_stage = mma_tile_cursor + mma_tile & 1;
                        int mk_stage = (mma_tile_cursor + mma_tile) % 5;
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
                            uint64_t _mma_ss_a_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135267472, 1);
                            }
                        }
                        elect_commit(k_empty_addr + ((int)((unsigned int)(mma_tile_cursor + mma_tile) % 5)) * 8);
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
                        int _mma_p32 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_scratch), "r"(_mma_p32), "r"(tmem_tmem_vt + 64 * pv_stage), "r"(((pv_first) ? 0 : 1)) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (32))), "r"(_mma_p32), "r"(tmem_tmem_vt + (64 * pv_stage + 16)), "r"(((pv_first) ? 0 : 1)) : "memory");
                        elect_commit(o_full_addr);
                        int _parity_o_empty_1 = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                        mbarrier_wait(o_empty_addr + 8, _parity_o_empty_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (64))), "r"(_mma_p32), "r"(tmem_tmem_vt + (64 * pv_stage + 32)), "r"(((pv_first) ? 0 : 1)) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (96))), "r"(_mma_p32), "r"(tmem_tmem_vt + (64 * pv_stage + 32 + 16)), "r"(((pv_first) ? 0 : 1)) : "memory");
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
                        int k_stage = (ld_tile_cursor + ld_tile) % 5;
                        int lt_on = ld_tile_cursor + ld_tile < 320;
                        int _stage_k_full = (int)((unsigned int)(ld_tile_cursor + ld_tile) % 5);
                        int _parity_k_full = (int)((unsigned int)(ld_tile_cursor + ld_tile) / 5 % 2);
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
                        int _stage_k_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile + 3) % 5);
                        int _parity_k_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile + 3) / 5 % 2) ^ 1;
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
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int ld_s[8];
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(ld_s[0]), "=r"(ld_s[1]), "=r"(ld_s[2]), "=r"(ld_s[3])
                            : "r"((unsigned int)sf_base + (unsigned int)(lane * 32))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(ld_s[4]), "=r"(ld_s[5]), "=r"(ld_s[6]), "=r"(ld_s[7])
                            : "r"((unsigned int)sf_base + (unsigned int)((32 + lane) * 32))
                            : "memory");
                        unsigned int ld_vt = taddr + (unsigned int)(ld_row_base << 16) + (unsigned int)(128 + 64 * p_stage);
                        unsigned int ld_codes = smem_k_a_addr + (unsigned int)(k_stage * 26624);
                        unsigned int ld_lo[16];
                        unsigned int ld_hi[16];
                        unsigned int c[8];
                        unsigned int swz = (unsigned int)((lane & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                            : "r"((unsigned int)ld_codes + (unsigned int)(lane * 128) + ((unsigned int)(ld_warp * 16) ^ swz))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
                            : "r"((unsigned int)ld_codes + (unsigned int)((32 + lane) * 128) + ((unsigned int)(ld_warp * 16) ^ swz))
                            : "memory");
                        int src_lo = 8 * (ld_warp & 3) + (lane & 3);
                        int src_hi = src_lo + 4;
                        if (kexp_u == 0) {
                            unsigned int sl_0 = __shfl_sync(0xFFFFFFFF, ld_s[0], src_lo);
                            unsigned int sh_0 = __shfl_sync(0xFFFFFFFF, ld_s[0], src_hi);
                            ld_lo[0] = qmul4_direct(c[0], sl_0);
                            ld_lo[2] = qmul4_direct(c[0] >> 4, sl_0);
                            ld_hi[0] = qmul4_direct(c[1], sh_0);
                            ld_hi[2] = qmul4_direct(c[1] >> 4, sh_0);
                            unsigned int sl_1 = __shfl_sync(0xFFFFFFFF, ld_s[2], src_lo);
                            unsigned int sh_1 = __shfl_sync(0xFFFFFFFF, ld_s[2], src_hi);
                            ld_lo[1] = qmul4_direct(c[2], sl_1);
                            ld_lo[3] = qmul4_direct(c[2] >> 4, sl_1);
                            ld_hi[1] = qmul4_direct(c[3], sh_1);
                            ld_hi[3] = qmul4_direct(c[3] >> 4, sh_1);
                            unsigned int sl_2 = __shfl_sync(0xFFFFFFFF, ld_s[4], src_lo);
                            unsigned int sh_2 = __shfl_sync(0xFFFFFFFF, ld_s[4], src_hi);
                            ld_lo[4] = qmul4_direct(c[4], sl_2);
                            ld_lo[6] = qmul4_direct(c[4] >> 4, sl_2);
                            ld_hi[4] = qmul4_direct(c[5], sh_2);
                            ld_hi[6] = qmul4_direct(c[5] >> 4, sh_2);
                            unsigned int sl_3 = __shfl_sync(0xFFFFFFFF, ld_s[6], src_lo);
                            unsigned int sh_3 = __shfl_sync(0xFFFFFFFF, ld_s[6], src_hi);
                            ld_lo[5] = qmul4_direct(c[6], sl_3);
                            ld_lo[7] = qmul4_direct(c[6] >> 4, sl_3);
                            ld_hi[5] = qmul4_direct(c[7], sh_3);
                            ld_hi[7] = qmul4_direct(c[7] >> 4, sh_3);
                        } else {
                            unsigned int sl_4 = __shfl_sync(0xFFFFFFFF, ld_s[0], src_lo);
                            unsigned int sh_4 = __shfl_sync(0xFFFFFFFF, ld_s[0], src_hi);
                            uint32_t _cvt_f16x2_e4m3x2_9;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_9) : "h"((uint16_t)(sl_4)));
                            uint32_t sl01_0;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_0) : "r"(_cvt_f16x2_e4m3x2_9), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_10;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_10) : "h"((uint16_t)(sl_4 >> 16)));
                            uint32_t sl23_0;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_0) : "r"(_cvt_f16x2_e4m3x2_10), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_11;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_11) : "h"((uint16_t)(sh_4)));
                            uint32_t sh01_0;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_0) : "r"(_cvt_f16x2_e4m3x2_11), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_12;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_12) : "h"((uint16_t)(sh_4 >> 16)));
                            uint32_t sh23_0;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_0) : "r"(_cvt_f16x2_e4m3x2_12), "r"(kexp_f16x2));
                            unsigned int x = c[0] & 252645135;
                            unsigned int p = x | x >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_13;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_13) : "r"((uint32_t)(p)));
                            uint32_t prod_lo;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo) : "r"(_cvt_f16x2_e2m1x2_13), "r"(sl01_0));
                            uint32_t _cvt_f16x2_e2m1x2_14;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_14) : "r"((uint32_t)(p >> 16)));
                            uint32_t prod_hi;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi) : "r"(_cvt_f16x2_e2m1x2_14), "r"(sl23_0));
                            uint16_t lo;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(prod_lo));
                            uint16_t hi;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(prod_hi));
                            uint32_t _mov_b32_pack_u16x2_15;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_15) : "h"(lo), "h"(hi));
                            ld_lo[0] = _mov_b32_pack_u16x2_15;
                            unsigned int x_0 = c[0] >> 4 & 252645135;
                            unsigned int p_0 = x_0 | x_0 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_16;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_16) : "r"((uint32_t)(p_0)));
                            uint32_t prod_lo_0;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_0) : "r"(_cvt_f16x2_e2m1x2_16), "r"(sl01_0));
                            uint32_t _cvt_f16x2_e2m1x2_17;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_17) : "r"((uint32_t)(p_0 >> 16)));
                            uint32_t prod_hi_0;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_0) : "r"(_cvt_f16x2_e2m1x2_17), "r"(sl23_0));
                            uint16_t lo_0;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_0) : "r"(prod_lo_0));
                            uint16_t hi_0;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_0) : "r"(prod_hi_0));
                            uint32_t _mov_b32_pack_u16x2_18;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_18) : "h"(lo_0), "h"(hi_0));
                            ld_lo[2] = _mov_b32_pack_u16x2_18;
                            unsigned int x_1 = c[1] & 252645135;
                            unsigned int p_1 = x_1 | x_1 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_19;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_19) : "r"((uint32_t)(p_1)));
                            uint32_t prod_lo_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_1) : "r"(_cvt_f16x2_e2m1x2_19), "r"(sh01_0));
                            uint32_t _cvt_f16x2_e2m1x2_20;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_20) : "r"((uint32_t)(p_1 >> 16)));
                            uint32_t prod_hi_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_1) : "r"(_cvt_f16x2_e2m1x2_20), "r"(sh23_0));
                            uint16_t lo_1;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_1) : "r"(prod_lo_1));
                            uint16_t hi_1;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_1) : "r"(prod_hi_1));
                            uint32_t _mov_b32_pack_u16x2_21;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_21) : "h"(lo_1), "h"(hi_1));
                            ld_hi[0] = _mov_b32_pack_u16x2_21;
                            unsigned int x_2 = c[1] >> 4 & 252645135;
                            unsigned int p_2 = x_2 | x_2 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_22;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_22) : "r"((uint32_t)(p_2)));
                            uint32_t prod_lo_2;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_2) : "r"(_cvt_f16x2_e2m1x2_22), "r"(sh01_0));
                            uint32_t _cvt_f16x2_e2m1x2_23;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_23) : "r"((uint32_t)(p_2 >> 16)));
                            uint32_t prod_hi_2;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_2) : "r"(_cvt_f16x2_e2m1x2_23), "r"(sh23_0));
                            uint16_t lo_2;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_2) : "r"(prod_lo_2));
                            uint16_t hi_2;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_2) : "r"(prod_hi_2));
                            uint32_t _mov_b32_pack_u16x2_24;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_24) : "h"(lo_2), "h"(hi_2));
                            ld_hi[2] = _mov_b32_pack_u16x2_24;
                            unsigned int sl_5 = __shfl_sync(0xFFFFFFFF, ld_s[2], src_lo);
                            unsigned int sh_5 = __shfl_sync(0xFFFFFFFF, ld_s[2], src_hi);
                            uint32_t _cvt_f16x2_e4m3x2_25;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_25) : "h"((uint16_t)(sl_5)));
                            uint32_t sl01_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_1) : "r"(_cvt_f16x2_e4m3x2_25), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_26;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_26) : "h"((uint16_t)(sl_5 >> 16)));
                            uint32_t sl23_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_1) : "r"(_cvt_f16x2_e4m3x2_26), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_27;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_27) : "h"((uint16_t)(sh_5)));
                            uint32_t sh01_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_1) : "r"(_cvt_f16x2_e4m3x2_27), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_28;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_28) : "h"((uint16_t)(sh_5 >> 16)));
                            uint32_t sh23_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_1) : "r"(_cvt_f16x2_e4m3x2_28), "r"(kexp_f16x2));
                            unsigned int x_3 = c[2] & 252645135;
                            unsigned int p_3 = x_3 | x_3 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_29;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_29) : "r"((uint32_t)(p_3)));
                            uint32_t prod_lo_3;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_3) : "r"(_cvt_f16x2_e2m1x2_29), "r"(sl01_1));
                            uint32_t _cvt_f16x2_e2m1x2_30;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_30) : "r"((uint32_t)(p_3 >> 16)));
                            uint32_t prod_hi_3;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_3) : "r"(_cvt_f16x2_e2m1x2_30), "r"(sl23_1));
                            uint16_t lo_3;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_3) : "r"(prod_lo_3));
                            uint16_t hi_3;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_3) : "r"(prod_hi_3));
                            uint32_t _mov_b32_pack_u16x2_31;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_31) : "h"(lo_3), "h"(hi_3));
                            ld_lo[1] = _mov_b32_pack_u16x2_31;
                            unsigned int x_4 = c[2] >> 4 & 252645135;
                            unsigned int p_4 = x_4 | x_4 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_32;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_32) : "r"((uint32_t)(p_4)));
                            uint32_t prod_lo_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_4) : "r"(_cvt_f16x2_e2m1x2_32), "r"(sl01_1));
                            uint32_t _cvt_f16x2_e2m1x2_33;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_33) : "r"((uint32_t)(p_4 >> 16)));
                            uint32_t prod_hi_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_4) : "r"(_cvt_f16x2_e2m1x2_33), "r"(sl23_1));
                            uint16_t lo_4;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_4) : "r"(prod_lo_4));
                            uint16_t hi_4;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_4) : "r"(prod_hi_4));
                            uint32_t _mov_b32_pack_u16x2_34;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_34) : "h"(lo_4), "h"(hi_4));
                            ld_lo[3] = _mov_b32_pack_u16x2_34;
                            unsigned int x_5 = c[3] & 252645135;
                            unsigned int p_5 = x_5 | x_5 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_35;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_35) : "r"((uint32_t)(p_5)));
                            uint32_t prod_lo_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_5) : "r"(_cvt_f16x2_e2m1x2_35), "r"(sh01_1));
                            uint32_t _cvt_f16x2_e2m1x2_36;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_36) : "r"((uint32_t)(p_5 >> 16)));
                            uint32_t prod_hi_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_5) : "r"(_cvt_f16x2_e2m1x2_36), "r"(sh23_1));
                            uint16_t lo_5;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_5) : "r"(prod_lo_5));
                            uint16_t hi_5;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_5) : "r"(prod_hi_5));
                            uint32_t _mov_b32_pack_u16x2_37;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_37) : "h"(lo_5), "h"(hi_5));
                            ld_hi[1] = _mov_b32_pack_u16x2_37;
                            unsigned int x_6 = c[3] >> 4 & 252645135;
                            unsigned int p_6 = x_6 | x_6 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_38;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_38) : "r"((uint32_t)(p_6)));
                            uint32_t prod_lo_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_6) : "r"(_cvt_f16x2_e2m1x2_38), "r"(sh01_1));
                            uint32_t _cvt_f16x2_e2m1x2_39;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_39) : "r"((uint32_t)(p_6 >> 16)));
                            uint32_t prod_hi_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_6) : "r"(_cvt_f16x2_e2m1x2_39), "r"(sh23_1));
                            uint16_t lo_6;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_6) : "r"(prod_lo_6));
                            uint16_t hi_6;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_6) : "r"(prod_hi_6));
                            uint32_t _mov_b32_pack_u16x2_40;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_40) : "h"(lo_6), "h"(hi_6));
                            ld_hi[3] = _mov_b32_pack_u16x2_40;
                            unsigned int sl_6 = __shfl_sync(0xFFFFFFFF, ld_s[4], src_lo);
                            unsigned int sh_6 = __shfl_sync(0xFFFFFFFF, ld_s[4], src_hi);
                            uint32_t _cvt_f16x2_e4m3x2_41;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_41) : "h"((uint16_t)(sl_6)));
                            uint32_t sl01_2;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_2) : "r"(_cvt_f16x2_e4m3x2_41), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_42;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_42) : "h"((uint16_t)(sl_6 >> 16)));
                            uint32_t sl23_2;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_2) : "r"(_cvt_f16x2_e4m3x2_42), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_43;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_43) : "h"((uint16_t)(sh_6)));
                            uint32_t sh01_2;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_2) : "r"(_cvt_f16x2_e4m3x2_43), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_44;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_44) : "h"((uint16_t)(sh_6 >> 16)));
                            uint32_t sh23_2;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_2) : "r"(_cvt_f16x2_e4m3x2_44), "r"(kexp_f16x2));
                            unsigned int x_7 = c[4] & 252645135;
                            unsigned int p_7 = x_7 | x_7 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_45;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_45) : "r"((uint32_t)(p_7)));
                            uint32_t prod_lo_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_7) : "r"(_cvt_f16x2_e2m1x2_45), "r"(sl01_2));
                            uint32_t _cvt_f16x2_e2m1x2_46;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_46) : "r"((uint32_t)(p_7 >> 16)));
                            uint32_t prod_hi_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_7) : "r"(_cvt_f16x2_e2m1x2_46), "r"(sl23_2));
                            uint16_t lo_7;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_7) : "r"(prod_lo_7));
                            uint16_t hi_7;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_7) : "r"(prod_hi_7));
                            uint32_t _mov_b32_pack_u16x2_47;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_47) : "h"(lo_7), "h"(hi_7));
                            ld_lo[4] = _mov_b32_pack_u16x2_47;
                            unsigned int x_8 = c[4] >> 4 & 252645135;
                            unsigned int p_8 = x_8 | x_8 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_48;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_48) : "r"((uint32_t)(p_8)));
                            uint32_t prod_lo_8;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_8) : "r"(_cvt_f16x2_e2m1x2_48), "r"(sl01_2));
                            uint32_t _cvt_f16x2_e2m1x2_49;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_49) : "r"((uint32_t)(p_8 >> 16)));
                            uint32_t prod_hi_8;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_8) : "r"(_cvt_f16x2_e2m1x2_49), "r"(sl23_2));
                            uint16_t lo_8;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_8) : "r"(prod_lo_8));
                            uint16_t hi_8;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_8) : "r"(prod_hi_8));
                            uint32_t _mov_b32_pack_u16x2_50;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_50) : "h"(lo_8), "h"(hi_8));
                            ld_lo[6] = _mov_b32_pack_u16x2_50;
                            unsigned int x_9 = c[5] & 252645135;
                            unsigned int p_9 = x_9 | x_9 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_51;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_51) : "r"((uint32_t)(p_9)));
                            uint32_t prod_lo_9;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_9) : "r"(_cvt_f16x2_e2m1x2_51), "r"(sh01_2));
                            uint32_t _cvt_f16x2_e2m1x2_52;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_52) : "r"((uint32_t)(p_9 >> 16)));
                            uint32_t prod_hi_9;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_9) : "r"(_cvt_f16x2_e2m1x2_52), "r"(sh23_2));
                            uint16_t lo_9;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_9) : "r"(prod_lo_9));
                            uint16_t hi_9;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_9) : "r"(prod_hi_9));
                            uint32_t _mov_b32_pack_u16x2_53;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_53) : "h"(lo_9), "h"(hi_9));
                            ld_hi[4] = _mov_b32_pack_u16x2_53;
                            unsigned int x_10 = c[5] >> 4 & 252645135;
                            unsigned int p_10 = x_10 | x_10 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_54;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_54) : "r"((uint32_t)(p_10)));
                            uint32_t prod_lo_10;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_10) : "r"(_cvt_f16x2_e2m1x2_54), "r"(sh01_2));
                            uint32_t _cvt_f16x2_e2m1x2_55;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_55) : "r"((uint32_t)(p_10 >> 16)));
                            uint32_t prod_hi_10;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_10) : "r"(_cvt_f16x2_e2m1x2_55), "r"(sh23_2));
                            uint16_t lo_10;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_10) : "r"(prod_lo_10));
                            uint16_t hi_10;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_10) : "r"(prod_hi_10));
                            uint32_t _mov_b32_pack_u16x2_56;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_56) : "h"(lo_10), "h"(hi_10));
                            ld_hi[6] = _mov_b32_pack_u16x2_56;
                            unsigned int sl_7 = __shfl_sync(0xFFFFFFFF, ld_s[6], src_lo);
                            unsigned int sh_7 = __shfl_sync(0xFFFFFFFF, ld_s[6], src_hi);
                            uint32_t _cvt_f16x2_e4m3x2_57;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_57) : "h"((uint16_t)(sl_7)));
                            uint32_t sl01_3;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_3) : "r"(_cvt_f16x2_e4m3x2_57), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_58;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_58) : "h"((uint16_t)(sl_7 >> 16)));
                            uint32_t sl23_3;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_3) : "r"(_cvt_f16x2_e4m3x2_58), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_59;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_59) : "h"((uint16_t)(sh_7)));
                            uint32_t sh01_3;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_3) : "r"(_cvt_f16x2_e4m3x2_59), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_60;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_60) : "h"((uint16_t)(sh_7 >> 16)));
                            uint32_t sh23_3;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_3) : "r"(_cvt_f16x2_e4m3x2_60), "r"(kexp_f16x2));
                            unsigned int x_11 = c[6] & 252645135;
                            unsigned int p_11 = x_11 | x_11 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_61;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_61) : "r"((uint32_t)(p_11)));
                            uint32_t prod_lo_11;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_11) : "r"(_cvt_f16x2_e2m1x2_61), "r"(sl01_3));
                            uint32_t _cvt_f16x2_e2m1x2_62;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_62) : "r"((uint32_t)(p_11 >> 16)));
                            uint32_t prod_hi_11;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_11) : "r"(_cvt_f16x2_e2m1x2_62), "r"(sl23_3));
                            uint16_t lo_11;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_11) : "r"(prod_lo_11));
                            uint16_t hi_11;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_11) : "r"(prod_hi_11));
                            uint32_t _mov_b32_pack_u16x2_63;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_63) : "h"(lo_11), "h"(hi_11));
                            ld_lo[5] = _mov_b32_pack_u16x2_63;
                            unsigned int x_12 = c[6] >> 4 & 252645135;
                            unsigned int p_12 = x_12 | x_12 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_64;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_64) : "r"((uint32_t)(p_12)));
                            uint32_t prod_lo_12;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_12) : "r"(_cvt_f16x2_e2m1x2_64), "r"(sl01_3));
                            uint32_t _cvt_f16x2_e2m1x2_65;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_65) : "r"((uint32_t)(p_12 >> 16)));
                            uint32_t prod_hi_12;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_12) : "r"(_cvt_f16x2_e2m1x2_65), "r"(sl23_3));
                            uint16_t lo_12;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_12) : "r"(prod_lo_12));
                            uint16_t hi_12;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_12) : "r"(prod_hi_12));
                            uint32_t _mov_b32_pack_u16x2_66;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_66) : "h"(lo_12), "h"(hi_12));
                            ld_lo[7] = _mov_b32_pack_u16x2_66;
                            unsigned int x_13 = c[7] & 252645135;
                            unsigned int p_13 = x_13 | x_13 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_67;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_67) : "r"((uint32_t)(p_13)));
                            uint32_t prod_lo_13;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_13) : "r"(_cvt_f16x2_e2m1x2_67), "r"(sh01_3));
                            uint32_t _cvt_f16x2_e2m1x2_68;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_68) : "r"((uint32_t)(p_13 >> 16)));
                            uint32_t prod_hi_13;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_13) : "r"(_cvt_f16x2_e2m1x2_68), "r"(sh23_3));
                            uint16_t lo_13;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_13) : "r"(prod_lo_13));
                            uint16_t hi_13;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_13) : "r"(prod_hi_13));
                            uint32_t _mov_b32_pack_u16x2_69;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_69) : "h"(lo_13), "h"(hi_13));
                            ld_hi[5] = _mov_b32_pack_u16x2_69;
                            unsigned int x_14 = c[7] >> 4 & 252645135;
                            unsigned int p_14 = x_14 | x_14 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_70;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_70) : "r"((uint32_t)(p_14)));
                            uint32_t prod_lo_14;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_14) : "r"(_cvt_f16x2_e2m1x2_70), "r"(sh01_3));
                            uint32_t _cvt_f16x2_e2m1x2_71;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_71) : "r"((uint32_t)(p_14 >> 16)));
                            uint32_t prod_hi_14;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_14) : "r"(_cvt_f16x2_e2m1x2_71), "r"(sh23_3));
                            uint16_t lo_14;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_14) : "r"(prod_lo_14));
                            uint16_t hi_14;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_14) : "r"(prod_hi_14));
                            uint32_t _mov_b32_pack_u16x2_72;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_72) : "h"(lo_14), "h"(hi_14));
                            ld_hi[7] = _mov_b32_pack_u16x2_72;
                        }
                        unsigned int c_0[8];
                        unsigned int swz_0 = (unsigned int)((lane & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c_0[0]), "=r"(c_0[1]), "=r"(c_0[2]), "=r"(c_0[3])
                            : "r"((unsigned int)ld_codes + (unsigned int)(lane * 128) + ((unsigned int)((ld_warp + 4) * 16) ^ swz_0))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c_0[4]), "=r"(c_0[5]), "=r"(c_0[6]), "=r"(c_0[7])
                            : "r"((unsigned int)ld_codes + (unsigned int)((32 + lane) * 128) + ((unsigned int)((ld_warp + 4) * 16) ^ swz_0))
                            : "memory");
                        int src_lo_0 = 8 * (ld_warp + 4 & 3) + (lane & 3);
                        int src_hi_0 = src_lo_0 + 4;
                        if (kexp_u == 0) {
                            unsigned int sl_8 = __shfl_sync(0xFFFFFFFF, ld_s[1], src_lo_0);
                            unsigned int sh_8 = __shfl_sync(0xFFFFFFFF, ld_s[1], src_hi_0);
                            ld_lo[8] = qmul4_direct(c_0[0], sl_8);
                            ld_lo[10] = qmul4_direct(c_0[0] >> 4, sl_8);
                            ld_hi[8] = qmul4_direct(c_0[1], sh_8);
                            ld_hi[10] = qmul4_direct(c_0[1] >> 4, sh_8);
                            unsigned int sl_9 = __shfl_sync(0xFFFFFFFF, ld_s[3], src_lo_0);
                            unsigned int sh_9 = __shfl_sync(0xFFFFFFFF, ld_s[3], src_hi_0);
                            ld_lo[9] = qmul4_direct(c_0[2], sl_9);
                            ld_lo[11] = qmul4_direct(c_0[2] >> 4, sl_9);
                            ld_hi[9] = qmul4_direct(c_0[3], sh_9);
                            ld_hi[11] = qmul4_direct(c_0[3] >> 4, sh_9);
                            unsigned int sl_10 = __shfl_sync(0xFFFFFFFF, ld_s[5], src_lo_0);
                            unsigned int sh_10 = __shfl_sync(0xFFFFFFFF, ld_s[5], src_hi_0);
                            ld_lo[12] = qmul4_direct(c_0[4], sl_10);
                            ld_lo[14] = qmul4_direct(c_0[4] >> 4, sl_10);
                            ld_hi[12] = qmul4_direct(c_0[5], sh_10);
                            ld_hi[14] = qmul4_direct(c_0[5] >> 4, sh_10);
                            unsigned int sl_11 = __shfl_sync(0xFFFFFFFF, ld_s[7], src_lo_0);
                            unsigned int sh_11 = __shfl_sync(0xFFFFFFFF, ld_s[7], src_hi_0);
                            ld_lo[13] = qmul4_direct(c_0[6], sl_11);
                            ld_lo[15] = qmul4_direct(c_0[6] >> 4, sl_11);
                            ld_hi[13] = qmul4_direct(c_0[7], sh_11);
                            ld_hi[15] = qmul4_direct(c_0[7] >> 4, sh_11);
                        } else {
                            unsigned int sl_12 = __shfl_sync(0xFFFFFFFF, ld_s[1], src_lo_0);
                            unsigned int sh_12 = __shfl_sync(0xFFFFFFFF, ld_s[1], src_hi_0);
                            uint32_t _cvt_f16x2_e4m3x2_73;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_73) : "h"((uint16_t)(sl_12)));
                            uint32_t sl01_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_4) : "r"(_cvt_f16x2_e4m3x2_73), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_74;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_74) : "h"((uint16_t)(sl_12 >> 16)));
                            uint32_t sl23_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_4) : "r"(_cvt_f16x2_e4m3x2_74), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_75;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_75) : "h"((uint16_t)(sh_12)));
                            uint32_t sh01_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_4) : "r"(_cvt_f16x2_e4m3x2_75), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_76;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_76) : "h"((uint16_t)(sh_12 >> 16)));
                            uint32_t sh23_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_4) : "r"(_cvt_f16x2_e4m3x2_76), "r"(kexp_f16x2));
                            unsigned int x_15 = c_0[0] & 252645135;
                            unsigned int p_15 = x_15 | x_15 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_77;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_77) : "r"((uint32_t)(p_15)));
                            uint32_t prod_lo_15;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_15) : "r"(_cvt_f16x2_e2m1x2_77), "r"(sl01_4));
                            uint32_t _cvt_f16x2_e2m1x2_78;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_78) : "r"((uint32_t)(p_15 >> 16)));
                            uint32_t prod_hi_15;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_15) : "r"(_cvt_f16x2_e2m1x2_78), "r"(sl23_4));
                            uint16_t lo_15;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_15) : "r"(prod_lo_15));
                            uint16_t hi_15;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_15) : "r"(prod_hi_15));
                            uint32_t _mov_b32_pack_u16x2_79;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_79) : "h"(lo_15), "h"(hi_15));
                            ld_lo[8] = _mov_b32_pack_u16x2_79;
                            unsigned int x_15_1 = c_0[0] >> 4 & 252645135;
                            unsigned int p_15_1 = x_15_1 | x_15_1 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_80;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_80) : "r"((uint32_t)(p_15_1)));
                            uint32_t prod_lo_15_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_15_1) : "r"(_cvt_f16x2_e2m1x2_80), "r"(sl01_4));
                            uint32_t _cvt_f16x2_e2m1x2_81;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_81) : "r"((uint32_t)(p_15_1 >> 16)));
                            uint32_t prod_hi_15_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_15_1) : "r"(_cvt_f16x2_e2m1x2_81), "r"(sl23_4));
                            uint16_t lo_15_1;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_15_1) : "r"(prod_lo_15_1));
                            uint16_t hi_15_1;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_15_1) : "r"(prod_hi_15_1));
                            uint32_t _mov_b32_pack_u16x2_82;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_82) : "h"(lo_15_1), "h"(hi_15_1));
                            ld_lo[10] = _mov_b32_pack_u16x2_82;
                            unsigned int x_16 = c_0[1] & 252645135;
                            unsigned int p_16 = x_16 | x_16 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_83;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_83) : "r"((uint32_t)(p_16)));
                            uint32_t prod_lo_16;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_16) : "r"(_cvt_f16x2_e2m1x2_83), "r"(sh01_4));
                            uint32_t _cvt_f16x2_e2m1x2_84;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_84) : "r"((uint32_t)(p_16 >> 16)));
                            uint32_t prod_hi_16;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_16) : "r"(_cvt_f16x2_e2m1x2_84), "r"(sh23_4));
                            uint16_t lo_16;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_16) : "r"(prod_lo_16));
                            uint16_t hi_16;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_16) : "r"(prod_hi_16));
                            uint32_t _mov_b32_pack_u16x2_85;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_85) : "h"(lo_16), "h"(hi_16));
                            ld_hi[8] = _mov_b32_pack_u16x2_85;
                            unsigned int x_17 = c_0[1] >> 4 & 252645135;
                            unsigned int p_17 = x_17 | x_17 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_86;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_86) : "r"((uint32_t)(p_17)));
                            uint32_t prod_lo_17;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_17) : "r"(_cvt_f16x2_e2m1x2_86), "r"(sh01_4));
                            uint32_t _cvt_f16x2_e2m1x2_87;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_87) : "r"((uint32_t)(p_17 >> 16)));
                            uint32_t prod_hi_17;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_17) : "r"(_cvt_f16x2_e2m1x2_87), "r"(sh23_4));
                            uint16_t lo_17;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_17) : "r"(prod_lo_17));
                            uint16_t hi_17;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_17) : "r"(prod_hi_17));
                            uint32_t _mov_b32_pack_u16x2_88;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_88) : "h"(lo_17), "h"(hi_17));
                            ld_hi[10] = _mov_b32_pack_u16x2_88;
                            unsigned int sl_13 = __shfl_sync(0xFFFFFFFF, ld_s[3], src_lo_0);
                            unsigned int sh_13 = __shfl_sync(0xFFFFFFFF, ld_s[3], src_hi_0);
                            uint32_t _cvt_f16x2_e4m3x2_89;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_89) : "h"((uint16_t)(sl_13)));
                            uint32_t sl01_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_5) : "r"(_cvt_f16x2_e4m3x2_89), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_90;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_90) : "h"((uint16_t)(sl_13 >> 16)));
                            uint32_t sl23_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_5) : "r"(_cvt_f16x2_e4m3x2_90), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_91;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_91) : "h"((uint16_t)(sh_13)));
                            uint32_t sh01_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_5) : "r"(_cvt_f16x2_e4m3x2_91), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_92;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_92) : "h"((uint16_t)(sh_13 >> 16)));
                            uint32_t sh23_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_5) : "r"(_cvt_f16x2_e4m3x2_92), "r"(kexp_f16x2));
                            unsigned int x_18 = c_0[2] & 252645135;
                            unsigned int p_18 = x_18 | x_18 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_93;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_93) : "r"((uint32_t)(p_18)));
                            uint32_t prod_lo_18;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_18) : "r"(_cvt_f16x2_e2m1x2_93), "r"(sl01_5));
                            uint32_t _cvt_f16x2_e2m1x2_94;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_94) : "r"((uint32_t)(p_18 >> 16)));
                            uint32_t prod_hi_18;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_18) : "r"(_cvt_f16x2_e2m1x2_94), "r"(sl23_5));
                            uint16_t lo_18;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_18) : "r"(prod_lo_18));
                            uint16_t hi_18;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_18) : "r"(prod_hi_18));
                            uint32_t _mov_b32_pack_u16x2_95;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_95) : "h"(lo_18), "h"(hi_18));
                            ld_lo[9] = _mov_b32_pack_u16x2_95;
                            unsigned int x_19 = c_0[2] >> 4 & 252645135;
                            unsigned int p_19 = x_19 | x_19 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_96;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_96) : "r"((uint32_t)(p_19)));
                            uint32_t prod_lo_19;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_19) : "r"(_cvt_f16x2_e2m1x2_96), "r"(sl01_5));
                            uint32_t _cvt_f16x2_e2m1x2_97;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_97) : "r"((uint32_t)(p_19 >> 16)));
                            uint32_t prod_hi_19;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_19) : "r"(_cvt_f16x2_e2m1x2_97), "r"(sl23_5));
                            uint16_t lo_19;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_19) : "r"(prod_lo_19));
                            uint16_t hi_19;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_19) : "r"(prod_hi_19));
                            uint32_t _mov_b32_pack_u16x2_98;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_98) : "h"(lo_19), "h"(hi_19));
                            ld_lo[11] = _mov_b32_pack_u16x2_98;
                            unsigned int x_20 = c_0[3] & 252645135;
                            unsigned int p_20 = x_20 | x_20 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_99;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_99) : "r"((uint32_t)(p_20)));
                            uint32_t prod_lo_20;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_20) : "r"(_cvt_f16x2_e2m1x2_99), "r"(sh01_5));
                            uint32_t _cvt_f16x2_e2m1x2_100;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_100) : "r"((uint32_t)(p_20 >> 16)));
                            uint32_t prod_hi_20;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_20) : "r"(_cvt_f16x2_e2m1x2_100), "r"(sh23_5));
                            uint16_t lo_20;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_20) : "r"(prod_lo_20));
                            uint16_t hi_20;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_20) : "r"(prod_hi_20));
                            uint32_t _mov_b32_pack_u16x2_101;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_101) : "h"(lo_20), "h"(hi_20));
                            ld_hi[9] = _mov_b32_pack_u16x2_101;
                            unsigned int x_21 = c_0[3] >> 4 & 252645135;
                            unsigned int p_21 = x_21 | x_21 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_102;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_102) : "r"((uint32_t)(p_21)));
                            uint32_t prod_lo_21;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_21) : "r"(_cvt_f16x2_e2m1x2_102), "r"(sh01_5));
                            uint32_t _cvt_f16x2_e2m1x2_103;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_103) : "r"((uint32_t)(p_21 >> 16)));
                            uint32_t prod_hi_21;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_21) : "r"(_cvt_f16x2_e2m1x2_103), "r"(sh23_5));
                            uint16_t lo_21;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_21) : "r"(prod_lo_21));
                            uint16_t hi_21;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_21) : "r"(prod_hi_21));
                            uint32_t _mov_b32_pack_u16x2_104;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_104) : "h"(lo_21), "h"(hi_21));
                            ld_hi[11] = _mov_b32_pack_u16x2_104;
                            unsigned int sl_14 = __shfl_sync(0xFFFFFFFF, ld_s[5], src_lo_0);
                            unsigned int sh_14 = __shfl_sync(0xFFFFFFFF, ld_s[5], src_hi_0);
                            uint32_t _cvt_f16x2_e4m3x2_105;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_105) : "h"((uint16_t)(sl_14)));
                            uint32_t sl01_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_6) : "r"(_cvt_f16x2_e4m3x2_105), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_106;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_106) : "h"((uint16_t)(sl_14 >> 16)));
                            uint32_t sl23_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_6) : "r"(_cvt_f16x2_e4m3x2_106), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_107;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_107) : "h"((uint16_t)(sh_14)));
                            uint32_t sh01_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_6) : "r"(_cvt_f16x2_e4m3x2_107), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_108;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_108) : "h"((uint16_t)(sh_14 >> 16)));
                            uint32_t sh23_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_6) : "r"(_cvt_f16x2_e4m3x2_108), "r"(kexp_f16x2));
                            unsigned int x_22 = c_0[4] & 252645135;
                            unsigned int p_22 = x_22 | x_22 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_109;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_109) : "r"((uint32_t)(p_22)));
                            uint32_t prod_lo_22;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_22) : "r"(_cvt_f16x2_e2m1x2_109), "r"(sl01_6));
                            uint32_t _cvt_f16x2_e2m1x2_110;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_110) : "r"((uint32_t)(p_22 >> 16)));
                            uint32_t prod_hi_22;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_22) : "r"(_cvt_f16x2_e2m1x2_110), "r"(sl23_6));
                            uint16_t lo_22;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_22) : "r"(prod_lo_22));
                            uint16_t hi_22;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_22) : "r"(prod_hi_22));
                            uint32_t _mov_b32_pack_u16x2_111;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_111) : "h"(lo_22), "h"(hi_22));
                            ld_lo[12] = _mov_b32_pack_u16x2_111;
                            unsigned int x_23 = c_0[4] >> 4 & 252645135;
                            unsigned int p_23 = x_23 | x_23 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_112;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_112) : "r"((uint32_t)(p_23)));
                            uint32_t prod_lo_23;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_23) : "r"(_cvt_f16x2_e2m1x2_112), "r"(sl01_6));
                            uint32_t _cvt_f16x2_e2m1x2_113;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_113) : "r"((uint32_t)(p_23 >> 16)));
                            uint32_t prod_hi_23;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_23) : "r"(_cvt_f16x2_e2m1x2_113), "r"(sl23_6));
                            uint16_t lo_23;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_23) : "r"(prod_lo_23));
                            uint16_t hi_23;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_23) : "r"(prod_hi_23));
                            uint32_t _mov_b32_pack_u16x2_114;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_114) : "h"(lo_23), "h"(hi_23));
                            ld_lo[14] = _mov_b32_pack_u16x2_114;
                            unsigned int x_24 = c_0[5] & 252645135;
                            unsigned int p_24 = x_24 | x_24 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_115;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_115) : "r"((uint32_t)(p_24)));
                            uint32_t prod_lo_24;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_24) : "r"(_cvt_f16x2_e2m1x2_115), "r"(sh01_6));
                            uint32_t _cvt_f16x2_e2m1x2_116;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_116) : "r"((uint32_t)(p_24 >> 16)));
                            uint32_t prod_hi_24;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_24) : "r"(_cvt_f16x2_e2m1x2_116), "r"(sh23_6));
                            uint16_t lo_24;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_24) : "r"(prod_lo_24));
                            uint16_t hi_24;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_24) : "r"(prod_hi_24));
                            uint32_t _mov_b32_pack_u16x2_117;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_117) : "h"(lo_24), "h"(hi_24));
                            ld_hi[12] = _mov_b32_pack_u16x2_117;
                            unsigned int x_25 = c_0[5] >> 4 & 252645135;
                            unsigned int p_25 = x_25 | x_25 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_118;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_118) : "r"((uint32_t)(p_25)));
                            uint32_t prod_lo_25;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_25) : "r"(_cvt_f16x2_e2m1x2_118), "r"(sh01_6));
                            uint32_t _cvt_f16x2_e2m1x2_119;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_119) : "r"((uint32_t)(p_25 >> 16)));
                            uint32_t prod_hi_25;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_25) : "r"(_cvt_f16x2_e2m1x2_119), "r"(sh23_6));
                            uint16_t lo_25;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_25) : "r"(prod_lo_25));
                            uint16_t hi_25;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_25) : "r"(prod_hi_25));
                            uint32_t _mov_b32_pack_u16x2_120;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_120) : "h"(lo_25), "h"(hi_25));
                            ld_hi[14] = _mov_b32_pack_u16x2_120;
                            unsigned int sl_15 = __shfl_sync(0xFFFFFFFF, ld_s[7], src_lo_0);
                            unsigned int sh_15 = __shfl_sync(0xFFFFFFFF, ld_s[7], src_hi_0);
                            uint32_t _cvt_f16x2_e4m3x2_121;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_121) : "h"((uint16_t)(sl_15)));
                            uint32_t sl01_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_7) : "r"(_cvt_f16x2_e4m3x2_121), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_122;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_122) : "h"((uint16_t)(sl_15 >> 16)));
                            uint32_t sl23_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_7) : "r"(_cvt_f16x2_e4m3x2_122), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_123;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_123) : "h"((uint16_t)(sh_15)));
                            uint32_t sh01_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_7) : "r"(_cvt_f16x2_e4m3x2_123), "r"(kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_124;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_124) : "h"((uint16_t)(sh_15 >> 16)));
                            uint32_t sh23_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_7) : "r"(_cvt_f16x2_e4m3x2_124), "r"(kexp_f16x2));
                            unsigned int x_26 = c_0[6] & 252645135;
                            unsigned int p_26 = x_26 | x_26 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_125;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_125) : "r"((uint32_t)(p_26)));
                            uint32_t prod_lo_26;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_26) : "r"(_cvt_f16x2_e2m1x2_125), "r"(sl01_7));
                            uint32_t _cvt_f16x2_e2m1x2_126;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_126) : "r"((uint32_t)(p_26 >> 16)));
                            uint32_t prod_hi_26;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_26) : "r"(_cvt_f16x2_e2m1x2_126), "r"(sl23_7));
                            uint16_t lo_26;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_26) : "r"(prod_lo_26));
                            uint16_t hi_26;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_26) : "r"(prod_hi_26));
                            uint32_t _mov_b32_pack_u16x2_127;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_127) : "h"(lo_26), "h"(hi_26));
                            ld_lo[13] = _mov_b32_pack_u16x2_127;
                            unsigned int x_27 = c_0[6] >> 4 & 252645135;
                            unsigned int p_27 = x_27 | x_27 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_128;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_128) : "r"((uint32_t)(p_27)));
                            uint32_t prod_lo_27;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_27) : "r"(_cvt_f16x2_e2m1x2_128), "r"(sl01_7));
                            uint32_t _cvt_f16x2_e2m1x2_129;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_129) : "r"((uint32_t)(p_27 >> 16)));
                            uint32_t prod_hi_27;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_27) : "r"(_cvt_f16x2_e2m1x2_129), "r"(sl23_7));
                            uint16_t lo_27;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_27) : "r"(prod_lo_27));
                            uint16_t hi_27;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_27) : "r"(prod_hi_27));
                            uint32_t _mov_b32_pack_u16x2_130;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_130) : "h"(lo_27), "h"(hi_27));
                            ld_lo[15] = _mov_b32_pack_u16x2_130;
                            unsigned int x_28 = c_0[7] & 252645135;
                            unsigned int p_28 = x_28 | x_28 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_131;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_131) : "r"((uint32_t)(p_28)));
                            uint32_t prod_lo_28;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_28) : "r"(_cvt_f16x2_e2m1x2_131), "r"(sh01_7));
                            uint32_t _cvt_f16x2_e2m1x2_132;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_132) : "r"((uint32_t)(p_28 >> 16)));
                            uint32_t prod_hi_28;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_28) : "r"(_cvt_f16x2_e2m1x2_132), "r"(sh23_7));
                            uint16_t lo_28;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_28) : "r"(prod_lo_28));
                            uint16_t hi_28;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_28) : "r"(prod_hi_28));
                            uint32_t _mov_b32_pack_u16x2_133;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_133) : "h"(lo_28), "h"(hi_28));
                            ld_hi[13] = _mov_b32_pack_u16x2_133;
                            unsigned int x_29 = c_0[7] >> 4 & 252645135;
                            unsigned int p_29 = x_29 | x_29 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_134;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_134) : "r"((uint32_t)(p_29)));
                            uint32_t prod_lo_29;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_29) : "r"(_cvt_f16x2_e2m1x2_134), "r"(sh01_7));
                            uint32_t _cvt_f16x2_e2m1x2_135;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_135) : "r"((uint32_t)(p_29 >> 16)));
                            uint32_t prod_hi_29;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_29) : "r"(_cvt_f16x2_e2m1x2_135), "r"(sh23_7));
                            uint16_t lo_29;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_29) : "r"(prod_lo_29));
                            uint16_t hi_29;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_29) : "r"(prod_hi_29));
                            uint32_t _mov_b32_pack_u16x2_136;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_136) : "h"(lo_29), "h"(hi_29));
                            ld_hi[15] = _mov_b32_pack_u16x2_136;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(ld_vt), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[3])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[4])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[5])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[6])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[7])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[8])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[9])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[10])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[11])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[12])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[13])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[14])), "r"(*reinterpret_cast<const uint32_t*>(&ld_lo[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(ld_vt + (unsigned int)(16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[3])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[4])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[5])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[6])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[7])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[8])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[9])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[10])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[11])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[12])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[13])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[14])), "r"(*reinterpret_cast<const uint32_t*>(&ld_hi[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (ld_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(kstage_free_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 5)) * 8);
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
                int _stage_k_empty_1 = (int)((unsigned int)g_tile_cursor % 5);
                int _parity_k_empty_1 = (int)((unsigned int)g_tile_cursor / 5 % 2) ^ 1;
                mbarrier_wait(k_empty_addr + (_stage_k_empty_1) * 8, _parity_k_empty_1);
                int _stage_kstage_free = (int)((unsigned int)g_tile_cursor % 5);
                int _parity_kstage_free = (int)((unsigned int)g_tile_cursor / 5 % 2) ^ 1;
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
                int _stage_k_empty_2 = (int)((unsigned int)(g_tile_cursor + 1) % 5);
                int _parity_k_empty_2 = (int)((unsigned int)(g_tile_cursor + 1) / 5 % 2) ^ 1;
                mbarrier_wait(k_empty_addr + (_stage_k_empty_2) * 8, _parity_k_empty_2);
                int _stage_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + 1) % 5);
                int _parity_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + 1) / 5 % 2) ^ 1;
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
                        int g_stage = (g_tile_cursor + g_tile) % 5;
                        int g_slot_tile = g_tile & 3;
                        if (g_slot_tile == 0) {
                            mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_1);
                        }
                        int _stage_k_empty_3 = (int)((unsigned int)(g_tile_cursor + g_tile) % 5);
                        int _parity_k_empty_3 = (int)((unsigned int)(g_tile_cursor + g_tile) / 5 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty_3) * 8, _parity_k_empty_3);
                        int _stage_kstage_free_2 = (int)((unsigned int)(g_tile_cursor + g_tile) % 5);
                        int _parity_kstage_free_2 = (int)((unsigned int)(g_tile_cursor + g_tile) / 5 % 2) ^ 1;
                        mbarrier_wait(kstage_free_addr + (_stage_kstage_free_2) * 8, _parity_kstage_free_2);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (g_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 5)) * 8, 26624);
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
                                    tma_gather4_gmem2smem(g_dst, (&tmap_swa_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 5)) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_compressed_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 5)) * 8);
                                }
                            } else {
                                int g_fdst = g_stage_base + 24576 + g_group * 128;
                                if (g_use_main != 0) {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_swa_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 5)) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_compressed_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 5)) * 8);
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
                        int xk_stage = (vx_tile_cursor + vx_tile) % 5;
                        int _stage_k_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) % 5);
                        int _parity_k_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) / 5 % 2);
                        mbarrier_wait(k_full_addr + (_stage_k_full_1) * 8, _parity_k_full_1);
                        int _stage_kexp_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) % 2);
                        int _parity_kexp_full_1 = (int)((unsigned int)(vx_tile_cursor + vx_tile) / 2 % 2);
                        mbarrier_wait(kexp_full_addr + (_stage_kexp_full_1) * 8, _parity_kexp_full_1);
                        unsigned int x_kexp = smem_kexp[x_stage * 4];
                        unsigned int x_kexp_f16x2 = (15 - x_kexp << 10) * 65537;
                        const int vx_warp = warp - 20;
                        unsigned int x_kexp_bf16x2 = (127 - x_kexp << 7) * 65537;
                        unsigned int x_stage_base = smem_k_a_addr + (unsigned int)(xk_stage * 26624);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int vx_s[8];
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(vx_s[0]), "=r"(vx_s[1]), "=r"(vx_s[2]), "=r"(vx_s[3])
                            : "r"((unsigned int)(smem_k_sf_addr + (unsigned int)(xk_stage * 26624) + 16) + (unsigned int)(lane * 32))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(vx_s[4]), "=r"(vx_s[5]), "=r"(vx_s[6]), "=r"(vx_s[7])
                            : "r"((unsigned int)(smem_k_sf_addr + (unsigned int)(xk_stage * 26624) + 16) + (unsigned int)((32 + lane) * 32))
                            : "memory");
                        unsigned int vx_vt = taddr + (unsigned int)(vx_warp * 32 << 16) + (unsigned int)(128 + 64 * x_stage);
                        unsigned int vx_lo[16];
                        unsigned int vx_hi[16];
                        unsigned int c_1[8];
                        unsigned int swz_1 = (unsigned int)((lane & 7) << 4);
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c_1[0]), "=r"(c_1[1]), "=r"(c_1[2]), "=r"(c_1[3])
                            : "r"((unsigned int)(x_stage_base + 8192) + (unsigned int)(lane * 128) + ((unsigned int)(vx_warp * 16) ^ swz_1))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c_1[4]), "=r"(c_1[5]), "=r"(c_1[6]), "=r"(c_1[7])
                            : "r"((unsigned int)(x_stage_base + 8192) + (unsigned int)((32 + lane) * 128) + ((unsigned int)(vx_warp * 16) ^ swz_1))
                            : "memory");
                        int src_lo_1 = 8 * (vx_warp & 3) + (lane & 3);
                        int src_hi_1 = src_lo_1 + 4;
                        if (x_kexp == 0) {
                            unsigned int sl_16 = __shfl_sync(0xFFFFFFFF, vx_s[0], src_lo_1);
                            unsigned int sh_16 = __shfl_sync(0xFFFFFFFF, vx_s[0], src_hi_1);
                            vx_lo[0] = qmul4_direct(c_1[0], sl_16);
                            vx_lo[2] = qmul4_direct(c_1[0] >> 4, sl_16);
                            vx_hi[0] = qmul4_direct(c_1[1], sh_16);
                            vx_hi[2] = qmul4_direct(c_1[1] >> 4, sh_16);
                            unsigned int sl_17 = __shfl_sync(0xFFFFFFFF, vx_s[2], src_lo_1);
                            unsigned int sh_17 = __shfl_sync(0xFFFFFFFF, vx_s[2], src_hi_1);
                            vx_lo[1] = qmul4_direct(c_1[2], sl_17);
                            vx_lo[3] = qmul4_direct(c_1[2] >> 4, sl_17);
                            vx_hi[1] = qmul4_direct(c_1[3], sh_17);
                            vx_hi[3] = qmul4_direct(c_1[3] >> 4, sh_17);
                            unsigned int sl_18 = __shfl_sync(0xFFFFFFFF, vx_s[4], src_lo_1);
                            unsigned int sh_18 = __shfl_sync(0xFFFFFFFF, vx_s[4], src_hi_1);
                            vx_lo[4] = qmul4_direct(c_1[4], sl_18);
                            vx_lo[6] = qmul4_direct(c_1[4] >> 4, sl_18);
                            vx_hi[4] = qmul4_direct(c_1[5], sh_18);
                            vx_hi[6] = qmul4_direct(c_1[5] >> 4, sh_18);
                            unsigned int sl_19 = __shfl_sync(0xFFFFFFFF, vx_s[6], src_lo_1);
                            unsigned int sh_19 = __shfl_sync(0xFFFFFFFF, vx_s[6], src_hi_1);
                            vx_lo[5] = qmul4_direct(c_1[6], sl_19);
                            vx_lo[7] = qmul4_direct(c_1[6] >> 4, sl_19);
                            vx_hi[5] = qmul4_direct(c_1[7], sh_19);
                            vx_hi[7] = qmul4_direct(c_1[7] >> 4, sh_19);
                        } else {
                            unsigned int sl_20 = __shfl_sync(0xFFFFFFFF, vx_s[0], src_lo_1);
                            unsigned int sh_20 = __shfl_sync(0xFFFFFFFF, vx_s[0], src_hi_1);
                            uint32_t _cvt_f16x2_e4m3x2_6;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_6) : "h"((uint16_t)(sl_20)));
                            uint32_t sl01_8;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_8) : "r"(_cvt_f16x2_e4m3x2_6), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_7;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_7) : "h"((uint16_t)(sl_20 >> 16)));
                            uint32_t sl23_8;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_8) : "r"(_cvt_f16x2_e4m3x2_7), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_8;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_8) : "h"((uint16_t)(sh_20)));
                            uint32_t sh01_8;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_8) : "r"(_cvt_f16x2_e4m3x2_8), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_9_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_9_1) : "h"((uint16_t)(sh_20 >> 16)));
                            uint32_t sh23_8;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_8) : "r"(_cvt_f16x2_e4m3x2_9_1), "r"(x_kexp_f16x2));
                            unsigned int x_30 = c_1[0] & 252645135;
                            unsigned int p_30 = x_30 | x_30 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_10;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_10) : "r"((uint32_t)(p_30)));
                            uint32_t prod_lo_30;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_30) : "r"(_cvt_f16x2_e2m1x2_10), "r"(sl01_8));
                            uint32_t _cvt_f16x2_e2m1x2_11;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_11) : "r"((uint32_t)(p_30 >> 16)));
                            uint32_t prod_hi_30;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_30) : "r"(_cvt_f16x2_e2m1x2_11), "r"(sl23_8));
                            uint16_t lo_30;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_30) : "r"(prod_lo_30));
                            uint16_t hi_30;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_30) : "r"(prod_hi_30));
                            uint32_t _mov_b32_pack_u16x2_12;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_12) : "h"(lo_30), "h"(hi_30));
                            vx_lo[0] = _mov_b32_pack_u16x2_12;
                            unsigned int x_30_1 = c_1[0] >> 4 & 252645135;
                            unsigned int p_30_1 = x_30_1 | x_30_1 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_13_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_13_1) : "r"((uint32_t)(p_30_1)));
                            uint32_t prod_lo_30_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_30_1) : "r"(_cvt_f16x2_e2m1x2_13_1), "r"(sl01_8));
                            uint32_t _cvt_f16x2_e2m1x2_14_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_14_1) : "r"((uint32_t)(p_30_1 >> 16)));
                            uint32_t prod_hi_30_1;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_30_1) : "r"(_cvt_f16x2_e2m1x2_14_1), "r"(sl23_8));
                            uint16_t lo_30_1;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_30_1) : "r"(prod_lo_30_1));
                            uint16_t hi_30_1;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_30_1) : "r"(prod_hi_30_1));
                            uint32_t _mov_b32_pack_u16x2_15_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_15_1) : "h"(lo_30_1), "h"(hi_30_1));
                            vx_lo[2] = _mov_b32_pack_u16x2_15_1;
                            unsigned int x_31 = c_1[1] & 252645135;
                            unsigned int p_31 = x_31 | x_31 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_16_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_16_1) : "r"((uint32_t)(p_31)));
                            uint32_t prod_lo_31;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_31) : "r"(_cvt_f16x2_e2m1x2_16_1), "r"(sh01_8));
                            uint32_t _cvt_f16x2_e2m1x2_17_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_17_1) : "r"((uint32_t)(p_31 >> 16)));
                            uint32_t prod_hi_31;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_31) : "r"(_cvt_f16x2_e2m1x2_17_1), "r"(sh23_8));
                            uint16_t lo_31;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_31) : "r"(prod_lo_31));
                            uint16_t hi_31;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_31) : "r"(prod_hi_31));
                            uint32_t _mov_b32_pack_u16x2_18_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_18_1) : "h"(lo_31), "h"(hi_31));
                            vx_hi[0] = _mov_b32_pack_u16x2_18_1;
                            unsigned int x_32 = c_1[1] >> 4 & 252645135;
                            unsigned int p_32 = x_32 | x_32 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_19_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_19_1) : "r"((uint32_t)(p_32)));
                            uint32_t prod_lo_32;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_32) : "r"(_cvt_f16x2_e2m1x2_19_1), "r"(sh01_8));
                            uint32_t _cvt_f16x2_e2m1x2_20_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_20_1) : "r"((uint32_t)(p_32 >> 16)));
                            uint32_t prod_hi_32;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_32) : "r"(_cvt_f16x2_e2m1x2_20_1), "r"(sh23_8));
                            uint16_t lo_32;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_32) : "r"(prod_lo_32));
                            uint16_t hi_32;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_32) : "r"(prod_hi_32));
                            uint32_t _mov_b32_pack_u16x2_21_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_21_1) : "h"(lo_32), "h"(hi_32));
                            vx_hi[2] = _mov_b32_pack_u16x2_21_1;
                            unsigned int sl_21 = __shfl_sync(0xFFFFFFFF, vx_s[2], src_lo_1);
                            unsigned int sh_21 = __shfl_sync(0xFFFFFFFF, vx_s[2], src_hi_1);
                            uint32_t _cvt_f16x2_e4m3x2_22;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_22) : "h"((uint16_t)(sl_21)));
                            uint32_t sl01_9;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_9) : "r"(_cvt_f16x2_e4m3x2_22), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_23;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_23) : "h"((uint16_t)(sl_21 >> 16)));
                            uint32_t sl23_9;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_9) : "r"(_cvt_f16x2_e4m3x2_23), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_24;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_24) : "h"((uint16_t)(sh_21)));
                            uint32_t sh01_9;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_9) : "r"(_cvt_f16x2_e4m3x2_24), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_25_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_25_1) : "h"((uint16_t)(sh_21 >> 16)));
                            uint32_t sh23_9;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_9) : "r"(_cvt_f16x2_e4m3x2_25_1), "r"(x_kexp_f16x2));
                            unsigned int x_33 = c_1[2] & 252645135;
                            unsigned int p_33 = x_33 | x_33 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_26;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_26) : "r"((uint32_t)(p_33)));
                            uint32_t prod_lo_33;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_33) : "r"(_cvt_f16x2_e2m1x2_26), "r"(sl01_9));
                            uint32_t _cvt_f16x2_e2m1x2_27;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_27) : "r"((uint32_t)(p_33 >> 16)));
                            uint32_t prod_hi_33;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_33) : "r"(_cvt_f16x2_e2m1x2_27), "r"(sl23_9));
                            uint16_t lo_33;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_33) : "r"(prod_lo_33));
                            uint16_t hi_33;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_33) : "r"(prod_hi_33));
                            uint32_t _mov_b32_pack_u16x2_28;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_28) : "h"(lo_33), "h"(hi_33));
                            vx_lo[1] = _mov_b32_pack_u16x2_28;
                            unsigned int x_34 = c_1[2] >> 4 & 252645135;
                            unsigned int p_34 = x_34 | x_34 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_29_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_29_1) : "r"((uint32_t)(p_34)));
                            uint32_t prod_lo_34;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_34) : "r"(_cvt_f16x2_e2m1x2_29_1), "r"(sl01_9));
                            uint32_t _cvt_f16x2_e2m1x2_30_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_30_1) : "r"((uint32_t)(p_34 >> 16)));
                            uint32_t prod_hi_34;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_34) : "r"(_cvt_f16x2_e2m1x2_30_1), "r"(sl23_9));
                            uint16_t lo_34;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_34) : "r"(prod_lo_34));
                            uint16_t hi_34;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_34) : "r"(prod_hi_34));
                            uint32_t _mov_b32_pack_u16x2_31_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_31_1) : "h"(lo_34), "h"(hi_34));
                            vx_lo[3] = _mov_b32_pack_u16x2_31_1;
                            unsigned int x_35 = c_1[3] & 252645135;
                            unsigned int p_35 = x_35 | x_35 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_32_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_32_1) : "r"((uint32_t)(p_35)));
                            uint32_t prod_lo_35;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_35) : "r"(_cvt_f16x2_e2m1x2_32_1), "r"(sh01_9));
                            uint32_t _cvt_f16x2_e2m1x2_33_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_33_1) : "r"((uint32_t)(p_35 >> 16)));
                            uint32_t prod_hi_35;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_35) : "r"(_cvt_f16x2_e2m1x2_33_1), "r"(sh23_9));
                            uint16_t lo_35;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_35) : "r"(prod_lo_35));
                            uint16_t hi_35;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_35) : "r"(prod_hi_35));
                            uint32_t _mov_b32_pack_u16x2_34_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_34_1) : "h"(lo_35), "h"(hi_35));
                            vx_hi[1] = _mov_b32_pack_u16x2_34_1;
                            unsigned int x_36 = c_1[3] >> 4 & 252645135;
                            unsigned int p_36 = x_36 | x_36 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_35_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_35_1) : "r"((uint32_t)(p_36)));
                            uint32_t prod_lo_36;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_36) : "r"(_cvt_f16x2_e2m1x2_35_1), "r"(sh01_9));
                            uint32_t _cvt_f16x2_e2m1x2_36_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_36_1) : "r"((uint32_t)(p_36 >> 16)));
                            uint32_t prod_hi_36;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_36) : "r"(_cvt_f16x2_e2m1x2_36_1), "r"(sh23_9));
                            uint16_t lo_36;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_36) : "r"(prod_lo_36));
                            uint16_t hi_36;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_36) : "r"(prod_hi_36));
                            uint32_t _mov_b32_pack_u16x2_37_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_37_1) : "h"(lo_36), "h"(hi_36));
                            vx_hi[3] = _mov_b32_pack_u16x2_37_1;
                            unsigned int sl_22 = __shfl_sync(0xFFFFFFFF, vx_s[4], src_lo_1);
                            unsigned int sh_22 = __shfl_sync(0xFFFFFFFF, vx_s[4], src_hi_1);
                            uint32_t _cvt_f16x2_e4m3x2_38;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_38) : "h"((uint16_t)(sl_22)));
                            uint32_t sl01_10;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_10) : "r"(_cvt_f16x2_e4m3x2_38), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_39;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_39) : "h"((uint16_t)(sl_22 >> 16)));
                            uint32_t sl23_10;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_10) : "r"(_cvt_f16x2_e4m3x2_39), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_40;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_40) : "h"((uint16_t)(sh_22)));
                            uint32_t sh01_10;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_10) : "r"(_cvt_f16x2_e4m3x2_40), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_41_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_41_1) : "h"((uint16_t)(sh_22 >> 16)));
                            uint32_t sh23_10;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_10) : "r"(_cvt_f16x2_e4m3x2_41_1), "r"(x_kexp_f16x2));
                            unsigned int x_37 = c_1[4] & 252645135;
                            unsigned int p_37 = x_37 | x_37 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_42;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_42) : "r"((uint32_t)(p_37)));
                            uint32_t prod_lo_37;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_37) : "r"(_cvt_f16x2_e2m1x2_42), "r"(sl01_10));
                            uint32_t _cvt_f16x2_e2m1x2_43;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_43) : "r"((uint32_t)(p_37 >> 16)));
                            uint32_t prod_hi_37;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_37) : "r"(_cvt_f16x2_e2m1x2_43), "r"(sl23_10));
                            uint16_t lo_37;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_37) : "r"(prod_lo_37));
                            uint16_t hi_37;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_37) : "r"(prod_hi_37));
                            uint32_t _mov_b32_pack_u16x2_44;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_44) : "h"(lo_37), "h"(hi_37));
                            vx_lo[4] = _mov_b32_pack_u16x2_44;
                            unsigned int x_38 = c_1[4] >> 4 & 252645135;
                            unsigned int p_38 = x_38 | x_38 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_45_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_45_1) : "r"((uint32_t)(p_38)));
                            uint32_t prod_lo_38;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_38) : "r"(_cvt_f16x2_e2m1x2_45_1), "r"(sl01_10));
                            uint32_t _cvt_f16x2_e2m1x2_46_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_46_1) : "r"((uint32_t)(p_38 >> 16)));
                            uint32_t prod_hi_38;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_38) : "r"(_cvt_f16x2_e2m1x2_46_1), "r"(sl23_10));
                            uint16_t lo_38;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_38) : "r"(prod_lo_38));
                            uint16_t hi_38;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_38) : "r"(prod_hi_38));
                            uint32_t _mov_b32_pack_u16x2_47_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_47_1) : "h"(lo_38), "h"(hi_38));
                            vx_lo[6] = _mov_b32_pack_u16x2_47_1;
                            unsigned int x_39 = c_1[5] & 252645135;
                            unsigned int p_39 = x_39 | x_39 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_48_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_48_1) : "r"((uint32_t)(p_39)));
                            uint32_t prod_lo_39;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_39) : "r"(_cvt_f16x2_e2m1x2_48_1), "r"(sh01_10));
                            uint32_t _cvt_f16x2_e2m1x2_49_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_49_1) : "r"((uint32_t)(p_39 >> 16)));
                            uint32_t prod_hi_39;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_39) : "r"(_cvt_f16x2_e2m1x2_49_1), "r"(sh23_10));
                            uint16_t lo_39;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_39) : "r"(prod_lo_39));
                            uint16_t hi_39;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_39) : "r"(prod_hi_39));
                            uint32_t _mov_b32_pack_u16x2_50_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_50_1) : "h"(lo_39), "h"(hi_39));
                            vx_hi[4] = _mov_b32_pack_u16x2_50_1;
                            unsigned int x_40 = c_1[5] >> 4 & 252645135;
                            unsigned int p_40 = x_40 | x_40 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_51_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_51_1) : "r"((uint32_t)(p_40)));
                            uint32_t prod_lo_40;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_40) : "r"(_cvt_f16x2_e2m1x2_51_1), "r"(sh01_10));
                            uint32_t _cvt_f16x2_e2m1x2_52_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_52_1) : "r"((uint32_t)(p_40 >> 16)));
                            uint32_t prod_hi_40;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_40) : "r"(_cvt_f16x2_e2m1x2_52_1), "r"(sh23_10));
                            uint16_t lo_40;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_40) : "r"(prod_lo_40));
                            uint16_t hi_40;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_40) : "r"(prod_hi_40));
                            uint32_t _mov_b32_pack_u16x2_53_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_53_1) : "h"(lo_40), "h"(hi_40));
                            vx_hi[6] = _mov_b32_pack_u16x2_53_1;
                            unsigned int sl_23 = __shfl_sync(0xFFFFFFFF, vx_s[6], src_lo_1);
                            unsigned int sh_23 = __shfl_sync(0xFFFFFFFF, vx_s[6], src_hi_1);
                            uint32_t _cvt_f16x2_e4m3x2_54;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_54) : "h"((uint16_t)(sl_23)));
                            uint32_t sl01_11;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_11) : "r"(_cvt_f16x2_e4m3x2_54), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_55;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_55) : "h"((uint16_t)(sl_23 >> 16)));
                            uint32_t sl23_11;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_11) : "r"(_cvt_f16x2_e4m3x2_55), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_56;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_56) : "h"((uint16_t)(sh_23)));
                            uint32_t sh01_11;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_11) : "r"(_cvt_f16x2_e4m3x2_56), "r"(x_kexp_f16x2));
                            uint32_t _cvt_f16x2_e4m3x2_57_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_57_1) : "h"((uint16_t)(sh_23 >> 16)));
                            uint32_t sh23_11;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_11) : "r"(_cvt_f16x2_e4m3x2_57_1), "r"(x_kexp_f16x2));
                            unsigned int x_41 = c_1[6] & 252645135;
                            unsigned int p_41 = x_41 | x_41 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_58;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_58) : "r"((uint32_t)(p_41)));
                            uint32_t prod_lo_41;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_41) : "r"(_cvt_f16x2_e2m1x2_58), "r"(sl01_11));
                            uint32_t _cvt_f16x2_e2m1x2_59;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_59) : "r"((uint32_t)(p_41 >> 16)));
                            uint32_t prod_hi_41;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_41) : "r"(_cvt_f16x2_e2m1x2_59), "r"(sl23_11));
                            uint16_t lo_41;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_41) : "r"(prod_lo_41));
                            uint16_t hi_41;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_41) : "r"(prod_hi_41));
                            uint32_t _mov_b32_pack_u16x2_60;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_60) : "h"(lo_41), "h"(hi_41));
                            vx_lo[5] = _mov_b32_pack_u16x2_60;
                            unsigned int x_42 = c_1[6] >> 4 & 252645135;
                            unsigned int p_42 = x_42 | x_42 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_61_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_61_1) : "r"((uint32_t)(p_42)));
                            uint32_t prod_lo_42;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_42) : "r"(_cvt_f16x2_e2m1x2_61_1), "r"(sl01_11));
                            uint32_t _cvt_f16x2_e2m1x2_62_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_62_1) : "r"((uint32_t)(p_42 >> 16)));
                            uint32_t prod_hi_42;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_42) : "r"(_cvt_f16x2_e2m1x2_62_1), "r"(sl23_11));
                            uint16_t lo_42;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_42) : "r"(prod_lo_42));
                            uint16_t hi_42;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_42) : "r"(prod_hi_42));
                            uint32_t _mov_b32_pack_u16x2_63_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_63_1) : "h"(lo_42), "h"(hi_42));
                            vx_lo[7] = _mov_b32_pack_u16x2_63_1;
                            unsigned int x_43 = c_1[7] & 252645135;
                            unsigned int p_43 = x_43 | x_43 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_64_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_64_1) : "r"((uint32_t)(p_43)));
                            uint32_t prod_lo_43;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_43) : "r"(_cvt_f16x2_e2m1x2_64_1), "r"(sh01_11));
                            uint32_t _cvt_f16x2_e2m1x2_65_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_65_1) : "r"((uint32_t)(p_43 >> 16)));
                            uint32_t prod_hi_43;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_43) : "r"(_cvt_f16x2_e2m1x2_65_1), "r"(sh23_11));
                            uint16_t lo_43;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_43) : "r"(prod_lo_43));
                            uint16_t hi_43;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_43) : "r"(prod_hi_43));
                            uint32_t _mov_b32_pack_u16x2_66_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_66_1) : "h"(lo_43), "h"(hi_43));
                            vx_hi[5] = _mov_b32_pack_u16x2_66_1;
                            unsigned int x_44 = c_1[7] >> 4 & 252645135;
                            unsigned int p_44 = x_44 | x_44 >> 4;
                            uint32_t _cvt_f16x2_e2m1x2_67_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_67_1) : "r"((uint32_t)(p_44)));
                            uint32_t prod_lo_44;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_44) : "r"(_cvt_f16x2_e2m1x2_67_1), "r"(sh01_11));
                            uint32_t _cvt_f16x2_e2m1x2_68_1;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_68_1) : "r"((uint32_t)(p_44 >> 16)));
                            uint32_t prod_hi_44;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_44) : "r"(_cvt_f16x2_e2m1x2_68_1), "r"(sh23_11));
                            uint16_t lo_44;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_44) : "r"(prod_lo_44));
                            uint16_t hi_44;
                            asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_44) : "r"(prod_hi_44));
                            uint32_t _mov_b32_pack_u16x2_69_1;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_69_1) : "h"(lo_44), "h"(hi_44));
                            vx_hi[7] = _mov_b32_pack_u16x2_69_1;
                        }
                        if (vx_warp < 2) {
                            unsigned int c_1_1[8];
                            unsigned int swz_1_1 = (unsigned int)((lane & 7) << 4);
                            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(c_1_1[0]), "=r"(c_1_1[1]), "=r"(c_1_1[2]), "=r"(c_1_1[3])
                                : "r"((unsigned int)(x_stage_base + 8192) + (unsigned int)(lane * 128) + ((unsigned int)((vx_warp + 4) * 16) ^ swz_1_1))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(c_1_1[4]), "=r"(c_1_1[5]), "=r"(c_1_1[6]), "=r"(c_1_1[7])
                                : "r"((unsigned int)(x_stage_base + 8192) + (unsigned int)((32 + lane) * 128) + ((unsigned int)((vx_warp + 4) * 16) ^ swz_1_1))
                                : "memory");
                            int src_lo_1_1 = 8 * (vx_warp + 4 & 3) + (lane & 3);
                            int src_hi_1_1 = src_lo_1_1 + 4;
                            if (x_kexp == 0) {
                                unsigned int sl_24 = __shfl_sync(0xFFFFFFFF, vx_s[1], src_lo_1_1);
                                unsigned int sh_24 = __shfl_sync(0xFFFFFFFF, vx_s[1], src_hi_1_1);
                                vx_lo[8] = qmul4_direct(c_1_1[0], sl_24);
                                vx_lo[10] = qmul4_direct(c_1_1[0] >> 4, sl_24);
                                vx_hi[8] = qmul4_direct(c_1_1[1], sh_24);
                                vx_hi[10] = qmul4_direct(c_1_1[1] >> 4, sh_24);
                                unsigned int sl_25 = __shfl_sync(0xFFFFFFFF, vx_s[3], src_lo_1_1);
                                unsigned int sh_25 = __shfl_sync(0xFFFFFFFF, vx_s[3], src_hi_1_1);
                                vx_lo[9] = qmul4_direct(c_1_1[2], sl_25);
                                vx_lo[11] = qmul4_direct(c_1_1[2] >> 4, sl_25);
                                vx_hi[9] = qmul4_direct(c_1_1[3], sh_25);
                                vx_hi[11] = qmul4_direct(c_1_1[3] >> 4, sh_25);
                                unsigned int sl_26 = __shfl_sync(0xFFFFFFFF, vx_s[5], src_lo_1_1);
                                unsigned int sh_26 = __shfl_sync(0xFFFFFFFF, vx_s[5], src_hi_1_1);
                                vx_lo[12] = qmul4_direct(c_1_1[4], sl_26);
                                vx_lo[14] = qmul4_direct(c_1_1[4] >> 4, sl_26);
                                vx_hi[12] = qmul4_direct(c_1_1[5], sh_26);
                                vx_hi[14] = qmul4_direct(c_1_1[5] >> 4, sh_26);
                                unsigned int sl_27 = __shfl_sync(0xFFFFFFFF, vx_s[7], src_lo_1_1);
                                unsigned int sh_27 = __shfl_sync(0xFFFFFFFF, vx_s[7], src_hi_1_1);
                                vx_lo[13] = qmul4_direct(c_1_1[6], sl_27);
                                vx_lo[15] = qmul4_direct(c_1_1[6] >> 4, sl_27);
                                vx_hi[13] = qmul4_direct(c_1_1[7], sh_27);
                                vx_hi[15] = qmul4_direct(c_1_1[7] >> 4, sh_27);
                            } else {
                                unsigned int sl_28 = __shfl_sync(0xFFFFFFFF, vx_s[1], src_lo_1_1);
                                unsigned int sh_28 = __shfl_sync(0xFFFFFFFF, vx_s[1], src_hi_1_1);
                                uint32_t _cvt_f16x2_e4m3x2_70;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_70) : "h"((uint16_t)(sl_28)));
                                uint32_t sl01_12;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_12) : "r"(_cvt_f16x2_e4m3x2_70), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_71;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_71) : "h"((uint16_t)(sl_28 >> 16)));
                                uint32_t sl23_12;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_12) : "r"(_cvt_f16x2_e4m3x2_71), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_72;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_72) : "h"((uint16_t)(sh_28)));
                                uint32_t sh01_12;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_12) : "r"(_cvt_f16x2_e4m3x2_72), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_73_1;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_73_1) : "h"((uint16_t)(sh_28 >> 16)));
                                uint32_t sh23_12;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_12) : "r"(_cvt_f16x2_e4m3x2_73_1), "r"(x_kexp_f16x2));
                                unsigned int x_45 = c_1_1[0] & 252645135;
                                unsigned int p_45 = x_45 | x_45 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_74;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_74) : "r"((uint32_t)(p_45)));
                                uint32_t prod_lo_45;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_45) : "r"(_cvt_f16x2_e2m1x2_74), "r"(sl01_12));
                                uint32_t _cvt_f16x2_e2m1x2_75;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_75) : "r"((uint32_t)(p_45 >> 16)));
                                uint32_t prod_hi_45;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_45) : "r"(_cvt_f16x2_e2m1x2_75), "r"(sl23_12));
                                uint16_t lo_45;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_45) : "r"(prod_lo_45));
                                uint16_t hi_45;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_45) : "r"(prod_hi_45));
                                uint32_t _mov_b32_pack_u16x2_76;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_76) : "h"(lo_45), "h"(hi_45));
                                vx_lo[8] = _mov_b32_pack_u16x2_76;
                                unsigned int x_45_1 = c_1_1[0] >> 4 & 252645135;
                                unsigned int p_45_1 = x_45_1 | x_45_1 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_77_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_77_1) : "r"((uint32_t)(p_45_1)));
                                uint32_t prod_lo_45_1;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_45_1) : "r"(_cvt_f16x2_e2m1x2_77_1), "r"(sl01_12));
                                uint32_t _cvt_f16x2_e2m1x2_78_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_78_1) : "r"((uint32_t)(p_45_1 >> 16)));
                                uint32_t prod_hi_45_1;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_45_1) : "r"(_cvt_f16x2_e2m1x2_78_1), "r"(sl23_12));
                                uint16_t lo_45_1;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_45_1) : "r"(prod_lo_45_1));
                                uint16_t hi_45_1;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_45_1) : "r"(prod_hi_45_1));
                                uint32_t _mov_b32_pack_u16x2_79_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_79_1) : "h"(lo_45_1), "h"(hi_45_1));
                                vx_lo[10] = _mov_b32_pack_u16x2_79_1;
                                unsigned int x_46 = c_1_1[1] & 252645135;
                                unsigned int p_46 = x_46 | x_46 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_80_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_80_1) : "r"((uint32_t)(p_46)));
                                uint32_t prod_lo_46;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_46) : "r"(_cvt_f16x2_e2m1x2_80_1), "r"(sh01_12));
                                uint32_t _cvt_f16x2_e2m1x2_81_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_81_1) : "r"((uint32_t)(p_46 >> 16)));
                                uint32_t prod_hi_46;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_46) : "r"(_cvt_f16x2_e2m1x2_81_1), "r"(sh23_12));
                                uint16_t lo_46;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_46) : "r"(prod_lo_46));
                                uint16_t hi_46;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_46) : "r"(prod_hi_46));
                                uint32_t _mov_b32_pack_u16x2_82_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_82_1) : "h"(lo_46), "h"(hi_46));
                                vx_hi[8] = _mov_b32_pack_u16x2_82_1;
                                unsigned int x_47 = c_1_1[1] >> 4 & 252645135;
                                unsigned int p_47 = x_47 | x_47 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_83_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_83_1) : "r"((uint32_t)(p_47)));
                                uint32_t prod_lo_47;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_47) : "r"(_cvt_f16x2_e2m1x2_83_1), "r"(sh01_12));
                                uint32_t _cvt_f16x2_e2m1x2_84_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_84_1) : "r"((uint32_t)(p_47 >> 16)));
                                uint32_t prod_hi_47;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_47) : "r"(_cvt_f16x2_e2m1x2_84_1), "r"(sh23_12));
                                uint16_t lo_47;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_47) : "r"(prod_lo_47));
                                uint16_t hi_47;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_47) : "r"(prod_hi_47));
                                uint32_t _mov_b32_pack_u16x2_85_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_85_1) : "h"(lo_47), "h"(hi_47));
                                vx_hi[10] = _mov_b32_pack_u16x2_85_1;
                                unsigned int sl_29 = __shfl_sync(0xFFFFFFFF, vx_s[3], src_lo_1_1);
                                unsigned int sh_29 = __shfl_sync(0xFFFFFFFF, vx_s[3], src_hi_1_1);
                                uint32_t _cvt_f16x2_e4m3x2_86;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_86) : "h"((uint16_t)(sl_29)));
                                uint32_t sl01_13;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_13) : "r"(_cvt_f16x2_e4m3x2_86), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_87;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_87) : "h"((uint16_t)(sl_29 >> 16)));
                                uint32_t sl23_13;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_13) : "r"(_cvt_f16x2_e4m3x2_87), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_88;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_88) : "h"((uint16_t)(sh_29)));
                                uint32_t sh01_13;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_13) : "r"(_cvt_f16x2_e4m3x2_88), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_89_1;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_89_1) : "h"((uint16_t)(sh_29 >> 16)));
                                uint32_t sh23_13;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_13) : "r"(_cvt_f16x2_e4m3x2_89_1), "r"(x_kexp_f16x2));
                                unsigned int x_48 = c_1_1[2] & 252645135;
                                unsigned int p_48 = x_48 | x_48 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_90;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_90) : "r"((uint32_t)(p_48)));
                                uint32_t prod_lo_48;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_48) : "r"(_cvt_f16x2_e2m1x2_90), "r"(sl01_13));
                                uint32_t _cvt_f16x2_e2m1x2_91;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_91) : "r"((uint32_t)(p_48 >> 16)));
                                uint32_t prod_hi_48;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_48) : "r"(_cvt_f16x2_e2m1x2_91), "r"(sl23_13));
                                uint16_t lo_48;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_48) : "r"(prod_lo_48));
                                uint16_t hi_48;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_48) : "r"(prod_hi_48));
                                uint32_t _mov_b32_pack_u16x2_92;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_92) : "h"(lo_48), "h"(hi_48));
                                vx_lo[9] = _mov_b32_pack_u16x2_92;
                                unsigned int x_49 = c_1_1[2] >> 4 & 252645135;
                                unsigned int p_49 = x_49 | x_49 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_93_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_93_1) : "r"((uint32_t)(p_49)));
                                uint32_t prod_lo_49;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_49) : "r"(_cvt_f16x2_e2m1x2_93_1), "r"(sl01_13));
                                uint32_t _cvt_f16x2_e2m1x2_94_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_94_1) : "r"((uint32_t)(p_49 >> 16)));
                                uint32_t prod_hi_49;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_49) : "r"(_cvt_f16x2_e2m1x2_94_1), "r"(sl23_13));
                                uint16_t lo_49;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_49) : "r"(prod_lo_49));
                                uint16_t hi_49;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_49) : "r"(prod_hi_49));
                                uint32_t _mov_b32_pack_u16x2_95_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_95_1) : "h"(lo_49), "h"(hi_49));
                                vx_lo[11] = _mov_b32_pack_u16x2_95_1;
                                unsigned int x_50 = c_1_1[3] & 252645135;
                                unsigned int p_50 = x_50 | x_50 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_96_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_96_1) : "r"((uint32_t)(p_50)));
                                uint32_t prod_lo_50;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_50) : "r"(_cvt_f16x2_e2m1x2_96_1), "r"(sh01_13));
                                uint32_t _cvt_f16x2_e2m1x2_97_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_97_1) : "r"((uint32_t)(p_50 >> 16)));
                                uint32_t prod_hi_50;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_50) : "r"(_cvt_f16x2_e2m1x2_97_1), "r"(sh23_13));
                                uint16_t lo_50;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_50) : "r"(prod_lo_50));
                                uint16_t hi_50;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_50) : "r"(prod_hi_50));
                                uint32_t _mov_b32_pack_u16x2_98_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_98_1) : "h"(lo_50), "h"(hi_50));
                                vx_hi[9] = _mov_b32_pack_u16x2_98_1;
                                unsigned int x_51 = c_1_1[3] >> 4 & 252645135;
                                unsigned int p_51 = x_51 | x_51 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_99_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_99_1) : "r"((uint32_t)(p_51)));
                                uint32_t prod_lo_51;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_51) : "r"(_cvt_f16x2_e2m1x2_99_1), "r"(sh01_13));
                                uint32_t _cvt_f16x2_e2m1x2_100_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_100_1) : "r"((uint32_t)(p_51 >> 16)));
                                uint32_t prod_hi_51;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_51) : "r"(_cvt_f16x2_e2m1x2_100_1), "r"(sh23_13));
                                uint16_t lo_51;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_51) : "r"(prod_lo_51));
                                uint16_t hi_51;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_51) : "r"(prod_hi_51));
                                uint32_t _mov_b32_pack_u16x2_101_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_101_1) : "h"(lo_51), "h"(hi_51));
                                vx_hi[11] = _mov_b32_pack_u16x2_101_1;
                                unsigned int sl_30 = __shfl_sync(0xFFFFFFFF, vx_s[5], src_lo_1_1);
                                unsigned int sh_30 = __shfl_sync(0xFFFFFFFF, vx_s[5], src_hi_1_1);
                                uint32_t _cvt_f16x2_e4m3x2_102;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_102) : "h"((uint16_t)(sl_30)));
                                uint32_t sl01_14;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_14) : "r"(_cvt_f16x2_e4m3x2_102), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_103;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_103) : "h"((uint16_t)(sl_30 >> 16)));
                                uint32_t sl23_14;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_14) : "r"(_cvt_f16x2_e4m3x2_103), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_104;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_104) : "h"((uint16_t)(sh_30)));
                                uint32_t sh01_14;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_14) : "r"(_cvt_f16x2_e4m3x2_104), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_105_1;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_105_1) : "h"((uint16_t)(sh_30 >> 16)));
                                uint32_t sh23_14;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_14) : "r"(_cvt_f16x2_e4m3x2_105_1), "r"(x_kexp_f16x2));
                                unsigned int x_52 = c_1_1[4] & 252645135;
                                unsigned int p_52 = x_52 | x_52 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_106;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_106) : "r"((uint32_t)(p_52)));
                                uint32_t prod_lo_52;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_52) : "r"(_cvt_f16x2_e2m1x2_106), "r"(sl01_14));
                                uint32_t _cvt_f16x2_e2m1x2_107;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_107) : "r"((uint32_t)(p_52 >> 16)));
                                uint32_t prod_hi_52;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_52) : "r"(_cvt_f16x2_e2m1x2_107), "r"(sl23_14));
                                uint16_t lo_52;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_52) : "r"(prod_lo_52));
                                uint16_t hi_52;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_52) : "r"(prod_hi_52));
                                uint32_t _mov_b32_pack_u16x2_108;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_108) : "h"(lo_52), "h"(hi_52));
                                vx_lo[12] = _mov_b32_pack_u16x2_108;
                                unsigned int x_53 = c_1_1[4] >> 4 & 252645135;
                                unsigned int p_53 = x_53 | x_53 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_109_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_109_1) : "r"((uint32_t)(p_53)));
                                uint32_t prod_lo_53;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_53) : "r"(_cvt_f16x2_e2m1x2_109_1), "r"(sl01_14));
                                uint32_t _cvt_f16x2_e2m1x2_110_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_110_1) : "r"((uint32_t)(p_53 >> 16)));
                                uint32_t prod_hi_53;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_53) : "r"(_cvt_f16x2_e2m1x2_110_1), "r"(sl23_14));
                                uint16_t lo_53;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_53) : "r"(prod_lo_53));
                                uint16_t hi_53;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_53) : "r"(prod_hi_53));
                                uint32_t _mov_b32_pack_u16x2_111_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_111_1) : "h"(lo_53), "h"(hi_53));
                                vx_lo[14] = _mov_b32_pack_u16x2_111_1;
                                unsigned int x_54 = c_1_1[5] & 252645135;
                                unsigned int p_54 = x_54 | x_54 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_112_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_112_1) : "r"((uint32_t)(p_54)));
                                uint32_t prod_lo_54;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_54) : "r"(_cvt_f16x2_e2m1x2_112_1), "r"(sh01_14));
                                uint32_t _cvt_f16x2_e2m1x2_113_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_113_1) : "r"((uint32_t)(p_54 >> 16)));
                                uint32_t prod_hi_54;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_54) : "r"(_cvt_f16x2_e2m1x2_113_1), "r"(sh23_14));
                                uint16_t lo_54;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_54) : "r"(prod_lo_54));
                                uint16_t hi_54;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_54) : "r"(prod_hi_54));
                                uint32_t _mov_b32_pack_u16x2_114_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_114_1) : "h"(lo_54), "h"(hi_54));
                                vx_hi[12] = _mov_b32_pack_u16x2_114_1;
                                unsigned int x_55 = c_1_1[5] >> 4 & 252645135;
                                unsigned int p_55 = x_55 | x_55 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_115_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_115_1) : "r"((uint32_t)(p_55)));
                                uint32_t prod_lo_55;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_55) : "r"(_cvt_f16x2_e2m1x2_115_1), "r"(sh01_14));
                                uint32_t _cvt_f16x2_e2m1x2_116_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_116_1) : "r"((uint32_t)(p_55 >> 16)));
                                uint32_t prod_hi_55;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_55) : "r"(_cvt_f16x2_e2m1x2_116_1), "r"(sh23_14));
                                uint16_t lo_55;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_55) : "r"(prod_lo_55));
                                uint16_t hi_55;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_55) : "r"(prod_hi_55));
                                uint32_t _mov_b32_pack_u16x2_117_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_117_1) : "h"(lo_55), "h"(hi_55));
                                vx_hi[14] = _mov_b32_pack_u16x2_117_1;
                                unsigned int sl_31 = __shfl_sync(0xFFFFFFFF, vx_s[7], src_lo_1_1);
                                unsigned int sh_31 = __shfl_sync(0xFFFFFFFF, vx_s[7], src_hi_1_1);
                                uint32_t _cvt_f16x2_e4m3x2_118;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_118) : "h"((uint16_t)(sl_31)));
                                uint32_t sl01_15;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl01_15) : "r"(_cvt_f16x2_e4m3x2_118), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_119;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_119) : "h"((uint16_t)(sl_31 >> 16)));
                                uint32_t sl23_15;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sl23_15) : "r"(_cvt_f16x2_e4m3x2_119), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_120;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_120) : "h"((uint16_t)(sh_31)));
                                uint32_t sh01_15;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh01_15) : "r"(_cvt_f16x2_e4m3x2_120), "r"(x_kexp_f16x2));
                                uint32_t _cvt_f16x2_e4m3x2_121_1;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_121_1) : "h"((uint16_t)(sh_31 >> 16)));
                                uint32_t sh23_15;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(sh23_15) : "r"(_cvt_f16x2_e4m3x2_121_1), "r"(x_kexp_f16x2));
                                unsigned int x_56 = c_1_1[6] & 252645135;
                                unsigned int p_56 = x_56 | x_56 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_122;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_122) : "r"((uint32_t)(p_56)));
                                uint32_t prod_lo_56;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_56) : "r"(_cvt_f16x2_e2m1x2_122), "r"(sl01_15));
                                uint32_t _cvt_f16x2_e2m1x2_123;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_123) : "r"((uint32_t)(p_56 >> 16)));
                                uint32_t prod_hi_56;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_56) : "r"(_cvt_f16x2_e2m1x2_123), "r"(sl23_15));
                                uint16_t lo_56;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_56) : "r"(prod_lo_56));
                                uint16_t hi_56;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_56) : "r"(prod_hi_56));
                                uint32_t _mov_b32_pack_u16x2_124;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_124) : "h"(lo_56), "h"(hi_56));
                                vx_lo[13] = _mov_b32_pack_u16x2_124;
                                unsigned int x_57 = c_1_1[6] >> 4 & 252645135;
                                unsigned int p_57 = x_57 | x_57 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_125_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_125_1) : "r"((uint32_t)(p_57)));
                                uint32_t prod_lo_57;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_57) : "r"(_cvt_f16x2_e2m1x2_125_1), "r"(sl01_15));
                                uint32_t _cvt_f16x2_e2m1x2_126_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_126_1) : "r"((uint32_t)(p_57 >> 16)));
                                uint32_t prod_hi_57;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_57) : "r"(_cvt_f16x2_e2m1x2_126_1), "r"(sl23_15));
                                uint16_t lo_57;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_57) : "r"(prod_lo_57));
                                uint16_t hi_57;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_57) : "r"(prod_hi_57));
                                uint32_t _mov_b32_pack_u16x2_127_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_127_1) : "h"(lo_57), "h"(hi_57));
                                vx_lo[15] = _mov_b32_pack_u16x2_127_1;
                                unsigned int x_58 = c_1_1[7] & 252645135;
                                unsigned int p_58 = x_58 | x_58 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_128_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_128_1) : "r"((uint32_t)(p_58)));
                                uint32_t prod_lo_58;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_58) : "r"(_cvt_f16x2_e2m1x2_128_1), "r"(sh01_15));
                                uint32_t _cvt_f16x2_e2m1x2_129_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_129_1) : "r"((uint32_t)(p_58 >> 16)));
                                uint32_t prod_hi_58;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_58) : "r"(_cvt_f16x2_e2m1x2_129_1), "r"(sh23_15));
                                uint16_t lo_58;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_58) : "r"(prod_lo_58));
                                uint16_t hi_58;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_58) : "r"(prod_hi_58));
                                uint32_t _mov_b32_pack_u16x2_130_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_130_1) : "h"(lo_58), "h"(hi_58));
                                vx_hi[13] = _mov_b32_pack_u16x2_130_1;
                                unsigned int x_59 = c_1_1[7] >> 4 & 252645135;
                                unsigned int p_59 = x_59 | x_59 >> 4;
                                uint32_t _cvt_f16x2_e2m1x2_131_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_131_1) : "r"((uint32_t)(p_59)));
                                uint32_t prod_lo_59;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_lo_59) : "r"(_cvt_f16x2_e2m1x2_131_1), "r"(sh01_15));
                                uint32_t _cvt_f16x2_e2m1x2_132_1;
                                asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_132_1) : "r"((uint32_t)(p_59 >> 16)));
                                uint32_t prod_hi_59;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(prod_hi_59) : "r"(_cvt_f16x2_e2m1x2_132_1), "r"(sh23_15));
                                uint16_t lo_59;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo_59) : "r"(prod_lo_59));
                                uint16_t hi_59;
                                asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi_59) : "r"(prod_hi_59));
                                uint32_t _mov_b32_pack_u16x2_133_1;
                                asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_133_1) : "h"(lo_59), "h"(hi_59));
                                vx_hi[15] = _mov_b32_pack_u16x2_133_1;
                            }
                        } else {
                            unsigned int r[32];
                            int kk = 4 * ((lane & 7) >> 1) + 2 * (lane >> 3 & 1) + (lane & 1);
                            unsigned int swz_2 = (unsigned int)((kk & 7) << 4);
                            unsigned int seg = (unsigned int)((4 * (vx_warp - 2) + (lane >> 4)) * 16);
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)(kk * 128) + (seg ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)(kk * 128) + (seg + 32 ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)((16 + kk) * 128) + (seg ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)((16 + kk) * 128) + (seg + 32 ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)((32 + kk) * 128) + (seg ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)((32 + kk) * 128) + (seg + 32 ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)((48 + kk) * 128) + (seg ^ swz_2))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                                : "r"((unsigned int)(x_stage_base + 16384) + (unsigned int)((48 + kk) * 128) + (seg + 32 ^ swz_2))
                                : "memory");
                            if (x_kexp != 0) {
                                uint32_t r_scaled_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_0) : "r"(r[0]), "r"(x_kexp_bf16x2));
                                r[0] = r_scaled_0;
                                uint32_t r_scaled_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_1) : "r"(r[1]), "r"(x_kexp_bf16x2));
                                r[1] = r_scaled_1;
                                uint32_t r_scaled_2;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_2) : "r"(r[2]), "r"(x_kexp_bf16x2));
                                r[2] = r_scaled_2;
                                uint32_t r_scaled_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_3) : "r"(r[3]), "r"(x_kexp_bf16x2));
                                r[3] = r_scaled_3;
                                uint32_t r_scaled_4;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_4) : "r"(r[4]), "r"(x_kexp_bf16x2));
                                r[4] = r_scaled_4;
                                uint32_t r_scaled_5;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_5) : "r"(r[5]), "r"(x_kexp_bf16x2));
                                r[5] = r_scaled_5;
                                uint32_t r_scaled_6;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_6) : "r"(r[6]), "r"(x_kexp_bf16x2));
                                r[6] = r_scaled_6;
                                uint32_t r_scaled_7;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_7) : "r"(r[7]), "r"(x_kexp_bf16x2));
                                r[7] = r_scaled_7;
                                uint32_t r_scaled_8;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_8) : "r"(r[8]), "r"(x_kexp_bf16x2));
                                r[8] = r_scaled_8;
                                uint32_t r_scaled_9;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_9) : "r"(r[9]), "r"(x_kexp_bf16x2));
                                r[9] = r_scaled_9;
                                uint32_t r_scaled_10;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_10) : "r"(r[10]), "r"(x_kexp_bf16x2));
                                r[10] = r_scaled_10;
                                uint32_t r_scaled_11;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_11) : "r"(r[11]), "r"(x_kexp_bf16x2));
                                r[11] = r_scaled_11;
                                uint32_t r_scaled_12;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_12) : "r"(r[12]), "r"(x_kexp_bf16x2));
                                r[12] = r_scaled_12;
                                uint32_t r_scaled_13;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_13) : "r"(r[13]), "r"(x_kexp_bf16x2));
                                r[13] = r_scaled_13;
                                uint32_t r_scaled_14;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_14) : "r"(r[14]), "r"(x_kexp_bf16x2));
                                r[14] = r_scaled_14;
                                uint32_t r_scaled_15;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_15) : "r"(r[15]), "r"(x_kexp_bf16x2));
                                r[15] = r_scaled_15;
                                uint32_t r_scaled_16;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_16) : "r"(r[16]), "r"(x_kexp_bf16x2));
                                r[16] = r_scaled_16;
                                uint32_t r_scaled_17;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_17) : "r"(r[17]), "r"(x_kexp_bf16x2));
                                r[17] = r_scaled_17;
                                uint32_t r_scaled_18;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_18) : "r"(r[18]), "r"(x_kexp_bf16x2));
                                r[18] = r_scaled_18;
                                uint32_t r_scaled_19;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_19) : "r"(r[19]), "r"(x_kexp_bf16x2));
                                r[19] = r_scaled_19;
                                uint32_t r_scaled_20;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_20) : "r"(r[20]), "r"(x_kexp_bf16x2));
                                r[20] = r_scaled_20;
                                uint32_t r_scaled_21;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_21) : "r"(r[21]), "r"(x_kexp_bf16x2));
                                r[21] = r_scaled_21;
                                uint32_t r_scaled_22;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_22) : "r"(r[22]), "r"(x_kexp_bf16x2));
                                r[22] = r_scaled_22;
                                uint32_t r_scaled_23;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_23) : "r"(r[23]), "r"(x_kexp_bf16x2));
                                r[23] = r_scaled_23;
                                uint32_t r_scaled_24;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_24) : "r"(r[24]), "r"(x_kexp_bf16x2));
                                r[24] = r_scaled_24;
                                uint32_t r_scaled_25;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_25) : "r"(r[25]), "r"(x_kexp_bf16x2));
                                r[25] = r_scaled_25;
                                uint32_t r_scaled_26;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_26) : "r"(r[26]), "r"(x_kexp_bf16x2));
                                r[26] = r_scaled_26;
                                uint32_t r_scaled_27;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_27) : "r"(r[27]), "r"(x_kexp_bf16x2));
                                r[27] = r_scaled_27;
                                uint32_t r_scaled_28;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_28) : "r"(r[28]), "r"(x_kexp_bf16x2));
                                r[28] = r_scaled_28;
                                uint32_t r_scaled_29;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_29) : "r"(r[29]), "r"(x_kexp_bf16x2));
                                r[29] = r_scaled_29;
                                uint32_t r_scaled_30;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_30) : "r"(r[30]), "r"(x_kexp_bf16x2));
                                r[30] = r_scaled_30;
                                uint32_t r_scaled_31;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_scaled_31) : "r"(r[31]), "r"(x_kexp_bf16x2));
                                r[31] = r_scaled_31;
                            }
                            unsigned int w[16];
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_135;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_135) : "r"(r[0]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[0]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_135) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_136;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_136) : "r"(r[1]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[1]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_136) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_134;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_134) : "h"(_cvt_satfinite_e4m3x2_bf16x2_135), "h"(_cvt_satfinite_e4m3x2_bf16x2_136));
                            w[0] = _mov_b32_pack_u16x2_134;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_138;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_138) : "r"(r[8]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[8]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_138) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_139;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_139) : "r"(r[9]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[9]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_139) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_137;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_137) : "h"(_cvt_satfinite_e4m3x2_bf16x2_138), "h"(_cvt_satfinite_e4m3x2_bf16x2_139));
                            w[1] = _mov_b32_pack_u16x2_137;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_141;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_141) : "r"(r[16]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[16]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_141) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_142;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_142) : "r"(r[17]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[17]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_142) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_140;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_140) : "h"(_cvt_satfinite_e4m3x2_bf16x2_141), "h"(_cvt_satfinite_e4m3x2_bf16x2_142));
                            w[2] = _mov_b32_pack_u16x2_140;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_144;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_144) : "r"(r[24]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[24]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_144) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_145;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_145) : "r"(r[25]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[25]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_145) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_143;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_143) : "h"(_cvt_satfinite_e4m3x2_bf16x2_144), "h"(_cvt_satfinite_e4m3x2_bf16x2_145));
                            w[3] = _mov_b32_pack_u16x2_143;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_147;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "r"(r[2]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[2]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_147) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_148;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_148) : "r"(r[3]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[3]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_148) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_146;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_146) : "h"(_cvt_satfinite_e4m3x2_bf16x2_147), "h"(_cvt_satfinite_e4m3x2_bf16x2_148));
                            w[4] = _mov_b32_pack_u16x2_146;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_150;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_150) : "r"(r[10]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[10]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_150) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_151;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "r"(r[11]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[11]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_151) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_149;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_149) : "h"(_cvt_satfinite_e4m3x2_bf16x2_150), "h"(_cvt_satfinite_e4m3x2_bf16x2_151));
                            w[5] = _mov_b32_pack_u16x2_149;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_153;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "r"(r[18]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[18]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_153) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_154;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_154) : "r"(r[19]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[19]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_154) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_152;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_152) : "h"(_cvt_satfinite_e4m3x2_bf16x2_153), "h"(_cvt_satfinite_e4m3x2_bf16x2_154));
                            w[6] = _mov_b32_pack_u16x2_152;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_156;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_156) : "r"(r[26]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[26]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_156) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_157;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "r"(r[27]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[27]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_157) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_155;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_155) : "h"(_cvt_satfinite_e4m3x2_bf16x2_156), "h"(_cvt_satfinite_e4m3x2_bf16x2_157));
                            w[7] = _mov_b32_pack_u16x2_155;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_159;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "r"(r[4]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[4]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_160;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_160) : "r"(r[5]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[5]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_160) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_158;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_158) : "h"(_cvt_satfinite_e4m3x2_bf16x2_159), "h"(_cvt_satfinite_e4m3x2_bf16x2_160));
                            w[8] = _mov_b32_pack_u16x2_158;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_162;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_162) : "r"(r[12]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[12]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_162) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_163;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "r"(r[13]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[13]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_161;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_161) : "h"(_cvt_satfinite_e4m3x2_bf16x2_162), "h"(_cvt_satfinite_e4m3x2_bf16x2_163));
                            w[9] = _mov_b32_pack_u16x2_161;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_165;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "r"(r[20]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[20]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_166;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_166) : "r"(r[21]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[21]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_166) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_164;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_164) : "h"(_cvt_satfinite_e4m3x2_bf16x2_165), "h"(_cvt_satfinite_e4m3x2_bf16x2_166));
                            w[10] = _mov_b32_pack_u16x2_164;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_168;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_168) : "r"(r[28]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[28]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_168) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_169;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_169) : "r"(r[29]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[29]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_169) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_167;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_167) : "h"(_cvt_satfinite_e4m3x2_bf16x2_168), "h"(_cvt_satfinite_e4m3x2_bf16x2_169));
                            w[11] = _mov_b32_pack_u16x2_167;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_171;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "r"(r[6]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[6]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_171) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_172;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_172) : "r"(r[7]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[7]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_172) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_170;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_170) : "h"(_cvt_satfinite_e4m3x2_bf16x2_171), "h"(_cvt_satfinite_e4m3x2_bf16x2_172));
                            w[12] = _mov_b32_pack_u16x2_170;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_174;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_174) : "r"(r[14]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[14]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_174) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_175;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "r"(r[15]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[15]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_175) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_173;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_173) : "h"(_cvt_satfinite_e4m3x2_bf16x2_174), "h"(_cvt_satfinite_e4m3x2_bf16x2_175));
                            w[13] = _mov_b32_pack_u16x2_173;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_177;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "r"(r[22]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[22]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_177) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_178;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_178) : "r"(r[23]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[23]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_178) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_176;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_176) : "h"(_cvt_satfinite_e4m3x2_bf16x2_177), "h"(_cvt_satfinite_e4m3x2_bf16x2_178));
                            w[14] = _mov_b32_pack_u16x2_176;
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_180;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_180) : "r"(r[30]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[30]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_180) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_181;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "r"(r[31]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(r[31]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_181) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _mov_b32_pack_u16x2_179;
                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_179) : "h"(_cvt_satfinite_e4m3x2_bf16x2_180), "h"(_cvt_satfinite_e4m3x2_bf16x2_181));
                            w[15] = _mov_b32_pack_u16x2_179;
                            vx_lo[8] = w[0];
                            vx_hi[8] = w[8];
                            vx_lo[9] = w[1];
                            vx_hi[9] = w[9];
                            vx_lo[10] = w[4];
                            vx_hi[10] = w[12];
                            vx_lo[11] = w[5];
                            vx_hi[11] = w[13];
                            vx_lo[12] = w[2];
                            vx_hi[12] = w[10];
                            vx_lo[13] = w[3];
                            vx_hi[13] = w[11];
                            vx_lo[14] = w[6];
                            vx_hi[14] = w[14];
                            vx_lo[15] = w[7];
                            vx_hi[15] = w[15];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(vx_vt + 32), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[3])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[4])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[5])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[6])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[7])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[8])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[9])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[10])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[11])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[12])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[13])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[14])), "r"(*reinterpret_cast<const uint32_t*>(&vx_lo[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(vx_vt + 32 + (unsigned int)(16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[3])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[4])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[5])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[6])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[7])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[8])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[9])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[10])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[11])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[12])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[13])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[14])), "r"(*reinterpret_cast<const uint32_t*>(&vx_hi[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (vx_thread == 0) {
                            mbarrier_arrive(kstage_free_addr + ((int)((unsigned int)(vx_tile_cursor + vx_tile) % 5)) * 8);
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
