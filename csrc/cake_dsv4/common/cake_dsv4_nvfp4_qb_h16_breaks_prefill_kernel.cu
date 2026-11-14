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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
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
kernel_cake_dsv4_nvfp4_qb_h16_breaks_prefill(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, const __grid_constant__ CUtensorMap tmap_o, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int has_sinks, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens, const __grid_constant__ CUtensorMap tmap_o16)
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
    #define index_empty_addr (mbar_base + 40)
    #define k_full_addr (mbar_base + 56)
    #define k_empty_addr (mbar_base + 80)
    #define kstage_free_addr (mbar_base + 104)
    #define sf_full_addr (mbar_base + 128)
    #define kexp_full_addr (mbar_base + 152)
    #define v_full_addr (mbar_base + 168)
    #define v_empty_addr (mbar_base + 184)
    #define s_full_addr (mbar_base + 200)
    #define s_empty_addr (mbar_base + 216)
    #define p_full_addr (mbar_base + 232)
    #define p_recycle_addr (mbar_base + 248)
    #define stats_addr (mbar_base + 264)
    #define o_empty_addr (mbar_base + 280)
    #define o_full_addr (mbar_base + 296)
    #define sum_ready_addr (mbar_base + 312)
    #define sum_empty_addr (mbar_base + 320)
    #define tmem_dealloc_addr (mbar_base + 328)
    #define work_full_addr (mbar_base + 336)
    #define work_empty_addr (mbar_base + 352)

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
    float* smem_scale = reinterpret_cast<float*>(smem_raw + 210048);
    const int smem_scale_addr = smem + 210048;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 211072);
    const int smem_exch_addr = smem + 211072;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 212096);
    const int smem_sum_addr = smem + 212096;
    float* smem_fmax = reinterpret_cast<float*>(smem_raw + 212608);
    const int smem_fmax_addr = smem + 212608;
    unsigned int* smem_kexp = reinterpret_cast<unsigned int*>(smem_raw + 213120);
    const int smem_kexp_addr = smem + 213120;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 213152);
    const int work_response_addr = smem + 213152;
    __nv_bfloat16* smem_o_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int smem_o_stage_addr = smem + 214016;
    __nv_bfloat16* smem_o_last0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_o_last0_addr = smem + 9216;
    __nv_bfloat16* smem_o_last1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_o_last1_addr = smem + 25600;
    int* smem_first = reinterpret_cast<int*>(smem_raw + 213184);
    const int smem_first_addr = smem + 213184;

    // Kernel setup ops
    if (warp == 10) {
        int _max_0 = (((int)blockIdx.x - (int)gridDim.x + (num_query_tokens + 3 >> 2)) > (0) ? ((int)blockIdx.x - (int)gridDim.x + (num_query_tokens + 3 >> 2)) : (0));
        int pf_t0 = ((_max_0 * 4) < (num_query_tokens - 1) ? (_max_0 * 4) : (num_query_tokens - 1));
        if (lane < 16) {
            int _min_1 = ((pf_t0 + (lane >> 2)) < (num_query_tokens - 1) ? (pf_t0 + (lane >> 2)) : (num_query_tokens - 1));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(swa_indices + (_min_1 * swa_index_stride + (lane & 3) * 32))));
        }
        if (lane == 31) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(sparse_topk_lens + pf_t0)));
        }
    }
    int pc_fallback[1];
    pc_fallback[0] = 0;
    int pi_rows[8];
    int pi_len[1];
    if (warp == 20) {
        int _max_2 = (((int)blockIdx.x - (int)gridDim.x + (num_query_tokens + 3 >> 2)) > (0) ? ((int)blockIdx.x - (int)gridDim.x + (num_query_tokens + 3 >> 2)) : (0));
        int pg_t0 = ((_max_2 * 4) < (num_query_tokens - 1) ? (_max_2 * 4) : (num_query_tokens - 1));
        {
            const int4* _ivptr_3 = reinterpret_cast<const int4*>(swa_indices + (pg_t0 * swa_index_stride + (lane & 15) * 4) + 0);
            int4 _ivld_3;
            _ivld_3 = *_ivptr_3;
            pi_rows[0 + 0] = _ivld_3.x;
            pi_rows[0 + 1] = _ivld_3.y;
            pi_rows[0 + 2] = _ivld_3.z;
            pi_rows[0 + 3] = _ivld_3.w;
        }
        {
            const int4* _ivptr_4 = reinterpret_cast<const int4*>(swa_indices + (pg_t0 * swa_index_stride + 64 + (lane & 15) * 4) + 0);
            int4 _ivld_4;
            _ivld_4 = *_ivptr_4;
            pi_rows[4 + 0] = _ivld_4.x;
            pi_rows[4 + 1] = _ivld_4.y;
            pi_rows[4 + 2] = _ivld_4.z;
            pi_rows[4 + 3] = _ivld_4.w;
        }
        pi_len[0] = sparse_topk_lens[pg_t0];
    }

    // Mbarrier init (46 barriers at smem_raw[0..368))

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
            // index_full: 2 barriers, init_count=32
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            // index_empty: 2 barriers, init_count=704
            mbarrier_init(smem + 40, 704);
            mbarrier_init(smem + 48, 704);
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // kstage_free: 3 barriers, init_count=2
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // sf_full: 3 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 128);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            mbarrier_init(smem + 240, 128);
            // p_recycle: 2 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 264, 128);
            mbarrier_init(smem + 272, 128);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 312, 128);
            // sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 320, 128);
            // tmem_dealloc: 1 barriers, init_count=576
            mbarrier_init(smem + 328, 576);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            // work_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 352, 32);
            mbarrier_init(smem + 360, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 368);
    if (warp == 0) {
        int _tmem_hold = smem + 368;
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
    if (warp == 20) {
        if (lane < 16) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_first_addr + (unsigned int)(lane * 16)), "r"(pi_rows[0]), "r"(pi_rows[1]), "r"(pi_rows[2]), "r"(pi_rows[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_first_addr + (unsigned int)(256 + lane * 16)), "r"(pi_rows[4]), "r"(pi_rows[5]), "r"(pi_rows[6]), "r"(pi_rows[7]) : "memory");
        }
        if (lane == 0) {
            smem_first[128] = pi_len[0];
        }
    }
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
            int q = blockIdx.x;
            unsigned int _phase_index_full = 0;
            if (blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                q = smem_indices[(int)sm_index_stage * 528 + 512 + 15];
            }
            int sm_query = q;
            int sm_last = blockIdx.x;
            int sm_last_t0 = 0;
            int sm_last_nb = 0;
            #pragma unroll 1
            for (unsigned int sm_work = 0; sm_work < 1048576; sm_work++) {
                sm_last = sm_query;
                int sm_ret = -1;
                int sm_t0 = 0;
                int sm_nb = 0;
                if (sm_query >= 0) {
                    mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                    sm_t0 = smem_indices[(int)sm_index_stage * 528 + 512 + 12];
                    sm_nb = smem_indices[(int)sm_index_stage * 528 + 512 + 13];
                    int nb = sm_nb;
                    int rest = smem_indices[(int)sm_index_stage * 528 + 512] > 1;
                    int tiles = ((smem_indices[(int)sm_index_stage * 528 + 512] > 1) ? 3 + 2 * (nb - smem_indices[(int)sm_index_stage * 528 + 512]) : 2 * nb);
                    int sm_lo = smem_indices[(int)sm_index_stage * 528 + 512 + (2 + 2 * (sm_head >> 4))];
                    int sm_span = smem_indices[(int)sm_index_stage * 528 + 512 + (3 + 2 * (sm_head >> 4))];
                    int sm_hi = sm_lo + (sm_span & 65535);
                    int sm_shift = sm_span >> 16;
                    int _parity_q_load_full = (int)((unsigned int)sm_item % 2);
                    mbarrier_wait(q_load_full_addr, _parity_q_load_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sfa_regs[8];
                    int sfa_row = lane + (sm_row_base & 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[(0) + 3]))
                        : "r"(smem_q_sf_addr + (unsigned int)(sfa_row * 32)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfa_regs[(4) + 3]))
                        : "r"(smem_q_sf_addr + (unsigned int)(sfa_row * 32 + 16)));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(384 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[0])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(388 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 1)[0])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(392 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 2)[0])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(396 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 3)[0])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(400 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 4)[0])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(404 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 5)[0])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(408 + (sm_row_base >> 5)) + (unsigned int)(sm_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 6)[0])));
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
                    if (has_sinks != 0) {
                        row_max_val = sinks[sm_head & 15] * 1.442695f / softmax_scale_log2;
                        row_sum_val = ((sm_half == 0) ? 1.0f : 0.0f);
                    }
                    #pragma unroll 1
                    for (int sm_tile = 0; sm_tile < tiles; sm_tile++) {
                        int s_stage = (sm_tile_cursor + sm_tile) % 2;
                        int sm_col0 = sm_tile * 64 + sm_half * 32;
                        int staged_index[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[0])) : "r"(smem_indices_addr + sm_index_stage * 2112 + (unsigned int)((sm_tile * 64 + sm_half * 32 + lane) * 4)));
                        unsigned int invalid_cols = __ballot_sync(0xFFFFFFFF, staged_index[0] < 0);
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
                        int sm_c0 = sm_col0 - ((sm_tile >= 2) ? sm_shift : 0);
                        uint32_t _thresh_mask_0;
                        {
                            int _lim_1 = sm_hi - sm_c0;
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
                        uint32_t _thresh_mask_2;
                        {
                            int _lim_3 = sm_lo - sm_c0;
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
                        unsigned int keep_cols = _thresh_mask_0 & (_thresh_mask_2 ^ 4294967295u) & (invalid_cols ^ 4294967295u);
                        if (keep_cols != 4294967295u) {
                            if (!(keep_cols & (1u << 0))) sv[0] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 1))) sv[1] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 2))) sv[2] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 3))) sv[3] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 4))) sv[4] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 5))) sv[5] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 6))) sv[6] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 7))) sv[7] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 8))) sv[8] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 9))) sv[9] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 10))) sv[10] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 11))) sv[11] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 12))) sv[12] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 13))) sv[13] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 14))) sv[14] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 15))) sv[15] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 16))) sv[16] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 17))) sv[17] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 18))) sv[18] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 19))) sv[19] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 20))) sv[20] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 21))) sv[21] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 22))) sv[22] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 23))) sv[23] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 24))) sv[24] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 25))) sv[25] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 26))) sv[26] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 27))) sv[27] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 28))) sv[28] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 29))) sv[29] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 30))) sv[30] = -CUDART_INF_F;
                            if (!(keep_cols & (1u << 31))) sv[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_4 = {-CUDART_INF_F, -CUDART_INF_F};
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[0], sv[1]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[2], sv[3]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[4], sv[5]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[6], sv[7]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[8], sv[9]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[10], sv[11]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[12], sv[13]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[14], sv[15]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[16], sv[17]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[18], sv[19]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[20], sv[21]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[22], sv[23]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[24], sv[25]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[26], sv[27]));
                        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[28], sv[29]));
                        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[30], sv[31]));
                        float local_max = row_max_reduce(_reg_reduce_max2_4);
                        smem_exch[s_stage * 128 + sm_thread] = local_max;
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(max_noftz(local_max, smem_exch[s_stage * 128 + (sm_thread ^ 64)])), "f"(row_max_val));
                        if (row_max_val > -CUDART_INF_F && (new_max - row_max_val) * softmax_scale_log2 <= 5.0f) {
                            new_max = row_max_val;
                        }
                        float delta = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max * softmax_scale_log2);
                        float _exp2_5 = approx_exp2(delta);
                        float acc_scale = ((row_max_val > -CUDART_INF_F && new_max != row_max_val) ? _exp2_5 : 1.0f);
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
                        const float2 _fma_b2_6 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_7 = {kexp_f - max_scaled, kexp_f - max_scaled};
                        float2 _fma_pair_8 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_6, _fma_c2_7);
                        sv[0] = _fma_pair_8.x;
                        sv[1] = _fma_pair_8.y;
                        float2 _fma_pair_9 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_6, _fma_c2_7);
                        sv[2] = _fma_pair_9.x;
                        sv[3] = _fma_pair_9.y;
                        float2 _fma_pair_10 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_6, _fma_c2_7);
                        sv[4] = _fma_pair_10.x;
                        sv[5] = _fma_pair_10.y;
                        float2 _fma_pair_11 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_6, _fma_c2_7);
                        sv[6] = _fma_pair_11.x;
                        sv[7] = _fma_pair_11.y;
                        float2 _fma_pair_12 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_6, _fma_c2_7);
                        sv[8] = _fma_pair_12.x;
                        sv[9] = _fma_pair_12.y;
                        float2 _fma_pair_13 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_6, _fma_c2_7);
                        sv[10] = _fma_pair_13.x;
                        sv[11] = _fma_pair_13.y;
                        float2 _fma_pair_14 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_6, _fma_c2_7);
                        sv[12] = _fma_pair_14.x;
                        sv[13] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_6, _fma_c2_7);
                        sv[14] = _fma_pair_15.x;
                        sv[15] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_6, _fma_c2_7);
                        sv[16] = _fma_pair_16.x;
                        sv[17] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_6, _fma_c2_7);
                        sv[18] = _fma_pair_17.x;
                        sv[19] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_6, _fma_c2_7);
                        sv[20] = _fma_pair_18.x;
                        sv[21] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_6, _fma_c2_7);
                        sv[22] = _fma_pair_19.x;
                        sv[23] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_6, _fma_c2_7);
                        sv[24] = _fma_pair_20.x;
                        sv[25] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_6, _fma_c2_7);
                        sv[26] = _fma_pair_21.x;
                        sv[27] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_6, _fma_c2_7);
                        sv[28] = _fma_pair_22.x;
                        sv[29] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_6, _fma_c2_7);
                        sv[30] = _fma_pair_23.x;
                        sv[31] = _fma_pair_23.y;
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
                        float2 _reg_reduce_sum2_25 = make_float2(0.0f, 0.0f);
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[0], sv[1]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[2], sv[3]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[4], sv[5]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[6], sv[7]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[8], sv[9]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[10], sv[11]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[12], sv[13]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[14], sv[15]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[16], sv[17]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[18], sv[19]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[20], sv[21]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[22], sv[23]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[24], sv[25]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[26], sv[27]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[28], sv[29]));
                        _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[30], sv[31]));
                        float _reg_array_reduce_24 = _reg_reduce_sum2_25.x + _reg_reduce_sum2_25.y;
                        float block_sum = _reg_array_reduce_24 * kexp_inv;
                        row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                    }
                    mbarrier_arrive(index_empty_addr + (sm_index_stage) * 8);
                    sm_index_stage += 1;
                    if (sm_index_stage == 2) { sm_index_stage = 0; _phase_index_full ^= 1; }
                    int _parity_sum_empty = (int)((unsigned int)sm_item % 2) ^ 1;
                    mbarrier_wait(sum_empty_addr, _parity_sum_empty);
                    smem_sum[sm_thread] = row_sum_val;
                    smem_fmax[sm_thread] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    sm_tile_cursor += tiles;
                    sm_item += 1;
                }
                sm_last_t0 = sm_t0;
                sm_last_nb = sm_nb;
                mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                sm_query = smem_indices[(int)sm_index_stage * 528 + 512 + 15];
                if (sm_query < 0) {
                    break;
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            float le_sum = smem_sum[sm_thread] + smem_sum[sm_thread ^ 64];
            float _rcp_26 = approx_rcp(le_sum);
            float le_scale = ((le_sum > 0.0f) ? _rcp_26 : 0.0f) * bmm2_scale[0];
            int _stage_p_recycle_1 = (int)((unsigned int)(sm_tile_cursor - 2) % 2);
            int _parity_p_recycle_1 = (int)((unsigned int)(sm_tile_cursor - 2) / 2 % 2);
            mbarrier_wait(p_recycle_addr + (_stage_p_recycle_1) * 8, _parity_p_recycle_1);
            int _parity_o_full = (int)((unsigned int)sm_tile_cursor % 2) ^ 1;
            mbarrier_wait(o_full_addr + 8, _parity_o_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int le_box_row = sm_last_t0 * 16 + (sm_thread >> 5 & 1) * 32;
            unsigned int le_box = ((sm_half == 0) ? smem_o_last0_addr : smem_o_last1_addr) + (unsigned int)((sm_thread >> 5 & 1) * 4096);
            unsigned int le_row_addr = le_box + (unsigned int)(lane * 128);
            unsigned int le_swz = (unsigned int)((lane & 7) << 4);
            #pragma unroll 1
            for (int le_q = 0; le_q < 2; le_q++) {
                unsigned int le_pack[32];
                float le_buf_0[16];
                tmem_ld_x16(&le_buf_0[0], taddr + (unsigned int)(sm_row_base << 16) + (unsigned int)(128 + 64 * le_q));
                const float2 _scale2_27 = {le_scale, le_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(le_buf_0)[_ls], _scale2_27);
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(le_buf_0[_lp*2 + 0], le_buf_0[_lp*2+1 + 0]));
                    le_pack[_lp] = *(uint32_t*)&_bf2;
                }
                float le_buf_1[16];
                tmem_ld_x16(&le_buf_1[0], taddr + (unsigned int)(sm_row_base << 16) + (unsigned int)(128 + 64 * le_q + 16));
                const float2 _scale2_28 = {le_scale, le_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(le_buf_1)[_ls], _scale2_28);
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(le_buf_1[_lp*2 + 0], le_buf_1[_lp*2+1 + 0]));
                    le_pack[_lp + 8] = *(uint32_t*)&_bf2;
                }
                float le_buf_2[16];
                tmem_ld_x16(&le_buf_2[0], taddr + (unsigned int)(sm_row_base << 16) + (unsigned int)(128 + 64 * le_q + 32));
                const float2 _scale2_29 = {le_scale, le_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(le_buf_2)[_ls], _scale2_29);
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(le_buf_2[_lp*2 + 0], le_buf_2[_lp*2+1 + 0]));
                    le_pack[_lp + 16] = *(uint32_t*)&_bf2;
                }
                float le_buf_3[16];
                tmem_ld_x16(&le_buf_3[0], taddr + (unsigned int)(sm_row_base << 16) + (unsigned int)(128 + 64 * le_q + 48));
                const float2 _scale2_30 = {le_scale, le_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(le_buf_3)[_ls], _scale2_30);
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(le_buf_3[_lp*2 + 0], le_buf_3[_lp*2+1 + 0]));
                    le_pack[_lp + 24] = *(uint32_t*)&_bf2;
                }
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                __syncwarp();
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (0 ^ le_swz)), "r"(le_pack[0]), "r"(le_pack[1]), "r"(le_pack[2]), "r"(le_pack[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (16 ^ le_swz)), "r"(le_pack[4]), "r"(le_pack[5]), "r"(le_pack[6]), "r"(le_pack[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (32 ^ le_swz)), "r"(le_pack[8]), "r"(le_pack[9]), "r"(le_pack[10]), "r"(le_pack[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (48 ^ le_swz)), "r"(le_pack[12]), "r"(le_pack[13]), "r"(le_pack[14]), "r"(le_pack[15]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (64 ^ le_swz)), "r"(le_pack[16]), "r"(le_pack[17]), "r"(le_pack[18]), "r"(le_pack[19]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (80 ^ le_swz)), "r"(le_pack[20]), "r"(le_pack[21]), "r"(le_pack[22]), "r"(le_pack[23]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (96 ^ le_swz)), "r"(le_pack[24]), "r"(le_pack[25]), "r"(le_pack[26]), "r"(le_pack[27]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(le_row_addr + (112 ^ le_swz)), "r"(le_pack[28]), "r"(le_pack[29]), "r"(le_pack[30]), "r"(le_pack[31]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (lane == 0) {
                    if ((sm_thread >> 5 & 1) * 32 + 31 >> 4 < sm_last_nb) {
                        tma_store_2d((&tmap_o), 256 + (sm_thread >> 6) * 128 + 64 * le_q, le_box_row, le_box);
                    }
                    if ((sm_thread >> 5 & 1) * 2 + 1 == sm_last_nb) {
                        tma_store_2d((&tmap_o16), 256 + (sm_thread >> 6) * 128 + 64 * le_q, le_box_row, le_box);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            if (lane == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: correction_wg ----
    } else if (warp >= 4 && warp <= 7) {
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
            unsigned int co_info_stage = 0;
            int q_1 = blockIdx.x;
            unsigned int _phase_index_full_1 = 0;
            if (blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                mbarrier_wait(index_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                q_1 = smem_indices[(int)co_info_stage * 528 + 512 + 15];
            }
            int co_query = q_1;
            mbarrier_arrive(o_empty_addr);
            #pragma unroll 1
            for (unsigned int co_work = 0; co_work < 1048576; co_work++) {
                int co_ret = -1;
                if (co_query >= 0) {
                    mbarrier_wait(index_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                    int co_t0 = smem_indices[(int)co_info_stage * 528 + 512 + 12];
                    int co_nb = smem_indices[(int)co_info_stage * 528 + 512 + 13];
                    int nb_1 = co_nb;
                    int rest_1 = smem_indices[(int)co_info_stage * 528 + 512] > 1;
                    int tiles_1 = ((smem_indices[(int)co_info_stage * 528 + 512] > 1) ? 3 + 2 * (nb_1 - smem_indices[(int)co_info_stage * 528 + 512]) : 2 * nb_1);
                    mbarrier_arrive(index_empty_addr + (co_info_stage) * 8);
                    co_info_stage += 1;
                    if (co_info_stage == 2) { co_info_stage = 0; _phase_index_full_1 ^= 1; }
                    #pragma unroll 1
                    for (int co_tile = 0; co_tile < tiles_1; co_tile++) {
                        int c_stage = (co_tile_cursor + co_tile) % 2;
                        int _stage_stats = (int)((unsigned int)(co_tile_cursor + co_tile) % 2);
                        int _parity_stats = (int)((unsigned int)(co_tile_cursor + co_tile) / 2 % 2);
                        mbarrier_wait(stats_addr + (_stage_stats) * 8, _parity_stats);
                        float acc_scale_1 = smem_scale[c_stage * 128 + co_thread];
                        int any_rescale = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (co_tile > 0) {
                            int _parity_o_full_1 = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                            mbarrier_wait(o_full_addr, _parity_o_full_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (any_rescale != 0) {
                                int o_base = taddr + (unsigned int)co_lane_addr;
                                #pragma unroll 2
                                for (int c = 0; c < 128; c += 16) {
                                    float _tmem_load_0[16];
                                    tmem_ld_x16(&_tmem_load_0[0], o_base + c);
                                    const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                                    tmem_st_x16_f32(o_base + c, _tmem_load_0);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
                        }
                        if (co_tile > 0) {
                            int _parity_o_full_2 = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                            mbarrier_wait(o_full_addr + 8, _parity_o_full_2);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (any_rescale != 0) {
                                int o_base_1 = taddr + 128 + (unsigned int)co_lane_addr;
                                #pragma unroll 2
                                for (int c_1 = 0; c_1 < 128; c_1 += 16) {
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], o_base_1 + c_1);
                                    const float2 _scale2_1 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                                    tmem_st_x16_f32(o_base_1 + c_1, _tmem_load_1);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                        }
                        mbarrier_arrive(o_empty_addr + 8);
                    }
                    co_tile_cursor += tiles_1;
                    int _parity_o_full_3 = (int)((unsigned int)co_tile_cursor % 2) ^ 1;
                    mbarrier_wait(o_full_addr, _parity_o_full_3);
                    int _parity_o_full_4 = (int)((unsigned int)co_tile_cursor % 2) ^ 1;
                    mbarrier_wait(o_full_addr + 8, _parity_o_full_4);
                    int _parity_sum_ready = (int)((unsigned int)co_item % 2);
                    mbarrier_wait(sum_ready_addr, _parity_sum_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = smem_sum[co_thread] + smem_sum[co_thread ^ 64];
                    float final_max = smem_fmax[co_thread];
                    mbarrier_arrive(sum_empty_addr);
                    int ep_row = co_t0 * 16 + co_head;
                    if (co_half == 0 && co_head >> 4 < co_nb) {
                        float _log2_2;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(total_sum));
                        LSE[ep_row] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_2) : -CUDART_INF_F);
                    }
                    float _rcp_3 = approx_rcp(total_sum);
                    float ep_scale = ((total_sum > 0.0f) ? _rcp_3 : 0.0f) * output_scale;
                    int ep_box_row = co_t0 * 16 + (co_thread >> 5 & 1) * 32;
                    unsigned int ep_box = smem_o_stage_addr + (unsigned int)((co_thread >> 5) * 4096);
                    unsigned int ep_row_addr = ep_box + (unsigned int)(lane * 128);
                    unsigned int ep_swz = (unsigned int)((lane & 7) << 4);
                    #pragma unroll 1
                    for (int ep_i = 0; ep_i < 4; ep_i++) {
                        int ep_part = ep_i >> 1;
                        int ep_q = ep_i & 1;
                        unsigned int ep_pack[32];
                        float ep_buf_0[16];
                        tmem_ld_x16(&ep_buf_0[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q));
                        const float2 _scale2_4 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_0)[_ls], _scale2_4);
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_0[_lp*2 + 0], ep_buf_0[_lp*2+1 + 0]));
                            ep_pack[_lp] = *(uint32_t*)&_bf2;
                        }
                        float ep_buf_1[16];
                        tmem_ld_x16(&ep_buf_1[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q + 16));
                        const float2 _scale2_5 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_1)[_ls], _scale2_5);
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_1[_lp*2 + 0], ep_buf_1[_lp*2+1 + 0]));
                            ep_pack[_lp + 8] = *(uint32_t*)&_bf2;
                        }
                        float ep_buf_2[16];
                        tmem_ld_x16(&ep_buf_2[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q + 32));
                        const float2 _scale2_6 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_2)[_ls], _scale2_6);
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_2[_lp*2 + 0], ep_buf_2[_lp*2+1 + 0]));
                            ep_pack[_lp + 16] = *(uint32_t*)&_bf2;
                        }
                        float ep_buf_3[16];
                        tmem_ld_x16(&ep_buf_3[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q + 48));
                        const float2 _scale2_7 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_3)[_ls], _scale2_7);
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_3[_lp*2 + 0], ep_buf_3[_lp*2+1 + 0]));
                            ep_pack[_lp + 24] = *(uint32_t*)&_bf2;
                        }
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (0 ^ ep_swz)), "r"(ep_pack[0]), "r"(ep_pack[1]), "r"(ep_pack[2]), "r"(ep_pack[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (16 ^ ep_swz)), "r"(ep_pack[4]), "r"(ep_pack[5]), "r"(ep_pack[6]), "r"(ep_pack[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (32 ^ ep_swz)), "r"(ep_pack[8]), "r"(ep_pack[9]), "r"(ep_pack[10]), "r"(ep_pack[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (48 ^ ep_swz)), "r"(ep_pack[12]), "r"(ep_pack[13]), "r"(ep_pack[14]), "r"(ep_pack[15]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (64 ^ ep_swz)), "r"(ep_pack[16]), "r"(ep_pack[17]), "r"(ep_pack[18]), "r"(ep_pack[19]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (80 ^ ep_swz)), "r"(ep_pack[20]), "r"(ep_pack[21]), "r"(ep_pack[22]), "r"(ep_pack[23]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (96 ^ ep_swz)), "r"(ep_pack[24]), "r"(ep_pack[25]), "r"(ep_pack[26]), "r"(ep_pack[27]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ep_row_addr + (112 ^ ep_swz)), "r"(ep_pack[28]), "r"(ep_pack[29]), "r"(ep_pack[30]), "r"(ep_pack[31]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (lane == 0) {
                            if ((co_thread >> 5 & 1) * 32 + 31 >> 4 < co_nb) {
                                tma_store_2d((&tmap_o), 256 * ep_part + co_half * 128 + 64 * ep_q, ep_box_row, ep_box);
                            }
                            if ((co_thread >> 5 & 1) * 2 + 1 == co_nb) {
                                tma_store_2d((&tmap_o16), 256 * ep_part + co_half * 128 + 64 * ep_q, ep_box_row, ep_box);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (ep_i == 1) {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
                            mbarrier_wait(index_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                            if (smem_indices[(int)co_info_stage * 528 + 512 + 15] < 0) {
                                break;
                            }
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    co_item += 1;
                }
                mbarrier_wait(index_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                co_query = smem_indices[(int)co_info_stage * 528 + 512 + 15];
                if (co_query < 0) {
                    break;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (lane == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 8) {
        { // mma_warp_main
            int mma_tile_cursor = 0;
            int mma_item = 0;
            unsigned int mma_info_stage = 0;
            int q_2 = blockIdx.x;
            unsigned int _phase_index_full_2 = 0;
            if (blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                mbarrier_wait(index_full_addr + (mma_info_stage) * 8, _phase_index_full_2);
                q_2 = smem_indices[(int)mma_info_stage * 528 + 512 + 15];
            }
            int mma_query = q_2;
            #pragma unroll 1
            for (unsigned int mma_work = 0; mma_work < 1048576; mma_work++) {
                int mma_t0 = (mma_query - ((int)gridDim.x - (num_query_tokens + 3 >> 2))) * 4;
                int mma_ret = -1;
                if (mma_query >= 0) {
                    if (mma_query < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                        mbarrier_wait(index_full_addr + (mma_info_stage) * 8, _phase_index_full_2);
                        mma_t0 = smem_indices[(int)mma_info_stage * 528 + 512 + 12];
                    }
                    int _parity_q_empty = (int)((unsigned int)mma_item % 2) ^ 1;
                    mbarrier_wait(q_empty_addr, _parity_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_load_full_addr, 34816);
                        tma_3d_gmem2smem(smem_q_a_addr, (&tmap_q), 0, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_b_addr, (&tmap_q), 128, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_rope_addr, (&tmap_q), 224, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_rope_addr + 8192, (&tmap_q), 224, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_sf_addr, (&tmap_q_sf), 352, 0, mma_t0, q_load_full_addr);
                    }
                    mbarrier_wait(index_full_addr + (mma_info_stage) * 8, _phase_index_full_2);
                    int shared = smem_indices[(int)mma_info_stage * 528 + 512];
                    int nb_read = smem_indices[(int)mma_info_stage * 528 + 512 + 13];
                    int nxt = 0;
                    mbarrier_arrive(index_empty_addr + (mma_info_stage) * 8);
                    mma_info_stage += 1;
                    if (mma_info_stage == 2) { mma_info_stage = 0; _phase_index_full_2 ^= 1; }
                    int nb_2 = nb_read;
                    int rest_2 = shared > 1;
                    int tiles_2 = ((shared > 1) ? 3 + 2 * (nb_2 - shared) : 2 * nb_2);
                    mma_ret = nxt;
                    int _parity_q_full = (int)((unsigned int)mma_item % 2);
                    mbarrier_wait(q_full_addr, _parity_q_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_0 = elect_sync();
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_q_a_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 0)), "l"(_tcgen05_cp_desc_1), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_q_a_addr + 16)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 4)), "l"(_tcgen05_cp_desc_2), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_q_a_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 8)), "l"(_tcgen05_cp_desc_3), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_q_a_addr + 48)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 12)), "l"(_tcgen05_cp_desc_4), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_q_a_addr + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 16)), "l"(_tcgen05_cp_desc_5), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_q_a_addr + 80)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 20)), "l"(_tcgen05_cp_desc_6), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_q_a_addr + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 24)), "l"(_tcgen05_cp_desc_7), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_q_a_addr + 112)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa0 + 28)), "l"(_tcgen05_cp_desc_8), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_q_b_addr)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 0)), "l"(_tcgen05_cp_desc_9), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_q_b_addr + 16)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 4)), "l"(_tcgen05_cp_desc_10), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_q_b_addr + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 8)), "l"(_tcgen05_cp_desc_11), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(smem_q_b_addr + 48)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 12)), "l"(_tcgen05_cp_desc_12), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(smem_q_b_addr + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 16)), "l"(_tcgen05_cp_desc_13), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(smem_q_b_addr + 80)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_tmem_qa1 + 20)), "l"(_tcgen05_cp_desc_14), "r"(_elected_0)
                            : "memory");
                    }
                    #pragma unroll 1
                    for (int mma_tile = 0; mma_tile < tiles_2; mma_tile++) {
                        int m_stage = mma_tile_cursor + mma_tile & 1;
                        int mk_stage = (mma_tile_cursor + mma_tile) % 3;
                        int _stage_sf_full = (int)((unsigned int)(mma_tile_cursor + mma_tile) % 3);
                        int _parity_sf_full = (int)((unsigned int)(mma_tile_cursor + mma_tile) / 3 % 2);
                        mbarrier_wait(sf_full_addr + (_stage_sf_full) * 8, _parity_sf_full);
                        int _stage_s_empty = (int)((unsigned int)(mma_tile_cursor + mma_tile) % 2);
                        int _parity_s_empty = (int)((unsigned int)(mma_tile_cursor + mma_tile) / 2 % 2) ^ 1;
                        mbarrier_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                        int mma_n = ((rest_2 != 0 && mma_tile == 2) ? 16 : 64);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_0_0 = make_warp_uniform((((smem_k_a_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"(tmem_tmem_qa0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"(tmem_tmem_sfa + 0), "r"(tmem_tmem_sfb + m_stage * 16), "r"(0) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa0 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"((tmem_tmem_sfa + 0 + 4)), "r"((tmem_tmem_sfb + (m_stage * 16 - 2) + 4)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa0 + 16)), "l"(((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"((tmem_tmem_sfa + 0 + 8)), "r"((tmem_tmem_sfb + (m_stage * 16 - 4) + 8)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa0 + 24)), "l"(((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"((tmem_tmem_sfa + 0 + 12)), "r"((tmem_tmem_sfb + (m_stage * 16 - 6) + 12)), "r"(1) : "memory");
                            }
                        }
                        int _mma_b_lo_0_1 = make_warp_uniform((((smem_k_b_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"(tmem_tmem_qa1), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"(tmem_tmem_sfa + 16), "r"(tmem_tmem_sfb + (m_stage * 16 + 8)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa1 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"((tmem_tmem_sfa + 16 + 4)), "r"((tmem_tmem_sfb + (m_stage * 16 + 8 - 2) + 4)), "r"(1) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_mma_b_lo_0_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_scratch + (256 + m_stage * 64))), "r"((tmem_tmem_qa1 + 16)), "l"(((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32))), "r"(((0x8100480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17))), "r"((tmem_tmem_sfa + 16 + 8)), "r"((tmem_tmem_sfb + (m_stage * 16 + 8 - 4) + 8)), "r"(1) : "memory");
                            }
                        }
                        int _mma_a_lo_2 = make_warp_uniform((((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 1024);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_k_c_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        {
                            uint64_t _mma_ss_a_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            const uint32_t _mma_ss_idesc_15 = ((0x8100490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(mma_n) >> 3) << 17));
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, _mma_ss_idesc_15, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, _mma_ss_idesc_15, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, _mma_ss_idesc_15, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, _mma_ss_idesc_15, 1);
                            }
                        }
                        elect_commit(k_empty_addr + ((int)((unsigned int)(mma_tile_cursor + mma_tile) % 3)) * 8);
                        elect_commit(s_full_addr + ((int)((unsigned int)(mma_tile_cursor + mma_tile) % 2)) * 8);
                    }
                    elect_commit(q_empty_addr);
                    mma_tile_cursor += tiles_2;
                    mma_item += 1;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_wait(index_full_addr + (mma_info_stage) * 8, _phase_index_full_2);
                mma_query = smem_indices[(int)mma_info_stage * 528 + 512 + 15];
                if (mma_query < 0) {
                    break;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    // ---- Role: index_warp ----
    } else if (warp == 10) {
        { // index_warp_main
            unsigned int ix_stage = 0;
            unsigned int ix_work_stage = 0;
            int ix_query = blockIdx.x;
            int ix_open = blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2);
            int ix_fresh = 1;
            int ix_gb = 0;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int ix_work = 0; ix_work < 1048576; ix_work++) {
                int ix_block = ix_query >= (int)gridDim.x - (num_query_tokens + 3 >> 2);
                int ix_t0 = (ix_query - ((int)gridDim.x - (num_query_tokens + 3 >> 2))) * 4;
                int ix_n = ((num_query_tokens - ix_t0) < (4) ? (num_query_tokens - ix_t0) : (4));
                int ix_more = 0;
                if (ix_block == 0) {
                    ix_n = 0;
                    int ix_nblk = num_query_tokens + 3 >> 2;
                    int ix_per = (ix_nblk + ((int)gridDim.x - (num_query_tokens + 3 >> 2)) - 1) / ((int)gridDim.x - (num_query_tokens + 3 >> 2));
                    int ix_b1 = ((ix_query * ix_per + ix_per) < (ix_nblk) ? (ix_query * ix_per + ix_per) : (ix_nblk));
                    if (ix_fresh != 0) {
                        ix_gb = ix_query * ix_per;
                    }
                    #pragma unroll 1
                    for (int ix_scan = 0; ix_scan < 1048576; ix_scan++) {
                        if (ix_gb >= ix_b1) {
                            break;
                        }
                        int _min_2 = ((ix_b1 - ix_gb) < (8) ? (ix_b1 - ix_gb) : (8));
                        int _min_1_1 = ((_min_2 * 4) < (num_query_tokens - ix_gb * 4) ? (_min_2 * 4) : (num_query_tokens - ix_gb * 4));
                        int _min_0 = ((lane) < (_min_1_1 - 1) ? (lane) : (_min_1_1 - 1));
                        int bk_t = ix_gb * 4 + _min_0;
                        int _max_3 = ((sparse_topk_lens[bk_t] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[bk_t] + sparse_topk_lens_offset) : (0));
                        int bk_len = ((_max_3) < (swa_width) ? (_max_3) : (swa_width));
                        int bk_row[2];
                        {
                            const int2* _ivptr_4 = reinterpret_cast<const int2*>(swa_indices + bk_t * swa_index_stride + 0);
                            int2 _ivld_4;
                            _ivld_4 = *_ivptr_4;
                            bk_row[0 + 0] = _ivld_4.x;
                            bk_row[0 + 1] = _ivld_4.y;
                        }
                        int bk_plen = __shfl_up_sync(0xFFFFFFFF, bk_len, 1, 32);
                        int bk_p0 = __shfl_up_sync(0xFFFFFFFF, bk_row[0], 1, 32);
                        int bk_p1 = __shfl_up_sync(0xFFFFFFFF, bk_row[1], 1, 32);
                        int _min_6 = ((ix_b1 - ix_gb) < (8) ? (ix_b1 - ix_gb) : (8));
                        int _min_5 = ((_min_6 * 4) < (num_query_tokens - ix_gb * 4) ? (_min_6 * 4) : (num_query_tokens - ix_gb * 4));
                        int bk_brk = (lane & 3) != 0 && lane < _min_5 && (int)(bk_len == 128 && bk_plen == 128 && bk_row[0] == bk_p1 || bk_plen < 128 && bk_row[0] == bk_p0 || (bk_plen < 128 || bk_len == 128) && bk_row[1] == bk_p0) == 0;
                        unsigned int ix_brk = __ballot_sync(0xFFFFFFFF, bk_brk != 0);
                        if (ix_brk != 0) {
                            int _find_msb_7;
                            asm volatile("bfind.u32 %0, %1;" : "=r"(_find_msb_7) : "r"(ix_brk & (ix_brk ^ 4294967295u) + 1));
                            int ix_j = (int)_find_msb_7;
                            ix_t0 = ix_gb * 4 + ix_j;
                            int _min_8 = ((num_query_tokens - (ix_gb * 4 + (ix_j & -4))) < (4) ? (num_query_tokens - (ix_gb * 4 + (ix_j & -4))) : (4));
                            ix_n = _min_8 - (ix_j & 3);
                            ix_gb = ix_gb + (ix_j >> 2) + 1;
                            ix_more = 1;
                            break;
                        }
                        ix_gb = ix_gb + 8;
                    }
                }
                ix_fresh = 0;
                int ix_next = -1;
                #pragma unroll 1
                for (int _ix_pub = 0; _ix_pub < (int)(ix_n > 0); _ix_pub++) {
                    int ix_next_0 = -1;
                    int ix_next_1 = -1;
                    int ix_len[4];
                    int ix_rows[16];
                    int _min_11 = ((ix_t0) < (num_query_tokens - 1) ? (ix_t0) : (num_query_tokens - 1));
                    int _max_10 = ((sparse_topk_lens[_min_11] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[_min_11] + sparse_topk_lens_offset) : (0));
                    int _min_9 = ((_max_10) < (swa_width) ? (_max_10) : (swa_width));
                    ix_len[0] = _min_9;
                    int _min_12 = ((ix_t0) < (num_query_tokens - 1) ? (ix_t0) : (num_query_tokens - 1));
                    {
                        const int4* _ivptr_13 = reinterpret_cast<const int4*>(swa_indices + (_min_12 * swa_index_stride + lane * 4) + 0);
                        int4 _ivld_13;
                        _ivld_13 = *_ivptr_13;
                        ix_rows[0 + 0] = _ivld_13.x;
                        ix_rows[0 + 1] = _ivld_13.y;
                        ix_rows[0 + 2] = _ivld_13.z;
                        ix_rows[0 + 3] = _ivld_13.w;
                    }
                    int _min_16 = ((ix_t0 + 1) < (num_query_tokens - 1) ? (ix_t0 + 1) : (num_query_tokens - 1));
                    int _max_15 = ((sparse_topk_lens[_min_16] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[_min_16] + sparse_topk_lens_offset) : (0));
                    int _min_14 = ((_max_15) < (swa_width) ? (_max_15) : (swa_width));
                    ix_len[1] = _min_14;
                    int _min_17 = ((ix_t0 + 1) < (num_query_tokens - 1) ? (ix_t0 + 1) : (num_query_tokens - 1));
                    {
                        const int4* _ivptr_18 = reinterpret_cast<const int4*>(swa_indices + (_min_17 * swa_index_stride + lane * 4) + 0);
                        int4 _ivld_18;
                        _ivld_18 = *_ivptr_18;
                        ix_rows[4 + 0] = _ivld_18.x;
                        ix_rows[4 + 1] = _ivld_18.y;
                        ix_rows[4 + 2] = _ivld_18.z;
                        ix_rows[4 + 3] = _ivld_18.w;
                    }
                    int _min_21 = ((ix_t0 + 2) < (num_query_tokens - 1) ? (ix_t0 + 2) : (num_query_tokens - 1));
                    int _max_20 = ((sparse_topk_lens[_min_21] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[_min_21] + sparse_topk_lens_offset) : (0));
                    int _min_19 = ((_max_20) < (swa_width) ? (_max_20) : (swa_width));
                    ix_len[2] = _min_19;
                    int _min_22 = ((ix_t0 + 2) < (num_query_tokens - 1) ? (ix_t0 + 2) : (num_query_tokens - 1));
                    {
                        const int4* _ivptr_23 = reinterpret_cast<const int4*>(swa_indices + (_min_22 * swa_index_stride + lane * 4) + 0);
                        int4 _ivld_23;
                        _ivld_23 = *_ivptr_23;
                        ix_rows[8 + 0] = _ivld_23.x;
                        ix_rows[8 + 1] = _ivld_23.y;
                        ix_rows[8 + 2] = _ivld_23.z;
                        ix_rows[8 + 3] = _ivld_23.w;
                    }
                    int _min_26 = ((ix_t0 + 3) < (num_query_tokens - 1) ? (ix_t0 + 3) : (num_query_tokens - 1));
                    int _max_25 = ((sparse_topk_lens[_min_26] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[_min_26] + sparse_topk_lens_offset) : (0));
                    int _min_24 = ((_max_25) < (swa_width) ? (_max_25) : (swa_width));
                    ix_len[3] = _min_24;
                    int _min_27 = ((ix_t0 + 3) < (num_query_tokens - 1) ? (ix_t0 + 3) : (num_query_tokens - 1));
                    {
                        const int4* _ivptr_28 = reinterpret_cast<const int4*>(swa_indices + (_min_27 * swa_index_stride + lane * 4) + 0);
                        int4 _ivld_28;
                        _ivld_28 = *_ivptr_28;
                        ix_rows[12 + 0] = _ivld_28.x;
                        ix_rows[12 + 1] = _ivld_28.y;
                        ix_rows[12 + 2] = _ivld_28.z;
                        ix_rows[12 + 3] = _ivld_28.w;
                    }
                    int ix_nb = ix_n;
                    int ix_cut_at = ix_n;
                    if ((ix_block & (int)((int)gridDim.x - (num_query_tokens + 3 >> 2) > 0)) != 0) {
                        if (3 < ix_n) {
                            if ((int)(ix_len[3] == 128 && ix_len[2] == 128 && ix_rows[12] == ix_rows[9] || ix_len[2] < 128 && ix_rows[12] == ix_rows[8] || (ix_len[2] < 128 || ix_len[3] == 128) && ix_rows[13] == ix_rows[8]) == 0) {
                                ix_cut_at = 3;
                            }
                        }
                        if (2 < ix_n) {
                            if ((int)(ix_len[2] == 128 && ix_len[1] == 128 && ix_rows[8] == ix_rows[5] || ix_len[1] < 128 && ix_rows[8] == ix_rows[4] || (ix_len[1] < 128 || ix_len[2] == 128) && ix_rows[9] == ix_rows[4]) == 0) {
                                ix_cut_at = 2;
                            }
                        }
                        if (1 < ix_n) {
                            if ((int)(ix_len[1] == 128 && ix_len[0] == 128 && ix_rows[4] == ix_rows[1] || ix_len[0] < 128 && ix_rows[4] == ix_rows[0] || (ix_len[0] < 128 || ix_len[1] == 128) && ix_rows[5] == ix_rows[0]) == 0) {
                                ix_cut_at = 1;
                            }
                        }
                    }
                    int _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, ix_cut_at, 0);
                    ix_nb = _shfl_sync_29;
                    if (0 >= ix_nb) {
                        ix_len[0] = 0;
                    }
                    if (1 >= ix_nb) {
                        ix_len[1] = 0;
                    }
                    if (2 >= ix_nb) {
                        ix_len[2] = 0;
                    }
                    if (3 >= ix_nb) {
                        ix_len[3] = 0;
                    }
                    int ix_nf = __shfl_sync(0xFFFFFFFF, (int)(ix_rows[5] == ix_rows[0] && ix_rows[4] != ix_rows[1]), 0);
                    if (ix_nf == 0) {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        unsigned int ix_stage_base = smem_indices_addr + ix_stage * 2112;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(lane * 16)), "r"(ix_rows[0]), "r"(ix_rows[1]), "r"(ix_rows[2]), "r"(ix_rows[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(512 + lane * 16)), "r"(-1), "r"(-1), "r"(-1), "r"(-1) : "memory");
                        __syncwarp();
                        if (lane == 31) {
                            smem_indices[(int)ix_stage * 528 + 127 + 1] = ((1 < ix_nb) ? ix_rows[7] : -1);
                            smem_indices[(int)ix_stage * 528 + 127 + 2] = ((2 < ix_nb) ? ix_rows[11] : -1);
                            smem_indices[(int)ix_stage * 528 + 127 + 3] = ((3 < ix_nb) ? ix_rows[15] : -1);
                        }
                        __syncwarp();
                        int ix_bad = ix_len[0] != 128;
                        int ix_bad0 = __any_sync(0xFFFFFFFF, ix_bad != 0);
                        int ix_shared = ((ix_bad0 != 0) ? 1 : ix_nb);
                        int ix_bad_i_0 = ix_len[3] != 128;
                        int ix_want = smem_indices[(int)ix_stage * 528 + 3 + lane * 4];
                        if (!(ix_rows[12] == ix_want)) {
                            ix_bad_i_0 = 1;
                        }
                        int ix_want_0 = smem_indices[(int)ix_stage * 528 + 3 + lane * 4 + 1];
                        if (!(ix_rows[13] == ix_want_0)) {
                            ix_bad_i_0 = 1;
                        }
                        int ix_want_1 = smem_indices[(int)ix_stage * 528 + 3 + lane * 4 + 2];
                        if (!(ix_rows[14] == ix_want_1)) {
                            ix_bad_i_0 = 1;
                        }
                        int ix_want_2 = smem_indices[(int)ix_stage * 528 + 3 + lane * 4 + 3];
                        if (!(ix_rows[15] == ix_want_2)) {
                            ix_bad_i_0 = 1;
                        }
                        int ix_vote_i_0 = __any_sync(0xFFFFFFFF, ix_bad_i_0 != 0);
                        if (3 < ix_shared && ix_vote_i_0 != 0) {
                            ix_shared = 3;
                        }
                        int ix_bad_i_1 = ix_len[2] != 128;
                        int ix_want_3 = smem_indices[(int)ix_stage * 528 + 2 + lane * 4];
                        if (!(ix_rows[8] == ix_want_3)) {
                            ix_bad_i_1 = 1;
                        }
                        int ix_want_4 = smem_indices[(int)ix_stage * 528 + 2 + lane * 4 + 1];
                        if (!(ix_rows[9] == ix_want_4)) {
                            ix_bad_i_1 = 1;
                        }
                        int ix_want_5 = smem_indices[(int)ix_stage * 528 + 2 + lane * 4 + 2];
                        if (!(ix_rows[10] == ix_want_5)) {
                            ix_bad_i_1 = 1;
                        }
                        int ix_want_6 = smem_indices[(int)ix_stage * 528 + 2 + lane * 4 + 3];
                        if (!(ix_rows[11] == ix_want_6)) {
                            ix_bad_i_1 = 1;
                        }
                        int ix_vote_i_1 = __any_sync(0xFFFFFFFF, ix_bad_i_1 != 0);
                        if (2 < ix_shared && ix_vote_i_1 != 0) {
                            ix_shared = 2;
                        }
                        int ix_bad_i_2 = ix_len[1] != 128;
                        int ix_want_7 = smem_indices[(int)ix_stage * 528 + 1 + lane * 4];
                        if (!(ix_rows[4] == ix_want_7)) {
                            ix_bad_i_2 = 1;
                        }
                        int ix_want_8 = smem_indices[(int)ix_stage * 528 + 1 + lane * 4 + 1];
                        if (!(ix_rows[5] == ix_want_8)) {
                            ix_bad_i_2 = 1;
                        }
                        int ix_want_9 = smem_indices[(int)ix_stage * 528 + 1 + lane * 4 + 2];
                        if (!(ix_rows[6] == ix_want_9)) {
                            ix_bad_i_2 = 1;
                        }
                        int ix_want_10 = smem_indices[(int)ix_stage * 528 + 1 + lane * 4 + 3];
                        if (!(ix_rows[7] == ix_want_10)) {
                            ix_bad_i_2 = 1;
                        }
                        int ix_vote_i_2 = __any_sync(0xFFFFFFFF, ix_bad_i_2 != 0);
                        if (1 < ix_shared && ix_vote_i_2 != 0) {
                            ix_shared = 1;
                        }
                        int ix_off_stride = ((ix_shared > 1) ? 128 : 0);
                        int ix_off_base = ((ix_shared > 1) ? 192 - 128 * ix_shared : 0);
                        if (ix_shared < ix_nb) {
                            __syncwarp();
                            if (ix_shared > 1 && lane == 31) {
                                if (1 >= ix_shared) {
                                    smem_indices[(int)ix_stage * 528 + 127 + 1] = -1;
                                }
                                if (2 >= ix_shared) {
                                    smem_indices[(int)ix_stage * 528 + 127 + 2] = -1;
                                }
                                if (3 >= ix_shared) {
                                    smem_indices[(int)ix_stage * 528 + 127 + 3] = -1;
                                }
                            }
                            if (0 >= ix_shared || ix_shared == 1) {
                                int ix_own[4];
                                #pragma unroll
                                for (int ix_j_1 = 0; ix_j_1 < 4; ix_j_1++) {
                                    ix_own[ix_j_1] = ((lane * 4 + ix_j_1 < ix_len[0]) ? ix_rows[ix_j_1] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(4 * ((ix_shared > 1) ? ix_off_base : 0) + lane * 16)), "r"(ix_own[0]), "r"(ix_own[1]), "r"(ix_own[2]), "r"(ix_own[3]) : "memory");
                            }
                            if (1 >= ix_shared || ix_shared == 1) {
                                int ix_own_1[4];
                                #pragma unroll
                                for (int ix_j_2 = 0; ix_j_2 < 4; ix_j_2++) {
                                    ix_own_1[ix_j_2] = ((lane * 4 + ix_j_2 < ix_len[1]) ? ix_rows[4 + ix_j_2] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(4 * ((ix_shared > 1) ? ix_off_base + ix_off_stride : 128) + lane * 16)), "r"(ix_own_1[0]), "r"(ix_own_1[1]), "r"(ix_own_1[2]), "r"(ix_own_1[3]) : "memory");
                            }
                            if (2 >= ix_shared || ix_shared == 1) {
                                int ix_own_2[4];
                                #pragma unroll
                                for (int ix_j_3 = 0; ix_j_3 < 4; ix_j_3++) {
                                    ix_own_2[ix_j_3] = ((lane * 4 + ix_j_3 < ix_len[2]) ? ix_rows[8 + ix_j_3] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(4 * ((ix_shared > 1) ? ix_off_base + 2 * ix_off_stride : 256) + lane * 16)), "r"(ix_own_2[0]), "r"(ix_own_2[1]), "r"(ix_own_2[2]), "r"(ix_own_2[3]) : "memory");
                            }
                            if (3 >= ix_shared || ix_shared == 1) {
                                int ix_own_3[4];
                                #pragma unroll
                                for (int ix_j_4 = 0; ix_j_4 < 4; ix_j_4++) {
                                    ix_own_3[ix_j_4] = ((lane * 4 + ix_j_4 < ix_len[3]) ? ix_rows[12 + ix_j_4] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(4 * ((ix_shared > 1) ? ix_off_base + 3 * ix_off_stride : 384) + lane * 16)), "r"(ix_own_3[0]), "r"(ix_own_3[1]), "r"(ix_own_3[2]), "r"(ix_own_3[3]) : "memory");
                            }
                        }
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 528 + 512] = ix_shared;
                            smem_indices[(int)ix_stage * 528 + 512 + 15] = ix_query;
                            smem_indices[(int)ix_stage * 528 + 512 + 12] = ix_t0;
                            smem_indices[(int)ix_stage * 528 + 512 + 13] = ix_nb;
                            int ix_lo_0 = ((ix_shared > 1) ? ((0 < ix_shared) ? 0 : ix_off_base) : 0);
                            int ix_span_0 = ix_len[0];
                            smem_indices[(int)ix_stage * 528 + 512 + 2] = ix_lo_0;
                            smem_indices[(int)ix_stage * 528 + 512 + 3] = ix_span_0;
                            int ix_lo_1 = ((ix_shared > 1) ? ((1 < ix_shared) ? 1 : ix_off_base + ix_off_stride) : 128);
                            int ix_span_1 = ix_len[1];
                            smem_indices[(int)ix_stage * 528 + 512 + 2 + 2] = ix_lo_1;
                            smem_indices[(int)ix_stage * 528 + 512 + 3 + 2] = ix_span_1;
                            int ix_lo_2 = ((ix_shared > 1) ? ((2 < ix_shared) ? 2 : ix_off_base + 2 * ix_off_stride) : 256);
                            int ix_span_2 = ix_len[2];
                            smem_indices[(int)ix_stage * 528 + 512 + 2 + 4] = ix_lo_2;
                            smem_indices[(int)ix_stage * 528 + 512 + 3 + 4] = ix_span_2;
                            int ix_lo_3 = ((ix_shared > 1) ? ((3 < ix_shared) ? 3 : ix_off_base + 3 * ix_off_stride) : 384);
                            int ix_span_3 = ix_len[3];
                            smem_indices[(int)ix_stage * 528 + 512 + 2 + 6] = ix_lo_3;
                            smem_indices[(int)ix_stage * 528 + 512 + 3 + 6] = ix_span_3;
                        }
                    } else {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        unsigned int ix_stage_base_1 = smem_indices_addr + ix_stage * 2112;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(lane * 16)), "r"(ix_rows[0]), "r"(ix_rows[1]), "r"(ix_rows[2]), "r"(ix_rows[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(512 + lane * 16)), "r"(-1), "r"(-1), "r"(-1), "r"(-1) : "memory");
                        __syncwarp();
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 528 + 127 + 1] = ((1 < ix_nb) ? ix_rows[4] : -1);
                            smem_indices[(int)ix_stage * 528 + 127 + 2] = ((2 < ix_nb) ? ix_rows[8] : -1);
                            smem_indices[(int)ix_stage * 528 + 127 + 3] = ((3 < ix_nb) ? ix_rows[12] : -1);
                        }
                        __syncwarp();
                        int ix_bad_1 = ix_len[0] != 128;
                        int ix_bad0_1 = __any_sync(0xFFFFFFFF, ix_bad_1 != 0);
                        int ix_shared_1 = ((ix_bad0_1 != 0) ? 1 : ix_nb);
                        int ix_bad_i_3 = ix_len[3] != 128;
                        int _max_30 = ((lane * 4 + -3) > (0) ? (lane * 4 + -3) : (0));
                        int ix_want_11 = smem_indices[(int)ix_stage * 528 + _max_30];
                        if (lane == 0) {
                            ix_want_11 = ix_rows[12];
                        }
                        if (!(ix_rows[12] == ix_want_11)) {
                            ix_bad_i_3 = 1;
                        }
                        int _max_31 = ((lane * 4 + -2) > (0) ? (lane * 4 + -2) : (0));
                        int ix_want_11_1 = smem_indices[(int)ix_stage * 528 + _max_31];
                        if (lane == 0) {
                            ix_want_11_1 = ix_rows[8];
                        }
                        if (!(ix_rows[13] == ix_want_11_1)) {
                            ix_bad_i_3 = 1;
                        }
                        int _max_32 = ((lane * 4 + -1) > (0) ? (lane * 4 + -1) : (0));
                        int ix_want_12 = smem_indices[(int)ix_stage * 528 + _max_32];
                        if (lane == 0) {
                            ix_want_12 = ix_rows[4];
                        }
                        if (!(ix_rows[14] == ix_want_12)) {
                            ix_bad_i_3 = 1;
                        }
                        int ix_want_13 = smem_indices[(int)ix_stage * 528 + -3 + lane * 4 + 3];
                        if (!(ix_rows[15] == ix_want_13)) {
                            ix_bad_i_3 = 1;
                        }
                        int ix_vote_i_3 = __any_sync(0xFFFFFFFF, ix_bad_i_3 != 0);
                        if (3 < ix_shared_1 && ix_vote_i_3 != 0) {
                            ix_shared_1 = 3;
                        }
                        int ix_bad_i_4 = ix_len[2] != 128;
                        int _max_33 = ((lane * 4 + -2) > (0) ? (lane * 4 + -2) : (0));
                        int ix_want_14 = smem_indices[(int)ix_stage * 528 + _max_33];
                        if (lane == 0) {
                            ix_want_14 = ix_rows[8];
                        }
                        if (!(ix_rows[8] == ix_want_14)) {
                            ix_bad_i_4 = 1;
                        }
                        int _max_34 = ((lane * 4 + -1) > (0) ? (lane * 4 + -1) : (0));
                        int ix_want_15 = smem_indices[(int)ix_stage * 528 + _max_34];
                        if (lane == 0) {
                            ix_want_15 = ix_rows[4];
                        }
                        if (!(ix_rows[9] == ix_want_15)) {
                            ix_bad_i_4 = 1;
                        }
                        int ix_want_16 = smem_indices[(int)ix_stage * 528 + -2 + lane * 4 + 2];
                        if (!(ix_rows[10] == ix_want_16)) {
                            ix_bad_i_4 = 1;
                        }
                        int ix_want_17 = smem_indices[(int)ix_stage * 528 + -2 + lane * 4 + 3];
                        if (!(ix_rows[11] == ix_want_17)) {
                            ix_bad_i_4 = 1;
                        }
                        int ix_vote_i_4 = __any_sync(0xFFFFFFFF, ix_bad_i_4 != 0);
                        if (2 < ix_shared_1 && ix_vote_i_4 != 0) {
                            ix_shared_1 = 2;
                        }
                        int ix_bad_i_5 = ix_len[1] != 128;
                        int _max_35 = ((lane * 4 + -1) > (0) ? (lane * 4 + -1) : (0));
                        int ix_want_18 = smem_indices[(int)ix_stage * 528 + _max_35];
                        if (lane == 0) {
                            ix_want_18 = ix_rows[4];
                        }
                        if (!(ix_rows[4] == ix_want_18)) {
                            ix_bad_i_5 = 1;
                        }
                        int ix_want_19 = smem_indices[(int)ix_stage * 528 + -1 + lane * 4 + 1];
                        if (!(ix_rows[5] == ix_want_19)) {
                            ix_bad_i_5 = 1;
                        }
                        int ix_want_20 = smem_indices[(int)ix_stage * 528 + -1 + lane * 4 + 2];
                        if (!(ix_rows[6] == ix_want_20)) {
                            ix_bad_i_5 = 1;
                        }
                        int ix_want_21 = smem_indices[(int)ix_stage * 528 + -1 + lane * 4 + 3];
                        if (!(ix_rows[7] == ix_want_21)) {
                            ix_bad_i_5 = 1;
                        }
                        int ix_vote_i_5 = __any_sync(0xFFFFFFFF, ix_bad_i_5 != 0);
                        if (1 < ix_shared_1 && ix_vote_i_5 != 0) {
                            ix_shared_1 = 1;
                        }
                        int ix_off_stride_1 = ((ix_shared_1 > 1) ? 128 : 0);
                        int ix_off_base_1 = ((ix_shared_1 > 1) ? 192 - 128 * ix_shared_1 : 0);
                        if (ix_shared_1 < ix_nb) {
                            __syncwarp();
                            if (ix_shared_1 > 1 && lane == 0) {
                                if (1 >= ix_shared_1) {
                                    smem_indices[(int)ix_stage * 528 + 127 + 1] = -1;
                                }
                                if (2 >= ix_shared_1) {
                                    smem_indices[(int)ix_stage * 528 + 127 + 2] = -1;
                                }
                                if (3 >= ix_shared_1) {
                                    smem_indices[(int)ix_stage * 528 + 127 + 3] = -1;
                                }
                            }
                            if (0 >= ix_shared_1 || ix_shared_1 == 1) {
                                int ix_own_4[4];
                                #pragma unroll
                                for (int ix_j_5 = 0; ix_j_5 < 4; ix_j_5++) {
                                    ix_own_4[ix_j_5] = ((lane * 4 + ix_j_5 < ix_len[0]) ? ix_rows[ix_j_5] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(4 * ((ix_shared_1 > 1) ? ix_off_base_1 : 0) + lane * 16)), "r"(ix_own_4[0]), "r"(ix_own_4[1]), "r"(ix_own_4[2]), "r"(ix_own_4[3]) : "memory");
                            }
                            if (1 >= ix_shared_1 || ix_shared_1 == 1) {
                                int ix_own_5[4];
                                #pragma unroll
                                for (int ix_j_6 = 0; ix_j_6 < 4; ix_j_6++) {
                                    ix_own_5[ix_j_6] = ((lane * 4 + ix_j_6 < ix_len[1]) ? ix_rows[4 + ix_j_6] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(4 * ((ix_shared_1 > 1) ? ix_off_base_1 + ix_off_stride_1 : 128) + lane * 16)), "r"(ix_own_5[0]), "r"(ix_own_5[1]), "r"(ix_own_5[2]), "r"(ix_own_5[3]) : "memory");
                            }
                            if (2 >= ix_shared_1 || ix_shared_1 == 1) {
                                int ix_own_6[4];
                                #pragma unroll
                                for (int ix_j_7 = 0; ix_j_7 < 4; ix_j_7++) {
                                    ix_own_6[ix_j_7] = ((lane * 4 + ix_j_7 < ix_len[2]) ? ix_rows[8 + ix_j_7] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(4 * ((ix_shared_1 > 1) ? ix_off_base_1 + 2 * ix_off_stride_1 : 256) + lane * 16)), "r"(ix_own_6[0]), "r"(ix_own_6[1]), "r"(ix_own_6[2]), "r"(ix_own_6[3]) : "memory");
                            }
                            if (3 >= ix_shared_1 || ix_shared_1 == 1) {
                                int ix_own_7[4];
                                #pragma unroll
                                for (int ix_j_8 = 0; ix_j_8 < 4; ix_j_8++) {
                                    ix_own_7[ix_j_8] = ((lane * 4 + ix_j_8 < ix_len[3]) ? ix_rows[12 + ix_j_8] : -1);
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(4 * ((ix_shared_1 > 1) ? ix_off_base_1 + 3 * ix_off_stride_1 : 384) + lane * 16)), "r"(ix_own_7[0]), "r"(ix_own_7[1]), "r"(ix_own_7[2]), "r"(ix_own_7[3]) : "memory");
                            }
                        }
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 528 + 512] = ix_shared_1;
                            smem_indices[(int)ix_stage * 528 + 512 + 15] = ix_query;
                            smem_indices[(int)ix_stage * 528 + 512 + 12] = ix_t0;
                            smem_indices[(int)ix_stage * 528 + 512 + 13] = ix_nb;
                            int ix_lo_4 = ((ix_shared_1 > 1) ? ((0 < ix_shared_1) ? 0 : ix_off_base_1) : 0);
                            int ix_span_4 = ix_len[0];
                            smem_indices[(int)ix_stage * 528 + 512 + 2] = ix_lo_4;
                            smem_indices[(int)ix_stage * 528 + 512 + 3] = ix_span_4;
                            int ix_lo_5 = ((ix_shared_1 > 1) ? ((1 < ix_shared_1) ? 1 : ix_off_base_1 + ix_off_stride_1) : 128);
                            int ix_span_5 = ix_len[1];
                            if (1 < ix_shared_1 && ix_shared_1 > 1) {
                                ix_lo_5 = 0;
                                ix_span_5 = 131199;
                            }
                            smem_indices[(int)ix_stage * 528 + 512 + 2 + 2] = ix_lo_5;
                            smem_indices[(int)ix_stage * 528 + 512 + 3 + 2] = ix_span_5;
                            int ix_lo_6 = ((ix_shared_1 > 1) ? ((2 < ix_shared_1) ? 2 : ix_off_base_1 + 2 * ix_off_stride_1) : 256);
                            int ix_span_6 = ix_len[2];
                            if (2 < ix_shared_1 && ix_shared_1 > 1) {
                                ix_lo_6 = 0;
                                ix_span_6 = 262270;
                            }
                            smem_indices[(int)ix_stage * 528 + 512 + 2 + 4] = ix_lo_6;
                            smem_indices[(int)ix_stage * 528 + 512 + 3 + 4] = ix_span_6;
                            int ix_lo_7 = ((ix_shared_1 > 1) ? ((3 < ix_shared_1) ? 3 : ix_off_base_1 + 3 * ix_off_stride_1) : 384);
                            int ix_span_7 = ix_len[3];
                            if (3 < ix_shared_1 && ix_shared_1 > 1) {
                                ix_lo_7 = 0;
                                ix_span_7 = 393341;
                            }
                            smem_indices[(int)ix_stage * 528 + 512 + 2 + 6] = ix_lo_7;
                            smem_indices[(int)ix_stage * 528 + 512 + 3 + 6] = ix_span_7;
                        }
                    }
                    mbarrier_arrive(index_full_addr + (ix_stage) * 8);
                    ix_stage += 1;
                    if (ix_stage == 2) { ix_stage = 0; _phase_index_empty ^= 1; }
                    ix_next_0 = ix_next_1;
                    ix_next = ix_next_0;
                    if (ix_block == 0) {
                        ix_open = 1;
                    }
                }
                if (ix_more == 0) {
                    ix_fresh = 1;
                    mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                    mbarrier_wait(work_empty_addr + (ix_work_stage) * 8, _phase_work_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(work_full_addr + (ix_work_stage) * 8, 16);
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + ix_work_stage * 16 + 0 * 16), "r"(work_full_addr + ix_work_stage * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (ix_work_stage) * 8, _phase_work_full);
                    uint32_t ixn_valid = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(ixn_valid)
                        : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t ixn_ctaid = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(ixn_ctaid)
                        : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (ix_work_stage) * 8);
                    ix_work_stage += 1;
                    if (ix_work_stage == 2) { ix_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    ix_query = ((ixn_valid != 0) ? (int)ixn_ctaid : -1);
                    if (ix_query < 0) {
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 528 + 512 + 15] = -1;
                        }
                        mbarrier_arrive(index_full_addr + (ix_stage) * 8);
                        ix_stage += 1;
                        if (ix_stage == 2) { ix_stage = 0; _phase_index_empty ^= 1; }
                        break;
                    }
                }
            }
        }
    // ---- Role: pv_issuer ----
    } else if (warp == 11) {
        { // pv_issuer_main
            int pv_tile_cursor = 0;
            unsigned int pv_info_stage = 0;
            int q_3 = blockIdx.x;
            unsigned int _phase_index_full_3 = 0;
            if (blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                mbarrier_wait(index_full_addr + (pv_info_stage) * 8, _phase_index_full_3);
                q_3 = smem_indices[(int)pv_info_stage * 528 + 512 + 15];
            }
            int pv_query = q_3;
            #pragma unroll 1
            for (unsigned int pv_work = 0; pv_work < 1048576; pv_work++) {
                int pv_ret = -1;
                if (pv_query >= 0) {
                    mbarrier_wait(index_full_addr + (pv_info_stage) * 8, _phase_index_full_3);
                    int shared_1 = smem_indices[(int)pv_info_stage * 528 + 512];
                    int nb_read_1 = smem_indices[(int)pv_info_stage * 528 + 512 + 13];
                    int nxt_1 = 0;
                    mbarrier_arrive(index_empty_addr + (pv_info_stage) * 8);
                    pv_info_stage += 1;
                    if (pv_info_stage == 2) { pv_info_stage = 0; _phase_index_full_3 ^= 1; }
                    int nb_3 = nb_read_1;
                    int rest_3 = shared_1 > 1;
                    int tiles_3 = ((shared_1 > 1) ? 3 + 2 * (nb_3 - shared_1) : 2 * nb_3);
                    pv_ret = nxt_1;
                    int pv_first = 1;
                    #pragma unroll 1
                    for (int pv_tile = 0; pv_tile < tiles_3; pv_tile++) {
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
                        int pv_narrow = rest_3 != 0 && pv_tile == 2;
                        if (pv_narrow != 0) {
                            int pv_a_lo = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                            int pv_b_lo = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
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
                    "}\n"
                    :: "r"(pv_a_lo), "r"(pv_b_lo), "r"(tmem_tmem_scratch), "r"(((pv_first) ? 0 : 1)) : "memory");
                            elect_commit(o_full_addr);
                        } else {
                            int pv_a_lo_1 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                            int pv_b_lo_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
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
                    :: "r"(pv_a_lo_1), "r"(pv_b_lo_1), "r"(tmem_tmem_scratch), "r"(((pv_first) ? 0 : 1)) : "memory");
                            elect_commit(o_full_addr);
                        }
                        int _parity_o_empty_1 = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                        mbarrier_wait(o_empty_addr + 8, _parity_o_empty_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (pv_narrow != 0) {
                            int pv_a_lo_2 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                            int pv_b_lo_2 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
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
                    "}\n"
                    :: "r"(pv_a_lo_2), "r"(pv_b_lo_2), "r"((tmem_tmem_scratch + (128))), "r"(((pv_first) ? 0 : 1)) : "memory");
                            elect_commit(o_full_addr + 8);
                        } else {
                            int pv_a_lo_3 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pv_stage) * 256);
                            int pv_b_lo_3 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (pv_stage) * 2048);
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
                    :: "r"(pv_a_lo_3), "r"(pv_b_lo_3), "r"((tmem_tmem_scratch + (128))), "r"(((pv_first) ? 0 : 1)) : "memory");
                            elect_commit(o_full_addr + 8);
                        }
                        elect_commit(v_empty_addr + ((int)((unsigned int)(pv_tile_cursor + pv_tile) % 2)) * 8);
                        elect_commit(p_recycle_addr + ((int)((unsigned int)(pv_tile_cursor + pv_tile) % 2)) * 8);
                        pv_first = 0;
                    }
                    pv_tile_cursor += tiles_3;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_wait(index_full_addr + (pv_info_stage) * 8, _phase_index_full_3);
                pv_query = smem_indices[(int)pv_info_stage * 528 + 512 + 15];
                if (pv_query < 0) {
                    break;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: load_warp ----
    } else if (warp >= 12 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_warp_main
            const int x_grp = warp - 12 >> 2;
            const int ld_warp = warp - 12 & 3;
            const int ld_thread = ld_warp * 32 + lane;
            const int ld_row_base = ld_warp * 32;
            const int v_key = ld_thread & 63;
            const int v_half = ld_thread >> 6;
            const int x_u_lo = x_grp * (3 - v_half);
            const int x_u_hi = 3 - v_half + x_grp * (5 - v_half);
            const int x_r_lo = 2 * x_grp;
            int ld_tile_cursor = 0;
            unsigned int ld_info_stage = 0;
            int q_4 = blockIdx.x;
            unsigned int _phase_index_full_4 = 0;
            if (blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                mbarrier_wait(index_full_addr + (ld_info_stage) * 8, _phase_index_full_4);
                q_4 = smem_indices[(int)ld_info_stage * 528 + 512 + 15];
            }
            int ld_query = q_4;
            #pragma unroll 1
            for (unsigned int ld_work = 0; ld_work < 1048576; ld_work++) {
                int ld_next = -1;
                if (ld_query >= 0) {
                    mbarrier_wait(index_full_addr + (ld_info_stage) * 8, _phase_index_full_4);
                    int shared_2 = smem_indices[(int)ld_info_stage * 528 + 512];
                    int nb_read_2 = smem_indices[(int)ld_info_stage * 528 + 512 + 13];
                    int nxt_2 = 0;
                    mbarrier_arrive(index_empty_addr + (ld_info_stage) * 8);
                    ld_info_stage += 1;
                    if (ld_info_stage == 2) { ld_info_stage = 0; _phase_index_full_4 ^= 1; }
                    int nb_4 = nb_read_2;
                    int rest_4 = shared_2 > 1;
                    int tiles_4 = ((shared_2 > 1) ? 3 + 2 * (nb_4 - shared_2) : 2 * nb_4);
                    ld_next = nxt_2;
                    #pragma unroll 1
                    for (int ld_tile = 0; ld_tile < tiles_4; ld_tile++) {
                        int p_stage = ld_tile_cursor + ld_tile & 1;
                        int k_stage = (ld_tile_cursor + ld_tile) % 3;
                        int _stage_k_full = (int)((unsigned int)(ld_tile_cursor + ld_tile) % 3);
                        int _parity_k_full = (int)((unsigned int)(ld_tile_cursor + ld_tile) / 3 % 2);
                        mbarrier_wait(k_full_addr + (_stage_k_full) * 8, _parity_k_full);
                        int sf_base = smem_k_sf_addr + (unsigned int)(k_stage * 26624);
                        unsigned int kexp_u = 0;
                        if (x_grp == 0) {
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
                            int ld_rows = ((rest_4 != 0 && ld_tile == 2) ? 4 : 64);
                            #pragma unroll
                            for (int sf_s = 0; sf_s < 7; sf_s++) {
                                sf_a[sf_s] = ((lane < ld_rows) ? sf_a[sf_s] : 0);
                                sf_b[sf_s] = ((lane + 32 < ld_rows) ? sf_b[sf_s] : 0);
                            }
                            int _stage_k_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile + 1) % 3);
                            int _parity_k_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile + 1) / 3 % 2) ^ 1;
                            mbarrier_wait(k_empty_addr + (_stage_k_empty) * 8, _parity_k_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int sfb_regs[16];
                            #pragma unroll
                            for (int sf_s_1 = 0; sf_s_1 < 7; sf_s_1++) {
                                sfb_regs[2 * sf_s_1] = sf_a[sf_s_1];
                                sfb_regs[2 * sf_s_1 + 1] = sf_b[sf_s_1];
                            }
                            sfb_regs[14] = 0;
                            sfb_regs[15] = 0;
                            int sfb_col = 416 + p_stage * 16;
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr + (unsigned int)sfb_col + (unsigned int)(ld_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[15])));
                            unsigned int k68 = 0;
                            unsigned int k70 = 0;
                            unsigned int k78 = 0;
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 7, 128;" ::: "memory");
                            if (ld_warp == 0) {
                                if (elect_sync()) {
                                    mbarrier_arrive(sf_full_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 3)) * 8);
                                }
                            }
                            #pragma unroll
                            for (int sf_s_2 = 0; sf_s_2 < 7; sf_s_2++) {
                                k68 = k68 | sf_a[sf_s_2] + 404232216 | sf_b[sf_s_2] + 404232216;
                                k70 = k70 | sf_a[sf_s_2] + 269488144 | sf_b[sf_s_2] + 269488144;
                                k78 = k78 | sf_a[sf_s_2] + 134744072 | sf_b[sf_s_2] + 134744072;
                            }
                            int _warp_vote_0 = __any_sync(0xFFFFFFFF, (k68 & 2155905152u) != 0);
                            int _warp_vote_1 = __any_sync(0xFFFFFFFF, (k70 & 2155905152u) != 0);
                            int _warp_vote_2 = __any_sync(0xFFFFFFFF, (k78 & 2155905152u) != 0);
                            kexp_u = (unsigned int)(_warp_vote_0 != 0) + (unsigned int)(_warp_vote_1 != 0) + (unsigned int)(_warp_vote_2 != 0);
                            int _stage_v_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile) % 2);
                            int _parity_v_empty = (int)((unsigned int)(ld_tile_cursor + ld_tile) / 2 % 2) ^ 1;
                            mbarrier_wait(v_empty_addr + (_stage_v_empty) * 8, _parity_v_empty);
                            if (ld_warp == 0) {
                                if (elect_sync()) {
                                    smem_kexp[p_stage * 4] = kexp_u;
                                    mbarrier_arrive(kexp_full_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 2)) * 8);
                                }
                            }
                        } else {
                            int _stage_kexp_full_1 = (int)((unsigned int)(ld_tile_cursor + ld_tile) % 2);
                            int _parity_kexp_full_1 = (int)((unsigned int)(ld_tile_cursor + ld_tile) / 2 % 2);
                            mbarrier_wait(kexp_full_addr + (_stage_kexp_full_1) * 8, _parity_kexp_full_1);
                            kexp_u = smem_kexp[p_stage * 4];
                        }
                        unsigned int kexp_f16x2 = (15 - kexp_u << 10) * 65537;
                        float kexp_scale = __uint_as_float(127 - kexp_u << 23);
                        if (rest_4 == 0 || ld_tile != 2 || v_key < 4) {
                            int swz16 = (v_key & 7) << 4;
                            int v_row_off = v_key * 128;
                            int v_sf_addr = sf_base + v_key * 32 + v_half * 16;
                            int v_codes_src = smem_k_a_addr + (unsigned int)(k_stage * 26624) + (unsigned int)(v_half * 8192) + (unsigned int)v_row_off;
                            int v_dst_base0 = p_stage * 32768 + v_key * 128 + 2 * v_half * 8192;
                            int v_dst_base1 = v_dst_base0 + 8192;
                            unsigned int v_sc[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&v_sc[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_sc[(0) + 3]))
                                : "r"(v_sf_addr));
                            if ((int)(kexp_u == 0) != 0) {
                                #pragma unroll 1
                                for (int u = x_u_lo; u < x_u_hi; u++) {
                                    unsigned int v_codes[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&v_codes[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_codes[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_codes[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_codes[(0) + 3]))
                                        : "r"(v_codes_src + (u << 4 ^ swz16)));
                                    int v_w_idx = u >> 1;
                                    unsigned int v_sc_lo = ((v_w_idx == 0) ? v_sc[0] : v_sc[1]);
                                    unsigned int v_sc_hi = ((v_w_idx == 2) ? v_sc[2] : v_sc[3]);
                                    unsigned int v_sc_word = ((v_w_idx < 2) ? v_sc_lo : v_sc_hi) >> (unsigned int)((u & 1) * 16);
                                    unsigned int v_out[8];
                                    #pragma unroll
                                    for (int v_w = 0; v_w < 4; v_w++) {
                                        unsigned int v_sc_w = ((v_w < 2) ? v_sc_word : v_sc_word >> 8);
                                        v_out[2 * v_w] = qmul4_lower4(v_codes[v_w], v_sc_w);
                                        v_out[2 * v_w + 1] = qmul4_higher4(v_codes[v_w], v_sc_w);
                                    }
                                    int v_dst = ((u >> 2 == 0) ? v_dst_base0 : v_dst_base1);
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst + ((u & 3) * 2 << 4 ^ swz16))), "r"(v_out[0]), "r"(v_out[1]), "r"(v_out[2]), "r"(v_out[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst + ((u & 3) * 2 + 1 << 4 ^ swz16))), "r"(v_out[4]), "r"(v_out[5]), "r"(v_out[6]), "r"(v_out[7]) : "memory");
                                }
                            } else {
                                #pragma unroll 1
                                for (int u_1 = x_u_lo; u_1 < x_u_hi; u_1++) {
                                    unsigned int v_in[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&v_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 3]))
                                        : "r"(v_codes_src + (u_1 << 4 ^ swz16)));
                                    int v_w_idx_1 = u_1 >> 1;
                                    unsigned int v_sc_lo_1 = ((v_w_idx_1 == 0) ? v_sc[0] : v_sc[1]);
                                    unsigned int v_sc_hi_1 = ((v_w_idx_1 == 2) ? v_sc[2] : v_sc[3]);
                                    unsigned int v_sc_word_1 = ((v_w_idx_1 < 2) ? v_sc_lo_1 : v_sc_hi_1);
                                    uint32_t v_sc_pair;
                                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(v_sc_pair) : "h"((uint16_t)(v_sc_word_1 >> (unsigned int)((u_1 & 1) * 16))));
                                    uint32_t _f16x2_mul_0;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(v_sc_pair), "r"(kexp_f16x2));
                                    v_sc_pair = _f16x2_mul_0;
                                    uint32_t v_sc0;
                                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(v_sc0) : "r"(v_sc_pair), "r"(v_sc_pair));
                                    uint32_t v_sc1;
                                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(v_sc1) : "r"(v_sc_pair), "r"(v_sc_pair));
                                    unsigned int v_out_1[8];
                                    #pragma unroll
                                    for (int v_w_1 = 0; v_w_1 < 4; v_w_1++) {
                                        unsigned int v_sc_w_1 = ((v_w_1 < 2) ? v_sc0 : v_sc1);
                                        unsigned int v_word = v_in[v_w_1];
                                        uint32_t _cvt_f16x2_e2m1x2_3;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_3) : "r"((uint32_t)(v_word)));
                                        uint32_t f16x2_mul;
                                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul) : "r"(_cvt_f16x2_e2m1x2_3), "r"(v_sc_w_1));
                                        uint16_t e4m3x2;
                                        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2) : "r"(f16x2_mul));
                                        uint32_t _cvt_f16x2_e2m1x2_4;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_4) : "r"((uint32_t)(v_word >> 8)));
                                        uint32_t f16x2_mul_0;
                                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_0) : "r"(_cvt_f16x2_e2m1x2_4), "r"(v_sc_w_1));
                                        uint16_t e4m3x2_0;
                                        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_0) : "r"(f16x2_mul_0));
                                        uint32_t _cvt_f16x2_e2m1x2_5;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_5) : "r"((uint32_t)(v_word >> 16)));
                                        uint32_t f16x2_mul_1;
                                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_1) : "r"(_cvt_f16x2_e2m1x2_5), "r"(v_sc_w_1));
                                        uint16_t e4m3x2_1;
                                        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_1) : "r"(f16x2_mul_1));
                                        uint32_t _cvt_f16x2_e2m1x2_6;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_6) : "r"((uint32_t)(v_word >> 24)));
                                        uint32_t f16x2_mul_2;
                                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_2) : "r"(_cvt_f16x2_e2m1x2_6), "r"(v_sc_w_1));
                                        uint16_t e4m3x2_2;
                                        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_2) : "r"(f16x2_mul_2));
                                        uint32_t _mov_b32_pack_u16x2_7;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_7) : "h"(e4m3x2), "h"(e4m3x2_0));
                                        v_out_1[2 * v_w_1] = _mov_b32_pack_u16x2_7;
                                        uint32_t _mov_b32_pack_u16x2_8;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_8) : "h"(e4m3x2_1), "h"(e4m3x2_2));
                                        v_out_1[2 * v_w_1 + 1] = _mov_b32_pack_u16x2_8;
                                    }
                                    int v_dst_1 = ((u_1 >> 2 == 0) ? v_dst_base0 : v_dst_base1);
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_1 + ((u_1 & 3) * 2 << 4 ^ swz16))), "r"(v_out_1[0]), "r"(v_out_1[1]), "r"(v_out_1[2]), "r"(v_out_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(v_dst_1 + ((u_1 & 3) * 2 + 1 << 4 ^ swz16))), "r"(v_out_1[4]), "r"(v_out_1[5]), "r"(v_out_1[6]), "r"(v_out_1[7]) : "memory");
                                }
                            }
                            if (v_half != 0) {
                                int r_src_row = smem_k_a_addr + (unsigned int)(k_stage * 26624) + 16384 + (unsigned int)v_row_off;
                                int r_dst = p_stage * 32768 + v_key * 128 + 24576;
                                #pragma unroll
                                for (int r_i = 0; r_i < 2; r_i++) {
                                    unsigned int r_out[4];
                                    #pragma unroll
                                    for (int r_half = 0; r_half < 2; r_half++) {
                                        unsigned int r_in[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&r_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 3]))
                                            : "r"(r_src_row + (2 * (x_r_lo + r_i) + r_half << 4 ^ swz16)));
                                        #pragma unroll
                                        for (int r_w = 0; r_w < 2; r_w++) {
                                            float r_lo0 = __uint_as_float(r_in[2 * r_w] << 16);
                                            float r_hi0 = __uint_as_float(r_in[2 * r_w] & 4294901760u);
                                            float r_lo1 = __uint_as_float(r_in[2 * r_w + 1] << 16);
                                            float r_hi1 = __uint_as_float(r_in[2 * r_w + 1] & 4294901760u);
                                            if ((int)(kexp_u == 0) == 0) {
                                                r_lo0 = r_lo0 * kexp_scale;
                                                r_hi0 = r_hi0 * kexp_scale;
                                                r_lo1 = r_lo1 * kexp_scale;
                                                r_hi1 = r_hi1 * kexp_scale;
                                            }
                                            uint16_t _cvt_satfinite_e4m3x2_f32_10;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_10) : "f"(r_hi0), "f"(r_lo0));
                                            uint16_t _cvt_satfinite_e4m3x2_f32_11;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(r_hi1), "f"(r_lo1));
                                            uint32_t _mov_b32_pack_u16x2_9;
                                            asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_9) : "h"(_cvt_satfinite_e4m3x2_f32_10), "h"(_cvt_satfinite_e4m3x2_f32_11));
                                            r_out[2 * r_half + r_w] = _mov_b32_pack_u16x2_9;
                                        }
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(r_dst + (4 + (x_r_lo + r_i) << 4 ^ swz16))), "r"(r_out[0]), "r"(r_out[1]), "r"(r_out[2]), "r"(r_out[3]) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (x_grp == 0) {
                            asm volatile("barrier.sync 7, 128;" ::: "memory");
                        } else {
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
                        }
                        if (ld_thread == 0) {
                            mbarrier_arrive(kstage_free_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 3)) * 8);
                            mbarrier_arrive(v_full_addr + ((int)((unsigned int)(ld_tile_cursor + ld_tile) % 2)) * 8);
                        }
                    }
                    ld_tile_cursor += tiles_4;
                }
                mbarrier_wait(index_full_addr + (ld_info_stage) * 8, _phase_index_full_4);
                ld_query = smem_indices[(int)ld_info_stage * 528 + 512 + 15];
                if (ld_query < 0) {
                    break;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: gather ----
    } else if (warp >= 20 && warp <= 23) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // gather_main
            const int g_warp = warp - 20;
            const int g_early = (int)(blockIdx.x >= (int)gridDim.x - (num_query_tokens + 3 >> 2));
            int g_tile_cursor = 0;
            int g_item = 0;
            unsigned int g_index_stage = 0;
            int q_5 = blockIdx.x;
            unsigned int _phase_index_full_5 = 0;
            if (blockIdx.x < (int)gridDim.x - (num_query_tokens + 3 >> 2)) {
                mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_5);
                q_5 = smem_indices[(int)g_index_stage * 528 + 512 + 15];
            }
            int g_query = q_5;
            int g_prev_nt = 0;
            asm volatile("barrier.sync 11, 128;" ::: "memory");
            int _max_0_1 = ((smem_first[128] + sparse_topk_lens_offset) > (0) ? (smem_first[128] + sparse_topk_lens_offset) : (0));
            int gp_len = ((_max_0_1) < (swa_width) ? (_max_0_1) : (swa_width));
            if (g_early != 0) {
                #pragma unroll 1
                for (int g_pt = 0; g_pt < 2; g_pt++) {
                    int _stage_k_empty_1 = (int)((unsigned int)(g_tile_cursor + g_pt) % 3);
                    int _parity_k_empty_1 = (int)((unsigned int)(g_tile_cursor + g_pt) / 3 % 2) ^ 1;
                    mbarrier_wait(k_empty_addr + (_stage_k_empty_1) * 8, _parity_k_empty_1);
                    int _stage_kstage_free = (int)((unsigned int)(g_tile_cursor + g_pt) % 3);
                    int _parity_kstage_free = (int)((unsigned int)(g_tile_cursor + g_pt) / 3 % 2) ^ 1;
                    mbarrier_wait(kstage_free_addr + (_stage_kstage_free) * 8, _parity_kstage_free);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (g_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(k_full_addr + (g_pt) * 8, 26624);
                        }
                    }
                    if (lane < 16) {
                        int gp_rows[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gp_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&gp_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gp_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gp_rows[(0) + 3]))
                            : "r"(smem_first_addr + (unsigned int)(256 * g_pt + lane * 16)));
                        int gp_data_rows[4];
                        int gp_foot_rows[4];
                        #pragma unroll
                        for (int g_j = 0; g_j < 4; g_j++) {
                            int gp_row = ((g_pt * 64 + lane * 4 + g_j < gp_len) ? gp_rows[g_j] : -1);
                            int gp_tok = ((gp_row >= 0) ? gp_row : 0);
                            int gp_page = gp_tok >> swa_page_log2;
                            int gp_pslot = gp_tok - (gp_page << swa_page_log2);
                            gp_data_rows[g_j] = ((gp_row >= 0) ? gp_page * swa_pitch_units + gp_pslot * 11 : 1073741824);
                            gp_foot_rows[g_j] = ((gp_row >= 0) ? gp_page * swa_pitch_units + swa_footer_units + gp_pslot : 1073741824);
                        }
                        int gp_dst = smem_k_a_addr + (unsigned int)(g_pt * 26624);
                        if (g_warp < 3) {
                            tma_gather4_gmem2smem(gp_dst + g_warp * 8192 + lane * 512, (&tmap_swa_kv), ((g_warp == 2) ? 224 : g_warp * 128), gp_data_rows[0], gp_data_rows[1], gp_data_rows[2], gp_data_rows[3], k_full_addr + (g_pt) * 8);
                        } else {
                            tma_gather4_gmem2smem(gp_dst + 24576 + lane * 128, (&tmap_swa_sf), 0, gp_foot_rows[0], gp_foot_rows[1], gp_foot_rows[2], gp_foot_rows[3], k_full_addr + (g_pt) * 8);
                        }
                    }
                }
            }
            #pragma unroll 1
            for (unsigned int g_work = 0; g_work < 1048576; g_work++) {
                int g_ret = 0;
                int g_next = -1;
                if (g_query >= 0) {
                    mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_5);
                    int nb_5 = smem_indices[(int)g_index_stage * 528 + 512 + 13];
                    int rest_5 = smem_indices[(int)g_index_stage * 528 + 512] > 1;
                    int tiles_5 = ((smem_indices[(int)g_index_stage * 528 + 512] > 1) ? 3 + 2 * (nb_5 - smem_indices[(int)g_index_stage * 528 + 512]) : 2 * nb_5);
                    #pragma unroll 1
                    for (int g_tile = 2 * (int)(g_item == 0 && g_early != 0); g_tile < tiles_5; g_tile++) {
                        int g_stage = (g_tile_cursor + g_tile) % 3;
                        int _stage_k_empty_2 = (int)((unsigned int)(g_tile_cursor + g_tile) % 3);
                        int _parity_k_empty_2 = (int)((unsigned int)(g_tile_cursor + g_tile) / 3 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty_2) * 8, _parity_k_empty_2);
                        int _stage_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + g_tile) % 3);
                        int _parity_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + g_tile) / 3 % 2) ^ 1;
                        mbarrier_wait(kstage_free_addr + (_stage_kstage_free_1) * 8, _parity_kstage_free_1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int g_narrow = rest_5 != 0 && g_tile == 2;
                        if (g_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8, 26624 - 24960 * g_narrow);
                            }
                        }
                        if (lane < 16 - 15 * g_narrow) {
                            int g_group = lane;
                            int g_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&g_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 3]))
                                : "r"(smem_indices_addr + g_index_stage * 2112 + (unsigned int)((g_tile * 64 + g_group * 4) * 4)));
                            int g_page_log2 = swa_page_log2;
                            int g_pitch = swa_pitch_units;
                            int g_footer_units = swa_footer_units;
                            int g_data_rows[4];
                            int g_foot_rows[4];
                            #pragma unroll
                            for (int g_j_1 = 0; g_j_1 < 4; g_j_1++) {
                                int g_tok = ((g_rows[g_j_1] >= 0) ? g_rows[g_j_1] : 0);
                                int g_page = g_tok >> g_page_log2;
                                int g_pslot = g_tok - (g_page << g_page_log2);
                                g_data_rows[g_j_1] = ((g_rows[g_j_1] >= 0) ? g_page * g_pitch + g_pslot * 11 : 1073741824);
                                g_foot_rows[g_j_1] = ((g_rows[g_j_1] >= 0) ? g_page * g_pitch + g_footer_units + g_pslot : 1073741824);
                            }
                            int g_stage_base = smem_k_a_addr + (unsigned int)(g_stage * 26624);
                            if (g_warp < 3) {
                                int g_col = ((g_warp == 2) ? 224 : g_warp * 128);
                                int g_dst = g_stage_base + g_warp * 8192 + g_group * 512;
                                tma_gather4_gmem2smem(g_dst, (&tmap_swa_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8);
                            } else {
                                int g_fdst = g_stage_base + 24576 + g_group * 128;
                                tma_gather4_gmem2smem(g_fdst, (&tmap_swa_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(g_tile_cursor + g_tile) % 3)) * 8);
                            }
                        }
                    }
                    mbarrier_arrive(index_empty_addr + (g_index_stage) * 8);
                    g_item += 1;
                    g_index_stage += 1;
                    if (g_index_stage == 2) { g_index_stage = 0; _phase_index_full_5 ^= 1; }
                    g_tile_cursor += tiles_5;
                    g_ret = rest_5;
                }
                g_prev_nt = g_ret;
                mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_5);
                g_query = smem_indices[(int)g_index_stage * 528 + 512 + 15];
                if (g_query < 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
