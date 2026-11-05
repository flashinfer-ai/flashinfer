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
kernel_cake_dsv4_nvfp4_qb_h32_late_claim_prefill(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, const __grid_constant__ CUtensorMap tmap_o, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int has_sinks, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens, const __grid_constant__ CUtensorMap tmap_o16)
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
    #define q_read_addr (mbar_base + 8)
    #define q_full_addr (mbar_base + 16)
    #define q_empty_addr (mbar_base + 24)
    #define index_full_addr (mbar_base + 32)
    #define index_empty_addr (mbar_base + 48)
    #define tail_full_addr (mbar_base + 64)
    #define tdot_full_addr (mbar_base + 80)
    #define k_full_addr (mbar_base + 96)
    #define k_empty_addr (mbar_base + 120)
    #define kstage_free_addr (mbar_base + 144)
    #define sf_full_addr (mbar_base + 168)
    #define kexp_full_addr (mbar_base + 192)
    #define v_full_addr (mbar_base + 208)
    #define v_empty_addr (mbar_base + 224)
    #define s_full_addr (mbar_base + 240)
    #define s_empty_addr (mbar_base + 256)
    #define p_full_addr (mbar_base + 272)
    #define p_recycle_addr (mbar_base + 288)
    #define stats_addr (mbar_base + 304)
    #define o_empty_addr (mbar_base + 320)
    #define o_full_addr (mbar_base + 336)
    #define sum_ready_addr (mbar_base + 352)
    #define sum_empty_addr (mbar_base + 360)
    #define tmem_dealloc_addr (mbar_base + 368)
    #define work_full_addr (mbar_base + 376)
    #define work_empty_addr (mbar_base + 392)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* smem_q_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_a_addr = smem + 1024;
    uint8_t* smem_q_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_q_b_addr = smem + 17408;
    uint8_t* smem_kt = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_kt_addr = smem + 9216;
    float* smem_tdot = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_tdot_addr = smem + 13312;
    float* smem_ptail = reinterpret_cast<float*>(smem_raw + 231424);
    const int smem_ptail_addr = smem + 231424;
    float* smem_vt = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_vt_addr = smem + 25600;
    float* smem_vzero = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_vzero_addr = smem + 230400;
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
    int* smem_first = reinterpret_cast<int*>(smem_raw + 213184);
    const int smem_first_addr = smem + 213184;

    // Kernel setup ops
    if (tid < 64) {
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vzero_addr + (unsigned int)(tid * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
    }
    if (warp == 10) {
        int pf_t0 = ((blockIdx.x * 2) < (num_query_tokens - 1) ? (blockIdx.x * 2) : (num_query_tokens - 1));
        if (lane < 8) {
            int _min_0 = ((pf_t0 + (lane >> 2)) < (num_query_tokens - 1) ? (pf_t0 + (lane >> 2)) : (num_query_tokens - 1));
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(swa_indices + (_min_0 * swa_index_stride + (lane & 3) * 32))));
        }
        if (lane == 31) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(sparse_topk_lens + pf_t0)));
        }
    }
    int pc_fallback[1];
    pc_fallback[0] = 0;
    if (warp == 10) {
        int pc_t0 = blockIdx.x * 2;
        if (pc_t0 + 1 < num_query_tokens) {
            int pc_len0 = sparse_topk_lens[pc_t0] + sparse_topk_lens_offset;
            int pc_len1 = sparse_topk_lens[pc_t0 + 1] + sparse_topk_lens_offset;
            int pc_r0[2];
            int pc_r1[2];
            {
                const int2* _ivptr_1 = reinterpret_cast<const int2*>(swa_indices + pc_t0 * swa_index_stride + 0);
                int2 _ivld_1;
                _ivld_1 = *_ivptr_1;
                pc_r0[0 + 0] = _ivld_1.x;
                pc_r0[0 + 1] = _ivld_1.y;
            }
            {
                const int2* _ivptr_2 = reinterpret_cast<const int2*>(swa_indices + (pc_t0 + 1) * swa_index_stride + 0);
                int2 _ivld_2;
                _ivld_2 = *_ivptr_2;
                pc_r1[0 + 0] = _ivld_2.x;
                pc_r1[0 + 1] = _ivld_2.y;
            }
            int _min_3 = ((pc_len0) < (swa_width) ? (pc_len0) : (swa_width));
            int _min_4 = ((pc_len1) < (swa_width) ? (pc_len1) : (swa_width));
            pc_fallback[0] = (int)(_min_3 != 128 || _min_4 != 128 || pc_r1[0] != pc_r0[1] && pc_r1[1] != pc_r0[0]);
        }
    }
    int pi_rows[8];
    int pi_len[1];
    if (warp == 20) {
        int pg_t0 = ((blockIdx.x * 2) < (num_query_tokens - 1) ? (blockIdx.x * 2) : (num_query_tokens - 1));
        {
            const int4* _ivptr_5 = reinterpret_cast<const int4*>(swa_indices + (pg_t0 * swa_index_stride + (lane & 15) * 4) + 0);
            int4 _ivld_5;
            _ivld_5 = *_ivptr_5;
            pi_rows[0 + 0] = _ivld_5.x;
            pi_rows[0 + 1] = _ivld_5.y;
            pi_rows[0 + 2] = _ivld_5.z;
            pi_rows[0 + 3] = _ivld_5.w;
        }
        {
            const int4* _ivptr_6 = reinterpret_cast<const int4*>(swa_indices + (pg_t0 * swa_index_stride + 64 + (lane & 15) * 4) + 0);
            int4 _ivld_6;
            _ivld_6 = *_ivptr_6;
            pi_rows[4 + 0] = _ivld_6.x;
            pi_rows[4 + 1] = _ivld_6.y;
            pi_rows[4 + 2] = _ivld_6.z;
            pi_rows[4 + 3] = _ivld_6.w;
        }
        pi_len[0] = sparse_topk_lens[pg_t0];
    }

    // Mbarrier init (51 barriers at smem_raw[0..408))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_load_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_read: 1 barriers, init_count=128
            mbarrier_init(smem + 8, 128);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'index_pipe' ---
            // index_full: 2 barriers, init_count=32
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            // index_empty: 2 barriers, init_count=704
            mbarrier_init(smem + 48, 704);
            mbarrier_init(smem + 56, 704);
            // tail_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // tdot_full: 2 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // kstage_free: 3 barriers, init_count=2
            mbarrier_init(smem + 144, 2);
            mbarrier_init(smem + 152, 2);
            mbarrier_init(smem + 160, 2);
            // sf_full: 3 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 208, 2);
            mbarrier_init(smem + 216, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 272, 128);
            mbarrier_init(smem + 280, 128);
            // p_recycle: 2 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            mbarrier_init(smem + 312, 128);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 320, 128);
            mbarrier_init(smem + 328, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 352, 128);
            // sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            // tmem_dealloc: 1 barriers, init_count=576
            mbarrier_init(smem + 368, 576);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 376, 1);
            mbarrier_init(smem + 384, 1);
            // work_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 392, 32);
            mbarrier_init(smem + 400, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 408);
    if (warp == 0) {
        int _tmem_hold = smem + 408;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
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
            int sm_query = q;
            int sm_last = blockIdx.x;
            int sm_last_t0 = 0;
            int sm_last_nb = 0;
            unsigned int _phase_index_full = 0;
            #pragma unroll 1
            for (unsigned int sm_work = 0; sm_work < 1048576; sm_work++) {
                sm_last = sm_query;
                int sm_ret = -1;
                int sm_t0 = 0;
                int sm_nb = 0;
                if (sm_query * 2 < num_query_tokens) {
                    mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                    sm_t0 = sm_query * 2;
                    int _min_0_1 = ((num_query_tokens - sm_query * 2) < (2) ? (num_query_tokens - sm_query * 2) : (2));
                    sm_nb = _min_0_1;
                    int nb = sm_nb;
                    int rest = ((smem_indices[(int)sm_index_stage * 272 + 256] != 0) ? nb - 1 : 0);
                    int tiles = ((smem_indices[(int)sm_index_stage * 272 + 256] != 0) ? 2 : 2 * nb);
                    sm_ret = smem_indices[(int)sm_index_stage * 272 + 256 + 14];
                    int sm_lo = smem_indices[(int)sm_index_stage * 272 + 256 + (2 + 2 * (sm_head >> 5))];
                    int sm_span = smem_indices[(int)sm_index_stage * 272 + 256 + (3 + 2 * (sm_head >> 5))];
                    int sm_hi = sm_lo + sm_span;
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
                        row_max_val = sinks[sm_head & 31] * 1.442695f / softmax_scale_log2;
                        row_sum_val = ((sm_half == 0) ? 1.0f : 0.0f);
                    }
                    #pragma unroll 1
                    for (int sm_tile = 0; sm_tile < tiles; sm_tile++) {
                        int s_stage = (sm_tile_cursor + sm_tile) % 2;
                        int sm_col0 = sm_tile * 64 + sm_half * 32;
                        int staged_index[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[0])) : "r"(smem_indices_addr + sm_index_stage * 1088 + (unsigned int)((sm_tile * 64 + sm_half * 32 + lane) * 4)));
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
                        int sm_c0 = sm_col0;
                        uint32_t _thresh_mask_1;
                        {
                            int _lim_2 = sm_hi - sm_c0;
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
                        uint32_t _thresh_mask_3;
                        {
                            int _lim_4 = sm_lo - sm_c0;
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
                        unsigned int keep_cols = _thresh_mask_1 & (_thresh_mask_3 ^ 4294967295u) & (invalid_cols ^ 4294967295u);
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
                        float2 _reg_reduce_max2_5 = {-CUDART_INF_F, -CUDART_INF_F};
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[0], sv[1]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[2], sv[3]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[4], sv[5]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[6], sv[7]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[8], sv[9]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[10], sv[11]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[12], sv[13]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[14], sv[15]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[16], sv[17]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[18], sv[19]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[20], sv[21]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[22], sv[23]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[24], sv[25]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[26], sv[27]));
                        _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv[28], sv[29]));
                        _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv[30], sv[31]));
                        float local_max = row_max_reduce(_reg_reduce_max2_5);
                        smem_exch[s_stage * 128 + sm_thread] = local_max;
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(max_noftz(local_max, smem_exch[s_stage * 128 + (sm_thread ^ 64)])), "f"(row_max_val));
                        if (row_max_val > -CUDART_INF_F && (new_max - row_max_val) * softmax_scale_log2 <= 5.0f) {
                            new_max = row_max_val;
                        }
                        float delta = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max * softmax_scale_log2);
                        float _exp2_6 = approx_exp2(delta);
                        float acc_scale = ((row_max_val > -CUDART_INF_F && new_max != row_max_val) ? _exp2_6 : 1.0f);
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
                        const float2 _fma_b2_7 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_8 = {kexp_f - max_scaled, kexp_f - max_scaled};
                        float2 _fma_pair_9 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_7, _fma_c2_8);
                        sv[0] = _fma_pair_9.x;
                        sv[1] = _fma_pair_9.y;
                        float2 _fma_pair_10 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_7, _fma_c2_8);
                        sv[2] = _fma_pair_10.x;
                        sv[3] = _fma_pair_10.y;
                        float2 _fma_pair_11 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_7, _fma_c2_8);
                        sv[4] = _fma_pair_11.x;
                        sv[5] = _fma_pair_11.y;
                        float2 _fma_pair_12 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_7, _fma_c2_8);
                        sv[6] = _fma_pair_12.x;
                        sv[7] = _fma_pair_12.y;
                        float2 _fma_pair_13 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_7, _fma_c2_8);
                        sv[8] = _fma_pair_13.x;
                        sv[9] = _fma_pair_13.y;
                        float2 _fma_pair_14 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_7, _fma_c2_8);
                        sv[10] = _fma_pair_14.x;
                        sv[11] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_7, _fma_c2_8);
                        sv[12] = _fma_pair_15.x;
                        sv[13] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_7, _fma_c2_8);
                        sv[14] = _fma_pair_16.x;
                        sv[15] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_7, _fma_c2_8);
                        sv[16] = _fma_pair_17.x;
                        sv[17] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_7, _fma_c2_8);
                        sv[18] = _fma_pair_18.x;
                        sv[19] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_7, _fma_c2_8);
                        sv[20] = _fma_pair_19.x;
                        sv[21] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_7, _fma_c2_8);
                        sv[22] = _fma_pair_20.x;
                        sv[23] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_7, _fma_c2_8);
                        sv[24] = _fma_pair_21.x;
                        sv[25] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_7, _fma_c2_8);
                        sv[26] = _fma_pair_22.x;
                        sv[27] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_7, _fma_c2_8);
                        sv[28] = _fma_pair_23.x;
                        sv[29] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_7, _fma_c2_8);
                        sv[30] = _fma_pair_24.x;
                        sv[31] = _fma_pair_24.y;
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
                        float2 _reg_reduce_sum2_26 = make_float2(0.0f, 0.0f);
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[0], sv[1]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[2], sv[3]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[4], sv[5]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[6], sv[7]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[8], sv[9]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[10], sv[11]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[12], sv[13]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[14], sv[15]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[16], sv[17]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[18], sv[19]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[20], sv[21]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[22], sv[23]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[24], sv[25]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[26], sv[27]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[28], sv[29]));
                        _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv[30], sv[31]));
                        float _reg_array_reduce_25 = _reg_reduce_sum2_26.x + _reg_reduce_sum2_26.y;
                        float block_sum = _reg_array_reduce_25 * kexp_inv;
                        row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                    }
                    int sm_tok = sm_head >> 5;
                    int sm_kw = ((rest) < ((sm_row_base & 63) + 31 >> 5) ? (rest) : ((sm_row_base & 63) + 31 >> 5));
                    mbarrier_wait(tdot_full_addr + (sm_index_stage) * 8, _phase_index_full);
                    float t_s[1];
                    float t_mx = row_max_val;
                    int t_u_0 = ((sm_head >= 32) ? 0 : 0);
                    int t_i_0 = (int)sm_index_stage * 128 + t_u_0 * 128 + (sm_head & 31);
                    t_s[0] = smem_tdot[t_i_0] + smem_tdot[t_i_0 + 32] + (smem_tdot[t_i_0 + 64] + smem_tdot[t_i_0 + 96]);
                    if (0 < sm_kw && 0 < sm_tok) {
                        t_mx = max_noftz(t_mx, t_s[0]);
                    }
                    mbarrier_arrive(index_empty_addr + (sm_index_stage) * 8);
                    sm_index_stage += 1;
                    if (sm_index_stage == 2) { sm_index_stage = 0; _phase_index_full ^= 1; }
                    if (row_max_val > -CUDART_INF_F && (t_mx - row_max_val) * softmax_scale_log2 <= 5.0f) {
                        t_mx = row_max_val;
                    }
                    float _exp2_27 = approx_exp2(__fmaf_rn(row_max_val, softmax_scale_log2, -t_mx * softmax_scale_log2));
                    float t_alpha = ((row_max_val > -CUDART_INF_F && t_mx != row_max_val) ? _exp2_27 : 1.0f);
                    row_sum_val = row_sum_val * t_alpha;
                    row_max_val = t_mx;
                    int _parity_sum_empty = (int)((unsigned int)sm_item % 2) ^ 1;
                    mbarrier_wait(sum_empty_addr, _parity_sum_empty);
                    float t_fms = ((row_max_val == -CUDART_INF_F) ? 0.0f : row_max_val) * softmax_scale_log2;
                    float _exp2_28 = approx_exp2(__fmaf_rn(t_s[0], softmax_scale_log2, -t_fms));
                    float t_p_0 = ((0 < sm_kw && 0 < sm_tok) ? _exp2_28 : 0.0f);
                    if (sm_half == 0) {
                        row_sum_val = row_sum_val + t_p_0;
                        uint16_t _cvt_satfinite_e4m3x2_f32_31;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_31) : "f"(0.0f), "f"(t_p_0));
                        uint32_t _cvt_f16x2_e4m3x2_30;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_30) : "h"((uint16_t)(_cvt_satfinite_e4m3x2_f32_31)));
                        float _cvt_f32_f16_29;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(_cvt_f16x2_e4m3x2_30)));
                        smem_ptail[sm_head] = _cvt_f32_f16_29;
                    }
                    if (sm_half == 0) {
                        smem_ptail[64 + sm_head] = t_alpha;
                    }
                    smem_sum[sm_thread] = row_sum_val;
                    smem_fmax[sm_thread] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    sm_tile_cursor += tiles;
                    sm_item += 1;
                }
                int q_4 = sm_ret;
                if (sm_ret == -2) {
                    mbarrier_wait(index_full_addr + (sm_index_stage) * 8, _phase_index_full);
                    q_4 = smem_indices[(int)sm_index_stage * 272 + 256 + 15];
                }
                sm_query = q_4;
                if (sm_query < 0) {
                    break;
                }
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
            int co_query = q_1;
            mbarrier_arrive(o_empty_addr);
            unsigned int _phase_index_full_1 = 0;
            #pragma unroll 1
            for (unsigned int co_work = 0; co_work < 1048576; co_work++) {
                int co_ret = -1;
                if (co_query * 2 < num_query_tokens) {
                    mbarrier_wait(index_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                    int co_t0 = co_query * 2;
                    int co_nb = ((num_query_tokens - co_query * 2) < (2) ? (num_query_tokens - co_query * 2) : (2));
                    int nb_1 = co_nb;
                    int rest_1 = ((smem_indices[(int)co_info_stage * 272 + 256] != 0) ? nb_1 - 1 : 0);
                    int tiles_1 = ((smem_indices[(int)co_info_stage * 272 + 256] != 0) ? 2 : 2 * nb_1);
                    co_ret = smem_indices[(int)co_info_stage * 272 + 256 + 14];
                    #pragma unroll 1
                    for (int co_tile = 0; co_tile < tiles_1; co_tile++) {
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
                            int _parity_o_full_1 = (int)((unsigned int)(co_tile_cursor + co_tile) % 2) ^ 1;
                            mbarrier_wait(o_full_addr + 8, _parity_o_full_1);
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
                    int co_kw = ((rest_1) < ((co_thread & 32) + 31 >> 5) ? (rest_1) : ((co_thread & 32) + 31 >> 5));
                    unsigned int co_vt = smem_vt_addr + (unsigned int)(co_half * 1024);
                    mbarrier_wait(tail_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                    unsigned int t_base = smem_kt_addr + co_info_stage * 2048;
                    int t_d = (lane >> 4) * 256 + co_half * 128 + (lane & 15) * 8;
                    #pragma unroll 1
                    for (int t_j = 0; t_j < co_kw; t_j++) {
                        float t_out[8];
                        if (t_d < 448) {
                            unsigned int t_row = t_base + (unsigned int)((t_d >> 8) * 512 + t_j * 128);
                            unsigned int t_code[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_code[0])) : "r"(t_row + (unsigned int)((t_d >> 5 & 7 ^ (int)(t_row >> 7 & 7)) << 4 | t_d >> 1 & 12)));
                            unsigned int t_sw[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_sw[0])) : "r"(t_base + (unsigned int)(1536 + t_j * 32 + (t_d >> 4 & 28))));
                            unsigned int t_sb = t_sw[0] >> (unsigned int)((t_d >> 4 & 3) * 8) & 255;
                            uint32_t t_sc;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(t_sc) : "h"((uint16_t)(t_sb * 257)));
                            uint32_t _cvt_f16x2_e2m1x2_2;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_2) : "r"((uint32_t)(t_code[0] >> 0)));
                            uint32_t t_h2_4;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(t_h2_4) : "r"(_cvt_f16x2_e2m1x2_2), "r"(t_sc));
                            float _cvt_f32_f16_3;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(t_h2_4)));
                            t_out[0] = _cvt_f32_f16_3;
                            float _cvt_f32_f16_4;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(t_h2_4 >> 16)));
                            t_out[1] = _cvt_f32_f16_4;
                            uint32_t _cvt_f16x2_e2m1x2_5;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_5) : "r"((uint32_t)(t_code[0] >> 8)));
                            uint32_t t_h2_5;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(t_h2_5) : "r"(_cvt_f16x2_e2m1x2_5), "r"(t_sc));
                            float _cvt_f32_f16_6;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(t_h2_5)));
                            t_out[2] = _cvt_f32_f16_6;
                            float _cvt_f32_f16_7;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(t_h2_5 >> 16)));
                            t_out[3] = _cvt_f32_f16_7;
                            uint32_t _cvt_f16x2_e2m1x2_8;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_8) : "r"((uint32_t)(t_code[0] >> 16)));
                            uint32_t t_h2_6;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(t_h2_6) : "r"(_cvt_f16x2_e2m1x2_8), "r"(t_sc));
                            float _cvt_f32_f16_9;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(t_h2_6)));
                            t_out[4] = _cvt_f32_f16_9;
                            float _cvt_f32_f16_10;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(t_h2_6 >> 16)));
                            t_out[5] = _cvt_f32_f16_10;
                            uint32_t _cvt_f16x2_e2m1x2_11;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_11) : "r"((uint32_t)(t_code[0] >> 24)));
                            uint32_t t_h2_7;
                            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(t_h2_7) : "r"(_cvt_f16x2_e2m1x2_11), "r"(t_sc));
                            float _cvt_f32_f16_12;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(t_h2_7)));
                            t_out[6] = _cvt_f32_f16_12;
                            float _cvt_f32_f16_13;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(t_h2_7 >> 16)));
                            t_out[7] = _cvt_f32_f16_13;
                        } else {
                            unsigned int t_row_1 = t_base + (unsigned int)(1024 + t_j * 128);
                            unsigned int t_rv[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_rv[(0) + 3]))
                                : "r"(t_row_1 + (unsigned int)((t_d - 448 >> 3 ^ (int)(t_row_1 >> 7 & 7)) << 4)));
                            t_out[0] = __uint_as_float(t_rv[0] << 16);
                            t_out[1] = __uint_as_float(t_rv[0] & 4294901760u);
                            t_out[2] = __uint_as_float(t_rv[1] << 16);
                            t_out[3] = __uint_as_float(t_rv[1] & 4294901760u);
                            t_out[4] = __uint_as_float(t_rv[2] << 16);
                            t_out[5] = __uint_as_float(t_rv[2] & 4294901760u);
                            t_out[6] = __uint_as_float(t_rv[3] << 16);
                            t_out[7] = __uint_as_float(t_rv[3] & 4294901760u);
                        }
                        unsigned int t_dst = co_vt + (unsigned int)(t_j * 1024 + lane * 32);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(t_dst), "r"(__float_as_uint(t_out[0])), "r"(__float_as_uint(t_out[1])), "r"(__float_as_uint(t_out[2])), "r"(__float_as_uint(t_out[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(t_dst + 16), "r"(__float_as_uint(t_out[4])), "r"(__float_as_uint(t_out[5])), "r"(__float_as_uint(t_out[6])), "r"(__float_as_uint(t_out[7])) : "memory");
                    }
                    __syncwarp();
                    mbarrier_arrive(index_empty_addr + (co_info_stage) * 8);
                    co_info_stage += 1;
                    if (co_info_stage == 2) { co_info_stage = 0; _phase_index_full_1 ^= 1; }
                    int _parity_o_full_2 = (int)((unsigned int)co_tile_cursor % 2) ^ 1;
                    mbarrier_wait(o_full_addr, _parity_o_full_2);
                    int _parity_o_full_3 = (int)((unsigned int)co_tile_cursor % 2) ^ 1;
                    mbarrier_wait(o_full_addr + 8, _parity_o_full_3);
                    int _parity_sum_ready = (int)((unsigned int)co_item % 2);
                    mbarrier_wait(sum_ready_addr, _parity_sum_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = smem_sum[co_thread] + smem_sum[co_thread ^ 64];
                    float final_max = smem_fmax[co_thread];
                    float ep_pt[1];
                    ep_pt[0] = ((0 < co_kw) ? smem_ptail[co_head] : 0.0f);
                    float ep_alpha = smem_ptail[64 + co_head];
                    mbarrier_arrive(sum_empty_addr);
                    int ep_row = co_query * 64 + co_head;
                    if (co_half == 0 && ep_row < num_query_tokens * 32) {
                        float _log2_14;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_14) : "f"(total_sum));
                        LSE[ep_row] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_14) : -CUDART_INF_F);
                    }
                    float _rcp_15 = approx_rcp(total_sum);
                    float ep_scale = ((total_sum > 0.0f) ? _rcp_15 : 0.0f) * output_scale;
                    float ep_w[1];
                    ep_w[0] = ep_pt[0] * ep_scale;
                    ep_scale = ep_scale * ep_alpha;
                    int ep_box_row = co_query * 64 + (co_thread >> 5 & 1) * 32;
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
                        const float2 _scale2_16 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_0)[_ls], _scale2_16);
                        float ep_v_0[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(0) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(4) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(8) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 8))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_0[(12) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 12))));
                        {
                            unsigned long long _fma_acc_scale2_17;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_17) : "f"(ep_w[0]));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[0]), "+f"(ep_buf_0[1]) : "f"(ep_v_0[0]), "f"(ep_v_0[1]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[2]), "+f"(ep_buf_0[3]) : "f"(ep_v_0[2]), "f"(ep_v_0[3]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[4]), "+f"(ep_buf_0[5]) : "f"(ep_v_0[4]), "f"(ep_v_0[5]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[6]), "+f"(ep_buf_0[7]) : "f"(ep_v_0[6]), "f"(ep_v_0[7]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[8]), "+f"(ep_buf_0[9]) : "f"(ep_v_0[8]), "f"(ep_v_0[9]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[10]), "+f"(ep_buf_0[11]) : "f"(ep_v_0[10]), "f"(ep_v_0[11]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[12]), "+f"(ep_buf_0[13]) : "f"(ep_v_0[12]), "f"(ep_v_0[13]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_0[14]), "+f"(ep_buf_0[15]) : "f"(ep_v_0[14]), "f"(ep_v_0[15]), "l"(_fma_acc_scale2_17));
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_0[_lp*2 + 0], ep_buf_0[_lp*2+1 + 0]));
                            ep_pack[_lp] = *(uint32_t*)&_bf2;
                        }
                        float ep_buf_1[16];
                        tmem_ld_x16(&ep_buf_1[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q + 16));
                        const float2 _scale2_18 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_1)[_ls], _scale2_18);
                        float ep_v_1[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(0) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 16))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(4) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 16 + 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(8) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 16 + 8))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_1[(12) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 16 + 12))));
                        {
                            unsigned long long _fma_acc_scale2_19;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_19) : "f"(ep_w[0]));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[0]), "+f"(ep_buf_1[1]) : "f"(ep_v_1[0]), "f"(ep_v_1[1]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[2]), "+f"(ep_buf_1[3]) : "f"(ep_v_1[2]), "f"(ep_v_1[3]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[4]), "+f"(ep_buf_1[5]) : "f"(ep_v_1[4]), "f"(ep_v_1[5]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[6]), "+f"(ep_buf_1[7]) : "f"(ep_v_1[6]), "f"(ep_v_1[7]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[8]), "+f"(ep_buf_1[9]) : "f"(ep_v_1[8]), "f"(ep_v_1[9]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[10]), "+f"(ep_buf_1[11]) : "f"(ep_v_1[10]), "f"(ep_v_1[11]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[12]), "+f"(ep_buf_1[13]) : "f"(ep_v_1[12]), "f"(ep_v_1[13]), "l"(_fma_acc_scale2_19));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_1[14]), "+f"(ep_buf_1[15]) : "f"(ep_v_1[14]), "f"(ep_v_1[15]), "l"(_fma_acc_scale2_19));
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_1[_lp*2 + 0], ep_buf_1[_lp*2+1 + 0]));
                            ep_pack[_lp + 8] = *(uint32_t*)&_bf2;
                        }
                        float ep_buf_2[16];
                        tmem_ld_x16(&ep_buf_2[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q + 32));
                        const float2 _scale2_20 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_2)[_ls], _scale2_20);
                        float ep_v_2[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(0) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 32))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(4) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 32 + 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(8) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 32 + 8))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_2[(12) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 32 + 12))));
                        {
                            unsigned long long _fma_acc_scale2_21;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_21) : "f"(ep_w[0]));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[0]), "+f"(ep_buf_2[1]) : "f"(ep_v_2[0]), "f"(ep_v_2[1]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[2]), "+f"(ep_buf_2[3]) : "f"(ep_v_2[2]), "f"(ep_v_2[3]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[4]), "+f"(ep_buf_2[5]) : "f"(ep_v_2[4]), "f"(ep_v_2[5]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[6]), "+f"(ep_buf_2[7]) : "f"(ep_v_2[6]), "f"(ep_v_2[7]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[8]), "+f"(ep_buf_2[9]) : "f"(ep_v_2[8]), "f"(ep_v_2[9]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[10]), "+f"(ep_buf_2[11]) : "f"(ep_v_2[10]), "f"(ep_v_2[11]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[12]), "+f"(ep_buf_2[13]) : "f"(ep_v_2[12]), "f"(ep_v_2[13]), "l"(_fma_acc_scale2_21));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_2[14]), "+f"(ep_buf_2[15]) : "f"(ep_v_2[14]), "f"(ep_v_2[15]), "l"(_fma_acc_scale2_21));
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ep_buf_2[_lp*2 + 0], ep_buf_2[_lp*2+1 + 0]));
                            ep_pack[_lp + 16] = *(uint32_t*)&_bf2;
                        }
                        float ep_buf_3[16];
                        tmem_ld_x16(&ep_buf_3[0], taddr + (unsigned int)co_lane_addr + (unsigned int)(128 * ep_part + 64 * ep_q + 48));
                        const float2 _scale2_22 = {ep_scale, ep_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ep_buf_3)[_ls], _scale2_22);
                        float ep_v_3[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(0) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 48))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(4) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 48 + 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(8) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 48 + 8))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ep_v_3[(12) + 3]))
                            : "r"(((0 < co_kw) ? co_vt + 0 : smem_vzero_addr) + (unsigned int)(4 * (128 * ep_part + 64 * ep_q + 48 + 12))));
                        {
                            unsigned long long _fma_acc_scale2_23;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_23) : "f"(ep_w[0]));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[0]), "+f"(ep_buf_3[1]) : "f"(ep_v_3[0]), "f"(ep_v_3[1]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[2]), "+f"(ep_buf_3[3]) : "f"(ep_v_3[2]), "f"(ep_v_3[3]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[4]), "+f"(ep_buf_3[5]) : "f"(ep_v_3[4]), "f"(ep_v_3[5]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[6]), "+f"(ep_buf_3[7]) : "f"(ep_v_3[6]), "f"(ep_v_3[7]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[8]), "+f"(ep_buf_3[9]) : "f"(ep_v_3[8]), "f"(ep_v_3[9]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[10]), "+f"(ep_buf_3[11]) : "f"(ep_v_3[10]), "f"(ep_v_3[11]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[12]), "+f"(ep_buf_3[13]) : "f"(ep_v_3[12]), "f"(ep_v_3[13]), "l"(_fma_acc_scale2_23));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(ep_buf_3[14]), "+f"(ep_buf_3[15]) : "f"(ep_v_3[14]), "f"(ep_v_3[15]), "l"(_fma_acc_scale2_23));
                        }
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
                            tma_store_2d((&tmap_o), 256 * ep_part + co_half * 128 + 64 * ep_q, ep_box_row, ep_box);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (ep_i == 1) {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(o_empty_addr);
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    co_item += 1;
                }
                int q_5 = co_ret;
                if (co_ret == -2) {
                    mbarrier_wait(index_full_addr + (co_info_stage) * 8, _phase_index_full_1);
                    q_5 = smem_indices[(int)co_info_stage * 272 + 256 + 15];
                }
                co_query = q_5;
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
            int mma_query = q_2;
            unsigned int _phase_index_full_2 = 0;
            #pragma unroll 1
            for (unsigned int mma_work = 0; mma_work < 1048576; mma_work++) {
                int mma_t0 = (mma_query - 0) * 2;
                int mma_ret = -1;
                if (mma_t0 < num_query_tokens) {
                    int _parity_q_empty = (int)((unsigned int)mma_item % 2) ^ 1;
                    mbarrier_wait(q_empty_addr, _parity_q_empty);
                    int _parity_q_read = (int)((unsigned int)mma_item % 2) ^ 1;
                    mbarrier_wait(q_read_addr, _parity_q_read);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_load_full_addr, 34816);
                        tma_3d_gmem2smem(smem_q_a_addr, (&tmap_q), 0, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_b_addr, (&tmap_q), 128, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_rope_addr, (&tmap_q), 224, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_rope_addr + 8192, (&tmap_q), 224, 0, mma_t0, q_load_full_addr);
                        tma_3d_gmem2smem(smem_q_sf_addr, (&tmap_q_sf), 352, 0, mma_t0, q_load_full_addr);
                    }
                    mbarrier_wait(index_full_addr + (mma_info_stage) * 8, _phase_index_full_2);
                    int shared = smem_indices[(int)mma_info_stage * 272 + 256];
                    int nb_read = 0;
                    int nxt = smem_indices[(int)mma_info_stage * 272 + 256 + 14];
                    mbarrier_arrive(index_empty_addr + (mma_info_stage) * 8);
                    mma_info_stage += 1;
                    if (mma_info_stage == 2) { mma_info_stage = 0; _phase_index_full_2 ^= 1; }
                    int nb_2 = ((num_query_tokens - mma_t0) < (2) ? (num_query_tokens - mma_t0) : (2));
                    int rest_2 = ((shared != 0) ? nb_2 - 1 : 0);
                    int tiles_2 = ((shared != 0) ? 2 : 2 * nb_2);
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
                        int mma_n = 64;
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
                        int _mma_a_lo_2 = make_warp_uniform((((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 1024);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_k_c_addr) >> 4) & 0x3FFF) + (mk_stage) * 1664);
                        {
                            uint64_t _mma_ss_a_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_scratch + (256 + m_stage * 64)), _mma_ss_a_desc_15, _mma_ss_b_desc_15, 135267472, 1);
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
                int q_2_1 = mma_ret;
                if (mma_ret == -2) {
                    mbarrier_wait(index_full_addr + (mma_info_stage) * 8, _phase_index_full_2);
                    q_2_1 = smem_indices[(int)mma_info_stage * 272 + 256 + 15];
                }
                mma_query = q_2_1;
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
            int ix_late = pc_fallback[0];
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int ix_work = 0; ix_work < 1048576; ix_work++) {
                if (ix_late == 0) {
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
                }
                int ix_next = -1;
                if (ix_query * 2 < num_query_tokens) {
                    int ix_next_0 = -1;
                    int ix_len[2];
                    int ix_rows[8];
                    int _min_2 = ((ix_query * 2) < (num_query_tokens - 1) ? (ix_query * 2) : (num_query_tokens - 1));
                    int _max_1 = ((sparse_topk_lens[_min_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[_min_2] + sparse_topk_lens_offset) : (0));
                    int _min_0_2 = ((_max_1) < (swa_width) ? (_max_1) : (swa_width));
                    ix_len[0] = _min_0_2;
                    int _min_3_1 = ((ix_query * 2) < (num_query_tokens - 1) ? (ix_query * 2) : (num_query_tokens - 1));
                    {
                        const int4* _ivptr_4 = reinterpret_cast<const int4*>(swa_indices + (_min_3_1 * swa_index_stride + lane * 4) + 0);
                        int4 _ivld_4;
                        _ivld_4 = *_ivptr_4;
                        ix_rows[0 + 0] = _ivld_4.x;
                        ix_rows[0 + 1] = _ivld_4.y;
                        ix_rows[0 + 2] = _ivld_4.z;
                        ix_rows[0 + 3] = _ivld_4.w;
                    }
                    int _min_7 = ((ix_query * 2 + 1) < (num_query_tokens - 1) ? (ix_query * 2 + 1) : (num_query_tokens - 1));
                    int _max_6 = ((sparse_topk_lens[_min_7] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[_min_7] + sparse_topk_lens_offset) : (0));
                    int _min_5 = ((_max_6) < (swa_width) ? (_max_6) : (swa_width));
                    ix_len[1] = _min_5;
                    int _min_8 = ((ix_query * 2 + 1) < (num_query_tokens - 1) ? (ix_query * 2 + 1) : (num_query_tokens - 1));
                    {
                        const int4* _ivptr_9 = reinterpret_cast<const int4*>(swa_indices + (_min_8 * swa_index_stride + lane * 4) + 0);
                        int4 _ivld_9;
                        _ivld_9 = *_ivptr_9;
                        ix_rows[4 + 0] = _ivld_9.x;
                        ix_rows[4 + 1] = _ivld_9.y;
                        ix_rows[4 + 2] = _ivld_9.z;
                        ix_rows[4 + 3] = _ivld_9.w;
                    }
                    int ix_nb = ((num_query_tokens - ix_query * 2) < (2) ? (num_query_tokens - ix_query * 2) : (2));
                    if (0 >= ix_nb) {
                        ix_len[0] = 0;
                    }
                    if (1 >= ix_nb) {
                        ix_len[1] = 0;
                    }
                    int ix_nf = __shfl_sync(0xFFFFFFFF, (int)(ix_rows[5] == ix_rows[0] && ix_rows[4] != ix_rows[1]), 0);
                    if (ix_nf == 0) {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        unsigned int ix_stage_base = smem_indices_addr + ix_stage * 1088;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(lane * 16)), "r"(ix_rows[0]), "r"(ix_rows[1]), "r"(ix_rows[2]), "r"(ix_rows[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(512 + lane * 16)), "r"(-1), "r"(-1), "r"(-1), "r"(-1) : "memory");
                        __syncwarp();
                        if (lane == 31) {
                            smem_indices[(int)ix_stage * 272 + 127 + 1] = ((1 < ix_nb) ? ix_rows[7] : -1);
                        }
                        __syncwarp();
                        int ix_bad = ix_len[0] != 128;
                        if (1 < ix_nb) {
                            if (ix_len[1] != 128) {
                                ix_bad = 1;
                            }
                            int ix_want = smem_indices[(int)ix_stage * 272 + 1 + lane * 4];
                            if (!(ix_rows[4] == ix_want)) {
                                ix_bad = 1;
                            }
                            int ix_want_0 = smem_indices[(int)ix_stage * 272 + 1 + lane * 4 + 1];
                            if (!(ix_rows[5] == ix_want_0)) {
                                ix_bad = 1;
                            }
                            int ix_want_1 = smem_indices[(int)ix_stage * 272 + 1 + lane * 4 + 2];
                            if (!(ix_rows[6] == ix_want_1)) {
                                ix_bad = 1;
                            }
                            int ix_want_2 = smem_indices[(int)ix_stage * 272 + 1 + lane * 4 + 3];
                            if (!(ix_rows[7] == ix_want_2)) {
                                ix_bad = 1;
                            }
                        }
                        int _warp_vote_10 = __any_sync(0xFFFFFFFF, ix_bad != 0);
                        int ix_shared = _warp_vote_10 == 0;
                        if (ix_shared == 0) {
                            __syncwarp();
                            int ix_own_0[4];
                            #pragma unroll
                            for (int ix_j_0 = 0; ix_j_0 < 4; ix_j_0++) {
                                ix_own_0[ix_j_0] = ((lane * 4 + ix_j_0 < ix_len[0]) ? ix_rows[ix_j_0] : -1);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(lane * 16)), "r"(ix_own_0[0]), "r"(ix_own_0[1]), "r"(ix_own_0[2]), "r"(ix_own_0[3]) : "memory");
                            int ix_own_1[4];
                            #pragma unroll
                            for (int ix_j_1 = 0; ix_j_1 < 4; ix_j_1++) {
                                ix_own_1[ix_j_1] = ((lane * 4 + ix_j_1 < ix_len[1]) ? ix_rows[4 + ix_j_1] : -1);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base + (unsigned int)(512 + lane * 16)), "r"(ix_own_1[0]), "r"(ix_own_1[1]), "r"(ix_own_1[2]), "r"(ix_own_1[3]) : "memory");
                        }
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 272 + 256] = ix_shared;
                            smem_indices[(int)ix_stage * 272 + 256 + 15] = ix_query;
                            int ix_lo_0 = ((ix_shared != 0) ? 0 : 0);
                            int ix_span_0 = ix_len[0];
                            smem_indices[(int)ix_stage * 272 + 256 + 2] = ix_lo_0;
                            smem_indices[(int)ix_stage * 272 + 256 + 3] = ix_span_0;
                            int ix_lo_1 = ((ix_shared != 0) ? 1 : 128);
                            int ix_span_1 = ix_len[1];
                            smem_indices[(int)ix_stage * 272 + 256 + 2 + 2] = ix_lo_1;
                            smem_indices[(int)ix_stage * 272 + 256 + 3 + 2] = ix_span_1;
                        }
                    } else {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        unsigned int ix_stage_base_1 = smem_indices_addr + ix_stage * 1088;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(lane * 16)), "r"(ix_rows[0]), "r"(ix_rows[1]), "r"(ix_rows[2]), "r"(ix_rows[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(512 + lane * 16)), "r"(-1), "r"(-1), "r"(-1), "r"(-1) : "memory");
                        __syncwarp();
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 272 + 127 + 1] = ((1 < ix_nb) ? ix_rows[4] : -1);
                        }
                        __syncwarp();
                        int ix_bad_1 = ix_len[0] != 128;
                        if (1 < ix_nb) {
                            if (ix_len[1] != 128) {
                                ix_bad_1 = 1;
                            }
                            int _max_11 = ((lane * 4 + -1) > (0) ? (lane * 4 + -1) : (0));
                            int ix_want_3 = smem_indices[(int)ix_stage * 272 + _max_11];
                            if (lane == 0) {
                                ix_want_3 = ix_rows[4];
                            }
                            if (!(ix_rows[4] == ix_want_3)) {
                                ix_bad_1 = 1;
                            }
                            int ix_want_3_1 = smem_indices[(int)ix_stage * 272 + -1 + lane * 4 + 1];
                            if (!(ix_rows[5] == ix_want_3_1)) {
                                ix_bad_1 = 1;
                            }
                            int ix_want_4 = smem_indices[(int)ix_stage * 272 + -1 + lane * 4 + 2];
                            if (!(ix_rows[6] == ix_want_4)) {
                                ix_bad_1 = 1;
                            }
                            int ix_want_5 = smem_indices[(int)ix_stage * 272 + -1 + lane * 4 + 3];
                            if (!(ix_rows[7] == ix_want_5)) {
                                ix_bad_1 = 1;
                            }
                        }
                        int _warp_vote_12 = __any_sync(0xFFFFFFFF, ix_bad_1 != 0);
                        int ix_shared_1 = _warp_vote_12 == 0;
                        if (ix_shared_1 == 0) {
                            __syncwarp();
                            int ix_own_2[4];
                            #pragma unroll
                            for (int ix_j_2 = 0; ix_j_2 < 4; ix_j_2++) {
                                ix_own_2[ix_j_2] = ((lane * 4 + ix_j_2 < ix_len[0]) ? ix_rows[ix_j_2] : -1);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(lane * 16)), "r"(ix_own_2[0]), "r"(ix_own_2[1]), "r"(ix_own_2[2]), "r"(ix_own_2[3]) : "memory");
                            int ix_own_3[4];
                            #pragma unroll
                            for (int ix_j_3 = 0; ix_j_3 < 4; ix_j_3++) {
                                ix_own_3[ix_j_3] = ((lane * 4 + ix_j_3 < ix_len[1]) ? ix_rows[4 + ix_j_3] : -1);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ix_stage_base_1 + (unsigned int)(512 + lane * 16)), "r"(ix_own_3[0]), "r"(ix_own_3[1]), "r"(ix_own_3[2]), "r"(ix_own_3[3]) : "memory");
                        }
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 272 + 256] = ix_shared_1;
                            smem_indices[(int)ix_stage * 272 + 256 + 15] = ix_query;
                            int ix_lo_2 = ((ix_shared_1 != 0) ? 0 : 0);
                            int ix_span_2 = ix_len[0];
                            smem_indices[(int)ix_stage * 272 + 256 + 2] = ix_lo_2;
                            smem_indices[(int)ix_stage * 272 + 256 + 3] = ix_span_2;
                            int ix_lo_3 = ((ix_shared_1 != 0) ? 0 : 128);
                            int ix_span_3 = ix_len[1];
                            if (ix_shared_1 != 0) {
                                ix_span_3 = 127;
                            }
                            smem_indices[(int)ix_stage * 272 + 256 + 2 + 2] = ix_lo_3;
                            smem_indices[(int)ix_stage * 272 + 256 + 3 + 2] = ix_span_3;
                        }
                    }
                    int ix_next_late = -2;
                    if (1 - ix_late != 0) {
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
                        ix_next_0 = ((ixn_valid != 0) ? (int)ixn_ctaid : -1);
                        ix_next_late = ix_next_0;
                    }
                    if (lane == 0) {
                        smem_indices[(int)ix_stage * 272 + 256 + 14] = ix_next_late;
                    }
                    mbarrier_arrive(index_full_addr + (ix_stage) * 8);
                    ix_stage += 1;
                    if (ix_stage == 2) { ix_stage = 0; _phase_index_empty ^= 1; }
                    ix_next = ix_next_0;
                }
                if (ix_late != 0) {
                    ix_late = 0;
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
                    uint32_t ixn_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(ixn_valid_1)
                        : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t ixn_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(ixn_ctaid_1)
                        : "r"(work_response_addr + ix_work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (ix_work_stage) * 8);
                    ix_work_stage += 1;
                    if (ix_work_stage == 2) { ix_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    ix_next = ((ixn_valid_1 != 0) ? (int)ixn_ctaid_1 : -1);
                    if (ix_next < 0) {
                        mbarrier_wait(index_empty_addr + (ix_stage) * 8, _phase_index_empty);
                        if (lane == 0) {
                            smem_indices[(int)ix_stage * 272 + 256 + 15] = -1;
                        }
                        mbarrier_arrive(index_full_addr + (ix_stage) * 8);
                        ix_stage += 1;
                        if (ix_stage == 2) { ix_stage = 0; _phase_index_empty ^= 1; }
                    }
                }
                if (ix_next < 0) {
                    break;
                }
                ix_query = ix_next;
            }
        }
    // ---- Role: pv_issuer ----
    } else if (warp == 11) {
        { // pv_issuer_main
            int pv_tile_cursor = 0;
            unsigned int pv_info_stage = 0;
            int q_3 = blockIdx.x;
            int pv_query = q_3;
            unsigned int _phase_index_full_3 = 0;
            #pragma unroll 1
            for (unsigned int pv_work = 0; pv_work < 1048576; pv_work++) {
                int pv_ret = -1;
                if (pv_query * 2 < num_query_tokens) {
                    mbarrier_wait(index_full_addr + (pv_info_stage) * 8, _phase_index_full_3);
                    int shared_1 = smem_indices[(int)pv_info_stage * 272 + 256];
                    int nb_read_1 = 0;
                    int nxt_1 = smem_indices[(int)pv_info_stage * 272 + 256 + 14];
                    mbarrier_arrive(index_empty_addr + (pv_info_stage) * 8);
                    pv_info_stage += 1;
                    if (pv_info_stage == 2) { pv_info_stage = 0; _phase_index_full_3 ^= 1; }
                    int nb_3 = ((num_query_tokens - pv_query * 2) < (2) ? (num_query_tokens - pv_query * 2) : (2));
                    int rest_3 = ((shared_1 != 0) ? nb_3 - 1 : 0);
                    int tiles_3 = ((shared_1 != 0) ? 2 : 2 * nb_3);
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
                        int pv_narrow = 0;
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
                int q_3_1 = pv_ret;
                if (pv_ret == -2) {
                    mbarrier_wait(index_full_addr + (pv_info_stage) * 8, _phase_index_full_3);
                    q_3_1 = smem_indices[(int)pv_info_stage * 272 + 256 + 15];
                }
                pv_query = q_3_1;
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
            int q_6 = blockIdx.x;
            int ld_query = q_6;
            unsigned int _phase_index_full_4 = 0;
            #pragma unroll 1
            for (unsigned int ld_work = 0; ld_work < 1048576; ld_work++) {
                int ld_next = -1;
                if (ld_query * 2 < num_query_tokens) {
                    mbarrier_wait(index_full_addr + (ld_info_stage) * 8, _phase_index_full_4);
                    int shared_2 = smem_indices[(int)ld_info_stage * 272 + 256];
                    int nb_read_2 = 0;
                    int nxt_2 = smem_indices[(int)ld_info_stage * 272 + 256 + 14];
                    mbarrier_arrive(index_empty_addr + (ld_info_stage) * 8);
                    ld_info_stage += 1;
                    if (ld_info_stage == 2) { ld_info_stage = 0; _phase_index_full_4 ^= 1; }
                    int nb_4 = ((num_query_tokens - ld_query * 2) < (2) ? (num_query_tokens - ld_query * 2) : (2));
                    int rest_4 = ((shared_2 != 0) ? nb_4 - 1 : 0);
                    int tiles_4 = ((shared_2 != 0) ? 2 : 2 * nb_4);
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
                            for (int sf_s_1 = 0; sf_s_1 < 7; sf_s_1++) {
                                k68 = k68 | sf_a[sf_s_1] + 404232216 | sf_b[sf_s_1] + 404232216;
                                k70 = k70 | sf_a[sf_s_1] + 269488144 | sf_b[sf_s_1] + 269488144;
                                k78 = k78 | sf_a[sf_s_1] + 134744072 | sf_b[sf_s_1] + 134744072;
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
                                    uint32_t _cvt_f16x2_e2m1x2_5_1;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_5_1) : "r"((uint32_t)(v_word >> 16)));
                                    uint32_t f16x2_mul_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_1) : "r"(_cvt_f16x2_e2m1x2_5_1), "r"(v_sc_w_1));
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
                            #pragma unroll 1
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
                int q_1_1 = ld_next;
                if (ld_next == -2) {
                    mbarrier_wait(index_full_addr + (ld_info_stage) * 8, _phase_index_full_4);
                    q_1_1 = smem_indices[(int)ld_info_stage * 272 + 256 + 15];
                }
                ld_query = q_1_1;
                if (ld_query < 0) {
                    break;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: gather ----
    } else if (warp >= 20 && warp <= 23) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // gather_main
            const int g_warp = warp - 20;
            const int g_early = (int)(blockIdx.x >= 0);
            int g_tile_cursor = 0;
            int g_item = 0;
            unsigned int g_index_stage = 0;
            int q_7 = blockIdx.x;
            int g_query = q_7;
            int g_prev_nt = 0;
            asm volatile("barrier.sync 11, 128;" ::: "memory");
            int _max_0 = ((smem_first[128] + sparse_topk_lens_offset) > (0) ? (smem_first[128] + sparse_topk_lens_offset) : (0));
            int gp_len = ((_max_0) < (swa_width) ? (_max_0) : (swa_width));
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
            unsigned int _phase_index_full_5 = 0;
            unsigned int _phase_tail_full = 1;
            #pragma unroll 1
            for (unsigned int g_work = 0; g_work < 1048576; g_work++) {
                int g_ret = 0;
                int g_next = -1;
                if (g_query * 2 < num_query_tokens) {
                    mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_5);
                    int nb_5 = ((num_query_tokens - g_query * 2) < (2) ? (num_query_tokens - g_query * 2) : (2));
                    int rest_5 = ((smem_indices[(int)g_index_stage * 272 + 256] != 0) ? nb_5 - 1 : 0);
                    int tiles_5 = ((smem_indices[(int)g_index_stage * 272 + 256] != 0) ? 2 : 2 * nb_5);
                    g_next = smem_indices[(int)g_index_stage * 272 + 256 + 14];
                    if (g_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(tail_full_addr + (g_index_stage) * 8, 1664 * (int)(rest_5 > 0));
                        }
                    }
                    if (rest_5 > 0 && lane == 16) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int t_rows[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_rows[(0) + 3]))
                            : "r"(smem_indices_addr + g_index_stage * 1088 + 512));
                        int t_data_rows[4];
                        int t_foot_rows[4];
                        #pragma unroll
                        for (int t_j_1 = 0; t_j_1 < 4; t_j_1++) {
                            int t_tok = ((t_rows[t_j_1] >= 0) ? t_rows[t_j_1] : 0);
                            int t_page = t_tok >> swa_page_log2;
                            int t_pslot = t_tok - (t_page << swa_page_log2);
                            t_data_rows[t_j_1] = ((t_rows[t_j_1] >= 0) ? t_page * swa_pitch_units + t_pslot * 11 : 1073741824);
                            t_foot_rows[t_j_1] = ((t_rows[t_j_1] >= 0) ? t_page * swa_pitch_units + swa_footer_units + t_pslot : 1073741824);
                        }
                        int t_base_1 = smem_kt_addr + g_index_stage * 2048;
                        if (g_warp < 3) {
                            tma_gather4_gmem2smem(t_base_1 + g_warp * 512, (&tmap_swa_kv), ((g_warp == 2) ? 224 : g_warp * 128), t_data_rows[0], t_data_rows[1], t_data_rows[2], t_data_rows[3], tail_full_addr + (g_index_stage) * 8);
                        } else {
                            tma_gather4_gmem2smem(t_base_1 + 1536, (&tmap_swa_sf), 0, t_foot_rows[0], t_foot_rows[1], t_foot_rows[2], t_foot_rows[3], tail_full_addr + (g_index_stage) * 8);
                        }
                    }
                    #pragma unroll 1
                    for (int g_tile = 2 * (int)(g_item == 0); g_tile < tiles_5; g_tile++) {
                        int g_stage = (g_tile_cursor + g_tile) % 3;
                        int _stage_k_empty_2 = (int)((unsigned int)(g_tile_cursor + g_tile) % 3);
                        int _parity_k_empty_2 = (int)((unsigned int)(g_tile_cursor + g_tile) / 3 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty_2) * 8, _parity_k_empty_2);
                        int _stage_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + g_tile) % 3);
                        int _parity_kstage_free_1 = (int)((unsigned int)(g_tile_cursor + g_tile) / 3 % 2) ^ 1;
                        mbarrier_wait(kstage_free_addr + (_stage_kstage_free_1) * 8, _parity_kstage_free_1);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int g_narrow = 0;
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
                                : "r"(smem_indices_addr + g_index_stage * 1088 + (unsigned int)((g_tile * 64 + g_group * 4) * 4)));
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
                        if (g_tile == 0 && g_item > 0) {
                            int _parity_q_load_full_1 = (int)((unsigned int)g_item % 2) ^ 1;
                            mbarrier_wait(q_load_full_addr, _parity_q_load_full_1);
                            mbarrier_wait(tail_full_addr + (((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0)) * 8, ((g_index_stage + 1 < 2) ? _phase_tail_full : _phase_tail_full ^ 1));
                            unsigned int t_kb = smem_kt_addr + ((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0) * 2048;
                            int t_part = g_warp >> 1;
                            int t_c0 = (g_warp & 1) * 4;
                            int t_c1 = ((t_c0 + 4) < (8 - 2 * t_part) ? (t_c0 + 4) : (8 - 2 * t_part));
                            #pragma unroll 1
                            for (int t_u = 0; t_u < 1; t_u++) {
                                int t_grp = t_u >= 0;
                                int t_j_2 = t_u;
                                if (t_j_2 < g_prev_nt) {
                                    int t_row_2 = t_grp * 32 + lane;
                                    float t_acc = 0.0f;
                                    unsigned int t_q_row = ((t_part != 0) ? smem_q_b_addr : smem_q_a_addr) + (unsigned int)(t_row_2 * 128);
                                    unsigned int t_k_row = t_kb + (unsigned int)(t_part * 512 + t_j_2 * 128);
                                    unsigned int t_q_sf = smem_q_sf_addr + (unsigned int)(t_row_2 * 32 + t_part * 16);
                                    unsigned int t_k_sf = t_kb + (unsigned int)(1536 + t_j_2 * 32 + t_part * 16);
                                    #pragma unroll 1
                                    for (int t_c = t_c0; t_c < t_c1; t_c++) {
                                        unsigned int t_qv[4];
                                        unsigned int t_kv[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&t_qv[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv[(0) + 3]))
                                            : "r"(t_q_row + (unsigned int)((t_c ^ t_row_2 & 7) << 4)));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&t_kv[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv[(0) + 3]))
                                            : "r"(t_k_row + (unsigned int)((t_c ^ (int)(t_k_row >> 7 & 7)) << 4)));
                                        unsigned int t_qs[1];
                                        unsigned int t_ks[1];
                                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_qs[0])) : "r"(t_q_sf + (unsigned int)(t_c >> 1 << 2)));
                                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_ks[0])) : "r"(t_k_sf + (unsigned int)(t_c >> 1 << 2)));
                                        uint32_t t_qsc;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(t_qsc) : "h"((uint16_t)(t_qs[0] >> (unsigned int)((t_c & 1) * 16))));
                                        uint32_t t_ksc;
                                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(t_ksc) : "h"((uint16_t)(t_ks[0] >> (unsigned int)((t_c & 1) * 16))));
                                        unsigned int t_h2_0 = 0;
                                        uint32_t _cvt_f16x2_e2m1x2_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_1) : "r"((uint32_t)(t_qv[0] >> 0)));
                                        uint32_t _cvt_f16x2_e2m1x2_2_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_2_1) : "r"((uint32_t)(t_kv[0] >> 0)));
                                        uint32_t t_fma_0;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_0) : "r"(_cvt_f16x2_e2m1x2_1), "r"(_cvt_f16x2_e2m1x2_2_1), "r"(t_h2_0));
                                        t_h2_0 = t_fma_0;
                                        uint32_t _cvt_f16x2_e2m1x2_3_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_3_1) : "r"((uint32_t)(t_qv[0] >> 8)));
                                        uint32_t _cvt_f16x2_e2m1x2_4_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_4_1) : "r"((uint32_t)(t_kv[0] >> 8)));
                                        uint32_t t_fma_1;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_1) : "r"(_cvt_f16x2_e2m1x2_3_1), "r"(_cvt_f16x2_e2m1x2_4_1), "r"(t_h2_0));
                                        t_h2_0 = t_fma_1;
                                        uint32_t _cvt_f16x2_e2m1x2_5_2;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_5_2) : "r"((uint32_t)(t_qv[0] >> 16)));
                                        uint32_t _cvt_f16x2_e2m1x2_6_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_6_1) : "r"((uint32_t)(t_kv[0] >> 16)));
                                        uint32_t t_fma_2;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_2) : "r"(_cvt_f16x2_e2m1x2_5_2), "r"(_cvt_f16x2_e2m1x2_6_1), "r"(t_h2_0));
                                        t_h2_0 = t_fma_2;
                                        uint32_t _cvt_f16x2_e2m1x2_7;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_7) : "r"((uint32_t)(t_qv[0] >> 24)));
                                        uint32_t _cvt_f16x2_e2m1x2_8_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_8_1) : "r"((uint32_t)(t_kv[0] >> 24)));
                                        uint32_t t_fma_3;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_3) : "r"(_cvt_f16x2_e2m1x2_7), "r"(_cvt_f16x2_e2m1x2_8_1), "r"(t_h2_0));
                                        t_h2_0 = t_fma_3;
                                        uint32_t _cvt_f16x2_e2m1x2_9;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_9) : "r"((uint32_t)(t_qv[1] >> 0)));
                                        uint32_t _cvt_f16x2_e2m1x2_10;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_10) : "r"((uint32_t)(t_kv[1] >> 0)));
                                        uint32_t t_fma_4;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_4) : "r"(_cvt_f16x2_e2m1x2_9), "r"(_cvt_f16x2_e2m1x2_10), "r"(t_h2_0));
                                        t_h2_0 = t_fma_4;
                                        uint32_t _cvt_f16x2_e2m1x2_11_1;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_11_1) : "r"((uint32_t)(t_qv[1] >> 8)));
                                        uint32_t _cvt_f16x2_e2m1x2_12;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_12) : "r"((uint32_t)(t_kv[1] >> 8)));
                                        uint32_t t_fma_5;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_5) : "r"(_cvt_f16x2_e2m1x2_11_1), "r"(_cvt_f16x2_e2m1x2_12), "r"(t_h2_0));
                                        t_h2_0 = t_fma_5;
                                        uint32_t _cvt_f16x2_e2m1x2_13;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_13) : "r"((uint32_t)(t_qv[1] >> 16)));
                                        uint32_t _cvt_f16x2_e2m1x2_14;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_14) : "r"((uint32_t)(t_kv[1] >> 16)));
                                        uint32_t t_fma_6;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_6) : "r"(_cvt_f16x2_e2m1x2_13), "r"(_cvt_f16x2_e2m1x2_14), "r"(t_h2_0));
                                        t_h2_0 = t_fma_6;
                                        uint32_t _cvt_f16x2_e2m1x2_15;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_15) : "r"((uint32_t)(t_qv[1] >> 24)));
                                        uint32_t _cvt_f16x2_e2m1x2_16;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_16) : "r"((uint32_t)(t_kv[1] >> 24)));
                                        uint32_t t_fma_7;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_7) : "r"(_cvt_f16x2_e2m1x2_15), "r"(_cvt_f16x2_e2m1x2_16), "r"(t_h2_0));
                                        t_h2_0 = t_fma_7;
                                        float _cvt_f32_f16_17;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(t_h2_0)));
                                        float _cvt_f32_f16_18;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(t_h2_0 >> 16)));
                                        float t_blk_0 = _cvt_f32_f16_17 + _cvt_f32_f16_18;
                                        float _cvt_f32_f16_19;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(t_qsc >> 0)));
                                        float _cvt_f32_f16_20;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(t_ksc >> 0)));
                                        float t_sab_0 = _cvt_f32_f16_19 * _cvt_f32_f16_20;
                                        t_acc = __fmaf_rn(t_blk_0, t_sab_0, t_acc);
                                        unsigned int t_h2_1 = 0;
                                        uint32_t _cvt_f16x2_e2m1x2_21;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_21) : "r"((uint32_t)(t_qv[2] >> 0)));
                                        uint32_t _cvt_f16x2_e2m1x2_22;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_22) : "r"((uint32_t)(t_kv[2] >> 0)));
                                        uint32_t t_fma_8;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_8) : "r"(_cvt_f16x2_e2m1x2_21), "r"(_cvt_f16x2_e2m1x2_22), "r"(t_h2_1));
                                        t_h2_1 = t_fma_8;
                                        uint32_t _cvt_f16x2_e2m1x2_23;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_23) : "r"((uint32_t)(t_qv[2] >> 8)));
                                        uint32_t _cvt_f16x2_e2m1x2_24;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_24) : "r"((uint32_t)(t_kv[2] >> 8)));
                                        uint32_t t_fma_9;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_9) : "r"(_cvt_f16x2_e2m1x2_23), "r"(_cvt_f16x2_e2m1x2_24), "r"(t_h2_1));
                                        t_h2_1 = t_fma_9;
                                        uint32_t _cvt_f16x2_e2m1x2_25;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_25) : "r"((uint32_t)(t_qv[2] >> 16)));
                                        uint32_t _cvt_f16x2_e2m1x2_26;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_26) : "r"((uint32_t)(t_kv[2] >> 16)));
                                        uint32_t t_fma_10;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_10) : "r"(_cvt_f16x2_e2m1x2_25), "r"(_cvt_f16x2_e2m1x2_26), "r"(t_h2_1));
                                        t_h2_1 = t_fma_10;
                                        uint32_t _cvt_f16x2_e2m1x2_27;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_27) : "r"((uint32_t)(t_qv[2] >> 24)));
                                        uint32_t _cvt_f16x2_e2m1x2_28;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_28) : "r"((uint32_t)(t_kv[2] >> 24)));
                                        uint32_t t_fma_11;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_11) : "r"(_cvt_f16x2_e2m1x2_27), "r"(_cvt_f16x2_e2m1x2_28), "r"(t_h2_1));
                                        t_h2_1 = t_fma_11;
                                        uint32_t _cvt_f16x2_e2m1x2_29;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_29) : "r"((uint32_t)(t_qv[3] >> 0)));
                                        uint32_t _cvt_f16x2_e2m1x2_30;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_30) : "r"((uint32_t)(t_kv[3] >> 0)));
                                        uint32_t t_fma_12;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_12) : "r"(_cvt_f16x2_e2m1x2_29), "r"(_cvt_f16x2_e2m1x2_30), "r"(t_h2_1));
                                        t_h2_1 = t_fma_12;
                                        uint32_t _cvt_f16x2_e2m1x2_31;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_31) : "r"((uint32_t)(t_qv[3] >> 8)));
                                        uint32_t _cvt_f16x2_e2m1x2_32;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_32) : "r"((uint32_t)(t_kv[3] >> 8)));
                                        uint32_t t_fma_13;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_13) : "r"(_cvt_f16x2_e2m1x2_31), "r"(_cvt_f16x2_e2m1x2_32), "r"(t_h2_1));
                                        t_h2_1 = t_fma_13;
                                        uint32_t _cvt_f16x2_e2m1x2_33;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_33) : "r"((uint32_t)(t_qv[3] >> 16)));
                                        uint32_t _cvt_f16x2_e2m1x2_34;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_34) : "r"((uint32_t)(t_kv[3] >> 16)));
                                        uint32_t t_fma_14;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_14) : "r"(_cvt_f16x2_e2m1x2_33), "r"(_cvt_f16x2_e2m1x2_34), "r"(t_h2_1));
                                        t_h2_1 = t_fma_14;
                                        uint32_t _cvt_f16x2_e2m1x2_35;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_35) : "r"((uint32_t)(t_qv[3] >> 24)));
                                        uint32_t _cvt_f16x2_e2m1x2_36;
                                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_36) : "r"((uint32_t)(t_kv[3] >> 24)));
                                        uint32_t t_fma_15;
                                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_15) : "r"(_cvt_f16x2_e2m1x2_35), "r"(_cvt_f16x2_e2m1x2_36), "r"(t_h2_1));
                                        t_h2_1 = t_fma_15;
                                        float _cvt_f32_f16_37;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_37) : "h"((uint16_t)(t_h2_1)));
                                        float _cvt_f32_f16_38;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_38) : "h"((uint16_t)(t_h2_1 >> 16)));
                                        float t_blk_1 = _cvt_f32_f16_37 + _cvt_f32_f16_38;
                                        float _cvt_f32_f16_39;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(t_qsc >> 16)));
                                        float _cvt_f32_f16_40;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_40) : "h"((uint16_t)(t_ksc >> 16)));
                                        float t_sab_1 = _cvt_f32_f16_39 * _cvt_f32_f16_40;
                                        t_acc = __fmaf_rn(t_blk_1, t_sab_1, t_acc);
                                    }
                                    unsigned int t_qr_row = smem_q_rope_addr + (unsigned int)(t_row_2 * 128);
                                    unsigned int t_kr_row = t_kb + (unsigned int)(1024 + t_j_2 * 128);
                                    #pragma unroll 1
                                    for (int t_c_1 = 0; t_c_1 < 8 * (int)(g_warp == 3); t_c_1++) {
                                        unsigned int t_qv_1[4];
                                        unsigned int t_kv_1[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&t_qv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_1[(0) + 3]))
                                            : "r"(t_qr_row + (unsigned int)((t_c_1 ^ t_row_2 & 7) << 4)));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&t_kv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_1[(0) + 3]))
                                            : "r"(t_kr_row + (unsigned int)((t_c_1 ^ (int)(t_kr_row >> 7 & 7)) << 4)));
                                        float _dot_41;
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                            "}\n"
                                            : "=f"(_dot_41) : "r"(t_qv_1[0]), "r"(t_kv_1[0]), "f"(t_acc));
                                        t_acc = _dot_41;
                                        float _dot_42;
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                            "}\n"
                                            : "=f"(_dot_42) : "r"(t_qv_1[1]), "r"(t_kv_1[1]), "f"(t_acc));
                                        t_acc = _dot_42;
                                        float _dot_43;
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                            "}\n"
                                            : "=f"(_dot_43) : "r"(t_qv_1[2]), "r"(t_kv_1[2]), "f"(t_acc));
                                        t_acc = _dot_43;
                                        float _dot_44;
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                            "}\n"
                                            : "=f"(_dot_44) : "r"(t_qv_1[3]), "r"(t_kv_1[3]), "f"(t_acc));
                                        t_acc = _dot_44;
                                    }
                                    smem_tdot[(int)((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0) * 128 + t_u * 128 + g_warp * 32 + lane] = t_acc;
                                }
                            }
                            mbarrier_arrive(tdot_full_addr + (((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0)) * 8);
                            mbarrier_arrive(q_read_addr);
                        }
                    }
                    mbarrier_arrive(index_empty_addr + (g_index_stage) * 8);
                    g_item += 1;
                    g_index_stage += 1;
                    if (g_index_stage == 2) { g_index_stage = 0; _phase_index_full_5 ^= 1; _phase_tail_full ^= 1; }
                    g_tile_cursor += tiles_5;
                    g_ret = rest_5;
                }
                g_prev_nt = g_ret;
                int q_0 = g_next;
                if (g_next == -2) {
                    mbarrier_wait(index_full_addr + (g_index_stage) * 8, _phase_index_full_5);
                    q_0 = smem_indices[(int)g_index_stage * 272 + 256 + 15];
                }
                g_query = q_0;
                if (g_query < 0) {
                    break;
                }
            }
            if (g_item > 0) {
                int _parity_q_load_full_2 = (int)((unsigned int)g_item % 2) ^ 1;
                mbarrier_wait(q_load_full_addr, _parity_q_load_full_2);
                mbarrier_wait(tail_full_addr + (((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0)) * 8, ((g_index_stage + 1 < 2) ? _phase_tail_full : _phase_tail_full ^ 1));
                unsigned int t_kb_1 = smem_kt_addr + ((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0) * 2048;
                int t_part_1 = g_warp >> 1;
                int t_c0_1 = (g_warp & 1) * 4;
                int t_c1_1 = ((t_c0_1 + 4) < (8 - 2 * t_part_1) ? (t_c0_1 + 4) : (8 - 2 * t_part_1));
                #pragma unroll 1
                for (int t_u_1 = 0; t_u_1 < 1; t_u_1++) {
                    int t_grp_1 = t_u_1 >= 0;
                    int t_j_3 = t_u_1;
                    if (t_j_3 < g_prev_nt) {
                        int t_row_3 = t_grp_1 * 32 + lane;
                        float t_acc_1 = 0.0f;
                        unsigned int t_q_row_1 = ((t_part_1 != 0) ? smem_q_b_addr : smem_q_a_addr) + (unsigned int)(t_row_3 * 128);
                        unsigned int t_k_row_1 = t_kb_1 + (unsigned int)(t_part_1 * 512 + t_j_3 * 128);
                        unsigned int t_q_sf_1 = smem_q_sf_addr + (unsigned int)(t_row_3 * 32 + t_part_1 * 16);
                        unsigned int t_k_sf_1 = t_kb_1 + (unsigned int)(1536 + t_j_3 * 32 + t_part_1 * 16);
                        #pragma unroll 1
                        for (int t_c_2 = t_c0_1; t_c_2 < t_c1_1; t_c_2++) {
                            unsigned int t_qv_2[4];
                            unsigned int t_kv_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_qv_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_2[(0) + 3]))
                                : "r"(t_q_row_1 + (unsigned int)((t_c_2 ^ t_row_3 & 7) << 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_kv_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_2[(0) + 3]))
                                : "r"(t_k_row_1 + (unsigned int)((t_c_2 ^ (int)(t_k_row_1 >> 7 & 7)) << 4)));
                            unsigned int t_qs_1[1];
                            unsigned int t_ks_1[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_qs_1[0])) : "r"(t_q_sf_1 + (unsigned int)(t_c_2 >> 1 << 2)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&t_ks_1[0])) : "r"(t_k_sf_1 + (unsigned int)(t_c_2 >> 1 << 2)));
                            uint32_t t_qsc_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(t_qsc_1) : "h"((uint16_t)(t_qs_1[0] >> (unsigned int)((t_c_2 & 1) * 16))));
                            uint32_t t_ksc_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(t_ksc_1) : "h"((uint16_t)(t_ks_1[0] >> (unsigned int)((t_c_2 & 1) * 16))));
                            unsigned int t_h2_2 = 0;
                            uint32_t _cvt_f16x2_e2m1x2_45;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_45) : "r"((uint32_t)(t_qv_2[0] >> 0)));
                            uint32_t _cvt_f16x2_e2m1x2_46;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_46) : "r"((uint32_t)(t_kv_2[0] >> 0)));
                            uint32_t t_fma_16;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_16) : "r"(_cvt_f16x2_e2m1x2_45), "r"(_cvt_f16x2_e2m1x2_46), "r"(t_h2_2));
                            t_h2_2 = t_fma_16;
                            uint32_t _cvt_f16x2_e2m1x2_47;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_47) : "r"((uint32_t)(t_qv_2[0] >> 8)));
                            uint32_t _cvt_f16x2_e2m1x2_48;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_48) : "r"((uint32_t)(t_kv_2[0] >> 8)));
                            uint32_t t_fma_17;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_17) : "r"(_cvt_f16x2_e2m1x2_47), "r"(_cvt_f16x2_e2m1x2_48), "r"(t_h2_2));
                            t_h2_2 = t_fma_17;
                            uint32_t _cvt_f16x2_e2m1x2_49;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_49) : "r"((uint32_t)(t_qv_2[0] >> 16)));
                            uint32_t _cvt_f16x2_e2m1x2_50;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_50) : "r"((uint32_t)(t_kv_2[0] >> 16)));
                            uint32_t t_fma_18;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_18) : "r"(_cvt_f16x2_e2m1x2_49), "r"(_cvt_f16x2_e2m1x2_50), "r"(t_h2_2));
                            t_h2_2 = t_fma_18;
                            uint32_t _cvt_f16x2_e2m1x2_51;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_51) : "r"((uint32_t)(t_qv_2[0] >> 24)));
                            uint32_t _cvt_f16x2_e2m1x2_52;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_52) : "r"((uint32_t)(t_kv_2[0] >> 24)));
                            uint32_t t_fma_19;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_19) : "r"(_cvt_f16x2_e2m1x2_51), "r"(_cvt_f16x2_e2m1x2_52), "r"(t_h2_2));
                            t_h2_2 = t_fma_19;
                            uint32_t _cvt_f16x2_e2m1x2_53;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_53) : "r"((uint32_t)(t_qv_2[1] >> 0)));
                            uint32_t _cvt_f16x2_e2m1x2_54;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_54) : "r"((uint32_t)(t_kv_2[1] >> 0)));
                            uint32_t t_fma_20;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_20) : "r"(_cvt_f16x2_e2m1x2_53), "r"(_cvt_f16x2_e2m1x2_54), "r"(t_h2_2));
                            t_h2_2 = t_fma_20;
                            uint32_t _cvt_f16x2_e2m1x2_55;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_55) : "r"((uint32_t)(t_qv_2[1] >> 8)));
                            uint32_t _cvt_f16x2_e2m1x2_56;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_56) : "r"((uint32_t)(t_kv_2[1] >> 8)));
                            uint32_t t_fma_21;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_21) : "r"(_cvt_f16x2_e2m1x2_55), "r"(_cvt_f16x2_e2m1x2_56), "r"(t_h2_2));
                            t_h2_2 = t_fma_21;
                            uint32_t _cvt_f16x2_e2m1x2_57;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_57) : "r"((uint32_t)(t_qv_2[1] >> 16)));
                            uint32_t _cvt_f16x2_e2m1x2_58;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_58) : "r"((uint32_t)(t_kv_2[1] >> 16)));
                            uint32_t t_fma_22;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_22) : "r"(_cvt_f16x2_e2m1x2_57), "r"(_cvt_f16x2_e2m1x2_58), "r"(t_h2_2));
                            t_h2_2 = t_fma_22;
                            uint32_t _cvt_f16x2_e2m1x2_59;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_59) : "r"((uint32_t)(t_qv_2[1] >> 24)));
                            uint32_t _cvt_f16x2_e2m1x2_60;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_60) : "r"((uint32_t)(t_kv_2[1] >> 24)));
                            uint32_t t_fma_23;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_23) : "r"(_cvt_f16x2_e2m1x2_59), "r"(_cvt_f16x2_e2m1x2_60), "r"(t_h2_2));
                            t_h2_2 = t_fma_23;
                            float _cvt_f32_f16_61;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_61) : "h"((uint16_t)(t_h2_2)));
                            float _cvt_f32_f16_62;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_62) : "h"((uint16_t)(t_h2_2 >> 16)));
                            float t_blk_2 = _cvt_f32_f16_61 + _cvt_f32_f16_62;
                            float _cvt_f32_f16_63;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_63) : "h"((uint16_t)(t_qsc_1 >> 0)));
                            float _cvt_f32_f16_64;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_64) : "h"((uint16_t)(t_ksc_1 >> 0)));
                            float t_sab_2 = _cvt_f32_f16_63 * _cvt_f32_f16_64;
                            t_acc_1 = __fmaf_rn(t_blk_2, t_sab_2, t_acc_1);
                            unsigned int t_h2_3 = 0;
                            uint32_t _cvt_f16x2_e2m1x2_65;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_65) : "r"((uint32_t)(t_qv_2[2] >> 0)));
                            uint32_t _cvt_f16x2_e2m1x2_66;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_66) : "r"((uint32_t)(t_kv_2[2] >> 0)));
                            uint32_t t_fma_24;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_24) : "r"(_cvt_f16x2_e2m1x2_65), "r"(_cvt_f16x2_e2m1x2_66), "r"(t_h2_3));
                            t_h2_3 = t_fma_24;
                            uint32_t _cvt_f16x2_e2m1x2_67;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_67) : "r"((uint32_t)(t_qv_2[2] >> 8)));
                            uint32_t _cvt_f16x2_e2m1x2_68;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_68) : "r"((uint32_t)(t_kv_2[2] >> 8)));
                            uint32_t t_fma_25;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_25) : "r"(_cvt_f16x2_e2m1x2_67), "r"(_cvt_f16x2_e2m1x2_68), "r"(t_h2_3));
                            t_h2_3 = t_fma_25;
                            uint32_t _cvt_f16x2_e2m1x2_69;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_69) : "r"((uint32_t)(t_qv_2[2] >> 16)));
                            uint32_t _cvt_f16x2_e2m1x2_70;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_70) : "r"((uint32_t)(t_kv_2[2] >> 16)));
                            uint32_t t_fma_26;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_26) : "r"(_cvt_f16x2_e2m1x2_69), "r"(_cvt_f16x2_e2m1x2_70), "r"(t_h2_3));
                            t_h2_3 = t_fma_26;
                            uint32_t _cvt_f16x2_e2m1x2_71;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_71) : "r"((uint32_t)(t_qv_2[2] >> 24)));
                            uint32_t _cvt_f16x2_e2m1x2_72;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_72) : "r"((uint32_t)(t_kv_2[2] >> 24)));
                            uint32_t t_fma_27;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_27) : "r"(_cvt_f16x2_e2m1x2_71), "r"(_cvt_f16x2_e2m1x2_72), "r"(t_h2_3));
                            t_h2_3 = t_fma_27;
                            uint32_t _cvt_f16x2_e2m1x2_73;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_73) : "r"((uint32_t)(t_qv_2[3] >> 0)));
                            uint32_t _cvt_f16x2_e2m1x2_74;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_74) : "r"((uint32_t)(t_kv_2[3] >> 0)));
                            uint32_t t_fma_28;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_28) : "r"(_cvt_f16x2_e2m1x2_73), "r"(_cvt_f16x2_e2m1x2_74), "r"(t_h2_3));
                            t_h2_3 = t_fma_28;
                            uint32_t _cvt_f16x2_e2m1x2_75;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_75) : "r"((uint32_t)(t_qv_2[3] >> 8)));
                            uint32_t _cvt_f16x2_e2m1x2_76;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_76) : "r"((uint32_t)(t_kv_2[3] >> 8)));
                            uint32_t t_fma_29;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_29) : "r"(_cvt_f16x2_e2m1x2_75), "r"(_cvt_f16x2_e2m1x2_76), "r"(t_h2_3));
                            t_h2_3 = t_fma_29;
                            uint32_t _cvt_f16x2_e2m1x2_77;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_77) : "r"((uint32_t)(t_qv_2[3] >> 16)));
                            uint32_t _cvt_f16x2_e2m1x2_78;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_78) : "r"((uint32_t)(t_kv_2[3] >> 16)));
                            uint32_t t_fma_30;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_30) : "r"(_cvt_f16x2_e2m1x2_77), "r"(_cvt_f16x2_e2m1x2_78), "r"(t_h2_3));
                            t_h2_3 = t_fma_30;
                            uint32_t _cvt_f16x2_e2m1x2_79;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_79) : "r"((uint32_t)(t_qv_2[3] >> 24)));
                            uint32_t _cvt_f16x2_e2m1x2_80;
                            asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_80) : "r"((uint32_t)(t_kv_2[3] >> 24)));
                            uint32_t t_fma_31;
                            asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(t_fma_31) : "r"(_cvt_f16x2_e2m1x2_79), "r"(_cvt_f16x2_e2m1x2_80), "r"(t_h2_3));
                            t_h2_3 = t_fma_31;
                            float _cvt_f32_f16_81;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_81) : "h"((uint16_t)(t_h2_3)));
                            float _cvt_f32_f16_82;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_82) : "h"((uint16_t)(t_h2_3 >> 16)));
                            float t_blk_3 = _cvt_f32_f16_81 + _cvt_f32_f16_82;
                            float _cvt_f32_f16_83;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_83) : "h"((uint16_t)(t_qsc_1 >> 16)));
                            float _cvt_f32_f16_84;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_84) : "h"((uint16_t)(t_ksc_1 >> 16)));
                            float t_sab_3 = _cvt_f32_f16_83 * _cvt_f32_f16_84;
                            t_acc_1 = __fmaf_rn(t_blk_3, t_sab_3, t_acc_1);
                        }
                        unsigned int t_qr_row_1 = smem_q_rope_addr + (unsigned int)(t_row_3 * 128);
                        unsigned int t_kr_row_1 = t_kb_1 + (unsigned int)(1024 + t_j_3 * 128);
                        #pragma unroll 1
                        for (int t_c_3 = 0; t_c_3 < 8 * (int)(g_warp == 3); t_c_3++) {
                            unsigned int t_qv_3[4];
                            unsigned int t_kv_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_qv_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_qv_3[(0) + 3]))
                                : "r"(t_qr_row_1 + (unsigned int)((t_c_3 ^ t_row_3 & 7) << 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&t_kv_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_kv_3[(0) + 3]))
                                : "r"(t_kr_row_1 + (unsigned int)((t_c_3 ^ (int)(t_kr_row_1 >> 7 & 7)) << 4)));
                            float _dot_85;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                "}\n"
                                : "=f"(_dot_85) : "r"(t_qv_3[0]), "r"(t_kv_3[0]), "f"(t_acc_1));
                            t_acc_1 = _dot_85;
                            float _dot_86;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                "}\n"
                                : "=f"(_dot_86) : "r"(t_qv_3[1]), "r"(t_kv_3[1]), "f"(t_acc_1));
                            t_acc_1 = _dot_86;
                            float _dot_87;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                "}\n"
                                : "=f"(_dot_87) : "r"(t_qv_3[2]), "r"(t_kv_3[2]), "f"(t_acc_1));
                            t_acc_1 = _dot_87;
                            float _dot_88;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                "}\n"
                                : "=f"(_dot_88) : "r"(t_qv_3[3]), "r"(t_kv_3[3]), "f"(t_acc_1));
                            t_acc_1 = _dot_88;
                        }
                        smem_tdot[(int)((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0) * 128 + t_u_1 * 128 + g_warp * 32 + lane] = t_acc_1;
                    }
                }
                mbarrier_arrive(tdot_full_addr + (((g_index_stage + 1 < 2) ? g_index_stage + 1 : (unsigned int)0)) * 8);
                mbarrier_arrive(q_read_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
