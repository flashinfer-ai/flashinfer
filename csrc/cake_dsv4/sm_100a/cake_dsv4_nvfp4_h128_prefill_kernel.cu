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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
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

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(640, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_nvfp4_h128_prefill_sm100a(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_q_sf, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, const __grid_constant__ CUtensorMap tmap_o, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items, int max_q_len, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens)
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
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 48)
    #define v_empty_addr (mbar_base + 64)
    #define nvfp4_sf_full_addr (mbar_base + 80)
    #define nvfp4_kexp_full_addr (mbar_base + 96)
    #define nvfp4_kstage_free_addr (mbar_base + 112)
    #define nvfp4_q_load_full_addr (mbar_base + 128)
    #define index_full_addr (mbar_base + 136)
    #define index_empty_addr (mbar_base + 184)
    #define s_full_addr (mbar_base + 232)
    #define p_full_addr (mbar_base + 248)
    #define split_pv_p_empty_addr (mbar_base + 264)
    #define s_empty_addr (mbar_base + 280)
    #define stats_addr (mbar_base + 296)
    #define stats_empty_addr (mbar_base + 312)
    #define sum_ready_addr (mbar_base + 328)
    #define source_sum_empty_addr (mbar_base + 336)
    #define o_empty_addr (mbar_base + 344)
    #define o_full_addr (mbar_base + 352)
    #define s_seeded_addr (mbar_base + 360)
    #define q_pair_ready_addr (mbar_base + 368)
    #define kv_pair_ready_addr (mbar_base + 376)
    #define pv_pair_ready_addr (mbar_base + 384)
    #define tmem_dealloc_addr (mbar_base + 400)
    #define tmem_dealloc_peer_addr (mbar_base + 408)
    #define source_work_full_addr (mbar_base + 416)
    #define source_work_empty_addr (mbar_base + 448)
    #define source_throttle_full_addr (mbar_base + 480)
    #define source_throttle_empty_addr (mbar_base + 496)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_kv_addr = smem + 27648;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_v_addr = smem + 27648;
    uint8_t* smem_q_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_a_addr = smem + 1024;
    uint8_t* smem_q_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_q_b_addr = smem + 9216;
    __nv_bfloat16* smem_q_rope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_rope_addr = smem + 17408;
    unsigned int* smem_q_sf = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int smem_q_sf_addr = smem + 25600;
    uint8_t* smem_k_a = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_k_a_addr = smem + 27648;
    uint8_t* smem_k_b = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_k_b_addr = smem + 35840;
    __nv_bfloat16* smem_k_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 44032);
    const int smem_k_c_addr = smem + 44032;
    unsigned int* smem_k_sf = reinterpret_cast<unsigned int*>(smem_raw + 52224);
    const int smem_k_sf_addr = smem + 52224;
    uint8_t* smem_k_vraw = reinterpret_cast<uint8_t*>(smem_raw + 56320);
    const int smem_k_vraw_addr = smem + 56320;
    uint8_t* smem_v_full = reinterpret_cast<uint8_t*>(smem_raw + 117760);
    const int smem_v_full_addr = smem + 117760;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 183296);
    const int smem_stats_max_addr = smem + 183296;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 184320);
    const int smem_stats_sum_addr = smem + 184320;
    float* smem_stats_final_max = reinterpret_cast<float*>(smem_raw + 184832);
    const int smem_stats_final_max_addr = smem + 184832;
    float* smem_softmax_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 185344);
    const int smem_softmax_warp_pair_exchange_addr = smem + 185344;
    float* smem_corr_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 186368);
    const int smem_corr_warp_pair_exchange_addr = smem + 186368;
    unsigned int* smem_epi_o_pad = reinterpret_cast<unsigned int*>(smem_raw + 210016);
    const int smem_epi_o_pad_addr = smem + 210016;
    __nv_bfloat16* smem_epi_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_epi_o_addr = smem + 210944;
    int* smem_sparse_indices = reinterpret_cast<int*>(smem_raw + 186880);
    const int smem_sparse_indices_addr = smem + 186880;
    uint8_t* smem_p_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 193536);
    const int smem_p_fp8_addr = smem + 193536;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 229376);
    const int work_response_addr = smem + 229376;
    unsigned int* smem_kexp = reinterpret_cast<unsigned int*>(smem_raw + 209952);
    const int smem_kexp_addr = smem + 209952;
    float* smem_kmax = reinterpret_cast<float*>(smem_raw + 209984);
    const int smem_kmax_addr = smem + 209984;

    // Kernel setup ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (64 barriers at smem_raw[0..512))

    if (warp == 12) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'k_pipe' ---
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'k_pipe' ---
            // nvfp4_sf_full: 2 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // nvfp4_kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // nvfp4_kstage_free: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // nvfp4_q_load_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'index_pipe' ---
            // index_full: 6 barriers, init_count=32
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            // index_empty: 6 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
            mbarrier_init(smem + 208, 256);
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 248, 256);
            mbarrier_init(smem + 256, 256);
            // split_pv_p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 280, 256);
            mbarrier_init(smem + 288, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 5) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            mbarrier_init(smem + 304, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 312, 128);
            mbarrier_init(smem + 320, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            // source_sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 336, 128);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 408, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 4) {
        uint32_t leader = elect_sync();
        if (leader) {
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 344, 256);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 352, 1);
            // pv_pair_ready: stages (1,), init_count=64
            mbarrier_init(smem + 392, 64);
            // tmem_dealloc: 1 barriers, init_count=448
            mbarrier_init(smem + 400, 448);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 360, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 368, 64);
            // --- pipeline 'kv_pipe' ---
            // kv_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 376, 64);
            // pv_pair_ready: stages (0,), init_count=64
            mbarrier_init(smem + 384, 64);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 10) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'source_work_pipe' ---
            // source_work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 416, 1);
            mbarrier_init(smem + 424, 1);
            mbarrier_init(smem + 432, 1);
            mbarrier_init(smem + 440, 1);
            // source_work_empty: 4 barriers, init_count=1216
            mbarrier_init(smem + 448, 1216);
            mbarrier_init(smem + 456, 1216);
            mbarrier_init(smem + 464, 1216);
            mbarrier_init(smem + 472, 1216);
            // --- pipeline 'source_throttle_pipe' ---
            // source_throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 480, 128);
            mbarrier_init(smem + 488, 128);
            // source_throttle_empty: 2 barriers, init_count=32
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
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem_scratch = taddr;
    const unsigned int tmem_tmem_sfa = taddr + 64;
    const unsigned int tmem_tmem_sfb = taddr + 96;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: index_warp ----
    if (warp == 9) {
        { // index_warp_main
            const int index_dummy = 0;
            int all_num_kv_tiles = (sparse_topk + 128 - 1) / 128;
            unsigned int index_prod_stage = 0;
            unsigned int source_work_stage = 0;
            unsigned int source_work_x = blockIdx.x;
            unsigned int source_work_z = blockIdx.z;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_source_work_full = 0;
            #pragma unroll 1
            for (unsigned int static_work = 0; static_work < 514; static_work++) {
                int split_idx = 0;
                int query_idx = source_work_x >> 1;
                int source_work_valid = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1);
                if (source_work_valid != 0) {
                    int _max_0 = ((sparse_topk_lens[mapped_query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx] + sparse_topk_lens_offset) : (0));
                    int main_active = ((_max_0) < (swa_width) ? (_max_0) : (swa_width));
                    int extra_active = 0;
                    if (compressed_width > 0) {
                        int _max_2 = ((extra_topk_lens[mapped_query_idx]) > (0) ? (extra_topk_lens[mapped_query_idx]) : (0));
                        int _min_1 = ((_max_2) < (compressed_width) ? (_max_2) : (compressed_width));
                        extra_active = _min_1;
                    }
                    int n_main_tiles = (main_active + 128 - 1) / 128;
                    int n_extra_tiles = (extra_active + 128 - 1) / 128;
                    int _max_3 = ((n_main_tiles + n_extra_tiles) > (1) ? (n_main_tiles + n_extra_tiles) : (1));
                    int active_topk = _max_3 * 128;
                    all_num_kv_tiles = (active_topk + 128 - 1) / 128;
                    int tiles_per_split = all_num_kv_tiles + 1 - 1;
                    int first_tile = split_idx * tiles_per_split;
                    int num_index_passes = (tiles_per_split + 1) / 2;
                    #pragma unroll 1
                    for (int index_pass = 0; index_pass < num_index_passes; index_pass++) {
                        mbarrier_wait(index_empty_addr + (index_prod_stage) * 8, _phase_index_empty);
                        int index_stage_base = smem_sparse_indices_addr + index_prod_stage * 1024;
                        int stage_row_base = index_pass * 256;
                        #pragma unroll
                        for (int index_half = 0; index_half < 2; index_half++) {
                            int index_offset = index_half * 128 + lane * 4;
                            int stage_tile = index_pass * 2 + index_half;
                            int seg_in_main = stage_tile < n_main_tiles || n_extra_tiles == 0;
                            int seg_tile = ((seg_in_main != 0) ? stage_tile : stage_tile - n_main_tiles);
                            int seg_width = ((seg_in_main != 0) ? swa_width : compressed_width);
                            int tile_col_base = seg_tile * 128;
                            int* row_tile_ptr = ((seg_in_main != 0) ? (swa_indices + (mapped_query_idx * swa_index_stride + tile_col_base)) : (compressed_indices + (mapped_query_idx * compressed_index_stride + tile_col_base)));
                            int lane_col = tile_col_base + lane * 4;
                            if (seg_width >= lane_col + 4) {
                                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;"
                                    :: "r"(index_stage_base + index_offset * 4), "l"(row_tile_ptr + lane * 4));
                            } else {
                                int tail_rows[4];
                                int tail_fill = -1;
                                if (tile_col_base < seg_width) {
                                    tail_fill = row_tile_ptr[0];
                                }
                                #pragma unroll
                                for (int row_i = 0; row_i < 4; row_i++) {
                                    tail_rows[row_i] = tail_fill;
                                }
                                #pragma unroll
                                for (int row_i_1 = 0; row_i_1 < 4; row_i_1++) {
                                    if (seg_width > lane_col + row_i_1) {
                                        tail_rows[row_i_1] = row_tile_ptr[lane * 4 + row_i_1];
                                    }
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sparse_indices_addr + ((unsigned int)index_stage_base - smem_sparse_indices_addr + (unsigned int)(index_offset * 4))), "r"(tail_rows[0]), "r"(tail_rows[1]), "r"(tail_rows[2]), "r"(tail_rows[3]) : "memory");
                                asm volatile("fence.release.cta;" ::: "memory");
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(index_full_addr + (index_prod_stage) * 8) : "memory");
                        index_prod_stage += 1;
                        if (index_prod_stage == 6) { index_prod_stage = 0; _phase_index_empty ^= 1; }
                    }
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage) * 8, _phase_source_work_full);
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
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
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
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1)
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2)
                    : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage) * 8), "r"(0) : "memory");
                source_work_stage += 1;
                if (source_work_stage == 4) { source_work_stage = 0; _phase_source_work_full ^= 1; }
                if (clc_valid == 0) {
                    break;
                }
                source_work_x = clc_ctaid + (unsigned int)cta_rank;
                source_work_z = clc_ctaid_2;
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // load_warp_main
            const int wg2_dummy = 0;
            const int load_warp_rank = warp - 12;
            const int tmem_row_base = load_warp_rank * 32;
            int peer_rank = bid % 2 ^ 1;
            int k_cta_offset = bid % 2 * 64;
            const int v_key = load_warp_rank * 32 + lane;
            int v_own = v_key >> 6 == bid % 2;
            const int v_swz = v_key & 7;
            const int sf_tok_a = (load_warp_rank >> 1) * 64 + lane;
            const int sf_tok_b = sf_tok_a + 32;
            int tile_cursor = 0;
            int item_cursor = 0;
            unsigned int load_k_index_stage = 0;
            unsigned int source_work_stage_1 = 0;
            unsigned int source_throttle_stage = 0;
            unsigned int source_work_x_1 = blockIdx.x;
            unsigned int source_work_z_1 = blockIdx.z;
            int source_work_valid_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1) < (unsigned int)num_query_tokens;
            int mapped_query_idx_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
            int main_active_1 = 0;
            int extra_active_1 = 0;
            int n_main_tiles_1 = 0;
            int n_extra_tiles_1 = 0;
            int num_kv_tiles = 1;
            if (source_work_valid_1 != 0) {
                int _max_1 = ((sparse_topk_lens[mapped_query_idx_1] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_1] + sparse_topk_lens_offset) : (0));
                int _min_0 = ((_max_1) < (swa_width) ? (_max_1) : (swa_width));
                main_active_1 = _min_0;
                if (compressed_width > 0) {
                    int _max_3_1 = ((extra_topk_lens[mapped_query_idx_1]) > (0) ? (extra_topk_lens[mapped_query_idx_1]) : (0));
                    int _min_2 = ((_max_3_1) < (compressed_width) ? (_max_3_1) : (compressed_width));
                    extra_active_1 = _min_2;
                }
                n_main_tiles_1 = (main_active_1 + 128 - 1) / 128;
                n_extra_tiles_1 = (extra_active_1 + 128 - 1) / 128;
                int _max_4 = ((n_main_tiles_1 + n_extra_tiles_1) > (1) ? (n_main_tiles_1 + n_extra_tiles_1) : (1));
                num_kv_tiles = _max_4;
            }
            int has_next = 1;
            int next_valid = 0;
            int next_query_idx = 0;
            int next_main_active = 0;
            int next_extra_active = 0;
            int next_n_main_tiles = 0;
            int next_n_extra_tiles = 0;
            int next_num_kv_tiles = 1;
            unsigned int _phase_source_throttle_empty = 1;
            unsigned int _phase_source_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = 0; work_idx < 514; work_idx++) {
                if (bid % 2 == 0) {
                    mbarrier_wait(source_throttle_empty_addr + (source_throttle_stage) * 8, _phase_source_throttle_empty);
                    mbarrier_arrive(source_throttle_full_addr + (source_throttle_stage) * 8);
                    source_throttle_stage += 1;
                    if (source_throttle_stage == 2) { source_throttle_stage = 0; _phase_source_throttle_empty ^= 1; }
                }
                if (source_work_valid_1 != 0) {
                    int ld_rec = item_cursor;
                    int ld_on = item_cursor < 48;
                    #pragma unroll 1
                    for (int tile = 0; tile < num_kv_tiles; tile++) {
                        int p_stage = tile_cursor + tile & 1;
                        int _stage_k_full = (int)((unsigned int)(tile_cursor + tile) % 2);
                        int _parity_k_full = (int)((unsigned int)(tile_cursor + tile) / 2 % 2);
                        mbarrier_wait(k_full_addr + (_stage_k_full) * 8, _parity_k_full);
                        int lt_on = tile_cursor + tile < 160;
                        int ld_t0 = ld_on & (int)(tile == 0);
                        int sf_base = smem_k_a_addr + (unsigned int)(p_stage * 45056) + 24576;
                        unsigned int sf_a[8];
                        unsigned int sf_b[8];
                        int sf_par = lane >> 2 & 1;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 3]))
                            : "r"(sf_base + sf_tok_a * 32 + 16 * sf_par));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 3]))
                            : "r"(sf_base + sf_tok_a * 32 + (16 - 16 * sf_par)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 3]))
                            : "r"(sf_base + sf_tok_b * 32 + 16 * sf_par));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 3]))
                            : "r"(sf_base + sf_tok_b * 32 + (16 - 16 * sf_par)));
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
                        int _stage_k_empty = (int)((unsigned int)(tile_cursor + tile) % 2);
                        int _parity_k_empty = (int)((unsigned int)(tile_cursor + tile) / 2 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty) * 8, _parity_k_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int sfb_regs[32];
                        #pragma unroll
                        for (int sf_s = 0; sf_s < 7; sf_s++) {
                            sfb_regs[4 * sf_s] = sf_a[sf_s];
                            sfb_regs[4 * sf_s + 1] = sf_b[sf_s];
                            sfb_regs[4 * sf_s + 2] = 0;
                            sfb_regs[4 * sf_s + 3] = 0;
                        }
                        #pragma unroll
                        for (int sf_pad = 28; sf_pad < 32; sf_pad++) {
                            sfb_regs[sf_pad] = 0;
                        }
                        int sfb_col = 96 + p_stage * 128;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)sfb_col + (unsigned int)(tmem_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + (unsigned int)sfb_col + 16 + (unsigned int)(tmem_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs + 16)[15])));
                        unsigned int k68 = 0;
                        unsigned int k70 = 0;
                        unsigned int k78 = 0;
                        #pragma unroll
                        for (int sf_s_1 = 0; sf_s_1 < 7; sf_s_1++) {
                            k68 = k68 | sf_a[sf_s_1] + 404232216 | sf_b[sf_s_1] + 404232216;
                            k70 = k70 | sf_a[sf_s_1] + 269488144 | sf_b[sf_s_1] + 269488144;
                            k78 = k78 | sf_a[sf_s_1] + 134744072 | sf_b[sf_s_1] + 134744072;
                        }
                        int _warp_vote_5 = __any_sync(0xFFFFFFFF, (k68 & 2155905152u) != 0);
                        int _warp_vote_6 = __any_sync(0xFFFFFFFF, (k70 & 2155905152u) != 0);
                        int _warp_vote_7 = __any_sync(0xFFFFFFFF, (k78 & 2155905152u) != 0);
                        float e_lvl = (float)(_warp_vote_5 != 0) + (float)(_warp_vote_6 != 0) + (float)(_warp_vote_7 != 0);
                        if (elect_sync()) {
                            smem_kmax[p_stage * 4 + load_warp_rank] = e_lvl;
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (load_warp_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(nvfp4_sf_full_addr + ((int)((unsigned int)(tile_cursor + tile) % 2)) * 8), "r"(0) : "memory");
                            }
                        }
                        float k_lvl;
                        asm("max.f32 %0, %1, %2;" : "=f"(k_lvl) : "f"(max_noftz(smem_kmax[p_stage * 4], smem_kmax[p_stage * 4 + 1])), "f"(max_noftz(smem_kmax[p_stage * 4 + 2], smem_kmax[p_stage * 4 + 3])));
                        int kexp = (int)k_lvl;
                        unsigned int kexp_u = (unsigned int)kexp;
                        unsigned int kexp_f16x2 = (15 - kexp_u << 10) * 65537;
                        unsigned int kexp_bf16x2 = (127 - kexp_u << 7) * 65537;
                        int kexp_zero = (int)(kexp_u == 0);
                        int _stage_v_empty = (int)((unsigned int)(tile_cursor + tile) % 2);
                        int _parity_v_empty = (int)((unsigned int)(tile_cursor + tile) / 2 % 2) ^ 1;
                        mbarrier_wait(v_empty_addr + (_stage_v_empty) * 8, _parity_v_empty);
                        if (load_warp_rank == 0) {
                            if (elect_sync()) {
                                smem_kexp[p_stage * 4] = kexp_u;
                                mbarrier_arrive(nvfp4_kexp_full_addr + ((int)((unsigned int)(tile_cursor + tile) % 2)) * 8);
                            }
                        }
                        int v_stage_base = smem_k_a_addr + (unsigned int)(p_stage * 45056);
                        int v_row_off = (v_key & 63) * 128;
                        int v_sf_addr = sf_base + v_key * 32 + bid % 2 * 16;
                        int v_dst_row = p_stage * 32768 + v_key * 128;
                        int v_codes_src = v_stage_base + v_row_off + ((v_own != 0) ? bid % 2 * 8192 : 28672);
                        int v_units = 8 - bid % 2 * 2;
                        unsigned int v_next[4];
                        unsigned int v_sc_next[1];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_next[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_next[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_next[(0) + 3]))
                            : "r"(v_codes_src + (v_swz << 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&v_sc_next[0])) : "r"(v_sf_addr));
                        #pragma unroll 1
                        for (int v_u = 0; v_u < v_units; v_u++) {
                            unsigned int v_cur[4];
                            #pragma unroll
                            for (int v_i = 0; v_i < 4; v_i++) {
                                v_cur[v_i] = v_next[v_i];
                            }
                            unsigned int v_sc_cur = v_sc_next[0];
                            int v_u_next = ((v_u + 1) < (v_units - 1) ? (v_u + 1) : (v_units - 1));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&v_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_next[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_next[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_next[(0) + 3]))
                                : "r"(v_codes_src + ((v_u_next ^ v_swz) << 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&v_sc_next[0])) : "r"(v_sf_addr + (v_u_next >> 1 << 2)));
                            unsigned int v_out[8];
                            if (kexp_zero != 0) {
                                unsigned int v_sc_shift = (unsigned int)((v_u & 1) * 16);
                                unsigned int v_sc_lo = v_sc_cur >> v_sc_shift;
                                unsigned int v_sc_hi = v_sc_cur >> v_sc_shift + 8;
                                #pragma unroll
                                for (int v_w = 0; v_w < 4; v_w++) {
                                    unsigned int v_sc = ((v_w < 2) ? v_sc_lo : v_sc_hi);
                                    v_out[2 * v_w] = qmul4_lower4(v_cur[v_w], v_sc);
                                    v_out[2 * v_w + 1] = qmul4_higher4(v_cur[v_w], v_sc);
                                }
                            } else {
                                uint32_t v_sc_pair;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(v_sc_pair) : "h"((uint16_t)(v_sc_cur >> (unsigned int)((v_u & 1) * 16))));
                                uint32_t _f16x2_mul_0;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(v_sc_pair), "r"(kexp_f16x2));
                                v_sc_pair = _f16x2_mul_0;
                                uint32_t v_sc0;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(v_sc0) : "r"(v_sc_pair), "r"(v_sc_pair));
                                uint32_t v_sc1;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(v_sc1) : "r"(v_sc_pair), "r"(v_sc_pair));
                                #pragma unroll
                                for (int v_w_1 = 0; v_w_1 < 4; v_w_1++) {
                                    unsigned int v_sc_1 = ((v_w_1 < 2) ? v_sc0 : v_sc1);
                                    unsigned int v_word = v_cur[v_w_1];
                                    uint32_t _cvt_f16x2_e2m1x2_8;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_8) : "r"((uint32_t)(v_word)));
                                    uint32_t f16x2_mul;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul) : "r"(_cvt_f16x2_e2m1x2_8), "r"(v_sc_1));
                                    uint16_t e4m3x2;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2) : "r"(f16x2_mul));
                                    uint32_t _cvt_f16x2_e2m1x2_9;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_9) : "r"((uint32_t)(v_word >> 8)));
                                    uint32_t f16x2_mul_0;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_0) : "r"(_cvt_f16x2_e2m1x2_9), "r"(v_sc_1));
                                    uint16_t e4m3x2_0;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_0) : "r"(f16x2_mul_0));
                                    uint32_t _cvt_f16x2_e2m1x2_10;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_10) : "r"((uint32_t)(v_word >> 16)));
                                    uint32_t f16x2_mul_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_1) : "r"(_cvt_f16x2_e2m1x2_10), "r"(v_sc_1));
                                    uint16_t e4m3x2_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_1) : "r"(f16x2_mul_1));
                                    uint32_t _cvt_f16x2_e2m1x2_11;
                                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_11) : "r"((uint32_t)(v_word >> 24)));
                                    uint32_t f16x2_mul_2;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f16x2_mul_2) : "r"(_cvt_f16x2_e2m1x2_11), "r"(v_sc_1));
                                    uint16_t e4m3x2_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(e4m3x2_2) : "r"(f16x2_mul_2));
                                    uint32_t _mov_b32_pack_u16x2_12;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_12) : "h"(e4m3x2), "h"(e4m3x2_0));
                                    v_out[2 * v_w_1] = _mov_b32_pack_u16x2_12;
                                    uint32_t _mov_b32_pack_u16x2_13;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_13) : "h"(e4m3x2_1), "h"(e4m3x2_2));
                                    v_out[2 * v_w_1 + 1] = _mov_b32_pack_u16x2_13;
                                }
                            }
                            int v_o0 = (v_u & 3) * 2;
                            int v_o1 = v_o0 + 1;
                            int v_dst_u = v_dst_row + (v_u >> 2) * 16384;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_u + ((v_o0 ^ v_swz) << 4))), "r"(v_out[0]), "r"(v_out[1]), "r"(v_out[2]), "r"(v_out[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_u + ((v_o1 ^ v_swz) << 4))), "r"(v_out[4]), "r"(v_out[5]), "r"(v_out[6]), "r"(v_out[7]) : "memory");
                        }
                        if (bid % 2 != 0) {
                            int r_src_row = v_stage_base + v_row_off + ((v_own != 0) ? 16384 : 36864);
                            int v_dst_off = v_dst_row + 16384;
                            #pragma unroll 1
                            for (int r_u = 0; r_u < 4; r_u++) {
                                unsigned int r_out[4];
                                #pragma unroll
                                for (int r_half = 0; r_half < 2; r_half++) {
                                    unsigned int r_in[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&r_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 3]))
                                        : "r"(r_src_row + ((2 * r_u + r_half ^ v_swz) << 4)));
                                    #pragma unroll
                                    for (int r_w = 0; r_w < 2; r_w++) {
                                        uint32_t r_s0;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_s0) : "r"(r_in[2 * r_w]), "r"(kexp_bf16x2));
                                        uint32_t r_s1;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r_s1) : "r"(r_in[2 * r_w + 1]), "r"(kexp_bf16x2));
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_15;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_15) : "r"(r_s0));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(r_s0);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint16_t _cvt_satfinite_e4m3x2_bf16x2_16;
                                        #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_16) : "r"(r_s1));
                                        #else
                                        {
                                        const uint32_t _bf16x2 = (uint32_t)(r_s1);
                                        const float _lo = __uint_as_float(_bf16x2 << 16);
                                        const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
                                        }
                                        #endif
                                        uint32_t _mov_b32_pack_u16x2_14;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_14) : "h"(_cvt_satfinite_e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_16));
                                        r_out[2 * r_half + r_w] = _mov_b32_pack_u16x2_14;
                                    }
                                }
                                int r_o = 4 + r_u;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_off + ((r_o ^ v_swz) << 4))), "r"(r_out[0]), "r"(r_out[1]), "r"(r_out[2]), "r"(r_out[3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (load_warp_rank == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(nvfp4_kstage_free_addr + ((int)((unsigned int)(tile_cursor + tile) % 2)) * 8);
                            }
                            if (elect_sync()) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(v_full_addr + ((int)((unsigned int)(tile_cursor + tile) % 2)) * 8), "r"(0) : "memory");
                            }
                        }
                    }
                    tile_cursor += num_kv_tiles;
                    item_cursor += 1;
                    mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
                    uint32_t clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid_1)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_3 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_3)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_1_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_1_1)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_2_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_2_1)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + (source_work_stage_1) * 8), "r"(0) : "memory");
                    source_work_stage_1 += 1;
                    if (source_work_stage_1 == 4) { source_work_stage_1 = 0; _phase_source_work_full_1 ^= 1; }
                    has_next = 0;
                    next_valid = 0;
                    if (clc_valid_1 != 0) {
                        has_next = 1;
                        source_work_x_1 = clc_ctaid_3 + (unsigned int)cta_rank;
                        source_work_z_1 = clc_ctaid_2_1;
                        next_query_idx = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
                        next_valid = next_query_idx < num_query_tokens;
                        if (next_valid != 0) {
                            int _max_18 = ((sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) : (0));
                            int _min_17 = ((_max_18) < (swa_width) ? (_max_18) : (swa_width));
                            next_main_active = _min_17;
                            next_extra_active = 0;
                            if (compressed_width > 0) {
                                int _max_20 = ((extra_topk_lens[next_query_idx]) > (0) ? (extra_topk_lens[next_query_idx]) : (0));
                                int _min_19 = ((_max_20) < (compressed_width) ? (_max_20) : (compressed_width));
                                next_extra_active = _min_19;
                            }
                            next_n_main_tiles = (next_main_active + 128 - 1) / 128;
                            next_n_extra_tiles = (next_extra_active + 128 - 1) / 128;
                            int _max_21 = ((next_n_main_tiles + next_n_extra_tiles) > (1) ? (next_n_main_tiles + next_n_extra_tiles) : (1));
                            next_num_kv_tiles = _max_21;
                        }
                    }
                } else {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
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
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
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
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
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
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
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
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + (source_work_stage_1) * 8), "r"(0) : "memory");
                    source_work_stage_1 += 1;
                    if (source_work_stage_1 == 4) { source_work_stage_1 = 0; _phase_source_work_full_1 ^= 1; }
                    has_next = 0;
                    next_valid = 0;
                    if (_clc_valid_2 != 0) {
                        has_next = 1;
                        source_work_x_1 = _clc_ctaid_6 + (unsigned int)cta_rank;
                        source_work_z_1 = _clc_ctaid_8;
                        next_query_idx = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
                        next_valid = next_query_idx < num_query_tokens;
                        if (next_valid != 0) {
                            int _max_23 = ((sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) : (0));
                            int _min_22 = ((_max_23) < (swa_width) ? (_max_23) : (swa_width));
                            next_main_active = _min_22;
                            next_extra_active = 0;
                            if (compressed_width > 0) {
                                int _max_25 = ((extra_topk_lens[next_query_idx]) > (0) ? (extra_topk_lens[next_query_idx]) : (0));
                                int _min_24 = ((_max_25) < (compressed_width) ? (_max_25) : (compressed_width));
                                next_extra_active = _min_24;
                            }
                            next_n_main_tiles = (next_main_active + 128 - 1) / 128;
                            next_n_extra_tiles = (next_extra_active + 128 - 1) / 128;
                            int _max_26 = ((next_n_main_tiles + next_n_extra_tiles) > (1) ? (next_n_main_tiles + next_n_extra_tiles) : (1));
                            next_num_kv_tiles = _max_26;
                        }
                    }
                }
                if (has_next == 0) {
                    break;
                }
                source_work_valid_1 = next_valid;
                mapped_query_idx_1 = next_query_idx;
                main_active_1 = next_main_active;
                extra_active_1 = next_extra_active;
                n_main_tiles_1 = next_n_main_tiles;
                n_extra_tiles_1 = next_n_extra_tiles;
                num_kv_tiles = next_num_kv_tiles;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.442695f;
            const int wg_dummy_inc = 0;
            int all_num_kv_tiles_1 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base_1 = warp % 2 * 32;
            const int tmem_score_row_base = (int)(warp % 4 * 32);
            const int n_half = (int)(warp % 4 / 2);
            const int my_row = tmem_row_base_1 + lane;
            const int stats_row = n_half * 64 + my_row;
            int softmax_tile_cursor = 0;
            int sm_item = 0;
            unsigned int softmax_index_stage = 0;
            unsigned int source_work_stage_2 = 0;
            unsigned int source_work_x_2 = blockIdx.x;
            unsigned int source_work_z_2 = blockIdx.z;
            unsigned int _phase_source_sum_empty_0 = 1;
            unsigned int _phase_source_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = 0; work_idx_1 < 514; work_idx_1++) {
                int split_idx_1 = 0;
                int query_idx_1 = source_work_x_2 >> 1 >> 1;
                split_idx_1 = 0;
                query_idx_1 = source_work_x_2 >> 1;
                int source_work_valid_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1);
                if (source_work_valid_2 != 0) {
                    int _max_0_1 = ((sparse_topk_lens[mapped_query_idx_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_2] + sparse_topk_lens_offset) : (0));
                    int main_active_2 = ((_max_0_1) < (swa_width) ? (_max_0_1) : (swa_width));
                    int extra_active_2 = 0;
                    if (compressed_width > 0) {
                        int _max_2_1 = ((extra_topk_lens[mapped_query_idx_2]) > (0) ? (extra_topk_lens[mapped_query_idx_2]) : (0));
                        int _min_1_1 = ((_max_2_1) < (compressed_width) ? (_max_2_1) : (compressed_width));
                        extra_active_2 = _min_1_1;
                    }
                    int n_main_tiles_2 = (main_active_2 + 128 - 1) / 128;
                    int n_extra_tiles_2 = (extra_active_2 + 128 - 1) / 128;
                    int _max_3_2 = ((n_main_tiles_2 + n_extra_tiles_2) > (1) ? (n_main_tiles_2 + n_extra_tiles_2) : (1));
                    int active_topk_1 = _max_3_2 * 128;
                    all_num_kv_tiles_1 = (active_topk_1 + 128 - 1) / 128;
                    int tiles_per_split_1 = all_num_kv_tiles_1 + 1 - 1;
                    int first_tile_1 = split_idx_1 * tiles_per_split_1;
                    int num_kv_tiles_1 = tiles_per_split_1;
                    int _parity_nvfp4_q_load_full = (int)((unsigned int)sm_item % 2);
                    mbarrier_wait(nvfp4_q_load_full_addr, _parity_nvfp4_q_load_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sfa_regs[32];
                    #pragma unroll
                    for (int sfa_s = 0; sfa_s < 7; sfa_s++) {
                        sfa_regs[4 * sfa_s] = smem_q_sf[lane * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 1] = smem_q_sf[(lane + 32) * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 2] = 0;
                        sfa_regs[4 * sfa_s + 3] = 0;
                    }
                    #pragma unroll
                    for (int sfa_pad = 28; sfa_pad < 32; sfa_pad++) {
                        sfa_regs[sfa_pad] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 64 + (unsigned int)(tmem_score_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 64 + 16 + (unsigned int)(tmem_score_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(q_full_addr), "r"(0) : "memory");
                        }
                    }
                    float row_max_val = -CUDART_INF_F;
                    float row_sum_val = 0.0f;
                    int sink_head = bid % 2 * 64 + my_row;
                    if (has_sinks != 0 && sink_head < num_heads && split_idx_1 == 0) {
                        row_max_val = sinks[sink_head] * 1.442695f / softmax_scale_log2;
                        row_sum_val = ((n_half == 0) ? 1.0f : 0.0f);
                    }
                    #pragma unroll 1
                    for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                        int phase = softmax_tile_cursor + tile_1 & 1;
                        int sm_t0 = sm_item < 48 && tile_1 == 0;
                        int s_off = ((phase != 0) ? 128 : 0);
                        int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_score_row_base << 16);
                        float new_max = row_max_val;
                        int seg_in_main_1 = n_main_tiles_2 > tile_1 || n_extra_tiles_2 == 0;
                        int seg_active = ((seg_in_main_1 != 0) ? main_active_2 : extra_active_2);
                        int seg_tile_1 = ((seg_in_main_1 != 0) ? tile_1 : tile_1 - n_main_tiles_2);
                        int valid_sparse_cols = seg_active - seg_tile_1 * 128 - n_half * 64;
                        if (valid_sparse_cols < 0) {
                            valid_sparse_cols = 0;
                        }
                        if (valid_sparse_cols > 64) {
                            valid_sparse_cols = 64;
                        }
                        int _stage_s_full = (int)((unsigned int)(softmax_tile_cursor + tile_1) % 2);
                        int _parity_s_full = (int)((unsigned int)(softmax_tile_cursor + tile_1) / 2 % 2);
                        uint32_t _mbar_token_4 = mbarrier_try_wait(s_full_addr + (_stage_s_full) * 8, _parity_s_full);
                        int _stage_s_full_1 = (int)((unsigned int)(softmax_tile_cursor + tile_1) % 2);
                        int _parity_s_full_1 = (int)((unsigned int)(softmax_tile_cursor + tile_1) / 2 % 2);
                        mbarrier_wait_token(s_full_addr + (_stage_s_full_1) * 8, _parity_s_full_1, _mbar_token_4);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int index_tile_addr = smem_sparse_indices_addr + softmax_index_stage * 1024 + (unsigned int)((tile_1 & 1) * 512);
                        int index_lane_addr = index_tile_addr + (n_half * 64 + lane) * 4;
                        int staged_index[2];
                        unsigned int invalid_cols[2];
                        unsigned int any_invalid = 0;
                        #pragma unroll
                        for (int w = 0; w < 2; w++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[w])) : "r"(index_lane_addr + w * 128));
                            unsigned int _warp_vote_4 = __ballot_sync(0xFFFFFFFF, staged_index[w] < 0);
                            invalid_cols[w] = _warp_vote_4;
                            any_invalid = any_invalid | invalid_cols[w];
                        }
                        if ((tile_1 & 1) != 0 || tile_1 + 1 == num_kv_tiles_1) {
                            mbarrier_arrive(index_empty_addr + (softmax_index_stage) * 8);
                            softmax_index_stage += 1;
                            if (softmax_index_stage == 6) { softmax_index_stage = 0; }
                        }
                        float _tmem_load_0[4];
                        tmem_ld_x4(&_tmem_load_0[0], s_base);
                        float score_tile_max = -CUDART_INF_F;
                        float score_tile_max_hi = -CUDART_INF_F;
                        float sv_split[64];
                        if (valid_sparse_cols == 64 && any_invalid == 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                                : "=f"(sv_split[0]), "=f"(sv_split[1]), "=f"(sv_split[2]), "=f"(sv_split[3]), "=f"(sv_split[4]), "=f"(sv_split[5]), "=f"(sv_split[6]), "=f"(sv_split[7]), "=f"(sv_split[8]), "=f"(sv_split[9]), "=f"(sv_split[10]), "=f"(sv_split[11]), "=f"(sv_split[12]), "=f"(sv_split[13]), "=f"(sv_split[14]), "=f"(sv_split[15]), "=f"(sv_split[16]), "=f"(sv_split[17]), "=f"(sv_split[18]), "=f"(sv_split[19]), "=f"(sv_split[20]), "=f"(sv_split[21]), "=f"(sv_split[22]), "=f"(sv_split[23]), "=f"(sv_split[24]), "=f"(sv_split[25]), "=f"(sv_split[26]), "=f"(sv_split[27]), "=f"(sv_split[28]), "=f"(sv_split[29]), "=f"(sv_split[30]), "=f"(sv_split[31]), "=f"(sv_split[32]), "=f"(sv_split[33]), "=f"(sv_split[34]), "=f"(sv_split[35]), "=f"(sv_split[36]), "=f"(sv_split[37]), "=f"(sv_split[38]), "=f"(sv_split[39]), "=f"(sv_split[40]), "=f"(sv_split[41]), "=f"(sv_split[42]), "=f"(sv_split[43]), "=f"(sv_split[44]), "=f"(sv_split[45]), "=f"(sv_split[46]), "=f"(sv_split[47]), "=f"(sv_split[48]), "=f"(sv_split[49]), "=f"(sv_split[50]), "=f"(sv_split[51]), "=f"(sv_split[52]), "=f"(sv_split[53]), "=f"(sv_split[54]), "=f"(sv_split[55]), "=f"(sv_split[56]), "=f"(sv_split[57]), "=f"(sv_split[58]), "=f"(sv_split[59]), "=f"(sv_split[60]), "=f"(sv_split[61]), "=f"(sv_split[62]), "=f"(sv_split[63])
                                : "r"(s_base));
                            float _max3_8;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(sv_split[0]), "f"(sv_split[1]), "f"(sv_split[2]));
                            float _max3_9;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(sv_split[3]), "f"(sv_split[4]), "f"(sv_split[5]));
                            float _max3_10;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(sv_split[6]), "f"(sv_split[7]), "f"(sv_split[8]));
                            float _max3_7;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(_max3_8), "f"(_max3_9), "f"(_max3_10));
                            float _max3_12;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_12) : "f"(sv_split[9]), "f"(sv_split[10]), "f"(sv_split[11]));
                            float _max3_13;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_13) : "f"(sv_split[12]), "f"(sv_split[13]), "f"(sv_split[14]));
                            float _max3_14;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_14) : "f"(sv_split[15]), "f"(sv_split[16]), "f"(sv_split[17]));
                            float _max3_11;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(_max3_12), "f"(_max3_13), "f"(_max3_14));
                            float _max3_16;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(sv_split[18]), "f"(sv_split[19]), "f"(sv_split[20]));
                            float _max3_17;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(sv_split[21]), "f"(sv_split[22]), "f"(sv_split[23]));
                            float _max3_18;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(sv_split[24]), "f"(sv_split[25]), "f"(sv_split[26]));
                            float _max3_15;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_15) : "f"(_max3_16), "f"(_max3_17), "f"(_max3_18));
                            float _max3_6;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(_max3_7), "f"(_max3_11), "f"(_max3_15));
                            float _max3_21;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(sv_split[27]), "f"(sv_split[28]), "f"(sv_split[29]));
                            float _max3_22;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(sv_split[30]), "f"(sv_split[31]), "f"(sv_split[32]));
                            float _max3_23;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_23) : "f"(sv_split[33]), "f"(sv_split[34]), "f"(sv_split[35]));
                            float _max3_20;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_20) : "f"(_max3_21), "f"(_max3_22), "f"(_max3_23));
                            float _max3_25;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(sv_split[36]), "f"(sv_split[37]), "f"(sv_split[38]));
                            float _max3_26;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(sv_split[39]), "f"(sv_split[40]), "f"(sv_split[41]));
                            float _max3_27;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(sv_split[42]), "f"(sv_split[43]), "f"(sv_split[44]));
                            float _max3_24;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(_max3_25), "f"(_max3_26), "f"(_max3_27));
                            float _max3_29;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(sv_split[45]), "f"(sv_split[46]), "f"(sv_split[47]));
                            float _max3_30;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(sv_split[48]), "f"(sv_split[49]), "f"(sv_split[50]));
                            float _max3_31;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_31) : "f"(sv_split[51]), "f"(sv_split[52]), "f"(sv_split[53]));
                            float _max3_28;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(_max3_29), "f"(_max3_30), "f"(_max3_31));
                            float _max3_19;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_19) : "f"(_max3_20), "f"(_max3_24), "f"(_max3_28));
                            float _max3_33;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(sv_split[54]), "f"(sv_split[55]), "f"(sv_split[56]));
                            float _max3_34;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_34) : "f"(sv_split[57]), "f"(sv_split[58]), "f"(sv_split[59]));
                            float _max3_35;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_35) : "f"(sv_split[60]), "f"(sv_split[61]), "f"(sv_split[62]));
                            float _max3_32;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(_max3_33), "f"(_max3_34), "f"(_max3_35));
                            float _max3_5;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(_max3_6), "f"(_max3_19), "f"(max_noftz(_max3_32, sv_split[63])));
                            score_tile_max = _max3_5;
                        } else if (valid_sparse_cols != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_split[0]), "=f"(sv_split[1]), "=f"(sv_split[2]), "=f"(sv_split[3]), "=f"(sv_split[4]), "=f"(sv_split[5]), "=f"(sv_split[6]), "=f"(sv_split[7]), "=f"(sv_split[8]), "=f"(sv_split[9]), "=f"(sv_split[10]), "=f"(sv_split[11]), "=f"(sv_split[12]), "=f"(sv_split[13]), "=f"(sv_split[14]), "=f"(sv_split[15]), "=f"(sv_split[16]), "=f"(sv_split[17]), "=f"(sv_split[18]), "=f"(sv_split[19]), "=f"(sv_split[20]), "=f"(sv_split[21]), "=f"(sv_split[22]), "=f"(sv_split[23]), "=f"(sv_split[24]), "=f"(sv_split[25]), "=f"(sv_split[26]), "=f"(sv_split[27]), "=f"(sv_split[28]), "=f"(sv_split[29]), "=f"(sv_split[30]), "=f"(sv_split[31])
                                : "r"(s_base));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_split[32]), "=f"(sv_split[33]), "=f"(sv_split[34]), "=f"(sv_split[35]), "=f"(sv_split[36]), "=f"(sv_split[37]), "=f"(sv_split[38]), "=f"(sv_split[39]), "=f"(sv_split[40]), "=f"(sv_split[41]), "=f"(sv_split[42]), "=f"(sv_split[43]), "=f"(sv_split[44]), "=f"(sv_split[45]), "=f"(sv_split[46]), "=f"(sv_split[47]), "=f"(sv_split[48]), "=f"(sv_split[49]), "=f"(sv_split[50]), "=f"(sv_split[51]), "=f"(sv_split[52]), "=f"(sv_split[53]), "=f"(sv_split[54]), "=f"(sv_split[55]), "=f"(sv_split[56]), "=f"(sv_split[57]), "=f"(sv_split[58]), "=f"(sv_split[59]), "=f"(sv_split[60]), "=f"(sv_split[61]), "=f"(sv_split[62]), "=f"(sv_split[63])
                                : "r"(s_base + 32));
                            uint32_t _thresh_mask_36;
                            {
                                int _lim_37 = valid_sparse_cols;
                                if (_lim_37 <= 0) { _thresh_mask_36 = 0u; }
                                else if (_lim_37 >= 32) { _thresh_mask_36 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_36) : "r"(_lim_37));
                                }
                            }
                            if (!(_thresh_mask_36 & (1u << 0))) sv_split[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 1))) sv_split[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 2))) sv_split[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 3))) sv_split[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 4))) sv_split[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 5))) sv_split[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 6))) sv_split[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 7))) sv_split[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 8))) sv_split[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 9))) sv_split[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 10))) sv_split[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 11))) sv_split[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 12))) sv_split[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 13))) sv_split[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 14))) sv_split[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 15))) sv_split[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 16))) sv_split[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 17))) sv_split[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 18))) sv_split[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 19))) sv_split[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 20))) sv_split[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 21))) sv_split[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 22))) sv_split[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 23))) sv_split[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 24))) sv_split[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 25))) sv_split[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 26))) sv_split[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 27))) sv_split[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 28))) sv_split[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 29))) sv_split[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 30))) sv_split[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_36 & (1u << 31))) sv_split[31] = -CUDART_INF_F;
                            uint32_t _thresh_mask_38;
                            {
                                int _lim_39 = valid_sparse_cols - 32;
                                if (_lim_39 <= 0) { _thresh_mask_38 = 0u; }
                                else if (_lim_39 >= 32) { _thresh_mask_38 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_38) : "r"(_lim_39));
                                }
                            }
                            if (!(_thresh_mask_38 & (1u << 0))) sv_split[32] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 1))) sv_split[33] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 2))) sv_split[34] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 3))) sv_split[35] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 4))) sv_split[36] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 5))) sv_split[37] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 6))) sv_split[38] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 7))) sv_split[39] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 8))) sv_split[40] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 9))) sv_split[41] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 10))) sv_split[42] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 11))) sv_split[43] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 12))) sv_split[44] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 13))) sv_split[45] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 14))) sv_split[46] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 15))) sv_split[47] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 16))) sv_split[48] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 17))) sv_split[49] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 18))) sv_split[50] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 19))) sv_split[51] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 20))) sv_split[52] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 21))) sv_split[53] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 22))) sv_split[54] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 23))) sv_split[55] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 24))) sv_split[56] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 25))) sv_split[57] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 26))) sv_split[58] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 27))) sv_split[59] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 28))) sv_split[60] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 29))) sv_split[61] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 30))) sv_split[62] = -CUDART_INF_F;
                            if (!(_thresh_mask_38 & (1u << 31))) sv_split[63] = -CUDART_INF_F;
                            if (any_invalid != 0) {
                                #pragma unroll
                                for (int w_1 = 0; w_1 < 2; w_1++) {
                                    unsigned int mask_word = invalid_cols[w_1];
                                    sv_split[w_1 * 32] = (((mask_word & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32]);
                                    sv_split[w_1 * 32 + 1] = (((mask_word >> 1 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 1]);
                                    sv_split[w_1 * 32 + 2] = (((mask_word >> 2 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 2]);
                                    sv_split[w_1 * 32 + 3] = (((mask_word >> 3 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 3]);
                                    sv_split[w_1 * 32 + 4] = (((mask_word >> 4 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 4]);
                                    sv_split[w_1 * 32 + 5] = (((mask_word >> 5 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 5]);
                                    sv_split[w_1 * 32 + 6] = (((mask_word >> 6 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 6]);
                                    sv_split[w_1 * 32 + 7] = (((mask_word >> 7 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 7]);
                                    sv_split[w_1 * 32 + 8] = (((mask_word >> 8 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 8]);
                                    sv_split[w_1 * 32 + 9] = (((mask_word >> 9 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 9]);
                                    sv_split[w_1 * 32 + 10] = (((mask_word >> 10 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 10]);
                                    sv_split[w_1 * 32 + 11] = (((mask_word >> 11 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 11]);
                                    sv_split[w_1 * 32 + 12] = (((mask_word >> 12 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 12]);
                                    sv_split[w_1 * 32 + 13] = (((mask_word >> 13 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 13]);
                                    sv_split[w_1 * 32 + 14] = (((mask_word >> 14 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 14]);
                                    sv_split[w_1 * 32 + 15] = (((mask_word >> 15 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 15]);
                                    sv_split[w_1 * 32 + 16] = (((mask_word >> 16 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 16]);
                                    sv_split[w_1 * 32 + 17] = (((mask_word >> 17 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 17]);
                                    sv_split[w_1 * 32 + 18] = (((mask_word >> 18 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 18]);
                                    sv_split[w_1 * 32 + 19] = (((mask_word >> 19 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 19]);
                                    sv_split[w_1 * 32 + 20] = (((mask_word >> 20 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 20]);
                                    sv_split[w_1 * 32 + 21] = (((mask_word >> 21 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 21]);
                                    sv_split[w_1 * 32 + 22] = (((mask_word >> 22 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 22]);
                                    sv_split[w_1 * 32 + 23] = (((mask_word >> 23 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 23]);
                                    sv_split[w_1 * 32 + 24] = (((mask_word >> 24 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 24]);
                                    sv_split[w_1 * 32 + 25] = (((mask_word >> 25 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 25]);
                                    sv_split[w_1 * 32 + 26] = (((mask_word >> 26 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 26]);
                                    sv_split[w_1 * 32 + 27] = (((mask_word >> 27 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 27]);
                                    sv_split[w_1 * 32 + 28] = (((mask_word >> 28 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 28]);
                                    sv_split[w_1 * 32 + 29] = (((mask_word >> 29 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 29]);
                                    sv_split[w_1 * 32 + 30] = (((mask_word >> 30 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 30]);
                                    sv_split[w_1 * 32 + 31] = (((mask_word >> 31 & 1) != 0) ? -CUDART_INF_F : sv_split[w_1 * 32 + 31]);
                                }
                            }
                            float2 _reg_reduce_max2_41 = {-CUDART_INF_F, -CUDART_INF_F};
                            row_max_x32_accum(&sv_split[0], _reg_reduce_max2_41);
                            row_max_x32_accum(&sv_split[32], _reg_reduce_max2_41);
                            float _reg_array_reduce_40 = row_max_reduce(_reg_reduce_max2_41);
                            score_tile_max = _reg_array_reduce_40;
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(s_empty_addr + ((int)((unsigned int)(softmax_tile_cursor + tile_1) % 2)) * 8), "r"(0) : "memory");
                        new_max = max_noftz(new_max, score_tile_max);
                        int _stage_split_pv_p_empty = (int)((unsigned int)(softmax_tile_cursor + tile_1) % 2);
                        int _parity_split_pv_p_empty = (int)((unsigned int)(softmax_tile_cursor + tile_1) / 2 % 2) ^ 1;
                        mbarrier_wait(split_pv_p_empty_addr + (_stage_split_pv_p_empty) * 8, _parity_split_pv_p_empty);
                        smem_softmax_warp_pair_exchange[phase * 128 + stats_row] = new_max;
                        asm volatile("barrier.sync %0, 64;" :: "r"(2 + warp % 2) : "memory");
                        new_max = max_noftz(new_max, smem_softmax_warp_pair_exchange[phase * 128 + (stats_row ^ 64)]);
                        if (row_max_val > -CUDART_INF_F && (new_max - row_max_val) * softmax_scale_log2 <= 5.0f) {
                            new_max = row_max_val;
                        }
                        float delta = __fmaf_rn(row_max_val, softmax_scale_log2, -new_max * softmax_scale_log2);
                        float exp_delta = approx_exp2(delta);
                        float acc_scale = ((row_max_val > -CUDART_INF_F && new_max != row_max_val) ? exp_delta : 1.0f);
                        int _stage_stats_empty = (int)((unsigned int)(softmax_tile_cursor + tile_1) % 2);
                        int _parity_stats_empty = (int)((unsigned int)(softmax_tile_cursor + tile_1) / 2 % 2) ^ 1;
                        mbarrier_wait(stats_empty_addr + (_stage_stats_empty) * 8, _parity_stats_empty);
                        smem_stats_max[phase * 128 + stats_row] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(stats_addr + ((int)((unsigned int)(softmax_tile_cursor + tile_1) % 2)) * 8);
                        row_max_val = new_max;
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float max_scaled = safe_max * softmax_scale_log2;
                        float block_sum = 0.0f;
                        float kexp_inv = 1.0f;
                        int _stage_nvfp4_kexp_full = (int)((unsigned int)(softmax_tile_cursor + tile_1) % 2);
                        int _parity_nvfp4_kexp_full = (int)((unsigned int)(softmax_tile_cursor + tile_1) / 2 % 2);
                        mbarrier_wait(nvfp4_kexp_full_addr + (_stage_nvfp4_kexp_full) * 8, _parity_nvfp4_kexp_full);
                        unsigned int kexp_bits = smem_kexp[phase * 4];
                        float kexp_f = (float)kexp_bits;
                        float p_bias = kexp_f - max_scaled;
                        unsigned int kexp_inv_bits = 127 - kexp_bits << 23;
                        kexp_inv = __uint_as_float(kexp_inv_bits);
                        if (valid_sparse_cols != 0) {
                            const float2 _fma_b2_42 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_43 = {p_bias, p_bias};
                            #pragma unroll
                            for (int _lf = 0; _lf < 32; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(sv_split)[_lf], _fma_b2_42, _fma_c2_43);
                            #pragma unroll
                            for (int _le = 0; _le < 64; _le++) {
                                sv_split[_le] = approx_exp2(sv_split[_le]);
                            }
                            unsigned int _fp8_0[16];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv_split[0]), "f"(sv_split[1]),
                                                       "f"(sv_split[2]), "f"(sv_split[3]));
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
                                    : "=r"(_packed) : "f"(sv_split[4]), "f"(sv_split[5]),
                                                       "f"(sv_split[6]), "f"(sv_split[7]));
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
                                    : "=r"(_packed) : "f"(sv_split[8]), "f"(sv_split[9]),
                                                       "f"(sv_split[10]), "f"(sv_split[11]));
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
                                    : "=r"(_packed) : "f"(sv_split[12]), "f"(sv_split[13]),
                                                       "f"(sv_split[14]), "f"(sv_split[15]));
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
                                    : "=r"(_packed) : "f"(sv_split[16]), "f"(sv_split[17]),
                                                       "f"(sv_split[18]), "f"(sv_split[19]));
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
                                    : "=r"(_packed) : "f"(sv_split[20]), "f"(sv_split[21]),
                                                       "f"(sv_split[22]), "f"(sv_split[23]));
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
                                    : "=r"(_packed) : "f"(sv_split[24]), "f"(sv_split[25]),
                                                       "f"(sv_split[26]), "f"(sv_split[27]));
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
                                    : "=r"(_packed) : "f"(sv_split[28]), "f"(sv_split[29]),
                                                       "f"(sv_split[30]), "f"(sv_split[31]));
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
                                    : "=r"(_packed) : "f"(sv_split[32]), "f"(sv_split[33]),
                                                       "f"(sv_split[34]), "f"(sv_split[35]));
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
                                    : "=r"(_packed) : "f"(sv_split[36]), "f"(sv_split[37]),
                                                       "f"(sv_split[38]), "f"(sv_split[39]));
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
                                    : "=r"(_packed) : "f"(sv_split[40]), "f"(sv_split[41]),
                                                       "f"(sv_split[42]), "f"(sv_split[43]));
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
                                    : "=r"(_packed) : "f"(sv_split[44]), "f"(sv_split[45]),
                                                       "f"(sv_split[46]), "f"(sv_split[47]));
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
                                    : "=r"(_packed) : "f"(sv_split[48]), "f"(sv_split[49]),
                                                       "f"(sv_split[50]), "f"(sv_split[51]));
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
                                    : "=r"(_packed) : "f"(sv_split[52]), "f"(sv_split[53]),
                                                       "f"(sv_split[54]), "f"(sv_split[55]));
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
                                    : "=r"(_packed) : "f"(sv_split[56]), "f"(sv_split[57]),
                                                       "f"(sv_split[58]), "f"(sv_split[59]));
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
                                    : "=r"(_packed) : "f"(sv_split[60]), "f"(sv_split[61]),
                                                       "f"(sv_split[62]), "f"(sv_split[63]));
                                _fp8_0[15] = _packed;
                            }
                            int p_stage_base = smem_p_fp8_addr + (unsigned int)(phase * 8192);
                            #pragma unroll
                            for (int p_vec = 0; p_vec < 4; p_vec++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_stage_base + (my_row * 128 + (n_half * 4 + p_vec) * 16 ^ (my_row * 128 + (n_half * 4 + p_vec) * 16 >> 7 & 7) << 4))), "r"(_fp8_0[p_vec * 4]), "r"(_fp8_0[p_vec * 4 + 1]), "r"(_fp8_0[p_vec * 4 + 2]), "r"(_fp8_0[p_vec * 4 + 3]) : "memory");
                            }
                        } else {
                            int zero_p_stage = smem_p_fp8_addr + (unsigned int)(phase * 8192);
                            #pragma unroll
                            for (int zero_p_vec = 0; zero_p_vec < 4; zero_p_vec++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((zero_p_stage + (my_row * 128 + (n_half * 4 + zero_p_vec) * 16 ^ (my_row * 128 + (n_half * 4 + zero_p_vec) * 16 >> 7 & 7) << 4))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(p_full_addr + ((int)((unsigned int)(softmax_tile_cursor + tile_1) % 2)) * 8), "r"(0) : "memory");
                        asm volatile("" ::: "memory");
                        if (valid_sparse_cols != 0) {
                            const float2* _reg_reduce_src2_45 = reinterpret_cast<const float2*>(&sv_split[0]);
                            float2 _reg_reduce_sum2_45_0 = make_float2(0.0f, 0.0f);
                            float2 _reg_reduce_sum2_45_1 = make_float2(0.0f, 0.0f);
                            float2 _reg_reduce_sum2_45_2 = make_float2(0.0f, 0.0f);
                            float2 _reg_reduce_sum2_45_3 = make_float2(0.0f, 0.0f);
                            #pragma unroll
                            for (int _rr = 0; _rr < 8; _rr++) {
                                add_f32x2_inplace(&_reg_reduce_sum2_45_0, _reg_reduce_src2_45[_rr * 4 + 0]);
                                add_f32x2_inplace(&_reg_reduce_sum2_45_1, _reg_reduce_src2_45[_rr * 4 + 1]);
                                add_f32x2_inplace(&_reg_reduce_sum2_45_2, _reg_reduce_src2_45[_rr * 4 + 2]);
                                add_f32x2_inplace(&_reg_reduce_sum2_45_3, _reg_reduce_src2_45[_rr * 4 + 3]);
                            }
                            add_f32x2_inplace(&_reg_reduce_sum2_45_0, _reg_reduce_sum2_45_1);
                            add_f32x2_inplace(&_reg_reduce_sum2_45_2, _reg_reduce_sum2_45_3);
                            add_f32x2_inplace(&_reg_reduce_sum2_45_0, _reg_reduce_sum2_45_2);
                            float _reg_array_reduce_44 = _reg_reduce_sum2_45_0.x + _reg_reduce_sum2_45_0.y;
                            block_sum = _reg_array_reduce_44;
                        }
                        block_sum = block_sum * kexp_inv;
                        row_sum_val = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                    }
                    mbarrier_wait(source_sum_empty_addr, _phase_source_sum_empty_0);
                    _phase_source_sum_empty_0 ^= 1;
                    smem_stats_sum[stats_row] = row_sum_val;
                    smem_stats_final_max[stats_row] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    softmax_tile_cursor += num_kv_tiles_1;
                    sm_item += 1;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_2) * 8, _phase_source_work_full_2);
                uint32_t clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_2)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_4)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_2)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_2)
                    : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_2) * 8), "r"(0) : "memory");
                source_work_stage_2 += 1;
                if (source_work_stage_2 == 4) { source_work_stage_2 = 0; _phase_source_work_full_2 ^= 1; }
                if (clc_valid_2 == 0) {
                    break;
                }
                source_work_x_2 = clc_ctaid_4 + (unsigned int)cta_rank;
                source_work_z_2 = clc_ctaid_2_2;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.442695f;
            float output_scale = bmm2_scale[0];
            const int wg_dummy_inc_1 = 0;
            int all_num_kv_tiles_2 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base_2 = warp % 2 * 32;
            const int n_half_1 = (int)(warp % 4 / 2);
            const int my_row_1 = tmem_row_base_2 + lane;
            const int stats_row_1 = n_half_1 * 64 + my_row_1;
            const int corr_row = tmem_row_base_2 << 16;
            int correction_tile_cursor = 0;
            int ep_item = 0;
            unsigned int source_work_stage_3 = 0;
            unsigned int source_work_x_3 = blockIdx.x;
            unsigned int source_work_z_3 = blockIdx.z;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(o_empty_addr), "r"(0) : "memory");
            unsigned int _phase_sum_ready_0 = 0;
            unsigned int _phase_source_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = 0; work_idx_2 < 514; work_idx_2++) {
                int split_idx_2 = 0;
                int query_idx_2 = source_work_x_3 >> 1 >> 1;
                int v_chunk = source_work_x_3 >> 1 & 1;
                split_idx_2 = 0;
                query_idx_2 = source_work_x_3 >> 1;
                v_chunk = bid % 2;
                int source_work_valid_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1);
                if (source_work_valid_3 != 0) {
                    int _max_0_2 = ((sparse_topk_lens[mapped_query_idx_3] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_3] + sparse_topk_lens_offset) : (0));
                    int main_active_3 = ((_max_0_2) < (swa_width) ? (_max_0_2) : (swa_width));
                    int extra_active_3 = 0;
                    if (compressed_width > 0) {
                        int _max_2_2 = ((extra_topk_lens[mapped_query_idx_3]) > (0) ? (extra_topk_lens[mapped_query_idx_3]) : (0));
                        int _min_1_2 = ((_max_2_2) < (compressed_width) ? (_max_2_2) : (compressed_width));
                        extra_active_3 = _min_1_2;
                    }
                    int n_main_tiles_3 = (main_active_3 + 128 - 1) / 128;
                    int n_extra_tiles_3 = (extra_active_3 + 128 - 1) / 128;
                    int _max_3_3 = ((n_main_tiles_3 + n_extra_tiles_3) > (1) ? (n_main_tiles_3 + n_extra_tiles_3) : (1));
                    int active_topk_2 = _max_3_3 * 128;
                    all_num_kv_tiles_2 = (active_topk_2 + 128 - 1) / 128;
                    int tiles_per_split_2 = all_num_kv_tiles_2 + 1 - 1;
                    int first_tile_2 = split_idx_2 * tiles_per_split_2;
                    int num_kv_tiles_2 = tiles_per_split_2;
                    int first_stats_tile = correction_tile_cursor;
                    int first_stats_phase = first_stats_tile & 1;
                    int _stage_stats = (int)((unsigned int)correction_tile_cursor % 2);
                    int _parity_stats = (int)((unsigned int)correction_tile_cursor / 2 % 2);
                    mbarrier_wait(stats_addr + (_stage_stats) * 8, _parity_stats);
                    mbarrier_arrive(stats_empty_addr + ((int)((unsigned int)correction_tile_cursor % 2)) * 8);
                    #pragma unroll 1
                    for (int tile_2 = 1; tile_2 < num_kv_tiles_2; tile_2++) {
                        int phase_1 = correction_tile_cursor + tile_2 & 1;
                        int _stage_stats_1 = (int)((unsigned int)(correction_tile_cursor + tile_2) % 2);
                        int _parity_stats_1 = (int)((unsigned int)(correction_tile_cursor + tile_2) / 2 % 2);
                        mbarrier_wait(stats_addr + (_stage_stats_1) * 8, _parity_stats_1);
                        int ep_t0 = ep_item < 48 && tile_2 == 1;
                        float acc_scale_1 = smem_stats_max[phase_1 * 128 + stats_row_1];
                        mbarrier_arrive(stats_empty_addr + ((int)((unsigned int)(correction_tile_cursor + tile_2) % 2)) * 8);
                        int _parity_o_full = (int)((unsigned int)(correction_tile_cursor + tile_2) % 2) ^ 1;
                        uint32_t _mbar_token_5 = mbarrier_try_wait(o_full_addr, _parity_o_full);
                        int _parity_o_full_1 = (int)((unsigned int)(correction_tile_cursor + tile_2) % 2) ^ 1;
                        mbarrier_wait_token(o_full_addr, _parity_o_full_1, _mbar_token_5);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int any_rescale = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (any_rescale != 0) {
                            #pragma unroll
                            for (int vs = 0; vs < 2; vs++) {
                                int o_base = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                                #pragma unroll
                                for (int c = 0; c < 128; c += 16) {
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], o_base + c);
                                    const float2 _scale2_4 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_4);
                                    tmem_st_x16_f32(o_base + c, _tmem_load_1);
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_empty_addr), "r"(0) : "memory");
                    }
                    correction_tile_cursor += num_kv_tiles_2;
                    int _parity_o_full_2 = (int)((unsigned int)correction_tile_cursor % 2) ^ 1;
                    uint32_t _mbar_token_6 = mbarrier_try_wait(o_full_addr, _parity_o_full_2);
                    int _parity_o_full_3 = (int)((unsigned int)correction_tile_cursor % 2) ^ 1;
                    mbarrier_wait_token(o_full_addr, _parity_o_full_3, _mbar_token_6);
                    mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
                    _phase_sum_ready_0 ^= 1;
                    float total_sum = smem_stats_sum[stats_row_1];
                    float final_max = smem_stats_final_max[stats_row_1];
                    mbarrier_arrive(source_sum_empty_addr);
                    smem_corr_warp_pair_exchange[stats_row_1] = total_sum;
                    asm volatile("barrier.sync %0, 64;" :: "r"(4 + warp % 2) : "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    total_sum = total_sum + smem_corr_warp_pair_exchange[stats_row_1 ^ 64];
                    float _rcp_5 = approx_rcp(total_sum);
                    float inv_sum = ((total_sum > 0.0f) ? _rcp_5 : 0.0f);
                    int head_idx = my_row_1;
                    head_idx = bid % 2 * 64 + my_row_1;
                    long long direct_o_offset = ((long long)mapped_query_idx_3 * (long long)num_heads + (long long)head_idx) * 512ll + (long long)(v_chunk * 256);
                    long long partial_o_offset = ((long long)mapped_query_idx_3 * (long long)num_heads + (long long)(head_idx + split_idx_2)) * 512ll + (long long)(n_half_1 * 256);
                    long long o_offset = partial_o_offset;
                    if (n_half_1 == 0 && head_idx < num_heads) {
                        float _log2_6;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(total_sum));
                        LSE[mapped_query_idx_3 * num_heads + head_idx] = ((total_sum > 0.0f) ? __fmaf_rn(final_max, softmax_scale_log2_1, _log2_6) : -CUDART_INF_F);
                    }
                    float epi_scale = inv_sum * output_scale;
                    int epi_head0 = bid % 2 * 64 + tmem_row_base_2;
                    int epi_issue_box = epi_head0 < num_heads;
                    int epi_stage_base = smem_epi_o_addr + (unsigned int)(warp % 4 * 4096);
                    int epi_swz = (lane & 7) << 4;
                    #pragma unroll 1
                    for (int box = 0; box < 4; box++) {
                        if (box >= 1) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            __syncwarp();
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        #pragma unroll
                        for (int epi_h = 0; epi_h < 2; epi_h++) {
                            int o_base_epi = taddr + 256 + (unsigned int)(box * 64 + epi_h * 32) + (unsigned int)corr_row;
                            float epi_ahead[32];
                            tmem_ld_x16(&epi_ahead[0], o_base_epi);
                            tmem_ld_x16(&epi_ahead[16], o_base_epi + 16);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 32; c_1 += 16) {
                                float epi_scaled4[16];
                                #pragma unroll
                                for (int elem = 0; elem < 16; elem++) {
                                    epi_scaled4[elem] = epi_ahead[c_1 + elem] * epi_scale;
                                }
                                unsigned int epi_packed4[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi_scaled4[_lp*2 + 0], epi_scaled4[_lp*2+1 + 0]));
                                    epi_packed4[_lp] = *(uint32_t*)&_bf2;
                                }
                                int epi_chunk = epi_h * 4 + c_1 / 8;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(epi_stage_base + lane * 128 + (epi_chunk * 16 ^ epi_swz)), "r"(epi_packed4[0]), "r"(epi_packed4[1]), "r"(epi_packed4[2]), "r"(epi_packed4[3]) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(epi_stage_base + lane * 128 + ((epi_chunk + 1) * 16 ^ epi_swz)), "r"(epi_packed4[4]), "r"(epi_packed4[5]), "r"(epi_packed4[6]), "r"(epi_packed4[7]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (epi_issue_box != 0) {
                            if (elect_sync()) {
                                tma_store_3d((&tmap_o), 0, mapped_query_idx_3 * num_heads + epi_head0, n_half_1 * 4 + box, epi_stage_base);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(o_empty_addr), "r"(0) : "memory");
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    ep_item += 1;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_3) * 8, _phase_source_work_full_3);
                uint32_t clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_3)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_5)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_3)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_3)
                    : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_3) * 8), "r"(0) : "memory");
                source_work_stage_3 += 1;
                if (source_work_stage_3 == 4) { source_work_stage_3 = 0; _phase_source_work_full_3 ^= 1; }
                if (clc_valid_3 == 0) {
                    break;
                }
                source_work_x_3 = clc_ctaid_5 + (unsigned int)cta_rank;
                source_work_z_3 = clc_ctaid_2_3;
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg2_dummy_1 = 0;
            int all_num_kv_tiles_3 = (sparse_topk + 128 - 1) / 128;
            unsigned int mma_k_stage = 0;
            unsigned int mma_v_stage = 0;
            unsigned int mma_kv_stage = 0;
            int mma_tile_cursor = 0;
            int mma_item = 0;
            unsigned int source_work_stage_4 = 0;
            unsigned int source_work_x_4 = blockIdx.x;
            unsigned int source_work_z_4 = blockIdx.z;
            if (cta_rank == 0) {
                uint32_t _mbar_token_0 = mbarrier_try_wait(s_empty_addr, 1);
                mbarrier_wait_token(s_empty_addr, 1, _mbar_token_0);
                uint32_t _mbar_token_1 = mbarrier_try_wait(s_empty_addr + 8, 1);
                mbarrier_wait_token(s_empty_addr + 8, 1, _mbar_token_1);
            }
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_nvfp4_sf_full = 0;
            unsigned int _phase_source_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = 0; work_idx_3 < 514; work_idx_3++) {
                int split_idx_3 = 0;
                split_idx_3 = 0;
                int query_idx_3 = source_work_x_4 >> 1;
                int source_work_valid_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1);
                if (source_work_valid_4 != 0) {
                    if (warp == 8) {
                        int _parity_q_empty = (int)((unsigned int)mma_item % 2) ^ 1;
                        mbarrier_wait(q_empty_addr, _parity_q_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(nvfp4_q_load_full_addr, 26624);
                            int row0 = bid % 2 * 64;
                            tma_3d_gmem2smem(smem_q_a_addr, (&tmap_q), 0, row0, mapped_query_idx_4, nvfp4_q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_b_addr, (&tmap_q), 128, row0, mapped_query_idx_4, nvfp4_q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_rope_addr, (&tmap_q), 224, row0, mapped_query_idx_4, nvfp4_q_load_full_addr);
                            tma_3d_gmem2smem(smem_q_sf_addr, (&tmap_q_sf), 352, row0, mapped_query_idx_4, nvfp4_q_load_full_addr);
                        }
                    }
                    int _max_0_3 = ((sparse_topk_lens[mapped_query_idx_4] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_4] + sparse_topk_lens_offset) : (0));
                    int main_active_4 = ((_max_0_3) < (swa_width) ? (_max_0_3) : (swa_width));
                    int extra_active_4 = 0;
                    if (compressed_width > 0) {
                        int _max_2_3 = ((extra_topk_lens[mapped_query_idx_4]) > (0) ? (extra_topk_lens[mapped_query_idx_4]) : (0));
                        int _min_1_3 = ((_max_2_3) < (compressed_width) ? (_max_2_3) : (compressed_width));
                        extra_active_4 = _min_1_3;
                    }
                    int n_main_tiles_4 = (main_active_4 + 128 - 1) / 128;
                    int n_extra_tiles_4 = (extra_active_4 + 128 - 1) / 128;
                    int _max_3_4 = ((n_main_tiles_4 + n_extra_tiles_4) > (1) ? (n_main_tiles_4 + n_extra_tiles_4) : (1));
                    int active_topk_3 = _max_3_4 * 128;
                    all_num_kv_tiles_3 = (active_topk_3 + 128 - 1) / 128;
                    int tiles_per_split_3 = all_num_kv_tiles_3 + 1 - 1;
                    int first_tile_3 = split_idx_3 * tiles_per_split_3;
                    int num_kv_tiles_3 = tiles_per_split_3;
                    if (cta_rank == 0) {
                        mbarrier_wait(q_full_addr, _phase_q_full_0);
                        _phase_q_full_0 ^= 1;
                    }
                    int first_pv = 1;
                    int mma_on = mma_item < 48;
                    int first_pipeline_tile = mma_tile_cursor;
                    int first_phase = first_pipeline_tile & 1;
                    int first_score_col = ((first_phase != 0) ? 128 : 0);
                    if (cta_rank == 0) {
                        mbarrier_wait(nvfp4_sf_full_addr + (mma_k_stage) * 8, _phase_nvfp4_sf_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int mma_a_lo = (((smem_q_a_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int mma_b_lo = (((smem_k_a_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2816;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 0, b_desc + 0,
                                    0x8200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + first_phase * 128 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 2, b_desc + 2,
                                    0x8200480U, tmem_tmem_sfa + 4, tmem_tmem_sfb + first_phase * 128 + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 4, b_desc + 4,
                                    0x8200480U, tmem_tmem_sfa + 8, tmem_tmem_sfb + first_phase * 128 + 8, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 6, b_desc + 6,
                                    0x8200480U, tmem_tmem_sfa + 12, tmem_tmem_sfb + first_phase * 128 + 12, 1);
                            }
                        }
                        int mma_a_lo_1 = (((smem_q_b_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int mma_b_lo_1 = (((smem_k_b_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2816;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 0, b_desc + 0,
                                    0x8200480U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + (first_phase * 128 + 16) + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 2, b_desc + 2,
                                    0x8200480U, tmem_tmem_sfa + 16 + 4, tmem_tmem_sfb + (first_phase * 128 + 16) + 4, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 4, b_desc + 4,
                                    0x8200480U, tmem_tmem_sfa + 16 + 8, tmem_tmem_sfb + (first_phase * 128 + 16) + 8, 1);
                            }
                        }
                        int mma_a_lo_2 = (((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int mma_b_lo_2 = (((smem_k_c_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2816;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(mma_a_lo_2), "r"(mma_b_lo_2), "r"((tmem_tmem_scratch + (first_score_col))), "r"(1) : "memory");
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                        }
                        mma_k_stage += 1;
                        if (mma_k_stage == 2) { mma_k_stage = 0; _phase_nvfp4_sf_full ^= 1; }
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(s_full_addr + ((int)((unsigned int)mma_tile_cursor % 2)) * 8, (uint16_t)(3));
                        }
                    }
                    #pragma unroll 1
                    for (int tile_3 = 1; tile_3 < num_kv_tiles_3; tile_3++) {
                        int phase_2 = mma_tile_cursor + tile_3 & 1;
                        int score_col = ((phase_2 != 0) ? 128 : 0);
                        if (cta_rank == 0) {
                            mbarrier_wait(nvfp4_sf_full_addr + (mma_k_stage) * 8, _phase_nvfp4_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int mma_a_lo_3 = (((smem_q_a_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int mma_b_lo_3 = (((smem_k_a_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2816;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 0, b_desc + 0,
                                        0x8200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + phase_2 * 128 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 2, b_desc + 2,
                                        0x8200480U, tmem_tmem_sfa + 4, tmem_tmem_sfb + phase_2 * 128 + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 4, b_desc + 4,
                                        0x8200480U, tmem_tmem_sfa + 8, tmem_tmem_sfb + phase_2 * 128 + 8, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 6, b_desc + 6,
                                        0x8200480U, tmem_tmem_sfa + 12, tmem_tmem_sfb + phase_2 * 128 + 12, 1);
                                }
                            }
                            int mma_a_lo_1_1 = (((smem_q_b_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int mma_b_lo_1_1 = (((smem_k_b_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2816;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_1_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_1_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 0, b_desc + 0,
                                        0x8200480U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + (phase_2 * 128 + 16) + 0, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 2, b_desc + 2,
                                        0x8200480U, tmem_tmem_sfa + 16 + 4, tmem_tmem_sfb + (phase_2 * 128 + 16) + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 4, b_desc + 4,
                                        0x8200480U, tmem_tmem_sfa + 16 + 8, tmem_tmem_sfb + (phase_2 * 128 + 16) + 8, 1);
                                }
                            }
                            int mma_a_lo_2_1 = (((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int mma_b_lo_2_1 = (((smem_k_c_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2816;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(mma_a_lo_2_1), "r"(mma_b_lo_2_1), "r"((tmem_tmem_scratch + (score_col))), "r"(1) : "memory");
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                            }
                            mma_k_stage += 1;
                            if (mma_k_stage == 2) { mma_k_stage = 0; _phase_nvfp4_sf_full ^= 1; }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(s_full_addr + ((int)((unsigned int)(mma_tile_cursor + tile_3) % 2)) * 8, (uint16_t)(3));
                            }
                            int next_s_stage = mma_tile_cursor + tile_3 + 1 & 1;
                            int _stage_s_empty = (int)((unsigned int)(mma_tile_cursor + tile_3 + 1) % 2);
                            int _parity_s_empty = (int)((unsigned int)(mma_tile_cursor + tile_3 + 1) / 2 % 2) ^ 1;
                            uint32_t _mbar_token_2 = mbarrier_try_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                            int _stage_s_empty_1 = (int)((unsigned int)(mma_tile_cursor + tile_3 + 1) % 2);
                            int _parity_s_empty_1 = (int)((unsigned int)(mma_tile_cursor + tile_3 + 1) / 2 % 2) ^ 1;
                            mbarrier_wait_token(s_empty_addr + (_stage_s_empty_1) * 8, _parity_s_empty_1, _mbar_token_2);
                        }
                    }
                    if (cta_rank == 0) {
                        elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    }
                    mma_tile_cursor += num_kv_tiles_3;
                    int last_pipeline_tile = mma_tile_cursor - 1;
                    int last_phase = last_pipeline_tile & 1;
                    int drain_wait_phase = last_pipeline_tile >> 1 & 1;
                    int final_s_stage = mma_tile_cursor + 1 & 1;
                    if (cta_rank == 0) {
                        int _stage_s_empty_2 = (int)((unsigned int)(mma_tile_cursor + 1) % 2);
                        int _parity_s_empty_2 = (int)((unsigned int)(mma_tile_cursor + 1) / 2 % 2) ^ 1;
                        uint32_t _mbar_token_3 = mbarrier_try_wait(s_empty_addr + (_stage_s_empty_2) * 8, _parity_s_empty_2);
                        int _stage_s_empty_3 = (int)((unsigned int)(mma_tile_cursor + 1) % 2);
                        int _parity_s_empty_3 = (int)((unsigned int)(mma_tile_cursor + 1) / 2 % 2) ^ 1;
                        mbarrier_wait_token(s_empty_addr + (_stage_s_empty_3) * 8, _parity_s_empty_3, _mbar_token_3);
                    }
                    mma_item += 1;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_4) * 8, _phase_source_work_full_4);
                uint32_t clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_4)
                    : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_6)
                    : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_4)
                    : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_4)
                    : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_4) * 8), "r"(0) : "memory");
                source_work_stage_4 += 1;
                if (source_work_stage_4 == 4) { source_work_stage_4 = 0; _phase_source_work_full_4 ^= 1; }
                if (clc_valid_4 == 0) {
                    break;
                }
                source_work_x_4 = clc_ctaid_6 + (unsigned int)cta_rank;
                source_work_z_4 = clc_ctaid_2_4;
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int tmem_dealloc_peer_rank = bid % 2 ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_dealloc_peer_addr), "r"(tmem_dealloc_peer_rank) : "memory");
            mbarrier_wait(tmem_dealloc_peer_addr, 0);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            const int wg2_dummy_2 = 0;
            unsigned int source_work_stage_5 = 0;
            unsigned int source_throttle_stage_1 = 0;
            int claim_ahead = 0;
            unsigned int _phase_source_throttle_full = 0;
            unsigned int _phase_source_work_empty = 1;
            unsigned int _phase_source_work_full_5 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _source_work = 0; _source_work < 514; _source_work++) {
                    if (_source_work != 1 || claim_ahead == 0) {
                        mbarrier_wait(source_throttle_full_addr + (source_throttle_stage_1) * 8, _phase_source_throttle_full);
                        mbarrier_arrive(source_throttle_empty_addr + (source_throttle_stage_1) * 8);
                        source_throttle_stage_1 += 1;
                        if (source_throttle_stage_1 == 2) { source_throttle_stage_1 = 0; _phase_source_throttle_full ^= 1; }
                    }
                    mbarrier_wait(source_work_empty_addr + (source_work_stage_5) * 8, _phase_source_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(source_work_full_addr + (source_work_stage_5) * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16), "r"(source_work_full_addr + source_work_stage_5 * 8)
                            : "memory");
                    }
                    mbarrier_wait(source_work_full_addr + (source_work_stage_5) * 8, _phase_source_work_full_5);
                    uint32_t clc_valid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid_5)
                        : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_7 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_7)
                        : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_1_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_1_5)
                        : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_2_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_2_5)
                        : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    source_work_stage_5 += 1;
                    if (source_work_stage_5 == 4) { source_work_stage_5 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_5 ^= 1; }
                    if (clc_valid_5 == 0) {
                        break;
                    }
                    if (_source_work == 0 && sparse_topk <= 256 && clc_ctaid_7 < (unsigned int)gridDim.x / 2) {
                        claim_ahead = 1;
                    }
                }
                #pragma unroll
                for (unsigned int _source_tail = 0; _source_tail < 4; _source_tail++) {
                    mbarrier_wait(source_work_empty_addr + (source_work_stage_5) * 8, _phase_source_work_empty);
                    source_work_stage_5 += 1;
                    if (source_work_stage_5 == 4) { source_work_stage_5 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_5 ^= 1; }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            const int wg3_dummy = 0;
            unsigned int source_work_stage_6 = 0;
            unsigned int source_work_x_5 = blockIdx.x;
            unsigned int source_work_z_5 = blockIdx.z;
            int pv_tile_cursor = 0;
            unsigned int pv_v_stage = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_source_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _pv_work = 0; _pv_work < 514; _pv_work++) {
                int pv_valid = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1) < (unsigned int)num_query_tokens;
                if (pv_valid != 0) {
                    int pv_query = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1);
                    int _max_0_4 = ((sparse_topk_lens[pv_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[pv_query] + sparse_topk_lens_offset) : (0));
                    int pv_main = ((_max_0_4) < (swa_width) ? (_max_0_4) : (swa_width));
                    int pv_extra = 0;
                    if (compressed_width > 0) {
                        int _max_2_4 = ((extra_topk_lens[pv_query]) > (0) ? (extra_topk_lens[pv_query]) : (0));
                        int _min_1_4 = ((_max_2_4) < (compressed_width) ? (_max_2_4) : (compressed_width));
                        pv_extra = _min_1_4;
                    }
                    int pv_tiles = (((pv_main + 128 - 1) / 128 + (pv_extra + 128 - 1) / 128) > (1) ? ((pv_main + 128 - 1) / 128 + (pv_extra + 128 - 1) / 128) : (1));
                    int pv_first = 1;
                    #pragma unroll 1
                    for (int pv_tile = 0; pv_tile < pv_tiles; pv_tile++) {
                        int pv_phase = pv_tile_cursor + pv_tile & 1;
                        if (cta_rank == 0) {
                            int _stage_p_full = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                            int _parity_p_full = (int)((unsigned int)(pv_tile_cursor + pv_tile) / 2 % 2);
                            mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                            int _parity_o_empty = (int)((unsigned int)(pv_tile_cursor + pv_tile) % 2);
                            mbarrier_wait(o_empty_addr, _parity_o_empty);
                            mbarrier_wait(v_full_addr + (pv_v_stage) * 8, _phase_v_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_6 = (((smem_p_fp8_addr) >> 4) & 0x3FFF) + (pv_phase) * 512;
                            int _mma_b_lo_6 = ((((smem_v_full_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_v_stage) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_tmem_scratch + (256))), "r"(((pv_first) ? 0 : 1)) : "memory");
                            int _mma_a_lo_7 = (((smem_p_fp8_addr) >> 4) & 0x3FFF) + (pv_phase) * 512;
                            int _mma_b_lo_7 = ((((smem_v_full_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (pv_v_stage) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_tmem_scratch + (384))), "r"(((pv_first) ? 0 : 1)) : "memory");
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(v_empty_addr + (pv_v_stage) * 8, (uint16_t)(3));
                            }
                            pv_v_stage += 1;
                            if (pv_v_stage == 2) { pv_v_stage = 0; _phase_v_full ^= 1; }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(split_pv_p_empty_addr + ((int)((unsigned int)(pv_tile_cursor + pv_tile) % 2)) * 8, (uint16_t)(3));
                            }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                            }
                        }
                        pv_first = 0;
                    }
                    pv_tile_cursor += pv_tiles;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_6) * 8, _phase_source_work_full_6);
                uint32_t clc_valid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_6)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_8)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_1_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_1_6)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2_6)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_6) * 8), "r"(0) : "memory");
                source_work_stage_6 += 1;
                if (source_work_stage_6 == 4) { source_work_stage_6 = 0; _phase_source_work_full_6 ^= 1; }
                if (clc_valid_6 == 0) {
                    break;
                }
                source_work_x_5 = clc_ctaid_8 + (unsigned int)cta_rank;
                source_work_z_5 = clc_ctaid_2_6;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: gather ----
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // gather_main
            const int load_warp_rank_1 = warp - 16;
            int k_cta_offset_1 = bid % 2 * 64;
            int tile_cursor_1 = 0;
            unsigned int load_k_index_stage_1 = 0;
            unsigned int source_work_stage_7 = 0;
            unsigned int source_work_x_6 = blockIdx.x;
            unsigned int source_work_z_6 = blockIdx.z;
            int source_work_valid_5 = source_work_z_6 * (unsigned int)max_q_len + (source_work_x_6 >> 1) < (unsigned int)num_query_tokens;
            int mapped_query_idx_5 = source_work_z_6 * (unsigned int)max_q_len + (source_work_x_6 >> 1);
            unsigned int _phase_index_full = 0;
            unsigned int _phase_source_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = 0; work_idx_4 < 514; work_idx_4++) {
                if (source_work_valid_5 != 0) {
                    int _max_0_5 = ((sparse_topk_lens[mapped_query_idx_5] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_5] + sparse_topk_lens_offset) : (0));
                    int main_active_5 = ((_max_0_5) < (swa_width) ? (_max_0_5) : (swa_width));
                    int extra_active_5 = 0;
                    if (compressed_width > 0) {
                        int _max_2_5 = ((extra_topk_lens[mapped_query_idx_5]) > (0) ? (extra_topk_lens[mapped_query_idx_5]) : (0));
                        int _min_1_5 = ((_max_2_5) < (compressed_width) ? (_max_2_5) : (compressed_width));
                        extra_active_5 = _min_1_5;
                    }
                    int n_main_tiles_5 = (main_active_5 + 128 - 1) / 128;
                    int n_extra_tiles_5 = (extra_active_5 + 128 - 1) / 128;
                    int num_kv_tiles_4 = ((n_main_tiles_5 + n_extra_tiles_5) > (1) ? (n_main_tiles_5 + n_extra_tiles_5) : (1));
                    #pragma unroll 1
                    for (int tile_4 = 0; tile_4 < num_kv_tiles_4; tile_4++) {
                        int g_stage = tile_cursor_1 + tile_4 & 1;
                        if ((tile_4 & 1) == 0) {
                            mbarrier_wait(index_full_addr + (load_k_index_stage_1) * 8, _phase_index_full);
                        }
                        int _stage_k_empty_1 = (int)((unsigned int)(tile_cursor_1 + tile_4) % 2);
                        int _parity_k_empty_1 = (int)((unsigned int)(tile_cursor_1 + tile_4) / 2 % 2) ^ 1;
                        mbarrier_wait(k_empty_addr + (_stage_k_empty_1) * 8, _parity_k_empty_1);
                        int _stage_nvfp4_kstage_free = (int)((unsigned int)(tile_cursor_1 + tile_4) % 2);
                        int _parity_nvfp4_kstage_free = (int)((unsigned int)(tile_cursor_1 + tile_4) / 2 % 2) ^ 1;
                        mbarrier_wait(nvfp4_kstage_free_addr + (_stage_nvfp4_kstage_free) * 8, _parity_nvfp4_kstage_free);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (load_warp_rank_1 == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(k_full_addr + ((int)((unsigned int)(tile_cursor_1 + tile_4) % 2)) * 8, 28672 + ((k_cta_offset_1 >> 6) + 1) * 8192);
                            }
                        }
                        int g_group = lane & 15;
                        int g_peer = lane >> 4;
                        int g_rank = k_cta_offset_1 >> 6;
                        int g_vraw = load_warp_rank_1 == 3 || load_warp_rank_1 == 0 && g_rank == 0 || load_warp_rank_1 != 0 && load_warp_rank_1 < 3 && g_rank == 1;
                        if (g_peer == 0 || g_vraw != 0) {
                            int g_tok0 = k_cta_offset_1 ^ g_peer << 6;
                            int g_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&g_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 3]))
                                : "r"(smem_sparse_indices_addr + load_k_index_stage_1 * 1024 + (unsigned int)(((tile_4 & 1) * 128 + g_tok0 + g_group * 4) * 4)));
                            int g_use_swa = n_main_tiles_5 > tile_4 || n_extra_tiles_5 == 0;
                            int g_page_log2 = ((g_use_swa != 0) ? swa_page_log2 : compressed_page_log2);
                            int g_pitch_units = ((g_use_swa != 0) ? swa_pitch_units : compressed_pitch_units);
                            int g_footer_units = ((g_use_swa != 0) ? swa_footer_units : compressed_footer_units);
                            int g_data_rows[4];
                            int g_foot_rows[4];
                            #pragma unroll
                            for (int g_j = 0; g_j < 4; g_j++) {
                                int g_tok = ((g_rows[g_j] >= 0) ? g_rows[g_j] : 0);
                                int g_page = g_tok >> g_page_log2;
                                int g_slot = g_tok - (g_page << g_page_log2);
                                g_data_rows[g_j] = g_page * g_pitch_units + g_slot * 11;
                                g_foot_rows[g_j] = g_page * g_pitch_units + g_footer_units + g_slot;
                            }
                            int g_stage_base = smem_k_a_addr + (unsigned int)(g_stage * 45056);
                            if (load_warp_rank_1 < 3) {
                                int g_col = ((load_warp_rank_1 == 2) ? 224 : load_warp_rank_1 * 128);
                                int g_chunk_off = ((g_peer == 0) ? load_warp_rank_1 * 8192 : 28672 + (load_warp_rank_1 - g_rank) * 8192);
                                int g_dst = g_stage_base + g_chunk_off + g_group * 512;
                                if (g_use_swa != 0) {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_swa_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(tile_cursor_1 + tile_4) % 2)) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_dst, (&tmap_compressed_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + ((int)((unsigned int)(tile_cursor_1 + tile_4) % 2)) * 8);
                                }
                            } else {
                                int g_fdst = g_stage_base + 24576 + (g_tok0 + g_group * 4) * 32;
                                if (g_use_swa != 0) {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_swa_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(tile_cursor_1 + tile_4) % 2)) * 8);
                                } else {
                                    tma_gather4_gmem2smem(g_fdst, (&tmap_compressed_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + ((int)((unsigned int)(tile_cursor_1 + tile_4) % 2)) * 8);
                                }
                            }
                        }
                        if ((tile_4 & 1) != 0 || tile_4 + 1 == num_kv_tiles_4) {
                            mbarrier_arrive(index_empty_addr + (load_k_index_stage_1) * 8);
                            load_k_index_stage_1 += 1;
                            if (load_k_index_stage_1 == 6) { load_k_index_stage_1 = 0; _phase_index_full ^= 1; }
                        }
                    }
                    tile_cursor_1 += num_kv_tiles_4;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_7) * 8, _phase_source_work_full_7);
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
                    : "r"(work_response_addr + source_work_stage_7 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + source_work_stage_7 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + source_work_stage_7 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + source_work_stage_7 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + (source_work_stage_7) * 8), "r"(0) : "memory");
                source_work_stage_7 += 1;
                if (source_work_stage_7 == 4) { source_work_stage_7 = 0; _phase_source_work_full_7 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                source_work_x_6 = _clc_ctaid_9 + (unsigned int)cta_rank;
                source_work_z_6 = _clc_ctaid_11;
                mapped_query_idx_5 = source_work_z_6 * (unsigned int)max_q_len + (source_work_x_6 >> 1);
                source_work_valid_5 = mapped_query_idx_5 < num_query_tokens;
            }
        }
    }

    // Cleanup
}

} // extern "C"
